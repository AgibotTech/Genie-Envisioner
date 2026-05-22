import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import random
import traceback
import warnings

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from torch.utils.data.dataset import Dataset

from data.utils.domain_table import DomainTable
from data.utils.get_actions import parse_h5
from data.utils.statistics import StatisticInfo
from data.utils.utils import gen_crop_config, intrin_crop_transform, intrinsic_transform
from utils import zero_rank_print
from utils.get_ray_maps import get_ray_maps
from utils.get_traj_maps import get_traj_maps, simple_radius_gen_func

warnings.filterwarnings("ignore", category=FutureWarning)


class ChallengeWMDataset(Dataset):
    def __init__(
        self,
        data_roots,
        domains,
        split="train",
        sample_size=(320, 512),
        sample_n_frames=64,
        preprocess="resize",
        valid_cam=("head",),
        chunk=16,
        action_chunk=None,
        n_previous=4,
        previous_pick_mode="random",
        random_crop=True,
        min_sep=1,
        max_sep=3,
        fps=2,
        dataset_info_cache_path=None,
        use_unified_prompt=True,
        unified_prompt="best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx=None,
        fix_sidx=None,
        fix_mem_idx=None,
        action_space="eef",
        normalize_actions=False,
        stat_file=None,
        radius_mode="simple",
    ):
        zero_rank_print("loading challenge wm annotations...")

        if action_space != "eef":
            raise ValueError("ChallengeWMDataset currently only supports eef action_space.")

        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam]
        if len(valid_cam) != 1:
            raise ValueError("ChallengeWMDataset expects single-view input.")

        self.valid_cam = list(valid_cam)
        self.data_roots = data_roots
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.preprocess = preprocess
        self.chunk = chunk
        self.action_chunk = action_chunk or chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        if self.chunk * self.video_temporal_stride != self.action_chunk:
            raise ValueError("action_chunk should be an integer multiple of chunk.")

        self.n_previous = n_previous
        self.previous_pick_mode = previous_pick_mode
        self.random_crop = random_crop
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.fps = fps
        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx
        self.action_space = action_space
        self.normalize_actions = normalize_actions
        self.radius_mode = radius_mode

        if preprocess == "center_crop_resize":
            self.pixel_transforms_resize = transforms.Compose(
                [
                    transforms.Resize(min(sample_size)),
                    transforms.CenterCrop(sample_size),
                ]
            )
        elif preprocess == "resize":
            self.pixel_transforms_resize = transforms.Compose([transforms.Resize(sample_size)])
        else:
            raise NotImplementedError
        self.pixel_transforms_norm = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

        self.dataset = []
        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load cache dataset information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            for data_root, domain_name in zip(self.data_roots, domains):
                split_root = os.path.join(data_root, split)
                if not os.path.isdir(split_root):
                    raise FileNotFoundError(f"Challenge WM split not found: {split_root}")

                file_list = sorted(os.listdir(split_root))
                for file_name in file_list:
                    sample_root = os.path.join(split_root, file_name)
                    if not os.path.isdir(sample_root):
                        continue
                    info = [
                        sample_root,
                        domain_name,
                        DomainTable.get(domain_name, 0),
                        file_name,
                    ]
                    self.dataset.append(info)

            if dataset_info_cache_path is not None:
                zero_rank_print(f"Save cache dataset information to {dataset_info_cache_path}")
                cache_dir = os.path.dirname(dataset_info_cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(dataset_info_cache_path, "w") as f:
                    json.dump(self.dataset, f)

        self.length = len(self.dataset)
        zero_rank_print(f"challenge wm data scale: {self.length}")

    def __len__(self):
        return self.length

    def get_total_timesteps(self, sample_root, cam_name):
        with open(os.path.join(sample_root, f"{cam_name}_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        return len(info)

    def get_frame_indexes(self, total_frames, sep=1):
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.action_chunk))
            frame_indexes = action_indexes[self.video_temporal_stride - 1 :: self.video_temporal_stride]
            return self.fix_mem_idx + frame_indexes

        if total_frames > self.action_chunk * sep:
            chunk_end = random.randint(self.action_chunk * sep, total_frames)
        else:
            chunk_end = total_frames

        indexes = np.array(list(range(chunk_end - self.sample_n_frames * sep, chunk_end, sep)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames - 1).tolist()

        video_end = indexes[-self.action_chunk :]
        mem_candidates = indexes[: self.sample_n_frames - self.action_chunk]

        if len(mem_candidates) == 0:
            mem_candidates = [indexes[0]]

        if self.previous_pick_mode == "uniform":
            mem_indexes = [
                mem_candidates[int(i)]
                for i in np.linspace(0, len(mem_candidates) - 1, self.n_previous).tolist()
            ]
        elif self.previous_pick_mode == "random":
            if len(mem_candidates) <= self.n_previous:
                mem_indexes = mem_candidates[: self.n_previous]
                while len(mem_indexes) < self.n_previous:
                    mem_indexes.insert(0, mem_indexes[0])
            else:
                sampled = sorted(
                    np.random.choice(
                        list(range(0, len(mem_candidates) - 1)),
                        size=self.n_previous - 1,
                        replace=False,
                    ).tolist()
                )
                mem_indexes = [mem_candidates[i] for i in sampled] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")

        frame_indexes = mem_indexes + video_end[self.video_temporal_stride - 1 :: self.video_temporal_stride]
        return frame_indexes

    def get_action_bias_std(self, domain_name, stat_key):
        full_key = f"{domain_name}_{stat_key}"
        if full_key not in self.StatisticInfo:
            return None, None
        info = self.StatisticInfo[full_key]
        return (
            torch.tensor(info["mean"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(info["std"], dtype=torch.float32).unsqueeze(0),
        )

    def get_action(self, h5_file, slices, domain_name):
        delta_action = parse_h5(
            h5_file,
            slices=slices,
            delta_act_sidx=self.n_previous,
            action_space=self.action_space,
        )[1]
        delta_action = torch.FloatTensor(delta_action)

        with h5py.File(h5_file, "r") as fid:
            all_abs_gripper = np.array(fid["state/effector/position"], dtype=np.float32)
            all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
            all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

        pose_action = []
        for i in slices:
            pose_action.append(
                np.concatenate(
                    (
                        all_ends_p[i, 0],
                        all_ends_o[i, 0],
                        all_abs_gripper[i, :1],
                        all_ends_p[i, 1],
                        all_ends_o[i, 1],
                        all_abs_gripper[i, 1:],
                    ),
                    axis=0,
                )
            )
        action = torch.FloatTensor(np.stack(pose_action, axis=0))

        if self.normalize_actions:
            delta_mean, delta_std = self.get_action_bias_std(domain_name, f"delta_{self.action_space}")
            if delta_mean is not None and delta_std is not None:
                delta_action = (delta_action - delta_mean) / (delta_std + 1e-6)

        return action, delta_action

    def seek_mp4(self, sample_root, cam_name, slices):
        video_reader = VideoFileClip(os.path.join(sample_root, f"{cam_name}_color.mp4"))
        fps = video_reader.fps
        video = []
        for idx in slices:
            video.append(video_reader.get_frame(float(idx) / fps))
        video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
        video = video.float() / 255.0
        video_reader.close()
        return video

    def get_intrin_and_extrin(self, sample_root, cam_name, slices):
        with open(os.path.join(sample_root, f"{cam_name}_intrinsic_params.json"), "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0] = info["fx"]
        intrinsic[1, 1] = info["fy"]
        intrinsic[0, 2] = info["ppx"]
        intrinsic[1, 2] = info["ppy"]

        with open(os.path.join(sample_root, f"{cam_name}_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        c2ws = []
        w2cs = []
        for frame_idx in slices:
            frame_info = info[frame_idx]
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = torch.FloatTensor(frame_info["extrinsic"]["rotation_matrix"])
            c2w[:3, -1] = torch.FloatTensor(frame_info["extrinsic"]["translation_vector"])
            c2ws.append(c2w)
            w2cs.append(torch.linalg.inv(c2w))

        return intrinsic, torch.stack(c2ws, dim=0), torch.stack(w2cs, dim=0)

    def transform_video(self, video, intrinsic):
        c, _, h, w = video.shape
        if self.random_crop:
            h_start, w_start, h_crop, w_crop = gen_crop_config(video)
            video = video[:, :, h_start : h_start + h_crop, w_start : w_start + w_crop]
            intrinsic = intrin_crop_transform(intrinsic, h_start, w_start)
            h, w = h_crop, w_crop

        intrinsic = intrinsic_transform(intrinsic, (h, w), self.sample_size, self.preprocess)
        video = self.pixel_transforms_resize(video)
        return video, intrinsic

    def normalize_video(self, video):
        return self.pixel_transforms_norm(video.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

    def build_ray_maps(self, intrinsic, c2ws):
        intrinsics = intrinsic.unsqueeze(0)
        c2ws = c2ws.unsqueeze(0)
        rays_o, rays_d = get_ray_maps(
            intrinsics.unsqueeze(1).repeat(1, c2ws.shape[1], 1, 1).reshape(-1, 3, 3),
            c2ws.reshape(-1, 4, 4),
            self.sample_size[0],
            self.sample_size[1],
        )
        rays = torch.cat((rays_o, rays_d), dim=-1).reshape(
            intrinsics.shape[0], c2ws.shape[1], rays_o.shape[1], rays_o.shape[2], -1
        )
        return rays.permute(4, 0, 1, 2, 3).float()

    def get_caption(self, sample_name):
        if self.use_unified_prompt:
            return self.unified_prompt
        return f"world model prediction for {sample_name}"

    def get_batch(self, idx):
        sample_root, domain_name, domain_id, sample_name = self.dataset[idx]
        cam_name = self.valid_cam[0]
        h5_file = os.path.join(sample_root, "proprio_stats.h5")

        total_frames = self.get_total_timesteps(sample_root, cam_name)
        sep = random.randint(self.min_sep, self.max_sep)
        frame_indexes = self.get_frame_indexes(total_frames, sep=sep)

        action, delta_action = self.get_action(h5_file, frame_indexes, domain_name)
        intrinsic, c2ws, w2cs = self.get_intrin_and_extrin(sample_root, cam_name, frame_indexes)

        video = self.seek_mp4(sample_root, cam_name, frame_indexes)
        video, intrinsic = self.transform_video(video, intrinsic)
        video = self.normalize_video(video).unsqueeze(1)

        radius_gen_func = simple_radius_gen_func if self.radius_mode == "simple" else None
        traj = get_traj_maps(
            action,
            w2cs.unsqueeze(0),
            c2ws.unsqueeze(0),
            intrinsic.unsqueeze(0),
            self.sample_size,
            radius_gen_func=radius_gen_func,
        )
        traj = traj * 2.0 - 1.0
        ray = self.build_ray_maps(intrinsic, c2ws)
        cond_to_concat = torch.cat((traj, ray), dim=0)

        state = action[self.n_previous - 1 : self.n_previous]
        caption = self.get_caption(sample_name)

        sample = dict(
            video=video,
            caption=caption,
            actions=action,
            state=state,
            action=action,
            delta_action=delta_action,
            intrinsic=intrinsic.unsqueeze(0),
            extrinsic=c2ws.unsqueeze(0),
            traj=traj,
            ray=ray,
            cond_to_concat=cond_to_concat,
            cond_id=-(self.n_previous + self.chunk),
            path=sample_root,
            domain_id=domain_id,
            fps=self.fps,
        )
        return sample

    def __getitem__(self, idx):
        if self.fix_epiidx is not None:
            idx = self.fix_epiidx

        while True:
            try:
                return self.get_batch(idx)
            except Exception:
                traceback.print_exc()
                idx = random.randint(0, self.length - 1)
