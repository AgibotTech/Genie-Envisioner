
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import traceback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from data.utils.domain_table import DomainTable
from data.utils.statistics import StatisticInfo
from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform


class AgiBotWorld(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_info_root,
        task_recap_file = None,
        step_recap_file = None,
        specific_tasks=None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['head', 'hand_left', 'hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=False,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        ignore_seek = False,
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
        stat_file = None,
    ):

        """
        data_roots:               directory of AgiBotWorld dataset
        task_info_root:           directory of dataset task info json files, like files in https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/tree/main/task_info
        task_recap_file:          json file of augmented task captions:
                                    {
                                        'ori_task_caption_1': ['new_caption_1', 'new_caption_2', ...],
                                        'ori_task_caption_2': ['new_caption_1', 'new_caption_2', ...],
                                    }
        step_recap_file:          json file of augmented step captions:
                                    {
                                        'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                        'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                    }
        specific_tasks:           list of selected task ids
        sample_size:              video frame size
        sample_n_frames:          number of frames used to randomly or uniformly select memories
        preprocess:               frame preprocessing strategy, resize or center_crop_resize
        valid_cam:                list of cam names 
        chunk:                    number of video frames to predict
        action_chunk:             number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:               number of memory frames
        previous_pick_mode:       how to select memories
        random_crop:              randomly crop images
        dataset_info_cache_path:  path to save dataset meta information cache
        action_type:              action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:             joint or eef
        ignore_seek:              if True, load the first furture frame only
        use_unified_prompt:       if set all input prompt the same
        unified_prompt:           unified prompt
        fix_epiidx:               used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                 used in validation stage only, set start index to fix_sidx
        fix_mem_idx:              used in validation stage only, set memory indexes to fix_mem_idx
        """

        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space

        self.random_crop = random_crop
        
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        self.data_roots = data_roots
        self.task_info_root = task_info_root
        self.dataset = []

        if specific_tasks is None and dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            for _data_root, _domain_name in zip(self.data_roots, domains):
                valid_tasks = os.listdir(os.path.join(_data_root, "observations"))
                valid_tasks.sort()
                for task in tqdm(valid_tasks):
                    if (specific_tasks is None) or (int(task) in specific_tasks):
                        if not os.path.exists(os.path.join(task_info_root, f"task_{task}.json")):
                            continue
                        zero_rank_print(f"preparing data info: task-{task}")
                        task_infos = dict()
                        with open(os.path.join(task_info_root, f"task_{task}.json"), "r") as f:
                            for info in json.load(f):
                                task_infos.update({str(info["episode_id"]): (info["label_info"], info["task_name"])})
                        episode_list = list(task_infos.keys())
                        episode_list.sort()
                        episode_list = [os.path.join(_data_root, "observations", task, str(_)) for _ in episode_list]
                        n_episode = len(episode_list)
                        zero_rank_print(f"{n_episode} episodes in task-{task}")

                        for episode in episode_list:
                            episode_id = os.path.basename(episode)
                            info = [
                                episode,
                                os.path.join(_data_root, "parameters", task, episode_id),
                                os.path.join(_data_root, "proprio_stats", task, episode_id),
                                _domain_name, DomainTable[_domain_name],
                                task_infos[episode_id][0], task_infos[episode_id][1]
                            ]
                            self.dataset.append(info)

        if specific_tasks is None:                
            if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
                zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
                with open(dataset_info_cache_path, "w") as f:
                    json.dump(self.dataset, f)


        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert(self.chunk * self.video_temporal_stride == self.action_chunk)

        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        elif preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        else:
            raise NotImplementedError
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = 'uniform'

        self.ignore_seek = ignore_seek

        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None

        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None

        self.use_unified_prompt = use_unified_prompt

        ### validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx


        ### load stat_file if provided
        self.StatisticInfo = StatisticInfo
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)

    def get_total_timesteps(self, data_root, cam_name):
        with open(os.path.join(data_root, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        total_frames = len(info)
        return total_frames


    def get_frame_indexes(self, total_frames, ):
        """
        select self.n_previous memory frames and self.action_chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-action_chunk} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames} to {end-action_chunk}
        """

        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx+self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            # print(self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes)
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes

        chunk_end = random.randint(self.action_chunk, total_frames+self.action_chunk)
        indexes = np.array(list(range(chunk_end-self.sample_n_frames, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames-1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = [
            indexes[int(i)] for i in range(0, self.sample_n_frames-self.action_chunk-1)
        ]
        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")       

        frame_indexes = mem_indexes + video_end[self.video_temporal_stride-1::self.video_temporal_stride]
        
        action_indexes = mem_indexes + video_end

        return frame_indexes, action_indexes


    def get_action_bias_std(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]['mean']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]['std']).unsqueeze(0)+1e-6


    def get_action(self, h5_file, slices, domain_name):
        """
        1. extract actions from .h5 files
        2. obatin End Effector Action / Delta Actoin / Relative Action:
           action (t, 14): {xyz, rpy,  gripper} * 2
        """
        
        action, delta_action = parse_h5(h5_file, slices=slices, delta_act_sidx=0, action_space=self.action_space)

        act_meanv, act_stdv = self.get_action_bias_std(domain_name)
        state = torch.FloatTensor(action[self.n_previous-1:self.n_previous])
        state = (state - act_meanv) / act_stdv

        if self.action_type == "absolute":
            action = torch.FloatTensor(action)
            action = (action - act_meanv) / act_stdv
            return action, state

        elif self.action_type == "delta":
            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name + "_delta")
            delta_action = torch.FloatTensor(delta_action)
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            return delta_action, state

        elif self.action_type == "relative":
            act_meanv, act_stdv = self.get_action_bias_std(domain_name)
            action = torch.FloatTensor(action)
            action = (action - act_meanv) / act_stdv
            rel_action = action - state
            return rel_action, state

        else:
            raise NotImplementedError


    def seek_mp4(self, video_root, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(os.path.join(video_root, "videos", cam_name+'_color.mp4'))
            fps = video_reader.fps
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(float(idx)/fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            video_reader.close()
            video_list.append(video)
        video_list = torch.stack(video_list, dim=1)
        return video_list


    def get_intrin_and_extrin(self, cam_name_list, data_root, slices,):
        """
        get the intrinsic (Vx3x3), c2ws (VxTx4x4) tensors
        """
        intrinsic_list = []
        c2ws_list = []
        for cam_name in cam_name_list:
            with open(os.path.join(data_root, "parameters", "camera", cam_name+"_intrinsic_params.json"), "r") as f:
                info = json.load(f)["intrinsic"]
            intrinsic = torch.eye(3, dtype=torch.float)
            intrinsic[0,0] = info["fx"]
            intrinsic[1,1] = info["fy"]
            intrinsic[0,2] = info["ppx"]
            intrinsic[1,2] = info["ppy"]
            intrinsic_list.append(intrinsic)

            with open(os.path.join(data_root, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json"), "r") as f:
                info = json.load(f)
            c2ws = []
            for _i in slices:
                _i_info = info[_i]
                c2w = torch.eye(4, dtype=torch.float)
                c2w[:3, :3] = torch.FloatTensor(_i_info["extrinsic"]["rotation_matrix"])
                c2w[:3, -1] = torch.FloatTensor(_i_info["extrinsic"]["translation_vector"])
                w2c = torch.linalg.inv(c2w)
                c2ws.append(c2w)
            c2ws = torch.stack(c2ws, dim=0)
            c2ws_list.append(c2ws)
        intrinsic_list = torch.stack(intrinsic_list, dim=0)
        c2ws_list = torch.stack(c2ws_list, dim=0)
        return intrinsic_list, c2ws_list


    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        c, v, t, h, w = videos.shape
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[:, iv]
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
                intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                h, w = h_crop, w_crop
            else:
                intrinsic = intrinsics[iv]
            intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
            video = specific_transforms_resize(video)
            new_videos.append(video)
            new_intrinsics.append(intrinsic)
        new_videos = torch.stack(new_videos, dim=1)
        new_intrinsics = torch.stack(new_intrinsics, dim=0)
        return new_videos, new_intrinsics


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,v,t,h,w)
        """
        c,v,t,h,w = video.shape
        video = specific_transforms_norm(video.permute(1,2,0,3,4).reshape(-1,c,h,w)).reshape(v,t,c,h,w).permute(2,0,1,3,4)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        # find = []
        for step_caption in step_captions:
            # if step_caption in find:
            #     continue
            # else:
            #     find.append(step_caption)
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption,[])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list,1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)

        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the "," ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption,[])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list,1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0,2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap


    def get_caption(self, label_info, task_caption):
        if self.use_unified_prompt:
            caption = self.unified_prompt
        else:
            step_captions = [_["action_text"] for _ in label_info["action_config"]]
            caption = self.get_long_recaption(step_captions, task_caption)
        return caption


    def get_batch(self, idx, debug=False):

        video_root = self.dataset[idx][0]
        caminfo_root = self.dataset[idx][1]
        h5_file = os.path.join(self.dataset[idx][2], "proprio_stats.h5")
        domain_name = self.dataset[idx][3]
        # domain_id = self.dataset[idx][4]
        label_info = self.dataset[idx][5]
        task_name = self.dataset[idx][6]

        total_frames = self.get_total_timesteps(caminfo_root, self.valid_cam[0])

        caption = self.get_caption(label_info, task_name)

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()

        vid_indexes, indexes = self.get_frame_indexes(total_frames, )

        action, state = self.get_action(h5_file, indexes, domain_name)
        
        intrinsics, c2ws = self.get_intrin_and_extrin(self.valid_cam, caminfo_root, vid_indexes)

        ### c, n_view, total_frames, h, w
        if self.ignore_seek:
            ### used in the action-training stage to avoid seek future frames
            vid_indexes = vid_indexes[:self.n_previous+1]
            end_frames = 1

        videos = self.seek_mp4(video_root, self.valid_cam, vid_indexes)
        videos, intrinsics = self.transform_video(
            videos, specific_transforms_resize, intrinsics, sample_size
        )
        videos = self.normalize_video(videos, specific_transforms_norm)

        return videos, video_root, intrinsics, c2ws, action, state, caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if self.fix_epiidx is not None:
            video, video_root, intrinsics, extrinsics, actions, state, caption = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, video_root, intrinsics, extrinsics, actions, state, caption = self.get_batch(idx)
                    break
                except:
                    ### print error information to debug
                    traceback.print_exc()
                    ### 
                    idx = random.randint(0, self.length-1)

        sample = dict(
            video=video,
            actions=actions,
            state=state,
            caption=caption,
        )
        return sample
