
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image

# from data.utils.domain_table import DomainTable
from data.utils.statistics import StatisticInfo
# from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform


def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_recap_file = None,
        step_recap_file = None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        ignore_seek = False,
        train_dataset=True,
        action_key = "action",
        state_key = "observation.state",
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
        stat_file = None,
        extra_parquet_index = False,
        valid_act_dim = None,
        valid_sta_dim = None,
    ):
        """
        data_roots:              directory of LeRoBot dataset
        domains:                 name of your dataset, used to index different statistics
        task_recap_file:         json file of augmented task captions:
                                 {
                                    'ori_task_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_task_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        step_recap_file:         json file of augmented step captions:
                                 {
                                    'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        sample_size:             video frame size
        sample_n_frames:         number of frames used to randomly or uniformly select memories
        preprocess:              frame preprocessing strategy, resize or center_crop_resize
        valid_cam:               list of cam names 
        chunk:                   number of video frames to predict
        action_chunk:            number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:              number of memory frames
        previous_pick_mode:      how to select memories
        random_crop:             randomly crop images
        dataset_info_cache_path: path to save dataset meta information cache
        action_type:             action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:            joint or eef, which is used to determinate the statistics values only in this dataset
        ignore_seek:             if True, load the first furture frame only
        use_unified_prompt:      if set all prompt the same
        unified_prompt:          unified prompt
        fix_epiidx:              used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                used in validation stage only, set start index to fix_sidx
        fix_mem_idx:             used in validation stage only, set memory indexes to fix_mem_idx
        stat_file:               used to specific statistics
        extra_parquet_index:     when extra_parquet_index=True, the shape of the action/state arrary saved in .parquet files should be [T,1,C]; when extra_parquet_index=False, the shape of the action/state arrary saved in .parquet files should be [T,C].
        valid_act_dim:           when valid_act_dim is not None, only the first $valid_act_dim dimenssions of actions will be used.
        valid_sta_dim:           when valid_sta_dim is not None, only the first $valid_sta_dim dimenssions of actions will be used.

        """
        
        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space



        self.action_key = action_key
        self.state_key = state_key
        self.extra_parquet_index = extra_parquet_index
        self.valid_act_dim = valid_act_dim
        self.valid_sta_dim = valid_sta_dim

        self.random_crop = random_crop
        
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots
        self.dataset = []
        
        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            # construct the dataset_info
            for _data_root, _domain_name in zip(self.data_roots, domains):

                print(f"Loading {_domain_name} data from {_data_root}")
                
                # into the meta folder
                if os.path.exists(os.path.join(_data_root, _domain_name, "meta", "tasks.jsonl")):
                    meta_folder = os.path.join(_data_root, _domain_name, "meta")
                    data_folder = os.path.join(_data_root, _domain_name, "data")
                    video_folder = os.path.join(_data_root, _domain_name, "videos")
                else:
                    meta_folder = os.path.join(_data_root, "meta")
                    data_folder = os.path.join(_data_root, "data")
                    video_folder = os.path.join(_data_root, "videos")
                    
                tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
                task_index_task_str = load_jsonl(tasks_jsonl)
                task_index_task_str_dict = {}
                for item in task_index_task_str:
                    task_index_task_str_dict[item['task_index']] = item['task']


                with open(os.path.join(meta_folder, "info.json"), "r") as f:
                    metainfo = json.load(f)
                    total_chunks = metainfo["total_chunks"]
                    chunks_size = metainfo["chunks_size"]

                episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
                epiosdes_data = load_jsonl(episodes_jsonl) # episode_index  tasks  length


                for episode_data in tqdm(epiosdes_data):

                    episode_index = episode_data['episode_index']
                    tasks = episode_data['tasks']
                    if len(tasks) > 1:
                        task = random.choice(tasks)
                    else:
                        task = tasks[0]
                    length = episode_data['length']
                    
                    episode_chunk = int(episode_index//chunks_size)

                    parquet_path = os.path.join(data_folder, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet")
                    if not os.path.exists(parquet_path):
                        zero_rank_print(f"parquet file not found: {parquet_path}")
                        continue

                    video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                    
                    info = [
                        video_path,
                        None, # no need for camera_info
                        parquet_path,
                        _domain_name, "", # DomainTable[_domain_name],
                        None, task, # no task_info
                        length,
                    ]
                    self.dataset.append(info)

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
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = int(n_previous)
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = int(self.sample_n_frames - self.chunk)
            self.previous_pick_mode = 'uniform'

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

        self.ignore_seek = ignore_seek

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
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames-1).tolist()
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames-1).tolist()
            fix_mem_idx = np.clip(self.fix_mem_idx, a_min=0, a_max=total_frames-1).tolist()
            return fix_mem_idx + frame_indexes, fix_mem_idx + action_indexes

        chunk_end = random.randint(self.action_chunk, total_frames+self.action_chunk)
        indexes = np.array(list(range(max(-100, chunk_end-self.sample_n_frames), chunk_end)))
        indexes = np.clip(indexes, a_min=0, a_max=total_frames-1).tolist()
        video_end = indexes[-self.action_chunk:]
        # mem_candidates = [
        #     indexes[int(i)] for i in range(0, self.sample_n_frames-self.action_chunk-1)
        # ]
        mem_candidates = indexes[:-self.action_chunk]
        if len(mem_candidates)<self.n_previous-1:
            mem_candidates = [1,]*(self.n_previous-1) + mem_candidates

        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")       
        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride-1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end

        return frame_indexes, action_indexes


    def get_action_bias_std(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['mean']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['std']).unsqueeze(0)+1e-6


    def get_action_q01_q99(self, domain_name):
        return torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['q01']).unsqueeze(0), torch.tensor(self.StatisticInfo[domain_name+"_"+self.action_space]['q99']).unsqueeze(0)


    def seek_mp4(self, video_path, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            video_reader = VideoFileClip(video_path.format(cam_name))
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
                if intrinsics is not None:
                    intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                
                h, w = h_crop, w_crop
            if intrinsics is not None:
                intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
                new_intrinsics.append(intrinsic)
                
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        if len(new_intrinsics) > 0:
            new_intrinsics = torch.stack(new_intrinsics, dim=0)
        else:
            new_intrinsics = None
        return new_videos, None


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



    def get_batch(self, idx):
        
        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        # domain_id = self.dataset[idx][4]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames, )
        
        data = pd.read_parquet(parquet_path)


        action_min, action_max = self.get_action_q01_q99(domain_name)
        state_min, state_max = self.get_action_q01_q99(domain_name + "_state")
        
        ###
        ### example data
        ### data[self.action_key] with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        ### data[self.state_key]  with the shape of T*C: [[1.0, 1.0, 1.0, ...], ...]
        try:
            if self.extra_parquet_index:
                action = np.stack([data[self.action_key][i][0] for i in range(data[self.action_key].shape[0])])
                state = np.stack([data[self.state_key][i][0] for i in range(data[self.state_key].shape[0])])
            else:
                action = np.stack([data[self.action_key][i] for i in range(data[self.action_key].shape[0])])
                state = np.stack([data[self.state_key][i] for i in range(data[self.state_key].shape[0])])
        except:
            raise ValueError("We currently only support action and state data with the shape of T*C!")

        action = action.astype(np.float32)
        state = state.astype(np.float32)

        state = torch.FloatTensor(state)[indexes]

        if self.valid_act_dim is not None:
            action = action[:, :self.valid_act_dim]
            action_min = action_min[:, :self.valid_act_dim]
            action_max = action_max[:, :self.valid_act_dim]

        if self.valid_sta_dim is not None:
            state = state[:, :self.valid_sta_dim]
            state_min = state_min[:, :self.valid_sta_dim]
            state_max = state_max[:, :self.valid_sta_dim]

        state = (state - state_min) / (state_max - state_min + 1e-6)
        state = state * 2.0 - 1.0

        assert(self.action_type == "absolute")

        ### act = norm(act)
        action = action[indexes].astype(np.float32)
        action = torch.FloatTensor(action)
        action = (action - action_min) / (action_max - action_min + 1e-6)
        action = action * 2.0 - 1.0

        ori_act_dim = action.shape[1]

        action = torch.cat((action, state), dim=1)
        state = torch.cat((torch.zeros([1,ori_act_dim]), state[self.n_previous-1:self.n_previous]), dim=1)

        # videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)

        video_list = []
        for cam in self.valid_cam:
            cam_img_bytes = data[cam].to_list()
            video = []
            for index in vid_indexes:
                img = Image.open(io.BytesIO(cam_img_bytes[index]["bytes"]))
                video.append(img)
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float()/255.
            video_list.append(video)
        videos = torch.stack(video_list, dim=1) 
        videos, _ = self.transform_video(
            videos, specific_transforms_resize, None, sample_size
        )
        videos = self.normalize_video(videos, specific_transforms_norm)

        return videos, action, caption, state



    def __len__(self):
        return self.length



    def __getitem__(self, idx):        
        
        # video, actions, caption, state = self.get_batch(idx)

        if self.fix_epiidx is not None:
            video, actions, caption, state = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state = self.get_batch(idx)
                    break
                except:
                    ### print error information to debug
                    traceback.print_exc()
                    ### 
                    idx = random.randint(0, self.length-1)
                    
        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        return sample

