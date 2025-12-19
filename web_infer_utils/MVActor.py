import argparse
import os
import pdb
import sys
import time

import cv2
from yaml import Dumper, Loader, dump, load

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

root_dir = os.path.dirname(os.path.dirname(root_dir))
sys.path.insert(0, root_dir)

sys.path.insert(0, current_dir)

import os
import pdb
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from data.utils.statistics import StatisticInfo

from einops import rearrange

from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model

from utils import init_logging, import_custom_class, save_video

from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video


from PIL import Image
from torch.utils.data import DataLoader
from decord import VideoReader
from typing import Any, Dict, List

import json

class MVActor:
    def __init__(
        self,
        config_file,
        transformer_file,
        threshold=None,
        n_prev=4,
        action_dim=14,
        gripper_dim=1,
        domain_name="agibotworld",
        load_weights=True,
        num_inference_steps=None,
        device = torch.device("cuda:0"),
        dtype = torch.bfloat16,
        norm_type = "meanstd",
    ):
        self.device = device
        self.dtype = dtype
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim

        self.transformer_file = transformer_file

        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)

        self.action_type = args.data["train"]["action_type"]
        self.action_space = args.data["train"]["action_space"]
        self.add_state = getattr(args, "add_state", False)

        print(self.action_type, self.action_space)
        if self.add_state:
            print("Add state")

        if num_inference_steps is not None:
            args.num_inference_steps = num_inference_steps

        self.args = args


        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.prepare_models()

        state_statistic_name = domain_name + "_" + "state" + "_" + self.action_space
        if self.action_type == "delta":
            action_statistic_name = domain_name + "_" + "delta" + "_" + self.action_space
        else:
            action_statistic_name = domain_name + "_" + self.action_space

        self.StatisticInfo = StatisticInfo
        if self.args.data['val'].get('stat_file', None) is not None:
            with open(self.args.data['val']['stat_file'], "r") as f:
                self.StatisticInfo = json.load(f)

        self.norm_type = norm_type
        assert(self.norm_type in ["minmax", "meanstd"])

        if self.norm_type == "meanstd":
            ### (1,1,C)
            self.act_mean = torch.tensor(self.StatisticInfo[action_statistic_name]["mean"]).unsqueeze(0).unsqueeze(0)
            self.act_std = torch.tensor(self.StatisticInfo[action_statistic_name]["std"]).unsqueeze(0).unsqueeze(0)+1e-6
            ### (C, )
            self.sta_mean = np.array(self.StatisticInfo[state_statistic_name]["mean"])
            self.sta_std = np.array(self.StatisticInfo[state_statistic_name]["std"])+1e-6
            if self.args.data['val'].get('valid_act_dim', None) is not None:
                self.valid_act_dim = self.args.data['val']['valid_act_dim']
                self.act_mean = self.act_mean[:,:,:self.valid_act_dim]
                self.act_std = self.act_std[:,:,:self.valid_act_dim]
            if self.args.data['val'].get('valid_sta_dim', None) is not None:
                self.valid_sta_dim = self.args.data['val']['valid_sta_dim']
                self.sta_mean = self.sta_mean[:self.valid_sta_dim]
                self.sta_std = self.sta_std[:self.valid_sta_dim]
        elif self.norm_type == "minmax":
            ### (1,1,C)
            self.act_min = torch.tensor(self.StatisticInfo[action_statistic_name]["q01"]).unsqueeze(0).unsqueeze(0)
            self.act_max = torch.tensor(self.StatisticInfo[action_statistic_name]["q99"]).unsqueeze(0).unsqueeze(0)
            ### (C, )
            self.sta_min = np.array(self.StatisticInfo[state_statistic_name]["q01"])
            self.sta_max = np.array(self.StatisticInfo[state_statistic_name]["q99"])
            if self.args.data['val'].get('valid_act_dim', None) is not None:
                self.valid_act_dim = self.args.data['val']['valid_act_dim']
                self.act_min = self.act_min[:,:,:self.valid_act_dim]
                self.act_max = self.act_max[:,:,:self.valid_act_dim]
            if self.args.data['val'].get('valid_sta_dim', None) is not None:
                self.valid_sta_dim = self.args.data['val']['valid_sta_dim']
                self.sta_min = self.sta_min[:self.valid_sta_dim]
                self.sta_max = self.sta_max[:self.valid_sta_dim]

        
        self.obs = []
        self.buffer = []
        self.prev_chunk_buffer = None
        self.n_prev = n_prev

        if threshold is None:
            self.threshold = args.threshold
        else:
            self.threshold = threshold
            
        self.num_inference_steps = args.num_inference_steps

        self.reset()


    def prepare_models(self,):

        print("Initializing models")
        device = self.device
        dtype = self.dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')

        self.chunk = (self.args.data["train"]["chunk"]-1)//self.TEMPORAL_DOWN_RATIO+1
        self.action_chunk = self.args.data["train"]["action_chunk"]
        self.resize = self.args.data["train"]["sample_size"]

        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=torch.bfloat16)
    
        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.transformer_file,
            load_weights=True,
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        self.pipeline = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

    @torch.no_grad()
    def play(self, obs, prompt, idx=0, num_inference_steps=None, execution_step=1, state=None, state_zeropadding=[0,0], ndim_action=None):
        """
        obs: One of the followings
            1. torch.tensor of shape: {v, 3, h, w}, ranging from -1 to 1
            2. np.array of shape: {v, h, w, 3}, ranging from 0 to 255 (np.uin8)
        prompt: task description
        execution_step: excution step of the past play
        return the raw action
        """
        assert execution_step >= 1 and execution_step <= 100, "execution_step should be in [1, 100]"


        if obs.dtype == np.uint8:
            ### obs / 255 * 2 - 1
            obs = obs.astype(np.float32) / 127.5 - 1
            obs = np.transpose(obs, (0,3,1,2))

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)

        if ndim_action is None:
            ndim_action = self.action_dim

        v, c, h, w = obs.shape
        obs = obs.to(self.device, dtype=self.dtype)


        if self.add_state:
            
            if self.norm_type == "meanstd":
                ### C -> 1,C
                sta_mean = np.concatenate([np.zeros(state_zeropadding[0]), self.sta_mean, np.zeros(state_zeropadding[1])])
                sta_std = np.concatenate([np.ones(state_zeropadding[0]), self.sta_std, np.ones(state_zeropadding[1])])
                normed_state = np.expand_dims((state-sta_mean)/sta_std, axis=0)
            
            elif self.norm_type == "minmax":
                ### C -> 1,C
                sta_min = np.concatenate([np.zeros(state_zeropadding[0]), self.sta_min, np.zeros(state_zeropadding[1])])
                sta_max = np.concatenate([np.ones(state_zeropadding[0])-1e-6, self.sta_max, np.ones(state_zeropadding[1])-1e-6])
                normed_state = np.expand_dims((state-sta_min)/(sta_max-sta_min+1e-6), axis=0)
                normed_state = normed_state * 2 -1
                if state_zeropadding[0] > 0:
                    normed_state[:,:state_zeropadding[0]] *= 0
                if state_zeropadding[1] > 0:
                    normed_state[:,-state_zeropadding[1]:] *= 0

            history_action_state = torch.from_numpy(normed_state).to(self.device, dtype=self.dtype)
            ### 1,1,C
            history_action_state = history_action_state.unsqueeze(dim=0)
            assert(len(history_action_state.shape) == 3 and history_action_state.shape[-1]==self.action_dim)

        else:
            history_action_state = None

        if not self.obs:
            self.obs = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.buffer = [self.obs[-1]]
        else:
            if execution_step == 0:
                return self.action_buffer
            self.count += execution_step
            if self.count >= self.threshold:
                self.count = 0
                self.obs.pop(0)
                self.obs[-1] = self.buffer[0]
                self.obs.append(obs)
            else:
                self.obs[-1] = obs
            self.buffer = [self.obs[-1]]

        obs_tensor = torch.stack(self.obs, dim=1)  # from v, c, h, w to v, t, c, h, w
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")  # c,t,h,w
        obs_tensor = obs_tensor.unsqueeze(0)  # b,c,v,t,h,w
        obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

        negative_prompt = ''

        pred_all = self.pipeline.infer(
            image=obs_tensor,
            n_prev=self.n_prev,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps if num_inference_steps is not None else self.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0, ### close CFG for action-prediction
            height=h,
            width=w,
            n_view=v,
            return_action=True,
            return_video=False,
            chunk=self.chunk,
            action_chunk=self.action_chunk,
            history_action_state=history_action_state if self.add_state else None,
            noise_seed=42,
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            n_chunk=1,
            return_dict=False,
            action_dim=self.args.diffusion_model["config"]["action_in_channels"],
        )[0]

        ### 1,t,c
        actions_pred = pred_all["action"].detach().cpu()[:,:,:ndim_action]

        ### original state: 1,1,C
        state = torch.from_numpy(state).unsqueeze(dim=0).unsqueeze(dim=0)

        ### for dual-arm
        gripper_dim = self.gripper_dim
        arm_dim = (self.action_dim - 2*self.gripper_dim)//2

        if self.action_type == "absolute":
            ### train:
            ### abs_act = norm(act)
            ### infer:
            ### denorm(abs_act)
            if self.norm_type == "meanstd":
                final_actions_pred = actions_pred[:, :execution_step, :] * self.act_std + self.act_mean
            elif self.norm_type == "minmax":
                final_actions_pred = (actions_pred + 1)/2
                final_actions_pred = final_actions_pred[:, :execution_step, :] * (self.act_max - self.act_min + 1e-6) + self.act_min

        elif self.action_type == "delta":
            ### train:
            ### delta_act = act_t - act_{t-1}
            ### delta_act = norm(delta_act)
            ### infer:
            ### cumsum(denorm(output)
            if self.norm_type == "meanstd":
                final_actions_pred = actions_pred[:, :execution_step, :] * self.act_std + self.act_mean
            elif self.norm_type == "minmax":
                final_actions_pred = (actions_pred + 1)/2
                final_actions_pred = final_actions_pred[:, :execution_step, :] * (self.act_max - self.act_min + 1e-6) + self.act_min
            ### left arm
            final_actions_pred[:, :, :arm_dim] = torch.cumsum(final_actions_pred[:, :, :arm_dim], dim=1) + state[:, :, :arm_dim]
            ### right arm
            final_actions_pred[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim] = torch.cumsum(final_actions_pred[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim], dim=1) + state[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim]

        elif self.action_type == "relative":
            ### train:
            ### rel_act = norm(act) - norm(state)
            ### infer:
            ### denorm(output + norm(state))

            final_actions_pred = actions_pred[:, :execution_step, :]
            if self.norm_type == "meanstd":
                sta_mean = torch.from_numpy(self.sta_mean).unsqueeze(dim=0).unsqueeze(dim=0)
                sta_std = torch.from_numpy(self.sta_std).unsqueeze(dim=0).unsqueeze(dim=0)
                normed_state = (state[:,:,:ndim_action]-sta_mean[:,:,:ndim_action])/sta_std[:,:,:ndim_action]
            elif self.norm_type == "minmax":
                sta_min = torch.from_numpy(self.sta_min).unsqueeze(dim=0).unsqueeze(dim=0)
                sta_max = torch.from_numpy(self.sta_max).unsqueeze(dim=0).unsqueeze(dim=0)
                normed_state = (state[:,:,:ndim_action]-sta_min[:,:,:ndim_action])/(sta_max[:,:,:ndim_action]-sta_min[:,:,:ndim_action]+1e-6)
            final_actions_pred[:, :, :arm_dim] = final_actions_pred[:, :, :arm_dim] + normed_state[:, :, :arm_dim]
            final_actions_pred[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim] = final_actions_pred[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim] + normed_state[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim]
            
            if self.norm_type == "meanstd":
                final_actions_pred = final_actions_pred * self.act_std + self.act_mean
            elif self.norm_type == "minmax":
                final_actions_pred = (final_actions_pred + 1)/2
                final_actions_pred = final_actions_pred * (self.act_max - self.act_min + 1e-6) + self.act_min

        else:
            raise NotImplementedError

        ### T, C
        final_actions_pred = final_actions_pred[0]

        self.action_buffer = final_actions_pred.clone()

        return final_actions_pred.data.cpu().numpy()


    def reset(self):
        self.obs = []
        self.buffer = []
        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=torch.bfloat16)
        self.cur_step = 0


    def change_step(self, pb_pred):
        return pb_pred.median() >= 0.99
