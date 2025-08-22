import os, random, math
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Any, Dict, List
import argparse

from datetime import datetime, timedelta
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging
import cv2



# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video


def load_config(config_file):
    cd = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**cd)
    return args

def prepare_model(args, dtype=torch.bfloat16, device="cuda:0"):

    ### Load Tokenizer
    tokenizer_class = import_custom_class(
        args.tokenizer_class, getattr(args, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        args.textenc_class, getattr(args, "textenc_class_path", "transformers")
    )
    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        args.pretrained_model_name_or_path if not hasattr(args, "tokenizer_pretrained_model_name_or_path") else args.tokenizer_pretrained_model_name_or_path,
        load_weights=args.load_weights
    )
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype).eval()

    ### Load VAE
    vae_class = import_custom_class(
        args.vae_class, getattr(args, "vae_class_path", "transformers")
    )
    if getattr(args, 'vae_path', False):
        vae = load_vae_models(vae_class, args.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
    if isinstance(vae.latents_mean, List):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, List):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if vae is not None:
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

    ### Load Diffusion Model
    diffusion_model_class = import_custom_class(
        args.diffusion_model_class, getattr(args, "diffusion_model_class_path", "transformers")
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=args.diffusion_model['model_path'],
        load_weights=args.load_weights and getattr(args, "load_diffusion_model_weights", True),
        **args.diffusion_model['config']
    ).to(device, dtype=dtype)
    total_params = count_model_parameters(diffusion_model)
    print(f'Total parameters for transfomer model:{total_params}')


    ### Load Diffuser Scheduler
    diffusion_scheduler_class = import_custom_class(
        args.diffusion_scheduler_class, getattr(args, "diffusion_scheduler_class_path", "diffusers")
    )
    if hasattr(args, "diffusion_scheduler_args"):
        scheduler = diffusion_scheduler_class(**args.diffusion_scheduler_args)
    else:
        scheduler = diffusion_scheduler_class()

    ### Import Inference Pipeline Class
    pipeline_class = import_custom_class(
        args.pipeline_class, getattr(args, "pipeline_class_path", "diffusers")
    )

    pipe = pipeline_class(
        scheduler, vae, text_encoder, tokenizer, diffusion_model
    )

    return tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe


def load_images(args, image_root, valid_cams, size=(256,192)):
    n_mem = args.data["train"]["n_previous"]
    mv_images = []
    for cam in valid_cams:
        images = []
        for i in range(n_mem):
            img = cv2.imread(os.path.join(image_root, cam, str(i)+".png"))[:,:,::-1]
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            img = torch.from_numpy(np.transpose(img, (2,0,1)))
            images.append(img)
        ### c,t,h,w
        images = torch.stack(images, dim=1)
        mv_images.append(images)
    ### v,c,t,h,w
    mv_images = torch.stack(mv_images, dim=0)
    return mv_images


def infer(
    config_file, image_root, prompt, save_path, n_chunk=1, normed_state=None,
    num_denois_steps=50, seed=42, device="cuda", default_fps=30
):

    args = load_config(config_file)

    if "action_chunk" in args.data["train"]:
        args.data['train']['chunk']
        video_fps = default_fps // (args.data['train']['action_chunk'] // args.data['train']['chunk'])
    else:
        video_fps = default_fps

    tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe = prepare_model(args, device=device)

    valid_cams = [_+"_color" for _ in args.data["train"]["valid_cam"]]

    obs = load_images(args, image_root, valid_cams, size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0]))

    v,c,t,h,w = obs.shape

    SPATIAL_DOWN_RATIO = vae.spatial_compression_ratio
    TEMPORAL_DOWN_RATIO = vae.temporal_compression_ratio
    
    # negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    negative_prompt = ''

    if normed_state is not None:
        ### 1, 1, c_act
        normed_state = torch.from_numpy(normed_state).unsqueeze(dim=0).unsqueeze(dim=0)

    preds = pipe.infer(
        image=obs.to(device),
        prompt=[prompt, ],
        negative_prompt=negative_prompt,
        num_inference_steps=num_denois_steps,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        height=h,
        width=w,
        n_view=v,
        guidance_scale=1.0,
        return_action=args.return_action,
        n_prev=args.data['train']['n_previous'],
        chunk=(args.data['train']['chunk']-1)//TEMPORAL_DOWN_RATIO+1,
        return_video=args.return_video,
        noise_seed=seed,
        action_chunk=args.data['train']['action_chunk'],
        history_action_state = normed_state,
        pixel_wise_timestep = args.pixel_wise_timestep,
        n_chunk=n_chunk,
    )[0]

    os.makedirs(save_path, exist_ok=True)

    if args.return_video:
        video = preds['video'].data.cpu()
        save_video(
            rearrange(video, '(b v) c t h w -> b c t h (v w)', v=v)[0],
            os.path.join(save_path, "video.mp4"),
            fps=video_fps
        )

    if args.return_action:
        ### shape: {action_chunk, c_act}
        pred_action = preds['action'].detach().cpu().float().squeeze(0).numpy()
        np.save(os.path.join(save_path, "action.npy"), pred_action)



def args_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--image_root', type=str, required=True, help='Path to observation images')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_txt_file', type=str, default=None)
    parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--state_path', type=str, default=None, help='Path to load state')
    parser.add_argument('--n_chunk', type=int, default=1, help='num of chunks to predict, used in inference stage only')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = args_parser()
    if args.state_path is not None:
        normed_state = np.load(state_path)
    else:
        normed_state = None
    
    if args.prompt is None:
        asser(args.prompt_txt_file is not None)
        with open(args.prompt_txt_file, "r") as f:
            args.prompt = f.readline().strip()
    
    infer(
        args.config_file, args.image_root, args.prompt, args.output_path,
        args.n_chunk, normed_state,
    )
