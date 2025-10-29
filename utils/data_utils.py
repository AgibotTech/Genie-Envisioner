from typing import Dict, List, Optional, Union
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

import torchvision
import torchvision.transforms as transforms



def _encode_prompt(
    tokenizer,
    text_encoder,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length,
) -> torch.Tensor:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


def read_img(img_file, target_shape=(512,384)):
    """
    traget_shape: tuple of width, height
    """
    image = Image.open(img_file)
    image = image.resize(target_shape)
    image_array = np.array(image)
    if image_array.shape[0] != 3:
        image_array = np.transpose(image_array, (2,0,1))
    image = torch.from_numpy(image_array)
    image = image.float()/255.0 * 2 - 1
    return image


@torch.no_grad()
def get_text_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


@torch.no_grad()
def get_latents(vae,
                mem: torch.Tensor,
                video: torch.Tensor,
                patch_size: int = 1,
                patch_size_t: int = 1,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                generator: Optional[torch.Generator] = None):
    """
    mem: (b v) c m h w [-1,1]
    video: (b v) c f h w [-1,1]
    Returns:
        mem_latents: (b v m) (1 h_latent w_latent) c
        video_latents: (b v) (f_latent h_latent w_latent) c
    """

    device = device or vae.device

    video_latents = vae.encode(video).latent_dist.sample(generator=generator)
    video_latents = video_latents.to(dtype=dtype)
    video_latents = _normalize_latents(video_latents, vae.latents_mean, vae.latents_std)
    video_latents = _pack_latents(video_latents, patch_size, patch_size_t)

    ### seperately encode memory frmaes since memory frames are randomly or uniformly sampled
    mem = rearrange(mem, 'b c m h w -> (b m) c h w').unsqueeze(2)
    mem_latents = vae.encode(mem).latent_dist.sample(generator=generator)
    mem_latents = mem_latents.to(dtype=dtype)
    mem_latents = _normalize_latents(mem_latents, vae.latents_mean, vae.latents_std)
    mem_latents = _pack_latents(mem_latents, patch_size, patch_size_t)

    return mem_latents, video_latents


@torch.no_grad()
def decode_latents(vae, latents, decode_timestep=0.0, decode_noise_scale=None, dtype=torch.bfloat16, generator=None,):
    """Input shape b,c,t,h,w, return latents shape b,c,t,h,w ranging from -1 to 1"""
    batch_size = latents.shape[0]
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor, reverse=True)
    latents = latents.to(dtype)

    if not vae.config.timestep_conditioning:
        timestep = None
    else:
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        if not isinstance(decode_timestep, list):
            decode_timestep = [decode_timestep] * batch_size
        if decode_noise_scale is None:
            decode_noise_scale = decode_timestep
        elif not isinstance(decode_noise_scale, list):
            decode_noise_scale = [decode_noise_scale] * batch_size

        timestep = torch.tensor(decode_timestep, device=latents.device, dtype=latents.dtype)
        decode_noise_scale = torch.tensor(decode_noise_scale, device=latents.device, dtype=latents.dtype)[
            :, None, None, None, None
        ]
        latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

    video = vae.decode(latents, temb=timestep, return_dict=False)[0]

    return video.clamp(-1, 1)



def prepare_latents(
    vae,
    image_or_video: torch.Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
) -> torch.Tensor:
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents.to(dtype=dtype)
        _, _, num_frames, height, width = latents.shape
        latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)
        latents = _pack_latents(latents, patch_size, patch_size_t)
        return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        _, _, num_frames, height, width = h.shape

        # TODO(aryan): This is very stupid that we might possibly be storing the latents_mean and latents_std in every file
        # if precomputation is enabled. We should probably have a single file where re-usable properties like this are stored
        # so as to reduce the disk memory requirements of the precomputed files.
        return {
            "latents": h,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "latents_mean": vae.latents_mean,
            "latents_std": vae.latents_std,
        }


def post_latent_preparation(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    latents = _normalize_latents(latents, latents_mean, latents_std)
    latents = _pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}



def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def gen_noise_from_condition_frame_latent(
    condition_frame_latent, latent_num_frames,
    latent_height=12, latent_width=16,
    generator=None, noise_to_condition_frames=0.05
):

    """
    To train the model for memory-frames conditioning,
    we occasionally set the timestep of the tokens
    belonging to the condition video frames to a small
    random value and noise these tokens to the corresponding
    level. The model quickly learns to utilize
    this new information (when provided) as a conditioning signal
    
    condition_frame_latent: (b v) c m h w

    """

    mem_size = condition_frame_latent.shape[2]
    num_channels_latents = condition_frame_latent.shape[1] # 128
    batch_size = condition_frame_latent.size(0)   # bv
    # latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

    shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
    mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)

    init_latents = condition_frame_latent[:,:,:1].repeat(1, 1, latent_num_frames, 1, 1)
    init_latents[:,:,:mem_size] = condition_frame_latent
    conditioning_mask = torch.zeros(mask_shape, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    conditioning_mask[:, :, :mem_size] = 1.0

    # similar to conditioning mask but useful to timesteps
    cond_indicator = torch.zeros((1, 1, latent_num_frames, 1, 1), device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    cond_indicator[:, :, :mem_size] = 1.0

    rand_noise_ff = random.random() * noise_to_condition_frames

    first_frame_mask = conditioning_mask.clone()
    first_frame_mask[:, :, :mem_size] = 1.0 - rand_noise_ff

    noise = randn_tensor(shape, generator=generator, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    latents = init_latents * first_frame_mask + noise * (1 - first_frame_mask)

    conditioning_mask = _pack_latents(conditioning_mask).squeeze(-1)
    cond_indicator = _pack_latents(cond_indicator).squeeze(-1)

    latents = _pack_latents(latents)

    # pack_latents: b c f h w -> b (f h w) c
    # unpack_latents: b (f h w) c -> b c f h w

    return latents, conditioning_mask, cond_indicator


def apply_color_jitter_to_video(tensor, jitter=None):
    """
    inputs:
        tensor (torch.Tensor): {b,c,t,h,w}, range [-1, 1]
        jitter (ColorJitter) : torchvision.transforms.ColorJitter
    output:
        augmented video tensor
    """
    B, C, T, H, W = tensor.shape
    assert C == 3
    if jitter is None:
        # jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1)
    tensor = (tensor + 1.0) / 2.0
    tensor = rearrange(tensor, 'b c t h w -> b t c h w')
    for b in range(B):
        tensor[b, :, :] = jitter(tensor[b, :, :])
    tensor = rearrange(tensor, 'b t c h w -> b c t h w')
    tensor = tensor * 2.0 - 1.0
    return tensor


@torch.no_grad()
def prepare_ray_map(intrinsic, c2w, H, W):
    """
    inputs:
        intrinsic: b,3,3
        c2w:       b,4,4
    outputs:
        rays:      b, H, W, 3 and b, H, W, 3
    """
    batch_size = intrinsic.shape[0]
    fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(batch_size,1,1)
    j = j.unsqueeze(0).repeat(batch_size,1,1)
    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)
    rays_o = c2w[:, :3,-1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
    viewdir = rays_d/torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, viewdir


@torch.no_grad()
def get_ray_maps(intrinsic, extrinsic, h, w, n_view, t, device, dtype):
    """
    inputs:
        intrinsic: {b,v,t,3,3}
        extrinsic: {b,v,t,4,4}
    output:
        rays:      {b,c,v,t,h,w}
    """
    intrinsics = rearrange(intrinsic, "b v t i j -> (b v t) i j")
    extrinsics = rearrange(extrinsic, "b v t i j -> (b v t) i j")
    rays_o, rays_d = prepare_ray_map(intrinsics, extrinsics, H=h, W=w)
    ### (b v t) h w c -> b c v t h w
    rays = rearrange(torch.cat((rays_o, rays_d), dim=-1), "(b v t) h w c -> b c v t h w", v=n_view, t=t)
    rays = rays.to(device, dtype=dtype).contiguous()
    return rays

