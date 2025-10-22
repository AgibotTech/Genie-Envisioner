# Copyright 2024 Lightricks and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import time
import numpy as np
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


from einops import rearrange
from utils.data_utils import gen_noise_from_condition_frame_latent

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CustomPipelineOutput(BaseOutput):
    r"""
    Output class for custom pipelines.
    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """
    frames: torch.Tensor


def cross_attn_judgement(timesteps):
    return timesteps>=0

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def get_end_id(latents, eps=0.001):
    r"""
    Check if we can stop video generation
    """
    b, c, t, h, w = latents.shape
    end_frame = torch.ones((b, c, 1, h, w)).to(latents.device, dtype=latents.dtype) * -1
    assert b == 1, "do not support multi-video AR generation"
    diff = torch.abs(latents - end_frame)
    mask = torch.mean(diff, dim=(0, 1, 3, 4)) < eps
    indices = torch.where(mask)[0]
    if indices.numel() == 0:
        return -1
    else:
        return indices[0].item()


def update_init_latents(init_latents, n_prev, new_latents):
    init_latents = torch.cat((init_latents, new_latents), dim=2)
    return init_latents[:, :, -n_prev:]


### Modified From diffusers/pipelines/ltx/pipeline_ltx.py
class CustomPipeline(DiffusionPipeline, FromSingleFileMixin):

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler_action=FlowMatchEulerDiscreteScheduler(),
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            scheduler_action=scheduler_action,
        )

        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio if hasattr(self, "vae") else 32
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio if hasattr(self, "vae") else 8

        self.transformer_spatial_patch_size = None
        self.transformer_temporal_patch_size = None
        if hasattr(self, "transformer"):
            if getattr(self.transformer, "unpack_in_forward", False):
                self.transformer_spatial_patch_size = 1
                self.transformer_temporal_patch_size = 1
            else:
                if hasattr(self.transformer.config, "patch_size"):
                    if isinstance(self.transformer.config.patch_size, (list, tuple)):
                        if len(self.transformer.config.patch_size) == 3:
                            assert(self.transformer.config.patch_size[1] ==  self.transformer.config.patch_size[2])
                            self.transformer_spatial_patch_size = self.transformer.config.patch_size[1]
                            self.transformer_temporal_patch_size = self.transformer.config.patch_size[0]
                        elif len(self.transformer.config.patch_size) == 2:
                            assert(self.transformer.config.patch_size[0] ==  self.transformer.config.patch_size[1])
                            self.transformer_spatial_patch_size = self.transformer.config.patch_size[0]
                    else:
                        self.transformer_spatial_patch_size = self.transformer.config.patch_size
                if self.transformer_temporal_patch_size is None:
                    if hasattr(self.transformer.config, "patch_size_t"):
                        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t
        if self.transformer_spatial_patch_size is None:
            self.transformer_spatial_patch_size = 1
        if self.transformer_temporal_patch_size is None:
            self.transformer_temporal_patch_size = 1


        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 128


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
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

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt with 256->128
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
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

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1 if latents is None else latents.size(2)

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            conditioning_mask = latents.new_zeros(shape)
            conditioning_mask[:, :, 0] = 1.0
            conditioning_mask = self._pack_latents(conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            return latents.to(device=device, dtype=dtype), conditioning_mask

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [retrieve_latents(self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)), generator[i]) for i in range(batch_size)]
        else:
            init_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), generator) for img in image]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)
        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = self._pack_latents(conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size).squeeze(
            -1
        )
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        return latents, conditioning_mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt


    def decode_to_frames(self, latents, dtype, generator, device, batch_size, decode_timestep, decode_noise_scale):
        latents = self._denormalize_latents(latents, self.vae.latents_mean, self.vae.latents_std, getattr(self.vae.config, "scaling_factor", 1.0))
        latents = latents.to(dtype)

        if not getattr(self.vae.config, "timestep_conditioning", False):
            timestep = None
        else:
            noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size

            timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[:, None, None, None, None]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        video = self.vae.decode(latents, temb=timestep, return_dict=False)[0]

        # return latents shape b,c,t,h,w ranging from -1 to 1
        return video.clamp(-1, 1)



    @torch.no_grad()
    def infer(
        self,
        image: PipelineImageInput = None,
        n_prev: int = 4,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 192,
        width: int = 256,
        chunk: int = 8,
        frame_rate: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 128,
        return_init=True,
        n_view: int = 3,
        return_action: bool = False,
        action_chunk: int = 57,
        action_dim: int = 14,
        return_video: bool = True,
        noise_seed: int = None,
        history_action_state: torch.Tensor = None,
        pixel_wise_timestep: bool = True,
        n_chunk: int = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `CustomPipelineOutput` instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`CustomPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CustomPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """


        if return_action:
            assert n_chunk==1, "action-inference pipeline only support single chunk prediction now"
        

        # pre-compute latent shape
        self.transformer.eval()
        latent_num_frames = n_prev + chunk
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        # 1. Check inputs. Raise error if not correct, skip for validation

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        preds = {}
        # 3. Prepare text embeddings
        if prompt_embeds is None:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            preds["text_embedding"] = {}
            preds["text_embedding"]["prompt_embeds"] = prompt_embeds
            preds["text_embedding"]["prompt_attention_mask"] = prompt_attention_mask
            preds["text_embedding"]["negative_prompt_embeds"] = negative_prompt_embeds
            preds["text_embedding"]["negative_prompt_attention_mask"] = negative_prompt_attention_mask

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)  # b,l,c ?
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if len(image.shape) == 4:  # in this case, a single image act as input
            image = image.unsqueeze(2)
        image = image.to(device=device, dtype=prompt_embeds.dtype)  # out_shape b, c, t, h, w, range(-1,1)

        video_list = None
        if return_init:
            video_list = image.clone()

        mem_size = image.shape[2]
        sample_mode = "argmax" if noise_seed else "sample"  # we use deterministic vae when noise is fixed

        image = rearrange(image, "bv c t h w -> (bv t) c h w").unsqueeze(2)
        init_latents = retrieve_latents(self.vae.encode(image), generator, sample_mode=sample_mode)
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        
        if mem_size == 1 and n_prev > 1:
            init_latents = init_latents.repeat(1, 1, n_prev, 1, 1)
        else:
            init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)

        if return_action:
            if noise_seed:
                action_generator = torch.Generator(device=device).manual_seed(noise_seed)
            else:
                action_generator = generator
            actions = randn_tensor((batch_size, action_chunk, action_dim), device=device, dtype=prompt_embeds.dtype, generator=action_generator)
        else:
            actions = None


        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )

        _, _ = retrieve_timesteps(
            self.scheduler_action,
            num_inference_steps,
            device,
            None,
            sigmas=sigmas,
            mu=mu,
        )

        if getattr(self.scheduler.config, "final_sigmas_type", None) == "sigma_min":
            print("Replace the last sigma (which is zero) with the minimum sigma value")
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions
        latent_frame_rate = frame_rate / self.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )


        # 7. Denoising loop
        if noise_seed:
            video_generator = torch.Generator(device=device).manual_seed(noise_seed)
        else:
            video_generator = generator
        video_attention_mask = None
        for i_chunk in range(n_chunk):

            video_states_buffer = None
            latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                init_latents, latent_num_frames, latent_height, latent_width, video_generator, noise_to_condition_frames=0
            )

            if self.do_classifier_free_guidance:
                conditioning_mask = torch.cat([conditioning_mask, conditioning_mask], dim=0)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # (g b v) l c
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents.clone()
                    latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                    # TODO: only compute video in the first most noisy timestep
                    compute_video = i == 0 or return_video
                    store_buffer = i == 0 and not return_video

                    if return_action:
                        action_timesteps = t.clone()
                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(actions.shape[0], actions.shape[1])
                        history_action_state_in = None
                        if self.do_classifier_free_guidance:
                            actions_in = torch.cat([actions, actions])
                            action_timesteps = torch.cat([action_timesteps, action_timesteps])
                            if history_action_state is not None:
                                history_action_state_in = torch.cat((history_action_state, history_action_state), dim=0)

                        else:
                            actions_in = actions.clone()
                            if history_action_state is not None:
                                history_action_state_in = history_action_state.clone()
                            
                        actions_in = actions_in.to(prompt_embeds.dtype)
                        if history_action_state is not None:
                            history_action_state_in = history_action_state_in.to(device=device, dtype=prompt_embeds.dtype)

                    else:
                        actions_in = None
                        action_timesteps = None
                        history_action_state_in = None

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])
                    
                    if pixel_wise_timestep:
                        # shape: bv, thw
                        timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        # shape: bv, t
                        timestep = timestep.unsqueeze(-1) * (1 - cond_indicator)
                        
                    
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        rope_interpolation_scale=rope_interpolation_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        action_states=actions_in,
                        action_timestep=action_timesteps,
                        return_video=compute_video,
                        return_action=return_action,
                        n_view=n_view,
                        video_states_buffer=video_states_buffer,
                        store_buffer=store_buffer,
                        video_attention_mask=video_attention_mask,
                        history_action_state=history_action_state_in,
                        condition_mask=conditioning_mask,
                    )[0]


                    if store_buffer:
                        video_states_buffer = noise_pred["video_states_buffer"]

                    if return_action:
                        if self.do_classifier_free_guidance:
                            action_noise_pred_uncond, action_noise_pred = noise_pred["action"].float().chunk(2)
                            action_noise_pred = action_noise_pred_uncond + self.guidance_scale * (action_noise_pred - action_noise_pred_uncond)
                        else:
                            action_noise_pred = noise_pred["action"].float()
                        
                        actions = self.scheduler_action.step(action_noise_pred, t, actions, return_dict=False)[0]
                        
                    if return_video:
                        video_noise_pred = noise_pred["video"].float()

                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = video_noise_pred.chunk(2)
                            video_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            timestep, _ = timestep.chunk(2)

                        # compute the previous noisy sample x_t -> x_t-1
                        video_noise_pred = self._unpack_latents(
                            video_noise_pred,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        latents = self._unpack_latents(
                            latents,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )

                        video_noise_pred = video_noise_pred[:, :, n_prev:]
                        noise_latents = latents[:, :, n_prev:]
                        pred_latents = self.scheduler.step(
                            video_noise_pred, t, noise_latents, return_dict=False
                        )[0]

                        latents = torch.cat([latents[:, :, :n_prev], pred_latents], dim=2)
                        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

                    progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if return_video:
                
                ### clean_frames: (b v) c t h w, -1~1
                clean_frames = self.decode_to_frames(
                    pred_latents,
                    dtype=prompt_embeds.dtype,
                    generator=generator,
                    device=device,
                    batch_size=batch_size * n_view,
                    decode_timestep=decode_timestep,
                    decode_noise_scale=decode_noise_scale,
                )
                if video_list is None:
                    video_list = clean_frames
                else:
                    video_list = torch.cat((video_list, clean_frames), dim=2)

                if i_chunk < n_chunk-1:
                    
                    ### reset scheduler
                    self.scheduler._step_index = None

                    ### prepare memories for next chunk
                    new_mem_idxs = torch.linspace(0, video_list.shape[2]-1, n_prev).round().long()
                    new_mems = video_list[:, :, new_mem_idxs, :, :].clone()
                    new_mems = rearrange(new_mems, "bv c t h w -> (bv t) c h w").unsqueeze(2)
                    init_latents = retrieve_latents(self.vae.encode(new_mems), generator, sample_mode=sample_mode)
                    init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
                    init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)
        

        # Offload all models
        self.maybe_free_model_hooks()

        if return_video:
            preds["video"] = video_list

        if return_action:
            preds["action"] = actions

        self.transformer.train()

        if not return_dict:
            return (preds,)

        return CustomPipelineOutput(frames=preds)
