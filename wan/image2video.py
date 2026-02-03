# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import json
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
try:
    from .modules.vae2_1 import Wan2_1_VAE
    _WAN21_VAE_AVAILABLE = True
except Exception as exc:
    Wan2_1_VAE = None
    _WAN21_VAE_AVAILABLE = False
    logging.warning("Wan2_1_VAE unavailable; falling back to WanVAE. Error: %s",
                    exc)
from .b10_model_loader import B10ModelLoader, Rank0First
from .b10_config import enable_b10_attn_cache, enable_b10_kernel
from .distributed.b10_attn_cache import B10UNCONDCACHE, B10CONDNCACHE
from accelerate import init_empty_weights
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
try:
    from blite_tracing.trace import start_event, end_event
except ImportError:
    start_event = end_event = lambda x: None


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        b10_model_loader: Optional[B10ModelLoader] = None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.b10_model_loader = b10_model_loader or B10ModelLoader(
            checkpoint_dir, num_shards=world_size)

        text_encoder_meta_path = self.b10_model_loader.b10fs_path(
            f"text_encoder_shard_info_{world_size}.json")
        text_encoder_shard_model_path = self.b10_model_loader.b10fs_path(
            f"text_encoder_shard{rank}-{world_size}.safetensors")
        delay_text_encoder_load = text_encoder_shard_model_path.exists()

        load_device = "cpu"
        if dist.is_initialized() and dist.get_backend() == "nccl":
            load_device = self.device

        start_event("create_text_encoder")
        if not delay_text_encoder_load:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir,
                                             config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir,
                                            config.t5_tokenizer),
                shard_fn=None,
            )
            with Rank0First():
                text_encoder_metadata = (
                    self.b10_model_loader.create_or_read_metadata_for_load(
                        self.text_encoder.model, text_encoder_meta_path))
            self.b10_model_loader.save_model_shard(
                self.text_encoder.model,
                text_encoder_shard_model_path,
                metadata=text_encoder_metadata,
                shard_id=rank,
            )
        else:
            with init_empty_weights():
                self.text_encoder = T5EncoderModel(
                    text_len=config.text_len,
                    dtype=config.t5_dtype,
                    device=torch.device('cpu'),
                    checkpoint_path=None,
                    tokenizer_path=os.path.join(checkpoint_dir,
                                                config.t5_tokenizer),
                    shard_fn=None,
                    skip_load=True,
                )
            with Rank0First():
                text_encoder_metadata = (
                    self.b10_model_loader.create_or_read_metadata_for_load(
                        self.text_encoder.model, text_encoder_meta_path))
            self.b10_model_loader.load_model_from_safetensors(
                self.text_encoder.model,
                text_encoder_shard_model_path,
                metadata=text_encoder_metadata,
                shard_id=rank,
                dtype=config.t5_dtype,
                device=load_device,
            )
        if t5_fsdp:
            self.text_encoder.model = shard_fn(self.text_encoder.model,
                                               sync_module_states=False)
        end_event("create_text_encoder")

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        start_event("create_vae")
        if enable_b10_kernel() and _WAN21_VAE_AVAILABLE:
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device)
        else:
            self.vae = WanVAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device)
        end_event("create_vae")

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        start_event("create_model")
        model_meta_path = self.b10_model_loader.b10fs_path(
            f"model_shard_info_{world_size}.json")
        model_path = Path(checkpoint_dir)
        with init_empty_weights():
            with open(model_path / "config.json", "r") as f:
                model_config = json.load(f)
            model_config = {
                k: v
                for k, v in model_config.items()
                if not str(k).startswith("_")
            }
            self.model = WanModel(**model_config)
        with Rank0First():
            model_metadata = (
                self.b10_model_loader.create_or_read_metadata_for_load(
                    self.model, model_meta_path, model_path=model_path))
        if dist.is_initialized() and dist.get_backend() == "nccl":
            load_device = self.device
        else:
            load_device = "cpu" if init_on_cpu else self.device
        self.b10_model_loader.load_model_from_safetensors(
            self.model,
            model_path,
            metadata=model_metadata,
            shard_id=rank,
            dtype=self.param_dtype,
            device=load_device,
        )
        self.model.eval().requires_grad_(False)
        end_event("create_model")

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            use_b10_sp = False
            if enable_b10_kernel():
                try:
                    from .distributed.b10_sequence_parallel import (
                        b10_sp_attn_forward,
                        b10_sp_block_forward_2_batch,
                        b10_sp_dit_forward_2_batch,
                    )
                    use_b10_sp = True
                except Exception as exc:
                    logging.warning(
                        "B10 sequence-parallel disabled; falling back to USP. "
                        "Error: %s", exc)
            if not use_b10_sp:
                from .distributed.xdit_context_parallel import (
                    usp_attn_forward,
                    usp_dit_forward,
                )
            for block in self.model.blocks:
                if use_b10_sp:
                    block.forward = types.MethodType(
                        b10_sp_block_forward_2_batch, block)
                    block.self_attn.forward = types.MethodType(
                        b10_sp_attn_forward, block.self_attn)
                else:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
            if use_b10_sp:
                self.model.forward = types.MethodType(
                    b10_sp_dit_forward_2_batch, self.model)
                self.sp_size = dist.get_world_size(
                ) if dist.is_initialized() else get_sequence_parallel_world_size(
                )
            else:
                self.model.forward = types.MethodType(usp_dit_forward,
                                                      self.model)
                self.sp_size = get_sequence_parallel_world_size()
            self.use_b10_sp = use_b10_sp
        else:
            self.sp_size = 1
            self.use_b10_sp = False

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        start_event("text_encoder")
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        end_event("text_encoder")

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }
            use_cfg_cache = enable_b10_attn_cache()
            use_b10_sp = self.use_b10_sp and self.sp_size > 1
            if use_cfg_cache:
                B10UNCONDCACHE.step_id_threshold = 10
                B10UNCONDCACHE.interval = 4
                B10CONDNCACHE.step_id_threshold = 10
                B10CONDNCACHE.interval = 3
                B10CONDNCACHE.max_step_id = len(timesteps)
                B10UNCONDCACHE.max_step_id = len(timesteps)
                B10CONDNCACHE.layer_id2attn_output = {}
                B10UNCONDCACHE.delta_high_freq = None
                B10UNCONDCACHE.delta_low_freq = None

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            start_event("diffusion_sampling")
            for step_id, t in enumerate(tqdm(timesteps)):
                if use_cfg_cache or use_b10_sp:
                    B10CONDNCACHE.force_update = False
                    B10UNCONDCACHE.force_update = False
                    B10CONDNCACHE.current_step_id = step_id
                    B10CONDNCACHE.current_ts = t
                    B10UNCONDCACHE.current_step_id = step_id
                    use_uncond_cache = (
                        use_cfg_cache
                        and B10UNCONDCACHE.if_use_cache_this_step())
                else:
                    use_uncond_cache = False
                if use_b10_sp:
                    latent_model_input = [latent.to(self.device)]
                    timestep = torch.stack([t]).to(self.device)
                    noise_pred_cond, noise_pred_uncond = self.model(
                        latent_model_input,
                        t=timestep,
                        context=context,
                        context_null=context_null,
                        seq_len=max_seq_len,
                        y=[y],
                        clip_fea=clip_context,
                    )
                    noise_pred_cond = noise_pred_cond[0]
                    noise_pred_uncond = noise_pred_uncond[0]
                    if offload_model:
                        noise_pred_cond = noise_pred_cond.to(
                            torch.device('cpu'))
                        noise_pred_uncond = noise_pred_uncond.to(
                            torch.device('cpu'))
                        torch.cuda.empty_cache()
                elif use_cfg_cache:
                    latent_model_input = [latent.to(self.device)]
                    timestep = torch.stack([t]).to(self.device)
                    noise_pred_cond = self.model(
                        latent_model_input,
                        t=timestep,
                        **arg_c,
                        use_attn_cache=True,
                    )[0]
                    if offload_model:
                        noise_pred_cond = noise_pred_cond.to(
                            torch.device('cpu'))
                        torch.cuda.empty_cache()
                    if use_uncond_cache:
                        noise_pred_uncond = B10UNCONDCACHE.get_cached_output(
                            noise_pred_cond)
                    else:
                        noise_pred_uncond = self.model(
                            latent_model_input,
                            t=timestep,
                            **arg_null,
                            use_attn_cache=False,
                        )[0]
                        if offload_model:
                            noise_pred_uncond = noise_pred_uncond.to(
                                torch.device('cpu'))
                            torch.cuda.empty_cache()
                        if B10UNCONDCACHE.if_use_cache_next_step():
                            B10UNCONDCACHE.set_cached_output(
                                noise_pred_cond, noise_pred_uncond)
                elif self.sp_size > 1:
                    # Dual-branch CFG in one forward (cond + uncond)
                    latent_model_input = [latent.to(self.device), latent.to(self.device)]
                    timestep = torch.stack([t, t]).to(self.device)
                    context_batched = [context[0], context_null[0]]
                    clip_batched = torch.cat([clip_context, clip_context], dim=0)
                    y_batched = [y, y]
                    preds = self.model(
                        latent_model_input,
                        t=timestep,
                        context=context_batched,
                        clip_fea=clip_batched,
                        seq_len=max_seq_len,
                        y=y_batched)
                    noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
                    if offload_model:
                        noise_pred_cond = noise_pred_cond.to(torch.device('cpu'))
                        noise_pred_uncond = noise_pred_uncond.to(torch.device('cpu'))
                        torch.cuda.empty_cache()
                else:
                    latent_model_input = [latent.to(self.device)]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep
            end_event("diffusion_sampling")

            if offload_model:
                start_event("offload_model")
                self.model.cpu()
                torch.cuda.empty_cache()
                end_event("offload_model")

            if self.rank == 0:
                start_event("vae_decode")
                videos = self.vae.decode(x0)
                end_event("vae_decode")

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
