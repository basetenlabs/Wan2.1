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

import time
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .b10_model_loader import B10ModelLoader, Rank0First
from .b10_config import enable_b10_attn_cache
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


class WanT2V:

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
        b10_model_loader: Optional[B10ModelLoader] = None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

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
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
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
                shard_fn=None)
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
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        end_event("create_vae")

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

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

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
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            use_cfg_cache = enable_b10_attn_cache()
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

            enable_profile = int(os.getenv("ENABLE_PROFILE", "0"))
            profiler = None
            if enable_profile == 1:
                logging.info("Profiling WanT2V generate")
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(
                        wait=4,
                        warmup=1,
                        active=1,
                        repeat=1,
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                profiler.start()

            start_event("diffusion_sampling")
            for step_id, t in enumerate(tqdm(timesteps)):
                if profiler is not None:
                    profiler.step()
                if use_cfg_cache:
                    B10CONDNCACHE.force_update = False
                    B10UNCONDCACHE.force_update = False
                    B10CONDNCACHE.current_step_id = step_id
                    B10CONDNCACHE.current_ts = t
                    B10UNCONDCACHE.current_step_id = step_id
                    use_uncond_cache = B10UNCONDCACHE.if_use_cache_this_step()
                else:
                    use_uncond_cache = False
                self.model.to(self.device)
                if use_cfg_cache:
                    latent_model_input = latents
                    timestep = torch.stack([t])
                    noise_pred_cond = self.model(
                        latent_model_input,
                        t=timestep,
                        **arg_c,
                        use_attn_cache=True,
                    )[0]
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
                        if B10UNCONDCACHE.if_use_cache_next_step():
                            B10UNCONDCACHE.set_cached_output(
                                noise_pred_cond, noise_pred_uncond)
                elif self.sp_size > 1:
                    # Dual-branch CFG in one forward (cond + uncond)
                    latent_model_input = [latents[0], latents[0]]
                    timestep = torch.stack([t, t])
                    context_batched = [context[0], context_null[0]]
                    preds = self.model(
                        latent_model_input,
                        t=timestep,
                        context=context_batched,
                        seq_len=seq_len)
                    noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
                else:
                    latent_model_input = latents
                    timestep = torch.stack([t])
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            end_event("diffusion_sampling")

            x0 = latents
            if offload_model:
                start_event("offload_model")
                self.model.cpu()
                torch.cuda.empty_cache()
                end_event("offload_model")
            if self.rank == 0:
                start_event("vae_decode")
                videos = self.vae.decode(x0)
                end_event("vae_decode")

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if profiler is not None:
            s = time.perf_counter()
            profiler.stop()
            trace_name = os.getenv("TRACE_NAME", "wan_t2v_generate")
            rank = dist.get_rank() if dist.is_initialized() else self.rank
            profiler.export_chrome_trace(f"{trace_name}_rank{rank}_pid{os.getpid()}.json.gz")
            print(f"[{rank}]time taken to dump torch profiler: {time.perf_counter() - s}s")

        return videos[0] if self.rank == 0 else None
