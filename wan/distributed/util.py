# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
import torch.distributed as dist
from .b10_low_contention_alltoall import B10LowContentionAlltoall
from ..kernels.permute import b10_permute_mnk

ENABLE_LOW_CONTENDING_ALLTOALL = os.environ.get("ENABLE_LOW_CONTENDING_ALLTOALL", "1") == "1"


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


@torch.profiler.record_function("all_to_all")
def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x

low_contending_alltoall = {}
low_contending_alltoall_streams = {}
def b10_collect_tokens(x, group=None, group_name="0", launch_barrier=True, reuse_streams_group=None):
    """
    args: 
        x: [B, S, CP * N, H]
        S: local sequence length
        CP: context parallel size
        N: number of local attention heads
    """
    if group is None:
        group = dist.group.WORLD
    cp_size = get_world_size()
    if cp_size > 1:
        with torch.profiler.record_function("b10_collect_tokens"):
            B, S, N, H = x.shape
            N = N // cp_size
            x = x.view(B * S, cp_size, N * H)
            x = b10_permute_mnk(x).view(cp_size, B, S, N, H)
            # x = einops.rearrange(x, "B S (CP N) H -> CP B S N H", CP=cp_size).contiguous()
        if ENABLE_LOW_CONTENDING_ALLTOALL:
            if reuse_streams_group is not None and reuse_streams_group in low_contending_alltoall_streams:
                streams = low_contending_alltoall_streams[reuse_streams_group]
            else:
                streams = [torch.cuda.Stream() for _ in range(1)]
                low_contending_alltoall_streams[group_name] = streams
            if group_name not in low_contending_alltoall:
                low_contending_alltoall[group_name] = B10LowContentionAlltoall(
                    group=group,
                    tensor_numel=x.numel(), 
                    dtype=x.dtype, 
                    device=None, 
                    num_streams=1,
                    launch_barrier=launch_barrier,
                    streams=streams
                )
            output = low_contending_alltoall[group_name](x)
            return output, low_contending_alltoall[group_name]
        output = torch.empty_like(x) 
        handle = torch.distributed.all_to_all_single(output, x, group=group, async_op=True)
        return output, handle
    return x

def b10_collect_tokens_wait(output, handle):
    if handle is not None:
        handle.wait()
        with torch.profiler.record_function("b10_collect_tokens_wait"):
            CP, B, S, N, H = output.shape
            output = output.view(CP, B, S * N * H)
            output = b10_permute_mnk(output).view(B, CP * S, N, H)
            # output = einops.rearrange(output, "CP B S N H -> B (CP S) N H").contiguous()
    return output


def b10_collect_heads(x, group=None, group_name="0", launch_barrier=True, reuse_streams_group=None):
    """
    args: 
        x: [B, CP * S, N, H]
        S: local sequence length
        CP: context parallel size
        N: number of local attention heads
    """
    if group is None:
        group = dist.group.WORLD
    cp_size = get_world_size()
    if cp_size > 1:
        with torch.profiler.record_function("b10_collect_heads"):
            B, S, N, H = x.shape
            S = S // cp_size
            x = x.view(B, cp_size, S * N * H)
            x = b10_permute_mnk(x).view(cp_size, B, S, N, H)
            # x = einops.rearrange(x, "B (CP S) N H -> CP B S N H", CP=cp_size).contiguous()
        if ENABLE_LOW_CONTENDING_ALLTOALL:
            if reuse_streams_group is not None and reuse_streams_group in low_contending_alltoall_streams:
                streams = low_contending_alltoall_streams[reuse_streams_group]
            else:
                streams = [torch.cuda.Stream() for _ in range(1)]
                low_contending_alltoall_streams[group_name] = streams
            if group_name not in low_contending_alltoall:
                low_contending_alltoall[group_name] = B10LowContentionAlltoall(
                    group=group,
                    tensor_numel=x.numel(), 
                    dtype=x.dtype, 
                    device=None, 
                    num_streams=1,
                    launch_barrier=launch_barrier,
                    streams=streams
                )
            output = low_contending_alltoall[group_name](x)
            return output, low_contending_alltoall[group_name]
        output = torch.empty_like(x)
        handle = torch.distributed.all_to_all_single(output, x, group=group, async_op=True)
        return output, handle
    return x, None

def b10_collect_heads_wait(output, handle):
    if handle is not None:
        handle.wait()
        with torch.profiler.record_function("b10_collect_heads_wait"):
            CP, B, S, N, H = output.shape
            output = output.view(CP, B * S, N * H)
            output = b10_permute_mnk(output).view(B, S, CP * N, H)
            # output = einops.rearrange(output, "CP B S N H -> B S (CP N) H").contiguous()
    return output



def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
