"""
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 wan/distributed/b10_low_contention_alltoall.py
"""
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import rendezvous

class B10LowContentionAlltoall:
    def __init__(self, 
        group,
        tensor_numel,
        dtype=torch.bfloat16, 
        device=None, 
        num_streams=1,
        launch_barrier=True,
        streams=None,
    ):
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group=group)
        self.dtype = dtype
        self.device = device
        self.group_name = group.group_name
        self.num_streams = num_streams
        if streams is None:
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        else:
            assert len(streams) == num_streams, "streams must be of length num_streams"
            self.streams = streams
        self.local_buffer = _SymmetricMemory.empty_strided_p2p(
            (tensor_numel,),
            (1,),
            dtype=dtype,
            device=device,
            group_name=group.group_name,
        )
        self.x_ref = None
        self.launch_barrier = launch_barrier

    def __call__(self, x):
        # x: input tensor, [CP, H]
        self.x_ref = x # keep a reference to the input tensor so that it won't be freed
        symm_mem = rendezvous(self.local_buffer, self.group_name)
        original_shape = x.shape
        x = x.view(self.world_size, -1)
        shape_per_shard = (x.shape[0] // self.world_size, x.shape[1])
        numel_per_shard = x.numel() // self.world_size
        output = symm_mem.get_buffer(self.rank, x.shape, x.dtype)
        for stream in self.streams:
            stream.wait_stream(torch.cuda.current_stream())
        local_shard = symm_mem.get_buffer(self.rank, shape_per_shard, x.dtype, numel_per_shard*self.rank)
        local_shard.copy_(x[self.rank:self.rank+1, :])
        for i in range(1, self.world_size):
            dst_rank = (self.rank + i) % self.world_size
            with torch.cuda.stream(self.streams[i % self.num_streams]):
                local_shard = symm_mem.get_buffer(dst_rank, shape_per_shard, x.dtype, numel_per_shard*self.rank)
                local_shard.copy_(x[dst_rank:dst_rank+1, :])
        
        if self.launch_barrier:
            for i in range(self.num_streams):
                with torch.cuda.stream(self.streams[(i+1) % self.num_streams]):
                    symm_mem.barrier()
        return output.view(original_shape)
    
    def wait(self):
        assert self.x_ref is not None, "x_ref is not set"
        self.x_ref = None
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)
        

if __name__ == "__main__":
    world_size = 4
    B, H = 1024, 1024
    tensor_numel = B * H
    dtype = torch.bfloat16
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    alltoall = B10LowContentionAlltoall(dist.group.WORLD, tensor_numel, dtype, None, 1)
    x = torch.randn(B, H, device=torch.cuda.current_device(), dtype=dtype)
    b10_output = alltoall(x)
    torch_output = torch.empty_like(x)
    dist.all_to_all_single(torch_output, x)
    torch.testing.assert_close(b10_output, torch_output)

    alltoall_null = B10LowContentionAlltoall(dist.group.WORLD, tensor_numel, dtype, None, 1)
    x_null = torch.randn(B, H, device=torch.cuda.current_device(), dtype=dtype)
    b10_output_null = alltoall_null(x_null)
    torch_output_null = torch.empty_like(x_null)
    dist.all_to_all_single(torch_output_null, x_null)
    torch.testing.assert_close(b10_output_null, torch_output_null)
