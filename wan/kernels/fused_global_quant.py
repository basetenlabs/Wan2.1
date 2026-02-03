# This is a sample implementation but isn't used because global sf is computed via calibration.
import torch
import triton
import triton.language as tl

@triton.jit
def fused_global_quant_kernel(X, N, BLOCK_SIZE: tl.constexpr, OUT):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    x = tl.abs(x)
    block_max = tl.max(x, axis=0)
    tl.store(OUT + pid, block_max)

def fused_global_quant_triton(x: torch.Tensor, block_size=4096):
    assert x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    N = x.numel()
    num_blocks = (N + block_size - 1) // block_size
    assert x.is_contiguous()
    x_ = x.view(-1)
    out = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

    fused_global_quant_kernel[(num_blocks,)](
        x_, N,
        BLOCK_SIZE=block_size,
        OUT=out
    )
    gmax = out.max()
    return gmax