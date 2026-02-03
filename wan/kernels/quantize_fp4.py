import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_fp4_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    amax = tl.max(tl.abs(x_f32), axis=0)

    descale = amax / 6.0
    e = tl.ceil(tl.maximum(tl.log2(descale), -127.0))
    pow2e = tl.exp2(e)

    y = x_f32 / pow2e
    tl.store(out_ptr + offs, y, mask=mask)

    e_u8 = (e + 127.0).to(tl.uint8)
    tl.store(scale_ptr + pid, e_u8)

def quantize_fp4_triton(input: torch.Tensor, block_size: int = 16):
    x = input.contiguous()
    orig_shape = x.shape
    N = x.numel()
    if N == 0:
        return x.to(torch.float32), torch.empty(0, dtype=torch.uint8, device=x.device)

    x_flat = x.view(-1)

    y_flat = torch.empty_like(x_flat, dtype=torch.float32)
    num_blocks = (N + block_size - 1) // block_size
    scales = torch.empty(num_blocks, dtype=torch.uint8, device=x.device)

    grid = (num_blocks,)
    _quantize_fp4_kernel[grid](
        x_flat, y_flat, scales,
        N,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2
    )

    return y_flat.view(orig_shape), scales