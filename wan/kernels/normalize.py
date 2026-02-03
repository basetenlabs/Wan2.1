import torch
import pytest
import triton
import triton.language as tl
import torch.nn.functional as F


import torch
import triton
import triton.language as tl


@triton.jit
def normalize3d_nch_kernel(
    x_ptr, y_ptr,
    N, C, H,
    scale, gamma_ptr, bias_ptr,
    eps,                               # float32 scalar
    sxN, sxC, sxH,                     # input strides (elements)
    syN, syC, syH,                     # output strides (elements)
    HAS_BIASES: tl.constexpr,          # boolean indicating if biases are present
    BLOCK_H: tl.constexpr,             # number of (n,h) positions per program
    BLOCK_C: tl.constexpr,             # channels processed per inner iteration
):
    # Flatten (N, H) -> M positions
    n = tl.program_id(1)
    pid = tl.program_id(0)
    h_offsets = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Base pointers for each (n, :, h) vector across C
    x_base = x_ptr + n * sxN + h_offsets * sxH
    y_base = y_ptr + n * syN + h_offsets * syH

    # ---------- Pass 1: compute ||x||_p over C ----------
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        cmask = (c_idx < C)[None, :] & (h_mask[:, None])

        x_ptrs = x_base[:, None] + c_idx[None, :] * sxC
        x_chunk = tl.load(x_ptrs, mask=cmask, other=0)

        x_f32 = x_chunk.to(tl.float32)
        acc += tl.sum(x_f32 * x_f32, axis=1)

    denom = tl.sqrt(acc)

    inv = 1.0 / tl.maximum(denom, eps)
    inv = tl.where(h_mask, inv, 0.0)  # silence inactive lanes

    # ---------- Pass 2: y = x / ||x||_p ----------
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        cmask = (c_idx < C)[None, :]

        x_ptrs = x_base[:, None] + c_idx[None, :] * sxC
        y_ptrs = y_base[:, None] + c_idx[None, :] * syC
        gamma_ptrs = gamma_ptr + c_idx[None, :]
        bias_ptrs = bias_ptr + c_idx[None, :]
        gamma = tl.load(gamma_ptrs, mask=cmask, other=0).to(tl.float32)
        x_chunk = tl.load(x_ptrs, mask=(cmask & h_mask[:, None]), other=0)
        y_chunk = (x_chunk.to(tl.float32) * inv[:, None]) * gamma * scale
        if HAS_BIASES:
            bias = tl.load(bias_ptrs, mask=cmask, other=0).to(tl.float32)
            y_chunk += bias
        tl.store(y_ptrs, y_chunk.to(x_chunk.dtype), mask=(cmask & h_mask[:, None]))


def normalize3d_nch(x: torch.Tensor, eps: float = 1e-5, scale: float = 1.0, gamma: torch.Tensor = None, bias: torch.Tensor = None):
    """
    Channel-wise normalization for 3D NCH:
      y[n,c,h] = x[n,c,h] / max(eps, ||x[n,:,h]||_p)

    Works with any dtype and strides. Accumulation is fp32.
    """
    assert x.ndim == 3, "expected (N, C, H)"
    N, C, H = x.shape
    assert gamma.numel() == C, f"gamma must have {C} elements, but got {gamma.numel()}"
    assert isinstance(bias, torch.Tensor) or isinstance(bias, float), "bias must be a tensor or float or None"
    HAS_BIASES = isinstance(bias, torch.Tensor)
    if HAS_BIASES:
        assert bias.numel() == C, f"bias must have {C} elements, but got {bias.numel()}"
        bias = bias.view(-1)
    y = torch.empty_like(x)

    sxN, sxC, sxH = x.stride()
    syN, syC, syH = y.stride()

    BLOCK_H = min(triton.next_power_of_2(H), 2048)    # (n,h) positions per program
    BLOCK_C = min(triton.next_power_of_2(C), 2048 // BLOCK_H)    # channels per inner iteration (tune per GPU/shape)

    grid = (triton.cdiv(H, BLOCK_H), N)

    normalize3d_nch_kernel[grid](
        x, y,
        N, C, H,
        scale, gamma.view(-1), bias,
        eps,
        sxN, sxC, sxH,
        syN, syC, syH,
        BLOCK_H=BLOCK_H,
        BLOCK_C=BLOCK_C,
        HAS_BIASES=HAS_BIASES,
        num_warps=4,
        num_stages=2,
    )
    return y

@torch.profiler.record_function("b10_normalize3d_nch")
def b10_normalize3d_nch(x: torch.Tensor, dim=1, eps: float = 1e-5, scale: float = 1.0, gamma: torch.Tensor = None, bias: torch.Tensor = None):
    assert dim==1 or dim==-1, "dim must be 1 or -1"
    # print(f"b10_normalize3d_nch: {x.shape=}, {dim=}, {eps=}, {scale=}, {gamma=}, {bias=}")
    old_shape = x.shape
    N = x.shape[0]
    if dim == 1:
        x_ = x.reshape(N, x.shape[1], -1)
    else:
        x_ = x.reshape(-1, x.shape[-1], 1)
    y = normalize3d_nch(x_, scale=scale, gamma=gamma, bias=bias, eps=eps)
    return y.view(old_shape)

@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("C", [16])
@pytest.mark.parametrize("H", [1280])
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.no_grad()
def test_normalize3d_nch(N, C, H, dtype):
    scale = 8.0
    x = torch.randn(N, C, H, dtype=dtype).cuda()
    gamma = torch.randn(1, C, 1, dtype=dtype).cuda()
    bias = torch.randn(1, C, 1, dtype=dtype).cuda()
    y_b10_base = normalize3d_nch(x, scale=scale, gamma=gamma, bias=bias)
    y_torch = torch.nn.functional.normalize(x, dim=1, eps=1e-5) * scale * gamma + bias
    torch.testing.assert_close(y_b10_base, y_torch, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[512, 1280, 720 * 1280],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "b10",
        ],  # possible values for `line_arg`
        line_names=[
            "Torch",
            "B10",
        ],  # label name for the lines
        styles=[
            ("green", "-"),
            ("red", "--"),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "ZeroPad5D throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "N": 1,
            "C": 16,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_zero_pad5d(N, C, H, dtype, provider):
    """Benchmark ZeroPad5D throughput across different implementations."""
    x = torch.randn(N, C, H, dtype=dtype, device="cuda")
    scale = 8.0
    gamma = torch.randn(1, C, 1, dtype=dtype, device="cuda")
    bias = torch.randn(1, C, 1, dtype=dtype, device="cuda")

    def _b10_zero_pad5d():
        return normalize3d_nch(x, scale=scale, gamma=gamma, bias=bias)

    def torch_zero_pad5d():
        return F.normalize(x, dim=1, eps=1e-5) * scale * gamma + bias

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_zero_pad5d)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_zero_pad5d)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (N * C * H) * 2 * dtype.itemsize / ms * 1e-6
    return gb_s

if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_zero_pad5d.run(print_data=True)