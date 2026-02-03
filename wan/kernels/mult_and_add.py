import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

# Y = X * (W2d + W1d) + B2d + B1d
@triton.jit
def _mult_and_add_kernel(
    X, # pointer to the input A with shape [L, H]
    X_Bias, # pointer to the input X_Bias with shape [H]
    W1d, # pointer to the input W1d with shape [H]
    W2d, # pointer to the input W2d with shape [L, H]
    B1d, # pointer to the input B1d with shape [H]
    B2d, # pointer to the input B2d with shape [L, H]
    Y, # pointer to the input Y with shape [L, H]
    L: tl.int64,
    H: tl.int64,
    x_stride_l: tl.int64,
    w2d_stride_l: tl.int64,
    b2d_stride_l: tl.int64,
    HAS_W1D: tl.constexpr,
    HAS_W2D: tl.constexpr,
    HAS_B1D: tl.constexpr,
    HAS_B2D: tl.constexpr,
    HAS_X_BIAS: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SAVE_FP32: tl.constexpr,
):
    compute_dtype = tl.float32
    pid = tl.program_id(0)
    row = pid * BLOCK_L + tl.arange(0, BLOCK_L)[:, None]
    mask_rows = row < L

    row_offset = tl.program_id(0) * BLOCK_L
    for col_offset in range(0, H, BLOCK_H):
        cols = col_offset + tl.arange(0, BLOCK_H)[None, :]
        mask_cols = cols < H
        x = tl.load(X + row * x_stride_l + cols, mask=mask_cols & mask_rows, other=0.)
        x_compute = x.to(compute_dtype)
        if HAS_X_BIAS:
            x_bias = tl.load(X_Bias + cols, mask=mask_cols, other=0.).to(compute_dtype)
            x_compute += x_bias
        if HAS_W2D and HAS_W1D:
            w2d = tl.load(W2d + row * w2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            w1d = tl.load(W1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            w = w2d + w1d
        elif not HAS_W2D and HAS_W1D:
            w1d = tl.load(W1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            w = w1d
        elif HAS_W2D and not HAS_W1D:
            w2d = tl.load(W2d + row * w2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            w = w2d
        else:
            w = 1.0
        if HAS_B2D and HAS_B1D:
            b2d = tl.load(B2d + row * b2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            b1d = tl.load(B1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            b = b2d + b1d
        elif not HAS_B2D and HAS_B1D:
            b1d = tl.load(B1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            b = b1d
        elif HAS_B2D and not HAS_B1D:
            b2d = tl.load(B2d + row * b2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            b = b2d
        else:
            b = 0.0
        y = x_compute * w + b
        if not SAVE_FP32:
            y = y.to(x.dtype)
        tl.store(Y + row * x_stride_l + cols, y, mask=mask_cols & mask_rows)

# Y = X * (W2d + W1d) + B2d + B1d
def b10_mult_and_add(x: torch.Tensor, w1d: Optional[torch.Tensor], w2d: Optional[torch.Tensor], b1d: Optional[torch.Tensor], b2d: Optional[torch.Tensor], x_bias: Optional[torch.Tensor] = None, SAVE_FP32: bool = True):
    # print(f"{x.shape=}, {w1d.shape=}, {w2d.shape=}, {b2d.shape=}, {SAVE_FP32=}")
    if SAVE_FP32:
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = torch.empty_like(x)
    x_, y_ = x.view(-1, x.shape[-1]), y.view(-1, x.shape[-1])
    w1d_, w2d_, b1d_, b2d_, x_bias_ = None, None, None, None, None
    if x_bias is not None:
        x_bias_ = x_bias.view(-1, x.shape[-1])
    if w1d is not None:
        w1d_ = w1d.view(-1, x.shape[-1])
    if w2d is not None:
        w2d_ = w2d.view(-1, x.shape[-1])
    if b1d is not None:
        b1d_ = b1d.view(-1, x.shape[-1])
    if b2d is not None:
        b2d_ = b2d.view(-1, x.shape[-1])
    L, H = x_.shape[0], x_.shape[1]
    BLOCK_L = 8
    BLOCK_H = min(triton.next_power_of_2(H), 4096 // BLOCK_L)
    x_stride_l = x_.stride(0)
    w2d_stride_l = w2d_.stride(0) if w2d_ is not None else 0
    b2d_stride_l = b2d_.stride(0) if b2d_ is not None else 0
    _mult_and_add_kernel[(triton.cdiv(L, BLOCK_L),)](
        x_, 
        x_bias_,
        w1d_, 
        w2d_, 
        b1d_, 
        b2d_, 
        y_, 
        L, 
        H, 
        x_stride_l, 
        w2d_stride_l, 
        b2d_stride_l,
        w1d is not None, 
        w2d is not None, 
        b1d is not None, 
        b2d is not None,
        x_bias is not None,
        BLOCK_L, 
        BLOCK_H, 
        SAVE_FP32
    )
    return y

@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("S", [128])
@pytest.mark.parametrize("H", [5120])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_mult_and_add(B, S, H, dtype):
    x = torch.randn(B, S, H, dtype=dtype).cuda()
    w1d = torch.randn(H, dtype=dtype).cuda()
    w2d = torch.randn(B, S, H, dtype=dtype).cuda()
    b1d = torch.randn(H, dtype=dtype).cuda()
    b2d = torch.randn(B, S, H, dtype=dtype).cuda()
    y_b10_base = b10_mult_and_add(x, w1d, w2d, b1d, b2d, SAVE_FP32=False)
    y_torch = (x.float() * (w2d.float() + w1d.float()) + (b2d.float() + b1d.float())).type_as(x)
    torch.testing.assert_close(y_b10_base, y_torch, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[512, 9450, 18900],  # different possible values for `x_name`
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
        "Mult and Add throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 5120,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_mult_and_add(B, S, H, dtype, provider):
    """Benchmark Mult and Add throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    w1d = torch.randn(H, device=device, dtype=dtype).cuda()
    w2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()
    # b1d = torch.randn(H, device=device, dtype=dtype).cuda()
    b2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()

    def _b10_mult_and_add():
        return b10_mult_and_add(x, w1d, w2d, None, b2d, SAVE_FP32=True)

    def torch_mult_and_add():
        return x * (w2d + w1d) + (b2d)

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_mult_and_add)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_mult_and_add)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (4 * B * S * H) * 4 / ms * 1e-6
    return gb_s

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[512, 9450, 18900],  # different possible values for `x_name`
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
        "Just Add throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 5120,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_just_add(B, S, H, dtype, provider):
    """Benchmark Mult and Add throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    b2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()

    def _b10_mult_and_add():
        return b10_mult_and_add(x, None, None, None, b2d, SAVE_FP32=False)

    def torch_mult_and_add():
        return x + b2d

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_mult_and_add)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_mult_and_add)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (4 * B * S * H) * 4 / ms * 1e-6
    return gb_s



if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_mult_and_add.run(print_data=True)
    benchmark_just_add.run(print_data=True)
