import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    W_MODULATION,  # pointer to the weight modulation, [H]
    B_MODULATION,  # pointer to the bias modulation, [H]
    stride: tl.int64,  # how much to increase the pointer when moving by 1 row
    w_stride: tl.int64,  # how much to increase the pointer when moving by 1 row
    N: tl.int64,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    WITH_AFFINE: tl.constexpr,
    SAVE_FP32: tl.constexpr,
    USE_TIME_EMBED_AS_AFFINE: tl.constexpr,
):
    compute_dtype = tl.float32
    if SAVE_FP32:
        output_dtype = tl.float32
    else:
        output_dtype = tl.bfloat16
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    if WITH_AFFINE and USE_TIME_EMBED_AS_AFFINE:
        W += row * w_stride
        B += row * w_stride
    Y += row * stride
    X += row * stride
    # Compute mean
    _sum = tl.zeros((), dtype=compute_dtype)
    _sum_sq = tl.zeros((), dtype=compute_dtype)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(compute_dtype)
        _sum += tl.sum(x, axis=0)
        _sum_sq += tl.sum(x * x, axis=0)

    mean = _sum / N
    var = _sum_sq / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.)
        x_hat = ((x.to(compute_dtype) - mean) * rstd).to(x.dtype).to(compute_dtype)
        if WITH_AFFINE:
            w = tl.load(W + cols, mask=mask, other=0.).to(compute_dtype)
            b = tl.load(B + cols, mask=mask, other=0.).to(compute_dtype)
            if USE_TIME_EMBED_AS_AFFINE:
                w_modulation = tl.load(W_MODULATION + cols, mask=mask, other=0.).to(compute_dtype)
                b_modulation = tl.load(B_MODULATION + cols, mask=mask, other=0.).to(compute_dtype)
                w += (w_modulation + 1.0)
                b += b_modulation
            y = x_hat * w + b
        else:
            y = x_hat
        # Write output
        tl.store(Y + cols, y.to(output_dtype), mask=mask)


class B10LayerNorm(nn.LayerNorm):
    """
    GroupNorm applied per-frame.
    """

    def __init__(self,
                 dim,
                 eps=1e-6,
                 elementwise_affine=False,
                 save_fp32=False,
                 use_time_embed_as_affine=False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.save_fp32 = save_fp32
        self.use_time_embed_as_affine = use_time_embed_as_affine

    @torch.profiler.record_function("B10LayerNorm")
    def forward(self, x: torch.Tensor):
        assert x.is_contiguous(), "x must be contiguous"
        assert x.dtype == torch.float32 or x.dtype == torch.bfloat16, "x must be float32 or bfloat16"
        if self.use_time_embed_as_affine:
            assert self.time_weight.shape == x.shape, f"B10LayerNormError: {self.time_weight.shape=}, {x.shape=}"
            assert self.time_bias.shape == x.shape, f"B10LayerNormError: {self.time_bias.shape=}, {x.shape=}"
        
        y_dtype = torch.float32 if self.save_fp32 else x.dtype
        save_fp32 = y_dtype == torch.float32
        N = x.shape[-1]
        y = torch.empty_like(x, dtype=y_dtype)
        x_, y_ = x.view(-1, N), y.view(-1, N)
        w_, b_, w_modulation, b_modulation, w_stride = None, None, None, None, 0
        if self.elementwise_affine:
            if self.use_time_embed_as_affine:
                w_ = self.time_weight.view(-1, N)
                b_ = self.time_bias.view(-1, N)
                w_modulation = self.weight_modulation.view(N)
                b_modulation = self.bias_modulation.view(N)
                assert w_modulation.is_contiguous(), "w_modulation must be contiguous"
                assert b_modulation.is_contiguous(), "b_modulation must be contiguous"
                assert w_.stride(0) == b_.stride(0), "w_ and b_ must have the same stride"
            else:
                w_ = self.weight.view(-1, N)
                b_ = self.bias.view(-1, N)
            w_stride = w_.stride(0)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        _layer_norm_fused[(x_.shape[0], )](
            x_,
            y_,
            w_,
            b_,
            w_modulation,
            b_modulation,
            x_.stride(0),
            w_stride,
            N,
            self.eps,
            BLOCK_SIZE,
            self.elementwise_affine,
            save_fp32,
            self.use_time_embed_as_affine,
            num_warps=4,
            num_stages=2,
        )
        return y


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("S", [128])
@pytest.mark.parametrize("H", [5120])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_norm(B, S, H, eps, dtype):
    x = torch.randn(B, S, H).cuda().to(dtype)
    b10_norm_fn = B10LayerNorm(x.shape[-1],
                               eps=eps,
                               elementwise_affine=False).cuda()
    torch_norm_fn = nn.LayerNorm(x.shape[-1],
                                 eps=eps,
                                 elementwise_affine=False).cuda()
    e1 = torch.randn_like(x, dtype=torch.float32).cuda()
    e2 = torch.randn_like(x, dtype=torch.float32).cuda()
    m1 = torch.randn(H, dtype=torch.float32).cuda()
    m2 = torch.randn(H, dtype=torch.float32).cuda()
    y_b10_base = b10_norm_fn(x).float() * (1 + e1 + m1) + (e2 + m2)
    b10_norm_fn.use_time_embed_as_affine = True
    b10_norm_fn.save_fp32 = True
    b10_norm_fn.elementwise_affine = True
    b10_norm_fn.time_weight = e1
    b10_norm_fn.time_bias = e2
    b10_norm_fn.weight_modulation = m1
    b10_norm_fn.bias_modulation = m2
    y_b10 = b10_norm_fn(x)
    y_torch = torch_norm_fn(x).float() * (1 + e1 + m1) + (e2 + m2)
    if dtype == torch.float32:
        torch.testing.assert_close(y_b10, y_torch, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(y_b10, y_torch, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(y_b10_base, y_b10, rtol=1e-5, atol=1e-5)


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
        "LayerNorm throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 5120,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_layer_norm(B, S, H, dtype, provider):
    """Benchmark GroupNorm throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    eps, elementwise_affine = 1e-5, False
    b10_norm_fn = B10LayerNorm(x.shape[-1],
                               eps=eps,
                               elementwise_affine=elementwise_affine).cuda()
    torch_norm_fn = nn.LayerNorm(x.shape[-1],
                                 eps=eps,
                                 elementwise_affine=elementwise_affine).cuda()
    e1 = torch.randn_like(x, dtype=torch.float32).cuda()
    e2 = torch.randn_like(x, dtype=torch.float32).cuda()
    m1 = torch.randn(H, dtype=torch.float32).cuda()
    m2 = torch.randn(H, dtype=torch.float32).cuda()
    b10_norm_fn.time_weight = e1
    b10_norm_fn.time_bias = e2
    b10_norm_fn.weight_modulation = m1
    b10_norm_fn.bias_modulation = m2
    b10_norm_fn.use_time_embed_as_affine = True
    b10_norm_fn.save_fp32 = True
    b10_norm_fn.elementwise_affine = True

    def b10_norm():
        return b10_norm_fn(x)

    def torch_norm():
        return torch_norm_fn(x).float() * (1 + e1 + m1) + (e2 + m2)

    if provider == "b10":
        ms = triton.testing.do_bench(b10_norm)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_norm)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (4 * S * H) * 4 / ms * 1e-6
    return gb_s


if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_layer_norm.run(print_data=True)
    B, S, H = 1, 128, 5120
    eps = 1e-5
    dtype = torch.bfloat16
    x = torch.randn(B, S, H).cuda().to(dtype)
    b10_norm_fn = B10LayerNorm(x.shape[-1],
                               eps=eps,
                               elementwise_affine=False).cuda()
    torch_norm_fn = nn.LayerNorm(x.shape[-1],
                                 eps=eps,
                                 elementwise_affine=False).cuda()
    e1 = torch.randn_like(x, dtype=torch.float32).cuda()
    e2 = torch.randn_like(x, dtype=torch.float32).cuda()
    m1 = torch.randn(H, dtype=torch.float32).cuda()
    m2 = torch.randn(H, dtype=torch.float32).cuda()
    y_b10_base = b10_norm_fn(x).float() * (1 + e1 + m1) + (e2 + m2)
    b10_norm_fn.elementwise_affine = True
    b10_norm_fn.use_time_embed_as_affine = True
    b10_norm_fn.save_fp32 = True
    b10_norm_fn.time_weight = e1
    b10_norm_fn.time_bias = e2
    b10_norm_fn.weight_modulation = m1
    b10_norm_fn.bias_modulation = m2
    y_b10 = b10_norm_fn(x)
    y_torch = torch_norm_fn(x).float() * (1 + e1 + m1) + (e2 + m2)
    # torch.testing.assert_close(y_b10, y_torch, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(y_b10_base, y_b10, rtol=1e-5, atol=1e-5)