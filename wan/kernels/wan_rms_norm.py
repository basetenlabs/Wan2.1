import pytest
import torch
import torch.nn as nn
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _rms_norm_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride: tl.int64,  # how much to increase the pointer when moving by 1 row
    N: tl.int64,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    _rms = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _rms += tl.sum(x * x, axis=0)
    
    rms = tl.sqrt(_rms / N + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.)
        x_rms = x.to(tl.float32) / rms
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        y = x_rms * w
        tl.store(Y + cols, y.to(x.dtype), mask=mask)


class B10RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.profiler.record_function("B10RMSNorm")
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        N = x.shape[-1]
        y = torch.empty_like(x)
        x_, y_ = x.view(-1, N), y.view(-1, N)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        _rms_norm_fused[(x_.shape[0],)](x_, y_, self.weight, x_.stride(0), N, self.eps, BLOCK_SIZE)
        return y

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.profiler.record_function("WanRMSNorm")
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("S", [128, 18900])
@pytest.mark.parametrize("H", [5120])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_norm(B, S, H, eps, dtype):
    x = torch.randn(B, S, H, dtype=dtype).cuda()
    b10_norm_fn = B10RMSNorm(x.shape[-1], eps=eps).cuda()
    torch_norm_fn = WanRMSNorm(x.shape[-1], eps=eps).cuda()
    random_weight = torch.empty_like(torch_norm_fn.weight).random_(-5, 5).to(dtype).cuda()
    b10_norm_fn.weight.data = random_weight
    torch_norm_fn.weight.data = random_weight
    y_b10 = b10_norm_fn(x)
    y_torch = torch_norm_fn(x)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(y_b10, y_torch, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(y_b10, y_torch, rtol=1e-3, atol=1e-3)


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
        "RMSNorm throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 5120,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_rms_norm(B, S, H, dtype, provider):
    """Benchmark RMSNorm throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    eps = 1e-5
    b10_norm_fn = B10RMSNorm(x.shape[-1], eps=eps).cuda()
    torch_norm_fn = WanRMSNorm(x.shape[-1], eps=eps).cuda()
    random_weight = torch.randn_like(torch_norm_fn.weight).cuda()
    b10_norm_fn.weight.data = random_weight
    torch_norm_fn.weight.data = random_weight

    def b10_norm():
        return b10_norm_fn(x)

    def torch_norm():
        return torch_norm_fn(x)

    if provider == "b10":
        ms = triton.testing.do_bench(b10_norm)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_norm)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (2 * S * H) * 4 / ms * 1e-6
    return gb_s


if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_rms_norm.run(print_data=True)
