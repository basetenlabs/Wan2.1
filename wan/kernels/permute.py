import pytest
import torch
import triton
import triton.language as tl

# B = A.permute(0, 1, 2)
@triton.jit
def permute_mnk_kernel(
    a_ptr, b_ptr,
    M, N, K,
    stride_am, stride_an, stride_ak,   # strides for a: (M, N, K)
    stride_bn, stride_bm, stride_bk,   # strides for b: (N, M, K)
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BK):
        k_ids = k0 + offs_k
        mask_k = k_ids < K

        # pointers into a[m, n, k]
        a_ptrs = (
            a_ptr
            + (offs_m[:, None, None] * stride_am)
            + (offs_n[None, :, None] * stride_an)
            + (k_ids[None, None, :] * stride_ak)
        )
        # pointers into b[n, m, k]  (axes 0 and 1 swapped)
        b_ptrs = (
            b_ptr
            + (offs_n[None, :, None] * stride_bn)
            + (offs_m[:, None, None] * stride_bm)
            + (k_ids[None, None, :] * stride_bk)
        )

        mask = mask_m[:, None, None] & mask_n[None, :, None] & mask_k[None, None, :]
        vals = tl.load(a_ptrs, mask=mask, other=0)
        tl.store(b_ptrs, vals, mask=mask)


def b10_permute_mnk(a: torch.Tensor):
    M, N, K = a.shape
    if N == 1 or M == 1:
        return a
    b = torch.empty((N, M, K), device=a.device, dtype=a.dtype)

    # Grab (row-major) strides from PyTorch; Triton uses element strides (not bytes)
    sa_m, sa_n, sa_k = a.stride()
    sb_n, sb_m, sb_k = b.stride()

    # Tune these for your GPU / sizes
    BK = min(triton.next_power_of_2(K), 2048)
    BM, BN = 1, 4096 // BK
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    permute_mnk_kernel[grid](
        a, b, M, N, K,
        sa_m, sa_n, sa_k,
        sb_n, sb_m, sb_k,
        BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=2,
    )
    return b

@pytest.mark.parametrize("M", [18900])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("K", [1280])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_permute_mnk(M, N, K, dtype):
    x = torch.randn(M, N, K, dtype=dtype).cuda()
    y_b10_base = b10_permute_mnk(x)
    y_torch = x.permute(1, 0, 2).contiguous()
    torch.testing.assert_close(y_b10_base, y_torch, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
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
        "Permute MNK throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "N": 4,
            "K": 1280,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_permute_mnk(M, N, K, dtype, provider):
    """Benchmark permute_mnk throughput across different implementations."""
    x = torch.randn(M, N, K, dtype=dtype, device="cuda")

    def _b10_permute_mnk():
        return b10_permute_mnk(x)

    def torch_permute_mnk():
        return x.permute(1, 0, 2).contiguous()

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_permute_mnk)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_permute_mnk)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (M * N * K) * dtype.itemsize / ms * 1e-6
    return gb_s


if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_permute_mnk.run(print_data=True)
