import torch
import pytest
import triton
import triton.language as tl
import torch.nn.functional as F

import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl


@triton.jit
def _zero_pad5d_kernel(
    x_ptr, y_ptr, z_ptr,
    N, C, D, H, W,                 # input sizes
    PD_FRONT, PD_BACK,              # depth padding
    PH_FRONT, PH_BACK,              # height padding
    PW_FRONT, PW_BACK,              # width padding
    zD: tl.int64,
    sxN, sxC, sxD, sxH, sxW,        # input strides (elements)
    syN, syC, syD, syH, syW,        # output strides (elements)
    szN, szC, szD, szH, szW,        # cache_x strides (elements)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # program ids:
    #  - pid_w tiles over W_out
    #  - pid_h tiles over H_out
    #  - pid_z tiles over (D_out tiles) × (N*C)
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    D_out = D + PD_FRONT + PD_BACK
    H_out = H + PH_FRONT + PH_BACK
    W_out = W + PW_FRONT + PW_BACK

    # decode pid_z -> (d_tile_id, n, c)
    d_tile = pid_z % D_out
    nc_id = pid_z // D_out
    n = nc_id // C
    c = nc_id % C

    # tile coordinates in output
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    od = d_tile
    out_mask = (ow < W_out)[None, :] & (oh < H_out)[:, None]
    OW = ow[None, :]
    OH = oh[:, None]

    IW = OW - PW_FRONT
    IH = OH - PH_FRONT
    in_mask_w = (IW >= 0) & (IW < W)
    in_mask_h = (IH >= 0) & (IH < H)
    in_mask = in_mask_w & in_mask_h

    if d_tile < PD_FRONT - zD or d_tile >= D_out - PD_BACK:
        ID = (d_tile).to(tl.int64)
        x_ptrs = (
            x_ptr
            + n * sxN
            + c * sxC
            + ID * sxD
            + IH * sxH
            + IW * sxW
        )
        vals = tl.zeros([BLOCK_H, BLOCK_W], dtype=DTYPE)
    elif d_tile >= PD_FRONT - zD and d_tile < PD_FRONT:
        ID = (d_tile - PD_FRONT + zD).to(tl.int64)
        x_ptrs = (
            z_ptr
            + n * szN
            + c * szC
            + ID * szD
            + IH * szH
            + IW * szW
        )
        vals = tl.load(x_ptrs, mask=in_mask, other=0.0)
    else:
        ID = (d_tile - PD_FRONT).to(tl.int64)
        x_ptrs = (
            x_ptr
            + n * sxN
            + c * sxC
            + ID * sxD
            + IH * sxH
            + IW * sxW
        )
        vals = tl.load(x_ptrs, mask=in_mask, other=0.0)

    y_ptrs = (
        y_ptr
        + n * syN
        + c * syC
        + od * syD
        + OH * syH
        + OW * syW
    )
    tl.store(y_ptrs, vals, mask=out_mask)


def b10_zero_pad5d(x: torch.Tensor, pad: tuple[int, int, int, int, int, int], z=None):
    """
    Zero-pad a 5D tensor (N, C, D, H, W) along D/H/W.
    pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back)
    """
    assert x.ndim == 5, "Expected 5D NCDHW tensor"
    pwf, pwb, phf, phb, pdf, pdb = pad
    N, C, D, H, W = x.shape

    D_out = D + pdf + pdb
    H_out = H + phf + phb
    W_out = W + pwf + pwb

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    sxN, sxC, sxD, sxH, sxW = x.stride()
    syN, syC, syD, syH, syW = y.stride()
    szN, szC, szD, szH, szW = 0, 0, 0, 0, 0
    assert x.dtype == torch.float32 or x.dtype == torch.bfloat16, f"{x.dtype=}, support only float32 and bfloat16"
    if z is not None:
        zN, zC, zD, zH, zW = z.shape
        assert zN == N and zC == C and zH == H and zW == W, f"{z.shape=}, {x.shape=}"
        assert zD <= pdf, f"{zD=} > {pdf=}"
        szN, szC, szD, szH, szW = z.stride()
    else:
        z = x
        zD = 0

    BLOCK_H, BLOCK_W = 4, min(triton.next_power_of_2(W_out), 1024)  # good starting point; tune per workload

    grid = (
        triton.cdiv(W_out, BLOCK_W),                # along W_out
        triton.cdiv(H_out, BLOCK_H),                # along H_out
        D_out * N * C,        # pack D-tiles with (N*C)
    )

    _zero_pad5d_kernel[grid](
        x, y, z,
        N, C, D, H, W,
        pdf, pdb, phf, phb, pwf, pwb, zD,
        sxN, sxC, sxD, sxH, sxW,
        syN, syC, syD, syH, syW,
        szN, szC, szD, szH, szW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, DTYPE=tl.float32 if x.dtype == torch.float32 else tl.bfloat16,
        num_warps=4, num_stages=2,
    )
    return y


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("C", [16])
@pytest.mark.parametrize("D", [21])
@pytest.mark.parametrize("H", [90])
@pytest.mark.parametrize("W", [160])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1, 2, 0)])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_zero_pad5d(B, C, D, H, W, padding, use_cache, dtype):
    x = torch.randn(B, C, D, H, W, dtype=dtype).cuda()
    cache_x = torch.randn(B, C, 2, H, W, dtype=dtype).cuda() if use_cache else None
    y_b10_base = b10_zero_pad5d(x, padding, z=cache_x)
    if use_cache:
        y_torch = torch.cat([cache_x, x], dim=2)
        wl, wr, hl, hr, dl, dr = padding
        y_torch = F.pad(y_torch, (wl, wr, hl, hr, dl-2, dr))
    else:
        y_torch = F.pad(x, padding)
    torch.testing.assert_close(y_b10_base, y_torch, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],
        x_vals=[21, 81],  # different possible values for `x_name`
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
            "B": 1,
            "C": 3,
            "H": 720,
            "W": 1280,
            "padding": (1, 1, 1, 1, 2, 0),
            "use_cache": True,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_zero_pad5d(B, C, D, H, W, padding, use_cache, dtype, provider):
    """Benchmark ZeroPad5D throughput across different implementations."""
    x = torch.randn(B, C, D, H, W, dtype=dtype, device="cuda")
    if use_cache:
        cache_x = torch.randn(B, C, 2, H, W, dtype=dtype, device="cuda")
    else:
        cache_x = None

    def _b10_zero_pad5d():
        return b10_zero_pad5d(x, padding, z=cache_x)

    def torch_zero_pad5d():
        if not use_cache:
            return F.pad(x, padding)
        wl, wr, hl, hr, dl, dr = padding
        cache_x = torch.randn(B, C, 2, H, W, dtype=dtype, device="cuda")
        return F.pad(torch.cat([cache_x, x], dim=2), (wl, wr, hl, hr, dl-2, dr))

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_zero_pad5d)
    elif provider == "torch":
        ms = triton.testing.do_bench(torch_zero_pad5d)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (B * C * D * H * W) * dtype.itemsize / ms * 1e-6
    return gb_s

if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_zero_pad5d.run(print_data=True)