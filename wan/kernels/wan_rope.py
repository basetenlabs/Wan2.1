from termios import B1000000
from sympy.printing.pretty.pretty_symbology import d
import torch
import triton
from typing import List
import triton.language as tl

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

@torch.amp.autocast('cuda', enabled=False)
@torch.profiler.record_function("rope_params")
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, sp_size, sp_rank):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        # sp_size = get_world_size()
        # sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class WanRope(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, grid_sizes: List[int], sp_size, sp_rank):
        return rope_apply(x, grid_sizes, self.freqs, sp_size, sp_rank)

@triton.jit
def _b10_wan_rope_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    FREQS, 
    N: tl.int64, # number of cols in X
    x_stride_s: tl.int64,
    x_stride_n: tl.int64,
    freqs_stride_s: tl.int64,
    BLOCK_N: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SAVE_FP16: tl.constexpr,
):
    BLOCK_D: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)

    compute_dtype = tl.float64
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * x_stride_s
    X += row * x_stride_s
    FREQS += row * freqs_stride_s
    rd = tl.arange(0, BLOCK_D)
    freqs = tl.load(FREQS + rd, mask=rd < ROTARY_DIM, other=0.)
    freqs_real, freqs_imag = tl.split(tl.reshape(freqs.to(compute_dtype), [1, BLOCK_D // 2, 2]))

    for off in range(0, N, BLOCK_N):
        off_n = off + tl.arange(0, BLOCK_N)
        cols = off_n[:, None] * x_stride_n + rd[None, :] # x_stride_d should be 1
        mask = (off_n[:, None] < N) & (rd[None, :] < ROTARY_DIM)
        x = tl.load(X + cols, mask=mask, other=0.)
        x_real, x_imag = tl.split(tl.reshape(x.to(compute_dtype), [BLOCK_N, BLOCK_D // 2, 2]))
        y_real = x_real * freqs_real - x_imag * freqs_imag
        y_imag = x_real * freqs_imag + x_imag * freqs_real
        y = tl.reshape(tl.join(y_real, y_imag), [BLOCK_N, BLOCK_D])
        if SAVE_FP16:
            tl.store(Y + cols, y.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Y + cols, y.to(x.dtype), mask=mask)


class B10WanRope(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        d = dim // num_heads
        self.d = d
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.dim = dim
        self.num_heads = num_heads
    
    @staticmethod
    def preprocess_freqs(freqs, grid_sizes, sp_size, sp_rank, seq_len_per_rank):
        c = freqs.shape[1]
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        grid_sizes = grid_sizes.tolist()
        assert len(grid_sizes) == 1, "For Now, Let us assume that bs=1"
        f, h, w = grid_sizes[0]
        seq_len = f * h * w
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        freqs_i = pad_freqs(freqs_i, seq_len_per_rank * sp_size)
        s_per_rank = seq_len_per_rank
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
        return torch.view_as_real(freqs_i_rank) # [s, 1, d//2, 2]
    
    @staticmethod
    def b10_rope_apply(x: torch.Tensor, freqs: torch.Tensor, save_fp16=False):
        B, S, N, D = x.shape
        assert B == 1, "For Now, B10WanRope only supports bs=1"
        assert x.is_contiguous(), "x must be contiguous"
        if save_fp16:
            dtype = torch.bfloat16
        else:
            dtype = x.dtype
        y = torch.empty_like(x, dtype=dtype)
        BLOCK_N = min(triton.next_power_of_2(N), 4096 // triton.next_power_of_2(D))
        x_, freqs_ = x.view(B*S, N, D), freqs.view(B*S, D)
        # print(f"{x_.shape=}, {freqs_.shape=}, {x_.stride(0)=}, {x_.stride(1)=}, {freqs_.stride(0)=}")
        _b10_wan_rope_fused[(x_.shape[0],)](
            x_, 
            y.view(B*S, N, D), 
            freqs_,
            N, 
            x_.stride(0), 
            x_.stride(1),
            freqs_.stride(0),
            BLOCK_N,
            D,
            SAVE_FP16=save_fp16)
        return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["F"],
        x_vals=[16, 32],  # different possible values for `x_name`
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
        "Rope throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "H": 45,
            "W": 80,
            "D": 5120,
            "num_heads": 40,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_rms_norm( F, H, W, D, num_heads, dtype, provider):
    """Benchmark GroupNorm throughput across different implementations."""
    device = torch.device("cuda")
    sp_size, sp_rank = 4, 0
    x = torch.randn(1, F * H * W // sp_size, num_heads, D // num_heads, device=device, dtype=dtype)
    grid_sizes = torch.tensor([[F, H, W]])
    rope = WanRope(dim, num_heads)
    rope.freqs = rope.freqs.cuda()

    b10_rope = B10WanRope(dim, num_heads)
    b10_rope.freqs = b10_rope.freqs.cuda()
    freqs_i_rank = B10WanRope.preprocess_freqs(b10_rope.freqs, grid_sizes, sp_size, sp_rank, F * H * W // sp_size)

    def _torch_rope():
        return rope(x, grid_sizes, sp_size, sp_rank)

    def _b10_rope():
        return b10_rope.b10_rope_apply(x, freqs_i_rank)

    if provider == "b10":
        ms = triton.testing.do_bench(_b10_rope)
    elif provider == "torch":
        ms = triton.testing.do_bench(_torch_rope)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (2 * F * H * W * D) // sp_size * dtype.itemsize / ms * 1e-6
    return gb_s

if __name__ == "__main__":
    f, h, w, dim, num_heads = 21, 45, 80, 5120, 40
    sp_size, sp_rank = 4, 0
    x = torch.randn(1, f * h * w // sp_size, num_heads, dim // num_heads).cuda()
    print("finished generating x with shape", x.shape)
    grid_sizes = torch.tensor([[f, h, w]])
    rope = WanRope(dim, num_heads)
    rope.freqs = rope.freqs.cuda()
    base_result = rope(x, grid_sizes, sp_size, sp_rank)
    print("finished generating rope with shape", rope.freqs.shape)

    b10_rope = B10WanRope(dim, num_heads)
    b10_rope.freqs = b10_rope.freqs.cuda()
    freqs_i_rank = B10WanRope.preprocess_freqs(b10_rope.freqs, grid_sizes, sp_size, sp_rank, f * h * w // sp_size)
    b10_result = b10_rope.b10_rope_apply(x, freqs_i_rank)
    print("finished generating b10_rope with shape", freqs_i_rank.shape)

    assert torch.allclose(base_result, b10_result), "base_result and b10_result are not close"
    torch.testing.assert_close(base_result, b10_result)
    benchmark_rms_norm.run(print_data=True)