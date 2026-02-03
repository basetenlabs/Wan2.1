import torch
from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
# from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from fp4_blockscaled_gemm import grouped_gemm_nt_masked
from sgl_kernel.gemm import scaled_fp4_grouped_quant

torch.manual_seed(0)

L = 1
M = 18900
N = 5120
K = 13824

a = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
b = torch.randn([N, K], device="cuda", dtype=torch.bfloat16)
bias = torch.randn([N], device="cuda", dtype=torch.bfloat16)

c = torch.matmul(a, b.T)
c = c + bias.view(1, N).expand(M, N)

print(c)

a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True)
out = mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1.0/(a_global_sf * b_global_sf), torch.bfloat16, None, backend="trtllm")

out = out + bias.view(1, N).expand(M, N)
print(out)

def ceil_div(a, b):
    return (a + b - 1) // b

masked_m = torch.ones([L], device="cuda", dtype=torch.int32).fill_(M)
masked_n = torch.ones([L], device="cuda", dtype=torch.int32).fill_(N)
a = a.view(L, M, K)
b = b.view(L, N, K)
a_global_sf = a_global_sf.view(L)
b_global_sf = b_global_sf.view(L)
a_fp4, a_sf = scaled_fp4_grouped_quant(a, a_global_sf, masked_m)
b_fp4, b_sf = scaled_fp4_grouped_quant(b, b_global_sf, masked_n)

print(f"a_fp4 shape: {a_fp4.shape}, {a_fp4.dtype}, contiguous: {a_fp4.is_contiguous()}, stride: {a_fp4.stride()}")
print(f"a_sf shape: {a_sf.shape}, {a_sf.dtype}, contiguous: {a_sf.is_contiguous()}, stride: {a_sf.stride()}")

out_cute = torch.zeros([L, M, N], device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)

grouped_gemm_nt_masked(
    (a_fp4, a_sf),
    (b_fp4, b_sf),
    out_cute,
    masked_m,
    ab_dtype="float4_e2m1fn",
    sf_dtype="float8_e4m3fn",
    c_dtype="bfloat16",
    sf_vec_size=16,
    alpha=1.0/(a_global_sf * b_global_sf),
    alpha_dtype="float32",
    bias=bias,
    bias_dtype="bfloat16",
) 

print(out_cute.permute(2, 0, 1).view(M, N))
