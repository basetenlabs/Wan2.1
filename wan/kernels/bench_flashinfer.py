#!/usr/bin/env python3
"""
FlashInfer benchmark using triton.benchmark
"""
import torch
import triton
from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from sgl_kernel.gemm import scaled_fp4_grouped_quant

def benchmark():
    """Benchmark FlashInfer kernels with different M sizes"""
    print("=== FlashInfer FP4 GEMM Benchmark ===")
    
    # Test different M sizes
    M_sizes = [1024, 4096, 16384, 65536, 131072]
    N, K = 2048, 8192
    
    print(f"Testing with N={N}, K={K}")
    print("M\tPyTorch (ms)\t\tFlashInfer (ms)\t\tGrouped (ms)\t\tSpeedup1\t\tSpeedup2")
    print("-" * 120)
    
    for M in M_sizes:
        # Create test tensors
        a = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
        b = torch.randn([N, K], device="cuda", dtype=torch.bfloat16)
        
        # Pre-quantize for FlashInfer (outside benchmark)
        a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
        b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
        a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True)
        
        # Pre-quantize for Grouped GEMM (outside benchmark)
        L = 1
        masked_m = torch.ones([L], device="cuda", dtype=torch.int32).fill_(M)
        masked_n = torch.ones([L], device="cuda", dtype=torch.int32).fill_(N)
        a_grouped = a.view(L, M, K)
        b_grouped = b.view(L, N, K)
        a_global_sf_grouped = a_global_sf.view(L)
        b_global_sf_grouped = b_global_sf.view(L)
        a_fp4_grouped, a_sf_grouped = scaled_fp4_grouped_quant(a_grouped, a_global_sf_grouped, masked_m)
        b_fp4_grouped, b_sf_grouped = scaled_fp4_grouped_quant(b_grouped, b_global_sf_grouped, masked_n)
        
        # Benchmark PyTorch
        def pytorch_gemm():
            return torch.matmul(a, b.T)
        
        pytorch_ms = triton.testing.do_bench(pytorch_gemm, warmup=10, rep=100)
        
        # Benchmark FlashInfer (only GEMM)
        def flashinfer_gemm():
            return mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1.0/(a_global_sf * b_global_sf), torch.bfloat16, None, backend="trtllm")
        
        flashinfer_ms = triton.testing.do_bench(flashinfer_gemm, warmup=10, rep=100)
        
        # Benchmark Grouped GEMM (only GEMM)
        def grouped_gemm():
            out_cute = torch.zeros([L, M, N], device="cuda", dtype=torch.bfloat16).permute(1, 2, 0)
            grouped_gemm_nt_masked(
                (a_fp4_grouped, a_sf_grouped),
                (b_fp4_grouped, b_sf_grouped),
                out_cute,
                masked_m,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                alpha=1.0/(a_global_sf_grouped * b_global_sf_grouped),
                alpha_dtype="float32",
            )
            return out_cute
        
        grouped_ms = triton.testing.do_bench(grouped_gemm, warmup=10, rep=100)
        
        # Calculate speedups
        speedup1 = pytorch_ms / flashinfer_ms
        speedup2 = pytorch_ms / grouped_ms
        
        print(f"{M:8d}\t{pytorch_ms:8.2f}\t\t{flashinfer_ms:8.2f}\t\t{grouped_ms:8.2f}\t\t{speedup1:8.2f}\t{speedup2:8.2f}")

if __name__ == "__main__":
    benchmark()
