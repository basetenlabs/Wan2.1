import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append('/workdir/Wan2.2')
from wan.modules.model import B10FP4Linear
from wan.kernels.quantize_fp4 import quantize_fp4_triton

def test_b10fp4_linear_profile():
    batch_size = 1
    seq_len = 512
    in_features = 1024
    out_features = 1024
    block_size = 16
    
    model = B10FP4Linear(in_features, out_features, block_size=block_size, bias=True)
    model.eval()
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16, device='cuda')

    print(f"warmup...")
    for i in range(10):
        with torch.no_grad():
            output = model(x)
    
    print("forward pass profile...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("B10FP4Linear_forward"):
            with torch.no_grad():
                output = model(x)

    print(f"output: {output}")
    
    prof.export_chrome_trace('/workdir/Wan2.2/b10fp4_profile_trace.json')

if __name__ == "__main__":
    test_b10fp4_linear_profile()
