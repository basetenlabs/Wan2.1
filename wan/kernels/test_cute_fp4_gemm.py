import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from mxfp4_util import MXFP4QuantizeUtil
from fp4_gemm import Sm100BlockScaledPersistentDenseGemmKernel

def ceil_div(a, b):
    return (a + b - 1) // b

torch.manual_seed(42)

M = 1024 
N = 128
K = 256
L = 1

sf_vec_size = 16

x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
y = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
quantized_x, scale_x = MXFP4QuantizeUtil.quantize(x, sf_vec_size)
scale_x = scale_x.reshape(M, ceil_div(K, sf_vec_size))
quantized_y, scale_y = MXFP4QuantizeUtil.quantize(y, sf_vec_size)
scale_y = scale_y.reshape(N, ceil_div(K, sf_vec_size))

@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

def convert(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sf_a_tensor: torch.Tensor,
    sf_b_tensor: torch.Tensor,
    sf_vec_size: int,
):

    def unfuse_uint8_to_uint4(x):
        # Extract the lower 4 bits (even indices)
        left_side = x & 0x0F
        # Extract the upper 4 bits (odd indices)  
        right_side = (x >> 4) & 0x0F
        
        # Create a new tensor with alternating values
        shape = list(x.shape)
        shape[-1] = shape[-1] * 2
        result = torch.zeros(shape, dtype=torch.uint8, device=x.device)
        
        # Fill in the values - even indices get low bits, odd indices get high bits
        result[..., 0::2] = left_side  # Even indices from low bits
        result[..., 1::2] = right_side  # Odd indices from high bits
        
        return result

    def create_fp4_tensor(tensor):
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            tensor,
            cutlass.Float4E2M1FN,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        return cute_tensor
    
    m, k = a_tensor.shape
    n, k = b_tensor.shape
    l = 1
    a_tensor = a_tensor.view(l, m, k).permute(1, 2, 0)
    b_tensor = b_tensor.view(l, n, k).permute(1, 2, 0)

    # a_fp4 = unfuse_uint8_to_uint4(a_tensor).permute(1, 2, 0) # uint8, (m, k, l)
    # b_fp4 = unfuse_uint8_to_uint4(b_tensor).permute(1, 2, 0) # uint8, (n, k, l)

    # cute_a_tensor = from_dlpack(a_fp4, assumed_align=16)
    # cute_b_tensor = from_dlpack(b_fp4, assumed_align=16)

    cute_a_tensor = create_fp4_tensor(a_tensor)
    cute_b_tensor = create_fp4_tensor(b_tensor)
    
    cute_a_tensor.element_type = cutlass.Float4E2M1FN
    cute_b_tensor.element_type = cutlass.Float4E2M1FN

    def create_scale_factor_tensor(ori_sf_tensor):

        mn, sf_k, l = ori_sf_tensor.shape

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # (atom_m, atom_k, rest_m, atom_k, rest_k, l)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        cute_sf_mma_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.uint8,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
        )

        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ori_sf_tensor),
            from_dlpack(cute_sf_mma_torch_tensor),
        )

        return cute_sf_mma_torch_tensor


    sf_a_tensor = sf_a_tensor.view(l, m, ceil_div(k, sf_vec_size)).contiguous().permute(1, 2, 0).cpu()
    sf_b_tensor = sf_b_tensor.view(l, n, ceil_div(k, sf_vec_size)).contiguous().permute(1, 2, 0).cpu()

    sf_a_tensor = create_scale_factor_tensor(sf_a_tensor)
    sf_b_tensor = create_scale_factor_tensor(sf_b_tensor)

    cute_sf_a_tensor = from_dlpack(sf_a_tensor.cuda(), assumed_align=16)
    cute_sf_b_tensor = from_dlpack(sf_b_tensor.cuda(), assumed_align=16)

    cute_sf_a_tensor.element_type = cutlass.Float8E8M0FNU
    cute_sf_b_tensor.element_type = cutlass.Float8E8M0FNU

    return cute_a_tensor, cute_b_tensor, cute_sf_a_tensor, cute_sf_b_tensor


quantized_x_cute, quantized_y_cute, scale_x_cute, scale_y_cute = convert(quantized_x, quantized_y, scale_x, scale_y, sf_vec_size)
print(f"quantized_x_cute shape: {quantized_x_cute.shape}, quantized_x_cute dtype: {quantized_x_cute.element_type}")
print(f"quantized_y_cute shape: {quantized_y_cute.shape}, quantized_y_cute dtype: {quantized_y_cute.element_type}")
print(f"scale_x_cute shape: {scale_x_cute.shape}, scale_x_cute dtype: {scale_x_cute.element_type}")
print(f"scale_y_cute shape: {scale_y_cute.shape}, scale_y_cute dtype: {scale_y_cute.element_type}")

print(f"x: {x}")

@cute.kernel
def print_tensor_gpu(tensor: cute.Tensor):
    cute.print_tensor(tensor)

@cute.jit
def print_tensor_host(src: cute.Tensor):
    print_tensor_gpu(src).launch(grid=(1,1,1), block=(1,1,1))
    # cute.print_tensor(src)

def tensor_print():
    print_tensor_host(quantized_x_cute)

# tensor_print()


mma_tiler_mn = (256, 128)
cluster_shape_mn = (2, 1)

gemm = Sm100BlockScaledPersistentDenseGemmKernel(
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
)

# Compute max active clusters on current device
hardware_info = cutlass.utils.HardwareInfo()
max_active_clusters = hardware_info.get_max_active_clusters(
    cluster_shape_mn[0] * cluster_shape_mn[1]
)

# Initialize Stream
current_stream = cutlass_torch.default_stream()

c_tensor_torch = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda').view(L, M, N).permute(1, 2, 0)
c_tensor = from_dlpack(c_tensor_torch)

print(f"quantized_x_cute stride: {quantized_x_cute.stride}")
print(f"quantized_y_cute stride: {quantized_y_cute.stride}")
print(f"scale_x_cute stride: {scale_x_cute.stride}")
print(f"scale_y_cute stride: {scale_y_cute.stride}")
print(f"c_tensor stride: {c_tensor.stride}")

compiled_gemm = cute.compile(
    gemm,
    quantized_x_cute,
    quantized_y_cute,
    scale_x_cute,
    scale_y_cute,
    c_tensor,
    max_active_clusters,
    current_stream,
)

compiled_gemm(
    quantized_x_cute, quantized_y_cute, scale_x_cute, scale_y_cute, c_tensor, current_stream
)

torch.cuda.synchronize()

c_ref = torch.matmul(x, y.T)

print(f"c_ref: {c_ref} c_ref dtype: {c_ref.dtype} c_ref shape: {c_ref.shape}")

dequantized_x = MXFP4QuantizeUtil.dequantize(quantized_x, torch.bfloat16, scale_x.reshape(M*ceil_div(K, sf_vec_size)), [16])
dequantized_y = MXFP4QuantizeUtil.dequantize(quantized_y, torch.bfloat16, scale_y.reshape(N*ceil_div(K, sf_vec_size)), [16])
c_ref1 = torch.matmul(dequantized_x, dequantized_y.T)

print(f"c_ref1: {c_ref1} c_ref1 dtype: {c_ref1.dtype} c_ref1 shape: {c_ref1.shape}")

c_tensor_torch = c_tensor_torch.view(M, N)

print(f"c_tensor_torch: {c_tensor_torch} c_tensor_torch dtype: {c_tensor_torch.dtype} c_tensor_torch shape: {c_tensor_torch.shape}")