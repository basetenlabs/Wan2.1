import os

ENABLE_B10_KERNEL = os.getenv('ENABLE_B10_KERNEL', '1') == '1'

ENABLE_B10_FP4_LINEAR_FFN = os.getenv('ENABLE_B10_FP4_LINEAR_FFN', '0') == '1'
ENABLE_B10_FP4_LINEAR_ATTN = os.getenv('ENABLE_B10_FP4_LINEAR_ATTN',
                                       '0') == '1'
LOAD_GLOBAL_SF = os.getenv('LOAD_GLOBAL_SF', '0') == '1'
ENABLE_CALIBRATION = os.getenv('ENABLE_CALIBRATION', '0') == '1'
CALIBRATION_OUTPUT_PATH = os.getenv('CALIBRATION_OUTPUT_PATH',
                                    'global_sfs.txt')

ENABLE_B10_ATTN_CACHE = os.getenv('ENABLE_B10_ATTN_CACHE', '0') == '1'
ENABLE_PARALLEL_DECODER = os.getenv('ENABLE_PARALLEL_DECODER',
                                    '0') == '1'


def set_b10_kernel(enable):
    global ENABLE_B10_KERNEL
    ENABLE_B10_KERNEL = enable


def enable_b10_kernel():
    global ENABLE_B10_KERNEL
    return ENABLE_B10_KERNEL


def set_b10_fp4_linear_ffn(enable):
    global ENABLE_B10_FP4_LINEAR_FFN
    ENABLE_B10_FP4_LINEAR_FFN = enable


def enable_b10_fp4_linear_ffn():
    global ENABLE_B10_FP4_LINEAR_FFN
    return ENABLE_B10_FP4_LINEAR_FFN


def set_b10_fp4_linear_attn(enable):
    global ENABLE_B10_FP4_LINEAR_ATTN
    ENABLE_B10_FP4_LINEAR_ATTN = enable


def enable_b10_fp4_linear_attn():
    global ENABLE_B10_FP4_LINEAR_ATTN
    return ENABLE_B10_FP4_LINEAR_ATTN


def set_load_global_sf(enable):
    global LOAD_GLOBAL_SF
    LOAD_GLOBAL_SF = enable


def load_global_sf():
    global LOAD_GLOBAL_SF
    return LOAD_GLOBAL_SF


def set_enable_calibration(enable):
    global ENABLE_CALIBRATION
    ENABLE_CALIBRATION = enable


def enable_calibration():
    global ENABLE_CALIBRATION
    return ENABLE_CALIBRATION


def set_calibration_output_path(path):
    global CALIBRATION_OUTPUT_PATH
    CALIBRATION_OUTPUT_PATH = path


def calibration_output_path():
    global CALIBRATION_OUTPUT_PATH
    return CALIBRATION_OUTPUT_PATH


def set_b10_attn_cache(enable):
    global ENABLE_B10_ATTN_CACHE
    ENABLE_B10_ATTN_CACHE = enable


def enable_b10_attn_cache():
    global ENABLE_B10_ATTN_CACHE
    return ENABLE_B10_ATTN_CACHE


def set_parallel_decoder(enable):
    global ENABLE_PARALLEL_DECODER
    ENABLE_PARALLEL_DECODER = enable


def enable_parallel_decoder():
    global ENABLE_PARALLEL_DECODER
    return ENABLE_PARALLEL_DECODER
