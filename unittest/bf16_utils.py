"""
BF16 helpers shared by the BF16-specific unit test suite.

Provides conversions between float32 arrays and the uint16 bit patterns
expected by the BF16 kernels, along with a ctypes pointer helper.
"""
import ctypes

import numpy as np


def float32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float32 numpy array to the raw uint16 bit pattern used by BF16.

    The conversion matches `float_to_bf16` in `include/bf16_utils.h` (simply shifts
    the top 16 bits) so kernels and reference code stay consistent.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    bits = arr.view(np.uint32)
    return np.right_shift(bits, 16).astype(np.uint16)


def bf16_to_float32(arr: np.ndarray) -> np.ndarray:
    """
    Convert a BF16 bit pattern array back to float32.
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint16)
    bits = arr.astype(np.uint32)
    return np.left_shift(bits, 16).view(np.float32)


def numpy_to_uint16_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint16):
    """
    Return a ctypes pointer to a uint16 numpy array (used for BF16 inputs/outputs).
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint16)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
