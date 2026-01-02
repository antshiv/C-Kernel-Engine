"""
Add kernel unit tests with PyTorch parity verification.

Tests element-wise addition kernels (for residual connections):
  - add_forward_bf16: y = a + b
  - add_scaled_forward_bf16: y = a + alpha * b
  - add_inplace_bf16: a += b
  - add_backward_bf16: gradient passthrough
  - add_forward_2d_bf16: 2D tensor version

Reports accuracy against PyTorch reference and timing.
"""
import ctypes
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_add.so", "libckernel_engine.so")


# =============================================================================
# BF16 utility functions (match C implementation)
# =============================================================================

def float_to_bf16(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to bf16 (stored as uint16)."""
    # Truncate lower 16 bits of float32 mantissa
    x_f32 = x.astype(np.float32)
    x_bytes = x_f32.view(np.uint32)
    x_bf16 = (x_bytes >> 16).astype(np.uint16)
    return x_bf16


def bf16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert bf16 (stored as uint16) back to float32."""
    x_u32 = x.astype(np.uint32) << 16
    return x_u32.view(np.float32)


# =============================================================================
# Function signatures
# =============================================================================

# void add_forward_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t n)
lib.add_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_size_t,                   # n
]
lib.add_forward_bf16.restype = None

# void add_scaled_forward_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, float alpha, size_t n)
lib.add_scaled_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_float,                    # alpha
    ctypes.c_size_t,                   # n
]
lib.add_scaled_forward_bf16.restype = None

# void add_inplace_bf16(uint16_t *a, const uint16_t *b, size_t n)
lib.add_inplace_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.c_size_t,                   # n
]
lib.add_inplace_bf16.restype = None

# void add_backward_bf16(const uint16_t *d_y, uint16_t *d_a, uint16_t *d_b, size_t n)
lib.add_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # d_y
    ctypes.POINTER(ctypes.c_uint16),  # d_a
    ctypes.POINTER(ctypes.c_uint16),  # d_b
    ctypes.c_size_t,                   # n
]
lib.add_backward_bf16.restype = None

# void add_forward_2d_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, int tokens, int dim, int aligned_dim)
lib.add_forward_2d_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # a
    ctypes.POINTER(ctypes.c_uint16),  # b
    ctypes.POINTER(ctypes.c_uint16),  # y
    ctypes.c_int,                      # tokens
    ctypes.c_int,                      # dim
    ctypes.c_int,                      # aligned_dim
]
lib.add_forward_2d_bf16.restype = None

# FP32 versions for comparison
lib.add_forward_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # a
    ctypes.POINTER(ctypes.c_float),   # b
    ctypes.POINTER(ctypes.c_float),   # y
    ctypes.c_size_t,                   # n
]
lib.add_forward_f32.restype = None


def numpy_to_ptr_u16(arr: np.ndarray):
    """Get ctypes pointer to uint16 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def numpy_to_ptr_f32(arr: np.ndarray):
    """Get ctypes pointer to float32 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# =============================================================================
# Test: Forward BF16 (y = a + b)
# =============================================================================

def test_forward_bf16(N=4096, warmup=10, iterations=100):
    """Test add_forward_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_forward_bf16 (y = a + b)")
    print("=" * 60)

    np.random.seed(42)

    # Generate random test data
    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    # Convert to BF16
    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)
    y_bf16 = np.zeros(N, dtype=np.uint16)

    # Get actual BF16 values (with truncation)
    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference (using actual BF16 values)
    a_torch = torch.from_numpy(a_bf16_f32)
    b_torch = torch.from_numpy(b_bf16_f32)
    ref = (a_torch + b_torch).numpy()

    # C kernel
    lib.add_forward_bf16(
        numpy_to_ptr_u16(a_bf16),
        numpy_to_ptr_u16(b_bf16),
        numpy_to_ptr_u16(y_bf16),
        ctypes.c_size_t(N)
    )

    # Convert result back to float for comparison
    y_f32 = bf16_to_float(y_bf16)

    # Compute error
    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    rel_err = max_err / (np.max(np.abs(ref)) + 1e-10)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")
    print(f"  Rel error: {rel_err:.2e}")

    # Timing
    def c_add():
        lib.add_forward_bf16(
            numpy_to_ptr_u16(a_bf16),
            numpy_to_ptr_u16(b_bf16),
            numpy_to_ptr_u16(y_bf16),
            ctypes.c_size_t(N)
        )

    def torch_add():
        return a_torch + b_torch

    # Warmup
    for _ in range(warmup):
        c_add()
        torch_add()

    # Time C kernel
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        c_add()
    c_time = (time.perf_counter() - start) / iterations * 1e6  # microseconds

    # Time PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        torch_add()
    torch_time = (time.perf_counter() - start) / iterations * 1e6

    print(f"  C kernel:  {c_time:.2f} us")
    print(f"  PyTorch:   {torch_time:.2f} us")
    print(f"  Speedup:   {torch_time/c_time:.2f}x")

    # Pass/Fail
    # BF16 has ~3 decimal digits of precision, so we allow some error
    passed = max_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: {status}")

    return passed


# =============================================================================
# Test: Scaled Forward BF16 (y = a + alpha * b)
# =============================================================================

def test_scaled_forward_bf16(N=4096, alpha=0.5):
    """Test add_scaled_forward_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print(f"TEST: add_scaled_forward_bf16 (y = a + {alpha} * b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)
    y_bf16 = np.zeros(N, dtype=np.uint16)

    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference
    ref = a_bf16_f32 + alpha * b_bf16_f32

    # C kernel
    lib.add_scaled_forward_bf16(
        numpy_to_ptr_u16(a_bf16),
        numpy_to_ptr_u16(b_bf16),
        numpy_to_ptr_u16(y_bf16),
        ctypes.c_float(alpha),
        ctypes.c_size_t(N)
    )

    y_f32 = bf16_to_float(y_bf16)

    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"  Shape:     N={N}, alpha={alpha}")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")

    passed = max_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: Inplace BF16 (a += b)
# =============================================================================

def test_inplace_bf16(N=4096):
    """Test add_inplace_bf16 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_inplace_bf16 (a += b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32)
    b_bf16 = float_to_bf16(b_f32)

    a_bf16_f32 = bf16_to_float(a_bf16)
    b_bf16_f32 = bf16_to_float(b_bf16)

    # PyTorch reference
    ref = a_bf16_f32 + b_bf16_f32

    # C kernel (modifies a in-place)
    a_bf16_copy = a_bf16.copy()
    lib.add_inplace_bf16(
        numpy_to_ptr_u16(a_bf16_copy),
        numpy_to_ptr_u16(b_bf16),
        ctypes.c_size_t(N)
    )

    y_f32 = bf16_to_float(a_bf16_copy)

    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")

    passed = max_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: Backward BF16 (gradient passthrough)
# =============================================================================

def test_backward_bf16(N=4096):
    """Test add_backward_bf16 - gradients should pass through unchanged."""
    print("\n" + "=" * 60)
    print("TEST: add_backward_bf16 (gradient passthrough)")
    print("=" * 60)

    np.random.seed(42)

    # For y = a + b:
    #   d_a = d_y (gradient passes through)
    #   d_b = d_y (gradient passes through)

    d_y_f32 = np.random.randn(N).astype(np.float32)
    d_y_bf16 = float_to_bf16(d_y_f32)

    d_a_bf16 = np.zeros(N, dtype=np.uint16)
    d_b_bf16 = np.zeros(N, dtype=np.uint16)

    # C kernel
    lib.add_backward_bf16(
        numpy_to_ptr_u16(d_y_bf16),
        numpy_to_ptr_u16(d_a_bf16),
        numpy_to_ptr_u16(d_b_bf16),
        ctypes.c_size_t(N)
    )

    # Both d_a and d_b should equal d_y
    d_a_f32 = bf16_to_float(d_a_bf16)
    d_b_f32 = bf16_to_float(d_b_bf16)
    d_y_actual = bf16_to_float(d_y_bf16)

    diff_a = np.abs(d_a_f32 - d_y_actual)
    diff_b = np.abs(d_b_f32 - d_y_actual)

    max_err_a = np.max(diff_a)
    max_err_b = np.max(diff_b)

    print(f"  Shape:        N={N}")
    print(f"  Max diff d_a: {max_err_a:.2e}")
    print(f"  Max diff d_b: {max_err_b:.2e}")

    # Should be exact (just a copy)
    passed = max_err_a == 0 and max_err_b == 0
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: 2D Forward BF16 (for transformer residuals)
# =============================================================================

def test_forward_2d_bf16(T=32, D=896, aligned_D=896):
    """Test add_forward_2d_bf16 with 2D tensor shapes."""
    print("\n" + "=" * 60)
    print(f"TEST: add_forward_2d_bf16 (shape [{T}, {D}])")
    print("=" * 60)

    np.random.seed(42)

    # Generate [T, aligned_D] shaped data
    a_f32 = np.random.randn(T, aligned_D).astype(np.float32)
    b_f32 = np.random.randn(T, aligned_D).astype(np.float32)

    a_bf16 = float_to_bf16(a_f32.flatten()).reshape(T, aligned_D)
    b_bf16 = float_to_bf16(b_f32.flatten()).reshape(T, aligned_D)
    y_bf16 = np.zeros((T, aligned_D), dtype=np.uint16)

    a_bf16_f32 = bf16_to_float(a_bf16.flatten()).reshape(T, aligned_D)
    b_bf16_f32 = bf16_to_float(b_bf16.flatten()).reshape(T, aligned_D)

    # PyTorch reference (only compare first D elements per row)
    ref = a_bf16_f32[:, :D] + b_bf16_f32[:, :D]

    # C kernel
    lib.add_forward_2d_bf16(
        numpy_to_ptr_u16(a_bf16.flatten()),
        numpy_to_ptr_u16(b_bf16.flatten()),
        numpy_to_ptr_u16(y_bf16.flatten()),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned_D)
    )

    y_f32 = bf16_to_float(y_bf16.flatten()).reshape(T, aligned_D)[:, :D]

    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    print(f"  Shape:     [{T}, {D}] (aligned: {aligned_D})")
    print(f"  Max diff:  {max_err:.2e}")
    print(f"  Mean diff: {mean_err:.2e}")

    passed = max_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Test: FP32 version
# =============================================================================

def test_forward_f32(N=4096):
    """Test add_forward_f32 against PyTorch."""
    print("\n" + "=" * 60)
    print("TEST: add_forward_f32 (y = a + b)")
    print("=" * 60)

    np.random.seed(42)

    a_f32 = np.random.randn(N).astype(np.float32)
    b_f32 = np.random.randn(N).astype(np.float32)
    y_f32 = np.zeros(N, dtype=np.float32)

    # PyTorch reference
    ref = a_f32 + b_f32

    # C kernel
    lib.add_forward_f32(
        numpy_to_ptr_f32(a_f32),
        numpy_to_ptr_f32(b_f32),
        numpy_to_ptr_f32(y_f32),
        ctypes.c_size_t(N)
    )

    diff = np.abs(y_f32 - ref)
    max_err = np.max(diff)

    print(f"  Shape:     N={N}")
    print(f"  Max diff:  {max_err:.2e}")

    # Should be exact for FP32
    passed = max_err < 1e-6
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")

    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ADD KERNEL UNIT TESTS")
    print("=" * 60)

    cpu_info = get_cpu_info()
    print(f"\nCPU: {cpu_info.model_name}")
    print(f"SIMD: {cpu_info.best_simd}")

    results = []

    # Run all tests
    results.append(("add_forward_bf16", test_forward_bf16()))
    results.append(("add_scaled_forward_bf16", test_scaled_forward_bf16()))
    results.append(("add_inplace_bf16", test_inplace_bf16()))
    results.append(("add_backward_bf16", test_backward_bf16()))
    results.append(("add_forward_2d_bf16", test_forward_2d_bf16()))
    results.append(("add_forward_f32", test_forward_f32()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:30s} {status}")

    print(f"\n  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print("\n  SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
