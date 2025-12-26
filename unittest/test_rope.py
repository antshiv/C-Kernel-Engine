"""
RoPE (Rotary Position Embedding) kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_rope.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # max_seq_len
    ctypes.c_int,                    # head_dim
    ctypes.c_float,                  # base
]
lib.rope_precompute_cache.restype = None

lib.rope_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x (in-place)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
]
lib.rope_forward.restype = None

lib.rope_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_out
    ctypes.POINTER(ctypes.c_float),  # d_x (output)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
]
lib.rope_backward.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementations
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis_pytorch(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """PyTorch reference: compute cos/sin cache for RoPE."""
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def rope_forward_pytorch(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """PyTorch reference RoPE forward (rotate-half)."""
    H, T, D = x.shape
    half_dim = D // 2

    x_out = x.clone()
    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]
            sin_row = sin_cache[pos]

            for i in range(half_dim):
                x0 = x[h, t, i]
                x1 = x[h, t, i + half_dim]
                c = cos_row[i]
                s = sin_row[i]

                x_out[h, t, i] = x0 * c - x1 * s
                x_out[h, t, i + half_dim] = x0 * s + x1 * c

    return x_out


def rope_forward_pytorch_vectorized(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """Vectorized PyTorch reference for fair timing comparison."""
    H, T, D = x.shape
    half_dim = D // 2

    # Get relevant cache slices
    cos = cos_cache[pos_offset:pos_offset + T]  # [T, half_dim]
    sin = sin_cache[pos_offset:pos_offset + T]  # [T, half_dim]

    # Split x into two halves
    x1 = x[..., :half_dim]  # [H, T, half_dim]
    x2 = x[..., half_dim:]  # [H, T, half_dim]

    # Apply rotation (broadcast cos/sin over heads)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


def rope_backward_pytorch(d_out: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """PyTorch reference RoPE backward (inverse rotation)."""
    H, T, D = d_out.shape
    half_dim = D // 2

    d_x = torch.zeros_like(d_out)
    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]
            sin_row = sin_cache[pos]

            for i in range(half_dim):
                d0 = d_out[h, t, i]
                d1 = d_out[h, t, i + half_dim]
                c = cos_row[i]
                s = sin_row[i]

                d_x[h, t, i] = d0 * c + d1 * s
                d_x[h, t, i + half_dim] = -d0 * s + d1 * c

    return d_x


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_cache_tests(max_seq_len=128, head_dim=64, warmup=10, iterations=1000):
    """Test cos/sin cache precomputation."""
    np.random.seed(0)
    half_dim = head_dim // 2

    # Pre-allocate numpy arrays
    cos_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)
    sin_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)

    cos_ptr = numpy_to_ptr(cos_np)
    sin_ptr = numpy_to_ptr(sin_np)

    report = TestReport(
        test_name="RoPE Cache Precompute",
        dtype="fp32",
        shape=f"max_seq_len={max_seq_len}, head_dim={head_dim}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    cos_ref, sin_ref = precompute_freqs_cis_pytorch(head_dim, max_seq_len)

    # C kernel
    def c_precompute():
        lib.rope_precompute_cache(
            cos_ptr, sin_ptr,
            ctypes.c_int(max_seq_len), ctypes.c_int(head_dim),
            ctypes.c_float(10000.0)
        )

    c_precompute()
    cos_c = torch.from_numpy(cos_np.copy())
    sin_c = torch.from_numpy(sin_np.copy())

    diff_cos = max_diff(cos_c, cos_ref)
    diff_sin = max_diff(sin_c, sin_ref)

    cache_tolerance = 5e-6
    report.add_result(TestResult(
        name="cos_cache",
        passed=diff_cos <= cache_tolerance,
        max_diff=diff_cos,
        tolerance=cache_tolerance,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="sin_cache",
        passed=diff_sin <= cache_tolerance,
        max_diff=diff_sin,
        tolerance=cache_tolerance,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)
    half_dim = D // 2

    # Pre-allocate numpy arrays
    x_np = np.random.randn(H, T, D).astype(np.float32)
    cos_np = np.zeros((T, half_dim), dtype=np.float32)
    sin_np = np.zeros((T, half_dim), dtype=np.float32)

    # Precompute cache
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0)
    )

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())

    report = TestReport(
        test_name="RoPE Forward",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference (vectorized for fair comparison)
    ref = rope_forward_pytorch_vectorized(x, cos_cache, sin_cache)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: rope_forward_pytorch_vectorized(x, cos_cache, sin_cache),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel (in-place)
    x_c_np = x_np.copy()
    x_c_ptr = numpy_to_ptr(x_c_np)

    def c_rope_forward():
        np.copyto(x_c_np, x_np)
        lib.rope_forward(
            x_c_ptr, numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

    c_rope_forward()
    out = torch.from_numpy(x_c_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_rope_forward, warmup=warmup, iterations=iterations, name="C RoPE")

    report.add_result(TestResult(
        name="RoPE Forward",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_backward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)
    half_dim = D // 2

    # Pre-allocate numpy arrays
    d_out_np = np.random.randn(H, T, D).astype(np.float32)
    d_x_np = np.zeros((H, T, D), dtype=np.float32)
    cos_np = np.zeros((T, half_dim), dtype=np.float32)
    sin_np = np.zeros((T, half_dim), dtype=np.float32)

    # Precompute cache
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0)
    )

    # Get pointers
    d_out_ptr = numpy_to_ptr(d_out_np)
    d_x_ptr = numpy_to_ptr(d_x_np)

    # Torch tensors
    d_out = torch.from_numpy(d_out_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())

    report = TestReport(
        test_name="RoPE Backward",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    d_x_ref = rope_backward_pytorch(d_out, cos_cache, sin_cache)

    # C kernel
    def c_rope_backward():
        lib.rope_backward(
            d_out_ptr, d_x_ptr,
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

    c_rope_backward()
    d_x_c = torch.from_numpy(d_x_np.copy())
    diff = max_diff(d_x_c, d_x_ref)

    kernel_time = time_function(c_rope_backward, warmup=warmup, iterations=iterations, name="C RoPE Bwd")

    report.add_result(TestResult(
        name="d_input",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=kernel_time
    ))

    return report


def run_accuracy_tests():
    """Run accuracy tests at various sizes."""
    report = TestReport(
        test_name="RoPE Accuracy (Various Sizes)",
        dtype="fp32",
        shape="Multiple configurations",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, 16, "Tiny"),
        (4, 16, 32, "Small"),
        (8, 32, 64, "Medium"),
        (8, 64, 128, "Large"),
    ]

    for H, T, D, name in test_configs:
        np.random.seed(42)
        half_dim = D // 2

        x_np = np.random.randn(H, T, D).astype(np.float32)
        cos_np = np.zeros((T, half_dim), dtype=np.float32)
        sin_np = np.zeros((T, half_dim), dtype=np.float32)

        lib.rope_precompute_cache(
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0)
        )

        x = torch.from_numpy(x_np.copy())
        cos_cache = torch.from_numpy(cos_np.copy())
        sin_cache = torch.from_numpy(sin_np.copy())

        # Use loop reference for exact comparison
        ref = rope_forward_pytorch(x, cos_cache, sin_cache)

        x_c_np = x_np.copy()
        lib.rope_forward(
            numpy_to_ptr(x_c_np), numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

        out = torch.from_numpy(x_c_np)
        diff = max_diff(out, ref)

        report.add_result(TestResult(
            name=f"{name} (H={H},T={T},D={D})",
            passed=diff <= 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Cache tests
    cache_report = run_cache_tests()
    cache_report.print_report()

    # Accuracy tests
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Forward performance tests
    fwd_report = run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    bwd_report.print_report()

    # Exit with error if any tests failed
    all_passed = (cache_report.all_passed() and acc_report.all_passed() and
                  fwd_report.all_passed() and bwd_report.all_passed())
    if not all_passed:
        exit(1)
