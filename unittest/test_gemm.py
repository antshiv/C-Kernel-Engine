"""
GEMM kernel unit tests with performance metrics.

Tests multiple GEMM variants against PyTorch reference.
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
lib = load_lib("libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

_gemm_sig = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]

lib.gemm_naive_parallel.argtypes = _gemm_sig
lib.gemm_naive_parallel.restype = None
lib.gemm_avx512_parallel.argtypes = _gemm_sig
lib.gemm_avx512_parallel.restype = None
lib.gemm_fine_grained_parallel.argtypes = _gemm_sig
lib.gemm_fine_grained_parallel.restype = None
lib.gemm_blocked_serial.argtypes = _gemm_sig
lib.gemm_blocked_serial.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_gemm_tests(M, N, K, desc, warmup=10, iterations=1000):
    """Run GEMM tests for all variants with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(N, K).astype(np.float32)  # N x K (rows are output channels)
    bias_np = np.random.randn(N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    # Get pointers
    A_ptr = numpy_to_ptr(A_np)
    B_ptr = numpy_to_ptr(B_np)
    bias_ptr = numpy_to_ptr(bias_np)
    C_ptr = numpy_to_ptr(C_np)

    # Torch tensors
    A = torch.from_numpy(A_np)
    B = torch.from_numpy(B_np)
    bias = torch.from_numpy(bias_np)

    report = TestReport(
        test_name=f"GEMM {desc}",
        dtype="fp32",
        shape=f"M={M}, N={N}, K={K}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference: C = A @ B^T + bias
    ref = A @ B.t() + bias

    # Time PyTorch
    pytorch_time = time_function(
        lambda: A @ B.t() + bias,
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # Test each kernel variant
    kernels = [
        ("Naive (parallel)", lib.gemm_naive_parallel),
        ("AVX-512 (parallel)", lib.gemm_avx512_parallel),
        ("Fine-grained", lib.gemm_fine_grained_parallel),
        ("Blocked (serial)", lib.gemm_blocked_serial),
    ]

    for name, kernel_fn in kernels:
        # Run kernel
        def run_kernel():
            kernel_fn(A_ptr, B_ptr, bias_ptr, C_ptr, M, N, K)

        run_kernel()
        out = torch.from_numpy(C_np.copy())
        diff = max_diff(out, ref)

        # Time kernel
        kernel_time = time_function(run_kernel, warmup=warmup, iterations=iterations, name=name)

        report.add_result(TestResult(
            name=name,
            passed=diff <= 1e-4,  # GEMM can have higher numerical error
            max_diff=diff,
            tolerance=1e-4,
            pytorch_time=pytorch_time,
            kernel_time=kernel_time
        ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # MLP-style: [T,D] · [D,4D] - typical intermediate projection
    T, D = 32, 128
    mlp_report = run_gemm_tests(T, 4 * D, D, "[T,D]x[D,4D] MLP", warmup=10, iterations=1000)
    mlp_report.print_report()

    # Attention QK^T-like: [T,d] · [T,d]^T
    T, d = 64, 64
    qk_report = run_gemm_tests(T, T, d, "[T,d]x[T,d] QK^T", warmup=10, iterations=1000)
    qk_report.print_report()

    # Attention SV: [T,T] · [T,d] → [T,d]
    T, d = 64, 64
    sv_report = run_gemm_tests(T, d, T, "[T,T]x[T,d] SV", warmup=10, iterations=1000)
    sv_report.print_report()

    # Larger MLP for better timing
    T, D = 128, 256
    large_report = run_gemm_tests(T, 4 * D, D, "[128,256]x[256,1024]", warmup=10, iterations=500)
    large_report.print_report()

    # Exit with error if any tests failed
    all_passed = (mlp_report.all_passed() and qk_report.all_passed() and
                  sv_report.all_passed() and large_report.all_passed())
    if not all_passed:
        exit(1)
