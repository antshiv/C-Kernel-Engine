"""
Fused GEMM kernel unit tests with performance metrics.

Tests fused GEMM + bias + activation operations against PyTorch reference.
Reports accuracy, timing, and system information.

Fused operations tested:
- gemm_bias_relu_fused: GEMM + bias + ReLU
- gemm_bias_gelu_fused: GEMM + bias + GELU
- gemm_bias_silu_fused: GEMM + bias + SiLU
- gemm_swiglu_fused: Two GEMMs + SwiGLU (LLaMA MLP style)
"""
import ctypes

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")


# Function signatures
# C signature: gemm_bias_relu_fused(const float *A, const float *B,
#                                   const float *bias, float *C,
#                                   int M, int N, int K)
lib.gemm_bias_relu_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]
lib.gemm_bias_relu_fused.restype = None

lib.gemm_bias_gelu_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]
lib.gemm_bias_gelu_fused.restype = None

lib.gemm_bias_silu_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]
lib.gemm_bias_silu_fused.restype = None

# gemm_swiglu_fused: output = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
lib.gemm_swiglu_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # W_gate
    ctypes.POINTER(ctypes.c_float),  # W_up
    ctypes.POINTER(ctypes.c_float),  # b_gate (can be NULL)
    ctypes.POINTER(ctypes.c_float),  # b_up (can be NULL)
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]
lib.gemm_swiglu_fused.restype = None


def pytorch_gemm_bias_relu(A, B, bias):
    """PyTorch reference: GEMM + bias + ReLU."""
    # B is stored transposed (N x K), so we need B^T for matmul
    C = torch.matmul(A, B.T) + bias
    return F.relu(C)


def pytorch_gemm_bias_gelu(A, B, bias):
    """PyTorch reference: GEMM + bias + GELU."""
    C = torch.matmul(A, B.T) + bias
    return F.gelu(C, approximate="tanh")


def pytorch_gemm_bias_silu(A, B, bias):
    """PyTorch reference: GEMM + bias + SiLU."""
    C = torch.matmul(A, B.T) + bias
    return F.silu(C)


def pytorch_swiglu(x, W_gate, W_up, b_gate=None, b_up=None):
    """PyTorch reference: SwiGLU = SiLU(x @ W_gate) * (x @ W_up)."""
    gate = torch.matmul(x, W_gate.T)
    up = torch.matmul(x, W_up.T)
    if b_gate is not None:
        gate = gate + b_gate
    if b_up is not None:
        up = up + b_up
    return F.silu(gate) * up


def run_fused_relu_tests(M=64, N=128, K=256, warmup=10, iterations=500):
    """Run GEMM + bias + ReLU fused kernel tests."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(N, K).astype(np.float32)  # Transposed layout (N x K)
    bias_np = np.random.randn(N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    # Get pointers
    A_ptr = numpy_to_ptr(A_np)
    B_ptr = numpy_to_ptr(B_np)
    bias_ptr = numpy_to_ptr(bias_np)
    C_ptr = numpy_to_ptr(C_np)

    # Torch tensors
    A = torch.from_numpy(A_np.copy())
    B = torch.from_numpy(B_np.copy())
    bias = torch.from_numpy(bias_np.copy())

    report = TestReport(
        test_name="Fused GEMM + Bias + ReLU",
        dtype="fp32",
        shape=f"M={M}, N={N}, K={K}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = pytorch_gemm_bias_relu(A, B, bias)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: pytorch_gemm_bias_relu(A, B, bias),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel
    def c_fused_relu():
        lib.gemm_bias_relu_fused(
            A_ptr, B_ptr, bias_ptr, C_ptr,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )

    # Run once for accuracy
    c_fused_relu()
    out = torch.from_numpy(C_np.copy())
    diff = max_diff(out, ref)

    # Time C kernel
    kernel_time = time_function(c_fused_relu, warmup=warmup, iterations=iterations, name="C Fused ReLU")

    report.add_result(TestResult(
        name="GEMM+Bias+ReLU",
        passed=diff <= 1e-4,
        max_diff=diff,
        tolerance=1e-4,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_fused_gelu_tests(M=64, N=128, K=256, warmup=10, iterations=500):
    """Run GEMM + bias + GELU fused kernel tests."""
    np.random.seed(1)

    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(N, K).astype(np.float32)
    bias_np = np.random.randn(N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    A_ptr = numpy_to_ptr(A_np)
    B_ptr = numpy_to_ptr(B_np)
    bias_ptr = numpy_to_ptr(bias_np)
    C_ptr = numpy_to_ptr(C_np)

    A = torch.from_numpy(A_np.copy())
    B = torch.from_numpy(B_np.copy())
    bias = torch.from_numpy(bias_np.copy())

    report = TestReport(
        test_name="Fused GEMM + Bias + GELU",
        dtype="fp32",
        shape=f"M={M}, N={N}, K={K}",
        cpu_info=get_cpu_info()
    )

    ref = pytorch_gemm_bias_gelu(A, B, bias)

    pytorch_time = time_function(
        lambda: pytorch_gemm_bias_gelu(A, B, bias),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    def c_fused_gelu():
        lib.gemm_bias_gelu_fused(
            A_ptr, B_ptr, bias_ptr, C_ptr,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )

    c_fused_gelu()
    out = torch.from_numpy(C_np.copy())
    diff = max_diff(out, ref)

    # GELU uses QuickGELU (1.702*sigmoid) approximation, which differs from PyTorch tanh approx
    # The two approximations have max error ~0.03 from each other, which is acceptable
    kernel_time = time_function(c_fused_gelu, warmup=warmup, iterations=iterations, name="C Fused GELU")

    report.add_result(TestResult(
        name="GEMM+Bias+GELU",
        passed=diff <= 0.05,  # QuickGELU vs tanh-GELU can differ by ~0.03
        max_diff=diff,
        tolerance=0.05,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_fused_silu_tests(M=64, N=128, K=256, warmup=10, iterations=500):
    """Run GEMM + bias + SiLU fused kernel tests."""
    np.random.seed(2)

    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(N, K).astype(np.float32)
    bias_np = np.random.randn(N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    A_ptr = numpy_to_ptr(A_np)
    B_ptr = numpy_to_ptr(B_np)
    bias_ptr = numpy_to_ptr(bias_np)
    C_ptr = numpy_to_ptr(C_np)

    A = torch.from_numpy(A_np.copy())
    B = torch.from_numpy(B_np.copy())
    bias = torch.from_numpy(bias_np.copy())

    report = TestReport(
        test_name="Fused GEMM + Bias + SiLU",
        dtype="fp32",
        shape=f"M={M}, N={N}, K={K}",
        cpu_info=get_cpu_info()
    )

    ref = pytorch_gemm_bias_silu(A, B, bias)

    pytorch_time = time_function(
        lambda: pytorch_gemm_bias_silu(A, B, bias),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    def c_fused_silu():
        lib.gemm_bias_silu_fused(
            A_ptr, B_ptr, bias_ptr, C_ptr,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )

    c_fused_silu()
    out = torch.from_numpy(C_np.copy())
    diff = max_diff(out, ref)

    # SiLU uses fast sigmoid approximation
    kernel_time = time_function(c_fused_silu, warmup=warmup, iterations=iterations, name="C Fused SiLU")

    report.add_result(TestResult(
        name="GEMM+Bias+SiLU",
        passed=diff <= 1e-3,  # Higher tolerance for fast sigmoid approximation
        max_diff=diff,
        tolerance=1e-3,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_swiglu_tests(M=64, N=128, K=256, warmup=10, iterations=500):
    """Run GEMM + SwiGLU fused kernel tests (LLaMA MLP style)."""
    np.random.seed(3)

    # Allocate arrays
    x_np = np.random.randn(M, K).astype(np.float32)
    W_gate_np = np.random.randn(N, K).astype(np.float32)
    W_up_np = np.random.randn(N, K).astype(np.float32)
    b_gate_np = np.random.randn(N).astype(np.float32)
    b_up_np = np.random.randn(N).astype(np.float32)
    out_np = np.zeros((M, N), dtype=np.float32)

    # Pointers
    x_ptr = numpy_to_ptr(x_np)
    W_gate_ptr = numpy_to_ptr(W_gate_np)
    W_up_ptr = numpy_to_ptr(W_up_np)
    b_gate_ptr = numpy_to_ptr(b_gate_np)
    b_up_ptr = numpy_to_ptr(b_up_np)
    out_ptr = numpy_to_ptr(out_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    W_gate = torch.from_numpy(W_gate_np.copy())
    W_up = torch.from_numpy(W_up_np.copy())
    b_gate = torch.from_numpy(b_gate_np.copy())
    b_up = torch.from_numpy(b_up_np.copy())

    report = TestReport(
        test_name="Fused GEMM + SwiGLU (LLaMA MLP)",
        dtype="fp32",
        shape=f"M={M}, N={N}, K={K}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = pytorch_swiglu(x, W_gate, W_up, b_gate, b_up)

    # Time PyTorch (two separate GEMMs + SwiGLU)
    pytorch_time = time_function(
        lambda: pytorch_swiglu(x, W_gate, W_up, b_gate, b_up),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel (fused)
    def c_swiglu_fused():
        lib.gemm_swiglu_fused(
            x_ptr, W_gate_ptr, W_up_ptr, b_gate_ptr, b_up_ptr, out_ptr,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )

    c_swiglu_fused()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_swiglu_fused, warmup=warmup, iterations=iterations, name="C SwiGLU Fused")

    report.add_result(TestResult(
        name="GEMM+SwiGLU",
        passed=diff <= 2e-3,  # Two GEMMs + SiLU accumulates more FP error
        max_diff=diff,
        tolerance=2e-3,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_accuracy_tests():
    """Test fused kernels at various sizes."""
    report = TestReport(
        test_name="Fused GEMM Accuracy (Various Sizes)",
        dtype="fp32",
        shape="Multiple configurations",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (8, 16, 32, "Tiny"),
        (32, 64, 128, "Small"),
        (64, 128, 256, "Medium"),
        (128, 256, 512, "Large"),
    ]

    for M, N, K, name in test_configs:
        np.random.seed(42)

        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(N, K).astype(np.float32)
        bias_np = np.random.randn(N).astype(np.float32)
        C_np = np.zeros((M, N), dtype=np.float32)

        A = torch.from_numpy(A_np.copy())
        B = torch.from_numpy(B_np.copy())
        bias = torch.from_numpy(bias_np.copy())

        # Test ReLU fused
        ref = pytorch_gemm_bias_relu(A, B, bias)
        lib.gemm_bias_relu_fused(
            numpy_to_ptr(A_np), numpy_to_ptr(B_np),
            numpy_to_ptr(bias_np), numpy_to_ptr(C_np),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )
        out = torch.from_numpy(C_np.copy())
        diff = max_diff(out, ref)

        report.add_result(TestResult(
            name=f"{name} (M={M},N={N},K={K})",
            passed=diff <= 1e-4,
            max_diff=diff,
            tolerance=1e-4,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_unfused_comparison(M=64, N=128, K=256, warmup=10, iterations=500):
    """Compare fused vs unfused (separate) operations."""
    np.random.seed(3)

    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(N, K).astype(np.float32)
    bias_np = np.random.randn(N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    A = torch.from_numpy(A_np.copy())
    B = torch.from_numpy(B_np.copy())
    bias = torch.from_numpy(bias_np.copy())

    # Unfused: separate GEMM, add, ReLU
    def pytorch_unfused():
        C = torch.matmul(A, B.T)
        C = C + bias
        return F.relu(C)

    # Fused
    def c_fused():
        lib.gemm_bias_relu_fused(
            numpy_to_ptr(A_np), numpy_to_ptr(B_np),
            numpy_to_ptr(bias_np), numpy_to_ptr(C_np),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
        )

    unfused_time = time_function(pytorch_unfused, warmup=warmup, iterations=iterations, name="PyTorch Unfused")
    fused_time = time_function(c_fused, warmup=warmup, iterations=iterations, name="C Fused")

    print("\n  FUSED vs UNFUSED COMPARISON")
    print("  " + "-" * 60)
    print(f"  {'Operation':<30} {'Time (us)':<15} {'Speedup':<10}")
    print("  " + "-" * 60)
    print(f"  {'PyTorch (GEMM + bias + ReLU)':<30} {unfused_time.mean_us:<15.1f} {'1.00x':<10}")
    print(f"  {'C Fused (single pass)':<30} {fused_time.mean_us:<15.1f} {unfused_time.mean_us/fused_time.mean_us:.2f}x")
    print("  " + "-" * 60)


if __name__ == "__main__":
    print_system_info()

    # Accuracy tests at various sizes
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Individual kernel tests with timing
    relu_report = run_fused_relu_tests(M=64, N=128, K=256, warmup=10, iterations=500)
    relu_report.print_report()

    gelu_report = run_fused_gelu_tests(M=64, N=128, K=256, warmup=10, iterations=500)
    gelu_report.print_report()

    silu_report = run_fused_silu_tests(M=64, N=128, K=256, warmup=10, iterations=500)
    silu_report.print_report()

    # SwiGLU test (LLaMA MLP style - two GEMMs fused)
    swiglu_report = run_swiglu_tests(M=64, N=128, K=256, warmup=10, iterations=500)
    swiglu_report.print_report()

    # Fused vs unfused comparison
    run_unfused_comparison(M=64, N=128, K=256, warmup=10, iterations=500)

    # Exit with error if any tests failed
    all_passed = (acc_report.all_passed() and relu_report.all_passed() and
                  swiglu_report.all_passed() and
                  gelu_report.all_passed() and silu_report.all_passed())
    if not all_passed:
        exit(1)
