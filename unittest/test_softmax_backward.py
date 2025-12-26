"""
Causal Softmax Backward kernel unit tests with performance metrics.

Note: This is a focused backward-only test. See test_softmax.py for
comprehensive forward+backward tests with timing breakdown.
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

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.causal_softmax_head_major.restype = None

lib.backward_causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.backward_causal_softmax_head_major.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def backward_causal_softmax_ref(dY: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal softmax backward."""
    H, T, _ = Y.shape
    dX = torch.zeros_like(Y)
    for h in range(H):
        for i in range(T):
            y_row = Y[h, i, : i + 1]
            dy_row = dY[h, i, : i + 1]
            dot = (y_row * dy_row).sum()
            dX[h, i, : i + 1] = y_row * (dy_row - dot)
            dX[h, i, i + 1 :] = 0.0
    return dX


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_backward_tests(H=8, T=64, warmup=10, iterations=500):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    scores_np = np.random.randn(H, T, T).astype(np.float32)
    weights_np = scores_np.copy()
    dY_np = np.random.randn(H, T, T).astype(np.float32)
    dX_np = dY_np.copy()

    # Get pointers
    weights_ptr = numpy_to_ptr(weights_np)
    dX_ptr = numpy_to_ptr(dX_np)

    # Run forward to get weights
    lib.causal_softmax_head_major(weights_ptr, H, T, T)

    # Torch tensors
    Y = torch.from_numpy(weights_np.copy())
    dY = torch.from_numpy(dY_np.copy())

    report = TestReport(
        test_name="Causal Softmax Backward",
        dtype="fp32",
        shape=f"H={H}, T={T}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    dX_ref = backward_causal_softmax_ref(dY, Y)

    # C kernel
    def c_softmax_backward():
        np.copyto(dX_np, dY_np)
        lib.backward_causal_softmax_head_major(dX_ptr, weights_ptr, H, T, T)

    c_softmax_backward()
    dX_c = torch.from_numpy(dX_np.copy())
    diff = max_diff(dX_c, dX_ref)

    kernel_time = time_function(c_softmax_backward, warmup=warmup, iterations=iterations, name="C Softmax Bwd")

    report.add_result(TestResult(
        name="d_scores",
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
        test_name="Causal Softmax Backward Accuracy",
        dtype="fp32",
        shape="Multiple sizes",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, "Small"),
        (4, 16, "Medium"),
        (8, 32, "Large"),
    ]

    for H, T, name in test_configs:
        np.random.seed(42)

        scores_np = np.random.randn(H, T, T).astype(np.float32)
        weights_np = scores_np.copy()
        dY_np = np.random.randn(H, T, T).astype(np.float32)
        dX_np = dY_np.copy()

        weights_ptr = numpy_to_ptr(weights_np)
        dX_ptr = numpy_to_ptr(dX_np)

        lib.causal_softmax_head_major(weights_ptr, H, T, T)

        Y = torch.from_numpy(weights_np.copy())
        dY = torch.from_numpy(dY_np.copy())
        dX_ref = backward_causal_softmax_ref(dY, Y)

        lib.backward_causal_softmax_head_major(dX_ptr, weights_ptr, H, T, T)
        dX_c = torch.from_numpy(dX_np.copy())
        diff = max_diff(dX_c, dX_ref)

        report.add_result(TestResult(
            name=f"{name} (H={H},T={T})",
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

    # Accuracy tests
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Performance tests
    perf_report = run_backward_tests(H=8, T=64, warmup=10, iterations=500)
    perf_report.print_report()

    # Exit with error if any tests failed
    if not acc_report.all_passed() or not perf_report.all_passed():
        exit(1)
