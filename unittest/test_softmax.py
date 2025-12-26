"""
Causal Softmax kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
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
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.causal_softmax_head_major.restype = None

lib.backward_causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_scores (in/out)
    ctypes.POINTER(ctypes.c_float),  # weights
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.backward_causal_softmax_head_major.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementations
# ═══════════════════════════════════════════════════════════════════════════════

def softmax_causal_reference(scores: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal softmax."""
    H, T, _ = scores.shape
    ref = scores.clone()
    for h in range(H):
        for i in range(T):
            row = ref[h, i, : i + 1]
            ref[h, i, : i + 1] = F.softmax(row, dim=-1)
            ref[h, i, i + 1 :] = 0.0
    return ref


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

def run_forward_tests(H=8, T=64, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    scores_np = np.random.randn(H, T, T).astype(np.float32)
    out_np = scores_np.copy()

    # Get pointer
    out_ptr = numpy_to_ptr(out_np)

    # Torch tensor
    scores = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Causal Softmax Forward",
        dtype="fp32",
        shape=f"heads={H}, tokens={T}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = softmax_causal_reference(scores)

    # Time PyTorch (using vectorized mask approach for fair comparison)
    def pytorch_causal_softmax():
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s = scores.clone()
        s.masked_fill_(mask, float('-inf'))
        return F.softmax(s, dim=-1)

    pytorch_time = time_function(
        pytorch_causal_softmax,
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel (inplace)
    def c_softmax():
        np.copyto(out_np, scores_np)
        lib.causal_softmax_head_major(out_ptr, H, T, T)

    # Run once for accuracy
    c_softmax()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    # Time C kernel
    kernel_time = time_function(c_softmax, warmup=warmup, iterations=iterations, name="C Softmax")

    report.add_result(TestResult(
        name="Causal Softmax",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_backward_tests(H=8, T=64, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    scores_np = np.random.randn(H, T, T).astype(np.float32)
    weights_np = np.zeros((H, T, T), dtype=np.float32)
    dY_np = np.random.randn(H, T, T).astype(np.float32)
    dX_np = dY_np.copy()

    # Get pointers
    scores_ptr = numpy_to_ptr(scores_np)
    weights_ptr = numpy_to_ptr(weights_np)
    dX_ptr = numpy_to_ptr(dX_np)

    # Torch tensors
    scores = torch.from_numpy(scores_np.copy())
    dY = torch.from_numpy(dY_np.copy())

    report = TestReport(
        test_name="Causal Softmax Backward",
        dtype="fp32",
        shape=f"heads={H}, tokens={T}",
        cpu_info=get_cpu_info()
    )

    # Get forward output for backward
    ref_Y = softmax_causal_reference(scores)

    # C forward first
    np.copyto(weights_np, scores_np)
    lib.causal_softmax_head_major(weights_ptr, H, T, T)
    Y = torch.from_numpy(weights_np.copy())

    # PyTorch reference backward
    dX_ref = backward_causal_softmax_ref(dY, ref_Y)

    # PyTorch forward only
    def pytorch_forward():
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s = scores.clone()
        s.masked_fill_(mask, float('-inf'))
        return F.softmax(s, dim=-1)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        s = scores.clone().requires_grad_(True)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        s_masked = s.masked_fill(mask, float('-inf'))
        y = F.softmax(s_masked, dim=-1)
        y.backward(dY)
        return s.grad

    # C backward
    def c_backward():
        np.copyto(dX_np, dY_np)
        lib.backward_causal_softmax_head_major(dX_ptr, weights_ptr, H, T, T)

    # C forward + backward
    def c_fwd_bwd():
        np.copyto(weights_np, scores_np)
        lib.causal_softmax_head_major(weights_ptr, H, T, T)
        np.copyto(dX_np, dY_np)
        lib.backward_causal_softmax_head_major(dX_ptr, weights_ptr, H, T, T)

    # Run once for accuracy
    c_backward()
    dX_c = torch.from_numpy(dX_np.copy())
    diff = max_diff(dX_c, dX_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_fwd_time = time_function(
        lambda: (np.copyto(weights_np, scores_np), lib.causal_softmax_head_major(weights_ptr, H, T, T)),
        warmup=warmup, iterations=iterations, name="C Fwd"
    )
    c_bwd_time = time_function(c_backward, warmup=warmup, iterations=iterations, name="C Bwd")
    c_fwd_bwd_time = time_function(c_fwd_bwd, warmup=warmup, iterations=iterations, name="C Fwd+Bwd")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    report.add_result(TestResult(
        name="d_scores",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_fwd_bwd_time
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_fwd': c_fwd_time.mean_us,
        'c_bwd': c_bwd_time.mean_us,
        'c_fwd_bwd': c_fwd_bwd_time.mean_us,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Forward tests
    fwd_report = run_forward_tests(H=8, T=64, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(H=8, T=64, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Forward vs Backward)")
        print("  " + "-" * 60)
        print(f"  {'Operation':<20} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 60)
        print(f"  {'Forward':<20} {t['pt_fwd']:<15.1f} {t['c_fwd']:<15.1f} {t['pt_fwd']/t['c_fwd']:.2f}x")
        print(f"  {'Backward (est)':<20} {t['pt_bwd_est']:<15.1f} {t['c_bwd']:<15.1f} {t['pt_bwd_est']/t['c_bwd']:.2f}x")
        print(f"  {'Forward+Backward':<20} {t['pt_fwd_bwd']:<15.1f} {t['c_fwd_bwd']:<15.1f} {t['pt_fwd_bwd']/t['c_fwd_bwd']:.2f}x")
        print("  " + "-" * 60)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
