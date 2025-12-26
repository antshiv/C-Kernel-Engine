"""
ReLU kernel unit tests with performance metrics.

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
lib = load_lib("libckernel_relu.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.relu_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # n
]
lib.relu_forward.restype = None

lib.relu_forward_inplace.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # data (in/out)
    ctypes.c_size_t,                 # n
]
lib.relu_forward_inplace.restype = None

lib.relu_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.relu_backward.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(N=4096, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(N).astype(np.float32)
    out_np = np.zeros(N, dtype=np.float32)
    out_inplace_np = x_np.copy()

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    out_ptr = numpy_to_ptr(out_np)
    out_inplace_ptr = numpy_to_ptr(out_inplace_np)

    # Torch tensor
    x = torch.from_numpy(x_np.copy())

    report = TestReport(
        test_name="ReLU Forward",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = F.relu(x)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: F.relu(x),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel out-of-place
    def c_relu():
        lib.relu_forward(x_ptr, out_ptr, ctypes.c_size_t(N))

    c_relu()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_relu, warmup=warmup, iterations=iterations, name="C ReLU")

    report.add_result(TestResult(
        name="ReLU (out-of-place)",
        passed=diff <= 1e-7,
        max_diff=diff,
        tolerance=1e-7,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    # C kernel inplace
    def c_relu_inplace():
        np.copyto(out_inplace_np, x_np)
        lib.relu_forward_inplace(out_inplace_ptr, ctypes.c_size_t(N))

    c_relu_inplace()
    out_inplace = torch.from_numpy(out_inplace_np.copy())
    diff_inplace = max_diff(out_inplace, ref)

    inplace_time = time_function(c_relu_inplace, warmup=warmup, iterations=iterations, name="C ReLU Inplace")

    report.add_result(TestResult(
        name="ReLU (inplace)",
        passed=diff_inplace <= 1e-7,
        max_diff=diff_inplace,
        tolerance=1e-7,
        pytorch_time=pytorch_time,
        kernel_time=inplace_time
    ))

    return report


def run_backward_tests(N=4096, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(N).astype(np.float32)
    upstream_np = np.random.randn(N).astype(np.float32)
    dx_np = np.zeros(N, dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    upstream_ptr = numpy_to_ptr(upstream_np)
    dx_ptr = numpy_to_ptr(dx_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="ReLU Backward",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return F.relu(x)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        y = F.relu(x_ref)
        y.backward(upstream)
        return x_ref.grad

    # Get reference grad
    dx_ref = pytorch_fwd_bwd()

    # C backward
    def c_backward():
        lib.relu_backward(x_ptr, upstream_ptr, dx_ptr, ctypes.c_size_t(N))

    # Run once for accuracy
    c_backward()
    dx_c = torch.from_numpy(dx_np.copy())
    diff = max_diff(dx_c, dx_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_bwd_time = time_function(c_backward, warmup=warmup, iterations=iterations, name="C Bwd")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    report.add_result(TestResult(
        name="d_input",
        passed=diff <= 1e-7,
        max_diff=diff,
        tolerance=1e-7,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_bwd_time
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_bwd': c_bwd_time.mean_us,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Forward tests
    fwd_report = run_forward_tests(N=4096, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(N=4096, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Forward vs Backward)")
        print("  " + "-" * 60)
        print(f"  {'Operation':<20} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 60)
        print(f"  {'Forward':<20} {t['pt_fwd']:<15.1f} {'(see above)':<15} {'-':<10}")
        print(f"  {'Backward':<20} {t['pt_bwd_est']:<15.1f} {t['c_bwd']:<15.1f} {t['pt_bwd_est']/t['c_bwd']:.2f}x")
        print("  " + "-" * 60)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
