"""
RMSNorm kernel unit tests with performance metrics.

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
lib = load_lib("libckernel_rmsnorm.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.POINTER(ctypes.c_float),  # rstd_cache
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_forward.restype = None

lib.rmsnorm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # rstd_cache
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_gamma
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
]
lib.rmsnorm_backward.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference: x: [T,D], gamma: [D]"""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    x_hat = x * rstd
    return x_hat * gamma


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(T=64, D=256, eps=1e-5, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    out_np = np.zeros((T, D), dtype=np.float32)
    rstd_np = np.zeros(T, dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    gamma_ptr = numpy_to_ptr(gamma_np)
    out_ptr = numpy_to_ptr(out_np)
    rstd_ptr = numpy_to_ptr(rstd_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    gamma = torch.from_numpy(gamma_np.copy())

    report = TestReport(
        test_name="RMSNorm Forward",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = rmsnorm_torch(x, gamma, eps)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: rmsnorm_torch(x, gamma, eps),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel
    def c_rmsnorm():
        lib.rmsnorm_forward(
            x_ptr, gamma_ptr, out_ptr, rstd_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D),
            ctypes.c_float(eps)
        )

    c_rmsnorm()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_rmsnorm, warmup=warmup, iterations=iterations, name="C RMSNorm")

    report.add_result(TestResult(
        name="RMSNorm",
        passed=diff <= 1e-6,
        max_diff=diff,
        tolerance=1e-6,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    # C kernel without rstd cache (inference-only path)
    out_no_cache = np.zeros((T, D), dtype=np.float32)
    out_no_cache_ptr = numpy_to_ptr(out_no_cache)

    def c_rmsnorm_no_cache():
        lib.rmsnorm_forward(
            x_ptr, gamma_ptr, out_no_cache_ptr, None,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D),
            ctypes.c_float(eps)
        )

    c_rmsnorm_no_cache()
    out_nc = torch.from_numpy(out_no_cache.copy())
    diff_nc = max_diff(out_nc, ref)

    report.add_result(TestResult(
        name="RMSNorm (no rstd cache)",
        passed=diff_nc <= 1e-6,
        max_diff=diff_nc,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_backward_tests(T=64, D=256, eps=1e-5, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)
    out_np = np.zeros((T, D), dtype=np.float32)
    rstd_np = np.zeros(T, dtype=np.float32)
    dx_np = np.zeros((T, D), dtype=np.float32)
    dgamma_np = np.zeros(D, dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    gamma_ptr = numpy_to_ptr(gamma_np)
    upstream_ptr = numpy_to_ptr(upstream_np)
    out_ptr = numpy_to_ptr(out_np)
    rstd_ptr = numpy_to_ptr(rstd_np)
    dx_ptr = numpy_to_ptr(dx_np)
    dgamma_ptr = numpy_to_ptr(dgamma_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    gamma = torch.from_numpy(gamma_np.copy())
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="RMSNorm Backward",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return rmsnorm_torch(x, gamma, eps)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        gamma_ref = gamma.clone().detach().requires_grad_(True)
        y = rmsnorm_torch(x_ref, gamma_ref, eps)
        y.backward(upstream)
        return x_ref.grad, gamma_ref.grad

    # Get reference grads
    dx_ref, dgamma_ref = pytorch_fwd_bwd()

    # C forward (needed for rstd cache)
    lib.rmsnorm_forward(
        x_ptr, gamma_ptr, out_ptr, rstd_ptr,
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D),
        ctypes.c_float(eps)
    )

    # C backward
    def c_backward():
        lib.rmsnorm_backward(
            upstream_ptr, x_ptr, gamma_ptr, rstd_ptr,
            dx_ptr, dgamma_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
        )

    # C forward + backward combined
    def c_fwd_bwd():
        lib.rmsnorm_forward(
            x_ptr, gamma_ptr, out_ptr, rstd_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D),
            ctypes.c_float(eps)
        )
        lib.rmsnorm_backward(
            upstream_ptr, x_ptr, gamma_ptr, rstd_ptr,
            dx_ptr, dgamma_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
        )

    # Run once for accuracy
    c_backward()
    dx_c = torch.from_numpy(dx_np.copy())
    dgamma_c = torch.from_numpy(dgamma_np.copy())
    diff_dx = max_diff(dx_c, dx_ref)
    diff_dgamma = max_diff(dgamma_c, dgamma_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_bwd_time = time_function(c_backward, warmup=warmup, iterations=iterations, name="C Bwd")
    c_fwd_bwd_time = time_function(c_fwd_bwd, warmup=warmup, iterations=iterations, name="C Fwd+Bwd")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    report.add_result(TestResult(
        name="d_input",
        passed=diff_dx <= 1e-5,
        max_diff=diff_dx,
        tolerance=1e-5,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_fwd_bwd_time
    ))

    report.add_result(TestResult(
        name="d_gamma",
        passed=diff_dgamma <= 1e-5,
        max_diff=diff_dgamma,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=None
    ))

    # Store timing data
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
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
    fwd_report = run_forward_tests(T=64, D=256, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(T=64, D=256, warmup=10, iterations=1000)
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
        print(f"  {'Forward+Backward':<20} {t['pt_fwd_bwd']:<15.1f} {t['c_fwd_bwd']:<15.1f} {t['pt_fwd_bwd']/t['c_fwd_bwd']:.2f}x")
        print("  " + "-" * 60)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
