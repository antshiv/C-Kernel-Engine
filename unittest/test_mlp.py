"""
MLP kernel unit tests with performance metrics.

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

lib.mlp_token_parallel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # W_fc1
    ctypes.POINTER(ctypes.c_float),  # b_fc1
    ctypes.POINTER(ctypes.c_float),  # W_fc2
    ctypes.POINTER(ctypes.c_float),  # b_fc2
    ctypes.POINTER(ctypes.c_float),  # fc1_output
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_dim
    ctypes.c_int,                    # num_threads
]
lib.mlp_token_parallel.restype = None

# Exact version using scalar GELU with standard library tanhf
lib.mlp_token_parallel_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # W_fc1
    ctypes.POINTER(ctypes.c_float),  # b_fc1
    ctypes.POINTER(ctypes.c_float),  # W_fc2
    ctypes.POINTER(ctypes.c_float),  # b_fc2
    ctypes.POINTER(ctypes.c_float),  # fc1_output
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_dim
    ctypes.c_int,                    # num_threads
]
lib.mlp_token_parallel_exact.restype = None

# Scalar exact GELU backward using standard library tanhf
lib.gelu_backward_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_scalar.restype = None

lib.gelu_backward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_exact.restype = None

lib.fc2_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # fc2_input
    ctypes.POINTER(ctypes.c_float),  # W_fc2
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_W_fc2
    ctypes.POINTER(ctypes.c_float),  # d_b_fc2
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_in
    ctypes.c_int,                    # aligned_out
    ctypes.c_int,                    # num_threads
]
lib.fc2_backward_kernel.restype = None

lib.fc1_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # fc1_input
    ctypes.POINTER(ctypes.c_float),  # W_fc1
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_W_fc1
    ctypes.POINTER(ctypes.c_float),  # d_b_fc1
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_in
    ctypes.c_int,                    # aligned_out
    ctypes.c_int,                    # num_threads
]
lib.fc1_backward_kernel.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(T=64, D=128, warmup=10, iterations=500):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)
    fourD = 4 * D

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, D).astype(np.float32)
    W1_np = np.random.randn(fourD, D).astype(np.float32) * 0.02
    b1_np = np.zeros(fourD, dtype=np.float32)
    W2_np = np.random.randn(D, fourD).astype(np.float32) * 0.02
    b2_np = np.zeros(D, dtype=np.float32)
    fc1_out_fast_np = np.zeros((T, fourD), dtype=np.float32)
    fc1_out_exact_np = np.zeros((T, fourD), dtype=np.float32)
    out_fast_np = np.zeros((T, D), dtype=np.float32)
    out_exact_np = np.zeros((T, D), dtype=np.float32)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    W1_ptr = numpy_to_ptr(W1_np)
    b1_ptr = numpy_to_ptr(b1_np)
    W2_ptr = numpy_to_ptr(W2_np)
    b2_ptr = numpy_to_ptr(b2_np)
    fc1_out_fast_ptr = numpy_to_ptr(fc1_out_fast_np)
    fc1_out_exact_ptr = numpy_to_ptr(fc1_out_exact_np)
    out_fast_ptr = numpy_to_ptr(out_fast_np)
    out_exact_ptr = numpy_to_ptr(out_exact_np)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    W1 = torch.from_numpy(W1_np.copy())
    b1 = torch.from_numpy(b1_np.copy())
    W2 = torch.from_numpy(W2_np.copy())
    b2 = torch.from_numpy(b2_np.copy())

    report = TestReport(
        test_name="MLP Forward",
        dtype="fp32",
        shape=f"T={T}, D={D}, 4D={fourD}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference: y = fc2(gelu(fc1(x)))
    def pytorch_mlp():
        h = F.linear(x, W1, b1)
        h = F.gelu(h, approximate="tanh")
        return F.linear(h, W2, b2)

    ref = pytorch_mlp()

    # Time PyTorch
    pytorch_time = time_function(pytorch_mlp, warmup=warmup, iterations=iterations, name="PyTorch")

    # === Fast version (uses gelu_fast_inplace with tanh512_fast) ===
    def c_mlp_fast():
        lib.mlp_token_parallel(
            x_ptr, W1_ptr, b1_ptr, W2_ptr, b2_ptr,
            fc1_out_fast_ptr, out_fast_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(1)
        )

    c_mlp_fast()
    out_fast = torch.from_numpy(out_fast_np.copy())
    diff_fast = max_diff(out_fast, ref)
    kernel_time_fast = time_function(c_mlp_fast, warmup=warmup, iterations=iterations, name="C MLP Fast")

    # Fast version: trades accuracy for speedup on AVX-512
    report.add_result(TestResult(
        name="Fast (SIMD GELU)",
        passed=diff_fast <= 1e-2,
        max_diff=diff_fast,
        tolerance=1e-2,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time_fast
    ))

    # === Exact version (uses gelu_exact_inplace with standard tanhf) ===
    def c_mlp_exact():
        lib.mlp_token_parallel_exact(
            x_ptr, W1_ptr, b1_ptr, W2_ptr, b2_ptr,
            fc1_out_exact_ptr, out_exact_ptr,
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(1)
        )

    c_mlp_exact()
    out_exact = torch.from_numpy(out_exact_np.copy())
    diff_exact = max_diff(out_exact, ref)
    kernel_time_exact = time_function(c_mlp_exact, warmup=warmup, iterations=iterations, name="C MLP Exact")

    # Exact version: full precision using standard library tanhf
    report.add_result(TestResult(
        name="Exact (scalar GELU)",
        passed=diff_exact <= 1e-4,
        max_diff=diff_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=kernel_time_exact
    ))

    return report


def run_backward_tests(T=64, D=128, warmup=10, iterations=500):
    """Run backward pass tests with accuracy and timing.

    Tests both fast (SIMD with tanh approximation) and exact (scalar with tanhf) versions.
    """
    np.random.seed(1)
    fourD = 4 * D

    # Pre-allocate numpy arrays
    x_np = np.random.randn(T, D).astype(np.float32)
    W1_np = np.random.randn(fourD, D).astype(np.float32) * 0.02
    b1_np = np.zeros(fourD, dtype=np.float32)
    W2_np = np.random.randn(D, fourD).astype(np.float32) * 0.02
    b2_np = np.zeros(D, dtype=np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    W1 = torch.from_numpy(W1_np.copy())
    b1 = torch.from_numpy(b1_np.copy())
    W2 = torch.from_numpy(W2_np.copy())
    b2 = torch.from_numpy(b2_np.copy())
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="MLP Backward",
        dtype="fp32",
        shape=f"T={T}, D={D}, 4D={fourD}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        W1_ref = W1.clone().detach().requires_grad_(True)
        b1_ref = b1.clone().detach().requires_grad_(True)
        W2_ref = W2.clone().detach().requires_grad_(True)
        b2_ref = b2.clone().detach().requires_grad_(True)

        h = F.linear(x_ref, W1_ref, b1_ref)
        h = F.gelu(h, approximate="tanh")
        y = F.linear(h, W2_ref, b2_ref)
        y.backward(upstream)

        return x_ref.grad, W1_ref.grad, b1_ref.grad, W2_ref.grad, b2_ref.grad

    # Get reference grads
    dx_ref, dW1_ref, db1_ref, dW2_ref, db2_ref = pytorch_fwd_bwd()

    # Timing for PyTorch
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")

    # Recompute Z1 for GELU backward (FC1 output before GELU)
    Z1_np = (x_np @ W1_np.T + b1_np).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fast version: uses gelu_backward_exact (SIMD with tanh512_fast approximation)
    # ═══════════════════════════════════════════════════════════════════════════

    # C fast forward to get intermediate values
    fc1_out_fast_np = np.zeros((T, fourD), dtype=np.float32)
    out_fast_np = np.zeros((T, D), dtype=np.float32)

    lib.mlp_token_parallel(
        numpy_to_ptr(x_np), numpy_to_ptr(W1_np), numpy_to_ptr(b1_np),
        numpy_to_ptr(W2_np), numpy_to_ptr(b2_np),
        numpy_to_ptr(fc1_out_fast_np), numpy_to_ptr(out_fast_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(1)
    )

    # Pre-allocate gradient arrays for fast version
    dH_fast_np = np.zeros((T, fourD), dtype=np.float32)
    dW2_fast_np = np.zeros_like(W2_np)
    db2_fast_np = np.zeros_like(b2_np)
    dZ1_fast_np = np.zeros((T, fourD), dtype=np.float32)
    dx_fast_np = np.zeros((T, D), dtype=np.float32)
    dW1_fast_np = np.zeros_like(W1_np)
    db1_fast_np = np.zeros_like(b1_np)

    # C backward fast
    def c_backward_fast():
        # Reset gradients
        dH_fast_np.fill(0)
        dW2_fast_np.fill(0)
        db2_fast_np.fill(0)
        dZ1_fast_np.fill(0)
        dx_fast_np.fill(0)
        dW1_fast_np.fill(0)
        db1_fast_np.fill(0)

        # FC2 backward
        lib.fc2_backward_kernel(
            numpy_to_ptr(upstream_np), numpy_to_ptr(fc1_out_fast_np), numpy_to_ptr(W2_np),
            numpy_to_ptr(dH_fast_np), numpy_to_ptr(dW2_fast_np), numpy_to_ptr(db2_fast_np),
            ctypes.c_int(T), ctypes.c_int(fourD), ctypes.c_int(D), ctypes.c_int(1)
        )

        # GELU backward (fast - uses tanh512_fast approximation)
        lib.gelu_backward_exact(
            numpy_to_ptr(Z1_np), numpy_to_ptr(dH_fast_np), numpy_to_ptr(dZ1_fast_np),
            ctypes.c_size_t(T * fourD)
        )

        # FC1 backward
        lib.fc1_backward_kernel(
            numpy_to_ptr(dZ1_fast_np), numpy_to_ptr(x_np), numpy_to_ptr(W1_np),
            numpy_to_ptr(dx_fast_np), numpy_to_ptr(dW1_fast_np), numpy_to_ptr(db1_fast_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(fourD), ctypes.c_int(1)
        )

    # Run once for accuracy
    c_backward_fast()
    dx_fast = torch.from_numpy(dx_fast_np.copy())
    dW1_fast = torch.from_numpy(dW1_fast_np.copy())
    db1_fast = torch.from_numpy(db1_fast_np.copy())
    dW2_fast = torch.from_numpy(dW2_fast_np.copy())
    db2_fast = torch.from_numpy(db2_fast_np.copy())

    diff_dx_fast = max_diff(dx_fast, dx_ref)
    diff_dW1_fast = max_diff(dW1_fast, dW1_ref)
    diff_db1_fast = max_diff(db1_fast, db1_ref)
    diff_dW2_fast = max_diff(dW2_fast, dW2_ref)
    diff_db2_fast = max_diff(db2_fast, db2_ref)

    c_bwd_fast_time = time_function(c_backward_fast, warmup=warmup, iterations=iterations, name="C Bwd Fast")

    # ═══════════════════════════════════════════════════════════════════════════
    # Exact version: uses gelu_backward_scalar (standard library tanhf)
    # ═══════════════════════════════════════════════════════════════════════════

    # C exact forward to get intermediate values (uses exact GELU)
    fc1_out_exact_np = np.zeros((T, fourD), dtype=np.float32)
    out_exact_np = np.zeros((T, D), dtype=np.float32)

    lib.mlp_token_parallel_exact(
        numpy_to_ptr(x_np), numpy_to_ptr(W1_np), numpy_to_ptr(b1_np),
        numpy_to_ptr(W2_np), numpy_to_ptr(b2_np),
        numpy_to_ptr(fc1_out_exact_np), numpy_to_ptr(out_exact_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(1)
    )

    # Pre-allocate gradient arrays for exact version
    dH_exact_np = np.zeros((T, fourD), dtype=np.float32)
    dW2_exact_np = np.zeros_like(W2_np)
    db2_exact_np = np.zeros_like(b2_np)
    dZ1_exact_np = np.zeros((T, fourD), dtype=np.float32)
    dx_exact_np = np.zeros((T, D), dtype=np.float32)
    dW1_exact_np = np.zeros_like(W1_np)
    db1_exact_np = np.zeros_like(b1_np)

    # C backward exact
    def c_backward_exact():
        # Reset gradients
        dH_exact_np.fill(0)
        dW2_exact_np.fill(0)
        db2_exact_np.fill(0)
        dZ1_exact_np.fill(0)
        dx_exact_np.fill(0)
        dW1_exact_np.fill(0)
        db1_exact_np.fill(0)

        # FC2 backward
        lib.fc2_backward_kernel(
            numpy_to_ptr(upstream_np), numpy_to_ptr(fc1_out_exact_np), numpy_to_ptr(W2_np),
            numpy_to_ptr(dH_exact_np), numpy_to_ptr(dW2_exact_np), numpy_to_ptr(db2_exact_np),
            ctypes.c_int(T), ctypes.c_int(fourD), ctypes.c_int(D), ctypes.c_int(1)
        )

        # GELU backward (exact - uses standard library tanhf)
        lib.gelu_backward_scalar(
            numpy_to_ptr(Z1_np), numpy_to_ptr(dH_exact_np), numpy_to_ptr(dZ1_exact_np),
            ctypes.c_size_t(T * fourD)
        )

        # FC1 backward
        lib.fc1_backward_kernel(
            numpy_to_ptr(dZ1_exact_np), numpy_to_ptr(x_np), numpy_to_ptr(W1_np),
            numpy_to_ptr(dx_exact_np), numpy_to_ptr(dW1_exact_np), numpy_to_ptr(db1_exact_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(fourD), ctypes.c_int(1)
        )

    # Run once for accuracy
    c_backward_exact()
    dx_exact = torch.from_numpy(dx_exact_np.copy())
    dW1_exact = torch.from_numpy(dW1_exact_np.copy())
    db1_exact = torch.from_numpy(db1_exact_np.copy())
    dW2_exact = torch.from_numpy(dW2_exact_np.copy())
    db2_exact = torch.from_numpy(db2_exact_np.copy())

    diff_dx_exact = max_diff(dx_exact, dx_ref)
    diff_dW1_exact = max_diff(dW1_exact, dW1_ref)
    diff_db1_exact = max_diff(db1_exact, db1_ref)
    diff_dW2_exact = max_diff(dW2_exact, dW2_ref)
    diff_db2_exact = max_diff(db2_exact, db2_ref)

    c_bwd_exact_time = time_function(c_backward_exact, warmup=warmup, iterations=iterations, name="C Bwd Exact")

    # ═══════════════════════════════════════════════════════════════════════════
    # Add results - Fast version (relaxed tolerance for SIMD approximation)
    # Error accumulates through chain rule: GELU ~1e-2 error amplifies to ~15%
    # ═══════════════════════════════════════════════════════════════════════════

    report.add_result(TestResult(
        name="Fast d_input",
        passed=diff_dx_fast <= 0.15,
        max_diff=diff_dx_fast,
        tolerance=0.15,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_bwd_fast_time
    ))

    report.add_result(TestResult(
        name="Fast d_W1",
        passed=diff_dW1_fast <= 0.15,
        max_diff=diff_dW1_fast,
        tolerance=0.15,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Fast d_b1",
        passed=diff_db1_fast <= 0.15,
        max_diff=diff_db1_fast,
        tolerance=0.15,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Fast d_W2",
        passed=diff_dW2_fast <= 0.15,
        max_diff=diff_dW2_fast,
        tolerance=0.15,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Fast d_b2",
        passed=diff_db2_fast <= 0.15,
        max_diff=diff_db2_fast,
        tolerance=0.15,
        pytorch_time=None,
        kernel_time=None
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # Add results - Exact version (strict tolerance for scalar tanhf)
    # ═══════════════════════════════════════════════════════════════════════════

    report.add_result(TestResult(
        name="Exact d_input",
        passed=diff_dx_exact <= 1e-4,
        max_diff=diff_dx_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=c_bwd_exact_time
    ))

    report.add_result(TestResult(
        name="Exact d_W1",
        passed=diff_dW1_exact <= 1e-4,
        max_diff=diff_dW1_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Exact d_b1",
        passed=diff_db1_exact <= 1e-4,
        max_diff=diff_db1_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Exact d_W2",
        passed=diff_dW2_exact <= 1e-4,
        max_diff=diff_dW2_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="Exact d_b2",
        passed=diff_db2_exact <= 1e-4,
        max_diff=diff_db2_exact,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Forward tests
    fwd_report = run_forward_tests(T=64, D=128, warmup=10, iterations=500)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(T=64, D=128, warmup=10, iterations=500)
    bwd_report.print_report()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
