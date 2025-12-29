"""
Attention Backward kernel unit tests with performance metrics.

Tests backward pass for both standard and GQA attention against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes
import math

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_attention.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.attention_forward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major.restype = None

# Exact version (uses standard library expf for softmax)
lib.attention_forward_causal_head_major_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major_exact.restype = None

lib.attention_forward_causal_head_major_gqa.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa.restype = None

# Exact GQA version
lib.attention_forward_causal_head_major_gqa_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa_exact.restype = None

lib.attention_backward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_backward_causal_head_major.restype = None

lib.attention_backward_causal_head_major_gqa.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_backward_causal_head_major_gqa.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def causal_attention_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """PyTorch reference causal attention."""
    H, T, D = q.shape
    H_kv = k.shape[0]
    scale = 1.0 / math.sqrt(D)

    attn_weights = torch.zeros(H, T, T, dtype=q.dtype)
    output = torch.zeros_like(q)

    for h in range(H):
        kv_h = h * H_kv // H
        for i in range(T):
            qi = q[h, i]
            kj = k[kv_h, :i+1, :]
            scores = torch.mv(kj, qi) * scale
            weights = torch.softmax(scores, dim=-1)
            attn_weights[h, i, :i+1] = weights
            vj = v[kv_h, :i+1, :]
            output[h, i, :] = torch.matmul(weights, vj)

    return output, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_backward_tests(H=4, T=32, D=32, warmup=10, iterations=200):
    """Run backward pass tests (non-GQA) with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H, T, D).astype(np.float32)
    v_np = np.random.randn(H, T, D).astype(np.float32)
    d_out_np = np.random.randn(H, T, D).astype(np.float32)
    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_np = np.zeros((H, T, D), dtype=np.float32)
    d_q_np = np.zeros((H, T, D), dtype=np.float32)
    d_k_np = np.zeros((H, T, D), dtype=np.float32)
    d_v_np = np.zeros((H, T, D), dtype=np.float32)
    d_scores_np = np.zeros((H, T, T), dtype=np.float32)

    # Torch tensors
    q = torch.from_numpy(q_np.copy()).requires_grad_(True)
    k = torch.from_numpy(k_np.copy()).requires_grad_(True)
    v = torch.from_numpy(v_np.copy()).requires_grad_(True)
    d_out = torch.from_numpy(d_out_np)

    report = TestReport(
        test_name="Attention Backward (Non-GQA)",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        out_ref, _ = causal_attention_pytorch(q_ref, k_ref, v_ref)
        out_ref.backward(d_out)
        return q_ref.grad, k_ref.grad, v_ref.grad

    d_q_ref, d_k_ref, d_v_ref = pytorch_fwd_bwd()

    # C kernel forward + backward (use exact forward for accurate weights)
    def c_fwd_bwd():
        # Forward (exact version for accurate attention weights)
        lib.attention_forward_causal_head_major_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )
        # Backward
        lib.attention_backward_causal_head_major(
            numpy_to_ptr(d_out_np), numpy_to_ptr(q_np), numpy_to_ptr(k_np),
            numpy_to_ptr(v_np), numpy_to_ptr(scores_np),
            numpy_to_ptr(d_q_np), numpy_to_ptr(d_k_np), numpy_to_ptr(d_v_np),
            numpy_to_ptr(d_scores_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

    # Run once for accuracy
    c_fwd_bwd()
    d_q_c = torch.from_numpy(d_q_np.copy())
    d_k_c = torch.from_numpy(d_k_np.copy())
    d_v_c = torch.from_numpy(d_v_np.copy())

    diff_dq = max_diff(d_q_c, d_q_ref)
    diff_dk = max_diff(d_k_c, d_k_ref)
    diff_dv = max_diff(d_v_c, d_v_ref)

    # Timing
    pt_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_fwd_bwd, warmup=warmup, iterations=iterations, name="C Kernel")

    report.add_result(TestResult(
        name="d_q",
        passed=diff_dq <= 1e-4,
        max_diff=diff_dq,
        tolerance=1e-4,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    report.add_result(TestResult(
        name="d_k",
        passed=diff_dk <= 1e-4,
        max_diff=diff_dk,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="d_v",
        passed=diff_dv <= 1e-4,
        max_diff=diff_dv,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_gqa_backward_tests(H=8, H_kv=2, T=32, D=32, warmup=10, iterations=200):
    """Run backward pass tests (GQA) with accuracy and timing."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H_kv, T, D).astype(np.float32)
    v_np = np.random.randn(H_kv, T, D).astype(np.float32)
    d_out_np = np.random.randn(H, T, D).astype(np.float32)
    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_np = np.zeros((H, T, D), dtype=np.float32)
    d_q_np = np.zeros((H, T, D), dtype=np.float32)
    d_k_np = np.zeros((H_kv, T, D), dtype=np.float32)
    d_v_np = np.zeros((H_kv, T, D), dtype=np.float32)
    d_scores_np = np.zeros((H, T, T), dtype=np.float32)

    # Torch tensors
    q = torch.from_numpy(q_np.copy()).requires_grad_(True)
    k = torch.from_numpy(k_np.copy()).requires_grad_(True)
    v = torch.from_numpy(v_np.copy()).requires_grad_(True)
    d_out = torch.from_numpy(d_out_np)

    report = TestReport(
        test_name="Attention Backward (GQA)",
        dtype="fp32",
        shape=f"H={H}, H_kv={H_kv}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        out_ref, _ = causal_attention_pytorch(q_ref, k_ref, v_ref)
        out_ref.backward(d_out)
        return q_ref.grad, k_ref.grad, v_ref.grad

    d_q_ref, d_k_ref, d_v_ref = pytorch_fwd_bwd()

    # C kernel forward + backward (use exact forward for accurate weights)
    def c_fwd_bwd():
        # Reset gradients
        d_q_np.fill(0)
        d_k_np.fill(0)
        d_v_np.fill(0)

        # Forward (exact version for accurate attention weights)
        lib.attention_forward_causal_head_major_gqa_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(D), ctypes.c_int(T)
        )
        # Backward
        lib.attention_backward_causal_head_major_gqa(
            numpy_to_ptr(d_out_np), numpy_to_ptr(q_np), numpy_to_ptr(k_np),
            numpy_to_ptr(v_np), numpy_to_ptr(scores_np),
            numpy_to_ptr(d_q_np), numpy_to_ptr(d_k_np), numpy_to_ptr(d_v_np),
            numpy_to_ptr(d_scores_np),
            ctypes.c_int(H), ctypes.c_int(H_kv), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(D), ctypes.c_int(T)
        )

    # Run once for accuracy
    c_fwd_bwd()
    d_q_c = torch.from_numpy(d_q_np.copy())
    d_k_c = torch.from_numpy(d_k_np.copy())
    d_v_c = torch.from_numpy(d_v_np.copy())

    diff_dq = max_diff(d_q_c, d_q_ref)
    diff_dk = max_diff(d_k_c, d_k_ref)
    diff_dv = max_diff(d_v_c, d_v_ref)

    # Timing
    pt_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_fwd_bwd, warmup=warmup, iterations=iterations, name="C Kernel")

    report.add_result(TestResult(
        name="d_q (GQA)",
        passed=diff_dq <= 1e-4,
        max_diff=diff_dq,
        tolerance=1e-4,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    report.add_result(TestResult(
        name="d_k (GQA)",
        passed=diff_dk <= 1e-4,
        max_diff=diff_dk,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="d_v (GQA)",
        passed=diff_dv <= 1e-4,
        max_diff=diff_dv,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_accuracy_tests():
    """Run accuracy tests at various sizes."""
    report = TestReport(
        test_name="Attention Backward Accuracy",
        dtype="fp32",
        shape="Multiple configurations",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, 16, "Tiny"),
        (4, 16, 32, "Small"),
        (8, 32, 64, "Medium"),
    ]

    for H, T, D, name in test_configs:
        np.random.seed(42)

        q_np = np.random.randn(H, T, D).astype(np.float32)
        k_np = np.random.randn(H, T, D).astype(np.float32)
        v_np = np.random.randn(H, T, D).astype(np.float32)
        d_out_np = np.random.randn(H, T, D).astype(np.float32)
        scores_np = np.zeros((H, T, T), dtype=np.float32)
        out_np = np.zeros((H, T, D), dtype=np.float32)
        d_q_np = np.zeros((H, T, D), dtype=np.float32)
        d_k_np = np.zeros((H, T, D), dtype=np.float32)
        d_v_np = np.zeros((H, T, D), dtype=np.float32)
        d_scores_np = np.zeros((H, T, T), dtype=np.float32)

        q = torch.from_numpy(q_np.copy()).requires_grad_(True)
        k = torch.from_numpy(k_np.copy()).requires_grad_(True)
        v = torch.from_numpy(v_np.copy()).requires_grad_(True)
        d_out = torch.from_numpy(d_out_np)

        out_ref, _ = causal_attention_pytorch(q, k, v)
        out_ref.backward(d_out)

        # Use exact forward for accurate attention weights
        lib.attention_forward_causal_head_major_exact(
            numpy_to_ptr(q_np), numpy_to_ptr(k_np), numpy_to_ptr(v_np),
            numpy_to_ptr(scores_np), numpy_to_ptr(out_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )
        lib.attention_backward_causal_head_major(
            numpy_to_ptr(d_out_np), numpy_to_ptr(q_np), numpy_to_ptr(k_np),
            numpy_to_ptr(v_np), numpy_to_ptr(scores_np),
            numpy_to_ptr(d_q_np), numpy_to_ptr(d_k_np), numpy_to_ptr(d_v_np),
            numpy_to_ptr(d_scores_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

        diff_dq = max_diff(torch.from_numpy(d_q_np), q.grad)
        diff_dk = max_diff(torch.from_numpy(d_k_np), k.grad)
        diff_dv = max_diff(torch.from_numpy(d_v_np), v.grad)
        max_err = max(diff_dq, diff_dk, diff_dv)

        report.add_result(TestResult(
            name=f"{name} (H={H},T={T},D={D})",
            passed=max_err <= 1e-4,
            max_diff=max_err,
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

    # Accuracy tests
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Non-GQA backward tests
    non_gqa_report = run_backward_tests(H=4, T=32, D=32, warmup=10, iterations=200)
    non_gqa_report.print_report()

    # GQA backward tests
    gqa_report = run_gqa_backward_tests(H=8, H_kv=2, T=32, D=32, warmup=10, iterations=200)
    gqa_report.print_report()

    # Exit with error if any tests failed
    all_passed = (acc_report.all_passed() and non_gqa_report.all_passed() and
                  gqa_report.all_passed())
    if not all_passed:
        exit(1)
