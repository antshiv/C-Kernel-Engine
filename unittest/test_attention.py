"""
Attention kernel unit tests with performance metrics.

Tests causal attention forward pass against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes
import math

import numpy as np
import torch
import torch.nn.functional as F

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
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # aligned_context_window
]
lib.attention_forward_causal_head_major.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for causal attention (head-major layout)."""
    H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Compute scores: [H, T, T]
    scores = torch.bmm(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))

    # Softmax
    weights = F.softmax(scores, dim=-1)

    # Output: [H, T, D]
    return torch.bmm(weights, v)


def attention_reference_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Loop-based reference for exact numerical comparison."""
    H, T, D = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for h in range(H):
        for i in range(T):
            qi = q[h, i]
            kj = k[h, : i + 1, :]
            scores = torch.mv(kj, qi) * scale
            weights = torch.softmax(scores, dim=-1)
            vj = v[h, : i + 1, :]
            out[h, i, :] = torch.matmul(weights, vj)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)

    # Pre-allocate numpy arrays
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H, T, D).astype(np.float32)
    v_np = np.random.randn(H, T, D).astype(np.float32)
    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_np = np.zeros((H, T, D), dtype=np.float32)

    # Get pointers
    q_ptr = numpy_to_ptr(q_np)
    k_ptr = numpy_to_ptr(k_np)
    v_ptr = numpy_to_ptr(v_np)
    scores_ptr = numpy_to_ptr(scores_np)
    out_ptr = numpy_to_ptr(out_np)

    # Torch tensors
    q = torch.from_numpy(q_np.copy())
    k = torch.from_numpy(k_np.copy())
    v = torch.from_numpy(v_np.copy())

    report = TestReport(
        test_name="Attention Forward (Causal)",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference (vectorized)
    ref = attention_reference(q, k, v)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: attention_reference(q, k, v),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel
    def c_attention():
        lib.attention_forward_causal_head_major(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

    c_attention()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_attention, warmup=warmup, iterations=iterations, name="C Attention")

    report.add_result(TestResult(
        name="Causal Attention",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_accuracy_tests():
    """Run small tests for numerical accuracy verification."""
    np.random.seed(42)

    report = TestReport(
        test_name="Attention Accuracy (Small)",
        dtype="fp32",
        shape="Various small sizes",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, 16, "Tiny"),
        (4, 16, 32, "Small"),
        (8, 32, 64, "Medium"),
    ]

    for H, T, D, name in test_configs:
        q_np = np.random.randn(H, T, D).astype(np.float32)
        k_np = np.random.randn(H, T, D).astype(np.float32)
        v_np = np.random.randn(H, T, D).astype(np.float32)
        scores_np = np.zeros((H, T, T), dtype=np.float32)
        out_np = np.zeros((H, T, D), dtype=np.float32)

        q_ptr = numpy_to_ptr(q_np)
        k_ptr = numpy_to_ptr(k_np)
        v_ptr = numpy_to_ptr(v_np)
        scores_ptr = numpy_to_ptr(scores_np)
        out_ptr = numpy_to_ptr(out_np)

        q = torch.from_numpy(q_np.copy())
        k = torch.from_numpy(k_np.copy())
        v = torch.from_numpy(v_np.copy())

        # Use loop reference for exact comparison
        ref = attention_reference_loop(q, k, v)

        lib.attention_forward_causal_head_major(
            q_ptr, k_ptr, v_ptr, scores_ptr, out_ptr,
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(T)
        )

        out = torch.from_numpy(out_np.copy())
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

    # Accuracy tests (small sizes)
    acc_report = run_accuracy_tests()
    acc_report.print_report()

    # Performance tests (larger size)
    perf_report = run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    perf_report.print_report()

    # Exit with error if any tests failed
    if not acc_report.all_passed() or not perf_report.all_passed():
        exit(1)
