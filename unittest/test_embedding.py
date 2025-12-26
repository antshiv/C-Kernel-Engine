"""
Embedding kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.embedding_forward.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # token_ids
    ctypes.c_int,                    # token_count
    ctypes.c_int,                    # vocab_size
    ctypes.POINTER(ctypes.c_float),  # token_embeddings
    ctypes.POINTER(ctypes.c_float),  # pos_embeddings
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_int,                    # context_window
    ctypes.c_int,                    # add_pos
]
lib.embedding_forward.restype = None

lib.embedding_backward.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # token_ids
    ctypes.c_int,                    # token_count
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_token_embeddings
    ctypes.POINTER(ctypes.c_float),  # d_pos_embeddings
    ctypes.c_int,                    # vocab_size
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_int,                    # context_window
    ctypes.c_int,                    # add_pos
]
lib.embedding_backward.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def align_up(n, a):
    return (n + a - 1) // a * a


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    return arr


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_int32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_tests(V=1000, T=128, D=256, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)
    aligned_D = align_up(D, 16)

    # Pre-allocate arrays
    token_ids = np.random.randint(0, V, (T,), dtype=np.int32)
    token_emb = aligned_empty((V, aligned_D))
    pos_emb = aligned_empty((T, aligned_D))
    out = aligned_empty((T, aligned_D))
    token_emb.fill(0.0)
    pos_emb.fill(0.0)

    # Initialize with random values
    token_emb[:, :D] = np.random.randn(V, D).astype(np.float32)
    pos_emb[:, :D] = np.random.randn(T, D).astype(np.float32)

    # Torch tensors
    token_ref = torch.from_numpy(token_emb[:, :D].copy())
    pos_ref = torch.from_numpy(pos_emb[:, :D].copy())
    token_ids_torch = torch.from_numpy(token_ids).long()

    report = TestReport(
        test_name="Embedding Forward",
        dtype="fp32",
        shape=f"V={V}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # Test with positional embeddings
    def pytorch_embedding():
        return token_ref[token_ids_torch] + pos_ref

    ref = pytorch_embedding()

    pytorch_time = time_function(pytorch_embedding, warmup=warmup, iterations=iterations, name="PyTorch")

    def c_embedding():
        lib.embedding_forward(
            ptr_int32(token_ids), ctypes.c_int(T), ctypes.c_int(V),
            ptr(token_emb), ptr(pos_emb), ptr(out),
            ctypes.c_int(D), ctypes.c_int(aligned_D), ctypes.c_int(T),
            ctypes.c_int(1)
        )

    c_embedding()
    out_t = torch.from_numpy(out[:, :D].copy())
    diff = max_diff(out_t, ref)

    kernel_time = time_function(c_embedding, warmup=warmup, iterations=iterations, name="C Embedding")

    report.add_result(TestResult(
        name="Embedding + Pos",
        passed=diff <= 1e-6,
        max_diff=diff,
        tolerance=1e-6,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    # Test without positional embeddings
    def pytorch_embedding_no_pos():
        return token_ref[token_ids_torch]

    ref_no_pos = pytorch_embedding_no_pos()

    pytorch_no_pos_time = time_function(pytorch_embedding_no_pos, warmup=warmup, iterations=iterations, name="PyTorch")

    def c_embedding_no_pos():
        lib.embedding_forward(
            ptr_int32(token_ids), ctypes.c_int(T), ctypes.c_int(V),
            ptr(token_emb), ctypes.POINTER(ctypes.c_float)(), ptr(out),
            ctypes.c_int(D), ctypes.c_int(aligned_D), ctypes.c_int(T),
            ctypes.c_int(0)
        )

    c_embedding_no_pos()
    out_t_no_pos = torch.from_numpy(out[:, :D].copy())
    diff_no_pos = max_diff(out_t_no_pos, ref_no_pos)

    kernel_no_pos_time = time_function(c_embedding_no_pos, warmup=warmup, iterations=iterations, name="C Embedding")

    report.add_result(TestResult(
        name="Embedding Only",
        passed=diff_no_pos <= 1e-6,
        max_diff=diff_no_pos,
        tolerance=1e-6,
        pytorch_time=pytorch_no_pos_time,
        kernel_time=kernel_no_pos_time
    ))

    return report


def run_backward_tests(V=1000, T=128, D=256, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)
    aligned_D = align_up(D, 16)

    # Pre-allocate arrays
    token_ids = np.random.randint(0, V, (T,), dtype=np.int32)
    d_output = aligned_empty((T, aligned_D))
    d_output.fill(0.0)
    d_out_np = np.random.randn(T, D).astype(np.float32)
    d_output[:, :D] = d_out_np

    d_tok = aligned_empty((V, aligned_D))
    d_pos = aligned_empty((T, aligned_D))
    d_tok.fill(0.0)
    d_pos.fill(0.0)

    # Torch tensors
    token_ref = torch.randn(V, D, dtype=torch.float32, requires_grad=True)
    pos_ref = torch.randn(T, D, dtype=torch.float32, requires_grad=True)
    token_ids_torch = torch.from_numpy(token_ids).long()
    d_out_torch = torch.from_numpy(d_out_np)

    report = TestReport(
        test_name="Embedding Backward",
        dtype="fp32",
        shape=f"V={V}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return token_ref[token_ids_torch] + pos_ref

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        t_ref = token_ref.clone().detach().requires_grad_(True)
        p_ref = pos_ref.clone().detach().requires_grad_(True)
        out = t_ref[token_ids_torch] + p_ref
        out.backward(d_out_torch)
        return t_ref.grad, p_ref.grad

    # Get reference grads
    d_tok_ref, d_pos_ref = pytorch_fwd_bwd()

    # C backward
    def c_backward():
        d_tok.fill(0.0)
        d_pos.fill(0.0)
        lib.embedding_backward(
            ptr_int32(token_ids), ctypes.c_int(T),
            ptr(d_output), ptr(d_tok), ptr(d_pos),
            ctypes.c_int(V), ctypes.c_int(D), ctypes.c_int(aligned_D),
            ctypes.c_int(T), ctypes.c_int(1)
        )

    # Run once for accuracy
    c_backward()
    d_tok_c = torch.from_numpy(d_tok[:, :D].copy())
    d_pos_c = torch.from_numpy(d_pos[:, :D].copy())
    diff_tok = max_diff(d_tok_c, d_tok_ref)
    diff_pos = max_diff(d_pos_c, d_pos_ref)

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_bwd_time = time_function(c_backward, warmup=warmup, iterations=iterations, name="C Bwd")

    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    report.add_result(TestResult(
        name="d_token_emb",
        passed=diff_tok <= 1e-6,
        max_diff=diff_tok,
        tolerance=1e-6,
        pytorch_time=pt_fwd_bwd_time,
        kernel_time=c_bwd_time
    ))

    report.add_result(TestResult(
        name="d_pos_emb",
        passed=diff_pos <= 1e-6,
        max_diff=diff_pos,
        tolerance=1e-6,
        pytorch_time=None,
        kernel_time=None
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
    fwd_report = run_forward_tests(V=1000, T=128, D=256, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests
    bwd_report = run_backward_tests(V=1000, T=128, D=256, warmup=10, iterations=1000)
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
