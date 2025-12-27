"""
Attention BF16 kernel tests for both forward and backward causal passes.
"""
import ctypes
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)
from bf16_utils import (
    float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr
)

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)


lib = load_lib("libckernel_attention.so", "libckernel_engine.so")

lib.attention_forward_causal_head_major_gqa_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa_bf16.restype = None

lib.attention_backward_causal_head_major_gqa_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_float),   # d_x (unused)
    ctypes.POINTER(ctypes.c_uint16),  # q
    ctypes.POINTER(ctypes.c_uint16),  # k
    ctypes.POINTER(ctypes.c_uint16),  # v
    ctypes.POINTER(ctypes.c_float),   # attn_weights
    ctypes.POINTER(ctypes.c_float),   # d_q
    ctypes.POINTER(ctypes.c_float),   # d_k
    ctypes.POINTER(ctypes.c_float),   # d_v
    ctypes.POINTER(ctypes.c_float),   # d_scores
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.attention_backward_causal_head_major_gqa_bf16.restype = None


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.bmm(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores.masked_fill_(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


def causal_attention_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    H, T, D = q.shape
    H_kv = k.shape[0]
    scale = 1.0 / math.sqrt(D)

    attn_weights = torch.zeros(H, T, T, dtype=q.dtype)
    output = torch.zeros_like(q)

    for h in range(H):
        kv_h = h * H_kv // H
        for i in range(T):
            qi = q[h, i]
            kj = k[kv_h, : i + 1, :]
            scores = torch.mv(kj, qi) * scale
            weights = torch.softmax(scores, dim=-1)
            attn_weights[h, i, :i + 1] = weights
            vj = v[kv_h, : i + 1, :]
            output[h, i, :] = torch.matmul(weights, vj)

    return output, attn_weights


def run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    np.random.seed(0)
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H, T, D).astype(np.float32)
    v_np = np.random.randn(H, T, D).astype(np.float32)
    scores_np = np.zeros((H, T, T), dtype=np.float32)
    out_np = np.zeros((H, T, D), dtype=np.float32)

    q_bf = float32_to_bf16(q_np)
    k_bf = float32_to_bf16(k_np)
    v_bf = float32_to_bf16(v_np)

    q = torch.from_numpy(q_np.copy()).to(dtype=torch.bfloat16)
    k = torch.from_numpy(k_np.copy()).to(dtype=torch.bfloat16)
    v = torch.from_numpy(v_np.copy()).to(dtype=torch.bfloat16)

    report = TestReport(
        test_name="Attention Forward (BF16)",
        dtype="bf16",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    def pytorch_forward():
        return attention_reference(q, k, v)

    ref = pytorch_forward()

    def c_attention():
        lib.attention_forward_causal_head_major_gqa_bf16(
            numpy_to_uint16_ptr(q_bf),
            numpy_to_uint16_ptr(k_bf),
            numpy_to_uint16_ptr(v_bf),
            numpy_to_ptr(scores_np),
            numpy_to_ptr(out_np),
            ctypes.c_int(H),
            ctypes.c_int(H),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(D),
            ctypes.c_int(T),
        )

    c_attention()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref.to(dtype=torch.float32))

    report.add_result(TestResult(
        name="Causal Attention",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_attention, warmup=warmup, iterations=iterations, name="C Attention BF16")
    ))

    return report


def run_backward_tests(H=4, T=32, D=32, warmup=10, iterations=200):
    np.random.seed(0)
    q_np = np.random.randn(H, T, D).astype(np.float32)
    k_np = np.random.randn(H, T, D).astype(np.float32)
    v_np = np.random.randn(H, T, D).astype(np.float32)
    d_out_np = np.random.randn(H, T, D).astype(np.float32)

    q = torch.from_numpy(q_np.copy()).to(dtype=torch.bfloat16)
    k = torch.from_numpy(k_np.copy()).to(dtype=torch.bfloat16)
    v = torch.from_numpy(v_np.copy()).to(dtype=torch.bfloat16)
    d_out = torch.from_numpy(d_out_np.copy()).to(dtype=torch.bfloat16)

    _, attn_weights = causal_attention_pytorch(q, k, v)
    attn_weights_np = attn_weights.to(dtype=torch.float32).numpy()

    def pytorch_fwd_bwd():
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        out_ref, _ = causal_attention_pytorch(q_ref, k_ref, v_ref)
        out_ref.backward(d_out)
        return q_ref.grad, k_ref.grad, v_ref.grad

    d_q_ref, d_k_ref, d_v_ref = pytorch_fwd_bwd()

    scores_np = np.zeros((H, T, T), dtype=np.float32)
    d_q_np = np.zeros((H, T, D), dtype=np.float32)
    d_k_np = np.zeros((H, T, D), dtype=np.float32)
    d_v_np = np.zeros((H, T, D), dtype=np.float32)
    d_scores_np = np.zeros((H, T, T), dtype=np.float32)

    q_bf = float32_to_bf16(q_np)
    k_bf = float32_to_bf16(k_np)
    v_bf = float32_to_bf16(v_np)
    d_out_bf = float32_to_bf16(d_out_np)

    report = TestReport(
        test_name="Attention Backward (BF16)",
        dtype="bf16",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    def c_backward():
        lib.attention_backward_causal_head_major_gqa_bf16(
            numpy_to_uint16_ptr(d_out_bf),
            numpy_to_ptr(d_q_np),  # d_x unused, but we pass a float pointer
            numpy_to_uint16_ptr(q_bf),
            numpy_to_uint16_ptr(k_bf),
            numpy_to_uint16_ptr(v_bf),
            numpy_to_ptr(attn_weights_np),
            numpy_to_ptr(d_q_np),
            numpy_to_ptr(d_k_np),
            numpy_to_ptr(d_v_np),
            numpy_to_ptr(d_scores_np),
            ctypes.c_int(H),
            ctypes.c_int(H),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(D),
            ctypes.c_int(T),
        )

    c_backward()

    report.add_result(TestResult(
        name="d_q",
        passed=max_diff(torch.from_numpy(d_q_np.copy()), d_q_ref.to(dtype=torch.float32)) <= 1e-2,
        max_diff=max_diff(torch.from_numpy(d_q_np.copy()), d_q_ref.to(dtype=torch.float32)),
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C Attention BF16")
    ))

    report.add_result(TestResult(
        name="d_k",
        passed=max_diff(torch.from_numpy(d_k_np.copy()), d_k_ref.to(dtype=torch.float32)) <= 1e-2,
        max_diff=max_diff(torch.from_numpy(d_k_np.copy()), d_k_ref.to(dtype=torch.float32)),
        tolerance=1e-2,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="d_v",
        passed=max_diff(torch.from_numpy(d_v_np.copy()), d_v_ref.to(dtype=torch.float32)) <= 1e-2,
        max_diff=max_diff(torch.from_numpy(d_v_np.copy()), d_v_ref.to(dtype=torch.float32)),
        tolerance=1e-2,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    fwd_report = run_forward_tests()
    fwd_report.print_report()

    bwd_report = run_backward_tests()
    bwd_report.print_report()

    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
