import ctypes
import math

import torch

from lib_loader import load_lib


lib = load_lib("libckernel_attention.so")

lib.attention_forward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # aligned_context_window
]
lib.attention_forward_causal_head_major.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return (
        t.contiguous()
        .view(-1)
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )


def run_c_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    H, T, D = q.shape
    aligned_head_dim = D
    aligned_context_window = T

    q_buf = q.contiguous().float()
    k_buf = k.contiguous().float()
    v_buf = v.contiguous().float()

    scores = torch.empty(H, T, T, dtype=torch.float32)
    out = torch.empty(H, T, D, dtype=torch.float32)

    lib.attention_forward_causal_head_major(
        tensor_to_ptr(q_buf),
        tensor_to_ptr(k_buf),
        tensor_to_ptr(v_buf),
        tensor_to_ptr(scores),
        tensor_to_ptr(out),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned_head_dim),
        ctypes.c_int(aligned_context_window),
    )

    return out


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    H, T, D = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for h in range(H):
        for i in range(T):
            # Scores over j <= i
            # scores[j] = (q[h,i,:] · k[h,j,:]) / sqrt(D)
            qi = q[h, i]  # [D]
            kj = k[h, : i + 1, :]  # [i+1, D]
            scores = torch.mv(kj, qi) * scale  # [i+1]
            weights = torch.softmax(scores, dim=-1)  # [i+1]

            vj = v[h, : i + 1, :]  # [i+1, D]
            out[h, i, :] = torch.matmul(weights, vj)  # [D]

    return out


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_test(H=2, T=8, D=16):
    torch.manual_seed(0)
    q = torch.randn(H, T, D, dtype=torch.float32)
    k = torch.randn(H, T, D, dtype=torch.float32)
    v = torch.randn(H, T, D, dtype=torch.float32)

    ref = attention_reference(q, k, v)
    out = run_c_attention(q, k, v)

    diff = max_diff(out, ref)
    print(f"Attention (causal, head-major) vs PyTorch ref max diff: {diff:.2e}")

    tol = 1e-5
    if diff > tol:
        print("Attention forward mismatch on small test.")
        raise AssertionError(f"Attention forward mismatch: max diff {diff}")


if __name__ == "__main__":
    run_single_test()
