# GEMM correctness tests against PyTorch for the LLM-specific shapes.

import ctypes

import torch

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

_gemm_sig = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]

lib.gemm_naive_parallel.argtypes = _gemm_sig
lib.gemm_naive_parallel.restype = None
lib.gemm_avx512_parallel.argtypes = _gemm_sig
lib.gemm_avx512_parallel.restype = None
lib.gemm_fine_grained_parallel.argtypes = _gemm_sig
lib.gemm_fine_grained_parallel.restype = None
lib.gemm_blocked_serial.argtypes = _gemm_sig
lib.gemm_blocked_serial.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_kernel(fn, A, B, bias, M, N, K) -> torch.Tensor:
    A_f = A.contiguous().float()
    B_f = B.contiguous().float()
    bias_f = bias.contiguous().float()
    C = torch.empty(M, N, dtype=torch.float32)

    fn(
        tensor_to_ptr(A_f),
        tensor_to_ptr(B_f),
        tensor_to_ptr(bias_f),
        tensor_to_ptr(C),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_shape(M, N, K, desc: str):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(N, K, dtype=torch.float32)  # note: N x K (rows are output channels)
    bias = torch.randn(N, dtype=torch.float32)

    ref = A @ B.t() + bias  # C[i,j] = dot(A[i,:], B[j,:]) + bias[j]

    for name, fn in [
        ("naive", lib.gemm_naive_parallel),
        ("avx512", lib.gemm_avx512_parallel),
        ("fine_grained", lib.gemm_fine_grained_parallel),
        ("blocked_serial", lib.gemm_blocked_serial),
    ]:
        out = run_kernel(fn, A, B, bias, M, N, K)
        d = max_diff(out, ref)
        print(f"{desc}: {name:13s} max diff = {d:.2e}")


def run_all():
    # MLP-style: [T,D] · [D,4D]
    T, D = 16, 64
    run_single_shape(T, 4 * D, D, "MLP [T,D]x[D,4D]")

    # Attention QK^T-like: [T,d] · [T,d]^T (but our kernel uses [M,K]*[N,K])
    T, d = 16, 32
    run_single_shape(T, T, d, "QK^T [T,d]x[T,d]")

    # Attention SV: [T,T] · [T,d] → [T,d]
    T, d = 16, 32
    run_single_shape(T, d, T, "SV [T,T]x[T,d]")


if __name__ == "__main__":
    run_all()
