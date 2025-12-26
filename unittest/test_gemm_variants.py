# Tests for GEMM variants: gemm_nn (C = A @ B) and gemm_tn (C = A.T @ B)
# These are used in backward passes for MLP layers.

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

# gemm_nn: C[M,N] = A[M,K] @ B[K,N] + bias[N]
lib.gemm_nn_parallel.argtypes = _gemm_sig
lib.gemm_nn_parallel.restype = None
lib.gemm_nn_avx512.argtypes = _gemm_sig
lib.gemm_nn_avx512.restype = None
lib.gemm_nn_blocked.argtypes = _gemm_sig
lib.gemm_nn_blocked.restype = None

# gemm_tn: C[M,N] = A[K,M].T @ B[K,N] + bias[N]
lib.gemm_tn_parallel.argtypes = _gemm_sig
lib.gemm_tn_parallel.restype = None
lib.gemm_tn_avx512.argtypes = _gemm_sig
lib.gemm_tn_avx512.restype = None
lib.gemm_tn_blocked.argtypes = _gemm_sig
lib.gemm_tn_blocked.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_gemm_nn(fn, A, B, bias, M, N, K) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[K,N] + bias[N]"""
    A_f = A.contiguous().float()
    B_f = B.contiguous().float()
    bias_f = bias.contiguous().float() if bias is not None else None
    C = torch.empty(M, N, dtype=torch.float32)

    fn(
        tensor_to_ptr(A_f),
        tensor_to_ptr(B_f),
        tensor_to_ptr(bias_f) if bias_f is not None else None,
        tensor_to_ptr(C),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def run_gemm_tn(fn, A, B, bias, M, N, K) -> torch.Tensor:
    """C[M,N] = A[K,M].T @ B[K,N] + bias[N]"""
    A_f = A.contiguous().float()
    B_f = B.contiguous().float()
    bias_f = bias.contiguous().float() if bias is not None else None
    C = torch.empty(M, N, dtype=torch.float32)

    fn(
        tensor_to_ptr(A_f),
        tensor_to_ptr(B_f),
        tensor_to_ptr(bias_f) if bias_f is not None else None,
        tensor_to_ptr(C),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


def test_gemm_nn(M, N, K, desc: str):
    """Test gemm_nn: C[M,N] = A[M,K] @ B[K,N] + bias[N]"""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)  # [K, N] for gemm_nn
    bias = torch.randn(N, dtype=torch.float32)

    # PyTorch reference: C = A @ B + bias
    ref = A @ B + bias

    for name, fn in [
        ("nn_parallel", lib.gemm_nn_parallel),
        ("nn_avx512", lib.gemm_nn_avx512),
        ("nn_blocked", lib.gemm_nn_blocked),
    ]:
        out = run_gemm_nn(fn, A, B, bias, M, N, K)
        d = max_diff(out, ref)
        print(f"{desc}: {name:13s} max diff = {d:.2e}")


def test_gemm_tn(M, N, K, desc: str):
    """Test gemm_tn: C[M,N] = A[K,M].T @ B[K,N] + bias[N]"""
    torch.manual_seed(0)
    A = torch.randn(K, M, dtype=torch.float32)  # [K, M] for gemm_tn
    B = torch.randn(K, N, dtype=torch.float32)  # [K, N]
    bias = torch.randn(N, dtype=torch.float32)

    # PyTorch reference: C = A.T @ B + bias = [M, K] @ [K, N] + [N] = [M, N]
    ref = A.t() @ B + bias

    for name, fn in [
        ("tn_parallel", lib.gemm_tn_parallel),
        ("tn_avx512", lib.gemm_tn_avx512),
        ("tn_blocked", lib.gemm_tn_blocked),
    ]:
        out = run_gemm_tn(fn, A, B, bias, M, N, K)
        d = max_diff(out, ref)
        print(f"{desc}: {name:13s} max diff = {d:.2e}")


def test_gemm_nn_no_bias(M, N, K, desc: str):
    """Test gemm_nn without bias: C[M,N] = A[M,K] @ B[K,N]"""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    ref = A @ B

    for name, fn in [
        ("nn_parallel", lib.gemm_nn_parallel),
        ("nn_avx512", lib.gemm_nn_avx512),
        ("nn_blocked", lib.gemm_nn_blocked),
    ]:
        out = run_gemm_nn(fn, A, B, None, M, N, K)
        d = max_diff(out, ref)
        print(f"{desc} (no bias): {name:13s} max diff = {d:.2e}")


def test_gemm_tn_no_bias(M, N, K, desc: str):
    """Test gemm_tn without bias: C[M,N] = A[K,M].T @ B[K,N]"""
    torch.manual_seed(0)
    A = torch.randn(K, M, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    ref = A.t() @ B

    for name, fn in [
        ("tn_parallel", lib.gemm_tn_parallel),
        ("tn_avx512", lib.gemm_tn_avx512),
        ("tn_blocked", lib.gemm_tn_blocked),
    ]:
        out = run_gemm_tn(fn, A, B, None, M, N, K)
        d = max_diff(out, ref)
        print(f"{desc} (no bias): {name:13s} max diff = {d:.2e}")


def run_all():
    print("=== GEMM_NN tests (C = A @ B + bias) ===")
    # Backward d_input shape: [T, in] = [T, out] @ [out, in]
    T, out_dim, in_dim = 16, 64, 32
    test_gemm_nn(T, in_dim, out_dim, f"d_input [{T},{in_dim}]=[{T},{out_dim}]@[{out_dim},{in_dim}]")

    # MLP backward shape: [T, D] = [T, 4D] @ [4D, D]
    T, D = 16, 64
    test_gemm_nn(T, D, 4*D, f"MLP bwd [{T},{D}]=[{T},{4*D}]@[{4*D},{D}]")

    # No bias variant (used in actual backward)
    test_gemm_nn_no_bias(T, D, 4*D, f"MLP bwd [{T},{D}]")

    print("\n=== GEMM_TN tests (C = A.T @ B + bias) ===")
    # Backward d_W shape: [out, in] = [T, out].T @ [T, in]
    T, out_dim, in_dim = 16, 64, 32
    test_gemm_tn(out_dim, in_dim, T, f"d_W [{out_dim},{in_dim}]=[{T},{out_dim}].T@[{T},{in_dim}]")

    # MLP backward d_W shape: [4D, D] = [T, 4D].T @ [T, D]
    T, D = 16, 64
    test_gemm_tn(4*D, D, T, f"MLP d_W [{4*D},{D}]=[{T},{4*D}].T@[{T},{D}]")

    # No bias variant (used in actual backward)
    test_gemm_tn_no_bias(4*D, D, T, f"MLP d_W [{4*D},{D}]")

    print("\n=== Large shape tests ===")
    # Larger shapes to stress-test vectorization
    T, D = 128, 256
    test_gemm_nn_no_bias(T, D, 4*D, f"Large [{T},{D}]=[{T},{4*D}]@[{4*D},{D}]")
    test_gemm_tn_no_bias(4*D, D, T, f"Large d_W [{4*D},{D}]=[{T},{4*D}].T@[{T},{D}]")


if __name__ == "__main__":
    run_all()
