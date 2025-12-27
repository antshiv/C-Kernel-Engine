"""
Embedding BF16 kernel unit test (forward).

Compares BF16 embedding lookup (+ optional positional add) against a PyTorch BF16 reference.
"""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport,
    TestResult,
    get_cpu_info,
    max_diff,
    time_function,
    print_system_info,
)
from bf16_utils import float32_to_bf16, bf16_to_float32, numpy_to_uint16_ptr

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)

lib = load_lib("libckernel_engine.so")

lib.embedding_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # token_ids
    ctypes.c_int,                      # token_count
    ctypes.c_int,                      # vocab_size
    ctypes.POINTER(ctypes.c_uint16),   # token_embeddings
    ctypes.POINTER(ctypes.c_uint16),   # pos_embeddings
    ctypes.POINTER(ctypes.c_uint16),   # output
    ctypes.c_int,                      # embed_dim
    ctypes.c_int,                      # aligned_embed_dim
    ctypes.c_int,                      # context_window
    ctypes.c_int,                      # add_pos
]
lib.embedding_forward_bf16.restype = None


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def run_forward_tests(V=2048, T=128, D=96, warmup=10, iterations=1000):
    np.random.seed(0)
    aligned_D = align_up(D, 32)

    token_ids = np.random.randint(0, V, (T,), dtype=np.int32)
    token_emb = np.zeros((V, aligned_D), dtype=np.float32)
    pos_emb = np.zeros((T, aligned_D), dtype=np.float32)

    token_emb[:, :D] = np.random.randn(V, D).astype(np.float32) * 0.02
    pos_emb[:, :D] = np.random.randn(T, D).astype(np.float32) * 0.02

    token_emb_bf16 = float32_to_bf16(token_emb.reshape(-1))
    pos_emb_bf16 = float32_to_bf16(pos_emb.reshape(-1))
    out_bf16 = np.zeros((T, aligned_D), dtype=np.uint16)

    token_ids_t = torch.from_numpy(token_ids).long()
    token_ref = torch.from_numpy(token_emb[:, :D].copy()).to(dtype=torch.bfloat16)
    pos_ref = torch.from_numpy(pos_emb[:, :D].copy()).to(dtype=torch.bfloat16)

    report = TestReport(
        test_name="Embedding Forward (BF16)",
        dtype="bf16",
        shape=f"V={V}, T={T}, D={D}, aligned={aligned_D}",
        cpu_info=get_cpu_info(),
    )

    def pytorch_ref():
        out = token_ref[token_ids_t] + pos_ref
        if aligned_D == D:
            return out
        pad = torch.zeros((T, aligned_D - D), dtype=torch.bfloat16)
        return torch.cat([out, pad], dim=-1)

    ref = pytorch_ref()

    def c_embedding():
        lib.embedding_forward_bf16(
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(T),
            ctypes.c_int(V),
            numpy_to_uint16_ptr(token_emb_bf16),
            numpy_to_uint16_ptr(pos_emb_bf16),
            numpy_to_uint16_ptr(out_bf16.reshape(-1)),
            ctypes.c_int(D),
            ctypes.c_int(aligned_D),
            ctypes.c_int(T),
            ctypes.c_int(1),
        )

    c_embedding()
    out = torch.from_numpy(bf16_to_float32(out_bf16.copy()))
    diff = max_diff(out, ref.to(dtype=torch.float32))

    report.add_result(
        TestResult(
            name="Embedding",
            passed=diff <= 1e-2,
            max_diff=diff,
            tolerance=1e-2,
            pytorch_time=time_function(pytorch_ref, warmup=warmup, iterations=iterations, name="PyTorch"),
            kernel_time=time_function(c_embedding, warmup=warmup, iterations=iterations, name="C Embedding BF16"),
        )
    )

    return report


if __name__ == "__main__":
    print_system_info()

    report = run_forward_tests()
    report.print_report()

    if not report.all_passed():
        sys.exit(1)

