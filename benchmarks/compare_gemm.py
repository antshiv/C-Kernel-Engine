#!/usr/bin/env python3
"""Quick GEMM benchmark: C Kernel vs PyTorch (and optionally oneDNN)."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Callable, Iterable, Optional

import numpy as np
import torch

# Keep OpenMP single-threaded so the results are stable in this sandbox.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKERNEL_LIB = os.path.join(REPO_ROOT, "build", "libckernel_engine.so")
ONEDNN_LIB = os.path.join(REPO_ROOT, "3rd-Party", "MathLibrary", "oneDNN", "build", "src", "libdnnl.so")


def load_ckernel() -> ctypes.CDLL:
    if not os.path.isfile(CKERNEL_LIB):
        raise FileNotFoundError("Could not find libckernel_engine.so; run `make` first.")
    lib = ctypes.cdll.LoadLibrary(CKERNEL_LIB)
    lib.gemm_blocked_serial.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_blocked_serial.restype = None
    return lib


def load_onednn() -> Optional[ctypes.CDLL]:
    """Return libdnnl.so if it exists, otherwise None."""
    if not os.path.isfile(ONEDNN_LIB):
        return None
    libdnnl = ctypes.cdll.LoadLibrary(ONEDNN_LIB)
    libdnnl.dnnl_sgemm.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    libdnnl.dnnl_sgemm.restype = None
    return libdnnl


def bench_runner(
    name: str,
    runner: Callable[[], None],
    iterations: int = 5,
    skip_first: bool = False,
) -> float:
    """Run ``runner`` several times and return the minimum duration."""
    durations: list[float] = []
    for idx in range(iterations):
        start = time.perf_counter()
        runner()
        elapsed = time.perf_counter() - start
        if skip_first and idx == 0:
            continue
        durations.append(elapsed)
    best = min(durations) if durations else 0.0
    print(f"{name:<20} best: {best * 1e3:.2f} ms over {iterations} runs")
    return best


def main():
    M, N, K = 128, 1024, 256
    rng = np.random.default_rng(0)
    A = np.ascontiguousarray(rng.standard_normal((M, K), dtype=np.float32))
    B = np.ascontiguousarray(rng.standard_normal((K, N), dtype=np.float32))
    B_transposed = B.T.copy(order="C")
    C_ckernel = np.zeros((M, N), dtype=np.float32, order="C")

    # Torch reference.
    torch_a = torch.from_numpy(A)
    torch_b = torch.from_numpy(B)
    torch_out = torch.matmul(torch_a, torch_b)
    ref = torch_out.numpy()

    lib = load_ckernel()
    bias_ptr = ctypes.POINTER(ctypes.c_float)()

    def run_ckernel():
        lib.gemm_blocked_serial(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B_transposed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bias_ptr,
            C_ckernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
        )

    # Validate accuracy once.
    run_ckernel()
    max_diff = np.max(np.abs(C_ckernel - ref))
    print(f"reference max diff (C kernel vs Torch): {max_diff:.2e}")

    # Run comparison table.
    bench_runner("C kernel", run_ckernel)

    bench_runner(
        "PyTorch",
        lambda: torch.matmul(torch_a, torch_b),
    )

    libdnnl = load_onednn()
    if libdnnl is not None:
        C_dnnl = np.zeros((M, N), dtype=np.float32, order="C")

        def run_dnnl():
            libdnnl.dnnl_sgemm(
                b"N",
                b"N",
                ctypes.c_int64(M),
                ctypes.c_int64(N),
                ctypes.c_int64(K),
                ctypes.c_float(1.0),
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(K),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(N),
                ctypes.c_float(0.0),
                C_dnnl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(N),
            )

        run_dnnl()
        print(f"reference max diff (oneDNN vs Torch): {np.max(np.abs(C_dnnl - ref)):.2e}")
        bench_runner("oneDNN", run_dnnl)
    else:
        print("oneDNN binary not found; build under 3rd-Party/MathLibrary/oneDNN to enable this branch.")


if __name__ == "__main__":
    main()
