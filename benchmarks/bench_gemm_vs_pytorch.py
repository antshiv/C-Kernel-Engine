#!/usr/bin/env python3
"""
GEMM Performance Benchmark: CKernel vs PyTorch

Usage:
    make bench_gemm    # Compares native and MKL (if available) vs PyTorch
"""

import ctypes
import numpy as np
import torch
import time
import os
import sys

def benchmark_gemm(lib, sizes=[512, 1024, 2048, 4096]):
    """Run GEMM benchmark and return results."""
    lib.gemm_microkernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    results = []
    for sz in sizes:
        A = np.random.randn(sz, sz).astype(np.float32)
        B = np.random.randn(sz, sz).astype(np.float32)
        C = np.zeros((sz, sz), dtype=np.float32)

        At = torch.from_numpy(A.copy())
        Bt = torch.from_numpy(B.copy())

        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Warmup
        for _ in range(3):
            lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, sz, sz, sz, 0)
            _ = torch.mm(At, Bt)

        iters = max(3, 20 // (sz // 512))

        # Benchmark CKernel
        start = time.perf_counter()
        for _ in range(iters):
            lib.gemm_microkernel(A_ptr, B_ptr, C_ptr, sz, sz, sz, 0)
        ck_ms = (time.perf_counter() - start) / iters * 1000

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(iters):
            _ = torch.mm(At, Bt)
        pt_ms = (time.perf_counter() - start) / iters * 1000

        flops = 2.0 * sz * sz * sz
        results.append({
            'size': sz,
            'ck_ms': ck_ms,
            'pt_ms': pt_ms,
            'ck_gflops': flops / (ck_ms * 1e6),
            'pt_gflops': flops / (pt_ms * 1e6),
            'ratio': ck_ms / pt_ms
        })
    return results


def print_results(backend_name, results):
    """Print benchmark results table."""
    print(f"\n{'Size':>10} | {backend_name:>20} | {'PyTorch':>20} | {'Ratio':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']}x{r['size']:>4} | {r['ck_ms']:6.2f}ms {r['ck_gflops']:6.1f}GF/s | "
              f"{r['pt_ms']:6.2f}ms {r['pt_gflops']:6.1f}GF/s | {r['ratio']:6.2f}x")


def main():
    print("=" * 70)
    print("GEMM Performance Benchmark: CKernel vs PyTorch")
    print("=" * 70)

    # PyTorch info
    pt_backend = "MKL" if torch.backends.mkl.is_available() else "OpenBLAS"
    print(f"PyTorch {torch.__version__} backend: {pt_backend}")

    # Check for libraries
    native_lib = os.environ.get('CK_NATIVE_LIB')
    mkl_lib = os.environ.get('CK_MKL_LIB')

    sizes = [512, 1024, 2048]

    # Benchmark Native kernels
    if native_lib and os.path.exists(native_lib):
        try:
            lib = ctypes.CDLL(native_lib)
            lib.gemm_get_backend.restype = ctypes.c_char_p
            backend = lib.gemm_get_backend().decode()
            print(f"\nCKernel Native backend: {backend}")
            results = benchmark_gemm(lib, sizes)
            print_results("CKernel Native", results)
        except Exception as e:
            print(f"Error loading native library: {e}")

    # Benchmark MKL kernels
    if mkl_lib and os.path.exists(mkl_lib):
        try:
            lib = ctypes.CDLL(mkl_lib)
            lib.gemm_get_backend.restype = ctypes.c_char_p
            backend = lib.gemm_get_backend().decode()
            print(f"\nCKernel MKL backend: {backend}")
            results = benchmark_gemm(lib, sizes)
            print_results("CKernel MKL", results)
        except Exception as e:
            print(f"Error loading MKL library: {e}")
    elif not native_lib:
        # Single library mode (original behavior)
        lib_path = os.environ.get('CK_LIB_PATH', 'build/libckernel_engine.so')
        try:
            lib = ctypes.CDLL(lib_path)
            lib.gemm_get_backend.restype = ctypes.c_char_p
            backend = lib.gemm_get_backend().decode()
            print(f"CKernel backend: {backend}")
            results = benchmark_gemm(lib, sizes)
            print_results(f"CKernel {backend}", results)
        except Exception as e:
            print(f"Error loading library: {e}")
            sys.exit(1)

    print()
    print("Ratio < 1.0 = CKernel faster, > 1.0 = PyTorch faster")

    # Check if MKL message needed
    if os.environ.get('CK_MKL_MISSING'):
        print()
        print("=" * 70)
        print("TIP: Install Intel oneAPI for MKL-accelerated GEMM:")
        print("  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html")
        print("  Then run: make bench_gemm_mkl")
        print("=" * 70)


if __name__ == "__main__":
    main()
