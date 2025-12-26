"""
MLP forward BF16 wrapper test that compares the converted kernel against PyTorch.
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
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, time_function, print_system_info
)
from bf16_utils import (
    float32_to_bf16, numpy_to_uint16_ptr
)

cpu = get_cpu_info()
if not cpu.avx512bf16:
    print("BF16 kernels require AVX-512 BF16; skipping this test on the current CPU.")
    sys.exit(0)


lib = load_lib("libckernel_engine.so")

lib.mlp_token_parallel_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # W_fc1
    ctypes.POINTER(ctypes.c_uint16),  # b_fc1
    ctypes.POINTER(ctypes.c_uint16),  # W_fc2
    ctypes.POINTER(ctypes.c_uint16),  # b_fc2
    ctypes.POINTER(ctypes.c_float),   # fc1_output
    ctypes.POINTER(ctypes.c_float),   # output
    ctypes.c_int,                     # T
    ctypes.c_int,                     # aligned_dim
    ctypes.c_int,                     # num_threads
]
lib.mlp_token_parallel_bf16.restype = None


def run_forward_tests(T=64, D=128, warmup=10, iterations=500):
    np.random.seed(0)
    fourD = 4 * D

    x_np = np.random.randn(T, D).astype(np.float32)
    W1_np = np.random.randn(fourD, D).astype(np.float32) * 0.02
    b1_np = np.zeros(fourD, dtype=np.float32)
    W2_np = np.random.randn(D, fourD).astype(np.float32) * 0.02
    b2_np = np.zeros(D, dtype=np.float32)
    fc1_out_np = np.zeros((T, fourD), dtype=np.float32)
    out_np = np.zeros((T, D), dtype=np.float32)

    x_bf = float32_to_bf16(x_np)
    W1_bf = float32_to_bf16(W1_np)
    b1_bf = float32_to_bf16(b1_np)
    W2_bf = float32_to_bf16(W2_np)
    b2_bf = float32_to_bf16(b2_np)

    x = torch.from_numpy(x_np.copy())
    W1 = torch.from_numpy(W1_np.copy())
    b1 = torch.from_numpy(b1_np.copy())
    W2 = torch.from_numpy(W2_np.copy())
    b2 = torch.from_numpy(b2_np.copy())

    report = TestReport(
        test_name="MLP Forward (BF16)",
        dtype="bf16",
        shape=f"T={T}, D={D}, 4D={fourD}",
        cpu_info=get_cpu_info()
    )

    def pytorch_mlp():
        h = F.linear(x, W1, b1)
        h = F.gelu(h, approximate="tanh")
        return F.linear(h, W2, b2)

    ref = pytorch_mlp()

    def c_mlp():
        lib.mlp_token_parallel_bf16(
            numpy_to_uint16_ptr(x_bf),
            numpy_to_uint16_ptr(W1_bf),
            numpy_to_uint16_ptr(b1_bf),
            numpy_to_uint16_ptr(W2_bf),
            numpy_to_uint16_ptr(b2_bf),
            fc1_out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(1)
        )

    c_mlp()
    out = torch.from_numpy(out_np.copy())
    diff = max_diff(out, ref)

    report.add_result(TestResult(
        name="MLP Forward",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_mlp, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_mlp, warmup=warmup, iterations=iterations, name="C MLP BF16")
    ))

    return report


if __name__ == "__main__":
    print_system_info()

    report = run_forward_tests()
    report.print_report()

    if not report.all_passed():
        exit(1)
