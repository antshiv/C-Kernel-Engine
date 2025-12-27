"""
SwiGLU BF16 kernel unit tests.

Validates BF16 forward/backward kernels against PyTorch BF16 reference.

Note: our BF16 SwiGLU kernels treat BF16 as a storage format and do the math in
FP32, then round back to BF16. The references below match that contract by
using BF16-quantized inputs, FP32 compute, and BF16-quantized outputs.
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

lib = load_lib("libckernel_swiglu.so", "libckernel_engine.so")

lib.swiglu_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # output
    ctypes.c_int,                     # tokens
    ctypes.c_int,                     # dim
]
lib.swiglu_forward_bf16.restype = None

lib.swiglu_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_uint16),  # d_input
    ctypes.c_int,                     # tokens
    ctypes.c_int,                     # dim
]
lib.swiglu_backward_bf16.restype = None


def run_forward_tests(T=64, D=256, warmup=10, iterations=1000):
    np.random.seed(0)
    x_np = np.random.randn(T, 2 * D).astype(np.float32)
    out_bf16 = np.zeros((T, D), dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np.reshape(-1))
    x_ptr = numpy_to_uint16_ptr(x_bf16)
    out_ptr = numpy_to_uint16_ptr(out_bf16.reshape(-1))

    x_q = torch.from_numpy(bf16_to_float32(x_bf16.reshape(T, 2 * D))).to(dtype=torch.float32)
    gate = x_q[:, :D]
    value = x_q[:, D:]
    ref = (F.silu(gate) * value).to(dtype=torch.bfloat16).to(dtype=torch.float32)

    report = TestReport(
        test_name="SwiGLU Forward (BF16)",
        dtype="bf16",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info(),
    )

    def c_swiglu():
        lib.swiglu_forward_bf16(x_ptr, out_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_swiglu()
    out = torch.from_numpy(bf16_to_float32(out_bf16.copy()))
    diff = max_diff(out, ref)

    report.add_result(
        TestResult(
            name="SwiGLU",
            passed=diff <= 2e-2,
            max_diff=diff,
            tolerance=2e-2,
            pytorch_time=time_function(lambda: (F.silu(gate) * value), warmup=warmup, iterations=iterations, name="PyTorch"),
            kernel_time=time_function(c_swiglu, warmup=warmup, iterations=iterations, name="C SwiGLU BF16"),
        )
    )
    return report


def run_backward_tests(T=64, D=256, warmup=10, iterations=500):
    np.random.seed(1)
    x_np = np.random.randn(T, 2 * D).astype(np.float32)
    dy_np = np.random.randn(T, D).astype(np.float32)
    dx_bf16 = np.zeros((T, 2 * D), dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np.reshape(-1))
    dy_bf16 = float32_to_bf16(dy_np.reshape(-1))

    x_ptr = numpy_to_uint16_ptr(x_bf16)
    dy_ptr = numpy_to_uint16_ptr(dy_bf16)
    dx_ptr = numpy_to_uint16_ptr(dx_bf16.reshape(-1))

    x_q = torch.from_numpy(bf16_to_float32(x_bf16.reshape(T, 2 * D))).to(dtype=torch.float32)
    dy_q = torch.from_numpy(bf16_to_float32(dy_bf16.reshape(T, D))).to(dtype=torch.float32)

    report = TestReport(
        test_name="SwiGLU Backward (BF16)",
        dtype="bf16",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info(),
    )

    def pytorch_fwd_bwd():
        x_ref = x_q.clone().detach().requires_grad_(True)
        gate = x_ref[:, :D]
        value = x_ref[:, D:]
        y = F.silu(gate) * value
        y.backward(dy_q)
        # Kernel returns BF16 gradients; quantize the reference too.
        return x_ref.grad.to(dtype=torch.bfloat16).to(dtype=torch.float32)

    dx_ref = pytorch_fwd_bwd()

    def c_backward():
        lib.swiglu_backward_bf16(x_ptr, dy_ptr, dx_ptr, ctypes.c_int(T), ctypes.c_int(D))

    c_backward()
    dx = torch.from_numpy(bf16_to_float32(dx_bf16.copy()))
    diff = max_diff(dx, dx_ref)

    report.add_result(
        TestResult(
            name="d_input",
            passed=diff <= 3e-2,
            max_diff=diff,
            tolerance=3e-2,
            pytorch_time=time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd"),
            kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C SwiGLU Bwd BF16"),
        )
    )
    return report


if __name__ == "__main__":
    print_system_info()

    fwd_report = run_forward_tests()
    fwd_report.print_report()

    bwd_report = run_backward_tests()
    bwd_report.print_report()

    if not fwd_report.all_passed() or not bwd_report.all_passed():
        sys.exit(1)
