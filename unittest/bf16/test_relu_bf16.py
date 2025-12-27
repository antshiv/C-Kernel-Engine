"""
ReLU BF16 kernel unit tests.

Compares BF16 forward/backward kernels against PyTorch BF16 reference.
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

lib = load_lib("libckernel_relu.so", "libckernel_engine.so")

lib.relu_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_size_t,
]
lib.relu_forward_bf16.restype = None

lib.relu_forward_inplace_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_size_t,
]
lib.relu_forward_inplace_bf16.restype = None

lib.relu_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_size_t,
]
lib.relu_backward_bf16.restype = None


def run_forward_tests(N=4096, warmup=10, iterations=2000):
    np.random.seed(0)
    x_np = np.random.randn(N).astype(np.float32)
    out_bf16 = np.zeros(N, dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np)
    x_ptr = numpy_to_uint16_ptr(x_bf16)
    out_ptr = numpy_to_uint16_ptr(out_bf16)

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    ref = F.relu(x)

    report = TestReport(
        test_name="ReLU Forward (BF16)",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info(),
    )

    def c_relu():
        lib.relu_forward_bf16(x_ptr, out_ptr, ctypes.c_size_t(N))

    c_relu()
    out = torch.from_numpy(bf16_to_float32(out_bf16.copy()))
    diff = max_diff(out, ref.to(dtype=torch.float32))

    report.add_result(
        TestResult(
            name="ReLU",
            passed=diff <= 1e-2,
            max_diff=diff,
            tolerance=1e-2,
            pytorch_time=time_function(lambda: F.relu(x), warmup=warmup, iterations=iterations, name="PyTorch"),
            kernel_time=time_function(c_relu, warmup=warmup, iterations=iterations, name="C ReLU BF16"),
        )
    )
    return report


def run_backward_tests(N=4096, warmup=10, iterations=2000):
    np.random.seed(1)
    x_np = np.random.randn(N).astype(np.float32)
    upstream_np = np.random.randn(N).astype(np.float32)
    dx_bf16 = np.zeros(N, dtype=np.uint16)

    x_bf16 = float32_to_bf16(x_np)
    upstream_bf16 = float32_to_bf16(upstream_np)

    x_ptr = numpy_to_uint16_ptr(x_bf16)
    upstream_ptr = numpy_to_uint16_ptr(upstream_bf16)
    dx_ptr = numpy_to_uint16_ptr(dx_bf16)

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    upstream = torch.from_numpy(upstream_np.copy()).to(dtype=torch.bfloat16)

    report = TestReport(
        test_name="ReLU Backward (BF16)",
        dtype="bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info(),
    )

    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        y = F.relu(x_ref)
        y.backward(upstream)
        return x_ref.grad

    dx_ref = pytorch_fwd_bwd()

    def c_backward():
        lib.relu_backward_bf16(x_ptr, upstream_ptr, dx_ptr, ctypes.c_size_t(N))

    c_backward()
    dx = torch.from_numpy(bf16_to_float32(dx_bf16.copy()))
    diff = max_diff(dx, dx_ref.to(dtype=torch.float32))

    report.add_result(
        TestResult(
            name="d_input",
            passed=diff <= 1e-2,
            max_diff=diff,
            tolerance=1e-2,
            pytorch_time=time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd"),
            kernel_time=time_function(c_backward, warmup=warmup, iterations=iterations, name="C ReLU Bwd BF16"),
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

