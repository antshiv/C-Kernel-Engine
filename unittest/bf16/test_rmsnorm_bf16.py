"""
RMSNorm BF16 kernel unit tests comparing forward and backward wrappers against PyTorch.
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


lib = load_lib("libckernel_rmsnorm.so", "libckernel_engine.so")

lib.rmsnorm_forward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_float),   # gamma
    ctypes.POINTER(ctypes.c_uint16),  # output
    ctypes.POINTER(ctypes.c_float),   # rstd_cache
    ctypes.c_int,                     # tokens
    ctypes.c_int,                     # d_model
    ctypes.c_int,                     # aligned_embed_dim
    ctypes.c_float,                   # eps
]
lib.rmsnorm_forward_bf16.restype = None

lib.rmsnorm_backward_bf16.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # d_output
    ctypes.POINTER(ctypes.c_uint16),  # input
    ctypes.POINTER(ctypes.c_float),   # gamma
    ctypes.POINTER(ctypes.c_float),   # rstd_cache
    ctypes.POINTER(ctypes.c_uint16),  # d_input
    ctypes.POINTER(ctypes.c_float),   # d_gamma
    ctypes.c_int,                     # tokens
    ctypes.c_int,                     # d_model
    ctypes.c_int,                     # aligned_embed_dim
]
lib.rmsnorm_backward_bf16.restype = None


def rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def run_forward_tests(T=64, D=256, eps=1e-5, warmup=10, iterations=1000):
    np.random.seed(0)
    x_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    out_bf16 = np.zeros((T, D), dtype=np.uint16)
    rstd_np = np.zeros(T, dtype=np.float32)

    x_bf16 = float32_to_bf16(x_np)

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    gamma = torch.from_numpy(gamma_np.copy()).to(dtype=torch.bfloat16)

    report = TestReport(
        test_name="RMSNorm Forward (BF16)",
        dtype="bf16",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    def pytorch_ref():
        return rmsnorm_torch(x, gamma, eps)

    pytorch_out = pytorch_ref()

    def c_rmsnorm():
        lib.rmsnorm_forward_bf16(
            numpy_to_uint16_ptr(x_bf16),
            numpy_to_ptr(gamma_np),
            numpy_to_uint16_ptr(out_bf16),
            numpy_to_ptr(rstd_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(eps)
        )

    c_rmsnorm()
    out = torch.from_numpy(bf16_to_float32(out_bf16.copy()))
    diff = max_diff(out, pytorch_out.to(dtype=torch.float32))

    report.add_result(TestResult(
        name="RMSNorm",
        passed=diff <= 1e-2,
        max_diff=diff,
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_ref, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(c_rmsnorm, warmup=warmup, iterations=iterations, name="C RMSNorm BF16")
    ))

    return report


def run_backward_tests(T=64, D=256, eps=1e-5, warmup=10, iterations=1000):
    np.random.seed(1)
    x_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)

    x = torch.from_numpy(x_np.copy()).to(dtype=torch.bfloat16)
    gamma = torch.from_numpy(gamma_np.copy()).to(dtype=torch.bfloat16)
    upstream = torch.from_numpy(upstream_np.copy()).to(dtype=torch.bfloat16)

    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        gamma_ref = gamma.clone().detach().requires_grad_(True)
        out_ref = rmsnorm_torch(x_ref, gamma_ref, eps)
        out_ref.backward(upstream)
        return x_ref.grad, gamma_ref.grad

    dx_ref, dgamma_ref = pytorch_fwd_bwd()

    dx_bf16 = np.zeros((T, D), dtype=np.uint16)
    dgamma_np = np.zeros(D, dtype=np.float32)
    rstd_np = np.zeros(T, dtype=np.float32)
    x_bf16 = float32_to_bf16(x_np)

    upstream_bf16 = float32_to_bf16(upstream_np)
    lib.rmsnorm_forward_bf16(
        numpy_to_uint16_ptr(x_bf16),
        numpy_to_ptr(gamma_np),
        numpy_to_uint16_ptr(dx_bf16),  # reuse buffer to prime rstd cache
        numpy_to_ptr(rstd_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(eps)
    )

    lib.rmsnorm_backward_bf16(
        numpy_to_uint16_ptr(upstream_bf16),
        numpy_to_uint16_ptr(x_bf16),
        numpy_to_ptr(gamma_np),
        numpy_to_ptr(rstd_np),
        numpy_to_uint16_ptr(dx_bf16),
        numpy_to_ptr(dgamma_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
    )

    report = TestReport(
        test_name="RMSNorm Backward (BF16)",
        dtype="bf16",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    dx_c = torch.from_numpy(bf16_to_float32(dx_bf16.copy()))

    report.add_result(TestResult(
        name="d_input",
        passed=max_diff(dx_c, dx_ref) <= 1e-2,
        max_diff=max_diff(dx_c, dx_ref),
        tolerance=1e-2,
        pytorch_time=time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch"),
        kernel_time=time_function(
            lambda: lib.rmsnorm_backward_bf16(
                numpy_to_uint16_ptr(upstream_bf16),
                numpy_to_uint16_ptr(x_bf16),
                numpy_to_ptr(gamma_np),
                numpy_to_ptr(rstd_np),
                numpy_to_uint16_ptr(dx_bf16),
                numpy_to_ptr(dgamma_np),
                ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
            ),
            warmup=warmup, iterations=iterations, name="C RMSNorm BF16 Bwd"
        )
    ))

    report.add_result(TestResult(
        name="d_gamma",
        passed=max_diff(torch.from_numpy(dgamma_np), dgamma_ref) <= 1e-2,
        max_diff=max_diff(torch.from_numpy(dgamma_np), dgamma_ref),
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
