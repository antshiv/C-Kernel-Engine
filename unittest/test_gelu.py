import ctypes
import os

import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_gelu.so", "libckernel_engine.so")
lib.gelu_fast_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.gelu_fast_inplace.restype = None

lib.gelu_backward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_exact.restype = None

lib.gelu_backward_fast.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_fast.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_c_gelu(x: torch.Tensor) -> torch.Tensor:
    x_f = x.contiguous().float().clone()
    ptr = tensor_to_ptr(x_f)
    n = x_f.numel()
    lib.gelu_fast_inplace(ptr, ctypes.c_size_t(n))
    return x_f


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_test(N=1024):
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)

    ref = F.gelu(x, approximate="tanh")
    out = run_c_gelu(x)

    diff = max_diff(out, ref)
    tol = float(os.environ.get("CK_GELU_TOL", "1e-7"))
    print(f"GELU fast vs PyTorch(tanh) max diff: {diff:.2e}")
    # Treat any non-zero diff above tiny FP32 noise as a failure
    if diff > tol:
        print("Forward GELU mismatch detected. Showing first few elements:")
        for i in range(min(5, N)):
            print(f"i={i}: x={x[i].item():.6f}, ref={ref[i].item():.6f}, c={out[i].item():.6f}")
        raise AssertionError(f"GELU forward mismatch: max diff {diff}")


def run_backward_test(N=1024, check_fast: bool = False):
    torch.manual_seed(1)
    x = torch.randn(N, dtype=torch.float32, requires_grad=True)
    upstream = torch.randn(N, dtype=torch.float32)

    # PyTorch reference grad
    y = F.gelu(x, approximate="tanh")
    y.backward(upstream)
    dx_ref = x.grad.detach()

    # C exact backward
    x_in = x.detach().contiguous().float()
    dY = upstream.contiguous().float()
    dX_exact = torch.zeros_like(x_in)

    lib.gelu_backward_exact(
        tensor_to_ptr(x_in),
        tensor_to_ptr(dY),
        tensor_to_ptr(dX_exact),
        ctypes.c_size_t(x_in.numel()),
    )

    # C fast backward
    dX_fast = torch.zeros_like(x_in)
    lib.gelu_backward_fast(
        tensor_to_ptr(x_in),
        tensor_to_ptr(dY),
        tensor_to_ptr(dX_fast),
        ctypes.c_size_t(x_in.numel()),
    )

    exact_diff = max_diff(dX_exact, dx_ref)
    fast_diff = max_diff(dX_fast, dx_ref)

    print(f"GELU backward exact vs PyTorch max diff: {exact_diff:.2e}")
    print(f"GELU backward fast  vs PyTorch max diff: {fast_diff:.2e}")

    # Exact path must match within tiny FP32 noise.
    tol = float(os.environ.get("CK_GELU_TOL", "1e-7"))
    if exact_diff > tol:
        print("Backward GELU (exact) mismatch. Showing first few elements:")
        for i in range(min(5, N)):
            print(
                f"i={i}: x={x_in[i].item():.6f}, "
                f"dY={dY[i].item():.6f}, "
                f"ref={dx_ref[i].item():.6f}, "
                f"c_exact={dX_exact[i].item():.6f}"
            )
        raise AssertionError(f"GELU backward exact mismatch: max diff {exact_diff}")

    # Fast path is approximate; only enforce if explicitly requested.
    if check_fast and fast_diff > 1e-7:
        print("Backward GELU (fast) mismatch. Showing first few elements:")
        for i in range(min(5, N)):
            print(
                f"i={i}: x={x_in[i].item():.6f}, "
                f"dY={dY[i].item():.6f}, "
                f"ref={dx_ref[i].item():.6f}, "
                f"c_fast={dX_fast[i].item():.6f}"
            )
        raise AssertionError(f"GELU backward fast mismatch: max diff {fast_diff}")


if __name__ == "__main__":
    run_single_test()
    # In CI / make test we typically care only about the exact path.
    # Set check_fast=True when explicitly evaluating the approximation quality.
    run_backward_test(check_fast=False)
