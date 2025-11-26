import ctypes
import os

import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sig_only = os.path.join(root, "libckernel_sigmoid.so")
    full = os.path.join(root, "libckernel_engine.so")
    lib_path = sig_only if os.path.exists(sig_only) else full
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

lib.sigmoid_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # n
]
lib.sigmoid_forward.restype = None

lib.sigmoid_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.sigmoid_backward.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_forward_test(N=1024):
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)
    ref = torch.sigmoid(x)

    x_f = x.contiguous().float()
    out = torch.empty_like(x_f)

    lib.sigmoid_forward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(out),
        ctypes.c_size_t(N),
    )

    diff = max_diff(out, ref)
    print(f"Sigmoid forward max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("Sigmoid forward mismatch. Showing first few elements:")
        for i in range(min(5, N)):
            print(
                f"i={i}: x={x[i].item():.6f}, "
                f"ref={ref[i].item():.6f}, "
                f"c={out[i].item():.6f}"
            )
        raise AssertionError(f"Sigmoid forward mismatch: max diff {diff}")


def run_backward_test(N=1024):
    torch.manual_seed(1)
    x = torch.randn(N, dtype=torch.float32, requires_grad=True)
    upstream = torch.randn(N, dtype=torch.float32)

    y = torch.sigmoid(x)
    y.backward(upstream)
    dx_ref = x.grad.detach()

    x_f = x.detach().contiguous().float()
    dY = upstream.contiguous().float()
    dX = torch.zeros_like(x_f)

    lib.sigmoid_backward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(dY),
        tensor_to_ptr(dX),
        ctypes.c_size_t(N),
    )

    diff = max_diff(dX, dx_ref)
    print(f"Sigmoid backward d_input max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("Sigmoid backward mismatch. Showing first few elements:")
        for i in range(min(5, N)):
            print(
                f"i={i}: upstream={upstream[i].item():.6f}, "
                f"ref_dx={dx_ref[i].item():.6f}, "
                f"c_dx={dX[i].item():.6f}"
            )
        raise AssertionError(f"Sigmoid backward mismatch: max diff {diff}")


if __name__ == "__main__":
    run_forward_test()
    run_backward_test()

