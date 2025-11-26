import ctypes
import os

import torch
import torch.nn.functional as F


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    swiglu_only = os.path.join(root, "libckernel_swiglu.so")
    full = os.path.join(root, "libckernel_engine.so")
    lib_path = swiglu_only if os.path.exists(swiglu_only) else full
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

lib.swiglu_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # output [T × D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_forward.restype = None

lib.swiglu_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input [T × 2D]
    ctypes.POINTER(ctypes.c_float),  # d_output [T × D]
    ctypes.POINTER(ctypes.c_float),  # d_input [T × 2D]
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # dim
]
lib.swiglu_backward.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def swiglu_torch(x: torch.Tensor) -> torch.Tensor:
    # x: [T, 2D] => gate, value: [T, D]
    T, twoD = x.shape
    D = twoD // 2
    gate, value = x[:, :D], x[:, D:]
    return F.silu(gate) * value


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_forward_test(T=16, D=32):
    torch.manual_seed(0)
    x = torch.randn(T, 2 * D, dtype=torch.float32)

    ref = swiglu_torch(x)

    x_f = x.contiguous().float()
    out = torch.empty(T, D, dtype=torch.float32)

    lib.swiglu_forward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(out),
        ctypes.c_int(T),
        ctypes.c_int(D),
    )

    diff = max_diff(out, ref)
    print(f"SwiGLU forward max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("SwiGLU forward mismatch. Showing first few elements:")
        for t in range(min(2, T)):
            for d in range(min(5, D)):
                print(
                    f"t={t}, d={d}: x_gate={x[t,d].item():.6f}, "
                    f"x_val={x[t,D+d].item():.6f}, "
                    f"ref={ref[t,d].item():.6f}, "
                    f"c={out[t,d].item():.6f}"
                )
        raise AssertionError(f"SwiGLU forward mismatch: max diff {diff}")


def run_backward_test(T=16, D=32):
    torch.manual_seed(1)
    x = torch.randn(T, 2 * D, dtype=torch.float32)
    upstream = torch.randn(T, D, dtype=torch.float32)

    # PyTorch reference grads
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = swiglu_torch(x_ref)
    y_ref.backward(upstream)
    dx_ref = x_ref.grad.detach()

    # C forward/backward
    x_f = x.contiguous().float()
    dY = upstream.contiguous().float()
    dX = torch.zeros_like(x_f)

    lib.swiglu_backward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(dY),
        tensor_to_ptr(dX),
        ctypes.c_int(T),
        ctypes.c_int(D),
    )

    diff_dx = max_diff(dX, dx_ref)
    print(f"SwiGLU backward d_input max diff: {diff_dx:.2e}")

    tol = 1e-6
    if diff_dx > tol:
        print("SwiGLU backward mismatch. Showing first few elements:")
        for t in range(min(2, T)):
            for d in range(min(5, 2 * D)):
                print(
                    f"t={t}, d={d}: upstream={upstream[t,d % D].item() if d < D else 0.0:.6f}, "
                    f"ref_dx={dx_ref[t,d].item():.6f}, "
                    f"c_dx={dX[t,d].item():.6f}"
                )
        raise AssertionError(f"SwiGLU backward mismatch: d_input max diff {diff_dx}")


if __name__ == "__main__":
    run_forward_test()
    run_backward_test()

