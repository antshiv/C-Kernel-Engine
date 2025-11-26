import ctypes
import os

import torch
import torch.nn.functional as F


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    lib_path = os.path.join(root, "libckernel_engine.so")
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

lib.causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.causal_softmax_head_major.restype = None

lib.backward_causal_softmax_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.backward_causal_softmax_head_major.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def forward_causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    H, T, _ = scores.shape
    aligned = T
    s = scores.contiguous().float().clone()
    ptr = tensor_to_ptr(s)
    lib.causal_softmax_head_major(ptr, ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(aligned))
    return s


def backward_causal_softmax_ref(dY: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    H, T, _ = Y.shape
    dX = torch.zeros_like(Y)
    for h in range(H):
        for i in range(T):
            y_row = Y[h, i, : i + 1]
            dy_row = dY[h, i, : i + 1]
            dot = (y_row * dy_row).sum()
            dX[h, i, : i + 1] = y_row * (dy_row - dot)
            dX[h, i, i + 1 :] = 0.0
    return dX


def backward_causal_softmax_c(dY: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    H, T, _ = Y.shape
    aligned = T
    d_scores = dY.contiguous().float().clone()
    weights = Y.contiguous().float()

    lib.backward_causal_softmax_head_major(
        tensor_to_ptr(d_scores),
        tensor_to_ptr(weights),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(aligned),
    )
    return d_scores


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_test(H=2, T=8):
    torch.manual_seed(2)
    scores = torch.randn(H, T, T, dtype=torch.float32)

    Y = forward_causal_softmax(scores)
    dY = torch.randn_like(Y)

    dX_ref = backward_causal_softmax_ref(dY, Y)
    dX_c = backward_causal_softmax_c(dY, Y)

    diff = max_diff(dX_c, dX_ref)
    print(f"Backward causal softmax max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("Softmax backward mismatch. Showing first row for first head:")
        h = 0
        i = 0
        for j in range(min(5, T)):
            print(
                f"j={j}: Y={Y[h,i,j].item():.6f}, "
                f"dY={dY[h,i,j].item():.6f}, "
                f"ref_dX={dX_ref[h,i,j].item():.6f}, "
                f"c_dX={dX_c[h,i,j].item():.6f}"
            )
        raise AssertionError(f"Causal softmax backward mismatch: max diff {diff}")


if __name__ == "__main__":
    run_single_test()
