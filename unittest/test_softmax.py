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
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # aligned_context_window
]
lib.causal_softmax_head_major.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_c_softmax(scores: torch.Tensor) -> torch.Tensor:
    H, T, _ = scores.shape
    aligned = T
    s = scores.contiguous().float()
    ptr = tensor_to_ptr(s)
    lib.causal_softmax_head_major(
        ptr,
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(aligned),
    )
    return s


def softmax_causal_reference(scores: torch.Tensor) -> torch.Tensor:
    H, T, _ = scores.shape
    ref = scores.clone()
    for h in range(H):
        for i in range(T):
            row = ref[h, i, : i + 1]
            ref[h, i, : i + 1] = F.softmax(row, dim=-1)
            ref[h, i, i + 1 :] = 0.0
    return ref


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_test(H=2, T=8):
    torch.manual_seed(0)
    scores = torch.randn(H, T, T, dtype=torch.float32)

    ref = softmax_causal_reference(scores)
    out = run_c_softmax(scores)

    diff = max_diff(out, ref)
    print(f"Causal softmax head-major vs PyTorch ref max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("Softmax forward mismatch. Showing first row for first head:")
        h = 0
        i = 0
        for j in range(min(5, T)):
            print(
                f"j={j}: score={scores[h,i,j].item():.6f}, "
                f"ref={ref[h,i,j].item():.6f}, "
                f"c={out[h,i,j].item():.6f}"
            )
        raise AssertionError(f"Causal softmax forward mismatch: max diff {diff}")


if __name__ == "__main__":
    run_single_test()
