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
lib.gelu_fast_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.gelu_fast_inplace.restype = None


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

    print(f"GELU fast vs PyTorch(tanh) max diff: {max_diff(out, ref):.2e}")


if __name__ == "__main__":
    run_single_test()

