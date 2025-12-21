import ctypes

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

lib.softmax_cross_entropy_loss.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # logits
    ctypes.POINTER(ctypes.c_int32),  # targets
    ctypes.c_int,  # tokens
    ctypes.c_int,  # vocab_size
    ctypes.POINTER(ctypes.c_float),  # d_logits
    ctypes.POINTER(ctypes.c_float),  # loss_out
]
lib.softmax_cross_entropy_loss.restype = None


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_test(T=6, V=13):
    torch.manual_seed(0)
    logits = torch.randn(T, V, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (T,), dtype=torch.int64)

    loss_ref = F.cross_entropy(logits, targets, reduction="mean")
    loss_ref.backward()
    dlogits_ref = logits.grad.detach().cpu().numpy().astype(np.float32)

    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    targets_np = targets.detach().cpu().numpy().astype(np.int32)
    dlogits_np = np.zeros_like(logits_np, dtype=np.float32)
    loss_c = ctypes.c_float(0.0)

    lib.softmax_cross_entropy_loss(
        logits_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        targets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(T),
        ctypes.c_int(V),
        ptr(dlogits_np),
        ctypes.byref(loss_c),
    )

    loss_diff = abs(loss_c.value - float(loss_ref))
    grad_diff = np.max(np.abs(dlogits_np - dlogits_ref))

    print(f"Cross-entropy loss diff: {loss_diff:.2e}")
    print(f"Cross-entropy d_logits max diff: {grad_diff:.2e}")

    if loss_diff > 1e-5 or grad_diff > 1e-5:
        raise AssertionError("softmax_cross_entropy_loss mismatch")


if __name__ == "__main__":
    run_test()
