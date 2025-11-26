import ctypes
import os

import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    rms_only = os.path.join(root, "libckernel_rmsnorm.so")
    full = os.path.join(root, "libckernel_engine.so")
    lib_path = rms_only if os.path.exists(rms_only) else full
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.POINTER(ctypes.c_float),  # rstd_cache
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_forward.restype = None

lib.rmsnorm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # rstd_cache
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_gamma
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
]
lib.rmsnorm_backward.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    # x: [T,D], gamma: [D]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    x_hat = x * rstd
    return x_hat * gamma


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_forward_test(T=32, D=128, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(T, D, dtype=torch.float32)
    gamma = torch.randn(D, dtype=torch.float32)

    ref = rmsnorm_torch(x, gamma, eps)

    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    out = torch.empty_like(x_f)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.rmsnorm_forward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(out),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )

    diff = max_diff(out, ref)
    print(f"RMSNorm forward max diff: {diff:.2e}")

    tol = 1e-6
    if diff > tol:
        print("RMSNorm forward mismatch. Showing first few elements:")
        for t in range(min(2, T)):
            for d in range(min(5, D)):
                print(
                    f"t={t}, d={d}: x={x[t,d].item():.6f}, "
                    f"ref={ref[t,d].item():.6f}, "
                    f"c={out[t,d].item():.6f}"
                )
        raise AssertionError(f"RMSNorm forward mismatch: max diff {diff}")


def run_backward_test(T=16, D=32, eps=1e-5):
    torch.manual_seed(1)
    x = torch.randn(T, D, dtype=torch.float32)
    gamma = torch.randn(D, dtype=torch.float32)
    upstream = torch.randn(T, D, dtype=torch.float32)

    # PyTorch reference grads
    x_ref = x.clone().detach().requires_grad_(True)
    gamma_ref = gamma.clone().detach().requires_grad_(True)

    y_ref = rmsnorm_torch(x_ref, gamma_ref, eps)
    y_ref.backward(upstream)

    dx_ref = x_ref.grad.detach()
    dgamma_ref = gamma_ref.grad.detach()

    # C forward to get rstd, then backward
    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    out = torch.empty_like(x_f)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.rmsnorm_forward(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(out),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )

    d_input = torch.zeros_like(x_f)
    d_gamma = torch.zeros_like(g_f)

    lib.rmsnorm_backward(
        tensor_to_ptr(upstream),
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(rstd),
        tensor_to_ptr(d_input),
        tensor_to_ptr(d_gamma),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
    )

    diff_dx = max_diff(d_input, dx_ref)
    diff_dgamma = max_diff(d_gamma, dgamma_ref)

    print(f"RMSNorm backward d_input max diff: {diff_dx:.2e}")
    print(f"RMSNorm backward d_gamma max diff: {diff_dgamma:.2e}")

    tol = 1e-6
    if diff_dx > tol or diff_dgamma > tol:
        print("RMSNorm backward mismatch. Showing first few elements for d_input vs PyTorch:")
        for t in range(min(2, T)):
            for d in range(min(5, D)):
                print(
                    f"t={t}, d={d}: upstream={upstream[t,d].item():.6f}, "
                    f"ref_dx={dx_ref[t,d].item():.6f}, "
                    f"c_dx={d_input[t,d].item():.6f}"
                )
        raise AssertionError(
            f"RMSNorm backward mismatch: d_input={diff_dx}, d_gamma={diff_dgamma}"
        )


if __name__ == "__main__":
    run_forward_test()
    run_backward_test()

