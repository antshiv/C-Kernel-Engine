import ctypes
import os

import numpy as np
import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    lib_path = os.path.join(root, "libckernel_engine.so")
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

lib.layernorm_naive_serial.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_naive_serial.restype = None

lib.layernorm_forward_rolled_slice.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_forward_rolled_slice.restype = None

lib.layernorm_forward_unrolled_slice.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_forward_unrolled_slice.restype = None

lib.layernorm_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # mean
    ctypes.POINTER(ctypes.c_float),  # rstd
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_gamma
    ctypes.POINTER(ctypes.c_float),  # d_beta
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
]
lib.layernorm_backward_kernel.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_c_layernorm_naive(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_naive_serial(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def run_c_layernorm_rolled(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_forward_rolled_slice(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def run_c_layernorm_unrolled(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_forward_unrolled_slice(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_single_test(T=32, D=128, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(T, D, dtype=torch.float32)
    gamma = torch.randn(D, dtype=torch.float32)
    beta = torch.randn(D, dtype=torch.float32)

    ref = torch.layer_norm(x, (D,), gamma, beta, eps)

    out_naive, _, _ = run_c_layernorm_naive(x, gamma, beta, eps)
    out_rolled, _, _ = run_c_layernorm_rolled(x, gamma, beta, eps)
    out_unrolled, _, _ = run_c_layernorm_unrolled(x, gamma, beta, eps)

    diff_naive = max_diff(out_naive, ref)
    diff_rolled = max_diff(out_rolled, ref)
    diff_unrolled = max_diff(out_unrolled, ref)

    print(f"Naive vs PyTorch max diff:      {diff_naive:.2e}")
    print(f"Rolled vs PyTorch max diff:     {diff_rolled:.2e}")
    print(f"Unrolled vs PyTorch max diff:   {diff_unrolled:.2e}")

    # Enforce near-exactness; if this fails, print a few entries.
    tol = 1e-6
    if diff_naive > tol or diff_rolled > tol or diff_unrolled > tol:
        print("LayerNorm forward mismatch detected. Showing first few elements for naive vs PyTorch:")
        for t in range(min(2, T)):
            for d in range(min(5, D)):
                print(
                    f"t={t}, d={d}: x={x[t,d].item():.6f}, "
                    f"ref={ref[t,d].item():.6f}, "
                    f"naive={out_naive[t,d].item():.6f}"
                )
        raise AssertionError(
            f"LayerNorm forward mismatch: "
            f"diff_naive={diff_naive}, diff_rolled={diff_rolled}, diff_unrolled={diff_unrolled}"
        )


def run_backward_test(T=16, D=32, eps=1e-5):
    torch.manual_seed(1)
    x = torch.randn(T, D, dtype=torch.float32)
    gamma = torch.randn(D, dtype=torch.float32)
    beta = torch.randn(D, dtype=torch.float32)
    upstream = torch.randn(T, D, dtype=torch.float32)

    # PyTorch reference grads
    x_ref = x.clone().detach().requires_grad_(True)
    gamma_ref = gamma.clone().detach().requires_grad_(True)
    beta_ref = beta.clone().detach().requires_grad_(True)

    y_ref = torch.layer_norm(x_ref, (D,), gamma_ref, beta_ref, eps)
    y_ref.backward(upstream)

    dx_ref = x_ref.grad.detach()
    dgamma_ref = gamma_ref.grad.detach()
    dbeta_ref = beta_ref.grad.detach()

    # C forward to get mean/rstd, then C backward
    _, mean, rstd = run_c_layernorm_naive(x, gamma, beta, eps)

    d_input = torch.zeros_like(x)
    d_gamma = torch.zeros_like(gamma)
    d_beta = torch.zeros_like(beta)

    lib.layernorm_backward_kernel(
        tensor_to_ptr(upstream),
        tensor_to_ptr(x),
        tensor_to_ptr(gamma),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        tensor_to_ptr(d_input),
        tensor_to_ptr(d_gamma),
        tensor_to_ptr(d_beta),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
    )

    diff_d_input = max_diff(d_input, dx_ref)
    diff_d_gamma = max_diff(d_gamma, dgamma_ref)
    diff_d_beta = max_diff(d_beta, dbeta_ref)

    print(f"Backward d_input max diff:  {diff_d_input:.2e}")
    print(f"Backward d_gamma max diff:  {diff_d_gamma:.2e}")
    print(f"Backward d_beta max diff:   {diff_d_beta:.2e}")

    tol_bwd = 1e-6
    if diff_d_input > tol_bwd or diff_d_gamma > tol_bwd or diff_d_beta > tol_bwd:
        print("LayerNorm backward mismatch. Showing first few elements for d_input vs PyTorch:")
        for t in range(min(2, T)):
            for d in range(min(5, D)):
                print(
                    f"t={t}, d={d}: upstream={upstream[t,d].item():.6f}, "
                    f"ref_dx={dx_ref[t,d].item():.6f}, "
                    f"c_dx={d_input[t,d].item():.6f}"
                )
        raise AssertionError(
            f"LayerNorm backward mismatch: "
            f"d_input={diff_d_input}, d_gamma={diff_d_gamma}, d_beta={diff_d_beta}"
        )


if __name__ == "__main__":
    run_single_test()
    run_backward_test()
