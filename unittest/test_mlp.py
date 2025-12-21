import ctypes

import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

# Signatures from ckernel_engine.h
lib.mlp_token_parallel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # W_fc1
    ctypes.POINTER(ctypes.c_float),  # b_fc1
    ctypes.POINTER(ctypes.c_float),  # W_fc2
    ctypes.POINTER(ctypes.c_float),  # b_fc2
    ctypes.POINTER(ctypes.c_float),  # fc1_output
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_dim
    ctypes.c_int,                    # num_threads
]
lib.mlp_token_parallel.restype = None

lib.gelu_backward_exact.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.c_size_t,                 # n
]
lib.gelu_backward_exact.restype = None

lib.fc2_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # fc2_input
    ctypes.POINTER(ctypes.c_float),  # W_fc2
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_W_fc2
    ctypes.POINTER(ctypes.c_float),  # d_b_fc2
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_in
    ctypes.c_int,                    # aligned_out
    ctypes.c_int,                    # num_threads
]
lib.fc2_backward_kernel.restype = None

lib.fc1_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # fc1_input
    ctypes.POINTER(ctypes.c_float),  # W_fc1
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_W_fc1
    ctypes.POINTER(ctypes.c_float),  # d_b_fc1
    ctypes.c_int,                    # T
    ctypes.c_int,                    # aligned_in
    ctypes.c_int,                    # aligned_out
    ctypes.c_int,                    # num_threads
]
lib.fc1_backward_kernel.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.detach().contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_forward_test(T=8, D=16):
    torch.manual_seed(0)
    fourD = 4 * D

    x = torch.randn(T, D, dtype=torch.float32)
    fc1 = torch.nn.Linear(D, fourD, bias=True)
    fc2 = torch.nn.Linear(fourD, D, bias=True)

    # PyTorch reference
    ref = fc2(F.gelu(fc1(x), approximate="tanh"))

    # Extract weights in our layout:
    # GEMM uses A[M,K], B[N,K]; B is [out × in] with K=in.
    W1 = fc1.weight.detach().contiguous()       # [4D, D]
    b1 = fc1.bias.detach().contiguous()         # [4D]
    W2 = fc2.weight.detach().contiguous()       # [D, 4D]
    b2 = fc2.bias.detach().contiguous()         # [D]

    fc1_out = torch.empty(T, fourD, dtype=torch.float32)
    out = torch.empty(T, D, dtype=torch.float32)

    lib.mlp_token_parallel(
        tensor_to_ptr(x),
        tensor_to_ptr(W1),
        tensor_to_ptr(b1),
        tensor_to_ptr(W2),
        tensor_to_ptr(b2),
        tensor_to_ptr(fc1_out),
        tensor_to_ptr(out),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(1),  # num_threads at kernel level; OpenMP inside C can override
    )

    print(f"MLP forward max diff: {max_diff(out, ref):.2e}")


def run_backward_test(T=8, D=16):
    torch.manual_seed(1)
    fourD = 4 * D

    x = torch.randn(T, D, dtype=torch.float32)
    fc1 = torch.nn.Linear(D, fourD, bias=True)
    fc2 = torch.nn.Linear(fourD, D, bias=True)
    upstream = torch.randn(T, D, dtype=torch.float32)

    # PyTorch reference backward
    x_ref = x.clone().detach().requires_grad_(True)
    fc1_ref = torch.nn.Linear(D, fourD, bias=True)
    fc2_ref = torch.nn.Linear(fourD, D, bias=True)
    fc1_ref.load_state_dict(fc1.state_dict())
    fc2_ref.load_state_dict(fc2.state_dict())

    y_ref = fc2_ref(F.gelu(fc1_ref(x_ref), approximate="tanh"))
    y_ref.backward(upstream)

    dx_ref = x_ref.grad.detach()
    dW1_ref = fc1_ref.weight.grad.detach()
    db1_ref = fc1_ref.bias.grad.detach()
    dW2_ref = fc2_ref.weight.grad.detach()
    db2_ref = fc2_ref.bias.grad.detach()

    # C forward pieces: need fc1_input, fc1_output (pre-GELU), fc2_input (post-GELU)
    x_c = x.contiguous().float()
    W1 = fc1.weight.detach().contiguous()
    b1 = fc1.bias.detach().contiguous()
    W2 = fc2.weight.detach().contiguous()
    b2 = fc2.bias.detach().contiguous()

    fc1_out = torch.empty(T, fourD, dtype=torch.float32)
    out = torch.empty(T, D, dtype=torch.float32)

    lib.mlp_token_parallel(
        tensor_to_ptr(x_c),
        tensor_to_ptr(W1),
        tensor_to_ptr(b1),
        tensor_to_ptr(W2),
        tensor_to_ptr(b2),
        tensor_to_ptr(fc1_out),
        tensor_to_ptr(out),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(1),
    )

    # Now backward: y = fc2(gelu(fc1(x)))
    # Let:
    #   Z1 = fc1(x)
    #   H  = gelu(Z1)
    #   Y  = fc2(H)
    #
    # We have upstream dY = upstream

    dY = upstream.contiguous().float()
    H = fc1_out.clone()  # after GELU in mlp_token_parallel

    # To run FC2 backward, we need fc2_input = H, d_output = dY
    dH = torch.zeros_like(H)
    dW2 = torch.zeros_like(W2)
    db2 = torch.zeros_like(b2)

    lib.fc2_backward_kernel(
        tensor_to_ptr(dY),
        tensor_to_ptr(H),
        tensor_to_ptr(W2),
        tensor_to_ptr(dH),
        tensor_to_ptr(dW2),
        tensor_to_ptr(db2),
        ctypes.c_int(T),
        ctypes.c_int(fourD),
        ctypes.c_int(D),
        ctypes.c_int(1),
    )

    # GELU backward: Z1 = pre-GELU activations
    # We need Z1; to reconstruct Z1 we can recompute fc1(x) once more.
    Z1 = fc1(x_c)  # [T, 4D]
    dZ1 = torch.zeros_like(Z1)

    lib.gelu_backward_exact(
        tensor_to_ptr(Z1),
        tensor_to_ptr(dH),
        tensor_to_ptr(dZ1),
        ctypes.c_size_t(Z1.numel()),
    )

    # FC1 backward: d_output = dZ1, fc1_input = x, W_fc1 = W1
    dx = torch.zeros_like(x_c)
    dW1 = torch.zeros_like(W1)
    db1 = torch.zeros_like(b1)

    lib.fc1_backward_kernel(
        tensor_to_ptr(dZ1),
        tensor_to_ptr(x_c),
        tensor_to_ptr(W1),
        tensor_to_ptr(dx),
        tensor_to_ptr(dW1),
        tensor_to_ptr(db1),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(fourD),
        ctypes.c_int(1),
    )

    print(f"MLP backward dX   max diff: {max_diff(dx, dx_ref):.2e}")
    print(f"MLP backward dW1  max diff: {max_diff(dW1, dW1_ref):.2e}")
    print(f"MLP backward db1  max diff: {max_diff(db1, db1_ref):.2e}")
    print(f"MLP backward dW2  max diff: {max_diff(dW2, dW2_ref):.2e}")
    print(f"MLP backward db2  max diff: {max_diff(db2, db2_ref):.2e}")


if __name__ == "__main__":
    run_forward_test()
    run_backward_test()
