"""
Test attention backward kernels against PyTorch autograd.

Validates:
- attention_backward_causal_head_major (non-GQA)
- attention_backward_causal_head_major_gqa (GQA)

Compares d_q, d_k, d_v gradients from C kernel against PyTorch.
"""
import ctypes
import math
import os

import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    lib_path = os.path.join(root, "build", "libckernel_attention.so")
    if not os.path.exists(lib_path):
        # Fallback to old location
        lib_path = os.path.join(root, "libckernel_attention.so")
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

# Forward function for non-GQA
lib.attention_forward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # aligned_context_window
]
lib.attention_forward_causal_head_major.restype = None

# Forward function for GQA
lib.attention_forward_causal_head_major_gqa.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_kv_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # aligned_context_window
]
lib.attention_forward_causal_head_major_gqa.restype = None

# Backward function for non-GQA
lib.attention_backward_causal_head_major.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # attn_weights
    ctypes.POINTER(ctypes.c_float),  # d_q (output)
    ctypes.POINTER(ctypes.c_float),  # d_k (output)
    ctypes.POINTER(ctypes.c_float),  # d_v (output)
    ctypes.POINTER(ctypes.c_float),  # d_scores (scratch)
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # aligned_context_window
]
lib.attention_backward_causal_head_major.restype = None

# Backward function for GQA
lib.attention_backward_causal_head_major_gqa.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # attn_weights
    ctypes.POINTER(ctypes.c_float),  # d_q (output)
    ctypes.POINTER(ctypes.c_float),  # d_k (output)
    ctypes.POINTER(ctypes.c_float),  # d_v (output)
    ctypes.POINTER(ctypes.c_float),  # d_scores (scratch)
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_kv_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # aligned_context_window
]
lib.attention_backward_causal_head_major_gqa.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return (
        t.contiguous()
        .view(-1)
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )


def causal_attention_pytorch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference causal attention.

    Args:
        q: [H, T, D] queries
        k: [H_kv, T, D] keys
        v: [H_kv, T, D] values

    Returns:
        output: [H, T, D]
        attn_weights: [H, T, T]
    """
    H, T, D = q.shape
    H_kv = k.shape[0]
    scale = 1.0 / math.sqrt(D)

    # Compute attention weights for each query head
    attn_weights = torch.zeros(H, T, T, dtype=q.dtype)
    output = torch.zeros_like(q)

    for h in range(H):
        # Map query head to KV head
        kv_h = h * H_kv // H

        for i in range(T):
            # Compute scores: q[h, i] @ k[kv_h, :i+1].T
            qi = q[h, i]  # [D]
            kj = k[kv_h, :i+1, :]  # [i+1, D]
            scores = torch.mv(kj, qi) * scale  # [i+1]

            # Softmax over valid positions
            weights = torch.softmax(scores, dim=-1)  # [i+1]
            attn_weights[h, i, :i+1] = weights

            # Weighted sum of values
            vj = v[kv_h, :i+1, :]  # [i+1, D]
            output[h, i, :] = torch.matmul(weights, vj)  # [D]

    return output, attn_weights


def run_c_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_output: torch.Tensor,
    gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run C kernel forward and backward.

    Returns:
        output, d_q, d_k, d_v
    """
    H, T, D = q.shape
    H_kv = k.shape[0]
    aligned_head_dim = D
    aligned_context_window = T

    # Prepare buffers
    q_buf = q.contiguous().float()
    k_buf = k.contiguous().float()
    v_buf = v.contiguous().float()
    d_out_buf = d_output.contiguous().float()

    scores = torch.zeros(H, T, T, dtype=torch.float32)
    output = torch.zeros(H, T, D, dtype=torch.float32)
    d_q = torch.zeros_like(q_buf)
    d_k = torch.zeros_like(k_buf)
    d_v = torch.zeros_like(v_buf)
    d_scores = torch.zeros(H, T, T, dtype=torch.float32)

    if gqa:
        # Forward pass
        lib.attention_forward_causal_head_major_gqa(
            tensor_to_ptr(q_buf),
            tensor_to_ptr(k_buf),
            tensor_to_ptr(v_buf),
            tensor_to_ptr(scores),
            tensor_to_ptr(output),
            ctypes.c_int(H),
            ctypes.c_int(H_kv),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
            ctypes.c_int(aligned_context_window),
        )

        # Backward pass
        lib.attention_backward_causal_head_major_gqa(
            tensor_to_ptr(d_out_buf),
            tensor_to_ptr(q_buf),
            tensor_to_ptr(k_buf),
            tensor_to_ptr(v_buf),
            tensor_to_ptr(scores),  # scores now contains attention weights after forward
            tensor_to_ptr(d_q),
            tensor_to_ptr(d_k),
            tensor_to_ptr(d_v),
            tensor_to_ptr(d_scores),
            ctypes.c_int(H),
            ctypes.c_int(H_kv),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
            ctypes.c_int(aligned_context_window),
        )
    else:
        # Forward pass (non-GQA: H == H_kv)
        lib.attention_forward_causal_head_major(
            tensor_to_ptr(q_buf),
            tensor_to_ptr(k_buf),
            tensor_to_ptr(v_buf),
            tensor_to_ptr(scores),
            tensor_to_ptr(output),
            ctypes.c_int(H),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
            ctypes.c_int(aligned_context_window),
        )

        # Backward pass
        lib.attention_backward_causal_head_major(
            tensor_to_ptr(d_out_buf),
            tensor_to_ptr(q_buf),
            tensor_to_ptr(k_buf),
            tensor_to_ptr(v_buf),
            tensor_to_ptr(scores),
            tensor_to_ptr(d_q),
            tensor_to_ptr(d_k),
            tensor_to_ptr(d_v),
            tensor_to_ptr(d_scores),
            ctypes.c_int(H),
            ctypes.c_int(T),
            ctypes.c_int(D),
            ctypes.c_int(aligned_head_dim),
            ctypes.c_int(aligned_context_window),
        )

    return output, d_q, d_k, d_v


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_backward_test_non_gqa(H=2, T=8, D=16, seed=0):
    """Test attention backward without GQA (num_heads == num_kv_heads)."""
    print(f"\n=== Non-GQA Backward Test: H={H}, T={T}, D={D} ===")
    torch.manual_seed(seed)

    # Create tensors with gradients
    q = torch.randn(H, T, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(H, T, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(H, T, D, dtype=torch.float32, requires_grad=True)

    # PyTorch forward
    output_ref, _ = causal_attention_pytorch(q, k, v)

    # Random upstream gradient
    d_output = torch.randn(H, T, D, dtype=torch.float32)

    # PyTorch backward
    output_ref.backward(d_output)
    d_q_ref = q.grad.clone()
    d_k_ref = k.grad.clone()
    d_v_ref = v.grad.clone()

    # C kernel forward + backward
    _, d_q_c, d_k_c, d_v_c = run_c_forward_backward(
        q.detach(), k.detach(), v.detach(), d_output, gqa=False
    )

    # Compare
    diff_dq = max_diff(d_q_c, d_q_ref)
    diff_dk = max_diff(d_k_c, d_k_ref)
    diff_dv = max_diff(d_v_c, d_v_ref)

    print(f"d_q max diff: {diff_dq:.2e}")
    print(f"d_k max diff: {diff_dk:.2e}")
    print(f"d_v max diff: {diff_dv:.2e}")

    tol = 1e-4  # Slightly relaxed for accumulated floating point errors
    passed = True

    if diff_dq > tol:
        print(f"FAIL: d_q mismatch (diff={diff_dq:.2e} > tol={tol:.2e})")
        passed = False
    if diff_dk > tol:
        print(f"FAIL: d_k mismatch (diff={diff_dk:.2e} > tol={tol:.2e})")
        passed = False
    if diff_dv > tol:
        print(f"FAIL: d_v mismatch (diff={diff_dv:.2e} > tol={tol:.2e})")
        passed = False

    if passed:
        print("PASS: All gradients match PyTorch!")

    return passed


def run_backward_test_gqa(H=8, H_kv=2, T=8, D=16, seed=0):
    """Test attention backward with GQA (num_heads > num_kv_heads)."""
    print(f"\n=== GQA Backward Test: H={H}, H_kv={H_kv}, T={T}, D={D} ===")
    torch.manual_seed(seed)

    # Create tensors with gradients
    q = torch.randn(H, T, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(H_kv, T, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(H_kv, T, D, dtype=torch.float32, requires_grad=True)

    # PyTorch forward
    output_ref, _ = causal_attention_pytorch(q, k, v)

    # Random upstream gradient
    d_output = torch.randn(H, T, D, dtype=torch.float32)

    # PyTorch backward
    output_ref.backward(d_output)
    d_q_ref = q.grad.clone()
    d_k_ref = k.grad.clone()
    d_v_ref = v.grad.clone()

    # C kernel forward + backward
    _, d_q_c, d_k_c, d_v_c = run_c_forward_backward(
        q.detach(), k.detach(), v.detach(), d_output, gqa=True
    )

    # Compare
    diff_dq = max_diff(d_q_c, d_q_ref)
    diff_dk = max_diff(d_k_c, d_k_ref)
    diff_dv = max_diff(d_v_c, d_v_ref)

    print(f"d_q max diff: {diff_dq:.2e}")
    print(f"d_k max diff: {diff_dk:.2e}")
    print(f"d_v max diff: {diff_dv:.2e}")

    tol = 1e-4
    passed = True

    if diff_dq > tol:
        print(f"FAIL: d_q mismatch (diff={diff_dq:.2e} > tol={tol:.2e})")
        passed = False
    if diff_dk > tol:
        print(f"FAIL: d_k mismatch (diff={diff_dk:.2e} > tol={tol:.2e})")
        passed = False
    if diff_dv > tol:
        print(f"FAIL: d_v mismatch (diff={diff_dv:.2e} > tol={tol:.2e})")
        passed = False

    if passed:
        print("PASS: All GQA gradients match PyTorch!")

    return passed


def run_all_tests():
    """Run all attention backward tests."""
    all_passed = True

    # Non-GQA tests
    all_passed &= run_backward_test_non_gqa(H=2, T=8, D=16)
    all_passed &= run_backward_test_non_gqa(H=4, T=16, D=32)
    all_passed &= run_backward_test_non_gqa(H=1, T=4, D=8)

    # GQA tests
    all_passed &= run_backward_test_gqa(H=8, H_kv=2, T=8, D=16)  # 4:1 ratio
    all_passed &= run_backward_test_gqa(H=4, H_kv=2, T=8, D=16)  # 2:1 ratio
    all_passed &= run_backward_test_gqa(H=6, H_kv=3, T=10, D=24) # 2:1 ratio

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL ATTENTION BACKWARD TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        raise AssertionError("Attention backward test failures")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
