"""
Test RoPE (Rotary Position Embedding) kernels against PyTorch.

Validates:
- rope_precompute_cache
- rope_forward
- rope_backward

Uses the standard Llama-style RoPE formulation.
"""
import ctypes
import math
import os

import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # Try build directory first, then root
    lib_path = os.path.join(root, "build", "libckernel_rope.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(root, "build", "libckernel_engine.so")
    return ctypes.cdll.LoadLibrary(lib_path)


lib = load_lib()

# Precompute cache
lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,  # max_seq_len
    ctypes.c_int,  # head_dim
    ctypes.c_float,  # base
]
lib.rope_precompute_cache.restype = None

# Forward
lib.rope_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x (in-place)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # pos_offset
]
lib.rope_forward.restype = None

# Backward
lib.rope_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_out
    ctypes.POINTER(ctypes.c_float),  # d_x (output)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,  # num_heads
    ctypes.c_int,  # num_tokens
    ctypes.c_int,  # head_dim
    ctypes.c_int,  # aligned_head_dim
    ctypes.c_int,  # pos_offset
]
lib.rope_backward.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return (
        t.contiguous()
        .view(-1)
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )


def precompute_freqs_cis_pytorch(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """
    PyTorch reference: compute cos/sin cache for RoPE.
    Returns cos_cache, sin_cache: [max_seq_len, head_dim/2]
    """
    half_dim = head_dim // 2
    # theta_i = 1 / (base^(2i/d))
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    # positions
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # angles[pos, i] = pos * theta_i
    angles = torch.outer(t, freqs)  # [max_seq_len, half_dim]
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache, sin_cache


def rope_forward_pytorch(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """
    PyTorch reference RoPE forward.
    x: [num_heads, num_tokens, head_dim]
    cos_cache, sin_cache: [max_seq_len, head_dim/2]
    """
    H, T, D = x.shape
    half_dim = D // 2

    x_out = x.clone()

    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]  # [half_dim]
            sin_row = sin_cache[pos]  # [half_dim]

            for i in range(half_dim):
                x0 = x[h, t, 2 * i]
                x1 = x[h, t, 2 * i + 1]
                c = cos_row[i]
                s = sin_row[i]

                x_out[h, t, 2 * i] = x0 * c - x1 * s
                x_out[h, t, 2 * i + 1] = x0 * s + x1 * c

    return x_out


def rope_backward_pytorch(d_out: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """
    PyTorch reference RoPE backward (inverse rotation).
    d_out: [num_heads, num_tokens, head_dim]
    """
    H, T, D = d_out.shape
    half_dim = D // 2

    d_x = torch.zeros_like(d_out)

    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]
            sin_row = sin_cache[pos]

            for i in range(half_dim):
                d0 = d_out[h, t, 2 * i]
                d1 = d_out[h, t, 2 * i + 1]
                c = cos_row[i]
                s = sin_row[i]

                # Inverse rotation: rotate by -theta
                d_x[h, t, 2 * i] = d0 * c + d1 * s
                d_x[h, t, 2 * i + 1] = -d0 * s + d1 * c

    return d_x


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_precompute_cache(max_seq_len: int = 64, head_dim: int = 32, base: float = 10000.0):
    """Test cos/sin cache precomputation."""
    print(f"\n=== Cache Test: max_seq_len={max_seq_len}, head_dim={head_dim} ===")

    half_dim = head_dim // 2

    # PyTorch reference
    cos_ref, sin_ref = precompute_freqs_cis_pytorch(head_dim, max_seq_len, base)

    # C kernel
    cos_c = torch.zeros(max_seq_len, half_dim, dtype=torch.float32)
    sin_c = torch.zeros(max_seq_len, half_dim, dtype=torch.float32)

    lib.rope_precompute_cache(
        tensor_to_ptr(cos_c),
        tensor_to_ptr(sin_c),
        ctypes.c_int(max_seq_len),
        ctypes.c_int(head_dim),
        ctypes.c_float(base),
    )

    diff_cos = max_diff(cos_c, cos_ref)
    diff_sin = max_diff(sin_c, sin_ref)

    print(f"cos cache max diff: {diff_cos:.2e}")
    print(f"sin cache max diff: {diff_sin:.2e}")

    tol = 1e-6
    if diff_cos > tol or diff_sin > tol:
        print("FAIL: Cache precomputation mismatch!")
        return False

    print("PASS: Cache matches PyTorch!")
    return True


def test_forward(H: int = 4, T: int = 16, D: int = 32, seed: int = 0):
    """Test RoPE forward pass."""
    print(f"\n=== Forward Test: H={H}, T={T}, D={D} ===")
    torch.manual_seed(seed)

    x = torch.randn(H, T, D, dtype=torch.float32)

    # Precompute cache
    cos_cache, sin_cache = precompute_freqs_cis_pytorch(D, T)

    # PyTorch reference
    x_ref = rope_forward_pytorch(x, cos_cache, sin_cache)

    # C kernel (in-place)
    x_c = x.clone()
    cos_c = cos_cache.clone()
    sin_c = sin_cache.clone()

    lib.rope_forward(
        tensor_to_ptr(x_c),
        tensor_to_ptr(cos_c),
        tensor_to_ptr(sin_c),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
        ctypes.c_int(0),
    )

    diff = max_diff(x_c, x_ref)
    print(f"Forward max diff: {diff:.2e}")

    tol = 1e-5
    if diff > tol:
        print("FAIL: Forward mismatch!")
        return False

    print("PASS: Forward matches PyTorch!")
    return True


def test_backward(H: int = 4, T: int = 16, D: int = 32, seed: int = 0):
    """Test RoPE backward pass."""
    print(f"\n=== Backward Test: H={H}, T={T}, D={D} ===")
    torch.manual_seed(seed)

    # Random upstream gradient
    d_out = torch.randn(H, T, D, dtype=torch.float32)

    # Precompute cache
    cos_cache, sin_cache = precompute_freqs_cis_pytorch(D, T)

    # PyTorch reference
    d_x_ref = rope_backward_pytorch(d_out, cos_cache, sin_cache)

    # C kernel
    d_x_c = torch.zeros_like(d_out)

    lib.rope_backward(
        tensor_to_ptr(d_out),
        tensor_to_ptr(d_x_c),
        tensor_to_ptr(cos_cache),
        tensor_to_ptr(sin_cache),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
        ctypes.c_int(0),
    )

    diff = max_diff(d_x_c, d_x_ref)
    print(f"Backward max diff: {diff:.2e}")

    tol = 1e-5
    if diff > tol:
        print("FAIL: Backward mismatch!")
        return False

    print("PASS: Backward matches PyTorch!")
    return True


def test_round_trip(H: int = 4, T: int = 16, D: int = 32, seed: int = 0):
    """Test that forward then backward recovers original gradients."""
    print(f"\n=== Round-trip Test: H={H}, T={T}, D={D} ===")
    torch.manual_seed(seed)

    x = torch.randn(H, T, D, dtype=torch.float32, requires_grad=True)

    # Precompute cache
    cos_cache, sin_cache = precompute_freqs_cis_pytorch(D, T)

    # PyTorch forward with autograd
    x_rot = rope_forward_pytorch(x, cos_cache, sin_cache)

    # Random upstream gradient
    d_out = torch.randn_like(x_rot)

    # Compute gradients via autograd manually using our backward reference
    d_x_ref = rope_backward_pytorch(d_out, cos_cache, sin_cache)

    # C kernel forward
    x_c = x.detach().clone()
    lib.rope_forward(
        tensor_to_ptr(x_c),
        tensor_to_ptr(cos_cache),
        tensor_to_ptr(sin_cache),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
        ctypes.c_int(0),
    )

    # C kernel backward
    d_x_c = torch.zeros_like(x)
    lib.rope_backward(
        tensor_to_ptr(d_out),
        tensor_to_ptr(d_x_c),
        tensor_to_ptr(cos_cache),
        tensor_to_ptr(sin_cache),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
        ctypes.c_int(0),
    )

    diff_fwd = max_diff(x_c, x_rot)
    diff_bwd = max_diff(d_x_c, d_x_ref)

    print(f"Forward max diff: {diff_fwd:.2e}")
    print(f"Backward max diff: {diff_bwd:.2e}")

    tol = 1e-5
    if diff_fwd > tol or diff_bwd > tol:
        print("FAIL: Round-trip mismatch!")
        return False

    print("PASS: Round-trip matches PyTorch!")
    return True


def test_pos_offset(H: int = 2, T: int = 8, D: int = 16, offset: int = 5, seed: int = 0):
    """Test RoPE with position offset (for KV cache continuation)."""
    print(f"\n=== Position Offset Test: H={H}, T={T}, D={D}, offset={offset} ===")
    torch.manual_seed(seed)

    x = torch.randn(H, T, D, dtype=torch.float32)

    # Precompute cache for max_seq_len = T + offset
    cos_cache, sin_cache = precompute_freqs_cis_pytorch(D, T + offset)

    # PyTorch reference with offset
    x_ref = rope_forward_pytorch(x, cos_cache, sin_cache, pos_offset=offset)

    # C kernel
    x_c = x.clone()

    lib.rope_forward(
        tensor_to_ptr(x_c),
        tensor_to_ptr(cos_cache),
        tensor_to_ptr(sin_cache),
        ctypes.c_int(H),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
        ctypes.c_int(offset),
    )

    diff = max_diff(x_c, x_ref)
    print(f"Position offset forward max diff: {diff:.2e}")

    tol = 1e-5
    if diff > tol:
        print("FAIL: Position offset mismatch!")
        return False

    print("PASS: Position offset matches PyTorch!")
    return True


def run_all_tests():
    """Run all RoPE tests."""
    all_passed = True

    all_passed &= test_precompute_cache()
    all_passed &= test_forward()
    all_passed &= test_backward()
    all_passed &= test_round_trip()
    all_passed &= test_pos_offset()

    # Additional sizes
    all_passed &= test_forward(H=8, T=32, D=64)
    all_passed &= test_backward(H=8, T=32, D=64)

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL ROPE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        raise AssertionError("RoPE test failures")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
