import ctypes

import numpy as np
import torch

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

lib.embedding_forward.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # token_ids
    ctypes.c_int,  # token_count
    ctypes.c_int,  # vocab_size
    ctypes.POINTER(ctypes.c_float),  # token_embeddings
    ctypes.POINTER(ctypes.c_float),  # pos_embeddings
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,  # embed_dim
    ctypes.c_int,  # aligned_embed_dim
    ctypes.c_int,  # context_window
    ctypes.c_int,  # add_pos
]
lib.embedding_forward.restype = None

lib.embedding_backward.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # token_ids
    ctypes.c_int,  # token_count
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # d_token_embeddings
    ctypes.POINTER(ctypes.c_float),  # d_pos_embeddings
    ctypes.c_int,  # vocab_size
    ctypes.c_int,  # embed_dim
    ctypes.c_int,  # aligned_embed_dim
    ctypes.c_int,  # context_window
    ctypes.c_int,  # add_pos
]
lib.embedding_backward.restype = None


def align_up(n, a):
    return (n + a - 1) // a * a


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    return arr


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_forward_test(add_pos: bool):
    torch.manual_seed(0)
    V = 16
    T = 8
    D = 12
    aligned_D = align_up(D, 16)

    token_ids = torch.randint(0, V, (T,), dtype=torch.int32).numpy()
    token_emb = aligned_empty((V, aligned_D))
    pos_emb = aligned_empty((T, aligned_D))
    out = aligned_empty((T, aligned_D))
    token_emb.fill(0.0)
    pos_emb.fill(0.0)

    token_ref = torch.randn(V, D, dtype=torch.float32)
    pos_ref = torch.randn(T, D, dtype=torch.float32)

    token_emb[:, :D] = token_ref.numpy()
    pos_emb[:, :D] = pos_ref.numpy()

    lib.embedding_forward(
        token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(T),
        ctypes.c_int(V),
        ptr(token_emb),
        ptr(pos_emb) if add_pos else ctypes.POINTER(ctypes.c_float)(),
        ptr(out),
        ctypes.c_int(D),
        ctypes.c_int(aligned_D),
        ctypes.c_int(T),
        ctypes.c_int(1 if add_pos else 0),
    )

    out_t = torch.from_numpy(out[:, :D]).float()
    ref = token_ref[token_ids]
    if add_pos:
        ref = ref + pos_ref

    diff = (out_t - ref).abs().max().item()
    print(f"Embedding forward add_pos={add_pos} max diff: {diff:.2e}")
    if diff > 1e-6:
        raise AssertionError(f"embedding_forward mismatch (add_pos={add_pos}): {diff}")

    pad = np.abs(out[:, D:]).max()
    if pad > 1e-6:
        raise AssertionError(f"embedding_forward padding not zero (add_pos={add_pos}): {pad}")


def run_backward_test():
    torch.manual_seed(1)
    V = 16
    T = 8
    D = 12
    aligned_D = align_up(D, 16)

    token_ids = torch.randint(0, V, (T,), dtype=torch.int32).numpy()
    d_output = aligned_empty((T, aligned_D))
    d_output.fill(0.0)
    d_out_ref = torch.randn(T, D, dtype=torch.float32)
    d_output[:, :D] = d_out_ref.numpy()

    d_tok = aligned_empty((V, aligned_D))
    d_pos = aligned_empty((T, aligned_D))
    d_tok.fill(0.0)
    d_pos.fill(0.0)

    lib.embedding_backward(
        token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(T),
        ptr(d_output),
        ptr(d_tok),
        ptr(d_pos),
        ctypes.c_int(V),
        ctypes.c_int(D),
        ctypes.c_int(aligned_D),
        ctypes.c_int(T),
        ctypes.c_int(1),
    )

    token_ref = torch.randn(V, D, dtype=torch.float32, requires_grad=True)
    pos_ref = torch.randn(T, D, dtype=torch.float32, requires_grad=True)
    out = token_ref[token_ids] + pos_ref
    loss = (out * d_out_ref).sum()
    loss.backward()

    diff_tok = np.max(np.abs(d_tok[:, :D] - token_ref.grad.numpy()))
    diff_pos = np.max(np.abs(d_pos[:, :D] - pos_ref.grad.numpy()))

    print(f"Embedding backward d_token max diff: {diff_tok:.2e}")
    print(f"Embedding backward d_pos max diff: {diff_pos:.2e}")

    if diff_tok > 1e-6 or diff_pos > 1e-6:
        raise AssertionError("embedding_backward mismatch")

    pad_tok = np.abs(d_tok[:, D:]).max()
    pad_pos = np.abs(d_pos[:, D:]).max()
    if pad_tok > 1e-6 or pad_pos > 1e-6:
        raise AssertionError("embedding_backward padding not zero")


if __name__ == "__main__":
    run_forward_test(add_pos=False)
    run_forward_test(add_pos=True)
    run_backward_test()
