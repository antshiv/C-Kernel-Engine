import ctypes
import math
import os

import numpy as np
import torch


def load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(root, "libckernel_engine.so"),
        os.path.join(root, "build", "libckernel_engine.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return ctypes.cdll.LoadLibrary(path)
    raise FileNotFoundError("libckernel_engine.so not found")


lib = load_lib()


class CKLayerForwardParams(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_int),
        ("embed_dim", ctypes.c_int),
        ("aligned_embed_dim", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("aligned_head_dim", ctypes.c_int),
        ("aligned_context_window", ctypes.c_int),
        ("intermediate_dim", ctypes.c_int),
        ("aligned_intermediate_dim", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("ln1_gamma", ctypes.POINTER(ctypes.c_float)),
        ("ln2_gamma", ctypes.POINTER(ctypes.c_float)),
        ("wq", ctypes.POINTER(ctypes.c_float)),
        ("bq", ctypes.POINTER(ctypes.c_float)),
        ("wk", ctypes.POINTER(ctypes.c_float)),
        ("bk", ctypes.POINTER(ctypes.c_float)),
        ("wv", ctypes.POINTER(ctypes.c_float)),
        ("bv", ctypes.POINTER(ctypes.c_float)),
        ("wo", ctypes.POINTER(ctypes.c_float)),
        ("bo", ctypes.POINTER(ctypes.c_float)),
        ("w1", ctypes.POINTER(ctypes.c_float)),
        ("b1", ctypes.POINTER(ctypes.c_float)),
        ("w2", ctypes.POINTER(ctypes.c_float)),
        ("b2", ctypes.POINTER(ctypes.c_float)),
        ("ln1_out", ctypes.POINTER(ctypes.c_float)),
        ("ln1_rstd", ctypes.POINTER(ctypes.c_float)),
        ("q", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.POINTER(ctypes.c_float)),
        ("v", ctypes.POINTER(ctypes.c_float)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("attn_out", ctypes.POINTER(ctypes.c_float)),
        ("proj_tmp", ctypes.POINTER(ctypes.c_float)),
        ("proj_scratch", ctypes.POINTER(ctypes.c_float)),
        ("residual1", ctypes.POINTER(ctypes.c_float)),
        ("ln2_out", ctypes.POINTER(ctypes.c_float)),
        ("ln2_rstd", ctypes.POINTER(ctypes.c_float)),
        ("fc1_out", ctypes.POINTER(ctypes.c_float)),
        ("swiglu_out", ctypes.POINTER(ctypes.c_float)),
        ("mlp_out", ctypes.POINTER(ctypes.c_float)),
        ("output", ctypes.POINTER(ctypes.c_float)),
    ]


lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [
    ctypes.POINTER(CKLayerForwardParams)
]
lib.ck_layer_forward_rmsnorm_swiglu.restype = None


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    return arr


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def align_up(n, a):
    return (n + a - 1) // a * a


def rmsnorm_ref(x, gamma, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def run_layer_test(num_kv_heads):
    torch.manual_seed(0)
    T = 8
    D = 32
    num_heads = 4
    head_dim = D // num_heads
    intermediate_dim = 64
    eps = 1e-5

    aligned_embed_dim = align_up(D, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(T, 16)

    x = aligned_empty((T, aligned_embed_dim))
    x[:, :D] = torch.randn(T, D, dtype=torch.float32).numpy()
    x[:, D:] = 0.0

    ln1_gamma = aligned_empty((aligned_embed_dim,))
    ln2_gamma = aligned_empty((aligned_embed_dim,))
    ln1_gamma[:D] = torch.randn(D, dtype=torch.float32).numpy()
    ln2_gamma[:D] = torch.randn(D, dtype=torch.float32).numpy()
    ln1_gamma[D:] = 0.0
    ln2_gamma[D:] = 0.0

    wq = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim))
    wk = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
    wv = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim))
    bq = aligned_empty((num_heads, aligned_head_dim))
    bk = aligned_empty((num_kv_heads, aligned_head_dim))
    bv = aligned_empty((num_kv_heads, aligned_head_dim))
    wq.fill(0.0)
    wk.fill(0.0)
    wv.fill(0.0)
    bq.fill(0.0)
    bk.fill(0.0)
    bv.fill(0.0)

    wq_t = torch.randn(num_heads, head_dim, D, dtype=torch.float32)
    wk_t = torch.randn(num_kv_heads, head_dim, D, dtype=torch.float32)
    wv_t = torch.randn(num_kv_heads, head_dim, D, dtype=torch.float32)
    bq_t = torch.randn(num_heads, head_dim, dtype=torch.float32)
    bk_t = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)
    bv_t = torch.randn(num_kv_heads, head_dim, dtype=torch.float32)

    wq[:, :head_dim, :D] = wq_t.numpy()
    wk[:, :head_dim, :D] = wk_t.numpy()
    wv[:, :head_dim, :D] = wv_t.numpy()
    bq[:, :head_dim] = bq_t.numpy()
    bk[:, :head_dim] = bk_t.numpy()
    bv[:, :head_dim] = bv_t.numpy()

    wo = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim))
    wo.fill(0.0)
    wo_t = torch.randn(num_heads, D, head_dim, dtype=torch.float32)
    wo[:, :D, :head_dim] = wo_t.numpy()

    bo = aligned_empty((aligned_embed_dim,))
    bo[:D] = torch.randn(D, dtype=torch.float32).numpy()
    bo[D:] = 0.0

    w1 = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim))
    b1 = aligned_empty((2 * aligned_intermediate_dim,))
    w2 = aligned_empty((aligned_embed_dim, aligned_intermediate_dim))
    b2 = aligned_empty((aligned_embed_dim,))
    w1.fill(0.0)
    w2.fill(0.0)
    b1.fill(0.0)
    b2.fill(0.0)

    w1_t = torch.randn(2 * intermediate_dim, D, dtype=torch.float32)
    b1_t = torch.randn(2 * intermediate_dim, dtype=torch.float32)
    w2_t = torch.randn(D, intermediate_dim, dtype=torch.float32)
    b2_t = torch.randn(D, dtype=torch.float32)

    w1[: 2 * intermediate_dim, :D] = w1_t.numpy()
    b1[: 2 * intermediate_dim] = b1_t.numpy()
    w2[:D, :intermediate_dim] = w2_t.numpy()
    b2[:D] = b2_t.numpy()

    ln1_out = aligned_empty((T, aligned_embed_dim))
    ln1_rstd = aligned_empty((T,))
    q = aligned_empty((num_heads, T, aligned_head_dim))
    k = aligned_empty((num_kv_heads, T, aligned_head_dim))
    v = aligned_empty((num_kv_heads, T, aligned_head_dim))
    scores = aligned_empty((num_heads, aligned_context_window, aligned_context_window))
    attn_out = aligned_empty((num_heads, T, aligned_head_dim))
    proj_tmp = aligned_empty((T, aligned_embed_dim))
    proj_scratch = aligned_empty((T, aligned_embed_dim))
    residual1 = aligned_empty((T, aligned_embed_dim))
    ln2_out = aligned_empty((T, aligned_embed_dim))
    ln2_rstd = aligned_empty((T,))
    fc1_out = aligned_empty((T, 2 * aligned_intermediate_dim))
    swiglu_out = aligned_empty((T, aligned_intermediate_dim))
    mlp_out = aligned_empty((T, aligned_embed_dim))
    output = aligned_empty((T, aligned_embed_dim))

    params = CKLayerForwardParams(
        tokens=T,
        embed_dim=D,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        wq=ptr(wq),
        bq=ptr(bq),
        wk=ptr(wk),
        bk=ptr(bk),
        wv=ptr(wv),
        bv=ptr(bv),
        wo=ptr(wo),
        bo=ptr(bo),
        w1=ptr(w1),
        b1=ptr(b1),
        w2=ptr(w2),
        b2=ptr(b2),
        ln1_out=ptr(ln1_out),
        ln1_rstd=ptr(ln1_rstd),
        q=ptr(q),
        k=ptr(k),
        v=ptr(v),
        scores=ptr(scores),
        attn_out=ptr(attn_out),
        proj_tmp=ptr(proj_tmp),
        proj_scratch=ptr(proj_scratch),
        residual1=ptr(residual1),
        ln2_out=ptr(ln2_out),
        ln2_rstd=ptr(ln2_rstd),
        fc1_out=ptr(fc1_out),
        swiglu_out=ptr(swiglu_out),
        mlp_out=ptr(mlp_out),
        output=ptr(output),
    )

    lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(params))

    x_ref = torch.from_numpy(x[:, :D]).float()
    ln1_gamma_ref = torch.from_numpy(ln1_gamma[:D]).float()
    ln2_gamma_ref = torch.from_numpy(ln2_gamma[:D]).float()
    bo_ref = torch.from_numpy(bo[:D]).float()
    b1_ref = torch.from_numpy(b1[: 2 * intermediate_dim]).float()
    b2_ref = torch.from_numpy(b2[:D]).float()

    h1 = rmsnorm_ref(x_ref, ln1_gamma_ref, eps)

    attn_heads = []
    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        wq_h = torch.from_numpy(wq[h, :head_dim, :D]).float()
        wk_h = torch.from_numpy(wk[kv_head, :head_dim, :D]).float()
        wv_h = torch.from_numpy(wv[kv_head, :head_dim, :D]).float()
        bq_h = torch.from_numpy(bq[h, :head_dim]).float()
        bk_h = torch.from_numpy(bk[kv_head, :head_dim]).float()
        bv_h = torch.from_numpy(bv[kv_head, :head_dim]).float()

        q_h = h1 @ wq_h.t() + bq_h
        k_h = h1 @ wk_h.t() + bk_h
        v_h = h1 @ wv_h.t() + bv_h

        scores_h = (q_h @ k_h.t()) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones_like(scores_h), diagonal=1).bool()
        scores_h = scores_h.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores_h, dim=-1)
        attn_h = weights @ v_h
        attn_heads.append(attn_h)

    proj = torch.zeros(T, D, dtype=torch.float32)
    for h in range(num_heads):
        wo_h = torch.from_numpy(wo[h, :D, :head_dim]).float()
        proj += attn_heads[h] @ wo_h.t()
    proj += bo_ref

    res1 = x_ref + proj
    h2 = rmsnorm_ref(res1, ln2_gamma_ref, eps)

    w1_ref = torch.from_numpy(w1[: 2 * intermediate_dim, :D]).float()
    w2_ref = torch.from_numpy(w2[:D, :intermediate_dim]).float()
    up = h2 @ w1_ref.t() + b1_ref
    gate, value = up[:, :intermediate_dim], up[:, intermediate_dim:]
    swiglu = torch.nn.functional.silu(gate) * value
    mlp = swiglu @ w2_ref.t() + b2_ref
    ref_out = res1 + mlp

    out_c = torch.from_numpy(output[:, :D]).float()
    diff = (out_c - ref_out).abs().max().item()
    print(f"Layer forward (kv_heads={num_kv_heads}) max diff: {diff:.2e}")
    if diff > 1e-4:
        raise AssertionError(f"Layer forward mismatch: max diff {diff}")


if __name__ == "__main__":
    run_layer_test(num_kv_heads=4)
    run_layer_test(num_kv_heads=2)
