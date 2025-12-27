import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import get_cpu_info, max_diff, print_system_info


lib = load_lib("libckernel_engine.so")


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
        ("rope_pos_offset", ctypes.c_int),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("ln1_gamma", ctypes.POINTER(ctypes.c_float)),
        ("ln2_gamma", ctypes.POINTER(ctypes.c_float)),
        ("rope_cos", ctypes.POINTER(ctypes.c_float)),
        ("rope_sin", ctypes.POINTER(ctypes.c_float)),
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
    ctypes.POINTER(CKLayerForwardParams),
]
lib.ck_layer_forward_rmsnorm_swiglu.restype = None

lib.ck_layer_forward_rmsnorm_swiglu_decode.argtypes = [
    ctypes.POINTER(CKLayerForwardParams),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ck_layer_forward_rmsnorm_swiglu_decode.restype = None

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.rope_precompute_cache.restype = None


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


def run_decode_parity_test(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cache_capacity = 16
    prompt_len = 8
    total_len = 12

    num_heads = 4
    num_kv_heads = 2
    embed_dim = 48
    head_dim = embed_dim // num_heads
    intermediate_dim = 50
    eps = 1e-5
    rope_theta = 10000.0

    aligned_embed_dim = align_up(embed_dim, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(cache_capacity, 16)

    x = aligned_empty((cache_capacity, aligned_embed_dim))
    x.fill(0.0)
    x[:total_len, :embed_dim] = torch.randn(total_len, embed_dim, dtype=torch.float32).numpy()

    ln1_gamma = aligned_empty((aligned_embed_dim,))
    ln2_gamma = aligned_empty((aligned_embed_dim,))
    ln1_gamma.fill(0.0)
    ln2_gamma.fill(0.0)
    ln1_gamma[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()
    ln2_gamma[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    wq = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim)); wq.fill(0.0)
    wk = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim)); wk.fill(0.0)
    wv = aligned_empty((num_kv_heads, aligned_head_dim, aligned_embed_dim)); wv.fill(0.0)
    bq = aligned_empty((num_heads, aligned_head_dim)); bq.fill(0.0)
    bk = aligned_empty((num_kv_heads, aligned_head_dim)); bk.fill(0.0)
    bv = aligned_empty((num_kv_heads, aligned_head_dim)); bv.fill(0.0)

    wq[:, :head_dim, :embed_dim] = torch.randn(num_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    wk[:, :head_dim, :embed_dim] = torch.randn(num_kv_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    wv[:, :head_dim, :embed_dim] = torch.randn(num_kv_heads, head_dim, embed_dim, dtype=torch.float32).numpy()
    bq[:, :head_dim] = torch.randn(num_heads, head_dim, dtype=torch.float32).numpy()
    bk[:, :head_dim] = torch.randn(num_kv_heads, head_dim, dtype=torch.float32).numpy()
    bv[:, :head_dim] = torch.randn(num_kv_heads, head_dim, dtype=torch.float32).numpy()

    wo = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim)); wo.fill(0.0)
    wo[:, :embed_dim, :head_dim] = torch.randn(num_heads, embed_dim, head_dim, dtype=torch.float32).numpy()
    bo = aligned_empty((aligned_embed_dim,)); bo.fill(0.0)
    bo[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    w1 = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim)); w1.fill(0.0)
    b1 = aligned_empty((2 * aligned_intermediate_dim,)); b1.fill(0.0)
    w2 = aligned_empty((aligned_embed_dim, aligned_intermediate_dim)); w2.fill(0.0)
    b2 = aligned_empty((aligned_embed_dim,)); b2.fill(0.0)

    w1[: 2 * intermediate_dim, :embed_dim] = torch.randn(2 * intermediate_dim, embed_dim, dtype=torch.float32).numpy()
    b1[: 2 * intermediate_dim] = torch.randn(2 * intermediate_dim, dtype=torch.float32).numpy()
    w2[:embed_dim, :intermediate_dim] = torch.randn(embed_dim, intermediate_dim, dtype=torch.float32).numpy()
    b2[:embed_dim] = torch.randn(embed_dim, dtype=torch.float32).numpy()

    rope_cos = aligned_empty((cache_capacity, head_dim // 2))
    rope_sin = aligned_empty((cache_capacity, head_dim // 2))
    lib.rope_precompute_cache(ptr(rope_cos), ptr(rope_sin),
                              ctypes.c_int(cache_capacity),
                              ctypes.c_int(head_dim),
                              ctypes.c_float(rope_theta))

    def alloc_buffers():
        return {
            "ln1_out": aligned_empty((cache_capacity, aligned_embed_dim)),
            "ln1_rstd": aligned_empty((cache_capacity,)),
            "q": aligned_empty((num_heads, cache_capacity, aligned_head_dim)),
            "k": aligned_empty((num_kv_heads, cache_capacity, aligned_head_dim)),
            "v": aligned_empty((num_kv_heads, cache_capacity, aligned_head_dim)),
            "attn_out": aligned_empty((num_heads, cache_capacity, aligned_head_dim)),
            "proj_tmp": aligned_empty((cache_capacity, aligned_embed_dim)),
            "proj_scratch": aligned_empty((cache_capacity, aligned_embed_dim)),
            "residual1": aligned_empty((cache_capacity, aligned_embed_dim)),
            "ln2_out": aligned_empty((cache_capacity, aligned_embed_dim)),
            "ln2_rstd": aligned_empty((cache_capacity,)),
            "fc1_out": aligned_empty((cache_capacity, 2 * aligned_intermediate_dim)),
            "swiglu_out": aligned_empty((cache_capacity, aligned_intermediate_dim)),
            "mlp_out": aligned_empty((cache_capacity, aligned_embed_dim)),
            "output": aligned_empty((cache_capacity, aligned_embed_dim)),
        }

    buf_full = alloc_buffers()
    buf_decode = alloc_buffers()

    for buf in (buf_full, buf_decode):
        for arr in buf.values():
            arr.fill(0.0)

    p_full = CKLayerForwardParams(
        tokens=total_len,
        embed_dim=embed_dim,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        rope_pos_offset=0,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos),
        rope_sin=ptr(rope_sin),
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
        ln1_out=ptr(buf_full["ln1_out"]),
        ln1_rstd=ptr(buf_full["ln1_rstd"]),
        q=ptr(buf_full["q"]),
        k=ptr(buf_full["k"]),
        v=ptr(buf_full["v"]),
        scores=None,
        attn_out=ptr(buf_full["attn_out"]),
        proj_tmp=ptr(buf_full["proj_tmp"]),
        proj_scratch=ptr(buf_full["proj_scratch"]),
        residual1=ptr(buf_full["residual1"]),
        ln2_out=ptr(buf_full["ln2_out"]),
        ln2_rstd=ptr(buf_full["ln2_rstd"]),
        fc1_out=ptr(buf_full["fc1_out"]),
        swiglu_out=ptr(buf_full["swiglu_out"]),
        mlp_out=ptr(buf_full["mlp_out"]),
        output=ptr(buf_full["output"]),
    )
    lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(p_full))

    p_decode = CKLayerForwardParams(
        tokens=prompt_len,
        embed_dim=embed_dim,
        aligned_embed_dim=aligned_embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        aligned_head_dim=aligned_head_dim,
        aligned_context_window=aligned_context_window,
        intermediate_dim=intermediate_dim,
        aligned_intermediate_dim=aligned_intermediate_dim,
        eps=eps,
        rope_pos_offset=0,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos),
        rope_sin=ptr(rope_sin),
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
        ln1_out=ptr(buf_decode["ln1_out"]),
        ln1_rstd=ptr(buf_decode["ln1_rstd"]),
        q=ptr(buf_decode["q"]),
        k=ptr(buf_decode["k"]),
        v=ptr(buf_decode["v"]),
        scores=None,
        attn_out=ptr(buf_decode["attn_out"]),
        proj_tmp=ptr(buf_decode["proj_tmp"]),
        proj_scratch=ptr(buf_decode["proj_scratch"]),
        residual1=ptr(buf_decode["residual1"]),
        ln2_out=ptr(buf_decode["ln2_out"]),
        ln2_rstd=ptr(buf_decode["ln2_rstd"]),
        fc1_out=ptr(buf_decode["fc1_out"]),
        swiglu_out=ptr(buf_decode["swiglu_out"]),
        mlp_out=ptr(buf_decode["mlp_out"]),
        output=ptr(buf_decode["output"]),
    )

    # Prefill for prompt tokens (fills KV cache for positions < prompt_len).
    lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(p_decode))

    # Decode remaining tokens one-by-one, updating KV cache.
    for t in range(prompt_len, total_len):
        p_decode.rope_pos_offset = t
        lib.ck_layer_forward_rmsnorm_swiglu_decode(ctypes.byref(p_decode),
                                                   ctypes.c_int(t),
                                                   ctypes.c_int(cache_capacity))

        if head_dim < aligned_head_dim:
            k_slice = buf_decode["k"][:, t, head_dim:aligned_head_dim]
            v_slice = buf_decode["v"][:, t, head_dim:aligned_head_dim]
            if np.any(k_slice != 0.0) or np.any(v_slice != 0.0):
                raise AssertionError("KV cache padded lanes are not zeroed")

    out_full = torch.from_numpy(buf_full["output"][:total_len])
    out_decode = torch.from_numpy(buf_decode["output"][:total_len])

    diff = max_diff(out_full, out_decode)
    return diff


if __name__ == "__main__":
    print_system_info()
    cpu = get_cpu_info()
    diff = run_decode_parity_test(seed=0)
    tol = 1e-5
    print("\n================================================================================")
    print("  TEST: KV-cache Layer Decode Parity")
    print("================================================================================\n")
    print("  SYSTEM INFO")
    print("  ----------------------------------------")
    print(f"  CPU:        {cpu['cpu']}")
    print(f"  Cores:      {cpu['cores']}")
    print(f"  SIMD:       {cpu['simd']}\n")
    print("  ACCURACY")
    print("  ----------------------------------------")
    print(f"  output  max_diff={diff:.2e}  tol={tol:.0e}  [{'PASS' if diff <= tol else 'FAIL'}]\n")

    if diff > tol:
        raise SystemExit(1)

