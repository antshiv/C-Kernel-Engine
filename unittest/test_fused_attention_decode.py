"""
Fused attention decode parity test (PyTorch vs C kernel).

Validates the fused attention decode path (QKV + RoPE + KV cache + attention + Wo)
against a PyTorch reference for a full layer decode run.
"""
import argparse
import ctypes
import math

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import TestReport, TestResult, get_cpu_info, max_diff, print_system_info


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


lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn.argtypes = [
    ctypes.POINTER(CKLayerForwardParams),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn.restype = None

lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp.argtypes = [
    ctypes.POINTER(CKLayerForwardParams),
    ctypes.c_int,
    ctypes.c_int,
]
lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp.restype = None


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


def rope_cache(head_dim, max_seq_len, base=10000.0):
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache, sin_cache


def rope_apply(x, cos_cache, sin_cache, pos_offset=0):
    t_len, d_model = x.shape
    half_dim = d_model // 2
    out = x.clone()
    for t in range(t_len):
        cos_row = cos_cache[pos_offset + t]
        sin_row = sin_cache[pos_offset + t]
        for i in range(half_dim):
            x0 = x[t, i]
            x1 = x[t, i + half_dim]
            c = cos_row[i]
            s = sin_row[i]
            out[t, i] = x0 * c - x1 * s
            out[t, i + half_dim] = x0 * s + x1 * c
    return out


def layer_forward_ref(x,
                      ln1_gamma,
                      ln2_gamma,
                      wq, bq,
                      wk, bk,
                      wv, bv,
                      wo, bo,
                      w1, b1,
                      w2, b2,
                      num_heads,
                      num_kv_heads,
                      eps,
                      cos_cache,
                      sin_cache,
                      rope_offset=0):
    t_len, d_model = x.shape
    head_dim = d_model // num_heads

    h1 = rmsnorm_ref(x, ln1_gamma, eps)

    attn = torch.zeros(num_heads, t_len, head_dim, dtype=torch.float32)
    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        wq_h = wq[h]
        wk_h = wk[kv_head]
        wv_h = wv[kv_head]
        bq_h = bq[h]
        bk_h = bk[kv_head]
        bv_h = bv[kv_head]

        q_h = h1 @ wq_h.t() + bq_h
        k_h = h1 @ wk_h.t() + bk_h
        v_h = h1 @ wv_h.t() + bv_h

        if cos_cache is not None and sin_cache is not None:
            q_h = rope_apply(q_h, cos_cache, sin_cache, pos_offset=rope_offset)
            k_h = rope_apply(k_h, cos_cache, sin_cache, pos_offset=rope_offset)

        scores = (q_h @ k_h.t()) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn[h] = weights @ v_h

    proj = torch.zeros(t_len, d_model, dtype=torch.float32)
    for h in range(num_heads):
        wo_h = wo[h]
        proj += attn[h] @ wo_h.t()
    proj = proj + bo

    res1 = x + proj
    h2 = rmsnorm_ref(res1, ln2_gamma, eps)

    up = h2 @ w1.t() + b1
    gate, value = up[:, : w2.shape[1]], up[:, w2.shape[1]:]
    swiglu = F.silu(gate) * value
    mlp = swiglu @ w2.t() + b2
    return res1 + mlp


def run_decode(func, params, total_len, cache_capacity):
    for t in range(total_len):
        params.rope_pos_offset = t
        func(ctypes.byref(params), t, cache_capacity)
    return params


def build_buffers(cache_capacity,
                  embed_dim,
                  aligned_embed_dim,
                  num_heads,
                  num_kv_heads,
                  aligned_head_dim,
                  aligned_context_window,
                  aligned_intermediate_dim):
    return {
        "ln1_out": aligned_empty((cache_capacity, aligned_embed_dim)),
        "ln1_rstd": aligned_empty((cache_capacity,)),
        "q": aligned_empty((num_heads, cache_capacity, aligned_head_dim)),
        "k": aligned_empty((num_kv_heads, cache_capacity, aligned_head_dim)),
        "v": aligned_empty((num_kv_heads, cache_capacity, aligned_head_dim)),
        "scores": aligned_empty((num_heads, aligned_context_window, aligned_context_window)),
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


def run_case(seed=0,
             cache_capacity=16,
             total_len=12,
             embed_dim=48,
             num_heads=4,
             num_kv_heads=2,
             intermediate_dim=64,
             eps=1e-5,
             rope_theta=10000.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    head_dim = embed_dim // num_heads

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

    cos_cache, sin_cache = rope_cache(head_dim, cache_capacity, base=rope_theta)
    cos_cache_np = cos_cache.numpy()
    sin_cache_np = sin_cache.numpy()

    x_ref = torch.from_numpy(x[:total_len, :embed_dim]).float()
    ln1_gamma_ref = torch.from_numpy(ln1_gamma[:embed_dim]).float()
    ln2_gamma_ref = torch.from_numpy(ln2_gamma[:embed_dim]).float()

    wq_ref = torch.from_numpy(wq[:, :head_dim, :embed_dim]).float()
    wk_ref = torch.from_numpy(wk[:, :head_dim, :embed_dim]).float()
    wv_ref = torch.from_numpy(wv[:, :head_dim, :embed_dim]).float()
    bq_ref = torch.from_numpy(bq[:, :head_dim]).float()
    bk_ref = torch.from_numpy(bk[:, :head_dim]).float()
    bv_ref = torch.from_numpy(bv[:, :head_dim]).float()
    wo_ref = torch.from_numpy(wo[:, :embed_dim, :head_dim]).float()
    bo_ref = torch.from_numpy(bo[:embed_dim]).float()
    w1_ref = torch.from_numpy(w1[: 2 * intermediate_dim, :embed_dim]).float()
    b1_ref = torch.from_numpy(b1[: 2 * intermediate_dim]).float()
    w2_ref = torch.from_numpy(w2[:embed_dim, :intermediate_dim]).float()
    b2_ref = torch.from_numpy(b2[:embed_dim]).float()

    ref_out = layer_forward_ref(x_ref,
                                ln1_gamma_ref,
                                ln2_gamma_ref,
                                wq_ref, bq_ref,
                                wk_ref, bk_ref,
                                wv_ref, bv_ref,
                                wo_ref, bo_ref,
                                w1_ref, b1_ref,
                                w2_ref, b2_ref,
                                num_heads,
                                num_kv_heads,
                                eps,
                                cos_cache,
                                sin_cache,
                                rope_offset=0)

    common_params = dict(
        tokens=cache_capacity,
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
        rope_cos=cos_cache_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rope_sin=sin_cache_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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
    )

    def run_decode_kernel(func):
        buffers = build_buffers(cache_capacity,
                                embed_dim,
                                aligned_embed_dim,
                                num_heads,
                                num_kv_heads,
                                aligned_head_dim,
                                aligned_context_window,
                                aligned_intermediate_dim)
        for buf in buffers.values():
            buf.fill(0.0)
        params = CKLayerForwardParams(
            **common_params,
            ln1_out=ptr(buffers["ln1_out"]),
            ln1_rstd=ptr(buffers["ln1_rstd"]),
            q=ptr(buffers["q"]),
            k=ptr(buffers["k"]),
            v=ptr(buffers["v"]),
            scores=ptr(buffers["scores"]),
            attn_out=ptr(buffers["attn_out"]),
            proj_tmp=ptr(buffers["proj_tmp"]),
            proj_scratch=ptr(buffers["proj_scratch"]),
            residual1=ptr(buffers["residual1"]),
            ln2_out=ptr(buffers["ln2_out"]),
            ln2_rstd=ptr(buffers["ln2_rstd"]),
            fc1_out=ptr(buffers["fc1_out"]),
            swiglu_out=ptr(buffers["swiglu_out"]),
            mlp_out=ptr(buffers["mlp_out"]),
            output=ptr(buffers["output"]),
        )
        run_decode(func, params, total_len, cache_capacity)
        out = torch.from_numpy(buffers["output"][:total_len, :embed_dim]).float()
        return out

    out_fused = run_decode_kernel(lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn)
    out_fused_mlp = run_decode_kernel(lib.ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp)

    diff_fused = max_diff(out_fused, ref_out)
    diff_fused_mlp = max_diff(out_fused_mlp, ref_out)
    return diff_fused, diff_fused_mlp


def main():
    parser = argparse.ArgumentParser(description="Fused attention decode parity test")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Max abs diff tolerance")
    args = parser.parse_args()

    print_system_info()

    diff_fused, diff_fused_mlp = run_case(seed=args.seed)

    report = TestReport(
        test_name="Fused Attention Decode",
        dtype="fp32",
        shape="decode",
        cpu_info=get_cpu_info(),
    )
    report.add_result(TestResult(
        name="fused_attention_decode",
        passed=diff_fused <= args.tolerance,
        max_diff=diff_fused,
        tolerance=args.tolerance,
    ))
    report.add_result(TestResult(
        name="fused_attention_decode_mlp",
        passed=diff_fused_mlp <= args.tolerance,
        max_diff=diff_fused_mlp,
        tolerance=args.tolerance,
    ))

    report.print_report()
    if not report.all_passed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
