#!/usr/bin/env python3
import argparse
import json
import os
import sys
import ctypes
import math

import numpy as np

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _cpu_flags():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    return set(line.split(":")[1].strip().split())
    except OSError:
        return set()
    return set()


def _maybe_force_safe_torch():
    if os.environ.get("CK_TORCH_SAFE") == "0":
        return
    flags = _cpu_flags()
    if os.environ.get("CK_TORCH_SAFE") == "1" or "avx2" not in flags:
        isa = "AVX" if "avx" in flags else "SSE4_2"
        os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
        os.environ.setdefault("DNNL_MAX_CPU_ISA", isa)
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", isa)
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", isa)
        os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
        os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")


_maybe_force_safe_torch()
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNITTEST_DIR = os.path.join(_ROOT, "unittest")
if _UNITTEST_DIR not in sys.path:
    sys.path.insert(0, _UNITTEST_DIR)
from lib_loader import load_lib


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


def rope_cache(head_dim, max_seq_len, base=10000.0):
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache, sin_cache


def rope_apply(x, cos_cache, sin_cache, pos_offset=0):
    T, D = x.shape
    half_dim = D // 2
    out = x.clone()
    for t in range(T):
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


def rmsnorm_ref(x, gamma, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def rmsnorm_rstd_ref(x, eps):
    var = x.pow(2).mean(dim=-1)
    return (var + eps).rsqrt()


def stage_diff(name, c_val, ref_val):
    diff = (c_val - ref_val).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    max_rel = (diff / (ref_val.abs() + 1e-9)).max().item() if diff.numel() else 0.0
    print(f"  {name:12s} max_abs={max_abs:.2e} mean_abs={mean_abs:.2e} max_rel={max_rel:.2e}")
    return max_abs


def prepare_tokens(tokenizer, text, context_len):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    if len(ids) < context_len:
        pad_id = tokenizer.eos_token_id or 0
        ids.extend([pad_id] * (context_len - len(ids)))
    ids = ids[:context_len]
    return np.array(ids, dtype=np.int32)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def override_context(cfg, context_len):
    if context_len is None:
        return cfg
    cfg = dict(cfg)
    cfg["max_position_embeddings"] = int(context_len)
    cfg["context_window"] = int(context_len)
    return cfg


def pack_qkv_rows(weight, num_heads, head_dim, aligned_head_dim, aligned_embed_dim):
    packed = aligned_empty((num_heads, aligned_head_dim, aligned_embed_dim))
    packed.fill(0.0)
    for h in range(num_heads):
        rows = weight[h * head_dim:(h + 1) * head_dim, :]
        packed[h, :head_dim, :rows.shape[1]] = rows
    return packed


def pack_wo_cols(weight, num_heads, head_dim, aligned_head_dim, aligned_embed_dim):
    packed = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim))
    packed.fill(0.0)
    for h in range(num_heads):
        cols = weight[:, h * head_dim:(h + 1) * head_dim]
        packed[h, :cols.shape[0], :head_dim] = cols
    return packed


def pack_w1_gate_up(gate, up, aligned_intermediate_dim, aligned_embed_dim):
    packed = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim))
    packed.fill(0.0)
    packed[: gate.shape[0], : gate.shape[1]] = gate
    packed[aligned_intermediate_dim:aligned_intermediate_dim + up.shape[0], : up.shape[1]] = up
    return packed


def pack_w2(down, aligned_embed_dim, aligned_intermediate_dim):
    packed = aligned_empty((aligned_embed_dim, aligned_intermediate_dim))
    packed.fill(0.0)
    packed[: down.shape[0], : down.shape[1]] = down
    return packed


def main():
    parser = argparse.ArgumentParser(description="Per-stage diff for SmolLM layer vs PyTorch ref")
    parser.add_argument("--config", default=None, help="Model config JSON (default: uses model-dir/config.json)")
    parser.add_argument("--model-dir", required=True, help="HF model directory")
    parser.add_argument("--download-model", action="store_true", help="Download model if missing")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="HF repo id")
    parser.add_argument("--context", type=int, default=5, help="Context length")
    parser.add_argument("--text", default="Once upon a time", help="Prompt text")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    parser.add_argument("--tol", type=float, default=1e-3, help="Max allowed diff")
    parser.add_argument("--dump-stages", action="store_true", help="Print per-stage diffs")
    args = parser.parse_args()

    if args.download_model or not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
        subprocess = __import__("subprocess")
        subprocess.check_call([
            sys.executable,
            "scripts/download_smollm.py",
            "--repo", args.repo,
            "--outdir", args.model_dir,
        ])

    # Use model's config.json by default
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(args.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise SystemExit(f"Config not found: {config_path}. Specify --config or ensure model-dir has config.json")

    cfg = override_context(load_cfg(config_path), args.context)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokens = prepare_tokens(tokenizer, args.text, args.context)

    model = LlamaForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Expected a Llama-style model with model.layers")

    layer = model.model.layers[args.layer]
    embed = model.model.embed_tokens.weight.detach().cpu().float().numpy()

    D = int(cfg["hidden_size"])
    T = int(args.context)
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = D // num_heads
    intermediate_dim = int(cfg["intermediate_size"])
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(cfg.get("rope_theta", 0.0))
    use_rope = rope_theta > 0.0

    aligned_embed_dim = align_up(D, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(T, 16)

    x = aligned_empty((T, aligned_embed_dim))
    x.fill(0.0)
    x[:, :D] = embed[tokens]

    ln1_gamma = aligned_empty((aligned_embed_dim,))
    ln2_gamma = aligned_empty((aligned_embed_dim,))
    ln1_gamma.fill(0.0)
    ln2_gamma.fill(0.0)
    ln1_gamma[:D] = layer.input_layernorm.weight.detach().cpu().float().numpy()
    ln2_gamma[:D] = layer.post_attention_layernorm.weight.detach().cpu().float().numpy()

    q_proj = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()
    k_proj = layer.self_attn.k_proj.weight.detach().cpu().float().numpy()
    v_proj = layer.self_attn.v_proj.weight.detach().cpu().float().numpy()
    o_proj = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()
    gate_proj = layer.mlp.gate_proj.weight.detach().cpu().float().numpy()
    up_proj = layer.mlp.up_proj.weight.detach().cpu().float().numpy()
    down_proj = layer.mlp.down_proj.weight.detach().cpu().float().numpy()

    wq = pack_qkv_rows(q_proj, num_heads, head_dim, aligned_head_dim, aligned_embed_dim)
    wk = pack_qkv_rows(k_proj, num_kv_heads, head_dim, aligned_head_dim, aligned_embed_dim)
    wv = pack_qkv_rows(v_proj, num_kv_heads, head_dim, aligned_head_dim, aligned_embed_dim)
    wo = pack_wo_cols(o_proj, num_heads, head_dim, aligned_head_dim, aligned_embed_dim)
    w1 = pack_w1_gate_up(gate_proj, up_proj, aligned_intermediate_dim, aligned_embed_dim)
    w2 = pack_w2(down_proj, aligned_embed_dim, aligned_intermediate_dim)

    bq = aligned_empty((num_heads, aligned_head_dim))
    bk = aligned_empty((num_kv_heads, aligned_head_dim))
    bv = aligned_empty((num_kv_heads, aligned_head_dim))
    bo = aligned_empty((aligned_embed_dim,))
    b1 = aligned_empty((2 * aligned_intermediate_dim,))
    b2 = aligned_empty((aligned_embed_dim,))
    bq.fill(0.0)
    bk.fill(0.0)
    bv.fill(0.0)
    bo.fill(0.0)
    b1.fill(0.0)
    b2.fill(0.0)

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

    rope_cos = None
    rope_sin = None
    if use_rope:
        cos_cache, sin_cache = rope_cache(head_dim, T, rope_theta)
        rope_cos = cos_cache.numpy().astype(np.float32)
        rope_sin = sin_cache.numpy().astype(np.float32)

    lib = load_lib("libckernel_engine.so")
    lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [ctypes.POINTER(CKLayerForwardParams)]
    lib.ck_layer_forward_rmsnorm_swiglu.restype = None

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
        rope_pos_offset=0,
        input=ptr(x),
        ln1_gamma=ptr(ln1_gamma),
        ln2_gamma=ptr(ln2_gamma),
        rope_cos=ptr(rope_cos) if use_rope else None,
        rope_sin=ptr(rope_sin) if use_rope else None,
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

    h1 = rmsnorm_ref(x_ref, ln1_gamma_ref, eps)

    q_ref = torch.zeros(num_heads, T, head_dim, dtype=torch.float32)
    k_ref = torch.zeros(num_kv_heads, T, head_dim, dtype=torch.float32)
    v_ref = torch.zeros(num_kv_heads, T, head_dim, dtype=torch.float32)
    weights_ref = torch.zeros(num_heads, T, T, dtype=torch.float32)
    attn_ref = torch.zeros(num_heads, T, head_dim, dtype=torch.float32)

    cos_cache = None
    sin_cache = None
    if use_rope:
        cos_cache, sin_cache = rope_cache(head_dim, T, rope_theta)

    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        wq_h = torch.from_numpy(q_proj[h * head_dim:(h + 1) * head_dim, :D]).float()
        wk_h = torch.from_numpy(k_proj[kv_head * head_dim:(kv_head + 1) * head_dim, :D]).float()
        wv_h = torch.from_numpy(v_proj[kv_head * head_dim:(kv_head + 1) * head_dim, :D]).float()

        q_h = h1 @ wq_h.t()
        k_h = h1 @ wk_h.t()
        v_h = h1 @ wv_h.t()

        if use_rope:
            q_h = rope_apply(q_h, cos_cache, sin_cache)
            k_h = rope_apply(k_h, cos_cache, sin_cache)

        q_ref[h] = q_h
        k_ref[kv_head] = k_h
        v_ref[kv_head] = v_h

        scores_h = (q_h @ k_h.t()) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones_like(scores_h), diagonal=1).bool()
        scores_h = scores_h.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores_h, dim=-1)
        weights_ref[h] = weights
        attn_ref[h] = weights @ v_h

    attn_concat = attn_ref.transpose(0, 1).reshape(T, D)
    o_proj_ref = torch.from_numpy(o_proj[:, :D]).float()
    proj = attn_concat @ o_proj_ref.t()

    res1 = x_ref + proj
    h2 = rmsnorm_ref(res1, ln2_gamma_ref, eps)

    gate = h2 @ torch.from_numpy(gate_proj[:, :D]).float().t()
    value = h2 @ torch.from_numpy(up_proj[:, :D]).float().t()
    swiglu = torch.nn.functional.silu(gate) * value
    mlp = swiglu @ torch.from_numpy(down_proj[:D, :]).float().t()
    ref_out = res1 + mlp

    fc1_ref = torch.zeros(T, 2 * aligned_intermediate_dim, dtype=torch.float32)
    fc1_ref[:, :intermediate_dim] = gate
    fc1_ref[:, aligned_intermediate_dim:aligned_intermediate_dim + intermediate_dim] = value

    stages = [
        ("ln1_out", torch.from_numpy(ln1_out[:, :D]).float(), h1),
        ("ln1_rstd", torch.from_numpy(ln1_rstd[:T]).float(), rmsnorm_rstd_ref(x_ref, eps)),
        ("q", torch.from_numpy(q[:, :, :head_dim]).float(), q_ref),
        ("k", torch.from_numpy(k[:, :, :head_dim]).float(), k_ref),
        ("v", torch.from_numpy(v[:, :, :head_dim]).float(), v_ref),
        ("scores", torch.from_numpy(scores[:, :T, :T]).float(), weights_ref),
        ("attn_out", torch.from_numpy(attn_out[:, :, :head_dim]).float(), attn_ref),
        ("proj", torch.from_numpy(proj_tmp[:, :D]).float(), proj),
        ("residual1", torch.from_numpy(residual1[:, :D]).float(), res1),
        ("ln2_out", torch.from_numpy(ln2_out[:, :D]).float(), h2),
        ("ln2_rstd", torch.from_numpy(ln2_rstd[:T]).float(), rmsnorm_rstd_ref(res1, eps)),
        ("fc1_out", torch.from_numpy(fc1_out[:, : 2 * intermediate_dim]).float(), fc1_ref[:, : 2 * intermediate_dim]),
        ("swiglu_out", torch.from_numpy(swiglu_out[:, :intermediate_dim]).float(), swiglu),
        ("mlp_out", torch.from_numpy(mlp_out[:, :D]).float(), mlp),
        ("output", torch.from_numpy(output[:, :D]).float(), ref_out),
    ]

    max_diff = 0.0
    if args.dump_stages:
        print("Stage diffs:")
        for name, c_val, ref_val in stages:
            max_diff = max(max_diff, stage_diff(name, c_val, ref_val))
    else:
        for _, c_val, ref_val in stages:
            diff = (c_val - ref_val).abs().max().item()
            max_diff = max(max_diff, diff)

    print(f"Layer {args.layer} max diff: {max_diff:.2e}")
    if max_diff > args.tol:
        raise SystemExit(f"Layer diff exceeds tol ({args.tol:g})")


if __name__ == "__main__":
    main()
