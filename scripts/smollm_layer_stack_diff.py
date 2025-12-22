#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import ctypes

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
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNITTEST_DIR = os.path.join(_ROOT, "unittest")
if _UNITTEST_DIR not in sys.path:
    sys.path.insert(0, _UNITTEST_DIR)
from lib_loader import load_lib


class _DummyDynamo:
    @staticmethod
    def is_compiling() -> bool:
        return False


torch.__dict__["_dynamo"] = _DummyDynamo()


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
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    w = weight.astype(np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :w.shape[1]] = w[src_base:src_base + head_dim, :]
    return buf


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
    parser = argparse.ArgumentParser(description="Per-layer output diffs for SmolLM stack vs PyTorch")
    parser.add_argument("--config", default="smolLM-135.json", help="Model config JSON")
    parser.add_argument("--model-dir", required=True, help="HF model directory")
    parser.add_argument("--context", type=int, default=5, help="Context length")
    parser.add_argument("--text", default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-layers", type=int, default=None, help="Limit layers to check")
    parser.add_argument("--tol", type=float, default=1e-3, help="Max allowed diff")
    args = parser.parse_args()

    cfg = override_context(load_cfg(args.config), args.context)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokens = prepare_tokens(tokenizer, args.text, args.context)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Expected a Llama-style model with model.layers")

    with torch.no_grad():
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        outputs = model.model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    num_layers = len(model.model.layers)
    max_layers = args.max_layers if args.max_layers is not None else num_layers
    max_layers = min(max_layers, num_layers)

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

    rope_cos = None
    rope_sin = None
    if use_rope:
        cos_cache, sin_cache = rope_cache(head_dim, T, rope_theta)
        rope_cos = cos_cache.numpy().astype(np.float32)
        rope_sin = sin_cache.numpy().astype(np.float32)

    lib = load_lib("libckernel_engine.so")
    lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [ctypes.POINTER(CKLayerForwardParams)]
    lib.ck_layer_forward_rmsnorm_swiglu.restype = None

    for layer_idx in range(max_layers):
        layer = model.model.layers[layer_idx]
        x_ref = hidden_states[layer_idx][0].cpu().float().numpy()
        y_ref = hidden_states[layer_idx + 1][0].cpu().float().numpy()

        x = aligned_empty((T, aligned_embed_dim))
        x.fill(0.0)
        x[:, :D] = x_ref

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

        out_c = output[:, :D].astype(np.float32)
        diff = np.max(np.abs(out_c - y_ref))
        print(f"Layer {layer_idx:02d} max_diff={diff:.3e}")
        if diff > args.tol:
            raise SystemExit(f"Layer {layer_idx} diff exceeds tol ({args.tol:g})")


if __name__ == "__main__":
    main()
