#!/usr/bin/env python3
import argparse
import json
import os
import struct

import numpy as np


CACHE_ALIGN = 64
FLOAT_SIZE = 4
HEADER_SIZE = 128


def align_up_elems(elems, elem_bytes=FLOAT_SIZE, align_bytes=CACHE_ALIGN):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned_bytes = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned_bytes // elem_bytes


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def pick(cfg, keys, default=None):
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise SystemExit("torch is required to compare weights") from exc


def require_transformers():
    try:
        from transformers import LlamaForCausalLM  # noqa: F401
        return LlamaForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required to compare weights") from exc


def get_state_dict(checkpoint, torch_dtype):
    torch = require_torch()
    LlamaForCausalLM = require_transformers()
    model = LlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    return model.state_dict(), model.config


def get_tensor(state, key, alt_keys=()):
    if key in state:
        return state[key]
    for alt in alt_keys:
        if alt in state:
            return state[alt]
    raise KeyError(f"Missing weight: {key}")


def get_optional(state, key, alt_keys=()):
    if key in state:
        return state[key]
    for alt in alt_keys:
        if alt in state:
            return state[alt]
    return None


def pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    if weight is None:
        return buf
    w = weight.astype(np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :embed_dim] = w[src_base:src_base + head_dim, :embed_dim]
    return buf


def pack_qkv_bias(bias, num_heads, head_dim, aligned_head_dim):
    buf = np.zeros(num_heads * aligned_head_dim, dtype=np.float32)
    if bias is None:
        return buf
    b = bias.astype(np.float32).reshape(-1)
    for h in range(num_heads):
        src_base = h * head_dim
        dst_base = h * aligned_head_dim
        buf[dst_base:dst_base + head_dim] = b[src_base:src_base + head_dim]
    return buf


def pack_wo(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    blocks = []
    w = weight.astype(np.float32) if weight is not None else None
    for h in range(num_heads):
        block = np.zeros((aligned_embed_dim, aligned_head_dim), dtype=np.float32)
        if w is not None:
            col_base = h * head_dim
            block[:embed_dim, :head_dim] = w[:embed_dim, col_base:col_base + head_dim]
        blocks.append(block.reshape(-1))
    return np.concatenate(blocks)


def pack_w1(gate, up, intermediate, aligned_intermediate, embed_dim, aligned_embed_dim):
    w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
    w1[:intermediate, :embed_dim] = gate[:intermediate, :embed_dim].astype(np.float32)
    w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up[:intermediate, :embed_dim].astype(np.float32)
    return w1


def pack_w2(down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim):
    buf = np.zeros((aligned_embed_dim, aligned_intermediate), dtype=np.float32)
    buf[:embed_dim, :intermediate] = down[:embed_dim, :intermediate].astype(np.float32)
    return buf


def skip_bump_header(f):
    magic = f.read(8)
    if magic == b"BUMPWGT2":
        f.seek(HEADER_SIZE, 0)
        return True
    f.seek(0, 0)
    return False


def read_floats(f, count):
    data = np.fromfile(f, dtype=np.float32, count=count)
    if data.size != count:
        raise SystemExit(f"Failed to read {count} floats (got {data.size})")
    return data


def max_diff(name, got, ref):
    diff = np.max(np.abs(got - ref))
    print(f"{name:24s} max_diff={diff:.3e}")
    return diff


def main():
    parser = argparse.ArgumentParser(description="Compare bump weights against HF weights")
    parser.add_argument("--checkpoint", required=True, help="HF model directory (local)")
    parser.add_argument("--bump", required=True, help="Bump weights file")
    parser.add_argument("--config", help="Optional config JSON (overrides model config)")
    parser.add_argument("--context", type=int, help="Override context length")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to compare")
    args = parser.parse_args()

    state_dict, hf_config = get_state_dict(args.checkpoint, torch_dtype=None)
    cfg = load_config(args.config) if args.config else hf_config.to_dict()

    num_layers = pick(cfg, ["num_hidden_layers", "num_layers"])
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    intermediate = pick(cfg, ["intermediate_size"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    num_kv_heads = pick(cfg, ["num_key_value_heads", "num_kv_heads"], num_heads)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)
    if args.context is not None:
        context_len = int(args.context)

    if not all([num_layers, embed_dim, intermediate, num_heads, vocab_size, context_len]):
        raise SystemExit("Config missing required fields")

    if args.layer < 0 or args.layer >= num_layers:
        raise SystemExit(f"Layer index {args.layer} out of range 0..{num_layers - 1}")

    head_dim = embed_dim // num_heads
    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)

    with open(args.bump, "rb") as f:
        skip_bump_header(f)

        read_floats(f, vocab_size * aligned_embed_dim)  # token emb
        read_floats(f, context_len * aligned_embed_dim)  # pos emb

        for layer in range(num_layers):
            ln1 = read_floats(f, aligned_embed_dim)
            ln2 = read_floats(f, aligned_embed_dim)
            wq = read_floats(f, num_heads * aligned_head_dim * aligned_embed_dim)
            bq = read_floats(f, num_heads * aligned_head_dim)
            wk = read_floats(f, num_kv_heads * aligned_head_dim * aligned_embed_dim)
            bk = read_floats(f, num_kv_heads * aligned_head_dim)
            wv = read_floats(f, num_kv_heads * aligned_head_dim * aligned_embed_dim)
            bv = read_floats(f, num_kv_heads * aligned_head_dim)
            wo = read_floats(f, num_heads * aligned_embed_dim * aligned_head_dim)
            bo = read_floats(f, aligned_embed_dim)
            w1 = read_floats(f, 2 * aligned_intermediate * aligned_embed_dim)
            b1 = read_floats(f, 2 * aligned_intermediate)
            w2 = read_floats(f, aligned_embed_dim * aligned_intermediate)
            b2 = read_floats(f, aligned_embed_dim)

            if layer != args.layer:
                continue

            prefix = f"model.layers.{layer}"
            ln1_ref = get_tensor(state_dict, f"{prefix}.input_layernorm.weight").detach().cpu().numpy()
            ln2_ref = get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight").detach().cpu().numpy()
            ln1_pad = np.zeros(aligned_embed_dim, dtype=np.float32)
            ln2_pad = np.zeros(aligned_embed_dim, dtype=np.float32)
            ln1_pad[:embed_dim] = ln1_ref.astype(np.float32)
            ln2_pad[:embed_dim] = ln2_ref.astype(np.float32)

            wq_ref = get_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight").detach().cpu().numpy()
            wk_ref = get_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight").detach().cpu().numpy()
            wv_ref = get_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight").detach().cpu().numpy()
            bq_ref = get_optional(state_dict, f"{prefix}.self_attn.q_proj.bias")
            bk_ref = get_optional(state_dict, f"{prefix}.self_attn.k_proj.bias")
            bv_ref = get_optional(state_dict, f"{prefix}.self_attn.v_proj.bias")

            wo_ref = get_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight").detach().cpu().numpy()
            bo_ref = get_optional(state_dict, f"{prefix}.self_attn.o_proj.bias")

            gate = get_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight").detach().cpu().numpy()
            up = get_tensor(state_dict, f"{prefix}.mlp.up_proj.weight").detach().cpu().numpy()
            down = get_tensor(state_dict, f"{prefix}.mlp.down_proj.weight").detach().cpu().numpy()

            b1_gate = get_optional(state_dict, f"{prefix}.mlp.gate_proj.bias")
            b1_up = get_optional(state_dict, f"{prefix}.mlp.up_proj.bias")
            b2_ref = get_optional(state_dict, f"{prefix}.mlp.down_proj.bias")

            max_diff("ln1_gamma", ln1, ln1_pad)
            max_diff("ln2_gamma", ln2, ln2_pad)
            max_diff("wq", wq, pack_qkv_weight(wq_ref, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim).ravel())
            max_diff("bq", bq, pack_qkv_bias(None if bq_ref is None else bq_ref.detach().cpu().numpy(),
                                            num_heads, head_dim, aligned_head_dim))
            max_diff("wk", wk, pack_qkv_weight(wk_ref, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim).ravel())
            max_diff("bk", bk, pack_qkv_bias(None if bk_ref is None else bk_ref.detach().cpu().numpy(),
                                            num_kv_heads, head_dim, aligned_head_dim))
            max_diff("wv", wv, pack_qkv_weight(wv_ref, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim).ravel())
            max_diff("bv", bv, pack_qkv_bias(None if bv_ref is None else bv_ref.detach().cpu().numpy(),
                                            num_kv_heads, head_dim, aligned_head_dim))
            max_diff("wo", wo, pack_wo(wo_ref, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            bo_pad = np.zeros(aligned_embed_dim, dtype=np.float32)
            if bo_ref is not None:
                bo_pad[:embed_dim] = bo_ref.detach().cpu().numpy().astype(np.float32)
            max_diff("bo", bo, bo_pad)

            w1_ref = pack_w1(gate, up, intermediate, aligned_intermediate, embed_dim, aligned_embed_dim).ravel()
            max_diff("w1", w1, w1_ref)

            b1_pad = np.zeros(2 * aligned_intermediate, dtype=np.float32)
            if b1_gate is not None:
                g = b1_gate.detach().cpu().numpy().astype(np.float32)
                b1_pad[:intermediate] = g[:intermediate]
            if b1_up is not None:
                u = b1_up.detach().cpu().numpy().astype(np.float32)
                b1_pad[aligned_intermediate:aligned_intermediate + intermediate] = u[:intermediate]
            max_diff("b1", b1, b1_pad)

            w2_ref = pack_w2(down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim).ravel()
            max_diff("w2", w2, w2_ref)

            b2_pad = np.zeros(aligned_embed_dim, dtype=np.float32)
            if b2_ref is not None:
                b2_pad[:embed_dim] = b2_ref.detach().cpu().numpy().astype(np.float32)
            max_diff("b2", b2, b2_pad)

            return

    raise SystemExit("Reached EOF without matching layer")


if __name__ == "__main__":
    main()
