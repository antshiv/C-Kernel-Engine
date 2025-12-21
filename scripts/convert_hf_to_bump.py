#!/usr/bin/env python3
import argparse
import hashlib
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


def write_zero_block(f, count):
    f.write(np.zeros(count, dtype=np.float32).tobytes())
    return count


def write_vector(f, vec, aligned_dim):
    buf = np.zeros(aligned_dim, dtype=np.float32)
    if vec is not None:
        flat = vec.astype(np.float32).reshape(-1)
        buf[:flat.size] = flat
    f.write(buf.tobytes())
    return buf.size


def write_matrix_padded(f, mat, out_dim, in_dim, aligned_in, aligned_out=None):
    if aligned_out is None:
        aligned_out = out_dim
    buf = np.zeros((aligned_out, aligned_in), dtype=np.float32)
    if mat is not None:
        buf[:out_dim, :in_dim] = mat[:out_dim, :in_dim].astype(np.float32)
    f.write(buf.ravel().tobytes())
    return buf.size


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


def write_qkv_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    buf = pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
    f.write(buf.ravel().tobytes())
    return buf.size


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


def write_qkv_bias_packed(f, bias, num_heads, head_dim, aligned_head_dim):
    buf = pack_qkv_bias(bias, num_heads, head_dim, aligned_head_dim)
    f.write(buf.tobytes())
    return buf.size


def write_wo_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    total = 0
    w = weight.astype(np.float32) if weight is not None else None
    for h in range(num_heads):
        block = np.zeros((aligned_embed_dim, aligned_head_dim), dtype=np.float32)
        if w is not None:
            col_base = h * head_dim
            block[:embed_dim, :head_dim] = w[:embed_dim, col_base:col_base + head_dim]
        f.write(block.ravel().tobytes())
        total += block.size
    return total


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
        raise SystemExit("torch is required to convert HF weights") from exc


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
        return AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required to convert HF weights") from exc


def get_state_dict(checkpoint, torch_dtype):
    torch = require_torch()
    AutoModelForCausalLM = require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
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


def main():
    parser = argparse.ArgumentParser(description="Convert HF Llama-style weights to bump format")
    parser.add_argument("--checkpoint", required=True, help="HF model directory (local)")
    parser.add_argument("--config", help="Optional config JSON (overrides model config)")
    parser.add_argument("--output", required=True, help="Output bump weights file")
    parser.add_argument("--context", type=int, help="Override context length (for small tests)")
    parser.add_argument("--dtype", default="float32", help="Output dtype (float32 only supported)")
    args = parser.parse_args()

    if args.dtype != "float32":
        raise SystemExit("Only float32 output is supported for now")

    state_dict, hf_config = get_state_dict(args.checkpoint, torch_dtype=None)

    cfg = {}
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = hf_config.to_dict()

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
        raise SystemExit("Config missing required fields for conversion")

    head_dim = embed_dim // num_heads
    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)
    aligned_context = align_up_elems(context_len)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w+b") as f:
        f.write(b"\x00" * HEADER_SIZE)
        total_elems = 0

        def write_and_count(count):
            nonlocal total_elems
            total_elems += count

        tok = get_tensor(
            state_dict,
            "model.embed_tokens.weight",
            alt_keys=("model.tok_embeddings.weight",),
        ).detach().cpu().numpy()
        write_and_count(write_matrix_padded(f, tok, vocab_size, embed_dim, aligned_embed_dim))

        write_and_count(write_zero_block(f, context_len * aligned_embed_dim))

        for layer in range(num_layers):
            prefix = f"model.layers.{layer}"
            ln1 = get_tensor(state_dict, f"{prefix}.input_layernorm.weight").detach().cpu().numpy()
            ln2 = get_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight").detach().cpu().numpy()
            write_and_count(write_vector(f, ln1, aligned_embed_dim))
            write_and_count(write_vector(f, ln2, aligned_embed_dim))

            wq = get_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight").detach().cpu().numpy()
            wk = get_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight").detach().cpu().numpy()
            wv = get_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight").detach().cpu().numpy()

            bq = get_optional(state_dict, f"{prefix}.self_attn.q_proj.bias")
            bk = get_optional(state_dict, f"{prefix}.self_attn.k_proj.bias")
            bv = get_optional(state_dict, f"{prefix}.self_attn.v_proj.bias")

            write_and_count(write_qkv_packed(f, wq, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_qkv_bias_packed(f, None if bq is None else bq.detach().cpu().numpy(),
                                                  num_heads, head_dim, aligned_head_dim))
            write_and_count(write_qkv_packed(f, wk, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_qkv_bias_packed(f, None if bk is None else bk.detach().cpu().numpy(),
                                                  num_kv_heads, head_dim, aligned_head_dim))
            write_and_count(write_qkv_packed(f, wv, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_qkv_bias_packed(f, None if bv is None else bv.detach().cpu().numpy(),
                                                  num_kv_heads, head_dim, aligned_head_dim))

            wo = get_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight").detach().cpu().numpy()
            bo = get_optional(state_dict, f"{prefix}.self_attn.o_proj.bias")
            write_and_count(write_wo_packed(f, wo, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_vector(f, None if bo is None else bo.detach().cpu().numpy(), aligned_embed_dim))

            gate = get_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight").detach().cpu().numpy()
            up = get_tensor(state_dict, f"{prefix}.mlp.up_proj.weight").detach().cpu().numpy()
            down = get_tensor(state_dict, f"{prefix}.mlp.down_proj.weight").detach().cpu().numpy()

            w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
            w1[:intermediate, :embed_dim] = gate[:intermediate, :embed_dim]
            w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up[:intermediate, :embed_dim]
            f.write(w1.ravel().tobytes())
            write_and_count(w1.size)

            b1_gate = get_optional(state_dict, f"{prefix}.mlp.gate_proj.bias")
            b1_up = get_optional(state_dict, f"{prefix}.mlp.up_proj.bias")
            b1 = np.zeros((2 * aligned_intermediate,), dtype=np.float32)
            if b1_gate is not None:
                g = b1_gate.detach().cpu().numpy()
                b1[:intermediate] = g[:intermediate]
            if b1_up is not None:
                u = b1_up.detach().cpu().numpy()
                b1[aligned_intermediate:aligned_intermediate + intermediate] = u[:intermediate]
            f.write(b1.tobytes())
            write_and_count(b1.size)

            write_and_count(write_matrix_padded(f, down, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim))

            b2 = get_optional(state_dict, f"{prefix}.mlp.down_proj.bias")
            write_and_count(write_vector(f, None if b2 is None else b2.detach().cpu().numpy(), aligned_embed_dim))

        ln_f = get_tensor(state_dict, "model.norm.weight").detach().cpu().numpy()
        write_and_count(write_vector(f, ln_f, aligned_embed_dim))
        write_and_count(write_zero_block(f, aligned_embed_dim))

        f.flush()
        f.seek(HEADER_SIZE)
        payload = f.read()
        checksum = hashlib.sha256(payload).digest()
        f.seek(0)
        f.write(b"BUMPWGT2")
        f.write(struct.pack("I", 2))
        f.write(struct.pack("I", 1))
        f.write(struct.pack("I", int(num_layers)))
        f.write(struct.pack("I", int(vocab_size)))
        f.write(struct.pack("I", int(embed_dim)))
        f.write(struct.pack("I", int(context_len)))
        f.write(struct.pack("I", int(num_heads)))
        f.write(struct.pack("I", int(head_dim)))
        f.write(struct.pack("Q", int(aligned_embed_dim)))
        f.write(struct.pack("Q", int(aligned_head_dim)))
        f.write(struct.pack("Q", int(aligned_context)))
        f.write(checksum)
        f.write(b"\x00" * 32)

    mb = (HEADER_SIZE + total_elems * FLOAT_SIZE) / (1024 * 1024)
    print(f"Wrote {args.output} ({mb:.2f} MB, ctx={context_len}, heads={num_heads}, kv={num_kv_heads})")


if __name__ == "__main__":
    main()
