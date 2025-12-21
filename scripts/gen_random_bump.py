#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct

import numpy as np

ALIGN_BYTES = 64
ELEM_BYTES = 4
HEADER_SIZE = 128


def align_up_elems(elems, elem_bytes=ELEM_BYTES, align_bytes=ALIGN_BYTES):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned // elem_bytes


def load_cfg(path):
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


def write_zero_block(f, count):
    f.write(np.zeros(count, dtype=np.float32).tobytes())
    return count


def write_vector(f, vec, aligned_dim):
    buf = np.zeros(aligned_dim, dtype=np.float32)
    flat = vec.astype(np.float32).reshape(-1)
    buf[: flat.size] = flat
    f.write(buf.tobytes())
    return buf.size


def write_matrix_padded(f, mat, out_dim, in_dim, aligned_in, aligned_out=None):
    if aligned_out is None:
        aligned_out = out_dim
    buf = np.zeros((aligned_out, aligned_in), dtype=np.float32)
    buf[:out_dim, :in_dim] = mat[:out_dim, :in_dim].astype(np.float32)
    f.write(buf.ravel().tobytes())
    return buf.size


def pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :embed_dim] = weight[src_base:src_base + head_dim, :embed_dim]
    return buf


def write_qkv_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    buf = pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
    f.write(buf.ravel().tobytes())
    return buf.size


def write_wo_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    total = 0
    for h in range(num_heads):
        block = np.zeros((aligned_embed_dim, aligned_head_dim), dtype=np.float32)
        col_base = h * head_dim
        block[:embed_dim, :head_dim] = weight[:embed_dim, col_base:col_base + head_dim]
        f.write(block.ravel().tobytes())
        total += block.size
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate random bump-format weights")
    parser.add_argument("--config", required=True, help="Model config JSON")
    parser.add_argument("--output", required=True, help="Output weights file")
    parser.add_argument("--npz", help="Optional NPZ dump of logical (unpadded) weights")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument("--std", type=float, default=0.02, help="Stddev for normal init")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    num_layers = pick(cfg, ["num_hidden_layers", "num_layers"])
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    intermediate = pick(cfg, ["intermediate_size"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    num_kv_heads = pick(cfg, ["num_key_value_heads", "num_kv_heads"], num_heads)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)

    if not all([num_layers, embed_dim, intermediate, num_heads, vocab_size, context_len]):
        raise ValueError("Config missing required fields for weight generation")

    head_dim = embed_dim // num_heads
    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)

    rng = np.random.default_rng(args.seed)
    scale = args.std

    npz_data = {}
    npz_data["token_emb"] = None
    npz_data["pos_emb"] = None
    npz_data["final_ln_weight"] = None
    npz_data["final_ln_bias"] = None

    with open(args.output, "w+b") as f:
        f.write(b"\x00" * HEADER_SIZE)
        total_elems = 0

        def write_and_count(count):
            nonlocal total_elems
            total_elems += count

        tok = rng.normal(0.0, scale, size=(vocab_size, embed_dim)).astype(np.float32)
        write_and_count(write_matrix_padded(f, tok, vocab_size, embed_dim, aligned_embed_dim))
        npz_data["token_emb"] = tok.copy()

        write_and_count(write_zero_block(f, context_len * aligned_embed_dim))
        npz_data["pos_emb"] = np.zeros((context_len, embed_dim), dtype=np.float32)

        for layer_idx in range(num_layers):
            ln1 = rng.normal(1.0, scale, size=(embed_dim,)).astype(np.float32)
            ln2 = rng.normal(1.0, scale, size=(embed_dim,)).astype(np.float32)
            write_and_count(write_vector(f, ln1, aligned_embed_dim))
            write_and_count(write_vector(f, ln2, aligned_embed_dim))
            npz_data[f"layer.{layer_idx}.ln1_gamma"] = ln1.copy()
            npz_data[f"layer.{layer_idx}.ln2_gamma"] = ln2.copy()

            wq = rng.normal(0.0, scale, size=(num_heads * head_dim, embed_dim)).astype(np.float32)
            wk = rng.normal(0.0, scale, size=(num_kv_heads * head_dim, embed_dim)).astype(np.float32)
            wv = rng.normal(0.0, scale, size=(num_kv_heads * head_dim, embed_dim)).astype(np.float32)
            write_and_count(write_qkv_packed(f, wq, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_zero_block(f, num_heads * aligned_head_dim))
            write_and_count(write_qkv_packed(f, wk, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_zero_block(f, num_kv_heads * aligned_head_dim))
            write_and_count(write_qkv_packed(f, wv, num_kv_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_zero_block(f, num_kv_heads * aligned_head_dim))
            npz_data[f"layer.{layer_idx}.wq"] = wq.copy()
            npz_data[f"layer.{layer_idx}.bq"] = np.zeros((num_heads * head_dim,), dtype=np.float32)
            npz_data[f"layer.{layer_idx}.wk"] = wk.copy()
            npz_data[f"layer.{layer_idx}.bk"] = np.zeros((num_kv_heads * head_dim,), dtype=np.float32)
            npz_data[f"layer.{layer_idx}.wv"] = wv.copy()
            npz_data[f"layer.{layer_idx}.bv"] = np.zeros((num_kv_heads * head_dim,), dtype=np.float32)

            wo = rng.normal(0.0, scale, size=(embed_dim, num_heads * head_dim)).astype(np.float32)
            write_and_count(write_wo_packed(f, wo, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim))
            write_and_count(write_zero_block(f, aligned_embed_dim))
            npz_data[f"layer.{layer_idx}.wo"] = wo.copy()
            npz_data[f"layer.{layer_idx}.bo"] = np.zeros((embed_dim,), dtype=np.float32)

            gate = rng.normal(0.0, scale, size=(intermediate, embed_dim)).astype(np.float32)
            up = rng.normal(0.0, scale, size=(intermediate, embed_dim)).astype(np.float32)
            w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
            w1[:intermediate, :embed_dim] = gate
            w1[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim] = up
            f.write(w1.ravel().tobytes())
            write_and_count(w1.size)
            write_and_count(write_zero_block(f, 2 * aligned_intermediate))
            npz_data[f"layer.{layer_idx}.w1"] = np.concatenate([gate, up], axis=0)
            npz_data[f"layer.{layer_idx}.b1"] = np.zeros((2 * intermediate,), dtype=np.float32)

            w2 = rng.normal(0.0, scale, size=(embed_dim, intermediate)).astype(np.float32)
            write_and_count(write_matrix_padded(f, w2, embed_dim, intermediate, aligned_intermediate, aligned_embed_dim))
            write_and_count(write_zero_block(f, aligned_embed_dim))
            npz_data[f"layer.{layer_idx}.w2"] = w2.copy()
            npz_data[f"layer.{layer_idx}.b2"] = np.zeros((embed_dim,), dtype=np.float32)

        ln_f = rng.normal(1.0, scale, size=(embed_dim,)).astype(np.float32)
        write_and_count(write_vector(f, ln_f, aligned_embed_dim))
        write_and_count(write_zero_block(f, aligned_embed_dim))
        npz_data["final_ln_weight"] = ln_f.copy()
        npz_data["final_ln_bias"] = np.zeros((embed_dim,), dtype=np.float32)

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
        f.write(struct.pack("Q", int(align_up_elems(context_len))))
        f.write(checksum)
        f.write(b"\x00" * 32)

    if args.npz:
        npz_data["config_json"] = json.dumps(cfg).encode("utf-8")
        np.savez(args.npz, **npz_data)
        print(f"Wrote {args.npz} (logical weights)")

    mb = (HEADER_SIZE + total_elems * ELEM_BYTES) / (1024 * 1024)
    print(f"Wrote {args.output} ({mb:.2f} MB, seed={args.seed})")


if __name__ == "__main__":
    main()
