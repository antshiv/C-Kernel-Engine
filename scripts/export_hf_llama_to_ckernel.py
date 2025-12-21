#!/usr/bin/env python3
"""
Export a HuggingFace LLaMA/SmolLM-style checkpoint into the C-Kernel-Engine
flat weight layout (bump-order). This mirrors the GPT-2 export flow used in
C-Transformer but packs weights for RMSNorm + RoPE + GQA + SwiGLU.
"""

import argparse
import os
import struct
import hashlib

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

ALIGN_BYTES = 64
ELEM_BYTES = 4


def align_up_elems(elems, elem_bytes=ELEM_BYTES, align_bytes=ALIGN_BYTES):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned // elem_bytes


def write_zero_block(f, count):
    f.write(np.zeros(count, dtype=np.float32).tobytes())
    return count


def write_vector(f, vec, aligned_dim):
    buf = np.zeros(aligned_dim, dtype=np.float32)
    flat = vec.detach().cpu().numpy().astype(np.float32).reshape(-1)
    buf[: flat.size] = flat
    f.write(buf.tobytes())
    return buf.size


def write_matrix_padded(f, mat, out_dim, in_dim, aligned_in, aligned_out=None):
    if aligned_out is None:
        aligned_out = out_dim
    buf = np.zeros((aligned_out, aligned_in), dtype=np.float32)
    mat_np = mat.detach().cpu().numpy().astype(np.float32)
    buf[:out_dim, :in_dim] = mat_np[:out_dim, :in_dim]
    f.write(buf.ravel().tobytes())
    return buf.size


def pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    w_np = weight.detach().cpu().numpy().astype(np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :embed_dim] = w_np[src_base:src_base + head_dim, :]
    return buf


def write_qkv_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    buf = pack_qkv_weight(weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim)
    f.write(buf.ravel().tobytes())
    return buf.size


def write_wo_packed(f, weight, num_heads, head_dim, aligned_head_dim, embed_dim, aligned_embed_dim):
    w_np = weight.detach().cpu().numpy().astype(np.float32)
    total = 0
    for h in range(num_heads):
        block = np.zeros((aligned_embed_dim, aligned_head_dim), dtype=np.float32)
        col_base = h * head_dim
        block[:embed_dim, :head_dim] = w_np[:, col_base:col_base + head_dim]
        f.write(block.ravel().tobytes())
        total += block.size
    return total


def export_llama_checkpoint(checkpoint_dir, output_file):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    print(f"Loading config from {checkpoint_dir}...")
    cfg = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)

    print(f"Loading model weights from {checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=torch.float32, local_files_only=True
    )
    state = model.state_dict()

    # Resolve key prefix
    prefix = "model."
    if f"{prefix}embed_tokens.weight" not in state:
        prefix = ""
    if f"{prefix}embed_tokens.weight" not in state:
        raise KeyError("Could not find embed_tokens.weight in checkpoint")

    num_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    embed_dim = getattr(cfg, "hidden_size", None) or cfg.n_embd
    intermediate = getattr(cfg, "intermediate_size", None) or cfg.n_inner
    num_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    vocab_size = cfg.vocab_size
    context_len = getattr(cfg, "max_position_embeddings", None) or cfg.n_positions
    head_dim = embed_dim // num_heads

    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)

    print("Model config:")
    print(f"  layers={num_layers} embed={embed_dim} inter={intermediate}")
    print(f"  heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim}")
    print(f"  vocab={vocab_size} ctx={context_len}")
    print(f"  aligned: embed={aligned_embed_dim} head={aligned_head_dim} inter={aligned_intermediate}")

    with open(output_file, "w+b") as f:
        header_size = 128
        f.write(b"\x00" * header_size)
        total_elems = 0

        def write_and_count(count):
            nonlocal total_elems
            total_elems += count

        # 1) Token embeddings [V x aligned_embed_dim]
        write_and_count(
            write_matrix_padded(
                f,
                state[f"{prefix}embed_tokens.weight"],
                vocab_size,
                embed_dim,
                aligned_embed_dim,
            )
        )

        # 2) Pos embeddings [T x aligned_embed_dim] (RoPE models use zeros)
        write_and_count(write_zero_block(f, context_len * aligned_embed_dim))

        # 3) Per-layer weights
        for layer in range(num_layers):
            lp = f"{prefix}layers.{layer}"
            print(f"  Layer {layer}")

            # RMSNorm weights
            write_and_count(write_vector(f, state[f"{lp}.input_layernorm.weight"], aligned_embed_dim))
            write_and_count(write_vector(f, state[f"{lp}.post_attention_layernorm.weight"], aligned_embed_dim))

            # QKV weights
            write_and_count(
                write_qkv_packed(
                    f,
                    state[f"{lp}.self_attn.q_proj.weight"],
                    num_heads,
                    head_dim,
                    aligned_head_dim,
                    embed_dim,
                    aligned_embed_dim,
                )
            )
            write_and_count(write_zero_block(f, num_heads * aligned_head_dim))  # bq

            write_and_count(
                write_qkv_packed(
                    f,
                    state[f"{lp}.self_attn.k_proj.weight"],
                    num_kv_heads,
                    head_dim,
                    aligned_head_dim,
                    embed_dim,
                    aligned_embed_dim,
                )
            )
            write_and_count(write_zero_block(f, num_kv_heads * aligned_head_dim))  # bk

            write_and_count(
                write_qkv_packed(
                    f,
                    state[f"{lp}.self_attn.v_proj.weight"],
                    num_kv_heads,
                    head_dim,
                    aligned_head_dim,
                    embed_dim,
                    aligned_embed_dim,
                )
            )
            write_and_count(write_zero_block(f, num_kv_heads * aligned_head_dim))  # bv

            # Attention output projection (packed by head)
            write_and_count(
                write_wo_packed(
                    f,
                    state[f"{lp}.self_attn.o_proj.weight"],
                    num_heads,
                    head_dim,
                    aligned_head_dim,
                    embed_dim,
                    aligned_embed_dim,
                )
            )
            write_and_count(write_zero_block(f, aligned_embed_dim))  # bo

            # SwiGLU (gate + up) packed into w1
            gate = state[f"{lp}.mlp.gate_proj.weight"]
            up = state[f"{lp}.mlp.up_proj.weight"]
            w1 = np.zeros((2 * aligned_intermediate, aligned_embed_dim), dtype=np.float32)
            w1[:intermediate, :embed_dim] = gate.detach().cpu().numpy().astype(np.float32)
            w1[aligned_intermediate : aligned_intermediate + intermediate, :embed_dim] = (
                up.detach().cpu().numpy().astype(np.float32)
            )
            f.write(w1.ravel().tobytes())
            write_and_count(w1.size)
            write_and_count(write_zero_block(f, 2 * aligned_intermediate))  # b1

            # Down proj
            write_and_count(
                write_matrix_padded(
                    f,
                    state[f"{lp}.mlp.down_proj.weight"],
                    embed_dim,
                    intermediate,
                    aligned_intermediate,
                    aligned_embed_dim,
                )
            )
            write_and_count(write_zero_block(f, aligned_embed_dim))  # b2

        # 4) Final RMSNorm
        write_and_count(write_vector(f, state[f"{prefix}norm.weight"], aligned_embed_dim))
        write_and_count(write_zero_block(f, aligned_embed_dim))  # final_ln_bias

        # Header
        f.flush()
        f.seek(header_size)
        payload = f.read()
        checksum = hashlib.sha256(payload).digest()
        f.seek(0)
        f.write(b"BUMPWGT2")
        f.write(struct.pack("I", 2))
        f.write(struct.pack("I", 1))  # model_type: 1=LLAMA
        f.write(struct.pack("I", num_layers))
        f.write(struct.pack("I", vocab_size))
        f.write(struct.pack("I", embed_dim))
        f.write(struct.pack("I", context_len))
        f.write(struct.pack("I", num_heads))
        f.write(struct.pack("I", head_dim))
        f.write(struct.pack("Q", aligned_embed_dim))
        f.write(struct.pack("Q", aligned_head_dim))
        f.write(struct.pack("Q", align_up_elems(context_len)))
        f.write(checksum)
        f.write(b"\x00" * 32)

    mb = (header_size + total_elems * ELEM_BYTES) / (1024 * 1024)
    print(f"✅ Wrote {output_file} ({mb:.2f} MB)")
    print(f"   checksum: {checksum.hex()[:16]}...")


def main():
    parser = argparse.ArgumentParser(description="Export HF LLaMA/SmolLM checkpoint -> C-Kernel-Engine weights")
    parser.add_argument("--checkpoint", required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--output", required=True, help="Output weights file")
    args = parser.parse_args()
    export_llama_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
