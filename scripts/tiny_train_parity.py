#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

import numpy as np


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


def align_up_elems(elems, elem_bytes=4, align_bytes=64):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned // elem_bytes


def detect_avx_flags():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":")[1].strip().split())
                    if "avx512f" in flags:
                        return ["-mavx512f"]
                    if "avx2" in flags:
                        return ["-mavx2"]
                    if "avx" in flags:
                        return ["-mavx"]
                    break
    except OSError:
        pass
    return []


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def read_bump_weights(path, cfg):
    num_layers = pick(cfg, ["num_hidden_layers", "num_layers"])
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    intermediate = pick(cfg, ["intermediate_size"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    num_kv_heads = pick(cfg, ["num_key_value_heads", "num_kv_heads"], num_heads)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)

    head_dim = embed_dim // num_heads
    aligned_embed_dim = align_up_elems(embed_dim)
    aligned_head_dim = align_up_elems(head_dim)
    aligned_intermediate = align_up_elems(intermediate)

    def read_floats(f, count):
        data = np.fromfile(f, dtype=np.float32, count=count)
        if data.size != count:
            raise ValueError("unexpected EOF while reading weights")
        return data

    weights = {}
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic == b"BUMPWGT2":
            f.seek(128)
        else:
            f.seek(0)

        tok = read_floats(f, vocab_size * aligned_embed_dim).reshape(vocab_size, aligned_embed_dim)
        pos = read_floats(f, context_len * aligned_embed_dim).reshape(context_len, aligned_embed_dim)
        weights["token_emb"] = tok[:, :embed_dim].copy()
        weights["pos_emb"] = pos[:, :embed_dim].copy()

        for layer in range(num_layers):
            ln1 = read_floats(f, aligned_embed_dim)[:embed_dim].copy()
            ln2 = read_floats(f, aligned_embed_dim)[:embed_dim].copy()

            q_w = read_floats(f, num_heads * aligned_head_dim * aligned_embed_dim)
            q_w = q_w.reshape(num_heads * aligned_head_dim, aligned_embed_dim)
            q_b = read_floats(f, num_heads * aligned_head_dim)

            k_w = read_floats(f, num_kv_heads * aligned_head_dim * aligned_embed_dim)
            k_w = k_w.reshape(num_kv_heads * aligned_head_dim, aligned_embed_dim)
            k_b = read_floats(f, num_kv_heads * aligned_head_dim)

            v_w = read_floats(f, num_kv_heads * aligned_head_dim * aligned_embed_dim)
            v_w = v_w.reshape(num_kv_heads * aligned_head_dim, aligned_embed_dim)
            v_b = read_floats(f, num_kv_heads * aligned_head_dim)

            wo_blocks = read_floats(f, num_heads * aligned_embed_dim * aligned_head_dim)
            wo_blocks = wo_blocks.reshape(num_heads, aligned_embed_dim, aligned_head_dim)
            bo = read_floats(f, aligned_embed_dim)

            w1_full = read_floats(f, 2 * aligned_intermediate * aligned_embed_dim)
            w1_full = w1_full.reshape(2 * aligned_intermediate, aligned_embed_dim)
            b1_full = read_floats(f, 2 * aligned_intermediate)

            w2_full = read_floats(f, aligned_embed_dim * aligned_intermediate)
            w2_full = w2_full.reshape(aligned_embed_dim, aligned_intermediate)
            b2_full = read_floats(f, aligned_embed_dim)

            wq = np.zeros((num_heads * head_dim, embed_dim), dtype=np.float32)
            bq = np.zeros((num_heads * head_dim,), dtype=np.float32)
            for h in range(num_heads):
                src = q_w[h * aligned_head_dim:h * aligned_head_dim + head_dim, :embed_dim]
                wq[h * head_dim:(h + 1) * head_dim, :] = src
                bq[h * head_dim:(h + 1) * head_dim] = q_b[h * aligned_head_dim:h * aligned_head_dim + head_dim]

            wk = np.zeros((num_kv_heads * head_dim, embed_dim), dtype=np.float32)
            bk = np.zeros((num_kv_heads * head_dim,), dtype=np.float32)
            wv = np.zeros((num_kv_heads * head_dim, embed_dim), dtype=np.float32)
            bv = np.zeros((num_kv_heads * head_dim,), dtype=np.float32)
            for h in range(num_kv_heads):
                src = k_w[h * aligned_head_dim:h * aligned_head_dim + head_dim, :embed_dim]
                wk[h * head_dim:(h + 1) * head_dim, :] = src
                bk[h * head_dim:(h + 1) * head_dim] = k_b[h * aligned_head_dim:h * aligned_head_dim + head_dim]
                src = v_w[h * aligned_head_dim:h * aligned_head_dim + head_dim, :embed_dim]
                wv[h * head_dim:(h + 1) * head_dim, :] = src
                bv[h * head_dim:(h + 1) * head_dim] = v_b[h * aligned_head_dim:h * aligned_head_dim + head_dim]

            wo = np.zeros((embed_dim, num_heads * head_dim), dtype=np.float32)
            for h in range(num_heads):
                block = wo_blocks[h, :embed_dim, :head_dim]
                wo[:, h * head_dim:(h + 1) * head_dim] = block

            gate = w1_full[:intermediate, :embed_dim]
            up = w1_full[aligned_intermediate:aligned_intermediate + intermediate, :embed_dim]
            w1 = np.concatenate([gate, up], axis=0)

            b1 = np.concatenate([
                b1_full[:intermediate],
                b1_full[aligned_intermediate:aligned_intermediate + intermediate],
            ])

            w2 = w2_full[:embed_dim, :intermediate].copy()
            b2 = b2_full[:embed_dim].copy()

            prefix = f"layer.{layer}"
            weights[f"{prefix}.ln1_gamma"] = ln1
            weights[f"{prefix}.ln2_gamma"] = ln2
            weights[f"{prefix}.wq"] = wq
            weights[f"{prefix}.bq"] = bq
            weights[f"{prefix}.wk"] = wk
            weights[f"{prefix}.bk"] = bk
            weights[f"{prefix}.wv"] = wv
            weights[f"{prefix}.bv"] = bv
            weights[f"{prefix}.wo"] = wo
            weights[f"{prefix}.bo"] = bo[:embed_dim].copy()
            weights[f"{prefix}.w1"] = w1
            weights[f"{prefix}.b1"] = b1
            weights[f"{prefix}.w2"] = w2
            weights[f"{prefix}.b2"] = b2

        weights["final_ln_weight"] = read_floats(f, aligned_embed_dim)[:embed_dim].copy()
        weights["final_ln_bias"] = read_floats(f, aligned_embed_dim)[:embed_dim].copy()

    return weights


def rope_cache(head_dim, max_seq_len, base):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def rope_apply(x, cos, sin):
    T, D = x.shape
    half = D // 2
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    cos = cos[:T]
    sin = sin[:T]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[:, 0::2] = out_even
    out[:, 1::2] = out_odd
    return out


def rmsnorm(x, gamma, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def torch_step(weights, cfg, tokens, targets, lr, rope_cos, rope_sin):
    num_layers = pick(cfg, ["num_hidden_layers", "num_layers"])
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    intermediate = pick(cfg, ["intermediate_size"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    num_kv_heads = pick(cfg, ["num_key_value_heads", "num_kv_heads"], num_heads)
    rope_theta = pick(cfg, ["rope_theta"], 0.0)
    eps = pick(cfg, ["rms_norm_eps"], 1e-5)
    head_dim = embed_dim // num_heads

    x = weights["token_emb"][tokens]
    if rope_theta <= 0.0:
        x = x + weights["pos_emb"][: x.shape[0]]

    for layer in range(num_layers):
        prefix = f"layer.{layer}"
        ln1 = rmsnorm(x, weights[f"{prefix}.ln1_gamma"], eps)

        wq = weights[f"{prefix}.wq"]
        wk = weights[f"{prefix}.wk"]
        wv = weights[f"{prefix}.wv"]
        bq = weights[f"{prefix}.bq"]
        bk = weights[f"{prefix}.bk"]
        bv = weights[f"{prefix}.bv"]

        q = ln1 @ wq.t() + bq
        k = ln1 @ wk.t() + bk
        v = ln1 @ wv.t() + bv

        q = q.view(-1, num_heads, head_dim).transpose(0, 1).contiguous()
        k = k.view(-1, num_kv_heads, head_dim).transpose(0, 1).contiguous()
        v = v.view(-1, num_kv_heads, head_dim).transpose(0, 1).contiguous()

        if rope_theta > 0.0:
            for h in range(num_heads):
                q[h] = rope_apply(q[h], rope_cos, rope_sin)
            for h in range(num_kv_heads):
                k[h] = rope_apply(k[h], rope_cos, rope_sin)

        attn_out = torch.zeros_like(q)
        for h in range(num_heads):
            kv_head = (h * num_kv_heads) // num_heads
            scores = q[h] @ k[kv_head].t()
            scores = scores / (head_dim ** 0.5)
            mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            weights_attn = torch.softmax(scores, dim=-1)
            attn_out[h] = weights_attn @ v[kv_head]

        wo = weights[f"{prefix}.wo"]
        bo = weights[f"{prefix}.bo"]
        proj = torch.zeros_like(x)
        for h in range(num_heads):
            wo_h = wo[:, h * head_dim:(h + 1) * head_dim]
            proj = proj + attn_out[h] @ wo_h.t()
        proj = proj + bo

        res1 = x + proj
        ln2 = rmsnorm(res1, weights[f"{prefix}.ln2_gamma"], eps)

        w1 = weights[f"{prefix}.w1"]
        b1 = weights[f"{prefix}.b1"]
        w2 = weights[f"{prefix}.w2"]
        b2 = weights[f"{prefix}.b2"]

        up = ln2 @ w1.t() + b1
        gate, value = up[:, :intermediate], up[:, intermediate:]
        swiglu = torch.nn.functional.silu(gate) * value
        mlp = swiglu @ w2.t() + b2

        x = res1 + mlp

    final = rmsnorm(x, weights["final_ln_weight"], eps)
    logits = final @ weights["token_emb"].t()
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    loss.backward()

    with torch.no_grad():
        for name, param in weights.items():
            if param.grad is None:
                continue
            param -= lr * param.grad
            param.grad = None

    return loss.detach().item()


def main():
    parser = argparse.ArgumentParser(description="End-to-end tiny training parity vs PyTorch")
    parser.add_argument("--config", default="tiny.config.json", help="Model config JSON")
    parser.add_argument("--steps", type=int, default=1, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="SGD learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument("--std", type=float, default=0.02, help="Stddev for weight init")
    parser.add_argument("--build-dir", default="build", help="Build output directory")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)
    rope_theta = pick(cfg, ["rope_theta"], 0.0)
    embed_dim = pick(cfg, ["hidden_size", "embed_dim"])
    num_heads = pick(cfg, ["num_attention_heads", "num_heads"])
    head_dim = embed_dim // num_heads

    os.makedirs(args.build_dir, exist_ok=True)
    weights_bin = os.path.join(args.build_dir, "tiny_weights.bin")
    weights_npz = os.path.join(args.build_dir, "tiny_weights.npz")
    tokens_bin = os.path.join(args.build_dir, "tiny_tokens.bin")
    targets_bin = os.path.join(args.build_dir, "tiny_targets.bin")
    out_weights = os.path.join(args.build_dir, "tiny_weights_after.bin")
    out_loss = os.path.join(args.build_dir, "tiny_loss.bin")
    gen_c = os.path.join(args.build_dir, "tiny_generated.c")
    kernel_manifest = gen_c + ".kernels"
    model_bin = os.path.join(args.build_dir, "tiny_model")

    run([
        sys.executable,
        "scripts/gen_random_bump.py",
        "--config", args.config,
        "--output", weights_bin,
        "--npz", weights_npz,
        "--seed", str(args.seed),
        "--std", str(args.std),
    ])

    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, vocab_size, size=(context_len,), dtype=np.int32)
    targets = (tokens.astype(np.int64) + 1) % vocab_size
    targets = targets.astype(np.int32)
    tokens.tofile(tokens_bin)
    targets.tofile(targets_bin)

    run(["make", "build/ck_ir_demo"])
    run(["./build/ck_ir_demo", args.config, "--emit", gen_c])

    with open(kernel_manifest, "r", encoding="utf-8") as f:
        kernels = f.read().split()

    cflags = ["-O3", "-fPIC", "-fopenmp", "-Wall"] + detect_avx_flags() + ["-Iinclude"]
    run(["gcc", *cflags, gen_c, *kernels, "-o", model_bin, "-lm"])

    run([
        model_bin,
        "--model-weights", weights_bin,
        "--tokens", tokens_bin,
        "--targets", targets_bin,
        "--backward",
        "--lr", str(args.lr),
        "--steps", str(args.steps),
        "--log-steps",
        "--out-loss", out_loss,
        "--out-weights", out_weights,
    ])

    npz = np.load(weights_npz)
    torch_weights = {}
    for name in npz.files:
        if name == "config_json":
            continue
        arr = npz[name]
        torch_weights[name] = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    torch_tokens = torch.tensor(tokens, dtype=torch.long)
    torch_targets = torch.tensor(targets, dtype=torch.long)

    rope_cos = None
    rope_sin = None
    if rope_theta and rope_theta > 0.0:
        rope_cos, rope_sin = rope_cache(head_dim, context_len, rope_theta)

    losses = []
    for _ in range(args.steps):
        loss = torch_step(torch_weights, cfg, torch_tokens, torch_targets, args.lr, rope_cos, rope_sin)
        losses.append(loss)

    c_loss = float(np.fromfile(out_loss, dtype=np.float32, count=1)[0])
    print(f"C loss: {c_loss:.6f} | Torch loss: {losses[-1]:.6f}")

    c_weights = read_bump_weights(out_weights, cfg)
    max_diff = 0.0
    for name, torch_val in torch_weights.items():
        if name == "config_json":
            continue
        if name not in c_weights:
            continue
        c_val = c_weights[name]
        t_val = torch_val.detach().cpu().numpy()
        diff = np.max(np.abs(c_val - t_val))
        max_diff = max(max_diff, diff)
        print(f"{name:24s} max_diff={diff:.3e}")

    print(f"Max weight diff: {max_diff:.3e}")


if __name__ == "__main__":
    main()
