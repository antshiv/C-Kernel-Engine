#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


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


def override_context(cfg, context_len):
    if context_len is None:
        return cfg
    cfg = dict(cfg)
    cfg["max_position_embeddings"] = int(context_len)
    cfg["context_window"] = int(context_len)
    return cfg


def write_cfg(path, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


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


def openmp_flag(cc: str) -> str:
    cc_lower = cc.lower()
    if "icx" in cc_lower or "icc" in cc_lower:
        return "-qopenmp"
    return "-fopenmp"


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def select_text_key(ds, user_key=None):
    if user_key:
        return user_key
    for key, feature in ds.features.items():
        if getattr(feature, "dtype", None) == "string":
            return key
    raise ValueError("could not infer text column; pass --text-key")


def build_tokens(tokenizer, ds, text_key, context_len, max_samples):
    tokens = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        text = row.get(text_key)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(ids)
        if len(tokens) >= context_len:
            break
    if not tokens:
        raise ValueError("no tokens extracted from dataset")
    if len(tokens) < context_len:
        pad_id = tokenizer.eos_token_id or 0
        tokens.extend([pad_id] * (context_len - len(tokens)))
    tokens = tokens[:context_len]
    targets = tokens[1:] + [tokenizer.eos_token_id or 0]
    return (
        np.array(tokens, dtype=np.int32),
        np.array(targets, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="SmolLM tiny training demo on a small dataset slice")
    parser.add_argument("--config", default=None, help="Model config JSON (default: uses model-dir/config.json)")
    parser.add_argument("--model-dir", required=True, help="HF model directory")
    parser.add_argument("--download-model", action="store_true", help="Download model if missing")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="HF repo id for download")
    parser.add_argument("--context", type=int, default=2, help="Context length for demo run")
    parser.add_argument("--dataset", default="roneneldan/TinyStories", help="HF dataset name")
    parser.add_argument("--dataset-config", help="HF dataset config name (optional)")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text-key", help="Text column name")
    parser.add_argument("--max-samples", type=int, default=4, help="Max samples to scan for tokens")
    parser.add_argument("--steps", type=int, default=1, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="SGD learning rate")
    parser.add_argument("--build-dir", default="build", help="Build output directory")
    parser.add_argument("--strict", action="store_true", help="Enable strict parity in C runtime")
    args = parser.parse_args()

    if args.download_model or not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
        run([
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
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)
    if context_len <= 0:
        raise SystemExit("invalid context length; pass --context")

    os.makedirs(args.build_dir, exist_ok=True)
    runtime_cfg = os.path.join(args.build_dir, "runtime.config.json")
    write_cfg(runtime_cfg, cfg)

    weights_bin = os.path.join(args.build_dir, "smollm_weights.bin")
    tokens_bin = os.path.join(args.build_dir, "smollm_tokens.bin")
    targets_bin = os.path.join(args.build_dir, "smollm_targets.bin")
    gen_c = os.path.join(args.build_dir, "smollm_generated.c")
    kernel_manifest = gen_c + ".kernels"
    model_bin = os.path.join(args.build_dir, "smollm_model")

    ds_kwargs = {}
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    ds = load_dataset(args.dataset, **ds_kwargs, split=args.split)
    text_key = select_text_key(ds, args.text_key)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokens, targets = build_tokens(tokenizer, ds, text_key, context_len, args.max_samples)
    tokens.tofile(tokens_bin)
    targets.tofile(targets_bin)

    run([
        sys.executable,
        "scripts/convert_hf_to_bump.py",
        "--checkpoint", args.model_dir,
        "--output", weights_bin,
        "--config", runtime_cfg,
        "--context", str(context_len),
    ])

    run(["make", "build/ck_ir_demo"])
    run(["./build/ck_ir_demo", runtime_cfg, "--emit", gen_c])

    with open(kernel_manifest, "r", encoding="utf-8") as f:
        kernels = f.read().split()

    cc = os.environ.get("CC", "gcc")
    cflags = ["-O3", "-fPIC", openmp_flag(cc), "-Wall"] + detect_avx_flags() + ["-Iinclude"]
    run([cc, *cflags, gen_c, *kernels, "-o", model_bin, "-lm"])

    cmd = [
        model_bin,
        "--model-weights", weights_bin,
        "--tokens", tokens_bin,
        "--targets", targets_bin,
        "--backward",
        "--lr", str(args.lr),
        "--steps", str(args.steps),
        "--log-steps",
    ]
    if args.strict:
        cmd.append("--strict")
    run(cmd)


if __name__ == "__main__":
    main()
