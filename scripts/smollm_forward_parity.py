#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

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


class _DummyDynamo:
    @staticmethod
    def is_compiling() -> bool:
        return False


torch.__dict__["_dynamo"] = _DummyDynamo()


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


def prepare_tokens(tokenizer, text, context_len):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    if len(ids) < context_len:
        pad_id = tokenizer.eos_token_id or 0
        ids.extend([pad_id] * (context_len - len(ids)))
    ids = ids[:context_len]
    return np.array(ids, dtype=np.int32)


def topk_ids(logits, k):
    idx = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
    rows = np.arange(logits.shape[0])[:, None]
    vals = logits[rows, idx]
    order = np.argsort(-vals, axis=-1)
    sorted_idx = idx[rows, order]
    sorted_vals = vals[rows, order]
    return sorted_idx, sorted_vals


def format_topk(tokenizer, ids, vals):
    pieces = []
    for token_id, val in zip(ids, vals):
        tok = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        tok = tok.replace("\n", "\\n")
        pieces.append(f"{int(token_id)}:{tok}:{val:.4f}")
    return ", ".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="SmolLM forward parity vs PyTorch")
    parser.add_argument("--config", default="smolLM-135.json", help="Model config JSON")
    parser.add_argument("--model-dir", required=True, help="HF model directory")
    parser.add_argument("--download-model", action="store_true", help="Download model if missing")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="HF repo id")
    parser.add_argument("--context", type=int, default=5, help="Context length")
    parser.add_argument("--text", default="Once upon a time", help="Prompt text")
    parser.add_argument("--topk", type=int, default=5, help="Top-k to display")
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

    cfg = override_context(load_cfg(args.config), args.context)
    os.makedirs(args.build_dir, exist_ok=True)
    runtime_cfg = os.path.join(args.build_dir, "runtime.config.json")
    write_cfg(runtime_cfg, cfg)

    weights_bin = os.path.join(args.build_dir, "smollm_weights.bin")
    tokens_bin = os.path.join(args.build_dir, "smollm_tokens.bin")
    logits_bin = os.path.join(args.build_dir, "smollm_logits.bin")
    gen_c = os.path.join(args.build_dir, "smollm_generated.c")
    kernel_manifest = gen_c + ".kernels"
    model_bin = os.path.join(args.build_dir, "smollm_model")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokens = prepare_tokens(tokenizer, args.text, args.context)
    tokens.tofile(tokens_bin)

    run([
        sys.executable,
        "scripts/convert_hf_to_bump.py",
        "--checkpoint", args.model_dir,
        "--output", weights_bin,
        "--config", runtime_cfg,
        "--context", str(args.context),
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
        "--out-logits", logits_bin,
    ]
    if args.strict:
        cmd.append("--strict")
    run(cmd)

    vocab_size = cfg.get("vocab_size")
    c_logits = np.fromfile(logits_bin, dtype=np.float32)
    c_logits = c_logits.reshape(args.context, vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        t_logits = model(input_ids).logits[0].cpu().numpy()

    max_abs = np.max(np.abs(c_logits - t_logits))
    mean_abs = np.mean(np.abs(c_logits - t_logits))
    print(f"Logits diff: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")

    c_topk, c_vals = topk_ids(c_logits, args.topk)
    t_topk, t_vals = topk_ids(t_logits, args.topk)

    for i in range(args.context):
        c_top1 = int(c_topk[i, 0])
        t_top1 = int(t_topk[i, 0])
        match = "OK" if c_top1 == t_top1 else "DIFF"
        print(f"pos {i} top1: C={c_top1} Torch={t_top1} [{match}]")
        print(f"  C   top{args.topk}: {format_topk(tokenizer, c_topk[i], c_vals[i])}")
        print(f"  Pyt top{args.topk}: {format_topk(tokenizer, t_topk[i], t_vals[i])}")


if __name__ == "__main__":
    main()
