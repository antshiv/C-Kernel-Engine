#!/usr/bin/env python3
import argparse
import json

import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description="Generate random token IDs")
    parser.add_argument("--config", required=True, help="Model config JSON")
    parser.add_argument("--output", required=True, help="Output int32 file")
    parser.add_argument("--targets", help="Optional output file for target tokens")
    parser.add_argument("--target-shift", type=int, default=1, help="Shift applied to tokens to make targets")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    vocab_size = pick(cfg, ["vocab_size"])
    context_len = pick(cfg, ["max_position_embeddings", "context_window", "ctx"], 0)

    if not vocab_size or not context_len:
        raise ValueError("Config missing vocab_size or context length")

    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, vocab_size, size=(context_len,), dtype=np.int32)

    with open(args.output, "wb") as f:
        f.write(tokens.tobytes())

    print(f"Wrote {args.output} ({context_len} tokens, vocab={vocab_size}, seed={args.seed})")

    if args.targets:
        shift = int(args.target_shift)
        targets = (tokens.astype(np.int64) + shift) % vocab_size
        targets = targets.astype(np.int32)
        with open(args.targets, "wb") as f:
            f.write(targets.tobytes())
        print(f"Wrote {args.targets} ({context_len} targets, shift={shift})")


if __name__ == "__main__":
    main()
