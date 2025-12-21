#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download


DEFAULT_PATTERNS = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SmolLM weights from the Hugging Face Hub")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="HF repo id")
    parser.add_argument(
        "--outdir",
        default="/home/antshiv/Workspace/HF-Models/SmolLM-135M",
        help="Output directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files (including ONNX artifacts)",
    )
    args = parser.parse_args()

    allow_patterns = None if args.all else DEFAULT_PATTERNS

    snapshot_download(
        repo_id=args.repo,
        local_dir=args.outdir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.repo} -> {args.outdir}")


if __name__ == "__main__":
    main()
