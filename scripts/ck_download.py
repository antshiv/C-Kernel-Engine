#!/usr/bin/env python3
"""
ck_download.py - HuggingFace model download helper for C-Kernel-Engine

Called by the C orchestrator (ck) to:
1. Download model from HuggingFace
2. Detect and validate architecture
3. Convert weights to bump format
4. Return model info as JSON

Usage:
    python3 ck_download.py --model "HuggingFaceTB/SmolLM-135M" --cache-dir ~/.cache/ck-engine/models
    python3 ck_download.py --model "https://huggingface.co/Qwen/Qwen2-0.5B" --cache-dir ~/.cache/ck-engine/models
    python3 ck_download.py --info-only --model "HuggingFaceTB/SmolLM-135M"  # Just print model info

Output (JSON):
    {
        "status": "ok",
        "model_id": "HuggingFaceTB/SmolLM-135M",
        "architecture": "LlamaForCausalLM",
        "supported": true,
        "model_type": "llama",
        "hidden_size": 576,
        "num_layers": 30,
        "num_heads": 9,
        "num_kv_heads": 3,
        "intermediate_size": 1536,
        "vocab_size": 49152,
        "param_count": 134515008,
        "cache_dir": "/home/user/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M",
        "config_path": "/home/user/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M/config.json",
        "weights_path": "/home/user/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M/weights.bump",
        "tokenizer_path": "/home/user/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M/tokenizer.json"
    }
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

# Supported architectures
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM": True,      # LLaMA, SmolLM, Mistral, etc.
    "MistralForCausalLM": True,    # Uses same arch as Llama
    "Qwen2ForCausalLM": True,      # Qwen2
    "GPT2LMHeadModel": False,      # TODO
    "GPTNeoXForCausalLM": False,   # TODO
    "PhiForCausalLM": False,       # TODO
    "GemmaForCausalLM": False,     # TODO
}

def parse_model_id(model_input: str) -> str:
    """
    Parse model input to get HuggingFace model ID.

    Accepts:
      - https://huggingface.co/org/model
      - org/model
      - model (assumes default org)
    """
    # HuggingFace URL
    if "huggingface.co/" in model_input:
        match = re.search(r"huggingface\.co/([^/]+/[^/]+)", model_input)
        if match:
            return match.group(1)

    # Already in org/model format
    if "/" in model_input and not model_input.startswith(("/", ".")):
        return model_input

    # Just model name - can't determine org
    return model_input

def get_cache_dir(model_id: str, base_cache_dir: str) -> Path:
    """Get the cache directory for a model."""
    safe_name = model_id.replace("/", "--")
    return Path(base_cache_dir) / safe_name

def download_model(model_id: str, cache_dir: Path, force: bool = False) -> dict:
    """
    Download model from HuggingFace.

    Returns dict with paths to downloaded files.
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        return {"error": "huggingface_hub not installed. Run: pip install huggingface-hub"}

    config_path = cache_dir / "config.json"
    tokenizer_path = cache_dir / "tokenizer.json"

    # Check if already downloaded
    if not force and config_path.exists():
        return {
            "status": "cached",
            "cache_dir": str(cache_dir),
            "config_path": str(config_path),
            "tokenizer_path": str(tokenizer_path) if tokenizer_path.exists() else None,
        }

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download model files
    try:
        snapshot_download(
            model_id,
            local_dir=str(cache_dir),
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "*.safetensors",
                "*.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ],
        )
    except Exception as e:
        return {"error": f"Failed to download model: {e}"}

    return {
        "status": "downloaded",
        "cache_dir": str(cache_dir),
        "config_path": str(config_path),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path.exists() else None,
    }

def load_config(config_path: Path) -> dict:
    """Load and parse config.json."""
    if not config_path.exists():
        return {"error": f"Config not found: {config_path}"}

    with open(config_path) as f:
        return json.load(f)

def detect_architecture(config: dict) -> dict:
    """
    Detect model architecture from config.json.

    Returns dict with architecture info.
    """
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "Unknown"

    supported = SUPPORTED_ARCHITECTURES.get(arch_name, False)

    # Get model parameters
    hidden_size = config.get("hidden_size", 0)
    num_layers = config.get("num_hidden_layers", 0)
    num_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", num_heads)  # Default to MHA
    intermediate_size = config.get("intermediate_size", 0)
    vocab_size = config.get("vocab_size", 0)
    model_type = config.get("model_type", "unknown")

    # Estimate parameter count
    # Rough formula: embedding + layers * (attention + mlp) + lm_head
    embed_params = vocab_size * hidden_size  # Embedding
    attn_params = num_layers * (
        4 * hidden_size * hidden_size  # Q, K, V, O (simplified)
    )
    mlp_params = num_layers * (
        3 * hidden_size * intermediate_size  # up, gate, down
    )
    lm_head_params = vocab_size * hidden_size  # Usually tied with embedding

    # More accurate for GQA models
    head_dim = hidden_size // num_heads if num_heads > 0 else 0
    q_params = num_layers * hidden_size * hidden_size
    kv_params = num_layers * 2 * num_kv_heads * head_dim * hidden_size
    o_params = num_layers * hidden_size * hidden_size
    attn_params = q_params + kv_params + o_params

    param_count = embed_params + attn_params + mlp_params

    return {
        "architecture": arch_name,
        "supported": supported,
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "param_count": param_count,
        "param_count_str": format_params(param_count),
    }

def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.0f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)

def convert_weights(cache_dir: Path, force: bool = False) -> dict:
    """
    Convert safetensors/bin weights to bump format.

    Calls the convert_hf_to_bump.py script.
    """
    weights_path = cache_dir / "weights.bump"

    # Check if already converted
    if not force and weights_path.exists():
        return {
            "status": "cached",
            "weights_path": str(weights_path),
        }

    # Find conversion script
    script_paths = [
        Path(__file__).parent / "convert_hf_to_bump.py",
        Path("scripts/convert_hf_to_bump.py"),
        Path("../scripts/convert_hf_to_bump.py"),
    ]

    convert_script = None
    for p in script_paths:
        if p.exists():
            convert_script = p
            break

    if not convert_script:
        return {"error": "Conversion script not found (convert_hf_to_bump.py)"}

    # Run conversion
    import subprocess
    result = subprocess.run(
        [sys.executable, str(convert_script), "--checkpoint", str(cache_dir), "--output", str(weights_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {"error": f"Conversion failed: {result.stderr}"}

    if not weights_path.exists():
        return {"error": "Conversion completed but weights.bump not created"}

    return {
        "status": "converted",
        "weights_path": str(weights_path),
    }

def main():
    parser = argparse.ArgumentParser(description="C-Kernel-Engine HuggingFace helper")
    parser.add_argument("--model", required=True, help="Model ID or HuggingFace URL")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/ck-engine/models"),
                        help="Cache directory for models")
    parser.add_argument("--info-only", action="store_true", help="Only show model info, don't download")
    parser.add_argument("--no-convert", action="store_true", help="Skip weight conversion")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-convert")
    parser.add_argument("--quiet", action="store_true", help="Only output JSON result")

    args = parser.parse_args()

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    # Parse model ID
    model_id = parse_model_id(args.model)
    log(f"Model ID: {model_id}")

    # Get cache directory
    cache_dir = get_cache_dir(model_id, args.cache_dir)
    log(f"Cache dir: {cache_dir}")

    result = {
        "status": "ok",
        "model_id": model_id,
        "cache_dir": str(cache_dir),
    }

    # Download model
    if not args.info_only:
        log("Downloading model...")
        download_result = download_model(model_id, cache_dir, force=args.force)

        if "error" in download_result:
            result["status"] = "error"
            result["error"] = download_result["error"]
            print(json.dumps(result, indent=2))
            sys.exit(1)

        result["download_status"] = download_result["status"]
        result["config_path"] = download_result["config_path"]
        result["tokenizer_path"] = download_result.get("tokenizer_path")
    else:
        # For info-only, still need config path
        result["config_path"] = str(cache_dir / "config.json")
        result["tokenizer_path"] = str(cache_dir / "tokenizer.json")

    # Load and analyze config
    config_path = Path(result["config_path"])
    if config_path.exists():
        config = load_config(config_path)

        if "error" in config:
            result["status"] = "error"
            result["error"] = config["error"]
            print(json.dumps(result, indent=2))
            sys.exit(1)

        # Detect architecture
        arch_info = detect_architecture(config)
        result.update(arch_info)

        log(f"Architecture: {arch_info['architecture']} ({'supported' if arch_info['supported'] else 'NOT supported'})")
        log(f"Parameters: {arch_info['param_count_str']}")

        if not arch_info["supported"]:
            log(f"WARNING: Architecture {arch_info['architecture']} is not yet supported!")
    else:
        log(f"Config not found at {config_path}")
        if args.info_only:
            result["status"] = "error"
            result["error"] = "Config not found. Download the model first."
            print(json.dumps(result, indent=2))
            sys.exit(1)

    # Convert weights
    if not args.info_only and not args.no_convert and result.get("supported", False):
        log("Converting weights...")
        convert_result = convert_weights(cache_dir, force=args.force)

        if "error" in convert_result:
            result["convert_status"] = "error"
            result["convert_error"] = convert_result["error"]
            log(f"Warning: {convert_result['error']}")
        else:
            result["convert_status"] = convert_result["status"]
            result["weights_path"] = convert_result["weights_path"]
    else:
        result["weights_path"] = str(cache_dir / "weights.bump")

    # Print result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
