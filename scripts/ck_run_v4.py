#!/usr/bin/env python3
"""
ck_run_v4.py - C-Kernel-Engine v4 Pipeline Runner

Unified CLI that chains: download -> convert -> IR -> codegen -> compile -> run

Usage:
  python scripts/ck_run_v4.py run HuggingFaceTB/SmolLM-135M
  python scripts/ck_run_v4.py run ./model.gguf
  python scripts/ck_run_v4.py run ./local/config.json
  python scripts/ck_run_v4.py run Qwen/Qwen2-0.5B --weight-dtype=q4_k
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path.home() / ".cache" / "ck-engine-v4" / "models"
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build"

# Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ORANGE = "\033[38;5;214m"
C_GREEN = "\033[38;5;114m"
C_BLUE = "\033[38;5;75m"
C_RED = "\033[38;5;203m"
C_CYAN = "\033[38;5;87m"


def log(msg: str, color: str = ""):
    """Print colored log message."""
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_step(step: int, msg: str):
    """Print pipeline step."""
    print(f"{C_ORANGE}[{step}/6]{C_RESET} {C_BOLD}{msg}{C_RESET}")


def log_error(msg: str):
    """Print error message."""
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


def run_cmd(cmd: list, cwd: Path = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run command with error handling."""
    try:
        if capture:
            return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        else:
            return subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Input Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_input_type(model_input: str) -> tuple[str, dict]:
    """
    Detect input type and return (type, info).
    Types: 'hf_id', 'hf_url', 'gguf', 'local_dir', 'local_config'
    """
    # Local GGUF file
    if model_input.endswith('.gguf') and Path(model_input).exists():
        return 'gguf', {'path': Path(model_input).resolve()}

    # Local config.json
    if model_input.endswith('.json') and Path(model_input).exists():
        return 'local_config', {'path': Path(model_input).resolve()}

    # Local directory with config.json
    local_path = Path(model_input)
    if local_path.is_dir() and (local_path / "config.json").exists():
        return 'local_dir', {'path': local_path.resolve()}

    # HuggingFace URL
    if model_input.startswith('https://huggingface.co/'):
        # Extract org/model from URL
        parts = model_input.replace('https://huggingface.co/', '').strip('/').split('/')
        if len(parts) >= 2:
            model_id = f"{parts[0]}/{parts[1]}"
            return 'hf_id', {'model_id': model_id, 'org': parts[0], 'name': parts[1]}

    # HuggingFace model ID (org/model or just model)
    if '/' in model_input:
        org, name = model_input.split('/', 1)
        return 'hf_id', {'model_id': model_input, 'org': org, 'name': name}

    # Assume single name is HF model (search common orgs)
    return 'hf_id', {'model_id': model_input, 'org': '', 'name': model_input}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step_download(model_id: str, cache_dir: Path, force: bool = False) -> Path:
    """Download model from HuggingFace Hub."""
    log_step(1, f"Downloading {model_id}")

    model_dir = cache_dir / model_id.replace('/', '--')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    config_path = model_dir / "config.json"
    if config_path.exists() and not force:
        log(f"  Using cached model at {model_dir}", C_DIM)
        return model_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log_error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    log(f"  Downloading to {model_dir}", C_DIM)
    snapshot_download(
        model_id,
        local_dir=str(model_dir),
        ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],  # Skip non-safetensors
    )

    return model_dir


def step_convert_hf(model_dir: Path, output_dir: Path, weight_dtype: str = "float32", force: bool = False) -> Path:
    """Convert HF safetensors to bump format."""
    log_step(2, f"Converting weights to bump format ({weight_dtype})")

    weights_path = output_dir / "weights.bump"
    manifest_path = output_dir / "weights_manifest.json"

    if weights_path.exists() and manifest_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        return weights_path

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_hf_to_bump_v4.py"),
        f"--checkpoint={model_dir}",
        f"--output={weights_path}",
        f"--dtype={weight_dtype}",
        f"--manifest-out={manifest_path}",
    ]

    run_cmd(cmd)
    log(f"  Created {weights_path}", C_GREEN)
    return weights_path


def step_convert_gguf(gguf_path: Path, output_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Convert GGUF to bump format."""
    log_step(2, f"Converting GGUF to bump format")

    weights_path = output_dir / "weights.bump"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "weights_manifest.json"

    if weights_path.exists() and config_path.exists() and not force:
        log(f"  Using cached weights at {weights_path}", C_DIM)
        return weights_path, config_path

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "convert_gguf_to_bump_v4.py"),
        f"--input={gguf_path}",
        f"--output={weights_path}",
        f"--config-out={config_path}",
        f"--manifest-out={manifest_path}",
    ]

    run_cmd(cmd)
    log(f"  Created {weights_path}", C_GREEN)
    return weights_path, config_path


def step_build_ir(config_path: Path, output_dir: Path, manifest_path: Path = None,
                  weight_dtype: str = None, modes: list = None, force: bool = False) -> Path:
    """Build IR and generate layout JSON."""
    log_step(3, "Building IR v4 and layout")

    layout_path = output_dir / "layout.json"

    if layout_path.exists() and not force:
        log(f"  Using cached layout at {layout_path}", C_DIM)
        return layout_path

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_ir_v4.py"),
        f"--config={config_path}",
        f"--prefix={output_dir}",
        "--emit=lib",
    ]

    if manifest_path and manifest_path.exists():
        cmd.append(f"--weights-manifest={manifest_path}")

    if weight_dtype:
        cmd.append(f"--weight-dtype={weight_dtype}")

    if modes:
        cmd.append(f"--modes={','.join(modes)}")
    else:
        cmd.append("--modes=prefill,decode")

    run_cmd(cmd)
    log(f"  Created {layout_path}", C_GREEN)
    return layout_path


def step_codegen(layout_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Generate C code from layout."""
    log_step(4, "Generating C code")

    model_c_path = output_dir / "model.c"

    if model_c_path.exists() and not force:
        log(f"  Using cached C code at {model_c_path}", C_DIM)
        return model_c_path

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "codegen_v4.py"),
        str(layout_path),
        f"--output={model_c_path}",
    ]

    run_cmd(cmd)
    log(f"  Created {model_c_path}", C_GREEN)
    return model_c_path


def step_compile(model_c_path: Path, output_dir: Path, force: bool = False) -> Path:
    """Compile C code to shared library."""
    log_step(5, "Compiling to shared library")

    lib_path = output_dir / "libmodel.so"

    if lib_path.exists() and not force:
        log(f"  Using cached library at {lib_path}", C_DIM)
        return lib_path

    # Find kernel sources
    kernel_list_path = model_c_path.with_suffix('.c.kernels')
    kernel_sources = []
    if kernel_list_path.exists():
        kernel_sources = kernel_list_path.read_text().strip().split('\n')

    # Default kernel sources if not specified
    if not kernel_sources:
        src_dir = PROJECT_ROOT / "src" / "kernels"
        kernel_sources = [str(f) for f in src_dir.glob("*.c")]

    # Build command
    cmd = [
        "gcc", "-O3", "-fPIC", "-fopenmp", "-shared",
        f"-I{PROJECT_ROOT / 'include'}",
        "-o", str(lib_path),
        str(model_c_path),
    ] + kernel_sources + ["-lm"]

    log(f"  Compiling with {len(kernel_sources)} kernel sources", C_DIM)
    run_cmd(cmd)
    log(f"  Created {lib_path}", C_GREEN)
    return lib_path


def step_run_chat(model_dir: Path, args: argparse.Namespace):
    """Run chat interface."""
    log_step(6, "Starting chat")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "ck_chat.py"),
        str(model_dir),
    ]

    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])

    # Replace current process with chat
    os.execvp(sys.executable, cmd)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace):
    """Run the full v4 pipeline."""
    model_input = args.model

    # Detect input type
    input_type, info = detect_input_type(model_input)
    log(f"{C_ORANGE}C-Kernel-Engine v4{C_RESET}")
    log(f"Input: {model_input} ({input_type})", C_DIM)

    # Determine working directory
    if input_type == 'hf_id':
        model_id = info['model_id']
        work_dir = CACHE_DIR / model_id.replace('/', '--')
        model_dir = step_download(model_id, CACHE_DIR, force=args.force_download)
        config_path = model_dir / "config.json"

        # Convert weights
        weights_path = step_convert_hf(
            model_dir, work_dir,
            weight_dtype=args.weight_dtype or "float32",
            force=args.force_convert
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'gguf':
        gguf_path = info['path']
        work_dir = CACHE_DIR / gguf_path.stem
        weights_path, config_path = step_convert_gguf(
            gguf_path, work_dir,
            force=args.force_convert
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'local_dir':
        model_dir = info['path']
        work_dir = model_dir / ".ck_build"
        config_path = model_dir / "config.json"

        # Convert weights
        weights_path = step_convert_hf(
            model_dir, work_dir,
            weight_dtype=args.weight_dtype or "float32",
            force=args.force_convert
        )
        manifest_path = work_dir / "weights_manifest.json"

    elif input_type == 'local_config':
        config_path = info['path']
        work_dir = config_path.parent / ".ck_build"
        manifest_path = None
        # No weight conversion for config-only (assume weights.bump exists)

    else:
        log_error(f"Unknown input type: {input_type}")
        sys.exit(1)

    # Build IR
    layout_path = step_build_ir(
        config_path, work_dir,
        manifest_path=manifest_path,
        weight_dtype=args.weight_dtype,
        force=args.force_compile
    )

    # Generate C code
    model_c_path = step_codegen(layout_path, work_dir, force=args.force_compile)

    # Compile
    lib_path = step_compile(model_c_path, work_dir, force=args.force_compile)

    # Copy tokenizer if available
    tokenizer_src = None
    if input_type in ('hf_id', 'local_dir'):
        tokenizer_src = (model_dir if input_type == 'hf_id' else info['path']) / "tokenizer.json"
    if tokenizer_src and tokenizer_src.exists():
        tokenizer_dst = work_dir / "tokenizer.json"
        if not tokenizer_dst.exists():
            shutil.copy(tokenizer_src, tokenizer_dst)

    # Run chat (unless generate-only)
    if args.generate_only:
        log(f"\n{C_GREEN}Generated:{C_RESET}")
        log(f"  Layout:  {layout_path}")
        log(f"  C code:  {model_c_path}")
        log(f"  Library: {lib_path}")
    else:
        print()
        step_run_chat(work_dir, args)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine v4 Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ck_run_v4.py run HuggingFaceTB/SmolLM-135M
  python scripts/ck_run_v4.py run ./model.gguf --weight-dtype=q4_k
  python scripts/ck_run_v4.py run Qwen/Qwen2-0.5B --generate-only
  python scripts/ck_run_v4.py run ./local/config.json --prompt "Hello"
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run model')
    run_parser.add_argument('model', help='Model ID, URL, GGUF file, or local path')
    run_parser.add_argument('--weight-dtype', choices=['float32', 'bf16', 'q4_k', 'q6_k'],
                           help='Weight dtype (default: float32)')
    run_parser.add_argument('--temperature', type=float, default=0.7,
                           help='Sampling temperature (default: 0.7)')
    run_parser.add_argument('--max-tokens', type=int, default=512,
                           help='Max tokens to generate (default: 512)')
    run_parser.add_argument('--prompt', help='Single prompt (non-interactive)')
    run_parser.add_argument('--force-download', action='store_true',
                           help='Re-download model files')
    run_parser.add_argument('--force-convert', action='store_true',
                           help='Re-convert weights')
    run_parser.add_argument('--force-compile', action='store_true',
                           help='Re-generate and recompile')
    run_parser.add_argument('--generate-only', action='store_true',
                           help='Generate C code only, do not run')

    # List command
    list_parser = subparsers.add_parser('list', help='List cached models')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cached models')
    clean_parser.add_argument('model', nargs='?', help='Model to clean (or all)')

    args = parser.parse_args()

    if args.command == 'run':
        run_pipeline(args)
    elif args.command == 'list':
        if CACHE_DIR.exists():
            models = list(CACHE_DIR.iterdir())
            if models:
                log(f"Cached models in {CACHE_DIR}:")
                for m in sorted(models):
                    if m.is_dir():
                        size = sum(f.stat().st_size for f in m.rglob('*') if f.is_file())
                        log(f"  {m.name.replace('--', '/')} ({size / 1e6:.1f} MB)")
            else:
                log("No cached models")
        else:
            log("No cached models")
    elif args.command == 'clean':
        if args.model:
            model_dir = CACHE_DIR / args.model.replace('/', '--')
            if model_dir.exists():
                shutil.rmtree(model_dir)
                log(f"Removed {args.model}")
            else:
                log_error(f"Model not found: {args.model}")
        else:
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                log("Cleaned all cached models")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
