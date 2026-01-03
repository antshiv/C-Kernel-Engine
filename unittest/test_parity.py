#!/usr/bin/env python3
"""
test_parity.py - Compare C-Kernel-Engine outputs against PyTorch reference.

Usage:
    # Generate parity dumps from C:
    python scripts/ck_run_v4.py run <model> --parity

    # Then run this script to validate:
    python unittest/test_parity.py <parity_dir> <model_path>

The script:
1. Loads the model in PyTorch
2. Runs the same input tokens
3. Compares layer-by-layer outputs against C dumps
4. Reports max diff and first divergence point
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DEBUG = os.environ.get("DEBUG", "0") == "1"


def load_c_buffer(path: Path) -> np.ndarray:
    """Load a binary buffer saved by C parity mode."""
    return np.fromfile(path, dtype=np.float32)


def print_buffer_stats(name: str, arr: np.ndarray):
    """Print debug stats for a buffer."""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    zero_count = (arr == 0).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {name}: size={len(arr)} nan={nan_count} inf={inf_count} zero={zero_count}")
    else:
        print(f"  {name}: size={len(arr)} range=[{arr.min():.3e}, {arr.max():.3e}] mean={arr.mean():.3e}")


def compare_buffers(name: str, c_buf: np.ndarray, pt_buf: np.ndarray, tol: float = 1e-4) -> dict:
    """Compare C and PyTorch buffers, return comparison stats."""
    if c_buf.shape != pt_buf.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "c_shape": c_buf.shape,
            "pt_shape": pt_buf.shape,
        }

    # Check for NaN in C output
    c_nan = np.isnan(c_buf).sum()
    pt_nan = np.isnan(pt_buf).sum()

    if c_nan > 0:
        return {
            "name": name,
            "status": "C_HAS_NAN",
            "c_nan_count": int(c_nan),
            "pt_nan_count": int(pt_nan),
        }

    # Compute diff
    diff = np.abs(c_buf - pt_buf)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Find first divergence point
    diverge_idx = None
    if max_diff > tol:
        diverge_indices = np.where(diff > tol)[0]
        if len(diverge_indices) > 0:
            diverge_idx = int(diverge_indices[0])

    status = "PASS" if max_diff <= tol else "FAIL"

    return {
        "name": name,
        "status": status,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "diverge_idx": diverge_idx,
        "c_range": [float(c_buf.min()), float(c_buf.max())],
        "pt_range": [float(pt_buf.min()), float(pt_buf.max())],
    }


def run_pytorch_reference(model_path: Path, token_ids: list) -> dict:
    """Run PyTorch reference model and return outputs at each layer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("ERROR: transformers and torch required for parity testing")
        print("  pip install transformers torch")
        sys.exit(1)

    print(f"Loading PyTorch model from {model_path}...")

    # Load model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    outputs = {}

    # Hook to capture intermediate outputs
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Get last token output
            outputs[name] = out[0, -1, :].detach().numpy().astype(np.float32)
        return hook

    # Register hooks for each layer
    hooks = []

    # Embedding
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_out")))

    # Each transformer layer
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}_output")))

    # Final layer norm
    hooks.append(model.model.norm.register_forward_hook(make_hook("final_rmsnorm_out")))

    # Run forward
    with torch.no_grad():
        logits = model(input_ids).logits

    # Save logits
    outputs["logits"] = logits[0, -1, :].detach().numpy().astype(np.float32)

    # Remove hooks
    for h in hooks:
        h.remove()

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Compare C-Kernel-Engine outputs against PyTorch")
    parser.add_argument("parity_dir", type=Path, help="Directory with C parity dumps")
    parser.add_argument("model_path", type=Path, help="Path to HuggingFace model")
    parser.add_argument("--token-idx", type=int, default=0, help="Token index to compare")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for comparison")
    parser.add_argument("--tokens", type=str, default="1,2,3", help="Comma-separated token IDs")
    args = parser.parse_args()

    token_ids = [int(t) for t in args.tokens.split(",")]

    print("=" * 70)
    print("  C-Kernel-Engine Parity Test")
    print("=" * 70)
    print(f"  Parity dir:  {args.parity_dir}")
    print(f"  Model:       {args.model_path}")
    print(f"  Token IDs:   {token_ids}")
    print(f"  Tolerance:   {args.tol}")
    print("=" * 70)

    # Find C buffers
    c_buffers = {}
    suffix = f"_tok{args.token_idx}.bin"
    for f in args.parity_dir.glob(f"*{suffix}"):
        name = f.name.replace(suffix, "")
        c_buffers[name] = load_c_buffer(f)
        print(f"Loaded C buffer: {name} ({len(c_buffers[name])} floats)")

    if not c_buffers:
        print(f"ERROR: No C buffers found in {args.parity_dir}")
        print(f"  Looking for files matching *{suffix}")
        sys.exit(1)

    # Run PyTorch reference
    pt_outputs = run_pytorch_reference(args.model_path, token_ids)

    print("\n" + "=" * 70)
    print("  PARITY COMPARISON")
    print("=" * 70)

    results = []
    for name in sorted(c_buffers.keys()):
        if name not in pt_outputs:
            print(f"  {name}: SKIPPED (no PyTorch reference)")
            continue

        c_buf = c_buffers[name]
        pt_buf = pt_outputs[name]

        # Handle size mismatch (aligned vs unaligned)
        if len(c_buf) > len(pt_buf):
            # C has aligned padding - trim to PT size
            c_buf = c_buf[:len(pt_buf)]
        elif len(pt_buf) > len(c_buf):
            pt_buf = pt_buf[:len(c_buf)]

        result = compare_buffers(name, c_buf, pt_buf, args.tol)
        results.append(result)

        status_color = "\033[92m" if result["status"] == "PASS" else "\033[91m"
        reset = "\033[0m"

        if result["status"] == "PASS":
            print(f"  {status_color}[PASS]{reset} {name}: max_diff={result['max_diff']:.3e}")
        elif result["status"] == "C_HAS_NAN":
            print(f"  {status_color}[FAIL]{reset} {name}: C has {result['c_nan_count']} NaN values!")
        elif result["status"] == "SHAPE_MISMATCH":
            print(f"  {status_color}[FAIL]{reset} {name}: shape mismatch C={result['c_shape']} PT={result['pt_shape']}")
        else:
            print(f"  {status_color}[FAIL]{reset} {name}: max_diff={result['max_diff']:.3e} (first diverge @ idx {result['diverge_idx']})")
            if DEBUG:
                print(f"         C range:  {result['c_range']}")
                print(f"         PT range: {result['pt_range']}")

    print("=" * 70)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)

    if passed == total:
        print(f"\033[92m  ALL TESTS PASSED ({passed}/{total})\033[0m")
        return 0
    else:
        print(f"\033[91m  TESTS FAILED ({passed}/{total} passed)\033[0m")

        # Find first failure
        for r in results:
            if r["status"] != "PASS":
                print(f"\n  First failure: {r['name']}")
                if r["status"] == "C_HAS_NAN":
                    print(f"    C output contains {r['c_nan_count']} NaN values")
                    print(f"    This indicates a bug in the C kernel or weight loading")
                elif r.get("diverge_idx") is not None:
                    print(f"    Divergence starts at index {r['diverge_idx']}")
                break

        return 1


if __name__ == "__main__":
    sys.exit(main())
