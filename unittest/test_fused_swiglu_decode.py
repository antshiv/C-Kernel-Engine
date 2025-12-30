"""
Fused SwiGLU decode MLP parity test (PyTorch vs C kernel).

Compares fused gate+up matvec + down projection against a PyTorch reference.
Optionally loads real weights from a local HF model directory.
"""
import argparse
import ctypes
import sys

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info, max_diff, numpy_to_ptr, print_system_info
)


CACHE_ALIGN = 64
FLOAT_SIZE = 4


def align_up_elems(elems, elem_bytes=FLOAT_SIZE, align_bytes=CACHE_ALIGN):
    if align_bytes == 0:
        return elems
    total_bytes = elems * elem_bytes
    aligned_bytes = ((total_bytes + align_bytes - 1) // align_bytes) * align_bytes
    return aligned_bytes // elem_bytes


def pad_matrix(mat, rows, cols):
    buf = np.zeros((rows, cols), dtype=np.float32)
    if mat is not None:
        r, c = mat.shape
        buf[:r, :c] = mat.astype(np.float32)
    return buf


def pad_vector(vec, size):
    buf = np.zeros((size,), dtype=np.float32)
    if vec is not None:
        v = vec.astype(np.float32).reshape(-1)
        buf[:v.size] = v
    return buf


def maybe_load_hf_weights(model_dir, layer_idx):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required for --model-dir") from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    state = model.state_dict()
    cfg = model.config.to_dict()
    if "text_config" in cfg:
        cfg = cfg["text_config"]

    num_layers = int(cfg.get("num_hidden_layers", 0))
    if layer_idx < 0 or layer_idx >= num_layers:
        raise SystemExit(f"Layer index {layer_idx} out of range 0..{num_layers - 1}")

    embed_dim = int(cfg.get("hidden_size", 0))
    intermediate = int(cfg.get("intermediate_size", 0))
    if embed_dim <= 0 or intermediate <= 0:
        raise SystemExit("Config missing hidden_size/intermediate_size")

    prefix = f"model.layers.{layer_idx}.mlp"
    w_gate = state[f"{prefix}.gate_proj.weight"].detach().cpu().numpy()
    w_up = state[f"{prefix}.up_proj.weight"].detach().cpu().numpy()
    w_down = state[f"{prefix}.down_proj.weight"].detach().cpu().numpy()
    b_gate = state.get(f"{prefix}.gate_proj.bias")
    b_up = state.get(f"{prefix}.up_proj.bias")
    b_down = state.get(f"{prefix}.down_proj.bias")
    b_gate = None if b_gate is None else b_gate.detach().cpu().numpy()
    b_up = None if b_up is None else b_up.detach().cpu().numpy()
    b_down = None if b_down is None else b_down.detach().cpu().numpy()

    return {
        "embed_dim": embed_dim,
        "intermediate": intermediate,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "b_gate": b_gate,
        "b_up": b_up,
        "b_down": b_down,
    }


def run_case(embed_dim, intermediate, seed=1234, weights=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(embed_dim).astype(np.float32)

    if weights is None:
        w_gate = rng.standard_normal((intermediate, embed_dim)).astype(np.float32)
        w_up = rng.standard_normal((intermediate, embed_dim)).astype(np.float32)
        w_down = rng.standard_normal((embed_dim, intermediate)).astype(np.float32)
        b_gate = rng.standard_normal((intermediate,)).astype(np.float32)
        b_up = rng.standard_normal((intermediate,)).astype(np.float32)
        b_down = rng.standard_normal((embed_dim,)).astype(np.float32)
    else:
        w_gate = weights["w_gate"]
        w_up = weights["w_up"]
        w_down = weights["w_down"]
        b_gate = weights["b_gate"]
        b_up = weights["b_up"]
        b_down = weights["b_down"]

    aligned_embed = align_up_elems(embed_dim)
    aligned_intermediate = align_up_elems(intermediate)

    # PyTorch reference
    x_t = torch.from_numpy(x).unsqueeze(0)
    gate = x_t @ torch.from_numpy(w_gate).T
    up = x_t @ torch.from_numpy(w_up).T
    if b_gate is not None:
        gate = gate + torch.from_numpy(b_gate)
    if b_up is not None:
        up = up + torch.from_numpy(b_up)
    swiglu = F.silu(gate) * up
    out_ref = swiglu @ torch.from_numpy(w_down).T
    if b_down is not None:
        out_ref = out_ref + torch.from_numpy(b_down)
    out_ref = out_ref.squeeze(0)

    # Pad to aligned sizes
    x_pad = pad_vector(x, aligned_embed)
    w_gate_pad = pad_matrix(w_gate, aligned_intermediate, aligned_embed)
    w_up_pad = pad_matrix(w_up, aligned_intermediate, aligned_embed)
    w_down_pad = pad_matrix(w_down, aligned_embed, aligned_intermediate)
    b_gate_pad = None if b_gate is None else pad_vector(b_gate, aligned_intermediate)
    b_up_pad = None if b_up is None else pad_vector(b_up, aligned_intermediate)
    b_down_pad = None if b_down is None else pad_vector(b_down, aligned_embed)

    # C kernel
    lib = load_lib("libckernel_engine.so")
    lib.gemm_swiglu_fused.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # W_gate
        ctypes.POINTER(ctypes.c_float),  # W_up
        ctypes.POINTER(ctypes.c_float),  # b_gate
        ctypes.POINTER(ctypes.c_float),  # b_up
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.gemm_blocked_serial.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # C
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    swiglu_out = np.zeros((aligned_intermediate,), dtype=np.float32)
    mlp_out = np.zeros((aligned_embed,), dtype=np.float32)

    lib.gemm_swiglu_fused(
        numpy_to_ptr(x_pad),
        numpy_to_ptr(w_gate_pad),
        numpy_to_ptr(w_up_pad),
        None if b_gate_pad is None else numpy_to_ptr(b_gate_pad),
        None if b_up_pad is None else numpy_to_ptr(b_up_pad),
        numpy_to_ptr(swiglu_out),
        1, aligned_intermediate, aligned_embed,
    )

    lib.gemm_blocked_serial(
        numpy_to_ptr(swiglu_out),
        numpy_to_ptr(w_down_pad),
        None if b_down_pad is None else numpy_to_ptr(b_down_pad),
        numpy_to_ptr(mlp_out),
        1, aligned_embed, aligned_intermediate,
    )

    out = torch.from_numpy(mlp_out[:embed_dim].copy())
    diff = max_diff(out, out_ref)
    return diff


def main():
    parser = argparse.ArgumentParser(description="Fused SwiGLU decode MLP parity test")
    parser.add_argument("--model-dir", help="Optional HF model directory for real weights")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (when using --model-dir)")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for random test")
    parser.add_argument("--embed", type=int, default=256, help="Embed dim for random test")
    parser.add_argument("--intermediate", type=int, default=512, help="Intermediate dim for random test")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Max abs diff tolerance")
    args = parser.parse_args()

    print_system_info()

    if args.model_dir:
        weights = maybe_load_hf_weights(args.model_dir, args.layer)
        embed_dim = weights["embed_dim"]
        intermediate = weights["intermediate"]
        diff = run_case(embed_dim, intermediate, seed=args.seed, weights=weights)
        shape = f"model={args.model_dir} layer={args.layer} D={embed_dim} Hff={intermediate}"
    else:
        diff = run_case(args.embed, args.intermediate, seed=args.seed, weights=None)
        shape = f"D={args.embed} Hff={args.intermediate}"

    report = TestReport(
        test_name="Fused SwiGLU Decode MLP",
        dtype="fp32",
        shape=shape,
        cpu_info=get_cpu_info()
    )
    report.add_result(TestResult(
        name="fused_swiglu_decode_mlp",
        passed=diff <= args.tolerance,
        max_diff=diff,
        tolerance=args.tolerance
    ))
    report.print_report()

    if not report.all_passed():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
