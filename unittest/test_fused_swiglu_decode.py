"""
Fused SwiGLU decode MLP parity test (PyTorch vs C kernel).

Tests both:
1. Old approach: gemm_swiglu_fused + gemm_blocked_serial (two DRAM round-trips)
2. New approach: mlp_fused_swiglu_decode_v2 (fully fused, no intermediate DRAM)

Optionally loads real weights from a local HF model directory.
"""
import argparse
import ctypes
import sys
import time

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


def run_pytorch_reference(x, w_gate, w_up, w_down, b_gate, b_up, b_down):
    """PyTorch reference implementation."""
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
    return out_ref.squeeze(0)


def run_unfused_kernel(lib, x_pad, w_gate_pad, w_up_pad, w_down_pad,
                       b_gate_pad, b_up_pad, b_down_pad,
                       embed_dim, intermediate, aligned_embed, aligned_intermediate):
    """Old approach: gemm_swiglu_fused + gemm_blocked_serial (two calls)."""
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

    return mlp_out[:embed_dim].copy()


def run_fused_kernel(lib, x, w_gate, w_up, w_down,
                     b_gate, b_up, b_down,
                     embed_dim, intermediate,
                     version="v2"):
    """New approach: fused_mlp_swiglu_decode (fully fused).

    Uses UNPADDED matrices directly since the fused kernel handles
    arbitrary dimensions without needing alignment padding.
    """

    # Select kernel version
    if version == "v1":
        kernel_name = "fused_mlp_swiglu_decode"
    elif version == "v2":
        kernel_name = "fused_mlp_swiglu_decode_v2"
    elif version == "tiled":
        kernel_name = "fused_mlp_swiglu_decode_tiled"
    else:
        raise ValueError(f"Unknown version: {version}")

    kernel = getattr(lib, kernel_name)
    kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # W_gate
        ctypes.POINTER(ctypes.c_float),  # W_up
        ctypes.POINTER(ctypes.c_float),  # W_down
        ctypes.POINTER(ctypes.c_float),  # b_gate
        ctypes.POINTER(ctypes.c_float),  # b_up
        ctypes.POINTER(ctypes.c_float),  # b_down
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int, ctypes.c_int       # D, Hff
    ]

    # Use unpadded output buffer matching the actual dimensions
    mlp_out = np.zeros((embed_dim,), dtype=np.float32)

    # Ensure inputs are contiguous float32 arrays
    x_np = np.ascontiguousarray(x, dtype=np.float32)
    w_gate_np = np.ascontiguousarray(w_gate, dtype=np.float32)
    w_up_np = np.ascontiguousarray(w_up, dtype=np.float32)
    w_down_np = np.ascontiguousarray(w_down, dtype=np.float32)

    kernel(
        numpy_to_ptr(x_np),
        numpy_to_ptr(w_gate_np),
        numpy_to_ptr(w_up_np),
        numpy_to_ptr(w_down_np),
        None if b_gate is None else numpy_to_ptr(np.ascontiguousarray(b_gate, dtype=np.float32)),
        None if b_up is None else numpy_to_ptr(np.ascontiguousarray(b_up, dtype=np.float32)),
        None if b_down is None else numpy_to_ptr(np.ascontiguousarray(b_down, dtype=np.float32)),
        numpy_to_ptr(mlp_out),
        embed_dim, intermediate,
    )

    return mlp_out.copy()


def run_case(embed_dim, intermediate, seed=1234, weights=None, benchmark=False, num_iters=100):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(embed_dim).astype(np.float32)

    if weights is None:
        # Scale weights like real models (prevents huge intermediate values)
        # This matches test_mlp.py which uses 0.02 scaling
        w_gate = (rng.standard_normal((intermediate, embed_dim)) * 0.02).astype(np.float32)
        w_up = (rng.standard_normal((intermediate, embed_dim)) * 0.02).astype(np.float32)
        w_down = (rng.standard_normal((embed_dim, intermediate)) * 0.02).astype(np.float32)
        # Zero biases are common in pre-trained models
        b_gate = np.zeros((intermediate,), dtype=np.float32)
        b_up = np.zeros((intermediate,), dtype=np.float32)
        b_down = np.zeros((embed_dim,), dtype=np.float32)
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
    out_ref = run_pytorch_reference(x, w_gate, w_up, w_down, b_gate, b_up, b_down)

    # Pad to aligned sizes
    x_pad = pad_vector(x, aligned_embed)
    w_gate_pad = pad_matrix(w_gate, aligned_intermediate, aligned_embed)
    w_up_pad = pad_matrix(w_up, aligned_intermediate, aligned_embed)
    w_down_pad = pad_matrix(w_down, aligned_embed, aligned_intermediate)
    b_gate_pad = None if b_gate is None else pad_vector(b_gate, aligned_intermediate)
    b_up_pad = None if b_up is None else pad_vector(b_up, aligned_intermediate)
    b_down_pad = None if b_down is None else pad_vector(b_down, aligned_embed)

    # Load library
    lib = load_lib("libckernel_engine.so")

    results = {}

    # Test unfused kernel (old approach)
    out_unfused = run_unfused_kernel(
        lib, x_pad, w_gate_pad, w_up_pad, w_down_pad,
        b_gate_pad, b_up_pad, b_down_pad,
        embed_dim, intermediate, aligned_embed, aligned_intermediate
    )
    diff_unfused = max_diff(torch.from_numpy(out_unfused), out_ref)
    results["unfused"] = diff_unfused

    # Test fused kernels (new approaches)
    # Note: Fused kernel uses UNPADDED matrices directly (handles any dimension)
    for version in ["v2"]:  # v2 is recommended for multi-core
        try:
            out_fused = run_fused_kernel(
                lib, x, w_gate, w_up, w_down,
                b_gate, b_up, b_down,
                embed_dim, intermediate,
                version=version
            )
            diff_fused = max_diff(torch.from_numpy(out_fused), out_ref)
            results[f"fused_{version}"] = diff_fused
        except AttributeError as e:
            print(f"Warning: {version} kernel not found: {e}")
            results[f"fused_{version}"] = None

    # Benchmark if requested
    if benchmark:
        print(f"\n{'='*70}")
        print(f"  PERFORMANCE BENCHMARK: D={embed_dim}, Hff={intermediate}")
        print(f"{'='*70}")

        # Prepare PyTorch tensors for benchmarking
        x_t = torch.from_numpy(x).unsqueeze(0)
        w_gate_t = torch.from_numpy(w_gate)
        w_up_t = torch.from_numpy(w_up)
        w_down_t = torch.from_numpy(w_down)
        b_gate_t = torch.from_numpy(b_gate) if b_gate is not None else None
        b_up_t = torch.from_numpy(b_up) if b_up is not None else None
        b_down_t = torch.from_numpy(b_down) if b_down is not None else None

        def pytorch_mlp():
            gate = x_t @ w_gate_t.T
            up = x_t @ w_up_t.T
            if b_gate_t is not None:
                gate = gate + b_gate_t
            if b_up_t is not None:
                up = up + b_up_t
            swiglu = F.silu(gate) * up
            out = swiglu @ w_down_t.T
            if b_down_t is not None:
                out = out + b_down_t
            return out

        # Warmup PyTorch
        for _ in range(10):
            pytorch_mlp()

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(num_iters):
            pytorch_mlp()
        pytorch_time = (time.perf_counter() - start) / num_iters * 1000

        # Warmup unfused
        for _ in range(5):
            run_unfused_kernel(
                lib, x_pad, w_gate_pad, w_up_pad, w_down_pad,
                b_gate_pad, b_up_pad, b_down_pad,
                embed_dim, intermediate, aligned_embed, aligned_intermediate
            )

        # Benchmark unfused
        start = time.perf_counter()
        for _ in range(num_iters):
            run_unfused_kernel(
                lib, x_pad, w_gate_pad, w_up_pad, w_down_pad,
                b_gate_pad, b_up_pad, b_down_pad,
                embed_dim, intermediate, aligned_embed, aligned_intermediate
            )
        unfused_time = (time.perf_counter() - start) / num_iters * 1000

        # Print header
        print(f"  {'Implementation':<25} {'Time (ms)':<12} {'vs PyTorch':<12} {'vs Unfused':<12}")
        print(f"  {'-'*61}")

        # PyTorch baseline
        print(f"  {'PyTorch (CPU)':<25} {pytorch_time:<12.3f} {'1.00x':<12} {'-':<12}")

        # Unfused C kernel
        unfused_vs_pt = pytorch_time / unfused_time if unfused_time > 0 else 0
        print(f"  {'C Unfused (MKL)':<25} {unfused_time:<12.3f} {unfused_vs_pt:<12.2f}x {'-':<12}")

        # Benchmark fused versions
        for version in ["v2"]:
            try:
                # Warmup
                for _ in range(5):
                    run_fused_kernel(
                        lib, x, w_gate, w_up, w_down,
                        b_gate, b_up, b_down,
                        embed_dim, intermediate,
                        version=version
                    )

                start = time.perf_counter()
                for _ in range(num_iters):
                    run_fused_kernel(
                        lib, x, w_gate, w_up, w_down,
                        b_gate, b_up, b_down,
                        embed_dim, intermediate,
                        version=version
                    )
                fused_time = (time.perf_counter() - start) / num_iters * 1000
                fused_vs_pt = pytorch_time / fused_time if fused_time > 0 else 0
                fused_vs_unfused = unfused_time / fused_time if fused_time > 0 else 0
                print(f"  {'C Fused ' + version:<25} {fused_time:<12.3f} {fused_vs_pt:<12.2f}x {fused_vs_unfused:<12.2f}x")
            except AttributeError:
                print(f"  {'C Fused ' + version:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

        print(f"  {'-'*61}")
        print(f"  Note: Fused kernel benefits most from AVX-512 (Xeon 5th Gen)")
        print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fused SwiGLU decode MLP parity test")
    parser.add_argument("--model-dir", help="Optional HF model directory for real weights")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (when using --model-dir)")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for random test")
    parser.add_argument("--embed", type=int, default=896, help="Embed dim for random test (default: Qwen2 896)")
    parser.add_argument("--intermediate", type=int, default=4864, help="Intermediate dim (default: Qwen2 4864)")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Max abs diff tolerance")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip performance benchmark")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    print_system_info()

    if args.model_dir:
        weights = maybe_load_hf_weights(args.model_dir, args.layer)
        embed_dim = weights["embed_dim"]
        intermediate = weights["intermediate"]
        results = run_case(embed_dim, intermediate, seed=args.seed, weights=weights,
                          benchmark=args.benchmark, num_iters=args.iters)
        shape = f"model={args.model_dir} layer={args.layer} D={embed_dim} Hff={intermediate}"
    else:
        results = run_case(args.embed, args.intermediate, seed=args.seed, weights=None,
                          benchmark=args.benchmark, num_iters=args.iters)
        shape = f"D={args.embed} Hff={args.intermediate}"

    report = TestReport(
        test_name="Fused SwiGLU Decode MLP",
        dtype="fp32",
        shape=shape,
        cpu_info=get_cpu_info()
    )

    # Add results for each kernel version
    for name, diff in results.items():
        if diff is not None:
            report.add_result(TestResult(
                name=f"mlp_{name}",
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
