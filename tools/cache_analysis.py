#!/usr/bin/env python3
"""
CPU Cache & Model Memory Hierarchy Analyzer

This tool analyzes the relationship between:
1. CPU cache hierarchy (L1, L2, L3)
2. Model weights and activations
3. Fusion opportunities per cache level

Usage:
    python tools/cache_analysis.py [model_config.json]
    python tools/cache_analysis.py --cpu-only
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict

# =============================================================================
# CPU Cache Detection
# =============================================================================

@dataclass
class CacheLevel:
    level: int          # 1, 2, or 3
    type: str           # "Data", "Instruction", "Unified"
    size_kb: int
    line_size: int      # bytes
    ways: int           # associativity
    sets: int
    shared_cores: int   # how many cores share this cache

@dataclass
class CPUInfo:
    model_name: str
    cores: int
    threads: int
    base_freq_mhz: float
    l1d: CacheLevel
    l1i: CacheLevel
    l2: CacheLevel
    l3: CacheLevel
    memory_bandwidth_gbps: float  # estimated
    simd_width: int  # 256 for AVX2, 512 for AVX-512


def detect_cpu_info() -> CPUInfo:
    """Detect CPU cache hierarchy from Linux sysfs."""

    # Get CPU model name
    model_name = "Unknown CPU"
    cores = 1
    threads = 1
    freq_mhz = 2000.0
    simd_width = 256

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    model_name = line.split(":")[1].strip()
                    break

            # Count cores and threads
            cores = cpuinfo.count("processor\t:")
            threads = cores  # Simplified

            for line in cpuinfo.split("\n"):
                if "cpu MHz" in line:
                    try:
                        freq_mhz = float(line.split(":")[1].strip())
                    except:
                        pass
                    break

            # Detect SIMD capability
            if "avx512" in cpuinfo.lower():
                simd_width = 512
            elif "avx2" in cpuinfo.lower() or "avx" in cpuinfo.lower():
                simd_width = 256
            else:
                simd_width = 128

    except Exception as e:
        print(f"Warning: Could not read /proc/cpuinfo: {e}")

    # Detect cache hierarchy from sysfs
    caches = {1: {}, 2: {}, 3: {}}

    try:
        cache_base = "/sys/devices/system/cpu/cpu0/cache"
        if os.path.exists(cache_base):
            for idx in range(10):
                cache_dir = f"{cache_base}/index{idx}"
                if not os.path.exists(cache_dir):
                    continue

                def read_file(name):
                    path = f"{cache_dir}/{name}"
                    if os.path.exists(path):
                        with open(path) as f:
                            return f.read().strip()
                    return None

                level = int(read_file("level") or "0")
                cache_type = read_file("type") or "Unknown"
                size_str = read_file("size") or "0K"
                line_size = int(read_file("coherency_line_size") or "64")
                ways = int(read_file("ways_of_associativity") or "8")
                sets = int(read_file("number_of_sets") or "64")
                shared = read_file("shared_cpu_list") or "0"

                # Parse size (e.g., "32K", "256K", "8192K")
                size_kb = int(size_str.replace("K", "").replace("M", "000"))

                # Count shared cores
                if "-" in shared:
                    parts = shared.split("-")
                    shared_cores = int(parts[1]) - int(parts[0]) + 1
                elif "," in shared:
                    shared_cores = len(shared.split(","))
                else:
                    shared_cores = 1

                if level in caches:
                    caches[level][cache_type] = CacheLevel(
                        level=level,
                        type=cache_type,
                        size_kb=size_kb,
                        line_size=line_size,
                        ways=ways,
                        sets=sets,
                        shared_cores=shared_cores
                    )
    except Exception as e:
        print(f"Warning: Could not read cache info from sysfs: {e}")

    # Build cache levels (use defaults if not detected)
    l1d = caches.get(1, {}).get("Data", CacheLevel(1, "Data", 32, 64, 8, 64, 1))
    l1i = caches.get(1, {}).get("Instruction", CacheLevel(1, "Instruction", 32, 64, 8, 64, 1))
    l2 = caches.get(2, {}).get("Unified", CacheLevel(2, "Unified", 256, 64, 8, 512, 1))
    l3 = caches.get(3, {}).get("Unified", CacheLevel(3, "Unified", 8192, 64, 16, 8192, cores))

    # Estimate memory bandwidth (rough heuristic based on DDR generation)
    # Modern DDR4: ~50 GB/s, DDR5: ~80 GB/s
    memory_bandwidth_gbps = 50.0  # Conservative estimate

    return CPUInfo(
        model_name=model_name,
        cores=cores,
        threads=threads,
        base_freq_mhz=freq_mhz,
        l1d=l1d,
        l1i=l1i,
        l2=l2,
        l3=l3,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        simd_width=simd_width
    )


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_context: int
    dtype_bytes: int  # 4 for fp32, 2 for fp16/bf16


def load_model_config(config_path: str) -> ModelConfig:
    """Load model config from HuggingFace-style config.json."""
    with open(config_path) as f:
        cfg = json.load(f)

    hidden = cfg.get("hidden_size", 896)
    heads = cfg.get("num_attention_heads", 14)

    return ModelConfig(
        name=cfg.get("_name_or_path", "unknown"),
        hidden_size=hidden,
        num_layers=cfg.get("num_hidden_layers", 24),
        num_heads=heads,
        num_kv_heads=cfg.get("num_key_value_heads", heads),
        head_dim=hidden // heads,
        intermediate_size=cfg.get("intermediate_size", 4864),
        vocab_size=cfg.get("vocab_size", 151936),
        max_context=cfg.get("max_position_embeddings", 131072),
        dtype_bytes=4  # Assume fp32 for now
    )


def default_qwen2_config() -> ModelConfig:
    """Default Qwen2-0.5B configuration."""
    return ModelConfig(
        name="Qwen2-0.5B",
        hidden_size=896,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=4864,
        vocab_size=151936,
        max_context=131072,
        dtype_bytes=4
    )


# =============================================================================
# Memory Analysis
# =============================================================================

@dataclass
class LayerMemory:
    """Memory requirements for one transformer layer."""
    # Weights (read-only)
    wq_bytes: int
    wk_bytes: int
    wv_bytes: int
    wo_bytes: int
    w1_bytes: int  # gate + up
    w2_bytes: int  # down
    ln1_bytes: int
    ln2_bytes: int
    total_weight_bytes: int

    # Activations (per token)
    ln_out_bytes: int
    q_bytes: int
    k_bytes: int
    v_bytes: int
    attn_out_bytes: int
    fc1_out_bytes: int
    swiglu_out_bytes: int
    total_activation_bytes: int

    # Per-head working set
    per_head_scratch_bytes: int


def analyze_layer_memory(cfg: ModelConfig) -> LayerMemory:
    """Calculate memory requirements for one layer."""
    D = cfg.hidden_size
    H = cfg.num_heads
    H_kv = cfg.num_kv_heads
    hd = cfg.head_dim
    I = cfg.intermediate_size
    B = cfg.dtype_bytes

    # Weights
    wq = H * hd * D * B
    wk = H_kv * hd * D * B
    wv = H_kv * hd * D * B
    wo = D * H * hd * B
    w1 = 2 * I * D * B  # gate + up fused
    w2 = D * I * B
    ln1 = D * B
    ln2 = D * B

    # Activations (per token, T=1)
    ln_out = D * B
    q = H * hd * B
    k = H_kv * hd * B
    v = H_kv * hd * B
    attn_out = H * hd * B
    fc1_out = 2 * I * B
    swiglu_out = I * B

    # Per-head scratch (for tiled GEMV)
    per_head = hd * B * 4  # q, partial_k, partial_v, scores tile

    return LayerMemory(
        wq_bytes=wq, wk_bytes=wk, wv_bytes=wv, wo_bytes=wo,
        w1_bytes=w1, w2_bytes=w2, ln1_bytes=ln1, ln2_bytes=ln2,
        total_weight_bytes=wq + wk + wv + wo + w1 + w2 + ln1 + ln2,
        ln_out_bytes=ln_out, q_bytes=q, k_bytes=k, v_bytes=v,
        attn_out_bytes=attn_out, fc1_out_bytes=fc1_out,
        swiglu_out_bytes=swiglu_out,
        total_activation_bytes=ln_out + q + k + v + attn_out + fc1_out + swiglu_out,
        per_head_scratch_bytes=per_head
    )


# =============================================================================
# Cache Mapping Analysis
# =============================================================================

def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024 * 1024 * 1024:
        return f"{b / (1024**3):.2f} GB"
    elif b >= 1024 * 1024:
        return f"{b / (1024**2):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b} B"


def analyze_cache_mapping(cpu: CPUInfo, model: ModelConfig, layer: LayerMemory):
    """Analyze what fits in each cache level."""

    l1_bytes = cpu.l1d.size_kb * 1024
    l2_bytes = cpu.l2.size_kb * 1024
    l3_bytes = cpu.l3.size_kb * 1024

    print("\n" + "=" * 80)
    print("CACHE HIERARCHY ANALYSIS")
    print("=" * 80)

    print(f"\n{'CPU:':<20} {cpu.model_name}")
    print(f"{'Cores:':<20} {cpu.cores}")
    print(f"{'SIMD Width:':<20} {cpu.simd_width}-bit ({'AVX-512' if cpu.simd_width == 512 else 'AVX2' if cpu.simd_width == 256 else 'SSE'})")
    print(f"{'Est. DRAM BW:':<20} {cpu.memory_bandwidth_gbps:.1f} GB/s")

    print(f"\n{'Model:':<20} {model.name}")
    print(f"{'Layers:':<20} {model.num_layers}")
    print(f"{'Hidden Size:':<20} {model.hidden_size}")
    print(f"{'Heads:':<20} {model.num_heads} Q, {model.num_kv_heads} KV")
    print(f"{'Head Dim:':<20} {model.head_dim}")
    print(f"{'Intermediate:':<20} {model.intermediate_size}")

    # L1 Analysis
    print("\n" + "-" * 80)
    print("L1 DATA CACHE ANALYSIS")
    print("-" * 80)
    print(f"{'L1D Size:':<30} {format_bytes(l1_bytes)} per core")
    print(f"{'L1D Line Size:':<30} {cpu.l1d.line_size} bytes")
    print(f"{'L1D Associativity:':<30} {cpu.l1d.ways}-way")

    print(f"\n{'What fits in L1:'}")
    print(f"  {'Per-head scratch:':<25} {format_bytes(layer.per_head_scratch_bytes):<12} {'✓ FITS' if layer.per_head_scratch_bytes < l1_bytes else '✗ TOO BIG'}")
    print(f"  {'One head Q output:':<25} {format_bytes(model.head_dim * model.dtype_bytes):<12} ✓ FITS")
    print(f"  {'Attention scores tile:':<25} {format_bytes(64 * model.dtype_bytes):<12} ✓ FITS (64 positions)")
    print(f"  {'RMSNorm gamma:':<25} {format_bytes(layer.ln1_bytes):<12} {'✓ FITS' if layer.ln1_bytes < l1_bytes else '✗'}")

    heads_in_l1 = l1_bytes // layer.per_head_scratch_bytes
    print(f"\n  → Can process {heads_in_l1} heads simultaneously in L1")

    # L2 Analysis
    print("\n" + "-" * 80)
    print("L2 CACHE ANALYSIS")
    print("-" * 80)
    print(f"{'L2 Size:':<30} {format_bytes(l2_bytes)} per core")

    print(f"\n{'What fits in L2:'}")
    print(f"  {'All layer activations:':<25} {format_bytes(layer.total_activation_bytes):<12} {'✓ FITS' if layer.total_activation_bytes < l2_bytes else '✗'}")
    print(f"  {'Q + K + V (all heads):':<25} {format_bytes(layer.q_bytes + layer.k_bytes + layer.v_bytes):<12} {'✓ FITS' if (layer.q_bytes + layer.k_bytes + layer.v_bytes) < l2_bytes else '✗'}")
    print(f"  {'fc1_out (gate+up):':<25} {format_bytes(layer.fc1_out_bytes):<12} {'✓ FITS' if layer.fc1_out_bytes < l2_bytes else '✗'}")

    # How much weight can fit in L2?
    l2_for_weights = l2_bytes - layer.total_activation_bytes
    print(f"\n  L2 remaining for weights: {format_bytes(max(0, l2_for_weights))}")

    # Calculate weight tile sizes for L2
    wq_tile_rows = l2_for_weights // (model.hidden_size * model.dtype_bytes * 2)  # *2 for safety margin
    print(f"  → Can tile Wq with {wq_tile_rows} output rows at a time")

    # Fusion opportunities in L2
    fusion_scratch = layer.ln_out_bytes + layer.q_bytes + layer.k_bytes + layer.v_bytes + layer.attn_out_bytes
    print(f"\n{'Fused attention block scratch:':<30} {format_bytes(fusion_scratch)}")
    print(f"  → {'✓ Full attention block fits in L2!' if fusion_scratch < l2_bytes else '✗ Need to tile'}")

    mlp_fusion_scratch = layer.ln_out_bytes + layer.fc1_out_bytes + layer.swiglu_out_bytes
    print(f"{'Fused MLP block scratch:':<30} {format_bytes(mlp_fusion_scratch)}")
    print(f"  → {'✓ Full MLP block fits in L2!' if mlp_fusion_scratch < l2_bytes else '✗ Need to tile'}")

    # L3 Analysis
    print("\n" + "-" * 80)
    print("L3 CACHE ANALYSIS (Shared)")
    print("-" * 80)
    print(f"{'L3 Size:':<30} {format_bytes(l3_bytes)} shared across {cpu.l3.shared_cores} cores")
    print(f"{'L3 per core:':<30} {format_bytes(l3_bytes // max(1, cpu.l3.shared_cores))}")

    print(f"\n{'What fits in L3:'}")
    print(f"  {'One layer weights:':<25} {format_bytes(layer.total_weight_bytes):<12} {'✓ FITS' if layer.total_weight_bytes < l3_bytes else '✗'}")

    layers_in_l3 = l3_bytes // layer.total_weight_bytes
    print(f"  {'Layers that fit:':<25} {layers_in_l3} of {model.num_layers}")

    total_model_weights = layer.total_weight_bytes * model.num_layers
    print(f"  {'Total model weights:':<25} {format_bytes(total_model_weights)}")
    print(f"  → {'✓ Full model fits in L3!' if total_model_weights < l3_bytes else f'✗ Need {format_bytes(total_model_weights - l3_bytes)} more L3'}")

    # KV Cache analysis
    print("\n" + "-" * 80)
    print("KV CACHE ANALYSIS")
    print("-" * 80)

    kv_per_token_per_layer = 2 * model.num_kv_heads * model.head_dim * model.dtype_bytes
    kv_per_token_total = kv_per_token_per_layer * model.num_layers

    print(f"{'KV cache per token/layer:':<30} {format_bytes(kv_per_token_per_layer)}")
    print(f"{'KV cache per token (all layers):':<30} {format_bytes(kv_per_token_total)}")

    tokens_in_l2 = l2_bytes // kv_per_token_per_layer
    tokens_in_l3 = l3_bytes // kv_per_token_total

    print(f"\n{'Tokens of KV cache in L2:':<30} {tokens_in_l2} (per layer)")
    print(f"{'Tokens of KV cache in L3:':<30} {tokens_in_l3} (all layers)")

    # Roofline-style analysis
    print("\n" + "-" * 80)
    print("ROOFLINE ANALYSIS")
    print("-" * 80)

    # Compute capability (rough estimate)
    # AVX-512: 32 FLOPs/cycle per core (2x FMA units × 16 floats)
    # AVX2: 16 FLOPs/cycle per core
    flops_per_cycle = 32 if cpu.simd_width == 512 else 16 if cpu.simd_width == 256 else 8
    peak_gflops = cpu.cores * flops_per_cycle * cpu.base_freq_mhz / 1000

    print(f"{'Peak Compute:':<30} {peak_gflops:.1f} GFLOPS ({flops_per_cycle} FLOPs/cycle × {cpu.cores} cores × {cpu.base_freq_mhz/1000:.2f} GHz)")
    print(f"{'Peak Memory BW:':<30} {cpu.memory_bandwidth_gbps:.1f} GB/s")

    # Arithmetic intensity for GEMV (matrix-vector)
    # FLOPs = 2*M*N, Bytes = M*N*4 (weight) + N*4 (input) + M*4 (output) ≈ M*N*4
    # AI = 2*M*N / (M*N*4) = 0.5 FLOPs/byte
    gemv_ai = 0.5

    # Arithmetic intensity for GEMM (matrix-matrix)
    # FLOPs = 2*M*N*K, Bytes ≈ (M*K + K*N + M*N)*4
    # For large square: AI ≈ N/2 FLOPs/byte
    gemm_ai = model.hidden_size / 6  # Rough estimate

    ridge_point = peak_gflops / cpu.memory_bandwidth_gbps

    print(f"\n{'Ridge Point:':<30} {ridge_point:.1f} FLOPs/byte")
    print(f"{'GEMV Arithmetic Intensity:':<30} {gemv_ai:.2f} FLOPs/byte → {'MEMORY BOUND' if gemv_ai < ridge_point else 'COMPUTE BOUND'}")
    print(f"{'GEMM Arithmetic Intensity:':<30} {gemm_ai:.1f} FLOPs/byte → {'MEMORY BOUND' if gemm_ai < ridge_point else 'COMPUTE BOUND'}")

    # Decode analysis
    print("\n" + "-" * 80)
    print("DECODE (T=1) PERFORMANCE ESTIMATE")
    print("-" * 80)

    bytes_per_layer = layer.total_weight_bytes  # Dominated by weight streaming
    time_per_layer_ms = (bytes_per_layer / (cpu.memory_bandwidth_gbps * 1e9)) * 1000
    time_all_layers_ms = time_per_layer_ms * model.num_layers

    print(f"{'Weight bytes per layer:':<30} {format_bytes(bytes_per_layer)}")
    print(f"{'Time per layer (mem bound):':<30} {time_per_layer_ms:.2f} ms")
    print(f"{'Time for all layers:':<30} {time_all_layers_ms:.1f} ms")
    print(f"{'Theoretical tok/s:':<30} {1000 / time_all_layers_ms:.1f}")

    # Fusion impact
    print("\n" + "-" * 80)
    print("FUSION IMPACT ANALYSIS")
    print("-" * 80)

    unfused_activation_rw = layer.total_activation_bytes * 2 * 9  # 9 ops, read+write
    fused_activation_rw = layer.total_activation_bytes * 2 * 2    # 2 ops after fusion

    print(f"{'Unfused activation traffic:':<30} {format_bytes(unfused_activation_rw)} per layer")
    print(f"{'Fused activation traffic:':<30} {format_bytes(fused_activation_rw)} per layer")
    print(f"{'Activation traffic reduction:':<30} {unfused_activation_rw / fused_activation_rw:.1f}x")

    # But weights still dominate for decode
    total_unfused = bytes_per_layer + unfused_activation_rw
    total_fused = bytes_per_layer + fused_activation_rw

    print(f"\n{'Total unfused traffic/layer:':<30} {format_bytes(total_unfused)}")
    print(f"{'Total fused traffic/layer:':<30} {format_bytes(total_fused)}")
    print(f"{'Overall speedup from fusion:':<30} {total_unfused / total_fused:.2f}x")

    return {
        "l1_bytes": l1_bytes,
        "l2_bytes": l2_bytes,
        "l3_bytes": l3_bytes,
        "layer_weights": layer.total_weight_bytes,
        "layer_activations": layer.total_activation_bytes,
        "layers_in_l3": layers_in_l3,
        "fusion_speedup": total_unfused / total_fused,
    }


def print_recommendations(cpu: CPUInfo, model: ModelConfig, layer: LayerMemory, analysis: dict):
    """Print optimization recommendations."""

    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    l2_bytes = analysis["l2_bytes"]

    # Check if full attention block fits in L2
    attn_scratch = layer.ln_out_bytes + layer.q_bytes + layer.k_bytes + layer.v_bytes + layer.attn_out_bytes
    if attn_scratch < l2_bytes:
        print("\n✓ RECOMMENDED: Fuse entire attention block")
        print("  RMSNorm → Q/K/V Proj → RoPE → Attention → Out Proj")
        print(f"  Scratch needed: {format_bytes(attn_scratch)} (L2 has {format_bytes(l2_bytes)})")
    else:
        print("\n⚠ Attention block too large for L2 - tile by heads")
        heads_per_tile = l2_bytes // (layer.per_head_scratch_bytes * 2)
        print(f"  Process {heads_per_tile} heads at a time")

    # Check if MLP block fits
    mlp_scratch = layer.ln_out_bytes + layer.fc1_out_bytes + layer.swiglu_out_bytes
    if mlp_scratch < l2_bytes:
        print("\n✓ RECOMMENDED: Fuse entire MLP block")
        print("  RMSNorm → Gate+Up → SwiGLU → Down")
        print(f"  Scratch needed: {format_bytes(mlp_scratch)} (L2 has {format_bytes(l2_bytes)})")
    else:
        print("\n⚠ MLP block too large for L2 - tile intermediate dimension")
        # How many intermediate dims fit?
        tile_size = (l2_bytes - layer.ln_out_bytes) // (3 * model.dtype_bytes)  # gate, up, swiglu
        print(f"  Tile intermediate dim to {tile_size} (full is {model.intermediate_size})")

    # L3 recommendations
    if analysis["layers_in_l3"] >= model.num_layers:
        print("\n✓ Full model fits in L3 - weights will be hot after first pass")
    else:
        print(f"\n⚠ Only {analysis['layers_in_l3']}/{model.num_layers} layers fit in L3")
        print("  Consider: Layer pipelining to keep weights warm")

    # Thread recommendations
    print(f"\n{'Threading Strategy:'}")
    print(f"  • Use {cpu.cores} threads for data parallelism")
    print(f"  • Each thread handles a subset of heads in attention")
    print(f"  • Parallelize MLP across intermediate dimension tiles")

    # SIMD recommendations
    if cpu.simd_width == 512:
        print(f"\n✓ AVX-512 detected - use 512-bit intrinsics")
    elif cpu.simd_width == 256:
        print(f"\n• AVX2 detected - use 256-bit intrinsics")
        print(f"  Consider upgrading to AVX-512 CPU for 2x throughput")


def main():
    # Parse arguments
    config_path = None
    cpu_only = False

    for arg in sys.argv[1:]:
        if arg == "--cpu-only":
            cpu_only = True
        elif arg.endswith(".json"):
            config_path = arg

    # Detect CPU
    print("Detecting CPU configuration...")
    cpu = detect_cpu_info()

    if cpu_only:
        print(f"\nCPU: {cpu.model_name}")
        print(f"Cores: {cpu.cores}")
        print(f"L1D: {cpu.l1d.size_kb} KB")
        print(f"L2: {cpu.l2.size_kb} KB")
        print(f"L3: {cpu.l3.size_kb} KB")
        print(f"SIMD: {cpu.simd_width}-bit")
        return

    # Load model config
    if config_path and os.path.exists(config_path):
        print(f"Loading model config from {config_path}...")
        model = load_model_config(config_path)
    else:
        print("Using default Qwen2-0.5B configuration...")
        model = default_qwen2_config()

    # Analyze layer memory
    layer = analyze_layer_memory(model)

    # Run analysis
    analysis = analyze_cache_mapping(cpu, model, layer)

    # Print recommendations
    print_recommendations(cpu, model, layer, analysis)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
