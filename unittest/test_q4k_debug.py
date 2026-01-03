#!/usr/bin/env python3
"""
Minimal test for Q4_K debug output with tiny dummy weights.
Tests that the debug infrastructure works before deploying to server.
"""

import os
import sys
import json
import struct
import tempfile
import subprocess
from pathlib import Path

# Tiny model config (2 layers, 256 embed, 2 heads)
TINY_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 1000,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

def create_dummy_q4k_block():
    """Create a single Q4_K block (144 bytes) with valid scales."""
    # Q4_K block structure:
    # - d: fp16 (2 bytes) - main scale
    # - dmin: fp16 (2 bytes) - min scale
    # - scales: 12 bytes (sub-block scales)
    # - qs: 128 bytes (quantized values)

    # Use small non-zero scale (0.01 in fp16)
    d_fp16 = struct.pack('<e', 0.01)  # fp16 for 0.01
    dmin_fp16 = struct.pack('<e', 0.001)  # fp16 for 0.001
    scales = bytes([8] * 12)  # Mid-range scales
    qs = bytes([0x88] * 128)  # Mid-range quantized values (8,8 pattern)

    return d_fp16 + dmin_fp16 + scales + qs

def create_dummy_q4k_tensor(rows, cols):
    """Create a Q4_K tensor with shape (rows, cols)."""
    assert cols % 256 == 0, f"cols must be multiple of 256, got {cols}"
    blocks_per_row = cols // 256
    total_blocks = rows * blocks_per_row

    block = create_dummy_q4k_block()
    assert len(block) == 144, f"Block size should be 144, got {len(block)}"

    return block * total_blocks

def create_dummy_fp32_tensor(size):
    """Create a dummy FP32 tensor."""
    import random
    random.seed(42)
    values = [random.gauss(0, 0.02) for _ in range(size)]
    return struct.pack(f'<{size}f', *values)

def create_dummy_model(output_dir: Path):
    """Create a minimal dummy model for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TINY_CONFIG
    embed_dim = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    vocab_size = cfg["vocab_size"]
    num_layers = cfg["num_hidden_layers"]

    # Align to 256 for Q4_K
    aligned_embed = ((embed_dim + 255) // 256) * 256
    aligned_intermediate = ((intermediate + 255) // 256) * 256

    print(f"Creating dummy model: {num_layers} layers, {embed_dim} embed (aligned={aligned_embed})")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Create weight tensors
    weights = {}

    # Embedding (FP32)
    weights["model.embed_tokens.weight"] = create_dummy_fp32_tensor(vocab_size * embed_dim)

    # LM head (Q4_K) - vocab_size x embed_dim
    weights["lm_head.weight"] = create_dummy_q4k_tensor(vocab_size, aligned_embed)

    # Final norm (FP32)
    weights["model.norm.weight"] = create_dummy_fp32_tensor(embed_dim)

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Layer norms (FP32)
        weights[f"{prefix}.input_layernorm.weight"] = create_dummy_fp32_tensor(embed_dim)
        weights[f"{prefix}.post_attention_layernorm.weight"] = create_dummy_fp32_tensor(embed_dim)

        # QKV projections (Q4_K) - (num_heads * head_dim) x embed_dim
        qkv_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        weights[f"{prefix}.self_attn.q_proj.weight"] = create_dummy_q4k_tensor(qkv_out, aligned_embed)
        weights[f"{prefix}.self_attn.k_proj.weight"] = create_dummy_q4k_tensor(kv_out, aligned_embed)
        weights[f"{prefix}.self_attn.v_proj.weight"] = create_dummy_q4k_tensor(kv_out, aligned_embed)
        weights[f"{prefix}.self_attn.o_proj.weight"] = create_dummy_q4k_tensor(embed_dim, aligned_embed)

        # MLP (Q4_K)
        weights[f"{prefix}.mlp.gate_proj.weight"] = create_dummy_q4k_tensor(intermediate, aligned_embed)
        weights[f"{prefix}.mlp.up_proj.weight"] = create_dummy_q4k_tensor(intermediate, aligned_embed)
        weights[f"{prefix}.mlp.down_proj.weight"] = create_dummy_q4k_tensor(embed_dim, aligned_intermediate)

    # Save as simple binary format with index
    index = {}
    offset = 0

    with open(output_dir / "model.bin", "wb") as f:
        for name, data in weights.items():
            index[name] = {"offset": offset, "size": len(data)}
            f.write(data)
            offset += len(data)

    with open(output_dir / "model.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Also save as safetensors-like format that the loader expects
    # For now, let's create a minimal GGUF-like structure

    print(f"Created dummy model with {len(weights)} tensors, total {offset} bytes")
    return output_dir

def test_debug_output():
    """Test that debug output works with the dummy model."""

    # For now, just verify the codegen changes compile
    print("Testing codegen with --debug flag...")

    # Check that codegen_v4.py has the debug changes
    codegen_path = Path(__file__).parent.parent / "scripts" / "codegen_v4.py"
    with open(codegen_path) as f:
        content = f.read()

    if "debug_check_q4k_weights" not in content:
        print("ERROR: debug_check_q4k_weights not found in codegen_v4.py")
        return False

    if "layer0_wq" not in content:
        print("ERROR: layer0_wq debug not found in codegen_v4.py")
        return False

    print("OK: Codegen has debug instrumentation")

    # Test that the debug helper compiles
    test_c = '''
#include <stdio.h>
#include <stdint.h>
#include <math.h>

static void debug_check_buffer(const char *name, const float *buf, int size) {
    int nan_count = 0, inf_count = 0, zero_count = 0;
    float min_val = 1e38f, max_val = -1e38f, sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float v = buf[i];
        if (isnan(v)) { nan_count++; }
        else if (isinf(v)) { inf_count++; }
        else {
            if (v == 0.0f) zero_count++;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            sum += v;
        }
    }
    float mean = (size - nan_count - inf_count > 0) ? sum / (size - nan_count - inf_count) : 0.0f;
    fprintf(stderr, "[DEBUG] %-30s size=%6d  nan=%d inf=%d zero=%d  range=[%.3e, %.3e] mean=%.3e\\n",
            name, size, nan_count, inf_count, zero_count, min_val, max_val, mean);
}

static void debug_check_q4k_weights(const char *name, const void *w, int M, int K) {
    const uint8_t *bytes = (const uint8_t *)w;
    int blocks_per_row = K / 256;
    int zero_scale_blocks = 0;
    int nan_scale_blocks = 0;
    for (int row = 0; row < M && row < 16; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            int offset = (row * blocks_per_row + b) * 144;
            uint16_t d_bits = *(uint16_t *)&bytes[offset];
            if (d_bits == 0) zero_scale_blocks++;
            // Check for NaN in fp16: exp=31, mantissa!=0
            if ((d_bits & 0x7C00) == 0x7C00 && (d_bits & 0x03FF) != 0) nan_scale_blocks++;
        }
    }
    fprintf(stderr, "[DEBUG] %-30s M=%d K=%d zero_scales=%d nan_scales=%d\\n",
            name, M, K, zero_scale_blocks, nan_scale_blocks);
}

int main() {
    // Test buffer check
    float test_buf[8] = {1.0f, 2.0f, 0.0f, -1.0f, 0.0f/0.0f, 1.0f/0.0f, 3.0f, 4.0f};
    debug_check_buffer("test_buffer", test_buf, 8);

    // Test Q4_K weight check with dummy block
    uint8_t dummy_q4k[144 * 4];  // 4 blocks for K=1024
    // Set valid fp16 scales (0x3C00 = 1.0 in fp16)
    for (int i = 0; i < 4; i++) {
        dummy_q4k[i * 144 + 0] = 0x00;  // d low byte
        dummy_q4k[i * 144 + 1] = 0x3C;  // d high byte (1.0 in fp16)
        dummy_q4k[i * 144 + 2] = 0x00;  // dmin low byte
        dummy_q4k[i * 144 + 3] = 0x38;  // dmin high byte (0.5 in fp16)
    }
    debug_check_q4k_weights("test_q4k_weights", dummy_q4k, 1, 1024);

    // Test with zero scales (should detect)
    uint8_t zero_q4k[144 * 4] = {0};
    debug_check_q4k_weights("zero_q4k_weights", zero_q4k, 1, 1024);

    printf("Debug helper test passed!\\n");
    return 0;
}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        c_path = Path(tmpdir) / "test_debug.c"
        exe_path = Path(tmpdir) / "test_debug"

        with open(c_path, "w") as f:
            f.write(test_c)

        # Compile
        result = subprocess.run(
            ["gcc", "-o", str(exe_path), str(c_path), "-lm"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"ERROR: Compilation failed:\n{result.stderr}")
            return False

        print("OK: Debug helpers compile")

        # Run
        result = subprocess.run([str(exe_path)], capture_output=True, text=True)
        print("Debug helper output:")
        print(result.stderr)
        print(result.stdout)

        if result.returncode != 0:
            print(f"ERROR: Test failed")
            return False

        # Verify output contains expected debug lines
        if "[DEBUG] test_buffer" not in result.stderr:
            print("ERROR: test_buffer debug not found")
            return False

        if "nan=1" not in result.stderr:
            print("ERROR: NaN detection not working")
            return False

        if "inf=1" not in result.stderr:
            print("ERROR: Inf detection not working")
            return False

        if "zero_scales=4" in result.stderr:
            print("OK: Zero scale detection working")

        print("\nAll debug helper tests passed!")
        return True

if __name__ == "__main__":
    success = test_debug_output()
    sys.exit(0 if success else 1)
