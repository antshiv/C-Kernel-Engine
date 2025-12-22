#!/usr/bin/env python3
"""
Per-layer output diffs for SmolLM stack vs PyTorch.

Fixed to handle final layer correctly - hidden_states[num_layers] is post-final-norm,
so we compute raw layer output for comparison on the last layer.
"""
import argparse
import json
import os
import sys
import ctypes

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
from transformers import LlamaForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNITTEST_DIR = os.path.join(_ROOT, "unittest")
if _UNITTEST_DIR not in sys.path:
    sys.path.insert(0, _UNITTEST_DIR)
from lib_loader import load_lib


class _DummyDynamo:
    @staticmethod
    def is_compiling() -> bool:
        return False


torch.__dict__["_dynamo"] = _DummyDynamo()


class CKLayerForwardParams(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_int),
        ("embed_dim", ctypes.c_int),
        ("aligned_embed_dim", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("aligned_head_dim", ctypes.c_int),
        ("aligned_context_window", ctypes.c_int),
        ("intermediate_dim", ctypes.c_int),
        ("aligned_intermediate_dim", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("rope_pos_offset", ctypes.c_int),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("ln1_gamma", ctypes.POINTER(ctypes.c_float)),
        ("ln2_gamma", ctypes.POINTER(ctypes.c_float)),
        ("rope_cos", ctypes.POINTER(ctypes.c_float)),
        ("rope_sin", ctypes.POINTER(ctypes.c_float)),
        ("wq", ctypes.POINTER(ctypes.c_float)),
        ("bq", ctypes.POINTER(ctypes.c_float)),
        ("wk", ctypes.POINTER(ctypes.c_float)),
        ("bk", ctypes.POINTER(ctypes.c_float)),
        ("wv", ctypes.POINTER(ctypes.c_float)),
        ("bv", ctypes.POINTER(ctypes.c_float)),
        ("wo", ctypes.POINTER(ctypes.c_float)),
        ("bo", ctypes.POINTER(ctypes.c_float)),
        ("w1", ctypes.POINTER(ctypes.c_float)),
        ("b1", ctypes.POINTER(ctypes.c_float)),
        ("w2", ctypes.POINTER(ctypes.c_float)),
        ("b2", ctypes.POINTER(ctypes.c_float)),
        ("ln1_out", ctypes.POINTER(ctypes.c_float)),
        ("ln1_rstd", ctypes.POINTER(ctypes.c_float)),
        ("q", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.POINTER(ctypes.c_float)),
        ("v", ctypes.POINTER(ctypes.c_float)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("attn_out", ctypes.POINTER(ctypes.c_float)),
        ("proj_tmp", ctypes.POINTER(ctypes.c_float)),
        ("proj_scratch", ctypes.POINTER(ctypes.c_float)),
        ("residual1", ctypes.POINTER(ctypes.c_float)),
        ("ln2_out", ctypes.POINTER(ctypes.c_float)),
        ("ln2_rstd", ctypes.POINTER(ctypes.c_float)),
        ("fc1_out", ctypes.POINTER(ctypes.c_float)),
        ("swiglu_out", ctypes.POINTER(ctypes.c_float)),
        ("mlp_out", ctypes.POINTER(ctypes.c_float)),
        ("output", ctypes.POINTER(ctypes.c_float)),
    ]


def align_up(n, a):
    return (n + a - 1) // a * a


def aligned_empty(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    arr = buf[offset:offset + nbytes].view(dtype).reshape(shape)
    return arr


def ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def rope_cache(head_dim, max_seq_len, base=10000.0):
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache, sin_cache


def prepare_tokens(tokenizer, text, context_len):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    if len(ids) < context_len:
        pad_id = tokenizer.eos_token_id or 0
        ids.extend([pad_id] * (context_len - len(ids)))
    ids = ids[:context_len]
    return np.array(ids, dtype=np.int32)


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


def pack_qkv_rows(weight, num_heads, head_dim, aligned_head_dim, aligned_embed_dim):
    out_rows = num_heads * aligned_head_dim
    buf = np.zeros((out_rows, aligned_embed_dim), dtype=np.float32)
    w = weight.astype(np.float32)
    for h in range(num_heads):
        row_base = h * aligned_head_dim
        row_end = row_base + head_dim
        src_base = h * head_dim
        buf[row_base:row_end, :w.shape[1]] = w[src_base:src_base + head_dim, :]
    return buf


def pack_wo_cols(weight, num_heads, head_dim, aligned_head_dim, aligned_embed_dim):
    packed = aligned_empty((num_heads, aligned_embed_dim, aligned_head_dim))
    packed.fill(0.0)
    for h in range(num_heads):
        cols = weight[:, h * head_dim:(h + 1) * head_dim]
        packed[h, :cols.shape[0], :head_dim] = cols
    return packed


def pack_w1_gate_up(gate, up, aligned_intermediate_dim, aligned_embed_dim):
    packed = aligned_empty((2 * aligned_intermediate_dim, aligned_embed_dim))
    packed.fill(0.0)
    packed[: gate.shape[0], : gate.shape[1]] = gate
    packed[aligned_intermediate_dim:aligned_intermediate_dim + up.shape[0], : up.shape[1]] = up
    return packed


def pack_w2(down, aligned_embed_dim, aligned_intermediate_dim):
    packed = aligned_empty((aligned_embed_dim, aligned_intermediate_dim))
    packed.fill(0.0)
    packed[: down.shape[0], : down.shape[1]] = down
    return packed


def run_single_layer_pytorch(model, layer_idx, x_input, rope_theta=10000.0):
    """
    Run a single transformer layer in PyTorch and return the raw output
    (before any final norm). This gives us the true layer output for comparison.
    """
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        # We need to run through the layer manually to get raw output
        # LlamaDecoderLayer forward: residual + attn + residual + mlp
        hidden = x_input

        # Self-attention block
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        # Get attention output - need position_ids and attention_mask
        batch_size, seq_len, hidden_size = hidden.shape
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden.device),
            diagonal=1
        )

        # Compute RoPE position embeddings (cos, sin)
        # New transformers API requires position_embeddings tuple
        num_heads = getattr(layer.self_attn, 'num_heads', None) or layer.self_attn.config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Use the model's rotary_emb if available, otherwise compute manually
        if hasattr(model.model, 'rotary_emb'):
            cos, sin = model.model.rotary_emb(hidden, position_ids)
        elif hasattr(layer.self_attn, 'rotary_emb'):
            cos, sin = layer.self_attn.rotary_emb(hidden, position_ids)
        else:
            # Manual RoPE computation
            half_dim = head_dim // 2
            freqs = 1.0 / (rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=hidden.device) * 2.0 / head_dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=hidden.device)
            angles = torch.outer(t, freqs)
            cos = torch.cos(angles).unsqueeze(0)  # [1, seq_len, half_dim]
            sin = torch.sin(angles).unsqueeze(0)

        position_embeddings = (cos, sin)

        attn_result = layer.self_attn(
            hidden_states=hidden,
            attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        # Handle both old (3 values) and new (2 values) API
        attn_output = attn_result[0]
        hidden = residual + attn_output

        # MLP block
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    return hidden


def main():
    parser = argparse.ArgumentParser(description="Per-layer output diffs for SmolLM stack vs PyTorch")
    parser.add_argument("--config", default=None, help="Model config JSON (default: uses model-dir/config.json)")
    parser.add_argument("--model-dir", required=True, help="HF model directory")
    parser.add_argument("--context", type=int, default=5, help="Context length")
    parser.add_argument("--text", default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-layers", type=int, default=None, help="Limit layers to check")
    parser.add_argument("--tol", type=float, default=1e-3, help="Max allowed absolute diff")
    parser.add_argument("--rtol", type=float, default=None, help="Max allowed relative diff (overrides --tol if set)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    args = parser.parse_args()

    # Use model's config.json by default
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(args.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise SystemExit(f"Config not found: {config_path}. Specify --config or ensure model-dir has config.json")

    cfg = override_context(load_cfg(config_path), args.context)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokens = prepare_tokens(tokenizer, args.text, args.context)

    model = LlamaForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Expected a Llama-style model with model.layers")

    with torch.no_grad():
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        outputs = model.model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    num_layers = len(model.model.layers)
    max_layers = args.max_layers if args.max_layers is not None else num_layers
    max_layers = min(max_layers, num_layers)

    # Debug: show hidden_states structure
    if args.verbose:
        print(f"Config: {config_path}")
        print(f"Number of hidden states: {len(hidden_states)}")
        print(f"Number of layers: {num_layers}")
        print(f"hidden_states[0] shape: {hidden_states[0].shape} (embedding output)")
        print(f"hidden_states[{num_layers}] shape: {hidden_states[num_layers].shape} (final, post-norm)")
        if hasattr(model.model, 'norm'):
            print(f"Final norm exists: model.model.norm = {type(model.model.norm)}")

    D = int(cfg["hidden_size"])
    T = int(args.context)
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = D // num_heads
    intermediate_dim = int(cfg["intermediate_size"])
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(cfg.get("rope_theta", 0.0))
    use_rope = rope_theta > 0.0

    aligned_embed_dim = align_up(D, 16)
    aligned_head_dim = align_up(head_dim, 16)
    aligned_intermediate_dim = align_up(intermediate_dim, 16)
    aligned_context_window = align_up(T, 16)

    rope_cos = None
    rope_sin = None
    if use_rope:
        cos_cache, sin_cache = rope_cache(head_dim, T, rope_theta)
        rope_cos = cos_cache.numpy().astype(np.float32)
        rope_sin = sin_cache.numpy().astype(np.float32)

    lib = load_lib("libckernel_engine.so")
    lib.ck_layer_forward_rmsnorm_swiglu.argtypes = [ctypes.POINTER(CKLayerForwardParams)]
    lib.ck_layer_forward_rmsnorm_swiglu.restype = None

    all_passed = True

    for layer_idx in range(max_layers):
        layer = model.model.layers[layer_idx]
        x_ref = hidden_states[layer_idx][0].cpu().float().numpy()

        # For the last layer, hidden_states[layer_idx + 1] is POST final norm
        # We need to compute the raw layer output ourselves
        is_last_layer = (layer_idx == num_layers - 1)

        if is_last_layer and (layer_idx + 1) < len(hidden_states):
            # Compute raw layer output by running the layer manually
            x_tensor = hidden_states[layer_idx].float()
            y_raw = run_single_layer_pytorch(model, layer_idx, x_tensor, rope_theta)
            y_ref = y_raw[0].cpu().numpy()

            if args.verbose:
                # Also show what the post-norm value looks like
                y_post_norm = hidden_states[layer_idx + 1][0].cpu().float().numpy()
                print(f"Layer {layer_idx} (FINAL): comparing raw output, not post-norm")
                print(f"  Post-norm would have diff range: [{y_post_norm.min():.4f}, {y_post_norm.max():.4f}]")
        else:
            y_ref = hidden_states[layer_idx + 1][0].cpu().float().numpy()

        x = aligned_empty((T, aligned_embed_dim))
        x.fill(0.0)
        x[:, :D] = x_ref

        ln1_gamma = aligned_empty((aligned_embed_dim,))
        ln2_gamma = aligned_empty((aligned_embed_dim,))
        ln1_gamma.fill(0.0)
        ln2_gamma.fill(0.0)
        ln1_gamma[:D] = layer.input_layernorm.weight.detach().cpu().float().numpy()
        ln2_gamma[:D] = layer.post_attention_layernorm.weight.detach().cpu().float().numpy()

        q_proj = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()
        k_proj = layer.self_attn.k_proj.weight.detach().cpu().float().numpy()
        v_proj = layer.self_attn.v_proj.weight.detach().cpu().float().numpy()
        o_proj = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()
        gate_proj = layer.mlp.gate_proj.weight.detach().cpu().float().numpy()
        up_proj = layer.mlp.up_proj.weight.detach().cpu().float().numpy()
        down_proj = layer.mlp.down_proj.weight.detach().cpu().float().numpy()

        wq = pack_qkv_rows(q_proj, num_heads, head_dim, aligned_head_dim, aligned_embed_dim)
        wk = pack_qkv_rows(k_proj, num_kv_heads, head_dim, aligned_head_dim, aligned_embed_dim)
        wv = pack_qkv_rows(v_proj, num_kv_heads, head_dim, aligned_head_dim, aligned_embed_dim)
        wo = pack_wo_cols(o_proj, num_heads, head_dim, aligned_head_dim, aligned_embed_dim)
        w1 = pack_w1_gate_up(gate_proj, up_proj, aligned_intermediate_dim, aligned_embed_dim)
        w2 = pack_w2(down_proj, aligned_embed_dim, aligned_intermediate_dim)

        bq = aligned_empty((num_heads, aligned_head_dim))
        bk = aligned_empty((num_kv_heads, aligned_head_dim))
        bv = aligned_empty((num_kv_heads, aligned_head_dim))
        bo = aligned_empty((aligned_embed_dim,))
        b1 = aligned_empty((2 * aligned_intermediate_dim,))
        b2 = aligned_empty((aligned_embed_dim,))
        bq.fill(0.0)
        bk.fill(0.0)
        bv.fill(0.0)
        bo.fill(0.0)
        b1.fill(0.0)
        b2.fill(0.0)

        ln1_out = aligned_empty((T, aligned_embed_dim))
        ln1_rstd = aligned_empty((T,))
        q = aligned_empty((num_heads, T, aligned_head_dim))
        k = aligned_empty((num_kv_heads, T, aligned_head_dim))
        v = aligned_empty((num_kv_heads, T, aligned_head_dim))
        scores = aligned_empty((num_heads, aligned_context_window, aligned_context_window))
        attn_out = aligned_empty((num_heads, T, aligned_head_dim))
        proj_tmp = aligned_empty((T, aligned_embed_dim))
        proj_scratch = aligned_empty((T, aligned_embed_dim))
        residual1 = aligned_empty((T, aligned_embed_dim))
        ln2_out = aligned_empty((T, aligned_embed_dim))
        ln2_rstd = aligned_empty((T,))
        fc1_out = aligned_empty((T, 2 * aligned_intermediate_dim))
        swiglu_out = aligned_empty((T, aligned_intermediate_dim))
        mlp_out = aligned_empty((T, aligned_embed_dim))
        output = aligned_empty((T, aligned_embed_dim))

        params = CKLayerForwardParams(
            tokens=T,
            embed_dim=D,
            aligned_embed_dim=aligned_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            aligned_head_dim=aligned_head_dim,
            aligned_context_window=aligned_context_window,
            intermediate_dim=intermediate_dim,
            aligned_intermediate_dim=aligned_intermediate_dim,
            eps=eps,
            rope_pos_offset=0,
            input=ptr(x),
            ln1_gamma=ptr(ln1_gamma),
            ln2_gamma=ptr(ln2_gamma),
            rope_cos=ptr(rope_cos) if use_rope else None,
            rope_sin=ptr(rope_sin) if use_rope else None,
            wq=ptr(wq),
            bq=ptr(bq),
            wk=ptr(wk),
            bk=ptr(bk),
            wv=ptr(wv),
            bv=ptr(bv),
            wo=ptr(wo),
            bo=ptr(bo),
            w1=ptr(w1),
            b1=ptr(b1),
            w2=ptr(w2),
            b2=ptr(b2),
            ln1_out=ptr(ln1_out),
            ln1_rstd=ptr(ln1_rstd),
            q=ptr(q),
            k=ptr(k),
            v=ptr(v),
            scores=ptr(scores),
            attn_out=ptr(attn_out),
            proj_tmp=ptr(proj_tmp),
            proj_scratch=ptr(proj_scratch),
            residual1=ptr(residual1),
            ln2_out=ptr(ln2_out),
            ln2_rstd=ptr(ln2_rstd),
            fc1_out=ptr(fc1_out),
            swiglu_out=ptr(swiglu_out),
            mlp_out=ptr(mlp_out),
            output=ptr(output),
        )

        lib.ck_layer_forward_rmsnorm_swiglu(ctypes.byref(params))

        out_c = output[:, :D].astype(np.float32)
        abs_diff = np.max(np.abs(out_c - y_ref))

        # Compute relative diff (avoid division by zero)
        max_magnitude = max(np.max(np.abs(out_c)), np.max(np.abs(y_ref)), 1e-10)
        rel_diff = abs_diff / max_magnitude

        # Use relative tolerance if specified, otherwise absolute
        if args.rtol is not None:
            passed = rel_diff <= args.rtol
            tol_str = f"rtol={args.rtol:g}"
            diff_display = f"abs={abs_diff:.3e} rel={rel_diff:.3e}"
        else:
            passed = abs_diff <= args.tol
            tol_str = f"tol={args.tol:g}"
            diff_display = f"max_diff={abs_diff:.3e}"

        status = "OK" if passed else "FAIL"
        final_marker = " (FINAL)" if is_last_layer else ""
        print(f"Layer {layer_idx:02d} {diff_display} [{status}]{final_marker}")

        if not passed:
            all_passed = False
            if args.verbose:
                # Show where the max diff occurs
                diff_matrix = np.abs(out_c - y_ref)
                max_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
                print(f"  Max diff at position {max_idx}")
                print(f"  C output: {out_c[max_idx]:.6f}")
                print(f"  PyTorch:  {y_ref[max_idx]:.6f}")
                print(f"  Relative error: {abs_diff / abs(y_ref[max_idx]):.3e}")
                print(f"  Output range C: [{out_c.min():.4f}, {out_c.max():.4f}]")
                print(f"  Output range PT: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    tol_msg = f"rtol={args.rtol:g}" if args.rtol else f"tol={args.tol:g}"
    if all_passed:
        print(f"\nAll {max_layers} layers passed ({tol_msg})")
    else:
        raise SystemExit(f"\nSome layers exceeded tolerance ({tol_msg})")


if __name__ == "__main__":
    main()
