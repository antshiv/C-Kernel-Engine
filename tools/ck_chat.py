#!/usr/bin/env python3
"""
C-Kernel-Engine Interactive Chat CLI

A llama.cpp-style interactive interface for SmolLM/LLaMA inference
using the C-Kernel-Engine.

Usage:
    python tools/ck_chat.py --model-dir ~/.cache/huggingface/hub/SmolLM-135M
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np

# Disable torch compile overhead
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from transformers import AutoTokenizer


VERSION = "0.1.0"
BANNER = r"""
   ____      _  __                    _   _____             _
  / ___|    | |/ /___ _ __ _ __   ___| | | ____|_ __   __ _(_)_ __   ___
 | |   _____| ' // _ \ '__| '_ \ / _ \ | |  _| | '_ \ / _` | | '_ \ / _ \
 | |__|_____| . \  __/ |  | | | |  __/ | | |___| | | | (_| | | | | |  __/
  \____|    |_|\_\___|_|  |_| |_|\___|_| |_____|_| |_|\__, |_|_| |_|\___|
                                                      |___/

  C-Kernel-Engine v{version}
  By Anthony Shivakumar

  Pure C Transformer Inference Engine
  ============================================================
"""

HELP_TEXT = """
Commands:
  /help     - Show this help message
  /clear    - Clear conversation history
  /exit     - Exit the chat
  /stats    - Show model statistics
  /temp N   - Set temperature (default: 0.7)
  /top_k N  - Set top-k sampling (default: 40)
  /top_p N  - Set top-p (nucleus) sampling (default: 0.9)
  /ctx N    - Set max context length

Type your prompt and press Enter to generate.
"""


def print_banner():
    print(BANNER.format(version=VERSION))


def load_config(model_dir):
    """Load model config from HuggingFace directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def detect_cc():
    """Detect best available C compiler."""
    if shutil.which("icx"):
        return "icx"
    if shutil.which("icc"):
        return "icc"
    return "gcc"


def detect_avx_flags():
    """Detect AVX capabilities."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":")[1].strip().split())
                    if "avx512f" in flags:
                        return ["-mavx512f"]
                    if "avx2" in flags:
                        return ["-mavx2"]
                    if "avx" in flags:
                        return ["-mavx"]
                    break
    except OSError:
        pass
    return []


def openmp_flag(cc):
    """Get OpenMP flag for compiler."""
    cc_lower = cc.lower()
    if "icx" in cc_lower or "icc" in cc_lower:
        return "-qopenmp"
    return "-fopenmp"


def build_model(project_root, config_path, build_dir):
    """Build the model binary from config."""
    gen_c = os.path.join(build_dir, "chat_model.c")
    kernel_manifest = gen_c + ".kernels"
    model_bin = os.path.join(build_dir, "chat_model")

    # Generate C code
    ir_demo = os.path.join(project_root, "build", "ck_ir_demo")
    if not os.path.exists(ir_demo):
        subprocess.check_call(["make", "build/ck_ir_demo"], cwd=project_root)

    subprocess.check_call([ir_demo, config_path, "--emit", gen_c], cwd=project_root)

    # Read kernel files
    with open(kernel_manifest, "r", encoding="utf-8") as f:
        kernels = f.read().split()

    # Compile
    cc = os.environ.get("CC", detect_cc())
    cflags = ["-O3", "-fPIC", openmp_flag(cc), "-Wall"] + detect_avx_flags() + ["-Iinclude"]

    cmd = [cc] + cflags + [gen_c] + kernels + ["-o", model_bin, "-lm"]
    subprocess.check_call(cmd, cwd=project_root)

    return model_bin


def prepare_weights(project_root, model_dir, config_path, build_dir, context_len):
    """Convert HuggingFace weights to bump format."""
    weights_bin = os.path.join(build_dir, "chat_weights.bin")

    subprocess.check_call([
        sys.executable,
        os.path.join(project_root, "scripts", "convert_hf_to_bump.py"),
        "--checkpoint", model_dir,
        "--output", weights_bin,
        "--config", config_path,
        "--context", str(context_len),
    ], cwd=project_root)

    return weights_bin


def sample_token(logits, temperature=0.7, top_k=40, top_p=0.9):
    """Sample next token from logits with temperature, top-k, and top-p."""
    if temperature <= 0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices = np.argpartition(-logits, top_k)[:top_k]
        top_logits = logits[indices]
        # Softmax
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / probs.sum()
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p) + 1
            sorted_idx = sorted_idx[:cutoff]
            probs_filtered = probs[sorted_idx]
            probs_filtered = probs_filtered / probs_filtered.sum()
            chosen = np.random.choice(sorted_idx, p=probs_filtered)
            return int(indices[chosen])
        else:
            chosen = np.random.choice(len(probs), p=probs)
            return int(indices[chosen])
    else:
        # Full softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))


class CKernelChat:
    def __init__(self, model_dir, build_dir, max_tokens=128, context_len=512):
        self.model_dir = model_dir
        self.build_dir = build_dir
        self.max_tokens = max_tokens
        self.context_len = context_len
        self.temperature = 0.7
        self.top_k = 40
        self.top_p = 0.9

        self.project_root = str(Path(__file__).parent.parent)

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # Load config
        print("Loading config...")
        self.cfg = load_config(model_dir)
        self.vocab_size = self.cfg.get("vocab_size", 32000)

        # Override context if needed
        self.cfg["max_position_embeddings"] = context_len
        self.cfg["context_window"] = context_len

        # Write runtime config
        os.makedirs(build_dir, exist_ok=True)
        self.config_path = os.path.join(build_dir, "chat.config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

        # Build model
        print("Building model binary...")
        self.model_bin = build_model(self.project_root, self.config_path, build_dir)

        # Prepare weights
        print("Converting weights...")
        self.weights_bin = prepare_weights(
            self.project_root, model_dir, self.config_path, build_dir, context_len
        )

        self.conversation = []
        print("Ready!\n")

    def encode(self, text):
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def run_forward(self, token_ids):
        """Run forward pass and get logits."""
        T = len(token_ids)
        if T > self.context_len:
            token_ids = token_ids[-self.context_len:]
            T = self.context_len

        # Pad if needed
        padded = list(token_ids)
        if len(padded) < self.context_len:
            pad_id = self.tokenizer.eos_token_id or 0
            padded.extend([pad_id] * (self.context_len - len(padded)))

        # Write tokens
        tokens_bin = os.path.join(self.build_dir, "chat_tokens.bin")
        logits_bin = os.path.join(self.build_dir, "chat_logits.bin")

        np.array(padded, dtype=np.int32).tofile(tokens_bin)

        # Run model
        cmd = [
            self.model_bin,
            "--model-weights", self.weights_bin,
            "--tokens", tokens_bin,
            "--out-logits", logits_bin,
            "--ctx", str(self.context_len),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Model forward pass failed (exit code {e.returncode})")
            if e.stderr:
                print(f"[STDERR] {e.stderr[:500]}")
            print("\n[HINT] Run diagnostics with: python scripts/smollm_forward_parity.py")
            print("[HINT] Check kernel tests with: make test")
            raise RuntimeError(f"Model execution failed: {e.returncode}") from e

        # Read logits
        if not os.path.exists(logits_bin):
            raise RuntimeError(f"Output logits file not created: {logits_bin}")

        logits = np.fromfile(logits_bin, dtype=np.float32)
        expected_size = self.context_len * self.vocab_size
        if logits.size != expected_size:
            raise RuntimeError(
                f"Logits size mismatch: got {logits.size}, expected {expected_size} "
                f"(ctx={self.context_len}, vocab={self.vocab_size})"
            )
        logits = logits.reshape(self.context_len, self.vocab_size)

        # Return logits for the last actual token position
        return logits[T - 1]

    def generate(self, prompt, max_new_tokens=None, stream=True):
        """Generate text from prompt."""
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens

        token_ids = self.encode(prompt)
        generated = []

        for i in range(max_new_tokens):
            logits = self.run_forward(token_ids + generated)
            next_token = sample_token(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

            if next_token == self.tokenizer.eos_token_id:
                break

            generated.append(next_token)

            if stream:
                # Print token incrementally
                token_text = self.decode([next_token])
                print(token_text, end="", flush=True)

        if stream:
            print()  # Newline at end

        return self.decode(generated)

    def chat(self, user_input):
        """Process user input and generate response."""
        # Build full prompt with conversation history
        full_prompt = ""
        for role, text in self.conversation:
            if role == "user":
                full_prompt += f"User: {text}\n"
            else:
                full_prompt += f"Assistant: {text}\n"

        full_prompt += f"User: {user_input}\nAssistant:"

        # Generate response
        print("\nAssistant: ", end="", flush=True)
        response = self.generate(full_prompt)

        # Update conversation
        self.conversation.append(("user", user_input))
        self.conversation.append(("assistant", response))

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.conversation = []
        print("Conversation history cleared.")

    def show_stats(self):
        """Show model statistics."""
        print(f"\nModel Statistics:")
        print(f"  Model directory: {self.model_dir}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Hidden size: {self.cfg.get('hidden_size', 'N/A')}")
        print(f"  Num layers: {self.cfg.get('num_hidden_layers', 'N/A')}")
        print(f"  Num heads: {self.cfg.get('num_attention_heads', 'N/A')}")
        print(f"  Context length: {self.context_len}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Top-p: {self.top_p}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine Interactive Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_TEXT
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "SmolLM-135M"),
        help="HuggingFace model directory"
    )
    parser.add_argument(
        "--build-dir",
        default="build/chat",
        help="Build output directory"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=256,
        help="Context window size"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show startup banner"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (non-interactive mode)"
    )
    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    # Check model exists
    if not os.path.exists(os.path.join(args.model_dir, "config.json")):
        print(f"Error: Model not found at {args.model_dir}")
        print("Download with: python scripts/download_smollm.py --outdir " + args.model_dir)
        return 1

    try:
        chat = CKernelChat(
            model_dir=args.model_dir,
            build_dir=args.build_dir,
            max_tokens=args.max_tokens,
            context_len=args.context
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 1

    # Single prompt mode
    if args.prompt:
        chat.generate(args.prompt)
        return 0

    # Interactive mode
    print("Type /help for commands, or enter your prompt.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/exit" or cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/clear":
                chat.clear_history()
            elif cmd == "/stats":
                chat.show_stats()
            elif cmd == "/temp" and len(parts) > 1:
                try:
                    chat.temperature = float(parts[1])
                    print(f"Temperature set to {chat.temperature}")
                except ValueError:
                    print("Invalid temperature value")
            elif cmd == "/top_k" and len(parts) > 1:
                try:
                    chat.top_k = int(parts[1])
                    print(f"Top-k set to {chat.top_k}")
                except ValueError:
                    print("Invalid top_k value")
            elif cmd == "/top_p" and len(parts) > 1:
                try:
                    chat.top_p = float(parts[1])
                    print(f"Top-p set to {chat.top_p}")
                except ValueError:
                    print("Invalid top_p value")
            elif cmd == "/ctx" and len(parts) > 1:
                try:
                    new_ctx = int(parts[1])
                    print(f"Rebuilding with context length {new_ctx}...")
                    chat = CKernelChat(
                        model_dir=args.model_dir,
                        build_dir=args.build_dir,
                        max_tokens=args.max_tokens,
                        context_len=new_ctx
                    )
                except ValueError:
                    print("Invalid context value")
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Generate response
        try:
            chat.chat(user_input)
        except Exception as e:
            print(f"\nError generating response: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
