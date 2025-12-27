#!/usr/bin/env python3
"""
C-Kernel-Engine Chat Interface

Uses the HuggingFace tokenizer and calls the compiled C model library.
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

# Try to import tokenizers
try:
    from tokenizers import Tokenizer
except ImportError:
    print("Error: tokenizers package not found. Install with: pip install tokenizers")
    sys.exit(1)


class CKModel:
    """Wrapper for the C model library."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.lib = None
        self.tokenizer = None
        self.vocab_size = 0
        self.context_window = 0
        self.has_kv_decode = False

    def load(self) -> bool:
        """Load model library and tokenizer."""
        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            print(f"Error: Tokenizer not found: {tokenizer_path}")
            return False
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Load C library
        lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            print(f"Error: Model library not found: {lib_path}")
            return False

        self.lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int

        self.lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int

        self.lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_forward.restype = ctypes.c_int

        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.ck_model_get_vocab_size.argtypes = []
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int

        self.lib.ck_model_get_context_window.argtypes = []
        self.lib.ck_model_get_context_window.restype = ctypes.c_int

        self.lib.ck_model_get_active_tokens.argtypes = []
        self.lib.ck_model_get_active_tokens.restype = ctypes.c_int

        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        # Optional KV-cache decode API (newer generated runtimes).
        try:
            self.lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
            self.lib.ck_model_kv_cache_enable.restype = ctypes.c_int
            self.lib.ck_model_kv_cache_reset.argtypes = []
            self.lib.ck_model_kv_cache_reset.restype = None
            self.lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
            self.lib.ck_model_decode.restype = ctypes.c_int
            self.has_kv_decode = True
        except AttributeError:
            self.has_kv_decode = False

        # Initialize model
        weights_path = self.model_dir / "weights.bump"
        if not weights_path.exists():
            print(f"Error: Weights not found: {weights_path}")
            return False

        ret = self.lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"Error: Failed to initialize model (code {ret})")
            return False

        self.vocab_size = self.lib.ck_model_get_vocab_size()
        self.context_window = self.lib.ck_model_get_context_window()

        return True

    def kv_cache_enable(self, capacity: int | None = None) -> bool:
        if not self.has_kv_decode:
            return False
        if capacity is None:
            capacity = self.context_window
        ret = self.lib.ck_model_kv_cache_enable(int(capacity))
        return ret == 0

    def kv_cache_reset(self):
        if self.has_kv_decode:
            self.lib.ck_model_kv_cache_reset()

    def encode(self, text: str) -> list:
        """Tokenize text."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def forward(self, token_ids: list) -> np.ndarray:
        """Run forward pass and return logits for last position."""
        n = len(token_ids)
        tokens = (ctypes.c_int32 * n)(*token_ids)

        self.lib.ck_model_embed_tokens(tokens, n)
        self.lib.ck_model_forward(None)

        # Get logits pointer
        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()

        # Get last position logits
        last_pos_offset = (active_tokens - 1) * self.vocab_size
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active_tokens * self.vocab_size,))
        last_logits = logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

        return last_logits

    def prefill(self, token_ids: list) -> np.ndarray:
        """Prefill KV cache by running a full forward once (returns last logits)."""
        return self.forward(token_ids)

    def decode_step(self, token_id: int) -> np.ndarray:
        """Decode one token using KV cache (returns logits for that token)."""
        ret = self.lib.ck_model_decode(ctypes.c_int32(int(token_id)), None)
        if ret != 0:
            raise RuntimeError(f"ck_model_decode failed (code {ret})")

        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()
        last_pos_offset = (active_tokens - 1) * self.vocab_size
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active_tokens * self.vocab_size,))
        return logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

    def free(self):
        """Free model resources."""
        if self.lib:
            self.lib.ck_model_free()


def sample_top_k(logits: np.ndarray, k: int = 40, temperature: float = 0.7) -> int:
    """Sample from top-k logits with temperature."""
    if temperature <= 0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]

    # Softmax
    max_logit = np.max(top_k_logits)
    exp_logits = np.exp(top_k_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    idx = np.random.choice(len(top_k_indices), p=probs)
    return int(top_k_indices[idx])


def generate(model: CKModel, prompt: str, max_tokens: int = 50,
             temperature: float = 0.7, verbose: bool = False) -> str:
    """Generate text from prompt."""
    # Tokenize prompt
    token_ids = model.encode(prompt)

    if verbose:
        print(f"[Prompt tokens: {len(token_ids)}]")

    generated = []
    start_time = time.time()

    if model.has_kv_decode and model.kv_cache_enable():
        # KV-cache path: prefill once, then decode token-by-token.
        logits = model.prefill(token_ids)
        for i in range(max_tokens):
            next_token = sample_top_k(logits, k=40, temperature=temperature)
            if next_token <= 2:
                break
            generated.append(next_token)
            token_ids.append(next_token)
            token_text = model.decode([next_token])
            print(token_text, end='', flush=True)
            if len(token_ids) >= model.context_window - 1:
                break
            logits = model.decode_step(next_token)
    else:
        for i in range(max_tokens):
            # Forward pass
            t0 = time.time()
            logits = model.forward(token_ids)
            fwd_time = time.time() - t0

            if verbose and i == 0:
                print(f"[First forward: {fwd_time:.2f}s]")

            # Sample next token
            next_token = sample_top_k(logits, k=40, temperature=temperature)

            # Check for EOS (typically 0, 1, or 2)
            if next_token <= 2:
                break

            generated.append(next_token)
            token_ids.append(next_token)

            # Decode and print incrementally
            token_text = model.decode([next_token])
            print(token_text, end='', flush=True)

            # Check context limit
            if len(token_ids) >= model.context_window - 1:
                break

    total_time = time.time() - start_time
    tokens_per_sec = len(generated) / total_time if total_time > 0 else 0

    if verbose:
        print(f"\n[Generated {len(generated)} tokens in {total_time:.2f}s ({tokens_per_sec:.1f} tok/s)]")

    return model.decode(generated)


def chat_loop(model: CKModel, temperature: float = 0.7, max_tokens: int = 100):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("  C-Kernel-Engine Chat")
    print("  Type your message and press Enter. Commands: /exit, /help")
    print("=" * 60 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
            print("Goodbye!")
            break

        if user_input.lower() == '/help':
            print("  Commands: /exit, /help")
            print("  Just type your message to chat with the model.")
            continue

        # Build prompt (simple format)
        prompt = user_input

        # Generate response
        print("\033[94mAssistant: \033[0m", end='', flush=True)
        response = generate(model, prompt, max_tokens=max_tokens,
                          temperature=temperature, verbose=False)
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="C-Kernel-Engine Chat Interface")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--prompt", help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = CKModel(args.model_dir)

    if not model.load():
        sys.exit(1)

    print(f"Model loaded! Vocab: {model.vocab_size}, Context: {model.context_window}")

    try:
        if args.prompt:
            # Single prompt mode
            print(f"\nPrompt: {args.prompt}")
            print("Response: ", end='', flush=True)
            generate(model, args.prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, verbose=args.verbose)
            print()
        else:
            # Interactive chat mode
            chat_loop(model, temperature=args.temperature, max_tokens=args.max_tokens)
    finally:
        model.free()


if __name__ == "__main__":
    main()
