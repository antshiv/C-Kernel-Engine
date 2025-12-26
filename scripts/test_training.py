#!/usr/bin/env python3
"""
C-Kernel-Engine Training Test

Tests the training API: forward, backward, and optimizer step.
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Error: tokenizers package not found. Install with: pip install tokenizers")
    sys.exit(1)


class CKModel:
    """Wrapper for the C model library with training support."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.lib = None
        self.tokenizer = None
        self.vocab_size = 0
        self.context_window = 0

    def load(self) -> bool:
        """Load model library and tokenizer."""
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            print(f"Error: Tokenizer not found: {tokenizer_path}")
            return False
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            print(f"Error: Model library not found: {lib_path}")
            return False

        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

        weights_path = self.model_dir / "weights.bump"
        ret = self.lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"Error: Failed to initialize model (code {ret})")
            return False

        self.vocab_size = self.lib.ck_model_get_vocab_size()
        self.context_window = self.lib.ck_model_get_context_window()
        return True

    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # Init/Free
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        # Forward
        self.lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int
        self.lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_forward.restype = ctypes.c_int
        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        # Getters
        self.lib.ck_model_get_vocab_size.argtypes = []
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int
        self.lib.ck_model_get_context_window.argtypes = []
        self.lib.ck_model_get_context_window.restype = ctypes.c_int
        self.lib.ck_model_get_active_tokens.argtypes = []
        self.lib.ck_model_get_active_tokens.restype = ctypes.c_int

        # Training
        self.lib.ck_model_enable_training.argtypes = [ctypes.c_float]
        self.lib.ck_model_enable_training.restype = ctypes.c_int
        self.lib.ck_model_disable_training.argtypes = []
        self.lib.ck_model_disable_training.restype = None
        self.lib.ck_model_is_training_enabled.argtypes = []
        self.lib.ck_model_is_training_enabled.restype = ctypes.c_int
        self.lib.ck_model_backward.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # tokens
            ctypes.POINTER(ctypes.c_int32),  # targets
            ctypes.POINTER(ctypes.c_float)   # loss_out
        ]
        self.lib.ck_model_backward.restype = ctypes.c_int
        self.lib.ck_model_optimizer_step.argtypes = []
        self.lib.ck_model_optimizer_step.restype = None
        self.lib.ck_model_set_learning_rate.argtypes = [ctypes.c_float]
        self.lib.ck_model_set_learning_rate.restype = None
        self.lib.ck_model_get_learning_rate.argtypes = []
        self.lib.ck_model_get_learning_rate.restype = ctypes.c_float

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list) -> str:
        return self.tokenizer.decode(token_ids)

    def enable_training(self, lr: float = 1e-4) -> int:
        return self.lib.ck_model_enable_training(ctypes.c_float(lr))

    def disable_training(self):
        self.lib.ck_model_disable_training()

    def is_training_enabled(self) -> bool:
        return bool(self.lib.ck_model_is_training_enabled())

    def forward(self, token_ids: list) -> np.ndarray:
        """Run forward pass."""
        n = len(token_ids)
        tokens = (ctypes.c_int32 * n)(*token_ids)
        self.lib.ck_model_embed_tokens(tokens, n)
        self.lib.ck_model_forward(None)

        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active_tokens * self.vocab_size,))
        return logits_array.reshape(active_tokens, self.vocab_size).copy()

    def backward(self, token_ids: list, target_ids: list) -> float:
        """Run backward pass and return loss."""
        n = len(token_ids)
        tokens = (ctypes.c_int32 * n)(*token_ids)
        targets = (ctypes.c_int32 * n)(*target_ids)
        loss = ctypes.c_float(0.0)

        ret = self.lib.ck_model_backward(tokens, targets, ctypes.byref(loss))
        if ret != 0:
            raise RuntimeError(f"Backward pass failed with code {ret}")
        return loss.value

    def optimizer_step(self):
        """Apply gradients to weights."""
        self.lib.ck_model_optimizer_step()

    def free(self):
        if self.lib:
            self.lib.ck_model_free()


def test_training_api(model: CKModel):
    """Test that training API functions exist and work."""
    print("\n=== Testing Training API ===")

    # Test enable/disable
    print(f"Training enabled (before): {model.is_training_enabled()}")
    ret = model.enable_training(lr=1e-4)
    print(f"Enable training returned: {ret}")
    print(f"Training enabled (after): {model.is_training_enabled()}")

    if not model.is_training_enabled():
        print("ERROR: Training not enabled!")
        return False

    model.disable_training()
    print(f"Training enabled (disabled): {model.is_training_enabled()}")

    print("Training API test PASSED")
    return True


def test_forward_backward(model: CKModel, text: str = "Hello world"):
    """Test forward and backward pass."""
    print(f"\n=== Testing Forward/Backward ===")
    print(f"Input text: '{text}'")

    # Tokenize
    tokens = model.encode(text)
    print(f"Tokens: {tokens}")

    # Create targets (shifted by 1 for language modeling)
    # For simplicity, targets = tokens shifted left + padding
    targets = tokens[1:] + [0]  # Next token prediction
    print(f"Targets: {targets}")

    # Enable training
    model.enable_training(lr=1e-4)

    # Forward pass
    print("\nRunning forward pass...")
    t0 = time.time()
    logits = model.forward(tokens)
    fwd_time = time.time() - t0
    print(f"Forward time: {fwd_time:.3f}s")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Check logits are reasonable
    if np.isnan(logits).any() or np.isinf(logits).any():
        print("ERROR: Logits contain NaN or Inf!")
        return False

    # Backward pass
    print("\nRunning backward pass...")
    t0 = time.time()
    try:
        loss = model.backward(tokens, targets)
        bwd_time = time.time() - t0
        print(f"Backward time: {bwd_time:.3f}s")
        print(f"Loss: {loss:.6f}")

        if np.isnan(loss) or np.isinf(loss):
            print("ERROR: Loss is NaN or Inf!")
            return False

    except Exception as e:
        print(f"ERROR: Backward pass failed: {e}")
        return False

    print("Forward/Backward test PASSED")
    return True


def test_training_loop(model: CKModel, text: str = "The quick brown fox", steps: int = 5):
    """Test a mini training loop."""
    print(f"\n=== Testing Training Loop ({steps} steps) ===")
    print(f"Training text: '{text}'")

    tokens = model.encode(text)
    targets = tokens[1:] + [0]

    model.enable_training(lr=1e-3)

    losses = []
    for step in range(steps):
        # Forward
        logits = model.forward(tokens)

        # Backward
        loss = model.backward(tokens, targets)
        losses.append(loss)

        # Optimizer step
        model.optimizer_step()

        print(f"Step {step + 1}: loss = {loss:.6f}")

    # Check if loss decreased (not guaranteed but good sign)
    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            print(f"Loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f}")
        else:
            print(f"Warning: Loss didn't decrease: {losses[0]:.6f} -> {losses[-1]:.6f}")

    model.disable_training()
    print("Training loop test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test C-Kernel-Engine Training")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--steps", type=int, default=5, help="Training steps to run")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = CKModel(args.model_dir)

    if not model.load():
        sys.exit(1)

    print(f"Model loaded! Vocab: {model.vocab_size}, Context: {model.context_window}")

    try:
        # Run tests
        all_passed = True

        if not test_training_api(model):
            all_passed = False

        if not test_forward_backward(model):
            all_passed = False

        if not test_training_loop(model, steps=args.steps):
            all_passed = False

        print("\n" + "=" * 50)
        if all_passed:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
            sys.exit(1)

    finally:
        model.free()


if __name__ == "__main__":
    main()
