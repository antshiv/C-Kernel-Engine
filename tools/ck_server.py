#!/usr/bin/env python3
"""
C-Kernel-Engine Streaming Inference Server

HTTP server with Server-Sent Events (SSE) for streaming token generation.
Compatible with OpenAI-style API endpoints.

Usage:
    python tools/ck_server.py --model-dir ~/.cache/huggingface/hub/SmolLM-135M --port 8080

API Endpoints:
    POST /v1/completions     - Generate completion (streaming or non-streaming)
    POST /v1/chat/completions - Chat completion (streaming or non-streaming)
    GET  /v1/models          - List available models
    GET  /health             - Health check
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import shutil

import numpy as np

# Disable torch compile overhead
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from transformers import AutoTokenizer


VERSION = "0.1.0"

BANNER = r"""
   ____      _  __                    _   ____
  / ___|    | |/ /___ _ __ _ __   ___| | / ___|  ___ _ ____   _____ _ __
 | |   _____| ' // _ \ '__| '_ \ / _ \ | \___ \ / _ \ '__\ \ / / _ \ '__|
 | |__|_____| . \  __/ |  | | | |  __/ |  ___) |  __/ |   \ V /  __/ |
  \____|    |_|\_\___|_|  |_| |_|\___|_| |____/ \___|_|    \_/ \___|_|

  C-Kernel-Engine Inference Server v{version}
  By Anthony Shivakumar

  Streaming API Server for C-based Transformer Inference
  ============================================================
"""


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


def sample_token(logits, temperature=0.7, top_k=40, top_p=0.9):
    """Sample next token from logits."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k > 0:
        indices = np.argpartition(-logits, min(top_k, len(logits) - 1))[:top_k]
        top_logits = logits[indices]
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / probs.sum()

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
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))


class CKernelModel:
    """Wrapper for C-Kernel-Engine model inference."""

    def __init__(self, model_dir, build_dir, context_len=512):
        self.model_dir = model_dir
        self.build_dir = build_dir
        self.context_len = context_len
        self.lock = threading.Lock()

        self.project_root = str(Path(__file__).parent.parent)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # Load config
        self.cfg = load_config(model_dir)
        self.vocab_size = self.cfg.get("vocab_size", 32000)
        self.model_name = os.path.basename(model_dir)

        # Override context
        self.cfg["max_position_embeddings"] = context_len
        self.cfg["context_window"] = context_len

        # Write runtime config
        os.makedirs(build_dir, exist_ok=True)
        self.config_path = os.path.join(build_dir, "server.config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

        # Build model
        self._build_model()

        # Prepare weights
        self._prepare_weights()

    def _build_model(self):
        """Build the model binary."""
        gen_c = os.path.join(self.build_dir, "server_model.c")
        kernel_manifest = gen_c + ".kernels"
        self.model_bin = os.path.join(self.build_dir, "server_model")

        ir_demo = os.path.join(self.project_root, "build", "ck_ir_demo")
        if not os.path.exists(ir_demo):
            subprocess.check_call(["make", "build/ck_ir_demo"], cwd=self.project_root)

        subprocess.check_call([ir_demo, self.config_path, "--emit", gen_c], cwd=self.project_root)

        with open(kernel_manifest, "r", encoding="utf-8") as f:
            kernels = f.read().split()

        cc = os.environ.get("CC", detect_cc())
        cflags = ["-O3", "-fPIC", openmp_flag(cc), "-Wall"] + detect_avx_flags() + ["-Iinclude"]

        cmd = [cc] + cflags + [gen_c] + kernels + ["-o", self.model_bin, "-lm"]
        subprocess.check_call(cmd, cwd=self.project_root)

    def _prepare_weights(self):
        """Convert weights to bump format."""
        self.weights_bin = os.path.join(self.build_dir, "server_weights.bin")

        subprocess.check_call([
            sys.executable,
            os.path.join(self.project_root, "scripts", "convert_hf_to_bump.py"),
            "--checkpoint", self.model_dir,
            "--output", self.weights_bin,
            "--config", self.config_path,
            "--context", str(self.context_len),
        ], cwd=self.project_root)

    def encode(self, text):
        """Encode text to tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        """Decode tokens to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def forward(self, token_ids):
        """Run forward pass, return logits for last position."""
        with self.lock:
            T = len(token_ids)
            if T > self.context_len:
                token_ids = token_ids[-self.context_len:]
                T = self.context_len

            padded = list(token_ids)
            if len(padded) < self.context_len:
                pad_id = self.tokenizer.eos_token_id or 0
                padded.extend([pad_id] * (self.context_len - len(padded)))

            tokens_bin = os.path.join(self.build_dir, f"tokens_{threading.current_thread().ident}.bin")
            logits_bin = os.path.join(self.build_dir, f"logits_{threading.current_thread().ident}.bin")

            np.array(padded, dtype=np.int32).tofile(tokens_bin)

            cmd = [
                self.model_bin,
                "--model-weights", self.weights_bin,
                "--tokens", tokens_bin,
                "--out-logits", logits_bin,
                "--ctx", str(self.context_len),
            ]

            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            logits = np.fromfile(logits_bin, dtype=np.float32)
            logits = logits.reshape(self.context_len, self.vocab_size)

            # Cleanup temp files
            try:
                os.remove(tokens_bin)
                os.remove(logits_bin)
            except OSError:
                pass

            return logits[T - 1]

    def generate_stream(self, prompt, max_tokens=128, temperature=0.7, top_k=40, top_p=0.9, stop=None):
        """Generate tokens as a stream."""
        token_ids = self.encode(prompt)

        for i in range(max_tokens):
            logits = self.forward(token_ids)
            next_token = sample_token(logits, temperature, top_k, top_p)

            if next_token == self.tokenizer.eos_token_id:
                break

            token_ids.append(next_token)
            token_text = self.decode([next_token])

            # Check stop sequences
            if stop:
                full_text = self.decode(token_ids[len(self.encode(prompt)):])
                for s in stop:
                    if s in full_text:
                        return

            yield {
                "token": next_token,
                "text": token_text,
                "index": i,
            }

    def generate(self, prompt, max_tokens=128, temperature=0.7, top_k=40, top_p=0.9, stop=None):
        """Generate complete response."""
        tokens = []
        for chunk in self.generate_stream(prompt, max_tokens, temperature, top_k, top_p, stop):
            tokens.append(chunk["token"])
        return self.decode(tokens)


class CKernelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the inference server."""

    model = None
    server_start_time = None

    def log_message(self, format, *args):
        """Custom logging format."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} - {format % args}")

    def send_json(self, data, status=200):
        """Send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_sse(self, data):
        """Send Server-Sent Event."""
        if isinstance(data, dict):
            data = json.dumps(data)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path == "/health":
            self.send_json({
                "status": "ok",
                "version": VERSION,
                "uptime": time.time() - self.server_start_time,
            })

        elif path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": self.model.model_name,
                    "object": "model",
                    "owned_by": "c-kernel-engine",
                    "permission": [],
                }]
            })

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if path == "/v1/completions":
            self.handle_completions(data)
        elif path == "/v1/chat/completions":
            self.handle_chat_completions(data)
        else:
            self.send_json({"error": "Not found"}, 404)

    def handle_completions(self, data):
        """Handle /v1/completions endpoint."""
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 40)
        stream = data.get("stream", False)
        stop = data.get("stop", [])

        if isinstance(stop, str):
            stop = [stop]

        request_id = f"cmpl-{int(time.time() * 1000)}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for chunk in self.model.generate_stream(prompt, max_tokens, temperature, top_k, top_p, stop):
                event = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model.model_name,
                    "choices": [{
                        "text": chunk["text"],
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }]
                }
                self.send_sse(event)

            # Send done event
            self.send_sse("[DONE]")

        else:
            text = self.model.generate(prompt, max_tokens, temperature, top_k, top_p, stop)
            self.send_json({
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model.model_name,
                "choices": [{
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(self.model.encode(prompt)),
                    "completion_tokens": len(self.model.encode(text)),
                    "total_tokens": len(self.model.encode(prompt)) + len(self.model.encode(text)),
                }
            })

    def handle_chat_completions(self, data):
        """Handle /v1/chat/completions endpoint."""
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 40)
        stream = data.get("stream", False)
        stop = data.get("stop", [])

        if isinstance(stop, str):
            stop = [stop]

        # Build prompt from messages
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"

        request_id = f"chatcmpl-{int(time.time() * 1000)}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for chunk in self.model.generate_stream(prompt, max_tokens, temperature, top_k, top_p, stop):
                event = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk["text"]},
                        "finish_reason": None,
                    }]
                }
                self.send_sse(event)

            self.send_sse("[DONE]")

        else:
            text = self.model.generate(prompt, max_tokens, temperature, top_k, top_p, stop)
            self.send_json({
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(self.model.encode(prompt)),
                    "completion_tokens": len(self.model.encode(text)),
                    "total_tokens": len(self.model.encode(prompt)) + len(self.model.encode(text)),
                }
            })


def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine Streaming Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "SmolLM-135M"),
        help="HuggingFace model directory"
    )
    parser.add_argument(
        "--build-dir",
        default="build/server",
        help="Build output directory"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=512,
        help="Context window size"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show startup banner"
    )
    args = parser.parse_args()

    if not args.no_banner:
        print(BANNER.format(version=VERSION))

    # Check model exists
    if not os.path.exists(os.path.join(args.model_dir, "config.json")):
        print(f"Error: Model not found at {args.model_dir}")
        print("Download with: python scripts/download_smollm.py --outdir " + args.model_dir)
        return 1

    print(f"Loading model from {args.model_dir}...")

    try:
        model = CKernelModel(
            model_dir=args.model_dir,
            build_dir=args.build_dir,
            context_len=args.context
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Set up handler
    CKernelHandler.model = model
    CKernelHandler.server_start_time = time.time()

    # Start server
    server = HTTPServer((args.host, args.port), CKernelHandler)
    print(f"\nServer running at http://{args.host}:{args.port}")
    print(f"API endpoints:")
    print(f"  POST /v1/completions      - Text completion")
    print(f"  POST /v1/chat/completions - Chat completion")
    print(f"  GET  /v1/models           - List models")
    print(f"  GET  /health              - Health check")
    print(f"\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
