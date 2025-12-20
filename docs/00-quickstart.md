# Quickstart Guide

This guide will get you up and running with the C-Kernel-Engine in less than 5 minutes.

## Prerequisites

- **Linux** (tested on Ubuntu 20.04+)
- **GCC** (with OpenMP support)
- **Make**
- **Python 3** (for running tests and PyTorch comparisons)
- **PyTorch** (only for running comparison tests)

## 1. Build the Engine

The project uses a standard Makefile. To build the shared library and the IR demo tool:

```bash
make
```

This will create:
- `build/libckernel_engine.so`: The main runtime library.
- `build/ck_ir_demo`: The compiler tool that converts `config.json` -> C code.

## 2. Run the Compiler Demo

The "Hello World" of this engine is compiling a standard Llama-style configuration into a C runtime.

We have a default configuration file ready: `default.config.json`.

```bash
make ck
```

**What just happened?**
1. The tool parsed `default.config.json`.
2. It generated an **Intermediate Representation (IR)** of the model's compute graph.
3. It emitted a `generated_model.c` file (conceptually) or printed the skeleton to stdout.

You should see output like:

```text
=== Forward IR ===
CKIRGraph: layers=32, hidden_size=4096 ...
  L0 N0 RMSNORM       outputs=[L0:N0:0]              inputs=[IN]
  L0 N1 LINEAR_QKV    outputs=[L0:N1:0]              inputs=[L0:N0]
  ...
```

## 3. Run the Unit Tests

We use Python to verify that our C kernels match PyTorch's output exactly.

```bash
make test
```

This command will:
1. Build individual shared libraries for each kernel family (e.g., `libckernel_gelu.so`).
2. Run the Python scripts in `unittest/`.
3. Report `OK` if the C implementation matches PyTorch within floating-point tolerance.

## 4. Generate a Standalone Runtime

To generate a full C file `ai.c` that you could compile and run (this feature is currently in active development):

```bash
./build/ck_ir_demo default.config.json --emit build/ai.c
```

Open `build/ai.c` to see how the engine structures the forward pass using the "Header / Block / Footer" pattern.
