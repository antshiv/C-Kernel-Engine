# Kernel Testing Guide

This document explains how to build and test individual kernels in C-Kernel-Engine
using the provided Python unittest scripts.

All tests compare the C kernels against PyTorch reference implementations
(forward and backward) and fail on any significant mismatch.

---

## 1. Build Shared Libraries

From the project root:

```bash
cd C-Kernel-Engine

# Build the full engine (all kernels)
make

# Build small, kernel-specific libraries
make libckernel_gelu.so
make libckernel_rmsnorm.so
make libckernel_layernorm.so   # requires AVX-512 CPU
make libckernel_softmax.so     # requires AVX-512 CPU
```

On machines without AVX-512 (e.g., older laptops), only `libckernel_gelu.so`
and `libckernel_rmsnorm.so` are safe to execute. The LayerNorm and Softmax
libraries use AVX-512 intrinsics and should be run on a Xeon or other
AVX-512-capable CPU.

There is also a helper target that builds all kernel-specific libs:

```bash
make test-libs
```

This is automatically invoked by `make test` on AVX-512 machines.

---

## 2. Running Kernel Tests (Python)

Each kernel family has a corresponding test script under `unittest/`.
These scripts:

- Load the appropriate shared library via `ctypes`.
- Generate random tensors with PyTorch.
- Compute PyTorch reference outputs/gradients.
- Call the C kernels and compare results.

### 2.1 GELU

Build the GELU-only library and run tests:

```bash
make libckernel_gelu.so
python3 unittest/test_gelu.py
```

This script checks:

- Forward: `gelu_fast_inplace` vs `torch.nn.functional.gelu(..., approximate="tanh")`
- Backward (exact): `gelu_backward_exact` vs PyTorch autograd
- Backward (fast): `gelu_backward_fast` is treated as approximate; its max diff
  is printed but not enforced in `make test`.

### 2.2 RMSNorm

Build the RMSNorm-only library and run tests:

```bash
make libckernel_rmsnorm.so
python3 unittest/test_rmsnorm.py
```

This script checks:

- Forward: `rmsnorm_forward` vs a PyTorch RMSNorm reference:
  ```python
  var = x.pow(2).mean(dim=-1, keepdim=True)
  rstd = (var + eps).rsqrt()
  y = x * rstd * gamma
  ```
- Backward: `rmsnorm_backward` vs PyTorch autograd for `dX` and `dGamma`.

### 2.3 LayerNorm

Requires AVX-512 CPU.

```bash
make libckernel_layernorm.so
python3 unittest/test_layernorm.py
```

Checks:

- Forward:
  - `layernorm_naive_serial`
  - `layernorm_forward_rolled_slice`
  - `layernorm_forward_unrolled_slice`
  vs `torch.layer_norm`.
- Backward:
  - `layernorm_backward_kernel` vs PyTorch autograd for `dX`, `dGamma`, `dBeta`.

### 2.4 Softmax (Causal)

Requires AVX-512 CPU.

```bash
make libckernel_softmax.so
python3 unittest/test_softmax.py
python3 unittest/test_softmax_backward.py
```

Checks:

- Forward: `causal_softmax_head_major` vs a PyTorch masked softmax reference.
- Backward: `backward_causal_softmax_head_major` vs a pure PyTorch
  Jacobian–vector-product implementation.

### 2.5 GEMM and MLP

GEMM tests:

```bash
python3 unittest/test_gemm.py
```

  Checks all GEMM variants (`gemm_naive_parallel`, `gemm_avx512_parallel`,
  `gemm_fine_grained_parallel`, `gemm_blocked_serial`) against
  `A @ B.T + bias` for LLM-relevant shapes.

MLP tests:

```bash
python3 unittest/test_mlp.py
```

  Checks:

  - Forward: `mlp_token_parallel` vs `Linear(D→4D) → GELU → Linear(4D→D)` in PyTorch.
  - Backward: FC1/FC2 + GELU derivatives vs full PyTorch autograd
    for `dX`, `dW1`, `db1`, `dW2`, `db2`.

### 2.6 SwiGLU

Build the SwiGLU-only library and run tests:

```bash
make libckernel_swiglu.so
python3 unittest/test_swiglu.py
```

Checks:

  - Forward: `swiglu_forward` vs a PyTorch reference:

  ```python
  gate, value = x[:, :D], x[:, D:]
  y = F.silu(gate) * value
  ```

  - Backward: `swiglu_backward` vs PyTorch autograd for `dX` (both gate and value parts).

### 2.7 Sigmoid

Build the Sigmoid-only library and run tests:

```bash
make libckernel_sigmoid.so
python3 unittest/test_sigmoid.py
```

Checks:

- Forward: `sigmoid_forward` vs `torch.sigmoid`.
- Backward: `sigmoid_backward` vs PyTorch autograd for `dX`.

---

## 3. Running the Full Test Suite

On an AVX-512-capable machine:

```bash
cd C-Kernel-Engine
make
make test
```

This will:

- Build the full engine library (`libckernel_engine.so`).
- Build all kernel-specific libs (`make test-libs`).
- Run all unittest scripts:
  - `test_layernorm.py`
  - `test_gelu.py`
  - `test_softmax.py`
  - `test_softmax_backward.py`
  - `test_gemm.py`
  - `test_mlp.py`
  - `test_rmsnorm.py`

On machines **without AVX-512**, only run the tests whose `.so` libraries
do not use AVX-512 (GELU, RMSNorm), as shown above.

---

## 4. End-to-End / Orchestration Tests

These test the full decoder layer orchestration and the tiny training loop.

### 4.1 Layer Parity (strict C ref)

Runs the decoder layer against the strict C reference path (naive GEMM),
which is tight enough for AVX-only CPUs:

```bash
make layer-parity-scalar TOL=2e-3 ARGS="--tokens 256 --embed 64 --heads 4 --kv-heads 2 --intermediate 128 --rope --strict-ref"
```

### 4.2 Tiny End-to-End Training Parity (C vs PyTorch)

Runs a tiny model for 1 step and compares loss + final weights vs PyTorch:

```bash
make tiny-parity
```

Multi-step:

```bash
python3 scripts/tiny_train_parity.py --config tiny.config.json --steps 5 --lr 1e-3
```

### 4.3 One-Command Test Sweep

Runs kernel tests, layer parity, and tiny training parity:

```bash
make all-tests
```

You can tune the layer parity args with:

```bash
make all-tests ALL_TEST_LAYER_ARGS="--tokens 256 --embed 64 --heads 4 --kv-heads 2 --intermediate 128 --rope --strict-ref" ALL_TEST_LAYER_TOL=1e-3
```

---

## 5. Notes

- All tests print max differences and sample tensor entries when mismatches
  exceed tight tolerances (~1e-6).
- Exact kernels (GELU exact backward, LayerNorm, RMSNorm, softmax) are treated
  as correctness-critical: any significant difference causes the test to fail.
- Approximate kernels (e.g., fast GELU backward) are considered experimental
  and do not affect `make test` unless explicitly enabled in their scripts.
