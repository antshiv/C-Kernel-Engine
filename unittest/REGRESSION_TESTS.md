# Regression Test Registry

This file tracks bugs that have been caught and are now covered by unit tests.
When fixing a new bug, add it here so we don't repeat mistakes.

## How to Run Tests

```bash
# Run all parity tests (auto-escalates to DEBUG on failure)
python scripts/ck_run_v4.py run <model> --test

# Run individual test with debug output
DEBUG=1 python unittest/test_kv_cache_layer_decode.py

# Run with valgrind (catches memory bugs)
valgrind python unittest/test_kv_cache_layer_decode.py
```

---

## test_kv_cache_layer_decode.py

### Bug: Non-contiguous numpy cache array (2024-01)
- **Symptom**: K/V cache corruption, wrong values at position N-1 when writing position N
- **Root cause**: Guard region allocation created non-contiguous view
  ```python
  buf = aligned_empty((num_kv_heads, cache_capacity + 1, aligned_head_dim))
  cache = buf[:, :cache_capacity, :]  # NON-CONTIGUOUS - wrong strides!
  ```
- **Fix**: Allocate cache and guard as separate contiguous arrays
- **Debug indicators**:
  - `C_CONTIGUOUS: False`
  - Stride mismatch (e.g., 1088 vs expected 1024)
  - K[1,N-1,*] gets overwritten when writing position N

### Bug: Flat copy mixing head boundaries
- **Symptom**: K/V values scrambled across heads
- **Root cause**: Using `np.copyto(cache.reshape(-1), buffer.reshape(-1))`
  when source and dest have different token strides
- **Fix**: Copy head-by-head to preserve layout
  ```python
  for h in range(num_kv_heads):
      k_cache[h, :prompt_len, :] = buf_prefill["k"][h, :, :]
  ```

### Bug: Q stride vs KV stride mismatch (GQA)
- **Symptom**: Heap corruption on server, works locally
- **Root cause**: Q needs `tokens` stride, K/V need `cache_capacity` stride for GQA
- **Fix**: Added `kv_stride_tokens` parameter to separate Q from K/V strides

---

## test_fused_attention_decode.py

### (Add bugs as they are caught)

---

## test_multi_layer_parity.py

Progressive layer test (1 → 2 → 4 layers) to catch inter-layer bugs.

### What it catches:
- Layer output shape mismatches
- Inter-layer data handoff bugs
- Residual connection accumulation errors
- Memory layout incompatibilities between layers

### Key metrics:
- Diff should grow slowly with layers (not exponentially)
- If 1 layer passes but 2+ layers fail catastrophically, there's a handoff bug
- Typical expected diff: ~0.1-0.2 (due to numerical differences)
- Bug threshold: diff > 0.5 indicates real issue

---

## Adding New Regression Tests

When you fix a bug:
1. Add DEBUG output to help diagnose similar issues
2. Document the bug in this file with:
   - Symptom (what you observed)
   - Root cause (why it happened)
   - Fix (what you changed)
   - Debug indicators (how DEBUG mode reveals it)
3. The test should fail if the bug regresses

## Test Coverage Checklist

- [x] KV cache contiguity
- [x] Head-by-head copy for different strides
- [x] GQA stride separation (Q vs K/V)
- [x] Multi-layer parity (1 → 2 → 4 layers)
- [x] Inter-layer data handoff
- [ ] Quantization alignment (Q4_K, Q8_K require 256-byte alignment)
- [ ] Multi-layer KV cache handoff (decode across layers)
- [ ] RoPE position offset correctness
