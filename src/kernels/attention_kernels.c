#include "ckernel_engine.h"
#include <math.h>

// Helpers for head-major layouts used in attention.
// Q/K/V layout: [head][token][head_dim] with stride aligned_head_dim.
static inline size_t qkv_index(int h,
                               int t,
                               int d,
                               int num_tokens,
                               int aligned_head_dim)
{
    return ((size_t)h * (size_t)num_tokens + (size_t)t) * (size_t)aligned_head_dim
         + (size_t)d;
}

// Scores layout matches causal_softmax_head_major:
// [head][query_token][key_token] with stride aligned_context_window.
static inline size_t score_index(int h,
                                 int i,
                                 int j,
                                 int aligned_context_window)
{
    return ((size_t)h * (size_t)aligned_context_window * (size_t)aligned_context_window)
         + (size_t)i * (size_t)aligned_context_window
         + (size_t)j;
}

// Naive, reference-quality scaled dot-product attention with causal mask.
//
// Q, K, V are head-major:
//   q[h, t, d] at q[h * T * aligned_head_dim + t * aligned_head_dim + d]
//   same for k and v.
//
// scores buffer must be at least:
//   num_heads * aligned_context_window * aligned_context_window floats.
//
// output has same layout as Q/V:
//   out[h, t, d] at out[h * T * aligned_head_dim + t * aligned_head_dim + d]
//
// Only the first head_dim elements of each vector participate in the math;
// aligned_head_dim allows callers to pad to cache-friendly widths.
void attention_forward_causal_head_major(const float *q,
                                         const float *k,
                                         const float *v,
                                         float *scores,
                                         float *output,
                                         int num_heads,
                                         int num_tokens,
                                         int head_dim,
                                         int aligned_head_dim,
                                         int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute scaled dot-product scores Q·K^T / sqrt(d_k),
    // lower triangle only (j <= i).
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            // Ensure upper triangle is zeroed so there are no stale values
            // before the softmax kernel runs.
            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Phase 2: apply causal row-wise softmax in-place over j <= i.
    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    // Phase 3: attention weights · V.
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);

            // Zero the output vector for this head/token.
            for (int d = 0; d < head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            // Weighted sum over causal positions.
            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

