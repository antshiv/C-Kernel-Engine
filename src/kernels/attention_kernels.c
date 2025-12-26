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

            // Zero the full aligned head slice so padded dims stay clean.
            for (int d = 0; d < aligned_head_dim; ++d) {
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

// GQA-aware scaled dot-product attention with causal mask.
// Q has num_heads; K/V have num_kv_heads. Each query head maps to a KV head.
void attention_forward_causal_head_major_gqa(const float *q,
                                             const float *k,
                                             const float *v,
                                             float *scores,
                                             float *output,
                                             int num_heads,
                                             int num_kv_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

// ============================================================================
// ATTENTION BACKWARD - Causal, Head-Major, GQA-aware
// ============================================================================
//
// Backward pass for scaled dot-product attention with causal mask.
//
// Given:
//   d_output: gradient from the layer above [num_heads, T, head_dim]
//   q, k, v: saved activations from forward pass
//   attn_weights: saved softmax output from forward [num_heads, T, T]
//
// Computes:
//   d_q: gradient w.r.t. queries  [num_heads, T, head_dim]
//   d_k: gradient w.r.t. keys     [num_kv_heads, T, head_dim]
//   d_v: gradient w.r.t. values   [num_kv_heads, T, head_dim]
//
// Math derivation:
//   Forward: scores = Q @ K^T / sqrt(d)
//            weights = causal_softmax(scores)
//            output = weights @ V
//
//   Backward through V multiply:
//     d_weights = d_output @ V^T           [H, T, T]
//     d_v = weights^T @ d_output           [H_kv, T, d]
//
//   Backward through softmax:
//     d_scores = softmax_backward(d_weights, weights)
//
//   Backward through Q @ K^T:
//     d_q = d_scores @ K / sqrt(d)         [H, T, d]
//     d_k = d_scores^T @ Q / sqrt(d)       [H_kv, T, d]
//
// For GQA: multiple query heads share the same KV head, so we accumulate
// gradients from all query heads that map to each KV head.
//
void attention_backward_causal_head_major_gqa(
    const float *d_output,      // [num_heads, T, aligned_head_dim]
    const float *q,             // [num_heads, T, aligned_head_dim]
    const float *k,             // [num_kv_heads, T, aligned_head_dim]
    const float *v,             // [num_kv_heads, T, aligned_head_dim]
    const float *attn_weights,  // [num_heads, T, aligned_context_window]
    float *d_q,                 // [num_heads, T, aligned_head_dim] output
    float *d_k,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_v,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_scores,            // [num_heads, T, aligned_context_window] scratch
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    int T = num_tokens;
    int H = num_heads;
    int H_kv = num_kv_heads;
    int hd = head_dim;
    int ad = aligned_head_dim;
    int aw = aligned_context_window;

    // Zero d_k and d_v (we accumulate from multiple query heads for GQA)
    for (int kv_h = 0; kv_h < H_kv; ++kv_h) {
        for (int t = 0; t < T; ++t) {
            size_t base = qkv_index(kv_h, t, 0, T, ad);
            for (int i = 0; i < hd; ++i) {
                d_k[base + i] = 0.0f;
                d_v[base + i] = 0.0f;
            }
            /* Clear padding so later GEMMs don't see stale data. */
            for (int i = hd; i < ad; ++i) {
                d_k[base + i] = 0.0f;
                d_v[base + i] = 0.0f;
            }
        }
    }

    // Process each query head
    for (int h = 0; h < H; ++h) {
        // Which KV head does this query head use?
        int kv_h = (int)((long long)h * (long long)H_kv / (long long)H);

        // ----------------------------------------------------------------
        // Step 1: d_weights = d_output @ V^T  and  d_v += weights^T @ d_output
        // ----------------------------------------------------------------
        // For each query position i, compute d_weights[i, j] for j <= i
        // and accumulate d_v[j] contributions

        for (int i = 0; i < T; ++i) {
            size_t d_out_base = qkv_index(h, i, 0, T, ad);

            for (int j = 0; j <= i; ++j) {
                size_t v_base = qkv_index(kv_h, j, 0, T, ad);
                size_t w_idx = score_index(h, i, j, aw);
                float w = attn_weights[w_idx];

                // d_weights[h, i, j] = d_output[h, i, :] @ v[kv_h, j, :]^T
                float dot = 0.0f;
                for (int dd = 0; dd < hd; ++dd) {
                    dot += d_output[d_out_base + dd] * v[v_base + dd];
                }
                d_scores[w_idx] = dot;

                // d_v[kv_h, j, :] += weights[h, i, j] * d_output[h, i, :]
                for (int dd = 0; dd < hd; ++dd) {
                    d_v[v_base + dd] += w * d_output[d_out_base + dd];
                }
            }

            // Zero out upper triangle of d_scores
            for (int j = i + 1; j < T; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
            /* Scores scratch uses aligned_context_window, zero the padded columns. */
            for (int j = T; j < aw; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
        }

        // ----------------------------------------------------------------
        // Step 2: Backward through softmax (in-place on d_scores for this head)
        // ----------------------------------------------------------------
        // d_scores = softmax_backward(d_scores, attn_weights)
        // Formula: d_score[i,j] = w[i,j] * (d_w[i,j] - sum_k(w[i,k] * d_w[i,k]))

        for (int i = 0; i < T; ++i) {
            int base = h * aw * aw + i * aw;

            // Compute dot product: sum_j w[i,j] * d_w[i,j]
            float dot_product = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                dot_product += wt * dw;
            }

            // Apply softmax backward formula
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                d_scores[base + j] = wt * (dw - dot_product);
            }
        }

        // ----------------------------------------------------------------
        // Step 3: d_q = d_scores @ K * scale
        //         d_k += d_scores^T @ Q * scale
        // ----------------------------------------------------------------

        for (int i = 0; i < T; ++i) {
            size_t d_q_base = qkv_index(h, i, 0, T, ad);
            size_t q_base = qkv_index(h, i, 0, T, ad);

            // Zero d_q for this position
            for (int dd = 0; dd < hd; ++dd) {
                d_q[d_q_base + dd] = 0.0f;
            }
            /* Zero padded head lanes before accumulation. */
            for (int dd = hd; dd < ad; ++dd) {
                d_q[d_q_base + dd] = 0.0f;
            }

            // d_q[h, i, :] = sum_j d_scores[h, i, j] * k[kv_h, j, :] * scale
            // d_k[kv_h, j, :] += d_scores[h, i, j] * q[h, i, :] * scale
            for (int j = 0; j <= i; ++j) {
                size_t k_base = qkv_index(kv_h, j, 0, T, ad);
                size_t d_k_base = qkv_index(kv_h, j, 0, T, ad);
                float ds = d_scores[score_index(h, i, j, aw)] * scale;

                for (int dd = 0; dd < hd; ++dd) {
                    d_q[d_q_base + dd] += ds * k[k_base + dd];
                    d_k[d_k_base + dd] += ds * q[q_base + dd];
                }
            }
        }
    }
}

// Non-GQA version (num_heads == num_kv_heads)
void attention_backward_causal_head_major(
    const float *d_output,
    const float *q,
    const float *k,
    const float *v,
    const float *attn_weights,
    float *d_q,
    float *d_k,
    float *d_v,
    float *d_scores,
    int num_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    attention_backward_causal_head_major_gqa(
        d_output, q, k, v, attn_weights,
        d_q, d_k, d_v, d_scores,
        num_heads, num_heads,  // num_kv_heads == num_heads
        num_tokens, head_dim, aligned_head_dim, aligned_context_window);
}
