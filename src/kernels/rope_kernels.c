/**
 * RoPE (Rotary Position Embedding) Kernels
 *
 * Applies rotary position embeddings to query and key vectors.
 * Used by Llama, SmolLM, and most modern transformer architectures.
 *
 * Math (Llama-style rotate-half):
 *   Split head_dim into two halves (0..half-1, half..head_dim-1).
 *   For each position m and index i in [0, half):
 *     x0 = x[i], x1 = x[i + half]
 *     x'[i]       = x0 * cos(m * θ_i) - x1 * sin(m * θ_i)
 *     x'[i+half]  = x0 * sin(m * θ_i) + x1 * cos(m * θ_i)
 *
 *   Where θ_i = 1 / (base^(2i/d)), typically base=10000.
 *
 * Layout:
 *   x: [num_heads, num_tokens, head_dim] head-major
 *   cos_cache, sin_cache: [max_seq_len, head_dim/2] precomputed
 */

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Precompute cos/sin cache for given sequence length and head dimension.
// cos_cache, sin_cache: [max_seq_len, head_dim/2]
void rope_precompute_cache(float *cos_cache,
                           float *sin_cache,
                           int max_seq_len,
                           int head_dim,
                           float base)
{
    int half_dim = head_dim / 2;

    double base_d = (double)base;
    double head_dim_d = (double)head_dim;
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            double theta = 1.0 / pow(base_d, (double)(2 * i) / head_dim_d);
            double angle = (double)pos * theta;
            cos_cache[pos * half_dim + i] = (float)cos(angle);
            sin_cache[pos * half_dim + i] = (float)sin(angle);
        }
    }
}

// Apply RoPE to a single head's Q or K tensor in-place.
// x: [num_tokens, head_dim] for one head
// cos_cache, sin_cache: [max_seq_len, head_dim/2]
// pos_offset: starting position (for KV cache continuation)
static inline void rope_apply_head(float *x,
                                   const float *cos_cache,
                                   const float *sin_cache,
                                   int num_tokens,
                                   int head_dim,
                                   int aligned_head_dim,
                                   int pos_offset)
{
    int half_dim = head_dim / 2;

    for (int t = 0; t < num_tokens; ++t) {
        int pos = pos_offset + t;
        const float *cos_row = cos_cache + pos * half_dim;
        const float *sin_row = sin_cache + pos * half_dim;
        float *x_row = x + (size_t)t * (size_t)aligned_head_dim;

        for (int i = 0; i < half_dim; ++i) {
            float x0 = x_row[i];
            float x1 = x_row[i + half_dim];
            float c = cos_row[i];
            float s = sin_row[i];

            x_row[i] = x0 * c - x1 * s;
            x_row[i + half_dim] = x0 * s + x1 * c;
        }
    }
}

// Apply RoPE forward to Q or K tensor (head-major layout).
// x: [num_heads, num_tokens, head_dim]
// cos_cache, sin_cache: [max_seq_len, head_dim/2]
// Modifies x in-place.
void rope_forward(float *x,
                  const float *cos_cache,
                  const float *sin_cache,
                  int num_heads,
                  int num_tokens,
                  int head_dim,
                  int aligned_head_dim,
                  int pos_offset)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        rope_apply_head(x + h * head_stride,
                        cos_cache, sin_cache,
                        num_tokens, head_dim, aligned_head_dim, pos_offset);
    }
}

// RoPE backward: inverse rotation (rotate by -θ).
// Since cos(-θ) = cos(θ) and sin(-θ) = -sin(θ), the inverse is:
//   d_x[2i]   =  d_x'[2i] * cos(θ) + d_x'[2i+1] * sin(θ)
//   d_x[2i+1] = -d_x'[2i] * sin(θ) + d_x'[2i+1] * cos(θ)
//
// d_x: [num_heads, num_tokens, head_dim] - gradient w.r.t. input (output)
// d_out: [num_heads, num_tokens, head_dim] - gradient from upstream (input)
// cos_cache, sin_cache: [max_seq_len, head_dim/2]
void rope_backward(const float *d_out,
                   float *d_x,
                   const float *cos_cache,
                   const float *sin_cache,
                   int num_heads,
                   int num_tokens,
                   int head_dim,
                   int aligned_head_dim,
                   int pos_offset)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    int half_dim = head_dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < num_tokens; ++t) {
            int pos = pos_offset + t;
            const float *cos_row = cos_cache + pos * half_dim;
            const float *sin_row = sin_cache + pos * half_dim;

            size_t idx = h * head_stride + (size_t)t * (size_t)aligned_head_dim;
            const float *d_out_row = d_out + idx;
            float *d_x_row = d_x + idx;

            for (int i = 0; i < half_dim; ++i) {
                float d0 = d_out_row[i];
                float d1 = d_out_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];

                // Inverse rotation: rotate by -θ
                d_x_row[i] = d0 * c + d1 * s;
                d_x_row[i + half_dim] = -d0 * s + d1 * c;
            }

            for (int i = head_dim; i < aligned_head_dim; ++i) {
                d_x_row[i] = 0.0f;
            }
        }
    }
}

// In-place backward: overwrite d_out with inverse-rotated gradients.
// Useful when d_x == d_out is acceptable.
void rope_backward_inplace(float *d_x,
                           const float *cos_cache,
                           const float *sin_cache,
                           int num_heads,
                           int num_tokens,
                           int head_dim,
                           int aligned_head_dim,
                           int pos_offset)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    int half_dim = head_dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < num_tokens; ++t) {
            int pos = pos_offset + t;
            const float *cos_row = cos_cache + pos * half_dim;
            const float *sin_row = sin_cache + pos * half_dim;

            float *d_row = d_x + h * head_stride + (size_t)t * (size_t)aligned_head_dim;

            for (int i = 0; i < half_dim; ++i) {
                float d0 = d_row[i];
                float d1 = d_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];

                // Inverse rotation: rotate by -θ
                d_row[i] = d0 * c + d1 * s;
                d_row[i + half_dim] = -d0 * s + d1 * c;
            }

            for (int i = head_dim; i < aligned_head_dim; ++i) {
                d_row[i] = 0.0f;
            }
        }
    }
}

// Combined RoPE forward for both Q and K (common pattern in inference).
// q: [num_heads, num_tokens, head_dim]
// k: [num_kv_heads, num_tokens, head_dim]
void rope_forward_qk(float *q,
                     float *k,
                     const float *cos_cache,
                     const float *sin_cache,
                     int num_heads,
                     int num_kv_heads,
                     int num_tokens,
                     int head_dim,
                     int aligned_head_dim,
                     int pos_offset)
{
    rope_forward(q, cos_cache, sin_cache, num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    rope_forward(k, cos_cache, sin_cache, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
}

// Combined RoPE backward for both d_q and d_k.
void rope_backward_qk(const float *d_q_out,
                      const float *d_k_out,
                      float *d_q,
                      float *d_k,
                      const float *cos_cache,
                      const float *sin_cache,
                      int num_heads,
                      int num_kv_heads,
                      int num_tokens,
                      int head_dim,
                      int aligned_head_dim,
                      int pos_offset)
{
    rope_backward(d_q_out, d_q, cos_cache, sin_cache, num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    rope_backward(d_k_out, d_k, cos_cache, sin_cache, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
}
