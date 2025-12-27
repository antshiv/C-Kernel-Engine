// KV-cache helper kernels (head-major layout).
//
// These are small, explicit helpers used by the runtime/orchestrator to maintain
// per-layer KV caches during autoregressive decoding.
//
// Layout:
//   k_cache[ kv_head, token, aligned_head_dim ]
//   v_cache[ kv_head, token, aligned_head_dim ]
// with contiguous row-major storage and stride aligned_head_dim.

#include "ckernel_engine.h"

#include <stddef.h>
#include <string.h>

void kv_cache_write_head_major(const float *k_token,
                               const float *v_token,
                               float *k_cache,
                               float *v_cache,
                               int num_kv_heads,
                               int token_index,
                               int cache_capacity,
                               int head_dim,
                               int aligned_head_dim)
{
    if (!k_token || !v_token || !k_cache || !v_cache) {
        return;
    }
    if (num_kv_heads <= 0 || token_index < 0 || cache_capacity <= 0) {
        return;
    }
    if (token_index >= cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    const size_t token_stride = (size_t)aligned_head_dim;

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *k_src = k_token + (size_t)h * token_stride;
        const float *v_src = v_token + (size_t)h * token_stride;

        float *k_dst = k_cache + (size_t)h * head_stride + (size_t)token_index * token_stride;
        float *v_dst = v_cache + (size_t)h * head_stride + (size_t)token_index * token_stride;

        if (head_dim == aligned_head_dim) {
            memcpy(k_dst, k_src, (size_t)aligned_head_dim * sizeof(float));
            memcpy(v_dst, v_src, (size_t)aligned_head_dim * sizeof(float));
        } else {
            for (int d = 0; d < head_dim; ++d) {
                k_dst[d] = k_src[d];
                v_dst[d] = v_src[d];
            }
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                k_dst[d] = 0.0f;
                v_dst[d] = 0.0f;
            }
        }
    }
}

