#include <stdlib.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

static float *alloc_float_buffer(size_t count)
{
    return (float *)malloc(count * sizeof(float));
}

static void free_float_buffer(float *buf)
{
    free(buf);
}

void rope_forward_bf16(uint16_t *x,
                       const float *cos_cache,
                       const float *sin_cache,
                       int num_heads,
                       int num_tokens,
                       int head_dim,
                       int aligned_head_dim,
                       int pos_offset)
{
    size_t total = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    float *tmp = alloc_float_buffer(total);
    if (!tmp) {
        return;
    }

    bf16_tensor_to_float(x, tmp, total);
    rope_forward(tmp, cos_cache, sin_cache,
                 num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    float_tensor_to_bf16(tmp, x, total);

    free_float_buffer(tmp);
}

void rope_backward_bf16(const uint16_t *d_out,
                        uint16_t *d_x,
                        const float *cos_cache,
                        const float *sin_cache,
                        int num_heads,
                        int num_tokens,
                        int head_dim,
                        int aligned_head_dim,
                        int pos_offset)
{
    size_t total = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    float *d_out_f = alloc_float_buffer(total);
    float *d_x_f = alloc_float_buffer(total);
    if (!d_out_f || !d_x_f) {
        free_float_buffer(d_out_f);
        free_float_buffer(d_x_f);
        return;
    }

    bf16_tensor_to_float(d_out, d_out_f, total);
    rope_backward(d_out_f, d_x_f, cos_cache, sin_cache,
                  num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    float_tensor_to_bf16(d_x_f, d_x, total);

    free_float_buffer(d_out_f);
    free_float_buffer(d_x_f);
}
