#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "ckernel_engine.h"

static void convert_int8_to_float(const int8_t *src,
                                  float *dst,
                                  size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (float)src[i];
    }
}

static int8_t clamp_int8(float value)
{
    int32_t q = (int32_t)lrintf(value);
    if (q > INT8_MAX) {
        q = INT8_MAX;
    } else if (q < INT8_MIN) {
        q = INT8_MIN;
    }
    return (int8_t)q;
}

static void convert_float_to_int8(const float *src,
                                  int8_t *dst,
                                  size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = clamp_int8(src[i]);
    }
}

static float *alloc_float_buffer(size_t count)
{
    return (float *)malloc(count * sizeof(float));
}

static void free_float_buffer(float *buf)
{
    free(buf);
}

void rmsnorm_forward_int8(const int8_t *input,
                          const float *gamma,
                          int8_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps)
{
    if (!input || !gamma || !output) {
        return;
    }

    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    float *input_f = alloc_float_buffer(total);
    float *output_f = alloc_float_buffer(total);
    if (!input_f || !output_f) {
        free_float_buffer(input_f);
        free_float_buffer(output_f);
        return;
    }

    convert_int8_to_float(input, input_f, total);
    rmsnorm_forward(input_f, gamma, output_f, rstd_cache,
                    tokens, d_model, aligned_embed_dim, eps);
    convert_float_to_int8(output_f, output, total);

    free_float_buffer(input_f);
    free_float_buffer(output_f);
}

void rmsnorm_backward_int8(const int8_t *d_output,
                           const int8_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           int8_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim)
{
    if (!d_output || !input || !gamma || !rstd_cache || !d_input || !d_gamma) {
        return;
    }

    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    float *d_output_f = alloc_float_buffer(total);
    float *input_f = alloc_float_buffer(total);
    float *d_input_f = alloc_float_buffer(total);
    if (!d_output_f || !input_f || !d_input_f) {
        free_float_buffer(d_output_f);
        free_float_buffer(input_f);
        free_float_buffer(d_input_f);
        return;
    }

    convert_int8_to_float(d_output, d_output_f, total);
    convert_int8_to_float(input, input_f, total);

    // Zero gamma gradient before accumulation.
    for (int d = 0; d < d_model; ++d) {
        d_gamma[d] = 0.0f;
    }

    rmsnorm_backward(d_output_f,
                     input_f,
                     gamma,
                     rstd_cache,
                     d_input_f,
                     d_gamma,
                     tokens,
                     d_model,
                     aligned_embed_dim);

    convert_float_to_int8(d_input_f, d_input, total);

    free_float_buffer(d_output_f);
    free_float_buffer(input_f);
    free_float_buffer(d_input_f);
}
