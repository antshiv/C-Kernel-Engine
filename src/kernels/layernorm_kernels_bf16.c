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

void layernorm_forward_rolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                         const float *__restrict gamma,
                                         const float *__restrict beta,
                                         uint16_t *__restrict output_slice_base,
                                         float *__restrict mean_cache_slice,
                                         float *__restrict rstd_cache_slice,
                                         int num_tokens_in_slice,
                                         int d_model,
                                         int aligned_embed_dim,
                                         float eps)
{
    size_t total = (size_t)num_tokens_in_slice * (size_t)aligned_embed_dim;
    float *input_f = alloc_float_buffer(total);
    float *output_f = alloc_float_buffer(total);
    if (!input_f || !output_f) {
        free_float_buffer(input_f);
        free_float_buffer(output_f);
        return;
    }

    bf16_tensor_to_float(input_slice_base, input_f, total);
    layernorm_forward_rolled_slice(input_f, gamma, beta,
                                   output_f, mean_cache_slice, rstd_cache_slice,
                                   num_tokens_in_slice, d_model, aligned_embed_dim, eps);

    float_tensor_to_bf16(output_f, output_slice_base, total);

    free_float_buffer(input_f);
    free_float_buffer(output_f);
}

void layernorm_forward_unrolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                           const float *__restrict gamma,
                                           const float *__restrict beta,
                                           uint16_t *__restrict output_slice_base,
                                           float *__restrict mean_cache_slice,
                                           float *__restrict rstd_cache_slice,
                                           int num_tokens_in_slice,
                                           int d_model,
                                           float eps)
{
    size_t total = (size_t)num_tokens_in_slice * (size_t)d_model;
    float *input_f = alloc_float_buffer(total);
    float *output_f = alloc_float_buffer(total);
    if (!input_f || !output_f) {
        free_float_buffer(input_f);
        free_float_buffer(output_f);
        return;
    }

    bf16_tensor_to_float(input_slice_base, input_f, total);
    layernorm_forward_unrolled_slice(input_f, gamma, beta,
                                     output_f, mean_cache_slice, rstd_cache_slice,
                                     num_tokens_in_slice, d_model, eps);

    float_tensor_to_bf16(output_f, output_slice_base, total);

    free_float_buffer(input_f);
    free_float_buffer(output_f);
}

void layernorm_backward_kernel_bf16(const uint16_t *d_output,
                                    const uint16_t *input,
                                    const float *gamma,
                                    const float *mean,
                                    const float *rstd,
                                    uint16_t *d_input,
                                    float *d_gamma,
                                    float *d_beta,
                                    int tokens, int d_model, int aligned_embed_dim)
{
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

    bf16_tensor_to_float(d_output, d_output_f, total);
    bf16_tensor_to_float(input, input_f, total);

    layernorm_backward_kernel(d_output_f, input_f, gamma, mean, rstd,
                              d_input_f, d_gamma, d_beta,
                              tokens, d_model, aligned_embed_dim);

    float_tensor_to_bf16(d_input_f, d_input, total);

    free_float_buffer(d_output_f);
    free_float_buffer(input_f);
    free_float_buffer(d_input_f);
}
