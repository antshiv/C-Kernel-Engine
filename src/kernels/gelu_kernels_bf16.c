#include <stdlib.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

static float *alloc_float_buffer(size_t count)
{
    return (float *)malloc(count * sizeof(float));
}

static void free_float_buffer(float *buf)
{
    free(buf);
}

void gelu_fast_inplace_bf16(uint16_t *data, size_t n)
{
    float *tmp = alloc_float_buffer(n);
    if (!tmp) {
        return;
    }

    bf16_tensor_to_float(data, tmp, n);
    // Use exact version to avoid fast tanh approximation error accumulating
    // with BF16 precision loss. Conversion overhead dominates anyway.
    gelu_exact_inplace(tmp, n);
    float_tensor_to_bf16(tmp, data, n);

    free_float_buffer(tmp);
}

void gelu_backward_exact_bf16(const uint16_t *input,
                              const uint16_t *d_output,
                              uint16_t *d_input,
                              size_t n)
{
    float *input_f = alloc_float_buffer(n);
    float *d_output_f = alloc_float_buffer(n);
    float *d_input_f = alloc_float_buffer(n);
    if (!input_f || !d_output_f || !d_input_f) {
        free_float_buffer(input_f);
        free_float_buffer(d_output_f);
        free_float_buffer(d_input_f);
        return;
    }

    bf16_tensor_to_float(input, input_f, n);
    bf16_tensor_to_float(d_output, d_output_f, n);

    // Use scalar exact version to avoid fast tanh approximation error
    // accumulating with BF16 precision loss.
    gelu_backward_scalar(input_f, d_output_f, d_input_f, n);

    float_tensor_to_bf16(d_input_f, d_input, n);

    free_float_buffer(input_f);
    free_float_buffer(d_output_f);
    free_float_buffer(d_input_f);
}

void gelu_backward_fast_bf16(const uint16_t *input,
                             const uint16_t *d_output,
                             uint16_t *d_input,
                             size_t n)
{
    float *input_f = alloc_float_buffer(n);
    float *d_output_f = alloc_float_buffer(n);
    float *d_input_f = alloc_float_buffer(n);
    if (!input_f || !d_output_f || !d_input_f) {
        free_float_buffer(input_f);
        free_float_buffer(d_output_f);
        free_float_buffer(d_input_f);
        return;
    }

    bf16_tensor_to_float(input, input_f, n);
    bf16_tensor_to_float(d_output, d_output_f, n);

    gelu_backward_fast(input_f, d_output_f, d_input_f, n);

    float_tensor_to_bf16(d_input_f, d_input, n);

    free_float_buffer(input_f);
    free_float_buffer(d_output_f);
    free_float_buffer(d_input_f);
}
