#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

static float *alloc_float_buffer(size_t count)
{
    return (float *)malloc(count * sizeof(float));
}

void sigmoid_forward_bf16(const uint16_t *input,
                          uint16_t *output,
                          size_t n)
{
    if (!input || !output || n == 0) {
        return;
    }

    float *input_f = alloc_float_buffer(n);
    float *output_f = alloc_float_buffer(n);
    if (!input_f || !output_f) {
        free(input_f);
        free(output_f);
        return;
    }

    bf16_tensor_to_float(input, input_f, n);
    sigmoid_forward(input_f, output_f, n);
    float_tensor_to_bf16(output_f, output, n);

    free(input_f);
    free(output_f);
}

void sigmoid_backward_bf16(const uint16_t *input,
                           const uint16_t *d_output,
                           uint16_t *d_input,
                           size_t n)
{
    if (!input || !d_output || !d_input || n == 0) {
        return;
    }

    float *input_f = alloc_float_buffer(n);
    float *d_output_f = alloc_float_buffer(n);
    float *d_input_f = alloc_float_buffer(n);
    if (!input_f || !d_output_f || !d_input_f) {
        free(input_f);
        free(d_output_f);
        free(d_input_f);
        return;
    }

    bf16_tensor_to_float(input, input_f, n);
    bf16_tensor_to_float(d_output, d_output_f, n);
    sigmoid_backward(input_f, d_output_f, d_input_f, n);
    float_tensor_to_bf16(d_input_f, d_input, n);

    free(input_f);
    free(d_output_f);
    free(d_input_f);
}

