#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

void relu_forward_bf16(const uint16_t *input, uint16_t *output, size_t n)
{
    if (!input || !output) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        float x = bf16_to_float(input[i]);
        output[i] = float_to_bf16(x > 0.0f ? x : 0.0f);
    }
}

void relu_forward_inplace_bf16(uint16_t *data, size_t n)
{
    if (!data) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        float x = bf16_to_float(data[i]);
        data[i] = float_to_bf16(x > 0.0f ? x : 0.0f);
    }
}

void relu_backward_bf16(const uint16_t *input,
                        const uint16_t *d_output,
                        uint16_t *d_input,
                        size_t n)
{
    if (!input || !d_output || !d_input) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        float x = bf16_to_float(input[i]);
        float dy = bf16_to_float(d_output[i]);
        d_input[i] = float_to_bf16(x > 0.0f ? dy : 0.0f);
    }
}

