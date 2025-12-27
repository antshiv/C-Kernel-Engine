#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

void swiglu_forward_bf16(const uint16_t *input,
                         uint16_t *output,
                         int tokens,
                         int dim)
{
    if (!input || !output || tokens <= 0 || dim <= 0) {
        return;
    }

    const int T = tokens;
    const int D = dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *row = input + (size_t)t * (size_t)(2 * D);
        uint16_t *out_row = output + (size_t)t * (size_t)D;

        for (int d = 0; d < D; ++d) {
            float a = bf16_to_float(row[d]);
            float b = bf16_to_float(row[D + d]);
            float s = sigmoid_scalar(a);
            float silu = a * s;
            out_row[d] = float_to_bf16(silu * b);
        }
    }
}

void swiglu_backward_bf16(const uint16_t *input,
                          const uint16_t *d_output,
                          uint16_t *d_input,
                          int tokens,
                          int dim)
{
    if (!input || !d_output || !d_input || tokens <= 0 || dim <= 0) {
        return;
    }

    const int T = tokens;
    const int D = dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *row = input + (size_t)t * (size_t)(2 * D);
        const uint16_t *dy_row = d_output + (size_t)t * (size_t)D;
        uint16_t *dx_row = d_input + (size_t)t * (size_t)(2 * D);

        for (int d = 0; d < D; ++d) {
            float a = bf16_to_float(row[d]);
            float b = bf16_to_float(row[D + d]);
            float dy = bf16_to_float(dy_row[d]);

            float s = sigmoid_scalar(a);
            float silu = a * s;
            float s_prime = s * (1.0f - s);
            float silu_prime = s + a * s_prime;

            float dA = dy * b * silu_prime;
            float dB = dy * silu;

            dx_row[d] = float_to_bf16(dA);
            dx_row[D + d] = float_to_bf16(dB);
        }
    }
}

