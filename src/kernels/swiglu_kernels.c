#include "ckernel_engine.h"
#include <math.h>
#include <stddef.h>

// SwiGLU forward:
// Input layout per token:
//   gate:  input[t][0..D-1]
//   value: input[t][D..2D-1]
// Output:
//   y[t][d] = silu(gate[t][d]) * value[t][d]
//
// where silu(x) = x * sigmoid(x).
void swiglu_forward(const float *input,
                    float *output,
                    int tokens,
                    int dim)
{
    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        float *out_row = output + (size_t)t * D;

        for (int d = 0; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value

            float s = sigmoid_scalar(a);         // sigmoid(a)
            float silu = a * s;                  // silu(a) = a * sigmoid(a)

            out_row[d] = silu * b;
        }
    }
}

// SwiGLU backward:
// Given dY, X (gate+value), compute dX in same layout [gate_grad, value_grad].
//
// y = b * silu(a), where silu(a) = a * s, s = sigmoid(a)
// dy/da = b * silu'(a)
// dy/db = silu(a)
//
// silu'(a) = s + a * s * (1 - s)
void swiglu_backward(const float *input,
                     const float *d_output,
                     float *d_input,
                     int tokens,
                     int dim)
{
    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        const float *dy_row = d_output + (size_t)t * D;
        float *dx_row = d_input + (size_t)t * (2 * D);

        for (int d = 0; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value
            float dy = dy_row[d];

            float s = sigmoid_scalar(a);               // sigmoid(a)
            float silu = a * s;                       // silu(a)
            float s_prime = s * (1.0f - s);           // sigmoid'(a)
            float silu_prime = s + a * s_prime;       // silu'(a)

            float dA = dy * b * silu_prime;
            float dB = dy * silu;

            dx_row[d] = dA;
            dx_row[D + d] = dB;
        }
    }
}
