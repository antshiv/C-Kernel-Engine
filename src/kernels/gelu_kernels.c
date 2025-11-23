#include <math.h>
#include <stddef.h>

// Fast GELU approximation, copied from C-Transformer's gelu_activation_token_parallel.
// Applies in-place to a contiguous buffer of length n.
void gelu_fast_inplace(float *data, size_t n)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

