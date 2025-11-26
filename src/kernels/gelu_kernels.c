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

// Exact GELU backward using the tanh-based approximation derivative, adapted
// from C-Transformer's backward_gelu. Operates element-wise on contiguous
// buffers.
void gelu_backward_exact(const float *input,
                         const float *d_output,
                         float *d_input,
                         size_t n)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float x = input[i];

        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);

        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);

        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative =
            0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;

        d_input[i] = d_output[i] * gelu_derivative;
    }
}

// Fast approximate GELU backward, adapted from C-Transformer's backward_gelu_fast.
void gelu_backward_fast(const float *input,
                        const float *d_output,
                        float *d_input,
                        size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-1.702f * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * 1.702f);
        d_input[i] = d_output[i] * gelu_derivative;
    }
}

