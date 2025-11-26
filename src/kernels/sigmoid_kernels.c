#include <math.h>
#include <stddef.h>

// Core sigmoid scalar kernel.
float sigmoid_scalar(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Vectorized (loop) sigmoid forward over a contiguous buffer.
void sigmoid_forward(const float *input,
                     float *output,
                     size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        output[i] = sigmoid_scalar(input[i]);
    }
}

// Sigmoid backward over a contiguous buffer:
// Given dY and X, compute dX = dY * s * (1 - s),
// where s = sigmoid(X).
void sigmoid_backward(const float *input,
                      const float *d_output,
                      float *d_input,
                      size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float s = sigmoid_scalar(x);
        float s_prime = s * (1.0f - s);
        d_input[i] = d_output[i] * s_prime;
    }
}

