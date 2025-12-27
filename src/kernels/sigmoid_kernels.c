#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif


// Core sigmoid scalar kernel.
float sigmoid_scalar(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

#if defined(__AVX512F__)
static inline __m512 sigmoid_avx512_vec(__m512 x)
{
    __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg = _mm512_exp_ps(neg);
    __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg);
    return _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
}

static void sigmoid_forward_avx512(const float *input,
                                   float *output,
                                   size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 in_vec = _mm512_loadu_ps(input + i);
        __m512 sig = sigmoid_avx512_vec(in_vec);
        _mm512_storeu_ps(output + i, sig);
    }

    for (; i < n; ++i) {
        output[i] = sigmoid_scalar(input[i]);
    }
}

static void sigmoid_backward_avx512(const float *input,
                                    const float *d_output,
                                    float *d_input,
                                    size_t n)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 in_vec = _mm512_loadu_ps(input + i);
        __m512 s = sigmoid_avx512_vec(in_vec);
        __m512 dout = _mm512_loadu_ps(d_output + i);
        __m512 grad = _mm512_mul_ps(_mm512_mul_ps(s, _mm512_sub_ps(one, s)), dout);
        _mm512_storeu_ps(d_input + i, grad);
    }

    for (; i < n; ++i) {
        float x = input[i];
        float s = sigmoid_scalar(x);
        float s_prime = s * (1.0f - s);
        d_input[i] = d_output[i] * s_prime;
    }
}
#endif

// Vectorized (loop) sigmoid forward over a contiguous buffer.
void sigmoid_forward(const float *input,
                     float *output,
                     size_t n)
{
#if defined(__AVX512F__)
    sigmoid_forward_avx512(input, output, n);
#else
    for (size_t i = 0; i < n; ++i) {
        output[i] = sigmoid_scalar(input[i]);
    }
#endif
}

// Sigmoid backward over a contiguous buffer:
// Given dY and X, compute dX = dY * s * (1 - s),
// where s = sigmoid(X).
void sigmoid_backward(const float *input,
                      const float *d_output,
                      float *d_input,
                      size_t n)
{
#if defined(__AVX512F__)
    sigmoid_backward_avx512(input, d_output, d_input, n);
#else
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float s = sigmoid_scalar(x);
        float s_prime = s * (1.0f - s);
        d_input[i] = d_output[i] * s_prime;
    }
#endif
}
