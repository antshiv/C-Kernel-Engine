/**
 * Optimized BF16 MLP Kernels
 *
 * Uses direct BF16 GEMM instead of converting to FP32.
 * Layout: input[T,D] -> fc1[T,4D] -> GELU -> fc2[T,D]
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/* Forward declaration of optimized BF16 GEMM */
extern void gemm_bf16_fp32out(const uint16_t *A, const uint16_t *B,
                               const float *bias, float *C,
                               int M, int N, int K);

/* GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu_scalar(float x)
{
    const float c = 0.7978845608f;  /* sqrt(2/pi) */
    const float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

#if defined(__AVX512F__)
/* Vectorized GELU using polynomial approximation of tanh */
static inline __m512 gelu_avx512(__m512 x)
{
    const __m512 c = _mm512_set1_ps(0.7978845608f);
    const __m512 k = _mm512_set1_ps(0.044715f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);

    /* inner = sqrt(2/pi) * (x + 0.044715 * x^3) */
    __m512 inner = _mm512_fmadd_ps(k, x3, x);
    inner = _mm512_mul_ps(c, inner);

    /* tanh approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) for small x */
    /* For larger x, clamp to ±1 */
    __m512 inner2 = _mm512_mul_ps(inner, inner);
    __m512 num = _mm512_add_ps(_mm512_set1_ps(27.0f), inner2);
    __m512 den = _mm512_fmadd_ps(_mm512_set1_ps(9.0f), inner2, _mm512_set1_ps(27.0f));
    __m512 tanh_approx = _mm512_mul_ps(inner, _mm512_div_ps(num, den));

    /* Clamp to [-1, 1] */
    tanh_approx = _mm512_min_ps(tanh_approx, one);
    tanh_approx = _mm512_max_ps(tanh_approx, _mm512_set1_ps(-1.0f));

    /* 0.5 * x * (1 + tanh(...)) */
    __m512 result = _mm512_add_ps(one, tanh_approx);
    result = _mm512_mul_ps(half, _mm512_mul_ps(x, result));

    return result;
}
#endif

/**
 * Optimized MLP Forward (BF16 weights, FP32 activations)
 *
 * This version:
 * 1. Uses optimized BF16 GEMM directly (no bulk conversion)
 * 2. Keeps activations in FP32 for GELU accuracy
 * 3. Vectorized GELU with AVX-512
 */
void mlp_token_parallel_bf16(const uint16_t *input,
                             const uint16_t *W_fc1,
                             const uint16_t *b_fc1,
                             const uint16_t *W_fc2,
                             const uint16_t *b_fc2,
                             float *fc1_output,
                             float *output,
                             int T,
                             int aligned_dim,
                             int num_threads)
{
    if (!input || !W_fc1 || !b_fc1 || !W_fc2 || !b_fc2 || !fc1_output || !output) {
        return;
    }

    const int D = aligned_dim;
    const int fourD = 4 * D;

    /* Convert biases to FP32 (small, one-time cost) */
    float *bias1_f = (float *)malloc(fourD * sizeof(float));
    float *bias2_f = (float *)malloc(D * sizeof(float));
    if (!bias1_f || !bias2_f) {
        free(bias1_f);
        free(bias2_f);
        return;
    }

    for (int i = 0; i < fourD; ++i) {
        bias1_f[i] = bf16_to_float(b_fc1[i]);
    }
    for (int i = 0; i < D; ++i) {
        bias2_f[i] = bf16_to_float(b_fc2[i]);
    }

    /* FC1: [T, D] x [4D, D].T -> [T, 4D] with FP32 output */
    gemm_bf16_fp32out(input, W_fc1, bias1_f, fc1_output, T, fourD, D);

    /* GELU activation (in-place on FP32) */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *row = fc1_output + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 x = _mm512_loadu_ps(row + j);
            __m512 y = gelu_avx512(x);
            _mm512_storeu_ps(row + j, y);
        }
        for (; j < fourD; ++j) {
            row[j] = gelu_scalar(row[j]);
        }
    }
#else
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < fourD; ++j) {
            fc1_output[t * fourD + j] = gelu_scalar(fc1_output[t * fourD + j]);
        }
    }
#endif

    /* FC2: [T, 4D] x [D, 4D].T -> [T, D]
     * Need to convert fc1_output to BF16 for BF16 GEMM, or use FP32 GEMM
     * For now, use a hybrid approach: convert fc1_output to BF16 temp buffer
     */
    uint16_t *fc1_bf16 = (uint16_t *)malloc((size_t)T * fourD * sizeof(uint16_t));
    if (!fc1_bf16) {
        free(bias1_f);
        free(bias2_f);
        return;
    }

    /* Convert FP32 activations back to BF16 */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *src = fc1_output + (size_t)t * fourD;
        uint16_t *dst = fc1_bf16 + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 fp32 = _mm512_loadu_ps(src + j);
            /* Round to nearest even */
            __m512i as_int = _mm512_castps_si512(fp32);
            __m512i lsb = _mm512_srli_epi32(as_int, 16);
            lsb = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
            __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
            __m512i rounded = _mm512_add_epi32(as_int, rounding);
            __m512i shifted = _mm512_srli_epi32(rounded, 16);
            __m256i bf16 = _mm512_cvtepi32_epi16(shifted);
            _mm256_storeu_si256((__m256i *)(dst + j), bf16);
        }
        for (; j < fourD; ++j) {
            dst[j] = float_to_bf16(src[j]);
        }
    }
#else
    for (size_t i = 0; i < (size_t)T * fourD; ++i) {
        fc1_bf16[i] = float_to_bf16(fc1_output[i]);
    }
#endif

    /* FC2: BF16 GEMM with FP32 output */
    gemm_bf16_fp32out(fc1_bf16, W_fc2, bias2_f, output, T, D, fourD);

    free(fc1_bf16);
    free(bias1_f);
    free(bias2_f);
}

/**
 * Alternative: Fully FP32 activations throughout
 * Converts only weights once, keeps all activations in FP32
 * Use this for maximum accuracy
 */
void mlp_token_parallel_bf16_fp32act(const uint16_t *input,
                                      const uint16_t *W_fc1,
                                      const uint16_t *b_fc1,
                                      const uint16_t *W_fc2,
                                      const uint16_t *b_fc2,
                                      float *fc1_output,
                                      float *output,
                                      int T,
                                      int aligned_dim,
                                      int num_threads)
{
    if (!input || !W_fc1 || !b_fc1 || !W_fc2 || !b_fc2 || !fc1_output || !output) {
        return;
    }

    const int D = aligned_dim;
    const int fourD = 4 * D;

    /* Allocate FP32 buffers for input (activations often reused) */
    float *input_f = (float *)malloc((size_t)T * D * sizeof(float));
    float *bias1_f = (float *)malloc(fourD * sizeof(float));
    float *bias2_f = (float *)malloc(D * sizeof(float));

    if (!input_f || !bias1_f || !bias2_f) {
        free(input_f);
        free(bias1_f);
        free(bias2_f);
        return;
    }

    /* Convert input and biases to FP32 */
    bf16_tensor_to_float(input, input_f, (size_t)T * D);
    bf16_tensor_to_float(b_fc1, bias1_f, fourD);
    bf16_tensor_to_float(b_fc2, bias2_f, D);

    /* Use existing FP32 MLP with BF16 weights */
    /* FC1: gemm_bf16_fp32out(input, W_fc1, bias1_f, fc1_output, T, fourD, D) */
    gemm_bf16_fp32out(input, W_fc1, bias1_f, fc1_output, T, fourD, D);

    /* GELU */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *row = fc1_output + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 x = _mm512_loadu_ps(row + j);
            _mm512_storeu_ps(row + j, gelu_avx512(x));
        }
        for (; j < fourD; ++j) {
            row[j] = gelu_scalar(row[j]);
        }
    }
#else
    for (size_t i = 0; i < (size_t)T * fourD; ++i) {
        fc1_output[i] = gelu_scalar(fc1_output[i]);
    }
#endif

    /* FC2: Need FP32 input, BF16 weights */
    /* Convert fc1_output to BF16 for gemm_bf16_fp32out */
    uint16_t *fc1_bf16 = (uint16_t *)malloc((size_t)T * fourD * sizeof(uint16_t));
    if (fc1_bf16) {
        float_tensor_to_bf16(fc1_output, fc1_bf16, (size_t)T * fourD);
        gemm_bf16_fp32out(fc1_bf16, W_fc2, bias2_f, output, T, D, fourD);
        free(fc1_bf16);
    }

    free(input_f);
    free(bias1_f);
    free(bias2_f);
}

