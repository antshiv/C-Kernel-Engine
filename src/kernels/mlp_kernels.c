#include "ckernel_engine.h"
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#include <omp.h>

// Forward MLP kernel (FC1 -> GELU -> FC2) adapted from C-Transformer's
// mlp_token_parallel but expressed in a model-agnostic form. We keep the
// familiar name `mlp_token_parallel` for reuse during decode/inference.
//
// Shapes:
//   input:       [T × D]             (row-major, stride = aligned_dim)
//   W_fc1:       [4D × D]            (row-major, stored as [out × in])
//   b_fc1:       [4D]
//   W_fc2:       [D × 4D]
//   b_fc2:       [D]
//   fc1_output:  [T × 4D]            (workspace, also becomes GELU input/output)
//   output:      [T × D]
//
// D is typically `aligned_dim` in your transformer; pass that value here.
void mlp_token_parallel(const float *input,
                        const float *W_fc1,
                        const float *b_fc1,
                        const float *W_fc2,
                        const float *b_fc2,
                        float *fc1_output,
                        float *output,
                        int T,
                        int aligned_dim,
                        int num_threads)
{
    int D = aligned_dim;
    int fourD = 4 * D;

    // FC1: [T × D] · [D × 4D] -> [T × 4D]
    // Our GEMM layout: A[M×K], B[N×K], so B is [4D × D].
    gemm_blocked_serial(input, W_fc1, b_fc1,
                        fc1_output,
                        T,      // M
                        fourD,  // N
                        D);     // K

    // GELU in-place on FC1 output
    gelu_fast_inplace(fc1_output, (size_t)T * (size_t)fourD);

    // FC2: [T × 4D] · [4D × D] -> [T × D]
    gemm_blocked_serial(fc1_output, W_fc2, b_fc2,
                        output,
                        T,  // M
                        D,  // N
                        fourD); // K
}

// Generic FC2 backward kernel adapted from C-Transformer's backward_fc2_feature_parallel.
// Shapes:
//   d_output:   [T × aligned_out]
//   fc2_input:  [T × aligned_in]
//   W_fc2:      [aligned_out × aligned_in] (row-major)
//   d_input:    [T × aligned_in]
//   d_W_fc2:    [aligned_out × aligned_in] (accumulated)
//   d_b_fc2:    [aligned_out] (accumulated)
void fc2_backward_kernel(const float *d_output,
                         const float *fc2_input,
                         const float *W_fc2,
                         float *d_input,
                         float *d_W_fc2,
                         float *d_b_fc2,
                         int T,
                         int aligned_in,
                         int aligned_out,
                         int num_threads)
{
    // 1. d_input = d_output @ W_fc2^T
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < T; ++t) {
        const float *d_out_row = d_output + (size_t)t * aligned_out;
        float *d_in_row = d_input + (size_t)t * aligned_in;

        for (int in_idx = 0; in_idx < aligned_in; ++in_idx) {
            float sum = 0.0f;
            for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
                sum += d_out_row[out_idx] * W_fc2[out_idx * aligned_in + in_idx];
            }
            d_in_row[in_idx] = sum;
        }
    }

    // 2. d_W_fc2 = d_output^T @ fc2_input  (feature-parallel, vectorized)
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float *dst_row = d_W_fc2 + (size_t)out_idx * aligned_in;

#if defined(__AVX512F__)
        for (int in_idx = 0; in_idx < aligned_in; in_idx += 16) {
            __m512 accum = _mm512_setzero_ps();

            for (int t = 0; t < T; ++t) {
                __m512 input_vec = _mm512_load_ps(fc2_input + (size_t)t * aligned_in + in_idx);
                __m512 grad_broadcast =
                    _mm512_set1_ps(d_output[(size_t)t * aligned_out + out_idx]);
                accum = _mm512_fmadd_ps(grad_broadcast, input_vec, accum);
            }

            __m512 prev = _mm512_load_ps(dst_row + in_idx);
            _mm512_store_ps(dst_row + in_idx, _mm512_add_ps(prev, accum));
        }
#elif defined(__AVX2__) || defined(__AVX__)
        for (int in_idx = 0; in_idx < aligned_in; in_idx += 8) {
            __m256 accum = _mm256_setzero_ps();

            for (int t = 0; t < T; ++t) {
                __m256 input_vec = _mm256_load_ps(fc2_input + (size_t)t * aligned_in + in_idx);
                __m256 grad_broadcast =
                    _mm256_set1_ps(d_output[(size_t)t * aligned_out + out_idx]);
#if defined(__FMA__)
                accum = _mm256_fmadd_ps(grad_broadcast, input_vec, accum);
#else
                accum = _mm256_add_ps(accum, _mm256_mul_ps(grad_broadcast, input_vec));
#endif
            }

            __m256 prev = _mm256_load_ps(dst_row + in_idx);
            _mm256_store_ps(dst_row + in_idx, _mm256_add_ps(prev, accum));
        }
#else
        for (int in_idx = 0; in_idx < aligned_in; ++in_idx) {
            float accum = 0.0f;
            for (int t = 0; t < T; ++t) {
                accum += d_output[(size_t)t * aligned_out + out_idx]
                       * fc2_input[(size_t)t * aligned_in + in_idx];
            }
            dst_row[in_idx] += accum;
        }
#endif
    }

    // 3. d_b_fc2 = sum_over_T(d_output)
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; ++t) {
            bias_grad += d_output[(size_t)t * aligned_out + out_idx];
        }
        d_b_fc2[out_idx] += bias_grad;
    }
}

// Generic FC1 backward kernel adapted from C-Transformer's backward_fc1_feature_parallel.
// Shapes:
//   d_output:   [T × aligned_out]
//   fc1_input:  [T × aligned_in]
//   W_fc1:      [aligned_out × aligned_in] (row-major)
//   d_input:    [T × aligned_in]
//   d_W_fc1:    [aligned_out × aligned_in] (accumulated)
//   d_b_fc1:    [aligned_out] (accumulated)
void fc1_backward_kernel(const float *d_output,
                         const float *fc1_input,
                         const float *W_fc1,
                         float *d_input,
                         float *d_W_fc1,
                         float *d_b_fc1,
                         int T,
                         int aligned_in,
                         int aligned_out,
                         int num_threads)
{
    // 1. d_input = d_output @ W_fc1^T
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < T; ++t) {
        const float *d_out_row = d_output + (size_t)t * aligned_out;
        float *d_in_row = d_input + (size_t)t * aligned_in;

        for (int in_idx = 0; in_idx < aligned_in; ++in_idx) {
            float sum = 0.0f;
            for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
                sum += d_out_row[out_idx] * W_fc1[out_idx * aligned_in + in_idx];
            }
            d_in_row[in_idx] = sum;
        }
    }

    // 2. d_W_fc1 = d_output^T @ fc1_input
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float *dst_row = d_W_fc1 + (size_t)out_idx * aligned_in;

#if defined(__AVX512F__)
        for (int in_idx = 0; in_idx < aligned_in; in_idx += 16) {
            __m512 accum = _mm512_setzero_ps();
            for (int t = 0; t < T; ++t) {
                __m512 input_vec = _mm512_load_ps(fc1_input + (size_t)t * aligned_in + in_idx);
                __m512 grad_broadcast =
                    _mm512_set1_ps(d_output[(size_t)t * aligned_out + out_idx]);
                accum = _mm512_fmadd_ps(grad_broadcast, input_vec, accum);
            }
            __m512 prev = _mm512_load_ps(dst_row + in_idx);
            _mm512_store_ps(dst_row + in_idx, _mm512_add_ps(prev, accum));
        }
#elif defined(__AVX2__) || defined(__AVX__)
        for (int in_idx = 0; in_idx < aligned_in; in_idx += 8) {
            __m256 accum = _mm256_setzero_ps();
            for (int t = 0; t < T; ++t) {
                __m256 input_vec = _mm256_load_ps(fc1_input + (size_t)t * aligned_in + in_idx);
                __m256 grad_broadcast =
                    _mm256_set1_ps(d_output[(size_t)t * aligned_out + out_idx]);
#if defined(__FMA__)
                accum = _mm256_fmadd_ps(grad_broadcast, input_vec, accum);
#else
                accum = _mm256_add_ps(accum, _mm256_mul_ps(grad_broadcast, input_vec));
#endif
            }
            __m256 prev = _mm256_load_ps(dst_row + in_idx);
            _mm256_store_ps(dst_row + in_idx, _mm256_add_ps(prev, accum));
        }
#else
        for (int in_idx = 0; in_idx < aligned_in; ++in_idx) {
            float accum = 0.0f;
            for (int t = 0; t < T; ++t) {
                accum += d_output[(size_t)t * aligned_out + out_idx]
                       * fc1_input[(size_t)t * aligned_in + in_idx];
            }
            dst_row[in_idx] += accum;
        }
#endif
    }

    // 3. d_b_fc1 = sum_over_T(d_output)
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; ++t) {
            bias_grad += d_output[(size_t)t * aligned_out + out_idx];
        }
        d_b_fc1[out_idx] += bias_grad;
    }
}
