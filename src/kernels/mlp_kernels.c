#include "bf16_utils.h"
#include "ckernel_engine.h"
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#include <omp.h>
#include <stdlib.h>

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

static float *convert_bf16_tensor(const uint16_t *src, size_t count)
{
    float *dst = (float *)malloc(count * sizeof(float));
    if (!dst) {
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
    return dst;
}

/* The BF16 wrappers keep a portable path: on hosts without native BF16/AMX,
 * we decode the packed values, run the familiar FP32 SIMD kernel, and re-encode
 * the results so the API still behaves like BF16. On tuned systems the registry
 * can instead point at a native `_bf16` kernel that uses BF16 intrinsics/tiles
 * directly for higher throughput, so both workflows coexist cleanly. */
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
    int D = aligned_dim;
    int fourD = 4 * D;

    size_t input_elems = (size_t)T * (size_t)aligned_dim;
    size_t w1_elems = (size_t)fourD * (size_t)aligned_dim;
    size_t b1_elems = (size_t)fourD;
    size_t w2_elems = (size_t)aligned_dim * (size_t)fourD;
    size_t b2_elems = (size_t)aligned_dim;

    float *input_f = convert_bf16_tensor(input, input_elems);
    if (!input_f) {
        return;
    }
    float *W1_f = convert_bf16_tensor(W_fc1, w1_elems);
    if (!W1_f) {
        free(input_f);
        return;
    }
    float *b1_f = convert_bf16_tensor(b_fc1, b1_elems);
    if (!b1_f) {
        free(input_f);
        free(W1_f);
        return;
    }
    float *W2_f = convert_bf16_tensor(W_fc2, w2_elems);
    if (!W2_f) {
        free(input_f);
        free(W1_f);
        free(b1_f);
        return;
    }
    float *b2_f = convert_bf16_tensor(b_fc2, b2_elems);
    if (!b2_f) {
        free(input_f);
        free(W1_f);
        free(b1_f);
        free(W2_f);
        return;
    }

    mlp_token_parallel(input_f, W1_f, b1_f, W2_f, b2_f,
                       fc1_output, output, T, aligned_dim, num_threads);

    free(input_f);
    free(W1_f);
    free(b1_f);
    free(W2_f);
    free(b2_f);
}

// Generic FC2 backward kernel adapted from C-Transformer's backward_fc2_feature_parallel.
// Now uses shared GEMM kernels for d_input and d_W computation.
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
    (void)num_threads;  // Threading handled by GEMM kernels

    // 1. d_input[T, in] = d_output[T, out] @ W[out, in]
    // Using gemm_nn: C[M,N] = A[M,K] @ B[K,N]
    // A = d_output [T, out], B = W [out, in], C = d_input [T, in]
    // M = T, N = aligned_in, K = aligned_out
    gemm_nn_avx512(d_output, W_fc2, NULL, d_input,
                   T, aligned_in, aligned_out);

    // 2. d_W[out, in] = d_output[T, out].T @ fc2_input[T, in]
    // Using gemm_tn: C[M,N] = A[K,M].T @ B[K,N]
    // A = d_output [T, out] (stored as [K=T, M=out]), B = fc2_input [T, in]
    // C = d_W [out, in], M = aligned_out, N = aligned_in, K = T
    // Note: gemm_tn overwrites, so we need to save and add if accumulating
    // For now, assume d_W starts zeroed (gradient accumulation handled at higher level)
    gemm_tn_avx512(d_output, fc2_input, NULL, d_W_fc2,
                   aligned_out, aligned_in, T);

    // 3. d_b_fc2 = sum_over_T(d_output)
#pragma omp parallel for schedule(static)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; ++t) {
            bias_grad += d_output[(size_t)t * aligned_out + out_idx];
        }
        d_b_fc2[out_idx] += bias_grad;
    }
}

// Generic FC1 backward kernel adapted from C-Transformer's backward_fc1_feature_parallel.
// Now uses shared GEMM kernels for d_input and d_W computation.
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
    (void)num_threads;  // Threading handled by GEMM kernels

    // 1. d_input[T, in] = d_output[T, out] @ W[out, in]
    // Using gemm_nn: C[M,N] = A[M,K] @ B[K,N]
    // A = d_output [T, out], B = W [out, in], C = d_input [T, in]
    // M = T, N = aligned_in, K = aligned_out
    gemm_nn_avx512(d_output, W_fc1, NULL, d_input,
                   T, aligned_in, aligned_out);

    // 2. d_W[out, in] = d_output[T, out].T @ fc1_input[T, in]
    // Using gemm_tn: C[M,N] = A[K,M].T @ B[K,N]
    // A = d_output [T, out] (stored as [K=T, M=out]), B = fc1_input [T, in]
    // C = d_W [out, in], M = aligned_out, N = aligned_in, K = T
    gemm_tn_avx512(d_output, fc1_input, NULL, d_W_fc1,
                   aligned_out, aligned_in, T);

    // 3. d_b_fc1 = sum_over_T(d_output)
#pragma omp parallel for schedule(static)
    for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; ++t) {
            bias_grad += d_output[(size_t)t * aligned_out + out_idx];
        }
        d_b_fc1[out_idx] += bias_grad;
    }
}
