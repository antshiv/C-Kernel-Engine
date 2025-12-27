/**
 * @file gemm_kernels_q4k.c
 * @brief GEMM/GEMV kernels with Q4_K quantized weights
 *
 * Implements matrix multiplication where:
 *   - Activations (input): FP32
 *   - Weights: Q4_K (4.5 bits/weight, nested scales)
 *   - Output: FP32
 *
 * Key optimization: Fused dequantization - weights are dequantized in
 * registers and immediately used in FMA, never written to memory.
 *
 * Operations:
 *   - gemv_q4_k: Matrix-vector multiply (batch=1, token generation)
 *   - gemm_q4_k: Matrix-matrix multiply (batch>1, prefill)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ckernel_quant.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ============================================================================
 * GEMV: y = W @ x  (W is Q4_K, x and y are FP32)
 *
 * For token generation (batch=1), this is the critical path.
 * Memory-bound: we're loading ~4GB of weights for a 7B model per token.
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with Q4_K weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in Q4_K format [M x K], stored row-major
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns (must be multiple of 256)
 */
void gemv_q4_k_ref(float *y,
                   const void *W,
                   const float *x,
                   int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;  /* QK_K = 256 */

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = GGML_FP16_TO_FP32(block->d);
            const float dmin = GGML_FP16_TO_FP32(block->dmin);

            /* Unpack sub-block scales */
            uint8_t sc[8], m[8];
            unpack_q4_k_scales(block->scales, sc, m);

            /* Process 8 sub-blocks of 32 weights each */
            for (int sub = 0; sub < 8; sub++) {
                const float scale = d * (float)sc[sub];
                const float min_val = dmin * (float)m[sub];
                const uint8_t *qs = &block->qs[sub * 16];
                const float *xp = &x[b * QK_K + sub * 32];

                for (int i = 0; i < 16; i++) {
                    const uint8_t packed = qs[i];
                    const int8_t q0 = (packed & 0x0F) - 8;
                    const int8_t q1 = (packed >> 4) - 8;

                    const float w0 = scale * (float)q0 + min_val;
                    const float w1 = scale * (float)q1 + min_val;

                    sum += w0 * xp[2*i + 0];
                    sum += w1 * xp[2*i + 1];
                }
            }
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with Q4_K weights (AVX-512 optimized)
 *
 * Fused dequant + FMA: weights dequantized in ZMM registers, never touch RAM.
 */
void gemv_q4_k_avx512(float *y,
                      const void *W,
                      const float *x,
                      int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = GGML_FP16_TO_FP32(block->d);
            const float dmin = GGML_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m_arr[8];
            unpack_q4_k_scales(block->scales, sc, m_arr);

            /* Process 8 sub-blocks */
            for (int sub = 0; sub < 8; sub++) {
                const float scale = d * (float)sc[sub];
                const float min_val = dmin * (float)m_arr[sub];
                const __m512 vscale = _mm512_set1_ps(scale);
                const __m512 vmin = _mm512_set1_ps(min_val);
                const __m512i offset = _mm512_set1_epi32(8);
                const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                const uint8_t *qs = &block->qs[sub * 16];
                const float *xp = &x[b * QK_K + sub * 32];

                /* Load 16 bytes = 32 x 4-bit weights */
                __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                __m512i bytes = _mm512_cvtepu8_epi32(packed);

                /* Extract lower nibbles (weights at even indices) */
                __m512i lo = _mm512_and_epi32(bytes, mask_lo);
                lo = _mm512_sub_epi32(lo, offset);
                __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);

                /* Extract upper nibbles (weights at odd indices) */
                __m512i hi = _mm512_srli_epi32(bytes, 4);
                hi = _mm512_sub_epi32(hi, offset);
                __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                /* Load input vectors */
                /* Note: weights are interleaved (lo=even indices, hi=odd indices)
                 * We need to load x accordingly */
                __m512 x_even = _mm512_set_ps(
                    xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                    xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
                __m512 x_odd = _mm512_set_ps(
                    xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                    xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

                /* FMA: acc += w * x */
                acc = _mm512_fmadd_ps(w_lo, x_even, acc);
                acc = _mm512_fmadd_ps(w_hi, x_odd, acc);
            }
        }

        /* Horizontal sum */
        y[row] = _mm512_reduce_add_ps(acc);
    }
}
#endif /* __AVX512F__ */

/**
 * @brief Auto-dispatch GEMV based on available SIMD
 */
void gemv_q4_k(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_k_avx512(y, W, x, M, K);
#else
    gemv_q4_k_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * GEMM: Y = W @ X  (W is Q4_K, X and Y are FP32)
 *
 * For prefill (batch > 1), we can amortize weight loading across batch.
 * More compute-bound than GEMV.
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with Q4_K weights (scalar reference)
 *
 * @param Y Output matrix [M x N]
 * @param W Weight matrix in Q4_K format [M x K]
 * @param X Input matrix [K x N] (column-major for cache efficiency)
 * @param M Number of output rows
 * @param N Batch size (number of columns)
 * @param K Hidden dimension
 */
void gemm_q4_k_ref(float *Y,
                   const void *W,
                   const float *X,
                   int M, int N, int K)
{
    /* For each column in batch */
    for (int n = 0; n < N; n++) {
        gemv_q4_k_ref(&Y[n * M], W, &X[n * K], M, K);
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-matrix multiply with Q4_K weights (AVX-512)
 *
 * Processes multiple batch elements to improve weight reuse.
 */
void gemm_q4_k_avx512(float *Y,
                      const void *W,
                      const float *X,
                      int M, int N, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    /* Process 4 batch elements at a time for better register utilization */
    const int N4 = N / 4 * 4;

    for (int row = 0; row < M; row++) {
        /* Batch of 4 */
        for (int n = 0; n < N4; n += 4) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_K *block = &blocks[row * blocks_per_row + b];
                const float d = GGML_FP16_TO_FP32(block->d);
                const float dmin = GGML_FP16_TO_FP32(block->dmin);

                uint8_t sc[8], m_arr[8];
                unpack_q4_k_scales(block->scales, sc, m_arr);

                for (int sub = 0; sub < 8; sub++) {
                    const float scale = d * (float)sc[sub];
                    const float min_val = dmin * (float)m_arr[sub];
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const __m512 vmin = _mm512_set1_ps(min_val);
                    const __m512i offset = _mm512_set1_epi32(8);
                    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                    const uint8_t *qs = &block->qs[sub * 16];
                    const int x_offset = b * QK_K + sub * 32;

                    /* Dequantize weights (same for all batch elements) */
                    __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);

                    __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
                    __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

                    __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
                    __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                    /* Load inputs for 4 batch elements and accumulate */
                    /* (simplified - full impl would handle interleaving) */
                    for (int bn = 0; bn < 4; bn++) {
                        const float *xp = &X[(n + bn) * K + x_offset];

                        __m512 x_even = _mm512_set_ps(
                            xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                            xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
                        __m512 x_odd = _mm512_set_ps(
                            xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                            xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

                        __m512 *acc = (bn == 0) ? &acc0 : (bn == 1) ? &acc1 :
                                      (bn == 2) ? &acc2 : &acc3;
                        *acc = _mm512_fmadd_ps(w_lo, x_even, *acc);
                        *acc = _mm512_fmadd_ps(w_hi, x_odd, *acc);
                    }
                }
            }

            Y[(n + 0) * M + row] = _mm512_reduce_add_ps(acc0);
            Y[(n + 1) * M + row] = _mm512_reduce_add_ps(acc1);
            Y[(n + 2) * M + row] = _mm512_reduce_add_ps(acc2);
            Y[(n + 3) * M + row] = _mm512_reduce_add_ps(acc3);
        }

        /* Remainder */
        for (int n = N4; n < N; n++) {
            __m512 acc = _mm512_setzero_ps();

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_K *block = &blocks[row * blocks_per_row + b];
                const float d = GGML_FP16_TO_FP32(block->d);
                const float dmin = GGML_FP16_TO_FP32(block->dmin);

                uint8_t sc[8], m_arr[8];
                unpack_q4_k_scales(block->scales, sc, m_arr);

                for (int sub = 0; sub < 8; sub++) {
                    const float scale = d * (float)sc[sub];
                    const float min_val = dmin * (float)m_arr[sub];
                    const __m512 vscale = _mm512_set1_ps(scale);
                    const __m512 vmin = _mm512_set1_ps(min_val);
                    const __m512i offset = _mm512_set1_epi32(8);
                    const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                    const uint8_t *qs = &block->qs[sub * 16];
                    const float *xp = &X[n * K + b * QK_K + sub * 32];

                    __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                    __m512i bytes = _mm512_cvtepu8_epi32(packed);

                    __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
                    __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

                    __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
                    __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                    __m512 x_even = _mm512_set_ps(
                        xp[30], xp[28], xp[26], xp[24], xp[22], xp[20], xp[18], xp[16],
                        xp[14], xp[12], xp[10], xp[8], xp[6], xp[4], xp[2], xp[0]);
                    __m512 x_odd = _mm512_set_ps(
                        xp[31], xp[29], xp[27], xp[25], xp[23], xp[21], xp[19], xp[17],
                        xp[15], xp[13], xp[11], xp[9], xp[7], xp[5], xp[3], xp[1]);

                    acc = _mm512_fmadd_ps(w_lo, x_even, acc);
                    acc = _mm512_fmadd_ps(w_hi, x_odd, acc);
                }
            }

            Y[n * M + row] = _mm512_reduce_add_ps(acc);
        }
    }
}
#endif /* __AVX512F__ */

/**
 * @brief Auto-dispatch GEMM based on available SIMD
 */
void gemm_q4_k(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K)
{
#ifdef __AVX512F__
    gemm_q4_k_avx512(Y, W, X, M, N, K);
#else
    gemm_q4_k_ref(Y, W, X, M, N, K);
#endif
}

/* ============================================================================
 * Dot Product: Single row dot product with Q4_K weights
 * Used internally and for testing.
 * ============================================================================ */

/**
 * @brief Compute dot product of Q4_K row with FP32 vector
 *
 * @param w_q4k Q4_K blocks for one row
 * @param x FP32 input vector
 * @param K Vector length (must be multiple of 256)
 * @return Dot product result
 */
float dot_q4_k(const void *w_q4k, const float *x, int K)
{
    float result;
    gemv_q4_k(&result, w_q4k, x, 1, K);
    return result;
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 *
 * Given: dL/dY (gradient of loss w.r.t. output)
 * Compute: dL/dX = W^T @ dL/dY
 *
 * For quantized weights, we dequantize on-the-fly during backprop.
 * Weight gradients are NOT computed (weights are frozen).
 * For fine-tuning, use LoRA adapters which maintain FP32 gradients separately.
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient (scalar reference)
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in Q4_K format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_q4_k_backward_ref(float *dX,
                            const void *W,
                            const float *dY,
                            int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    /* Zero output gradient */
    memset(dX, 0, K * sizeof(float));

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m[8];
            unpack_q4_k_scales(block->scales, sc, m);

            for (int sub = 0; sub < 8; sub++) {
                const float scale = d * (float)sc[sub];
                const float min_val = dmin * (float)m[sub];
                const uint8_t *qs = &block->qs[sub * 16];
                float *dxp = &dX[b * QK_K + sub * 32];

                for (int i = 0; i < 16; i++) {
                    const uint8_t packed = qs[i];
                    const int8_t q0 = (packed & 0x0F) - 8;
                    const int8_t q1 = (packed >> 4) - 8;

                    const float w0 = scale * (float)q0 + min_val;
                    const float w1 = scale * (float)q1 + min_val;

                    dxp[2*i + 0] += w0 * dy;
                    dxp[2*i + 1] += w1 * dy;
                }
            }
        }
    }
}

#ifdef __AVX512F__
/**
 * @brief Backward pass with AVX-512
 */
void gemv_q4_k_backward_avx512(float *dX,
                               const void *W,
                               const float *dY,
                               int M, int K)
{
    const block_q4_K *blocks = (const block_q4_K *)W;
    const int blocks_per_row = K / QK_K;

    /* Zero output */
    memset(dX, 0, K * sizeof(float));

    for (int row = 0; row < M; row++) {
        const __m512 vdy = _mm512_set1_ps(dY[row]);

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_K *block = &blocks[row * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);

            uint8_t sc[8], m_arr[8];
            unpack_q4_k_scales(block->scales, sc, m_arr);

            for (int sub = 0; sub < 8; sub++) {
                const float scale = d * (float)sc[sub];
                const float min_val = dmin * (float)m_arr[sub];
                const __m512 vscale = _mm512_set1_ps(scale);
                const __m512 vmin = _mm512_set1_ps(min_val);
                const __m512i offset = _mm512_set1_epi32(8);
                const __m512i mask_lo = _mm512_set1_epi32(0x0F);

                const uint8_t *qs = &block->qs[sub * 16];
                float *dxp = &dX[b * QK_K + sub * 32];

                /* Dequantize weights */
                __m128i packed = _mm_loadu_si128((const __m128i *)qs);
                __m512i bytes = _mm512_cvtepu8_epi32(packed);

                __m512i lo = _mm512_sub_epi32(_mm512_and_epi32(bytes, mask_lo), offset);
                __m512i hi = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), offset);

                __m512 w_lo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), vscale, vmin);
                __m512 w_hi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), vscale, vmin);

                /* Compute gradients: dX += W * dY */
                __m512 grad_lo = _mm512_mul_ps(w_lo, vdy);
                __m512 grad_hi = _mm512_mul_ps(w_hi, vdy);

                /* Scatter to interleaved positions */
                float grad_lo_arr[16], grad_hi_arr[16];
                _mm512_storeu_ps(grad_lo_arr, grad_lo);
                _mm512_storeu_ps(grad_hi_arr, grad_hi);

                for (int i = 0; i < 16; i++) {
                    dxp[2*i + 0] += grad_lo_arr[i];
                    dxp[2*i + 1] += grad_hi_arr[i];
                }
            }
        }
    }
}
#endif

/**
 * @brief Auto-dispatch backward
 */
void gemv_q4_k_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int K)
{
#ifdef __AVX512F__
    gemv_q4_k_backward_avx512(dX, W, dY, M, K);
#else
    gemv_q4_k_backward_ref(dX, W, dY, M, K);
#endif
}

/**
 * @brief Batched backward pass
 */
void gemm_q4_k_backward(float *dX,
                        const void *W,
                        const float *dY,
                        int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_q4_k_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}
