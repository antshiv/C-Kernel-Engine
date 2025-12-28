/**
 * GEMM Microkernel - High-Performance Register-Blocked Matrix Multiplication
 *
 * This file implements optimized GEMM microkernels inspired by oneDNN/BLIS.
 * The key insight: keep all accumulator values in registers across the K loop.
 *
 * Architecture:
 * 1. Microkernel: Fixed-size tile computed entirely in registers
 *    - AVX-512: 6x32 (uses 24 ZMM registers for accumulators)
 *    - AVX2: 6x16 (uses 12 YMM registers for accumulators)
 * 2. Cache blocking: Auto-tuned based on detected CPU cache sizes
 * 3. Packing: Parallel A/B packing for optimal memory access
 * 4. Threading: 2D thread partitioning across M and N
 *
 * Reference: oneDNN BRGEMM, BLIS framework
 *
 * Layout: C[M,N] = A[M,K] @ B[K,N] (row-major)
 */

#include "ckernel_engine.h"
#include "cpu_features.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Microkernel Configuration
//
// MR/NR are fixed at compile time (microkernel register usage)
// MC/NC/KC are determined at runtime based on detected cache sizes
// =============================================================================

#if defined(__AVX512F__)
    // AVX-512: 6x32 microkernel (6 rows, 32 cols = 2 ZMM per row = 12 ZMM accumulators)
    #define MR_FIXED 6
    #define NR_FIXED 32
#elif defined(__AVX2__) || defined(__AVX__)
    // AVX/AVX2: 6x16 microkernel
    #define MR_FIXED 6
    #define NR_FIXED 16
#else
    // Scalar fallback
    #define MR_FIXED 4
    #define NR_FIXED 4
#endif

// These macros use runtime-detected values (initialized once at startup)
#define MR (MR_FIXED)
#define NR (NR_FIXED)
#define MC (get_gemm_params()->MC)
#define NC (get_gemm_params()->NC)
#define KC (get_gemm_params()->KC)

// =============================================================================
// AVX-512 6x32 Microkernel - oneDNN style
//
// Computes: C[0:6, 0:32] += A[0:6, 0:K] @ B[0:K, 0:32]
//
// Register usage (32 ZMM registers available):
// - c0_lo, c0_hi, c1_lo, c1_hi, ... c5_hi: 12 accumulators (2 per row)
// - b_lo, b_hi: 2 registers for B row
// - a0-a5: 6 registers for A broadcasts
// - Remaining for prefetch and temp
// =============================================================================

#if defined(__AVX512F__)
static inline void gemm_microkernel_6x32_avx512(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // 12 accumulators: 6 rows x 2 ZMM (32 floats) per row
    __m512 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m512 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm512_setzero_ps(); c0_hi = _mm512_setzero_ps();
        c1_lo = _mm512_setzero_ps(); c1_hi = _mm512_setzero_ps();
        c2_lo = _mm512_setzero_ps(); c2_hi = _mm512_setzero_ps();
        c3_lo = _mm512_setzero_ps(); c3_hi = _mm512_setzero_ps();
        c4_lo = _mm512_setzero_ps(); c4_hi = _mm512_setzero_ps();
        c5_lo = _mm512_setzero_ps(); c5_hi = _mm512_setzero_ps();
    } else {
        c0_lo = _mm512_loadu_ps(&C[0 * ldc]);      c0_hi = _mm512_loadu_ps(&C[0 * ldc + 16]);
        c1_lo = _mm512_loadu_ps(&C[1 * ldc]);      c1_hi = _mm512_loadu_ps(&C[1 * ldc + 16]);
        c2_lo = _mm512_loadu_ps(&C[2 * ldc]);      c2_hi = _mm512_loadu_ps(&C[2 * ldc + 16]);
        c3_lo = _mm512_loadu_ps(&C[3 * ldc]);      c3_hi = _mm512_loadu_ps(&C[3 * ldc + 16]);
        c4_lo = _mm512_loadu_ps(&C[4 * ldc]);      c4_hi = _mm512_loadu_ps(&C[4 * ldc + 16]);
        c5_lo = _mm512_loadu_ps(&C[5 * ldc]);      c5_hi = _mm512_loadu_ps(&C[5 * ldc + 16]);
    }

    // Prefetch first cache lines
    _mm_prefetch((const char*)&B[0], _MM_HINT_T0);
    _mm_prefetch((const char*)&B[64], _MM_HINT_T0);

    // Main K loop - unrolled by 4 for better ILP
    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Prefetch ahead for next iteration
        _mm_prefetch((const char*)&B[(k + 8) * ldb], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 8) * ldb + 64], _MM_HINT_T0);

        // Unroll 0
        {
            __m512 b_lo = _mm512_loadu_ps(&B[k * ldb]);
            __m512 b_hi = _mm512_loadu_ps(&B[k * ldb + 16]);

            __m512 a0 = _mm512_set1_ps(A[0 * lda + k]);
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k]);
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k]);
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k]);
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k]);
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k]);

            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
        }
        // Unroll 1
        {
            __m512 b_lo = _mm512_loadu_ps(&B[(k+1) * ldb]);
            __m512 b_hi = _mm512_loadu_ps(&B[(k+1) * ldb + 16]);

            __m512 a0 = _mm512_set1_ps(A[0 * lda + k + 1]);
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k + 1]);
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k + 1]);
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k + 1]);
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k + 1]);
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k + 1]);

            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
        }
        // Unroll 2
        {
            __m512 b_lo = _mm512_loadu_ps(&B[(k+2) * ldb]);
            __m512 b_hi = _mm512_loadu_ps(&B[(k+2) * ldb + 16]);

            __m512 a0 = _mm512_set1_ps(A[0 * lda + k + 2]);
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k + 2]);
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k + 2]);
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k + 2]);
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k + 2]);
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k + 2]);

            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
        }
        // Unroll 3
        {
            __m512 b_lo = _mm512_loadu_ps(&B[(k+3) * ldb]);
            __m512 b_hi = _mm512_loadu_ps(&B[(k+3) * ldb + 16]);

            __m512 a0 = _mm512_set1_ps(A[0 * lda + k + 3]);
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k + 3]);
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k + 3]);
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k + 3]);
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k + 3]);
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k + 3]);

            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
        }
    }

    // Handle remaining K
    for (; k < K; k++) {
        __m512 b_lo = _mm512_loadu_ps(&B[k * ldb]);
        __m512 b_hi = _mm512_loadu_ps(&B[k * ldb + 16]);

        c0_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b_hi, c0_hi);
        c1_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b_hi, c1_hi);
        c2_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b_hi, c2_hi);
        c3_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b_hi, c3_hi);
        c4_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), b_hi, c4_hi);
        c5_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), b_hi, c5_hi);
    }

    // Store results
    _mm512_storeu_ps(&C[0 * ldc], c0_lo);      _mm512_storeu_ps(&C[0 * ldc + 16], c0_hi);
    _mm512_storeu_ps(&C[1 * ldc], c1_lo);      _mm512_storeu_ps(&C[1 * ldc + 16], c1_hi);
    _mm512_storeu_ps(&C[2 * ldc], c2_lo);      _mm512_storeu_ps(&C[2 * ldc + 16], c2_hi);
    _mm512_storeu_ps(&C[3 * ldc], c3_lo);      _mm512_storeu_ps(&C[3 * ldc + 16], c3_hi);
    _mm512_storeu_ps(&C[4 * ldc], c4_lo);      _mm512_storeu_ps(&C[4 * ldc + 16], c4_hi);
    _mm512_storeu_ps(&C[5 * ldc], c5_lo);      _mm512_storeu_ps(&C[5 * ldc + 16], c5_hi);
}

// Packed version for large matrices
static inline void gemm_microkernel_6x32_packed_avx512(
    int K,
    const float * __restrict__ Ap,  // Packed A: [MR, K] contiguous
    const float * __restrict__ Bp,  // Packed B: [K, NR] contiguous
    float * __restrict__ C, int ldc,
    int first_k
)
{
    __m512 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m512 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm512_setzero_ps(); c0_hi = _mm512_setzero_ps();
        c1_lo = _mm512_setzero_ps(); c1_hi = _mm512_setzero_ps();
        c2_lo = _mm512_setzero_ps(); c2_hi = _mm512_setzero_ps();
        c3_lo = _mm512_setzero_ps(); c3_hi = _mm512_setzero_ps();
        c4_lo = _mm512_setzero_ps(); c4_hi = _mm512_setzero_ps();
        c5_lo = _mm512_setzero_ps(); c5_hi = _mm512_setzero_ps();
    } else {
        c0_lo = _mm512_loadu_ps(&C[0 * ldc]);      c0_hi = _mm512_loadu_ps(&C[0 * ldc + 16]);
        c1_lo = _mm512_loadu_ps(&C[1 * ldc]);      c1_hi = _mm512_loadu_ps(&C[1 * ldc + 16]);
        c2_lo = _mm512_loadu_ps(&C[2 * ldc]);      c2_hi = _mm512_loadu_ps(&C[2 * ldc + 16]);
        c3_lo = _mm512_loadu_ps(&C[3 * ldc]);      c3_hi = _mm512_loadu_ps(&C[3 * ldc + 16]);
        c4_lo = _mm512_loadu_ps(&C[4 * ldc]);      c4_hi = _mm512_loadu_ps(&C[4 * ldc + 16]);
        c5_lo = _mm512_loadu_ps(&C[5 * ldc]);      c5_hi = _mm512_loadu_ps(&C[5 * ldc + 16]);
    }

    // Packed B is contiguous: B[k, 0:32] at Bp[k * 32]
    _mm_prefetch((const char*)Bp, _MM_HINT_T0);
    _mm_prefetch((const char*)(Bp + 16), _MM_HINT_T0);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        _mm_prefetch((const char*)(Bp + (k + 8) * NR), _MM_HINT_T0);
        _mm_prefetch((const char*)(Bp + (k + 8) * NR + 16), _MM_HINT_T0);

        #define PACKED_ITER(koff) { \
            __m512 b_lo = _mm512_load_ps(&Bp[(k + koff) * NR]); \
            __m512 b_hi = _mm512_load_ps(&Bp[(k + koff) * NR + 16]); \
            __m512 a0 = _mm512_set1_ps(Ap[0 * K + k + koff]); \
            __m512 a1 = _mm512_set1_ps(Ap[1 * K + k + koff]); \
            __m512 a2 = _mm512_set1_ps(Ap[2 * K + k + koff]); \
            __m512 a3 = _mm512_set1_ps(Ap[3 * K + k + koff]); \
            __m512 a4 = _mm512_set1_ps(Ap[4 * K + k + koff]); \
            __m512 a5 = _mm512_set1_ps(Ap[5 * K + k + koff]); \
            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi); \
        }

        PACKED_ITER(0);
        PACKED_ITER(1);
        PACKED_ITER(2);
        PACKED_ITER(3);

        #undef PACKED_ITER
    }

    for (; k < K; k++) {
        __m512 b_lo = _mm512_load_ps(&Bp[k * NR]);
        __m512 b_hi = _mm512_load_ps(&Bp[k * NR + 16]);

        c0_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[0 * K + k]), b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[0 * K + k]), b_hi, c0_hi);
        c1_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[1 * K + k]), b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[1 * K + k]), b_hi, c1_hi);
        c2_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[2 * K + k]), b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[2 * K + k]), b_hi, c2_hi);
        c3_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[3 * K + k]), b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[3 * K + k]), b_hi, c3_hi);
        c4_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[4 * K + k]), b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[4 * K + k]), b_hi, c4_hi);
        c5_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[5 * K + k]), b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[5 * K + k]), b_hi, c5_hi);
    }

    _mm512_storeu_ps(&C[0 * ldc], c0_lo);      _mm512_storeu_ps(&C[0 * ldc + 16], c0_hi);
    _mm512_storeu_ps(&C[1 * ldc], c1_lo);      _mm512_storeu_ps(&C[1 * ldc + 16], c1_hi);
    _mm512_storeu_ps(&C[2 * ldc], c2_lo);      _mm512_storeu_ps(&C[2 * ldc + 16], c2_hi);
    _mm512_storeu_ps(&C[3 * ldc], c3_lo);      _mm512_storeu_ps(&C[3 * ldc + 16], c3_hi);
    _mm512_storeu_ps(&C[4 * ldc], c4_lo);      _mm512_storeu_ps(&C[4 * ldc + 16], c4_hi);
    _mm512_storeu_ps(&C[5 * ldc], c5_lo);      _mm512_storeu_ps(&C[5 * ldc + 16], c5_hi);
}
#endif // __AVX512F__

// =============================================================================
// AVX/AVX2 6x16 Microkernel with FMA
//
// KEY FIX: Use _mm256_fmadd_ps instead of separate mul+add
// FMA fuses multiply-add into single instruction: c = a*b + c
// This gives ~2x throughput improvement on FMA-capable CPUs
// =============================================================================

#if defined(__AVX__)
static inline void gemm_microkernel_6x16_avx(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // 12 accumulators: 6 rows x 2 YMM (16 floats) per row
    __m256 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m256 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm256_setzero_ps(); c0_hi = _mm256_setzero_ps();
        c1_lo = _mm256_setzero_ps(); c1_hi = _mm256_setzero_ps();
        c2_lo = _mm256_setzero_ps(); c2_hi = _mm256_setzero_ps();
        c3_lo = _mm256_setzero_ps(); c3_hi = _mm256_setzero_ps();
        c4_lo = _mm256_setzero_ps(); c4_hi = _mm256_setzero_ps();
        c5_lo = _mm256_setzero_ps(); c5_hi = _mm256_setzero_ps();
    } else {
        c0_lo = _mm256_loadu_ps(&C[0 * ldc]);      c0_hi = _mm256_loadu_ps(&C[0 * ldc + 8]);
        c1_lo = _mm256_loadu_ps(&C[1 * ldc]);      c1_hi = _mm256_loadu_ps(&C[1 * ldc + 8]);
        c2_lo = _mm256_loadu_ps(&C[2 * ldc]);      c2_hi = _mm256_loadu_ps(&C[2 * ldc + 8]);
        c3_lo = _mm256_loadu_ps(&C[3 * ldc]);      c3_hi = _mm256_loadu_ps(&C[3 * ldc + 8]);
        c4_lo = _mm256_loadu_ps(&C[4 * ldc]);      c4_hi = _mm256_loadu_ps(&C[4 * ldc + 8]);
        c5_lo = _mm256_loadu_ps(&C[5 * ldc]);      c5_hi = _mm256_loadu_ps(&C[5 * ldc + 8]);
    }

    // Prefetch first cache lines of B
    _mm_prefetch((const char*)B, _MM_HINT_T0);
    _mm_prefetch((const char*)(B + 32), _MM_HINT_T0);

    // Main K loop - unrolled by 8 for better ILP and prefetch hiding
    int k = 0;

#if defined(__FMA__)
    // FMA path - uses fused multiply-add (single instruction)
    for (; k <= K - 8; k += 8) {
        // Prefetch ahead - 16 rows ahead for L1
        _mm_prefetch((const char*)&B[(k + 16) * ldb], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 16) * ldb + 32], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 17) * ldb], _MM_HINT_T0);

        // Software pipelining: load B for iteration 0 before the loop body
        __m256 b_lo_next = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi_next = _mm256_loadu_ps(&B[k * ldb + 8]);

        #define FMA_ITER(koff) { \
            __m256 b_lo = b_lo_next; \
            __m256 b_hi = b_hi_next; \
            if ((koff) < 7) { \
                b_lo_next = _mm256_loadu_ps(&B[(k + (koff) + 1) * ldb]); \
                b_hi_next = _mm256_loadu_ps(&B[(k + (koff) + 1) * ldb + 8]); \
            } \
            __m256 a0 = _mm256_set1_ps(A[0 * lda + k + (koff)]); \
            __m256 a1 = _mm256_set1_ps(A[1 * lda + k + (koff)]); \
            __m256 a2 = _mm256_set1_ps(A[2 * lda + k + (koff)]); \
            __m256 a3 = _mm256_set1_ps(A[3 * lda + k + (koff)]); \
            __m256 a4 = _mm256_set1_ps(A[4 * lda + k + (koff)]); \
            __m256 a5 = _mm256_set1_ps(A[5 * lda + k + (koff)]); \
            c0_lo = _mm256_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm256_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm256_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm256_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm256_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm256_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm256_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm256_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm256_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm256_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm256_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm256_fmadd_ps(a5, b_hi, c5_hi); \
        }

        FMA_ITER(0);
        FMA_ITER(1);
        FMA_ITER(2);
        FMA_ITER(3);
        FMA_ITER(4);
        FMA_ITER(5);
        FMA_ITER(6);
        FMA_ITER(7);

        #undef FMA_ITER
    }

    // Handle remaining K with FMA
    for (; k < K; k++) {
        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);

        c0_lo = _mm256_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm256_fmadd_ps(a0, b_hi, c0_hi);
        c1_lo = _mm256_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm256_fmadd_ps(a1, b_hi, c1_hi);
        c2_lo = _mm256_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm256_fmadd_ps(a2, b_hi, c2_hi);
        c3_lo = _mm256_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm256_fmadd_ps(a3, b_hi, c3_hi);
        c4_lo = _mm256_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm256_fmadd_ps(a4, b_hi, c4_hi);
        c5_lo = _mm256_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm256_fmadd_ps(a5, b_hi, c5_hi);
    }
#else
    // Non-FMA fallback (older CPUs without FMA)
    for (; k <= K - 4; k += 4) {
        _mm_prefetch((const char*)&B[(k + 8) * ldb], _MM_HINT_T0);

        #define AVX_ITER(koff) { \
            __m256 b_lo = _mm256_loadu_ps(&B[(k + koff) * ldb]); \
            __m256 b_hi = _mm256_loadu_ps(&B[(k + koff) * ldb + 8]); \
            __m256 a0 = _mm256_set1_ps(A[0 * lda + k + koff]); \
            __m256 a1 = _mm256_set1_ps(A[1 * lda + k + koff]); \
            __m256 a2 = _mm256_set1_ps(A[2 * lda + k + koff]); \
            __m256 a3 = _mm256_set1_ps(A[3 * lda + k + koff]); \
            __m256 a4 = _mm256_set1_ps(A[4 * lda + k + koff]); \
            __m256 a5 = _mm256_set1_ps(A[5 * lda + k + koff]); \
            c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo)); \
            c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi)); \
            c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo)); \
            c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi)); \
            c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo)); \
            c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi)); \
            c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo)); \
            c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi)); \
            c4_lo = _mm256_add_ps(c4_lo, _mm256_mul_ps(a4, b_lo)); \
            c4_hi = _mm256_add_ps(c4_hi, _mm256_mul_ps(a4, b_hi)); \
            c5_lo = _mm256_add_ps(c5_lo, _mm256_mul_ps(a5, b_lo)); \
            c5_hi = _mm256_add_ps(c5_hi, _mm256_mul_ps(a5, b_hi)); \
        }

        AVX_ITER(0);
        AVX_ITER(1);
        AVX_ITER(2);
        AVX_ITER(3);

        #undef AVX_ITER
    }

    for (; k < K; k++) {
        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);

        c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo));
        c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi));
        c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo));
        c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi));
        c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo));
        c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi));
        c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo));
        c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi));
        c4_lo = _mm256_add_ps(c4_lo, _mm256_mul_ps(a4, b_lo));
        c4_hi = _mm256_add_ps(c4_hi, _mm256_mul_ps(a4, b_hi));
        c5_lo = _mm256_add_ps(c5_lo, _mm256_mul_ps(a5, b_lo));
        c5_hi = _mm256_add_ps(c5_hi, _mm256_mul_ps(a5, b_hi));
    }
#endif

    _mm256_storeu_ps(&C[0 * ldc], c0_lo);      _mm256_storeu_ps(&C[0 * ldc + 8], c0_hi);
    _mm256_storeu_ps(&C[1 * ldc], c1_lo);      _mm256_storeu_ps(&C[1 * ldc + 8], c1_hi);
    _mm256_storeu_ps(&C[2 * ldc], c2_lo);      _mm256_storeu_ps(&C[2 * ldc + 8], c2_hi);
    _mm256_storeu_ps(&C[3 * ldc], c3_lo);      _mm256_storeu_ps(&C[3 * ldc + 8], c3_hi);
    _mm256_storeu_ps(&C[4 * ldc], c4_lo);      _mm256_storeu_ps(&C[4 * ldc + 8], c4_hi);
    _mm256_storeu_ps(&C[5 * ldc], c5_lo);      _mm256_storeu_ps(&C[5 * ldc + 8], c5_hi);
}
#endif

// =============================================================================
// Edge case handler for non-MRxNR aligned tiles
// =============================================================================

static void gemm_microkernel_edge(
    int m, int n, int K,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int first_k
)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = first_k ? 0.0f : C[i * ldc + j];
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// =============================================================================
// Matrix Packing Functions - Parallel for large matrices
// =============================================================================

// Pack A panel: A[m0:m0+mc, k0:k0+kc] -> Ap[mc, kc] in row-panel format
static void pack_a_panel(
    const float *A, int lda,
    float *Ap,
    int mc, int kc, int mr
)
{
    #pragma omp parallel for schedule(static) if(mc > 64)
    for (int i = 0; i < mc; i += mr) {
        int rows = (i + mr <= mc) ? mr : (mc - i);
        float *Ap_panel = &Ap[(i / mr) * mr * kc];

        for (int p = 0; p < rows; p++) {
            const float *A_row = &A[(i + p) * lda];
            float *Ap_row = &Ap_panel[p * kc];

            // Vectorized copy
            int k = 0;
#if defined(__AVX__)
            for (; k <= kc - 8; k += 8) {
                _mm256_storeu_ps(&Ap_row[k], _mm256_loadu_ps(&A_row[k]));
            }
#endif
            for (; k < kc; k++) {
                Ap_row[k] = A_row[k];
            }
        }
        // Zero pad if partial panel
        for (int p = rows; p < mr; p++) {
            memset(&Ap_panel[p * kc], 0, kc * sizeof(float));
        }
    }
}

// Pack B panel: B[k0:k0+kc, n0:n0+nc] -> Bp[kc, nc] in column-panel format
static void pack_b_panel(
    const float *B, int ldb,
    float *Bp,
    int kc, int nc, int nr
)
{
    #pragma omp parallel for schedule(static) if(nc > 128)
    for (int j = 0; j < nc; j += nr) {
        int cols = (j + nr <= nc) ? nr : (nc - j);
        float *Bp_panel = &Bp[(j / nr) * nr * kc];

        for (int k = 0; k < kc; k++) {
            const float *B_row = &B[k * ldb + j];
            float *Bp_row = &Bp_panel[k * nr];

            // Copy cols and zero-pad
            int c = 0;
#if defined(__AVX512F__)
            for (; c <= cols - 16; c += 16) {
                _mm512_store_ps(&Bp_row[c], _mm512_loadu_ps(&B_row[c]));
            }
#elif defined(__AVX__)
            for (; c <= cols - 8; c += 8) {
                _mm256_store_ps(&Bp_row[c], _mm256_loadu_ps(&B_row[c]));
            }
#endif
            for (; c < cols; c++) {
                Bp_row[c] = B_row[c];
            }
            for (; c < nr; c++) {
                Bp_row[c] = 0.0f;
            }
        }
    }
}

// =============================================================================
// High-Performance GEMM with 2D Threading and Packing
//
// This is the main entry point for large matrices. Uses:
// 1. 2D thread partitioning (across M and N blocks)
// 2. Parallel matrix packing
// 3. Optimized microkernels
// =============================================================================

void gemm_microkernel_packed(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
#if defined(__AVX512F__) || defined(__AVX__)
    const int mr = MR;
    const int nr = NR;

    // Zero C once at the start
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    // Allocate per-thread packing buffers
    // Each thread gets its own Ap buffer, Bp is shared (packed once per N block)
    float *Bp = (float*)aligned_alloc(64, (size_t)KC * NC * sizeof(float));
    if (!Bp) {
        gemm_microkernel_blocked(A, B, C, M, N, K);
        return;
    }

    // Block over K (outermost for B panel reuse across M blocks)
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Block over N
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            // Pack B panel (parallel)
            pack_b_panel(&B[k0 * N + n0], N, Bp, kb, nb, nr);

            // 2D parallel loop over M blocks
            // Each thread processes a subset of M blocks
            #pragma omp parallel
            {
                // Thread-local A packing buffer
                float *Ap_local = (float*)aligned_alloc(64, (size_t)MC * KC * sizeof(float));

                #pragma omp for schedule(dynamic, 1)
                for (int m0 = 0; m0 < M; m0 += MC) {
                    int mb = (m0 + MC <= M) ? MC : (M - m0);

                    if (!Ap_local) continue;

                    // Pack A panel for this M block
                    pack_a_panel(&A[m0 * K + k0], K, Ap_local, mb, kb, mr);

                    // Microkernel loop over tiles
                    for (int m1 = 0; m1 < mb; m1 += mr) {
                        int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                        for (int n1 = 0; n1 < nb; n1 += nr) {
                            int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                            float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                            if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                                const float *Ap_tile = &Ap_local[(m1 / mr) * mr * kb];
                                const float *Bp_tile = &Bp[(n1 / nr) * nr * kb];
                                gemm_microkernel_6x32_packed_avx512(kb, Ap_tile, Bp_tile, C_tile, N, first_k);
#else
                                // For AVX, fall back to non-packed version
                                gemm_microkernel_6x16_avx(kb, &A[(m0 + m1) * K + k0], K,
                                                         &B[k0 * N + (n0 + n1)], N,
                                                         C_tile, N, first_k);
#endif
                            } else {
                                // Edge tiles
                                gemm_microkernel_edge(mr_actual, nr_actual, kb,
                                                      &A[(m0 + m1) * K + k0], K,
                                                      &B[k0 * N + (n0 + n1)], N,
                                                      C_tile, N, first_k);
                            }
                        }
                    }
                }

                if (Ap_local) free(Ap_local);
            }
        }
    }

    free(Bp);
#else
    gemm_microkernel_blocked(A, B, C, M, N, K);
#endif
}

// =============================================================================
// Cache-Blocked GEMM without packing (for medium-sized matrices)
// =============================================================================

void gemm_microkernel_blocked(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Zero output first
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    const int mr = MR;
    const int nr = NR;

    // Block over K
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Parallel over N blocks (good cache reuse of A)
        #pragma omp parallel for schedule(dynamic)
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            for (int m0 = 0; m0 < M; m0 += MC) {
                int mb = (m0 + MC <= M) ? MC : (M - m0);

                for (int m1 = 0; m1 < mb; m1 += mr) {
                    int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                    for (int n1 = 0; n1 < nb; n1 += nr) {
                        int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                        const float *A_tile = &A[(m0 + m1) * K + k0];
                        const float *B_tile = &B[k0 * N + (n0 + n1)];
                        float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                        if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                            gemm_microkernel_6x32_avx512(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__AVX__)
                            gemm_microkernel_6x16_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#else
                            gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#endif
                        } else {
                            gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// B-transposed GEMM: C[M,N] = A[M,K] @ B[N,K].T
// =============================================================================

#if defined(__AVX512F__)
static inline void gemm_microkernel_6x32_bt_avx512(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,  // B is [N, K] transposed
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // For B transposed, we need different access pattern
    // C[i,j] = sum_k A[i,k] * B[j,k]

    if (first_k) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 32; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    }

    // Process K in chunks of 16 for SIMD
    int k = 0;
    for (; k <= K - 16; k += 16) {
        // Load A[0:6, k:k+16] - 6 rows
        __m512 a0 = _mm512_loadu_ps(&A[0 * lda + k]);
        __m512 a1 = _mm512_loadu_ps(&A[1 * lda + k]);
        __m512 a2 = _mm512_loadu_ps(&A[2 * lda + k]);
        __m512 a3 = _mm512_loadu_ps(&A[3 * lda + k]);
        __m512 a4 = _mm512_loadu_ps(&A[4 * lda + k]);
        __m512 a5 = _mm512_loadu_ps(&A[5 * lda + k]);

        // For each column j of C (row j of B)
        for (int j = 0; j < 32; j++) {
            __m512 b = _mm512_loadu_ps(&B[j * ldb + k]);

            // Compute dot products using reduction
            C[0 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a0, b));
            C[1 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a1, b));
            C[2 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a2, b));
            C[3 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a3, b));
            C[4 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a4, b));
            C[5 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a5, b));
        }
    }

    // Handle remaining K
    for (; k < K; k++) {
        for (int i = 0; i < 6; i++) {
            float a = A[i * lda + k];
            for (int j = 0; j < 32; j++) {
                C[i * ldc + j] += a * B[j * ldb + k];
            }
        }
    }
}
#endif

void gemm_microkernel_blocked_bt(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Zero output first
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    const int mr = MR;
    const int nr = NR;

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int m0 = 0; m0 < M; m0 += MC) {
        for (int n0 = 0; n0 < N; n0 += NC) {
            int mb = (m0 + MC <= M) ? MC : (M - m0);
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            for (int k0 = 0; k0 < K; k0 += KC) {
                int kb = (k0 + KC <= K) ? KC : (K - k0);
                int first_k = (k0 == 0);

                for (int m1 = 0; m1 < mb; m1 += mr) {
                    int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                    for (int n1 = 0; n1 < nb; n1 += nr) {
                        int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                        const float *A_tile = &A[(m0 + m1) * K + k0];
                        const float *B_tile = &B[(n0 + n1) * K + k0];
                        float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                        if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                            gemm_microkernel_6x32_bt_avx512(kb, A_tile, K, B_tile, K, C_tile, N, first_k);
#else
                            // Scalar fallback for B-transposed
                            for (int i = 0; i < mr; i++) {
                                for (int j = 0; j < nr; j++) {
                                    float sum = first_k ? 0.0f : C_tile[i * N + j];
                                    for (int kk = 0; kk < kb; kk++) {
                                        sum += A_tile[i * K + kk] * B_tile[j * K + kk];
                                    }
                                    C_tile[i * N + j] = sum;
                                }
                            }
#endif
                        } else {
                            // Edge case
                            for (int i = 0; i < mr_actual; i++) {
                                for (int j = 0; j < nr_actual; j++) {
                                    float sum = first_k ? 0.0f : C_tile[i * N + j];
                                    for (int kk = 0; kk < kb; kk++) {
                                        sum += A_tile[i * K + kk] * B_tile[j * K + kk];
                                    }
                                    C_tile[i * N + j] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

#define PACK_THRESHOLD 256  // Use packing for matrices >= 256

void gemm_microkernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int B_transposed
)
{
    if (B_transposed) {
        gemm_microkernel_blocked_bt(A, B, C, M, N, K);
    } else {
        // Use packed version for large matrices
        if (M >= PACK_THRESHOLD && N >= PACK_THRESHOLD && K >= PACK_THRESHOLD) {
            gemm_microkernel_packed(A, B, C, M, N, K);
        } else {
            gemm_microkernel_blocked(A, B, C, M, N, K);
        }
    }
}
