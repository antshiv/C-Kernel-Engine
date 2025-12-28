/**
 * GEMM Microkernel - High-Performance Register-Blocked Matrix Multiplication
 *
 * This file implements optimized GEMM microkernels inspired by oneDNN/BLIS.
 * The key insight: keep all accumulator values in registers across the K loop.
 *
 * Architecture:
 * 1. Microkernel: Fixed-size tile (8x8) computed entirely in registers
 * 2. Cache blocking: Outer loops tile for L1/L2/L3 cache
 * 3. Packing: Optional A/B packing for better memory access patterns
 *
 * Supported configurations:
 * - AVX1 (256-bit, no FMA): 8x8 microkernel using mul+add
 * - AVX2 (256-bit, FMA): 8x8 microkernel using FMA
 * - AVX-512 (512-bit, FMA): 8x16 or 16x16 microkernel
 *
 * Layout: C[M,N] = A[M,K] @ B[K,N] (row-major)
 * For transposed B: C[M,N] = A[M,K] @ B[N,K].T
 */

#include "ckernel_engine.h"
#include <string.h>
#include <stdlib.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Microkernel Configuration
// =============================================================================

// Microkernel tile sizes
#define MR 8   // Rows per microkernel
#define NR 8   // Cols per microkernel (AVX: 8 floats per register)

// Cache blocking sizes (tuned for typical L1=32KB, L2=256KB, L3=shared)
#define MC 64   // M block size (fits in L2)
#define NC 256  // N block size (fits in L3)
#define KC 256  // K block size (fits in L1)

// =============================================================================
// 8x8 Microkernel - AVX1 (no FMA)
//
// Computes: C[0:8, 0:8] += A[0:8, 0:K] @ B[0:K, 0:8]
//
// Register usage (16 YMM registers available):
// - c0-c7: 8 accumulators (one per row of C)
// - b: loaded B values
// - a: broadcast A values
//
// This keeps 8 rows x 8 cols = 64 floats in registers!
// =============================================================================

#if defined(__AVX__)
static inline void gemm_microkernel_8x8_avx(
    int K,
    const float *A, int lda,  // A is [MR, K], lda = stride between rows
    const float *B, int ldb,  // B is [K, NR], ldb = stride between rows
    float *C, int ldc,        // C is [MR, NR], ldc = stride between rows
    int first_k               // If true, zero C; else accumulate
)
{
    // Initialize accumulators - one __m256 per row of C
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    if (first_k) {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
    } else {
        // Load existing C values
        c0 = _mm256_loadu_ps(&C[0 * ldc]);
        c1 = _mm256_loadu_ps(&C[1 * ldc]);
        c2 = _mm256_loadu_ps(&C[2 * ldc]);
        c3 = _mm256_loadu_ps(&C[3 * ldc]);
        c4 = _mm256_loadu_ps(&C[4 * ldc]);
        c5 = _mm256_loadu_ps(&C[5 * ldc]);
        c6 = _mm256_loadu_ps(&C[6 * ldc]);
        c7 = _mm256_loadu_ps(&C[7 * ldc]);
    }

    // Main K loop - this is where the magic happens
    // All 64 accumulator values stay in registers!
    for (int k = 0; k < K; k++) {
        // Load one row of B: B[k, 0:8]
        __m256 b = _mm256_loadu_ps(&B[k * ldb]);

        // For each row of A, broadcast A[row, k] and multiply-accumulate
        // AVX1: use mul + add (no FMA)
        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);
        __m256 a6 = _mm256_set1_ps(A[6 * lda + k]);
        __m256 a7 = _mm256_set1_ps(A[7 * lda + k]);

        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
        c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
        c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
        c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
        c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
    }

    // Store results back to C
    _mm256_storeu_ps(&C[0 * ldc], c0);
    _mm256_storeu_ps(&C[1 * ldc], c1);
    _mm256_storeu_ps(&C[2 * ldc], c2);
    _mm256_storeu_ps(&C[3 * ldc], c3);
    _mm256_storeu_ps(&C[4 * ldc], c4);
    _mm256_storeu_ps(&C[5 * ldc], c5);
    _mm256_storeu_ps(&C[6 * ldc], c6);
    _mm256_storeu_ps(&C[7 * ldc], c7);
}
#endif

// =============================================================================
// 8x8 Microkernel - AVX2/FMA
//
// Same as AVX1 but uses FMA instructions for better performance
// =============================================================================

#if defined(__AVX2__) && defined(__FMA__)
static inline void gemm_microkernel_8x8_fma(
    int K,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int first_k
)
{
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    if (first_k) {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
    } else {
        c0 = _mm256_loadu_ps(&C[0 * ldc]);
        c1 = _mm256_loadu_ps(&C[1 * ldc]);
        c2 = _mm256_loadu_ps(&C[2 * ldc]);
        c3 = _mm256_loadu_ps(&C[3 * ldc]);
        c4 = _mm256_loadu_ps(&C[4 * ldc]);
        c5 = _mm256_loadu_ps(&C[5 * ldc]);
        c6 = _mm256_loadu_ps(&C[6 * ldc]);
        c7 = _mm256_loadu_ps(&C[7 * ldc]);
    }

    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(&B[k * ldb]);

        // Use FMA: c = a * b + c
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * lda + k]), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * lda + k]), b, c7);
    }

    _mm256_storeu_ps(&C[0 * ldc], c0);
    _mm256_storeu_ps(&C[1 * ldc], c1);
    _mm256_storeu_ps(&C[2 * ldc], c2);
    _mm256_storeu_ps(&C[3 * ldc], c3);
    _mm256_storeu_ps(&C[4 * ldc], c4);
    _mm256_storeu_ps(&C[5 * ldc], c5);
    _mm256_storeu_ps(&C[6 * ldc], c6);
    _mm256_storeu_ps(&C[7 * ldc], c7);
}
#endif

// =============================================================================
// 8x16 Microkernel - AVX-512
//
// AVX-512 has 512-bit registers (16 floats), so we can do 8x16 tiles
// Uses 8 ZMM registers for C, leaving plenty for A/B loads
// =============================================================================

#if defined(__AVX512F__)
static inline void gemm_microkernel_8x16_avx512(
    int K,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int first_k
)
{
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;

    if (first_k) {
        c0 = _mm512_setzero_ps();
        c1 = _mm512_setzero_ps();
        c2 = _mm512_setzero_ps();
        c3 = _mm512_setzero_ps();
        c4 = _mm512_setzero_ps();
        c5 = _mm512_setzero_ps();
        c6 = _mm512_setzero_ps();
        c7 = _mm512_setzero_ps();
    } else {
        c0 = _mm512_loadu_ps(&C[0 * ldc]);
        c1 = _mm512_loadu_ps(&C[1 * ldc]);
        c2 = _mm512_loadu_ps(&C[2 * ldc]);
        c3 = _mm512_loadu_ps(&C[3 * ldc]);
        c4 = _mm512_loadu_ps(&C[4 * ldc]);
        c5 = _mm512_loadu_ps(&C[5 * ldc]);
        c6 = _mm512_loadu_ps(&C[6 * ldc]);
        c7 = _mm512_loadu_ps(&C[7 * ldc]);
    }

    for (int k = 0; k < K; k++) {
        __m512 b = _mm512_loadu_ps(&B[k * ldb]);

        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), b, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), b, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(A[6 * lda + k]), b, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(A[7 * lda + k]), b, c7);
    }

    _mm512_storeu_ps(&C[0 * ldc], c0);
    _mm512_storeu_ps(&C[1 * ldc], c1);
    _mm512_storeu_ps(&C[2 * ldc], c2);
    _mm512_storeu_ps(&C[3 * ldc], c3);
    _mm512_storeu_ps(&C[4 * ldc], c4);
    _mm512_storeu_ps(&C[5 * ldc], c5);
    _mm512_storeu_ps(&C[6 * ldc], c6);
    _mm512_storeu_ps(&C[7 * ldc], c7);
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
    // Simple scalar fallback for edge tiles
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
// Matrix Packing Functions
//
// Packing transforms matrices into contiguous layouts optimized for the
// microkernel access pattern. This is critical for large matrix performance.
// =============================================================================

// Pack A panel: A[m0:m0+mc, k0:k0+kc] -> Ap[mc, kc] in row-panel format
// Packed layout: for each MR-row panel, store rows contiguously
static void pack_a_panel(
    const float *A, int lda,
    float *Ap,
    int mc, int kc, int mr
)
{
    for (int i = 0; i < mc; i += mr) {
        int rows = (i + mr <= mc) ? mr : (mc - i);
        for (int p = 0; p < rows; p++) {
            for (int k = 0; k < kc; k++) {
                Ap[(i / mr) * mr * kc + p * kc + k] = A[(i + p) * lda + k];
            }
        }
        // Zero pad if partial panel
        for (int p = rows; p < mr; p++) {
            for (int k = 0; k < kc; k++) {
                Ap[(i / mr) * mr * kc + p * kc + k] = 0.0f;
            }
        }
    }
}

// Pack B panel: B[k0:k0+kc, n0:n0+nc] -> Bp[kc, nc] in column-panel format
// Packed layout: for each NR-column panel, store columns contiguously by K
static void pack_b_panel(
    const float *B, int ldb,
    float *Bp,
    int kc, int nc, int nr
)
{
    for (int j = 0; j < nc; j += nr) {
        int cols = (j + nr <= nc) ? nr : (nc - j);
        for (int k = 0; k < kc; k++) {
            for (int q = 0; q < cols; q++) {
                Bp[(j / nr) * nr * kc + k * nr + q] = B[k * ldb + (j + q)];
            }
            // Zero pad if partial panel
            for (int q = cols; q < nr; q++) {
                Bp[(j / nr) * nr * kc + k * nr + q] = 0.0f;
            }
        }
    }
}

// =============================================================================
// Optimized 8x8 Microkernel for Packed Matrices (AVX1)
//
// A is packed: [MR, KC] contiguous
// B is packed: [KC, NR] contiguous (column-panel)
// =============================================================================

#if defined(__AVX__)
static inline void gemm_microkernel_8x8_packed_avx(
    int K,
    const float *Ap,   // Packed A: [MR, K] contiguous
    const float *Bp,   // Packed B: [K, NR] contiguous
    float *C, int ldc,
    int first_k
)
{
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    if (first_k) {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
    } else {
        c0 = _mm256_loadu_ps(&C[0 * ldc]);
        c1 = _mm256_loadu_ps(&C[1 * ldc]);
        c2 = _mm256_loadu_ps(&C[2 * ldc]);
        c3 = _mm256_loadu_ps(&C[3 * ldc]);
        c4 = _mm256_loadu_ps(&C[4 * ldc]);
        c5 = _mm256_loadu_ps(&C[5 * ldc]);
        c6 = _mm256_loadu_ps(&C[6 * ldc]);
        c7 = _mm256_loadu_ps(&C[7 * ldc]);
    }

    // Prefetch first B line
    _mm_prefetch((const char*)Bp, _MM_HINT_T0);

    // Unrolled K loop by 4 for better ILP
    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Prefetch ahead
        _mm_prefetch((const char*)(Bp + (k + 8) * NR), _MM_HINT_T0);

        // Iteration 0
        {
            __m256 b = _mm256_load_ps(&Bp[k * NR]);  // Aligned load from packed B
            __m256 a0 = _mm256_set1_ps(Ap[0 * K + k]);
            __m256 a1 = _mm256_set1_ps(Ap[1 * K + k]);
            __m256 a2 = _mm256_set1_ps(Ap[2 * K + k]);
            __m256 a3 = _mm256_set1_ps(Ap[3 * K + k]);
            __m256 a4 = _mm256_set1_ps(Ap[4 * K + k]);
            __m256 a5 = _mm256_set1_ps(Ap[5 * K + k]);
            __m256 a6 = _mm256_set1_ps(Ap[6 * K + k]);
            __m256 a7 = _mm256_set1_ps(Ap[7 * K + k]);
            c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
            c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
            c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
            c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
            c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
            c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
            c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
            c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
        }
        // Iteration 1
        {
            __m256 b = _mm256_load_ps(&Bp[(k+1) * NR]);
            __m256 a0 = _mm256_set1_ps(Ap[0 * K + k + 1]);
            __m256 a1 = _mm256_set1_ps(Ap[1 * K + k + 1]);
            __m256 a2 = _mm256_set1_ps(Ap[2 * K + k + 1]);
            __m256 a3 = _mm256_set1_ps(Ap[3 * K + k + 1]);
            __m256 a4 = _mm256_set1_ps(Ap[4 * K + k + 1]);
            __m256 a5 = _mm256_set1_ps(Ap[5 * K + k + 1]);
            __m256 a6 = _mm256_set1_ps(Ap[6 * K + k + 1]);
            __m256 a7 = _mm256_set1_ps(Ap[7 * K + k + 1]);
            c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
            c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
            c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
            c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
            c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
            c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
            c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
            c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
        }
        // Iteration 2
        {
            __m256 b = _mm256_load_ps(&Bp[(k+2) * NR]);
            __m256 a0 = _mm256_set1_ps(Ap[0 * K + k + 2]);
            __m256 a1 = _mm256_set1_ps(Ap[1 * K + k + 2]);
            __m256 a2 = _mm256_set1_ps(Ap[2 * K + k + 2]);
            __m256 a3 = _mm256_set1_ps(Ap[3 * K + k + 2]);
            __m256 a4 = _mm256_set1_ps(Ap[4 * K + k + 2]);
            __m256 a5 = _mm256_set1_ps(Ap[5 * K + k + 2]);
            __m256 a6 = _mm256_set1_ps(Ap[6 * K + k + 2]);
            __m256 a7 = _mm256_set1_ps(Ap[7 * K + k + 2]);
            c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
            c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
            c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
            c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
            c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
            c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
            c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
            c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
        }
        // Iteration 3
        {
            __m256 b = _mm256_load_ps(&Bp[(k+3) * NR]);
            __m256 a0 = _mm256_set1_ps(Ap[0 * K + k + 3]);
            __m256 a1 = _mm256_set1_ps(Ap[1 * K + k + 3]);
            __m256 a2 = _mm256_set1_ps(Ap[2 * K + k + 3]);
            __m256 a3 = _mm256_set1_ps(Ap[3 * K + k + 3]);
            __m256 a4 = _mm256_set1_ps(Ap[4 * K + k + 3]);
            __m256 a5 = _mm256_set1_ps(Ap[5 * K + k + 3]);
            __m256 a6 = _mm256_set1_ps(Ap[6 * K + k + 3]);
            __m256 a7 = _mm256_set1_ps(Ap[7 * K + k + 3]);
            c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
            c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
            c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
            c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
            c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
            c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
            c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
            c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
        }
    }

    // Handle remaining K
    for (; k < K; k++) {
        __m256 b = _mm256_load_ps(&Bp[k * NR]);
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(Ap[0 * K + k]), b));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(Ap[1 * K + k]), b));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(Ap[2 * K + k]), b));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(Ap[3 * K + k]), b));
        c4 = _mm256_add_ps(c4, _mm256_mul_ps(_mm256_set1_ps(Ap[4 * K + k]), b));
        c5 = _mm256_add_ps(c5, _mm256_mul_ps(_mm256_set1_ps(Ap[5 * K + k]), b));
        c6 = _mm256_add_ps(c6, _mm256_mul_ps(_mm256_set1_ps(Ap[6 * K + k]), b));
        c7 = _mm256_add_ps(c7, _mm256_mul_ps(_mm256_set1_ps(Ap[7 * K + k]), b));
    }

    _mm256_storeu_ps(&C[0 * ldc], c0);
    _mm256_storeu_ps(&C[1 * ldc], c1);
    _mm256_storeu_ps(&C[2 * ldc], c2);
    _mm256_storeu_ps(&C[3 * ldc], c3);
    _mm256_storeu_ps(&C[4 * ldc], c4);
    _mm256_storeu_ps(&C[5 * ldc], c5);
    _mm256_storeu_ps(&C[6 * ldc], c6);
    _mm256_storeu_ps(&C[7 * ldc], c7);
}
#endif

// =============================================================================
// Optimized GEMM with Matrix Packing (for large matrices)
// =============================================================================

void gemm_microkernel_packed(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
#if defined(__AVX__)
    const int mr = MR;
    const int nr = NR;

    // Allocate packing buffers (thread-local for OpenMP)
    // Ap: MC x KC, Bp: KC x NC
    float *Ap = NULL;
    float *Bp = NULL;

    // For simplicity, allocate max size buffers
    // In production, use thread-local storage
    Ap = (float*)aligned_alloc(32, MC * KC * sizeof(float));
    Bp = (float*)aligned_alloc(32, KC * NC * sizeof(float));

    if (!Ap || !Bp) {
        // Fallback to non-packed version
        if (Ap) free(Ap);
        if (Bp) free(Bp);
        gemm_microkernel_blocked(A, B, C, M, N, K);
        return;
    }

    // Zero C once at the start
    memset(C, 0, (size_t)M * N * sizeof(float));

    // Block over K (outermost for B panel reuse)
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Block over N
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            // Pack B panel: B[k0:k0+kb, n0:n0+nb]
            pack_b_panel(&B[k0 * N + n0], N, Bp, kb, nb, nr);

            // Block over M (parallelize this loop)
            #pragma omp parallel for schedule(static) firstprivate(Ap)
            for (int m0 = 0; m0 < M; m0 += MC) {
                int mb = (m0 + MC <= M) ? MC : (M - m0);

                // Each thread needs its own Ap buffer
                float *Ap_local = (float*)aligned_alloc(32, MC * KC * sizeof(float));
                if (!Ap_local) continue;

                // Pack A panel: A[m0:m0+mb, k0:k0+kb]
                pack_a_panel(&A[m0 * K + k0], K, Ap_local, mb, kb, mr);

                // Micro-kernel loop
                for (int m1 = 0; m1 < mb; m1 += mr) {
                    int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                    for (int n1 = 0; n1 < nb; n1 += nr) {
                        int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                        float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                        // Packed A: offset by panel index
                        const float *Ap_tile = &Ap_local[(m1 / mr) * mr * kb];
                        // Packed B: offset by panel index
                        const float *Bp_tile = &Bp[(n1 / nr) * nr * kb];

                        if (mr_actual == mr && nr_actual == nr) {
                            gemm_microkernel_8x8_packed_avx(kb, Ap_tile, Bp_tile, C_tile, N, first_k);
                        } else {
                            // Edge case - use scalar
                            gemm_microkernel_edge(mr_actual, nr_actual, kb,
                                                  &A[(m0 + m1) * K + k0], K,
                                                  &B[k0 * N + (n0 + n1)], N,
                                                  C_tile, N, first_k);
                        }
                    }
                }

                free(Ap_local);
            }
        }
    }

    free(Ap);
    free(Bp);
#else
    // Fallback
    gemm_microkernel_blocked(A, B, C, M, N, K);
#endif
}

// =============================================================================
// Cache-Blocked GEMM using Microkernels (original, no packing)
//
// C[M,N] = A[M,K] @ B[K,N]
//
// The blocking strategy:
// 1. Block K into KC chunks (fits A block in L1)
// 2. Block N into NC chunks (fits B block in L2)
// 3. Block M into MC chunks (fits C block in L2)
// 4. Call microkernel for each MRxNR tile
// =============================================================================

void gemm_microkernel_blocked(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Zero output first
    memset(C, 0, (size_t)M * N * sizeof(float));

#if defined(__AVX512F__)
    const int nr = 16;  // AVX-512: 16 floats per register
#else
    const int nr = NR;  // AVX/AVX2: 8 floats per register
#endif
    const int mr = MR;

    // Block over K (for L1 cache - A panel)
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Block over N (for L2/L3 cache - B panel)
        #pragma omp parallel for schedule(dynamic)
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            // Block over M (for L2 cache - A panel reuse)
            for (int m0 = 0; m0 < M; m0 += MC) {
                int mb = (m0 + MC <= M) ? MC : (M - m0);

                // Call microkernel for each MRxNR tile
                for (int m1 = 0; m1 < mb; m1 += mr) {
                    int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                    for (int n1 = 0; n1 < nb; n1 += nr) {
                        int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                        const float *A_tile = &A[(m0 + m1) * K + k0];
                        const float *B_tile = &B[k0 * N + (n0 + n1)];
                        float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                        // Use optimized microkernel for full tiles
                        if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                            gemm_microkernel_8x16_avx512(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__AVX2__) && defined(__FMA__)
                            gemm_microkernel_8x8_fma(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__AVX__)
                            gemm_microkernel_8x8_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#else
                            gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#endif
                        } else {
                            // Edge tiles: use scalar fallback
                            gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// GEMM with B transposed: C[M,N] = A[M,K] @ B[N,K].T
//
// This is the common layout for neural network weights (stored as [out, in])
// =============================================================================

#if defined(__AVX__)
static inline void gemm_microkernel_8x8_bt_avx(
    int K,
    const float *A, int lda,   // A is [MR, K], row-major
    const float *B, int ldb,   // B is [NR, K], row-major (transposed)
    float *C, int ldc,
    int first_k
)
{
    // For B transposed with first_k, zero C first
    if (first_k) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    }

    // For B transposed, we compute dot products differently
    // C[i,j] = sum_k A[i,k] * B[j,k]
    //
    // Strategy: For each k, load A[0:8, k] as 8 broadcasts,
    // load B[0:8, k] as 8 broadcasts, then compute outer product

    // Actually, for B transposed we need a different approach
    // Let's compute 8 dot products at once using horizontal operations

    // Alternative: Process K in chunks of 8, use SIMD for the K dimension
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Load 8 consecutive values from each row of A
        __m256 a0 = _mm256_loadu_ps(&A[0 * lda + k]);
        __m256 a1 = _mm256_loadu_ps(&A[1 * lda + k]);
        __m256 a2 = _mm256_loadu_ps(&A[2 * lda + k]);
        __m256 a3 = _mm256_loadu_ps(&A[3 * lda + k]);
        __m256 a4 = _mm256_loadu_ps(&A[4 * lda + k]);
        __m256 a5 = _mm256_loadu_ps(&A[5 * lda + k]);
        __m256 a6 = _mm256_loadu_ps(&A[6 * lda + k]);
        __m256 a7 = _mm256_loadu_ps(&A[7 * lda + k]);

        // For each column j of C (row j of B)
        for (int j = 0; j < 8; j++) {
            __m256 b = _mm256_loadu_ps(&B[j * ldb + k]);

            // Multiply and accumulate partial products
            // We need horizontal sum at the end
            __m256 p0 = _mm256_mul_ps(a0, b);
            __m256 p1 = _mm256_mul_ps(a1, b);
            __m256 p2 = _mm256_mul_ps(a2, b);
            __m256 p3 = _mm256_mul_ps(a3, b);
            __m256 p4 = _mm256_mul_ps(a4, b);
            __m256 p5 = _mm256_mul_ps(a5, b);
            __m256 p6 = _mm256_mul_ps(a6, b);
            __m256 p7 = _mm256_mul_ps(a7, b);

            // Horizontal sum each product vector
            // hadd pairs adjacent elements
            __m256 s01 = _mm256_hadd_ps(p0, p1);
            __m256 s23 = _mm256_hadd_ps(p2, p3);
            __m256 s45 = _mm256_hadd_ps(p4, p5);
            __m256 s67 = _mm256_hadd_ps(p6, p7);

            __m256 s0123 = _mm256_hadd_ps(s01, s23);
            __m256 s4567 = _mm256_hadd_ps(s45, s67);

            // Now we have partial sums, need to combine 128-bit halves
            __m128 lo0123 = _mm256_castps256_ps128(s0123);
            __m128 hi0123 = _mm256_extractf128_ps(s0123, 1);
            __m128 sum0123 = _mm_add_ps(lo0123, hi0123);

            __m128 lo4567 = _mm256_castps256_ps128(s4567);
            __m128 hi4567 = _mm256_extractf128_ps(s4567, 1);
            __m128 sum4567 = _mm_add_ps(lo4567, hi4567);

            // Extract individual sums and accumulate
            C[0 * ldc + j] += _mm_cvtss_f32(sum0123);
            C[1 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum0123, sum0123, 1));
            C[2 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum0123, sum0123, 2));
            C[3 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum0123, sum0123, 3));
            C[4 * ldc + j] += _mm_cvtss_f32(sum4567);
            C[5 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum4567, sum4567, 1));
            C[6 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum4567, sum4567, 2));
            C[7 * ldc + j] += _mm_cvtss_f32(_mm_shuffle_ps(sum4567, sum4567, 3));
        }
    }

    // Handle remaining K
    for (; k < K; k++) {
        for (int i = 0; i < 8; i++) {
            float a = A[i * lda + k];
            for (int j = 0; j < 8; j++) {
                C[i * ldc + j] += a * B[j * ldb + k];
            }
        }
    }
}
#endif

// =============================================================================
// Simple blocked GEMM for B transposed (common in NN)
// C[M,N] = A[M,K] @ B[N,K].T
// =============================================================================

void gemm_microkernel_blocked_bt(
    const float *A,    // [M, K]
    const float *B,    // [N, K] (transposed)
    float *C,          // [M, N]
    int M, int N, int K
)
{
    // Zero output first
    memset(C, 0, (size_t)M * N * sizeof(float));

    const int mr = MR;
    const int nr = NR;

    #pragma omp parallel for schedule(dynamic)
    for (int m0 = 0; m0 < M; m0 += MC) {
        int mb = (m0 + MC <= M) ? MC : (M - m0);

        for (int n0 = 0; n0 < N; n0 += NC) {
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
#if defined(__AVX__)
                            gemm_microkernel_8x8_bt_avx(kb, A_tile, K, B_tile, K, C_tile, N, first_k);
#else
                            // Scalar fallback
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
// Public API - wrapper that chooses best implementation
// =============================================================================

// Threshold for using packed version (packing overhead amortized)
#define PACK_THRESHOLD 128

void gemm_microkernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int B_transposed  // 0 = B is [K,N], 1 = B is [N,K] (transposed)
)
{
    if (B_transposed) {
        gemm_microkernel_blocked_bt(A, B, C, M, N, K);
    } else {
        // Use packed version for large matrices where packing overhead is worth it
        if (M >= PACK_THRESHOLD && N >= PACK_THRESHOLD && K >= PACK_THRESHOLD) {
            gemm_microkernel_packed(A, B, C, M, N, K);
        } else {
            gemm_microkernel_blocked(A, B, C, M, N, K);
        }
    }
}
