#include "ckernel_engine.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <omp.h>

static inline int ck_min(int a, int b) { return a < b ? a : b; }

static void gemm_naive_serial_double(const float *A,
                                     const float *B,
                                     const float *bias,
                                     float *C,
                                     int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[i * K + k] * (double)B[j * K + k];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

// Naive parallel GEMM (reference baseline) – copied from C-Transformer.
void gemm_naive_parallel(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            float bias_val = bias ? bias[j] : 0.0f;
            C[i * N + j] = sum + bias_val;
        }
    }
}

// AVX-512 optimized GEMM – copied from C-Transformer.
void gemm_avx512_parallel(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m512 sum_vec = _mm512_setzero_ps();
            int k;
            for (k = 0; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            float bias_val = bias ? bias[j] : 0.0f;
            C[i * N + j] = sum + bias_val;
        }
    }
#else
    gemm_naive_parallel(A, B, bias, C, M, N, K);
#endif
}

// Cache-blocked GEMM with fine-grained parallelism – copied from C-Transformer.
void gemm_fine_grained_parallel(const float *A,
                                const float *B,
                                const float *bias,
                                float *C,
                                int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    const int block_size = 64;
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
#pragma omp parallel for collapse(3)
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int j_end = ck_min(jj + block_size, N);
                int k_end = ck_min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#pragma omp atomic
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
#else
    gemm_naive_parallel(A, B, bias, C, M, N, K);
#endif
}

// =============================================================================
// GEMM_NN: C[M,N] = A[M,K] @ B[K,N] + bias[N]
// B is stored row-major as [K,N] (no transpose)
// Used for backward d_input computation: d_input = d_output @ W
// =============================================================================

static void gemm_nn_serial_double(const float *A,
                                  const float *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

void gemm_nn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gemm_nn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    // For gemm_nn, we can't vectorize over K easily since B[k,j] has stride N.
    // Instead, vectorize over N (output columns) when N >= 16.
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        // Process 16 output columns at a time
        for (; j <= N - 16; j += 16) {
            __m512 sum_vec = bias ? _mm512_loadu_ps(&bias[j]) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 a_broadcast = _mm512_set1_ps(A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                sum_vec = _mm512_fmadd_ps(a_broadcast, b_vec, sum_vec);
            }
            _mm512_storeu_ps(&C[i * N + j], sum_vec);
        }
        // Handle remaining columns
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#else
    gemm_nn_parallel(A, B, bias, C, M, N, K);
#endif
}

void gemm_nn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
    const int block_size = 64;
    // Initialize C with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    // Blocked multiply-accumulate
    for (int ii = 0; ii < M; ii += block_size) {
        for (int kk = 0; kk < K; kk += block_size) {
            for (int jj = 0; jj < N; jj += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int k_end = ck_min(kk + block_size, K);
                int j_end = ck_min(jj + block_size, N);

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[i * K + k];
#if defined(__AVX512F__)
                        __m512 a_broadcast = _mm512_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 16; j += 16) {
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                            c_vec = _mm512_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm512_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#else
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

// =============================================================================
// GEMM_TN: C[M,N] = A[K,M].T @ B[K,N] + bias[N]
// A is stored row-major as [K,M], B is stored row-major as [K,N]
// Used for backward d_W computation: d_W = d_output.T @ input
// =============================================================================

static void gemm_tn_serial_double(const float *A,
                                  const float *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                // A.T[i,k] = A[k,i] = A[k*M + i]
                sum += (double)A[k * M + i] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

void gemm_tn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gemm_tn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    // Vectorize over N (output columns)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 sum_vec = bias ? _mm512_loadu_ps(&bias[j]) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 a_broadcast = _mm512_set1_ps(A[k * M + i]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                sum_vec = _mm512_fmadd_ps(a_broadcast, b_vec, sum_vec);
            }
            _mm512_storeu_ps(&C[i * N + j], sum_vec);
        }
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#else
    gemm_tn_parallel(A, B, bias, C, M, N, K);
#endif
}

void gemm_tn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
    const int block_size = 64;
    // Initialize C with bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    // Blocked multiply-accumulate
    for (int ii = 0; ii < M; ii += block_size) {
        for (int kk = 0; kk < K; kk += block_size) {
            for (int jj = 0; jj < N; jj += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int k_end = ck_min(kk + block_size, K);
                int j_end = ck_min(jj + block_size, N);

                for (int k = kk; k < k_end; k++) {
                    for (int i = ii; i < i_end; i++) {
                        float a_val = A[k * M + i];
#if defined(__AVX512F__)
                        __m512 a_broadcast = _mm512_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 16; j += 16) {
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                            c_vec = _mm512_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm512_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#else
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

// =============================================================================
// Original GEMM_NT: C[M,N] = A[M,K] @ B[N,K].T + bias[N]
// B is stored row-major as [N,K] (transposed in the multiply)
// =============================================================================

// Serial cache-blocked GEMM – copied from C-Transformer (with NULL-safe bias).
void gemm_blocked_serial(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_double(A, B, bias, C, M, N, K);
        return;
    }
    const int block_size = 64;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int j_end = ck_min(jj + block_size, N);
                int k_end = ck_min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
#if defined(__AVX512F__)
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#else
                        float partial_sum = 0.0f;
                        for (int k = kk; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#endif
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}
