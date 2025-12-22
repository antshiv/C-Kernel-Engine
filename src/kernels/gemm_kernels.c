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
