#ifndef CKERNEL_ENGINE_H
#define CKERNEL_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Core math backend interface for C-Kernel-Engine.
 *
 * This is intentionally minimal and matches the conventions already used
 * in C-Transformer for GEMM kernels.
 *
 * Layout assumptions (LLM-style shapes):
 *  - A: [M x K], row-major,       A(i,k) = A[i*K + k]
 *  - B: [N x K], row-major,       B(j,k) = B[j*K + k]
 *  - C: [M x N], row-major,       C(i,j) = C[i*N + j]
 *  - bias: optional [N], added per output column j
 */
typedef struct {
    void (*sgemm)(int M, int N, int K,
                  const float *A, int lda,
                  const float *B, int ldb,
                  const float *bias,
                  float *C, int ldc);
} CKMathBackend;

/**
 * Obtain the built-in native backend (single-node CPU, C + intrinsics).
 */
CKMathBackend ckernel_backend_native(void);

// Expose the individual GEMM kernels copied from C-Transformer.
void gemm_naive_parallel(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K);

void gemm_avx512_parallel(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K);

void gemm_fine_grained_parallel(const float *A,
                                const float *B,
                                const float *bias,
                                float *C,
                                int M, int N, int K);

void gemm_blocked_serial(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K);

// LayerNorm forward kernels, copied from C-Transformer.
void layernorm_naive_serial(const float *input,
                            const float *gamma,
                            const float *beta,
                            float *output,
                            float *mean_cache,
                            float *rstd_cache,
                            int tokens, int d_model, int aligned_embed_dim,
                            float eps);

void layernorm_forward_rolled_slice(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                    float *__restrict mean_cache_slice,
                                    float *__restrict rstd_cache_slice,
                                    int num_tokens_in_slice,
                                    int d_model,
                                    int aligned_embed_dim,
                                    float eps);

void layernorm_forward_unrolled_slice(const float *__restrict input_slice_base,
                                      const float *__restrict gamma,
                                      const float *__restrict beta,
                                      float *__restrict output_slice_base,
                                      float *__restrict mean_cache_slice,
                                      float *__restrict rstd_cache_slice,
                                      int num_tokens_in_slice,
                                      int d_model,
                                      float eps);

void layernorm_naive_serial_matched_precision(const float *input,
                                              const float *gamma,
                                              const float *beta,
                                              float *output,
                                              float *mean_cache,
                                              float *rstd_cache,
                                              int tokens, int d_model, float eps);

void layernorm_backward_kernel(const float *d_output,
                               const float *input,
                               const float *gamma,
                               const float *mean,
                               const float *rstd,
                               float *d_input,
                               float *d_gamma,
                               float *d_beta,
                               int tokens, int d_model, int aligned_embed_dim);

// GELU forward kernel (fast approximation), copied from C-Transformer.
void gelu_fast_inplace(float *data, size_t n);

// Causal softmax kernel on head-major attention scores, copied from C-Transformer.
void causal_softmax_head_major(float *scores,
                               int num_heads,
                               int num_tokens,
                               int aligned_context_window);

void backward_causal_softmax_head_major(float *d_scores,
                                        const float *weights,
                                        int num_heads,
                                        int num_tokens,
                                        int aligned_context_window);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CKERNEL_ENGINE_H
