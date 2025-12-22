#ifndef CKERNEL_ENGINE_H
#define CKERNEL_ENGINE_H

#include <stddef.h>
#include <stdint.h>

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

// Enable stricter numeric parity (single-thread + double-accumulation GEMM).
void ck_set_strict_parity(int enabled);
int ck_strict_parity_enabled(void);

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

// RMSNorm forward/backward kernels.
void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd_cache,
                     int tokens,
                     int d_model,
                     int aligned_embed_dim,
                     float eps);

void rmsnorm_backward(const float *d_output,
                      const float *input,
                      const float *gamma,
                      const float *rstd_cache,
                      float *d_input,
                      float *d_gamma,
                      int tokens,
                      int d_model,
                      int aligned_embed_dim);

// GELU forward kernel (fast approximation), copied from C-Transformer.
void gelu_fast_inplace(float *data, size_t n);

void gelu_backward_exact(const float *input,
                         const float *d_output,
                         float *d_input,
                         size_t n);

void gelu_backward_fast(const float *input,
                        const float *d_output,
                        float *d_input,
                        size_t n);

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

// Scaled dot-product attention (causal) in head-major layout.
// Q/K/V layout: [head][token][head_dim] with stride aligned_head_dim.
// scores: [head][query_token][key_token] with stride aligned_context_window.
// output: same layout as Q/V.
void attention_forward_causal_head_major(const float *q,
                                         const float *k,
                                         const float *v,
                                         float *scores,
                                         float *output,
                                         int num_heads,
                                         int num_tokens,
                                         int head_dim,
                                         int aligned_head_dim,
                                         int aligned_context_window);

// GQA-aware attention: Q has num_heads, K/V have num_kv_heads.
void attention_forward_causal_head_major_gqa(const float *q,
                                             const float *k,
                                             const float *v,
                                             float *scores,
                                             float *output,
                                             int num_heads,
                                             int num_kv_heads,
                                         int num_tokens,
                                         int head_dim,
                                         int aligned_head_dim,
                                         int aligned_context_window);

// MLP forward kernel (FC1 -> GELU -> FC2), generic token-parallel version.
void mlp_token_parallel(const float *input,
                        const float *W_fc1,
                        const float *b_fc1,
                        const float *W_fc2,
                        const float *b_fc2,
                        float *fc1_output,
                        float *output,
                        int T,
                        int aligned_dim,
                        int num_threads);

// MLP FC1/FC2 backward kernels (generic), adapted from C-Transformer.
void fc2_backward_kernel(const float *d_output,
                         const float *fc2_input,
                         const float *W_fc2,
                         float *d_input,
                         float *d_W_fc2,
                         float *d_b_fc2,
                         int T,
                         int aligned_in,
                         int aligned_out,
                         int num_threads);

void fc1_backward_kernel(const float *d_output,
                         const float *fc1_input,
                         const float *W_fc1,
                         float *d_input,
                         float *d_W_fc1,
                         float *d_b_fc1,
                         int T,
                         int aligned_in,
                         int aligned_out,
                         int num_threads);

// Sigmoid kernels (scalar + vector forms).
float sigmoid_scalar(float x);

void sigmoid_forward(const float *input,
                     float *output,
                     size_t n);

void sigmoid_backward(const float *input,
                      const float *d_output,
                      float *d_input,
                      size_t n);

// SwiGLU activation kernels (forward + backward).
// Input layout per token: [gate[0..D-1], value[0..D-1]], size 2*D.
// Output: [D].
void swiglu_forward(const float *input,
                    float *output,
                    int tokens,
                    int dim);

void swiglu_backward(const float *input,
                     const float *d_output,
                     float *d_input,
                     int tokens,
                     int dim);

// Attention backward (GQA-aware): computes d_q, d_k, d_v.
void attention_backward_causal_head_major_gqa(
    const float *d_output,
    const float *q,
    const float *k,
    const float *v,
    const float *attn_weights,
    float *d_q,
    float *d_k,
    float *d_v,
    float *d_scores,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window);

// Attention backward (non-GQA): num_kv_heads == num_heads.
void attention_backward_causal_head_major(
    const float *d_output,
    const float *q,
    const float *k,
    const float *v,
    const float *attn_weights,
    float *d_q,
    float *d_k,
    float *d_v,
    float *d_scores,
    int num_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window);

// RoPE (Rotary Position Embedding) kernels.
// Precompute cos/sin cache: [max_seq_len, head_dim/2].
void rope_precompute_cache(float *cos_cache,
                           float *sin_cache,
                           int max_seq_len,
                           int head_dim,
                           float base);

// Apply RoPE forward in-place: x[num_heads, num_tokens, aligned_head_dim].
void rope_forward(float *x,
                  const float *cos_cache,
                  const float *sin_cache,
                  int num_heads,
                  int num_tokens,
                  int head_dim,
                  int aligned_head_dim,
                  int pos_offset);

// RoPE backward: inverse rotation.
void rope_backward(const float *d_out,
                   float *d_x,
                   const float *cos_cache,
                   const float *sin_cache,
                   int num_heads,
                   int num_tokens,
                   int head_dim,
                   int aligned_head_dim,
                   int pos_offset);

// RoPE backward in-place.
void rope_backward_inplace(float *d_x,
                           const float *cos_cache,
                           const float *sin_cache,
                           int num_heads,
                           int num_tokens,
                           int head_dim,
                           int aligned_head_dim,
                           int pos_offset);

// Combined RoPE for Q and K.
void rope_forward_qk(float *q,
                     float *k,
                     const float *cos_cache,
                     const float *sin_cache,
                     int num_heads,
                     int num_kv_heads,
                     int num_tokens,
                     int head_dim,
                     int aligned_head_dim,
                     int pos_offset);

void rope_backward_qk(const float *d_q_out,
                      const float *d_k_out,
                      float *d_q,
                      float *d_k,
                      const float *cos_cache,
                      const float *sin_cache,
                      int num_heads,
                      int num_kv_heads,
                      int num_tokens,
                      int head_dim,
                      int aligned_head_dim,
                      int pos_offset);

// Token embedding lookup (optionally adds positional embeddings).
// token_embeddings: [vocab_size x aligned_embed_dim]
// pos_embeddings: [context_window x aligned_embed_dim] or NULL if add_pos == 0.
// output: [context_window x aligned_embed_dim]
void embedding_forward(const int32_t *token_ids,
                       int token_count,
                       int vocab_size,
                       const float *token_embeddings,
                       const float *pos_embeddings,
                       float *output,
                       int embed_dim,
                       int aligned_embed_dim,
                       int context_window,
                       int add_pos);

// Embedding backward: accumulates into d_token_embeddings and d_pos_embeddings.
// d_output: [context_window x aligned_embed_dim]
// d_token_embeddings: [vocab_size x aligned_embed_dim]
// d_pos_embeddings: [context_window x aligned_embed_dim] (optional)
void embedding_backward(const int32_t *token_ids,
                        int token_count,
                        const float *d_output,
                        float *d_token_embeddings,
                        float *d_pos_embeddings,
                        int vocab_size,
                        int embed_dim,
                        int aligned_embed_dim,
                        int context_window,
                        int add_pos);

// Softmax cross-entropy loss + gradient w.r.t logits.
// logits: [tokens x vocab_size], targets: [tokens], d_logits: [tokens x vocab_size]
void softmax_cross_entropy_loss(const float *logits,
                                const int32_t *targets,
                                int tokens,
                                int vocab_size,
                                float *d_logits,
                                float *loss_out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CKERNEL_ENGINE_H
