#ifndef CKERNEL_ENGINE_H
#define CKERNEL_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "cpu_features.h"

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

// Thread configuration - call once at startup
// num_threads: 0 = auto-detect physical cores, >0 = use specified count
void ck_set_num_threads(int num_threads);
int ck_get_num_threads(void);
int ck_get_physical_cores(void);

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

	// Reference BF16 GEMM (A/B/bias in BF16, output BF16).
void gemm_blocked_serial_bf16(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M, int N, int K);

// =============================================================================
// Quantized (GGML-style) GEMM/GEMV helpers
// =============================================================================
//
// These kernels are used for weight-only quantized inference (e.g. Q4_K_M).
// The "NT" wrapper matches the engine's common layout:
//   A: [M x K] fp32 (token-major)
//   B: [N x K] quantized (row-major by output channel)
//   C: [M x N] fp32
//
// NOTE: Q4_K requires K to be a multiple of 256 (QK_K).

void gemv_q4_k(float *y,
               const void *W,
               const float *x,
               int M, int K);

void gemm_q4_k(float *Y,
               const void *W,
               const float *X,
               int M, int N, int K);

void gemm_nt_q4_k(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K);

void dequant_q4_k_row(const void *src, float *dst, size_t n_elements);

// GEMM_NN: C[M,N] = A[M,K] @ B[K,N] + bias[N]
// B is stored row-major as [K,N] (no transpose on B)
// Used for backward d_input = d_output @ W
void gemm_nn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K);

void gemm_nn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K);

void gemm_nn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K);

// GEMM_TN: C[M,N] = A[K,M].T @ B[K,N] + bias[N]
// A is stored row-major as [K,M], B is stored row-major as [K,N]
// Used for backward d_W = d_output.T @ input
void gemm_tn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K);

void gemm_tn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K);

void gemm_tn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K);

// Fused GEMM operations (GEMM + bias + activation in one pass)
void gemm_bias_relu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K);

void gemm_bias_gelu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K);

void gemm_bias_silu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K);

// Fused GEMM + SwiGLU (LLaMA/SmolLM MLP gate+up projection)
// Computes: output = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
// Two GEMMs + SwiGLU fused into one pass - intermediates stay in registers
void gemm_swiglu_fused(const float *x,
                       const float *W_gate,
                       const float *W_up,
                       const float *b_gate,  // can be NULL
                       const float *b_up,    // can be NULL
                       float *output,
                       int M, int N, int K);

// High-performance GEMM microkernel with 8x8 register blocking
// Inspired by oneDNN/BLIS - keeps all 64 accumulator values in registers
// C[M,N] = A[M,K] @ B[K,N] or C[M,N] = A[M,K] @ B[N,K].T
// B_transposed: 0 = B is [K,N], 1 = B is [N,K] (transposed, common in NN weights)
void gemm_microkernel(const float *A,
                      const float *B,
                      float *C,
                      int M, int N, int K,
                      int B_transposed);

// Cache-blocked GEMM using 8x8 microkernels (B not transposed)
void gemm_microkernel_blocked(const float *A,
                              const float *B,
                              float *C,
                              int M, int N, int K);

// Cache-blocked GEMM for B transposed (common in NN)
void gemm_microkernel_blocked_bt(const float *A,
                                 const float *B,
                                 float *C,
                                 int M, int N, int K);

// Optimized GEMM with matrix packing (best for large matrices)
// Packs A and B into contiguous layouts for optimal cache access
void gemm_microkernel_packed(const float *A,
                             const float *B,
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

void layernorm_forward_rolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                         const float *__restrict gamma,
                                         const float *__restrict beta,
                                         uint16_t *__restrict output_slice_base,
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

void layernorm_forward_unrolled_slice_bf16(const uint16_t *__restrict input_slice_base,
                                           const float *__restrict gamma,
                                           const float *__restrict beta,
                                           uint16_t *__restrict output_slice_base,
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

void layernorm_backward_kernel_bf16(const uint16_t *d_output,
                                    const uint16_t *input,
                                    const float *gamma,
                                    const float *mean,
                                    const float *rstd,
                                    uint16_t *d_input,
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

void rmsnorm_forward_bf16(const uint16_t *input,
                          const float *gamma,
                          uint16_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps);

void rmsnorm_backward_bf16(const uint16_t *d_output,
                           const uint16_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           uint16_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim);

void rmsnorm_forward_int8(const int8_t *input,
                          const float *gamma,
                          int8_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps);

void rmsnorm_backward_int8(const int8_t *d_output,
                           const int8_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           int8_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim);

void rmsnorm_forward_int4(const uint8_t *input,
                          const float *gamma,
                          uint8_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps);

void rmsnorm_backward_int4(const uint8_t *d_output,
                           const uint8_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           uint8_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim);

// GELU forward kernel (fast approximation), copied from C-Transformer.
void gelu_fast_inplace(float *data, size_t n);

// Scalar-only exact GELU forward using standard library tanhf.
// Slower but provides maximum accuracy. Used by BF16 wrapper.
void gelu_exact_inplace(float *data, size_t n);

// GELU backward using tanh-based derivative (vectorized, uses fast tanh approx).
void gelu_backward_exact(const float *input,
                         const float *d_output,
                         float *d_input,
                         size_t n);

// Scalar-only exact GELU backward using standard library tanhf.
// Slower but provides maximum accuracy. Used by BF16 wrapper.
void gelu_backward_scalar(const float *input,
                          const float *d_output,
                          float *d_input,
                          size_t n);

void gelu_backward_fast(const float *input,
                        const float *d_output,
                        float *d_input,
                        size_t n);

// BF16 variants relying on the same floating-point logic.
void gelu_fast_inplace_bf16(uint16_t *data, size_t n);
void gelu_backward_exact_bf16(const uint16_t *input,
                              const uint16_t *d_output,
                              uint16_t *d_input,
                              size_t n);
void gelu_backward_fast_bf16(const uint16_t *input,
                             const uint16_t *d_output,
                             uint16_t *d_input,
                             size_t n);

	// ReLU kernels.
	void relu_forward(const float *input, float *output, size_t n);
	void relu_forward_inplace(float *data, size_t n);
	void relu_backward(const float *input,
	                   const float *d_output,
	                   float *d_input,
	                   size_t n);

	void relu_forward_bf16(const uint16_t *input, uint16_t *output, size_t n);
	void relu_forward_inplace_bf16(uint16_t *data, size_t n);
	void relu_backward_bf16(const uint16_t *input,
	                        const uint16_t *d_output,
	                        uint16_t *d_input,
	                        size_t n);

	// Causal softmax kernel on head-major attention scores, copied from C-Transformer.
	void causal_softmax_head_major(float *scores,
	                               int num_heads,
	                               int num_tokens,
	                               int aligned_context_window);

	// Scalar-only exact causal softmax using standard library expf.
	// Slower but provides maximum accuracy. Used by BF16 attention wrapper.
	void causal_softmax_head_major_exact(float *scores,
	                                      int num_heads,
	                                      int num_tokens,
	                                      int aligned_context_window);

	void backward_causal_softmax_head_major(float *d_scores,
	                                        const float *weights,
	                                        int num_heads,
	                                        int num_tokens,
	                                        int aligned_context_window);

	void causal_softmax_head_major_bf16(uint16_t *scores,
	                                   int num_heads,
	                                   int num_tokens,
	                                   int aligned_context_window);

	void backward_causal_softmax_head_major_bf16(uint16_t *d_scores,
	                                            const uint16_t *weights,
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

// Exact version using standard library expf (slower but accurate).
void attention_forward_causal_head_major_exact(const float *q,
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

void attention_forward_causal_head_major_gqa_bf16(const uint16_t *q,
                                                  const uint16_t *k,
                                                  const uint16_t *v,
                                                  float *scores,
                                                  float *output,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int num_tokens,
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  int aligned_context_window);

// Flash-style causal attention forward (no score/weight matrix materialization).
// Head-major layout:
//   Q: [num_heads, num_tokens, aligned_head_dim]
//   K/V: [num_kv_heads, num_tokens, aligned_head_dim]
//   out: [num_heads, num_tokens, aligned_head_dim]
void attention_forward_causal_head_major_gqa_flash(const float *q,
                                                   const float *k,
                                                   const float *v,
                                                   float *output,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int num_tokens,
                                                   int head_dim,
                                                   int aligned_head_dim);

// Decode attention for a single token using a KV cache.
//   q_token: [num_heads, aligned_head_dim]
//   k_cache/v_cache: [num_kv_heads, cache_capacity, aligned_head_dim]
//   out_token: [num_heads, aligned_head_dim]
void attention_forward_decode_head_major_gqa_flash(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim);

// KV cache helper (write one token for all KV heads).
void kv_cache_write_head_major(const float *k_token,
                               const float *v_token,
                               float *k_cache,
                               float *v_cache,
                               int num_kv_heads,
                               int token_index,
                               int cache_capacity,
                               int head_dim,
                               int aligned_head_dim);

// Repack a head-major tensor from a packed `[head, tokens, aligned_head_dim]`
// layout into a KV-cache-compatible layout `[head, cache_capacity, aligned_head_dim]`
// in-place. This is used after prefill when forward kernels write head slices
// back-to-back using `tokens` as the head stride, but decode expects a fixed
// `cache_capacity` stride.
void kv_cache_repack_head_major_inplace(float *buf,
                                        int num_heads,
                                        int tokens,
                                        int cache_capacity,
                                        int aligned_head_dim);

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

// Exact version using scalar GELU with standard library tanhf.
// Slower but provides maximum accuracy. Used for correctness testing.
void mlp_token_parallel_exact(const float *input,
                               const float *W_fc1,
                               const float *b_fc1,
                               const float *W_fc2,
                               const float *b_fc2,
                               float *fc1_output,
                               float *output,
                               int T,
                               int aligned_dim,
                               int num_threads);

void mlp_token_parallel_bf16(const uint16_t *input,
                             const uint16_t *W_fc1,
                             const uint16_t *b_fc1,
                             const uint16_t *W_fc2,
                             const uint16_t *b_fc2,
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

void sigmoid_forward_bf16(const uint16_t *input,
                          uint16_t *output,
                          size_t n);

void sigmoid_backward_bf16(const uint16_t *input,
                           const uint16_t *d_output,
                           uint16_t *d_input,
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

	// Exact versions using standard library expf (slower but accurate)
	void swiglu_forward_exact(const float *input,
	                          float *output,
	                          int tokens,
	                          int dim);

	void swiglu_backward_exact(const float *input,
	                           const float *d_output,
	                           float *d_input,
	                           int tokens,
	                           int dim);

	void swiglu_forward_bf16(const uint16_t *input,
	                         uint16_t *output,
	                         int tokens,
	                         int dim);

	void swiglu_backward_bf16(const uint16_t *input,
	                          const uint16_t *d_output,
	                          uint16_t *d_input,
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

void attention_backward_causal_head_major_gqa_bf16(
    const uint16_t *d_output,
    float *d_x,
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
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

void rope_forward_bf16(uint16_t *x,
                       const float *cos_cache,
                       const float *sin_cache,
                       int num_heads,
                       int num_tokens,
                       int head_dim,
                       int aligned_head_dim,
                       int pos_offset);

void rope_backward_bf16(const uint16_t *d_out,
                        uint16_t *d_x,
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

	void rope_forward_qk_bf16(uint16_t *q,
	                          uint16_t *k,
	                          const float *cos_cache,
	                          const float *sin_cache,
	                          int num_heads,
	                          int num_kv_heads,
	                          int num_tokens,
	                          int head_dim,
	                          int aligned_head_dim,
	                          int pos_offset);

	void rope_backward_qk_bf16(const uint16_t *d_q_out,
	                           const uint16_t *d_k_out,
	                           uint16_t *d_q,
	                           uint16_t *d_k,
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

	void embedding_forward_bf16(const int32_t *token_ids,
	                            int token_count,
	                            int vocab_size,
	                            const uint16_t *token_embeddings,
	                            const uint16_t *pos_embeddings,
	                            uint16_t *output,
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

	void embedding_backward_bf16(const int32_t *token_ids,
	                             int token_count,
	                             const uint16_t *d_output,
	                             uint16_t *d_token_embeddings,
	                             uint16_t *d_pos_embeddings,
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

	void softmax_cross_entropy_loss_bf16(const uint16_t *logits,
	                                     const int32_t *targets,
	                                     int tokens,
	                                     int vocab_size,
	                                     uint16_t *d_logits,
	                                     float *loss_out);

	// Vision helpers (patchify/unpatchify).
	void im2patch(const float *image,
	              float *patches,
	              int C, int H, int W, int P);
	void patch2im(const float *d_patches,
	              float *d_image,
	              int C, int H, int W, int P);

	void im2patch_bf16(const uint16_t *image,
	                   uint16_t *patches,
	                   int C, int H, int W, int P);
	void patch2im_bf16(const uint16_t *d_patches,
	                   uint16_t *d_image,
	                   int C, int H, int W, int P);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CKERNEL_ENGINE_H
