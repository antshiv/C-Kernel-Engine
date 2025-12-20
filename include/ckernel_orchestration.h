#ifndef CKERNEL_ORCHESTRATION_H
#define CKERNEL_ORCHESTRATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int tokens;
    int embed_dim;
    int aligned_embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int aligned_head_dim;
    int aligned_context_window;
    int intermediate_dim;
    int aligned_intermediate_dim;
    float eps;
    int rope_pos_offset;

    const float *input;     /* [T x aligned_embed_dim] */
    const float *ln1_gamma; /* [aligned_embed_dim] */
    const float *ln2_gamma; /* [aligned_embed_dim] */

    const float *rope_cos; /* [max_seq_len x head_dim/2] */
    const float *rope_sin; /* [max_seq_len x head_dim/2] */

    const float *wq; /* [num_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bq; /* [num_heads x aligned_head_dim] */
    const float *wk; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bk; /* [num_kv_heads x aligned_head_dim] */
    const float *wv; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bv; /* [num_kv_heads x aligned_head_dim] */

    const float *wo; /* [H x aligned_embed_dim x aligned_head_dim] */
    const float *bo; /* [aligned_embed_dim] */

    const float *w1; /* [2*aligned_intermediate_dim x aligned_embed_dim] */
    const float *b1; /* [2*aligned_intermediate_dim] */
    const float *w2; /* [aligned_embed_dim x aligned_intermediate_dim] */
    const float *b2; /* [aligned_embed_dim] */

    float *ln1_out;   /* [T x aligned_embed_dim] */
    float *ln1_rstd;  /* [T] (optional) */
    float *q;         /* [num_heads x T x aligned_head_dim] */
    float *k;         /* [num_kv_heads x T x aligned_head_dim] */
    float *v;         /* [num_kv_heads x T x aligned_head_dim] */
    float *scores;    /* [num_heads x aligned_context_window x aligned_context_window] */
    float *attn_out;  /* [num_heads x T x aligned_head_dim] */
    float *proj_tmp;  /* [T x aligned_embed_dim] */
    float *proj_scratch; /* [T x aligned_embed_dim], required if num_heads > 1 */
    float *residual1; /* [T x aligned_embed_dim] */
    float *ln2_out;   /* [T x aligned_embed_dim] */
    float *ln2_rstd;  /* [T] (optional) */
    float *fc1_out;   /* [T x 2*aligned_intermediate_dim] */
    float *swiglu_out;/* [T x aligned_intermediate_dim] */
    float *mlp_out;   /* [T x aligned_embed_dim] */
    float *output;    /* [T x aligned_embed_dim] */
} CKLayerForwardParams;

void ck_residual_add_token_major(const float *a,
                                 const float *b,
                                 float *out,
                                 int tokens,
                                 int aligned_embed_dim);

void ck_qkv_project_head_major(const float *input,
                               const float *wq, const float *bq,
                               const float *wk, const float *bk,
                               const float *wv, const float *bv,
                               float *q, float *k, float *v,
                               int tokens,
                               int aligned_embed_dim,
                               int num_heads,
                               int num_kv_heads,
                               int aligned_head_dim);

void ck_attention_project_head_major(const float *attn_out,
                                     const float *wo,
                                     const float *bo,
                                     float *out,
                                     float *scratch,
                                     int tokens,
                                     int aligned_embed_dim,
                                     int num_heads,
                                     int aligned_head_dim);

void ck_mlp_swiglu_forward(const float *input,
                           const float *w1,
                           const float *b1,
                           const float *w2,
                           const float *b2,
                           float *fc1_out,
                           float *swiglu_out,
                           float *output,
                           int tokens,
                           int aligned_embed_dim,
                           int aligned_intermediate_dim);

void ck_layer_forward_rmsnorm_swiglu(const CKLayerForwardParams *p);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* CKERNEL_ORCHESTRATION_H */
