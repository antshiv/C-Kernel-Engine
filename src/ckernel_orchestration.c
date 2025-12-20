#include "ckernel_orchestration.h"

#include "ckernel_engine.h"

#include <stddef.h>

void ck_residual_add_token_major(const float *a,
                                 const float *b,
                                 float *out,
                                 int tokens,
                                 int aligned_embed_dim)
{
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        out[i] = a[i] + b[i];
    }
}

void ck_qkv_project_head_major(const float *input,
                               const float *wq, const float *bq,
                               const float *wk, const float *bk,
                               const float *wv, const float *bv,
                               float *q, float *k, float *v,
                               int tokens,
                               int aligned_embed_dim,
                               int num_heads,
                               int num_kv_heads,
                               int aligned_head_dim)
{
    if (!input || !wq || !wk || !wv || !q || !k || !v) {
        return;
    }

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    size_t head_out_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *wq_h = wq + (size_t)h * head_weight_stride;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * head_out_stride;

        gemm_blocked_serial(input, wq_h, bq_h, q_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * head_out_stride;
        float *v_h = v + (size_t)h * head_out_stride;

        gemm_blocked_serial(input, wk_h, bk_h, k_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
        gemm_blocked_serial(input, wv_h, bv_h, v_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }
}

static void ck_add_inplace(float *dst,
                           const float *src,
                           int tokens,
                           int aligned_embed_dim)
{
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        dst[i] += src[i];
    }
}

void ck_attention_project_head_major(const float *attn_out,
                                     const float *wo,
                                     const float *bo,
                                     float *out,
                                     float *scratch,
                                     int tokens,
                                     int aligned_embed_dim,
                                     int num_heads,
                                     int aligned_head_dim)
{
    if (!attn_out || !wo || !out) {
        return;
    }
    if (num_heads > 1 && !scratch) {
        return;
    }

    size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *head_in = attn_out + (size_t)h * head_in_stride;
        const float *wo_h = wo + (size_t)h * head_weight_stride;

        if (h == 0) {
            gemm_blocked_serial(head_in, wo_h, bo, out,
                                tokens, aligned_embed_dim, aligned_head_dim);
        } else {
            gemm_blocked_serial(head_in, wo_h, NULL, scratch,
                                tokens, aligned_embed_dim, aligned_head_dim);
            ck_add_inplace(out, scratch, tokens, aligned_embed_dim);
        }
    }
}

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
                           int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    gemm_blocked_serial(input, w1, b1, fc1_out,
                        tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_blocked_serial(swiglu_out, w2, b2, output,
                        tokens, aligned_embed_dim, aligned_intermediate_dim);
}

void ck_layer_forward_rmsnorm_swiglu(const CKLayerForwardParams *p)
{
    if (!p) {
        return;
    }

    rmsnorm_forward(p->input,
                    p->ln1_gamma,
                    p->ln1_out,
                    p->ln1_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_qkv_project_head_major(p->ln1_out,
                              p->wq, p->bq,
                              p->wk, p->bk,
                              p->wv, p->bv,
                              p->q, p->k, p->v,
                              p->tokens,
                              p->aligned_embed_dim,
                              p->num_heads,
                              p->num_kv_heads,
                              p->aligned_head_dim);

    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(p->q,
                        p->k,
                        p->rope_cos,
                        p->rope_sin,
                        p->num_heads,
                        p->num_kv_heads,
                        p->tokens,
                        p->head_dim,
                        p->aligned_head_dim,
                        p->rope_pos_offset);
    }

    attention_forward_causal_head_major_gqa(p->q,
                                            p->k,
                                            p->v,
                                            p->scores,
                                            p->attn_out,
                                            p->num_heads,
                                            p->num_kv_heads,
                                            p->tokens,
                                            p->head_dim,
                                            p->aligned_head_dim,
                                            p->aligned_context_window);

    ck_attention_project_head_major(p->attn_out,
                                    p->wo,
                                    p->bo,
                                    p->proj_tmp,
                                    p->proj_scratch,
                                    p->tokens,
                                    p->aligned_embed_dim,
                                    p->num_heads,
                                    p->aligned_head_dim);

    ck_residual_add_token_major(p->input,
                                p->proj_tmp,
                                p->residual1,
                                p->tokens,
                                p->aligned_embed_dim);

    rmsnorm_forward(p->residual1,
                    p->ln2_gamma,
                    p->ln2_out,
                    p->ln2_rstd,
                    p->tokens,
                    p->embed_dim,
                    p->aligned_embed_dim,
                    p->eps);

    ck_mlp_swiglu_forward(p->ln2_out,
                          p->w1,
                          p->b1,
                          p->w2,
                          p->b2,
                          p->fc1_out,
                          p->swiglu_out,
                          p->mlp_out,
                          p->tokens,
                          p->aligned_embed_dim,
                          p->aligned_intermediate_dim);

    ck_residual_add_token_major(p->residual1,
                                p->mlp_out,
                                p->output,
                                p->tokens,
                                p->aligned_embed_dim);
}
