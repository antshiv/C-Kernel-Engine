#include "ckernel_orchestration.h"

#include "ckernel_engine.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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

void ck_residual_add_backward(const float *d_out,
                              float *d_a,
                              float *d_b,
                              int tokens,
                              int aligned_embed_dim)
{
    if (!d_out || !d_a || !d_b) {
        return;
    }
    size_t total = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total; ++i) {
        float v = d_out[i];
        d_a[i] = v;
        d_b[i] = v;
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

static void ck_qkv_project_head_major_ref(const float *input,
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

        gemm_naive_parallel(input, wq_h, bq_h, q_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * head_out_stride;
        float *v_h = v + (size_t)h * head_out_stride;

        gemm_naive_parallel(input, wk_h, bk_h, k_h,
                            tokens, aligned_head_dim, aligned_embed_dim);
        gemm_naive_parallel(input, wv_h, bv_h, v_h,
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

static void ck_attention_project_head_major_ref(const float *attn_out,
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
            gemm_naive_parallel(head_in, wo_h, bo, out,
                                tokens, aligned_embed_dim, aligned_head_dim);
        } else {
            gemm_naive_parallel(head_in, wo_h, NULL, scratch,
                                tokens, aligned_embed_dim, aligned_head_dim);
            ck_add_inplace(out, scratch, tokens, aligned_embed_dim);
        }
    }
}

void ck_attention_project_head_major_backward(const float *d_out,
                                              const float *attn_out,
                                              const float *wo,
                                              float *d_attn_out,
                                              float *d_wo,
                                              float *d_bo,
                                              int tokens,
                                              int aligned_embed_dim,
                                              int num_heads,
                                              int aligned_head_dim)
{
    if (!d_out || !attn_out || !wo || !d_attn_out || !d_wo || !d_bo) {
        return;
    }

    // Bias gradient: sum over tokens once (bias is applied once in forward).
    for (int d = 0; d < aligned_embed_dim; ++d) {
        d_bo[d] = 0.0f;
    }
    for (int t = 0; t < tokens; ++t) {
        const float *row = d_out + (size_t)t * (size_t)aligned_embed_dim;
        for (int d = 0; d < aligned_embed_dim; ++d) {
            d_bo[d] += row[d];
        }
    }

    size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;
    size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    float *tmp_b = (float *)calloc((size_t)aligned_embed_dim, sizeof(float));
    if (!tmp_b) {
        return;
    }

    for (int h = 0; h < num_heads; ++h) {
        const float *head_in = attn_out + (size_t)h * head_in_stride;
        const float *wo_h = wo + (size_t)h * head_weight_stride;
        float *d_head_in = d_attn_out + (size_t)h * head_in_stride;
        float *d_wo_h = d_wo + (size_t)h * head_weight_stride;

        memset(tmp_b, 0, (size_t)aligned_embed_dim * sizeof(float));
        fc2_backward_kernel(d_out,
                            head_in,
                            wo_h,
                            d_head_in,
                            d_wo_h,
                            tmp_b,
                            tokens,
                            aligned_head_dim,
                            aligned_embed_dim,
                            1);
    }

    free(tmp_b);
}

void ck_qkv_project_head_major_backward(const float *d_q,
                                        const float *d_k,
                                        const float *d_v,
                                        const float *input,
                                        const float *wq,
                                        const float *bq,
                                        const float *wk,
                                        const float *bk,
                                        const float *wv,
                                        const float *bv,
                                        float *d_input,
                                        float *d_wq,
                                        float *d_bq,
                                        float *d_wk,
                                        float *d_bk,
                                        float *d_wv,
                                        float *d_bv,
                                        float *scratch,
                                        int tokens,
                                        int aligned_embed_dim,
                                        int num_heads,
                                        int num_kv_heads,
                                        int aligned_head_dim,
                                        int num_threads)
{
    if (!d_q || !d_k || !d_v || !input || !wq || !wk || !wv ||
        !d_input || !d_wq || !d_bq || !d_wk || !d_bk || !d_wv || !d_bv || !scratch) {
        return;
    }

    size_t total_in = (size_t)tokens * (size_t)aligned_embed_dim;
    for (size_t i = 0; i < total_in; ++i) {
        d_input[i] = 0.0f;
    }

    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    size_t head_out_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *d_q_h = d_q + (size_t)h * head_out_stride;
        const float *wq_h = wq + (size_t)h * head_weight_stride;
        float *d_wq_h = d_wq + (size_t)h * head_weight_stride;
        float *d_bq_h = d_bq + (size_t)h * (size_t)aligned_head_dim;

        fc2_backward_kernel(d_q_h,
                            input,
                            wq_h,
                            scratch,
                            d_wq_h,
                            d_bq_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *d_k_h = d_k + (size_t)h * head_out_stride;
        const float *d_v_h = d_v + (size_t)h * head_out_stride;

        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;

        float *d_wk_h = d_wk + (size_t)h * head_weight_stride;
        float *d_wv_h = d_wv + (size_t)h * head_weight_stride;

        float *d_bk_h = d_bk + (size_t)h * (size_t)aligned_head_dim;
        float *d_bv_h = d_bv + (size_t)h * (size_t)aligned_head_dim;

        fc2_backward_kernel(d_k_h,
                            input,
                            wk_h,
                            scratch,
                            d_wk_h,
                            d_bk_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);

        fc2_backward_kernel(d_v_h,
                            input,
                            wv_h,
                            scratch,
                            d_wv_h,
                            d_bv_h,
                            tokens,
                            aligned_embed_dim,
                            aligned_head_dim,
                            num_threads);
        ck_add_inplace(d_input, scratch, tokens, aligned_embed_dim);
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

static void ck_mlp_swiglu_forward_ref(const float *input,
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
    gemm_naive_parallel(input, w1, b1, fc1_out,
                        tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_naive_parallel(swiglu_out, w2, b2, output,
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

void ck_layer_forward_rmsnorm_swiglu_ref(const CKLayerForwardParams *p)
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

    ck_qkv_project_head_major_ref(p->ln1_out,
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

    ck_attention_project_head_major_ref(p->attn_out,
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

    ck_mlp_swiglu_forward_ref(p->ln2_out,
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

void ck_layer_backward_rmsnorm_swiglu(const CKLayerBackwardParams *p)
{
    if (!p) {
        return;
    }

    int T = p->tokens;
    int aligned_embed = p->aligned_embed_dim;
    int aligned_head = p->aligned_head_dim;
    int aligned_intermediate = p->aligned_intermediate_dim;
    int up_dim = 2 * aligned_intermediate;
    int num_threads = 1;

    // 1) Residual add (output = residual1 + mlp_out)
    ck_residual_add_backward(p->d_output, p->d_residual1, p->d_mlp_out, T, aligned_embed);

    // 2) MLP down proj backward
    fc2_backward_kernel(p->d_mlp_out,
                        p->swiglu_out,
                        p->w2,
                        p->d_swiglu_out,
                        p->d_w2,
                        p->d_b2,
                        T,
                        aligned_intermediate,
                        aligned_embed,
                        num_threads);

    // 3) SwiGLU backward
    swiglu_backward(p->fc1_out, p->d_swiglu_out, p->d_fc1_out, T, aligned_intermediate);

    // 4) MLP up proj backward
    fc1_backward_kernel(p->d_fc1_out,
                        p->ln2_out,
                        p->w1,
                        p->d_ln2_out,
                        p->d_w1,
                        p->d_b1,
                        T,
                        aligned_embed,
                        up_dim,
                        num_threads);

    // 5) RMSNorm (ln2) backward; reuse d_output as scratch for d_residual1_from_ln2
    rmsnorm_backward(p->d_ln2_out,
                     p->residual1,
                     p->ln2_gamma,
                     p->ln2_rstd,
                     p->d_output,
                     p->d_ln2_gamma,
                     T,
                     p->embed_dim,
                     aligned_embed);
    ck_add_inplace(p->d_residual1, p->d_output, T, aligned_embed);

    // 6) Residual add (residual1 = input + proj_tmp)
    ck_residual_add_backward(p->d_residual1, p->d_input, p->d_proj_tmp, T, aligned_embed);

    // 7) Attention projection backward
    ck_attention_project_head_major_backward(p->d_proj_tmp,
                                             p->attn_out,
                                             p->wo,
                                             p->d_attn_out,
                                             p->d_wo,
                                             p->d_bo,
                                             T,
                                             aligned_embed,
                                             p->num_heads,
                                             aligned_head);

    // 8) Attention backward
    attention_backward_causal_head_major_gqa(p->d_attn_out,
                                             p->q,
                                             p->k,
                                             p->v,
                                             p->scores,
                                             p->d_q,
                                             p->d_k,
                                             p->d_v,
                                             p->d_scores,
                                             p->num_heads,
                                             p->num_kv_heads,
                                             T,
                                             p->head_dim,
                                             aligned_head,
                                             p->aligned_context_window);

    // 9) RoPE backward (if enabled)
    if (p->rope_cos && p->rope_sin) {
        rope_backward_qk(p->d_q,
                         p->d_k,
                         p->d_q,
                         p->d_k,
                         p->rope_cos,
                         p->rope_sin,
                         p->num_heads,
                         p->num_kv_heads,
                         T,
                         p->head_dim,
                         aligned_head,
                         p->rope_pos_offset);
    }

    // 10) QKV projection backward (scratch uses d_proj_tmp)
    ck_qkv_project_head_major_backward(p->d_q,
                                       p->d_k,
                                       p->d_v,
                                       p->ln1_out,
                                       p->wq,
                                       p->bq,
                                       p->wk,
                                       p->bk,
                                       p->wv,
                                       p->bv,
                                       p->d_ln1_out,
                                       p->d_wq,
                                       p->d_bq,
                                       p->d_wk,
                                       p->d_bk,
                                       p->d_wv,
                                       p->d_bv,
                                       p->d_proj_tmp,
                                       T,
                                       aligned_embed,
                                       p->num_heads,
                                       p->num_kv_heads,
                                       aligned_head,
                                       num_threads);

    // 11) RMSNorm (ln1) backward; reuse d_ln1_out as scratch for d_input_from_ln1
    rmsnorm_backward(p->d_ln1_out,
                     p->input,
                     p->ln1_gamma,
                     p->ln1_rstd,
                     p->d_ln1_out,
                     p->d_ln1_gamma,
                     T,
                     p->embed_dim,
                     aligned_embed);
    ck_add_inplace(p->d_input, p->d_ln1_out, T, aligned_embed);
}
