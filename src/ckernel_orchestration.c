#include "ckernel_orchestration.h"

#include "ckernel_engine.h"
#include "ckernel_dtype.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

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

/* ============================================================================
 * Q4_K (Q4_K_M) forward-only paths
 * ============================================================================
 *
 * These helpers keep the same activation layouts as the fp32 code paths, but
 * accept weight matrices stored as GGML-compatible Q4_K blocks. This is meant
 * for weight-only quantized inference: activations remain fp32.
 *
 * Important constraints:
 *   - Q4_K kernels require K (the input dimension) to be a multiple of 256.
 *   - For attention output projection we assume the concatenated head vector
 *     has length aligned_embed_dim (i.e., num_heads * aligned_head_dim matches).
 */

static void ck_qkv_project_head_major_q4_k(const float *input,
                                          const void *wq, const float *bq,
                                          const void *wk, const float *bk,
                                          const void *wv, const float *bv,
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

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t head_w_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);
    const size_t head_out_stride = (size_t)tokens * (size_t)aligned_head_dim;

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = wq_bytes + (size_t)h * head_w_bytes;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q + (size_t)h * head_out_stride;

        gemm_nt_q4_k(input, wq_h, bq_h, q_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = wk_bytes + (size_t)h * head_w_bytes;
        const void *wv_h = wv_bytes + (size_t)h * head_w_bytes;

        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;

        float *k_h = k + (size_t)h * head_out_stride;
        float *v_h = v + (size_t)h * head_out_stride;

        gemm_nt_q4_k(input, wk_h, bk_h, k_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
        gemm_nt_q4_k(input, wv_h, bv_h, v_h,
                     tokens, aligned_head_dim, aligned_embed_dim);
    }
}

static void ck_attention_project_head_major_q4_k(const float *attn_out,
                                                 const void *wo,
                                                 const float *bo,
                                                 float *out,
                                                 float *scratch,
                                                 int tokens,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int aligned_head_dim)
{
    if (!attn_out || !wo || !out || !scratch) {
        return;
    }

    /* Flatten head-major [H, T, ad] into token-major [T, H*ad] */
    const int K = num_heads * aligned_head_dim;
    if (K != aligned_embed_dim) {
        return;
    }

    const size_t head_in_stride = (size_t)tokens * (size_t)aligned_head_dim;

    for (int t = 0; t < tokens; ++t) {
        float *dst = scratch + (size_t)t * (size_t)aligned_embed_dim;
        for (int h = 0; h < num_heads; ++h) {
            const float *src = attn_out + (size_t)h * head_in_stride + (size_t)t * (size_t)aligned_head_dim;
            memcpy(dst + (size_t)h * (size_t)aligned_head_dim,
                   src,
                   (size_t)aligned_head_dim * sizeof(float));
        }
    }

    gemm_nt_q4_k(scratch, wo, bo, out,
                 tokens, aligned_embed_dim, aligned_embed_dim);
}

static void ck_mlp_swiglu_forward_q4_k(const float *input,
                                      const void *w1,
                                      const float *b1,
                                      const void *w2,
                                      const float *b2,
                                      float *fc1_out,
                                      float *swiglu_out,
                                      float *output,
                                      int tokens,
                                      int aligned_embed_dim,
                                      int aligned_intermediate_dim)
{
    int up_dim = 2 * aligned_intermediate_dim;
    gemm_nt_q4_k(input, w1, b1, fc1_out,
                 tokens, up_dim, aligned_embed_dim);

    swiglu_forward(fc1_out, swiglu_out, tokens, aligned_intermediate_dim);

    gemm_nt_q4_k(swiglu_out, w2, b2, output,
                 tokens, aligned_embed_dim, aligned_intermediate_dim);
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

    if (p->scores) {
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
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

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

    if (p->scores) {
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
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

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

static void ck_qkv_project_head_major_token(const float *input_row,
                                            const float *wq, const float *bq,
                                            const float *wk, const float *bk,
                                            const float *wv, const float *bv,
                                            float *q_token,
                                            float *k_token,
                                            float *v_token,
                                            int aligned_embed_dim,
                                            int num_heads,
                                            int num_kv_heads,
                                            int aligned_head_dim)
{
    // Decode-time QKV is a matrix-vector multiply (tokens=1). Parallelizing over heads
    // caps parallelism at small head counts (e.g., H=9). Since Wq is stored head-major as
    // contiguous rows ([H*D_head, D_model] row-major), we can flatten Q heads and expose a
    // larger N to GEMM for better multi-core utilization.
    const int q_out = num_heads * aligned_head_dim;
    gemm_blocked_serial(input_row, wq, bq, q_token,
                        /*tokens=*/1, q_out, aligned_embed_dim);

    // K/V have fewer heads (GQA), so keep head-parallel projection here to avoid the
    // overhead of spawning many threads for small N.
    size_t head_weight_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
#pragma omp parallel for schedule(static) if(num_kv_heads > 1)
    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_weight_stride;
        const float *wv_h = wv + (size_t)h * head_weight_stride;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;

        gemm_blocked_serial(input_row, wk_h, bk_h, k_h,
                            /*tokens=*/1, aligned_head_dim, aligned_embed_dim);
        gemm_blocked_serial(input_row, wv_h, bv_h, v_h,
                            /*tokens=*/1, aligned_head_dim, aligned_embed_dim);
    }
}

#if defined(__AVX__) && !defined(__AVX512F__)
static inline float ck_hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

static inline float ck_dot_f32(const float *a, const float *b, int len)
{
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= len - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float sum = ck_hsum256_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// Decode-specialized output projection:
//   out[embed_dim] = sum_h (attn[h] @ Wo[h]^T) + bias
// This is equivalent to the loop in ck_attention_project_head_major for tokens=1,
// but exposes enough parallelism (over output channels) to use many cores.
static void ck_attention_project_head_major_decode_token(const float *attn_token,
                                                        const float *wo,
                                                        const float *bo,
                                                        float *out_token,
                                                        int embed_dim,
                                                        int aligned_embed_dim,
                                                        int num_heads,
                                                        int aligned_head_dim)
{
    const size_t head_in_stride = (size_t)aligned_head_dim;
    const size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

#pragma omp parallel for schedule(static)
    for (int j = 0; j < embed_dim; ++j) {
        float sum = bo ? bo[j] : 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const float *head_in = attn_token + (size_t)h * head_in_stride;
            const float *wo_row = wo + (size_t)h * head_weight_stride + (size_t)j * (size_t)aligned_head_dim;
            sum += ck_dot_f32(head_in, wo_row, aligned_head_dim);
        }
        out_token[j] = sum;
    }

    // Clear padded output lanes so downstream GEMMs never see uninitialized data.
    for (int j = embed_dim; j < aligned_embed_dim; ++j) {
        out_token[j] = 0.0f;
    }
}

static void ck_mlp_swiglu_forward_fused_token(const float *input_row,
                                              const float *w1,
                                              const float *b1,
                                              const float *w2,
                                              const float *b2,
                                              float *swiglu_row,
                                              float *output_row,
                                              int aligned_embed_dim,
                                              int aligned_intermediate_dim)
{
    if (!input_row || !w1 || !w2 || !swiglu_row || !output_row) {
        return;
    }

    const float *w_gate = w1;
    const float *w_up = w1 + (size_t)aligned_intermediate_dim * (size_t)aligned_embed_dim;
    const float *b_gate = b1;
    const float *b_up = b1 ? (b1 + aligned_intermediate_dim) : NULL;

    gemm_swiglu_fused(input_row,
                      w_gate,
                      w_up,
                      b_gate,
                      b_up,
                      swiglu_row,
                      /*M=*/1,
                      /*N=*/aligned_intermediate_dim,
                      /*K=*/aligned_embed_dim);

    gemm_blocked_serial(swiglu_row, w2, b2, output_row,
                        /*M=*/1,
                        /*N=*/aligned_embed_dim,
                        /*K=*/aligned_intermediate_dim);
}

void ck_layer_forward_rmsnorm_swiglu_decode(const CKLayerForwardParams *p,
                                           int token_index,
                                           int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v ||
        !p->proj_tmp || !p->residual1 || !p->fc1_out || !p->swiglu_out || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }
    if (p->num_heads <= 0 || p->num_kv_heads <= 0 || p->aligned_head_dim <= 0) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;

    const float *input_row = p->input + (size_t)token_index * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + (size_t)token_index * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + (size_t)token_index * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + (size_t)token_index * (size_t)aligned_D;
    float *residual_row = p->residual1 + (size_t)token_index * (size_t)aligned_D;
    float *mlp_row = p->mlp_out + (size_t)token_index * (size_t)aligned_D;
    float *out_row = p->output + (size_t)token_index * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_index) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_index) : &ln2_rstd_tmp;

    // Scratch for a single token in head-major layout: [head, aligned_head_dim].
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    // LN1 / RMSNorm.
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // Project Q/K/V for the new token.
    ck_qkv_project_head_major_token(ln1_row,
                                    p->wq, p->bq,
                                    p->wk, p->bk,
                                    p->wv, p->bv,
                                    q_token, k_token, v_token,
                                    aligned_D,
                                    H,
                                    H_kv,
                                    ad);

    // RoPE for the new token at absolute position `p->rope_pos_offset`.
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    // Update KV cache (stores k/v for this token and clears padded lanes).
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    // Decode attention for this token using the KV cache.
    attention_forward_decode_head_major_gqa_flash(q_token,
                                                  p->k,
                                                  p->v,
                                                  attn_token,
                                                  H,
                                                  H_kv,
                                                  /*kv_tokens=*/token_index + 1,
                                                  cache_capacity,
                                                  hd,
                                                  ad);

    // Output projection (Wo) into token-major buffer (decode-specialized).
    ck_attention_project_head_major_decode_token(attn_token,
                                                 p->wo,
                                                 p->bo,
                                                 proj_row,
                                                 D,
                                                 aligned_D,
                                                 H,
                                                 ad);

    // Residual + LN2 / RMSNorm.
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // MLP block for this token.
    int up_dim = 2 * aligned_intermediate;
    float *fc1_row = p->fc1_out + (size_t)token_index * (size_t)up_dim;
    float *swiglu_row = p->swiglu_out + (size_t)token_index * (size_t)aligned_intermediate;

    ck_mlp_swiglu_forward(ln2_row,
                          p->w1,
                          p->b1,
                          p->w2,
                          p->b2,
                          fc1_row,
                          swiglu_row,
                          mlp_row,
                          /*tokens=*/1,
                          aligned_D,
                          aligned_intermediate);

    // Final residual.
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

void ck_layer_forward_rmsnorm_swiglu_decode_fused(const CKLayerForwardParams *p,
                                                 int token_index,
                                                 int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v || !p->swiglu_out ||
        !p->proj_tmp || !p->residual1 || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }
    if (p->num_heads <= 0 || p->num_kv_heads <= 0 || p->aligned_head_dim <= 0) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;

    const float *input_row = p->input + (size_t)token_index * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + (size_t)token_index * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + (size_t)token_index * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + (size_t)token_index * (size_t)aligned_D;
    float *residual_row = p->residual1 + (size_t)token_index * (size_t)aligned_D;
    float *swiglu_row = p->swiglu_out + (size_t)token_index * (size_t)aligned_intermediate;
    float *mlp_row = p->mlp_out + (size_t)token_index * (size_t)aligned_D;
    float *out_row = p->output + (size_t)token_index * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_index) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_index) : &ln2_rstd_tmp;

    // Scratch for a single token in head-major layout: [head, aligned_head_dim].
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    // LN1 / RMSNorm.
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // Project Q/K/V for the new token.
    ck_qkv_project_head_major_token(ln1_row,
                                    p->wq, p->bq,
                                    p->wk, p->bk,
                                    p->wv, p->bv,
                                    q_token, k_token, v_token,
                                    aligned_D,
                                    H,
                                    H_kv,
                                    ad);

    // RoPE for the new token at absolute position `p->rope_pos_offset`.
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    // Update KV cache (stores k/v for this token and clears padded lanes).
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    // Decode attention for this token using the KV cache.
    attention_forward_decode_head_major_gqa_flash(q_token,
                                                  p->k,
                                                  p->v,
                                                  attn_token,
                                                  H,
                                                  H_kv,
                                                  /*kv_tokens=*/token_index + 1,
                                                  cache_capacity,
                                                  hd,
                                                  ad);

    // Output projection (Wo) into token-major buffer (decode-specialized).
    ck_attention_project_head_major_decode_token(attn_token,
                                                 p->wo,
                                                 p->bo,
                                                 proj_row,
                                                 D,
                                                 aligned_D,
                                                 H,
                                                 ad);

    // Residual + LN2 / RMSNorm.
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    // MLP block for this token (fused gate+up projection).
    ck_mlp_swiglu_forward_fused_token(ln2_row,
                                      p->w1,
                                      p->b1,
                                      p->w2,
                                      p->b2,
                                      swiglu_row,
                                      mlp_row,
                                      aligned_D,
                                      aligned_intermediate);

    // Final residual.
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
}

static void ck_qkv_project_head_major_token_q4_k(const float *input_row,
                                                 const void *wq, const float *bq,
                                                 const void *wk, const float *bk,
                                                 const void *wv, const float *bv,
                                                 float *q_token,
                                                 float *k_token,
                                                 float *v_token,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int num_kv_heads,
                                                 int aligned_head_dim)
{
    if (!input_row || !wq || !wk || !wv || !q_token || !k_token || !v_token) {
        return;
    }

    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t head_w_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, head_w_elems);

    const uint8_t *wq_bytes = (const uint8_t *)wq;
    const uint8_t *wk_bytes = (const uint8_t *)wk;
    const uint8_t *wv_bytes = (const uint8_t *)wv;

    for (int h = 0; h < num_heads; ++h) {
        const void *wq_h = wq_bytes + (size_t)h * head_w_bytes;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k(input_row, wq_h, bq_h, q_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const void *wk_h = wk_bytes + (size_t)h * head_w_bytes;
        const void *wv_h = wv_bytes + (size_t)h * head_w_bytes;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_token + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_token + (size_t)h * (size_t)aligned_head_dim;
        gemm_nt_q4_k(input_row, wk_h, bk_h, k_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
        gemm_nt_q4_k(input_row, wv_h, bv_h, v_h,
                     /*M=*/1, /*N=*/aligned_head_dim, /*K=*/aligned_embed_dim);
    }
}

void ck_layer_forward_rmsnorm_swiglu_q4_k(const CKLayerForwardParamsQ4K *p)
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

    ck_qkv_project_head_major_q4_k(p->ln1_out,
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

    if (p->scores) {
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
    } else {
        attention_forward_causal_head_major_gqa_flash(p->q,
                                                     p->k,
                                                     p->v,
                                                     p->attn_out,
                                                     p->num_heads,
                                                     p->num_kv_heads,
                                                     p->tokens,
                                                     p->head_dim,
                                                     p->aligned_head_dim);
    }

    ck_attention_project_head_major_q4_k(p->attn_out,
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

    ck_mlp_swiglu_forward_q4_k(p->ln2_out,
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

void ck_layer_forward_rmsnorm_swiglu_decode_q4_k(const CKLayerForwardParamsQ4K *p,
                                                int token_index,
                                                int cache_capacity)
{
    if (!p) {
        return;
    }
    if (!p->input || !p->ln1_gamma || !p->ln2_gamma || !p->ln1_out || !p->ln2_out ||
        !p->wq || !p->wk || !p->wv || !p->wo || !p->w1 || !p->w2 ||
        !p->k || !p->v ||
        !p->proj_tmp || !p->proj_scratch || !p->residual1 || !p->fc1_out || !p->swiglu_out || !p->mlp_out || !p->output) {
        return;
    }
    if (token_index < 0 || cache_capacity <= 0 || token_index >= cache_capacity) {
        return;
    }

    const int D = p->embed_dim;
    const int aligned_D = p->aligned_embed_dim;
    const int H = p->num_heads;
    const int H_kv = p->num_kv_heads;
    const int hd = p->head_dim;
    const int ad = p->aligned_head_dim;
    const int aligned_intermediate = p->aligned_intermediate_dim;
    const int K_concat = H * ad;

    const float *input_row = p->input + (size_t)token_index * (size_t)aligned_D;
    float *ln1_row = p->ln1_out + (size_t)token_index * (size_t)aligned_D;
    float *ln2_row = p->ln2_out + (size_t)token_index * (size_t)aligned_D;
    float *proj_row = p->proj_tmp + (size_t)token_index * (size_t)aligned_D;
    float *residual_row = p->residual1 + (size_t)token_index * (size_t)aligned_D;
    float *mlp_row = p->mlp_out + (size_t)token_index * (size_t)aligned_D;
    float *out_row = p->output + (size_t)token_index * (size_t)aligned_D;

    float ln1_rstd_tmp = 0.0f;
    float ln2_rstd_tmp = 0.0f;
    float *ln1_rstd = p->ln1_rstd ? (p->ln1_rstd + token_index) : &ln1_rstd_tmp;
    float *ln2_rstd = p->ln2_rstd ? (p->ln2_rstd + token_index) : &ln2_rstd_tmp;

    /* Scratch for a single token in head-major layout: [head, aligned_head_dim]. */
    size_t q_elems = (size_t)H * (size_t)ad;
    size_t kv_elems = (size_t)H_kv * (size_t)ad;
    float q_token[q_elems];
    float k_token[kv_elems];
    float v_token[kv_elems];
    float attn_token[q_elems];

    /* LN1 / RMSNorm. */
    rmsnorm_forward(input_row,
                    p->ln1_gamma,
                    ln1_row,
                    ln1_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    /* Project Q/K/V for the new token (Q4_K weights). */
    ck_qkv_project_head_major_token_q4_k(ln1_row,
                                         p->wq, p->bq,
                                         p->wk, p->bk,
                                         p->wv, p->bv,
                                         q_token, k_token, v_token,
                                         aligned_D,
                                         H,
                                         H_kv,
                                         ad);

    /* RoPE for the new token at absolute position `p->rope_pos_offset`. */
    if (p->rope_cos && p->rope_sin) {
        rope_forward_qk(q_token,
                        k_token,
                        p->rope_cos,
                        p->rope_sin,
                        H,
                        H_kv,
                        /*num_tokens=*/1,
                        hd,
                        ad,
                        p->rope_pos_offset);
    }

    /* Update KV cache. */
    kv_cache_write_head_major(k_token,
                              v_token,
                              p->k,
                              p->v,
                              H_kv,
                              token_index,
                              cache_capacity,
                              hd,
                              ad);

    /* Decode attention for this token using the KV cache. */
    attention_forward_decode_head_major_gqa_flash(q_token,
                                                  p->k,
                                                  p->v,
                                                  attn_token,
                                                  H,
                                                  H_kv,
                                                  /*kv_tokens=*/token_index + 1,
                                                  cache_capacity,
                                                  hd,
                                                  ad);

    /* Quantized output projection: Wo is stored as a flattened Q4_K matrix. */
    gemm_nt_q4_k(attn_token,
                 p->wo,
                 p->bo,
                 proj_row,
                 /*M=*/1,
                 aligned_D,
                 /*K=*/K_concat);

    for (int j = D; j < aligned_D; ++j) {
        proj_row[j] = 0.0f;
    }

    /* Residual + LN2 / RMSNorm. */
    ck_residual_add_token_major(input_row,
                                proj_row,
                                residual_row,
                                /*tokens=*/1,
                                aligned_D);

    rmsnorm_forward(residual_row,
                    p->ln2_gamma,
                    ln2_row,
                    ln2_rstd,
                    /*tokens=*/1,
                    D,
                    aligned_D,
                    p->eps);

    /* MLP block for this token. */
    int up_dim = 2 * aligned_intermediate;
    float *fc1_row = p->fc1_out + (size_t)token_index * (size_t)up_dim;
    float *swiglu_row = p->swiglu_out + (size_t)token_index * (size_t)aligned_intermediate;

    ck_mlp_swiglu_forward_q4_k(ln2_row,
                               p->w1,
                               p->b1,
                               p->w2,
                               p->b2,
                               fc1_row,
                               swiglu_row,
                               mlp_row,
                               /*tokens=*/1,
                               aligned_D,
                               aligned_intermediate);

    /* Final residual. */
    ck_residual_add_token_major(residual_row,
                                mlp_row,
                                out_row,
                                /*tokens=*/1,
                                aligned_D);
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
