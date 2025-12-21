#include "ckernel_model.h"

#include <stdlib.h>
#include <string.h>

#define CACHELINE_BYTES 64

static size_t align_up_bytes(size_t n, size_t align)
{
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}

static size_t bump_bytes(size_t *off, size_t bytes, size_t align)
{
    size_t start = align_up_bytes(*off, align);
    *off = start + bytes;
    return start;
}

void layout_transformer_from_ir(TransformerModel *m, const CKIRGraph *ir)
{
    if (!m) {
        return;
    }

    if (ir) {
        /* If IR is provided, copy its config. Otherwise, trust m->cfg. */
        m->cfg = ir->config;
    }

    const int L   = m->cfg.num_layers;
    const int H   = m->cfg.hidden_size;
    const int Hff = m->cfg.intermediate_size;
    const int V   = m->cfg.vocab_size > 0 ? m->cfg.vocab_size : 1;
    const int T   = m->cfg.context_window > 0 ? m->cfg.context_window : 1;

    /* Allocate per-layer layout array. */
    if (m->layers) {
        /* caller responsible for freeing if re-layout is needed */
    } else if (L > 0) {
        m->layers = (CKLayerLayout *)calloc((size_t)L, sizeof(CKLayerLayout));
    }

    size_t elem_bytes = m->elem_bytes ? m->elem_bytes : sizeof(float);
    m->elem_bytes = elem_bytes;

    size_t offset = 0;

    /* Token embeddings: [V × H] */
    m->token_emb_offset = bump_bytes(&offset,
                                     (size_t)V * (size_t)H * elem_bytes,
                                     CACHELINE_BYTES);

    /* Positional embeddings: [T × H] */
    m->pos_emb_offset = bump_bytes(&offset,
                                   (size_t)T * (size_t)H * elem_bytes,
                                   CACHELINE_BYTES);

    /* Embedded input buffer: [T × H] */
    m->embedded_input_offset = bump_bytes(&offset,
                                          (size_t)T * (size_t)H * elem_bytes,
                                          CACHELINE_BYTES);

    m->layers_start_offset = offset;

    /* Per-layer weights. This is a simple, linear layout:
     *  - LN1 gamma/beta           [H]
     *  - QKV weight/bias          [H × 3H], [3H]
     *  - Attention proj weight/bias [H × H], [H]
     *  - FC1 weight/bias          [H × Hff], [Hff]
     *  - FC2 weight/bias          [Hff × H], [H]
     *
     * Activations are not yet explicitly laid out here; this pass focuses
     * on weights. A later planner can layer activations and gradients on top.
     */
    for (int layer = 0; layer < L; ++layer) {
        CKLayerLayout *Lyt = &m->layers[layer];

        /* LN1 weights/bias */
        Lyt->ln1_weight_offset = bump_bytes(&offset,
                                            (size_t)H * elem_bytes,
                                            CACHELINE_BYTES);

        Lyt->ln1_bias_offset = bump_bytes(&offset,
                                          (size_t)H * elem_bytes,
                                          CACHELINE_BYTES);

        /* QKV weight: [H × 3H] */
        Lyt->qkv_weight_offset = bump_bytes(&offset,
                                            (size_t)H * (size_t)(3 * H) * elem_bytes,
                                            CACHELINE_BYTES);

        /* QKV bias: [3H] */
        Lyt->qkv_bias_offset = bump_bytes(&offset,
                                          (size_t)(3 * H) * elem_bytes,
                                          CACHELINE_BYTES);

        /* Attention output projection: [H × H] + [H] */
        Lyt->attn_proj_weight_offset = bump_bytes(&offset,
                                                  (size_t)H * (size_t)H * elem_bytes,
                                                  CACHELINE_BYTES);

        Lyt->attn_proj_bias_offset = bump_bytes(&offset,
                                                (size_t)H * elem_bytes,
                                                CACHELINE_BYTES);

        /* FC1: [H × Hff] + [Hff] */
        Lyt->fc1_weight_offset = bump_bytes(&offset,
                                            (size_t)H * (size_t)Hff * elem_bytes,
                                            CACHELINE_BYTES);

        Lyt->fc1_bias_offset = bump_bytes(&offset,
                                          (size_t)Hff * elem_bytes,
                                          CACHELINE_BYTES);

        /* FC2: [Hff × H] + [H] */
        Lyt->fc2_weight_offset = bump_bytes(&offset,
                                            (size_t)Hff * (size_t)H * elem_bytes,
                                            CACHELINE_BYTES);

        Lyt->fc2_bias_offset = bump_bytes(&offset,
                                          (size_t)H * elem_bytes,
                                          CACHELINE_BYTES);
    }

    /* Final LayerNorm: gamma/beta [H], mean/rstd [T] if needed. */
    m->final_ln_weight_offset = bump_bytes(&offset,
                                           (size_t)H * elem_bytes,
                                           CACHELINE_BYTES);

    m->final_ln_bias_offset = bump_bytes(&offset,
                                         (size_t)H * elem_bytes,
                                         CACHELINE_BYTES);

    /* Final normalized output: [T × H] */
    m->final_output_offset = bump_bytes(&offset,
                                        (size_t)T * (size_t)H * elem_bytes,
                                        CACHELINE_BYTES);

    /* LM head weight: [V × H] (often tied to token_emb_offset in logic) */
    m->lm_head_weight_offset = bump_bytes(&offset,
                                          (size_t)V * (size_t)H * elem_bytes,
                                          CACHELINE_BYTES);

    /* Logits buffer: [T × V] */
    m->logits_offset = bump_bytes(&offset,
                                  (size_t)T * (size_t)V * elem_bytes,
                                  CACHELINE_BYTES);

    m->total_bytes = align_up_bytes(offset, CACHELINE_BYTES);
    m->total_floats = m->total_bytes / elem_bytes;
}
