#ifndef CKERNEL_MODEL_H
#define CKERNEL_MODEL_H

#include "ckernel_ir.h"

#include <stddef.h>
#include <stdint.h>

/**
 * Simplified forward-only model layout.
 *
 * This is the first step toward the full TransformerModel / TrulyOptimalLayer
 * / GradientStorage design. It focuses on:
 *  - A single contiguous memory block for weights + activations
 *  - Per-layer weight offsets for the core decoder kernels
 *
 * Backward/optimizer offsets and KV cache layout will be layered on later.
 */

typedef struct {
    /* Per-layer weight offsets, measured in bytes from memory_base. */
    size_t ln1_weight_offset;
    size_t ln1_bias_offset;

    size_t qkv_weight_offset;
    size_t qkv_bias_offset;

    size_t attn_proj_weight_offset;
    size_t attn_proj_bias_offset;

    size_t fc1_weight_offset;
    size_t fc1_bias_offset;

    size_t fc2_weight_offset;
    size_t fc2_bias_offset;
} CKLayerLayout;

typedef struct {
    CKModelConfig cfg;   /* parsed from HF config / IR config */

    /* Unified memory block (weights + activations). */
    uint8_t *memory_base;
    size_t   total_bytes;
    size_t   total_floats; /* legacy: bytes / elem_bytes when elem_bytes==4 */
    size_t   elem_bytes;

    /* Global offsets (bytes) into memory_base. */
    size_t token_emb_offset;      /* [vocab_size × hidden_size] */
    size_t pos_emb_offset;        /* [context_window × hidden_size] */
    size_t embedded_input_offset; /* [context_window × hidden_size] */
    size_t layers_start_offset;   /* start of first layer block */

    size_t final_ln_weight_offset;
    size_t final_ln_bias_offset;
    size_t final_output_offset;   /* [context_window × hidden_size] */

    size_t lm_head_weight_offset; /* often tied to token_emb_offset */
    size_t logits_offset;         /* [context_window × vocab_size] */

    /* Per-layer layouts (length = cfg.num_layers). */
    CKLayerLayout *layers;
} TransformerModel;

/**
 * Compute a simple forward-only layout for TransformerModel based on:
 *  - CKModelConfig (dims, heads, vocab, context)
 *  - The IR graph structure (number of layers, op types)
 *
 * This function:
 *  - Fills token/pos embedding offsets
 *  - Assigns per-layer weight offsets for LN, QKV, attention proj, MLP
 *  - Sets final LN / LM head / logits offsets
 *  - Populates total_bytes with the required byte capacity
 *
 * Offsets are in bytes counted from memory_base. The exact shapes
 * and alignment strategy will evolve; this initial version focuses on
 * correctness and clarity over tight packing.
 */
/**
 * Layout the TransformerModel memory based on its cfg and (optionally) the IR.
 *
 * If `ir` is non-NULL, its config is copied into `m->cfg`. If `ir` is NULL,
 * the function trusts that `m->cfg` has already been populated.
 */
void layout_transformer_from_ir(TransformerModel *m, const CKIRGraph *ir);

/**
 * Load weights from a single flat binary file into model->memory_base.
 *
 * Expected layout in the file (float32, little-endian), in the same order
 * as layout_transformer_from_ir assigns weight offsets:
 *
 *   1) Token embeddings   [vocab_size × hidden_size]
 *   2) Pos embeddings     [context_window × hidden_size]
 *   3) For each layer L = 0..num_layers-1:
 *        - LN1 gamma      [hidden_size]
 *        - LN1 beta       [hidden_size]
 *        - QKV weight     [hidden_size × 3*hidden_size]
 *        - QKV bias       [3*hidden_size]
 *        - Attn proj W    [hidden_size × hidden_size]
 *        - Attn proj b    [hidden_size]
 *        - FC1 weight     [hidden_size × intermediate_size]
 *        - FC1 bias       [intermediate_size]
 *        - FC2 weight     [intermediate_size × hidden_size]
 *        - FC2 bias       [hidden_size]
 *   4) Final LN gamma     [hidden_size]
 *   5) Final LN beta      [hidden_size]
 *   6) LM head weight     [vocab_size × hidden_size]
 *
 * Activation buffers (embedded_input_offset, final_output_offset, logits_offset)
 * are NOT populated by this loader.
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_model_load_weights_flat(TransformerModel *m, const char *path);

#endif /* CKERNEL_MODEL_H */
