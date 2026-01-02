/**
 * @file ckernel_memory_layout.h
 * @brief Single-Arena Memory Layout for CPU-Optimized Inference & Training
 *
 * PHILOSOPHY:
 * -----------
 * The CPU doesn't care if data is a weight or activation - it just needs
 * sequential memory access. This design:
 *
 * 1. ONE contiguous allocation (hugepage-backed)
 * 2. ALL offsets from single base pointer
 * 3. Weights and activations INTERLEAVED in execution order
 * 4. Fusion = same offset (skip write, read directly)
 * 5. No fusion = stride over (memory allocated but unused)
 *
 * BENEFITS:
 * - No malloc/free at runtime = no corruption, no double-free
 * - Prefetch layer N+1 while computing layer N (different DRAM bank)
 * - NUMA-aware: 1GB hugepages map to different memory channels
 * - Offset arithmetic only, no pointer chasing
 *
 * FUSION EXAMPLE:
 * ---------------
 * Without fusion:
 *   rmsnorm writes to ln1_output (offset 1000)
 *   qkv_project reads from ln1_output (offset 1000)
 *
 * With fusion (rmsnorm + qkv fused):
 *   fused kernel reads from input, writes directly to q/k/v
 *   ln1_output memory (offset 1000) is SKIPPED but still allocated
 *   CPU prefetch streams over it, no penalty
 */

#ifndef CKERNEL_MEMORY_LAYOUT_H
#define CKERNEL_MEMORY_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ALIGNMENT CONSTANTS
 * ============================================================================ */

#define CK_CACHE_LINE       64ULL           /* CPU cache line */
#define CK_HUGE_2MB         (2ULL << 20)    /* 2MB hugepage */
#define CK_HUGE_1GB         (1ULL << 30)    /* 1GB hugepage (NUMA optimal) */
#define CK_SIMD_ALIGN       64ULL           /* AVX-512 alignment */

/* ============================================================================
 * SECTION CONFIG (each section = encoder/decoder/vision/audio)
 *
 * Multi-modal models have multiple sections. Each section has its own config
 * but shares the SAME memory arena. Bridges connect sections.
 * ============================================================================ */

typedef struct {
    /* Model dimensions for this section */
    int embed_dim;              /* hidden_size */
    int num_heads;              /* attention heads */
    int num_kv_heads;           /* key-value heads (GQA) */
    int head_dim;               /* embed_dim / num_heads */
    int intermediate_dim;       /* MLP hidden size */
    int num_layers;             /* transformer layers in this section */
    int vocab_size;             /* only for sections with embeddings */
    int max_seq_len;            /* context window */

    /* Computed alignments */
    int aligned_embed;          /* embed_dim aligned to SIMD */
    int aligned_head;           /* head_dim aligned to SIMD */
    int aligned_intermediate;   /* intermediate aligned to SIMD */
} CKSectionConfig;

/* ============================================================================
 * LAYER OFFSETS - Weights and Activations INTERLEAVED in execution order
 *
 * Pattern: weight -> activation -> weight -> activation ...
 * This is the KEY insight: CPU streams through memory sequentially.
 * Whether fused or not, data is laid out in the order it's needed.
 * ============================================================================ */

typedef struct {
    /* === PRE-ATTENTION NORM === */
    size_t ln1_gamma;           /* [embed] weight */
    size_t ln1_beta;            /* [embed] weight (optional, some use RMSNorm) */
    size_t ln1_output;          /* [tokens, embed] activation */

    /* === QKV PROJECTION === */
    size_t wq;                  /* [embed, num_heads * head_dim] weight */
    size_t wk;                  /* [embed, num_kv_heads * head_dim] weight */
    size_t wv;                  /* [embed, num_kv_heads * head_dim] weight */
    size_t bq;                  /* [num_heads * head_dim] bias (optional) */
    size_t bk;                  /* [num_kv_heads * head_dim] bias (optional) */
    size_t bv;                  /* [num_kv_heads * head_dim] bias (optional) */
    size_t q;                   /* [tokens, num_heads, head_dim] activation */
    size_t k;                   /* [tokens, num_kv_heads, head_dim] activation */
    size_t v;                   /* [tokens, num_kv_heads, head_dim] activation */

    /* === ROPE (if applicable) === */
    size_t q_rope;              /* [tokens, num_heads, head_dim] activation */
    size_t k_rope;              /* [tokens, num_kv_heads, head_dim] activation */

    /* === ATTENTION === */
    size_t attn_scores;         /* [num_heads, tokens, tokens] activation */
    size_t attn_probs;          /* [num_heads, tokens, tokens] activation (after softmax) */
    size_t attn_output;         /* [tokens, num_heads, head_dim] activation */

    /* === OUTPUT PROJECTION === */
    size_t wo;                  /* [num_heads * head_dim, embed] weight */
    size_t bo;                  /* [embed] bias (optional) */
    size_t proj_output;         /* [tokens, embed] activation */

    /* === RESIDUAL 1 === */
    size_t residual1;           /* [tokens, embed] activation */

    /* === POST-ATTENTION NORM === */
    size_t ln2_gamma;           /* [embed] weight */
    size_t ln2_beta;            /* [embed] weight (optional) */
    size_t ln2_output;          /* [tokens, embed] activation */

    /* === MLP === */
    size_t mlp_gate_w;          /* [embed, intermediate] weight (gated MLP) */
    size_t mlp_up_w;            /* [embed, intermediate] weight */
    size_t mlp_down_w;          /* [intermediate, embed] weight */
    size_t mlp_gate_b;          /* [intermediate] bias (optional) */
    size_t mlp_up_b;            /* [intermediate] bias (optional) */
    size_t mlp_down_b;          /* [embed] bias (optional) */
    size_t mlp_gate_out;        /* [tokens, intermediate] activation */
    size_t mlp_up_out;          /* [tokens, intermediate] activation */
    size_t mlp_act_out;         /* [tokens, intermediate] activation (after SiLU/GELU) */
    size_t mlp_down_out;        /* [tokens, embed] activation */

    /* === RESIDUAL 2 (layer output) === */
    size_t residual2;           /* [tokens, embed] activation = layer output */

    /* === KV CACHE (decode mode) === */
    size_t k_cache;             /* [max_seq, num_kv_heads, head_dim] */
    size_t v_cache;             /* [max_seq, num_kv_heads, head_dim] */

} CKLayerOffsets;

/* ============================================================================
 * GRADIENT OFFSETS - Same pattern, follows forward offsets
 *
 * For training: gradients laid out AFTER forward activations.
 * Same interleaved pattern: d_weight, d_activation, d_weight, d_activation...
 * ============================================================================ */

typedef struct {
    /* Gradients mirror forward structure */
    size_t d_ln1_gamma;
    size_t d_ln1_beta;
    size_t d_ln1_output;

    size_t d_wq, d_wk, d_wv;
    size_t d_bq, d_bk, d_bv;
    size_t d_q, d_k, d_v;

    size_t d_attn_scores;
    size_t d_attn_output;

    size_t d_wo, d_bo;
    size_t d_proj_output;

    size_t d_ln2_gamma, d_ln2_beta;
    size_t d_ln2_output;

    size_t d_mlp_gate_w, d_mlp_up_w, d_mlp_down_w;
    size_t d_mlp_gate_out, d_mlp_up_out, d_mlp_down_out;

    /* Adam optimizer state (m and v) - also in same arena */
    size_t m_ln1_gamma, v_ln1_gamma;
    size_t m_wq, v_wq;
    size_t m_wk, v_wk;
    size_t m_wv, v_wv;
    size_t m_wo, v_wo;
    size_t m_ln2_gamma, v_ln2_gamma;
    size_t m_mlp_gate, v_mlp_gate;
    size_t m_mlp_up, v_mlp_up;
    size_t m_mlp_down, v_mlp_down;

} CKLayerGradOffsets;

/* ============================================================================
 * SECTION - A complete encoder/decoder/vision/audio module
 *
 * Each section has:
 * - Its own config (embed_dim, heads, etc.)
 * - Embedding layer (optional - only first section usually)
 * - N transformer layers
 * - Bridge to next section (optional)
 * ============================================================================ */

typedef struct {
    CKSectionConfig config;

    /* === EMBEDDINGS (optional) === */
    size_t token_embed;         /* [vocab, embed] weight */
    size_t pos_embed;           /* [max_seq, embed] weight (if absolute) */
    size_t embed_output;        /* [tokens, embed] activation */

    /* === TRANSFORMER LAYERS === */
    int num_layers;
    CKLayerOffsets *layers;     /* Array of per-layer offsets */
    CKLayerGradOffsets *grads;  /* Array of per-layer gradient offsets (training) */

    /* === FINAL NORM === */
    size_t final_ln_gamma;      /* [embed] weight */
    size_t final_ln_beta;       /* [embed] weight */
    size_t final_ln_output;     /* [tokens, embed] activation */

    /* === OUTPUT HEAD (optional) === */
    size_t lm_head;             /* [embed, vocab] weight (may tie with token_embed) */
    size_t logits;              /* [tokens, vocab] activation */

    /* === BRIDGE TO NEXT SECTION (optional) === */
    size_t bridge_proj_w;       /* [this_embed, next_embed] weight */
    size_t bridge_proj_b;       /* [next_embed] bias */
    size_t bridge_output;       /* [tokens, next_embed] activation */

    /* === SECTION BOUNDARIES (for NUMA planning) === */
    size_t section_start;       /* Byte offset where this section starts */
    size_t section_end;         /* Byte offset where this section ends */

} CKSection;

/* ============================================================================
 * MODEL - The complete multi-section model
 *
 * ONE allocation. ONE base pointer. ALL offsets relative to base.
 * ============================================================================ */

typedef struct {
    /* === MEMORY === */
    void *base;                 /* THE one and only allocation */
    size_t total_bytes;         /* Total allocated size */
    size_t weight_bytes;        /* Bytes used for weights */
    size_t activation_bytes;    /* Bytes used for activations */
    size_t grad_bytes;          /* Bytes used for gradients (0 if inference) */

    /* === SECTIONS === */
    int num_sections;
    CKSection *sections;

    /* === GLOBAL BUFFERS (shared across sections) === */
    size_t rope_cos;            /* [max_seq, head_dim] RoPE cosine */
    size_t rope_sin;            /* [max_seq, head_dim] RoPE sine */
    size_t causal_mask;         /* [max_seq, max_seq] attention mask */

    /* === FUSION FLAGS === */
    uint32_t fusion_flags;      /* Bitmask of enabled fusions */

    /* === RUNTIME STATE (not offsets, actual values) === */
    int current_seq_len;        /* Tokens processed so far */
    int batch_size;             /* Always 1 for now */

} CKModel;

/* ============================================================================
 * FUSION FLAGS
 *
 * When a fusion is enabled, the intermediate activation is SKIPPED.
 * Memory is still allocated (for consistency), but kernel reads/writes bypass it.
 * ============================================================================ */

typedef enum {
    CK_FUSE_NONE            = 0,
    CK_FUSE_EMBED_NORM      = 1 << 0,   /* embed + first layernorm */
    CK_FUSE_NORM_QKV        = 1 << 1,   /* layernorm + qkv projection */
    CK_FUSE_QKV_ROPE        = 1 << 2,   /* qkv projection + rope */
    CK_FUSE_ATTN_PROJ       = 1 << 3,   /* attention + output projection */
    CK_FUSE_NORM_MLP        = 1 << 4,   /* layernorm + mlp gate/up */
    CK_FUSE_MLP_GATE_UP     = 1 << 5,   /* gate and up projections */
    CK_FUSE_MLP_ACT_DOWN    = 1 << 6,   /* activation + down projection */
    CK_FUSE_ADD_NORM        = 1 << 7,   /* residual add + layernorm */
} CKFusionFlags;

/* ============================================================================
 * MEMORY PLANNING API
 *
 * Two-pass allocation:
 * 1. Dry run: compute all offsets, calculate total size
 * 2. Allocate: single hugepage-backed mmap
 * 3. Return model with all offsets filled in
 * ============================================================================ */

/**
 * Plan memory layout for a model.
 *
 * @param sections      Array of section configs
 * @param num_sections  Number of sections
 * @param mode          0=inference, 1=training (includes gradients)
 * @param fusion_flags  Which operations to fuse
 * @param out_model     Output: model with all offsets computed
 * @return              Total bytes needed (0 on error)
 */
size_t ck_memory_plan(const CKSectionConfig *sections,
                      int num_sections,
                      int mode,
                      uint32_t fusion_flags,
                      CKModel *out_model);

/**
 * Allocate the planned memory.
 *
 * @param model         Model with planned offsets
 * @param use_hugepages 0=regular malloc, 1=2MB hugepages, 2=1GB hugepages
 * @return              0 on success, -1 on failure
 */
int ck_memory_allocate(CKModel *model, int use_hugepages);

/**
 * Free the model memory.
 *
 * @param model         Model to free
 */
void ck_memory_free(CKModel *model);

/* ============================================================================
 * ACCESSOR MACROS
 *
 * All access is: (float*)(model->base + offset)
 * These macros make it cleaner.
 * ============================================================================ */

#define CK_PTR(model, offset) ((float*)((char*)(model)->base + (offset)))
#define CK_LAYER(model, section, layer) (&(model)->sections[section].layers[layer])
#define CK_GRAD(model, section, layer) (&(model)->sections[section].grads[layer])

/* Example usage:
 *
 *   CKModel model;
 *   ck_memory_plan(&config, 1, 0, CK_FUSE_NORM_QKV, &model);
 *   ck_memory_allocate(&model, 1);  // 2MB hugepages
 *
 *   // Access layer 5 Q weights:
 *   float *wq = CK_PTR(&model, CK_LAYER(&model, 0, 5)->wq);
 *
 *   // Access layer 5 Q activation:
 *   float *q = CK_PTR(&model, CK_LAYER(&model, 0, 5)->q);
 *
 *   // With fusion (CK_FUSE_NORM_QKV), the kernel skips ln1_output
 *   // and writes directly to q/k/v. The ln1_output memory exists
 *   // but is never touched - CPU prefetch streams over it.
 */

#ifdef __cplusplus
}
#endif

#endif /* CKERNEL_MEMORY_LAYOUT_H */
