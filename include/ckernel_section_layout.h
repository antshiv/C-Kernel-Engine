/**
 * @file ckernel_section_layout.h
 * @brief Section-Based Memory Layout: Header / Body / Footer Pattern
 *
 * =============================================================================
 * PHILOSOPHY: WHY NO MEMORY REUSE?
 * =============================================================================
 *
 * 1. CPU PREFETCH NEEDS PREDICTABLE PATTERNS
 *    - Prefetcher learns: "he's streaming forward through memory"
 *    - Reuse breaks this: same address, different data, different time
 *    - Result: prefetcher gives up, cache misses explode
 *
 * 2. TRAINING NEEDS ALL ACTIVATIONS
 *    - Forward: compute activations
 *    - Backward: need EVERY activation to compute gradients
 *    - Can't reuse what you still need!
 *
 * 3. MEMORY IS CHEAP, BANDWIDTH IS EXPENSIVE
 *    - DDR5: ~$3/GB (1TB = $3000, trivial for training cluster)
 *    - Bandwidth: 100-400 GB/s per socket (the real bottleneck)
 *    - Optimize for streaming, not for saving bytes
 *
 * 4. SINGLE ALLOCATION = ZERO RUNTIME MALLOC
 *    - No fragmentation, no free(), no double-free bugs
 *    - Hugepage-backed: 1GB pages for NUMA optimization
 *    - One base pointer + offsets = maximum simplicity
 *
 * =============================================================================
 * SECTION ARCHITECTURE
 * =============================================================================
 *
 * Multi-modal models have multiple SECTIONS. Each section is a complete
 * encoder/decoder/vision/audio module with its own dimensions.
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │                        SINGLE ALLOCATION                        │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ SECTION 0: Vision Encoder                                       │
 *   │   ├── HEADER: patch_embed, pos_embed                           │
 *   │   ├── BODY:   layer[0..N] (weights + activations interleaved)  │
 *   │   └── FOOTER: final_norm, bridge_to_text                       │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ SECTION 1: Text Decoder                                         │
 *   │   ├── HEADER: token_embed, pos_embed                           │
 *   │   ├── BODY:   layer[0..N] (weights + activations interleaved)  │
 *   │   └── FOOTER: final_norm, lm_head, logits                      │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ SECTION 2: Audio Encoder (optional)                             │
 *   │   ├── HEADER: mel_embed                                         │
 *   │   ├── BODY:   layer[0..N]                                       │
 *   │   └── FOOTER: bridge_to_text                                    │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ GRADIENTS (if training)                                         │
 *   │   └── Same layout: section[0].grads, section[1].grads, ...     │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ OPTIMIZER STATE (if training with Adam)                         │
 *   │   └── m[], v[] for each weight                                  │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * =============================================================================
 * EXECUTION ORDER LAYOUT
 * =============================================================================
 *
 * Within each layer, memory is laid out in the ORDER operations execute:
 *
 *   weight → activation → weight → activation → ...
 *
 * This is critical for CPU cache efficiency. The CPU streams forward,
 * prefetching the next cache line while processing the current one.
 *
 */

#ifndef CKERNEL_SECTION_LAYOUT_H
#define CKERNEL_SECTION_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ALIGNMENT CONSTANTS
 * ============================================================================ */

#define CK_ALIGN_CACHE      64ULL           /* CPU cache line (64 bytes) */
#define CK_ALIGN_SIMD       64ULL           /* AVX-512 register (64 bytes) */
#define CK_ALIGN_HUGE_2MB   (2ULL << 20)    /* 2MB hugepage */
#define CK_ALIGN_HUGE_1GB   (1ULL << 30)    /* 1GB hugepage (NUMA optimal) */

/* ============================================================================
 * SECTION CONFIG
 *
 * Each section (encoder/decoder/vision/audio) has its own dimensions.
 * A vision encoder might have embed_dim=1024, while text decoder has 4096.
 * ============================================================================ */

typedef struct {
    /* === DIMENSIONS === */
    int embed_dim;              /* hidden_size for this section */
    int num_heads;              /* attention heads */
    int num_kv_heads;           /* key-value heads (for GQA) */
    int head_dim;               /* embed_dim / num_heads */
    int intermediate_dim;       /* MLP hidden dimension */
    int num_layers;             /* transformer layers in this section */

    /* === VOCABULARY (optional, only for sections with embeddings) === */
    int vocab_size;             /* 0 if no token embedding in this section */

    /* === SEQUENCE === */
    int max_seq_len;            /* maximum sequence length */

    /* === ALIGNED DIMENSIONS (computed, for SIMD) === */
    int aligned_embed;          /* embed_dim padded to SIMD boundary */
    int aligned_head;           /* head_dim padded to SIMD boundary */
    int aligned_intermediate;   /* intermediate_dim padded to SIMD boundary */

    /* === FEATURES === */
    int has_bias;               /* 1 if layers have bias terms */
    int has_rope;               /* 1 if using rotary position embeddings */
    int has_pos_embed;          /* 1 if using learned position embeddings */
    int gated_mlp;              /* 1 if MLP is gated (SwiGLU/GeGLU) */
    int norm_type;              /* 0=LayerNorm, 1=RMSNorm */

} CKSectionConfig;

/* ============================================================================
 * HEADER OFFSETS
 *
 * The HEADER contains embeddings that happen BEFORE the transformer layers.
 * For text: token embeddings + position embeddings
 * For vision: patch embeddings + position embeddings
 * For audio: mel-spectrogram projection
 * ============================================================================ */

typedef struct {
    /* === TOKEN/PATCH EMBEDDING === */
    size_t embed_weight;        /* [vocab_size, embed_dim] or [patch_dim, embed_dim] */
    size_t embed_output;        /* [seq_len, embed_dim] activation */

    /* === POSITION EMBEDDING (if learned) === */
    size_t pos_embed_weight;    /* [max_seq_len, embed_dim] weight */
    size_t pos_embed_output;    /* [seq_len, embed_dim] activation (added to embed) */

    /* === COMBINED OUTPUT === */
    size_t header_output;       /* [seq_len, embed_dim] = embed + pos (input to body) */

    /* === BYTE BOUNDARIES === */
    size_t header_start;        /* first byte of header */
    size_t header_end;          /* last byte + 1 of header */

} CKHeaderOffsets;

/* ============================================================================
 * LAYER OFFSETS (within BODY)
 *
 * Each layer's memory is laid out in EXECUTION ORDER:
 *   pre_norm → qkv_proj → rope → attention → out_proj → residual →
 *   post_norm → mlp_gate → mlp_up → activation → mlp_down → residual
 *
 * Weights and activations are INTERLEAVED, not separated.
 * ============================================================================ */

typedef struct {
    /* === LAYER INPUT === */
    size_t input;               /* [seq, embed] activation (from previous layer or header) */

    /* ========== ATTENTION BLOCK ========== */

    /* Pre-Attention Normalization */
    size_t ln1_gamma;           /* [embed] weight */
    size_t ln1_beta;            /* [embed] weight (NULL if RMSNorm) */
    size_t ln1_output;          /* [seq, embed] activation */

    /* Q Projection */
    size_t wq;                  /* [embed, num_heads * head_dim] weight */
    size_t bq;                  /* [num_heads * head_dim] bias (optional) */
    size_t q;                   /* [seq, num_heads, head_dim] activation */

    /* K Projection */
    size_t wk;                  /* [embed, num_kv_heads * head_dim] weight */
    size_t bk;                  /* [num_kv_heads * head_dim] bias (optional) */
    size_t k;                   /* [seq, num_kv_heads, head_dim] activation */

    /* V Projection */
    size_t wv;                  /* [embed, num_kv_heads * head_dim] weight */
    size_t bv;                  /* [num_kv_heads * head_dim] bias (optional) */
    size_t v;                   /* [seq, num_kv_heads, head_dim] activation */

    /* RoPE (if enabled) */
    size_t q_rope;              /* [seq, num_heads, head_dim] activation */
    size_t k_rope;              /* [seq, num_kv_heads, head_dim] activation */

    /* Attention Scores */
    size_t attn_scores;         /* [num_heads, seq, seq] activation (Q @ K^T) */
    size_t attn_probs;          /* [num_heads, seq, seq] activation (softmax) */

    /* Attention Output */
    size_t attn_out;            /* [seq, num_heads, head_dim] activation (probs @ V) */

    /* Output Projection */
    size_t wo;                  /* [num_heads * head_dim, embed] weight */
    size_t bo;                  /* [embed] bias (optional) */
    size_t proj_out;            /* [seq, embed] activation */

    /* Residual Connection 1 */
    size_t residual1;           /* [seq, embed] activation (input + proj_out) */

    /* ========== MLP BLOCK ========== */

    /* Post-Attention Normalization */
    size_t ln2_gamma;           /* [embed] weight */
    size_t ln2_beta;            /* [embed] weight (NULL if RMSNorm) */
    size_t ln2_output;          /* [seq, embed] activation */

    /* MLP Gate Projection (for gated MLP like SwiGLU) */
    size_t mlp_gate_w;          /* [embed, intermediate] weight */
    size_t mlp_gate_b;          /* [intermediate] bias (optional) */
    size_t mlp_gate_out;        /* [seq, intermediate] activation */

    /* MLP Up Projection */
    size_t mlp_up_w;            /* [embed, intermediate] weight */
    size_t mlp_up_b;            /* [intermediate] bias (optional) */
    size_t mlp_up_out;          /* [seq, intermediate] activation */

    /* MLP Activation (SiLU/GELU) */
    size_t mlp_act_out;         /* [seq, intermediate] activation (gate * silu(up)) */

    /* MLP Down Projection */
    size_t mlp_down_w;          /* [intermediate, embed] weight */
    size_t mlp_down_b;          /* [embed] bias (optional) */
    size_t mlp_down_out;        /* [seq, embed] activation */

    /* Residual Connection 2 */
    size_t residual2;           /* [seq, embed] activation (residual1 + mlp_down_out) */

    /* ========== LAYER OUTPUT ========== */
    size_t output;              /* [seq, embed] = residual2 (input to next layer) */

    /* ========== KV CACHE (decode mode) ========== */
    size_t k_cache;             /* [max_seq, num_kv_heads, head_dim] persistent */
    size_t v_cache;             /* [max_seq, num_kv_heads, head_dim] persistent */

    /* ========== BYTE BOUNDARIES ========== */
    size_t layer_start;         /* first byte of this layer */
    size_t layer_end;           /* last byte + 1 of this layer */

} CKLayerOffsets;

/* ============================================================================
 * LAYER GRADIENT OFFSETS (for training)
 *
 * Gradients follow the same interleaved pattern.
 * Laid out AFTER all forward activations (or interleaved per-layer).
 * ============================================================================ */

typedef struct {
    /* === ATTENTION BLOCK GRADIENTS === */
    size_t d_ln1_gamma, d_ln1_beta;
    size_t d_ln1_output;

    size_t d_wq, d_bq, d_q;
    size_t d_wk, d_bk, d_k;
    size_t d_wv, d_bv, d_v;

    size_t d_attn_scores, d_attn_probs;
    size_t d_attn_out;

    size_t d_wo, d_bo;
    size_t d_proj_out;
    size_t d_residual1;

    /* === MLP BLOCK GRADIENTS === */
    size_t d_ln2_gamma, d_ln2_beta;
    size_t d_ln2_output;

    size_t d_mlp_gate_w, d_mlp_gate_b, d_mlp_gate_out;
    size_t d_mlp_up_w, d_mlp_up_b, d_mlp_up_out;
    size_t d_mlp_act_out;
    size_t d_mlp_down_w, d_mlp_down_b, d_mlp_down_out;

    size_t d_residual2;
    size_t d_output;

    /* === BYTE BOUNDARIES === */
    size_t grad_start;
    size_t grad_end;

} CKLayerGradOffsets;

/* ============================================================================
 * LAYER OPTIMIZER STATE (Adam m and v)
 * ============================================================================ */

typedef struct {
    /* First moment (m) for each weight */
    size_t m_ln1_gamma, m_ln1_beta;
    size_t m_wq, m_bq, m_wk, m_bk, m_wv, m_bv;
    size_t m_wo, m_bo;
    size_t m_ln2_gamma, m_ln2_beta;
    size_t m_mlp_gate_w, m_mlp_gate_b;
    size_t m_mlp_up_w, m_mlp_up_b;
    size_t m_mlp_down_w, m_mlp_down_b;

    /* Second moment (v) for each weight */
    size_t v_ln1_gamma, v_ln1_beta;
    size_t v_wq, v_bq, v_wk, v_bk, v_wv, v_bv;
    size_t v_wo, v_bo;
    size_t v_ln2_gamma, v_ln2_beta;
    size_t v_mlp_gate_w, v_mlp_gate_b;
    size_t v_mlp_up_w, v_mlp_up_b;
    size_t v_mlp_down_w, v_mlp_down_b;

    /* Byte boundaries */
    size_t opt_start;
    size_t opt_end;

} CKLayerOptimizerOffsets;

/* ============================================================================
 * FOOTER OFFSETS
 *
 * The FOOTER contains layers AFTER the transformer body:
 * - Final normalization
 * - Output projection (lm_head for text, classifier for vision)
 * - Bridge projection (to connect to next section)
 * ============================================================================ */

typedef struct {
    /* === FINAL NORMALIZATION === */
    size_t final_ln_gamma;      /* [embed] weight */
    size_t final_ln_beta;       /* [embed] weight (NULL if RMSNorm) */
    size_t final_ln_output;     /* [seq, embed] activation */

    /* === OUTPUT HEAD (optional) === */
    size_t lm_head_w;           /* [embed, vocab] weight (may tie with embed_weight) */
    size_t lm_head_b;           /* [vocab] bias (optional) */
    size_t logits;              /* [seq, vocab] activation */

    /* === BRIDGE TO NEXT SECTION (optional) === */
    size_t bridge_w;            /* [this_embed, next_embed] weight */
    size_t bridge_b;            /* [next_embed] bias (optional) */
    size_t bridge_output;       /* [seq, next_embed] activation */

    /* === BYTE BOUNDARIES === */
    size_t footer_start;
    size_t footer_end;

} CKFooterOffsets;

/* ============================================================================
 * FOOTER GRADIENTS (for training)
 * ============================================================================ */

typedef struct {
    size_t d_final_ln_gamma, d_final_ln_beta;
    size_t d_final_ln_output;

    size_t d_lm_head_w, d_lm_head_b;
    size_t d_logits;

    size_t d_bridge_w, d_bridge_b;
    size_t d_bridge_output;

    size_t grad_start;
    size_t grad_end;

} CKFooterGradOffsets;

/* ============================================================================
 * SECTION - Complete Header + Body + Footer
 *
 * Each section is a self-contained module (encoder, decoder, vision, audio).
 * ============================================================================ */

typedef struct {
    /* === CONFIGURATION === */
    CKSectionConfig config;
    const char *name;           /* "vision_encoder", "text_decoder", etc. */
    int section_id;             /* 0, 1, 2, ... */

    /* === HEADER === */
    CKHeaderOffsets header;

    /* === BODY (transformer layers) === */
    int num_layers;
    CKLayerOffsets *layers;             /* Array of per-layer offsets */
    CKLayerGradOffsets *layer_grads;    /* Array of per-layer gradient offsets */
    CKLayerOptimizerOffsets *layer_opt; /* Array of per-layer optimizer state */

    /* === FOOTER === */
    CKFooterOffsets footer;
    CKFooterGradOffsets footer_grads;

    /* === GLOBAL BUFFERS FOR THIS SECTION === */
    size_t rope_cos;            /* [max_seq, head_dim/2] RoPE cosines */
    size_t rope_sin;            /* [max_seq, head_dim/2] RoPE sines */
    size_t causal_mask;         /* [max_seq, max_seq] causal attention mask */

    /* === SECTION BYTE BOUNDARIES === */
    size_t section_start;       /* First byte of this section */
    size_t section_end;         /* Last byte + 1 of this section */
    size_t section_weight_bytes;
    size_t section_activation_bytes;
    size_t section_grad_bytes;
    size_t section_opt_bytes;

} CKSection;

/* ============================================================================
 * MODEL - The Complete Multi-Section Model
 *
 * ONE allocation. ONE base pointer. ALL offsets relative to base.
 * ============================================================================ */

typedef struct {
    /* === MEMORY === */
    void *base;                 /* THE single allocation */
    size_t total_bytes;         /* Total size of allocation */

    /* === BREAKDOWN === */
    size_t weight_bytes;        /* Total weights across all sections */
    size_t activation_bytes;    /* Total activations */
    size_t grad_bytes;          /* Total gradients (0 if inference) */
    size_t opt_bytes;           /* Total optimizer state (0 if inference) */

    /* === SECTIONS === */
    int num_sections;
    CKSection *sections;

    /* === GLOBAL SHARED BUFFERS === */
    size_t shared_scratch;      /* Scratch buffer for temp computations */
    size_t shared_scratch_bytes;

    /* === MODE FLAGS === */
    int training_enabled;       /* 0=inference, 1=training */
    int kv_cache_enabled;       /* 0=prefill, 1=decode with cache */
    uint32_t fusion_flags;      /* Bitmask of enabled kernel fusions */

    /* === RUNTIME STATE (not offsets) === */
    int current_seq_len;        /* Tokens processed in current sequence */
    int current_pos;            /* Position for decode mode */

    /* === NUMA INFO === */
    int num_numa_nodes;         /* Number of NUMA nodes detected */
    size_t hugepage_size;       /* Size of hugepages (2MB or 1GB) */

} CKModel;

/* ============================================================================
 * FUSION FLAGS
 *
 * When a fusion is enabled, intermediate activations are SKIPPED.
 * Memory is still allocated, but the fused kernel bypasses it.
 * ============================================================================ */

typedef enum {
    CK_FUSE_NONE            = 0,
    CK_FUSE_EMBED_NORM      = 1 << 0,   /* embedding + first layernorm */
    CK_FUSE_NORM_QKV        = 1 << 1,   /* layernorm + QKV projection */
    CK_FUSE_QKV_ROPE        = 1 << 2,   /* QKV + rotary position encoding */
    CK_FUSE_ATTN_PROJ       = 1 << 3,   /* attention output + projection */
    CK_FUSE_NORM_MLP        = 1 << 4,   /* layernorm + MLP input */
    CK_FUSE_MLP_GATE_UP     = 1 << 5,   /* gate and up projections together */
    CK_FUSE_MLP_ACT_DOWN    = 1 << 6,   /* activation + down projection */
    CK_FUSE_RESIDUAL_NORM   = 1 << 7,   /* residual add + layernorm */
} CKFusionFlags;

/* ============================================================================
 * MEMORY PLANNER API
 *
 * Two-phase allocation:
 * 1. Plan: dry run to compute all offsets and total size
 * 2. Allocate: single hugepage-backed mmap
 * ============================================================================ */

/**
 * Initialize section config with computed alignments.
 */
void ck_section_config_init(CKSectionConfig *config, size_t simd_align);

/**
 * Plan memory layout for a single section.
 * Returns bytes needed for this section.
 */
size_t ck_section_plan(CKSection *section,
                       const CKSectionConfig *config,
                       int training_enabled,
                       size_t base_offset);

/**
 * Plan memory layout for complete model.
 * Returns total bytes needed.
 */
size_t ck_model_plan(CKModel *model,
                     const CKSectionConfig *configs,
                     int num_sections,
                     int training_enabled,
                     uint32_t fusion_flags);

/**
 * Allocate the planned memory.
 * @param hugepage_mode: 0=normal, 1=2MB hugepages, 2=1GB hugepages
 * @return 0 on success, -1 on failure
 */
int ck_model_allocate(CKModel *model, int hugepage_mode);

/**
 * Free the model (single free, since single allocation).
 */
void ck_model_free(CKModel *model);

/* ============================================================================
 * ACCESSOR MACROS
 *
 * Clean syntax for accessing tensors from offsets.
 * ============================================================================ */

/* Get pointer from offset */
#define CK_PTR(model, offset) \
    ((float*)((char*)(model)->base + (offset)))

#define CK_PTR_BF16(model, offset) \
    ((uint16_t*)((char*)(model)->base + (offset)))

/* Get layer offsets for section s, layer l */
#define CK_LAYER(model, s, l) \
    (&(model)->sections[s].layers[l])

/* Get layer gradient offsets */
#define CK_LAYER_GRAD(model, s, l) \
    (&(model)->sections[s].layer_grads[l])

/* Get header offsets for section s */
#define CK_HEADER(model, s) \
    (&(model)->sections[s].header)

/* Get footer offsets for section s */
#define CK_FOOTER(model, s) \
    (&(model)->sections[s].footer)

/* ============================================================================
 * EXAMPLE USAGE
 * ============================================================================
 *
 * // Define a vision-language model (2 sections)
 * CKSectionConfig configs[2] = {
 *     // Section 0: Vision Encoder (ViT-L)
 *     { .embed_dim = 1024, .num_heads = 16, .num_kv_heads = 16,
 *       .head_dim = 64, .intermediate_dim = 4096, .num_layers = 24,
 *       .vocab_size = 0, .max_seq_len = 576,  // 24x24 patches
 *       .has_bias = 1, .has_rope = 0, .has_pos_embed = 1,
 *       .gated_mlp = 0, .norm_type = 0 },
 *
 *     // Section 1: Text Decoder (LLaMA-7B style)
 *     { .embed_dim = 4096, .num_heads = 32, .num_kv_heads = 32,
 *       .head_dim = 128, .intermediate_dim = 11008, .num_layers = 32,
 *       .vocab_size = 32000, .max_seq_len = 2048,
 *       .has_bias = 0, .has_rope = 1, .has_pos_embed = 0,
 *       .gated_mlp = 1, .norm_type = 1 }
 * };
 *
 * CKModel model = {0};
 * size_t total = ck_model_plan(&model, configs, 2, 1, CK_FUSE_NORM_QKV);
 * printf("Total memory: %.2f GB\n", total / 1e9);
 *
 * ck_model_allocate(&model, 2);  // 1GB hugepages
 *
 * // Access vision encoder layer 5 Q weights:
 * float *wq = CK_PTR(&model, CK_LAYER(&model, 0, 5)->wq);
 *
 * // Access text decoder layer 10 attention output:
 * float *attn = CK_PTR(&model, CK_LAYER(&model, 1, 10)->attn_out);
 *
 * // When done:
 * ck_model_free(&model);  // Single free!
 *
 * ============================================================================
 */

#ifdef __cplusplus
}
#endif

#endif /* CKERNEL_SECTION_LAYOUT_H */
