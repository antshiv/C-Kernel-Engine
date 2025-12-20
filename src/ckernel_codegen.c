#include "ckernel_codegen.h"
#include "ckernel_registry.h"

#include <errno.h>
#include <string.h>

static const char *op_name(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:        return "RMSNORM";
    case CK_OP_LINEAR_QKV:     return "LINEAR_QKV";
    case CK_OP_ATTENTION:      return "ATTENTION";
    case CK_OP_ADD:            return "ADD";
    case CK_OP_LINEAR:         return "LINEAR";
    case CK_OP_SPLIT:          return "SPLIT";
    case CK_OP_SWIGLU:         return "SWIGLU";
    case CK_OP_RMSNORM_BWD:    return "RMSNORM_BWD";
    case CK_OP_LINEAR_QKV_BWD: return "LINEAR_QKV_BWD";
    case CK_OP_ATTENTION_BWD:  return "ATTENTION_BWD";
    case CK_OP_ADD_BWD:        return "ADD_BWD";
    case CK_OP_LINEAR_BWD:     return "LINEAR_BWD";
    case CK_OP_SPLIT_BWD:      return "SPLIT_BWD";
    case CK_OP_SWIGLU_BWD:     return "SWIGLU_BWD";
    default:                   return "UNKNOWN";
    }
}

void ck_codegen_c_skeleton(const CKIRGraph *forward,
                           const CKIRGraph *backward,
                           FILE *out)
{
    if (!forward || !out) {
        return;
    }

    fprintf(out,
            "/* Auto-generated skeleton from CKIRGraph.\n"
            " * This file sketches the structure of the forward and backward\n"
            " * execution for a decoder-only transformer. It is NOT yet a\n"
            " * complete, runnable implementation. You can use it as a\n"
            " * starting point to wire buffers, kernel calls, and memory layout.\n"
            " */\n\n");

    fprintf(out, "#include \"ckernel_engine.h\"\n");
    fprintf(out, "#include \"ckernel_model.h\"\n");
    fprintf(out, "#include \"ckernel_alloc.h\"\n\n");

    /* Forward function */
    fprintf(out,
            "void run_decoder_forward(TransformerModel *model /*, inputs, etc. */)\n"
            "{\n"
            "    for (int layer = 0; layer < model->cfg.num_layers; ++layer) {\n"
            "        /* Forward pass for layer */\n");

    int nodes_per_layer = 0;
    if (forward->num_nodes > 0) {
        int l0 = forward->nodes[0].id.layer;
        for (int i = 0; i < forward->num_nodes; ++i) {
            if (forward->nodes[i].id.layer != l0) {
                break;
            }
            nodes_per_layer++;
        }
    }

    if (nodes_per_layer <= 0) {
        nodes_per_layer = forward->num_nodes;
    }

    fprintf(out, "        /* This layer has %d IR nodes */\n", nodes_per_layer);

    for (int i = 0; i < nodes_per_layer; ++i) {
        const CKIRNode *n = &forward->nodes[i];
        fprintf(out, "        // L%%d: %s\n", op_name(n->op));
        fprintf(out,
                "        //   outputs: [");
        for (int o = 0; o < n->n_outputs; ++o) {
            if (o > 0) fprintf(out, ", ");
            fprintf(out, "L%%d:N%d:%d", n->id.node, o);
        }
        fprintf(out, "]\n");
        fprintf(out, "        //   inputs : [");
        for (int j = 0; j < n->n_inputs; ++j) {
            const CKInputRef *inp = &n->inputs[j];
            if (j > 0) fprintf(out, ", ");
            if (inp->producer.node == 0xFFFFu) {
                fprintf(out, "IN");
            } else {
                fprintf(out, "L%%d:N%u:%u",
                        (unsigned)inp->producer.node,
                        (unsigned)inp->out_index);
            }
        }
        fprintf(out, "]\n");
        fprintf(out,
                "        //   TODO: bind buffers/weights and call %s kernel here\n\n",
                op_name(n->op));
    }

    fprintf(out,
            "    } /* end for layer */\n"
            "}\n\n");

    /* Backward skeleton */
    if (backward && backward->nodes && backward->num_nodes > 0) {
        fprintf(out,
            "void run_decoder_backward(TransformerModel *model /*, grads, etc. */)\n"
            "{\n"
            "    for (int layer = model->cfg.num_layers - 1; layer >= 0; --layer) {\n"
            "        /* Backward pass for layer */\n");

        int bwd_per_layer = 0;
        int l0 = backward->nodes[0].id.layer;
        for (int i = 0; i < backward->num_nodes; ++i) {
            if (backward->nodes[i].id.layer != l0) break;
            bwd_per_layer++;
        }
        if (bwd_per_layer <= 0) bwd_per_layer = backward->num_nodes;

        fprintf(out, "        /* This layer has %d backward IR nodes */\n", bwd_per_layer);

        for (int i = 0; i < bwd_per_layer; ++i) {
            const CKIRNode *n = &backward->nodes[i];
            fprintf(out, "        // L%%d: %s\n", op_name(n->op));
            fprintf(out,
                    "        //   TODO: wire gradient tensors and call %s kernel here\n\n",
                    op_name(n->op));
        }

        fprintf(out,
                "    } /* end for layer */\n"
                "}\n\n");
    }

    fprintf(out,
            "int main(int argc, char **argv)\n"
            "{\n"
            "    (void)argc; (void)argv;\n"
            "    TransformerModel model = {0};\n"
            "    model.cfg.num_layers        = %d;\n"
            "    model.cfg.hidden_size       = %d;\n"
            "    model.cfg.intermediate_size = %d;\n"
            "    model.cfg.num_heads         = %d;\n"
            "    model.cfg.num_kv_heads      = %d;\n"
            "    model.cfg.vocab_size        = %d;\n"
            "    model.cfg.context_window    = %d;\n"
            "    model.cfg.rms_norm_eps      = %.9g;\n"
            "    model.cfg.rope_theta        = %.9g;\n"
            "    layout_transformer_from_ir(&model, NULL); /* TODO: pass IR if needed */\n"
            "    size_t bytes = model.total_floats * sizeof(float);\n"
            "    model.memory_base = (float *)ck_huge_alloc(bytes);\n"
            "    if (!model.memory_base) {\n"
            "        fprintf(stderr, \"Failed to allocate %%zu bytes for model\\n\", bytes);\n"
            "        return 1;\n"
            "    }\n"
            "    // TODO: load weights into model.memory_base based on offsets\n"
            "    run_decoder_forward(&model);\n"
            "    // TODO: run_decoder_backward(&model) when training\n"
            "    ck_huge_free(model.memory_base, bytes);\n"
            "    return 0;\n"
            "}\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads,
            forward->config.vocab_size,
            forward->config.context_window,
            forward->config.rms_norm_eps,
            forward->config.rope_theta);
}

static int emit_source_filtered(FILE *out, const char *relpath)
{
    const char *prefixes[] = { "", "../", "../../", "../../../", NULL };
    char fullpath[4096];
    FILE *in = NULL;

    for (int i = 0; prefixes[i]; ++i) {
        if (snprintf(fullpath, sizeof(fullpath), "%s%s", prefixes[i], relpath) < 0) {
            continue;
        }
        in = fopen(fullpath, "rb");
        if (in) {
            break;
        }
    }

    if (!in) {
        fprintf(stderr, "ck_codegen_emit_runtime: failed to open %s\n", relpath);
        return -1;
    }

    char line[4096];
    while (fgets(line, sizeof(line), in)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (strncmp(p, "#include", 8) == 0) {
            continue;
        }
        if (strncmp(p, "#define _GNU_SOURCE", 19) == 0) {
            continue;
        }
        fputs(line, out);
    }

    fputs("\n", out);
    fclose(in);
    return 0;
}

static void emit_gemm_blocked_serial(FILE *out)
{
    fprintf(out,
            "static inline int ck_min(int a, int b) { return a < b ? a : b; }\n\n"
            "void gemm_blocked_serial(const float *A,\n"
            "                         const float *B,\n"
            "                         const float *bias,\n"
            "                         float *C,\n"
            "                         int M, int N, int K)\n"
            "{\n"
            "    const int block_size = 64;\n"
            "    for (int i = 0; i < M; i++) {\n"
            "        for (int j = 0; j < N; j++) {\n"
            "            C[i * N + j] = bias ? bias[j] : 0.0f;\n"
            "        }\n"
            "    }\n"
            "#if defined(__AVX512F__)\n"
            "    for (int ii = 0; ii < M; ii += block_size) {\n"
            "        for (int jj = 0; jj < N; jj += block_size) {\n"
            "            for (int kk = 0; kk < K; kk += block_size) {\n"
            "                int i_end = ck_min(ii + block_size, M);\n"
            "                int j_end = ck_min(jj + block_size, N);\n"
            "                int k_end = ck_min(kk + block_size, K);\n"
            "\n"
            "                for (int i = ii; i < i_end; i++) {\n"
            "                    for (int j = jj; j < j_end; j++) {\n"
            "                        __m512 sum_vec = _mm512_setzero_ps();\n"
            "                        int k;\n"
            "                        for (k = kk; k <= k_end - 16; k += 16) {\n"
            "                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);\n"
            "                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);\n"
            "                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);\n"
            "                        }\n"
            "                        float partial_sum = _mm512_reduce_add_ps(sum_vec);\n"
            "                        for (; k < k_end; k++) {\n"
            "                            partial_sum += A[i * K + k] * B[j * K + k];\n"
            "                        }\n"
            "                        C[i * N + j] += partial_sum;\n"
            "                    }\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "#else\n"
            "    for (int ii = 0; ii < M; ii += block_size) {\n"
            "        for (int jj = 0; jj < N; jj += block_size) {\n"
            "            for (int kk = 0; kk < K; kk += block_size) {\n"
            "                int i_end = ck_min(ii + block_size, M);\n"
            "                int j_end = ck_min(jj + block_size, N);\n"
            "                int k_end = ck_min(kk + block_size, K);\n"
            "\n"
            "                for (int i = ii; i < i_end; i++) {\n"
            "                    for (int j = jj; j < j_end; j++) {\n"
            "                        float partial_sum = 0.0f;\n"
            "                        for (int k = kk; k < k_end; k++) {\n"
            "                            partial_sum += A[i * K + k] * B[j * K + k];\n"
            "                        }\n"
            "                        C[i * N + j] += partial_sum;\n"
            "                    }\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "#endif\n"
            "}\n\n");
}

static int emit_runtime_preamble(FILE *out)
{
    fprintf(out,
            "/* Auto-generated runtime from CKIRGraph.\n"
            " * This file wires the existing C-Kernel-Engine kernels into a\n"
            " * decoder-only transformer forward pass.\n"
            " *\n"
            " * Compile (scalar): gcc -O2 generated_model.c -lm -o generated_model\n"
            " * Compile (AVX-512): gcc -O3 -mavx512f -mfma generated_model.c -lm -o generated_model\n"
            " */\n\n");

    fprintf(out,
            "#define _GNU_SOURCE\n"
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "#include <string.h>\n"
            "#include <math.h>\n"
            "#include <errno.h>\n"
            "#include <sys/mman.h>\n"
            "#include <sys/types.h>\n"
            "#include <unistd.h>\n"
            "#if defined(__AVX512F__)\n"
            "#include <immintrin.h>\n"
            "#endif\n\n");

    fprintf(out,
            "#define ALIGN_F 16\n"
            "static size_t align_up_size(size_t n, size_t align) {\n"
            "    if (align == 0) return n;\n"
            "    return (n + align - 1) & ~(align - 1);\n"
            "}\n\n"
            "static size_t bump(size_t *off, size_t count, size_t align) {\n"
            "    size_t start = align_up_size(*off, align);\n"
            "    *off = start + count;\n"
            "    return start;\n"
            "}\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    int tokens;\n"
            "    int embed_dim;\n"
            "    int aligned_embed_dim;\n"
            "    int num_heads;\n"
            "    int num_kv_heads;\n"
            "    int head_dim;\n"
            "    int aligned_head_dim;\n"
            "    int aligned_context_window;\n"
            "    int intermediate_dim;\n"
            "    int aligned_intermediate_dim;\n"
            "    float eps;\n"
            "    int rope_pos_offset;\n"
            "\n"
            "    const float *input;\n"
            "    const float *ln1_gamma;\n"
            "    const float *ln2_gamma;\n"
            "    const float *rope_cos;\n"
            "    const float *rope_sin;\n"
            "\n"
            "    const float *wq;\n"
            "    const float *bq;\n"
            "    const float *wk;\n"
            "    const float *bk;\n"
            "    const float *wv;\n"
            "    const float *bv;\n"
            "\n"
            "    const float *wo;\n"
            "    const float *bo;\n"
            "\n"
            "    const float *w1;\n"
            "    const float *b1;\n"
            "    const float *w2;\n"
            "    const float *b2;\n"
            "\n"
            "    float *ln1_out;\n"
            "    float *ln1_rstd;\n"
            "    float *q;\n"
            "    float *k;\n"
            "    float *v;\n"
            "    float *scores;\n"
            "    float *attn_out;\n"
            "    float *proj_tmp;\n"
            "    float *proj_scratch;\n"
            "    float *residual1;\n"
            "    float *ln2_out;\n"
            "    float *ln2_rstd;\n"
            "    float *fc1_out;\n"
            "    float *swiglu_out;\n"
            "    float *mlp_out;\n"
            "    float *output;\n"
            "} CKLayerForwardParams;\n\n");

    emit_gemm_blocked_serial(out);

    if (emit_source_filtered(out, "src/ckernel_alloc.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/sigmoid_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/swiglu_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/rmsnorm_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/softmax_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/rope_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/kernels/attention_kernels.c") != 0) {
        return -1;
    }
    if (emit_source_filtered(out, "src/ckernel_orchestration.c") != 0) {
        return -1;
    }

    return 0;
}

int ck_codegen_emit_runtime(const CKIRGraph *forward, const char *path)
{
    if (!forward || !path) {
        return -1;
    }
    if (ck_ir_validate_supported(forward) != 0) {
        return -1;
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_codegen_emit_runtime: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    if (emit_runtime_preamble(out) != 0) {
        fclose(out);
        return -1;
    }

    fprintf(out,
            "typedef enum {\n"
            "    TASK_LM = 0,\n"
            "    TASK_SEQ_CLS = 1\n"
            "} TaskType;\n\n"
            "typedef enum {\n"
            "    OPTIMIZER_SGD = 0,\n"
            "    OPTIMIZER_ADAM = 1\n"
            "} OptimizerType;\n\n"
            "typedef struct {\n"
            "    size_t total_gradient_floats;\n"
            "} GradientStorage;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    size_t ln1_gamma_offset;\n"
            "    size_t ln2_gamma_offset;\n"
            "    size_t wq_offset;\n"
            "    size_t bq_offset;\n"
            "    size_t wk_offset;\n"
            "    size_t bk_offset;\n"
            "    size_t wv_offset;\n"
            "    size_t bv_offset;\n"
            "    size_t wo_offset;\n"
            "    size_t bo_offset;\n"
            "    size_t w1_offset;\n"
            "    size_t b1_offset;\n"
            "    size_t w2_offset;\n"
            "    size_t b2_offset;\n"
            "\n"
            "    size_t ln1_out_offset;\n"
            "    size_t ln1_rstd_offset;\n"
            "    size_t q_offset;\n"
            "    size_t k_offset;\n"
            "    size_t v_offset;\n"
            "    size_t scores_offset;\n"
            "    size_t attn_out_offset;\n"
            "    size_t proj_tmp_offset;\n"
            "    size_t proj_scratch_offset;\n"
            "    size_t residual1_offset;\n"
            "    size_t ln2_out_offset;\n"
            "    size_t ln2_rstd_offset;\n"
            "    size_t fc1_out_offset;\n"
            "    size_t swiglu_out_offset;\n"
            "    size_t mlp_out_offset;\n"
            "    size_t output_offset;\n"
            "} LayerOffsets;\n\n");

    fprintf(out,
            "typedef LayerOffsets TrulyOptimalLayer;\n\n"
            "typedef struct {\n"
            "    char magic[8];\n"
            "    uint32_t version;\n"
            "    uint32_t model_type;\n"
            "\n"
            "    int num_layers;\n"
            "    int vocab_size;\n"
            "    int embed_dim;\n"
            "    int context_window;\n"
            "    int intermediate_size;\n"
            "\n"
            "    size_t aligned_embed_dim;\n"
            "    size_t aligned_head_dim;\n"
            "    size_t aligned_attn_context_window;\n"
            "\n"
            "    int num_cores;\n"
            "    int tokens_per_core;\n"
            "    int num_attention_heads;\n"
            "    int num_kv_heads;\n"
            "    int head_dim;\n"
            "    float rms_norm_eps;\n"
            "    float rope_theta;\n"
            "    size_t rope_cos_cache_offset;\n"
            "    size_t rope_sin_cache_offset;\n"
            "\n"
            "    float *memory_base;\n"
            "    size_t total_floats;\n"
            "    size_t layer_stride;\n"
            "\n"
            "    size_t token_emb_offset;\n"
            "    size_t pos_emb_offset;\n"
            "    size_t embedded_input_offset;\n"
            "    size_t layers_start_offset;\n"
            "\n"
            "    TrulyOptimalLayer *layers;\n"
            "\n"
            "    size_t final_ln_weight_offset;\n"
            "    size_t final_ln_bias_offset;\n"
            "    size_t final_ln_mean_offset;\n"
            "    size_t final_ln_rstd_offset;\n"
            "    size_t final_output_offset;\n"
            "\n"
            "    size_t lm_head_weight_offset;\n"
            "    size_t logits_offset;\n"
            "\n"
            "    GradientStorage gradients;\n"
            "    bool training_enabled;\n"
            "    float learning_rate;\n"
            "    int lr_warmup_steps;\n"
            "    float lr_warmup_init;\n"
            "    float grad_clip;\n"
            "    size_t training_cache_samples;\n"
            "    int active_tokens;\n"
            "    TaskType task_type;\n"
            "    OptimizerType optimizer;\n"
            "    uint64_t optimizer_step;\n"
            "    float adam_beta1;\n"
            "    float adam_beta2;\n"
            "    float adam_eps;\n"
            "    float weight_decay;\n"
            "    bool ema_enabled;\n"
            "    float ema_decay;\n"
            "    bool optimizer_state_initialized;\n"
            "\n"
            "    bool seq_cls_enabled;\n"
            "    int seq_cls_num_classes;\n"
            "    int seq_cls_pooling;\n"
            "    size_t seq_cls_weight_offset;\n"
            "    size_t seq_cls_bias_offset;\n"
            "\n"
            "    bool kv_cache_enabled;\n"
            "    int kv_cache_capacity;\n"
            "    int kv_cache_tokens;\n"
            "\n"
            "    long *training_data_buffer;\n"
            "    long num_training_tokens;\n"
            "\n"
            "    uint8_t checksum[32];\n"
            "    uint8_t reserved[32];\n"
            "} TransformerModel;\n\n");

    fprintf(out,
            "static int layout_model(TransformerModel *m)\n"
            "{\n"
            "    if (!m) return -1;\n"
            "    if (m->num_attention_heads <= 0 || m->embed_dim <= 0) return -1;\n"
            "    if (m->num_kv_heads <= 0) m->num_kv_heads = m->num_attention_heads;\n"
            "    if (m->num_attention_heads %% m->num_kv_heads != 0) return -1;\n"
            "    if (m->context_window <= 0) m->context_window = 1;\n"
            "    if (m->vocab_size <= 0) m->vocab_size = 1;\n"
            "    if (m->intermediate_size <= 0) return -1;\n"
            "    m->head_dim = m->embed_dim / m->num_attention_heads;\n"
            "    if (m->rms_norm_eps <= 0.0f) m->rms_norm_eps = 1e-5f;\n"
            "    if (m->rope_theta < 0.0f) m->rope_theta = 0.0f;\n"
            "    if (m->rope_theta > 0.0f && (m->head_dim %% 2 != 0)) return -1;\n"
            "    m->aligned_embed_dim = align_up_size((size_t)m->embed_dim, ALIGN_F);\n"
            "    m->aligned_head_dim = align_up_size((size_t)m->head_dim, ALIGN_F);\n"
            "    m->aligned_attn_context_window = align_up_size((size_t)m->context_window, ALIGN_F);\n"
            "    size_t aligned_intermediate_dim = align_up_size((size_t)m->intermediate_size, ALIGN_F);\n"
            "\n"
            "    if (m->num_cores <= 0) m->num_cores = 1;\n"
            "    m->tokens_per_core = (m->context_window + m->num_cores - 1) / m->num_cores;\n"
            "\n"
            "    m->layers = (TrulyOptimalLayer *)calloc((size_t)m->num_layers, sizeof(TrulyOptimalLayer));\n"
            "    if (!m->layers) return -1;\n"
            "\n"
            "    size_t off = 0;\n"
            "    m->token_emb_offset = bump(&off, (size_t)m->vocab_size * m->aligned_embed_dim, ALIGN_F);\n"
            "    m->pos_emb_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "    m->embedded_input_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "    if (m->rope_theta > 0.0f) {\n"
            "        size_t rope_half = (size_t)m->head_dim / 2;\n"
            "        size_t rope_count = (size_t)m->context_window * rope_half;\n"
            "        m->rope_cos_cache_offset = bump(&off, rope_count, ALIGN_F);\n"
            "        m->rope_sin_cache_offset = bump(&off, rope_count, ALIGN_F);\n"
            "    } else {\n"
            "        m->rope_cos_cache_offset = 0;\n"
            "        m->rope_sin_cache_offset = 0;\n"
            "    }\n"
            "    m->layers_start_offset = off;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        size_t head_w_stride = m->aligned_head_dim * m->aligned_embed_dim;\n"
            "        size_t q_w = (size_t)m->num_attention_heads * head_w_stride;\n"
            "        size_t kv_w = (size_t)m->num_kv_heads * head_w_stride;\n"
            "        size_t q_b = (size_t)m->num_attention_heads * m->aligned_head_dim;\n"
            "        size_t kv_b = (size_t)m->num_kv_heads * m->aligned_head_dim;\n"
            "        size_t wo_w = (size_t)m->num_attention_heads * m->aligned_embed_dim * m->aligned_head_dim;\n"
            "        size_t w1_w = (size_t)(2 * aligned_intermediate_dim) * m->aligned_embed_dim;\n"
            "        size_t w2_w = m->aligned_embed_dim * aligned_intermediate_dim;\n"
            "\n"
            "        L->ln1_gamma_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "        L->ln2_gamma_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "        L->wq_offset = bump(&off, q_w, ALIGN_F);\n"
            "        L->bq_offset = bump(&off, q_b, ALIGN_F);\n"
            "        L->wk_offset = bump(&off, kv_w, ALIGN_F);\n"
            "        L->bk_offset = bump(&off, kv_b, ALIGN_F);\n"
            "        L->wv_offset = bump(&off, kv_w, ALIGN_F);\n"
            "        L->bv_offset = bump(&off, kv_b, ALIGN_F);\n"
            "        L->wo_offset = bump(&off, wo_w, ALIGN_F);\n"
            "        L->bo_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "        L->w1_offset = bump(&off, w1_w, ALIGN_F);\n"
            "        L->b1_offset = bump(&off, (size_t)(2 * aligned_intermediate_dim), ALIGN_F);\n"
            "        L->w2_offset = bump(&off, w2_w, ALIGN_F);\n"
            "        L->b2_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "\n"
            "        L->ln1_out_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->ln1_rstd_offset = bump(&off, (size_t)m->context_window, ALIGN_F);\n"
            "        L->q_offset = bump(&off, (size_t)m->num_attention_heads * (size_t)m->context_window * m->aligned_head_dim, ALIGN_F);\n"
            "        L->k_offset = bump(&off, (size_t)m->num_kv_heads * (size_t)m->context_window * m->aligned_head_dim, ALIGN_F);\n"
            "        L->v_offset = bump(&off, (size_t)m->num_kv_heads * (size_t)m->context_window * m->aligned_head_dim, ALIGN_F);\n"
            "        L->scores_offset = bump(&off, (size_t)m->num_attention_heads * m->aligned_attn_context_window * m->aligned_attn_context_window, ALIGN_F);\n"
            "        L->attn_out_offset = bump(&off, (size_t)m->num_attention_heads * (size_t)m->context_window * m->aligned_head_dim, ALIGN_F);\n"
            "        L->proj_tmp_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->proj_scratch_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->residual1_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->ln2_out_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->ln2_rstd_offset = bump(&off, (size_t)m->context_window, ALIGN_F);\n"
            "        L->fc1_out_offset = bump(&off, (size_t)m->context_window * (size_t)(2 * aligned_intermediate_dim), ALIGN_F);\n"
            "        L->swiglu_out_offset = bump(&off, (size_t)m->context_window * aligned_intermediate_dim, ALIGN_F);\n"
            "        L->mlp_out_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "        L->output_offset = bump(&off, (size_t)m->context_window * m->aligned_embed_dim, ALIGN_F);\n"
            "    }\n"
            "\n"
            "    if (m->num_layers > 1) {\n"
            "        m->layer_stride = m->layers[1].ln1_gamma_offset - m->layers[0].ln1_gamma_offset;\n"
            "    } else {\n"
            "        m->layer_stride = 0;\n"
            "    }\n"
            "    m->final_output_offset = m->layers[m->num_layers - 1].output_offset;\n"
            "    m->final_ln_weight_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "    m->final_ln_bias_offset = bump(&off, m->aligned_embed_dim, ALIGN_F);\n"
            "    m->final_ln_mean_offset = bump(&off, (size_t)m->context_window, ALIGN_F);\n"
            "    m->final_ln_rstd_offset = bump(&off, (size_t)m->context_window, ALIGN_F);\n"
            "    m->lm_head_weight_offset = m->token_emb_offset;\n"
            "    m->logits_offset = bump(&off, (size_t)m->context_window * (size_t)m->vocab_size, ALIGN_F);\n"
            "    m->total_floats = align_up_size(off, ALIGN_F);\n"
            "    m->memory_base = (float *)ck_huge_alloc(m->total_floats * sizeof(float));\n"
            "    if (!m->memory_base) return -1;\n"
            "    if (m->rope_theta > 0.0f) {\n"
            "        rope_precompute_cache(m->memory_base + m->rope_cos_cache_offset,\n"
            "                             m->memory_base + m->rope_sin_cache_offset,\n"
            "                             m->context_window,\n"
            "                             m->head_dim,\n"
            "                             m->rope_theta);\n"
            "    }\n"
            "    return 0;\n"
            "}\n\n");

    fprintf(out,
            "static void lm_head_forward(const float *hidden,\n"
            "                            const float *weights,\n"
            "                            float *logits,\n"
            "                            int T, int V, int D, int aligned_D);\n\n");

    fprintf(out,
            "static void run_model_forward(TransformerModel *m)\n"
            "{\n"
            "    float *base = m->memory_base;\n"
            "    float *current = base + m->embedded_input_offset;\n"
            "    int aligned_intermediate_dim = (int)align_up_size((size_t)m->intermediate_size, ALIGN_F);\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        CKLayerForwardParams p = {0};\n"
            "        p.tokens = m->context_window;\n"
            "        p.embed_dim = m->embed_dim;\n"
            "        p.aligned_embed_dim = (int)m->aligned_embed_dim;\n"
            "        p.num_heads = m->num_attention_heads;\n"
            "        p.num_kv_heads = m->num_kv_heads;\n"
            "        p.head_dim = m->head_dim;\n"
            "        p.aligned_head_dim = (int)m->aligned_head_dim;\n"
            "        p.aligned_context_window = (int)m->aligned_attn_context_window;\n"
            "        p.intermediate_dim = m->intermediate_size;\n"
            "        p.aligned_intermediate_dim = aligned_intermediate_dim;\n"
            "        p.eps = m->rms_norm_eps;\n"
            "        p.rope_pos_offset = 0;\n"
            "        p.rope_cos = (m->rope_theta > 0.0f) ? (base + m->rope_cos_cache_offset) : NULL;\n"
            "        p.rope_sin = (m->rope_theta > 0.0f) ? (base + m->rope_sin_cache_offset) : NULL;\n"
            "        p.input = current;\n"
            "        p.ln1_gamma = base + L->ln1_gamma_offset;\n"
            "        p.ln2_gamma = base + L->ln2_gamma_offset;\n"
            "        p.wq = base + L->wq_offset;\n"
            "        p.bq = base + L->bq_offset;\n"
            "        p.wk = base + L->wk_offset;\n"
            "        p.bk = base + L->bk_offset;\n"
            "        p.wv = base + L->wv_offset;\n"
            "        p.bv = base + L->bv_offset;\n"
            "        p.wo = base + L->wo_offset;\n"
            "        p.bo = base + L->bo_offset;\n"
            "        p.w1 = base + L->w1_offset;\n"
            "        p.b1 = base + L->b1_offset;\n"
            "        p.w2 = base + L->w2_offset;\n"
            "        p.b2 = base + L->b2_offset;\n"
            "        p.ln1_out = base + L->ln1_out_offset;\n"
            "        p.ln1_rstd = base + L->ln1_rstd_offset;\n"
            "        p.q = base + L->q_offset;\n"
            "        p.k = base + L->k_offset;\n"
            "        p.v = base + L->v_offset;\n"
            "        p.scores = base + L->scores_offset;\n"
            "        p.attn_out = base + L->attn_out_offset;\n"
            "        p.proj_tmp = base + L->proj_tmp_offset;\n"
            "        p.proj_scratch = base + L->proj_scratch_offset;\n"
            "        p.residual1 = base + L->residual1_offset;\n"
            "        p.ln2_out = base + L->ln2_out_offset;\n"
            "        p.ln2_rstd = base + L->ln2_rstd_offset;\n"
            "        p.fc1_out = base + L->fc1_out_offset;\n"
            "        p.swiglu_out = base + L->swiglu_out_offset;\n"
            "        p.mlp_out = base + L->mlp_out_offset;\n"
            "        p.output = base + L->output_offset;\n"
            "        ck_layer_forward_rmsnorm_swiglu(&p);\n"
            "        current = p.output;\n"
            "    }\n"
            "    m->final_output_offset = (size_t)(current - base);\n"
            "    float *final_out = base + m->final_output_offset;\n"
            "    rmsnorm_forward(current,\n"
            "                    base + m->final_ln_weight_offset,\n"
            "                    final_out,\n"
            "                    base + m->final_ln_rstd_offset,\n"
            "                    m->context_window,\n"
            "                    m->embed_dim,\n"
            "                    (int)m->aligned_embed_dim,\n"
            "                    m->rms_norm_eps);\n"
            "    if (m->vocab_size > 0) {\n"
            "        lm_head_forward(final_out,\n"
            "                        base + m->lm_head_weight_offset,\n"
            "                        base + m->logits_offset,\n"
            "                        m->context_window,\n"
            "                        m->vocab_size,\n"
            "                        m->embed_dim,\n"
            "                        (int)m->aligned_embed_dim);\n"
            "    }\n"
            "}\n\n");

    fprintf(out,
            "static int parse_int_arg(const char *s, int *out)\n"
            "{\n"
            "    if (!s || !out) return 0;\n"
            "    char *end = NULL;\n"
            "    long v = strtol(s, &end, 10);\n"
            "    if (!end || *end != '\\0') return 0;\n"
            "    *out = (int)v;\n"
            "    return 1;\n"
            "}\n\n"
            "static void print_usage(const char *prog)\n"
            "{\n"
            "    printf(\"Usage: %%s [options]\\n\", prog);\n"
            "    printf(\"  --dump             Print layout summary (layer 0 only)\\n\");\n"
            "    printf(\"  --dump-all         Print layout summary for all layers\\n\");\n"
            "    printf(\"  --no-forward       Skip forward pass (layout + alloc only)\\n\");\n"
            "    printf(\"  --layers N         Override num_layers\\n\");\n"
            "    printf(\"  --embed N          Override embed_dim\\n\");\n"
            "    printf(\"  --intermediate N   Override intermediate_size\\n\");\n"
            "    printf(\"  --heads N          Override num_attention_heads\\n\");\n"
            "    printf(\"  --kv-heads N       Override num_kv_heads\\n\");\n"
            "    printf(\"  --vocab N          Override vocab_size\\n\");\n"
            "    printf(\"  --ctx N            Override context_window\\n\");\n"
            "    printf(\"  --cores N          Override num_cores\\n\");\n"
            "    printf(\"  --litmus           Run LM head + CE + backward litmus\\n\");\n"
            "    printf(\"  --hidden PATH      Load hidden activations [T x aligned_D] f32\\n\");\n"
            "    printf(\"  --weights PATH     Load LM head weights [V x aligned_D] f32 (litmus)\\n\");\n"
            "    printf(\"  --targets PATH     Load target tokens [T] int32\\n\");\n"
            "    printf(\"  --model-weights PATH  Load full model weights (bump format)\\n\");\n"
            "    printf(\"  --tokens PATH      Load token IDs [T] int32 and build embeddings\\n\");\n"
            "    printf(\"  --out-logits PATH  Write logits [T x V] f32\\n\");\n"
            "    printf(\"  --out-dlogits PATH Write d_logits [T x V] f32\\n\");\n"
            "    printf(\"  --out-dhidden PATH Write d_hidden [T x aligned_D] f32\\n\");\n"
            "    printf(\"  --out-dweights PATH Write d_weights [V x aligned_D] f32\\n\");\n"
            "    printf(\"  --out-loss PATH    Write loss (single f32)\\n\");\n"
            "    printf(\"  --help             Show this help\\n\");\n"
            "}\n\n"
            "static int read_floats(const char *path, float *dst, size_t count)\n"
            "{\n"
            "    if (!path || !dst) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t got = fread(dst, sizeof(float), count, f);\n"
            "    fclose(f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int read_ints(const char *path, int32_t *dst, size_t count)\n"
            "{\n"
            "    if (!path || !dst) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t got = fread(dst, sizeof(int32_t), count, f);\n"
            "    fclose(f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int read_floats_file(FILE *f, float *dst, size_t count)\n"
            "{\n"
            "    if (!f || !dst) return -1;\n"
            "    size_t got = fread(dst, sizeof(float), count, f);\n"
            "    return got == count ? 0 : -1;\n"
            "}\n\n"
            "static int skip_bump_header(FILE *f)\n"
            "{\n"
            "    if (!f) return -1;\n"
            "    char magic[8];\n"
            "    if (fread(magic, 1, 8, f) != 8) return -1;\n"
            "    if (memcmp(magic, \"BUMPWGT2\", 8) == 0) {\n"
            "        if (fseek(f, 128, SEEK_SET) != 0) return -1;\n"
            "        return 1;\n"
            "    }\n"
            "    if (fseek(f, 0, SEEK_SET) != 0) return -1;\n"
            "    return 0;\n"
            "}\n\n"
            "static int load_model_weights(const char *path, TransformerModel *m)\n"
            "{\n"
            "    if (!path || !m || !m->memory_base) return -1;\n"
            "    FILE *f = fopen(path, \"rb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    if (skip_bump_header(f) < 0) {\n"
            "        fclose(f);\n"
            "        return -1;\n"
            "    }\n"
            "    float *base = m->memory_base;\n"
            "    size_t aligned_intermediate = align_up_size((size_t)m->intermediate_size, ALIGN_F);\n"
            "\n"
            "    if (read_floats_file(f, base + m->token_emb_offset,\n"
            "                        (size_t)m->vocab_size * m->aligned_embed_dim) != 0) goto fail;\n"
            "    if (read_floats_file(f, base + m->pos_emb_offset,\n"
            "                        (size_t)m->context_window * m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    for (int layer = 0; layer < m->num_layers; ++layer) {\n"
            "        TrulyOptimalLayer *L = &m->layers[layer];\n"
            "        size_t head_w_stride = m->aligned_head_dim * m->aligned_embed_dim;\n"
            "        size_t q_w = (size_t)m->num_attention_heads * head_w_stride;\n"
            "        size_t kv_w = (size_t)m->num_kv_heads * head_w_stride;\n"
            "        size_t q_b = (size_t)m->num_attention_heads * m->aligned_head_dim;\n"
            "        size_t kv_b = (size_t)m->num_kv_heads * m->aligned_head_dim;\n"
            "        size_t wo_w = (size_t)m->num_attention_heads * m->aligned_embed_dim * m->aligned_head_dim;\n"
            "        size_t w1_w = (size_t)(2 * aligned_intermediate) * m->aligned_embed_dim;\n"
            "        size_t w2_w = m->aligned_embed_dim * aligned_intermediate;\n"
            "\n"
            "        if (read_floats_file(f, base + L->ln1_gamma_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->ln2_gamma_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->wq_offset, q_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->bq_offset, q_b) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->wk_offset, kv_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->bk_offset, kv_b) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->wv_offset, kv_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->bv_offset, kv_b) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->wo_offset, wo_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->bo_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->w1_offset, w1_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->b1_offset, (size_t)(2 * aligned_intermediate)) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->w2_offset, w2_w) != 0) goto fail;\n"
            "        if (read_floats_file(f, base + L->b2_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "    }\n"
            "\n"
            "    if (read_floats_file(f, base + m->final_ln_weight_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "    if (read_floats_file(f, base + m->final_ln_bias_offset, m->aligned_embed_dim) != 0) goto fail;\n"
            "\n"
            "    fclose(f);\n"
            "    return 0;\n"
            "fail:\n"
            "    fclose(f);\n"
            "    return -1;\n"
            "}\n\n"
            "static void embed_tokens(const TransformerModel *m, const int32_t *tokens, int token_count)\n"
            "{\n"
            "    if (!m || !m->memory_base || !tokens) return;\n"
            "    float *out = m->memory_base + m->embedded_input_offset;\n"
            "    const float *tok = m->memory_base + m->token_emb_offset;\n"
            "    const float *pos = m->memory_base + m->pos_emb_offset;\n"
            "    int T = m->context_window;\n"
            "    int D = m->embed_dim;\n"
            "    int aligned_D = (int)m->aligned_embed_dim;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        float *dst = out + (size_t)t * aligned_D;\n"
            "        if (t < token_count) {\n"
            "            int id = tokens[t];\n"
            "            if (id < 0 || id >= m->vocab_size) id = 0;\n"
            "            const float *src = tok + (size_t)id * aligned_D;\n"
            "            memcpy(dst, src, (size_t)D * sizeof(float));\n"
            "            if (aligned_D > D) {\n"
            "                memset(dst + D, 0, (size_t)(aligned_D - D) * sizeof(float));\n"
            "            }\n"
            "            if (m->rope_theta <= 0.0f) {\n"
            "                const float *p = pos + (size_t)t * aligned_D;\n"
            "                for (int d = 0; d < D; ++d) {\n"
            "                    dst[d] += p[d];\n"
            "                }\n"
            "            }\n"
            "        } else {\n"
            "            memset(dst, 0, (size_t)aligned_D * sizeof(float));\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "static int write_floats(const char *path, const float *src, size_t count)\n"
            "{\n"
            "    if (!path || !src) return -1;\n"
            "    FILE *f = fopen(path, \"wb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t wrote = fwrite(src, sizeof(float), count, f);\n"
            "    fclose(f);\n"
            "    return wrote == count ? 0 : -1;\n"
            "}\n\n"
            "static int write_float_scalar(const char *path, float v)\n"
            "{\n"
            "    if (!path) return -1;\n"
            "    FILE *f = fopen(path, \"wb\");\n"
            "    if (!f) {\n"
            "        perror(\"fopen\");\n"
            "        return -1;\n"
            "    }\n"
            "    size_t wrote = fwrite(&v, sizeof(float), 1, f);\n"
            "    fclose(f);\n"
            "    return wrote == 1 ? 0 : -1;\n"
            "}\n\n"
            "static void lm_head_forward(const float *hidden,\n"
            "                            const float *weights,\n"
            "                            float *logits,\n"
            "                            int T, int V, int D, int aligned_D)\n"
            "{\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *h = hidden + (size_t)t * aligned_D;\n"
            "        float *out = logits + (size_t)t * V;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            const float *w = weights + (size_t)v * aligned_D;\n"
            "            float sum = 0.0f;\n"
            "            for (int d = 0; d < D; ++d) {\n"
            "                sum += h[d] * w[d];\n"
            "            }\n"
            "            out[v] = sum;\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "static void softmax_cross_entropy(const float *logits,\n"
            "                                  const int32_t *targets,\n"
            "                                  int T, int V,\n"
            "                                  float *d_logits,\n"
            "                                  float *loss_out)\n"
            "{\n"
            "    double total = 0.0;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *row = logits + (size_t)t * V;\n"
            "        float *drow = d_logits + (size_t)t * V;\n"
            "        int target = targets[t];\n"
            "        float max_logit = row[0];\n"
            "        for (int v = 1; v < V; ++v) {\n"
            "            if (row[v] > max_logit) max_logit = row[v];\n"
            "        }\n"
            "        double sum_exp = 0.0;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] = expf(row[v] - max_logit);\n"
            "            sum_exp += drow[v];\n"
            "        }\n"
            "        float inv_sum = 1.0f / (float)sum_exp;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] *= inv_sum;\n"
            "        }\n"
            "        double logsum = (double)max_logit + log(sum_exp);\n"
            "        total += logsum - (double)row[target];\n"
            "        drow[target] -= 1.0f;\n"
            "        float scale = 1.0f / (float)T;\n"
            "        for (int v = 0; v < V; ++v) {\n"
            "            drow[v] *= scale;\n"
            "        }\n"
            "    }\n"
            "    if (loss_out) {\n"
            "        *loss_out = (float)(total / (double)T);\n"
            "    }\n"
            "}\n\n"
            "static void lm_head_backward(const float *hidden,\n"
            "                             const float *weights,\n"
            "                             const float *d_logits,\n"
            "                             float *d_hidden,\n"
            "                             float *d_weights,\n"
            "                             int T, int V, int D, int aligned_D)\n"
            "{\n"
            "    size_t dh_count = (size_t)T * aligned_D;\n"
            "    size_t dw_count = (size_t)V * aligned_D;\n"
            "    for (size_t i = 0; i < dh_count; ++i) d_hidden[i] = 0.0f;\n"
            "    for (size_t i = 0; i < dw_count; ++i) d_weights[i] = 0.0f;\n"
            "    for (int t = 0; t < T; ++t) {\n"
            "        const float *dlog = d_logits + (size_t)t * V;\n"
            "        for (int d = 0; d < D; ++d) {\n"
            "            float sum = 0.0f;\n"
            "            for (int v = 0; v < V; ++v) {\n"
            "                sum += dlog[v] * weights[(size_t)v * aligned_D + d];\n"
            "            }\n"
            "            d_hidden[(size_t)t * aligned_D + d] = sum;\n"
            "        }\n"
            "    }\n"
            "    for (int v = 0; v < V; ++v) {\n"
            "        float *dw = d_weights + (size_t)v * aligned_D;\n"
            "        for (int d = 0; d < D; ++d) {\n"
            "            float sum = 0.0f;\n"
            "            for (int t = 0; t < T; ++t) {\n"
            "                sum += d_logits[(size_t)t * V + v] * hidden[(size_t)t * aligned_D + d];\n"
            "            }\n"
            "            dw[d] = sum;\n"
            "        }\n"
            "    }\n"
            "}\n\n");

    fprintf(out,
            "static void dump_layer_offsets(const TransformerModel *m, int layer)\n"
            "{\n"
            "    const TrulyOptimalLayer *L = &m->layers[layer];\n"
            "    printf(\"Layer %%d offsets (floats):\\n\", layer);\n"
            "    printf(\"  ln1_gamma=%%zu ln2_gamma=%%zu wq=%%zu wk=%%zu wv=%%zu wo=%%zu w1=%%zu w2=%%zu\\n\",\n"
            "           L->ln1_gamma_offset, L->ln2_gamma_offset, L->wq_offset, L->wk_offset,\n"
            "           L->wv_offset, L->wo_offset, L->w1_offset, L->w2_offset);\n"
            "    printf(\"  ln1_out=%%zu q=%%zu k=%%zu v=%%zu scores=%%zu attn_out=%%zu\\n\",\n"
            "           L->ln1_out_offset, L->q_offset, L->k_offset, L->v_offset,\n"
            "           L->scores_offset, L->attn_out_offset);\n"
            "    printf(\"  proj_tmp=%%zu residual1=%%zu ln2_out=%%zu fc1_out=%%zu swiglu_out=%%zu mlp_out=%%zu output=%%zu\\n\",\n"
            "           L->proj_tmp_offset, L->residual1_offset, L->ln2_out_offset,\n"
            "           L->fc1_out_offset, L->swiglu_out_offset, L->mlp_out_offset, L->output_offset);\n"
            "}\n\n"
            "static void dump_layout(const TransformerModel *m, int dump_all)\n"
            "{\n"
            "    size_t bytes = m->total_floats * sizeof(float);\n"
            "    printf(\"Model config:\\n\");\n"
            "    printf(\"  layers=%%d embed=%%d intermediate=%%d heads=%%d kv_heads=%%d\\n\",\n"
            "           m->num_layers, m->embed_dim, m->intermediate_size, m->num_attention_heads, m->num_kv_heads);\n"
            "    printf(\"  head_dim=%%d vocab=%%d ctx=%%d cores=%%d\\n\",\n"
            "           m->head_dim, m->vocab_size, m->context_window, m->num_cores);\n"
            "    printf(\"  eps=%%.6g rope_theta=%%.6g\\n\", m->rms_norm_eps, m->rope_theta);\n"
            "    printf(\"Aligned dims (floats): embed=%%zu head=%%zu ctx=%%zu\\n\",\n"
            "           m->aligned_embed_dim, m->aligned_head_dim, m->aligned_attn_context_window);\n"
            "    printf(\"Memory: total_floats=%%zu bytes=%%zu\\n\", m->total_floats, bytes);\n"
            "    printf(\"Global offsets (floats): token=%%zu pos=%%zu embedded=%%zu layers_start=%%zu\\n\",\n"
            "           m->token_emb_offset, m->pos_emb_offset, m->embedded_input_offset, m->layers_start_offset);\n"
            "    printf(\"Final offsets (floats): final_ln_w=%%zu final_ln_b=%%zu final_ln_mean=%%zu final_ln_rstd=%%zu\\n\",\n"
            "           m->final_ln_weight_offset, m->final_ln_bias_offset,\n"
            "           m->final_ln_mean_offset, m->final_ln_rstd_offset);\n"
            "    printf(\"LM/logits offsets (floats): lm_head=%%zu logits=%%zu\\n\",\n"
            "           m->lm_head_weight_offset, m->logits_offset);\n"
            "    if (m->num_layers > 0) {\n"
            "        dump_layer_offsets(m, 0);\n"
            "        if (dump_all) {\n"
            "            for (int i = 1; i < m->num_layers; ++i) {\n"
            "                dump_layer_offsets(m, i);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n\n");

    fprintf(out,
            "int main(int argc, char **argv)\n"
            "{\n"
            "    int dump = 0;\n"
            "    int dump_all = 0;\n"
            "    int no_forward = 0;\n"
            "    int run_litmus = 0;\n"
            "    const char *litmus_hidden = NULL;\n"
            "    const char *litmus_weights = NULL;\n"
            "    const char *litmus_targets = NULL;\n"
            "    const char *model_weights = NULL;\n"
            "    const char *tokens_path = NULL;\n"
            "    const char *out_logits = NULL;\n"
            "    const char *out_dlogits = NULL;\n"
            "    const char *out_dhidden = NULL;\n"
            "    const char *out_dweights = NULL;\n"
            "    const char *out_loss = NULL;\n"
            "    TransformerModel m = {0};\n"
            "    memcpy(m.magic, \"BUMPWGT2\", 8);\n"
            "    m.version = 2;\n"
            "    m.model_type = 0;\n"
            "    m.num_layers = %d;\n"
            "    m.embed_dim = %d;\n"
            "    m.intermediate_size = %d;\n"
            "    m.num_attention_heads = %d;\n"
            "    m.num_kv_heads = %d;\n"
            "    m.vocab_size = %d;\n"
            "    m.context_window = %d;\n"
            "    m.rms_norm_eps = %.9g;\n"
            "    m.rope_theta = %.9g;\n"
            "    m.num_cores = 1;\n"
            "    m.task_type = TASK_LM;\n"
            "    m.optimizer = OPTIMIZER_SGD;\n"
            "    for (int i = 1; i < argc; ++i) {\n"
            "        if (strcmp(argv[i], \"--dump\") == 0) {\n"
            "            dump = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--dump-all\") == 0) {\n"
            "            dump = 1;\n"
            "            dump_all = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--no-forward\") == 0) {\n"
            "            no_forward = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--litmus\") == 0) {\n"
            "            run_litmus = 1;\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--help\") == 0) {\n"
            "            print_usage(argv[0]);\n"
            "            return 0;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--hidden\") == 0 && i + 1 < argc) {\n"
            "            litmus_hidden = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--weights\") == 0 && i + 1 < argc) {\n"
            "            litmus_weights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--targets\") == 0 && i + 1 < argc) {\n"
            "            litmus_targets = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--model-weights\") == 0 && i + 1 < argc) {\n"
            "            model_weights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--tokens\") == 0 && i + 1 < argc) {\n"
            "            tokens_path = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-logits\") == 0 && i + 1 < argc) {\n"
            "            out_logits = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dlogits\") == 0 && i + 1 < argc) {\n"
            "            out_dlogits = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dhidden\") == 0 && i + 1 < argc) {\n"
            "            out_dhidden = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-dweights\") == 0 && i + 1 < argc) {\n"
            "            out_dweights = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--out-loss\") == 0 && i + 1 < argc) {\n"
            "            out_loss = argv[++i];\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--layers\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_layers);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--embed\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.embed_dim);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--intermediate\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.intermediate_size);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--heads\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_attention_heads);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--kv-heads\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_kv_heads);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--vocab\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.vocab_size);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--ctx\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.context_window);\n"
            "            continue;\n"
            "        }\n"
            "        if (strcmp(argv[i], \"--cores\") == 0 && i + 1 < argc) {\n"
            "            parse_int_arg(argv[++i], &m.num_cores);\n"
            "            continue;\n"
            "        }\n"
            "        fprintf(stderr, \"Unknown or invalid arg: %%s\\n\", argv[i]);\n"
            "        print_usage(argv[0]);\n"
            "        return 1;\n"
            "    }\n"
            "    if (layout_model(&m) != 0) {\n"
            "        fprintf(stderr, \"layout_model failed\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    if (model_weights) {\n"
            "        if (load_model_weights(model_weights, &m) != 0) {\n"
            "            fprintf(stderr, \"failed to load model weights\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "    }\n"
            "    if (tokens_path) {\n"
            "        int T = m.context_window;\n"
            "        int32_t *tokens = (int32_t *)malloc((size_t)T * sizeof(int32_t));\n"
            "        if (!tokens) {\n"
            "            fprintf(stderr, \"failed to alloc tokens\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_ints(tokens_path, tokens, (size_t)T) != 0) {\n"
            "            fprintf(stderr, \"failed to read tokens\\n\");\n"
            "            free(tokens);\n"
            "            return 1;\n"
            "        }\n"
            "        embed_tokens(&m, tokens, T);\n"
            "        free(tokens);\n"
            "    }\n"
            "    if (dump) {\n"
            "        dump_layout(&m, dump_all);\n"
            "    }\n"
            "    if (run_litmus) {\n"
            "        if (!litmus_hidden || !litmus_weights || !litmus_targets) {\n"
            "            fprintf(stderr, \"litmus requires --hidden, --weights, and --targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        int T = m.context_window;\n"
            "        int V = m.vocab_size;\n"
            "        int D = m.embed_dim;\n"
            "        int aligned_D = (int)m.aligned_embed_dim;\n"
            "        float *hidden = m.memory_base + m.final_output_offset;\n"
            "        float *weights = m.memory_base + m.lm_head_weight_offset;\n"
            "        float *logits = m.memory_base + m.logits_offset;\n"
            "        if (read_floats(litmus_hidden, hidden, (size_t)T * aligned_D) != 0) {\n"
            "            fprintf(stderr, \"failed to read hidden\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_floats(litmus_weights, weights, (size_t)V * aligned_D) != 0) {\n"
            "            fprintf(stderr, \"failed to read weights\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        int32_t *targets = (int32_t *)malloc((size_t)T * sizeof(int32_t));\n"
            "        if (!targets) {\n"
            "            fprintf(stderr, \"failed to alloc targets\\n\");\n"
            "            return 1;\n"
            "        }\n"
            "        if (read_ints(litmus_targets, targets, (size_t)T) != 0) {\n"
            "            fprintf(stderr, \"failed to read targets\\n\");\n"
            "            free(targets);\n"
            "            return 1;\n"
            "        }\n"
            "        float *d_logits = (float *)calloc((size_t)T * V, sizeof(float));\n"
            "        float *d_hidden = (float *)calloc((size_t)T * aligned_D, sizeof(float));\n"
            "        float *d_weights = (float *)calloc((size_t)V * aligned_D, sizeof(float));\n"
            "        if (!d_logits || !d_hidden || !d_weights) {\n"
            "            fprintf(stderr, \"failed to alloc grads\\n\");\n"
            "            free(targets);\n"
            "            free(d_logits);\n"
            "            free(d_hidden);\n"
            "            free(d_weights);\n"
            "            return 1;\n"
            "        }\n"
            "        lm_head_forward(hidden, weights, logits, T, V, D, aligned_D);\n"
            "        float loss = 0.0f;\n"
            "        softmax_cross_entropy(logits, targets, T, V, d_logits, &loss);\n"
            "        lm_head_backward(hidden, weights, d_logits, d_hidden, d_weights, T, V, D, aligned_D);\n"
            "        if (out_logits) write_floats(out_logits, logits, (size_t)T * V);\n"
            "        if (out_dlogits) write_floats(out_dlogits, d_logits, (size_t)T * V);\n"
            "        if (out_dhidden) write_floats(out_dhidden, d_hidden, (size_t)T * aligned_D);\n"
            "        if (out_dweights) write_floats(out_dweights, d_weights, (size_t)V * aligned_D);\n"
            "        if (out_loss) write_float_scalar(out_loss, loss);\n"
            "        if (!out_loss) printf(\"loss=%%.6f\\n\", loss);\n"
            "        free(targets);\n"
            "        free(d_logits);\n"
            "        free(d_hidden);\n"
            "        free(d_weights);\n"
            "        ck_huge_free(m.memory_base, m.total_floats * sizeof(float));\n"
            "        free(m.layers);\n"
            "        return 0;\n"
            "    }\n"
            "    // TODO: load weights into m.memory_base using the offsets above.\n"
            "    // TODO: write token/pos embeddings into embedded_input_offset.\n"
            "    if (!no_forward) {\n"
            "        run_model_forward(&m);\n"
            "    }\n"
            "    if (out_logits) {\n"
            "        write_floats(out_logits, m.memory_base + m.logits_offset,\n"
            "                     (size_t)m.context_window * (size_t)m.vocab_size);\n"
            "    }\n"
            "    ck_huge_free(m.memory_base, m.total_floats * sizeof(float));\n"
            "    free(m.layers);\n"
            "    return 0;\n"
            "}\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads,
            forward->config.vocab_size,
            forward->config.context_window,
            forward->config.rms_norm_eps,
            forward->config.rope_theta);

    fclose(out);
    return 0;
}
