#include "ckernel_codegen.h"

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
            " * starting point to wire buffers and kernel calls.\n"
            " */\n\n");

    fprintf(out, "#include \"ckernel_engine.h\"\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    int num_layers;\n"
            "    int hidden_size;\n"
            "    int intermediate_size;\n"
            "    int num_heads;\n"
            "    int num_kv_heads;\n"
            "} ModelConfig;\n\n");

    fprintf(out,
            "typedef struct {\n"
            "    float *act;  /* activation buffer (bump-allocated) */\n"
            "    float *grad; /* gradient buffer (optional)        */\n"
            "} ModelBuffers;\n\n");

    fprintf(out,
            "/* TODO: implement a real memory planner that assigns offsets into\n"
            " * act/grad for each IR node's outputs and their gradients.\n"
            " */\n\n");

    /* Forward function */
    fprintf(out,
            "void run_decoder_forward(const ModelConfig *cfg,\n"
            "                         ModelBuffers *buf /*, weights, inputs, etc. */)\n"
            "{\n"
            "    for (int layer = 0; layer < cfg->num_layers; ++layer) {\n"
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
                "void run_decoder_backward(const ModelConfig *cfg,\n"
                "                          ModelBuffers *buf /*, weights, grads, etc. */)\n"
                "{\n"
                "    for (int layer = cfg->num_layers - 1; layer >= 0; --layer) {\n"
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
            "    ModelConfig cfg = {\n"
            "        .num_layers = %d,\n"
            "        .hidden_size = %d,\n"
            "        .intermediate_size = %d,\n"
            "        .num_heads = %d,\n"
            "        .num_kv_heads = %d\n"
            "    };\n"
            "    ModelBuffers buf = {0};\n"
            "    // TODO: allocate buf.act / buf.grad based on memory planning\n"
            "    // TODO: load weights and inputs\n"
            "    run_decoder_forward(&cfg, &buf);\n"
            "    // TODO: run_decoder_backward(&cfg, &buf) when training\n"
            "    return 0;\n"
            "}\n",
            forward->config.num_layers,
            forward->config.hidden_size,
            forward->config.intermediate_size,
            forward->config.num_heads,
            forward->config.num_kv_heads);
}

