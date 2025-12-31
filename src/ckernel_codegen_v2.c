#include "ckernel_codegen_v2.h"
#include "ckernel_mem_plan.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int emit_preamble(FILE *out)
{
    fprintf(out,
            "/* Auto-generated runtime from CKIRV2Graph.\n"
            " * This v2 emitter currently writes the schedule and buffer layout\n"
            " * stubs. Kernel wiring will be layered on as IR v2 grows.\n"
            " */\n\n");
    fprintf(out,
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <string.h>\n\n");
    return 0;
}

static void emit_buffer_layout(FILE *out, const CKIRV2Graph *graph, const CKMemPlan *plan)
{
    fprintf(out, "typedef struct {\n"
                 "    size_t offset_bytes;\n"
                 "    size_t size_bytes;\n"
                 "    int arena;\n"
                 "} CKV2BufferLayout;\n\n");

    fprintf(out, "static const CKV2BufferLayout ck_v2_buffers[] = {\n");
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKMemSpan *span = &plan->spans[i];
        fprintf(out,
                "    { %zu, %zu, %d }, /* %s */\n",
                span->offset_bytes,
                span->size_bytes,
                (int)span->arena,
                graph->buffers[i].name ? graph->buffers[i].name : "unnamed");
    }
    fprintf(out, "};\n\n");

    fprintf(out, "static const size_t ck_v2_total_bytes[] = {\n");
    for (int i = 0; i < CK_MEM_ARENA_COUNT; ++i) {
        fprintf(out, "    %zu%s\n",
                plan->total_bytes[i],
                (i + 1 == CK_MEM_ARENA_COUNT) ? "" : ",");
    }
    fprintf(out, "};\n\n");
}

static void emit_schedule(FILE *out, const CKIRV2Graph *graph)
{
    fprintf(out, "void ck_v2_run_forward(void) {\n");
    fprintf(out, "    /* Node schedule (no-op placeholders). */\n");
    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRV2Node *node = &graph->nodes[i];
        fprintf(out,
                "    /* node %d: layer=%d op=%s kernel=%s */\n",
                i,
                (int)node->layer,
                node->op ? node->op : "none",
                node->kernel ? node->kernel : "none");
        for (int b = 0; b < node->n_bindings; ++b) {
            const CKIRV2Binding *bind = &node->bindings[b];
            const char *buf_name = "unknown";
            if (bind->buffer >= 0 && bind->buffer < graph->num_buffers) {
                buf_name = graph->buffers[bind->buffer].name ? graph->buffers[bind->buffer].name : "unnamed";
            }
            fprintf(out,
                    "    /*   bind: %s -> %s */\n",
                    bind->arg ? bind->arg : "arg",
                    buf_name);
        }
    }
    fprintf(out, "    (void)ck_v2_buffers;\n");
    fprintf(out, "}\n\n");
}

int ck_codegen_v2_emit_runtime(const CKIRV2Graph *graph,
                               const char *path,
                               CKEmitMode mode)
{
    if (!graph || !path) {
        return -1;
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_codegen_v2_emit_runtime: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    CKMemPlan plan = {0};
    if (ck_mem_plan_build_inference(graph, &plan, CK_MEM_PLAN_DEFAULT_ALIGN) != 0) {
        fclose(out);
        return -1;
    }

    emit_preamble(out);
    emit_buffer_layout(out, graph, &plan);
    emit_schedule(out, graph);

    if (mode == CK_EMIT_STANDALONE) {
        fprintf(out,
                "int main(void) {\n"
                "    ck_v2_run_forward();\n"
                "    return 0;\n"
                "}\n");
    }

    ck_mem_plan_free(&plan);
    fclose(out);
    return 0;
}
