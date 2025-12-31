#include "ckernel_ir_v2.h"
#include "ckernel_kernel_specs.h"

#include <stdlib.h>
#include <string.h>

static char *ck_ir_v2_strdup(const char *s)
{
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *out = (char *)malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1);
    return out;
}

static void ck_ir_v2_copy_shape(CKDimToken *dst, const CKDimToken *src)
{
    memcpy(dst, src, sizeof(CKDimToken) * CK_IR_V2_MAX_DIMS);
}

static int ck_ir_v2_copy_buffer_spec(const CKBufferSpec *spec, CKIRV2Buffer *out)
{
    if (!spec || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->name = ck_ir_v2_strdup(spec->name);
    out->scope = spec->scope;
    out->role = spec->role;
    out->dtype = spec->dtype;
    out->optional = spec->optional;
    out->alias_of = ck_ir_v2_strdup(spec->alias_of);
    out->condition = ck_ir_v2_strdup(spec->condition);
    ck_ir_v2_copy_shape(out->shape, spec->shape);
    return 0;
}

static int ck_ir_v2_find_buffer_index(const CKIRV2Graph *graph, const char *name)
{
    if (!graph || !name) {
        return -1;
    }
    for (int i = 0; i < graph->num_buffers; ++i) {
        if (graph->buffers[i].name && strcmp(graph->buffers[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static const CKKernelSpec *ck_ir_v2_find_kernel_spec(const char *name)
{
    if (!name) {
        return NULL;
    }
    for (size_t i = 0; i < ck_kernel_spec_count; ++i) {
        if (strcmp(ck_kernel_specs[i].name, name) == 0) {
            return &ck_kernel_specs[i];
        }
    }
    return NULL;
}

static const char *ck_ir_v2_select_kernel(const CKKernelSpec *spec, CKDataType dtype, int backward)
{
    if (!spec) {
        return NULL;
    }
    if (dtype < 0 || dtype >= CK_DT_COUNT) {
        dtype = spec->default_dtype;
    }
    const char *name = backward ? spec->backward[dtype] : spec->forward[dtype];
    if (name && name[0]) {
        return name;
    }
    for (int i = 0; i < CK_DT_COUNT; ++i) {
        name = backward ? spec->backward[i] : spec->forward[i];
        if (name && name[0]) {
            return name;
        }
    }
    return spec->name;
}

int ck_ir_v2_build_decoder(const CKModelConfig *cfg, CKIRV2Graph *graph)
{
    if (!cfg || !graph) {
        return -1;
    }
    memset(graph, 0, sizeof(*graph));
    graph->config = *cfg;

    graph->num_buffers = (int)ck_decoder_buffer_count;
    graph->buffers = (CKIRV2Buffer *)calloc((size_t)graph->num_buffers, sizeof(CKIRV2Buffer));
    if (!graph->buffers) {
        return -1;
    }
    for (int i = 0; i < graph->num_buffers; ++i) {
        if (ck_ir_v2_copy_buffer_spec(&ck_decoder_buffers[i], &graph->buffers[i]) != 0) {
            ck_ir_v2_free(graph);
            return -1;
        }
    }

    int plan_count = (int)ck_decoder_forward_plan_v2_count;
    graph->num_nodes = cfg->num_layers * plan_count;
    graph->nodes = (CKIRV2Node *)calloc((size_t)graph->num_nodes, sizeof(CKIRV2Node));
    if (!graph->nodes) {
        ck_ir_v2_free(graph);
        return -1;
    }

    int idx = 0;
    for (int layer = 0; layer < cfg->num_layers; ++layer) {
        for (int p = 0; p < plan_count; ++p) {
            const CKPlanStepV2 *step = &ck_decoder_forward_plan_v2[p];
            const CKKernelSpec *spec = ck_ir_v2_find_kernel_spec(step->kernel);
            CKDataType dtype = spec ? spec->default_dtype : CK_DT_FP32;
            const char *impl = ck_ir_v2_select_kernel(spec, dtype, 0);
            CKIRV2Node *node = &graph->nodes[idx++];
            node->layer = (uint16_t)layer;
            node->op = ck_ir_v2_strdup(step->kernel);
            node->kernel = ck_ir_v2_strdup(impl ? impl : step->kernel);
            node->condition = ck_ir_v2_strdup(step->condition);
            node->flags = 0;
            node->n_bindings = 0;
            if (step->bindings && step->num_bindings > 0) {
                int limit = (int)step->num_bindings;
                if (limit > CK_IR_V2_MAX_BINDINGS) {
                    limit = CK_IR_V2_MAX_BINDINGS;
                }
                for (int b = 0; b < limit; ++b) {
                    const CKPlanBinding *binding = &step->bindings[b];
                    node->bindings[node->n_bindings].arg = ck_ir_v2_strdup(binding->arg);
                    node->bindings[node->n_bindings].buffer =
                        ck_ir_v2_find_buffer_index(graph, binding->buffer);
                    node->n_bindings++;
                }
            }
            node->n_inputs = 0;
            node->n_outputs = 0;
        }
    }
    return 0;
}

int ck_ir_v2_build_decoder_backward(const CKIRV2Graph *forward, CKIRV2Graph *backward)
{
    if (!forward || !backward) {
        return -1;
    }
    memset(backward, 0, sizeof(*backward));
    backward->config = forward->config;

    backward->num_buffers = forward->num_buffers;
    backward->buffers = (CKIRV2Buffer *)calloc((size_t)backward->num_buffers, sizeof(CKIRV2Buffer));
    if (!backward->buffers) {
        return -1;
    }
    for (int i = 0; i < backward->num_buffers; ++i) {
        CKBufferSpec spec = {0};
        const CKIRV2Buffer *src = &forward->buffers[i];
        spec.name = src->name;
        spec.scope = src->scope;
        spec.role = src->role;
        spec.dtype = src->dtype;
        spec.optional = src->optional;
        spec.alias_of = src->alias_of;
        spec.condition = src->condition;
        memcpy(spec.shape, src->shape, sizeof(spec.shape));
        if (ck_ir_v2_copy_buffer_spec(&spec, &backward->buffers[i]) != 0) {
            ck_ir_v2_free(backward);
            return -1;
        }
    }

    int plan_count = (int)ck_decoder_backward_plan_v2_count;
    backward->num_nodes = forward->config.num_layers * plan_count;
    backward->nodes = (CKIRV2Node *)calloc((size_t)backward->num_nodes, sizeof(CKIRV2Node));
    if (!backward->nodes) {
        ck_ir_v2_free(backward);
        return -1;
    }

    int idx = 0;
    for (int layer = 0; layer < forward->config.num_layers; ++layer) {
        for (int p = 0; p < plan_count; ++p) {
            const CKPlanStepV2 *step = &ck_decoder_backward_plan_v2[p];
            const CKKernelSpec *spec = ck_ir_v2_find_kernel_spec(step->kernel);
            CKDataType dtype = spec ? spec->default_dtype : CK_DT_FP32;
            const char *impl = ck_ir_v2_select_kernel(spec, dtype, 1);
            CKIRV2Node *node = &backward->nodes[idx++];
            node->layer = (uint16_t)layer;
            node->op = ck_ir_v2_strdup(step->kernel);
            node->kernel = ck_ir_v2_strdup(impl ? impl : step->kernel);
            node->condition = ck_ir_v2_strdup(step->condition);
            node->flags = 0;
            node->n_bindings = 0;
            if (step->bindings && step->num_bindings > 0) {
                int limit = (int)step->num_bindings;
                if (limit > CK_IR_V2_MAX_BINDINGS) {
                    limit = CK_IR_V2_MAX_BINDINGS;
                }
                for (int b = 0; b < limit; ++b) {
                    const CKPlanBinding *binding = &step->bindings[b];
                    node->bindings[node->n_bindings].arg = ck_ir_v2_strdup(binding->arg);
                    node->bindings[node->n_bindings].buffer =
                        ck_ir_v2_find_buffer_index(backward, binding->buffer);
                    node->n_bindings++;
                }
            }
            node->n_inputs = 0;
            node->n_outputs = 0;
        }
    }
    return 0;
}
