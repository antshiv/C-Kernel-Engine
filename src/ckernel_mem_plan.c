#include "ckernel_mem_plan.h"

#include "ckernel_dtype.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t aligned_embed;
    size_t aligned_head;
    size_t aligned_intermediate;
    size_t aligned_context;
} CKIRV2AlignInfo;

static size_t align_up_bytes(size_t n, size_t align)
{
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}

static size_t align_up_elems(size_t elems, size_t elem_bytes, size_t align_bytes)
{
    size_t bytes = elems * elem_bytes;
    bytes = align_up_bytes(bytes, align_bytes);
    return bytes / elem_bytes;
}

static size_t resolve_dim(const CKModelConfig *cfg,
                          const CKIRV2AlignInfo *align,
                          CKDimKind kind,
                          int tokens_override)
{
    switch (kind) {
    case CK_DIM_TOKENS:
        if (tokens_override >= 0) {
            return (size_t)tokens_override;
        }
        return (size_t)cfg->context_window;
    case CK_DIM_EMBED:
        return (size_t)cfg->hidden_size;
    case CK_DIM_ALIGNED_EMBED:
        return align->aligned_embed;
    case CK_DIM_HEAD_DIM:
        return (size_t)(cfg->hidden_size / cfg->num_heads);
    case CK_DIM_ALIGNED_HEAD:
        return align->aligned_head;
    case CK_DIM_NUM_HEADS:
        return (size_t)cfg->num_heads;
    case CK_DIM_NUM_KV_HEADS:
        return (size_t)cfg->num_kv_heads;
    case CK_DIM_ALIGNED_CTX:
        return align->aligned_context;
    case CK_DIM_INTERMEDIATE:
        return (size_t)cfg->intermediate_size;
    case CK_DIM_ALIGNED_INTERMEDIATE:
        return align->aligned_intermediate;
    case CK_DIM_VOCAB:
        return (size_t)cfg->vocab_size;
    case CK_DIM_END:
    default:
        return 0;
    }
}

static size_t resolve_shape_elems(const CKModelConfig *cfg,
                                  const CKIRV2AlignInfo *align,
                                  const CKDimToken *shape,
                                  int tokens_override)
{
    size_t total = 1;
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        size_t dim = resolve_dim(cfg, align, shape[i].dim, tokens_override);
        size_t mult = (size_t)(shape[i].mult > 0 ? shape[i].mult : 1);
        size_t div = (size_t)(shape[i].div > 0 ? shape[i].div : 1);
        if (div == 0) div = 1;
        total = total * dim * mult / div;
    }
    return total;
}

static CKMemArenaKind arena_for_role(CKBufferRole role)
{
    switch (role) {
    case CK_ROLE_WEIGHT:
        return CK_MEM_ARENA_WEIGHTS;
    case CK_ROLE_GRAD:
        return CK_MEM_ARENA_GRADS;
    case CK_ROLE_SCRATCH:
    case CK_ROLE_ACTIVATION:
    case CK_ROLE_INPUT:
    case CK_ROLE_OUTPUT:
    default:
        return CK_MEM_ARENA_ACTIVATIONS;
    }
}

static int buffer_enabled(const CKIRV2Graph *graph,
                          const CKIRV2Buffer *buf,
                          int training_enabled)
{
    if (!graph || !buf) {
        return 0;
    }
    if (buf->condition && strcmp(buf->condition, "rope_theta") == 0) {
        if (graph->config.rope_theta <= 0.0f) {
            return 0;
        }
    }
    if (buf->condition && strcmp(buf->condition, "rope_disabled") == 0) {
        if (graph->config.rope_theta > 0.0f) {
            return 0;
        }
    }
    if (buf->condition && strcmp(buf->condition, "has_pos_emb") == 0) {
        if (graph->has_pos_emb == 0) {
            return 0;
        }
    }
    if (buf->condition && strcmp(buf->condition, "training_enabled") == 0) {
        if (!training_enabled) {
            return 0;
        }
    }
    if (buf->role == CK_ROLE_GRAD && !training_enabled) {
        return 0;
    }
    return 1;
}

static int find_buffer_by_name(const CKIRV2Graph *graph, const char *name)
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

static int build_plan(const CKIRV2Graph *graph,
                      CKMemPlan *plan,
                      size_t alignment_bytes,
                      int training_enabled,
                      int tokens_override)
{
    if (!graph || !plan) {
        return -1;
    }
    memset(plan, 0, sizeof(*plan));
    if (alignment_bytes == 0) {
        alignment_bytes = CK_MEM_PLAN_DEFAULT_ALIGN;
    }
    plan->alignment_bytes = alignment_bytes;

    plan->num_spans = graph->num_buffers;
    plan->spans = (CKMemSpan *)calloc((size_t)graph->num_buffers, sizeof(CKMemSpan));
    if (!plan->spans) {
        return -1;
    }

    size_t elem_bytes = sizeof(float);
    CKIRV2AlignInfo align = {0};
    align.aligned_embed = align_up_elems((size_t)graph->config.hidden_size, elem_bytes, alignment_bytes);
    align.aligned_head = align_up_elems((size_t)(graph->config.hidden_size / graph->config.num_heads),
                                        elem_bytes, alignment_bytes);
    align.aligned_intermediate = align_up_elems((size_t)graph->config.intermediate_size,
                                               elem_bytes, alignment_bytes);
    align.aligned_context = align_up_elems((size_t)graph->config.context_window, elem_bytes, alignment_bytes);

    size_t arena_offsets[CK_MEM_ARENA_COUNT] = {0};

    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        CKMemSpan *span = &plan->spans[i];
        span->buffer_id = i;
        span->arena = arena_for_role(buf->role);
        span->offset_bytes = 0;
        span->size_bytes = 0;

        if (!buffer_enabled(graph, buf, training_enabled)) {
            continue;
        }

        if (buf->alias_of) {
            int alias_id = find_buffer_by_name(graph, buf->alias_of);
            if (alias_id >= 0) {
                span->offset_bytes = plan->spans[alias_id].offset_bytes;
                span->size_bytes = plan->spans[alias_id].size_bytes;
                span->arena = plan->spans[alias_id].arena;
                continue;
            }
        }

        size_t n_elems = resolve_shape_elems(&graph->config, &align, buf->shape, tokens_override);
        size_t bytes = ck_dtype_row_bytes(buf->dtype, n_elems);
        size_t aligned = align_up_bytes(bytes, alignment_bytes);

        span->offset_bytes = arena_offsets[span->arena];
        span->size_bytes = bytes;
        arena_offsets[span->arena] += aligned;
    }

    for (int i = 0; i < CK_MEM_ARENA_COUNT; ++i) {
        plan->total_bytes[i] = arena_offsets[i];
    }

    return 0;
}

int ck_mem_plan_build_inference(const CKIRV2Graph *graph,
                                CKMemPlan *plan,
                                size_t alignment_bytes)
{
    return build_plan(graph, plan, alignment_bytes, 0, -1);
}

int ck_mem_plan_build_training(const CKIRV2Graph *graph,
                               CKMemPlan *plan,
                               size_t alignment_bytes)
{
    return build_plan(graph, plan, alignment_bytes, 1, -1);
}

int ck_mem_plan_build_inference_with_tokens(const CKIRV2Graph *graph,
                                            CKMemPlan *plan,
                                            size_t alignment_bytes,
                                            int tokens_override)
{
    return build_plan(graph, plan, alignment_bytes, 0, tokens_override);
}

int ck_mem_plan_build_training_with_tokens(const CKIRV2Graph *graph,
                                           CKMemPlan *plan,
                                           size_t alignment_bytes,
                                           int tokens_override)
{
    return build_plan(graph, plan, alignment_bytes, 1, tokens_override);
}

void ck_mem_plan_free(CKMemPlan *plan)
{
    if (!plan) {
        return;
    }
    free(plan->spans);
    memset(plan, 0, sizeof(*plan));
}
