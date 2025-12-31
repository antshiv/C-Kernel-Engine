#include "ckernel_ir_v2.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ck_ir_v2_free_buffer(CKIRV2Buffer *buf)
{
    if (!buf) {
        return;
    }
    free(buf->name);
    free(buf->alias_of);
    free(buf->condition);
    memset(buf, 0, sizeof(*buf));
}

static void ck_ir_v2_free_node(CKIRV2Node *node)
{
    if (!node) {
        return;
    }
    for (int i = 0; i < node->n_bindings; ++i) {
        free(node->bindings[i].arg);
    }
    free(node->op);
    free(node->kernel);
    free(node->condition);
    memset(node, 0, sizeof(*node));
}

void ck_ir_v2_free(CKIRV2Graph *graph)
{
    if (!graph) {
        return;
    }
    if (graph->buffers) {
        for (int i = 0; i < graph->num_buffers; ++i) {
            ck_ir_v2_free_buffer(&graph->buffers[i]);
        }
        free(graph->buffers);
    }
    if (graph->nodes) {
        for (int i = 0; i < graph->num_nodes; ++i) {
            ck_ir_v2_free_node(&graph->nodes[i]);
        }
        free(graph->nodes);
    }
    memset(graph, 0, sizeof(*graph));
}

static const char *ck_ir_v2_scope_name(CKBufferScope scope)
{
    switch (scope) {
    case CK_SCOPE_LAYER:
        return "layer";
    case CK_SCOPE_GLOBAL:
        return "global";
    default:
        return "unknown";
    }
}

static const char *ck_ir_v2_role_name(CKBufferRole role)
{
    switch (role) {
    case CK_ROLE_INPUT:
        return "input";
    case CK_ROLE_OUTPUT:
        return "output";
    case CK_ROLE_ACTIVATION:
        return "activation";
    case CK_ROLE_WEIGHT:
        return "weight";
    case CK_ROLE_SCRATCH:
        return "scratch";
    case CK_ROLE_GRAD:
        return "grad";
    default:
        return "unknown";
    }
}

static const char *ck_ir_v2_dtype_name(CKDataType dtype)
{
    switch (dtype) {
    case CK_DT_FP32:
        return "fp32";
    case CK_DT_BF16:
        return "bf16";
    case CK_DT_FP16:
        return "fp16";
    case CK_DT_Q4_K:
        return "q4_k";
    case CK_DT_Q6_K:
        return "q6_k";
    case CK_DT_Q8_0:
        return "q8_0";
    default:
        return "unknown";
    }
}

static int ck_ir_v2_emit_shape(FILE *out, const CKDimToken *shape)
{
    fprintf(out, "[");
    int first = 1;
    for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
        if (shape[i].dim == CK_DIM_END) {
            break;
        }
        if (!first) {
            fprintf(out, ", ");
        }
        fprintf(out, "{\"dim\":%d,\"mult\":%d,\"div\":%d}",
                (int)shape[i].dim, shape[i].mult, shape[i].div);
        first = 0;
    }
    fprintf(out, "]");
    return 0;
}

int ck_ir_v2_serialize_json(const CKIRV2Graph *graph, const char *path)
{
    if (!graph || !path) {
        return -1;
    }
    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ck_ir_v2_serialize_json: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"version\": 2,\n");
    fprintf(out, "  \"config\": {\n");
    fprintf(out, "    \"num_layers\": %d,\n", graph->config.num_layers);
    fprintf(out, "    \"hidden_size\": %d,\n", graph->config.hidden_size);
    fprintf(out, "    \"intermediate_size\": %d,\n", graph->config.intermediate_size);
    fprintf(out, "    \"num_attention_heads\": %d,\n", graph->config.num_heads);
    fprintf(out, "    \"num_kv_heads\": %d,\n", graph->config.num_kv_heads);
    fprintf(out, "    \"vocab_size\": %d,\n", graph->config.vocab_size);
    fprintf(out, "    \"context_window\": %d,\n", graph->config.context_window);
    fprintf(out, "    \"rms_norm_eps\": %.9g,\n", graph->config.rms_norm_eps);
    fprintf(out, "    \"rope_theta\": %.9g\n", graph->config.rope_theta);
    fprintf(out, "  },\n");

    fprintf(out, "  \"buffers\": [\n");
    for (int i = 0; i < graph->num_buffers; ++i) {
        const CKIRV2Buffer *buf = &graph->buffers[i];
        fprintf(out, "    {\n");
        fprintf(out, "      \"name\": \"%s\",\n", buf->name ? buf->name : "");
        fprintf(out, "      \"scope\": \"%s\",\n", ck_ir_v2_scope_name(buf->scope));
        fprintf(out, "      \"role\": \"%s\",\n", ck_ir_v2_role_name(buf->role));
        fprintf(out, "      \"dtype\": \"%s\",\n", ck_ir_v2_dtype_name(buf->dtype));
        fprintf(out, "      \"optional\": %d,\n", buf->optional ? 1 : 0);
        fprintf(out, "      \"shape\": ");
        ck_ir_v2_emit_shape(out, buf->shape);
        fprintf(out, ",\n");
        if (buf->alias_of) {
            fprintf(out, "      \"alias_of\": \"%s\",\n", buf->alias_of);
        } else {
            fprintf(out, "      \"alias_of\": null,\n");
        }
        if (buf->condition) {
            fprintf(out, "      \"condition\": \"%s\"\n", buf->condition);
        } else {
            fprintf(out, "      \"condition\": null\n");
        }
        fprintf(out, "    }%s\n", (i + 1 == graph->num_buffers) ? "" : ",");
    }
    fprintf(out, "  ],\n");

    fprintf(out, "  \"nodes\": [\n");
    for (int i = 0; i < graph->num_nodes; ++i) {
        const CKIRV2Node *node = &graph->nodes[i];
        fprintf(out, "    {\n");
        fprintf(out, "      \"layer\": %d,\n", (int)node->layer);
        fprintf(out, "      \"op\": \"%s\",\n", node->op ? node->op : "");
        fprintf(out, "      \"kernel\": \"%s\",\n", node->kernel ? node->kernel : "");
        fprintf(out, "      \"flags\": %u,\n", (unsigned)node->flags);
        if (node->condition) {
            fprintf(out, "      \"condition\": \"%s\",\n", node->condition);
        } else {
            fprintf(out, "      \"condition\": null,\n");
        }
        fprintf(out, "      \"bindings\": [\n");
        for (int b = 0; b < node->n_bindings; ++b) {
            const CKIRV2Binding *bind = &node->bindings[b];
            const char *buf_name = "";
            if (bind->buffer >= 0 && bind->buffer < graph->num_buffers) {
                buf_name = graph->buffers[bind->buffer].name ? graph->buffers[bind->buffer].name : "";
            }
            fprintf(out,
                    "        {\"arg\": \"%s\", \"buffer\": \"%s\"}%s\n",
                    bind->arg ? bind->arg : "",
                    buf_name,
                    (b + 1 == node->n_bindings) ? "" : ",");
        }
        fprintf(out, "      ]\n");
        fprintf(out, "    }%s\n", (i + 1 == graph->num_nodes) ? "" : ",");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");

    fclose(out);
    return 0;
}

static int ck_ir_v2_parse_string(const char *start,
                                 const char *end,
                                 char **out_str)
{
    if (!start || !out_str || start >= end || *start != '"') {
        return -1;
    }
    const char *cur = start + 1;
    while (cur < end && *cur != '"') {
        if (*cur == '\\' && (cur + 1) < end) {
            cur += 2;
            continue;
        }
        cur++;
    }
    if (cur >= end || *cur != '"') {
        return -1;
    }
    size_t len = (size_t)(cur - (start + 1));
    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        return -1;
    }
    memcpy(buf, start + 1, len);
    buf[len] = '\0';
    *out_str = buf;
    return 0;
}

static const char *ck_ir_v2_skip_string(const char *cur, const char *end)
{
    if (!cur || cur >= end || *cur != '"') {
        return cur;
    }
    cur++;
    while (cur < end) {
        if (*cur == '\\' && (cur + 1) < end) {
            cur += 2;
            continue;
        }
        if (*cur == '"') {
            return cur + 1;
        }
        cur++;
    }
    return end;
}

static const char *ck_ir_v2_next_object(const char *cur,
                                        const char *end,
                                        const char **obj_start,
                                        const char **obj_end)
{
    int depth = 0;
    const char *start = NULL;
    for (const char *p = cur; p < end; ++p) {
        if (*p == '"') {
            p = ck_ir_v2_skip_string(p, end) - 1;
            continue;
        }
        if (*p == '{') {
            if (depth == 0) {
                start = p;
            }
            depth++;
            continue;
        }
        if (*p == '}') {
            if (depth > 0) {
                depth--;
                if (depth == 0 && start) {
                    *obj_start = start;
                    *obj_end = p;
                    return p + 1;
                }
            }
        }
    }
    return NULL;
}

static const char *ck_ir_v2_find_array_end(const char *open, const char *end)
{
    if (!open || open >= end || *open != '[') {
        return NULL;
    }
    int depth = 0;
    for (const char *p = open; p < end; ++p) {
        if (*p == '"') {
            p = ck_ir_v2_skip_string(p, end) - 1;
            continue;
        }
        if (*p == '[') {
            depth++;
            continue;
        }
        if (*p == ']') {
            if (depth > 0) {
                depth--;
                if (depth == 0) {
                    return p;
                }
            }
        }
    }
    return NULL;
}

static const char *ck_ir_v2_find_key(const char *json,
                                     const char *key,
                                     const char *end)
{
    size_t key_len = strlen(key);
    const char *cur = json;
    while (cur + key_len < end) {
        if (memcmp(cur, key, key_len) == 0) {
            return cur;
        }
        cur++;
    }
    return NULL;
}

static int ck_ir_v2_parse_int(const char *json,
                              const char *key,
                              const char *end,
                              int *out_val)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    int value = 0;
    if (sscanf(colon + 1, "%d", &value) != 1) {
        return -1;
    }
    *out_val = value;
    return 0;
}

static int ck_ir_v2_parse_float(const char *json,
                                const char *key,
                                const char *end,
                                float *out_val)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    float value = 0.0f;
    if (sscanf(colon + 1, "%f", &value) != 1) {
        return -1;
    }
    *out_val = value;
    return 0;
}

static int ck_ir_v2_parse_string_field(const char *json,
                                       const char *key,
                                       const char *end,
                                       char **out_str)
{
    const char *p = ck_ir_v2_find_key(json, key, end);
    if (!p) {
        return -1;
    }
    const char *colon = strchr(p, ':');
    if (!colon || colon >= end) {
        return -1;
    }
    const char *cur = colon + 1;
    while (cur < end && (*cur == ' ' || *cur == '\t' || *cur == '\n' || *cur == '\r')) {
        cur++;
    }
    if (cur >= end) {
        return -1;
    }
    if (strncmp(cur, "null", 4) == 0) {
        *out_str = NULL;
        return 0;
    }
    if (*cur != '"') {
        return -1;
    }
    return ck_ir_v2_parse_string(cur, end, out_str);
}

static CKBufferScope ck_ir_v2_parse_scope(const char *s)
{
    if (!s) return CK_SCOPE_GLOBAL;
    if (strcmp(s, "layer") == 0) return CK_SCOPE_LAYER;
    if (strcmp(s, "global") == 0) return CK_SCOPE_GLOBAL;
    return CK_SCOPE_GLOBAL;
}

static CKBufferRole ck_ir_v2_parse_role(const char *s)
{
    if (!s) return CK_ROLE_ACTIVATION;
    if (strcmp(s, "input") == 0) return CK_ROLE_INPUT;
    if (strcmp(s, "output") == 0) return CK_ROLE_OUTPUT;
    if (strcmp(s, "activation") == 0) return CK_ROLE_ACTIVATION;
    if (strcmp(s, "weight") == 0) return CK_ROLE_WEIGHT;
    if (strcmp(s, "scratch") == 0) return CK_ROLE_SCRATCH;
    if (strcmp(s, "grad") == 0) return CK_ROLE_GRAD;
    return CK_ROLE_ACTIVATION;
}

static CKDataType ck_ir_v2_parse_dtype(const char *s)
{
    if (!s) return CK_DT_FP32;
    if (strcmp(s, "fp32") == 0) return CK_DT_FP32;
    if (strcmp(s, "bf16") == 0) return CK_DT_BF16;
    if (strcmp(s, "fp16") == 0) return CK_DT_FP16;
    if (strcmp(s, "q4_k") == 0) return CK_DT_Q4_K;
    if (strcmp(s, "q6_k") == 0) return CK_DT_Q6_K;
    if (strcmp(s, "q8_0") == 0) return CK_DT_Q8_0;
    return CK_DT_FP32;
}

static const CKBufferSpec *ck_ir_v2_find_buffer_spec(const char *name)
{
    if (!name) {
        return NULL;
    }
    for (size_t i = 0; i < ck_decoder_buffer_count; ++i) {
        if (strcmp(ck_decoder_buffers[i].name, name) == 0) {
            return &ck_decoder_buffers[i];
        }
    }
    return NULL;
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

static int ck_ir_v2_parse_buffers(const char *json,
                                  const char *end,
                                  CKIRV2Graph *graph)
{
    const char *start = ck_ir_v2_find_key(json, "\"buffers\"", end);
    if (!start) {
        return -1;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, end);
    if (!open || !close) {
        return -1;
    }

    int count = 0;
    const char *cur = open;
    const char *obj_start = NULL;
    const char *obj_end = NULL;
    while ((cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {
        count++;
    }
    if (count <= 0) {
        return -1;
    }

    CKIRV2Buffer *buffers = (CKIRV2Buffer *)calloc((size_t)count, sizeof(CKIRV2Buffer));
    if (!buffers) {
        return -1;
    }
    int idx = 0;
    cur = open;
    while (idx < count &&
           (cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {

        CKIRV2Buffer buf = {0};
        char *name = NULL;
        char *scope = NULL;
        char *role = NULL;
        char *dtype = NULL;
        char *alias = NULL;
        char *cond = NULL;
        int optional = 0;

        ck_ir_v2_parse_string_field(obj_start, "\"name\"", obj_end, &name);
        ck_ir_v2_parse_string_field(obj_start, "\"scope\"", obj_end, &scope);
        ck_ir_v2_parse_string_field(obj_start, "\"role\"", obj_end, &role);
        ck_ir_v2_parse_string_field(obj_start, "\"dtype\"", obj_end, &dtype);
        ck_ir_v2_parse_string_field(obj_start, "\"alias_of\"", obj_end, &alias);
        ck_ir_v2_parse_string_field(obj_start, "\"condition\"", obj_end, &cond);
        ck_ir_v2_parse_int(obj_start, "\"optional\"", obj_end, &optional);

        buf.name = name;
        buf.scope = ck_ir_v2_parse_scope(scope);
        buf.role = ck_ir_v2_parse_role(role);
        buf.dtype = ck_ir_v2_parse_dtype(dtype);
        buf.optional = optional;
        buf.alias_of = alias;
        buf.condition = cond;

        const CKBufferSpec *spec = ck_ir_v2_find_buffer_spec(name);
        if (spec) {
            memcpy(buf.shape, spec->shape, sizeof(buf.shape));
        } else {
            for (int i = 0; i < CK_IR_V2_MAX_DIMS; ++i) {
                buf.shape[i].dim = CK_DIM_END;
                buf.shape[i].mult = 0;
                buf.shape[i].div = 0;
            }
        }

        free(scope);
        free(role);
        free(dtype);

        buffers[idx++] = buf;
    }

    graph->buffers = buffers;
    graph->num_buffers = idx;
    return 0;
}

static int ck_ir_v2_parse_bindings(const char *obj_start,
                                   const char *obj_end,
                                   CKIRV2Graph *graph,
                                   CKIRV2Node *node)
{
    const char *start = ck_ir_v2_find_key(obj_start, "\"bindings\"", obj_end);
    if (!start) {
        return 0;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, obj_end);
    if (!open || !close || close > obj_end) {
        return -1;
    }

    const char *cur = open;
    const char *bstart = NULL;
    const char *bend = NULL;
    while (node->n_bindings < CK_IR_V2_MAX_BINDINGS &&
           (cur = ck_ir_v2_next_object(cur, close, &bstart, &bend)) != NULL) {

        char *arg = NULL;
        char *buf = NULL;
        ck_ir_v2_parse_string_field(bstart, "\"arg\"", bend, &arg);
        ck_ir_v2_parse_string_field(bstart, "\"buffer\"", bend, &buf);

        if (arg) {
            int buf_idx = ck_ir_v2_find_buffer_index(graph, buf);
            node->bindings[node->n_bindings].arg = arg;
            node->bindings[node->n_bindings].buffer = buf_idx;
            node->n_bindings++;
        } else {
            free(arg);
        }
        free(buf);
    }

    return 0;
}

static int ck_ir_v2_parse_nodes(const char *json,
                                const char *end,
                                CKIRV2Graph *graph)
{
    const char *start = ck_ir_v2_find_key(json, "\"nodes\"", end);
    if (!start) {
        return -1;
    }
    const char *open = strchr(start, '[');
    const char *close = ck_ir_v2_find_array_end(open, end);
    if (!open || !close) {
        return -1;
    }

    int count = 0;
    const char *cur = open;
    const char *obj_start = NULL;
    const char *obj_end = NULL;
    while ((cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {
        count++;
    }
    if (count <= 0) {
        return -1;
    }

    CKIRV2Node *nodes = (CKIRV2Node *)calloc((size_t)count, sizeof(CKIRV2Node));
    if (!nodes) {
        return -1;
    }
    int idx = 0;
    cur = open;
    while (idx < count &&
           (cur = ck_ir_v2_next_object(cur, close, &obj_start, &obj_end)) != NULL) {

        CKIRV2Node node = {0};
        char *op = NULL;
        char *kernel = NULL;
        char *cond = NULL;
        int layer = 0;
        int flags = 0;

        ck_ir_v2_parse_string_field(obj_start, "\"op\"", obj_end, &op);
        ck_ir_v2_parse_string_field(obj_start, "\"kernel\"", obj_end, &kernel);
        ck_ir_v2_parse_string_field(obj_start, "\"condition\"", obj_end, &cond);
        ck_ir_v2_parse_int(obj_start, "\"layer\"", obj_end, &layer);
        ck_ir_v2_parse_int(obj_start, "\"flags\"", obj_end, &flags);

        node.op = op;
        node.kernel = kernel;
        node.condition = cond;
        node.layer = (uint16_t)layer;
        node.flags = (uint8_t)flags;
        node.n_bindings = 0;
        node.n_inputs = 0;
        node.n_outputs = 0;

        if (ck_ir_v2_parse_bindings(obj_start, obj_end, graph, &node) != 0) {
            free(node.op);
            free(node.kernel);
            free(node.condition);
            for (int b = 0; b < node.n_bindings; ++b) {
                free(node.bindings[b].arg);
            }
            for (int j = 0; j < idx; ++j) {
                ck_ir_v2_free_node(&nodes[j]);
            }
            free(nodes);
            return -1;
        }

        nodes[idx++] = node;
    }

    graph->nodes = nodes;
    graph->num_nodes = idx;
    return 0;
}

int ck_ir_v2_parse_json(const char *path, CKIRV2Graph *graph)
{
    if (!path || !graph) {
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("ck_ir_v2_parse_json: fopen");
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }
    size_t nread = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[nread] = '\0';

    CKIRV2Graph tmp = {0};
    const char *end = buf + nread;

    if (ck_ir_v2_parse_int(buf, "\"num_layers\"", end, &tmp.config.num_layers) != 0 ||
        ck_ir_v2_parse_int(buf, "\"hidden_size\"", end, &tmp.config.hidden_size) != 0 ||
        ck_ir_v2_parse_int(buf, "\"intermediate_size\"", end, &tmp.config.intermediate_size) != 0 ||
        ck_ir_v2_parse_int(buf, "\"num_attention_heads\"", end, &tmp.config.num_heads) != 0 ||
        ck_ir_v2_parse_int(buf, "\"num_kv_heads\"", end, &tmp.config.num_kv_heads) != 0) {
        free(buf);
        return -1;
    }

    ck_ir_v2_parse_int(buf, "\"vocab_size\"", end, &tmp.config.vocab_size);
    ck_ir_v2_parse_int(buf, "\"context_window\"", end, &tmp.config.context_window);
    ck_ir_v2_parse_float(buf, "\"rms_norm_eps\"", end, &tmp.config.rms_norm_eps);
    ck_ir_v2_parse_float(buf, "\"rope_theta\"", end, &tmp.config.rope_theta);

    if (ck_ir_v2_parse_buffers(buf, end, &tmp) != 0 ||
        ck_ir_v2_parse_nodes(buf, end, &tmp) != 0) {
        ck_ir_v2_free(&tmp);
        free(buf);
        return -1;
    }

    *graph = tmp;
    free(buf);
    return 0;
}
