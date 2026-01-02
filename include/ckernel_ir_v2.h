#ifndef CKERNEL_IR_V2_H
#define CKERNEL_IR_V2_H

#include "ckernel_ir.h"
#include "ckernel_kernel_specs.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_IR_V2_MAX_DIMS 4
#define CK_IR_V2_MAX_INPUTS 8
#define CK_IR_V2_MAX_OUTPUTS 4
#define CK_IR_V2_MAX_BINDINGS 24

typedef enum {
    CK_IR_V2_NODE_NONE = 0,
    CK_IR_V2_NODE_FUSED = 1 << 0,
    CK_IR_V2_NODE_INFERENCE_ONLY = 1 << 1
} CKIRV2NodeFlags;

typedef struct {
    char *name;
    CKBufferScope scope;
    CKBufferRole role;
    CKDataType dtype;
    CKDimToken shape[CK_IR_V2_MAX_DIMS];
    int optional;
    char *alias_of;
    char *condition;
} CKIRV2Buffer;

typedef struct {
    char *arg;
    int32_t buffer;
} CKIRV2Binding;

typedef struct {
    char *op;
    char *kernel;
    CKDataType kernel_dtype;
    char *condition;
    uint16_t layer;
    uint8_t flags;
    CKIRV2Binding bindings[CK_IR_V2_MAX_BINDINGS];
    uint8_t n_bindings;
    int32_t inputs[CK_IR_V2_MAX_INPUTS];
    uint8_t n_inputs;
    int32_t outputs[CK_IR_V2_MAX_OUTPUTS];
    uint8_t n_outputs;
} CKIRV2Node;

typedef struct {
    CKModelConfig config;
    int has_pos_emb;
    int tie_word_embeddings;
    int fused_qkv;
    int gated_mlp;
    int num_buffers;
    CKIRV2Buffer *buffers;
    int num_nodes;
    CKIRV2Node *nodes;
} CKIRV2Graph;

struct CKMemPlan;

int ck_ir_v2_build_decoder(const CKModelConfig *cfg, CKIRV2Graph *graph);
int ck_ir_v2_build_decoder_backward(const CKIRV2Graph *forward, CKIRV2Graph *backward);
int ck_ir_v2_apply_meta(const char *path, CKIRV2Graph *graph);
int ck_ir_v2_serialize_json(const CKIRV2Graph *graph, const char *path);
int ck_ir_v2_serialize_json_with_plan(const CKIRV2Graph *graph,
                                      const struct CKMemPlan *plan,
                                      const char *mode,
                                      int tokens_override,
                                      int base_context_window,
                                      const char *path);
int ck_ir_v2_parse_json(const char *path, CKIRV2Graph *graph);
void ck_ir_v2_free(CKIRV2Graph *graph);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKERNEL_IR_V2_H */
