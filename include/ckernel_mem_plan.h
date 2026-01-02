#ifndef CKERNEL_MEM_PLAN_H
#define CKERNEL_MEM_PLAN_H

#include "ckernel_ir_v2.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_MEM_PLAN_DEFAULT_ALIGN 64

typedef enum {
    CK_MEM_ARENA_WEIGHTS = 0,
    CK_MEM_ARENA_ACTIVATIONS = 1,
    CK_MEM_ARENA_GRADS = 2,
    CK_MEM_ARENA_COUNT
} CKMemArenaKind;

typedef struct CKMemSpan {
    int buffer_id;
    CKMemArenaKind arena;
    size_t offset_bytes;
    size_t size_bytes;
} CKMemSpan;

typedef struct CKMemPlan {
    CKMemSpan *spans;
    int num_spans;
    size_t total_bytes[CK_MEM_ARENA_COUNT];
    size_t alignment_bytes;
} CKMemPlan;

int ck_mem_plan_build_inference(const CKIRV2Graph *graph,
                                CKMemPlan *plan,
                                size_t alignment_bytes);

int ck_mem_plan_build_training(const CKIRV2Graph *graph,
                               CKMemPlan *plan,
                               size_t alignment_bytes);

int ck_mem_plan_build_inference_with_tokens(const CKIRV2Graph *graph,
                                            CKMemPlan *plan,
                                            size_t alignment_bytes,
                                            int tokens_override);

int ck_mem_plan_build_training_with_tokens(const CKIRV2Graph *graph,
                                           CKMemPlan *plan,
                                           size_t alignment_bytes,
                                           int tokens_override);

void ck_mem_plan_free(CKMemPlan *plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKERNEL_MEM_PLAN_H */
