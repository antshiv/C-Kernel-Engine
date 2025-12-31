#ifndef CKERNEL_KERNEL_SPECS_H
#define CKERNEL_KERNEL_SPECS_H

#include <stddef.h>

#include "ckernel_dtype.h"

typedef enum {
    CK_DIM_TOKENS = 0,
    CK_DIM_EMBED,
    CK_DIM_ALIGNED_EMBED,
    CK_DIM_HEAD_DIM,
    CK_DIM_ALIGNED_HEAD,
    CK_DIM_NUM_HEADS,
    CK_DIM_NUM_KV_HEADS,
    CK_DIM_ALIGNED_CTX,
    CK_DIM_INTERMEDIATE,
    CK_DIM_ALIGNED_INTERMEDIATE,
    CK_DIM_VOCAB,
    CK_DIM_END
} CKDimKind;

#define CKERNEL_MAX_KERNEL_SOURCES 8

typedef struct {
    CKDimKind dim;
    int mult;
    int div;
} CKDimToken;

typedef enum {
    CK_SCOPE_LAYER = 0,
    CK_SCOPE_GLOBAL
} CKBufferScope;

typedef enum {
    CK_ROLE_INPUT = 0,
    CK_ROLE_OUTPUT,
    CK_ROLE_ACTIVATION,
    CK_ROLE_WEIGHT,
    CK_ROLE_SCRATCH,
    CK_ROLE_GRAD
} CKBufferRole;

typedef struct {
    const char *name;
    CKBufferScope scope;
    CKBufferRole role;
    CKDimToken shape[4];
    int optional;
    const char *alias_of;
    const char *condition;
    CKDataType dtype;
} CKBufferSpec;

typedef struct {
    const char *name;
    const char *forward[CK_DT_COUNT];
    const char *backward[CK_DT_COUNT];
    CKDataTypeMask dtype_mask;
    CKDataType default_dtype;
    const char *sources[CKERNEL_MAX_KERNEL_SOURCES];
} CKKernelSpec;

typedef struct {
    const char *kernel;
    const char *condition;
} CKPlanStep;

typedef struct {
    const char *arg;
    const char *buffer;
} CKPlanBinding;

typedef struct {
    const char *kernel;
    const char *condition;
    const CKPlanBinding *bindings;
    size_t num_bindings;
} CKPlanStepV2;

extern const CKBufferSpec ck_decoder_buffers[];
extern const size_t ck_decoder_buffer_count;

extern const CKKernelSpec ck_kernel_specs[];
extern const size_t ck_kernel_spec_count;

extern const CKPlanStep ck_decoder_forward_plan[];
extern const size_t ck_decoder_forward_plan_count;

extern const CKPlanStep ck_decoder_backward_plan[];
extern const size_t ck_decoder_backward_plan_count;

extern const CKPlanStepV2 ck_decoder_forward_plan_v2[];
extern const size_t ck_decoder_forward_plan_v2_count;

extern const CKPlanStepV2 ck_decoder_backward_plan_v2[];
extern const size_t ck_decoder_backward_plan_v2_count;

#endif /* CKERNEL_KERNEL_SPECS_H */
