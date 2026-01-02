#ifndef CKERNEL_IR_V2_LOWER_H
#define CKERNEL_IR_V2_LOWER_H

#include "ckernel_ir_v2.h"
#include "ckernel_mem_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CK_IR_V2_LOWER_PREFILL = 0,
    CK_IR_V2_LOWER_DECODE = 1,
    CK_IR_V2_LOWER_BACKWARD = 2
} CKIRV2LowerMode;

const char *ck_ir_v2_lower_mode_name(CKIRV2LowerMode mode);
int ck_ir_v2_lower_mode_from_string(const char *name, CKIRV2LowerMode *out_mode);

int ck_ir_v2_lower_graph(const CKIRV2Graph *input,
                         CKIRV2LowerMode mode,
                         CKIRV2Graph *output,
                         CKMemPlan *plan);

int ck_ir_v2_lower_emit_json(const CKIRV2Graph *input,
                             CKIRV2LowerMode mode,
                             const char *path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKERNEL_IR_V2_LOWER_H */
