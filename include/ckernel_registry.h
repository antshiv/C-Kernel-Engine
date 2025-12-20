#ifndef CKERNEL_REGISTRY_H
#define CKERNEL_REGISTRY_H

#include "ckernel_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

int ck_op_supported(CKOpType op);

int ck_ir_validate_supported(const CKIRGraph *graph);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* CKERNEL_REGISTRY_H */
