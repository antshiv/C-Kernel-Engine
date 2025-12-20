#ifndef CKERNEL_CODEGEN_H
#define CKERNEL_CODEGEN_H

#include "ckernel_ir.h"

#include <stdio.h>

/**
 * Emit a C skeleton for forward + backward execution based on the IR.
 *
 * This does not yet generate full pointer arithmetic or memory planning.
 * It is intended as a starting point that:
 *  - Defines a model config / runtime context
 *  - Shows a per-layer forward loop over IR nodes
 *  - Sketches a backward loop over the backward IR
 */
void ck_codegen_c_skeleton(const CKIRGraph *forward,
                           const CKIRGraph *backward,
                           FILE *out);

/**
 * Emit a C runtime file that stitches kernels for the given forward IR.
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_codegen_emit_runtime(const CKIRGraph *forward, const char *path);

#endif /* CKERNEL_CODEGEN_H */
