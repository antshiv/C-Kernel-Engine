#ifndef CKERNEL_CODEGEN_H
#define CKERNEL_CODEGEN_H

#include "ckernel_ir.h"

#include <stdio.h>

/**
 * Code generation output mode.
 */
typedef enum {
    CK_EMIT_STANDALONE = 0,  /* Emit with main() for standalone executable */
    CK_EMIT_LIBRARY = 1,     /* Emit as library with API functions, no main() */
} CKEmitMode;

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
 * @param forward The forward IR graph
 * @param path Output file path
 * @param mode CK_EMIT_STANDALONE for executable with main(),
 *             CK_EMIT_LIBRARY for shared object with API functions
 *
 * Returns 0 on success, non-zero on failure.
 */
int ck_codegen_emit_runtime(const CKIRGraph *forward, const char *path, CKEmitMode mode);

#endif /* CKERNEL_CODEGEN_H */
