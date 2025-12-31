#ifndef CKERNEL_CODEGEN_V2_H
#define CKERNEL_CODEGEN_V2_H

#include "ckernel_codegen.h"
#include "ckernel_ir_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Emit a C runtime file from a CKIRV2Graph.
 *
 * This v2 emitter is IR-driven and will evolve to generate fully wired
 * kernels once buffer bindings and fusion metadata are added to the IR.
 */
int ck_codegen_v2_emit_runtime(const CKIRV2Graph *graph,
                               const char *path,
                               CKEmitMode mode);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKERNEL_CODEGEN_V2_H */
