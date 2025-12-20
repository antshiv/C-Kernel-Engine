#include "ckernel_registry.h"

#include <stdio.h>

static const char *ck_op_name(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:        return "RMSNORM";
    case CK_OP_LINEAR_QKV:     return "LINEAR_QKV";
    case CK_OP_ATTENTION:      return "ATTENTION";
    case CK_OP_ADD:            return "ADD";
    case CK_OP_LINEAR:         return "LINEAR";
    case CK_OP_SPLIT:          return "SPLIT";
    case CK_OP_SWIGLU:         return "SWIGLU";
    case CK_OP_RMSNORM_BWD:    return "RMSNORM_BWD";
    case CK_OP_LINEAR_QKV_BWD: return "LINEAR_QKV_BWD";
    case CK_OP_ATTENTION_BWD:  return "ATTENTION_BWD";
    case CK_OP_ADD_BWD:        return "ADD_BWD";
    case CK_OP_LINEAR_BWD:     return "LINEAR_BWD";
    case CK_OP_SPLIT_BWD:      return "SPLIT_BWD";
    case CK_OP_SWIGLU_BWD:     return "SWIGLU_BWD";
    default:                   return "UNKNOWN";
    }
}

int ck_op_supported(CKOpType op)
{
    switch (op) {
    case CK_OP_RMSNORM:
    case CK_OP_LINEAR_QKV:
    case CK_OP_ATTENTION:
    case CK_OP_ADD:
    case CK_OP_LINEAR:
    case CK_OP_SPLIT:
    case CK_OP_SWIGLU:
        return 1;
    default:
        return 0;
    }
}

int ck_ir_validate_supported(const CKIRGraph *graph)
{
    if (!graph || !graph->nodes || graph->num_nodes <= 0) {
        return -1;
    }

    for (int i = 0; i < graph->num_nodes; ++i) {
        CKOpType op = graph->nodes[i].op;
        if (!ck_op_supported(op)) {
            fprintf(stderr,
                    "Unsupported op in IR: %s (layer=%u node=%u)\n",
                    ck_op_name(op),
                    (unsigned)graph->nodes[i].id.layer,
                    (unsigned)graph->nodes[i].id.node);
            return -1;
        }
    }
    return 0;
}
