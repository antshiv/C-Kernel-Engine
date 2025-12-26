#ifndef CKERNEL_DTYPE_H
#define CKERNEL_DTYPE_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    CK_DT_FP32 = 0,
    CK_DT_BF16,
    CK_DT_FP16,
    CK_DT_INT8,
    CK_DT_INT4,
    CK_DT_COUNT
} CKDataType;

typedef uint32_t CKDataTypeMask;

#define CK_DT_MASK(dt) (1u << (uint32_t)(dt))

static inline size_t ck_dtype_bytes(CKDataType dt)
{
    switch (dt) {
    case CK_DT_BF16:
    case CK_DT_FP16:
        return 2;
    case CK_DT_INT8:
        return 1;
    case CK_DT_INT4:
        return 1;
    case CK_DT_FP32:
    default:
        return 4;
    }
}

static inline int ck_dtype_supported(CKDataTypeMask mask, CKDataType dt)
{
    return (mask & CK_DT_MASK(dt)) != 0;
}

#endif /* CKERNEL_DTYPE_H */
