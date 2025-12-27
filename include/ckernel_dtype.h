#ifndef CKERNEL_DTYPE_H
#define CKERNEL_DTYPE_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Supported data types in C-Kernel-Engine
 *
 * Standard types:
 *   - CK_DT_FP32: 32-bit float (baseline, full precision)
 *   - CK_DT_BF16: Brain Float 16 (1+8+7 bits)
 *   - CK_DT_FP16: IEEE Half Precision (1+5+10 bits)
 *
 * Simple quantization:
 *   - CK_DT_INT8: 8-bit signed integer
 *   - CK_DT_INT4: 4-bit signed integer (simple, no scales)
 *
 * GGML-compatible quantization (block-based with scales):
 *   - CK_DT_Q4_0: 4-bit, 32 weights/block, 1 FP16 scale
 *   - CK_DT_Q4_K: 4-bit k-quant, 256 weights/block, nested scales (Q4_K_M)
 *   - CK_DT_Q8_0: 8-bit, 32 weights/block, 1 FP16 scale
 */
typedef enum {
    /* Standard floating-point types */
    CK_DT_FP32 = 0,      /* 4 bytes per element */
    CK_DT_BF16,          /* 2 bytes per element */
    CK_DT_FP16,          /* 2 bytes per element */

    /* Simple integer types (legacy) */
    CK_DT_INT8,          /* 1 byte per element */
    CK_DT_INT4,          /* 0.5 bytes per element (packed) */

    /* GGML-compatible block quantization */
    CK_DT_Q4_0,          /* 4.5 bits/weight (18 bytes per 32 weights) */
    CK_DT_Q4_K,          /* 4.5 bits/weight (144 bytes per 256 weights) - Q4_K_M */
    CK_DT_Q8_0,          /* 8.5 bits/weight (34 bytes per 32 weights) */

    CK_DT_COUNT
} CKDataType;

typedef uint32_t CKDataTypeMask;

#define CK_DT_MASK(dt) (1u << (uint32_t)(dt))

/**
 * @brief Check if a data type is block-quantized (GGML-style)
 */
static inline int ck_dtype_is_quantized(CKDataType dt)
{
    return dt == CK_DT_Q4_0 || dt == CK_DT_Q4_K || dt == CK_DT_Q8_0;
}

/**
 * @brief Get bytes per element for non-quantized types
 * @note For quantized types, use ck_dtype_block_bytes() and ck_dtype_block_size()
 */
static inline size_t ck_dtype_bytes(CKDataType dt)
{
    switch (dt) {
    case CK_DT_BF16:
    case CK_DT_FP16:
        return 2;
    case CK_DT_INT8:
        return 1;
    case CK_DT_INT4:
        return 1; /* Note: actually 0.5, but stored as pairs */
    case CK_DT_FP32:
    default:
        return 4;
    }
}

/**
 * @brief Get the number of elements per quantization block
 */
static inline size_t ck_dtype_block_size(CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q4_0:
    case CK_DT_Q8_0:
        return 32;
    case CK_DT_Q4_K:
        return 256;
    default:
        return 1; /* Non-quantized types: 1 element per "block" */
    }
}

/**
 * @brief Get bytes per block for quantized types
 */
static inline size_t ck_dtype_block_bytes(CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q4_0:
        return 18;   /* 2 (scale) + 16 (32 x 4-bit) */
    case CK_DT_Q4_K:
        return 144;  /* 2 + 2 + 12 + 128 */
    case CK_DT_Q8_0:
        return 34;   /* 2 (scale) + 32 (32 x 8-bit) */
    default:
        return ck_dtype_bytes(dt);
    }
}

/**
 * @brief Calculate total bytes for n_elements of given dtype
 */
static inline size_t ck_dtype_row_bytes(CKDataType dt, size_t n_elements)
{
    if (ck_dtype_is_quantized(dt)) {
        size_t block_size = ck_dtype_block_size(dt);
        size_t n_blocks = (n_elements + block_size - 1) / block_size;
        return n_blocks * ck_dtype_block_bytes(dt);
    }
    return n_elements * ck_dtype_bytes(dt);
}

static inline int ck_dtype_supported(CKDataTypeMask mask, CKDataType dt)
{
    return (mask & CK_DT_MASK(dt)) != 0;
}

#endif /* CKERNEL_DTYPE_H */
