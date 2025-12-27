#ifndef BF16_UTILS_H
#define BF16_UTILS_H

#include <stdint.h>
#include <stddef.h>

static inline float bf16_to_float(uint16_t v)
{
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = (uint32_t)v << 16;
    return tmp.f;
}

static inline uint16_t float_to_bf16(float f)
{
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.f = f;
    /* Round-to-nearest-even (matches common BF16 semantics and PyTorch CPU). */
    uint32_t lsb = (tmp.u >> 16) & 1u;
    tmp.u += 0x7FFFu + lsb;
    return (uint16_t)(tmp.u >> 16);
}

static inline void bf16_tensor_to_float(const uint16_t *src, float *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
}

static inline void float_tensor_to_bf16(const float *src, uint16_t *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
}

#endif /* BF16_UTILS_H */
