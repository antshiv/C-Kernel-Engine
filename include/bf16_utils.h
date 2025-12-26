#ifndef BF16_UTILS_H
#define BF16_UTILS_H

#include <stdint.h>

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
    return (uint16_t)(tmp.u >> 16);
}

#endif /* BF16_UTILS_H */
