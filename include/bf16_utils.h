#ifndef BF16_UTILS_H
#define BF16_UTILS_H

#include <stdint.h>
#include <stddef.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// ============================================================================
// Scalar BF16 <-> FP32 conversion
// ============================================================================

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

// ============================================================================
// AVX-512 BF16 <-> FP32 conversion (16 elements at a time)
// Must be defined before the tensor conversion functions that use them
// ============================================================================

#if defined(__AVX512F__)

/* Convert 16 BF16 values (in __m256i) to 16 FP32 values (in __m512) */
static inline __m512 bf16x16_to_fp32(__m256i bf16_vec)
{
    /* BF16 to FP32: zero-extend to 32-bit, then shift left by 16 bits */
    __m512i as_int = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(as_int, 16);
    return _mm512_castsi512_ps(shifted);
}

/* Convert 16 FP32 values (in __m512) to 16 BF16 values (in __m256i) with rounding */
static inline __m256i fp32x16_to_bf16(__m512 fp32_vec)
{
    /* Round-to-nearest-even, then truncate to BF16 */
    __m512i as_int = _mm512_castps_si512(fp32_vec);
    __m512i lsb = _mm512_srli_epi32(as_int, 16);
    lsb = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
    __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    __m512i rounded = _mm512_add_epi32(as_int, rounding);
    __m512i shifted = _mm512_srli_epi32(rounded, 16);
    return _mm512_cvtepi32_epi16(shifted);
}

/* Load 16 BF16 values and convert to FP32 */
static inline __m512 bf16_loadu_cvt_fp32(const uint16_t *ptr)
{
    __m256i bf16_vec = _mm256_loadu_si256((const __m256i *)ptr);
    return bf16x16_to_fp32(bf16_vec);
}

/* Convert FP32 to BF16 and store 16 values */
static inline void fp32_cvt_storeu_bf16(uint16_t *ptr, __m512 fp32_vec)
{
    __m256i bf16_vec = fp32x16_to_bf16(fp32_vec);
    _mm256_storeu_si256((__m256i *)ptr, bf16_vec);
}

#endif /* __AVX512F__ */

// ============================================================================
// Tensor conversion functions (use SIMD when available)
// ============================================================================

static inline void bf16_tensor_to_float(const uint16_t *src, float *dst, size_t count)
{
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 fp32_vec = bf16_loadu_cvt_fp32(&src[i]);
        _mm512_storeu_ps(&dst[i], fp32_vec);
    }
    for (; i < count; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
#endif
}

static inline void float_tensor_to_bf16(const float *src, uint16_t *dst, size_t count)
{
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 fp32_vec = _mm512_loadu_ps(&src[i]);
        fp32_cvt_storeu_bf16(&dst[i], fp32_vec);
    }
    for (; i < count; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
#endif
}

#endif /* BF16_UTILS_H */
