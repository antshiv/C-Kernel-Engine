#ifndef BF16_UTILS_H
#define BF16_UTILS_H

#include <stdint.h>
#include <stddef.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// ============================================================================
// BF16 (Brain Floating Point) Format Overview
// ============================================================================
//
// BF16 is a 16-bit floating-point format designed for machine learning:
//
//   FP32 (32 bits):  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
//                     1    8              23 mantissa bits
//
//   BF16 (16 bits):  [S][EEEEEEEE][MMMMMMM]
//                     1    8         7 mantissa bits
//
// Key insight: BF16 is the UPPER 16 bits of FP32. Same exponent range (-126 to
// +127), but only 7 mantissa bits instead of 23. This preserves the full
// dynamic range of FP32 while sacrificing precision.
//
// Why BF16 over FP16?
//   - FP16 has only 5 exponent bits (range ±65504), causing overflow in ML
//   - BF16 has 8 exponent bits (same as FP32), no overflow issues
//   - Conversion is trivial: just truncate/round the lower 16 bits
//
// ============================================================================
// Scalar BF16 <-> FP32 conversion
// ============================================================================

// BF16 to FP32: Zero-extend the 16-bit value to the upper half of FP32
// This is lossless - every BF16 value maps to exactly one FP32 value.
static inline float bf16_to_float(uint16_t v)
{
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = (uint32_t)v << 16;  // Place BF16 in upper 16 bits, lower bits = 0
    return tmp.f;
}

// FP32 to BF16: Extract upper 16 bits with ROUND-TO-NEAREST-EVEN rounding
//
// Why not just truncate (>> 16)?
// ─────────────────────────────
// Simple truncation always rounds DOWN, creating systematic negative bias.
// Over many operations, this compounds:
//   - 1.999... truncates to 1.0 (lost 0.999)
//   - 2.001... truncates to 2.0 (lost 0.001)
//   - Average error: -0.5 LSB (always negative!)
//
// Round-to-nearest-even algorithm:
// ────────────────────────────────
// We add a rounding bias before truncating. The bias depends on the LSB
// (least significant bit) of the result to implement "banker's rounding":
//
//   Fractional part < 0.5:  Round down (add 0x7FFF, doesn't overflow to next)
//   Fractional part > 0.5:  Round up   (add 0x7FFF, overflows to next)
//   Fractional part = 0.5:  Round to EVEN (add 0x7FFF + LSB)
//                           - If result LSB=0 (even), add 0x7FFF → stays same
//                           - If result LSB=1 (odd),  add 0x8000 → rounds up
//
// The magic constant 0x7FFF is "almost half" of the lower 16 bits (0xFFFF).
// Adding 0x7FFF + lsb effectively rounds to nearest, with ties going to even.
//
// Example: Converting FP32 1.5f (exactly representable in BF16)
//   FP32 bits: 0x3FC00000 = 0011 1111 1100 0000 | 0000 0000 0000 0000
//                          ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
//                          upper 16 (BF16)      lower 16 (to discard)
//   Lower 16 = 0x0000, LSB of upper = 0
//   Add 0x7FFF + 0 = 0x7FFF: 0x3FC07FFF
//   Shift >> 16: 0x3FC0 ✓ (no change, fraction was 0)
//
// Example: Converting FP32 1.0000001f (not exact in BF16)
//   FP32 bits: 0x3F800001
//   Lower 16 = 0x0001, LSB of upper = 0
//   Add 0x7FFF: 0x3F808000 → overflows upper bits
//   Shift >> 16: 0x3F80 (rounded down, fraction < 0.5)
//
// Example: Converting FP32 1.0078125f (BF16 boundary, exactly 0.5 between)
//   This is exactly halfway between two BF16 values
//   Round-to-even picks the one with LSB=0
//
static inline uint16_t float_to_bf16(float f)
{
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.f = f;
    // Extract bit 16 (will be the LSB of the BF16 result after truncation)
    uint32_t lsb = (tmp.u >> 16) & 1u;
    // Add rounding bias: 0x7FFF normally, 0x8000 if LSB=1 (rounds ties to even)
    tmp.u += 0x7FFFu + lsb;
    // Truncate lower 16 bits
    return (uint16_t)(tmp.u >> 16);
}

// ============================================================================
// AVX-512 BF16 <-> FP32 conversion (16 elements at a time)
// ============================================================================
//
// AVX-512 provides 512-bit registers (16 floats) for efficient vectorization.
// We process 16 BF16 values at once, stored in a 256-bit register (__m256i).
//
// Memory layout for 16 BF16 values (32 bytes):
//   __m256i: [bf16_0][bf16_1][bf16_2]...[bf16_15]  (16 × 16-bit)
//
// After conversion to FP32 (64 bytes):
//   __m512:  [fp32_0][fp32_1][fp32_2]...[fp32_15]  (16 × 32-bit)
//
// ============================================================================

#if defined(__AVX512F__)

// BF16 to FP32 vectorized: Zero-extend 16-bit → 32-bit, then shift left 16
//
// Step-by-step for one element (but done for all 16 in parallel):
//   Input BF16:   0x3F80 (represents 1.0f)
//   Zero-extend:  0x00003F80
//   Shift << 16:  0x3F800000 (IEEE-754 for 1.0f) ✓
//
static inline __m512 bf16x16_to_fp32(__m256i bf16_vec)
{
    // Zero-extend 16 × uint16 to 16 × uint32
    __m512i as_int = _mm512_cvtepu16_epi32(bf16_vec);
    // Shift each 32-bit value left by 16 bits (move BF16 to upper half)
    __m512i shifted = _mm512_slli_epi32(as_int, 16);
    // Reinterpret bits as float (no conversion, just type cast)
    return _mm512_castsi512_ps(shifted);
}

// FP32 to BF16 vectorized: Same round-to-nearest-even algorithm, SIMD version
//
// This is the vectorized equivalent of float_to_bf16() above.
// Each of the 16 lanes independently performs:
//   1. Extract LSB of would-be BF16 result
//   2. Add rounding bias (0x7FFF + lsb)
//   3. Truncate by right-shifting 16 bits
//
static inline __m256i fp32x16_to_bf16(__m512 fp32_vec)
{
    // Reinterpret float bits as integers
    __m512i as_int = _mm512_castps_si512(fp32_vec);

    // Extract bit 16 of each value (this becomes LSB of BF16 result)
    __m512i lsb = _mm512_srli_epi32(as_int, 16);
    lsb = _mm512_and_si512(lsb, _mm512_set1_epi32(1));

    // Compute rounding bias: 0x7FFF + lsb (0x7FFF or 0x8000)
    __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);

    // Add rounding bias and shift right to get BF16 value
    __m512i rounded = _mm512_add_epi32(as_int, rounding);
    __m512i shifted = _mm512_srli_epi32(rounded, 16);

    // Pack 16 × 32-bit down to 16 × 16-bit (truncates upper bits, which are 0)
    return _mm512_cvtepi32_epi16(shifted);
}

// Convenience: Load 16 BF16 values from memory and convert to FP32
static inline __m512 bf16_loadu_cvt_fp32(const uint16_t *ptr)
{
    __m256i bf16_vec = _mm256_loadu_si256((const __m256i *)ptr);
    return bf16x16_to_fp32(bf16_vec);
}

// Convenience: Convert 16 FP32 values to BF16 and store to memory
static inline void fp32_cvt_storeu_bf16(uint16_t *ptr, __m512 fp32_vec)
{
    __m256i bf16_vec = fp32x16_to_bf16(fp32_vec);
    _mm256_storeu_si256((__m256i *)ptr, bf16_vec);
}

#endif /* __AVX512F__ */

// ============================================================================
// Tensor conversion functions (use SIMD when available)
// ============================================================================
//
// These functions convert entire tensors between BF16 and FP32.
// Two common usage patterns in neural network inference:
//
// Pattern 1: Convert-Compute-Convert
//   - Load BF16 weights/activations
//   - Convert entire tensor to FP32
//   - Compute in FP32 for full precision
//   - Convert result back to BF16
//   - Good for: Simple ops, debugging, when memory isn't critical
//
// Pattern 2: Inline Conversion (preferred for performance)
//   - Load BF16 values directly in compute kernel
//   - Convert to FP32 in registers (no memory write)
//   - Compute in FP32
//   - Convert back to BF16 before storing
//   - Good for: GEMM, activations, fused kernels
//
// The functions below implement Pattern 1. For Pattern 2, use
// bf16_loadu_cvt_fp32() and fp32_cvt_storeu_bf16() directly in kernels.
//
// ============================================================================

// Convert BF16 tensor to FP32 tensor
// Uses AVX-512 when available (16 elements per iteration)
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

// Convert FP32 tensor to BF16 tensor
// Uses AVX-512 when available (16 elements per iteration)
// Applies round-to-nearest-even for each conversion
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
