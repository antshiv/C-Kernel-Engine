/**
 * @file gemm_kernels_q4k_q8k_vnni.c
 * @brief VNNI Q4_K x Q8_K matvec kernel (inference only)
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline void load_q8_even_odd_16(const int8_t *q8,
                                       __m128i *even8,
                                       __m128i *odd8) {
    const __m128i q8_lo = _mm_loadu_si128((const __m128i *)q8);
    const __m128i q8_hi = _mm_loadu_si128((const __m128i *)(q8 + 16));
    const __m128i even_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128i odd_mask = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);

    const __m128i q8_lo_even = _mm_shuffle_epi8(q8_lo, even_mask);
    const __m128i q8_hi_even = _mm_shuffle_epi8(q8_hi, even_mask);
    *even8 = _mm_unpacklo_epi64(q8_lo_even, q8_hi_even);

    const __m128i q8_lo_odd = _mm_shuffle_epi8(q8_lo, odd_mask);
    const __m128i q8_hi_odd = _mm_shuffle_epi8(q8_hi, odd_mask);
    *odd8 = _mm_unpacklo_epi64(q8_lo_odd, q8_hi_odd);
}

static inline int32_t dot_q4_q8_32_vnni(const uint8_t *q4,
                                        const int8_t *q8) {
    const __m128i q4_packed = _mm_loadu_si128((const __m128i *)q4);
    const __m256i q4_16 = _mm256_cvtepu8_epi16(q4_packed);
    const __m256i mask4 = _mm256_set1_epi16(0x0F);

    const __m256i q4_lo16 = _mm256_and_si256(q4_16, mask4);
    const __m256i q4_hi16 = _mm256_and_si256(_mm256_srli_epi16(q4_16, 4), mask4);

    const __m128i q4_lo8 = _mm_packus_epi16(_mm256_castsi256_si128(q4_lo16),
                                            _mm256_extracti128_si256(q4_lo16, 1));
    const __m128i q4_hi8 = _mm_packus_epi16(_mm256_castsi256_si128(q4_hi16),
                                            _mm256_extracti128_si256(q4_hi16, 1));
    const __m256i q4_bytes = _mm256_set_m128i(q4_hi8, q4_lo8);

    __m128i q8_even8;
    __m128i q8_odd8;
    load_q8_even_odd_16(q8, &q8_even8, &q8_odd8);
    const __m256i q8_bytes = _mm256_set_m128i(q8_odd8, q8_even8);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}
#endif

void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        float sumf = 0.0f;

        for (int i = 0; i < blocks_per_row; ++i) {
            uint8_t sc[8], m[8];
            unpack_q4_k_scales(w_row[i].scales, sc, m);

            int32_t sum_scale = 0;
            int32_t sum_min = 0;

            for (int sub = 0; sub < 8; ++sub) {
                const uint8_t *qs = &w_row[i].qs[sub * 16];
                const int8_t *q8 = &x[i].qs[sub * 32];

                const int32_t sum_q4q8 = dot_q4_q8_32_vnni(qs, q8);
                const int32_t sum_q8 = (int32_t)x[i].bsums[sub * 2] +
                                       (int32_t)x[i].bsums[sub * 2 + 1];

                sum_scale += (int32_t)sc[sub] * (sum_q4q8 - 8 * sum_q8);
                sum_min += (int32_t)m[sub] * sum_q8;
            }

            const float d = CK_FP16_TO_FP32(w_row[i].d) * x[i].d;
            const float dmin = CK_FP16_TO_FP32(w_row[i].dmin) * x[i].d;
            sumf += d * (float)sum_scale;
            sumf += dmin * (float)sum_min;
        }

        y[row] = sumf;
    }
#elif defined(__AVX2__)
    gemv_q4_k_q8_k_avx2(y, W, x_q8, M, K);
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}
