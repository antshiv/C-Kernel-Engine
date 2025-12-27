#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

/*
 * Reference GEMM for BF16 A/B/bias with BF16 output.
 *
 * Layout matches gemm_blocked_serial:
 *   A: [M x K] row-major
 *   B: [N x K] row-major (stored as [out x in])
 *   C: [M x N] row-major
 *
 * This is a correctness-first kernel; higher-performance BF16 paths can
 * replace it later (AVX-512 BF16 / AMX).
 */
void gemm_blocked_serial_bf16(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M,
                              int N,
                              int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = bias ? (double)bf16_to_float(bias[j]) : 0.0;
            const size_t a_row = (size_t)i * (size_t)K;
            const size_t b_row = (size_t)j * (size_t)K;
            for (int k = 0; k < K; ++k) {
                sum += (double)bf16_to_float(A[a_row + (size_t)k]) *
                       (double)bf16_to_float(B[b_row + (size_t)k]);
            }
            C[(size_t)i * (size_t)N + (size_t)j] = float_to_bf16((float)sum);
        }
    }
}

