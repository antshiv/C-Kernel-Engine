/**
 * Native GEMM backend that directly reuses the C-Transformer GEMM kernel.
 *
 * Layout assumptions (identical to C-Transformer/main.c):
 *  - A: [M x K], row-major,       A(i,k) = A[i*K + k]
 *  - B: [N x K], row-major,       B(j,k) = B[j*K + k]
 *  - C: [M x N], row-major,       C(i,j) = C[i*N + j]
 *
 * This is a straight copy of gemm_blocked_serial with a thin wrapper to match
 * the CKMathBackend.sgemm signature. It is intentionally minimal.
 */

#include "ckernel_engine.h"

// Thin wrapper matching CKMathBackend.sgemm. For now we deliberately assume
// lda = K, ldb = K, ldc = N (the dense LLM layouts) and ignore the lda/ldb/ldc
// parameters to keep the implementation identical to the original kernel.
static void ckernel_sgemm_native(int M, int N, int K,
                                 const float *A, int lda,
                                 const float *B, int ldb,
                                 const float *bias,
                                 float *C, int ldc)
{
    /* Honor caller-provided strides so padded matrices still compute correctly. */
    for (int i = 0; i < M; ++i) {
        const float *a_row = A + (size_t)i * lda;
        float *c_row = C + (size_t)i * ldc;
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            const float *b_row = B + (size_t)j * ldb;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[j] = sum;
        }
    }
}

CKMathBackend ckernel_backend_native(void)
{
    CKMathBackend b;
    b.sgemm = &ckernel_sgemm_native;
    return b;
}
