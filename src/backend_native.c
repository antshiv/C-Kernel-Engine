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
    (void)lda;
    (void)ldb;
    (void)ldc;
    gemm_blocked_serial(A, B, bias, C, M, N, K);
}

CKMathBackend ckernel_backend_native(void)
{
    CKMathBackend b;
    b.sgemm = &ckernel_sgemm_native;
    return b;
}
