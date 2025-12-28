/**
 * GEMM Benchmark: C-Kernel-Engine vs oneDNN
 *
 * Compares our GEMM implementations against Intel oneDNN to identify
 * performance gaps and optimization opportunities.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp benchmarks/bench_gemm_vs_onednn.c \
 *       -I/usr/local/include -L/usr/local/lib -ldnnl \
 *       -Iinclude -Lbuild -lckernel -o build/bench_gemm
 *
 * Run:
 *   LD_LIBRARY_PATH=build:/usr/local/lib ./build/bench_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// oneDNN headers
#include <dnnl.h>

// Declare our kernels (we'll link against libckernel)
#ifdef __cplusplus
extern "C" {
#endif
extern void gemm_blocked_serial(const float *A, const float *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_avx512_parallel(const float *A, const float *B, const float *bias,
                                 float *C, int M, int N, int K);
extern void gemm_fine_grained_parallel(const float *A, const float *B, const float *bias,
                                       float *C, int M, int N, int K);
extern int ck_strict_parity_enabled(void);
#ifdef __cplusplus
}
#endif

// Timing helper
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Random init
static void random_init(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Compute GFLOPS
static double compute_gflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
    return flops / (ms * 1e6);       // GFLOPS
}

// Check for errors
#define CHECK_DNNL(call) do { \
    dnnl_status_t status = (call); \
    if (status != dnnl_success) { \
        fprintf(stderr, "oneDNN error: status=%d at line %d\n", status, __LINE__); \
        exit(1); \
    } \
} while(0)

// oneDNN GEMM wrapper
typedef struct {
    dnnl_engine_t engine;
    dnnl_stream_t stream;
} dnnl_context_t;

static void init_dnnl(dnnl_context_t *ctx) {
    CHECK_DNNL(dnnl_engine_create(&ctx->engine, dnnl_cpu, 0));
    CHECK_DNNL(dnnl_stream_create(&ctx->stream, ctx->engine, dnnl_stream_default_flags));
}

static void cleanup_dnnl(dnnl_context_t *ctx) {
    dnnl_stream_destroy(ctx->stream);
    dnnl_engine_destroy(ctx->engine);
}

// Run oneDNN GEMM: C = A @ B.T (to match our layout)
static double run_dnnl_gemm(dnnl_context_t *ctx, float *A, float *B, float *C,
                            int M, int N, int K, int warmup_iters, int bench_iters) {
    // oneDNN uses row-major by default
    // A: [M x K], B: [N x K] (we want B transposed), C: [M x N]

    dnnl_memory_desc_t a_md, b_md, c_md;
    dnnl_memory_t a_mem, b_mem, c_mem;

    // Create memory descriptors - use plain format
    dnnl_dims_t a_dims = {M, K};
    dnnl_dims_t b_dims = {N, K};  // Stored as [N,K] but used transposed
    dnnl_dims_t c_dims = {M, N};

    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&a_md, 2, a_dims, dnnl_f32, dnnl_ab));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&b_md, 2, b_dims, dnnl_f32, dnnl_ab));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&c_md, 2, c_dims, dnnl_f32, dnnl_ab));

    CHECK_DNNL(dnnl_memory_create(&a_mem, a_md, ctx->engine, A));
    CHECK_DNNL(dnnl_memory_create(&b_mem, b_md, ctx->engine, B));
    CHECK_DNNL(dnnl_memory_create(&c_mem, c_md, ctx->engine, C));

    // Create matmul primitive descriptor
    // C = A @ B^T means transB = true
    dnnl_primitive_desc_t matmul_pd;
    dnnl_primitive_t matmul;

    CHECK_DNNL(dnnl_matmul_primitive_desc_create(&matmul_pd, ctx->engine,
                                                  a_md, b_md, NULL, c_md, NULL));
    CHECK_DNNL(dnnl_primitive_create(&matmul, matmul_pd));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        dnnl_exec_arg_t args[3] = {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        };
        CHECK_DNNL(dnnl_primitive_execute(matmul, ctx->stream, 3, args));
        CHECK_DNNL(dnnl_stream_wait(ctx->stream));
    }

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < bench_iters; i++) {
        dnnl_exec_arg_t args[3] = {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        };
        CHECK_DNNL(dnnl_primitive_execute(matmul, ctx->stream, 3, args));
        CHECK_DNNL(dnnl_stream_wait(ctx->stream));
    }
    double elapsed = get_time_ms() - start;

    // Cleanup
    dnnl_primitive_destroy(matmul);
    dnnl_primitive_desc_destroy(matmul_pd);
    dnnl_memory_destroy(c_mem);
    dnnl_memory_destroy(b_mem);
    dnnl_memory_destroy(a_mem);
    dnnl_memory_desc_destroy(c_md);
    dnnl_memory_desc_destroy(b_md);
    dnnl_memory_desc_destroy(a_md);

    return elapsed / bench_iters;
}

// Benchmark result
typedef struct {
    const char *name;
    double time_ms;
    double gflops;
} bench_result_t;

// Run benchmark for a given size
static void run_benchmark(dnnl_context_t *ctx, int M, int N, int K,
                          int warmup_iters, int bench_iters) {
    printf("\n=== GEMM Benchmark: M=%d, N=%d, K=%d ===\n", M, N, K);
    printf("    (%.1f MB matrices)\n", (M*K + N*K + M*N) * 4.0 / 1e6);

    // Allocate
    float *A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float *B = (float*)aligned_alloc(64, N * K * sizeof(float));
    float *C_ours = (float*)aligned_alloc(64, M * N * sizeof(float));
    float *C_dnnl = (float*)aligned_alloc(64, M * N * sizeof(float));

    random_init(A, M * K);
    random_init(B, N * K);

    bench_result_t results[5];
    int n_results = 0;

    // 1. Our blocked serial
    memset(C_ours, 0, M * N * sizeof(float));
    for (int i = 0; i < warmup_iters; i++)
        gemm_blocked_serial(A, B, NULL, C_ours, M, N, K);
    double start = get_time_ms();
    for (int i = 0; i < bench_iters; i++)
        gemm_blocked_serial(A, B, NULL, C_ours, M, N, K);
    double t_blocked = (get_time_ms() - start) / bench_iters;
    results[n_results++] = (bench_result_t){
        "CK blocked_serial", t_blocked, compute_gflops(M, N, K, t_blocked)
    };

    // 2. Our AVX-512 parallel
    memset(C_ours, 0, M * N * sizeof(float));
    for (int i = 0; i < warmup_iters; i++)
        gemm_avx512_parallel(A, B, NULL, C_ours, M, N, K);
    start = get_time_ms();
    for (int i = 0; i < bench_iters; i++)
        gemm_avx512_parallel(A, B, NULL, C_ours, M, N, K);
    double t_avx = (get_time_ms() - start) / bench_iters;
    results[n_results++] = (bench_result_t){
        "CK avx512_parallel", t_avx, compute_gflops(M, N, K, t_avx)
    };

    // 3. Our fine-grained parallel
    memset(C_ours, 0, M * N * sizeof(float));
    for (int i = 0; i < warmup_iters; i++)
        gemm_fine_grained_parallel(A, B, NULL, C_ours, M, N, K);
    start = get_time_ms();
    for (int i = 0; i < bench_iters; i++)
        gemm_fine_grained_parallel(A, B, NULL, C_ours, M, N, K);
    double t_fine = (get_time_ms() - start) / bench_iters;
    results[n_results++] = (bench_result_t){
        "CK fine_grained", t_fine, compute_gflops(M, N, K, t_fine)
    };

    // 4. oneDNN
    memset(C_dnnl, 0, M * N * sizeof(float));
    double t_dnnl = run_dnnl_gemm(ctx, A, B, C_dnnl, M, N, K, warmup_iters, bench_iters);
    results[n_results++] = (bench_result_t){
        "oneDNN matmul", t_dnnl, compute_gflops(M, N, K, t_dnnl)
    };

    // Verify correctness (compare our best vs oneDNN)
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(C_ours[i] - C_dnnl[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Print results table
    printf("\n%-22s %10s %10s %10s\n", "Kernel", "Time(ms)", "GFLOPS", "vs oneDNN");
    printf("%-22s %10s %10s %10s\n", "------", "--------", "------", "---------");
    for (int i = 0; i < n_results; i++) {
        double ratio = results[n_results-1].gflops / results[i].gflops;
        printf("%-22s %10.2f %10.1f %9.2fx\n",
               results[i].name, results[i].time_ms, results[i].gflops, ratio);
    }
    printf("\nMax difference vs oneDNN: %.2e\n", max_diff);

    free(A);
    free(B);
    free(C_ours);
    free(C_dnnl);
}

int main(int argc, char **argv) {
    printf("=================================================\n");
    printf("  GEMM Benchmark: C-Kernel-Engine vs oneDNN\n");
    printf("=================================================\n");
    printf("OMP threads: %d\n", omp_get_max_threads());

    dnnl_context_t ctx;
    init_dnnl(&ctx);

    // Get oneDNN version
    const dnnl_version_t *ver = dnnl_version();
    printf("oneDNN version: %d.%d.%d\n", ver->major, ver->minor, ver->patch);

    int warmup = 3;
    int iters = 10;

    // Test various sizes relevant for LLMs
    // Small (embedding lookup, small MLP)
    run_benchmark(&ctx, 1, 4096, 4096, warmup, iters);      // GEMV
    run_benchmark(&ctx, 32, 4096, 4096, warmup, iters);     // Small batch

    // Medium (typical hidden dimension)
    run_benchmark(&ctx, 128, 4096, 4096, warmup, iters);
    run_benchmark(&ctx, 256, 4096, 11008, warmup, iters);   // LLaMA MLP

    // Large (big batch, attention)
    run_benchmark(&ctx, 512, 4096, 4096, warmup, iters);
    run_benchmark(&ctx, 1024, 4096, 4096, warmup, iters);

    cleanup_dnnl(&ctx);

    printf("\n=================================================\n");
    printf("  Analysis\n");
    printf("=================================================\n");
    printf("\nKey optimizations in oneDNN we could adopt:\n");
    printf("  1. JIT code generation for specific sizes\n");
    printf("  2. Better cache blocking (L1/L2/L3 aware)\n");
    printf("  3. Microkernel approach (small fixed-size GEMMs)\n");
    printf("  4. Pack matrices for better memory access\n");
    printf("  5. AMX instructions on supported CPUs\n");

    return 0;
}
