/**
 * Standalone GEMM Microkernel Benchmark for perf profiling
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp benchmarks/perf_gemm_micro.c \
 *       -Iinclude -Lbuild -lckernel_engine -lm -o build/perf_gemm_micro
 *
 * Profile with perf:
 *   LD_LIBRARY_PATH=build perf record -g ./build/perf_gemm_micro
 *   perf report
 *
 * Generate flamegraph:
 *   perf script | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > gemm_flame.svg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Our kernel declaration
extern void gemm_microkernel(const float *A, const float *B, float *C,
                             int M, int N, int K, int B_transposed);
extern void print_cpu_info(void);

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void random_init(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

int main(int argc, char **argv) {
    // Default sizes (can be overridden via command line)
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int iters = 50;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        iters = atoi(argv[4]);
    }

    printf("=== GEMM Microkernel Profiling Benchmark ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Iterations: %d\n", iters);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("\n");

    // Print CPU info
    print_cpu_info();

    // Allocate matrices (aligned for SIMD)
    float *A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float *C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    // Initialize
    srand(42);
    random_init(A, M * K);
    random_init(B, K * N);
    memset(C, 0, (size_t)M * N * sizeof(float));

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 5; i++) {
        gemm_microkernel(A, B, C, M, N, K, 0);
    }

    // Benchmark
    printf("Running benchmark (%d iterations)...\n", iters);

    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        gemm_microkernel(A, B, C, M, N, K, 0);
    }
    double elapsed = get_time_ms() - start;

    double avg_time_ms = elapsed / iters;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_time_ms * 1e6);

    printf("\n=== Results ===\n");
    printf("Total time:   %.2f ms\n", elapsed);
    printf("Avg per iter: %.2f ms\n", avg_time_ms);
    printf("Performance:  %.2f GFLOP/s\n", gflops);

    // Quick accuracy check (just to make sure it's working)
    printf("\nC[0,0] = %f (sanity check)\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
