#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

static float *alloc_float_buffer(size_t count)
{
    return (float *)malloc(count * sizeof(float));
}

static float *bf16_to_float_alloc(const uint16_t *src, size_t count)
{
    float *dst = alloc_float_buffer(count);
    if (!dst) {
        return NULL;
    }
    bf16_tensor_to_float(src, dst, count);
    return dst;
}

void mlp_token_parallel_bf16(const uint16_t *input,
                             const uint16_t *W_fc1,
                             const uint16_t *b_fc1,
                             const uint16_t *W_fc2,
                             const uint16_t *b_fc2,
                             float *fc1_output,
                             float *output,
                             int T,
                             int aligned_dim,
                             int num_threads)
{
    if (!input || !W_fc1 || !b_fc1 || !W_fc2 || !b_fc2 || !fc1_output || !output) {
        return;
    }

    const int D = aligned_dim;
    const int fourD = 4 * D;

    const size_t input_elems = (size_t)T * (size_t)aligned_dim;
    const size_t w1_elems = (size_t)fourD * (size_t)aligned_dim;
    const size_t b1_elems = (size_t)fourD;
    const size_t w2_elems = (size_t)aligned_dim * (size_t)fourD;
    const size_t b2_elems = (size_t)aligned_dim;

    float *input_f = bf16_to_float_alloc(input, input_elems);
    float *W1_f = bf16_to_float_alloc(W_fc1, w1_elems);
    float *b1_f = bf16_to_float_alloc(b_fc1, b1_elems);
    float *W2_f = bf16_to_float_alloc(W_fc2, w2_elems);
    float *b2_f = bf16_to_float_alloc(b_fc2, b2_elems);

    if (!input_f || !W1_f || !b1_f || !W2_f || !b2_f) {
        free(input_f);
        free(W1_f);
        free(b1_f);
        free(W2_f);
        free(b2_f);
        return;
    }

    mlp_token_parallel(input_f, W1_f, b1_f, W2_f, b2_f,
                       fc1_output, output, T, aligned_dim, num_threads);

    free(input_f);
    free(W1_f);
    free(b1_f);
    free(W2_f);
    free(b2_f);
}

