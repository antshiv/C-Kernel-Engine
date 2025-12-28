#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

void causal_softmax_head_major_bf16(uint16_t *scores,
                                   int num_heads,
                                   int num_tokens,
                                   int aligned_context_window)
{
    if (!scores || num_heads <= 0 || num_tokens <= 0 || aligned_context_window <= 0) {
        return;
    }

    const size_t total = (size_t)num_heads *
                         (size_t)aligned_context_window *
                         (size_t)aligned_context_window;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp) {
        return;
    }

    bf16_tensor_to_float(scores, tmp, total);
    causal_softmax_head_major(tmp, num_heads, num_tokens, aligned_context_window);
    float_tensor_to_bf16(tmp, scores, total);

    free(tmp);
}

void backward_causal_softmax_head_major_bf16(uint16_t *d_scores,
                                            const uint16_t *weights,
                                            int num_heads,
                                            int num_tokens,
                                            int aligned_context_window)
{
    if (!d_scores || !weights || num_heads <= 0 || num_tokens <= 0 || aligned_context_window <= 0) {
        return;
    }

    const size_t total = (size_t)num_heads *
                         (size_t)aligned_context_window *
                         (size_t)aligned_context_window;
    float *d_scores_f = (float *)malloc(total * sizeof(float));
    float *weights_f = (float *)malloc(total * sizeof(float));
    if (!d_scores_f || !weights_f) {
        free(d_scores_f);
        free(weights_f);
        return;
    }

    bf16_tensor_to_float(d_scores, d_scores_f, total);
    bf16_tensor_to_float(weights, weights_f, total);
    backward_causal_softmax_head_major(d_scores_f, weights_f, num_heads, num_tokens, aligned_context_window);
    float_tensor_to_bf16(d_scores_f, d_scores, total);

    free(d_scores_f);
    free(weights_f);
}

