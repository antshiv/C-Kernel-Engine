#include <math.h>

// Causal softmax on head-major attention scores, copied and generalized
// from C-Transformer's apply_causal_softmax_head_major.
//
// scores layout: [head][query_token][key_token] with aligned_context_window stride:
//   index = h * aligned_context_window * aligned_context_window
//         + i * aligned_context_window
//         + j
void causal_softmax_head_major(float *scores,
                               int num_heads,
                               int num_tokens,
                               int aligned_context_window)
{
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            int base = h * aligned_context_window * aligned_context_window
                     + i * aligned_context_window;

            float max_val = scores[base + 0];
            for (int j = 1; j <= i; ++j) {
                float v = scores[base + j];
                if (v > max_val) {
                    max_val = v;
                }
            }

            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float v = scores[base + j];
                float e = expf(v - max_val);
                scores[base + j] = e;
                sum += e;
            }

            float inv_sum = 1.0f / sum;
            for (int j = 0; j <= i; ++j) {
                scores[base + j] *= inv_sum;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[base + j] = 0.0f;
            }
        }
    }
}

// Backward pass for causal softmax on head-major scores, adapted from
// C-Transformer's backward_causal_softmax. Operates in-place on d_scores,
// using the cached forward softmax output `weights`.
void backward_causal_softmax_head_major(float *d_scores,
                                        const float *weights,
                                        int num_heads,
                                        int num_tokens,
                                        int aligned_context_window)
{
    int H = num_heads;
    int T = num_tokens;

    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < T; ++i) {
            int base = h * aligned_context_window * aligned_context_window
                     + i * aligned_context_window;

            float dot_product = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float w = weights[base + j];
                float dw = d_scores[base + j];
                dot_product += w * dw;
            }

            for (int j = 0; j <= i; ++j) {
                float w = weights[base + j];
                float dw = d_scores[base + j];
                d_scores[base + j] = w * (dw - dot_product);
            }

            for (int j = i + 1; j < T; ++j) {
                d_scores[base + j] = 0.0f;
            }
        }
    }
}

