#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

void softmax_cross_entropy_loss_bf16(const uint16_t *logits,
                                     const int32_t *targets,
                                     int tokens,
                                     int vocab_size,
                                     uint16_t *d_logits,
                                     float *loss_out)
{
    if (!logits || !targets || !d_logits || tokens <= 0 || vocab_size <= 0) {
        if (loss_out) {
            *loss_out = 0.0f;
        }
        return;
    }

    const size_t count = (size_t)tokens * (size_t)vocab_size;
    float *logits_f = (float *)malloc(count * sizeof(float));
    float *d_logits_f = (float *)malloc(count * sizeof(float));
    if (!logits_f || !d_logits_f) {
        free(logits_f);
        free(d_logits_f);
        if (loss_out) {
            *loss_out = 0.0f;
        }
        return;
    }

    bf16_tensor_to_float(logits, logits_f, count);
    softmax_cross_entropy_loss(logits_f, targets, tokens, vocab_size, d_logits_f, loss_out);
    float_tensor_to_bf16(d_logits_f, d_logits, count);

    free(logits_f);
    free(d_logits_f);
}

