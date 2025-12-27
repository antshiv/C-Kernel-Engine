#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <math.h>
#include <stdint.h>

// RMSNorm forward for BF16 inputs/outputs; gamma stays in float for precision.
void rmsnorm_forward_bf16(const uint16_t *input,
                          const float *gamma,
                          uint16_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *x_bf16 = input + (size_t)t * aligned;
        float *rstd_ptr = rstd_cache ? (rstd_cache + t) : NULL;
        uint16_t *out_bf16 = output + (size_t)t * aligned;

        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            sum_sq += (double)x * (double)x;
        }
        double mean_sq = sum_sq / (double)D;
        double r = sqrt(mean_sq + (double)eps);
        float rstd = (float)(1.0 / r);
        if (rstd_ptr) {
            *rstd_ptr = rstd;
        }

        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float y = x_hat * gamma[d];
            out_bf16[d] = float_to_bf16(y);
        }

        for (int d = D; d < aligned; ++d) {
            out_bf16[d] = 0;
        }
    }
}

// RMSNorm backward for BF16 inputs/outputs; gradients accumulate in float.
void rmsnorm_backward_bf16(const uint16_t *d_output,
                           const uint16_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           uint16_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    if (!d_output || !input || !gamma || !rstd_cache || !d_input || !d_gamma) {
        return;
    }

    for (int d = 0; d < D; ++d) {
        d_gamma[d] = 0.0f;
    }

    for (int t = 0; t < T; ++t) {
        const uint16_t *x_bf16 = input + (size_t)t * aligned;
        const uint16_t *dY_bf16 = d_output + (size_t)t * aligned;
        uint16_t *dX_bf16 = d_input + (size_t)t * aligned;
        float rstd = rstd_cache[t];

        double sum_dY_g_xhat = 0.0;
        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            sum_dY_g_xhat += (double)dy * (double)gamma[d] * (double)x_hat;
        }
        float m = (float)(sum_dY_g_xhat / (double)D);

        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            float dx = rstd * (dy * gamma[d] - x_hat * m);
            dX_bf16[d] = float_to_bf16(dx);
            d_gamma[d] += dy * x_hat;
        }

        for (int d = D; d < aligned; ++d) {
            dX_bf16[d] = 0;
        }
    }
}
