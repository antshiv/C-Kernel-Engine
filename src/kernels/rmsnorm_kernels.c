#include <math.h>
#include <stddef.h>

// RMSNorm forward:
// For each token t:
//   r = sqrt( (1/D) * sum_i x_i^2 + eps )
//   rstd = 1 / r
//   x_hat_i = x_i * rstd
//   y_i = gamma_i * x_hat_i
//
// We cache rstd per token for use in backward.
void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
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
        const float *x = input + (size_t)t * aligned;
        float *y = output + (size_t)t * aligned;

        // Compute mean square
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double v = (double)x[d];
            sum_sq += v * v;
        }
        double mean_sq = sum_sq / (double)D;
        double r = sqrt(mean_sq + (double)eps);
        float rstd = (float)(1.0 / r);
        rstd_cache[t] = rstd;

        // Apply normalization and scale
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            y[d] = x_hat * gamma[d];
        }

        // Zero padding (if any)
        for (int d = D; d < aligned; ++d) {
            y[d] = 0.0f;
        }
    }
}

// RMSNorm backward:
// Given dY, X, gamma, and cached rstd per token, compute:
//   dX and dGamma.
//
// From derivation:
//   x_hat_i = x_i * rstd_t
//   m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
//   dX_i = rstd_t * (dY_i * gamma_i - x_hat_i * m)
//   dGamma_i = sum_t (dY_i * x_hat_i)
//
// We do not include a beta parameter for RMSNorm here.
void rmsnorm_backward(const float *d_output,
                      const float *input,
                      const float *gamma,
                      const float *rstd_cache,
                      float *d_input,
                      float *d_gamma,
                      int tokens,
                      int d_model,
                      int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    // Zero parameter gradients
    for (int d = 0; d < D; ++d) {
        d_gamma[d] = 0.0f;
    }

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * aligned;
        const float *dY = d_output + (size_t)t * aligned;
        float *dX = d_input + (size_t)t * aligned;

        float rstd = rstd_cache[t];

        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        double sum_dY_g_xhat = 0.0;
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += (double)dY[d] * (double)gamma[d] * (double)x_hat;
        }
        float m = (float)(sum_dY_g_xhat / (double)D);

        // Compute dX and accumulate dGamma
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }

        // Zero padding gradients (if any)
        for (int d = D; d < aligned; ++d) {
            dX[d] = 0.0f;
        }
    }
}

