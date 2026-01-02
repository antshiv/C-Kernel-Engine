/**
 * @file optimizer_kernels.c
 * @brief Optimizer kernels for training (AdamW, SGD)
 *
 * AdamW Algorithm:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_{t-1})
 *
 * Note: AdamW applies weight decay directly to weights, not to gradients.
 * This is different from L2 regularization (Adam with L2 adds decay to gradient).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/**
 * @brief AdamW optimizer update (fp32 version)
 *
 * Updates weights in-place using the AdamW algorithm.
 * Momentum (m) and variance (v) are stored in fp32 for numerical stability.
 *
 * @param grad       Gradient tensor (fp32) [numel]
 * @param weight     Weight tensor to update (fp32, in-place) [numel]
 * @param m          First moment (momentum) buffer (fp32, in-place) [numel]
 * @param v          Second moment (variance) buffer (fp32, in-place) [numel]
 * @param numel      Number of elements
 * @param lr         Learning rate
 * @param beta1      Exponential decay rate for first moment (typically 0.9)
 * @param beta2      Exponential decay rate for second moment (typically 0.999)
 * @param eps        Small constant for numerical stability (typically 1e-8)
 * @param weight_decay Weight decay coefficient (typically 0.01)
 * @param step       Current step number (1-indexed for bias correction)
 */
void adamw_update_f32(
    const float *grad,
    float *weight,
    float *m,
    float *v,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step)
{
    if (!grad || !weight || !m || !v || numel == 0) {
        return;
    }

    // Bias correction terms
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    // Precompute constants
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

#if defined(__AVX512F__)
    // Vectorized path: process 16 floats at a time
    __m512 v_beta1 = _mm512_set1_ps(beta1);
    __m512 v_beta2 = _mm512_set1_ps(beta2);
    __m512 v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    __m512 v_one_minus_beta2 = _mm512_set1_ps(one_minus_beta2);
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);
    __m512 v_bc1_inv = _mm512_set1_ps(1.0f / bias_correction1);
    __m512 v_bc2_inv = _mm512_set1_ps(1.0f / bias_correction2);

    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        // Load gradient, weight, m, v
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 w = _mm512_loadu_ps(&weight[i]);
        __m512 m_val = _mm512_loadu_ps(&m[i]);
        __m512 v_val = _mm512_loadu_ps(&v[i]);

        // Update m: m = beta1 * m + (1 - beta1) * g
        m_val = _mm512_fmadd_ps(v_beta1, m_val, _mm512_mul_ps(v_one_minus_beta1, g));

        // Update v: v = beta2 * v + (1 - beta2) * g^2
        __m512 g_sq = _mm512_mul_ps(g, g);
        v_val = _mm512_fmadd_ps(v_beta2, v_val, _mm512_mul_ps(v_one_minus_beta2, g_sq));

        // Bias-corrected estimates
        __m512 m_hat = _mm512_mul_ps(m_val, v_bc1_inv);
        __m512 v_hat = _mm512_mul_ps(v_val, v_bc2_inv);

        // Update weight: w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
        __m512 update = _mm512_div_ps(m_hat, denom);
        update = _mm512_fmadd_ps(v_weight_decay, w, update);  // + weight_decay * w
        w = _mm512_fnmadd_ps(v_lr, update, w);  // w - lr * update

        // Store updated values
        _mm512_storeu_ps(&weight[i], w);
        _mm512_storeu_ps(&m[i], m_val);
        _mm512_storeu_ps(&v[i], v_val);
    }

    // Scalar tail
    for (; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];

        // Update m and v
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;

        // Bias-corrected estimates
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        // Update weight
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }
#else
    // Scalar path
    for (size_t i = 0; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];

        // Update m and v
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;

        // Bias-corrected estimates
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        // Update weight
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }
#endif
}


/**
 * @brief SGD with momentum optimizer update (fp32 version)
 *
 * v_t = momentum * v_{t-1} + g_t
 * w_t = w_{t-1} - lr * (v_t + weight_decay * w_{t-1})
 *
 * @param grad       Gradient tensor (fp32) [numel]
 * @param weight     Weight tensor to update (fp32, in-place) [numel]
 * @param velocity   Velocity buffer (fp32, in-place) [numel]
 * @param numel      Number of elements
 * @param lr         Learning rate
 * @param momentum   Momentum coefficient (typically 0.9)
 * @param weight_decay Weight decay coefficient
 */
void sgd_momentum_update_f32(
    const float *grad,
    float *weight,
    float *velocity,
    size_t numel,
    float lr,
    float momentum,
    float weight_decay)
{
    if (!grad || !weight || !velocity || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_momentum = _mm512_set1_ps(momentum);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);

    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 w = _mm512_loadu_ps(&weight[i]);
        __m512 vel = _mm512_loadu_ps(&velocity[i]);

        // v = momentum * v + g
        vel = _mm512_fmadd_ps(v_momentum, vel, g);

        // w = w - lr * (v + weight_decay * w)
        __m512 update = _mm512_fmadd_ps(v_weight_decay, w, vel);
        w = _mm512_fnmadd_ps(v_lr, update, w);

        _mm512_storeu_ps(&weight[i], w);
        _mm512_storeu_ps(&velocity[i], vel);
    }

    for (; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }
#endif
}


/**
 * @brief Zero out gradient buffer (fp32)
 *
 * @param grad  Gradient tensor to zero [numel]
 * @param numel Number of elements
 */
void zero_gradients_f32(float *grad, size_t numel)
{
    if (!grad || numel == 0) {
        return;
    }
    memset(grad, 0, numel * sizeof(float));
}


/**
 * @brief Accumulate gradients: dst += src (fp32)
 *
 * Used for gradient accumulation across micro-batches.
 *
 * @param dst   Destination gradient buffer (in-place) [numel]
 * @param src   Source gradient buffer [numel]
 * @param numel Number of elements
 */
void gradient_accumulate_f32(float *dst, const float *src, size_t numel)
{
    if (!dst || !src || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 d = _mm512_loadu_ps(&dst[i]);
        __m512 s = _mm512_loadu_ps(&src[i]);
        _mm512_storeu_ps(&dst[i], _mm512_add_ps(d, s));
    }
    for (; i < numel; ++i) {
        dst[i] += src[i];
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        dst[i] += src[i];
    }
#endif
}


/**
 * @brief Scale gradients by a constant: grad *= scale (fp32)
 *
 * Used for averaging gradients after accumulation: grad /= batch_size
 *
 * @param grad  Gradient tensor to scale (in-place) [numel]
 * @param numel Number of elements
 * @param scale Scale factor (typically 1.0 / batch_size)
 */
void gradient_scale_f32(float *grad, size_t numel, float scale)
{
    if (!grad || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    __m512 v_scale = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        _mm512_storeu_ps(&grad[i], _mm512_mul_ps(g, v_scale));
    }
    for (; i < numel; ++i) {
        grad[i] *= scale;
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        grad[i] *= scale;
    }
#endif
}


/**
 * @brief Clip gradient norm (fp32)
 *
 * If ||grad||_2 > max_norm, scale grad so that ||grad||_2 = max_norm
 *
 * @param grad     Gradient tensor to clip (in-place) [numel]
 * @param numel    Number of elements
 * @param max_norm Maximum allowed L2 norm
 * @return         The original L2 norm before clipping
 */
float gradient_clip_norm_f32(float *grad, size_t numel, float max_norm)
{
    if (!grad || numel == 0 || max_norm <= 0.0f) {
        return 0.0f;
    }

    // Compute L2 norm
    double sum_sq = 0.0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        acc = _mm512_fmadd_ps(g, g, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
    for (; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#endif

    float norm = sqrtf((float)sum_sq);

    // Clip if necessary
    if (norm > max_norm) {
        float scale = max_norm / norm;
        gradient_scale_f32(grad, numel, scale);
    }

    return norm;
}
