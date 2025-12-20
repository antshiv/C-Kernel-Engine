#include "ckernel_model.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_floats(FILE *f, float *dst, size_t count)
{
    size_t n = fread(dst, sizeof(float), count, f);
    if (n != count) {
        if (ferror(f)) {
            fprintf(stderr, "ck_model_load_weights_flat: fread error: %s\n",
                    strerror(errno));
        } else {
            fprintf(stderr, "ck_model_load_weights_flat: unexpected EOF (wanted %zu floats, got %zu)\n",
                    count, n);
        }
        return -1;
    }
    return 0;
}

int ck_model_load_weights_flat(TransformerModel *m, const char *path)
{
    if (!m || !m->memory_base || !path) {
        fprintf(stderr, "ck_model_load_weights_flat: invalid arguments\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ck_model_load_weights_flat: failed to open %s: %s\n",
                path, strerror(errno));
        return -1;
    }

    const int L   = m->cfg.num_layers;
    const int H   = m->cfg.hidden_size;
    const int Hff = m->cfg.intermediate_size;
    const int V   = m->cfg.vocab_size;
    const int T   = m->cfg.context_window;

    if (L <= 0 || H <= 0 || Hff <= 0 || V <= 0 || T <= 0) {
        fprintf(stderr, "ck_model_load_weights_flat: invalid model cfg (L=%d, H=%d, Hff=%d, V=%d, T=%d)\n",
                L, H, Hff, V, T);
        fclose(f);
        return -1;
    }

    float *base = m->memory_base;

    /* 1) Token embeddings [V × H] */
    if (read_floats(f, base + m->token_emb_offset, (size_t)V * (size_t)H) != 0) {
        fclose(f);
        return -1;
    }

    /* 2) Positional embeddings [T × H] */
    if (read_floats(f, base + m->pos_emb_offset, (size_t)T * (size_t)H) != 0) {
        fclose(f);
        return -1;
    }

    /* 3) Per-layer weights */
    for (int layer = 0; layer < L; ++layer) {
        CKLayerLayout *Lyt = &m->layers[layer];

        /* LN1 gamma [H] */
        if (read_floats(f, base + Lyt->ln1_weight_offset, (size_t)H) != 0) {
            fclose(f);
            return -1;
        }

        /* LN1 beta [H] */
        if (read_floats(f, base + Lyt->ln1_bias_offset, (size_t)H) != 0) {
            fclose(f);
            return -1;
        }

        /* QKV weight [H × 3H] */
        if (read_floats(f, base + Lyt->qkv_weight_offset,
                        (size_t)H * (size_t)(3 * H)) != 0) {
            fclose(f);
            return -1;
        }

        /* QKV bias [3H] */
        if (read_floats(f, base + Lyt->qkv_bias_offset, (size_t)(3 * H)) != 0) {
            fclose(f);
            return -1;
        }

        /* Attention proj weight [H × H] */
        if (read_floats(f, base + Lyt->attn_proj_weight_offset,
                        (size_t)H * (size_t)H) != 0) {
            fclose(f);
            return -1;
        }

        /* Attention proj bias [H] */
        if (read_floats(f, base + Lyt->attn_proj_bias_offset, (size_t)H) != 0) {
            fclose(f);
            return -1;
        }

        /* FC1 weight [H × Hff] */
        if (read_floats(f, base + Lyt->fc1_weight_offset,
                        (size_t)H * (size_t)Hff) != 0) {
            fclose(f);
            return -1;
        }

        /* FC1 bias [Hff] */
        if (read_floats(f, base + Lyt->fc1_bias_offset, (size_t)Hff) != 0) {
            fclose(f);
            return -1;
        }

        /* FC2 weight [Hff × H] */
        if (read_floats(f, base + Lyt->fc2_weight_offset,
                        (size_t)Hff * (size_t)H) != 0) {
            fclose(f);
            return -1;
        }

        /* FC2 bias [H] */
        if (read_floats(f, base + Lyt->fc2_bias_offset, (size_t)H) != 0) {
            fclose(f);
            return -1;
        }
    }

    /* 4) Final LN gamma [H] */
    if (read_floats(f, base + m->final_ln_weight_offset, (size_t)H) != 0) {
        fclose(f);
        return -1;
    }

    /* 5) Final LN beta [H] */
    if (read_floats(f, base + m->final_ln_bias_offset, (size_t)H) != 0) {
        fclose(f);
        return -1;
    }

    /* 6) LM head weight [V × H] */
    if (read_floats(f, base + m->lm_head_weight_offset,
                    (size_t)V * (size_t)H) != 0) {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

