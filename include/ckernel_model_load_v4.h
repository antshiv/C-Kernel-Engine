#ifndef CKERNEL_MODEL_LOAD_V4_H
#define CKERNEL_MODEL_LOAD_V4_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load BUMPWGT4 weights into a v4 model buffer using a manifest map.
 *
 * The manifest format is line-based:
 *   name|dtype|file_offset|size|runtime_offset
 * Offsets/sizes accept decimal or hex (0x...).
 *
 * @param base           Base pointer for the generated model buffer.
 * @param weights_path   Path to BUMPWGT4 weights file.
 * @param manifest_path  Path to weights_manifest.map emitted by build_ir_v4.py.
 * @return 0 on success, non-zero on error.
 */
int ck_load_weights_manifest_v4(void *base,
                                const char *weights_path,
                                const char *manifest_path);

#ifdef __cplusplus
}
#endif

#endif /* CKERNEL_MODEL_LOAD_V4_H */
