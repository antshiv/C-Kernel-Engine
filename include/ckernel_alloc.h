#ifndef CKERNEL_ALLOC_H
#define CKERNEL_ALLOC_H

#include <stddef.h>

/**
 * Allocate a large, contiguous memory region for model weights/activations.
 *
 * Implementation strategy:
 *  - Try to allocate 2MB-aligned memory backed by huge pages where possible.
 *  - Fall back to aligned_alloc + madvise(MADV_HUGEPAGE) when explicit
 *    huge pages are not available.
 *
 * Returns NULL on failure.
 */
void *ck_huge_alloc(size_t bytes);

/**
 * Free memory allocated by ck_huge_alloc.
 *
 * The bytes parameter should be the same size passed to ck_huge_alloc.
 */
void ck_huge_free(void *ptr, size_t bytes);

#endif /* CKERNEL_ALLOC_H */

