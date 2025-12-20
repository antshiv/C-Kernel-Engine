#define _GNU_SOURCE
#include "ckernel_alloc.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

/* 2MB huge page size on Linux. */
#ifndef HUGE_PAGE_SIZE
#define HUGE_PAGE_SIZE (2UL * 1024UL * 1024UL)
#endif

static size_t align_up_bytes(size_t n, size_t align)
{
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}

void *ck_huge_alloc(size_t bytes)
{
    size_t len = align_up_bytes(bytes, HUGE_PAGE_SIZE);

    /* First, try explicit huge pages via mmap + MAP_HUGETLB. */
    void *p = mmap(NULL, len,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                   -1, 0);
    if (p != MAP_FAILED) {
        return p;
    }

    /* Fallback: aligned_alloc with transparent hugepage hint. */
    void *q = aligned_alloc(HUGE_PAGE_SIZE, len);
    if (!q) {
        fprintf(stderr, "ck_huge_alloc: aligned_alloc failed for %zu bytes: %s\n",
                len, strerror(errno));
        return NULL;
    }

    /* Best-effort hint; ignore errors. */
    (void)madvise(q, len, MADV_HUGEPAGE);
    return q;
}

void ck_huge_free(void *ptr, size_t bytes)
{
    if (!ptr || bytes == 0) {
        return;
    }

    size_t len = align_up_bytes(bytes, HUGE_PAGE_SIZE);

    /* Try to detect whether this looks like an mmap'ed region. If munmap
     * fails, fall back to free().
     */
    if (munmap(ptr, len) != 0) {
        free(ptr);
    }
}

