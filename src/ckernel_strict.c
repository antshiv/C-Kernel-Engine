#include "ckernel_engine.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static int ck_strict_parity = 0;

void ck_set_strict_parity(int enabled)
{
    ck_strict_parity = enabled ? 1 : 0;
#ifdef _OPENMP
    if (ck_strict_parity) {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }
#endif
}

int ck_strict_parity_enabled(void)
{
    return ck_strict_parity;
}
