/* C interface to SIMDSort.jl (lib/src/SIMDSortC.jl), built with JuliaC.
 *
 * simdsort_<t> sorts data in place; simdsortperm_<t> writes the stable sort
 * permutation (0-based) to perm. Order: isless (NaNs last, -0.0 before 0.0),
 * reversed if rev != 0. nthreads bounds the tasks running at once; the
 * runtime's threads are set when the library is loaded (JULIA_NUM_THREADS).
 * All functions return 0. */
#ifndef SIMDSORT_H
#define SIMDSORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SIMDSORT_DECL(t, T) \
    int simdsort_##t(T *data, size_t n, int rev, int nthreads); \
    int simdsortperm_##t(const T *data, size_t n, int64_t *perm, int rev, int nthreads);
SIMDSORT_DECL(i32, int32_t)
SIMDSORT_DECL(i64, int64_t)
SIMDSORT_DECL(u32, uint32_t)
SIMDSORT_DECL(u64, uint64_t)
SIMDSORT_DECL(f32, float)
SIMDSORT_DECL(f64, double)
#undef SIMDSORT_DECL
/* The number of threads of the library's Julia runtime. */
int simdsort_nthreads(void);
#ifdef __cplusplus
}
#endif
#endif
