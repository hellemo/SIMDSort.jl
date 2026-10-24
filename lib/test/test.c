/* Test and time libsimdsort:  lib/test/run.sh  (builds against lib/build). */
#include "simdsort.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t s = 88172645463325252ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static int cmp_i64(const void *a, const void *b) { int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return (x > y) - (x < y); }
static int cmp_f64(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    size_t n = argc > 1 ? strtoull(argv[1], 0, 10) : (1u << 24);
    int nt = argc > 2 ? atoi(argv[2]) : 1;
    int fails = 0;
    int64_t *a = malloc(n * 8), *b = malloc(n * 8), *perm = malloc(n * 8);
    double *x = malloc(n * 8), *y = malloc(n * 8);
    for (size_t i = 0; i < n; i++) { a[i] = (int64_t)rnd(); x[i] = (double)(int64_t)rnd() / 1e9; }

    double t0 = now(); simdsort_i64(a, 1000, 0, 1); double tfirst = now() - t0;   /* first call */
    for (size_t i = 0; i < n; i++) a[i] = (int64_t)rnd();
    memcpy(b, a, n * 8);
    t0 = now(); simdsort_i64(a, n, 0, nt); double ts = now() - t0;
    t0 = now(); qsort(b, n, 8, cmp_i64); double tq = now() - t0;
    if (memcmp(a, b, n * 8)) { printf("FAIL i64 sort\n"); fails++; }
    printf("runtime threads: %d\n", simdsort_nthreads());
    printf("int64,  n=%zu, %d threads: simdsort %.4f s, qsort %.4f s (%.1fx)\n", n, nt, ts, tq, tq / ts);

    memcpy(y, x, n * 8);
    t0 = now(); simdsortperm_f64(x, n, perm, 0, nt); double tp = now() - t0;
    for (size_t i = 1; i < n; i++)
        if (x[perm[i - 1]] > x[perm[i]] || (x[perm[i - 1]] == x[perm[i]] && perm[i - 1] > perm[i])) { printf("FAIL f64 sortperm at %zu\n", i); fails++; break; }
    t0 = now(); simdsort_f64(x, n, 1, nt); double tf = now() - t0;
    qsort(y, n, 8, cmp_f64);
    for (size_t i = 0; i < n; i++) if (x[i] != y[n - 1 - i]) { printf("FAIL f64 rev sort at %zu\n", i); fails++; break; }
    printf("float64 sortperm %.4f s, sort (rev) %.4f s; first call %.4f s\n", tp, tf, tfirst);

    int32_t c[8] = {5, -1, 3, 3, -7, 0, 2, 1}; int64_t pc[8];
    simdsortperm_i32(c, 8, pc, 0, 1);
    printf("int32 sortperm of {5,-1,3,3,-7,0,2,1}: ");
    for (int i = 0; i < 8; i++) printf("%lld ", (long long)pc[i]);
    printf("\n%s\n", fails ? "FAILED" : "all checks passed");
    return fails != 0;
}
