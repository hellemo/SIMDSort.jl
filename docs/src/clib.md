# C library

SIMDSort can be built into a shared library with C entry points using
[JuliaC](https://github.com/JuliaLang/JuliaC.jl) (Julia 1.12+), with
`--trim=safe`, which removes code not reachable from the entry points and
fails the build on any call it cannot resolve at compile time. The files are
in `lib/` of the repository:

| File | |
|---|---|
| `lib/src/SIMDSortC.jl` | the `Base.@ccallable` entry points |
| `lib/include/simdsort.h` | the C header |
| `lib/build.sh` | the build (`juliac --output-lib --trim=safe --bundle`) |
| `lib/test/test.c`, `run.sh` | a C program that checks and times the library |

## Interface

```c
#include "simdsort.h"
int simdsort_i64(int64_t *data, size_t n, int rev, int nthreads);          /* in place */
int simdsortperm_f64(const double *data, size_t n, int64_t *perm, int rev,
                     int nthreads);                                          /* 0-based, stable */
int simdsort_nthreads(void);                                                 /* runtime threads */
```

with the suffixes `i32`, `i64`, `u32`, `u64`, `f32`, `f64`. The order is
`isless` (NaNs last, `-0.0` before `0.0`); `nthreads` bounds the tasks
running at once.

## Building and running

```sh
lib/build.sh                  # trimmed (or: lib/build.sh no)
lib/test/run.sh 16777216 8    # n, nthreads
```

The bundle in `lib/build/` holds `lib/libsimdsort.so` and the Julia runtime
it needs, with relative rpaths.

## Results of the feasibility study

On the AVX2 laptop, Julia 1.13.2, JuliaC 0.3.10:

- **Trimming works.** The first build failed trim verification with two
  errors: `@sync` waits on a `Channel{Any}`, which is a dynamic call. The task
  helpers now collect their tasks in a `Vector{Task}` and wait on each, which
  is also simpler; after that `--trim=safe` verifies without errors or
  warnings.
- **Size.** `libsimdsort.so` is 18.7 MB. The bundle is 113 MB, most of it
  runtime libraries the sort does not use but the runtime loads with the
  standard library: OpenBLAS (35 MB), libstdc++ (21 MB), libjulia-internal
  (15 MB), libgfortran (11 MB).
- **Threads.** The runtime's thread count is fixed when the library is built.
  JuliaC builds libraries with one thread by default, and
  `JULIA_NUM_THREADS` has no effect. `lib/build.sh` passes
  `--jl-option threads=N` (`SIMDSORT_THREADS`, default: the CPU count), and
  `simdsort_nthreads()` reports it.
- **CPU.** The package chooses its vector code paths when it is compiled, so
  the build uses `JULIA_CPU_TARGET=native` instead of JuliaC's generic
  multi-target default. The library is for the CPU it is built on.
- **Speed.** Times for 2^24 keys (Julia times without the copy):

| | C library, 1 thread | Julia, 1 thread | C library, 8 threads | Julia, 8 threads |
|---|---:|---:|---:|---:|
| Int64 sort | 0.133 s | 0.122 s | 0.047 s | 0.038 s |
| Float64 sortperm | 0.615 s | 0.555 s | 0.223 s | 0.147 s |

  The library is 9–20% slower than the same calls in Julia, and Float64
  `sortperm` at 8 threads about 50% slower. Against C's `qsort` it is 16×
  faster on one thread and 45× faster on eight (Int64). The first call takes
  about 15 ms.
- **Not covered:** strings, wide types and the extensions (the C API is for
  the six primitive types).
