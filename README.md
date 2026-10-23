# SIMDQuickSort.jl

An experimental vectorized quicksort in pure Julia for `Int32`, `Int64`, `UInt32`,
`UInt64`, `Float32` and `Float64` vectors. It is built on the SIMD.jl fork with
runtime permutes and compress (`vpartition`, `vpermute`), branch
[`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress).

```julia
using SIMDQuickSort
v = rand(Int32, 10^8)
simdsort!(v)                                  # one thread
simdsort!(v; ntasks = Threads.nthreads())     # task parallel
simdsort(v; rev = true)                       # sorted copy, reverse order
```

The order is that of `isless`: NaNs last, `-0.0` before `0.0`. Other element types
and other array types fall back to `sort!`.

## How it works

The algorithm follows the published descriptions of vectorized quicksort. It was
written from the papers, without using code from Google Highway (vqsort) or Intel's
x86-simd-sort. [docs/DESIGN.md](docs/DESIGN.md) describes each part, with
citations, the measurements behind the parameter choices, and where the
implementation deviates from the papers.

- **Keys:**
  - integers are mapped in place to signed integers (AVX2 has no unsigned
    compares), and the map is undone after sorting;
  - floats are sorted natively, with NaNs moved out first and signed zeros
    restored by count afterwards;
  - small integer ranges are counted instead of sorted.
- **Partition:** in place and bidirectional (Bramas 2017; Wassenberg et al. 2022).
  Each vector is split with `vpartition`, which is a table lookup plus `vpermd` on
  AVX2, and written with two full-width stores. In cache this costs about 0.11 ns
  per 32-bit key. Ranges larger than the L2 cache use software prefetch.
- **Pivot:** median of lane-wise medians of nine randomly placed vectors.
  - When the sample shows many copies of the pivot, the block of equal keys is
    split off.
  - Ranges of equal keys are detected by a vector min/max scan.
- **Base case:** a vectorized sorting network for up to 256 32-bit or 128 64-bit
  keys: Batcher's network sorts the columns, then bitonic merges without
  transposes, and one final transpose.
- **Fallbacks and parallelism:**
  - beyond a depth of $2\log_2 n + 4$, `Base.sort!` takes over;
  - with `ntasks > 1`, large ranges are partitioned in parallel and the smaller
    side of each partition becomes a new task.

## Results

Seconds per sort, mean of 5 runs (best in bold). Measured on an Intel Core Ultra 7
268V (Lunar Lake: 4 P + 4 E cores, AVX2, 15 GB) with Julia 1.13.2 and 8 threads.
x86-simd-sort and vqsort are called through the C shim in ChapelSort.jl's
`bench/simd`. Run with `julia -t 8 --project=bench bench/bench.jl`.

| case | Base.sort! (1 thr) | ChapelSort.jl psort! (1 thr) | ChapelSort.jl psort! (8 thr) | SIMDQuickSort.jl (1 thr) | SIMDQuickSort.jl (8 thr) | x86-simd-sort (1 thr) | x86-simd-sort (8 thr) | Highway vqsort (1 thr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UInt64, 2^27 | 3.289 | 1.735 | **0.431** | 1.473 | 0.515 | 2.053 | 0.605 | 1.713 |
| Int64, 2^26 | 1.534 | 0.869 | **0.213** | 0.675 | 0.220 | 0.843 | 0.251 | 0.772 |
| Float64 (randn), 2^26 | 1.589 | 1.077 | 0.214 | 0.626 | 0.199 | 0.597 | **0.199** | 0.822 |
| UInt32, 2^26 | 0.553 | 0.546 | 0.146 | 0.362 | 0.129 | 0.328 | **0.116** | 0.320 |
| Int32, 2^26 | 0.533 | 0.507 | 0.112 | 0.304 | **0.097** | 0.346 | 0.100 | 0.313 |
| Float32 (randn), 2^26 | 0.771 | 0.548 | 0.122 | 0.359 | **0.102** | 0.348 | 0.119 | 0.345 |
| UInt64 in 0:1000, 2^26 | 0.081 | 0.084 | **0.029** | 0.107 | 0.083 | 0.374 | 0.148 | 0.284 |
| UInt64 sorted, 2^26 | 0.029 | 0.031 | **0.012** | 0.088 | 0.052 | 1.051 | 0.313 | 0.799 |
| UInt64 reversed, 2^26 | 0.059 | 0.058 | **0.022** | 0.102 | 0.079 | 1.030 | 0.301 | 0.769 |
| UInt64, 1000 random values, 2^26 | 1.707 | 1.039 | 0.183 | 0.324 | 0.167 | 0.356 | **0.148** | 0.293 |
| Int32, 2^20 | 0.006 | 0.009 | 0.004 | 0.004 | **0.002** | 0.004 | 0.003 | 0.004 |

- **One thread, random keys:**
  - fastest of all measured sorts for `Int32`, `Int64` and `UInt64`;
  - within 4% of x86-simd-sort and Highway vqsort for `Float32`;
  - 5% behind x86-simd-sort on `Float64`;
  - `UInt32` is 10–13% behind the C++ sorts, because it pays two key-map passes;
  - 1.5–2.5× faster than Base.
- **Eight threads:**
  - fastest for `Int32` and `Float32`;
  - tied with x86-simd-sort for `Float64`;
  - ChapelSort's radix sort stays ahead for 64-bit integers.
- **Few distinct values:**
  - small integer ranges use a counting sort (0.11 s at 2^26 keys);
  - ChapelSort's counting path is still faster on this input;
  - few distinct values spread over a wide range are handled by the quicksort
    with equal-key peeling.
- **Sorted and reversed inputs** are caught by a linear pre-check.

[docs/DESIGN.md §7](docs/DESIGN.md) has the profile breakdown, per-level costs,
the effect of each optimization and the ideas that were rejected.
`bench/profile.jl` reproduces the breakdown.

## Limitations

- Tuned and tested on one AVX2 machine only, with 256-bit vectors.
  - CPUs without AVX2 (and ARM) get SIMD.jl's portable lowering. It is correct
    and covered by the tests with `SIMD_JL_GENERIC=1`, but about 4× slower.
  - AVX-512 machines use LLVM's compress intrinsic. That path is untested here.
  - A tuned AVX-512 path (native `vpcompress`, 512-bit vectors) is the obvious
    next step.
- Only plain vectors of the six primitive types are vectorized. There is no
  `sortperm`, `by` or `lt`.
- The sort is not stable. This only matters for distinguishable equal keys, such
  as NaNs with different payloads.
- Depends on the unreleased SIMD.jl fork (through `[sources]`, which needs Julia
  1.11+; on 1.10, `Pkg.develop` both packages).

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=8 SIMD_JL_GENERIC=1 julia --project=. test/runtests.jl
```

Passes on Julia 1.10, 1.13 and 1.14-alpha, and with the portable lowering.

## References

- J. Wassenberg, M. Blacher, J. Giesen, P. Sanders. *Vectorized and
  performance-portable quicksort.* Software: Practice and Experience 52(12), 2022.
  [arXiv:2205.05982](https://arxiv.org/abs/2205.05982)
- B. Bramas. *A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
  Skylake.* IJACSA 8(10), 2017. [arXiv:1704.08579](https://arxiv.org/abs/1704.08579)
- K. E. Batcher. *Sorting networks and their applications.* AFIPS 1968.
- D. R. Musser. *Introspective sorting and selection algorithms.* SPE 27(8), 1997.

## License

MIT, Copyright (c) 2026 Lars Hellemo.
