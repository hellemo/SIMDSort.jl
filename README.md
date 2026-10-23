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

- **Keys** are mapped in place to signed integers, so one signed vector compare
  implements `isless` for every type (AVX2 has no unsigned compares). The map is
  undone after sorting.
- **Partition:** in place and bidirectional (Bramas 2017; Wassenberg et al. 2022).
  Each vector is split with `vpartition`, which is a table lookup plus `vpermd` on
  AVX2, and written with two full-width stores. In cache this costs 0.10–0.12 ns
  per 32-bit key.
- **Pivot:** median of lane-wise medians of nine randomly placed vectors. Ranges of
  equal keys are detected by a vector min/max scan.
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
| UInt64, 2^27 | 3.167 | 1.667 | **0.489** | 1.957 | 0.719 | 2.251 | 0.693 | 1.898 |
| Int64, 2^26 | 1.601 | 0.913 | **0.268** | 0.846 | 0.282 | 0.941 | 0.347 | 0.868 |
| Float64 (randn), 2^26 | 1.678 | 1.190 | 0.284 | 0.920 | 0.345 | 0.655 | **0.201** | 0.886 |
| UInt32, 2^26 | 0.538 | 0.502 | 0.126 | 0.407 | 0.162 | 0.353 | **0.110** | 0.346 |
| Int32, 2^26 | 0.536 | 0.535 | 0.112 | 0.373 | 0.124 | 0.357 | **0.102** | 0.339 |
| Float32 (randn), 2^26 | 0.769 | 0.588 | 0.135 | 0.419 | 0.162 | 0.371 | **0.116** | 0.374 |
| UInt64 in 0:1000, 2^26 | 0.087 | 0.089 | **0.031** | 0.451 | 0.181 | 0.376 | 0.152 | 0.311 |
| UInt64 sorted, 2^26 | 0.033 | 0.034 | **0.010** | 0.082 | 0.056 | 1.087 | 0.321 | 0.835 |
| UInt64 reversed, 2^26 | 0.060 | 0.059 | **0.021** | 0.106 | 0.083 | 1.158 | 0.301 | 0.827 |
| Int32, 2^20 | 0.006 | 0.008 | 0.002 | 0.004 | **0.002** | 0.004 | 0.003 | 0.004 |

- **One thread, random keys.**
  - Faster than Base on every type: 1.4–1.9×.
  - Faster than ChapelSort's radix sort, except `UInt64` at 2^27.
  - About level with Highway vqsort for 64-bit integer keys.
  - 5–12% behind vqsort and x86-simd-sort for 32-bit keys.
  - x86-simd-sort is clearly faster on `Float64` (0.66 against 0.92 s).
- **Eight threads.** Scaling is weaker than the C++ sorts and ChapelSort (2.6–3.0×,
  against 3.2–4.8×). x86-simd-sort wins for 32-bit keys and floats, ChapelSort for
  64-bit integers. The top-level partitions are memory-bound, and floats pay for
  two extra passes (key map and unmap).
- **Few distinct values** (`0:1000`): every quicksort is about 5× slower than the
  radix sort's counting-sort path.
- **Sorted and reversed inputs** are caught by a linear pre-check, as in
  ChapelSort.

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
