# SIMDQuickSort.jl

> [!WARNING]
> SIMDQuickSort.jl depends on an **unreleased fork of SIMD.jl**: branch
> [`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress)
> of `hellemo/SIMD.jl`. It adds `vpartition`, `vpermute` and a faster `vstorec`.
> The registered SIMD.jl does not have these functions, so the package does not
> work with it.
> - On Julia 1.11 and later, `Project.toml` takes the fork from GitHub through
>   `[sources]` (`url = "https://github.com/hellemo/SIMD.jl"`,
>   `rev = "permute-compress"`); `Pkg.instantiate()` fetches it.
> - `[sources]` only applies to the active project. When SIMDQuickSort.jl is
>   added to another environment, or on Julia 1.10, add the fork there first:
>   `Pkg.add(url = "https://github.com/hellemo/SIMD.jl", rev = "permute-compress")`.
>
> This will stay so until the changes are merged into SIMD.jl and released.

An experimental vectorized quicksort in pure Julia. It is built on the SIMD.jl
fork with runtime permutes and compress (`vpartition`, `vpermute`), branch
[`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress).

```julia
using SIMDQuickSort
v = rand(Int32, 10^8)
simdsort!(v)                                  # one thread
simdsort!(v; ntasks = Threads.nthreads())     # task parallel
simdsort(v; rev = true)                       # sorted copy, reverse order
p = simdsortperm(v)                           # stable, as sortperm
```

The order is that of `isless`: NaNs last, `-0.0` before `0.0`. `simdsortperm` is
stable, like `sortperm`.

Supported element types (in `Vector`s):

| Group | Types |
|---|---|
| Vectorized quicksort | `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64` |
| Strings | `String`, `SubString{String}`, `StringView` (Julia 1.14+), `Symbol` |
| Wide and arbitrary-precision numbers | `Int128`, `UInt128`, BitIntegers.jl types, `BigInt`, `BigFloat` |
| Through extensions (weak dependencies) | Dates (`Date`, `DateTime`, `Time`, Periods), DoubleFloats.jl, MultiFloats.jl, Quadmath.jl (`Float128`) |
| Composite | `Tuple`s and `Pair`s of the above, `VersionNumber` |

Other element types and other array types fall back to `sort!`/`sortperm`.

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
  - small integer ranges are counted instead of sorted, in one fused pass;
  - few distinct values over a wide range (a Chao1 estimate from 1024 samples)
    are counted in a cache-resident hash table, as in CAFS (Shlyk);
  - `UInt32` is sorted as is (AVX2 has unsigned 32-bit min/max).
- **Partition:** in place and bidirectional (Bramas 2017; Wassenberg et al. 2022).
  Each vector is split with `vpartition`, which is a table lookup plus `vpermd` on
  AVX2, and written with two full-width stores. In cache this costs about 0.11 ns
  per 32-bit key. Ranges larger than the L2 cache use software prefetch.
- **Pivot:** median of lane-wise medians of nine randomly placed vectors.
  - When the sample shows many copies of the pivot, the block of equal keys is
    split off.
  - Ranges of equal keys are detected by a vector min/max scan.
- **Base case:** vectorized sorting networks for up to 256 32-bit or 128 64-bit
  keys.
  - Batcher's network sorts the columns.
  - F5-style zigzag merges (Arman & Loguinov) merge them: odd-row reversal,
    odd-even column merges, then diagonal block exchanges.
  - The 32-row networks are built from two 16-row halves so that they fit in the
    16 AVX2 registers.
  - 64-bit integer leaves are sorted with double min/max (also from F5).
- **Other types:** a key window and the element's position are packed into a
  `UInt64`, the packed keys are sorted, and runs of equal windows are refined with
  the next window, as in an MSD radix sort (McIlroy, Bostic & McIlroy). Strings
  use a cached 16-byte prefix; Symbols use the ranks of their distinct names.
- **Fallbacks and parallelism:**
  - beyond a depth of $2\log_2 n + 4$, `Base.sort!` takes over;
  - with `ntasks > 1`, large ranges are partitioned in parallel and the smaller
    side of each partition becomes a new task.

## Results

Seconds per sort, best of 9 interleaved rounds (5 for `Base.sort!`); the best
time per row is in bold. Measured on an Intel Core Ultra 7 268V (Lunar Lake: 4 P
+ 4 E cores, AVX2, 15 GB) with Julia 1.13.2 and 8 threads.

Each round draws a fresh input, the same for all methods within the round, and
runs the methods in a shuffled order. No method is timed on an input whose branch
pattern the predictor has learnt in an earlier repetition, and thermal drift is
spread evenly. `bench/results/bench.csv` also has the medians.

x86-simd-sort and vqsort are called through the C shim in ChapelSort.jl's
`bench/simd`. Run with `julia -t 8 --project=bench bench/bench.jl`.

| case | Base.sort! (1 thr) | ChapelSort.jl psort! (1 thr) | ChapelSort.jl psort! (8 thr) | SIMDQuickSort.jl (1 thr) | SIMDQuickSort.jl (8 thr) | x86-simd-sort (1 thr) | x86-simd-sort (8 thr) | Highway vqsort (1 thr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UInt64, 2^27 | 3.226 | 1.496 | **0.345** | 1.453 | 0.518 | 1.983 | 0.612 | 1.634 |
| Int64, 2^26 | 1.608 | 0.860 | **0.192** | 0.633 | 0.205 | 0.825 | 0.249 | 0.749 |
| Float64 (randn), 2^26 | 1.539 | 1.006 | 0.227 | 0.609 | 0.189 | 0.570 | **0.185** | 0.771 |
| UInt32, 2^26 | 0.556 | 0.449 | 0.101 | 0.301 | **0.095** | 0.323 | 0.104 | 0.304 |
| Int32, 2^26 | 0.544 | 0.482 | 0.111 | 0.303 | **0.092** | 0.329 | 0.104 | 0.304 |
| Float32 (randn), 2^26 | 0.758 | 0.497 | 0.115 | 0.316 | 0.099 | 0.320 | **0.097** | 0.320 |
| UInt64 in 0:1000, 2^26 | 0.078 | 0.083 | 0.029 | 0.063 | **0.023** | 0.323 | 0.142 | 0.270 |
| UInt64, 1000 random values, 2^26 | 1.701 | 1.023 | **0.132** | 0.290 | 0.144 | 0.326 | 0.141 | 0.276 |
| UInt64 sorted, 2^26 | 0.028 | 0.029 | **0.008** | 0.026 | 0.026 | 0.990 | 0.276 | 0.740 |
| UInt64 reversed, 2^26 | 0.055 | 0.056 | **0.020** | 0.053 | 0.053 | 1.047 | 0.278 | 0.703 |
| Int32, 2^20 | 0.006 | 0.007 | 0.002 | 0.003 | **0.001** | 0.003 | 0.002 | 0.003 |

- **One thread:**
  - fastest of all measured sorts for `Int64`, `UInt64`, `UInt32`, `Float32`,
    small key ranges, and sorted/reversed inputs;
  - level with vqsort for `Int32` (0.303 vs 0.304 s);
  - behind on `Float64`: x86-simd-sort is 7% faster (0.570 vs 0.609 s);
  - behind on 1000 values spread over the whole `UInt64` range: vqsort is 5%
    faster (0.276 vs 0.290 s);
  - 1.8–5.9× faster than Base on random and duplicate-heavy inputs (level with
    it on presorted ones).
- **Eight threads:**
  - fastest for `Int32`, `UInt32` and small key ranges;
  - within 2% of x86-simd-sort for `Float32` and `Float64`;
  - ChapelSort's parallel radix sort leads for 64-bit integers (`UInt64` at
    2^27: 0.345 vs 0.518 s; `Int64`: 0.192 vs 0.205 s), wide-range duplicates
    and presorted inputs.

[docs/DESIGN.md §7](docs/DESIGN.md) has the profile breakdown, per-level costs,
the effect of each optimization, the analysis of the remaining gaps, and the
ideas that were rejected. `bench/profile.jl` reproduces the breakdown.

## Limitations

- Tuned and tested on one AVX2 machine only, with 256-bit vectors.
  - CPUs without AVX2 (and ARM) get SIMD.jl's portable lowering. It is correct
    and covered by the tests with `SIMD_JL_GENERIC=1`, but about 4× slower.
  - On AVX-512 the code selects the single 32-row network and native 64-bit
    min/max, but has not been measured there. [docs/AVX512.md](docs/AVX512.md)
    describes what remains to be measured, and `bench/avx512.jl` measures it.
- No `by` or `lt` keywords.
- `simdsort!` on the six primitive types is not stable. This only matters for
  distinguishable equal keys, such as NaNs with different payloads.
  `simdsortperm`, and `simdsort!` on all other types, are stable.
- Depends on the unreleased SIMD.jl fork (see the warning at the top).

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=8 SIMD_JL_GENERIC=1 julia --project=. test/runtests.jl
```

Passes on Julia 1.10, 1.13 and 1.14-alpha, and with the portable lowering.

## References

Papers whose ideas are used. The code names them where they are applied (see
the source map at the top of `src/SIMDQuickSort.jl`):

- **[WBGS22]** J. Wassenberg, M. Blacher, J. Giesen, P. Sanders. *Vectorized and
  performance-portable quicksort.* Software: Practice and Experience 52(12), 2022.
  [arXiv:2205.05982](https://arxiv.org/abs/2205.05982). Used for the recursion,
  partitioning and pivot sampling, and the register-matrix base case.
- **[Bra17]** B. Bramas. *A Novel Hybrid Quicksort Algorithm Vectorized using
  AVX-512 on Intel Skylake.* IJACSA 8(10), 2017.
  [arXiv:1704.08579](https://arxiv.org/abs/1704.08579). Used for the in-place
  bidirectional partition.
- **[F5]** A. Arman, D. Loguinov. *F5: A Robust SIMD-Accelerated MSD Radix Sort.*
  Texas A&M University; listed as ICDE 2026. Used for:
  - zigzag-merge networks;
  - double min/max for 64-bit integers;
  - skipping common key bits.
- **[Bat68]** K. E. Batcher. *Sorting networks and their applications.* AFIPS
  1968. Used for the odd-even merge sort and merge networks.
- **[Hoa62]** C. A. R. Hoare. *Quicksort.* The Computer Journal 5(1), 1962.
- **[Mus97]** D. R. Musser. *Introspective sorting and selection algorithms.* SPE
  27(8), 1997. Used for the depth limit with a fallback sort.
- **[MBM93]** P. M. McIlroy, K. Bostic, M. D. McIlroy. *Engineering radix sort.*
  Computing Systems 6(1), 1993. Used for the MSD refinement of string keys.
- **[CAFS]** V. S. Shlyk. *CAFS: A Cache-Aware Frequency Sort for Low-Cardinality
  Integer Data on x86-64.* [arXiv:2605.25040](https://arxiv.org/abs/2605.25040),
  2026. Used for the Chao1 estimate of distinct keys and hash counting.

Considered, not implemented (docs/DESIGN.md §7): IPS4o/IPS2Ra (Axtmann, Witt,
Ferizovic, Sanders; ACM TOPC 2022,
[arXiv:2009.13569](https://arxiv.org/abs/2009.13569)).

The implementation is written from these papers. It uses no code from Google
Highway (vqsort), Intel's x86-simd-sort or F5.

## License

MIT, Copyright (c) 2026 Lars Hellemo.
