# SIMDSort.jl

> [!WARNING]
> SIMDSort.jl depends on an **unreleased fork of SIMD.jl**: branch
> [`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress)
> of `hellemo/SIMD.jl`. It adds `vpartition`, `vpermute` and a faster `vstorec`.
> The registered SIMD.jl does not have these functions, so the package does not
> work with it.
> - On Julia 1.11 and later, `Project.toml` takes the fork from GitHub through
>   `[sources]` (`url = "https://github.com/hellemo/SIMD.jl"`,
>   `rev = "permute-compress"`); `Pkg.instantiate()` fetches it.
> - `[sources]` only applies to the active project. When SIMDSort.jl is
>   added to another environment, or on Julia 1.10, add the fork there first:
>   `Pkg.add(url = "https://github.com/hellemo/SIMD.jl", rev = "permute-compress")`.
>
> This will stay so until the changes are merged into SIMD.jl and released.

Experimental vectorized sorting in pure Julia: a vectorized quicksort for
primitive numbers, with counting, hash counting, MSD radix refinement and a
sample-sort step for other inputs, types and many threads. It is built on the
SIMD.jl fork with runtime permutes and compress (`vpartition`, `vpermute`),
branch [`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress).

```julia
using SIMDSort
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
x86-simd-sort. [docs/src/design.md](docs/src/design.md) describes each part, with
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

### Code structure

The kernels are written against key references with a safe (array) and a
raw (pointer) backing. Measured with the same code, the safe one costs
17–19% on one thread (the partition loop reloads the array's data address
around every SIMD.jl store), so the kernels use pointers. Everything around
them works on arrays and views, and the remaining unsafe code is in a few
documented places. docs/src/design.md §10 has the measurements.
`bench/ab.jl` compares a baseline source tree against the working tree, run
interleaved in one process; every refactoring step had to pass it.

### Documentation

The documentation (usage, API, benchmarks, the design document, AVX-512
notes, the measuring scripts, internals) is built with Documenter:
`julia --project=docs -e 'using Pkg; Pkg.instantiate()'`, then
`julia --project=docs docs/make.jl` (HTML in `docs/build/`).

A shared library with C entry points for the six primitive types, built with
JuliaC and trimmed, is in `lib/` (see the "C library" page of the docs).

## Results

Measured on an Intel Core Ultra 7 268V (Lunar Lake: 4 P + 4 E cores, AVX2,
15 GB) with Julia 1.13.2 and 8 threads.
- Each round draws a fresh input, the same for all methods within the round, and
  runs the methods in a shuffled order. No method is timed on an input whose
  branch pattern the predictor has already seen, and thermal drift is spread
  evenly.
- Times are the best of 9 rounds (5 for `Base.sort!`, 3 for the other types);
  the CSV files in `bench/results/` also have the medians.
- x86-simd-sort and vqsort are called through the C shim in ChapelSort.jl's
  `bench/simd`.

To reproduce, run `bench/bench.jl`, `bench/types.jl` and `bench/crossover.jl`,
then `bench/tables.jl` (these tables) and `bench/plots.jl` (the charts).

![Random keys, one thread](docs/src/assets/random1.svg)
![Random keys, 8 threads](docs/src/assets/random8.svg)

Seconds per sort; 2^26 keys (2^27 for `UInt64`). In every row the fastest
single-threaded and the fastest 8-thread result are bold.

<!-- table:main -->
| **case**                         | **Base 1 thr** | **ChapelSort 1 thr** | **SIMDSort 1 thr** | **x86-simd-sort 1 thr** | **vqsort 1 thr** | **ChapelSort 8 thr** | **SIMDSort 8 thr** | **x86-simd-sort 8 thr** |
|:---------------------------------|---------------:|---------------------:|-------------------:|------------------------:|-----------------:|---------------------:|-------------------:|------------------------:|
| UInt64, 2^27                     |          3.084 |                1.354 |          **1.140** |                   1.827 |            1.546 |            **0.305** |              0.399 |                   0.531 |
| Int64, 2^26                      |          1.495 |                0.800 |          **0.497** |                   0.742 |            0.696 |                0.175 |          **0.166** |                   0.232 |
| Float64 (randn), 2^26            |          1.519 |                0.954 |          **0.508** |                   0.536 |            0.735 |                0.197 |          **0.172** |                   0.173 |
| UInt32, 2^26                     |          0.481 |                0.438 |          **0.256** |                   0.300 |            0.282 |                0.089 |          **0.082** |                   0.092 |
| Int32, 2^26                      |          0.480 |                0.469 |          **0.258** |                   0.312 |            0.290 |                0.102 |          **0.084** |                   0.094 |
| Float32 (randn), 2^26            |          0.663 |                0.490 |          **0.280** |                   0.307 |            0.310 |                0.107 |          **0.089** |                   0.095 |
| UInt64 in 0:1000, 2^26           |          0.077 |                0.081 |          **0.061** |                   0.305 |            0.255 |                0.033 |          **0.020** |                   0.131 |
| UInt64, 1000 random values, 2^26 |          1.548 |                0.926 |          **0.111** |                   0.314 |            0.267 |                0.123 |          **0.041** |                   0.134 |
| UInt64 sorted, 2^26              |          0.027 |                0.028 |          **0.025** |                   0.913 |            0.684 |                0.009 |          **0.008** |                   0.266 |
| UInt64 reversed, 2^26            |          0.052 |                0.054 |          **0.051** |                   0.914 |            0.660 |                0.021 |          **0.019** |                   0.256 |
| Int32, 2^20                      |          0.005 |                0.007 |          **0.003** |                   0.003 |            0.003 |                0.002 |          **0.001** |                   0.002 |
<!-- /table:main -->

- **One thread:** fastest of all measured sorts for every numeric type, and for
  duplicates and presorted inputs.
  - Against the best C++ sort: Float64 0.508 vs 0.536 s (x86-simd-sort), Int32
    0.258 vs 0.290 s (vqsort), UInt64 1.14 vs 1.55 s (vqsort).
  - Few distinct values: small ranges are counted, other few-distinct inputs are
    counted in a hash table. 1000 random `UInt64` values: 0.111 s, against 0.267
    for vqsort.
- **Eight threads:** fastest for `Int64`, `Float64`, the 32-bit types,
  duplicate-heavy and presorted inputs. ChapelSort's parallel radix sort leads
  for `UInt64` at 2^27 keys (0.305 vs 0.399 s), where the laptop's memory
  bandwidth is the limit.
- **Many cores and AVX-512:** see [below](#avx-512-and-many-cores).

![Duplicates and presorted inputs](docs/src/assets/special.svg)

### AVX-512 and many cores

Measured on a 2 × 28-core Intel Xeon Gold 6348 (Ice Lake-SP, AVX-512,
2 NUMA nodes) with `bench/measure/scaling.jl`: best of 5 fresh inputs, seconds.
With AVX-512 the kernels use 512-bit vectors (native compress, 16-row
512-bit base-case networks), and `UInt64` needs no key map
(docs/src/avx512.md). From 16 threads and large inputs a sample-sort step
replaces the top partition levels, which stop scaling (docs/src/design.md §7j).

<!-- table:xeon -->
| **case**                       | **SIMDSort 1 thr** | **ChapelSort 1 thr** | **x86-simd-sort 1 thr** | **SIMDSort 56 thr** | **ChapelSort 56 thr** | **x86-simd-sort 56 thr** |
|:-------------------------------|-------------------:|---------------------:|------------------------:|--------------------:|----------------------:|-------------------------:|
| UInt64 2^27                    |          **1.445** |                3.005 |                   1.485 |           **0.102** |                 0.116 |                    0.275 |
| Int64 2^26                     |          **0.704** |                1.696 |                   0.738 |           **0.045** |                 0.083 |                    0.133 |
| Float64 randn 2^26             |              0.753 |                1.815 |               **0.674** |           **0.060** |                 0.097 |                    0.127 |
| Int32 2^26                     |              0.357 |                1.094 |               **0.347** |           **0.034** |                 0.047 |                    0.063 |
| Int128 10^7                    |          **0.270** |                0.445 |                       – |           **0.015** |                 0.021 |                        – |
| Float64 sortperm 10^7          |          **0.296** |                0.499 |                       – |           **0.021** |                 0.022 |                        – |
| String 8-20 sortperm 2\*10^6   |          **0.102** |                0.232 |                       – |           **0.012** |                 0.021 |                        – |
| Symbol 10^5 distinct sort 10^7 |          **0.073** |                0.403 |                       – |           **0.017** |                 0.028 |                        – |
<!-- /table:xeon -->

![Thread scaling on AVX2 and AVX-512](docs/src/assets/scaling.svg)

- **One thread:** SIMDSort leads x86-simd-sort for `Int64` (0.70 vs 0.74 s)
  and `UInt64` (1.45 vs 1.49 s) and trails it for `Int32` (3%) and `Float64`
  (11%: 512-bit floating point lowers the clock, docs/src/avx512.md).
- **56 threads:** SIMDSort leads every case; the parallel quicksorts
  (x86-simd-sort's included) stop scaling at about 16 threads.

### Other types

![sortperm speedup over Base](docs/src/assets/types_sortperm.svg)

10^7 elements (BigInt, BigFloat, Float64x4, VersionNumber: 10^6), best of 3
rounds:

<!-- table:types -->
| **case, operation**                   | **Base 1 thr** | **ChapelSort 1 thr** | **SIMDSort 1 thr** | **ChapelSort 8 thr** | **SIMDSort 8 thr** |
|:--------------------------------------|---------------:|---------------------:|-------------------:|---------------------:|-------------------:|
| String, random, length 8–20, sort     |          4.916 |                1.373 |          **0.707** |                0.304 |          **0.235** |
| String, random, length 8–20, sortperm |         11.578 |                1.739 |          **0.618** |                0.508 |          **0.159** |
| String, h2o id3 (id%010d), sort       |          3.199 |            **1.159** |              1.749 |            **0.367** |              0.495 |
| String, h2o id3 (id%010d), sortperm   |          7.138 |            **1.512** |              1.587 |                0.672 |          **0.365** |
| Symbol, 10^5 distinct, sort           |          1.082 |                0.298 |          **0.042** |                0.096 |          **0.029** |
| Symbol, 10^5 distinct, sortperm       |          3.301 |                0.468 |          **0.348** |                0.217 |          **0.117** |
| Int128, sort                          |          0.410 |                0.273 |          **0.187** |                0.073 |          **0.056** |
| Int128, sortperm                      |          1.688 |                0.426 |          **0.227** |                0.103 |          **0.071** |
| UInt256 (BitIntegers), sort           |          0.655 |                0.345 |          **0.272** |            **0.085** |              0.092 |
| UInt256 (BitIntegers), sortperm       |          2.853 |                0.604 |          **0.231** |                0.176 |          **0.079** |
| Double64 (DoubleFloats), sort         |          0.930 |                0.705 |          **0.249** |                0.163 |          **0.068** |
| Double64 (DoubleFloats), sortperm     |          2.346 |                0.520 |          **0.366** |            **0.141** |              0.200 |
| Float128 (Quadmath), sort             |          2.590 |                0.390 |          **0.214** |                0.087 |          **0.067** |
| Float128 (Quadmath), sortperm         |          7.756 |                0.502 |          **0.345** |            **0.121** |              0.173 |
| Float64x2 (MultiFloats), sort         |          0.808 |                0.718 |          **0.266** |                0.158 |          **0.065** |
| Float64x2 (MultiFloats), sortperm     |          3.933 |                0.461 |          **0.287** |                0.141 |          **0.117** |
| Float64x4 (MultiFloats), sort         |          0.069 |                0.047 |          **0.028** |                0.011 |          **0.008** |
| Float64x4 (MultiFloats), sortperm     |          0.296 |                0.057 |          **0.017** |                0.015 |          **0.008** |
| Date, sort                            |          0.168 |                0.086 |          **0.051** |                0.024 |          **0.021** |
| Date, sortperm                        |          1.528 |            **0.139** |              0.253 |            **0.037** |              0.085 |
| DateTime, sort                        |          0.227 |                0.208 |          **0.067** |                0.036 |          **0.023** |
| DateTime, sortperm                    |          1.556 |                0.324 |          **0.251** |            **0.079** |              0.082 |
| Tuple{Int64,Float64}, sort            |          0.707 |                0.497 |          **0.217** |                0.113 |          **0.060** |
| Tuple{Int64,Float64}, sortperm        |          2.298 |                0.560 |          **0.349** |                0.140 |          **0.109** |
| Pair{Int32,Int64}, sort               |          0.441 |                0.254 |          **0.196** |                0.069 |          **0.060** |
| Pair{Int32,Int64}, sortperm           |          1.969 |                0.445 |          **0.293** |                0.112 |          **0.094** |
| Tuple{Int64,String}, sort             |          0.633 |                0.390 |          **0.299** |            **0.164** |              0.193 |
| Tuple{Int64,String}, sortperm         |          1.409 |                0.355 |          **0.282** |            **0.133** |              0.171 |
| VersionNumber, sort                   |          0.140 |            **0.061** |              0.071 |                0.043 |          **0.039** |
| VersionNumber, sortperm               |          0.261 |            **0.045** |              0.058 |            **0.028** |              0.033 |
| BigInt (\~128-384 bits), sort         |          0.431 |                0.139 |          **0.137** |                0.081 |          **0.052** |
| BigInt (\~128-384 bits), sortperm     |          0.901 |                0.183 |          **0.166** |                0.092 |          **0.052** |
| BigFloat (256 bits), sort             |          0.495 |                0.213 |          **0.128** |                0.085 |          **0.054** |
| BigFloat (256 bits), sortperm         |          0.858 |                0.204 |          **0.118** |                0.089 |          **0.055** |
<!-- /table:types -->

- **`sort` for wide fixed-width types** (Int128, UInt256, Float128, Double64,
  MultiFloats, packed tuples and pairs): one sample-sort distribution pass, then
  cache-resident SIMD sorts of the buckets (docs/src/design.md §8). It used to go
  through a permutation and a gather.
- **One thread:** SIMDSort leads in 29 of 34 rows; ChapelSort in the h2o id
  strings (fixed prefix, then digits), VersionNumber, and Date `sortperm`.
- **8 threads:** SIMDSort leads in 25 of 34 rows, including random strings,
  Symbols (counted in hash tables by address: 0.029 s against 0.096), BigInt
  and BigFloat. ChapelSort's radix sort leads for the h2o id strings,
  `Tuple{Int64,String}`, UInt256 `sort` (a tie) and some `sortperm` rows
  (Double64, Float128, Date, DateTime, VersionNumber).

![sort speedup over Base](docs/src/assets/types_sort.svg)

### Quicksort or radix sort?

![Crossover between ChapelSort and SIMDSort](docs/src/assets/crossover.svg)

The radix sort is faster only with 4–8 tasks, for 64-bit integer keys
(`UInt64` mostly at ≥ 2^26 keys, `Int64` at 2^27). The quicksort is faster for everything else,
including all floats and all 32-bit keys. See docs/src/design.md §7.h.

[docs/src/design.md §7](docs/src/design.md) has the profile breakdown, per-level costs,
the effect of each optimization, the analysis of the remaining gaps, and the
ideas that were rejected. `bench/profile.jl` reproduces the breakdown.

### Tuning data from other machines

The kernels and thresholds were tuned on the two machines above. On another
processor (AMD Zen 4/5, newer Intel server parts, ARM, ...), please run
`SIMDSort.tuning_report()` in a Julia started with all threads and send the
file it writes with an issue; docs/src/measuring.md lists the processors
where data would help most.

## Limitations

- Tuned and tested on one AVX2 machine only, with 256-bit vectors.
  - CPUs without AVX2 (and ARM) get SIMD.jl's portable lowering. It is correct
    and covered by the tests with `SIMD_JL_GENERIC=1`, but about 4× slower.
  - On AVX-512 the code selects the single 32-row network and native 64-bit
    min/max, but has not been measured there. [docs/src/avx512.md](docs/src/avx512.md)
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
the source map at the top of `src/SIMDSort.jl`):

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

- **[IPS4o]** M. Axtmann, S. Witt, D. Ferizovic, P. Sanders. *Engineering
  in-place (shared-memory) sorting algorithms.* ACM TOPC, 2022.
  [arXiv:2009.13569](https://arxiv.org/abs/2009.13569). Used for the
  sample-sort distribution step in `sort!` for wide fixed-width types
  (splitters from a sorted sample, branch-free bucket search).

The implementation is written from these papers. It uses no code from Google
Highway (vqsort), Intel's x86-simd-sort or F5.

## License

MIT, Copyright (c) 2026 Lars Hellemo.
