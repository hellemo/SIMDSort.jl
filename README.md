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

![Random keys, one thread](docs/random1.svg)
![Random keys, 8 threads](docs/random8.svg)

Seconds per sort; 2^26 keys (2^27 for `UInt64`). In every row the fastest
single-threaded and the fastest 8-thread result are bold.

<!-- table:main -->
| **case**                         | **Base 1 thr** | **ChapelSort 1 thr** | **SIMDQuickSort 1 thr** | **x86-simd-sort 1 thr** | **vqsort 1 thr** | **ChapelSort 8 thr** | **SIMDQuickSort 8 thr** | **x86-simd-sort 8 thr** |
|:---------------------------------|---------------:|---------------------:|------------------------:|------------------------:|-----------------:|---------------------:|------------------------:|------------------------:|
| UInt64, 2^27                     |          3.132 |                1.533 |               **1.242** |                   2.071 |            1.726 |            **0.332** |                   0.427 |                   0.574 |
| Int64, 2^26                      |          1.589 |                0.855 |               **0.531** |                   0.806 |            0.747 |                0.183 |               **0.175** |                   0.241 |
| Float64 (randn), 2^26            |          1.666 |                1.017 |               **0.542** |                   0.579 |            0.788 |                0.209 |               **0.178** |                   0.183 |
| UInt32, 2^26                     |          0.551 |                0.450 |               **0.271** |                   0.323 |            0.313 |                0.094 |               **0.085** |                   0.101 |
| Int32, 2^26                      |          0.540 |                0.495 |               **0.269** |                   0.336 |            0.302 |                0.108 |               **0.086** |                   0.105 |
| Float32 (randn), 2^26            |          0.779 |                0.513 |               **0.304** |                   0.331 |            0.327 |                0.110 |               **0.093** |                   0.100 |
| UInt64 in 0:1000, 2^26           |          0.084 |                0.087 |               **0.064** |                   0.322 |            0.285 |                0.030 |               **0.024** |                   0.140 |
| UInt64, 1000 random values, 2^26 |          1.635 |                1.025 |               **0.126** |                   0.338 |            0.279 |                0.132 |               **0.036** |                   0.141 |
| UInt64 sorted, 2^26              |          0.031 |                0.031 |               **0.030** |                   1.007 |            0.751 |            **0.009** |                   0.029 |                   0.281 |
| UInt64 reversed, 2^26            |          0.056 |                0.058 |               **0.054** |                   0.998 |            0.757 |            **0.021** |                   0.056 |                   0.276 |
| Int32, 2^20                      |          0.005 |                0.007 |               **0.003** |                   0.003 |            0.003 |                0.002 |               **0.001** |                   0.002 |
<!-- /table:main -->

- **One thread:** fastest of all measured sorts for every numeric type, and for
  duplicates and presorted inputs.
  - Against the best C++ sort: Float64 0.542 vs 0.579 s (x86-simd-sort), Int32
    0.269 vs 0.302 s (vqsort), UInt64 1.24 vs 1.73 s (vqsort).
  - Few distinct values: small ranges are counted, other few-distinct inputs are
    counted in a hash table. 1000 random `UInt64` values: 0.126 s, against 0.279
    for vqsort.
- **Eight threads:** fastest for `Int64`, `Float64`, the 32-bit types and
  duplicate-heavy inputs. ChapelSort's parallel radix sort leads for `UInt64` at
  2^27 keys and for presorted inputs.

![Duplicates and presorted inputs](docs/special.svg)

### Other types

![sortperm speedup over Base](docs/types_sortperm.svg)

10^7 elements (BigInt, BigFloat, Float64x4, VersionNumber: 10^6), best of 3
rounds:

<!-- table:types -->
| **case, operation**                   | **Base 1 thr** | **ChapelSort 1 thr** | **SIMDQuickSort 1 thr** | **ChapelSort 8 thr** | **SIMDQuickSort 8 thr** |
|:--------------------------------------|---------------:|---------------------:|------------------------:|---------------------:|------------------------:|
| String, random, length 8–20, sort     |          5.910 |                1.620 |               **0.825** |            **0.364** |                   0.413 |
| String, random, length 8–20, sortperm |         14.980 |                1.944 |               **0.684** |                0.541 |               **0.219** |
| String, h2o id3 (id%010d), sort       |          3.655 |            **1.328** |                   2.037 |            **0.390** |                   0.669 |
| String, h2o id3 (id%010d), sortperm   |          8.643 |            **1.667** |                   1.796 |                0.578 |               **0.558** |
| Symbol, 10^5 distinct, sort           |          1.174 |                0.303 |               **0.176** |            **0.062** |                   0.171 |
| Symbol, 10^5 distinct, sortperm       |          3.762 |                0.519 |               **0.427** |            **0.238** |                   0.247 |
| Int128, sort                          |          0.415 |            **0.293** |                   0.448 |            **0.071** |                   0.157 |
| Int128, sortperm                      |          2.096 |                0.459 |               **0.250** |                0.112 |               **0.109** |
| UInt256 (BitIntegers), sort           |          0.702 |            **0.371** |                   0.540 |            **0.092** |                   0.211 |
| UInt256 (BitIntegers), sortperm       |          3.476 |                0.664 |               **0.247** |                0.176 |               **0.107** |
| Double64 (DoubleFloats), sort         |          1.024 |                0.751 |               **0.550** |            **0.151** |                   0.271 |
| Double64 (DoubleFloats), sortperm     |          2.678 |                0.556 |               **0.364** |            **0.153** |                   0.248 |
| Float128 (Quadmath), sort             |          2.880 |            **0.426** |                   0.591 |            **0.102** |                   0.274 |
| Float128 (Quadmath), sortperm         |          9.474 |                0.560 |               **0.361** |            **0.131** |                   0.219 |
| Float64x2 (MultiFloats), sort         |          0.850 |                0.760 |               **0.484** |            **0.164** |                   0.267 |
| Float64x2 (MultiFloats), sortperm     |          4.523 |                0.513 |               **0.318** |            **0.164** |                   0.199 |
| Float64x4 (MultiFloats), sort         |          0.082 |                0.054 |               **0.027** |            **0.013** |                   0.016 |
| Float64x4 (MultiFloats), sortperm     |          0.335 |                0.058 |               **0.016** |                0.016 |               **0.011** |
| Date, sort                            |          0.225 |                0.084 |               **0.059** |                0.025 |               **0.020** |
| Date, sortperm                        |          1.572 |            **0.141** |                   0.284 |            **0.043** |                   0.116 |
| DateTime, sort                        |          0.273 |                0.208 |               **0.069** |                0.040 |               **0.023** |
| DateTime, sortperm                    |          1.599 |                0.348 |               **0.256** |            **0.081** |                   0.106 |
| Tuple{Int64,Float64}, sort            |          0.803 |                0.552 |               **0.537** |            **0.122** |                   0.208 |
| Tuple{Int64,Float64}, sortperm        |          2.694 |                0.618 |               **0.347** |            **0.144** |                   0.145 |
| Pair{Int32,Int64}, sort               |          0.474 |            **0.275** |                   0.532 |            **0.064** |                   0.211 |
| Pair{Int32,Int64}, sortperm           |          2.324 |                0.452 |               **0.337** |            **0.116** |                   0.118 |
| Tuple{Int64,String}, sort             |          0.708 |                0.473 |               **0.325** |            **0.143** |                   0.272 |
| Tuple{Int64,String}, sortperm         |          1.718 |                0.363 |               **0.312** |            **0.132** |                   0.186 |
| VersionNumber, sort                   |          0.149 |            **0.060** |                   0.064 |                0.044 |               **0.040** |
| VersionNumber, sortperm               |          0.287 |            **0.047** |                   0.053 |            **0.028** |                   0.033 |
| BigInt (\~128-384 bits), sort         |          0.480 |                0.190 |               **0.128** |            **0.099** |                   0.110 |
| BigInt (\~128-384 bits), sortperm     |          1.141 |                0.210 |               **0.150** |                0.114 |               **0.112** |
| BigFloat (256 bits), sort             |          0.550 |                0.229 |               **0.147** |            **0.090** |                   0.119 |
| BigFloat (256 bits), sortperm         |          1.123 |                0.216 |               **0.144** |            **0.087** |                   0.112 |
<!-- /table:types -->

- **`sortperm`:** SIMDQuickSort wins most rows on one thread.
- **`sort` for wide fixed-width types** (Int128, UInt256, Float128, Pair): it
  sorts through a permutation, and applying the permutation costs more than
  ChapelSort's in-place radix sort.
- **8 threads:** ChapelSort's radix sort leads in most rows.

![sort speedup over Base](docs/types_sort.svg)

### Quicksort or radix sort?

![Crossover between ChapelSort and SIMDQuickSort](docs/crossover.svg)

The radix sort is faster only with 8 tasks, for 64-bit integer keys (and for
`UInt64` mostly at ≥ 2^26 keys). The quicksort is faster for everything else,
including all floats and all 32-bit keys. See docs/DESIGN.md §7.h.

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
