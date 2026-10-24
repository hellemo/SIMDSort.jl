# Benchmarks

Measured on an AVX2 laptop (Intel Core Ultra 7 268V, 4 performance and 4
efficiency cores) unless noted; fresh input every round, methods interleaved
in random order, best of the rounds. The tables with all cases are in the
README; the scripts are in `bench/` (`bench.jl`, `types.jl`, `crossover.jl`,
`tables.jl`, `plots.jl`).

![Random keys, one thread](assets/random1.svg)

![Random keys, 8 threads](assets/random8.svg)

![Duplicates and presorted inputs](assets/special.svg)

![sortperm speedup over Base](assets/types_sortperm.svg)

![sort speedup over Base](assets/types_sort.svg)

![Crossover between ChapelSort and SIMDSort](assets/crossover.svg)

Results on a 2 × 28-core Xeon Gold 6348 (AVX-512) are summarized in
[AVX-512](avx512.md); the design document's performance analysis
([Design](design.md), §7) has the per-level costs and the effect of each
optimization.
