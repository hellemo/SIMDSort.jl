# SIMDSort.jl

Vectorized sorting in pure Julia: a vectorized quicksort for primitive
numbers, with counting and hash counting for few distinct values, MSD radix
refinement for `sortperm`, strings and other key types, a sample-sort step
for wide fixed-width types and for many threads, and task parallelism
throughout.

```julia
using SIMDSort
v = rand(Int32, 10^8)
simdsort!(v)                                  # one thread
simdsort!(v; ntasks = Threads.nthreads())     # task parallel
simdsort(v; rev = true)                       # sorted copy, reverse order
p = simdsortperm(v)                           # stable, as sortperm
```

The order is that of `isless`: NaNs last, `-0.0` before `0.0`.
[`simdsortperm`](@ref) is stable, like `sortperm`.

!!! warning "Unreleased SIMD.jl fork"
    SIMDSort.jl depends on branch
    [`permute-compress`](https://github.com/hellemo/SIMD.jl/tree/permute-compress)
    of `hellemo/SIMD.jl`, which adds `vpartition`, `vpermute` and a faster
    `vstorec`. On Julia 1.11 and later, the package's `[sources]` entry
    fetches it; in other environments, or on Julia 1.10, add it first:
    `Pkg.add(url = "https://github.com/hellemo/SIMD.jl", rev = "permute-compress")`.

## Where to go next

- [Usage](@ref): supported types, options, extensions, instrumentation.
- [API](@ref): the public functions.
- [Benchmarks](@ref): against x86-simd-sort, Highway vqsort, ChapelSort.jl and Base.
- [Design](design.md): how it works, the papers it builds on, the measurements behind each choice.
- [AVX-512](avx512.md) and [Measuring](@ref): results on an AVX-512 server and the scripts behind them.
- [C library](@ref): a trimmed shared library built with JuliaC.
- [Internals](@ref): docstrings of the internal functions.
