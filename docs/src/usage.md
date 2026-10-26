# Usage

## Sorting

```jldoctest
julia> using SIMDSort

julia> v = [3.0, -0.0, NaN, 0.0, -2.5];

julia> simdsort(v)
5-element Vector{Float64}:
  -2.5
  -0.0
   0.0
   3.0
 NaN

julia> simdsortperm([:b, :a, :c, :a])
4-element Vector{Int64}:
 2
 4
 1
 3

julia> simdsort!([3, 1, 2]; rev = true)
3-element Vector{Int64}:
 3
 2
 1
```

All functions take the keywords `rev` (reverse order) and `ntasks` (the
largest number of tasks running at once; `Threads.nthreads()` uses all
threads). With `ntasks > 1`, large inputs are sorted in parallel.

## Supported types

Elements of a `Vector`. For the primitive numbers, `simdsort!` also sorts a
contiguous view `view(v, a:b)` in place; other arrays of these types (views,
matrix columns, `reinterpret`ed vectors, CSV.jl's `ChainedVector`, ...) are
sorted as a `Vector` copy, which costs two copies (`simdsortperm` of a pooled
array, such as a `PooledArray`, is left to its own `sortperm`, which sorts
the refs):

| Group | Types | How |
|---|---|---|
| Primitive numbers | `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64` | vectorized quicksort; counting sort for small ranges, hash counting for few distinct values; a sample-sort top level from 24 tasks |
| Strings | `String`, `SubString{String}`, `StringView` (Julia 1.14+) | windowed MSD refinement of 64-bit key windows |
| Symbols | `Symbol` | hash counting by identity (few distinct), else as strings |
| Wide numbers | `Int128`, `UInt128`, BitIntegers.jl types, `BigInt`, `BigFloat` | sample-sort distribution plus cache-resident sorts (`sort!`); windowed keys (`sortperm`) |
| Extensions | Dates (`Date`, `DateTime`, `Time`, Periods), DoubleFloats.jl, MultiFloats.jl, Quadmath.jl (`Float128`) | loaded with the package (weak dependencies) |
| Composite | `Tuple`s and `Pair`s of the above, `VersionNumber` | packed multi-word keys, or one stable pass per component |
| Missing values | `Union{Missing,T}` for any `T` above | the other values compacted (in parallel) and sorted as a `Vector{T}`; missings last (first with `rev`), as `isless` orders them |

Other element types, and other array types, fall back to `sort!`/`sortperm`.

## Types of your own

A type whose order is given by a few 64-bit words can join the fast
`sort!`/`sortperm` paths by defining [`SIMDSort.orderwords`](@ref) and
[`SIMDSort.has_orderwords`](@ref), as the extensions for DoubleFloats,
MultiFloats and Quadmath do:

```julia
struct Fixed128; hi::Int64; lo::UInt64; end
Base.isless(a::Fixed128, b::Fixed128) = (a.hi, a.lo) < (b.hi, b.lo)

SIMDSort.orderwords(x::Fixed128) = ((x.hi % UInt64) ⊻ (UInt64(1) << 63), x.lo)
SIMDSort.has_orderwords(::Type{Fixed128}) = true
```

## Instrumentation

[`SIMDSort.instrumented`](@ref) runs a sort with phase timers and event
counters switched on; [`SIMDSort.print_report`](@ref) prints them. This is how
weak spots were found on remote machines (see [Measuring](@ref)).

```julia
r = SIMDSort.instrumented(() -> simdsort!(rand(10^7); ntasks = 8))
SIMDSort.print_report(r)
```
