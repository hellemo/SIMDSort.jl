"""
    SIMDQuickSort

A vectorized quicksort for primitive numeric vectors, written in Julia on top of
SIMD.jl. See `docs/DESIGN.md` for the algorithm and its sources.
"""
# Sources, cited in the code by these tags (full references in README.md and
# docs/DESIGN.md):
#   [Hoa62]  Hoare, Quicksort (1962)                      sort.jl: qsort!
#   [Mus97]  Musser, introspective sorting (1997)         sort.jl: depth limit
#   [Bat68]  Batcher, sorting networks (1968)             network.jl
#   [Bra17]  Bramas, AVX-512 hybrid quicksort (2017)      partition.jl
#   [WBGS22] Wassenberg et al., vqsort (2022)             partition.jl, sort.jl, network.jl
#   [F5]     Arman & Loguinov, F5 MSD radix sort          network.jl (zigzag merges),
#                                                         sort.jl (double min/max leaves),
#                                                         perm.jl (common-prefix skip)
#   [MBM93]  McIlroy, Bostic & McIlroy, radix sort (1993) perm.jl (MSD refinement)
#   [CAFS]   Shlyk, cache-aware frequency sort (2026)     hashcount.jl
# Everything else (key maps, prefetch, three-way partition, parallel partition,
# counting sort, windowed keys, string prefix cache, bignum encodings) is this
# package's own design or textbook technique.
module SIMDQuickSort

using SIMD

export simdsort!, simdsort, simdsortperm

include("keys.jl")
include("network.jl")
include("partition.jl")
include("sort.jl")
include("parallel.jl")
include("perm.jl")
include("hashcount.jl")
include("wide.jl")
include("compound.jl")

"""
    simdsort!(v; rev=false, ntasks=1)

Sort `v` in place in the order of `isless` (NaNs last, `-0.0` before `0.0`), or its
reverse if `rev`. `Vector`s of `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`
and `Float64` use the vectorized quicksort; anything else falls back to `sort!`.

With `ntasks > 1`, large subranges are sorted by up to `ntasks` concurrent tasks;
`ntasks = Threads.nthreads()` uses all threads.
"""
function simdsort!(v::AbstractVector; rev::Bool=false, ntasks::Integer=1)
    sort!(v; rev)
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:SortTypes}
    n = length(v)
    n <= 1 && return v
    counting_sort!(v, rev, Int(ntasks)) && return v
    # Check for sorted input before the key map, which preserves the order.
    presorted!(v, rev) && return v
    hash_count_sort!(v, rev, Int(ntasks)) && return v
    S = signedkey(T)
    GC.@preserve v begin
        p = Ptr{S}(pointer(v))
        k = unsafe_wrap(Array, p, n)
        foreach_chunk(k -> map_keys!(k, T, rev), k, ntasks)
        sort_keys!(p, n, Int(ntasks))
        foreach_chunk(k -> unmap_keys!(k, T, rev), k, ntasks)
    end
    return v
end

# UInt32 keys are sorted as they are: AVX2 has unsigned 32-bit min/max, so the
# kernels need no key map. `rev` complements the values (an exact involution that
# reverses the unsigned order). UInt64 keeps the map: without unsigned 64-bit
# min/max, native kernels cost as much as the two map passes save.
function simdsort!(v::Vector{UInt32}; rev::Bool=false, ntasks::Integer=1)
    n = length(v)
    n <= 1 && return v
    counting_sort!(v, rev, Int(ntasks)) && return v
    hash_count_sort!(v, rev, Int(ntasks)) && return v
    rev && foreach_chunk(complement!, v, ntasks)
    GC.@preserve v (presorted!(v) || sort_keys!(pointer(v), n, Int(ntasks)))
    rev && foreach_chunk(complement!, v, ntasks)
    return v
end

# Floats are sorted as floats: NaNs are moved out first, `rev` negates the values
# (an exact involution), and since compares treat -0.0 == 0.0, the run of zeros
# is rewritten in `isless` order afterwards from a count of the -0.0s.
function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{Float32,Float64}}
    nnan, nneg, npos = float_scan(v, ntasks)
    r = nnan == 0 ? eachindex(v) : nan_partition!(v, rev)
    n = length(r)
    n <= 1 && return v
    GC.@preserve v begin
        p = pointer(v, first(r))
        k = unsafe_wrap(Array, p, n)
        hash_count_sort!(k, rev, Int(ntasks)) && return v
        rev && foreach_chunk(negate!, k, ntasks)
        presorted!(k) || sort_keys!(p, n, Int(ntasks))
        # After negation (rev) the +0.0s of the input are the -0.0s.
        order_zeros!(k, rev ? npos : nneg)
        rev && foreach_chunk(negate!, k, ntasks)
    end
    return v
end

"""
    simdsort(v; rev=false, ntasks=1)

Return a sorted copy of `v`; see [`simdsort!`](@ref).
"""
simdsort(v::AbstractVector; rev::Bool=false, ntasks::Integer=1) = simdsort!(copy(v); rev, ntasks)

end
