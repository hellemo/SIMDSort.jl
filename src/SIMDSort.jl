"""
    SIMDSort

Vectorized sorting in Julia on top of SIMD.jl: a vectorized quicksort for
primitive numbers, with counting and hash counting for few distinct values,
MSD radix refinement for `sortperm`, strings and other key types, and a
sample-sort step for wide fixed-width types. See `docs/src/design.md` for the
algorithms and their sources.
"""
# Sources, cited in the code by these tags (full references in README.md and
# docs/src/design.md):
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
#   [IPS4o]  Axtmann et al., in-place sample sort (2022)  msd.jl (sample-sort distribution)
# Everything else (key maps, prefetch, three-way partition, parallel partition,
# counting sort, windowed keys, string prefix cache, bignum encodings) is this
# package's own design or textbook technique.
module SIMDSort

using SIMD

export simdsort!, simdsort, simdsortperm

include("instrument.jl")
include("scratchmem.jl")
include("tasks.jl")
include("keys.jl")
include("keyref.jl")
include("network.jl")
include("partition.jl")
include("sort.jl")
include("parallel.jl")
include("perm.jl")
include("hashcount.jl")
include("symbols.jl")
include("msd.jl")
include("wide.jl")
include("compound.jl")
include("inputs.jl")
include("tuning.jl")

"""
    simdsort!(v; rev=false, ntasks=1)

Sort `v` in place in the order of `isless` (NaNs last, `-0.0` before `0.0`), or its
reverse if `rev`. `Vector`s of `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`
and `Float64`, and contiguous views of them (`view(v, a:b)`), use the
vectorized quicksort, in place. Other element types with a fast path
(strings, symbols, wide and arbitrary-precision numbers, tuples, Dates, ...)
are sorted in `Vector`s; other arrays of them (views, matrix columns, ...) as
a `Vector` copy. Anything else falls back to `sort!`.

With `ntasks > 1`, large subranges are sorted by up to `ntasks` concurrent tasks;
`ntasks = Threads.nthreads()` uses all threads.
"""
function simdsort!(v::AbstractVector; rev::Bool=false, ntasks::Integer=1)
    # Types whose order is given by order words (extensions define them): the
    # distribution pass and cache-resident word sorts.
    if v isa Vector && has_orderwords(eltype(v))
        length(v) <= 1 && return v
        return sort_by_words!(v, orderwords, rev, Int(ntasks))
    end
    length(v) <= 1 && return v
    # Union{Missing,T} elements: the values sorted as a Vector{T}, missings last.
    missingpath(eltype(v)) && return missing_sort!(v, rev, Int(ntasks))
    # Other arrays of an element type with a fast path for Vectors (views of
    # strings, matrix columns, strided views, CSV.jl's ChainedVector, ...):
    # sorted as a Vector copy, then copied back.
    if !(v isa Vector) && vectorpath(eltype(v))
        return copyto!(v, simdsort!(collect(v); rev, ntasks))
    end
    return sort!(v; rev)
end

"""
    pooled(v) -> Bool

Whether `v` keeps a pool of distinct values and integer refs into it
(`DataAPI.refpool`, through the DataAPI extension): `simdsortperm` then leaves
it to its own `sortperm`, which sorts the refs.
"""
pooled(v) = false

"""
    vectorpath(T) -> Bool

Whether `simdsort!`/`simdsortperm` have a fast path for a `Vector{T}`. Other
arrays with elements of type `T` are then sorted as a `Vector` copy (two
copies, a few ms at 10^6 elements, against Base's `sort!` at 7-30× the sort
time). Extensions add methods for their types.
"""
vectorpath(::Type{T}) where {T} =
    T <: Union{SortTypes,SmallTypes,ByteString,Symbol,BigInt,BigFloat,VersionNumber} || iswideint(T) ||
    (T <: TupleLike && isconcretetype(T)) || has_orderwords(T)

function simdsort!(v::Union{Vector{T},VectorView{T}}; rev::Bool=false, ntasks::Integer=1) where {T<:SortTypes}
    length(v) <= 1 && return v
    return sort_primitive!(keyclass(T), v, ordering(rev), Int(ntasks))
end

# The keys of v as the same memory reinterpreted as type S: for a contiguous
# view, the same range of the reinterpreted parent (a key vector, where the
# reinterpreted view would not be).
keys_as(::Type{S}, v::VectorView) where {S} = view(reinterpret(S, parent(v)), parentindices(v)[1])
keys_as(::Type{S}, v::AbstractVector) where {S} = reinterpret(S, v)

"""
    MappedKeys(), NativeKeys(), FloatKeys()

How the six primitive types are sorted (`keyclass(T)`): `Int32`, `Int64` and
`UInt64` are mapped to signed keys (`UInt64` is sorted as is with 512-bit
vectors); `UInt32` is sorted as is; `Float32` and `Float64` are sorted as
floats. Each class has its own `sort_primitive!`
method, which keeps its own (measured) order of the cheap checks for special
inputs (counting, presorted, hash counting).
"""
struct MappedKeys end
struct NativeKeys end
struct FloatKeys end
keyclass(::Type{<:Union{Int32,Int64}}) = MappedKeys()
keyclass(::Type{UInt32}) = NativeKeys()
keyclass(::Type{UInt64}) = wide() ? NativeKeys() : MappedKeys()
keyclass(::Type{<:Union{Float32,Float64}}) = FloatKeys()

# Integers are mapped to signed keys of the same width (an order-preserving
# involution on the bits, complemented for Reverse), sorted, and mapped back.
function sort_primitive!(::MappedKeys, v::KeyVector{T}, o::Ord, nt::Int) where {T}
    @phase(:counting_check, counting_sort!(v, o, nt)) && return v
    # Check for sorted input before the key map, which preserves the order.
    @phase(:presorted_check, presorted!(v, o, nt)) && return v
    @phase(:hash_check, hash_count_sort!(v, o, nt)) && return v
    k = keys_as(signedkey(T), v)         # the same memory, as signed keys
    @phase :keymap foreach_chunk(c -> map_keys!(c, T, o), k, nt)
    sort_auto!(k, nt)
    @phase :keyunmap foreach_chunk(c -> unmap_keys!(c, T, o), k, nt)
    return v
end

# UInt32 keys are sorted as they are: AVX2 has unsigned 32-bit min/max, so the
# kernels need no key map. Reverse complements the values (an exact involution
# that reverses the unsigned order). UInt64 keeps the map on AVX2: without
# unsigned 64-bit min/max, native kernels cost as much as the two map passes
# save. AVX-512 has them (vpminuq, vpcmpuq), so there UInt64 is native too.
function sort_primitive!(::NativeKeys, v::KeyVector{T}, o::Ord, nt::Int) where {T}
    @phase(:counting_check, counting_sort!(v, o, nt)) && return v
    @phase(:hash_check, hash_count_sort!(v, o, nt)) && return v
    isrev(o) && foreach_chunk(c -> mapkeys!(~, c), v, nt)
    @phase(:presorted_check, presorted!(v, Base.Order.Forward, nt)) || sort_auto!(v, nt)
    isrev(o) && foreach_chunk(c -> mapkeys!(~, c), v, nt)
    return v
end

# Floats are sorted as floats: NaNs are moved out first, Reverse negates the
# values (an exact involution), and since compares treat -0.0 == 0.0, the run of
# zeros is rewritten in `isless` order afterwards from a count of the -0.0s.
function sort_primitive!(::FloatKeys, v::Union{Vector{T},VectorView{T}}, o::Ord, nt::Int) where {T}
    nnan, nneg, npos = @phase :float_scan float_scan(v, nt)
    r = nnan == 0 ? (1:length(v)) : @phase(:nan_partition, nan_partition!(v, o))
    length(r) <= 1 && return v
    k = view(v, r)              # the values without NaNs
    @phase(:hash_check, hash_count_sort!(k, o, nt)) && return v
    isrev(o) && foreach_chunk(c -> mapkeys!(-, c), k, nt)
    @phase(:presorted_check, presorted!(k, Base.Order.Forward, nt)) || sort_auto!(k, nt)
    # After negation (Reverse) the +0.0s of the input are the -0.0s.
    @phase :order_zeros order_zeros!(k, isrev(o) ? npos : nneg)
    isrev(o) && foreach_chunk(c -> mapkeys!(-, c), k, nt)
    return v
end

"""
    simdsort(v; rev=false, ntasks=1)

Return a sorted copy of `v`; see [`simdsort!`](@ref).
"""
simdsort(v::AbstractVector; rev::Bool=false, ntasks::Integer=1) = simdsort!(copy(v); rev, ntasks)

end
