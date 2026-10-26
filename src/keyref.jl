# Where the sorting kernels read and write keys: a key reference r, addressed
# by 0-based element offsets, with a handful of primitives. Two kinds:
#
#   ArrRange(v, off)      element i is v[off + i + 1]: safe array accesses
#                         (SIMD.jl's vload/vstore on the Vector, @inbounds)
#   Ptr{T}                element i is at p + i*sizeof(T): raw pointer accesses
#
# The kernels are written once against the primitives below; which kind the
# sort uses is decided at its entry points (keyref, SAFE_KEYS), so that both
# can be measured against each other (docs/src/design.md).

struct ArrRange{T,V<:AbstractVector{T}}
    v::V
    off::Int
end
ArrRange(v::AbstractVector) = ArrRange(v, 0)
Base.eltype(::Type{<:ArrRange{T}}) where {T} = T

const KeyRef{T} = Union{Ptr{T},ArrRange{T}}

# Safe array accesses (true) or raw pointers (false) in the kernels. With
# pointers the caller must keep the vectors alive (GC.@preserve); the entry
# points (sort_auto!, sort_keys!, ppartition!, pqsort!) do.
#
# Measured (docs/src/design.md §10): with array accesses the partition kernel is
# 8-25% slower and the full sorts 17-19% on one thread, because every vector
# load and store recomputes the array's data address (SIMD.jl's llvmcall
# stores carry no alias information, so LLVM cannot keep it in a register)
# and copyto! keeps its bounds checks. The small sorts are within 1-6%. So the
# kernels use pointers; everything around them works on arrays and views.
const SAFE_KEYS = false

# Test mode: kernels on arrays with bounds checks on. The test matrix runs the
# whole test suite once with this set (a rewritten copy of the source), which
# checks that the pointer kernels never read or write outside their arrays.
const CHECK_KEYS = false

# The key reference for the keys of v (and for scratch buffers), and the
# array that owns them (to keep alive while raw pointers are used).
# Key vectors: vectors, contiguous views of them, and vectors reinterpreted as
# integer keys of the same size (floats, Dates).
const ReinterpretedVector{T} = Base.ReinterpretArray{T,1,S,Vector{S},false} where {S}
const KeyVector{T} = Union{Vector{T},SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true},ReinterpretedVector{T}}
@inline keyref(v::Vector) = (SAFE_KEYS || CHECK_KEYS) ? ArrRange(v) : pointer(v)
@inline keyref(v::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}) where {T} =
    (SAFE_KEYS || CHECK_KEYS) ? ArrRange(parent(v), first(parentindices(v)[1]) - 1) : pointer(v)
# A scratch buffer as a reference of the same kind as p (the caller keeps buf
# alive while raw pointers are used).
@inline scratchref(::Ptr, buf::Vector) = pointer(buf)
@inline scratchref(::ArrRange, buf::Vector) = ArrRange(buf)
@inline keyref(v::Base.ReinterpretArray) = (SAFE_KEYS || CHECK_KEYS) ? ArrRange(v) : pointer(v)
keyowner(v::Vector) = v
keyowner(v::Union{SubArray,Base.ReinterpretArray}) = parent(v)

"""
    shift(r, k)

The reference moved k elements on.
"""
@inline shift(p::Ptr{T}, k::Integer) where {T} = p + k * sizeof(T)
@inline shift(r::ArrRange, k::Integer) = ArrRange(r.v, r.off + k)

"""
    vld(Val(N), r, i) -> Vec{N}
    vst(x::Vec, r, i)

N keys as a vector from offset i, and back.
"""
@inline vld(::Val{N}, p::Ptr{T}, i::Int) where {N,T} = vload(Vec{N,T}, p + i * sizeof(T))
@inline vld(::Val{N}, r::ArrRange{T}, i::Int) where {N,T} =
    CHECK_KEYS ? vload(Vec{N,T}, r.v, r.off + i + 1) : @inbounds(vload(Vec{N,T}, r.v, r.off + i + 1))
@inline vst(x::Vec, p::Ptr{T}, i::Int) where {T} = vstore(x, p + i * sizeof(T))
@inline vst(x::Vec, r::ArrRange, i::Int) =
    (CHECK_KEYS ? vstore(x, r.v, r.off + i + 1) : @inbounds(vstore(x, r.v, r.off + i + 1)); nothing)

"""
    vstc(x::Vec, r, i, mask)

The lanes of x selected by mask, in order, stored from offset i (exactly
count(mask) keys): a compress-store, in one of two forms, by CPU
(`vstc_memory()`):

- the memory form (SIMD.jl's `vstorec`; with AVX-512 `vpcompressq` with a
  memory operand): the default;
- a compress into a register and a store masked to the first count(mask)
  lanes: on AMD Zen (LLVM CPU `znver*`), where the memory form is microcoded
  and very slow. In a report from a Zen 4 EPYC 9274F (issue #1) the 512-bit
  partition took 1.8 ns per key and level, the sort of 2^24 Int64 0.6 s on one
  thread; on an Ice Lake Xeon this form is 8-11% slower than the memory form
  (one thread, 2^24 keys).

`VSTC_MEMORY = true`/`false` forces a form (A/B runs); `nothing` chooses by CPU.
"""
const VSTC_MEMORY = nothing

# The memory form, unless the CPU is an AMD Zen (decided when compiled, as target()).
@generated vstc_memory() = VSTC_MEMORY === nothing ? !startswith(Sys.CPU_NAME, "znver") : VSTC_MEMORY

@inline firstlanes(::Val{N}, c::Int) where {N} = Vec{N,Int32}(ntuple(k -> Int32(k - 1), Val(N))) < Int32(c)

@inline vstc(x::Vec{N,T}, p::Ptr{T}, i::Int, m::Vec{N,Bool}) where {N,T} =
    vstc_memory() ? vstc_mem(x, p, i, m) : vstc_reg(x, p, i, m)
@inline vstc(x::Vec{N,T}, r::ArrRange{T}, i::Int, m::Vec{N,Bool}) where {N,T} =
    vstc_memory() ? vstc_mem(x, r, i, m) : vstc_reg(x, r, i, m)

@inline vstc_mem(x::Vec{N,T}, p::Ptr{T}, i::Int, m::Vec{N,Bool}) where {N,T} =
    (vstorec(x, p + i * sizeof(T), m); nothing)
@inline vstc_reg(x::Vec{N,T}, p::Ptr{T}, i::Int, m::Vec{N,Bool}) where {N,T} =
    (vstore(vcompress(x, m), p + i * sizeof(T), firstlanes(Val(N), count(m))); nothing)
# (the stores' own bounds checks cover N keys, more than are written)
@inline function vstc_mem(x::Vec{N,T}, r::ArrRange{T}, i::Int, m::Vec{N,Bool}) where {N,T}
    CHECK_KEYS && checkbounds(r.v, r.off+i+1:r.off+i+count(m))
    @inbounds vstorec(x, r.v, r.off + i + 1, m)
    return nothing
end
@inline function vstc_reg(x::Vec{N,T}, r::ArrRange{T}, i::Int, m::Vec{N,Bool}) where {N,T}
    CHECK_KEYS && checkbounds(r.v, r.off+i+1:r.off+i+count(m))
    @inbounds vstore(vcompress(x, m), r.v, r.off + i + 1, firstlanes(Val(N), count(m)))
    return nothing
end

"""
    ld(r, i)
    st!(r, x, i)

One key, 1-based like unsafe_load (element i - 1).
"""
@inline ld(p::Ptr, i::Integer) = unsafe_load(p, i)
@inline ld(r::ArrRange, i::Integer) = CHECK_KEYS ? r.v[r.off+i] : @inbounds(r.v[r.off+i])
@inline st!(p::Ptr, x, i::Integer) = (unsafe_store!(p, x, i); nothing)
@inline st!(r::ArrRange, x, i::Integer) = (CHECK_KEYS ? (r.v[r.off+i] = x) : (@inbounds r.v[r.off+i] = x); nothing)

"""
    copykeys!(dst, src, n)

n keys from src to dst (overlap allowed, as memmove).
"""
@inline copykeys!(dst::Ptr{T}, src::Ptr{T}, n::Integer) where {T} = (unsafe_copyto!(dst, src, n); nothing)
@inline copykeys!(dst::ArrRange{T}, src::ArrRange{T}, n::Integer) where {T} =
    (copyto!(dst.v, dst.off + 1, src.v, src.off + 1, n); nothing)

"""
    prefetch(r, i)

Prefetch the cache line of element i (the address is not dereferenced, so
offsets past the end are harmless).
"""
@inline prefetch(p::Ptr{T}, i::Int) where {T} = prefetch(p + i * sizeof(T))
@inline function prefetch(r::ArrRange{T}, i::Int) where {T}
    v = r.v
    GC.@preserve v prefetch(pointer(v) + (r.off + i) * sizeof(T))
end

"""
    keyarray(r, n)

The n keys as an array, for Base's sort! (the depth-limit fallback).
"""
@inline keyarray(p::Ptr{T}, n::Integer) where {T} = unsafe_wrap(Array, p, n)
@inline keyarray(r::ArrRange, n::Integer) = view(r.v, r.off+1:r.off+n)
