# sortperm, and sorting strings and symbols, on top of the UInt64 quicksort
# (docs/src/design.md §8).
#
# Refining runs of equal windows with the next window is the MSD idea of radix
# sorting strings [MBM93]; skipping bits that all keys of a run share is also
# used by F5's prefix extractor [F5 §I]. Packing a key window with the element's
# position, the prefix cache and the symbol ranks are our own design.
#
# Every key is viewed as a big-endian bit string: numbers as their 32 or 64 bits in
# an order-preserving unsigned form, strings as their bytes followed by zeros.
# A window of K bits of each key goes into the high bits of a UInt64, the
# element's position into the low bits, and the UInt64s are sorted. Runs whose
# windows are equal are refined with the next window, as in an MSD radix sort.
# Positions in the low bits make every step stable.

"""
Key sources. Each key is a big-endian bit string; the interface:

- `keybits(ks, i, off, K)`: bits `[off, off + K)` of key `i` (K ≤ 63), zero
  past the end, complemented for reverse order;
- `keywidth(ks)`: the key length in bits, for fixed-width keys; or
  `exhausted(ks, i, off)`: whether key `i` has no bits from `off` on;
- `length_tiebreak(ks, a, b)`: the order of keys whose bits are all equal
  (default 0; strings put the shorter first);
- `finish_ties!(ks, perm)`: the same for a run of equal keys (default: none).
"""
abstract type KeySource end

@inline exhausted(ks::KeySource, i::Int, off::Int) = off >= keywidth(ks)
length_tiebreak(::KeySource, a::Int, b::Int) = 0

# --- numbers -------------------------------------------------------------------

struct NumKeys{T,U,V<:AbstractVector{T}} <: KeySource
    v::V
    rev::Bool
end
NumKeys(v::AbstractVector{T}, rev::Bool) where {T} = NumKeys{T,unsigned(signedkey(T)),typeof(v)}(v, rev)

# Order-preserving map to unsigned integers. NaNs (any sign or payload) map to
# the largest value, where `isless` puts them.
@inline ukey(::Type{U}, x::Unsigned) where {U} = x % U
@inline ukey(::Type{U}, x::Signed) where {U} = (x % U) ⊻ (one(U) << (8sizeof(U) - 1))
@inline function ukey(::Type{U}, x::AbstractFloat) where {U}
    isnan(x) && return typemax(U)
    b = reinterpret(U, x)
    m = one(U) << (8sizeof(U) - 1)
    return b & m == 0 ? b | m : ~b
end

keywidth(::NumKeys{T,U}) where {T,U} = 8sizeof(U)

@inline function keybits(ks::NumKeys{T,U}, i::Int, off::Int, K::Int) where {T,U}
    u = UInt64(ukey(U, @inbounds ks.v[i])) << (64 - 8sizeof(U))   # left-aligned
    ks.rev && (u = ~u >> (64 - 8sizeof(U)) << (64 - 8sizeof(U)))
    return off >= 64 ? zero(UInt64) : (u << off) >> (64 - K)
end

# --- strings and symbols -------------------------------------------------------

# Byte strings: `bytesrc(ks, i)` gives key i's bytes as (source, base), byte j
# being byteat(source, base + j); `bytelen(ks, i)` its length; `lentie(ks)` whether
# byte-equal keys of different length differ (strings: the shorter first).
abstract type AbstractByteKeys <: KeySource end

# Strings and symbols, read in place. The lengths are kept in an array, so that
# reading a window does not touch the string object for its length.
struct ByteKeys{V<:AbstractVector} <: AbstractByteKeys
    src::V
    len::Vector{Int}
    rev::Bool
end

# Strings read in place: String, SubString{String}, and (Julia 1.14+) StringViews
# of contiguous byte vectors. Symbols: their names (NUL-free, NUL-terminated).
@static if isdefined(Base, :DenseStringView)
    const ByteString = Union{String,SubString{String},Base.DenseStringView}
else
    const ByteString = Union{String,SubString{String}}
end
# A symbol's name has no array of code units, so it is read through a pointer:
# symbols are never freed.
byteptr_of(s::Symbol) = Base.unsafe_convert(Ptr{UInt8}, s)
bytelen_of(s::AbstractString) = ncodeunits(s)
bytelen_of(s::Symbol) = Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), byteptr_of(s)))

function ByteKeys(v::AbstractVector, rev::Bool, ntasks::Int, len = Vector{Int}(undef, length(v)))
    tforeach(length(v), ntasks) do r
        @inbounds for i in r
            len[i] = bytelen_of(v[i])
        end
    end
    return ByteKeys(v, len, rev)
end

@inline bytesrc(ks::ByteKeys, i::Int) = (bytes_of(@inbounds ks.src[i]), 0)
@inline bytes_of(s::AbstractString) = codeunits(s)
@inline bytes_of(s::Symbol) = byteptr_of(s)
@inline bytelen(ks::ByteKeys, i::Int) = @inbounds ks.len[i]
lentie(::ByteKeys) = true
nkeys(ks::ByteKeys) = length(ks.len)

# Keys given as one byte string each, back to back in `bytes` (key i is
# bytes[start[i]+1:start[i+1]]); equal values have identical encodings.
struct EncodedKeys <: AbstractByteKeys
    bytes::Vector{UInt8}
    start::Vector{Int}
    rev::Bool
end
@inline bytesrc(ks::EncodedKeys, i::Int) = (ks.bytes, @inbounds ks.start[i])
@inline bytelen(ks::EncodedKeys, i::Int) = @inbounds(ks.start[i+1] - ks.start[i])
lentie(::EncodedKeys) = false
nkeys(ks::EncodedKeys) = length(ks.start) - 1

@inline exhausted(ks::AbstractByteKeys, i::Int, off::Int) = off >= 8 * bytelen(ks, i)
@inline keybits(ks::AbstractByteKeys, i::Int, off::Int, K::Int) =
    bytebits(bytesrc(ks, i)..., bytelen(ks, i), off, K, ks.rev)

# Byte j (1-based) after base, and 8 bytes from base + 1 as a big-endian UInt64,
# from code units (safe: one movbe, as the pointer load) or a symbol's name.
@inline byteat(cu::AbstractVector{UInt8}, j::Int) = @inbounds cu[j]
@inline byteat(p::Ptr{UInt8}, j::Int) = unsafe_load(p, j)
@inline function load_be64(cu::AbstractVector{UInt8}, base::Int)
    x = zero(UInt64)
    @inbounds for j in 1:8      # little-endian assembly + bswap: LLVM fuses it into one load
        x |= UInt64(cu[base+j]) << (8 * (j - 1))
    end
    return bswap(x)
end
@inline load_be64(p::Ptr{UInt8}, base::Int) = ntoh(unsafe_load(Ptr{UInt64}(p + base)))

# Bits [off, off + K) of the zero-padded byte string (bytes base+1:base+len of
# src), K ≤ 63. They lie within the 9 bytes from byte off ÷ 8: the first 8 come
# from one big-endian load when they are all inside the string, the ninth only
# supplies the low bits for a shift. Bytes past the end read as zero; no byte
# past the end is loaded.
@inline function bytebits(src, base::Int, len::Int, off::Int, K::Int, rev::Bool)
    b = off >> 3
    s = off & 7
    if b + 8 <= len
        hi = load_be64(src, base + b)
    else
        hi = zero(UInt64)
        for j in 0:min(7, len - b - 1)
            hi |= UInt64(byteat(src, base + b + j + 1)) << (56 - 8j)
        end
    end
    lo = b + 8 < len ? byteat(src, base + b + 9) : 0x00
    x = s == 0 ? hi : (hi << s) | (UInt64(lo) >> (8 - s))
    x >>= 64 - K
    return rev ? ~x & ((one(UInt64) << K) - one(UInt64)) : x
end

function length_tiebreak(ks::AbstractByteKeys, a::Int, b::Int)
    lentie(ks) || return 0
    la, lb = bytelen(ks, a), bytelen(ks, b)
    la == lb && return 0
    return (la < lb) != ks.rev ? -1 : 1
end

# Strings equal as zero-padded bytes differ only in length; `isless` puts the
# shorter first.
finish_ties!(ks::KeySource, perm) = perm
function finish_ties!(ks::AbstractByteKeys, perm)
    lentie(ks) || return perm
    # Usually all lengths are equal (identical strings): nothing to do.
    l1 = bytelen(ks, perm[1])
    all(i -> bytelen(ks, i) == l1, perm) && return perm
    return sort!(perm; by = i -> bytelen(ks, i), rev = ks.rev, alg = Base.Sort.MergeSort)
end

# Full comparison of keys a and b from bit `off` on, for small runs: -1, 0 or 1.
function cmp_from(ks::KeySource, a::Int, b::Int, off::Int)
    while !(exhausted(ks, a, off) && exhausted(ks, b, off))
        x, y = keybits(ks, a, off, 56), keybits(ks, b, off, 56)
        x != y && return x < y ? -1 : 1
        off += 56
    end
    return length_tiebreak(ks, a, b)
end

# --- the windowed sort ---------------------------------------------------------

const PERM_SMALL = 64       # runs up to this size are finished by insertion sort
const PERM_PAR_MIN = 1 << 14

# The first 16 bytes of each byte string, big-endian and zero-padded. Kept aligned
# with `perm` and permuted along with it, so that windows within them are read
# sequentially instead of through the string pointers.
function prefix_cache(ks::AbstractByteKeys, ntasks::Int, pc = Vector{UInt128}(undef, nkeys(ks)))
    n = nkeys(ks)
    tforeach(n, ntasks) do r
        @inbounds for i in r
            pc[i] = (UInt128(keybits_raw(ks, i, 0)) << 64) | UInt128(keybits_raw(ks, i, 64))
        end
    end
    return pc
end

# The 64 bits from bit `off` of byte string i, not complemented for `rev`.
@inline function keybits_raw(ks::AbstractByteKeys, i::Int, off::Int)
    src, base = bytesrc(ks, i)
    len = bytelen(ks, i)
    return (bytebits(src, base, len, off, 32, false) << 32) | bytebits(src, base, len, off + 32, 32, false)
end

# (A function, not a closure over `off`, which refine! reassigns.)
function all_exhausted(ks::KeySource, perm, off::Int)
    @inbounds for i in perm
        exhausted(ks, i, off) || return false
    end
    return true
end

@inline function window(ks::KeySource, pc, perm, j::Int, off::Int, K::Int)
    if pc !== nothing && off + K <= 128
        x = ((@inbounds(pc[j]) << off) >> (128 - K)) % UInt64
        return ks.rev ? ~x & ((one(UInt64) << K) - one(UInt64)) : x
    end
    return keybits(ks, @inbounds(perm[j]), off, K)
end

# The window width at bit off with ib position bits: 64 - ib, but a window
# that starts in the prefix cache ends with it, so that it is read sequentially,
# not through the strings (for keys up to 16 bytes it then holds all their
# remaining bits).
window_bits(pc, off::Int, ib::Int) = pc !== nothing && off < 128 ? min(64 - ib, 128 - off) : 64 - ib

"""
    refine!(ks, perm, off, ntasks; isidentity = false)

Reorder `perm` (positions whose keys agree on bits `[0, off)`, in a stable
order) by the keys' bits from `off` on, stably. With `isidentity` (the top
call, `perm == 1:n`) the sorted positions are written directly instead of
looked up in `perm`: 10^7 random reads saved (a third of a Date sortperm).
"""
function refine!(ks::KeySource, perm::AbstractVector{Int}, off::Int, ntasks::Int, pc = nothing;
                 isidentity::Bool = false)
    m = length(perm)
    m <= 1 && return perm
    if all_exhausted(ks, perm, off)
        return finish_ties!(ks, perm)
    end
    m <= PERM_SMALL && return small_refine!(ks, perm, off, pc)
    ib = 8sizeof(Int) - leading_zeros(m - 1)       # bits for local positions
    mask = (one(UInt64) << ib) - one(UInt64)
    arena = ScratchArena()
    keys = alloc!(arena, UInt64, m)
    try
        while true
            K = window_bits(pc, off, ib)
            # The windows, and the bits in which any of them differs from the
            # first. Skip bits that all keys of the run share (long common
            # prefixes, as in "id%010d"): a window with no differences needs no sort.
            d = @phase :window_build window_keys!(keys, ks, pc, perm, off, K, ib, ntasks)
            if d == 0
                off += K
                all_exhausted(ks, perm, off) && return finish_ties!(ks, perm)
                continue
            end
            c = leading_zeros(d) - (64 - K)          # common leading bits in the window
            c < 8 && break
            off += c
        end
        K = window_bits(pc, off, ib)
        count!(:window_sorts)
        count!(:window_keys, m)
        simdsort!(keys; ntasks)
        @phase :window_gather if isidentity
            tforeach(m, ntasks) do r
                @inbounds for j in r
                    perm[j] = (keys[j] & mask) % Int + 1
                end
            end
        else
            old = alloc!(arena, Int, m)
            tforeach(m, ntasks) do r
                @inbounds for j in r
                    old[j] = perm[j]
                end
            end
            tforeach(m, ntasks) do r
                @inbounds for j in r
                    perm[j] = old[(keys[j] & mask) % Int + 1]
                end
            end
        end
        # Refine the runs with equal windows.
        runs = @phase :window_runs equal_runs(keys, ib, ntasks)
        isempty(runs) && return perm
        next = off + K
        # The runs' prefix cache entries in the new order, in one buffer (only
        # the runs: with distinct windows most keys are in none); each run gets
        # its slice.
        usepc = pc !== nothing && next < 128
        pcb = usepc ? alloc!(arena, UInt128, m) : nothing
        usepc && gather_runs!(pcb, pc, keys, mask, runs, ntasks)
        sub(r) = usepc ? view(pcb, r) : nothing
        if ntasks <= 1 || length(runs) == 1
            for r in runs
                refine!(ks, view(perm, r), next, length(r) > PERM_PAR_MIN ? ntasks : 1, sub(r))
            end
        else
            # Groups of runs of about equal total size, one task each.
            total = sum(length, runs)
            groups = balanced_groups(length.(runs), cld(total, 4ntasks))
            run_limited(groups, ntasks) do gr
                for r in view(runs, gr)
                    refine!(ks, view(perm, r), next, 1, sub(r))
                end
            end
        end
        return perm
    finally
        release!(arena)
    end
end

# dst[j] = src[old position of j] for the j in the runs.
function gather_runs!(dst, src, keys, mask::UInt64, runs, ntasks::Int)
    tforeach(length(runs), ntasks; minlen = 1 << 10) do rr
        @inbounds for q in rr, j in runs[q]
            dst[j] = src[(keys[j] & mask) % Int + 1]
        end
    end
    return nothing
end

# collect(1:n), written in parallel (a single server core writes ~13 GB/s).
function identity_perm(n::Int, ntasks::Int)
    p = Vector{Int}(undef, n)
    tforeach(n, ntasks) do r
        @inbounds for i in r
            p[i] = i
        end
    end
    return p
end

# keys[j] = window j (bits [off, off + K) of key perm[j]) << ib | (j - 1), in
# parallel chunks; returns the OR over all j of window j ⊻ window 1.
function window_keys!(keys, ks, pc, perm, off::Int, K::Int, ib::Int, ntasks::Int)
    m = length(keys)
    k1 = window(ks, pc, perm, 1, off, K)
    cs = chunks(m, partasks(m, ntasks, PERM_PAR_MIN))
    ds = zeros(UInt64, length(cs))
    tforeach_index(length(cs)) do c
        d = zero(UInt64)
        @inbounds for j in cs[c]
            w = window(ks, pc, perm, j, off, K)
            keys[j] = (w << ib) | UInt64(j - 1)
            d |= w ⊻ k1
        end
        ds[c] = d
    end
    return reduce(|, ds)
end

# The runs (length ≥ 2) of equal windows keys[j] >> ib, in order. In
# parallel: each chunk reports the runs that start in it, following a run past
# its end.
function equal_runs(keys, ib::Int, ntasks::Int)
    m = length(keys)
    cs = chunks(m, partasks(m, ntasks, PERM_PAR_MIN))
    parts = [UnitRange{Int}[] for _ in cs]
    tforeach_index(length(cs)) do c
        out = parts[c]
        j = first(cs[c])
        @inbounds while j <= last(cs[c])
            h = keys[j] >> ib
            if j > 1 && keys[j-1] >> ib == h     # continues a run that started before
                j += 1
                continue
            end
            k = j
            while k < m && keys[k+1] >> ib == h
                k += 1
            end
            k > j && push!(out, j:k)
            j = k + 1
        end
    end
    return reduce(vcat, parts)
end

# Stable insertion sort of a small run with a full comparison from bit `off`. With
# a prefix cache, keys that agree on [0, off) compare by their cached first 16
# bytes, and by length if both fit in them; only longer strings are read.
function small_refine!(ks::KeySource, perm::AbstractVector{Int}, off::Int, pc)
    m = length(perm)
    if pc === nothing || off >= 128
        for j in 2:m
            x = perm[j]
            k = j - 1
            while k >= 1 && cmp_from(ks, x, perm[k], off) < 0
                perm[k+1] = perm[k]
                k -= 1
            end
            perm[k+1] = x
        end
    else
        for j in 2:m
            x, px = perm[j], pc[j]
            k = j - 1
            while k >= 1 && cmp_cached(ks, x, px, perm[k], pc[k]) < 0
                perm[k+1] = perm[k]
                pc[k+1] = pc[k]
                k -= 1
            end
            perm[k+1] = x
            pc[k+1] = px
        end
    end
    return perm
end

@inline function cmp_cached(ks::AbstractByteKeys, a::Int, pa::UInt128, b::Int, pb::UInt128)
    if pa != pb
        return (pa < pb) != ks.rev ? -1 : 1
    end
    la, lb = bytelen(ks, a), bytelen(ks, b)
    (la <= 16 && lb <= 16) || return cmp_from(ks, a, b, 128)
    (!lentie(ks) || la == lb) && return 0
    return (la < lb) != ks.rev ? -1 : 1
end
cmp_cached(ks::KeySource, a, pa, b, pb) = cmp_from(ks, a, b, 0)

"""
    permute_by!(v, p, ntasks)

`v .= v[p]`, gathering and copying back with up to `ntasks` tasks.
"""
function permute_by!(v::AbstractVector{T}, p::Vector{Int}, ntasks::Int) where {T}
    arena = ScratchArena()
    try
        w = alloc!(arena, T, length(v))
        tforeach(length(v), ntasks) do r
            @inbounds for i in r
                w[i] = v[p[i]]
            end
        end
        tforeach(length(v), ntasks) do r
            @inbounds for i in r
                v[i] = w[i]
            end
        end
        return v
    finally
        release!(arena)
    end
end

"""
    simdsortperm(v; rev=false, ntasks=1) -> Vector{Int}

The permutation that sorts `v` stably (as `sortperm(v; rev)`), for vectors of
`Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64`, `String` and
`Symbol` (and the other types of [`simdsort!`](@ref)). Positions and key bits
are packed into `UInt64`s and sorted with the vectorized quicksort. Other
arrays of these types (views, matrix columns, ...) are sorted as a `Vector`
copy; other inputs fall back to `sortperm`.
"""
function simdsortperm(v::AbstractVector; rev::Bool=false, ntasks::Integer=1)
    # Other arrays of an element type with a fast path for Vectors (views,
    # matrix columns, ...): the permutation of a Vector copy, in v's indices.
    # (Not pooled arrays: their sortperm sorts the refs.)
    if !(v isa Vector) && vectorpath(eltype(v)) && !pooled(v) && length(v) > 1
        p = simdsortperm(collect(v); rev, ntasks)
        return firstindex(v) == 1 ? p : p .+ (firstindex(v) - 1)
    end
    if v isa Vector && has_orderwords(eltype(v))
        length(v) <= 1 && return collect(1:length(v))
        return with_words(w -> wordkeys_sortperm(w; rev, ntasks), orderwords, v, Int(ntasks))
    end
    return sortperm(v; rev)
end

function simdsortperm(v::KeyVector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:SortTypes}
    length(v) <= 1 && return collect(1:length(v))
    return refine!(NumKeys(v, rev), identity_perm(length(v), Int(ntasks)), 0, Int(ntasks); isidentity = true)
end

function simdsortperm(v::Vector{T}; rev::Bool=false,
                      ntasks::Integer=1) where {T<:Union{ByteString,Symbol}}
    length(v) <= 1 && return collect(1:length(v))
    if T === Symbol
        codes = symbol_codes(v, Int(ntasks))
        codes === nothing || return simdsortperm(codes; rev, ntasks)
    end
    n = length(v)
    arena = ScratchArena()
    try
        len = alloc!(arena, Int, n)
        pc = alloc!(arena, UInt128, n)
        GC.@preserve v begin
            ks = ByteKeys(v, rev, Int(ntasks), len)
            return refine!(ks, identity_perm(n, Int(ntasks)), 0, Int(ntasks), prefix_cache(ks, Int(ntasks), pc);
                           isidentity = true)
        end
    finally
        release!(arena)
    end
end

"""
    simdsort!(v::Vector{String}; rev=false, ntasks=1)
    simdsort!(v::Vector{Symbol}; rev=false, ntasks=1)

Sort strings or symbols in the order of `isless` (bytewise, shorter first on a
common prefix), through [`simdsortperm`](@ref).
"""
function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{ByteString,Symbol}}
    T === Symbol && symbol_sort!(v, ordering(rev), Int(ntasks)) && return v
    p = simdsortperm(v; rev, ntasks)
    permute_by!(v, p, Int(ntasks))
    return v
end
