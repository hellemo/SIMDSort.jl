# sortperm, and sorting strings and symbols, on top of the UInt64 quicksort
# (docs/DESIGN.md §8).
#
# Every key is viewed as a big-endian bit string: numbers as their 32 or 64 bits in
# an order-preserving unsigned form, strings as their bytes followed by zeros.
# A window of K bits of each key goes into the high bits of a UInt64, the
# element's position into the low bits, and the UInt64s are sorted. Runs whose
# windows are equal are refined with the next window, as in an MSD radix sort.
# Positions in the low bits make every step stable.

"""Key sources: `keybits(ks, i, off, K)` returns bits `[off, off + K)` of key `i`."""
abstract type KeySource end

# --- numbers -------------------------------------------------------------------

struct NumKeys{T,U} <: KeySource
    v::Vector{T}
    rev::Bool
end
NumKeys(v::Vector{T}, rev::Bool) where {T} = NumKeys{T,unsigned(signedkey(T))}(v, rev)

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
@inline exhausted(ks::NumKeys, i::Int, off::Int) = off >= keywidth(ks)

@inline function keybits(ks::NumKeys{T,U}, i::Int, off::Int, K::Int) where {T,U}
    u = UInt64(ukey(U, @inbounds ks.v[i])) << (64 - 8sizeof(U))   # left-aligned
    ks.rev && (u = ~u >> (64 - 8sizeof(U)) << (64 - 8sizeof(U)))
    return off >= 64 ? zero(UInt64) : (u << off) >> (64 - K)
end

# --- strings and symbols -------------------------------------------------------

struct ByteKeys <: KeySource
    ptr::Vector{Ptr{UInt8}}
    len::Vector{Int}
    rev::Bool
    lentie::Bool    # byte-equal keys of different length differ (strings)
end

# The bytes of a String, or the name of a Symbol (NUL-free, NUL-terminated). The
# caller keeps the vector of values alive while the pointers are used.
bytes_of(s::String) = (pointer(s), ncodeunits(s))
bytes_of(s::Symbol) = (p = Base.unsafe_convert(Ptr{UInt8}, s); (p, Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), p))))

function ByteKeys(v::AbstractVector, rev::Bool, ntasks::Int)
    n = length(v)
    ptr = Vector{Ptr{UInt8}}(undef, n)
    len = Vector{Int}(undef, n)
    foreach_range(n, ntasks) do r
        @inbounds for i in r
            ptr[i], len[i] = bytes_of(v[i])
        end
    end
    return ByteKeys(ptr, len, rev, true)
end

@inline exhausted(ks::ByteKeys, i::Int, off::Int) = off >= 8 * @inbounds(ks.len[i])

# Bits [off, off + K) of the zero-padded byte string, K ≤ 63. They lie within the
# 9 bytes from byte off ÷ 8: the first 8 come from one big-endian load when they
# are all inside the string, the ninth only supplies the low bits for a shift.
# Bytes past the end read as zero; no byte past the end is loaded.
@inline function keybits(ks::ByteKeys, i::Int, off::Int, K::Int)
    p = @inbounds ks.ptr[i]
    len = @inbounds ks.len[i]
    b = off >> 3
    s = off & 7
    if b + 8 <= len
        hi = ntoh(unsafe_load(Ptr{UInt64}(p + b)))
    else
        hi = zero(UInt64)
        for j in 0:min(7, len - b - 1)
            hi |= UInt64(unsafe_load(p, b + j + 1)) << (56 - 8j)
        end
    end
    lo = b + 8 < len ? unsafe_load(p, b + 9) : 0x00
    x = s == 0 ? hi : (hi << s) | (UInt64(lo) >> (8 - s))
    x >>= 64 - K
    return ks.rev ? ~x & ((one(UInt64) << K) - one(UInt64)) : x
end

# Strings equal as zero-padded bytes differ only in length; `isless` puts the
# shorter first.
finish_ties!(ks::KeySource, perm) = perm
function finish_ties!(ks::ByteKeys, perm)
    ks.lentie || return perm
    # Usually all lengths are equal (identical strings): nothing to do.
    l1 = @inbounds ks.len[perm[1]]
    all(i -> @inbounds(ks.len[i]) == l1, perm) && return perm
    return sort!(perm; by = i -> @inbounds(ks.len[i]), rev = ks.rev, alg = Base.Sort.MergeSort)
end

# Full comparison of keys a and b from bit `off` on, for small runs: -1, 0 or 1.
function cmp_from(ks::KeySource, a::Int, b::Int, off::Int)
    while !(exhausted(ks, a, off) && exhausted(ks, b, off))
        x, y = keybits(ks, a, off, 56), keybits(ks, b, off, 56)
        x != y && return x < y ? -1 : 1
        off += 56
    end
    (ks isa ByteKeys && ks.lentie) || return 0
    la, lb = ks.len[a], ks.len[b]
    la == lb && return 0
    return (la < lb) != ks.rev ? -1 : 1
end

# --- the windowed sort ---------------------------------------------------------

const PERM_SMALL = 64       # runs up to this size are finished by insertion sort
const PERM_PAR_MIN = 1 << 14

# Apply `f(r)` to about `ntasks` contiguous ranges covering 1:n, concurrently.
function foreach_range(f::F, n::Int, ntasks::Int) where {F}
    if ntasks <= 1 || n < PERM_PAR_MIN
        f(1:n)
    else
        @sync for r in Iterators.partition(1:n, cld(n, ntasks))
            Threads.@spawn f(r)
        end
    end
    return nothing
end

"""
    refine!(ks, perm, off, ntasks)

Reorder `perm` (positions whose keys agree on bits `[0, off)`, in a stable
order) by the keys' bits from `off` on, stably.
"""
# The first 16 bytes of each byte string, big-endian and zero-padded. Kept aligned
# with `perm` and permuted along with it, so that windows within them are read
# sequentially instead of through the string pointers.
function prefix_cache(ks::ByteKeys, ntasks::Int)
    n = length(ks.ptr)
    pc = Vector{UInt128}(undef, n)
    foreach_range(n, ntasks) do r
        @inbounds for i in r
            pc[i] = (UInt128(keybits_raw(ks, i, 0)) << 64) | UInt128(keybits_raw(ks, i, 64))
        end
    end
    return pc
end

# The 64 bits from bit `off` of byte string i, not complemented for `rev`.
@inline keybits_raw(ks::ByteKeys, i::Int, off::Int) =
    (keybits(ByteKeys(ks.ptr, ks.len, false, ks.lentie), i, off, 32) << 32) |
    keybits(ByteKeys(ks.ptr, ks.len, false, ks.lentie), i, off + 32, 32)

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

function refine!(ks::KeySource, perm::AbstractVector{Int}, off::Int, ntasks::Int, pc = nothing)
    m = length(perm)
    m <= 1 && return perm
    if all_exhausted(ks, perm, off)
        return finish_ties!(ks, perm)
    end
    m <= PERM_SMALL && return small_refine!(ks, perm, off, pc)
    ib = 8sizeof(Int) - leading_zeros(m - 1)       # bits for local positions
    K = 64 - ib
    mask = (one(UInt64) << ib) - one(UInt64)
    keys = Vector{UInt64}(undef, m)
    while true
        woff = off                  # a fresh binding per iteration for the closure
        foreach_range(m, ntasks) do r
            @inbounds for j in r
                keys[j] = (window(ks, pc, perm, j, woff, K) << ib) | UInt64(j - 1)
            end
        end
        # Skip bits that all keys of the run share (long common prefixes, as in
        # "id%010d"): a window with no differences needs no sort.
        k1 = keys[1] >> ib
        d = mapreduce(x -> (x >> ib) ⊻ k1, |, keys)
        if d == 0
            off += K
            all_exhausted(ks, perm, off) && return finish_ties!(ks, perm)
            continue
        end
        c = leading_zeros(d) - (64 - K)          # common leading bits in the window
        c < 8 && break
        off += c
    end
    simdsort!(keys; ntasks)
    old = copy(perm)
    foreach_range(m, ntasks) do r
        @inbounds for j in r
            perm[j] = old[(keys[j] & mask) % Int + 1]
        end
    end
    # Refine the runs with equal windows.
    runs = UnitRange{Int}[]
    j = 1
    @inbounds while j <= m
        h = keys[j] >> ib
        k = j
        while k < m && keys[k+1] >> ib == h
            k += 1
        end
        k > j && push!(runs, j:k)
        j = k + 1
    end
    isempty(runs) && return perm
    next = off + K
    # Each run gets its own slice of the prefix cache, gathered in its new order.
    usepc = pc !== nothing && next < 128
    sub(r) = usepc ? UInt128[@inbounds pc[(keys[j] & mask) % Int + 1] for j in r] : nothing
    if ntasks <= 1 || length(runs) == 1
        for r in runs
            refine!(ks, view(perm, r), next, length(r) > PERM_PAR_MIN ? ntasks : 1, sub(r))
        end
    else
        # Groups of runs of about equal total size, one task each.
        total = sum(length, runs)
        target = cld(total, 4ntasks)
        @sync begin
            g = 1
            while g <= length(runs)
                h, s = g, 0
                while h <= length(runs) && (s == 0 || s + length(runs[h]) <= target)
                    s += length(runs[h])
                    h += 1
                end
                group = runs[g:h-1]
                Threads.@spawn for r in $group
                    refine!(ks, view(perm, r), next, 1, sub(r))
                end
                g = h
            end
        end
    end
    return perm
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

@inline function cmp_cached(ks::ByteKeys, a::Int, pa::UInt128, b::Int, pb::UInt128)
    if pa != pb
        return (pa < pb) != ks.rev ? -1 : 1
    end
    la, lb = @inbounds(ks.len[a]), @inbounds(ks.len[b])
    (la <= 16 && lb <= 16) || return cmp_from(ks, a, b, 128)
    (!ks.lentie || la == lb) && return 0
    return (la < lb) != ks.rev ? -1 : 1
end
cmp_cached(ks::KeySource, a, pa, b, pb) = cmp_from(ks, a, b, 0)

"""
    permute_by!(v, p, ntasks)

`v .= v[p]`, gathering and copying back with up to `ntasks` tasks.
"""
function permute_by!(v::AbstractVector, p::Vector{Int}, ntasks::Int)
    w = similar(v)
    foreach_range(length(v), ntasks) do r
        @inbounds for i in r
            w[i] = v[p[i]]
        end
    end
    foreach_range(length(v), ntasks) do r
        @inbounds for i in r
            v[i] = w[i]
        end
    end
    return v
end

"""
    simdsortperm(v; rev=false, ntasks=1) -> Vector{Int}

The permutation that sorts `v` stably (as `sortperm(v; rev)`), for vectors of
`Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64`, `String` and
`Symbol`. Positions and key bits are packed into `UInt64`s and sorted with the
vectorized quicksort; other inputs fall back to `sortperm`.
"""
simdsortperm(v::AbstractVector; rev::Bool=false, ntasks::Integer=1) = sortperm(v; rev)

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:SortTypes}
    length(v) <= 1 && return collect(1:length(v))
    return refine!(NumKeys(v, rev), collect(1:length(v)), 0, Int(ntasks))
end

function simdsortperm(v::Vector{T}; rev::Bool=false,
                      ntasks::Integer=1) where {T<:Union{String,Symbol}}
    length(v) <= 1 && return collect(1:length(v))
    if T === Symbol
        r = symbol_ranks(v, Int(ntasks))
        r === nothing || return simdsortperm(r[1]; rev, ntasks)
    end
    GC.@preserve v begin
        ks = ByteKeys(v, rev, Int(ntasks))
        return refine!(ks, collect(1:length(v)), 0, Int(ntasks), prefix_cache(ks, Int(ntasks)))
    end
end

"""
    symbol_ranks(v, ntasks) -> (ranks::Vector{Int32}, sorted_unique::Vector{Symbol}) or nothing

Symbols are interned, so equal names are the same object: a hash table from
symbol to code, the distinct symbols sorted by name, and their ranks. Gives up
(returns `nothing`) when there are more than `length(v) ÷ 4` distinct symbols.
"""
function symbol_ranks(v::Vector{Symbol}, ntasks::Int)
    n = length(v)
    n < 1024 && return nothing
    limit = n ÷ 4
    code = Dict{Symbol,Int32}()
    codes = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        c = get!(code, v[i], Int32(length(code) + 1))
        length(code) > limit && return nothing
        codes[i] = c
    end
    uniq = Vector{Symbol}(undef, length(code))
    for (s, c) in code
        uniq[c] = s
    end
    order = GC.@preserve uniq begin
        ks = ByteKeys(uniq, false, 1)
        refine!(ks, collect(1:length(uniq)), 0, 1, prefix_cache(ks, 1))
    end
    rank = Vector{Int32}(undef, length(uniq))
    for (r, c) in enumerate(order)
        rank[c] = Int32(r)
    end
    foreach_range(n, ntasks) do rg
        @inbounds for i in rg
            codes[i] = rank[codes[i]]
        end
    end
    return codes, uniq[order]
end

"""
    simdsort!(v::Vector{String}; rev=false, ntasks=1)
    simdsort!(v::Vector{Symbol}; rev=false, ntasks=1)

Sort strings or symbols in the order of `isless` (bytewise, shorter first on a
common prefix), through [`simdsortperm`](@ref).
"""
function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{String,Symbol}}
    if T === Symbol
        r = symbol_ranks(v, Int(ntasks))
        if r !== nothing
            ranks, sorted = r
            simdsort!(ranks; rev, ntasks)
            foreach_range(length(v), Int(ntasks)) do rg
                @inbounds for i in rg
                    v[i] = sorted[ranks[i]]
                end
            end
            return v
        end
    end
    p = simdsortperm(v; rev, ntasks)
    permute_by!(v, p, Int(ntasks))
    return v
end
