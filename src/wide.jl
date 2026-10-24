# Other key types on the windowed sort of perm.jl (docs/DESIGN.md §8): fixed-size
# integers wider than 64 bits (Int128, UInt128, BitIntegers.jl types), and
# BigInt and BigFloat through order-preserving byte encodings.

# --- wide fixed-size integers --------------------------------------------------

struct WideIntKeys{T,U} <: KeySource
    v::Vector{T}
    rev::Bool
end
WideIntKeys(v::Vector{T}, rev::Bool) where {T} = WideIntKeys{T,typeof(unsigned(zero(T)))}(v, rev)

keywidth(::WideIntKeys{T}) where {T} = 8sizeof(T)
@inline exhausted(ks::WideIntKeys, i::Int, off::Int) = off >= keywidth(ks)

@inline function keybits(ks::WideIntKeys{T,U}, i::Int, off::Int, K::Int) where {T,U}
    W = 8sizeof(T)
    x = @inbounds ks.v[i]
    u = x % U
    T <: Signed && (u ⊻= one(U) << (W - 1))
    ks.rev && (u = ~u)
    off >= W && return zero(UInt64)
    # Bits [off, off + K) of the W-bit key, zero beyond the end.
    if off + K <= W
        return ((u >> (W - off - K)) % UInt64) & ((one(UInt64) << K) - one(UInt64))
    else
        return ((u << (off + K - W)) % UInt64) & ((one(UInt64) << K) - one(UInt64))
    end
end

# Fixed-size primitive integers wider than 64 bits.
iswideint(::Type{T}) where {T} =
    T <: Union{Signed,Unsigned} && isbitstype(T) && sizeof(T) > 8 && isprimitivetype(T)

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{Signed,Unsigned}}
    iswideint(T) || return sortperm(v; rev)
    length(v) <= 1 && return collect(1:length(v))
    return refine!(WideIntKeys(v, rev), identity_perm(length(v), Int(ntasks)), 0, Int(ntasks))
end

# Order words of a wide integer: its unsigned form (sign bit flipped if signed),
# most significant 64 bits first.
# (A plain function, not @generated: a generator could not call methods of
# integer types defined in packages loaded later, such as BitIntegers.)
@generated nwords(::Type{T}) where {T} = :(Val($(8sizeof(T) ÷ 64)))   # sizeof is a builtin

@inline function intwords(x::T) where {T}
    W = 8sizeof(T)
    u0 = x % typeof(unsigned(x))
    # One assignment: the closure below must not capture a reassigned variable.
    u = T <: Signed ? u0 ⊻ (one(u0) << (W - 1)) : u0
    return ntuple(k -> (u >> (W - 64k)) % UInt64, nwords(T))
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{Signed,Unsigned}}
    iswideint(T) || return sort!(v; rev)
    (8sizeof(T)) % 64 == 0 || return permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return sort_by_words!(v, intwords, rev, Int(ntasks))
end

# The order words of every element, computed by up to `ntasks` tasks.
function words_of(f::F, v::AbstractVector, ntasks::Int) where {F}
    w = Vector{typeof(f(first(v)))}(undef, length(v))
    foreach_range(length(v), ntasks) do r
        @inbounds for i in r
            w[i] = f(v[i])
        end
    end
    return w
end

# --- byte encodings ------------------------------------------------------------

# Keys given as one byte string each, stored back to back in `bytes`; `lentie`
# is false because equal values have identical encodings.
function encoded_keys(bytes::Vector{UInt8}, start::Vector{Int}, rev::Bool)
    n = length(start) - 1
    base = pointer(bytes)
    ptr = [base + start[i] for i in 1:n]
    len = [start[i+1] - start[i] for i in 1:n]
    return ByteKeys(ptr, len, rev, false)
end

@inline put64!(bytes::Vector{UInt8}, pos::Int, x::UInt64) =
    GC.@preserve bytes unsafe_store!(Ptr{UInt64}(pointer(bytes, pos + 1)), hton(x))

const LIMB = Base.GMP.Limb

# BigInt: an 8-byte header 2^63 + nlimbs (≥ 0) or 2^63 - nlimbs (< 0), then the
# magnitude limbs from the most significant, complemented for negative numbers.
# A longer magnitude is larger (positive) or smaller (negative); equal lengths
# compare limb by limb.
function encode(v::Vector{BigInt})
    n = length(v)
    start = Vector{Int}(undef, n + 1)
    start[1] = 0
    for i in 1:n
        start[i+1] = start[i] + 8 + 8 * abs(v[i].size)
    end
    bytes = Vector{UInt8}(undef, start[end])
    for i in 1:n
        x = v[i]
        k = Int(abs(x.size))
        neg = x.size < 0
        put64!(bytes, start[i], neg ? (UInt64(1) << 63) - UInt64(k) : (UInt64(1) << 63) + UInt64(k))
        GC.@preserve x for j in 1:k
            limb = UInt64(unsafe_load(x.d, k - j + 1))
            put64!(bytes, start[i] + 8j, neg ? ~limb : limb)
        end
    end
    return bytes, start
end

# BigFloat: an 8-byte class/exponent header, then the significand as L limbs
# (L for the largest precision in the vector), aligned at the top bit and
# complemented for negative numbers. Equal lengths make the complemented
# encodings compare correctly.
#   -Inf: 1, negative: 2^62 - (E + 2^40), -0: 2^62, +0: 2^62 + 1,
#   positive: 2^62 + 2 + (E + 2^40), +Inf: 2^63, NaN: 2^64 - 1
# with E the binary exponent (x = 0.1xxx₂ × 2^E).
function encode(v::Vector{BigFloat})
    n = length(v)
    L = maximum(x -> cld(precision(x), 64), v; init = 1)
    w = 8 + 8L
    bytes = zeros(UInt8, n * w)
    start = collect(0:w:n*w)
    z = BigInt()
    for i in 1:n
        x = v[i]
        pos = start[i]
        if isnan(x)
            put64!(bytes, pos, typemax(UInt64))
        elseif isinf(x)
            put64!(bytes, pos, x > 0 ? UInt64(1) << 63 : UInt64(1))
        elseif iszero(x)
            put64!(bytes, pos, (UInt64(1) << 62) + UInt64(!signbit(x)))
        else
            e = ccall((:mpfr_get_z_2exp, Base.MPFR.libmpfr), Clong, (Ref{BigInt}, Ref{BigFloat}), z, x)
            p = precision(x)
            E = Int(e) + p                     # x = (z / 2^p) × 2^E, |z| in [2^(p-1), 2^p)
            neg = x < 0
            put64!(bytes, pos, neg ? (UInt64(1) << 62) - UInt64(E + (1 << 40)) :
                                     (UInt64(1) << 62) + 2 + UInt64(E + (1 << 40)))
            m = abs(z) << (64L - p)            # top-aligned in L limbs
            k = Int(abs(m.size))
            GC.@preserve m for j in 1:L
                limb = j <= k ? UInt64(unsafe_load(m.d, k - j + 1)) : zero(UInt64)
                put64!(bytes, pos + 8j, neg ? ~limb : limb)
            end
        end
    end
    return bytes, start
end

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{BigInt,BigFloat}}
    length(v) <= 1 && return collect(1:length(v))
    bytes, start = encode(v)
    GC.@preserve bytes begin
        return refine!(encoded_keys(bytes, start, rev), collect(1:length(v)), 0, Int(ntasks))
    end
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{BigInt,BigFloat}}
    permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return v
end

# --- multi-word keys -----------------------------------------------------------

"""
    WordKeys(w::Vector{NTuple{M,UInt64}}, rev)

Keys given as `M` big-endian 64-bit words each (most significant first), for
types whose order key is wider than 128 bits, such as MultiFloats with three or
more limbs. Package extensions build the words.
"""
struct WordKeys{M,V<:AbstractVector{NTuple{M,UInt64}}} <: KeySource
    w::V
    rev::Bool
end
WordKeys{M}(w::AbstractVector{NTuple{M,UInt64}}, rev::Bool) where {M} = WordKeys{M,typeof(w)}(w, rev)

keywidth(::WordKeys{M}) where {M} = 64M
@inline exhausted(ks::WordKeys, i::Int, off::Int) = off >= keywidth(ks)

# Bits [off, off + K) (K ≤ 63) lie within words q and q + 1, q = off ÷ 64.
@inline function keybits(ks::WordKeys{M}, i::Int, off::Int, K::Int) where {M}
    off >= 64M && return zero(UInt64)
    w = @inbounds ks.w[i]
    q = off >> 6
    hi = @inbounds w[q+1]
    lo = q + 2 <= M ? @inbounds(w[q+2]) : zero(UInt64)
    if ks.rev
        hi, lo = ~hi, q + 2 <= M ? ~lo : zero(UInt64)
    end
    x = ((UInt128(hi) << 64 | UInt128(lo)) << (off & 63)) >> (128 - K)
    return x % UInt64
end

"""
    wordkeys_sortperm(w::Vector{NTuple{M,UInt64}}; rev=false, ntasks=1)

Stable sort permutation of multi-word keys; see [`WordKeys`](@ref).
"""
function wordkeys_sortperm(w::AbstractVector{NTuple{M,UInt64}}; rev::Bool=false, ntasks::Integer=1) where {M}
    length(w) <= 1 && return collect(1:length(w))
    return refine!(WordKeys{M}(w, rev), collect(1:length(w)), 0, Int(ntasks))
end
