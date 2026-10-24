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
# nwords is generated so that the word count is a compile-time Val; it uses only
# sizeof, a builtin, because a generator cannot call methods of integer types
# defined in packages loaded later, such as BitIntegers.
@generated nwords(::Type{T}) where {T} = :(Val($(8sizeof(T) ÷ 64)))

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
    tforeach(length(v), ntasks) do r
        @inbounds for i in r
            w[i] = f(v[i])
        end
    end
    return w
end

# --- byte encodings ------------------------------------------------------------

# The number of 64-bit limbs of |x| (0 for 0).
nlimbs(x::BigInt) = iszero(x) ? 0 : cld(Int(Base.GMP.MPZ.sizeinbase(x, 2)), 64)

# x as 8 big-endian bytes at bytes[pos+1:pos+8].
@inline function put64!(bytes::Vector{UInt8}, pos::Int, x::UInt64)
    @inbounds for k in 1:8
        bytes[pos+k] = (x >> (64 - 8k)) % UInt8
    end
    return bytes
end

# The magnitude of x as nbytes big-endian bytes at bytes[pos+1:pos+nbytes],
# right-aligned (GMP's documented mpz_export; |x| < 2^(8nbytes)), complemented
# if neg.
function put_magnitude!(bytes::Vector{UInt8}, pos::Int, x::BigInt, nbytes::Int, neg::Bool)
    used = Base.GMP.MPZ.sizeinbase(x, 256)
    iszero(x) && (used = 0)
    fill!(view(bytes, pos+1:pos+nbytes-used), 0x00)
    if used > 0
        GC.@preserve bytes ccall((:__gmpz_export, Base.GMP.libgmp), Ptr{Cvoid},
                                 (Ptr{UInt8}, Ptr{Csize_t}, Cint, Csize_t, Cint, Csize_t, Ref{BigInt}),
                                 pointer(bytes, pos + nbytes - used + 1), C_NULL, 1, 1, 1, 0, x)
    end
    if neg
        @inbounds for k in pos+1:pos+nbytes
            bytes[k] = ~bytes[k]
        end
    end
    return bytes
end

# BigInt: an 8-byte header 2^63 + nlimbs (≥ 0) or 2^63 - nlimbs (< 0), then the
# magnitude limbs from the most significant, complemented for negative numbers.
# A longer magnitude is larger (positive) or smaller (negative); equal lengths
# compare limb by limb.
function encode(v::Vector{BigInt})
    n = length(v)
    start = Vector{Int}(undef, n + 1)
    start[1] = 0
    for i in 1:n
        start[i+1] = start[i] + 8 + 8 * nlimbs(v[i])
    end
    bytes = Vector{UInt8}(undef, start[end])
    for i in 1:n
        x = v[i]
        k = nlimbs(x)
        neg = x < 0
        put64!(bytes, start[i], neg ? (UInt64(1) << 63) - UInt64(k) : (UInt64(1) << 63) + UInt64(k))
        put_magnitude!(bytes, start[i] + 8, x, 8k, neg)
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
            put_magnitude!(bytes, pos + 8, m, 8L, neg)
        end
    end
    return bytes, start
end

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{BigInt,BigFloat}}
    length(v) <= 1 && return collect(1:length(v))
    bytes, start = encode(v)
    GC.@preserve bytes begin
        return refine!(EncodedKeys(bytes, start, rev), collect(1:length(v)), 0, Int(ntasks))
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

# Bits [off, off + K) (K ≤ 63) lie within words q and q + 1, q = off ÷ 64.
@inline keybits(ks::WordKeys, i::Int, off::Int, K::Int) = wordbits(@inbounds(ks.w[i]), off, K, ks.rev)

# Bits [off, off + K) (K ≤ 63) of the order words w, complemented for rev.
@inline function wordbits(w::NTuple{M,UInt64}, off::Int, K::Int, rev::Bool) where {M}
    off >= 64M && return zero(UInt64)
    q = off >> 6
    hi = @inbounds w[q+1]
    lo = q + 2 <= M ? @inbounds(w[q+2]) : zero(UInt64)
    if rev
        hi, lo = ~hi, q + 2 <= M ? ~lo : zero(UInt64)
    end
    return (((UInt128(hi) << 64 | UInt128(lo)) << (off & 63)) >> (128 - K)) % UInt64
end

"""
    wordkeys_sortperm(w::Vector{NTuple{M,UInt64}}; rev=false, ntasks=1)

Stable sort permutation of multi-word keys; see [`WordKeys`](@ref).
"""
function wordkeys_sortperm(w::AbstractVector{NTuple{M,UInt64}}; rev::Bool=false, ntasks::Integer=1) where {M}
    length(w) <= 1 && return collect(1:length(w))
    return refine!(WordKeys{M}(w, rev), collect(1:length(w)), 0, Int(ntasks))
end
