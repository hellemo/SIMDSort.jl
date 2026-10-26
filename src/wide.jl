# Other key types on the windowed sort of perm.jl (docs/src/design.md §8): fixed-size
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
    T <: SmallTypes && return small_sortperm(v, rev, Int(ntasks))
    iswideint(T) || return sortperm(v; rev)
    length(v) <= 1 && return collect(1:length(v))
    return refine!(WideIntKeys(v, rev), identity_perm(length(v), Int(ntasks)), 0, Int(ntasks); isidentity = true)
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
    length(v) <= 1 && return v
    (8sizeof(T)) % 64 == 0 || return permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return sort_by_words!(v, intwords, rev, Int(ntasks))
end

# The order words of every element, computed by up to `ntasks` tasks.
words_of(f::F, v::AbstractVector, ntasks::Int) where {F} =
    words_of!(Vector{typeof(f(first(v)))}(undef, length(v)), f, v, ntasks)

function words_of!(w, f::F, v::AbstractVector, ntasks::Int) where {F}
    tforeach(length(v), ntasks) do r
        @inbounds for i in r
            w[i] = f(v[i])
        end
    end
    return w
end

# g(words of v), the words in a buffer outside the GC heap (they are plain
# bits; on the GC heap their hundreds of MB triggered collections).
function with_words(g::G, f::F, v::AbstractVector, ntasks::Int) where {G,F}
    arena = ScratchArena()
    try
        return g(words_of!(alloc!(arena, typeof(f(first(v))), length(v)), f, v, ntasks))
    finally
        release!(arena)
    end
end

# --- byte encodings ------------------------------------------------------------

# The number of 64-bit limbs of |x| (0 for 0).
nlimbs(x::BigInt) = Base.GMP.Limb === UInt64 ?
    Int(ccall((:__gmpz_size, Base.GMP.libgmp), Csize_t, (Ref{BigInt},), x)) :
    iszero(x) ? 0 : cld(Int(Base.GMP.MPZ.sizeinbase(x, 2)), 64)

# x as 8 big-endian bytes at bytes[pos+1:pos+8].
@inline function put64!(bytes::Vector{UInt8}, pos::Int, x::UInt64)
    @inbounds for k in 1:8
        bytes[pos+k] = (x >> (64 - 8k)) % UInt8
    end
    return bytes
end

# The magnitude of x (k = nlimbs(x) limbs) as 8k big-endian bytes at
# bytes[pos+1:pos+8k], complemented if neg. With 64-bit GMP limbs (64-bit
# platforms) limb by limb through GMP's documented mpz_getlimbn; otherwise
# with mpz_export, which writes whole 64-bit words, most significant first,
# big-endian, but is generic and costs about twice as much per number.
function put_magnitude!(bytes::Vector{UInt8}, pos::Int, x::BigInt, k::Int, neg::Bool)
    k == 0 && return bytes
    checkbounds(bytes, pos+1:pos+8k)
    if Base.GMP.Limb === UInt64
        for j in 1:k
            limb = ccall((:__gmpz_getlimbn, Base.GMP.libgmp), UInt64, (Ref{BigInt}, Clong), x, k - j)
            put64!(bytes, pos + 8(j - 1), neg ? ~limb : limb)
        end
        return bytes
    end
    GC.@preserve bytes ccall((:__gmpz_export, Base.GMP.libgmp), Ptr{Cvoid},
                             (Ptr{UInt8}, Ptr{Csize_t}, Cint, Csize_t, Cint, Csize_t, Ref{BigInt}),
                             pointer(bytes, pos + 1), C_NULL, 1, 8, 1, 0, x)
    if neg
        @inbounds for j in pos+1:pos+8k
            bytes[j] = ~bytes[j]
        end
    end
    return bytes
end

# Encoding runs in parallel over chunks of the vector from this many elements.
const ENCODE_PAR_MIN = 1 << 14

# BigInt: an 8-byte header 2^63 + nlimbs (≥ 0) or 2^63 - nlimbs (< 0), then the
# magnitude limbs from the most significant, complemented for negative numbers.
# A longer magnitude is larger (positive) or smaller (negative); equal lengths
# compare limb by limb.
function encode(v::Vector{BigInt}, ntasks::Int = 1)
    n = length(v)
    ks = Vector{Int}(undef, n)
    tforeach(n, ntasks; minlen = ENCODE_PAR_MIN) do r
        for i in r
            ks[i] = nlimbs(v[i])
        end
    end
    start = Vector{Int}(undef, n + 1)
    start[1] = 0
    for i in 1:n
        start[i+1] = start[i] + 8 + 8 * ks[i]
    end
    bytes = Vector{UInt8}(undef, start[end])
    tforeach(n, ntasks; minlen = ENCODE_PAR_MIN) do r
        for i in r
            x, k = v[i], ks[i]
            neg = signbit(x)
            put64!(bytes, start[i], neg ? (UInt64(1) << 63) - UInt64(k) : (UInt64(1) << 63) + UInt64(k))
            put_magnitude!(bytes, start[i] + 8, x, k, neg)
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
function encode(v::Vector{BigFloat}, ntasks::Int = 1)
    n = length(v)
    L = maximum(x -> cld(precision(x), 64), v; init = 1)
    w = 8 + 8L
    bytes = zeros(UInt8, n * w)
    start = collect(0:w:n*w)
    tforeach(n, ntasks; minlen = ENCODE_PAR_MIN) do r
        encode_floats!(bytes, start, v, r, L)
    end
    return bytes, start
end

function encode_floats!(bytes, start, v::Vector{BigFloat}, r::UnitRange{Int}, L::Int)
    z = BigInt()                               # the significand, reused
    for i in r
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
            neg = signbit(x)
            put64!(bytes, pos, neg ? (UInt64(1) << 62) - UInt64(E + (1 << 40)) :
                                     (UInt64(1) << 62) + 2 + UInt64(E + (1 << 40)))
            # Top-aligned in L limbs, in place: |z| in [2^(64L-1), 2^64L).
            ccall((:__gmpz_mul_2exp, Base.GMP.libgmp), Cvoid, (Ref{BigInt}, Ref{BigInt}, Culong), z, z, 64L - p)
            put_magnitude!(bytes, pos + 8, z, L, neg)
        end
    end
    return nothing
end

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{BigInt,BigFloat}}
    length(v) <= 1 && return collect(1:length(v))
    bytes, start = encode(v, Int(ntasks))
    GC.@preserve bytes begin
        return refine!(EncodedKeys(bytes, start, rev), identity_perm(length(v), Int(ntasks)), 0, Int(ntasks);
                       isidentity = true)
    end
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{BigInt,BigFloat}}
    permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return v
end

# --- multi-word keys -----------------------------------------------------------

"""
    orderwords(x) -> NTuple{M,UInt64}

The order key of `x` as `M` big-endian 64-bit words (most significant first),
compared lexicographically, for element types `T` with `has_orderwords(T) ==
true`. Package extensions define both (DoubleFloats, MultiFloats, Quadmath);
`simdsort!` and `simdsortperm` then sort such vectors by their words.
"""
function orderwords end

"""
    has_orderwords(T) -> Bool

Whether elements of type `T` define [`orderwords`](@ref), so that
`simdsort!` and `simdsortperm` sort vectors of them by their words. `false`
unless a method is added for `T`.
"""
has_orderwords(::Type) = false

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
    return refine!(WordKeys{M}(w, rev), identity_perm(length(w), Int(ntasks)), 0, Int(ntasks); isidentity = true)
end
