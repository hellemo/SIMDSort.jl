# Tuples, Pairs and VersionNumbers (docs/src/design.md §8).
#
# Tuples and Pairs compare lexicographically, component by component with
# `isless`. Their sort permutation is built from stable permutations of the
# components, from the last to the first (an LSD sort over components), each
# with simdsortperm. Any mix of component types works; unsupported component
# types fall back to `sortperm` for that component.

const TupleLike = Union{Tuple,Pair}

ncomponents(::Type{T}) where {T<:Tuple} = fieldcount(T)
ncomponents(::Type{<:Pair}) = 2
component(x::Tuple, k::Int) = x[k]
component(x::Pair, k::Int) = k == 1 ? x.first : x.second

# Fixed-width order keys of scalar components: `fixedwidth(T)` bits (or
# `nothing`), `fixedkey(x)` an unsigned integer whose order is `isless`'s.
# Extensions add methods (Dates).
fixedwidth(::Type) = nothing
fixedwidth(::Type{Bool}) = 8
fixedwidth(::Type{Char}) = 32
fixedwidth(::Type{T}) where {T<:Union{Base.BitInteger,Float32,Float64}} = 8sizeof(T)
@inline fixedkey(x::Bool) = UInt8(x)
@inline fixedkey(x::Char) = reinterpret(UInt32, x)
@inline fixedkey(x::T) where {T<:Union{Base.BitInteger,Float32,Float64}} = ukey(unsigned(signedkey_or_self(T)), x)
signedkey_or_self(::Type{T}) where {T<:Base.BitInteger} = T
signedkey_or_self(::Type{T}) where {T<:Union{Float32,Float64}} = signedkey(T)

componenttypes(::Type{T}) where {T<:Tuple} = fieldtypes(T)
componenttypes(::Type{Pair{A,B}}) where {A,B} = (A, B)

# Total key width of a tuple type whose components all have fixed-width keys.
function packedwidth(::Type{T}) where {T<:TupleLike}
    ws = map(fixedwidth, componenttypes(T))
    any(isnothing, ws) && return nothing
    return sum(ws)
end

# Place the w-bit key k at bits [pos, pos + w) of the big-endian words.
@inline function putbits(acc::NTuple{M,UInt64}, pos::Int, k::UInt128, w::Int) where {M}
    while w > 0
        j, r = pos >> 6, pos & 63
        take = min(64 - r, w)
        part = (k >> (w - take)) & ((UInt128(1) << take) - 1)
        acc = Base.setindex(acc, acc[j+1] | (UInt64(part) << (64 - r - take)), j + 1)
        pos += take
        w -= take
    end
    return acc
end

@inline packcomponents(acc, pos, ::Tuple{}) = acc
@inline function packcomponents(acc, pos, cs::Tuple)
    c = first(cs)
    w = fixedwidth(typeof(c))
    acc = putbits(acc, pos, UInt128(fixedkey(c)), w)
    return packcomponents(acc, pos + w, Base.tail(cs))
end

# Keys of at most 128 bits: shift the components into one UInt128 (straight-line
# code; putbits' loop indexes the word tuple at run time and is 6× slower).
@inline pack128(acc::UInt128, pos::Int, ::Tuple{}) = (acc, pos)
@inline function pack128(acc::UInt128, pos::Int, cs::Tuple)
    c = first(cs)
    w = fixedwidth(typeof(c))
    k = UInt128(fixedkey(c)) & (w >= 128 ? typemax(UInt128) : (UInt128(1) << w) - 1)
    return pack128((acc << w) | k, pos + w, Base.tail(cs))
end
@inline function packwords(x, ::Val{M}) where {M}
    if M <= 2
        acc, B = pack128(zero(UInt128), 0, astuple(x))
        acc <<= 128 - B
        return M == 1 ? ((acc >> 64) % UInt64,) : ((acc >> 64) % UInt64, acc % UInt64)
    end
    return packcomponents(ntuple(_ -> zero(UInt64), Val(M)), 0, astuple(x))
end

@inline astuple(x::Tuple) = x
@inline astuple(x::Pair) = (x.first, x.second)

function packed_words(v::Vector{T}, ::Val{M}, ntasks::Int) where {T,M}
    w = Vector{NTuple{M,UInt64}}(undef, length(v))
    tforeach(length(v), ntasks) do r
        @inbounds for i in r
            w[i] = packwords(v[i], Val(M))
        end
    end
    return w
end

# The packed order words of one tuple or pair, as a callable for sort_by_words!.
struct Packer{M} end
packer(::Val{M}) where {M} = Packer{M}()
@inline (::Packer{M})(x) where {M} = packwords(x, Val(M))

packed_tuple_sortperm(v::Vector{T}, ::Val{M}, rev::Bool, ntasks::Int) where {T,M} =
    wordkeys_sortperm(packed_words(v, Val(M), ntasks); rev, ntasks)

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:TupleLike}
    n = length(v)
    (isconcretetype(T) && ncomponents(T) > 0) || return sortperm(v; rev)
    n <= 1 && return collect(1:n)
    # All components with fixed-width keys: one sort of the packed keys.
    B = packedwidth(T)
    B === nothing || return packed_tuple_sortperm(v, Val(cld(B, 64)), rev, Int(ntasks))
    p = collect(1:n)
    for k in ncomponents(T):-1:1
        col = [component(v[i], k) for i in p]
        q = simdsortperm(col; rev, ntasks)
        p = p[q]
    end
    return p
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:TupleLike}
    (isconcretetype(T) && ncomponents(T) > 0) || return sort!(v; rev)
    B = packedwidth(T)
    if B !== nothing && length(v) > 1
        # Packed keys: distribute, then sort the buckets in cache (msd.jl).
        return sort_by_words!(v, packer(Val(cld(B, 64))), rev, Int(ntasks))
    end
    permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return v
end

# VersionNumber: (major, minor, patch), 32 bits each, packed into two 64-bit
# words and sorted with WordKeys. Runs that tie on all three and contain
# prerelease or build tags are then sorted stably with `isless`, which handles
# their rules (a prerelease sorts before the release, build tags after).
function simdsortperm(v::Vector{VersionNumber}; rev::Bool=false, ntasks::Integer=1)
    n = length(v)
    n <= 1 && return collect(1:n)
    w = [((UInt64(x.major) << 32) | UInt64(x.minor), UInt64(x.patch) << 32) for x in v]
    p = wordkeys_sortperm(w; rev, ntasks)
    j = 1
    while j <= n
        k = j
        while k < n && w[p[k+1]] == w[p[j]]
            k += 1
        end
        if k > j && any(i -> !isempty(v[i].prerelease) || !isempty(v[i].build), view(p, j:k))
            sort!(view(p, j:k); by = i -> v[i], rev, alg = Base.Sort.MergeSort)
        end
        j = k + 1
    end
    return p
end

function simdsort!(v::Vector{VersionNumber}; rev::Bool=false, ntasks::Integer=1)
    permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return v
end
