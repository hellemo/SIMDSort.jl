# InlineStrings.jl (String1 ... String255, as CSV.jl reads short strings): the
# string order (bytewise, a prefix first) as order words: the content bytes
# big-endian and zero-padded, then the length in the last byte of the key, so
# that of two strings equal up to the padding the shorter comes first. Built
# from codeunit, not from the types' internal layout (which changed between
# InlineStrings 1 and 2); for String1 ... String15, InlineStrings 2 defines
# its own unsigned order key (Base.Sort.uint_map, for Base's radix sort),
# which is cheaper and used when present. Those types (up to 128 bits) are
# also fixed-width components of packed tuple keys.
module SIMDSortInlineStringsExt

using SIMDSort, InlineStrings
import SIMDSort: orderwords, has_orderwords, fixedwidth, fixedkey

const Short = Union{String1,String3,String7,String15}
const HAS_UINTMAP = hasmethod(Base.Sort.uint_map, Tuple{String7,Base.Order.ForwardOrdering})

@inline function orderwords(x::T) where {T<:Short}
    HAS_UINTMAP || return bytewords(x)
    u = Base.Sort.uint_map(x, Base.Order.Forward)             # an unsigned key of 8sizeof(T) bits
    k = UInt128(u) << (128 - 8sizeof(T))
    return sizeof(T) <= 8 ? ((k >> 64) % UInt64,) : ((k >> 64) % UInt64, k % UInt64)
end
@inline orderwords(x::InlineString) = bytewords(x)

@inline function bytewords(x::T) where {T<:InlineString}
    n = ncodeunits(x)
    B = sizeof(T)                       # key bytes: content, then the length
    return ntuple(Val(cld(sizeof(T), 8))) do k
        w = zero(UInt64)
        for j in 1:8
            i = 8(k - 1) + j            # byte i of the key
            b = i <= n ? UInt64(codeunit(x, i)) : i == B ? UInt64(n) : zero(UInt64)
            w |= b << (64 - 8j)
        end
        w
    end
end
has_orderwords(::Type{T}) where {T<:InlineString} = isconcretetype(T)

fixedwidth(::Type{T}) where {T<:InlineString} = isconcretetype(T) && sizeof(T) <= 16 ? 8sizeof(T) : nothing
@inline function fixedkey(x::T) where {T<:InlineString}
    w = orderwords(x)                   # (one or two words for these widths)
    k = length(w) == 1 ? UInt128(w[1]) << 64 : (UInt128(w[1]) << 64) | w[2]
    return k >> (128 - 8sizeof(T))      # the key's 8sizeof(T) bits
end

end
