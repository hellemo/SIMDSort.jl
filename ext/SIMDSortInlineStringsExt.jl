# InlineStrings.jl (String1 ... String255, as CSV.jl reads short strings): the
# string order (bytewise, a prefix first) as order words.
#
# An InlineString is a primitive value of 8sizeof(T) bits holding the bytes and
# the length, laid out so that one integer operation on the whole value gives an
# unsigned key in string order (what InlineStrings' own isless compares):
#   InlineStrings 2: byte i - 1 is codeunit i, the top byte the unused
#     capacity; key = bswap(x ⊻ (0xff in the top byte));
#   InlineStrings 1: codeunit 1 in the top byte, the length in the lowest;
#     key = x.
# Which one holds is detected when the extension is compiled, by comparing the
# keys' order with that of words built byte by byte (bytewords: the content
# big-endian, zero-padded, the length in the last byte) on sample strings of
# every type; if neither does, bytewords are used (5-10x slower: String31
# sortperm 72 ms against 11 ms, sort! 192 ms against 19 ms for 2e6 strings).
# The types of up to 128 bits (String1 ... String15) are also fixed-width
# components of tuple keys.
module SIMDSortInlineStringsExt

using SIMDSort, InlineStrings
import SIMDSort: orderwords, has_orderwords, lazy_orderwords, fixedwidth, fixedkey

# The types the layout is checked on: InlineStrings defines all widths with the
# same generated code, and checking String127/255 too (32-word tuples) made
# compiling the extension take 11 s instead of 2.
const TYPES = (String1, String3, String7, String15, String31, String63)

# The candidate keys, as values of the same primitive type.
@inline capmask(::Type{T}) where {T} = Base.shl_int(Base.zext_int(T, 0xff), 8 * (sizeof(T) - 1))
@inline key2(x::T) where {T<:InlineString} = Base.bswap_int(Base.xor_int(x, capmask(T)))
@inline key1(x::T) where {T<:InlineString} = x

# A key of 8sizeof(T) bits as big-endian 64-bit words.
@inline function keywords(k::T) where {T<:InlineString}
    if sizeof(T) == 2
        return (UInt64(reinterpret(UInt16, k)) << 48,)
    elseif sizeof(T) == 4
        return (UInt64(reinterpret(UInt32, k)) << 32,)
    end
    return reverse(reinterpret(NTuple{sizeof(T) ÷ 8,UInt64}, k))      # (little-endian words)
end

# The order words built byte by byte, independent of the layout.
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

# Sample strings, with NUL and high bytes, of the lengths where a layout could
# differ: the ends, and around each 8-byte word boundary (a few hundred per
# type; every length of String255 made compiling the extension take a minute).
function samples(::Type{T}) where {T}
    cap = sizeof(T) - 1
    lens = sort!(unique!(filter(n -> 0 <= n <= cap, [0; 1; 2; cap - 1; cap; [8k + d for k in 1:cap÷8 for d in -1:1]])))
    out = T[]
    for n in lens, c in ('\0', 'a', 'b', '\xff')
        push!(out, T(repeat("ab", cap)[1:n-min(n, 1)] * (n > 0 ? string(c) : "")))
        n > 1 && push!(out, T(string(c) * repeat("a", n - 1)))
    end
    return out
end

function orders_as_bytes(key)
    for T in TYPES
        s = samples(T)
        for a in s, b in s
            isless(keywords(key(a)), keywords(key(b))) == isless(bytewords(a), bytewords(b)) || return false
        end
    end
    return true
end

const LAYOUT = orders_as_bytes(key2) ? 2 : orders_as_bytes(key1) ? 1 : 0

@inline orderwords(x::InlineString) =
    LAYOUT == 2 ? keywords(key2(x)) : LAYOUT == 1 ? keywords(key1(x)) : bytewords(x)
has_orderwords(::Type{T}) where {T<:InlineString} = isconcretetype(T)
# sortperm computes the words where it reads them (one byte swap): faster than
# storing them for all strings first (2e6 String63s 11 ms against 24 ms).
lazy_orderwords(::Type{T}) where {T<:InlineString} = LAYOUT != 0

fixedwidth(::Type{T}) where {T<:InlineString} = isconcretetype(T) && sizeof(T) <= 16 ? 8sizeof(T) : nothing
@inline function fixedkey(x::T) where {T<:InlineString}
    w = orderwords(x)                   # (one or two words for these widths)
    k = length(w) == 1 ? UInt128(w[1]) << 64 : (UInt128(w[1]) << 64) | w[2]
    return k >> (128 - 8sizeof(T))      # the key's 8sizeof(T) bits
end

end
