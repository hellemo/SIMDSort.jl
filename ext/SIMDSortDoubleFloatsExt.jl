# Double64 and Double32 keys: the high word, then the low word, each in the
# order-preserving unsigned form, as order words (sort and sortperm both take
# the generic path for types with orderwords, which computes them in
# parallel). DoubleFloats compares the high words with `<` (so -0.0 and
# 0.0 tie) and the low words with `<`; zeros are canonicalized to match. NaNs
# are put last, as for Float64. (DoubleFloats' own `isless` is not a total order
# with NaNs, so Base.sort has no defined result to match there.)
module SIMDSortDoubleFloatsExt

using SIMDSort, DoubleFloats
import SIMDSort: orderwords, has_orderwords

const DF = Union{DoubleFloat{Float64},DoubleFloat{Float32}}

@inline word(::Type{U}, x::AbstractFloat) where {U} = SIMDSort.ukey(U, iszero(x) ? zero(x) : x)

@inline function orderwords(x::DoubleFloat{Float64})
    isnan(x) && return (typemax(UInt64), typemax(UInt64))
    return (word(UInt64, x.hi), word(UInt64, x.lo))
end
@inline function orderwords(x::DoubleFloat{Float32})
    isnan(x) && return (typemax(UInt64),)
    return ((UInt64(word(UInt32, x.hi)) << 32) | UInt64(word(UInt32, x.lo)),)
end

has_orderwords(::Type{<:DF}) = true

end
