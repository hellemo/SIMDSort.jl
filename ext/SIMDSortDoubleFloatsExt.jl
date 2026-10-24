# Double64 and Double32 keys: the high word, then the low word, each in the
# order-preserving unsigned form, packed into one UInt128 (UInt64) and sorted
# with simdsortperm. DoubleFloats compares the high words with `<` (so -0.0 and
# 0.0 tie) and the low words with `<`; zeros are canonicalized to match. NaNs
# are put last, as for Float64. (DoubleFloats' own `isless` is not a total order
# with NaNs, so Base.sort has no defined result to match there.)
module SIMDSortDoubleFloatsExt

using SIMDSort, DoubleFloats
import SIMDSort: simdsort!, simdsortperm

const DF = Union{DoubleFloat{Float64},DoubleFloat{Float32}}

@inline word(::Type{U}, x::AbstractFloat) where {U} = SIMDSort.ukey(U, iszero(x) ? zero(x) : x)

function packed(v::Vector{DoubleFloat{T}}) where {T}
    W = 8sizeof(T)
    U = T === Float64 ? UInt64 : UInt32
    P = T === Float64 ? UInt128 : UInt64
    return P[isnan(x) ? typemax(P) : (P(word(U, x.hi)) << W) | P(word(U, x.lo)) for x in v]
end

simdsortperm(v::Vector{<:DF}; rev::Bool=false, ntasks::Integer=1) =
    simdsortperm(packed(v); rev, ntasks)

# Order words for the MSD distribution of sort! (SIMDSort.sort_by_words!).
@inline function dfwords(x::DoubleFloat{Float64})
    isnan(x) && return (typemax(UInt64), typemax(UInt64))
    return (word(UInt64, x.hi), word(UInt64, x.lo))
end
@inline function dfwords(x::DoubleFloat{Float32})
    isnan(x) && return (typemax(UInt64),)
    return ((UInt64(word(UInt32, x.hi)) << 32) | UInt64(word(UInt32, x.lo)),)
end

function simdsort!(v::Vector{<:DF}; rev::Bool=false, ntasks::Integer=1)
    length(v) <= 1 && return v
    nt = Int(ntasks)
    return SIMDSort.sort_by_words!(v, dfwords, rev, nt)
end

end
