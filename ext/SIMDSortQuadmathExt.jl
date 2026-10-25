# Quadmath.jl's Float128: the order-preserving unsigned form of its 128 bits (as
# for Float64: flip all bits of negatives, set the sign bit of the others), with
# every NaN last, as two order words (sort and sortperm both take the generic
# path for types with orderwords).
module SIMDSortQuadmathExt

using SIMDSort, Quadmath
import SIMDSort: orderwords, has_orderwords

@inline function key(x::Float128)
    isnan(x) && return typemax(UInt128)
    b = reinterpret(UInt128, x)
    m = one(UInt128) << 127
    return b & m == 0 ? b | m : ~b
end

@inline orderwords(x::Float128) = (k = key(x); ((k >> 64) % UInt64, k % UInt64))
has_orderwords(::Type{Float128}) = true

end
