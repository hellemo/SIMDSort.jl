# Quadmath.jl's Float128: the order-preserving unsigned form of its 128 bits (as
# for Float64: flip all bits of negatives, set the sign bit of the others), with
# every NaN last, sorted as UInt128 keys.
module SIMDQuickSortQuadmathExt

using SIMDQuickSort, Quadmath
import SIMDQuickSort: simdsort!, simdsortperm

@inline function key(x::Float128)
    isnan(x) && return typemax(UInt128)
    b = reinterpret(UInt128, x)
    m = one(UInt128) << 127
    return b & m == 0 ? b | m : ~b
end

simdsortperm(v::Vector{Float128}; rev::Bool=false, ntasks::Integer=1) =
    simdsortperm(map(key, v); rev, ntasks)

function simdsort!(v::Vector{Float128}; rev::Bool=false, ntasks::Integer=1)
    SIMDQuickSort.permute_by!(v, simdsortperm(v; rev, ntasks), Int(ntasks))
    return v
end

end
