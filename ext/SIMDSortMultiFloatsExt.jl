# MultiFloats.jl's MultiFloat{T,N} (Float64x2, Float64x4, Float32x3, ...): the
# key is the order-preserving unsigned form of the leading limb, then of each
# lower limb, as big-endian 64-bit words. MultiFloats orders like Float64 (NaN
# last, -0.0 before 0.0 by the sign of the leading limb) and compares lower limbs
# numerically, so zeros in the lower limbs are canonicalized. NaNs go last.
module SIMDSortMultiFloatsExt

using SIMDSort, MultiFloats
import SIMDSort: orderwords, has_orderwords

const MF{T,N} = MultiFloat{T,N}

@inline ukey(x::Float64) = SIMDSort.ukey(UInt64, x)
@inline ukey(x::Float32) = SIMDSort.ukey(UInt32, x)
@inline canon(x) = iszero(x) ? zero(x) : x

# Number of 64-bit words for N limbs of type T.
nwords(::Type{Float64}, N) = N
nwords(::Type{Float32}, N) = cld(N, 2)

@generated function orderwords(x::MF{T,N}) where {T<:Union{Float32,Float64},N}
    M = nwords(T, N)
    B = 8sizeof(T)
    parts = [k == 1 ? :(ukey(l[1])) : :(ukey(canon(l[$k]))) for k in 1:N]
    if T === Float64
        tup = Expr(:tuple, parts...)
    else                        # two 32-bit limb keys per word
        tup = Expr(:tuple, [k + 1 <= N ? :((UInt64($(parts[k])) << 32) | UInt64($(parts[k+1]))) :
                                          :(UInt64($(parts[k])) << 32) for k in 1:2:N]...)
    end
    return quote
        l = x._limbs
        isnan(x) && return $(Expr(:tuple, fill(:(typemax(UInt64)), M)...))
        return $tup
    end
end

has_orderwords(::Type{<:MF{T}}) where {T<:Union{Float32,Float64}} = true

end
