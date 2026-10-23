# Order-preserving, in-place, invertible maps from the supported element types to
# signed integers of the same width. AVX2 only has signed integer compares, and
# integer compares order -0.0 before 0.0 like `isless` does.

const SortTypes = Union{Int32, Int64, UInt32, UInt64, Float32, Float64}

signedkey(::Type{T}) where {T<:Union{Int32, UInt32, Float32}} = Int32
signedkey(::Type{T}) where {T<:Union{Int64, UInt64, Float64}} = Int64

# Each map is an involution on the bit pattern, so the same function undoes it.
@inline keymap(::Type{<:Signed}, k::S) where {S} = k
@inline keymap(::Type{<:Unsigned}, k::S) where {S} = k ⊻ typemin(S)
# Sign-magnitude to two's complement: negative floats get their magnitude bits
# flipped. The sign bit is unchanged, so applying it twice gives back `k`.
@inline keymap(::Type{<:AbstractFloat}, k::S) where {S} =
    k ⊻ ((k >> (8 * sizeof(S) - 1)) >>> 1)

function map_keys!(k::AbstractVector{S}, ::Type{T}, rev::Bool) where {S, T}
    if rev
        @inbounds @simd for i in eachindex(k)
            k[i] = ~keymap(T, k[i])
        end
    else
        @inbounds @simd for i in eachindex(k)
            k[i] = keymap(T, k[i])
        end
    end
    return k
end

function unmap_keys!(k::AbstractVector{S}, ::Type{T}, rev::Bool) where {S, T}
    if rev
        @inbounds @simd for i in eachindex(k)
            k[i] = keymap(T, ~k[i])
        end
    else
        @inbounds @simd for i in eachindex(k)
            k[i] = keymap(T, k[i])
        end
    end
    return k
end

"""
    nan_partition!(v, rev) -> UnitRange

Move all NaNs to the end of `v` (the front if `rev`), where `isless` puts them, and
return the index range of the remaining values.
"""
function nan_partition!(v::AbstractVector{<:AbstractFloat}, rev::Bool)
    lo, hi = firstindex(v), lastindex(v)
    if rev
        j = lo
        @inbounds for i in lo:hi
            x = v[i]
            if isnan(x)
                v[i] = v[j]
                v[j] = x
                j += 1
            end
        end
        return j:hi
    else
        j = hi
        @inbounds for i in hi:-1:lo
            x = v[i]
            if isnan(x)
                v[i] = v[j]
                v[j] = x
                j -= 1
            end
        end
        return lo:j
    end
end
