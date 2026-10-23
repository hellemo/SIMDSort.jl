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

function negate!(k::AbstractVector{<:AbstractFloat})
    @inbounds @simd for i in eachindex(k)
        k[i] = -k[i]
    end
    return k
end

# Counts of NaNs, -0.0s and +0.0s in `k`, from the bit patterns (vectorizes), by
# up to `ntasks` concurrent tasks.
function float_scan(k::AbstractVector{T}, ntasks::Integer) where {T<:AbstractFloat}
    n = length(k)
    if ntasks <= 1 || n < 1 << 16
        return float_scan(k)
    end
    parts = [Threads.@spawn(float_scan(view(k, c)))
             for c in Iterators.partition(eachindex(k), cld(n, ntasks))]
    return reduce((a, b) -> a .+ b, fetch.(parts))
end

function float_scan(k::AbstractVector{T}) where {T<:AbstractFloat}
    U = signedkey(T)
    expmask = reinterpret(U, T(Inf))
    nz = reinterpret(U, -zero(T))
    nnan = nneg = npos = 0
    @inbounds @simd for i in eachindex(k)
        b = reinterpret(U, k[i])
        nnan += (b & ~nz) > expmask
        nneg += b == nz
        npos += b == zero(U)
    end
    return (nnan, nneg, npos)
end

"""
    order_zeros!(k, c)

`k` is sorted except that `-0.0` and `0.0` compare equal; also, vector min/max
may have changed the sign of zeros. Rewrite the run of zeros as the `c` -0.0s of
the input followed by 0.0s, as `isless` orders them.
"""
function order_zeros!(k::AbstractVector{T}, c::Int) where {T<:AbstractFloat}
    lo = searchsortedfirst(k, zero(T); lt=<)
    hi = searchsortedlast(k, zero(T); lt=<)
    fill!(view(k, lo:lo+c-1), -zero(T))
    fill!(view(k, lo+c:hi), zero(T))
    return k
end

# Apply `f!` to `ntasks` contiguous chunks of `k` concurrently.
function foreach_chunk(f!::F, k::AbstractVector, ntasks::Integer) where {F}
    n = length(k)
    if ntasks <= 1 || n < 1 << 16
        f!(k)
    else
        @sync for c in Iterators.partition(eachindex(k), cld(n, ntasks))
            Threads.@spawn f!(view(k, c))
        end
    end
    return nothing
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
