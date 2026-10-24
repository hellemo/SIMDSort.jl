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

# The sort direction as a type: Base's Forward and Reverse orderings. Public
# functions take `rev::Bool` and convert once (`ordering`); everything below
# dispatches on the ordering, so each direction gets its own specialized code.
const Ord = Union{Base.Order.ForwardOrdering,Base.Order.ReverseOrdering{Base.Order.ForwardOrdering}}
ordering(rev::Bool) = rev ? Base.Order.Reverse : Base.Order.Forward
isrev(::Base.Order.ForwardOrdering) = false
isrev(::Base.Order.ReverseOrdering) = true

# Encode a value's bits as a signed key in the given order (complemented for
# Reverse: an exact involution that reverses the order), and decode it again.
@inline encodekey(::Type{T}, ::Base.Order.ForwardOrdering, k) where {T} = keymap(T, k)
@inline encodekey(::Type{T}, ::Base.Order.ReverseOrdering, k) where {T} = ~keymap(T, k)
@inline decodekey(::Type{T}, ::Base.Order.ForwardOrdering, k) where {T} = keymap(T, k)
@inline decodekey(::Type{T}, ::Base.Order.ReverseOrdering, k) where {T} = keymap(T, ~k)

# k[i] = f(k[i]) for all i (vectorizes for simple f).
function mapkeys!(f::F, k::AbstractVector) where {F}
    @inbounds @simd for i in eachindex(k)
        k[i] = f(k[i])
    end
    return k
end

map_keys!(k::AbstractVector, ::Type{T}, o::Ord) where {T} = mapkeys!(x -> encodekey(T, o, x), k)
unmap_keys!(k::AbstractVector, ::Type{T}, o::Ord) where {T} = mapkeys!(x -> decodekey(T, o, x), k)
map_keys!(k::AbstractVector, ::Type{T}, rev::Bool) where {T} = map_keys!(k, T, ordering(rev))
unmap_keys!(k::AbstractVector, ::Type{T}, rev::Bool) where {T} = unmap_keys!(k, T, ordering(rev))

# Counts of NaNs, -0.0s and +0.0s in `k`, from the bit patterns (vectorizes), by
# up to `ntasks` concurrent tasks.
function float_scan(k::AbstractVector{T}, ntasks::Integer) where {T<:AbstractFloat}
    return tmapreduce(r -> float_scan(view(k, r)), (a, b) -> a .+ b, length(k), ntasks; minlen = 1 << 16)
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
foreach_chunk(f!::F, k::AbstractVector, ntasks::Integer) where {F} =
    tforeach(r -> f!(view(k, r)), length(k), ntasks; minlen = 1 << 16)

"""
    nan_partition!(v, order) -> UnitRange

Move all NaNs to the end of `v` (the front for Reverse), where `isless` puts
them, and return the index range of the remaining values.
"""
function nan_partition!(v::AbstractVector{<:AbstractFloat}, ::Base.Order.ForwardOrdering)
    lo, j = firstindex(v), lastindex(v)
    @inbounds for i in j:-1:lo
        x = v[i]
        if isnan(x)
            v[i] = v[j]
            v[j] = x
            j -= 1
        end
    end
    return lo:j
end

function nan_partition!(v::AbstractVector{<:AbstractFloat}, ::Base.Order.ReverseOrdering)
    j, hi = firstindex(v), lastindex(v)
    @inbounds for i in j:hi
        x = v[i]
        if isnan(x)
            v[i] = v[j]
            v[j] = x
            j += 1
        end
    end
    return j:hi
end
