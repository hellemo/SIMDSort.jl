"""
    SIMDQuickSort

A vectorized quicksort for primitive numeric vectors, written in Julia on top of
SIMD.jl. See `docs/DESIGN.md` for the algorithm and its sources.
"""
module SIMDQuickSort

using SIMD

export simdsort!, simdsort

include("keys.jl")
include("sort.jl")

"""
    simdsort!(v; rev=false)

Sort `v` in place in the order of `isless` (NaNs last, `-0.0` before `0.0`), or its
reverse if `rev`. `Vector`s of `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32` and
`Float64` use the vectorized quicksort; anything else falls back to `sort!`.
"""
function simdsort!(v::AbstractVector; rev::Bool=false)
    sort!(v; rev)
end

function simdsort!(v::Vector{T}; rev::Bool=false) where {T<:SortTypes}
    r = T <: AbstractFloat ? nan_partition!(v, rev) : eachindex(v)
    k = reinterpret(signedkey(T), view(v, r))
    map_keys!(k, T, rev)
    sort_keys!(k)
    unmap_keys!(k, T, rev)
    return v
end

"""
    simdsort(v; rev=false)

Return a sorted copy of `v`; see [`simdsort!`](@ref).
"""
simdsort(v::AbstractVector; rev::Bool=false) = simdsort!(copy(v); rev)

end
