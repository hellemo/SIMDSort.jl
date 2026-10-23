"""
    SIMDQuickSort

A vectorized quicksort for primitive numeric vectors, written in Julia on top of
SIMD.jl. See `docs/DESIGN.md` for the algorithm and its sources.
"""
module SIMDQuickSort

using SIMD

export simdsort!, simdsort

include("keys.jl")
include("network.jl")
include("partition.jl")
include("sort.jl")
include("parallel.jl")

"""
    simdsort!(v; rev=false, ntasks=1)

Sort `v` in place in the order of `isless` (NaNs last, `-0.0` before `0.0`), or its
reverse if `rev`. `Vector`s of `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`
and `Float64` use the vectorized quicksort; anything else falls back to `sort!`.

With `ntasks > 1`, large subranges are sorted by up to `ntasks` concurrent tasks;
`ntasks = Threads.nthreads()` uses all threads.
"""
function simdsort!(v::AbstractVector; rev::Bool=false, ntasks::Integer=1)
    sort!(v; rev)
end

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:SortTypes}
    r = T <: AbstractFloat ? nan_partition!(v, rev) : eachindex(v)
    n = length(r)
    n <= 1 && return v
    S = signedkey(T)
    GC.@preserve v begin
        p = Ptr{S}(pointer(v, first(r)))
        k = unsafe_wrap(Array, p, n)
        foreach_chunk(k -> map_keys!(k, T, rev), k, ntasks)
        presorted!(k) || sort_keys!(p, n, Int(ntasks))
        foreach_chunk(k -> unmap_keys!(k, T, rev), k, ntasks)
    end
    return v
end

"""
    simdsort(v; rev=false, ntasks=1)

Return a sorted copy of `v`; see [`simdsort!`](@ref).
"""
simdsort(v::AbstractVector; rev::Bool=false, ntasks::Integer=1) = simdsort!(copy(v); rev, ntasks)

end
