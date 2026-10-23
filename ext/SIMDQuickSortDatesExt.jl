# Dates: Date, DateTime, Time and every concrete Period type (Day, Millisecond,
# ...) are an Int64 inside and are ordered by it. They are sorted in place as
# Int64 keys, without copying.
module SIMDQuickSortDatesExt

using SIMDQuickSort, Dates
import SIMDQuickSort: simdsort!, simdsortperm

const Int64Like = Union{Date,DateTime,Time,Dates.FixedPeriod,Dates.OtherPeriod}

ok(::Type{T}) where {T} = isconcretetype(T) && isbitstype(T) && sizeof(T) == 8

# As components of tuples and pairs.
SIMDQuickSort.fixedwidth(::Type{T}) where {T<:Int64Like} = ok(T) ? 64 : nothing
@inline SIMDQuickSort.fixedkey(x::T) where {T<:Int64Like} =
    SIMDQuickSort.ukey(UInt64, reinterpret(Int64, x))

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Int64Like}
    ok(T) || return sort!(v; rev)
    GC.@preserve v simdsort!(unsafe_wrap(Array, Ptr{Int64}(pointer(v)), length(v)); rev, ntasks)
    return v
end

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Int64Like}
    ok(T) || return sortperm(v; rev)
    return GC.@preserve v simdsortperm(unsafe_wrap(Array, Ptr{Int64}(pointer(v)), length(v)); rev, ntasks)
end

end
