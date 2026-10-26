# Dates: Date, DateTime, Time and every concrete Period type (Day, Millisecond,
# ...) are an Int64 inside and are ordered by it. They are sorted in place as
# Int64 keys, without copying.
module SIMDSortDatesExt

using SIMDSort, Dates
import SIMDSort: simdsort!, simdsortperm

const Int64Like = Union{Date,DateTime,Time,Dates.FixedPeriod,Dates.OtherPeriod}

ok(::Type{T}) where {T} = isconcretetype(T) && isbitstype(T) && sizeof(T) == 8
SIMDSort.vectorpath(::Type{T}) where {T<:Int64Like} = ok(T)

# As components of tuples and pairs.
SIMDSort.fixedwidth(::Type{T}) where {T<:Int64Like} = ok(T) ? 64 : nothing
@inline SIMDSort.fixedkey(x::T) where {T<:Int64Like} =
    SIMDSort.ukey(UInt64, reinterpret(Int64, x))

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Int64Like}
    ok(T) || return sort!(v; rev)
    length(v) <= 1 && return v
    SIMDSort.sort_primitive!(SIMDSort.MappedKeys(), reinterpret(Int64, v), SIMDSort.ordering(rev), Int(ntasks))
    return v
end

function simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Int64Like}
    ok(T) || return sortperm(v; rev)
    return simdsortperm(reinterpret(Int64, v); rev, ntasks)
end

end
