# StringViews.jl on Julia before 1.14 (from 1.14 its StringView is Base's, which
# the string path covers already): StringViews of contiguous byte vectors are
# read in place, as Strings are. A column of StringViews over one shared byte
# buffer (as Arrow-style formats store strings) keeps each string's offset and
# length in the vector itself, so it sorts faster than a Vector{String}, whose
# strings are separate objects.
module SIMDSortStringViewsExt

using SIMDSort, StringViews
import SIMDSort: simdsort!, simdsortperm, vectorpath

@static if !isdefined(Base, :DenseStringView)

const DSV = StringViews.DenseStringView

vectorpath(::Type{T}) where {T<:DSV} = isconcretetype(T)

simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:DSV} =
    SIMDSort.bytes_sortperm(v, rev, Int(ntasks))

simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:DSV} =
    SIMDSort.bytes_sort!(v, rev, Int(ntasks))

end

end
