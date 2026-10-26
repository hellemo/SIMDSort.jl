# Arrays storing integer refs into a pool of values (DataAPI.refarray and
# DataAPI.refpool: PooledArrays, CategoricalArrays, ...): sorted through their
# refs (src/inputs.jl, "pooled arrays"). sortperm only reads the refs, so any
# such array qualifies; sort! rewrites them, so only arrays known to own a
# Vector of refs that is their storage (PooledArray, CategoricalArray).
module SIMDSortDataAPIExt

using SIMDSort, DataAPI

function SIMDSort.pooled_sortperm(v::AbstractArray, rev::Bool, ntasks::Int)
    pool = DataAPI.refpool(v)
    pool === nothing && return nothing
    p = SIMDSort.refs_sortperm(DataAPI.refarray(v), pool, rev, ntasks)
    p === nothing && return nothing
    return firstindex(v) == 1 ? p : p .+ (firstindex(v) - 1)
end

ownsrefs(v) = nameof(typeof(v)) in (:PooledArray, :CategoricalArray) &&
              nameof(parentmodule(typeof(v))) in (:PooledArrays, :CategoricalArrays) &&
              DataAPI.refarray(v) isa Vector

function SIMDSort.pooled_sort!(v::AbstractArray, rev::Bool, ntasks::Int)
    ownsrefs(v) || return nothing
    pool = DataAPI.refpool(v)
    pool === nothing && return nothing
    SIMDSort.refs_sort!(DataAPI.refarray(v), pool, rev, ntasks) === nothing && return nothing
    return v
end

end
