# Arrays with a pool of distinct values (DataAPI.refpool: PooledArrays,
# CategoricalArrays, ...): their own sortperm sorts the integer refs, much
# faster than the permutation of a copy of the values would be, so simdsortperm
# leaves them to it.
module SIMDSortDataAPIExt

using SIMDSort, DataAPI

SIMDSort.pooled(v::AbstractArray) = DataAPI.refpool(v) !== nothing

end
