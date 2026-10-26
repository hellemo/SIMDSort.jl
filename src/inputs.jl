# Other element types and arrays, mapped onto the fast paths (docs/src/usage.md,
# "Supported types").

# --- missing values --------------------------------------------------------------
#
# Elements of type Union{Missing,T}, T with a fast path: the values that are not
# missing are compacted into a Vector{T} (in parallel), sorted as one, and the
# missings go last (first for rev), as `isless` orders them.

"""
    missingpath(T) -> Bool

Whether `T` is `Union{Missing,S}` for an `S` with a fast path ([`vectorpath`](@ref)).
"""
missingpath(::Type{T}) where {T} = T !== Missing && Missing <: T && vectorpath(nonmissingtype(T))

"""
    split_missing(v, S, ntasks; positions) -> (w, pos, mpos)

The values of `v` that are not missing, in order, as a `Vector{S}`; with
`positions`, also their indices in `v` and those of the missing values.
"""
function split_missing(v::AbstractVector, ::Type{S}, ntasks::Int; positions::Bool) where {S}
    n = length(v)
    off = firstindex(v) - 1
    cs = chunks(n, partasks(n, ntasks, PAR_MIN_KEYS))
    cnt = zeros(Int, length(cs))
    tforeach_index(length(cs)) do c
        k = 0
        @inbounds for i in cs[c]
            k += v[i+off] !== missing
        end
        cnt[c] = k
    end
    m = sum(cnt)
    w = Vector{S}(undef, m)
    pos = positions ? Vector{Int}(undef, m) : Int[]
    mpos = positions ? Vector{Int}(undef, n - m) : Int[]
    start = cumsum(cnt) .- cnt .+ 1                           # chunk c's first value
    mstart = cumsum(length.(cs) .- cnt) .- (length.(cs) .- cnt) .+ 1
    tforeach_index(length(cs)) do c
        j, jm = start[c], mstart[c]
        @inbounds for i in cs[c]
            x = v[i+off]
            if x === missing
                positions && (mpos[jm] = i + off)
                jm += 1
            else
                w[j] = x
                positions && (pos[j] = i + off)
                j += 1
            end
        end
    end
    return w, pos, mpos
end

function missing_sort!(v::AbstractVector, rev::Bool, ntasks::Int)
    w, _, _ = split_missing(v, nonmissingtype(eltype(v)), ntasks; positions = false)
    simdsort!(w; rev, ntasks)
    n, m, f = length(v), length(w), firstindex(v)
    if rev
        fill!(view(v, f:f+n-m-1), missing)
        copyto!(v, f + n - m, w, 1, m)
    else
        copyto!(v, f, w, 1, m)
        fill!(view(v, f+m:f+n-1), missing)
    end
    return v
end

function missing_sortperm(v::AbstractVector, rev::Bool, ntasks::Int)
    w, pos, mpos = split_missing(v, nonmissingtype(eltype(v)), ntasks; positions = true)
    p = simdsortperm(w; rev, ntasks)
    out = Vector{Int}(undef, length(v))
    m, k = length(w), length(mpos)
    vo = rev ? k : 0                                          # where the values start
    copyto!(out, rev ? 1 : m + 1, mpos, 1, k)
    tforeach(m, ntasks) do r
        @inbounds for i in r
            out[vo+i] = pos[p[i]]
        end
    end
    return out
end
