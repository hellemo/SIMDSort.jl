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

# --- small element types -----------------------------------------------------------
#
# 8- and 16-bit integers, Char and Float16 are widened to 32-bit keys of the
# same order: Int32 for signed integers, UInt32 for unsigned ones and Char
# (isless compares a Char's UInt32 encoding), Float32 for Float16 (exact, NaNs
# stay NaNs). sortperm sorts the keys; sort! of Char and Float16 sorts them and
# narrows back (Base's sort! of the integers is a counting sort already). Bool
# is left to Base: its sortperm is a stable partition (6 ms against 10 ms here
# for 2e6 elements).

const SmallTypes = Union{Int8,Int16,UInt8,UInt16,Char,Float16}

widetype(::Type{<:Union{Int8,Int16}}) = Int32
widetype(::Type{<:Union{UInt8,UInt16,Char}}) = UInt32
widetype(::Type{Float16}) = Float32
@inline widekey(x::Union{Int8,Int16}) = Int32(x)
@inline widekey(x::Union{UInt8,UInt16}) = UInt32(x)
@inline widekey(x::Char) = reinterpret(UInt32, x)
@inline widekey(x::Float16) = Float32(x)
@inline narrowkey(::Type{Char}, k::UInt32) = reinterpret(Char, k)
@inline narrowkey(::Type{Float16}, k::Float32) = Float16(k)

function widekeys(v::Vector{T}, ntasks::Int) where {T<:SmallTypes}
    k = Vector{widetype(T)}(undef, length(v))
    tforeach(length(v), ntasks) do r
        @inbounds for i in r
            k[i] = widekey(v[i])
        end
    end
    return k
end

small_sortperm(v::Vector{<:SmallTypes}, rev::Bool, ntasks::Int) =
    length(v) <= 1 ? collect(1:length(v)) : simdsortperm(widekeys(v, ntasks); rev, ntasks)

# (Int8 ... UInt16 come through the Signed/Unsigned methods of wide.jl.)
simdsortperm(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{Char,Float16}} =
    small_sortperm(v, rev, Int(ntasks))

function simdsort!(v::Vector{T}; rev::Bool=false, ntasks::Integer=1) where {T<:Union{Char,Float16}}
    length(v) <= 1 && return v
    k = simdsort!(widekeys(v, Int(ntasks)); rev, ntasks)
    tforeach(length(v), Int(ntasks)) do r
        @inbounds for i in r
            v[i] = narrowkey(T, k[i])
        end
    end
    return v
end

# --- pooled arrays -------------------------------------------------------------------
#
# Arrays storing integer refs into a pool of distinct values (DataAPI's refarray
# and refpool, through the DataAPI extension: PooledArrays, CategoricalArrays):
# sorted by their refs. The pool values are ranked once (a sort of the pool, in
# isless order: a CategoricalArray's by its levels, missing last); sortperm is
# then a stable parallel counting sort of the refs by rank (with many distinct
# values, a sortperm of the ranks), and sort! of an array that owns its refs
# rewrites them, rank by rank.

# The extension's methods; `nothing` means: not a pooled array (or a ref
# outside the pool), take the other paths.
pooled_sortperm(v, rev::Bool, ntasks::Int) = nothing
pooled_sort!(v, rev::Bool, ntasks::Int) = nothing

"""
    pool_ranks(pool, ntasks) -> (rank, nr, refof)

`rank[j]` is the rank (1-based, equal values equal) of the `j`-th pool entry
(ref `first(axes(pool, 1)) + j - 1`), `nr` the number of distinct values, and
`refof[r]` a ref of rank `r`.
"""
function pool_ranks(pool::AbstractVector, ntasks::Int)
    vals = collect(pool)
    p = simdsortperm(vals; ntasks)
    lo = first(axes(pool, 1))
    rank = Vector{Int}(undef, length(vals))
    refof = Int[]
    @inbounds for (k, j) in enumerate(p)
        (k == 1 || !isequal(vals[j], vals[p[k-1]])) && push!(refof, lo + j - 1)
        rank[j] = length(refof)
    end
    return rank, length(refof), refof
end

# The ranks of a chunk's refs counted (count) or written (codes), or false on a
# ref outside the pool.
function rank_chunk!(f::F, refs, r, off, rank, lo) where {F}
    np = length(rank)
    @inbounds for i in r
        j = Int(refs[i+off]) - lo + 1
        1 <= j <= np || return false
        f(i, rank[j])
    end
    return true
end

"""
    refs_sortperm(refs, pool, rev, ntasks) -> Vector{Int} or nothing

The stable permutation (indices `1:length(refs)`) that sorts the values
`pool[refs[i]]`, or `nothing` if a ref is outside the pool.
"""
function refs_sortperm(refs::AbstractVector{<:Integer}, pool::AbstractVector, rev::Bool, ntasks::Int)
    n = length(refs)
    rank, nr, _ = pool_ranks(pool, ntasks)
    lo, off = first(axes(pool, 1)), firstindex(refs) - 1
    cs = chunks(n, partasks(n, ntasks, PAR_MIN_KEYS))
    nc = length(cs)
    ok = trues(nc)
    if nr * nc > max(n, 1 << 16) || nr >= typemax(UInt32)
        # many distinct values: the permutation of the ranks
        codes = Vector{UInt32}(undef, n)
        tforeach_index(nc) do c
            ok[c] = rank_chunk!((i, k) -> @inbounds(codes[i] = k % UInt32), refs, cs[c], off, rank, lo)
        end
        all(ok) || return nothing
        return simdsortperm(codes; rev, ntasks)
    end
    cnt = zeros(Int, nr, nc)
    tforeach_index(nc) do c
        ok[c] = rank_chunk!((i, k) -> @inbounds(cnt[k, c] += 1), refs, cs[c], off, rank, lo)
    end
    all(ok) || return nothing
    s = 1
    @inbounds for k in (rev ? (nr:-1:1) : (1:nr)), c in 1:nc       # rank-major starts
        t = cnt[k, c]
        cnt[k, c] = s
        s += t
    end
    out = Vector{Int}(undef, n)
    tforeach_index(nc) do c
        rank_chunk!(refs, cs[c], off, rank, lo) do i, k
            @inbounds p = cnt[k, c]
            @inbounds out[p] = i
            @inbounds cnt[k, c] = p + 1
        end
    end
    return out
end

"""
    refs_sort!(refs, pool, rev, ntasks) -> refs or nothing

Rewrite `refs` so that `pool[refs[i]]` is sorted: the refs of each rank in
turn, as many as there were (`nothing` if a ref is outside the pool).
"""
function refs_sort!(refs::AbstractVector{R}, pool::AbstractVector, rev::Bool, ntasks::Int) where {R<:Integer}
    n = length(refs)
    rank, nr, refof = pool_ranks(pool, ntasks)
    lo, off = first(axes(pool, 1)), firstindex(refs) - 1
    cs = chunks(n, partasks(n, ntasks, PAR_MIN_KEYS))
    cnts = [zeros(Int, nr) for _ in cs]
    ok = trues(length(cs))
    tforeach_index(length(cs)) do c
        cn = cnts[c]
        ok[c] = rank_chunk!((i, k) -> @inbounds(cn[k] += 1), refs, cs[c], off, rank, lo)
    end
    all(ok) || return nothing
    total = reduce(+, cnts)
    pos = firstindex(refs)
    for k in (rev ? (nr:-1:1) : (1:nr))
        fill!(view(refs, pos:pos+total[k]-1), R(refof[k]))
        pos += total[k]
    end
    return refs
end
