# One out-of-place distribution pass ahead of the cache-resident sorts
# (docs/src/design.md §8), the top level of a sample sort [IPS4o].
#
# For wide keys (Int128, UInt256, Float128, Double64, MultiFloats, packed tuples)
# `sort!` used to compute a permutation of all n keys and then gather v[p]: a
# random 16-32 byte access per element, about half the time. Here the elements
# are first distributed into 256 buckets by their leading order word, with
# splitters from a sorted random sample (radix digits would put skewed keys,
# such as random floats, into a handful of buckets). Counting stores each
# element's bucket as one byte, so the stable scatter of the values into a
# scratch array (in parallel) never recomputes a key. Each bucket is then
# sorted into `v` while it is in cache (finish_bucket!).

const MSD_BUCKETS = 256           # buckets of the distribution pass (bucket ids are UInt8)
const MSD_OVERSAMPLE = 32         # sample keys per bucket for the splitters
const MSD_MINN = 1 << 16          # below this, the plain permutation path

# ---------------------------------------------------------------------------
# A distribution pass, shared by the two users below: classify every element
# into one of nb buckets (its id stored as one byte), then scatter the elements
# stably into a scratch array, bucket by bucket, with one task per chunk.

"""
    bucket_of(spl, x, Val(nb)) -> Int

Branch-free search: the number of splitters spl[1:nb-1] that are <= x (spl
sorted, length nb; spl[nb] is never read). nb is a compile-time constant: the
loop unrolls into conditional moves; with a run-time nb it compiled to
branches, 8 mispredictions per key.
"""
@inline function bucket_of(spl::Vector{S}, x::S, ::Val{nb}) where {S,nb}
    i = 0
    step = nb >> 1
    while step > 0
        @inbounds i += ifelse(spl[i+step] <= x, step, 0)
        step >>= 1
    end
    return i
end

"""
    sample_splitters(key, v, nb) -> Vector or nothing

Splitters from a sorted random sample of key(v[i]), every MSD_OVERSAMPLE-th
[IPS4o] (radix digits would put skewed keys, such as random floats, into a
handful of buckets). `nothing` if the sample has too few distinct keys for
balanced buckets.
"""
function sample_splitters(key::F, v::AbstractVector, nb::Int) where {F}
    n = length(v)
    smp = [key(@inbounds v[1])]
    resize!(smp, nb * MSD_OVERSAMPLE)
    r = UInt64(n) | 1
    @inbounds for j in eachindex(smp)
        r = xorshift(r)
        smp[j] = key(v[1+Int(r % UInt64(n))])
    end
    sort!(smp)
    count(i -> smp[i] != smp[i-1], 2:length(smp)) < nb ÷ 4 && return nothing
    return [smp[j*MSD_OVERSAMPLE] for j in 1:nb]
end

"""
    WordBucket(f, spl, rev)

Bucket of a wide element by its leading order word f(x)[1] (mirrored for rev).
"""
struct WordBucket{F}
    f::F
    spl::Vector{UInt64}
    rev::Bool
end
@inline function (c::WordBucket)(x)
    d = bucket_of(c.spl, c.f(x)[1], Val(MSD_BUCKETS))
    return c.rev ? MSD_BUCKETS - 1 - d : d
end

"""
    KeyBucket(spl)

Bucket of a primitive key.
"""
struct KeyBucket{S}
    spl::Vector{S}
end
@inline (c::KeyBucket)(x) = bucket_of(c.spl, x, Val(DIST_BUCKETS))

"""
    Distribution

The chunks of the input, where chunk c writes bucket b (start[b, c]), and the
buckets (bucket b is bstart[b]:bstart[b+1]-1).
"""
struct Distribution
    cs::Vector{UnitRange{Int}}
    start::Matrix{Int}
    bstart::Vector{Int}
end

"""
    distribution(classify, src, ids, nb, ntasks) -> Distribution

Count: ids[i] = classify(src[i]) ∈ 0:nb-1 for every i, then bucket-major
prefix sums.
"""
function distribution(classify::C, src::AbstractVector, ids::AbstractVector{UInt8},
                      nb::Int, ntasks::Int) where {C}
    cs = chunks(length(src), ntasks)
    nc = length(cs)
    counts = zeros(Int, nb, nc)
    tforeach_index(nc) do c
        count_chunk!(counts, ids, classify, src, cs[c], c, nb)
    end
    start = similar(counts)
    bstart = Vector{Int}(undef, nb + 1)
    s = 1
    for b in 1:nb
        bstart[b] = s
        for c in 1:nc
            start[b, c] = s
            s += counts[b, c]
        end
    end
    bstart[nb+1] = s
    return Distribution(cs, start, bstart)
end

function count_chunk!(counts, ids, classify::C, src, r, c, nb) where {C}
    cnt = zeros(Int, nb)
    @inbounds for i in r
        d = classify(src[i])
        ids[i] = d % UInt8
        cnt[d+1] += 1
    end
    @inbounds counts[:, c] .= cnt
    return nothing
end

"""
    scatter!(dst, src, ids, dist)

Scatter: dst[...] = src[i] into the buckets ids[i], stably.
"""
function scatter!(dst::AbstractVector, src::AbstractVector, ids::AbstractVector{UInt8}, dist::Distribution)
    tforeach_index(length(dist.cs)) do c
        scatter_chunk!(dst, src, ids, dist.start, dist.cs[c], c)
    end
    return dst
end

function scatter_chunk!(dst, src, ids, start, r, c)
    pos = start[:, c]
    @inbounds for i in r
        d = ids[i] + 1
        dst[pos[d]] = src[i]
        pos[d] += 1
    end
    return nothing
end

# ---------------------------------------------------------------------------
# sort! for wide fixed-width keys.

"""
    sort_by_words!(v, f, rev, ntasks)

Sort `v` stably by its order words `f(v[i])` (a tuple of big-endian `UInt64`s,
compared lexicographically), in reverse if `rev`.

Only the values are moved: the words are recomputed from them where needed,
which halves the memory traffic of the scatter.
"""
function sort_by_words!(v::Vector{T}, f::F, rev::Bool, ntasks::Int) where {T,F}
    n = length(v)
    n < MSD_MINN && return sort_by_words_perm!(v, f, rev, ntasks)
    count!(:msd_sorts)
    nb = MSD_BUCKETS
    spl = @phase :msd_sample sample_splitters(x -> f(x)[1], v, nb)
    if spl === nothing          # few distinct leading words: unbalanced buckets
        count!(:msd_skipped_fewlead)
        return sort_by_words_perm!(v, f, rev, ntasks)
    end
    arena = ScratchArena()
    try
        ids = alloc!(arena, UInt8, n)
        tv = alloc!(arena, T, n)
        dist = @phase :msd_count distribution(WordBucket(f, spl, rev), v, ids, nb, ntasks)
        bstart = dist.bstart
        count_buckets(:msd, bstart)
        @phase :msd_scatter scatter!(tv, v, ids, dist)
        @phase :msd_finish begin
            # Sort every bucket from the scratch array back into v, in cache:
            # groups of buckets of about equal size on ntasks workers, each with
            # one pair of scratch buffers reused for all its buckets (fresh
            # buffers per group allocated ~64 MB at 8 tasks).
            sizes = diff(bstart)
            bmax = maximum(sizes)
            groups = ntasks <= 1 ? [1:nb] : balanced_groups(sizes, cld(n, 8ntasks))
            tworkers(groups, ntasks; init = () -> (Vector{UInt64}(undef, bmax), Vector{Int}(undef, bmax))) do (keys, idx), bs
                finish_buckets!(v, tv, bstart, bs, f, rev, keys, idx)
            end
        end
        return v
    finally
        release!(arena)
    end
end

"""
    sort_by_words_perm!(v, f, rev, ntasks)

Small inputs, and too few distinct leading words: the permutation of the
windowed sort, applied in place.
"""
sort_by_words_perm!(v, f::F, rev::Bool, ntasks::Int) where {F} =
    permute_by!(v, wordkeys_sortperm(words_of(f, v, ntasks); rev, ntasks), ntasks)

# Finish buckets bs with one pair of scratch buffers.

function finish_buckets!(v, tv, bstart, bs, f::F, rev::Bool, keys::Vector{UInt64}, idx::Vector{Int}) where {F}
    for b in bs
        finish_bucket!(v, tv, bstart[b], bstart[b+1] - 1, f, rev, keys, idx)
    end
end

"""
    finish_bucket!(v, tv, a, e, f, rev, keys, idx)

Sort tv[a:e] stably by order words into v[a:e]: one SIMD sort of 64-bit keys
(a window of the order words just below the bucket's common prefix, above the
local position), then a full comparison within the rare runs of equal windows.
"""
function finish_bucket!(v, tv, a::Int, e::Int, f::F, rev::Bool, keys, idx) where {F}
    m = e - a + 1
    m <= 0 && return
    if m == 1
        @inbounds v[a] = tv[a]
        return
    end
    lo, hi = typemax(UInt64), zero(UInt64)
    @inbounds for i in a:e
        x = f(tv[i])[1]
        lo = min(lo, x)
        hi = max(hi, x)
    end
    off = lo == hi ? 64 : leading_zeros(lo ⊻ hi)
    ib = 8sizeof(Int) - leading_zeros(m - 1)
    K = 64 - ib
    mask = (one(UInt64) << ib) - one(UInt64)
    resize!(keys, m)
    @inbounds for j in 1:m
        keys[j] = (wordbits(f(tv[a+j-1]), off, K, rev) << ib) | UInt64(j - 1)
    end
    simdsort!(keys)
    resize!(idx, m)
    @inbounds for j in 1:m
        idx[j] = (keys[j] & mask) % Int + 1
    end
    # Runs of equal windows: sort by the full words, then position (stable).
    j = 1
    @inbounds while j < m
        h = keys[j] >> ib
        k = j
        while k < m && keys[k+1] >> ib == h
            k += 1
        end
        if k > j
            count!(:msd_tie_runs)
            sort!(view(idx, j:k); by = i -> (rev ? map(~, f(tv[a+i-1])) : f(tv[a+i-1]), i))
        end
        j = k + 1
    end
    @inbounds for j in 1:m
        v[a+j-1] = tv[a+idx[j]-1]
    end
    return
end

# ---------------------------------------------------------------------------
# Sample-sort top level for many threads (docs/src/design.md §7j). The parallel
# quicksort stops scaling at about 16 threads: on a 2 × 28-core Xeon Gold 6348 it
# takes the same time at 16, 32 and 56 threads (UInt64 2^27: 0.29 s), while
# this step keeps scaling (0.28, 0.18, 0.15 s; Int64 2^26 at 56 threads:
# 0.059 s against 0.110). At 8 threads it loses (laptop and Xeon), so it is
# used from DIST_MINTASKS tasks.
#
# One distribution into DIST_BUCKETS buckets (a count and a scatter, both
# parallel) replaces about log2(DIST_BUCKETS) partition passes over the whole
# array; each bucket is then quicksorted on one thread and copied back while
# in cache.

const DIST_BUCKETS = 256          # bucket ids fit a UInt8
const DIST_MINN = Ref(1 << 22)      # keys
const DIST_MINTASKS = Ref(24)       # Xeon: ties or loses at 16 tasks, wins at 32 and 56

use_distribution(n::Int, ntasks::Int) = ntasks >= DIST_MINTASKS[] && n >= DIST_MINN[]

"""
    sort_auto!(k, ntasks)

The quicksort, or above DIST_MINTASKS tasks and DIST_MINN keys the
sample-sort top level (which falls back to the quicksort when the sample
has too few distinct keys).
"""
function sort_auto!(k::KeyVector, ntasks::Int)
    n = length(k)
    use_distribution(n, ntasks) && @phase(:distribution, distribute_sort!(k, ntasks)) && return nothing
    owner = keyowner(k)
    GC.@preserve owner @phase :quicksort sort_keys!(keyref(k), n, ntasks)
    return nothing
end

"""
    distribute_sort!(k, ntasks) -> Bool

Sort the keys `k` with one sample-sort distribution pass and per-bucket
quicksorts. Returns `false` without changing anything if the sample has too
few distinct keys for balanced buckets.
"""
function distribute_sort!(k::KeyVector{S}, ntasks::Int) where {S}
    nb = DIST_BUCKETS
    n = length(k)
    spl = sample_splitters(identity, k, nb)
    spl === nothing && return false
    count!(:dist_sorts)
    arena = ScratchArena()
    try
        ids = alloc!(arena, UInt8, n)
        tv = alloc!(arena, S, n)
        dist = @phase :dist_count distribution(KeyBucket(spl), k, ids, nb, ntasks)
        bstart = dist.bstart
        count_buckets(:dist, bstart)
        @phase :dist_scatter scatter!(tv, k, ids, dist)
        @phase :dist_finish begin
            groups = balanced_groups(diff(bstart), cld(n, 8ntasks))
            run_limited(bs -> dist_finish!(k, tv, bstart, bs), groups, ntasks)
        end
        return true
    finally
        release!(arena)
    end
end

function dist_finish!(k::KeyVector{S}, tv::Vector{S}, bstart, bs) where {S}
    GC.@preserve tv for b in bs
        a, m = bstart[b], bstart[b+1] - bstart[b]
        m == 0 && continue
        sort_keys!(shift(keyref(tv), a - 1), m, 1)
        copyto!(k, a, tv, a, m)
    end
    return nothing
end
