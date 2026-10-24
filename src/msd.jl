# One out-of-place distribution pass ahead of the cache-resident sorts
# (docs/DESIGN.md §8), the top level of a sample sort [IPS4o].
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

# Branch-free search: the number of splitters spl[1:nb-1] that are <= x (spl
# sorted, length nb = MSD_BUCKETS, a power of two; spl[nb] is never read).
@inline function bucket_of(spl::Vector{UInt64}, x::UInt64)
    i = 0
    step = MSD_BUCKETS >> 1
    while step > 0
        @inbounds i += ifelse(spl[i+step] <= x, step, 0)
        step >>= 1
    end
    return i
end

# Splitters from the leading order words of a sorted random sample, every
# MSD_OVERSAMPLE-th [IPS4o] (radix digits put skewed keys such as random floats
# into a handful of buckets): spl[j] for j in 1:nb-1, padded with typemax at
# index nb, so that bucket(x) = #{j : spl[j] <= x}. `nothing` if the sample has
# too few distinct leading words for balanced buckets.
function word_splitters(f::F, v::AbstractVector, nb::Int) where {F}
    n = length(v)
    smp = Vector{UInt64}(undef, nb * MSD_OVERSAMPLE)
    r = UInt64(n) | 1
    @inbounds for j in eachindex(smp)
        r = xorshift(r)
        smp[j] = f(v[1+Int(r % UInt64(n))])[1]
    end
    sort!(smp)
    count(i -> smp[i] != smp[i-1], 2:length(smp)) < nb ÷ 4 && return nothing
    spl = [smp[j*MSD_OVERSAMPLE] for j in 1:nb-1]
    push!(spl, typemax(UInt64))
    return spl
end

"""
    sort_by_words!(v, f, rev, ntasks)

Sort `v` stably by its order words `f(v[i])` (a tuple of big-endian `UInt64`s,
compared lexicographically), in reverse if `rev`.

Only the values are moved: the words are recomputed from them where needed,
which halves the memory traffic of the scatter.
"""
function sort_by_words!(v::Vector{T}, f::F, rev::Bool, ntasks::Int) where {T,F}
    n = length(v)
    n < MSD_MINN && return permute_by!(v, wordkeys_sortperm(words_of(f, v, ntasks); rev, ntasks), ntasks)
    count!(:msd_sorts)
    nb = MSD_BUCKETS
    spl = @phase :msd_sample word_splitters(f, v, nb)
    if spl === nothing          # few distinct leading words: unbalanced buckets
        count!(:msd_skipped_fewlead)
        return permute_by!(v, wordkeys_sortperm(words_of(f, v, ntasks); rev, ntasks), ntasks)
    end
    cs = chunks(n, ntasks)
    nc = length(cs)
    counts = zeros(Int, nb, nc)
    arena = ScratchArena()
    try
        bk = alloc!(arena, UInt8, n)
        tv = alloc!(arena, T, n)
        @phase :msd_count begin
            tforeach_index(nc) do c
                cnt = zeros(Int, nb)
                @inbounds for i in cs[c]
                    d = bucket_of(spl, f(v[i])[1])
                    d = rev ? nb - 1 - d : d
                    bk[i] = d % UInt8
                    cnt[d+1] += 1
                end
                @inbounds counts[:, c] .= cnt
            end
            # Bucket-major prefix sums: chunk c writes bucket b from start[b, c].
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
        end
        count_buckets(:msd, bstart)
        @phase :msd_scatter begin
            tforeach_index(nc) do c
                pos = start[:, c]
                @inbounds for i in cs[c]
                    d = bk[i] + 1
                    tv[pos[d]] = v[i]
                    pos[d] += 1
                end
            end
        end
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

# Finish buckets bs with one pair of scratch buffers.

function finish_buckets!(v, tv, bstart, bs, f::F, rev::Bool, keys::Vector{UInt64}, idx::Vector{Int}) where {F}
    for b in bs
        finish_bucket!(v, tv, bstart[b], bstart[b+1] - 1, f, rev, keys, idx)
    end
end

# Bits [off, off + K) (K ≤ 63) of the order words w, complemented for rev.
@inline function wordbits(w::NTuple{M,UInt64}, off::Int, K::Int, rev::Bool) where {M}
    off >= 64M && return zero(UInt64)
    q = off >> 6
    hi = @inbounds w[q+1]
    lo = q + 2 <= M ? @inbounds(w[q+2]) : zero(UInt64)
    if rev
        hi, lo = ~hi, q + 2 <= M ? ~lo : zero(UInt64)
    end
    return (((UInt128(hi) << 64 | UInt128(lo)) << (off & 63)) >> (128 - K)) % UInt64
end

# Sort tv[a:e] stably by order words into v[a:e]: one SIMD sort of 64-bit keys
# (a window of the order words just below the bucket's common prefix, above the
# local position), then a full comparison within the rare runs of equal windows.
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
# Sample-sort top level for many threads (docs/DESIGN.md §7j). The parallel
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
const DIST_MINTASKS = Ref(16)       # Xeon: wins from 16 threads, loses at 8

# The number of spl[1:nb-1] that are <= x (nb = length(spl), a power of two).
# (nb a compile-time constant: the loop unrolls into conditional moves; with a
# run-time nb it compiled to branches, 8 mispredictions per key.)
@inline function bucket_of_n(spl::Vector{S}, x::S, ::Val{nb}) where {S,nb}
    i = 0
    step = nb >> 1
    while step > 0
        @inbounds i += ifelse(spl[i+step] <= x, step, 0)
        step >>= 1
    end
    return i
end

use_distribution(n::Int, ntasks::Int) = ntasks >= DIST_MINTASKS[] && n >= DIST_MINN[]

# The quicksort, or above DIST_MINTASKS tasks and DIST_MINN keys the
# sample-sort top level (which falls back to the quicksort when the sample
# has too few distinct keys).
function sort_auto!(p::Ptr{S}, n::Int, ntasks::Int) where {S}
    use_distribution(n, ntasks) && @phase(:distribution, distribute_sort!(p, n, ntasks)) && return nothing
    @phase :quicksort sort_keys!(p, n, ntasks)
    return nothing
end

"""
    distribute_sort!(p::Ptr{S}, n, ntasks) -> Bool

Sort the n keys at `p` with one sample-sort distribution pass and per-bucket
quicksorts. Returns `false` without changing anything if the sample has too
few distinct keys for balanced buckets.
"""
function distribute_sort!(p::Ptr{S}, n::Int, ntasks::Int) where {S}
    nb = DIST_BUCKETS
    k = unsafe_wrap(Array, p, n)
    smp = Vector{S}(undef, nb * MSD_OVERSAMPLE)
    r = UInt64(n) | 1
    @inbounds for j in eachindex(smp)
        r = xorshift(r)
        smp[j] = k[1+Int(r % UInt64(n))]
    end
    sort!(smp)
    count(i -> smp[i] != smp[i-1], 2:length(smp)) < nb ÷ 4 && return false
    count!(:dist_sorts)
    spl = S[smp[j*MSD_OVERSAMPLE] for j in 1:nb]
    cs = chunks(n, ntasks)
    nc = length(cs)
    counts = zeros(Int, nb, nc)
    arena = ScratchArena()
    try
        bk = alloc!(arena, UInt8, n)
        tv = alloc!(arena, S, n)
        @phase :dist_count begin
            tforeach_index(nc) do c
                dist_count!(counts, bk, k, spl, cs[c], c, Val(nb))
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
        end
        count_buckets(:dist, bstart)
        @phase :dist_scatter begin
            tforeach_index(nc) do c
                dist_scatter!(tv, k, bk, start, cs[c], c, nb)
            end
        end
        @phase :dist_finish begin
            groups = balanced_groups(diff(bstart), cld(n, 8ntasks))
            run_limited(bs -> dist_finish!(k, tv, bstart, bs), groups, ntasks)
        end
        return true
    finally
        release!(arena)
    end
end

function dist_count!(counts, bk, k::Vector{S}, spl::Vector{S}, r, c, vnb::Val{nb}) where {S,nb}
    cnt = zeros(Int, nb)
    @inbounds for i in r
        d = bucket_of_n(spl, k[i], vnb)
        bk[i] = d % UInt8
        cnt[d+1] += 1
    end
    @inbounds counts[:, c] .= cnt
    return nothing
end

function dist_scatter!(tv::Vector{S}, k::Vector{S}, bk, start, r, c, nb) where {S}
    pos = start[:, c]
    @inbounds for i in r
        d = bk[i] + 1
        tv[pos[d]] = k[i]
        pos[d] += 1
    end
    return nothing
end

function dist_finish!(k::Vector{S}, tv::Vector{S}, bstart, bs) where {S}
    GC.@preserve k tv for b in bs
        a, m = bstart[b], bstart[b+1] - bstart[b]
        m == 0 && continue
        sort_keys!(pointer(tv, a), m, 1)
        unsafe_copyto!(pointer(k, a), pointer(tv, a), m)
    end
    return nothing
end
