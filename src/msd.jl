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

# Run f(c) for every chunk index c in 1:nc, one task each when nc > 1.
function foreach_chunk_index(f::F, nc::Int) where {F}
    if nc <= 1
        f(1)
    else
        @sync for c in 1:nc
            Threads.@spawn f(c)
        end
    end
    return nothing
end

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
    # Splitters: leading words of a sorted random sample, every MSD_OVERSAMPLE-th
    # [IPS4o] (radix digits put skewed keys such as random floats into a
    # handful of buckets).
    nb = MSD_BUCKETS
    ts0 = INSTR[] ? time_ns() : 0
    smp = Vector{UInt64}(undef, nb * MSD_OVERSAMPLE)
    r = UInt64(n) | 1
    @inbounds for j in eachindex(smp)
        r = xorshift(r)
        smp[j] = f(v[1+Int(r % UInt64(n))])[1]
    end
    sort!(smp)
    # Few distinct leading words: the buckets would be unbalanced.
    if smp[1] == smp[end] || count(i -> smp[i] != smp[i-1], 2:length(smp)) < nb ÷ 4
        INSTR[] && record_phase!(:msd_sample, time_ns() - ts0)
        count!(:msd_skipped_fewlead)
        return permute_by!(v, wordkeys_sortperm(words_of(f, v, ntasks); rev, ntasks), ntasks)
    end
    # spl[j] for j in 1:nb-1 (padded at index nb): bucket(x) = #{j : spl[j] <= x}.
    spl = [smp[j*MSD_OVERSAMPLE] for j in 1:nb-1]
    push!(spl, typemax(UInt64))
    INSTR[] && record_phase!(:msd_sample, time_ns() - ts0)
    chunks = collect(Iterators.partition(1:n, cld(n, max(1, ntasks))))
    nc = length(chunks)
    counts = zeros(Int, nb, nc)
    bk = Vector{UInt8}(undef, n)
    tc = INSTR[] ? time_ns() : 0
    foreach_chunk_index(nc) do c
        cnt = zeros(Int, nb)
        @inbounds for i in chunks[c]
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
    INSTR[] && record_phase!(:msd_count, time_ns() - tc)
    if INSTR[]
        sizes = [bstart[b+1] - bstart[b] for b in 1:nb]
        count!(:msd_buckets_nonempty, count(>(0), sizes))
        countmax!(:msd_bucket_max, maximum(sizes))
    end
    ts = INSTR[] ? time_ns() : 0
    tv = similar(v)
    foreach_chunk_index(nc) do c
        pos = start[:, c]
        @inbounds for i in chunks[c]
            d = bk[i] + 1
            tv[pos[d]] = v[i]
            pos[d] += 1
        end
    end
    INSTR[] && record_phase!(:msd_scatter, time_ns() - ts)
    tf = INSTR[] ? time_ns() : 0
    # Sort every bucket from the scratch array back into v, in cache, with
    # scratch buffers reused across the buckets of a task (fresh buffers of this
    # size are malloc'ed and page-faulted each time).
    bmax = maximum(b -> bstart[b+1] - bstart[b], 1:nb)
    if ntasks <= 1
        finish_buckets!(v, tv, bstart, 1:nb, f, rev, bmax)
    else
        # Groups of buckets of about equal size, several per task.
        target = cld(n, 8ntasks)
        @sync begin
            b = 1
            while b <= nb
                g, sz = b, 0
                while g <= nb && (sz == 0 || sz + bstart[g+1] - bstart[g] <= target)
                    sz += bstart[g+1] - bstart[g]
                    g += 1
                end
                bs = b:g-1
                Threads.@spawn finish_buckets!(v, tv, bstart, $bs, f, rev, bmax)
                b = g
            end
        end
    end
    INSTR[] && record_phase!(:msd_finish, time_ns() - tf)
    return v
end

# Finish buckets bs with one set of scratch buffers.
function finish_buckets!(v, tv, bstart, bs, f::F, rev::Bool, bmax::Int) where {F}
    keys, idx = Vector{UInt64}(undef, bmax), Vector{Int}(undef, bmax)
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
# Sample-sort top level for 64-bit keys at high thread counts (docs/DESIGN.md
# §7j). DISABLED by default: on the 4P+4E development laptop it loses to the
# parallel quicksort at 8 threads (UInt64 2^27: 0.51 s against 0.44 s; 0.42 s
# only if the 1 GB scratch array is kept between calls, which costs too much
# memory), because the quicksort's time there goes to in-cache partitions,
# not to its memory-bound top levels. Kept for measurements on many-core
# machines (bench/idun), where the memory-bound top levels weigh more:
# `DIST_MINN[] = 1 << 24` enables it.
#
# One distribution into DIST_BUCKETS buckets (a count and a scatter, both
# parallel) replaces about log2(DIST_BUCKETS) partition passes over the whole
# array; each bucket is then quicksorted on one thread and copied back while
# in cache.

const DIST_BUCKETS = Ref(256)
const DIST_MINN = Ref(typemax(Int)) # keys; disabled (see above)
const DIST_MINTASKS = Ref(8)

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

function use_distribution(::Type{S}, n::Int, ntasks::Int) where {S}
    return sizeof(S) == 8 && ntasks >= DIST_MINTASKS[] && n >= DIST_MINN[]
end

"""
    distribute_sort!(p::Ptr{S}, n, ntasks) -> Bool

Sort the n keys at `p` with one sample-sort distribution pass and per-bucket
quicksorts. Returns `false` without changing anything if the sample has too
few distinct keys for balanced buckets.
"""
function distribute_sort!(p::Ptr{S}, n::Int, ntasks::Int) where {S}
    nb = DIST_BUCKETS[]
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
    chunks = collect(Iterators.partition(1:n, cld(n, ntasks)))
    nc = length(chunks)
    counts = zeros(Int, nb, nc)
    bk = Vector{UInt16}(undef, n)
    tc = INSTR[] ? time_ns() : 0
    foreach_chunk_index(nc) do c
        dist_count!(counts, bk, k, spl, chunks[c], c, Val(nb))
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
    INSTR[] && record_phase!(:dist_count, time_ns() - tc)
    INSTR[] && countmax!(:dist_bucket_max, maximum(b -> bstart[b+1] - bstart[b], 1:nb))
    ts = INSTR[] ? time_ns() : 0
    tv = Vector{S}(undef, n)
    foreach_chunk_index(nc) do c
        dist_scatter!(tv, k, bk, start, chunks[c], c, nb)
    end
    INSTR[] && record_phase!(:dist_scatter, time_ns() - ts)
    tf = INSTR[] ? time_ns() : 0
    target = cld(n, 8ntasks)
    @sync begin
        b = 1
        while b <= nb
            g, sz = b, 0
            while g <= nb && (sz == 0 || sz + bstart[g+1] - bstart[g] <= target)
                sz += bstart[g+1] - bstart[g]
                g += 1
            end
            Threads.@spawn dist_finish!(k, tv, bstart, $(b:g-1))
            b = g
        end
    end
    INSTR[] && record_phase!(:dist_finish, time_ns() - tf)
    return true
end

function dist_count!(counts, bk, k::Vector{S}, spl::Vector{S}, r, c, vnb::Val{nb}) where {S,nb}
    cnt = zeros(Int, nb)
    @inbounds for i in r
        d = bucket_of_n(spl, k[i], vnb)
        bk[i] = d % UInt16
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
