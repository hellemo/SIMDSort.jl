# Pivot selection and the quicksort recursion (docs/DESIGN.md §2 and §4).
#
# [Hoa62] quicksort; [WBGS22 Alg. 1]: handling of an empty right side (min/max
# scan, then the minimum as pivot); [WBGS22 §2.2]: pivot from lane-wise medians
# of three over randomly placed vectors; [Mus97]: depth limit with a fallback
# sort; [F5 §III-C]: 64-bit integer leaves sorted with double min/max.

const UNROLL = 8

@inline function xorshift(s::UInt64)
    s ⊻= s << 13
    s ⊻= s >> 7
    s ⊻= s << 17
    return s
end

@inline med3(a, b, c) = vmax(vmin(a, b), vmin(vmax(a, b), c))

@inline function sample(p::Ptr{T}, n::Int, s::UInt64) where {T}
    N = lanes(T)
    s = xorshift(s)
    # Offset in [0, n - N] by multiply-high instead of a division.
    o = (UInt128(s) * UInt128(n - N + 1)) >> 64
    return vld(Val(N), p, o % Int), s
end

"""
    choose_pivot(p, n, s) -> (pivot, dup, s)

Median of the lane-wise medians of three of three vectors loaded at pseudo-random
offsets. `dup` is set when at least N of the 9N sampled keys equal the pivot,
which suggests a large block of equal keys. Needs `n ≥ N`; `s` is the generator
state.
"""
function choose_pivot(p::Ptr{T}, n::Int, s::UInt64) where {T}
    N = lanes(T)
    a1, s = sample(p, n, s); a2, s = sample(p, n, s); a3, s = sample(p, n, s)
    a4, s = sample(p, n, s); a5, s = sample(p, n, s); a6, s = sample(p, n, s)
    a7, s = sample(p, n, s); a8, s = sample(p, n, s); a9, s = sample(p, n, s)
    m = med3(med3(a1, a2, a3), med3(a4, a5, a6), med3(a7, a8, a9))
    pivot = median_lane(m)
    pv = Vec{N,T}(pivot)
    c = count(a1 == pv) + count(a2 == pv) + count(a3 == pv) + count(a4 == pv) +
        count(a5 == pv) + count(a6 == pv) + count(a7 == pv) + count(a8 == pv) +
        count(a9 == pv)
    return pivot, c >= N, s
end

# Ranges above this size take the exact median of a larger sample.
# Measured: exact sample medians improve the split only slightly (20.8 → 20.1
# passes for Float64 at 2^26 keys), but detect duplicates better.
const BIGPIVOT_MINN = 1 << 18

"""
    choose_pivot(p, n, s, buf) -> (pivot, dup, s)

For ranges above `BIGPIVOT_MINN` keys: the exact median of `basecase_max(T)` keys
loaded as vectors from pseudo-random offsets, sorted with the base-case network
in `buf`. `dup` is set when at least 1/9 of the sample equals the pivot.
Otherwise as `choose_pivot(p, n, s)`.
"""
function choose_pivot(p::Ptr{T}, n::Int, s::UInt64, buf::Ptr{T}) where {T}
    n > BIGPIVOT_MINN || return choose_pivot(p, n, s)
    N = lanes(T)
    m = basecase_max(T)
    for j in 0:N:m-1
        x, s = sample(p, n, s)
        vst(x, buf, j)
    end
    small_sort!(buf, m, buf)
    h = m ÷ 2
    pivot = unsafe_load(buf, h + 1)
    lo = h
    while lo > 0 && unsafe_load(buf, lo) == pivot
        lo -= 1
    end
    hi = h + 1
    while hi < m && unsafe_load(buf, hi + 1) == pivot
        hi += 1
    end
    return pivot, 9 * (hi - lo) >= m, s
end

# The largest key below `x`, if there is one (floats: -0.0 and 0.0 count as equal).
@inline has_prev(x::Integer) = x != typemin(x)
@inline has_prev(x::AbstractFloat) = x != typemin(x)
@inline prevkey(x::Integer) = x - one(x)
@inline prevkey(x::AbstractFloat) = prevfloat(x)

@inline function median_lane(v::Vec{N,T}) where {N,T}
    t = Tuple(v)
    # Insertion sort of N ≤ 8 scalars.
    for i in 2:N, j in i:-1:2
        t[j-1] <= t[j] && break
        t = Base.setindex(Base.setindex(t, t[j], j - 1), t[j-1], j)
    end
    return t[(N + 1) ÷ 2]
end

function vextrema(p::Ptr{T}, n::Int) where {T}
    N = lanes(T)
    lo = hi = vld(Val(N), p, 0)
    i = N
    while i + N <= n
        x = vld(Val(N), p, i)
        lo = min(lo, x)
        hi = max(hi, x)
        i += N
    end
    x = vld(Val(N), p, n - N)       # overlapping last vector
    return minimum(min(lo, x)), maximum(max(hi, x))
end

@inline partition_auto!(p::Ptr{T}, n::Int, pivot::T, buf::Ptr{T}) where {T} =
    use_prefetch(T, n) ? partition!(p, n, pivot, buf, Val(UNROLL), Val(true)) :
                         partition!(p, n, pivot, buf, Val(UNROLL))

@inline partition3_auto!(p::Ptr{T}, n::Int, pivot::T, buf::Ptr{T}) where {T} =
    use_prefetch(T, n) ? partition3!(p, n, pivot, buf, Val(UNROLL), Val(true)) :
                         partition3!(p, n, pivot, buf, Val(UNROLL))

"""
    split!(p, n, pivot, dup, buf, partition) -> (a, b)

Partition around `pivot`. Returns `(a, b)`: the keys in `[0, a)` are below the
pivot (or equal to it if `a == b`), those in `[b, n)` above it, and those in
`[a, b)` equal to it, so they are done. When `dup` is set (the sample suggests
many keys equal to the pivot), one three-way pass splits off the equal block;
otherwise `partition(p, n, pivot)` makes a two-way split and `a == b`.
"""
@inline function split!(p::Ptr{T}, n::Int, pivot::T, dup::Bool, buf::Ptr{T},
                        partition::F) where {F,T}
    dup && return partition3_auto!(p, n, pivot, buf)
    m = partition(p, n, pivot)
    return m, m
end

"""
    qsort!(p, n, buf, depth, s, lo=typemin(T), hi=typemax(T)) -> s

Sort `n` keys at `p`, all in `[lo, hi]`. `buf` holds `basecase_max(T)` keys;
past `depth` levels the range is handed to `Base.sort!`. The bounds narrow with
every partition and let 64-bit integer base cases use double min/max
(`leaf_sort!`).
"""
function qsort!(p::Ptr{T}, n::Int, buf::Ptr{T}, depth::Int, s::UInt64,
                lo::T=typemin(T), hi::T=typemax(T)) where {T}
    while n > basecase_max(T)
        if depth == 0
            count!(:depth_limit_fallbacks)
            sort!(unsafe_wrap(Array, p, n))
            return s
        end
        depth -= 1
        pivot, dup, s = @hotphase 1 n choose_pivot(p, n, s, buf)
        a, b = @partphase(n, dup, split!(p, n, pivot, dup, buf, (p, n, x) -> partition_auto!(p, n, x, buf)))
        if a == n
            # Nothing above the pivot: it is the maximum. Done if all keys are
            # equal; otherwise split off the copies of the minimum.
            count!(:pivot_was_maximum)
            mn, mx = vextrema(p, n)
            mn == mx && (count!(:equal_ranges); return s)
            m = partition_auto!(p, n, mn, buf)
            p += m * sizeof(T)
            n -= m
            lo, hi = mn, mx
            continue
        end
        # Recurse on the smaller side, loop on the larger one. Keys left of b
        # are ≤ pivot, keys from b on are ≥ pivot.
        if a <= n - b
            s = qsort!(p, a, buf, depth, s, lo, pivot)
            p += b * sizeof(T)
            n -= b
            lo = pivot
        else
            s = qsort!(p + b * sizeof(T), n - b, buf, depth, s, pivot, hi)
            n = a
            hi = pivot
        end
    end
    count_leaf!(n)
    @hotphase 2 n leaf_sort!(p, n, buf, lo, hi)
    return s
end

@inline leaf_sort!(p::Ptr{T}, n::Int, buf::Ptr{T}, lo::T, hi::T) where {T} = small_sort!(p, n, buf)

# [F5 §III-C] AVX2 has no 64-bit integer min/max (a compare and two blends), but it has
# vminpd/vmaxpd. If the keys of a leaf lie within [lo, hi] with hi - lo < 2^62,
# the offsets x - lo, read as Float64 bit patterns, are finite non-negative
# doubles in the same order as the integers (subnormals included, unless they are
# flushed to zero), so the leaf can be sorted with the Float64 networks.
# Decided when the code is compiled (SIMD.Targets is meant for generated code).
@generated has_avx() = SIMD.Targets.has_feature(:avx)
# AVX-512 (F + VL) has native 64-bit integer min/max (vpminsq/vpmaxsq), so the
# double-domain detour for Int64 leaves is not needed there.
@generated has_avx512() = SIMD.Targets.has_feature(:avx512f) && SIMD.Targets.has_feature(:avx512vl)

@inline function leaf_sort!(p::Ptr{Int64}, n::Int, buf::Ptr{Int64}, lo::Int64, hi::Int64)
    if n > 8 && (hi - lo) % UInt64 < UInt64(1) << 62 && has_avx() && !has_avx512() &&
            !get_zero_subnormals()
        small_sort_offsets!(p, n, buf, lo)
    else
        small_sort!(p, n, buf)
    end
    return nothing
end

@inline function small_sort_offsets!(p::Ptr{Int64}, n::Int, buf::Ptr{Int64}, lo::Int64)
    N = lanes(Int64)
    R = MAXROWS
    if n <= N * N
        netsort_offsets!(p, n, buf, lo, Val(N))
    elseif n <= 2N * N
        netsort_offsets!(p, n, buf, lo, Val(2N))
    elseif n <= 4N * N && 4N <= R
        netsort_offsets!(p, n, buf, lo, Val(4N))
    else
        netsort_offsets!(p, n, buf, lo, Val(R))
    end
    return nothing
end

@inline function netsort_offsets!(p::Ptr{Int64}, n::Int, buf::Ptr{Int64}, lo::Int64,
                                  ::Val{R}) where {R}
    N = lanes(Int64)
    unsafe_copyto!(buf, p, n)
    # Padding: lo + (bits of +Inf), so that its offset is +Inf.
    pad = lo + reinterpret(Int64, Inf)
    for i in n+1:R*N
        unsafe_store!(buf, pad, i)
    end
    lv = Vec{N,Int64}(lo)
    vs = ntuple(r -> reinterpret(Vec{N,Float64}, vload(Vec{N,Int64}, buf + (r - 1) * N * 8) - lv), Val(R))
    vs = sortnet(vs)
    nfull = n ÷ N
    for r in 1:R
        x = reinterpret(Vec{N,Int64}, vs[r]) + lv
        if r <= nfull
            vstore(x, p + (r - 1) * N * 8)
        elseif r == nfull + 1
            vstore(x, buf)
        end
    end
    unsafe_copyto!(p + nfull * N * 8, buf, n - nfull * N)
    return nothing
end

function sort_keys!(p::Ptr{T}, n::Int, ntasks::Int=1) where {T}
    n <= 1 && return nothing
    depth = 2 * (8 * sizeof(Int) - leading_zeros(n)) + 4
    s = 0x9e3779b97f4a7c15 ⊻ UInt64(n)
    if ntasks > 1 && n > PAR_MIN
        pqsort!(p, n, depth, s, ntasks, n)
    else
        buf = scratch(T)
        GC.@preserve buf qsort!(p, n, pointer(buf), depth, s)
    end
    return nothing
end

"""
    presorted!(k, rev=false, ntasks=1) -> Bool

Return `true` if `k` is sorted (in reverse if `rev`), after reversing it if it is
in the opposite order. Both scans stop at the first key out of order, so random inputs cost
almost nothing.
"""
function presorted!(k::AbstractVector, rev::Bool=false, ntasks::Int=1)
    if ntasks <= 1 || length(k) < PRESORTED_PAR_MIN
        issorted(k; rev) && return true
        if issorted(k; rev=!rev)
            reverse!(k)
            return true
        end
        return false
    end
    # In parallel: a single thread reads only about 13 GB/s on a server core,
    # so a serial scan of sorted input cost the same at every thread count.
    psorted(k, rev, ntasks) && return true
    if psorted(k, !rev, ntasks)
        preverse!(k, ntasks)
        return true
    end
    return false
end

const PRESORTED_PAR_MIN = 1 << 20

# issorted in ntasks chunks that overlap by one key (every chunk stops at its
# first key out of order, so random input still costs almost nothing).
function psorted(k::AbstractVector, rev::Bool, ntasks::Int)
    n = length(k)
    chunks = collect(Iterators.partition(1:n, cld(n, ntasks)))
    res = Vector{Bool}(undef, length(chunks))
    @sync for (i, c) in enumerate(chunks)
        Threads.@spawn res[i] = issorted(view(k, first(c):min(last(c) + 1, n)); rev)
    end
    return all(res)
end

function preverse!(k::AbstractVector, ntasks::Int)
    n = length(k)
    h = n ÷ 2
    @sync for c in Iterators.partition(1:h, cld(h, ntasks))
        Threads.@spawn @inbounds for i in c
            k[i], k[n+1-i] = k[n+1-i], k[i]
        end
    end
    return k
end

# Key ranges up to this size (and below n/4) are sorted by counting.
const COUNT_MAXRANGE = 1 << 16
const COUNT_MINN = 1 << 12
# Slack around the sample's range for values the sample missed.
const COUNT_MARGIN = 32

"""
    counting_sort!(v, rev, ntasks) -> Bool

If the integers in `v` span a range of at most `COUNT_MAXRANGE` values and fewer
than `length(v) ÷ 4`, sort `v` (in reverse if `rev`) by counting each value and
writing the runs, and return `true`. Otherwise leave `v` unchanged and return
`false`. Works on the original values, so no key map is needed.

A sample of 1024 keys rejects wide ranges and sets up a count table over the
sample's range plus a margin. Keys outside it go to an overflow slot; if any do,
the table is redone over the exact range from a min/max scan.
"""
function counting_sort!(v::Vector{T}, rev::Bool, ntasks::Int) where {T<:Integer}
    n = length(v)
    n < COUNT_MINN && return false
    # A sample's range is at most the full range, so a wide sample rejects early
    # without scanning all keys.
    smn = smx = v[1]
    @inbounds for i in 1:(n ÷ 1024):n
        smn = min(smn, v[i])
        smx = max(smx, v[i])
    end
    Int128(smx) - Int128(smn) > COUNT_MAXRANGE && return false
    # Table over [base, base + W), clamped to T's range so that it never wraps.
    lo = max(Int128(smn) - COUNT_MARGIN, Int128(typemin(T)))
    hi = min(Int128(smx) + COUNT_MARGIN, Int128(typemax(T)))
    counts = count_all(v, T(lo), Int(hi - lo + 1), ntasks)
    if counts[end] != 0
        # Keys outside the table: count again over the exact range.
        mn, mx = GC.@preserve v vextrema(pointer(v), n)
        r = Int128(mx) - Int128(mn)
        (r <= COUNT_MAXRANGE && r < n ÷ 4) || return false
        lo = Int128(mn)
        counts = count_all(v, mn, Int(r + 1), ntasks)
    end
    pop!(counts)                                # the (empty) overflow slot
    f = findfirst(!iszero, counts)
    l = findlast(!iszero, counts)
    (l - f <= COUNT_MAXRANGE && l - f < n ÷ 4) || return false
    fill_runs!(v, view(counts, f:l), T(lo + f - 1), rev, ntasks)
    count!(:counting_sorts)
    return true
end

# Counts per slot of [base, base + W) plus one overflow slot, by up to `ntasks`
# tasks.
function count_all(v::Vector{T}, base::T, W::Int, ntasks::Int) where {T}
    chunks = collect(Iterators.partition(1:length(v), cld(length(v), max(1, ntasks))))
    length(chunks) == 1 && return count_keys(v, chunks[1], base, W)
    parts = Vector{Vector{Int}}(undef, length(chunks))
    @sync for (i, c) in enumerate(chunks)
        Threads.@spawn parts[i] = count_keys(v, c, base, W)
    end
    return reduce(+, parts)
end

# Four interleaved tables, so that runs of equal keys do not serialize on one
# counter. The slot is (x - base) mod 2^w, which is below W exactly for the keys in
# [base, base + W) because that interval does not wrap; the others land in slot W.
function count_keys(v::Vector{T}, c::UnitRange{Int}, base::T, W::Int) where {T}
    U = unsigned(T)
    w = UInt(W)
    t = zeros(Int, W + 1, 4)
    slot(x) = min(UInt((x - base) % U), w) + 1
    i = first(c)
    @inbounds while i + 3 <= last(c)
        t[slot(v[i]), 1] += 1
        t[slot(v[i+1]), 2] += 1
        t[slot(v[i+2]), 3] += 1
        t[slot(v[i+3]), 4] += 1
        i += 4
    end
    @inbounds while i <= last(c)
        t[slot(v[i]), 1] += 1
        i += 1
    end
    return vec(sum(t; dims=2))
end

# Write counts[d] copies of first + (d - 1) for each d (descending if `rev`).
# With several tasks, each fills an equal slice of `v`.
function fill_runs!(v::Vector{T}, counts::AbstractVector{Int}, first::T, rev::Bool,
                    ntasks::Int) where {T}
    m = length(counts)
    ord = rev ? (m:-1:1) : (1:m)
    starts = cumsum([1; [counts[d] for d in ord][1:end-1]])
    function fill_slice(a, b)
        j = searchsortedlast(starts, a)
        i = a
        @inbounds while i <= b
            d = ord[j]
            e = min(b, starts[j] + counts[d] - 1)
            x = first + (d - 1) % T
            for k in i:e
                v[k] = x
            end
            i = e + 1
            j += 1
        end
    end
    n = length(v)
    if ntasks <= 1
        fill_slice(1, n)
    else
        @sync for t in 1:ntasks
            a, b = div(n * (t - 1), ntasks) + 1, div(n * t, ntasks)
            a <= b && Threads.@spawn fill_slice(a, b)
        end
    end
    return v
end
