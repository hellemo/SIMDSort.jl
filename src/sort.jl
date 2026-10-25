# Pivot selection and the quicksort recursion (docs/src/design.md §2 and §4).
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

@inline function sample(p::KeyRef{T}, n::Int, s::UInt64) where {T}
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
function choose_pivot(p::KeyRef{T}, n::Int, s::UInt64) where {T}
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
function choose_pivot(p::KeyRef{T}, n::Int, s::UInt64, buf::KeyRef{T}) where {T}
    n > BIGPIVOT_MINN || return choose_pivot(p, n, s)
    N = lanes(T)
    m = basecase_max(T)
    for j in 0:N:m-1
        x, s = sample(p, n, s)
        vst(x, buf, j)
    end
    small_sort!(buf, m, buf)
    h = m ÷ 2
    pivot = ld(buf, h + 1)
    lo = h
    while lo > 0 && ld(buf, lo) == pivot
        lo -= 1
    end
    hi = h + 1
    while hi < m && ld(buf, hi + 1) == pivot
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
    # Insertion sort of N ≤ 16 scalars.
    for i in 2:N, j in i:-1:2
        t[j-1] <= t[j] && break
        t = Base.setindex(Base.setindex(t, t[j], j - 1), t[j-1], j)
    end
    return t[(N + 1) ÷ 2]
end

function vextrema(p::KeyRef{T}, n::Int) where {T}
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

# f(Val(prefetch)) with prefetching for ranges above PF_MIN_BYTES.
@inline with_prefetch(f::F, ::Type{T}, n::Int) where {F,T} = use_prefetch(T, n) ? f(Val(true)) : f(Val(false))

# The unrolled partition needs 2 UNROLL vectors of keys; ranges just above the
# base case can be smaller than that (with 512-bit vectors and UNROLL > 8) and
# take one vector per step.
@inline function partition_auto!(p::KeyRef{T}, n::Int, pivot::T, buf::KeyRef{T}) where {T}
    n < 2UNROLL * lanes(T) && return partition!(p, n, pivot, buf, Val(1))
    return with_prefetch(pf -> partition!(p, n, pivot, buf, Val(UNROLL), pf), T, n)
end
@inline function partition3_auto!(p::KeyRef{T}, n::Int, pivot::T, buf::KeyRef{T}) where {T}
    n < 2UNROLL * lanes(T) && return partition3!(p, n, pivot, buf, Val(1))
    return with_prefetch(pf -> partition3!(p, n, pivot, buf, Val(UNROLL), pf), T, n)
end

"""
    split!(p, n, pivot, dup, buf, partition) -> (a, b)

Partition around `pivot`. Returns `(a, b)`: the keys in `[0, a)` are below the
pivot (or equal to it if `a == b`), those in `[b, n)` above it, and those in
`[a, b)` equal to it, so they are done. When `dup` is set (the sample suggests
many keys equal to the pivot), one three-way pass splits off the equal block;
otherwise `partition(p, n, pivot)` makes a two-way split and `a == b`.
"""
@inline function split!(p::KeyRef{T}, n::Int, pivot::T, dup::Bool, buf::KeyRef{T},
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
function qsort!(p::KeyRef{T}, n::Int, buf::KeyRef{T}, depth::Int, s::UInt64,
                lo::T=typemin(T), hi::T=typemax(T)) where {T}
    while n > basecase_max(T)
        if depth == 0
            count!(:depth_limit_fallbacks)
            sort!(keyarray(p, n))
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
            p = shift(p, m)
            n -= m
            lo, hi = mn, mx
            continue
        end
        # Recurse on the smaller side, loop on the larger one. Keys left of b
        # are ≤ pivot, keys from b on are ≥ pivot.
        if a <= n - b
            s = qsort!(p, a, buf, depth, s, lo, pivot)
            p = shift(p, b)
            n -= b
            lo = pivot
        else
            s = qsort!(shift(p, b), n - b, buf, depth, s, pivot, hi)
            n = a
            hi = pivot
        end
    end
    count_leaf!(n)
    @hotphase 2 n leaf_sort!(p, n, buf, lo, hi)
    return s
end

@inline leaf_sort!(p::KeyRef{T}, n::Int, buf::KeyRef{T}, lo::T, hi::T) where {T} = small_sort!(p, n, buf)

# [F5 §III-C] AVX2 has no 64-bit integer min/max (a compare and two blends), but it has
# vminpd/vmaxpd. If the keys of a leaf lie within [lo, hi] with hi - lo < 2^62,
# the offsets x - lo, read as Float64 bit patterns, are finite non-negative
# doubles in the same order as the integers (subnormals included, unless they are
# flushed to zero), so the leaf can be sorted with the Float64 networks.
# The CPU target, decided when the code is compiled (SIMD.Targets is meant for
# generated code). AVX-512 (F + VL) has native 64-bit integer min/max
# (vpminsq/vpmaxsq), so the double-domain detour for Int64 leaves is only for
# AVX targets without it.
struct AVX512Target end
struct AVXTarget end
struct GenericTarget end
@generated function target()
    SIMD.Targets.has_feature(:avx512f) && SIMD.Targets.has_feature(:avx512vl) && return :(AVX512Target())
    SIMD.Targets.has_feature(:avx) && return :(AVXTarget())
    return :(GenericTarget())
end
has_avx() = !(target() isa GenericTarget)
has_avx512() = target() isa AVX512Target
double_leaves(::AVXTarget) = true
double_leaves(_) = false

@inline function leaf_sort!(p::KeyRef{Int64}, n::Int, buf::KeyRef{Int64}, lo::Int64, hi::Int64)
    if double_leaves(target()) && n > 8 && (hi - lo) % UInt64 < UInt64(1) << 62 &&
            !get_zero_subnormals()
        small_sort!(p, n, buf, OffsetKeys(lo))
    else
        small_sort!(p, n, buf)
    end
    return nothing
end

function sort_keys!(p::KeyRef{T}, n::Int, ntasks::Int=1) where {T}
    n <= 1 && return nothing
    depth = 2 * (8 * sizeof(Int) - leading_zeros(n)) + 4
    s = 0x9e3779b97f4a7c15 ⊻ UInt64(n)
    if ntasks > 1 && n > PAR_MIN
        pqsort!(p, n, depth, s, ntasks, n)
    else
        buf = scratch(T)
        GC.@preserve buf qsort!(p, n, scratchref(p, buf), depth, s)
    end
    return nothing
end

"""
    presorted!(k, order=Forward, ntasks=1) -> Bool

Return `true` if `k` is sorted in `order`, after reversing it if it is in the
opposite order. Both scans stop at the first key out of order, so random inputs
cost almost nothing.
"""
presorted!(k::AbstractVector, rev::Bool = false, ntasks::Int = 1) = presorted!(k, ordering(rev), ntasks)

function presorted!(k::AbstractVector, o::Ord, ntasks::Int)
    rev = isrev(o)
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
    return tmapreduce(&, n, ntasks; minlen = 0) do r
        issorted(view(k, first(r):min(last(r) + 1, n)); rev)
    end
end

function preverse!(k::AbstractVector, ntasks::Int)
    n = length(k)
    tforeach(n ÷ 2, ntasks; minlen = 0) do r
        @inbounds for i in r
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
    counting_sort!(v, order, ntasks) -> Bool

If the integers in `v` span a range of at most `COUNT_MAXRANGE` values and fewer
than `length(v) ÷ 4`, sort `v` in `order` by counting each value and
writing the runs, and return `true`. Otherwise leave `v` unchanged and return
`false`. Works on the original values, so no key map is needed.

A sample of 1024 keys rejects wide ranges and sets up a count table over the
sample's range plus a margin. Keys outside it go to an overflow slot; if any do,
the table is redone over the exact range from a min/max scan.
"""
function counting_sort!(v::KeyVector{T}, o::Ord, ntasks::Int) where {T<:Integer}
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
        owner = keyowner(v)
        mn, mx = GC.@preserve owner vextrema(keyref(v), n)
        r = Int128(mx) - Int128(mn)
        (r <= COUNT_MAXRANGE && r < n ÷ 4) || return false
        lo = Int128(mn)
        counts = count_all(v, mn, Int(r + 1), ntasks)
    end
    pop!(counts)                                # the (empty) overflow slot
    f = findfirst(!iszero, counts)
    l = findlast(!iszero, counts)
    (l - f <= COUNT_MAXRANGE && l - f < n ÷ 4) || return false
    fill_runs!(v, view(counts, f:l), T(lo + f - 1), o, ntasks)
    count!(:counting_sorts)
    return true
end

# Counts per slot of [base, base + W) plus one overflow slot, by up to `ntasks`
# tasks.
function count_all(v::AbstractVector{T}, base::T, W::Int, ntasks::Int) where {T}
    return tmapreduce(r -> count_keys(v, r, base, W), +, length(v), ntasks; minlen = 0)
end

# Four interleaved tables, so that runs of equal keys do not serialize on one
# counter. The slot is (x - base) mod 2^w, which is below W exactly for the keys in
# [base, base + W) because that interval does not wrap; the others land in slot W.
function count_keys(v::AbstractVector{T}, c::UnitRange{Int}, base::T, W::Int) where {T}
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

# Write counts[d] copies of first + (d - 1) for each d (descending for
# Reverse). With several tasks, each fills an equal slice of `v`.
function fill_runs!(v::AbstractVector{T}, counts::AbstractVector{Int}, first::T, o::Ord,
                    ntasks::Int) where {T}
    m = length(counts)
    ord = runorder(o, m)
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
    tforeach(r -> fill_slice(r.start, r.stop), length(v), ntasks; minlen = 0)
    return v
end

runorder(::Base.Order.ForwardOrdering, m::Int) = 1:m
runorder(::Base.Order.ReverseOrdering, m::Int) = m:-1:1
