# Pivot selection and the quicksort recursion (docs/DESIGN.md §2 and §4).

const UNROLL = 8

@inline function xorshift(s::UInt64)
    s ⊻= s << 13
    s ⊻= s >> 7
    s ⊻= s << 17
    return s
end

@inline med3(a, b, c) = max(min(a, b), min(max(a, b), c))

@inline function sample(p::Ptr{T}, n::Int, s::UInt64) where {T}
    N = lanes(T)
    s = xorshift(s)
    # Offset in [0, n - N] by multiply-high instead of a division.
    o = (UInt128(s) * UInt128(n - N + 1)) >> 64
    return vld(Val(N), p, o % Int), s
end

"""
    choose_pivot(p, n, s) -> (pivot, s)

Median of the lane-wise medians of three of three vectors loaded at pseudo-random
offsets. Needs `n ≥ N`; `s` is the generator state.
"""
function choose_pivot(p::Ptr{T}, n::Int, s::UInt64) where {T}
    a1, s = sample(p, n, s); a2, s = sample(p, n, s); a3, s = sample(p, n, s)
    a4, s = sample(p, n, s); a5, s = sample(p, n, s); a6, s = sample(p, n, s)
    a7, s = sample(p, n, s); a8, s = sample(p, n, s); a9, s = sample(p, n, s)
    m = med3(med3(a1, a2, a3), med3(a4, a5, a6), med3(a7, a8, a9))
    return median_lane(m), s
end

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

"""
    qsort!(p, n, buf, depth, s) -> s

Sort `n` keys at `p`. `buf` holds `basecase_max(T)` keys; past `depth` levels the
range is handed to `Base.sort!`.
"""
function qsort!(p::Ptr{T}, n::Int, buf::Ptr{T}, depth::Int, s::UInt64) where {T}
    while n > basecase_max(T)
        if depth == 0
            sort!(unsafe_wrap(Array, p, n))
            return s
        end
        depth -= 1
        pivot, s = choose_pivot(p, n, s)
        m = partition!(p, n, pivot, buf, Val(UNROLL))
        if m == n
            # Nothing above the pivot: it is the maximum. Done if all keys are
            # equal; otherwise split off the copies of the minimum.
            mn, mx = vextrema(p, n)
            mn == mx && return s
            m = partition!(p, n, mn, buf, Val(UNROLL))
            p += m * sizeof(T)
            n -= m
            continue
        end
        # Recurse on the smaller side, loop on the larger one.
        if m <= n - m
            s = qsort!(p, m, buf, depth, s)
            p += m * sizeof(T)
            n -= m
        else
            s = qsort!(p + m * sizeof(T), n - m, buf, depth, s)
            n = m
        end
    end
    small_sort!(p, n, buf)
    return s
end

function sort_keys!(p::Ptr{T}, n::Int, ntasks::Int=1) where {T}
    n <= 1 && return nothing
    depth = 2 * (8 * sizeof(Int) - leading_zeros(n)) + 4
    s = 0x9e3779b97f4a7c15 ⊻ UInt64(n)
    if ntasks > 1 && n > PAR_MIN
        pqsort!(p, n, depth, s, ntasks)
    else
        buf = scratch(T)
        GC.@preserve buf qsort!(p, n, pointer(buf), depth, s)
    end
    return nothing
end

"""
    presorted!(k) -> Bool

Return `true` if `k` is sorted, after reversing it if it is in non-increasing
order. Both scans stop at the first key out of order, so random inputs cost
almost nothing.
"""
function presorted!(k::AbstractVector)
    issorted(k) && return true
    if issorted(k; rev=true)
        reverse!(k)
        return true
    end
    return false
end
