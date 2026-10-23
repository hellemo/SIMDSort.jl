# Task-parallel sorting: the smaller side of each large partition goes to a new
# task, and ranges above PPART_MIN are partitioned by several tasks at once.

scratch(::Type{T}) where {T} = Vector{T}(undef, max(basecase_max(T), (2UNROLL + 1) * lanes(T)))

# Ranges above this size may be sorted by a separate task.
const PAR_MIN = 1 << 16
# Ranges above this size are partitioned in parallel.
const PPART_MIN = 1 << 20

"""
    ppartition!(p, n, pivot, ntasks) -> m

Parallel version of `partition!`. Each of `ntasks` chunks is partitioned in place
by its own task. Then the keys `> pivot` that ended up in `[0, m)` are swapped
with the keys `≤ pivot` in `[m, n)`; both sets have the same size, `X`, and each
task swaps an equal slice of them.
"""
function ppartition!(p::Ptr{T}, n::Int, pivot::T, ntasks::Int) where {T}
    bounds = [div(n * i, ntasks) for i in 0:ntasks]
    counts = Vector{Int}(undef, ntasks)
    @sync for i in 1:ntasks
        Threads.@spawn begin
            buf = scratch(T)
            lo = bounds[i]
            GC.@preserve buf begin
                counts[i] = partition_auto!(p + lo * sizeof(T), bounds[i+1] - lo, pivot,
                                            pointer(buf))
            end
        end
    end
    m = sum(counts)
    # Misplaced runs, in order: `> pivot` runs inside [0, m), `≤ pivot` runs
    # inside [m, n). Run i of chunk c is [c_lo + count, c_hi) and [c_lo, c_lo + count).
    big = Tuple{Int,Int}[]      # (start, length)
    small = Tuple{Int,Int}[]
    for i in 1:ntasks
        lo, hi, mid = bounds[i], bounds[i+1], bounds[i] + counts[i]
        a, b = mid, min(hi, m)
        b > a && push!(big, (a, b - a))
        a, b = max(lo, m), mid
        b > a && push!(small, (a, b - a))
    end
    X = sum(last, big; init=0)
    X == 0 && return m
    @sync for i in 1:ntasks
        from, to = div(X * (i - 1), ntasks), div(X * i, ntasks)
        to > from && Threads.@spawn swap_runs!(p, big, small, from, to)
    end
    return m
end

# Swap the misplaced keys with ranks [from, to) of `big` and `small`.
function swap_runs!(p::Ptr{T}, big, small, from::Int, to::Int) where {T}
    i, oi = locate(big, from)
    j, oj = locate(small, from)
    k = from
    while k < to
        (sa, la), (sb, lb) = big[i], small[j]
        len = min(la - oi, lb - oj, to - k)
        swap_block!(p + (sa + oi) * sizeof(T), p + (sb + oj) * sizeof(T), len)
        k += len
        oi += len
        oj += len
        oi == la && (i += 1; oi = 0)
        oj == lb && (j += 1; oj = 0)
    end
    return nothing
end

# Run index and offset of the key with rank `r` in a list of runs.
function locate(runs, r::Int)
    for (i, (_, l)) in enumerate(runs)
        r < l && return i, r
        r -= l
    end
    return length(runs) + 1, 0
end

function swap_block!(a::Ptr{T}, b::Ptr{T}, len::Int) where {T}
    N = lanes(T)
    i = 0
    while i + N <= len
        x = vld(Val(N), a, i)
        vst(vld(Val(N), b, i), a, i)
        vst(x, b, i)
        i += N
    end
    while i < len
        x = unsafe_load(a, i + 1)
        unsafe_store!(a, unsafe_load(b, i + 1), i + 1)
        unsafe_store!(b, x, i + 1)
        i += 1
    end
    return nothing
end

# Ranges are handed to new tasks down to `total ÷ (GRAIN_FACTOR * ntasks)` keys:
# many more tasks than threads, so that the scheduler balances fast and slow
# cores.
const GRAIN_FACTOR = 16

"""
    pqsort!(p, n, depth, s, ntasks, total)

Like `qsort!`, but while a range is larger than the grain
`max(PAR_MIN, total ÷ (GRAIN_FACTOR * ntasks))` the smaller side of each
partition is sorted by a new task. Ranges larger than `PPART_MIN` are
partitioned with `ppartition!`, by as many tasks as their share
`ntasks * n / total`. Waits for all tasks it started.
"""
function pqsort!(p::Ptr{T}, n::Int, depth::Int, s::UInt64, ntasks::Int, total::Int) where {T}
    buf = scratch(T)
    tasks = Task[]
    alldone = false
    grain = max(PAR_MIN, total ÷ (GRAIN_FACTOR * ntasks))
    GC.@preserve buf begin
        while n > grain && depth > 0
            depth -= 1
            pivot, dup, s = choose_pivot(p, n, s)
            nt = clamp(round(Int, ntasks * n / total), 1, ntasks)
            if dup && nt > 1 && n > PPART_MIN && has_prev(pivot)
                # The three-way partition is serial. For a range this large, two
                # parallel partitions are faster: around the pivot, then the left
                # side around the next smaller key, which splits off the keys
                # equal to the pivot.
                b = part(p, n, pivot, pointer(buf), nt)
                a = b >= 2UNROLL * lanes(T) ? part(p, b, prevkey(pivot), pointer(buf), nt) : b
            else
                a, b = split!(p, n, pivot, dup, pointer(buf),
                              (p, n, x) -> part(p, n, x, pointer(buf), nt))
            end
            if a == n
                # As in qsort!: the pivot is the maximum.
                mn, mx = vextrema(p, n)
                if mn == mx
                    alldone = true
                    break
                end
                m = part(p, n, mn, pointer(buf), nt)
                p += m * sizeof(T)
                n -= m
                continue
            end
            # Hand the smaller side to a new task, keep the larger one. [a, b) is
            # done.
            small = min(a, n - b)
            ps, pl, nl = a <= n - b ? (p, p + b * sizeof(T), n - b) : (p + b * sizeof(T), p, a)
            s = xorshift(s)
            push!(tasks, Threads.@spawn pqsort!($ps, $small, $depth, $s, $ntasks, $total))
            p, n = pl, nl
        end
        alldone || qsort!(p, n, pointer(buf), depth, s)
    end
    foreach(wait, tasks)
    return nothing
end

@inline part(p::Ptr{T}, n, pivot, buf, ntasks) where {T} =
    n > PPART_MIN && ntasks > 1 ? ppartition!(p, n, pivot, ntasks) :
                                  partition_auto!(p, n, pivot, buf)
