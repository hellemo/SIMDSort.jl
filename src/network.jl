# Base case: a vectorized sorting network over an R × N matrix held in R registers
# of N lanes (docs/DESIGN.md §5). The network code is generated at compile time
# from the comparator lists below.

# Comparators of Batcher's odd-even merge sort on `n` (a power of two) inputs,
# 0-based. 63 comparators for n = 16.
function batcher_pairs(n::Int)
    pairs = Tuple{Int,Int}[]
    p = 1
    while p < n
        k = p
        while k >= 1
            for j in (k % p):(2k):(n - 1 - k)
                for i in 0:min(k - 1, n - j - k - 1)
                    if (i + j) ÷ 2p == (i + j + k) ÷ 2p
                        push!(pairs, (i + j, i + j + k))
                    end
                end
            end
            k ÷= 2
        end
        p *= 2
    end
    return pairs
end

_v(i) = Symbol(:v, i)

# Constant one- and two-input shuffles; lane indices are 0-based.
_shuf(x, perm) = :(shufflevector($x, Val($(Tuple(perm)))))
_shuf(x, y, perm) = :(shufflevector($x, $y, Val($(Tuple(perm)))))
# Lane c from `x` where `takex(c)`, else from `y`; lowers to a blend.
_blend(x, y, takex, N) = _shuf(x, y, [takex(c) ? c : N + c for c in 0:N-1])

function network_body(R::Int, N::Int)
    ex = Expr[]
    push!(ex, :(($(_v.(0:R-1)...),) = vs))

    # 1. Sort the columns: each compare-exchange is one min and one max.
    for (a, b) in batcher_pairs(R)
        push!(ex, :(($(_v(a)), $(_v(b))) = (min($(_v(a)), $(_v(b))), max($(_v(a)), $(_v(b))))))
    end

    # 2. Bitonic merges of sorted runs of length s in column-major numbering
    #    i = c*R + ρ, from s = R up to R*N/2.
    s = R
    while s < R * N
        M = 2s ÷ R - 1                       # lane part of i ⊻ (2s - 1)
        perm = [c ⊻ M for c in 0:N-1]
        lower = c -> c & (s ÷ R) == 0        # lane belongs to the lower run
        for ρ in 0:(R ÷ 2 - 1)
            a, b = _v(ρ), _v(R - 1 - ρ)
            push!(ex, quote
                w = $(_shuf(b, perm))
                lo = min($a, w)
                hi = max($a, w)
                $a = $(_blend(:lo, :hi, lower, N))
                $b = $(_shuf(_blend(:hi, :lo, lower, N), perm))
            end)
        end
        d = s ÷ 2
        while d >= 1
            if d >= R
                m = d ÷ R
                perm = [c ⊻ m for c in 0:N-1]
                for ρ in 0:R-1
                    a = _v(ρ)
                    push!(ex, quote
                        w = $(_shuf(a, perm))
                        $a = $(_blend(:(min($a, w)), :(max($a, w)), c -> c & m == 0, N))
                    end)
                end
            else
                for ρ in 0:R-1
                    ρ & d == 0 || continue
                    a, b = _v(ρ), _v(ρ + d)
                    push!(ex, :(($a, $b) = (min($a, $b), max($a, $b))))
                end
            end
            d ÷= 2
        end
        s *= 2
    end

    # 3. Transpose each N × N block (rows bN .. bN+N-1) by swapping off-diagonal
    #    h × h sub-blocks, so that every register holds N consecutive outputs.
    for b in 0:(R ÷ N - 1)
        h = N ÷ 2
        while h >= 1
            for j in 0:N-1
                j & h == 0 || continue
                x, y = _v(b * N + j), _v(b * N + j + h)
                push!(ex, quote
                    t = $(_shuf(x, y, [c & h == 0 ? c : N + c - h for c in 0:N-1]))
                    $y = $(_shuf(x, y, [c & h == 0 ? c + h : N + c for c in 0:N-1]))
                    $x = t
                end)
            end
            h ÷= 2
        end
    end
    # Column j of block b holds outputs (j*R + b*N) .. + N-1.
    out = [_v(b * N + j) for j in 0:N-1 for b in 0:(R ÷ N - 1)]
    push!(ex, :(return ($(out...),)))
    return Expr(:block, ex...)
end

"""
    sortnet(vs::NTuple{R,Vec{N,T}}) -> NTuple{R,Vec{N,T}}

Sort the R*N keys in `vs`; the result holds them in order, lane by lane and
register by register. `R` must be a power-of-two multiple of `N`.
"""
@generated function sortnet(vs::NTuple{R,Vec{N,T}}) where {R,N,T}
    return network_body(R, N)
end

# Copy `n ≤ R*N` keys at `p` to `buf`, pad with typemax, sort, and copy back.
@inline function netsort!(p::Ptr{T}, n::Int, buf::Ptr{T}, ::Val{R}, ::Val{N}) where {T,R,N}
    W = R * N
    unsafe_copyto!(buf, p, n)
    for i in n+1:W
        unsafe_store!(buf, typemax(T), i)
    end
    vs = ntuple(r -> vload(Vec{N,T}, buf + (r - 1) * N * sizeof(T)), Val(R))
    vs = sortnet(vs)
    nfull = n ÷ N
    # Whole output vectors go straight back; the partial one through the buffer.
    for r in 1:R
        if r <= nfull
            vstore(vs[r], p + (r - 1) * N * sizeof(T))
        elseif r == nfull + 1
            vstore(vs[r], buf)
        end
    end
    unsafe_copyto!(p + nfull * N * sizeof(T), buf, n - nfull * N)
    return nothing
end

@inline function insertion_sort!(p::Ptr{T}, n::Int) where {T}
    for i in 2:n
        x = unsafe_load(p, i)
        j = i - 1
        while j >= 1
            y = unsafe_load(p, j)
            y <= x && break
            unsafe_store!(p, y, j + 1)
            j -= 1
        end
        unsafe_store!(p, x, j + 1)
    end
    return nothing
end

lanes(::Type{T}) where {T} = 32 ÷ sizeof(T)   # 256-bit vectors

# The base case uses up to MAXROWS registers: 128 × 32-bit or 64 × 64-bit keys.
const MAXROWS = 16
basecase_max(::Type{T}) where {T} = MAXROWS * lanes(T)

"""
    small_sort!(p, n, buf)

Sort `n ≤ basecase_max(T)` keys at `p`, using `buf` (at least that many keys) as
scratch. The smallest network with R ∈ {N, 2N, 4N} rows that fits is used.
"""
@inline function small_sort!(p::Ptr{T}, n::Int, buf::Ptr{T}) where {T}
    N = lanes(T)
    if n <= 8
        insertion_sort!(p, n)
    elseif n <= N * N
        netsort!(p, n, buf, Val(N), Val(N))
    elseif n <= 2N * N
        netsort!(p, n, buf, Val(min(2N, MAXROWS)), Val(N))
    else
        netsort!(p, n, buf, Val(MAXROWS), Val(N))
    end
    return nothing
end
