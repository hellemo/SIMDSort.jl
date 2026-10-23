# Base case: a vectorized sorting network over an R × N matrix held in R registers
# of N lanes (docs/DESIGN.md §5). The network code is generated at compile time
# from the comparator lists below.
#
# [Bat68]: odd-even merge sort and merge networks (batcher_pairs,
# batcher_merge_pairs); [WBGS22 §2.3, §3]: base case in a register matrix,
# column sort by one vector min/max per comparator, merging columns without
# transposing first (network_body); [F5 §III, Algorithm 3]: zigzag merges
# (zigzag_body). Our own: the two-halves split (halves_body) and the generator
# that derives the zigzag column permutations by simulation.

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

# Vector min/max for the networks. Inputs never contain NaNs (they are moved out
# first), so for 256-bit float vectors the plain x86 vminps/vminpd (vmaxps/vmaxpd)
# suffice. Base `min` adds NaN handling, and LLVM does the same for a
# compare-and-select. Other vectors and targets use `min`/`max`. The x86 forms may
# change the sign of equal zeros; `order_zeros!` repairs that.
@inline vmin(a::Vec, b::Vec) = min(a, b)
@inline vmax(a::Vec, b::Vec) = max(a, b)
for (T, N, sfx, llt) in ((Float32, 8, "ps", "float"), (Float64, 4, "pd", "double"))
    for (f, op) in ((:vmin, "min"), (:vmax, "max"))
        intr = "llvm.x86.avx.$op.$sfx.256"
        ir = ("""
            declare <$N x $llt> @$intr(<$N x $llt>, <$N x $llt>)
            define <$N x $llt> @entry(<$N x $llt> %a, <$N x $llt> %b) #0 {
              %r = call <$N x $llt> @$intr(<$N x $llt> %a, <$N x $llt> %b)
              ret <$N x $llt> %r
            }
            attributes #0 = { alwaysinline }
            """, "entry")
        @eval @generated function $f(a::Vec{$N,$T}, b::Vec{$N,$T})
            if SIMD.Targets.has_feature(:avx)
                return :(Vec(Base.llvmcall($$ir, SIMD.LVec{$$N,$$T},
                                           Tuple{SIMD.LVec{$$N,$$T}, SIMD.LVec{$$N,$$T}},
                                           a.data, b.data)))
            else
                return :($$(op == "min" ? :min : :max)(a, b))
            end
        end
    end
end

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
        push!(ex, :(($(_v(a)), $(_v(b))) = (vmin($(_v(a)), $(_v(b))), vmax($(_v(a)), $(_v(b))))))
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
                lo = vmin($a, w)
                hi = vmax($a, w)
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
                        $a = $(_blend(:(vmin($a, w)), :(vmax($a, w)), c -> c & m == 0, N))
                    end)
                end
            else
                for ρ in 0:R-1
                    ρ & d == 0 || continue
                    a, b = _v(ρ), _v(ρ + d)
                    push!(ex, :(($a, $b) = (vmin($a, $b), vmax($a, $b))))
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

# --- Zigzag-merge networks (after F5, Arman & Loguinov, ICDE 2026, §III) -------
#
# Columns are sorted first (Batcher). Runs are then merged in "zigzag" order: a
# sorted run of length r·k in r rows and k lanes keeps elements 2j and 2j+1 of
# each block of 2k in rows 2t and 2t+1. Merging two such runs A and B (k lanes
# each, side by side) is (F5, Algorithm 3):
#   1. reverse every odd row within the 2k lanes;
#   2. merge every column: its even rows (from one run) and odd rows (from the
#      other) are two sorted sequences, so Batcher's odd-even merge across the
#      registers suffices -- vertical min/max only, no shuffles;
#   3. for exchange distances d = 1, 2, ..., k: in every pair of rows swap
#      lane block t of the upper row with block t-1 of the lower row (odd t,
#      blocks of d lanes), then compare-exchange the two rows.
# The result is a 2k-zigzag run, with its columns in a fixed permutation; the
# next level's reversal masks and the final transpose take it into account.
# Compared with the bitonic merges of network_body this needs about 14% fewer
# min/max and no blends.

# Switch for experiments (read when a network is generated).
const ZIGZAG = Ref(true)

function batcher_merge_pairs(n::Int)          # the last level (p = n/2) of Batcher
    pairs = Tuple{Int,Int}[]
    p = n ÷ 2
    k = p
    while k >= 1
        for j in (k % p):(2k):(n - 1 - k), i in 0:min(k - 1, n - j - k - 1)
            (i + j) ÷ 2p == (i + j + k) ÷ 2p && push!(pairs, (i + j, i + j + k))
        end
        k ÷= 2
    end
    return pairs
end

# The data-independent schedule for an R×N zigzag network: for each merge level
# (G = 2k lanes per merged run) the reversal mask (new lane <- old lane) and the
# column permutation of the result, found by simulating the network on ranks.
function zigzag_plan(R::Int, N::Int)
    M = [collect(Float64, (r - 1) * N + 1:r * N) for r in 1:R]
    for c in 1:N                                  # distinct, unsorted columns
        col = shuffle_det([M[r][c] for r in 1:R], c)
        for r in 1:R; M[r][c] = col[r]; end
    end
    rows = collect(0:R-1)
    cex!(i, j) = (a = M[rows[i+1]+1]; b = M[rows[j+1]+1];
                  M[rows[i+1]+1] = min.(a, b); M[rows[j+1]+1] = max.(a, b))
    for (a, b) in batcher_pairs(R); cex!(a, b); end
    levels = []
    pos = [0]
    G = 2
    while G <= N
        jpos = [pos; pos .+ G ÷ 2]
        rev = zeros(Int, N)
        for g in 0:(N ÷ G - 1), j in 0:G-1
            rev[g*G + jpos[j+1] + 1] = g*G + jpos[G-j]
        end
        for ρ in 1:2:R-1; M[rows[ρ+1]+1] = M[rows[ρ+1]+1][rev .+ 1]; end
        h = R ÷ 2
        wire = [[2t for t in 0:h-1]; [2t + 1 for t in 0:h-1]]
        for (x, y) in batcher_merge_pairs(R); cex!(wire[x+1], wire[y+1]); end
        rows = [rows[wire[q+1]+1] for q in 0:R-1]
        dists = [1 << i for i in 0:Int(log2(G))-1]
        for d in dists, i in 0:2:R-1
            x, y = M[rows[i+1]+1], M[rows[i+2]+1]
            nx, ny = copy(x), copy(y)
            for g in 0:(N ÷ G - 1), t in 1:2:(G ÷ d - 1), e in 0:d-1
                cx, cy = g*G + t*d + e, g*G + (t-1)*d + e
                nx[cx+1], ny[cy+1] = y[cy+1], x[cx+1]
            end
            M[rows[i+1]+1], M[rows[i+2]+1] = nx, ny
            cex!(i, i + 1)
        end
        # Conceptual column of each lane: row 0 holds elements 0, 2, ..., 2G-2.
        S = sort(vcat([M[rows[r+1]+1][1:G] for r in 0:R-1]...))
        col = [(searchsortedfirst(S, M[rows[1]+1][c]) - 1) ÷ 2 for c in 1:G]
        pos = invperm(col .+ 1) .- 1
        push!(levels, (G = G, rev = rev, dists = dists, pos = copy(pos)))
        G *= 2
    end
    return levels
end

# A fixed permutation (deterministic, no RNG state needed in generators).
shuffle_det(v, seed) = v[sortperm([hash((seed, i)) for i in eachindex(v)])]

function zigzag_body(R::Int, N::Int)
    ex = Expr[]
    names = [_v(ρ) for ρ in 0:R-1]                # logical row -> variable
    push!(ex, :(($(names...),) = vs))
    cnt = Ref(0)
    fresh() = Symbol(:z, cnt[] += 1)
    function cex(i, j)
        a, b = names[i+1], names[j+1]
        lo, hi = fresh(), fresh()
        push!(ex, :(($lo, $hi) = (vmin($a, $b), vmax($a, $b))))
        names[i+1], names[j+1] = lo, hi
    end
    for (a, b) in batcher_pairs(R); cex(a, b); end
    plan = zigzag_plan(R, N)
    for lv in plan
        G = lv.G
        for ρ in 1:2:R-1
            z = fresh()
            push!(ex, :($z = $(_shuf(names[ρ+1], lv.rev))))
            names[ρ+1] = z
        end
        h = R ÷ 2
        wire = [[2t for t in 0:h-1]; [2t + 1 for t in 0:h-1]]
        for (x, y) in batcher_merge_pairs(R); cex(wire[x+1], wire[y+1]); end
        names = [names[wire[q+1]+1] for q in 0:R-1]
        for d in lv.dists, i in 0:2:R-1
            mx = collect(0:N-1)                   # lanes of x: own or y's (N + lane)
            my = collect(N:2N-1)
            for g in 0:(N ÷ G - 1), t in 1:2:(G ÷ d - 1), e in 0:d-1
                cx, cy = g*G + t*d + e, g*G + (t-1)*d + e
                mx[cx+1] = N + cy
                my[cy+1] = cx
            end
            x, y = names[i+1], names[i+2]
            nx, ny = fresh(), fresh()
            push!(ex, :($nx = $(_shuf(x, y, mx))))
            push!(ex, :($ny = $(_shuf(x, y, my))))
            names[i+1], names[i+2] = nx, ny
            cex(i, i + 1)
        end
    end
    # N-zigzag (columns permuted by pos) to row-major: rows 2t, 2t+1 hold keys
    # 2tN + 2j (row 2t) and 2tN + 2j + 1 (row 2t+1) in lane pos[j].
    pos = plan[end].pos
    out = Symbol[]
    for t in 0:R÷2-1
        x, y = names[2t+1], names[2t+2]
        for half in 0:1
            mask = [begin m = half * N + q; j = m ÷ 2; iseven(m) ? pos[j+1] : N + pos[j+1] end for q in 0:N-1]
            o = fresh()
            push!(ex, :($o = $(_shuf(x, y, mask))))
            push!(out, o)
        end
    end
    push!(ex, :(return ($(out...),)))
    return Expr(:block, ex...)
end

"""
    halves_body(R, N)

Code for an R×N network (R = 32, more registers than AVX2 has) built from two
R/2-row networks, which fit in the 16 registers, and a bitonic merge done half
by half (docs/DESIGN.md §5):

1. sort rows 0..R/2-1 (A) and rows R/2..R-1 (B) as R/2×N matrices; each
   comes back as N·R/2 sorted keys, register by register;
2. compare A[i] with B[last - i] (register R/2-1-r of B, lanes reversed): the
   minima stay in A, the maxima go to B, register by register. Afterwards
   every key in A is ≤ every key in B, and both halves are bitonic;
3. sort each half with the half-cleaners of a bitonic merge: register pairs
   at distances R/4, ..., 1, then lane pairs at distances N/2, ..., 1.

Steps 2 and 3 only ever need two (step 2) or R/2 (step 3) registers live,
instead of all R, which avoids most register spills.
"""
function halves_body(R::Int, N::Int)
    H = R ÷ 2
    ex = Expr[]
    push!(ex, :(a = sortnet($(Expr(:tuple, [:(vs[$r]) for r in 1:H]...)))))
    push!(ex, :(b = sortnet($(Expr(:tuple, [:(vs[$r]) for r in H+1:R]...)))))
    push!(ex, :(($(_v.(0:H-1)...),) = a))
    push!(ex, :(($(_v.(H:R-1)...),) = b))
    rev = [N - 1 - c for c in 0:N-1]
    # 2. A[i] against B[N*H - 1 - i].
    for r in 0:H-1
        a, b = _v(r), _v(H + H - 1 - r)
        push!(ex, quote
            w = $(_shuf(b, rev))
            lo = vmin($a, w)
            hi = vmax($a, w)
            $a = lo
            $b = $(_shuf(:hi, rev))
        end)
    end
    # 3. Half-cleaners within each half (sequence index i = register*N + lane).
    for base in (0, H)
        d = H ÷ 2
        while d >= 1                       # register distance d (key distance d*N)
            for r in 0:H-1
                r & d == 0 || continue
                a, b = _v(base + r), _v(base + r + d)
                push!(ex, :(($a, $b) = (vmin($a, $b), vmax($a, $b))))
            end
            d ÷= 2
        end
        m = N ÷ 2
        while m >= 1                       # lane distance m
            perm = [c ⊻ m for c in 0:N-1]
            for r in 0:H-1
                a = _v(base + r)
                push!(ex, quote
                    w = $(_shuf(a, perm))
                    $a = $(_blend(:(vmin($a, w)), :(vmax($a, w)), c -> c & m == 0, N))
                end)
            end
            m ÷= 2
        end
    end
    push!(ex, :(return ($(_v.(0:R-1)...),)))
    return Expr(:block, ex...)
end

"""
    sortnet(vs::NTuple{R,Vec{N,T}}) -> NTuple{R,Vec{N,T}}

Sort the R*N keys in `vs`; the result holds them in order, lane by lane and
register by register. `R` must be a power-of-two multiple of `N`.
"""
@generated function sortnet(vs::NTuple{R,Vec{N,T}}) where {R,N,T}
    # 64-bit integers keep the single network: their min/max (compare and two
    # blends) make the split version slower (11-16% in the base case).
    # Split only where the target has 16 vector registers (AVX2); AVX-512 has 32.
    # 64-bit integers keep the single network on AVX2 (their min/max is a
    # compare and two blends there).
    split = R > 16 && R ÷ 2 >= N && !(T <: Integer && sizeof(T) == 8) &&
            !SIMD.Targets.has_feature(:avx512f)
    # The 16-row (and smaller) networks use zigzag merges; the 32-row ones are
    # two zigzag halves joined by halves_body.
    body = split ? halves_body(R, N) :
           (R >= 2 && iseven(R) && R >= N ÷ 2 && ZIGZAG[]) ? zigzag_body(R, N) : network_body(R, N)
    # Inline: a call would pass all R vectors through memory.
    return Expr(:block, Expr(:meta, :inline), body)
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

# The base case uses up to MAXROWS registers: 256 32-bit or 128 64-bit keys.
const MAXROWS = 32
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
    elseif n <= 4N * N && 4N <= MAXROWS
        netsort!(p, n, buf, Val(min(4N, MAXROWS)), Val(N))
    else
        netsort!(p, n, buf, Val(MAXROWS), Val(N))
    end
    return nothing
end
