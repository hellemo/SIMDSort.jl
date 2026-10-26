using SIMDSort
using SIMD: Vec
using Test
using Random
using BitIntegers
using DoubleFloats
using MultiFloats
using Quadmath
using Dates

const TYPES = (Int32, UInt32, Int64, UInt64, Float32, Float64)

# Inputs that stress pivoting, duplicates and the special float values.
function gen(::Type{T}, dist::Symbol, n::Int, rng) where {T}
    if dist === :uniform
        T <: AbstractFloat ? T.(randn(rng, n)) : rand(rng, T, n)
    elseif dist === :sorted
        sort!(gen(T, :uniform, n, rng))
    elseif dist === :reversed
        sort!(gen(T, :uniform, n, rng); rev=true)
    elseif dist === :equal
        fill(rand(rng, T), n)
    elseif dist === :two
        rand(rng, T[typemin(T), typemax(T)], n)
    elseif dist === :few
        rand(rng, rand(rng, T, 5), n)
    elseif dist === :sawtooth
        T[T(i % 37) for i in 1:n]
    elseif dist === :organpipe
        T[T(min(i, n - i)) for i in 1:n]
    elseif dist === :extremes
        rand(rng, T[typemin(T), typemax(T), zero(T), one(T)], n)
    elseif dist === :special
        T <: AbstractFloat || return gen(T, :extremes, n, rng)
        rand(rng, T[NaN, -NaN, Inf, -Inf, 0.0, -0.0, 1.0, -1.0, floatmin(T),
                    -floatmin(T), nextfloat(T(0)), floatmax(T)], n)
    elseif dist === :zeros
        T <: AbstractFloat || return gen(T, :two, n, rng)
        rand(rng, T[0.0, -0.0, 0.0, -0.0, 1.0, -1.0, NaN], n)
    else
        error("unknown distribution $dist")
    end
end

const DISTS = (:uniform, :sorted, :reversed, :equal, :two, :few, :sawtooth,
               :organpipe, :extremes, :special, :zeros)

function check(v; rev=false)
    expected = sort(v; rev)
    got = simdsort(v; rev)
    ok = isequal(got, expected)
    ok || @info "mismatch" eltype(v) length(v) rev
    return ok
end

@testset "SIMDSort" begin
    rng = Xoshiro(42)
    @testset "keys round trip $T" for T in TYPES
        v = gen(T, :special, 1000, rng)
        for rev in (false, true)
            S = SIMDSort.signedkey(T)
            k = copy(reinterpret(S, filter(x -> !(x isa AbstractFloat && isnan(x)), v)))
            k0 = copy(k)
            SIMDSort.map_keys!(k, T, rev)
            SIMDSort.unmap_keys!(k, T, rev)
            @test k == k0
        end
    end

    @testset "network" begin
        # 0-1 principle: the column network sorts every 0/1 input of 16 keys.
        @test all(0:(2^16 - 1)) do m
            x = [(m >> i) & 1 for i in 0:15]
            for (a, b) in SIMDSort.batcher_pairs(16)
                x[a+1] > x[b+1] && ((x[a+1], x[b+1]) = (x[b+1], x[a+1]))
            end
            issorted(x)
        end
        @testset "small_sort! $T" for T in (Int32, Int64)
            buf = Vector{T}(undef, SIMDSort.basecase_max(T))
            @test all(Iterators.product(0:SIMDSort.basecase_max(T), 1:20)) do (n, t)
                a = isodd(t) ? rand(rng, T, n) : rand(rng, T(-2):T(2), n)
                t == 2 && (a = rand(rng, [typemin(T), typemax(T)], n))
                b = copy(a)
                GC.@preserve b buf SIMDSort.small_sort!(pointer(b), n, pointer(buf))
                b == sort(a)
            end
        end
    end

    @testset "partition! $T U=$U PF=$PF" for T in (Int32, Int64), U in (1, 2, 4, 8), PF in (false, true)
        N = SIMDSort.lanes(T)
        buf = Vector{T}(undef, (2U + 1) * N)
        @test all(Iterators.product(2U*N:2U*N+200, 1:6)) do (n, t)
            a = t <= 2 ? rand(rng, T(-3):T(3), n) : rand(rng, T, n)
            piv = t == 3 ? maximum(a) : t == 4 ? minimum(a) : a[rand(rng, 1:n)]
            b = copy(a)
            m = GC.@preserve b buf SIMDSort.partition!(pointer(b), n, piv, pointer(buf), Val(U), Val(PF))
            all(<=(piv), view(b, 1:m)) && all(>(piv), view(b, m+1:n)) && sort(b) == sort(a)
        end
    end

    @testset "$T" for T in TYPES
        for dist in DISTS, rev in (false, true)
            @test all(n -> check(gen(T, dist, n, rng); rev), 0:300)
            for n in (511, 512, 513, 1023, 1024, 4095, 10^4, 10^5)
                @test check(gen(T, dist, n, rng); rev)
            end
        end
        @test check(gen(T, :uniform, 10^6, rng))
    end

    @testset "ntasks $T" for T in TYPES
        for dist in (:uniform, :few, :equal, :sorted, :special, :zeros), rev in (false, true)
            v = gen(T, dist, 3 * 10^6 + 17, rng)
            @test isequal(simdsort(v; rev, ntasks=8), sort(v; rev))
        end
    end

    @testset "compress-store forms $T" for T in TYPES
        # both forms of vstc (memory compress-store; register compress and
        # masked store), whichever this CPU uses, at 256 and 512 bits
        for N in (32 ÷ sizeof(T), 64 ÷ sizeof(T)), _ in 1:50
            x = Vec{N,T}(ntuple(_ -> rand(rng, T), N))
            m = Vec{N,Bool}(ntuple(_ -> rand(rng, Bool), N))
            c = count(m)
            want = [x[k] for k in 1:N if m[k]]
            for f in (SIMDSort.vstc_mem, SIMDSort.vstc_reg)
                v = fill(T(7), N + 4)
                GC.@preserve v f(x, pointer(v), 2, m)
                @test v[3:2+c] == want && all(==(T(7)), v[[1:2; 3+c:N+4]])
                w = fill(T(7), N + 4)
                f(x, SIMDSort.ArrRange(w), 2, m)
                @test w == v
            end
        end
        @test SIMDSort.vstc_memory() == !startswith(Sys.CPU_NAME, "znver")
    end

    @testset "partition3! $T PF=$PF" for T in (Int32, Int64, UInt32, Float32, Float64), PF in (false, true)
        N = SIMDSort.lanes(T)
        U = SIMDSort.UNROLL
        buf = SIMDSort.scratch(T)
        vals = T <: AbstractFloat ? T[-1, -0.0, 0.0, 1, 2] : T[1, 2, 3, 5]
        @test all(Iterators.product(2U*N:2U*N+300, 1:3)) do (n, t)
            a = t == 1 ? rand(rng, vals, n) : t == 2 ? rand(rng, T, n) : fill(vals[2], n)
            piv = a[rand(rng, 1:n)]
            b = copy(a)
            x, y = GC.@preserve b buf SIMDSort.partition3!(pointer(b), n, piv, pointer(buf),
                                                                  Val(U), Val(PF))
            # Equal keys are refilled with the pivot value, so for floats the
            # signs of zeros may change; compare as numbers.
            all(<(piv), view(b, 1:x)) && all(==(piv), view(b, x+1:y)) &&
                all(>(piv), view(b, y+1:n)) && sort(b) == sort(a)
        end
    end

    @testset "ppartition! $T" for T in (Int32, Int64)
        for n in (2^20 + 1, 3 * 10^6), ntasks in (2, 3, 8), dist in (:uniform, :few, :sorted, :reversed)
            a = gen(T, dist, n, rng)
            piv = a[rand(rng, 1:n)]
            b = copy(a)
            m = GC.@preserve b SIMDSort.ppartition!(pointer(b), n, piv, ntasks)
            @test all(<=(piv), view(b, 1:m)) && all(>(piv), view(b, m+1:n))
            @test sort!(b) == sort!(a)
        end
    end

    @testset "counting sort and duplicates $T" for T in (Int32, UInt32, Int64, UInt64)
        n = 2^14
        for r in (0, 1, n ÷ 4 - 1, n ÷ 4, n ÷ 4 + 1, 2^16, 2^16 + 1)
            lo = rand(rng, (typemin(T), typemax(T) - T(min(r, 2^16 + 1)), zero(T)))
            hi = lo + T(r)
            v = rand(rng, lo:hi, n)
            v[1], v[2] = lo, hi             # the range is exactly r
            @test check(v) && check(v; rev=true)
            @test isequal(simdsort(v; ntasks=3), sort(v))
        end
        # Rare keys the 1024-key sample misses: inside the margin, outside it (but
        # in range), far outside (range too wide), and at the ends of T's range.
        for extra in (T[1005], T[1100, 3000], T[typemax(T)], T[typemin(T), 7])
            v = rand(rng, T(0):T(1000), n)
            v[rand(rng, 1:n, length(extra))] = extra
            for nt in (1, 3, 8)
                @test isequal(simdsort(v; ntasks=nt), sort(v))
                @test isequal(simdsort(v; rev=true, ntasks=nt), sort(v; rev=true))
            end
        end
        for v in (rand(rng, [typemax(T) - T(3), typemax(T)], n), rand(rng, typemin(T):typemin(T)+T(9), n))
            @test check(v) && check(v; rev=true)
        end
        # Few distinct values spread over the whole range, and one dominant value.
        for v in (rand(rng, rand(rng, T, 1000), 10^6), rand(rng, rand(rng, T, 3), 10^5),
                  [rand(rng) < 0.7 ? T(42) : rand(rng, T) for _ in 1:10^6])
            @test check(v) && check(v; rev=true)
            @test isequal(simdsort(v; ntasks=8), sort(v))
        end
    end

    @testset "hash counting of few distinct values $T" for T in TYPES
        n = 2^17
        sample(k) = T <: AbstractFloat ? T.(randn(rng, k)) : rand(rng, T, k)
        cases = [rand(rng, sample(1000), n), rand(rng, sample(3), n), fill(sample(1)[1], n),
                 # The sample sees few values, but 1% are unique: the table overflows
                 # and the quicksort takes over.
                 [rand(rng) < 0.99 ? x : y for (x, y) in zip(rand(rng, sample(500), n), sample(n))]]
        T <: AbstractFloat && push!(cases, rand(rng, T[0.0, -0.0, 1.0, -1.0, NaN, -NaN, Inf, -Inf], n))
        T <: AbstractFloat || push!(cases, rand(rng, [typemin(T), typemax(T), zero(T), one(T)], n))
        for v in cases, rev in (false, true), nt in (1, 8)
            @test isequal(simdsort(v; rev, ntasks=nt), sort(v; rev))
        end
    end

    @testset "Int64 leaves as doubles, also with subnormals flushed" begin
        # Small key ranges give offsets that are subnormal as doubles.
        for v in (rand(rng, Int64(-50):Int64(50), 10^5), rand(rng, Int64, 10^5),
                  rand(rng, UInt64(0):UInt64(2)^61, 10^5), rand(rng, [typemin(Int64), typemax(Int64), 0], 10^4))
            @test check(v) && check(v; rev=true)
            old = get_zero_subnormals()
            try
                set_zero_subnormals(true)
                @test simdsort(v) == sort(v)
            finally
                set_zero_subnormals(old)
            end
        end
    end

    @testset "depth limit" begin
        # A zero depth budget sends every range to the fallback.
        v = rand(rng, Int32, 10^4)
        w = copy(v)
        buf = SIMDSort.scratch(Int32)
        GC.@preserve w buf SIMDSort.qsort!(pointer(w), length(w), pointer(buf), 0, UInt64(1))
        @test w == sort(v)
        GC.@preserve w SIMDSort.pqsort!(pointer(v), length(v), 1, UInt64(1), 4, length(v))
        @test v == sort(w)
    end

    @testset "simdsortperm $T" for T in TYPES
        for dist in DISTS, rev in (false, true)
            @test all(n -> (v = gen(T, dist, n, rng); simdsortperm(v; rev) == sortperm(v; rev)), 0:300)
            for n in (1000, 10^5), nt in (1, 8)
                v = gen(T, dist, n, rng)
                @test simdsortperm(v; rev, ntasks=nt) == sortperm(v; rev)
            end
        end
    end

    @testset "strings and symbols" begin
        randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))
        tricky = ["a", "a\0", "a\0\0", "", "\0", "ab", "a\xff", "\xff", "é", "e", "a"^30, "a"^30 * "b"]
        inputs = [[randstr(rand(rng, 0:20)) for _ in 1:10^5],
                  rand(rng, ["id" * lpad(string(i), 10, "0") for i in 1:1000], 10^5),
                  rand(rng, tricky, 10^4),
                  [randstr(40) * "x" for _ in 1:5000],
                  [randstr(rand(rng, 0:3)) for _ in 1:2000]]
        for v in inputs, rev in (false, true), nt in (1, 8)
            @test simdsortperm(v; rev, ntasks=nt) == sortperm(v; rev)
            @test isequal(simdsort(v; rev, ntasks=nt), sort(v; rev))
            sy = Symbol.(filter(s -> !occursin('\0', s) && isvalid(s), v))
            @test simdsortperm(sy; rev, ntasks=nt) == sortperm(sy; rev)
            @test isequal(simdsort(sy; rev, ntasks=nt), sort(sy; rev))
        end
        # Symbols by hash counting: few and many distinct, more distinct than
        # n/4 (byte path), a table that overflows its estimate (it grows),
        # and several tasks merging the per-task tables.
        names = [Symbol(randstr(rand(rng, 1:12))) for _ in 1:30_000]
        symcases = [rand(rng, names[1:3], 50_000), rand(rng, names[1:2000], 200_000),
                    rand(rng, names, 60_000),                      # > n/4 distinct
                    [names[rand(rng) < 0.5 ? 1 : rand(rng, 1:20_000)] for _ in 1:100_000],
                    fill(:only, 5000), Symbol[]]
        for sy in symcases, rev in (false, true), nt in (1, 3, 8, 56)
            @test simdsortperm(sy; rev, ntasks=nt) == sortperm(sy; rev)
            @test simdsort(sy; rev, ntasks=nt) == sort(sy; rev)
        end
        # The merge of nearly half-full tables in short slot ranges: keys
        # pushed past the end of a range by linear probing, and wrapping around.
        for np in (2, 7, 64)
            cap = 1024
            tables = [SIMDSort.CountTable{Symbol}(cap) for _ in 1:np]
            want = Dict{Symbol,Int}()
            for t in tables, x in rand(rng, names[1:2000], 450)
                SIMDSort.add_count!(t, x, 1)
                want[x] = get(want, x, 0) + 1
            end
            got = Dict(zip(SIMDSort.merge_tables(tables, cap)...))
            @test got == want && length(SIMDSort.merge_tables(tables, cap)[1]) == length(want)
        end
        r = SIMDSort.instrumented(() -> simdsort!(rand(rng, names[1:50], 100_000); ntasks = 4))
        @test get(r.counters, :symbol_sorts, 0) == 1
        # Tables that outgrow the estimate in some chunks only: they are
        # doubled, and brought to one capacity for the merge. The sampled
        # positions (a stride of n ÷ HASH_SAMPLE) hold one symbol, so the
        # estimate is far too low; the many distinct ones are in the first 30%.
        sy = fill(names[1], 100_000)
        stride = 100_000 ÷ SIMDSort.HASH_SAMPLE
        for i in 1:30_000
            (i - 1) % stride == 0 || (sy[i] = rand(rng, names[1:20_000]))
        end
        for nt in (1, 4, 8)
            r = SIMDSort.instrumented(() -> simdsort!(copy(sy); ntasks = nt))
            @test get(r.counters, :symbol_table_grows, 0) >= 1 && get(r.counters, :symbol_sorts, 0) == 1
            @test simdsort(sy; ntasks = nt) == sort(sy) && simdsortperm(sy; ntasks = nt) == sortperm(sy)
        end
        # Views: SubString{String}, and StringView (Julia 1.14+).
        long = [randstr(rand(rng, 0:30)) * "\xff" for _ in 1:20_000]
        subs = [SubString(s, 1, max(0, ncodeunits(s) - rand(rng, 1:3))) for s in long]
        for rev in (false, true), nt in (1, 8)
            @test simdsortperm(subs; rev, ntasks=nt) == sortperm(subs; rev)
            @test isequal(simdsort(subs; rev, ntasks=nt), sort(subs; rev))
        end
        @static if isdefined(Base, :StringView)
            views = [Base.StringView(Vector{UInt8}(s)) for s in long]
            for rev in (false, true), nt in (1, 8)
                @test simdsortperm(views; rev, ntasks=nt) == sortperm(views; rev)
                @test isequal(simdsort(views; rev, ntasks=nt), sort(views; rev))
            end
        end
        for n in 0:100
            v = rand(rng, tricky, n)
            @test simdsortperm(v) == sortperm(v) && isequal(simdsort(v; rev=true), sort(v; rev=true))
        end
    end

    @testset "wide and arbitrary-precision numbers" begin
        same(v; rev=false, nt=1) = simdsortperm(v; rev, ntasks=nt) == sortperm(v; rev) &&
                                   isequal(simdsort(v; rev, ntasks=nt), sort(v; rev))
        for T in (Int128, UInt128, Int256, UInt256, Int512, UInt1024)
            for v in (rand(rng, T, 2 * 10^4), rand(rng, T(0):T(5), 5000),
                      rand(rng, [typemin(T), typemax(T), zero(T), one(T)], 1000), rand(rng, T, 7))
                @test all(same(v; rev, nt) for rev in (false, true), nt in (1, 8))
            end
        end
        bi = [big(rand(rng, Int128)) * big(rand(rng, Int128))^rand(rng, 0:3) * rand(rng, (-1, 1)) for _ in 1:5000]
        append!(bi, big.(rand(rng, -3:3, 1000)))
        bf = [BigFloat(randn(rng)) * big(10.0)^rand(rng, -40:40) for _ in 1:5000]
        append!(bf, BigFloat[NaN, Inf, -Inf, 0.0, -0.0, 1.5, -1.5, 1.5])
        append!(bf, [BigFloat(1.5, precision=64), BigFloat(-1.5, precision=512),
                     BigFloat(-1.5000001, precision=512), BigFloat(2.0, precision=1000)])
        shuffle!(rng, bf)
        for v in (bi, bf), rev in (false, true), nt in (1, 8)
            @test same(v; rev, nt)
        end
        # Quadmath's Float128 and MultiFloats (2-4 limbs, Float64 and Float32).
        q = Float128.(randn(rng, 20_000)) .* Float128(10) .^ rand(rng, -30:30, 20_000)
        append!(q, Float128[0.0, -0.0, Inf, -Inf, NaN, -NaN, 1, 1, -1])
        shuffle!(rng, q)
        @test all(same(q; rev, nt) for rev in (false, true), nt in (1, 8))
        for T in (Float64x2, Float64x3, Float64x4, Float32x2, Float32x3)
            v = T.(randn(rng, 10_000)) .+ T.(randn(rng, 10_000)) .* T(1e-12)
            append!(v, T[0.0, -0.0, Inf, -Inf, NaN, 1, 1, -1])
            shuffle!(rng, v)
            @test all(same(v; rev, nt) for rev in (false, true), nt in (1, 8))
        end
        for T in (Double64, Double32)
            v = T.(randn(rng, 2 * 10^4)) .+ T.(randn(rng, 2 * 10^4)) .* T(1e-10)
            append!(v, T[0.0, -0.0, Inf, -Inf, 1, 1, -1])
            shuffle!(rng, v)
            @test all(same(v; rev, nt) for rev in (false, true), nt in (1, 8))
            # DoubleFloats' isless is no total order with NaNs; NaNs go last.
            w = [v; T(NaN)]
            shuffle!(rng, w)
            r = simdsort(w)
            @test isnan(r[end]) && issorted(r[1:end-1]) && sort(filter(!isnan, w)) == r[1:end-1]
        end
    end

    @testset "MSD distribution for wide keys (sort!)" begin
        n = 3 * 10^5 + 11       # above MSD_MINN
        sameS(v; rev=false, nt=1) = isequal(simdsort(v; rev, ntasks=nt), sort(v; rev))
        cases = Any[rand(rng, Int128, n), rand(rng, Int128(-1000):Int128(1000), n),
                    rand(rng, UInt256, n), rand(rng, rand(rng, Int128, 7), n),
                    Int128(2)^100 .+ rand(rng, Int128(0):Int128(10)^6, n),     # long common prefix
                    rand(rng, UInt128(0):UInt128(3), n) .<< 64 .+ rand(rng, UInt128, n) .>> 64,
                    Float128.(randn(rng, n)), [Float128(rand(rng, [0.0, -0.0, 1.0, NaN, -Inf])) for _ in 1:n],
                    Double64.(randn(rng, n)) .+ Double64.(randn(rng, n)) .* 1e-20,
                    Float64x3.(randn(rng, n)),
                    collect(zip(rand(rng, 1:50, n), randn(rng, n))),
                    [rand(rng, Int16) => rand(rng, UInt64) for _ in 1:n],
                    collect(zip(rand(rng, Bool, n), rand(rng, 'a':'z', n), rand(rng, Int32, n))),
                    # Equal windows within a bucket (tie runs), resolved by the low word.
                    rand(rng, UInt128(1):UInt128(1000), n) .<< 64 .| rand(rng, UInt128(0):UInt128(3), n),
                    rand(rng, UInt128(1):UInt128(1000), n) .<< 64 .| rand(rng, UInt64, n),
                    sort(rand(rng, Int128, n)), sort(rand(rng, Int128, n); rev=true), fill(Int128(5), n),
                    [rand(rng, Int32(1):Int32(500)) => rand(rng, Int64(0):Int64(2)) for _ in 1:n]]
        for v in cases, rev in (false, true), nt in (1, 3, 8)
            @test sameS(v; rev, nt)
        end
    end

    @testset "distribution top level (64-bit keys)" begin
        # The rule (src/msd.jl): from 16 tasks (64-bit keys) or 20 (32-bit),
        # and keys × bytes × tasks >= 2^33.
        ud = SIMDSort.use_distribution
        @test ud(2^26, 8, 16) && !ud(2^25, 8, 16) && !ud(2^28, 8, 8)
        @test ud(2^25, 8, 32) && !ud(2^24, 8, 32) && ud(2^25, 8, 56)
        @test !ud(2^28, 4, 16) && ud(2^27, 4, 24) && !ud(2^25, 4, 32)
        old = SIMDSort.DIST_MODE[]
        SIMDSort.DIST_MODE[] = :on
        try
            n = 2 * 10^5 + 3
            fl = randn(rng, n); fl[1:7:end] .= 0.0; fl[2:7:end] .= -0.0; fl[3:101:end] .= NaN; fl[4:997:end] .= Inf
            for v in Any[rand(rng, Int64, n), rand(rng, UInt64, n), rand(rng, Int64(-50):Int64(50), n) .* 10^15,
                         round.(Int64, randn(rng, n) .* 1e9), [rand(rng, Bool) ? typemin(Int64) : rand(rng, Int64) for _ in 1:n],
                         rand(rng, Int32, n), rand(rng, UInt32, n), fl, Float32.(fl), randn(rng, Float32, n)],
                rev in (false, true), nt in (8, 13)
                @test isequal(simdsort(v; rev, ntasks = nt), sort(v; rev))
            end
            r = SIMDSort.instrumented(() -> simdsort!(rand(rng, UInt64, n); ntasks = 8))
            @test get(r.counters, :dist_sorts, 0) == 1
        finally
            SIMDSort.DIST_MODE[] = old
        end
    end

    @testset "parallel presorted check" begin
        n = 2^20 + 7
        for T in (Int64, UInt32, Float64), nt in (1, 3, 8), rev in (false, true)
            base = sort(T <: AbstractFloat ? randn(rng, n) : rand(rng, T, n))
            cases = Any[copy(base), reverse(base), fill(base[1], n)]
            # One key out of order at a chunk border, and in the middle of a chunk.
            for pos in unique(min.((cld(n, nt) + 1, n ÷ 2 + 3), n))
                v = copy(base); v[pos], v[pos-1] = v[pos-1], v[pos]
                v[pos] != v[pos-1] && push!(cases, v)
                w = reverse(base); w[pos], w[pos-1] = w[pos-1], w[pos]
                w[pos] != w[pos-1] && push!(cases, w)
            end
            push!(cases, sort(rand(rng, T(0):T(3), n)))
            for v in cases
                @test isequal(simdsort(v; rev, ntasks = nt), sort(v; rev))
            end
        end
    end

    @testset "ntasks bounds the concurrent tasks" begin
        # Peak number of tasks inside the leaf sorts at once, from a probe
        # wrapped around the sort of a large array.
        n = 2^22
        for nt in (1, 2, 3)
            v = rand(rng, Int64, n)
            r = SIMDSort.instrumented(() -> simdsort!(v; ntasks = nt))
            @test issorted(v)
            @test get(r.counters, :concurrent_tasks_max, 1) <= nt
        end
    end

    @testset "instrumentation" begin
        r = SIMDSort.instrumented(() -> simdsort!(rand(rng, Int128, 10^5); ntasks = 4))
        @test issorted(r.result)
        @test r.counters[:msd_sorts] == 1 && haskey(r.phases, :msd_finish)
        io = IOBuffer(); SIMDSort.print_report(io, r)
        @test occursin("msd_scatter", String(take!(io)))
        @test !SIMDSort.INSTR[]
        # The tuning report runs and writes its sections (tiny inputs).
        f = tempname()
        @test SIMDSort.tuning_report(f; ntasks = 2, rounds = 1, sizelog = 12) == f
        t = read(f, String)
        @test occursin("features:", t) && occursin("time, Int64, 2, sample-sort", t) &&
              occursin("String 8-20 sortperm", t) && SIMDSort.DIST_MODE[] === :auto
        rm(f)
    end

    @testset "Dates, Tuples, Pairs, VersionNumber" begin
        same(v; rev=false, nt=1) = simdsortperm(v; rev, ntasks=nt) == sortperm(v; rev) &&
                                   isequal(simdsort(v; rev, ntasks=nt), sort(v; rev))
        ds = Date(1900, 1, 1) .+ Day.(rand(rng, 0:80_000, 30_000))
        dts = DateTime(1950) .+ Millisecond.(rand(rng, Int64(0):Int64(10)^12, 30_000))
        ts = Time.(Nanosecond.(rand(rng, 0:86_399_999_999_999, 30_000)))
        periods = (Day.(rand(rng, -10^6:10^6, 30_000)), Millisecond.(rand(rng, Int64, 30_000)),
                   Month.(rand(rng, -50:50, 30_000)), Nanosecond.(rand(rng, Int64, 3000)))
        for v in (ds, dts, ts, periods...)
            @test all(same(v; rev, nt) for rev in (false, true), nt in (1, 8))
        end
        randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))
        tuples = (collect(zip(rand(rng, 1:20, 30_000), rand(rng, Int32, 30_000))),
                  collect(zip(rand(rng, ["a", "b", "ab", ""], 30_000), rand(rng, [0.0, -0.0, 1.0, NaN], 30_000))),
                  collect(zip(rand(rng, 1:3, 5000), [randstr(rand(rng, 0:4)) for _ in 1:5000], rand(rng, Bool, 5000))),
                  [((rand(rng, 1:3), rand(rng, 'a':'c')), rand(rng, UInt8)) for _ in 1:5000],
                  [rand(rng, 1:5) => randstr(2) for _ in 1:10_000],
                  [rand(rng, ds) => rand(rng, -2.0:1.0:2.0) for _ in 1:10_000])
        for v in tuples
            @test all(same(v; rev, nt) for rev in (false, true), nt in (1, 8))
        end
        pre = ["", "-alpha", "-alpha.1", "-beta", "-rc.2", "-0"]
        bld = ["", "+b1", "+b2.3"]
        vs = [VersionNumber("$(rand(rng, 0:3)).$(rand(rng, 0:3)).$(rand(rng, 0:3))$(rand(rng, pre))$(rand(rng, bld))")
              for _ in 1:20_000]
        @test all(same(vs; rev, nt) for rev in (false, true), nt in (1, 8))
    end

    @testset "fallback" begin
        v = rand(rng, Int16, 1000)
        @test simdsort(v) == sort(v)
        s = ["b", "a", "c"]
        @test simdsort(s; rev=true) == ["c", "b", "a"]
    end
end
