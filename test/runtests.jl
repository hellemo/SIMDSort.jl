using SIMDQuickSort
using Test
using Random

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

@testset "SIMDQuickSort" begin
    rng = Xoshiro(42)
    @testset "keys round trip $T" for T in TYPES
        v = gen(T, :special, 1000, rng)
        for rev in (false, true)
            S = SIMDQuickSort.signedkey(T)
            k = copy(reinterpret(S, filter(x -> !(x isa AbstractFloat && isnan(x)), v)))
            k0 = copy(k)
            SIMDQuickSort.map_keys!(k, T, rev)
            SIMDQuickSort.unmap_keys!(k, T, rev)
            @test k == k0
        end
    end

    @testset "network" begin
        # 0-1 principle: the column network sorts every 0/1 input of 16 keys.
        @test all(0:(2^16 - 1)) do m
            x = [(m >> i) & 1 for i in 0:15]
            for (a, b) in SIMDQuickSort.batcher_pairs(16)
                x[a+1] > x[b+1] && ((x[a+1], x[b+1]) = (x[b+1], x[a+1]))
            end
            issorted(x)
        end
        @testset "small_sort! $T" for T in (Int32, Int64)
            buf = Vector{T}(undef, SIMDQuickSort.basecase_max(T))
            @test all(Iterators.product(0:SIMDQuickSort.basecase_max(T), 1:20)) do (n, t)
                a = isodd(t) ? rand(rng, T, n) : rand(rng, T(-2):T(2), n)
                t == 2 && (a = rand(rng, [typemin(T), typemax(T)], n))
                b = copy(a)
                GC.@preserve b buf SIMDQuickSort.small_sort!(pointer(b), n, pointer(buf))
                b == sort(a)
            end
        end
    end

    @testset "partition! $T U=$U PF=$PF" for T in (Int32, Int64), U in (1, 2, 4, 8), PF in (false, true)
        N = SIMDQuickSort.lanes(T)
        buf = Vector{T}(undef, (2U + 1) * N)
        @test all(Iterators.product(2U*N:2U*N+200, 1:6)) do (n, t)
            a = t <= 2 ? rand(rng, T(-3):T(3), n) : rand(rng, T, n)
            piv = t == 3 ? maximum(a) : t == 4 ? minimum(a) : a[rand(rng, 1:n)]
            b = copy(a)
            m = GC.@preserve b buf SIMDQuickSort.partition!(pointer(b), n, piv, pointer(buf), Val(U), Val(PF))
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

    @testset "partition3! $T PF=$PF" for T in (Int32, Int64, UInt32, Float32, Float64), PF in (false, true)
        N = SIMDQuickSort.lanes(T)
        U = SIMDQuickSort.UNROLL
        buf = SIMDQuickSort.scratch(T)
        vals = T <: AbstractFloat ? T[-1, -0.0, 0.0, 1, 2] : T[1, 2, 3, 5]
        @test all(Iterators.product(2U*N:2U*N+300, 1:3)) do (n, t)
            a = t == 1 ? rand(rng, vals, n) : t == 2 ? rand(rng, T, n) : fill(vals[2], n)
            piv = a[rand(rng, 1:n)]
            b = copy(a)
            x, y = GC.@preserve b buf SIMDQuickSort.partition3!(pointer(b), n, piv, pointer(buf),
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
            m = GC.@preserve b SIMDQuickSort.ppartition!(pointer(b), n, piv, ntasks)
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

    @testset "depth limit" begin
        # A zero depth budget sends every range to the fallback.
        v = rand(rng, Int32, 10^4)
        w = copy(v)
        buf = SIMDQuickSort.scratch(Int32)
        GC.@preserve w buf SIMDQuickSort.qsort!(pointer(w), length(w), pointer(buf), 0, UInt64(1))
        @test w == sort(v)
        GC.@preserve w SIMDQuickSort.pqsort!(pointer(v), length(v), 1, UInt64(1), 4, length(v))
        @test v == sort(w)
    end

    @testset "fallback" begin
        v = rand(rng, Int16, 1000)
        @test simdsort(v) == sort(v)
        s = ["b", "a", "c"]
        @test simdsort(s; rev=true) == ["c", "b", "a"]
    end
end
