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
    else
        error("unknown distribution $dist")
    end
end

const DISTS = (:uniform, :sorted, :reversed, :equal, :two, :few, :sawtooth,
               :organpipe, :extremes, :special)

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

    @testset "partition! $T U=$U" for T in (Int32, Int64), U in (1, 2, 4)
        N = SIMDQuickSort.lanes(T)
        buf = Vector{T}(undef, 2N)
        @test all(Iterators.product(2U*N:2U*N+200, 1:6)) do (n, t)
            a = t <= 2 ? rand(rng, T(-3):T(3), n) : rand(rng, T, n)
            piv = t == 3 ? maximum(a) : t == 4 ? minimum(a) : a[rand(rng, 1:n)]
            b = copy(a)
            m = GC.@preserve b buf SIMDQuickSort.partition!(pointer(b), n, piv, pointer(buf), Val(U))
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

    @testset "fallback" begin
        v = rand(rng, Int16, 1000)
        @test simdsort(v) == sort(v)
        s = ["b", "a", "c"]
        @test simdsort(s; rev=true) == ["c", "b", "a"]
    end
end
