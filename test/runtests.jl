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
