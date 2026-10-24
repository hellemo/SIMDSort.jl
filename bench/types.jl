# sort! and sortperm for strings, symbols, wide and arbitrary-precision numbers:
# SIMDQuickSort against Base and ChapelSort.jl (which falls back to Base for the
# types it does not radix sort).
#
#   julia -t 8 --project=bench bench/types.jl [rounds=3]
#
# Every round draws a fresh input (the same for all methods in the round) and
# runs the methods in a shuffled order; the best round is reported. Writes
# bench/results/types.csv.

using SIMDQuickSort, ChapelSort, BitIntegers, DoubleFloats, MultiFloats, Quadmath, Dates, Printf, Random, Statistics

const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const NT = Threads.nthreads()

randstr(rng, n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))

# (name, n, rng -> generator)
const CASES = [
    ("String, random, length 8–20", 10^7, rng -> () -> [randstr(rng, rand(rng, 8:20)) for _ in 1:10^7]),
    ("String, h2o id3 (id%010d)", 10^7,
     rng -> (ids = ["id" * lpad(string(i), 10, "0") for i in 1:10^6]; () -> rand(rng, ids, 10^7))),
    ("Symbol, 10^5 distinct", 10^7,
     rng -> (s = [Symbol(randstr(rng, rand(rng, 4:12))) for _ in 1:10^5]; () -> rand(rng, s, 10^7))),
    ("Int128", 10^7, rng -> () -> rand(rng, Int128, 10^7)),
    ("UInt256 (BitIntegers)", 10^7, rng -> () -> rand(rng, UInt256, 10^7)),
    ("Double64 (DoubleFloats)", 10^7, rng -> () -> Double64.(randn(rng, 10^7)) .+ Double64.(randn(rng, 10^7)) .* 1e-17),
    ("Float128 (Quadmath)", 10^7, rng -> () -> Float128.(randn(rng, 10^7))),
    ("Float64x2 (MultiFloats)", 10^7, rng -> () -> Float64x2.(randn(rng, 10^7)) .+ Float64x2.(randn(rng, 10^7)) .* 1e-17),
    ("Float64x4 (MultiFloats)", 10^6, rng -> () -> Float64x4.(randn(rng, 10^6)) .+ Float64x4.(randn(rng, 10^6)) .* 1e-17),
    ("Date", 10^7, rng -> () -> Date(1900) .+ Day.(rand(rng, 0:80_000, 10^7))),
    ("DateTime", 10^7, rng -> () -> DateTime(1950) .+ Millisecond.(rand(rng, Int64(0):Int64(10)^12, 10^7))),
    ("Tuple{Int64,Float64}", 10^7, rng -> () -> collect(zip(rand(rng, 1:1000, 10^7), randn(rng, 10^7)))),
    ("Pair{Int32,Int64}", 10^7, rng -> () -> [rand(rng, Int32) => rand(rng, Int64) for _ in 1:10^7]),
    ("Tuple{Int64,String}", 2 * 10^6,
     rng -> () -> collect(zip(rand(rng, 1:100, 2 * 10^6), [string(rand(rng, UInt32)) for _ in 1:2 * 10^6]))),
    ("VersionNumber", 10^6, rng -> () -> [VersionNumber(rand(rng, 0:9), rand(rng, 0:20), rand(rng, 0:50)) for _ in 1:10^6]),
    ("BigInt (~128-384 bits)", 10^6,
     rng -> () -> [big(rand(rng, Int128)) * big(rand(rng, Int128))^rand(rng, 0:2) for _ in 1:10^6]),
    ("BigFloat (256 bits)", 10^6, rng -> () -> BigFloat.(randn(rng, 10^6)) .* big(10.0) .^ rand(rng, -20:20, 10^6)),
]

methods(sortonly) = [
    ("Base", 1, sortonly ? sort! : sortperm),
    ("ChapelSort.jl", 1, sortonly ? (v -> psort!(v; ntasks = 1)) : (v -> psortperm(v; ntasks = 1))),
    ("ChapelSort.jl", NT, sortonly ? (v -> psort!(v; ntasks = NT)) : (v -> psortperm(v; ntasks = NT))),
    ("SIMDQuickSort.jl", 1, sortonly ? (v -> simdsort!(v; ntasks = 1)) : (v -> simdsortperm(v; ntasks = 1))),
    ("SIMDQuickSort.jl", NT, sortonly ? (v -> simdsort!(v; ntasks = NT)) : (v -> simdsortperm(v; ntasks = NT))),
]

const ROWS = NamedTuple{(:case, :op, :method, :nthreads, :n, :best, :median),
                        Tuple{String,String,String,Int,Int,Float64,Float64}}[]

function run_case(name, n, mkgen)
    rng = Xoshiro(hash(name))
    gen = mkgen(rng)
    for (op, sortonly) in (("sort", true), ("sortperm", false))
        ms = methods(sortonly)
        x = gen()
        ref = sortonly ? sort(x) : sortperm(x)
        for (m, t, f) in ms                     # compile and check
            r = f(copy(x))
            isequal(r, ref) || error("$m ($t thr) on $name $op: wrong result")
        end
        times = [Float64[] for _ in ms]
        for _ in 1:ROUNDS
            x = gen()
            for i in randperm(rng, length(ms))
                y = copy(x)
                GC.gc(false)
                push!(times[i], @elapsed ms[i][3](y))
            end
        end
        for (i, (m, t, _)) in enumerate(ms)
            push!(ROWS, (; case = name, op, method = m, nthreads = t, n, best = minimum(times[i]),
                         median = median(times[i])))
        end
        @printf("%-30s %-8s %s\n", name, op,
                join([@sprintf("%s %dthr %.3f", m, t, minimum(times[i])) for (i, (m, t, _)) in enumerate(ms)], " | "))
        flush(stdout)
    end
end

function main()
    println("Julia $VERSION, $NT threads, best of $ROUNDS interleaved rounds, fresh input per round")
    for (name, n, mkgen) in CASES
        run_case(name, n, mkgen)
    end
    out = joinpath(get(ENV, "BENCH_RESULTS", joinpath(@__DIR__, "results")), "types.csv")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "case,op,method,nthreads,n,best,median")
        for r in ROWS
            println(io, "\"", r.case, "\",", r.op, ",\"", r.method, "\",", r.nthreads, ",", r.n, ",", r.best, ",", r.median)
        end
    end
    println("wrote ", out)
end

main()
