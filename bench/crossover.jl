# Crossover between SIMDSort's quicksort and ChapelSort.jl's parallel radix
# sort, over input size, number of tasks and key type, to see whether a rule
# "use the radix sort above n₀ keys with at least t₀ tasks" exists.
#
#   julia -t 8 --project=bench bench/crossover.jl [rounds=5] [maxlog2=28]
#
# Protocol as in bench.jl: every round draws a fresh input (the same for both
# methods) and runs the two methods in a random order. Correctness is checked
# with `issorted` here (the full multiset check is in the tests and bench.jl). Writes
# bench/results/crossover.csv and prints, per key type, the ratio
# radix / quicksort of the best times (< 1: the radix sort is faster).

using SIMDSort, ChapelSort, Printf, Random, Statistics

const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const MAXLOG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 28
const NT = Threads.nthreads()
const TASKS = unique([filter(<=(NT), [1, 2, 4, 8, 16, 32]); NT])
const LOGS = filter(<=(MAXLOG), [20, 22, 24, 26, 27, 28])

# (name, element type, rng -> in-place generator)
const CASES = [
    ("Int64", Int64, rng -> x -> rand!(rng, x)),
    ("UInt64", UInt64, rng -> x -> rand!(rng, x)),
    ("Float64 (randn)", Float64, rng -> x -> randn!(rng, x)),
    ("UInt64, 1000 random values", UInt64, rng -> (vals = rand(rng, UInt64, 1000); x -> rand!(rng, x, vals))),
    ("Int32", Int32, rng -> x -> rand!(rng, x)),
]


# BENCH_ONLY=<substring>: run only the cases whose name contains it and replace
# just their rows in the existing CSV.
const ONLY = get(ENV, "BENCH_ONLY", "")
selected(name) = isempty(ONLY) || occursin(ONLY, name)
function write_rows(out, header, rows, names)
    keep = String[]
    if !isempty(ONLY) && isfile(out)
        for l in readlines(out)[2:end]
            any(nm -> startswith(l, "\"" * nm * "\""), names) || push!(keep, l)
        end
    end
    open(out, "w") do io
        println(io, header)
        foreach(l -> println(io, l), keep)
        foreach(l -> println(io, l), rows)
    end
end

const ROWS = NamedTuple{(:case, :method, :ntasks, :n, :best, :median),
                        Tuple{String,String,Int,Int,Float64,Float64}}[]

# Inputs are refilled in place: at 2^28 keys every array is 2 GB.
function measure(name, T, mkgen, n, t)
    rng = Xoshiro(hash((name, n, t)))
    refill! = mkgen(rng)
    methods = [("quicksort", v -> simdsort!(v; ntasks = t)), ("radix", v -> psort!(v; ntasks = t))]
    x = refill!(Vector{T}(undef, n))
    y = similar(x)
    for (_, f!) in methods                      # compile and check
        copyto!(y, x)
        f!(y)
        issorted(y) || error("$name n=$n t=$t: not sorted")
    end
    times = [Float64[], Float64[]]
    for _ in 1:ROUNDS
        refill!(x)
        for i in randperm(rng, 2)
            copyto!(y, x)
            GC.gc(false)
            push!(times[i], @elapsed methods[i][2](y))
        end
    end
    for (i, (m, _)) in enumerate(methods)
        push!(ROWS, (; case = name, method = m, ntasks = t, n, best = minimum(times[i]),
                     median = median(times[i])))
    end
    return minimum(times[2]) / minimum(times[1])
end

function main()
    println("Julia $VERSION, $NT threads, best of $ROUNDS interleaved rounds, fresh input per round")
    for (name, T, mkgen) in CASES
        selected(name) || continue
        println("\n$name: radix / quicksort, best times (< 1: radix faster)")
        println("  n     ", join([@sprintf("%8s", "$t thr") for t in TASKS]))
        for k in LOGS
            ratios = [measure(name, T, mkgen, 2^k, t) for t in TASKS]
            println(@sprintf("  2^%-3d ", k), join([@sprintf("%8.2f", r) for r in ratios]))
            flush(stdout)
        end
    end
    out = joinpath(get(ENV, "BENCH_RESULTS", joinpath(@__DIR__, "results")), "crossover.csv")
    mkpath(dirname(out))
    write_rows(out, "case,method,ntasks,n,best,median",
               [string("\"", r.case, "\",", r.method, ",", r.ntasks, ",", r.n, ",", r.best, ",", r.median)
                for r in ROWS], unique(r.case for r in ROWS))
    println("\nwrote ", out)
end

main()
