# Crossover between SIMDSort's two top levels for primitive keys: the parallel
# quicksort and the sample-sort distribution step (src/msd.jl), over input size
# and number of tasks, to choose the rule use_distribution(n, ntasks).
#
#   OUT=dir julia -t <n> --project=bench/measure bench/measure/distcross.jl [rounds=5] [maxlog2=28]
#
# Every round draws a fresh input and runs the two top levels (forced through
# DIST_MINN and DIST_MINTASKS) in random order, into one preallocated buffer.
# Writes $OUT/distcross.csv (best and median per type, n, ntasks and method),
# prints per type the ratio sample sort / quicksort of the best times (< 1:
# the sample-sort step is faster), and fits a rule of the form
#
#   ntasks >= t0 && n >= c * ntasks
#
# A cost model gives that form: the quicksort makes about log2(ntasks)
# parallel partition passes over all keys before its tasks are independent,
# the sample-sort step a fixed ~3 (sample, scatter, copy back) plus a cost
# per task (count tables, synchronisation) that must be small next to the
# work per task, n / ntasks. The constants are measured.

using SIMDSort, Random, Statistics, Printf
const Q = SIMDSort

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results"))
mkpath(OUT)
const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const MAXLOG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 28
const NT = Threads.nthreads()
const TASKS = unique([filter(<=(NT), [4, 8, 12, 16, 20, 24, 32, 40, 48]); NT])
const LOGS = 20:MAXLOG

rng = Xoshiro(17)
const CASES = [
    ("Int64", Int64, x -> rand!(rng, x)),
    ("UInt64", UInt64, x -> rand!(rng, x)),
    ("Float64 randn", Float64, x -> randn!(rng, x)),
    ("Int32", Int32, x -> rand!(rng, x)),
]

# SIMDSort with the sample-sort top level forced on or off.
function sort_with!(v, nt, dist::Bool)
    old = (Q.DIST_MINN[], Q.DIST_MINTASKS[])
    Q.DIST_MINN[], Q.DIST_MINTASKS[] = dist ? (1, 1) : (typemax(Int), typemax(Int))
    try
        simdsort!(v; ntasks = nt)
    finally
        Q.DIST_MINN[], Q.DIST_MINTASKS[] = old
    end
end

const METHODS = ["quicksort" => false, "dist" => true]

# The smallest n from which the sample-sort step wins at this task count, at
# that size and at all larger ones (Inf if it never does).
function crossover(ratio::Dict, case, nt)
    n0 = Inf
    for l in reverse(LOGS)
        r = get(ratio, (case, l, nt), NaN)
        isnan(r) && continue
        r < 1 || break
        n0 = 2.0^l
    end
    return n0
end

function main()
    out = joinpath(OUT, "distcross.csv")
    io = open(out, "w")
    println(io, "type,n,ntasks,method,best,median")
    ratio = Dict{Tuple{String,Int,Int},Float64}()
    for (case, T, fill!) in CASES, l in LOGS
        n = 2^l
        x = fill!(Vector{T}(undef, n))
        y = similar(x)
        for nt in TASKS
            t1 = 0.0
            for (_, d) in METHODS                  # compile and check
                copyto!(y, x)
                sort_with!(y, nt, d)
                issorted(y) || error("$case 2^$l, $nt tasks: not sorted")
                copyto!(y, x)
                t1 = max(t1, @elapsed sort_with!(y, nt, d))
            end
            # Repetitions so that each timing fills about 20 ms (small inputs
            # take a few ms, where timer and scheduling noise dominate).
            reps = clamp(ceil(Int, 0.02 / t1), 1, 100)
            ts = Dict(k => Float64[] for (k, _) in METHODS)
            for _ in 1:ROUNDS
                fill!(x)
                for (k, d) in shuffle(rng, METHODS)
                    GC.gc(false)
                    t = 0.0
                    for _ in 1:reps
                        copyto!(y, x)
                        t += @elapsed sort_with!(y, nt, d)
                    end
                    push!(ts[k], t / reps)
                end
            end
            for (k, _) in METHODS
                println(io, "\"", case, "\",", n, ",", nt, ",", k, ",", minimum(ts[k]), ",", median(ts[k]))
            end
            flush(io)
            ratio[(case, l, nt)] = minimum(ts["dist"]) / minimum(ts["quicksort"])
        end
        @printf("%-14s 2^%d  %s\n", case, l, join([@sprintf("%d: %.2f", t, ratio[(case, l, t)]) for t in TASKS], "  "))
        x = y = nothing
        GC.gc()
    end
    close(io)
    println("wrote ", out)

    println("\nsample sort / quicksort (best times; < 1: the sample-sort step wins)")
    for (case, _, _) in CASES
        println("\n", case)
        println(rpad("n \\ ntasks", 11), join([lpad(t, 6) for t in TASKS]))
        for l in LOGS
            println(rpad("2^$l", 11), join([@sprintf("%6.2f", ratio[(case, l, t)]) for t in TASKS]))
        end
        println(rpad("crossover", 11), join([(n0 = crossover(ratio, case, t); lpad(isinf(n0) ? "-" : "2^$(round(Int, log2(n0)))", 6)) for t in TASKS]))
    end

    # The fit: t0 is the smallest task count from which every type has a
    # crossover; c the largest n0 / ntasks over the types and those task counts.
    found = [t for t in TASKS if all(!isinf(crossover(ratio, case, t)) for (case, _, _) in CASES)]
    if isempty(found)
        println("\nno task count up to $NT at which the sample-sort step wins for all types")
    else
        t0 = minimum(found)
        c = maximum(crossover(ratio, case, t) / t for (case, _, _) in CASES for t in TASKS if t >= t0)
        @printf("\nfitted rule: ntasks >= %d && n >= %.3g * ntasks (= 2^%.1f keys per task)\n", t0, c, log2(c))
        println("current rule: ntasks >= $(Q.DIST_MINTASKS[]) && n >= 2^$(round(Int, log2(Q.DIST_MINN[])))")
    end
end

main()
