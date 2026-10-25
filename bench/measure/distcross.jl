# Crossover between SIMDSort's two top levels for primitive keys: the parallel
# quicksort and the sample-sort distribution step (src/msd.jl), over input size
# and number of tasks, to choose the rule use_distribution(n, ntasks).
#
#   OUT=dir julia -t <n> --project=bench/measure bench/measure/distcross.jl [rounds=5] [maxlog2=28]
#   julia --project=bench/measure bench/measure/distcross.jl <distcross.csv>    # tables and fit only
#
# Every round draws a fresh input and runs the two top levels (forced through
# DIST_MODE) in random order, into one preallocated buffer.
# Writes $OUT/distcross.csv (best and median per type, n, ntasks and method),
# prints per type the ratio sample sort / quicksort of the best times (< 1:
# the sample-sort step is faster), and fits a rule of the form
#
#   ntasks >= t0(bytes per key) && n * bytes per key * ntasks >= W
#
# The quicksort makes about log2(ntasks) parallel partition passes over all
# keys before its tasks are independent, the sample-sort step a fixed ~3
# (sample, scatter, copy back) plus a cost per task, so the crossover size
# falls as the number of tasks grows (measured on the Xeon: Int64 from 2^26
# keys at 16 tasks to 2^24 at 56). The fit picks t0 per key width and one W
# that minimise the mean loss against always taking the faster method.

using SIMDSort, Random, Statistics, Printf
const Q = SIMDSort

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results"))
mkpath(OUT)
const CSV_IN = length(ARGS) >= 1 && endswith(ARGS[1], ".csv") ? ARGS[1] : ""
const ROUNDS = length(ARGS) >= 1 && isempty(CSV_IN) ? parse(Int, ARGS[1]) : 5
const MAXLOG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 28
const NT = Threads.nthreads()
TASKS = unique([filter(<=(NT), [4, 8, 12, 16, 20, 24, 32, 40, 48]); NT])
LOGS = 20:MAXLOG

rng = Xoshiro(17)
const CASES = [
    ("Int64", Int64, x -> rand!(rng, x)),
    ("UInt64", UInt64, x -> rand!(rng, x)),
    ("Float64 randn", Float64, x -> randn!(rng, x)),
    ("Int32", Int32, x -> rand!(rng, x)),
]

# SIMDSort with the sample-sort top level forced on or off.
function sort_with!(v, nt, dist::Bool)
    old = Q.DIST_MODE[]
    Q.DIST_MODE[] = dist ? :on : :off
    try
        simdsort!(v; ntasks = nt)
    finally
        Q.DIST_MODE[] = old
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

# The ratios of a measured distcross.csv, with its task counts and sizes.
function read_csv(path)
    best = Dict{Tuple{String,Int,Int,String},Float64}()
    for line in Iterators.drop(eachline(path), 1)
        m = match(r"^\"(.*)\",(\d+),(\d+),(\w+),([^,]+),", line)
        best[(m[1], trailing_zeros(parse(Int, m[2])), parse(Int, m[3]), m[4])] = parse(Float64, m[5])
    end
    global TASKS = sort(unique(k[3] for k in keys(best)))
    global LOGS = minimum(k[2] for k in keys(best)):maximum(k[2] for k in keys(best))
    return Dict((c, l, t) => best[(c, l, t, "dist")] / best[(c, l, t, "quicksort")]
                for (c, l, t, m) in keys(best) if m == "dist")
end

function measure()
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
    return ratio
end

function report(ratio)

    println("\nsample sort / quicksort (best times; < 1: the sample-sort step wins)")
    for (case, _, _) in CASES
        println("\n", case)
        println(rpad("n \\ ntasks", 11), join([lpad(t, 6) for t in TASKS]))
        for l in LOGS
            println(rpad("2^$l", 11), join([@sprintf("%6.2f", ratio[(case, l, t)]) for t in TASKS]))
        end
        println(rpad("crossover", 11), join([(n0 = crossover(ratio, case, t); lpad(isinf(n0) ? "-" : "2^$(round(Int, log2(n0)))", 6)) for t in TASKS]))
    end

    # The fit: a grid search over t0 per key width and W (bytes × tasks, in
    # half powers of two), minimising the geometric mean of chosen / faster.
    widths = sort(unique(sizeof(T) for (_, T, _) in CASES))
    loss(rule) = exp(mean(begin
        r = ratio[(case, l, t)]
        log((rule(sizeof(T), 2^l, t) ? r : 1.0) / min(r, 1.0))
    end for (case, T, _) in CASES for l in LOGS for t in TASKS))
    rule_of(t0s, W) = (b, n, t) -> t >= t0s[b] && n * b * t >= W
    best = (Inf, nothing, nothing)
    for W in 2.0 .^ (28:0.5:40), t0v in Iterators.product((TASKS for _ in widths)...)
        t0s = Dict(zip(widths, t0v))
        l = loss(rule_of(t0s, W))
        l < best[1] && (best = (l, t0s, W))
    end
    l, t0s, W = best
    current = (b, n, t) -> t >= Q.dist_mintasks(b) && n * b * t >= Q.DIST_MINWORK
    println("\nfitted rule: ntasks >= ", join(["$(t0s[b]) ($(8b)-bit keys)" for b in widths], ", "),
            @sprintf(" && n * bytes per key * ntasks >= 2^%.1f", log2(W)),
            @sprintf("; mean loss %.1f%%", 100 * (l - 1)))
    println("current rule: ntasks >= ", join(["$(Q.dist_mintasks(b)) ($(8b)-bit keys)" for b in widths], ", "),
            " && n * bytes per key * ntasks >= 2^$(round(Int, log2(Q.DIST_MINWORK)))",
            @sprintf("; mean loss %.1f%%", 100 * (loss(current) - 1)))
    @printf("never the sample-sort step: mean loss %.1f%%\n", 100 * (loss((b, n, t) -> false) - 1))
end

report(isempty(CSV_IN) ? measure() : read_csv(CSV_IN))
