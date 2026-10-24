# SIMDSort vs Base, ChapelSort.jl, and two C++ SIMD quicksorts: Intel's
# x86-simd-sort (OpenMP parallel) and Google Highway's vqsort (single-threaded).
# The C++ sorts are called through the shim library built by ChapelSort.jl's
# bench/simd/build.sh (set SIMDSORT_LIB to use another copy).
#
#   julia -t 8 --project=bench bench/bench.jl [rounds=9]
#
# Each round draws a fresh input (a new seed, the same input for every method in
# that round) and runs all methods on it in a freshly shuffled order. Fresh data
# means no method is timed on an input whose branch pattern the predictor has
# already seen in an earlier repetition; interleaving spreads thermal and power
# drift evenly over the methods. The table reports the best round per method; the
# CSV also has the median.
#
# Writes bench/results/bench.csv and prints a Markdown table.

using SIMDSort, ChapelSort, Libdl, Printf, Random, Statistics

const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 9
const BASE_ROUNDS = min(ROUNDS, 5)      # Base.sort! is slow; fewer rounds
const LIBPATH = get(ENV, "SIMDSORT_LIB",
                    joinpath(@__DIR__, "..", "..", "ChapelSort.jl", "bench", "simd", "libsimdsort.so"))
const LIB = dlopen(LIBPATH)
const NT = Threads.nthreads()

suffix(::Type{UInt64}) = "u64"
suffix(::Type{Int64}) = "i64"
suffix(::Type{Float64}) = "f64"
suffix(::Type{UInt32}) = "u32"
suffix(::Type{Int32}) = "i32"
suffix(::Type{Float32}) = "f32"

function xss!(v::Vector{T}, nt) where {T}
    f = dlsym(LIB, Symbol("xss_sort_", suffix(T)))
    ccall(f, Cvoid, (Ptr{T}, Csize_t, Cint), v, length(v), nt)
    return v
end
function vqsort!(v::Vector{T}) where {T}
    f = dlsym(LIB, Symbol("vqsort_", suffix(T)))
    ccall(f, Cvoid, (Ptr{T}, Csize_t), v, length(v))
    return v
end

const METHODS = [
    ("Base.sort!", 1, sort!),
    ("ChapelSort.jl psort!", 1, v -> psort!(v; ntasks = 1)),
    ("ChapelSort.jl psort!", NT, v -> psort!(v; ntasks = NT)),
    ("SIMDSort.jl", 1, v -> simdsort!(v; ntasks = 1)),
    ("SIMDSort.jl", NT, v -> simdsort!(v; ntasks = NT)),
    ("x86-simd-sort", 1, v -> xss!(v, 1)),
    ("x86-simd-sort", NT, v -> xss!(v, NT)),
    ("Highway vqsort", 1, vqsort!),
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

const ROWS = NamedTuple{(:case, :method, :nthreads, :n, :best, :median),
                        Tuple{String,String,Int,Int,Float64,Float64}}[]

# `gen(rng)` returns a fresh input.
function case(name, gen)
    selected(name) || return
    rng = Xoshiro(hash(name))
    x = gen(rng)
    ref = sort(x)
    y = similar(x)
    for (method, nt, f!) in METHODS                     # compile and check
        copyto!(y, x)
        f!(y)
        isequal(y, ref) || error("$method on $name: wrong result")
    end
    ref = nothing
    times = [Float64[] for _ in METHODS]
    for round in 1:ROUNDS
        x = gen(rng)
        for i in randperm(rng, length(METHODS))
            method, nt, f! = METHODS[i]
            method == "Base.sort!" && round > BASE_ROUNDS && continue
            copyto!(y, x)
            GC.gc(false)
            push!(times[i], @elapsed f!(y))
        end
    end
    for (i, (method, nt, _)) in enumerate(METHODS)
        best, med = minimum(times[i]), median(times[i])
        @printf("%-34s %-22s %2d thr  best %7.3f s  median %7.3f s  (%d rounds)\n",
                name, method, nt, best, med, length(times[i]))
        push!(ROWS, (; case = name, method, nthreads = nt, n = length(x), best, median = med))
    end
end

function markdown(io)
    cols = unique((r.method, r.nthreads) for r in ROWS)
    println(io, "| case | ", join(("$m ($t thr)" for (m, t) in cols), " | "), " |")
    println(io, "|---|", repeat("---:|", length(cols)))
    for c in unique(r.case for r in ROWS)
        ts = [only(r.best for r in ROWS if r.case == c && (r.method, r.nthreads) == mt)
              for mt in cols]
        best = minimum(ts)
        cells = [t == best ? @sprintf("**%.3f**", t) : @sprintf("%.3f", t) for t in ts]
        println(io, "| ", c, " | ", join(cells, " | "), " |")
    end
end

function main()
    println("$(unsafe_string(ccall(dlsym(LIB, :simdsort_info), Cstring, ()))); ",
            "Julia $VERSION, $NT threads, best of $ROUNDS interleaved rounds ",
            "($BASE_ROUNDS for Base.sort!), fresh input per round")
    case("UInt64, 2^27", rng -> rand(rng, UInt64, 2^27))
    case("Int64, 2^26", rng -> rand(rng, Int64, 2^26))
    case("Float64 (randn), 2^26", rng -> randn(rng, 2^26))
    case("UInt32, 2^26", rng -> rand(rng, UInt32, 2^26))
    case("Int32, 2^26", rng -> rand(rng, Int32, 2^26))
    case("Float32 (randn), 2^26", rng -> randn(rng, Float32, 2^26))
    case("UInt64 in 0:1000, 2^26", rng -> rand(rng, UInt64(0):UInt64(1000), 2^26))
    case("UInt64, 1000 random values, 2^26", rng -> rand(rng, rand(rng, UInt64, 1000), 2^26))
    case("UInt64 sorted, 2^26", rng -> sort!(rand(rng, UInt64, 2^26)))
    case("UInt64 reversed, 2^26", rng -> sort!(rand(rng, UInt64, 2^26); rev = true))
    case("Int32, 2^20", rng -> rand(rng, Int32, 2^20))
    out = joinpath(get(ENV, "BENCH_RESULTS", joinpath(@__DIR__, "results")), "bench.csv")
    mkpath(dirname(out))
    write_rows(out, "case,method,nthreads,n,best,median",
               [string("\"", r.case, "\",\"", r.method, "\",", r.nthreads, ",", r.n, ",", r.best, ",", r.median)
                for r in ROWS], unique(r.case for r in ROWS))
    println("wrote ", out)
    markdown(stdout)
end

main()
