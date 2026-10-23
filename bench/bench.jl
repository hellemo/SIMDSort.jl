# SIMDQuickSort vs Base, ChapelSort.jl, and two C++ SIMD quicksorts: Intel's
# x86-simd-sort (OpenMP parallel) and Google Highway's vqsort (single-threaded).
# The C++ sorts are called through the shim library built by ChapelSort.jl's
# bench/simd/build.sh (set SIMDSORT_LIB to use another copy).
#
#   julia -t 8 --project=bench bench/bench.jl [reps=5]
#
# Writes bench/results/bench.csv and prints a Markdown table.

using SIMDQuickSort, ChapelSort, Libdl, Printf, Random

const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
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
    ("SIMDQuickSort.jl", 1, v -> simdsort!(v; ntasks = 1)),
    ("SIMDQuickSort.jl", NT, v -> simdsort!(v; ntasks = NT)),
    ("x86-simd-sort", 1, v -> xss!(v, 1)),
    ("x86-simd-sort", NT, v -> xss!(v, NT)),
    ("Highway vqsort", 1, vqsort!),
]

const ROWS = NamedTuple{(:case, :method, :nthreads, :n, :seconds),
                        Tuple{String,String,Int,Int,Float64}}[]

function case(name, x)
    ref = sort(x)
    y = similar(x)
    for (method, nt, f!) in METHODS
        copyto!(y, x)
        f!(y)                                           # compile and check
        isequal(y, ref) || error("$method on $name: wrong result")
        reps = method == "Base.sort!" ? min(REPS, 3) : REPS
        ts = map(1:reps) do _
            copyto!(y, x)
            GC.gc()
            @elapsed f!(y)
        end
        t = sum(ts) / reps
        @printf("%-26s %-22s %2d thr %8.3f s %8.1f M elem/s\n", name, method, nt, t,
                length(x) / t / 1e6)
        push!(ROWS, (; case = name, method, nthreads = nt, n = length(x), seconds = t))
    end
end

function markdown(io)
    cols = unique((r.method, r.nthreads) for r in ROWS)
    println(io, "| case | ", join(("$m ($t thr)" for (m, t) in cols), " | "), " |")
    println(io, "|---|", repeat("---:|", length(cols)))
    for c in unique(r.case for r in ROWS)
        ts = [only(r.seconds for r in ROWS if r.case == c && (r.method, r.nthreads) == mt)
              for mt in cols]
        best = minimum(ts)
        cells = [t == best ? @sprintf("**%.3f**", t) : @sprintf("%.3f", t) for t in ts]
        println(io, "| ", c, " | ", join(cells, " | "), " |")
    end
end

function main()
    Random.seed!(42)
    println("$(unsafe_string(ccall(dlsym(LIB, :simdsort_info), Cstring, ()))); ",
            "Julia $VERSION, $NT threads, mean of $REPS runs")
    case("UInt64, 2^27", rand(UInt64, 2^27))
    case("Int64, 2^26", rand(Int64, 2^26))
    case("Float64 (randn), 2^26", randn(2^26))
    case("UInt32, 2^26", rand(UInt32, 2^26))
    case("Int32, 2^26", rand(Int32, 2^26))
    case("Float32 (randn), 2^26", randn(Float32, 2^26))
    case("UInt64 in 0:1000, 2^26", rand(UInt64(0):UInt64(1000), 2^26))
    case("UInt64 sorted, 2^26", sort!(rand(UInt64, 2^26)))
    case("UInt64 reversed, 2^26", sort!(rand(UInt64, 2^26); rev = true))
    case("Int32, 2^20", rand(Int32, 2^20))
    out = joinpath(@__DIR__, "results", "bench.csv")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "case,method,nthreads,n,seconds")
        for r in ROWS
            println(io, "\"", r.case, "\",\"", r.method, "\",", r.nthreads, ",", r.n, ",", r.seconds)
        end
    end
    println("wrote ", out)
    markdown(stdout)
end

main()
