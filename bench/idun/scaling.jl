# Thread scaling (bench/idun/run.sh step `scaling`): best and median time per
# case, method and thread count (1, 2, 4, ... and all threads), fresh input per
# round, methods interleaved in random order. Writes $OUT/scaling.csv.
#
#   OUT=dir julia -t <n> --project=bench bench/idun/scaling.jl [rounds]
#
# "SIMDQuickSort dist" is the sample-sort top level for 64-bit keys
# (src/msd.jl), disabled by default because it loses on the 8-thread
# development laptop; this measures whether it wins at higher thread counts.

using SIMDQuickSort, ChapelSort, Libdl, Random, Statistics, Printf
const Q = SIMDQuickSort

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results", "local"))
mkpath(OUT)
const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const NT = Threads.nthreads()
const THREADS = unique([filter(<=(NT), 2 .^ (0:10)); NT])
const LIBPATH = get(ENV, "SIMDSORT_LIB", joinpath(@__DIR__, "..", "..", "..", "ChapelSort.jl", "bench", "simd", "libsimdsort.so"))
const LIB = isfile(LIBPATH) ? dlopen(LIBPATH) : nothing
LIB === nothing && @warn "no $LIBPATH: x86-simd-sort is skipped"

suffix(::Type{UInt64}) = "u64"; suffix(::Type{Int64}) = "i64"; suffix(::Type{Float64}) = "f64"
suffix(::Type{UInt32}) = "u32"; suffix(::Type{Int32}) = "i32"; suffix(::Type{Float32}) = "f32"
function xss!(v::Vector{T}, nt) where {T}
    ccall(dlsym(LIB, Symbol("xss_sort_", suffix(T))), Cvoid, (Ptr{T}, Csize_t, Cint), v, length(v), nt)
    return v
end

function dist!(v, nt)
    Q.DIST_MINN[] = 1 << 20; Q.DIST_MINTASKS[] = 1
    try
        simdsort!(v; ntasks = nt)
    finally
        Q.DIST_MINN[] = typemax(Int); Q.DIST_MINTASKS[] = 8
    end
end

rng = Xoshiro(11)
const L = parse(Int, get(ENV, "SIZE_LOG", "26"))
# (case, generator, operation, 64-bit primitive?)
const CASES = [
    ("UInt64 2^$(L+1)", () -> rand(rng, UInt64, 2^(L + 1)), :sort),
    ("Int64 2^$L", () -> rand(rng, Int64, 2^L), :sort),
    ("Float64 randn 2^$L", () -> randn(rng, 2^L), :sort),
    ("Int32 2^$L", () -> rand(rng, Int32, 2^L), :sort),
    ("Int128 10^7", () -> rand(rng, Int128, 10^7), :sort),
    ("Float64 sortperm 10^7", () -> randn(rng, 10^7), :perm),
]

function methods_for(x, op, nt)
    T = eltype(x)
    ms = Pair{String,Function}[]
    if op === :perm
        push!(ms, "SIMDQuickSort" => v -> simdsortperm(v; ntasks = nt), "ChapelSort" => v -> psortperm(v; ntasks = nt))
        return ms
    end
    push!(ms, "SIMDQuickSort" => v -> simdsort!(v; ntasks = nt), "ChapelSort" => v -> psort!(v; ntasks = nt))
    T in (Int64, UInt64) && push!(ms, "SIMDQuickSort dist" => v -> dist!(v, nt))
    LIB !== nothing && T in (UInt64, Int64, Float64, UInt32, Int32, Float32) && push!(ms, "x86-simd-sort" => v -> xss!(v, nt))
    return ms
end

function main()
    out = joinpath(OUT, "scaling.csv")
    io = open(out, "w")
    println(io, "case,method,nthreads,n,best,median")
    for (case, gen, op) in CASES
        x0 = gen()
        n = length(x0)
        ref = op === :sort ? sort(x0) : sortperm(x0)
        for nt in THREADS
            ms = methods_for(x0, op, nt)
            for (k, f) in ms                       # compile and check
                y = copy(x0)
                r = f(y)
                ok = op === :sort ? isequal(y, ref) : r == ref
                ok || @warn "wrong result" case k nt
            end
            ts = Dict(k => Float64[] for (k, _) in ms)
            for _ in 1:ROUNDS
                x = gen()
                for (k, f) in shuffle(rng, ms)
                    y = copy(x)
                    GC.gc(false)
                    push!(ts[k], @elapsed f(y))
                end
            end
            for (k, _) in ms
                println(io, "\"", case, "\",\"", k, "\",", nt, ",", n, ",", minimum(ts[k]), ",", median(ts[k]))
            end
            flush(io)
            @printf("%-24s %3d thr  %s\n", case, nt,
                    join([@sprintf("%s %.4f", k, minimum(ts[k])) for (k, _) in ms], " | "))
        end
        x0 = ref = nothing; GC.gc()
    end
    close(io)
    println("wrote ", out)
end

main()
