# Thread scaling (bench/idun/run.sh step `scaling`): best and median time per
# case, method and thread count (1, 2, 4, ... and all threads), fresh input per
# round, methods interleaved in random order. Writes $OUT/scaling.csv.
#
#   OUT=dir julia -t <n> --project=bench/idun bench/idun/scaling.jl [rounds]
#
# "SIMDSort dist" is the sample-sort top level for 64-bit keys
# (src/msd.jl), disabled by default because it loses on the 8-thread
# development laptop; this measures whether it wins at higher thread counts.

using SIMDSort, ChapelSort, Libdl, Random, Statistics, Printf
const Q = SIMDSort

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

# SIMDSort with the sample-sort top level forced on, and forced off.
function with_dist(f, on::Bool)
    old = (Q.DIST_MINN[], Q.DIST_MINTASKS[])
    Q.DIST_MINN[], Q.DIST_MINTASKS[] = on ? (1 << 20, 1) : (typemax(Int), typemax(Int))
    try
        f()
    finally
        Q.DIST_MINN[], Q.DIST_MINTASKS[] = old
    end
end
dist!(v, nt) = with_dist(() -> simdsort!(v; ntasks = nt), true)
qs!(v, nt) = with_dist(() -> simdsort!(v; ntasks = nt), false)

rng = Xoshiro(11)
randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))
const SYMS = [Symbol(randstr(rand(rng, 4:12))) for _ in 1:10^5]
const L = parse(Int, get(ENV, "SIZE_LOG", "26"))
# (case, generator, operation, 64-bit primitive?)
const CASES = [
    ("UInt64 2^$(L+1)", () -> rand(rng, UInt64, 2^(L + 1)), :sort),
    ("Int64 2^$L", () -> rand(rng, Int64, 2^L), :sort),
    ("Float64 randn 2^$L", () -> randn(rng, 2^L), :sort),
    ("Int32 2^$L", () -> rand(rng, Int32, 2^L), :sort),
    ("Int128 10^7", () -> rand(rng, Int128, 10^7), :sort),
    ("Float64 sortperm 10^7", () -> randn(rng, 10^7), :perm),
    ("String 8-20 sortperm 2*10^6", () -> [randstr(rand(rng, 8:20)) for _ in 1:2 * 10^6], :perm),
    ("Symbol 10^5 distinct sort 10^7", () -> rand(rng, SYMS, 10^7), :sort),
]

function methods_for(x, op, nt)
    T = eltype(x)
    ms = Pair{String,Function}[]
    if op === :perm
        push!(ms, "SIMDSort" => v -> simdsortperm(v; ntasks = nt), "ChapelSort" => v -> psortperm(v; ntasks = nt))
        return ms
    end
    push!(ms, "SIMDSort" => v -> simdsort!(v; ntasks = nt), "ChapelSort" => v -> psort!(v; ntasks = nt))
    # "SIMDSort" is the default (sample-sort top level from DIST_MINTASKS tasks);
    # the two forced variants show the crossover.
    T <: Union{Int32,UInt32,Float32,Int64,UInt64,Float64} &&
        push!(ms, "SIMDSort dist" => v -> dist!(v, nt), "SIMDSort quicksort" => v -> qs!(v, nt))
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
