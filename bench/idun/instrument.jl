# Phase timers and event counters (src/instrument.jl) for the main cases at 1,
# NT/4 and NT threads (bench/idun/run.sh step `instrument`). Writes
# $OUT/instrument.txt (readable reports) and $OUT/instrument.csv (one row per
# case, thread count and phase or counter).
#
#   OUT=dir julia -t <n> --project=bench/idun bench/idun/instrument.jl
#
# Each case: a warm-up, the best of 3 plain runs ("wall_plain") and the best of
# 3 instrumented runs; the report is from the fastest instrumented run, so the
# difference of the two walls shows the probe overhead. Phase times are task
# times summed over threads; nested phases overlap their parents.

using SIMDSort, Random, Printf, Dates, DoubleFloats, Quadmath, MultiFloats
const Q = SIMDSort

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results", "local"))
mkpath(OUT)
const NT = Threads.nthreads()
const L = parse(Int, get(ENV, "SIZE_LOG", "26"))     # 2^L keys for 4- and 8-byte types
rng = Xoshiro(2026)
randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))

# (name, generator, operation); operation :sort (simdsort!) or :perm (simdsortperm)
const CASES = [
    ("UInt64 2^$(L+1)", () -> rand(rng, UInt64, 2^(L + 1)), :sort),
    ("Int64 2^$L", () -> rand(rng, Int64, 2^L), :sort),
    ("Float64 randn 2^$L", () -> randn(rng, 2^L), :sort),
    ("UInt32 2^$L", () -> rand(rng, UInt32, 2^L), :sort),
    ("Int32 2^$L", () -> rand(rng, Int32, 2^L), :sort),
    ("Float32 randn 2^$L", () -> randn(rng, Float32, 2^L), :sort),
    ("UInt64 in 0:1000 2^$L", () -> rand(rng, UInt64(0):UInt64(1000), 2^L), :sort),
    ("UInt64 1000 values 2^$L", () -> rand(rng, rand(rng, UInt64, 1000), 2^L), :sort),
    ("Int64 range 2^40 2^$L", () -> rand(rng, Int64(0):Int64(2)^40, 2^L), :sort),
    ("Float64 few distinct 2^$L", () -> rand(rng, randn(rng, 300), 2^L), :sort),
    ("UInt64 sorted 2^$L", () -> sort!(rand(rng, UInt64, 2^L)), :sort),
    ("Float64 sortperm 10^7", () -> randn(rng, 10^7), :perm),
    ("Int128 10^7", () -> rand(rng, Int128, 10^7), :sort),
    ("Float128 10^7", () -> Float128.(randn(rng, 10^7)), :sort),
    ("Double64 10^7", () -> Double64.(randn(rng, 10^7)), :sort),
    ("Float64x2 10^7", () -> Float64x2.(randn(rng, 10^7)), :sort),
    ("Pair{Int32,Int64} 10^7", () -> [rand(rng, Int32) => rand(rng, Int64) for _ in 1:10^7], :sort),
    ("String 8-24 sortperm 2*10^6", () -> [randstr(rand(rng, 8:24)) for _ in 1:2 * 10^6], :perm),
]

run!(v, op, nt) = op === :sort ? simdsort!(v; ntasks = nt) : simdsortperm(v; ntasks = nt)

function measure(name, gen, op, nt; label = name)
    run!(gen(), op, nt)                          # compile
    plain = minimum(1:3) do _
        v = gen(); GC.gc(false)
        @elapsed run!(v, op, nt)
    end
    best = nothing
    for _ in 1:3
        v = gen(); GC.gc(false)
        r = Q.instrumented(() -> run!(v, op, nt))
        (best === nothing || r.wall < best.wall) && (best = r)
    end
    return plain, best
end

function main()
    txt = open(joinpath(OUT, "instrument.txt"), "w")
    csv = open(joinpath(OUT, "instrument.csv"), "w")
    println(csv, "case,nthreads,kind,name,value,calls")
    threads = unique(filter(>(0), [1, NT ÷ 4, NT]))
    row(case, nt, kind, name, value, calls) =
        println(csv, "\"", case, "\",", nt, ",", kind, ",", name, ",", value, ",", calls)
    jobs = [(c..., nt, c[1]) for c in CASES for nt in threads]
    # The distribution top level (disabled by default) for 64-bit keys at NT.
    push!(jobs, (CASES[1]..., NT, CASES[1][1] * " [dist]"), (CASES[2]..., NT, CASES[2][1] * " [dist]"))
    for (name, gen, op, nt, label) in jobs
        dist = endswith(label, "[dist]")
        dist && (Q.DIST_MINN[] = 1 << 20)
        plain, r = try
            measure(name, gen, op, nt)
        catch err
            println(txt, "== $label, $nt threads: ERROR $err"); flush(txt)
            continue
        finally
            Q.DIST_MINN[] = typemax(Int)
        end
        hdr = @sprintf("== %s, %d threads: wall %.4f s plain, %.4f s instrumented", label, nt, plain, r.wall)
        println(hdr); println(txt, hdr)
        Q.print_report(txt, r); flush(txt)
        row(label, nt, "wall", "plain", plain, 1)
        row(label, nt, "wall", "instrumented", r.wall, 1)
        for (k, (t, c)) in r.phases
            row(label, nt, "phase", k, t, c)
        end
        for (k, v) in r.counters
            row(label, nt, "counter", k, v, 1)
        end
        flush(csv)
        GC.gc()
    end
    close(txt); close(csv)
    println("wrote ", joinpath(OUT, "instrument.txt"), " and instrument.csv")
end

main()
