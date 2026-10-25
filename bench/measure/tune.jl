# Tuning constants on this machine. Each variant is a copy of src/ with one
# constant changed, loaded as its own module (variants.jl), so that all
# variants run interleaved in one process. Writes $OUT/tune.csv with the best and
# median time per case and variant, and prints each variant relative to the
# baseline (the committed values).
#
#   OUT=dir julia -t <n> --project=bench/measure bench/measure/tune.jl [rounds]

using SIMD, Random, Statistics, Printf

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results"))
mkpath(OUT)
const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const NT = Threads.nthreads()
const SRC = joinpath(@__DIR__, "..", "..", "src")

# (label, file, constant, new value); one constant per variant.
const VARIANTS = [
    ("baseline", "", "", ""),
    ("UNROLL=4", "sort.jl", "UNROLL", "4"),
    ("UNROLL=16", "sort.jl", "UNROLL", "16"),
    ("PFDIST=256", "partition.jl", "PFDIST", "256"),
    ("PFDIST=1024", "partition.jl", "PFDIST", "1024"),
    ("PF_MIN_BYTES=512K", "partition.jl", "PF_MIN_BYTES", "512 << 10"),
    ("PF_MIN_BYTES=8M", "partition.jl", "PF_MIN_BYTES", "8 << 20"),
    ("no prefetch", "partition.jl", "PF_MIN_BYTES", "typemax(Int)"),
    ("GRAIN_FACTOR=4", "parallel.jl", "GRAIN_FACTOR", "4"),
    ("GRAIN_FACTOR=64", "parallel.jl", "GRAIN_FACTOR", "64"),
    ("PPART_MIN=2^18", "parallel.jl", "PPART_MIN", "1 << 18"),
    ("PPART_MIN=2^22", "parallel.jl", "PPART_MIN", "1 << 22"),
    ("BIGPIVOT_MINN=2^16", "sort.jl", "BIGPIVOT_MINN", "1 << 16"),
    ("BIGPIVOT_MINN=2^21", "sort.jl", "BIGPIVOT_MINN", "1 << 21"),
    ("MAXROWS=16", "network.jl", "MAXROWS", "16"),
    ("no zigzag", "network.jl", "ZIGZAG", "false"),
    ("MEDIAN_LANES=16", "sort.jl", "MEDIAN_LANES", "16"),
]

include("variants.jl")

rng = Xoshiro(5)
const L = parse(Int, get(ENV, "SIZE_LOG", "26"))
const CASES = [
    ("Int64 2^$L, 1 thr", () -> rand(rng, Int64, 2^L), 1),
    ("Float64 2^$L, 1 thr", () -> randn(rng, 2^L), 1),
    ("Int32 2^$L, 1 thr", () -> rand(rng, Int32, 2^L), 1),
    ("Int64 range 2^40, 2^$L, 1 thr", () -> rand(rng, Int64(0):Int64(2)^40, 2^L), 1),
    ("UInt64 2^$(L+1), $NT thr", () -> rand(rng, UInt64, 2^(L + 1)), NT),
    ("Float64 2^$L, $NT thr", () -> randn(rng, 2^L), NT),
    ("Int32 2^$L, $NT thr", () -> rand(rng, Int32, 2^L), NT),
]

function main(mods)
    io = open(joinpath(OUT, "tune.csv"), "w")
    println(io, "case,variant,best,median")
    for (case, gen, nt) in CASES
        x = gen()
        ref = sort(x)
        # A variant that fails (an error or a wrong result) is reported and
        # left out of this case.
        ok = filter(mods) do (lab, M)
            y = copy(x)
            try
                M.simdsort!(y; ntasks = nt)
            catch err
                println("  $lab: ERROR ", sprint(showerror, err)[1:min(end, 200)])
                return false
            end
            isequal(y, ref) || (println("  $lab: WRONG RESULT"); return false)
            return true
        end
        ts = Dict(lab => Float64[] for (lab, _) in ok)
        for _ in 1:ROUNDS
            x = gen()
            for (lab, M) in shuffle(rng, ok)
                y = copy(x)
                GC.gc(false)
                push!(ts[lab], @elapsed M.simdsort!(y; ntasks = nt))
            end
        end
        base = minimum(ts["baseline"])
        println("== $case (baseline $(round(base; digits = 4)) s)")
        for (lab, _) in ok
            b = minimum(ts[lab])
            println(io, "\"", case, "\",\"", lab, "\",", b, ",", median(ts[lab]))
            @printf("  %-22s %.4f s  %+6.1f%%\n", lab, b, 100 * (b / base - 1))
        end
        flush(io)
        x = ref = nothing; GC.gc()
    end
    close(io)
    println("wrote ", joinpath(OUT, "tune.csv"))
end

# Loaded at top level: methods defined inside main() would be too new for it.
const MODS = [(lab, load_variant(SRC, "SQS_v$i"; file, name, value))
              for (i, (lab, file, name, value)) in enumerate(VARIANTS)]
main(MODS)
