# A/B regression check: a baseline SIMDSort source tree against the working
# tree, both loaded as module copies in one process (bench/measure/variants.jl)
# and run interleaved: fresh input every round, shuffled order, best and median
# of the rounds. Used as the gate for refactoring steps: a case is slower if
# both its median and its best time are worse by more than GATE (2%); rerun
# flagged cases (AB_ONLY) to confirm before acting on them.
#
#   git worktree add /workspace/envs/ab-base <commit>      # once per baseline
#   julia -t 8 --project=bench/measure bench/ab.jl /workspace/envs/ab-base/src [rounds]
#
# Env: SIZE_LOG (2^k keys for the primitive cases, default 24), NT (threads
# for the parallel cases, default all), OUT (if set, writes $OUT/ab.csv),
# AB_ONLY (run only the cases whose name contains this string).

using SIMD, Random, Statistics, Printf

include(joinpath(@__DIR__, "measure", "variants.jl"))

length(ARGS) >= 1 || error("usage: bench/ab.jl <baseline src dir> [rounds]")
const BASE_SRC = ARGS[1]
const ROUNDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 9
const NEW_SRC = joinpath(@__DIR__, "..", "src")
const GATE = 1.02
const L = parse(Int, get(ENV, "SIZE_LOG", "24"))
const NT = parse(Int, get(ENV, "NT", string(Threads.nthreads())))

const BASE = load_variant(BASE_SRC, "SIMDSort_base")
const NEW = load_variant(NEW_SRC, "SIMDSort_new")

rng = Xoshiro(2024)
randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))
const SYMS = [Symbol(randstr(rand(rng, 4:12))) for _ in 1:10^4]

# (name, generator, :sort or :perm)
const CASES = [
    ("Int64 2^$L", () -> rand(rng, Int64, 2^L), :sort),
    ("Float64 2^$L", () -> randn(rng, 2^L), :sort),
    ("Int32 2^$L", () -> rand(rng, Int32, 2^L), :sort),
    ("UInt64 in 0:1000 2^$L", () -> rand(rng, UInt64(0):UInt64(1000), 2^L), :sort),
    ("UInt64 sorted 2^$L", () -> sort!(rand(rng, UInt64, 2^L)), :sort),
    ("Float64 sortperm 2^20", () -> randn(rng, 2^20), :perm),
    ("String sortperm 2^20", () -> [randstr(rand(rng, 8:20)) for _ in 1:2^20], :perm),
    ("Int128 2^20", () -> rand(rng, Int128, 2^20), :sort),
    ("Pair{Int32,Int64} 2^20", () -> [rand(rng, Int32) => rand(rng, Int64) for _ in 1:2^20], :sort),
    ("Symbol 2^20", () -> rand(rng, SYMS, 2^20), :sort),
]

run!(M, v, op, nt) = op === :sort ? M.simdsort!(v; ntasks = nt) : M.simdsortperm(v; ntasks = nt)

function main()
    rows = []
    worst = 1.0
    println(rpad("case", 30), rpad("thr", 5), "   base best/median  |   new best/median  |  new/base (median)")
    only = get(ENV, "AB_ONLY", "")
    for (name, gen, op) in CASES, nt in unique([1, NT])
        isempty(only) || occursin(only, name) || continue
        x = gen()
        a = run!(BASE, copy(x), op, nt); b = run!(NEW, copy(x), op, nt)
        isequal(a, b) || error("$name, $nt threads: results differ")
        # Repetitions per timing so that each fills about 20 ms (short cases
        # are otherwise dominated by timer and scheduling noise); one
        # preallocated buffer, refilled with copyto!, so no page faults.
        y = copy(x)
        t1 = @elapsed run!(NEW, y, op, nt)
        reps = clamp(ceil(Int, 0.02 / max(t1, 1e-6)), 1, 1000)
        ts = Dict(:base => Float64[], :new => Float64[])
        for _ in 1:ROUNDS
            x = gen()
            for (k, M) in shuffle(rng, [(:base, BASE), (:new, NEW)])
                GC.gc(false)
                t = 0.0
                for _ in 1:reps
                    copyto!(y, x)
                    t += @elapsed run!(M, y, op, nt)
                end
                push!(ts[k], t / reps)
            end
        end
        mb, mn = median(ts[:base]), median(ts[:new])
        r = mn / mb
        rbest = minimum(ts[:new]) / minimum(ts[:base])
        slower = r > GATE && rbest > GATE
        slower && (worst = max(worst, min(r, rbest)))
        flag = slower ? "  <-- slower" : ""
        @printf("%-30s%-5d  %8.4f / %8.4f | %8.4f / %8.4f | %6.3f (best %5.3f)%s\n",
                name, nt, minimum(ts[:base]), mb, minimum(ts[:new]), mn, r, rbest, flag)
        push!(rows, (name, nt, minimum(ts[:base]), mb, minimum(ts[:new]), mn, r))
    end
    out = get(ENV, "OUT", "")
    if !isempty(out)
        mkpath(out)
        open(joinpath(out, "ab.csv"), "w") do io
            println(io, "case,nthreads,base_best,base_median,new_best,new_median,ratio")
            for r in rows
                println(io, "\"", r[1], "\",", join(r[2:end], ","))
            end
        end
    end
    println(worst > GATE ? "GATE: FAIL (worst slowdown $(round(worst; digits = 3)); rerun those cases with AB_ONLY)" :
                           "GATE: pass")
end

main()
