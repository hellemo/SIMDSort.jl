# One case, many rounds, every round printed: to look into run-to-run swings
# (e.g. NUMA placement on a two-socket machine). The protocol is that of
# scaling.jl (fresh input each round, sorted in a fresh copy after
# GC.gc(false)), so the same swings show; per round it records the time, the
# GC time, and on Linux the CPU (and its NUMA node) where the input was made
# and where the sort started and ended.
#
#   OUT=dir LABEL=name julia -t <n> --project=bench/measure bench/measure/repeat.jl [case=symbol] [rounds=20] [ntasks=1]
#
# Writes $OUT/repeat-$LABEL.csv and prints a summary per NUMA node.

using SIMDSort, Random, Statistics, Printf

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results"))
mkpath(OUT)
const LABEL = get(ENV, "LABEL", "default")
const CASE = length(ARGS) >= 1 ? ARGS[1] : "symbol"
const ROUNDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
const NTASKS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

rng = Xoshiro(11)
randstr(n) = String(rand(rng, UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a], n))
const SYMS = [Symbol(randstr(rand(rng, 4:12))) for _ in 1:10^5]
const CASES = Dict(
    "symbol" => () -> rand(rng, SYMS, 10^7),               # as in scaling.jl
    "int64" => () -> rand(rng, Int64, 2^26),
    "float64" => () -> randn(rng, 2^26),
    "string" => () -> [randstr(rand(rng, 8:20)) for _ in 1:2 * 10^6],
)

cpu() = Sys.islinux() ? Int(ccall(:sched_getcpu, Cint, ())) : -1
# The NUMA node of a CPU, from sysfs (-1 if unknown).
function node(c::Int)
    c < 0 && return -1
    d = "/sys/devices/system/cpu/cpu$c"
    isdir(d) || return -1
    for f in readdir(d)
        m = match(r"^node(\d+)$", f)
        m === nothing || return parse(Int, m[1])
    end
    return -1
end

function main()
    gen = CASES[CASE]
    op = CASE == "string" ? (v -> simdsortperm(v; ntasks = NTASKS)) : (v -> simdsort!(v; ntasks = NTASKS))
    x = gen(); op(copy(x))                       # compile
    out = joinpath(OUT, "repeat-$LABEL.csv")
    rows = []
    open(out, "w") do io
        println(io, "label,case,ntasks,round,seconds,gc_seconds,gen_cpu,gen_node,start_cpu,start_node,end_cpu,end_node")
        for r in 1:ROUNDS
            c0 = cpu()
            x = gen()
            y = copy(x)
            GC.gc(false)
            c1 = cpu()
            st = @timed op(y)
            c2 = cpu()
            row = (r, st.time, st.gctime, c0, node(c0), c1, node(c1), c2, node(c2))
            push!(rows, row)
            println(io, LABEL, ",", CASE, ",", NTASKS, ",", join(row, ","))
            @printf("%-10s round %2d  %.4f s  gc %.4f  cpu gen %d (node %d) start %d (node %d) end %d (node %d)\n",
                    LABEL, row...)
        end
    end
    ts = [r[2] for r in rows]
    @printf("\n%s, %s, %d tasks: best %.4f  median %.4f  max %.4f  (max/best %.2f)\n",
            LABEL, CASE, NTASKS, minimum(ts), median(ts), maximum(ts), maximum(ts) / minimum(ts))
    for nd in sort(unique(r[7] for r in rows))
        t = [r[2] for r in rows if r[7] == nd]
        @printf("  sort started on node %2d: %2d rounds, median %.4f\n", nd, length(t), median(t))
    end
    moved = count(r -> r[5] != r[7] || r[7] != r[9], rows)
    println("  rounds where input, sort start and sort end were not on one node: $moved of $ROUNDS")
    println("wrote ", out)
end

main()
