# Phase breakdown of two variants side by side: where an A/B difference comes
# from. The variants are module copies (variants.jl), each with at most one
# constant changed, as in bench/ab.jl:
#
#   AB_BASE_EDIT="network.jl:VEC_BITS=256" julia -t <n> --project=bench/measure bench/measure/abphases.jl [rounds=5]
#
# Env: AB_BASE_EDIT, AB_NEW_EDIT ("file.jl:NAME=value"), NT (threads; default
# 1), OUT (if set, writes $OUT/abphases.csv, or abphases-$LABEL.csv). Per
# case and variant, the instrumented run with the median wall time of the
# rounds (fresh input each round) is reported.

using SIMD, Random, Statistics, Printf

include(joinpath(@__DIR__, "variants.jl"))

const SRC = joinpath(@__DIR__, "..", "..", "src")
const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const NT = parse(Int, get(ENV, "NT", "1"))

function variant(env, mod)
    edit = get(ENV, env, "")
    isempty(edit) && return load_variant(SRC, mod), "as is"
    file, rest = split(edit, ":"; limit = 2)
    name, value = split(rest, "="; limit = 2)
    return load_variant(SRC, mod; file, name, value), "$name = $value"
end
const (BASE, BLABEL) = variant("AB_BASE_EDIT", "SIMDSort_base")
const (NEW, NLABEL) = variant("AB_NEW_EDIT", "SIMDSort_new")

rng = Xoshiro(23)
const CASES = [
    ("Int64 2^24", () -> rand(rng, Int64, 2^24)),
    ("Float64 2^24", () -> randn(rng, 2^24)),
    ("Int32 2^24", () -> rand(rng, Int32, 2^24)),
]

# The instrumented run with the median wall time.
function median_run(M, gen)
    M.simdsort!(gen(); ntasks = NT)
    runs = [M.instrumented(() -> M.simdsort!(gen(); ntasks = NT)) for _ in 1:ROUNDS]
    sort!(runs; by = r -> r.wall)
    return runs[(ROUNDS + 1) ÷ 2]
end

phases(r) = Dict(String(k) => v for (k, v) in pairs(r.phases))
time_of(v) = v isa Tuple ? Float64(first(v)) : Float64(v)

function main()
    println("base: ", BLABEL, "   new: ", NLABEL, "   ", NT, " thread(s), median of ", ROUNDS)
    rows = []
    for (name, gen) in CASES
        rb, rn = median_run(BASE, gen), median_run(NEW, gen)
        pb, pn = phases(rb), phases(rn)
        @printf("\n%s: wall %.4f -> %.4f s (%.3f)\n", name, rb.wall, rn.wall, rn.wall / rb.wall)
        ks = sort(collect(union(keys(pb), keys(pn))); by = k -> -time_of(get(pb, k, 0.0)))
        for k in ks
            a, b = time_of(get(pb, k, 0.0)), time_of(get(pn, k, 0.0))
            max(a, b) < 0.001 && continue
            @printf("  %-22s %8.4f %8.4f  %+8.4f\n", k, a, b, b - a)
            push!(rows, (name, k, a, b))
        end
    end
    out = get(ENV, "OUT", "")
    if !isempty(out)
        mkpath(out)
        label = get(ENV, "LABEL", "")
        open(joinpath(out, isempty(label) ? "abphases.csv" : "abphases-$label.csv"), "w") do io
            println(io, "case,phase,base,new")
            foreach(r -> println(io, "\"", r[1], "\",", r[2], ",", r[3], ",", r[4]), rows)
        end
    end
end

main()
