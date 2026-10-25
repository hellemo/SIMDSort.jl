# Float keys through the float kernels or through the integer kernels, at 256
# and 512 bits: are 512-bit floating-point instructions what keeps Float64 from
# gaining on AVX-512? Intel server cores lower their clock further for "heavy"
# 512-bit instructions (floating point) than for integer ones, and the lower
# clock slows everything, scalar code included.
#
#   julia --project=bench/measure bench/measure/floatkernels.jl [rounds=7]
#
# The same random input is sorted by sort_auto! as floats, and as the
# order-preserving signed keys of their bits (keymap; the map passes are not
# timed), one thread, best and median of fresh inputs; the one-thread phase
# times of the median runs show whether the scalar parts (pivot) slow down.

using SIMD, Random, Statistics, Printf

include(joinpath(@__DIR__, "variants.jl"))

const SRC = joinpath(@__DIR__, "..", "..", "src")
const ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 7
const V256 = load_variant(SRC, "SIMDSort_256"; file = "network.jl", name = "VEC_BITS", value = "256")
const V512 = load_variant(SRC, "SIMDSort_512"; file = "network.jl", name = "VEC_BITS", value = "512")

rng = Xoshiro(29)

# Seconds and phases of one sort of fresh keys; as = :float or :int.
function one(M, ::Type{T}, n, as) where {T}
    x = T === Float64 ? randn(rng, n) : randn(rng, Float32, n)
    k = as === :float ? x : M.map_keys!(reinterpret(M.signedkey(T), x), T, false)
    r = M.instrumented(() -> M.sort_auto!(k, 1))
    return r.wall, r.phases
end

ph(p, k) = haskey(p, k) ? first(p[k]) : 0.0

function main()
    println("one thread; best / median of ", ROUNDS, " (s); pivot and leaf_sort from the median run")
    for T in (Float64, Float32), (wname, M) in (("256", V256), ("512", V512)), as in (:float, :int)
        n = T === Float64 ? 2^24 : 2^25
        one(M, T, 2^12, as)
        runs = [one(M, T, n, as) for _ in 1:ROUNDS]
        sort!(runs; by = first)
        t = first.(runs); med = runs[(ROUNDS + 1) ÷ 2][2]
        partition = sum(v -> first(v), (v for (k, v) in med if startswith(String(k), "partition_")); init = 0.0)
        @printf("  %-7s %s-bit as %-5s  %.4f / %.4f   pivot %.4f  leaf %.4f  partitions %.4f\n",
                T, wname, as, minimum(t), median(t), ph(med, :pivot), ph(med, :leaf_sort), partition)
    end
end

main()
