# Machine code of the hot kernels on this CPU: writes $OUT/asm/<kernel>_<T>.s
# and a summary table $OUT/codegen.csv with instruction counts that tell
# whether the AVX-512 decisions in docs/AVX512.md hold, e.g. native vpminsq/vpmaxsq in the Int64
# networks, the size of the 32-row networks, and spills (stack traffic).
#
#   OUT=dir julia --project=bench/measure bench/measure/codegen.jl

using SIMDSort, SIMD, InteractiveUtils, Printf
const Q = SIMDSort

const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results"))
const ASM = joinpath(OUT, "asm")
mkpath(ASM)

const PATTERNS = [
    "instructions" => r"^\s+[a-z]",
    "zmm" => r"zmm",
    "ymm" => r"ymm",
    "vpcompress" => r"vpcompress|vcompress",
    "vpermd/ps/q" => r"vperm(d|ps|q|pd|i2|t2)\b",
    "vpminsq/maxsq" => r"vp(min|max)[su]q",
    "vminp/vmaxp" => r"v(min|max)p[sd]",
    "vpminsd/ud" => r"vp(min|max)[su]d",
    "vpcmpgtq" => r"vpcmpgtq|vpcmpq",
    "blendv" => r"blendv",
    "kmask" => r"\bk[0-7]\b",
    "stack loads" => r"^\s+v?mov\w*\s+[^,]+,\s*\w+ ptr \[(rsp|rbp)",
    "stack stores" => r"^\s+v?mov\w*\s+\w+ ptr \[(rsp|rbp)",
    "calls" => r"^\s+call",
]

function asm(f, types)
    io = IOBuffer()
    code_native(io, f, types; syntax = :intel, debuginfo = :none)
    return String(take!(io))
end

function main()
    rows = Tuple{String,String,Vector{Int}}[]
    U = Q.UNROLL
    kernels = Any[]
    for T in (Int32, UInt32, Float32, Int64, Float64)
        P = Ptr{T}
        push!(kernels, ("partition", T, Q.partition!, (P, Int, T, P, Val{U}, Val{false}, Val{false})))
        push!(kernels, ("partition_prefetch", T, Q.partition!, (P, Int, T, P, Val{U}, Val{true}, Val{false})))
        push!(kernels, ("partition3", T, Q.partition3!, (P, Int, T, P, Val{U}, Val{false})))
        push!(kernels, ("small_sort", T, Q.small_sort!, (P, Int, P)))
    end
    push!(kernels, ("leaf_sort", Int64, Q.leaf_sort!, (Ptr{Int64}, Int, Ptr{Int64}, Int64, Int64)))
    # The 32-row and 16-row networks alone.
    for T in (Int32, Float32, Int64, Float64)
        N = Q.lanes(T)
        for R in (16, 32)
            push!(kernels, ("sortnet$R", T, Q.sortnet, (NTuple{R,Vec{N,T}},)))
        end
    end
    for (name, T, f, types) in kernels
        s = try
            asm(f, types)
        catch err
            @warn "no code for $name $T" err
            continue
        end
        write(joinpath(ASM, "$(name)_$T.s"), s)
        lines = split(s, '\n')
        push!(rows, (name, string(T), [count(l -> occursin(re, l), lines) for (_, re) in PATTERNS]))
    end
    open(joinpath(OUT, "codegen.csv"), "w") do io
        println(io, "kernel,type,", join(first.(PATTERNS), ","))
        for (k, t, c) in rows
            println(io, k, ",", t, ",", join(c, ","))
        end
    end
    @printf("%-20s %-8s", "kernel", "type")
    for (h, _) in PATTERNS
        @printf(" %8s", first(h, 8))
    end
    println()
    for (k, t, c) in rows
        @printf("%-20s %-8s", k, t)
        foreach(x -> @printf(" %8d", x), c)
        println()
    end
    println("asm in ", ASM)
end

main()
