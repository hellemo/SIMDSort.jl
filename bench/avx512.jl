# Measurements for AVX-512 machines (e.g. an NTNU Idun node); see docs/AVX512.md.
#
#   julia -t <n> --project=bench bench/avx512.jl [part|net|sort|all]
#
# 1. part: our partition step (vpartition: permutation table + vpermd, also on
#    AVX-512) against a Bramas-style step with native compress (vcompress for
#    the low part, vstorec, i.e. vpcompress to memory, for the high part).
# 2. net:  the 32-row base-case network as one network (AVX-512 default: 32
#    registers) against the two-halves version used on AVX2 (ns per call,
#    including the loads and stores of the register matrix).
# 3. sort: full sorts against x86-simd-sort and Highway vqsort (rebuild the shim
#    on the node: ChapelSort.jl/bench/simd/build.sh uses -march=native).

using SIMDSort, SIMD, Libdl, Printf, Random, Statistics
const Q = SIMDSort
const what = isempty(ARGS) ? "all" : ARGS[1]

println("Julia $VERSION, $(Threads.nthreads()) threads")
for f in (:avx2, :avx512f, :avx512vl, :avx512dq, :avx512bw, :avx512vbmi2)
    @printf("  %-12s %s\n", f, SIMD.Targets.has_feature(f))
end
try
    println("  CPU: ", strip(read(pipeline(`lscpu`, `grep "Model name"`), String)))
catch
end

# --- 1. partition kernel -------------------------------------------------------

# The partition step with AVX-512's compress instead of the permutation table.
Q.eval(quote
    @inline function part1_compress(x::Vec{N,T}, pv::Vec{N,T}, p::Ptr{T}, lw::Int, rw::Int) where {N,T}
        m = x <= pv
        c = count(m)
        vst(vcompress(x, m), p, lw)                       # low part, full width
        vstorec(x, p + (rw - (N - c)) * sizeof(T), !m)    # high part, exactly N - c keys
        return lw + c, rw - (N - c)
    end
end)

# One pass of the kernel over n keys from p into q (type-stable, f static).
Q.eval(quote
    function part_loop!(f::F, p::Ptr{T}, q::Ptr{T}, n::Int, ::Val{N}) where {F,T,N}
        pv = Vec{N,T}(zero(T)); lw, rw = 0, n + 2N
        for i in 0:N:n-N
            lw, rw = f(vld(Val(N), p, i), pv, q, lw, rw)
        end
        return lw
    end
end)

# ns/key of the partition step alone (out of place, so the input is unchanged
# between repetitions; random keys, pivot 0, i.e. unpredictable sides).
function part_cost(::Type{T}, n, kernel) where {T}
    N = Q.lanes(T)
    a = T <: AbstractFloat ? T.(randn(n)) : rand(T, n)
    tmp = Vector{T}(undef, n + 2N)
    reps = max(1, 2^24 ÷ n)
    f = kernel === :compress ? Q.part1_compress : Q.part1
    run() = GC.@preserve a tmp Base.invokelatest(Q.part_loop!, f, pointer(a), pointer(tmp), n, Val(N))
    run()
    best = minimum(1:5) do _
        @elapsed for _ in 1:reps
            run()
        end
    end
    return best / reps / n * 1e9
end

if what in ("part", "all")
    println("\n1. Partition kernel, ns/key (permutation table / native compress)")
    for T in (Int32, Float32, Int64, Float64), k in (14, 24)
        try
            @printf("  %-8s 2^%d: %.3f / %.3f\n", T, k, part_cost(T, 2^k, :table), part_cost(T, 2^k, :compress))
        catch e
            println("  $T 2^$k: ", sprint(showerror, e))
        end
    end
end

# --- 2. base-case network ------------------------------------------------------

# The two variants of the 32-row network: one network (zigzag merges, what
# sortnet uses without the split) and two 16-row halves plus a merge.
Q.eval(quote
    @generated net_single(vs::NTuple{R,Vec{N,T}}) where {R,N,T} =
        Expr(:block, Expr(:meta, :inline), zigzag_body(R, N))
    @generated net_halves(vs::NTuple{R,Vec{N,T}}) where {R,N,T} =
        Expr(:block, Expr(:meta, :inline), halves_body(R, N))
    # Load, sort, store M register matrices; type-stable (R, N static).
    function net_loop!(f::F, p::Ptr{T}, M::Int, ::Val{R}, ::Val{N}) where {F,T,R,N}
        for j in 0:M-1
            q = p + j * R * N * sizeof(T)
            vs = ntuple(r -> vload(Vec{N,T}, q + (r - 1) * N * sizeof(T)), Val(R))
            out = f(vs)
            for r in 1:R
                vstore(out[r], q + (r - 1) * N * sizeof(T))
            end
        end
    end
end)

function net_cost(::Type{T}, f) where {T}
    N = Q.lanes(T); R = 32; M = 4096
    a = T <: AbstractFloat ? T.(randn(R * N * M)) : rand(T, R * N * M)
    run(b) = GC.@preserve b Base.invokelatest(Q.net_loop!, f, pointer(b), M, Val(R), Val(N))
    run(copy(a))
    # Fresh (unsorted) input every round: a sorted matrix would be cheaper.
    return minimum(_ -> (b = copy(a); @elapsed run(b)), 1:7) / M * 1e9
end

if what in ("net", "all")
    println("\n2. 32-row base-case network, ns per call (single / two halves)")
    for T in (Int32, Float32, Int64, Float64)
        @printf("  %-8s %3d keys: %.0f / %.0f\n", T, 32 * Q.lanes(T), net_cost(T, Q.net_single), net_cost(T, Q.net_halves))
    end
end

# --- 3. full sorts --------------------------------------------------------------

# (Functions, so that the ccall argument types are static.)
xss_call(f, v::Vector{T}, nt) where {T} = ccall(f, Cvoid, (Ptr{T}, Csize_t, Cint), v, length(v), nt)
vq_call(f, v::Vector{T}) where {T} = ccall(f, Cvoid, (Ptr{T}, Csize_t), v, length(v))

if what in ("sort", "all")
    lib = get(ENV, "SIMDSORT_LIB", joinpath(@__DIR__, "..", "..", "ChapelSort.jl", "bench", "simd", "libsimdsort.so"))
    have = isfile(lib)
    L = have ? dlopen(lib) : nothing
    have && println("\n3. Full sorts, 2^26 keys, best of 5 interleaved rounds (fresh input); ",
                    unsafe_string(ccall(dlsym(L, :simdsort_info), Cstring, ())))
    have || println("\n3. Full sorts (no shim at $lib: SIMDSort only)")
    sfx = Dict(Int32 => "i32", Float32 => "f32", Int64 => "i64", Float64 => "f64", UInt64 => "u64", UInt32 => "u32")
    nt = Threads.nthreads()
    for T in (Int32, UInt32, Float32, Int64, UInt64, Float64)
        fs = Pair{String,Function}["SIMDSort 1" => v -> simdsort!(v), "SIMDSort $nt" => v -> simdsort!(v; ntasks = nt)]
        if have
            xs = dlsym(L, Symbol("xss_sort_", sfx[T])); vq = dlsym(L, Symbol("vqsort_", sfx[T]))
            push!(fs, "x86-simd-sort 1" => v -> xss_call(xs, v, 1),
                      "x86-simd-sort $nt" => v -> xss_call(xs, v, nt),
                      "vqsort 1" => v -> vq_call(vq, v))
        end
        rng = Xoshiro(1); n = 2^26
        gen() = T <: AbstractFloat ? T.(randn(rng, n)) : rand(rng, T, n)
        for (_, f) in fs; f(gen()[1:10^5]); end
        ts = Dict(k => Float64[] for (k, _) in fs)
        for _ in 1:5
            x = gen()
            for (k, f) in shuffle(rng, fs)
                y = copy(x); push!(ts[k], @elapsed f(y)); issorted(y) || error("$k: not sorted")
            end
        end
        println("  ", rpad(T, 8), join([@sprintf("%s %.3f", k, minimum(ts[k])) for (k, _) in fs], " | "))
    end
end
