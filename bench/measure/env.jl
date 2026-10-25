# What the Julia side sees of the machine, and which code paths SIMDSort
# compiles for it.
#
#   julia --project=bench/measure bench/measure/env.jl

using SIMDSort, SIMD, InteractiveUtils, Printf
const Q = SIMDSort

versioninfo(verbose = false)
println("LLVM: ", Base.libllvm_version, ", CPU name: ", Sys.CPU_NAME,
        ", CPU threads: ", Sys.CPU_THREADS, ", Julia threads: ", Threads.nthreads(),
        " (interactive: ", Threads.nthreads(:interactive), ")")
@printf("memory: %.1f GiB total, %.1f GiB free\n", Sys.total_memory() / 2^30, Sys.free_memory() / 2^30)

println("\nSIMD.Targets features:")
for f in (:sse42, :avx, :avx2, :fma, :bmi2, :avx512f, :avx512vl, :avx512dq, :avx512bw,
          :avx512cd, :avx512vbmi, :avx512vbmi2, :avx512vpopcntdq, :avx512bf16, :avx512fp16, :avx10_1_512)
    v = try
        SIMD.Targets.has_feature(f)
    catch
        "?"
    end
    @printf("  %-16s %s\n", f, v)
end

println("\nSIMDSort code paths:")
println("  has_avx = ", Q.has_avx(), ", has_avx512 = ", Q.has_avx512())
println("  Int64 leaves as Float64 offsets (F5): ", Q.has_avx() && !Q.has_avx512())
println("  32-row networks split in halves (non-64-bit-int types): ", !SIMD.Targets.has_feature(:avx512f))
for T in (Int32, UInt32, Float32, Int64, UInt64, Float64)
    @printf("  %-8s lanes %d (%d-bit vectors), base case up to %d keys\n", T, Q.lanes(T), Q.vecbits(), Q.basecase_max(T))
end
println("  constants: UNROLL=$(Q.UNROLL) PFDIST=$(Q.PFDIST) PF_MIN_BYTES=$(Q.PF_MIN_BYTES) ",
        "PAR_MIN=$(Q.PAR_MIN) PPART_MIN=$(Q.PPART_MIN) GRAIN_FACTOR=$(Q.GRAIN_FACTOR) ",
        "BIGPIVOT_MINN=$(Q.BIGPIVOT_MINN) MAXROWS=$(Q.MAXROWS) MSD_BUCKETS=$(Q.MSD_BUCKETS) ",
        "DIST_MINWORK=2^$(round(Int, log2(Q.DIST_MINWORK))) HASH_MAXK=$(Q.HASH_MAXK)")

println("\ncaches (sysfs, cpu0):")
for d in sort(filter(startswith("index"), readdir("/sys/devices/system/cpu/cpu0/cache"; join = false)))
    p = joinpath("/sys/devices/system/cpu/cpu0/cache", d)
    rd(f) = isfile(joinpath(p, f)) ? strip(read(joinpath(p, f), String)) : "?"
    println("  L", rd("level"), " ", rpad(rd("type"), 12), lpad(rd("size"), 8),
            "  shared by CPUs ", rd("shared_cpu_list"))
end

# Memory bandwidth, roughly: one thread and all threads, copy of 1 GiB.
function bw(nt)
    n = 2^27
    a = rand(n); b = similar(a)
    chunks = collect(Iterators.partition(1:n, cld(n, nt)))
    f() = @sync for c in chunks
        Threads.@spawn @views copyto!(b[c], a[c])
    end
    f()
    t = minimum(@elapsed(f()) for _ in 1:5)
    return 2 * 8n / t / 1e9
end
@printf("\ncopy bandwidth (read+write): 1 thread %.1f GB/s, %d threads %.1f GB/s\n",
        bw(1), Threads.nthreads(), bw(Threads.nthreads()))
