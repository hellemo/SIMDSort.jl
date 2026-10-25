# Tuning data from other machines: one instrumented benchmark run whose report
# file can be sent back (docs/src/measuring.md, "Tuning data from your
# machine"). Only the package itself is needed.

"""
    SIMDSort.tuning_report(path = "simdsort-tuning-<host>.txt"; ntasks = Threads.nthreads(),
                           rounds = 5, sizelog = 24) -> path

Run a fixed set of instrumented sorts and write what is needed to tune
SIMDSort for this machine to the text file `path`:

- the machine: CPU model, threads, caches (Linux), the SIMD features found, and
  the vector width and lanes the kernels use;
- per case (random 64- and 32-bit integers and floats, few distinct values,
  `Int128`, `String` `sortperm`): best and median wall time of `rounds` fresh
  inputs, with one task and with `ntasks`; for the primitive types with
  `ntasks`, also with each top level forced (parallel quicksort, sample-sort
  step), which is what the threshold rule `use_distribution` is fitted to;
- the phase times and counters of the median run (as `print_report`).

Takes a minute or two with the defaults (`2^sizelog` keys for the primitive
cases). Start Julia with all threads (`julia -t auto`); `ntasks = 1` measures
one thread only.

```julia
julia> using SIMDSort

julia> SIMDSort.tuning_report()
"simdsort-tuning-myhost.txt"
```
"""
function tuning_report(path::AbstractString = "simdsort-tuning-$(gethostname()).txt";
                       ntasks::Integer = Threads.nthreads(), rounds::Integer = 5, sizelog::Integer = 24)
    n = 2^Int(sizelog)
    alnum = UInt8[0x30:0x39; 0x41:0x5a; 0x61:0x7a]
    randstr(k) = String(rand(alnum, k))
    vals = rand(UInt64, 1000)
    # (name, generator, operation, primitive?)
    cases = [
        ("Int64", () -> rand(Int64, n), :sort, true),
        ("UInt64", () -> rand(UInt64, n), :sort, true),
        ("Float64 randn", () -> randn(n), :sort, true),
        ("Int32", () -> rand(Int32, n), :sort, true),
        ("Float32 randn", () -> randn(Float32, n), :sort, true),
        ("UInt64 1000 values", () -> rand(vals, n), :sort, false),
        ("Int128", () -> rand(Int128, n ÷ 16), :sort, false),
        ("String 8-20 sortperm", () -> [randstr(rand(8:20)) for _ in 1:n÷16], :perm, false),
    ]
    open(path, "w") do io
        tuning_header(io, Int(ntasks))
        println(io, "\n# times: case, ntasks, top level, keys, best s, median s")
        for (name, gen, op, primitive) in cases, nt in unique([1, Int(ntasks)])
            modes = primitive && nt > 1 ? (:auto, :off, :on) : (:auto,)
            for mode in modes
                best, med, r, len = tuning_case(gen, op, nt, mode, Int(rounds))
                label = mode === :auto ? "auto" : mode === :on ? "sample-sort" : "quicksort"
                println(io, "time, ", name, ", ", nt, ", ", label, ", ", len, ", ", best, ", ", med)
                println(io, "## ", name, ", ", nt, " tasks, ", label, " (median run)")
                print_report(io, r)
                flush(io)
            end
        end
    end
    return path
end

# Best and median wall time of `rounds` fresh inputs, the instrumented result
# of the median run, and the input length.
function tuning_case(gen, op::Symbol, nt::Int, mode::Symbol, rounds::Int)
    f(x) = op === :perm ? simdsortperm(x; ntasks = nt) : simdsort!(x; ntasks = nt)
    old = DIST_MODE[]
    DIST_MODE[] = mode
    try
        x = gen()
        f(copy(x))                                  # compile
        runs = map(1:rounds) do _
            y = gen()
            GC.gc(false)
            instrumented(() -> f(y))
        end
        sort!(runs; by = r -> r.wall)
        walls = [r.wall for r in runs]
        return minimum(walls), walls[(rounds + 1) ÷ 2], runs[(rounds + 1) ÷ 2], length(x)
    finally
        DIST_MODE[] = old
    end
end

function tuning_header(io::IO, ntasks::Int)
    println(io, "# SIMDSort tuning report")
    println(io, "date: ", Libc.strftime("%Y-%m-%d %H:%M", time()))
    println(io, "julia: ", VERSION, ", SIMDSort ", pkgversion(@__MODULE__))
    println(io, "cpu: ", Sys.cpu_info()[1].model, ", ", Sys.CPU_THREADS, " logical CPUs, ",
            Threads.nthreads(), " Julia threads, ntasks ", ntasks)
    println(io, "os: ", Sys.KERNEL, " ", Sys.MACHINE)
    feats = [f for f in (:sse4_2, :avx, :avx2, :bmi2, :avx512f, :avx512vl, :avx512dq, :avx512bw,
                         :avx512vbmi2, :neon, :sve) if SIMD.Targets.has_feature(f)]
    println(io, "features: ", join(feats, " "))
    println(io, "target: ", nameof(typeof(target())), ", ", vecbits(), "-bit vectors, lanes ",
            join(["$T $(lanes(T))" for T in (Int32, Int64)], ", "))
    if isdir("/sys/devices/system/cpu/cpu0/cache")
        for d in sort(filter(startswith("index"), readdir("/sys/devices/system/cpu/cpu0/cache")))
            p = joinpath("/sys/devices/system/cpu/cpu0/cache", d)
            rd(f) = isfile(joinpath(p, f)) ? strip(read(joinpath(p, f), String)) : "?"
            println(io, "cache: L", rd("level"), " ", rd("type"), " ", rd("size"))
        end
    end
end
