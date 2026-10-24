# Optional instrumentation: phase timers and event counters, for finding weak
# spots on machines where interactive profiling is impractical (bench/measure/).
# Off by default; then every probe costs one load and branch, and probes sit only
# at coarse steps (per pass or per partition call, never per key).
#
#   r = SIMDSort.instrumented() do
#       simdsort!(v; ntasks = 8)
#   end
#   SIMDSort.print_report(r)
#
# Times from concurrent tasks add up (they are CPU time, not wall time). The
# pivot, leaf_sort and partition_2e* times of small ranges are sampled
# estimates (SAMPLE_BELOW).

const INSTR = Ref(false)
# One accumulator per thread, so probes in concurrent tasks never wait on a lock
# (a shared lock inflated nested phase times at 8 threads). Recording cannot
# yield, so the thread id stays valid for the whole update.
#
# The per-partition probes in qsort! use fixed slots instead: hot phases
# (HOT_PHASES) and partition statistics by size class floor(log2 n), in
# arrays. Ranges below SAMPLE_BELOW keys are timed in one call of SAMPLE_EVERY
# (a time_ns pair costs as much as a small partition) and the times scaled up;
# calls and keys are always counted exactly.
const HOT_PHASES = (:pivot, :leaf_sort)
const SAMPLE_BELOW = 1 << 12
const SAMPLE_EVERY = 16
mutable struct Acc
    ns::Dict{Symbol,Int}
    calls::Dict{Symbol,Int}
    counters::Dict{Symbol,Int}
    hot_ns::Vector{Int}       # per HOT_PHASES entry
    hot_calls::Vector{Int}
    part_ns::Vector{Int}      # per size class 0:63 (index + 1)
    part_calls::Vector{Int}
    part_keys::Vector{Int}
    part3::Vector{Int}        # three-way partitions per size class
    leaf_calls::Int
    leaf_keys::Int
    tick::Int
end
Acc() = Acc(Dict{Symbol,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}(),
            zeros(Int, length(HOT_PHASES)), zeros(Int, length(HOT_PHASES)),
            zeros(Int, 64), zeros(Int, 64), zeros(Int, 64), zeros(Int, 64), 0, 0, 0)
const ACCS = Acc[]

@inline acc() = @inbounds ACCS[Threads.threadid()]

function record_phase!(name::Symbol, ns::Integer)
    a = acc()
    a.ns[name] = get(a.ns, name, 0) + ns
    a.calls[name] = get(a.calls, name, 0) + 1
    return nothing
end

function count!(name::Symbol, n::Integer = 1)
    INSTR[] || return nothing
    c = acc().counters
    c[name] = get(c, name, 0) + n
    return nothing
end

# Whether to time this call of a probe on a range of n keys (see SAMPLE_BELOW).
@inline function sample_now(a::Acc, n::Int)
    n >= SAMPLE_BELOW && return 1
    a.tick += 1
    return a.tick % SAMPLE_EVERY == 0 ? SAMPLE_EVERY : 0
end

"""
    @hotphase i n expr

Time `expr` in hot slot `i` (an index into HOT_PHASES) for a range of n keys,
sampled below SAMPLE_BELOW keys, when instrumentation is on.
"""
macro hotphase(i, n, ex)
    return quote
        if INSTR[]
            local a = acc()
            local w = sample_now(a, $(esc(n)))
            @inbounds a.hot_calls[$(esc(i))] += 1
            if w > 0
                local t0 = time_ns()
                local r = $(esc(ex))
                @inbounds a.hot_ns[$(esc(i))] += w * Int(time_ns() - t0)
                r
            else
                $(esc(ex))
            end
        else
            $(esc(ex))
        end
    end
end

"""
    @partphase n dup expr

Time and count the partition `expr` of n keys by size class.
"""
macro partphase(n, dup, ex)
    return quote
        if INSTR[]
            local a = acc()
            local k = sizeclass($(esc(n))) + 1
            @inbounds a.part_calls[k] += 1
            @inbounds a.part_keys[k] += $(esc(n))
            $(esc(dup)) && @inbounds(a.part3[k] += 1)
            local w = sample_now(a, $(esc(n)))
            if w > 0
                local t0 = time_ns()
                local r = $(esc(ex))
                @inbounds a.part_ns[k] += w * Int(time_ns() - t0)
                r
            else
                $(esc(ex))
            end
        else
            $(esc(ex))
        end
    end
end

@inline function count_leaf!(n::Int)
    INSTR[] || return nothing
    a = acc()
    a.leaf_calls += 1
    a.leaf_keys += n
    return nothing
end

# Tasks of a sort that hold a slot of its semaphore (src/parallel.jl), for the
# counter concurrent_tasks_max: checks that ntasks bounds the parallelism.
const ACTIVE_TASKS = Threads.Atomic{Int}(0)
@inline function enter_task!()
    INSTR[] || return nothing
    countmax!(:concurrent_tasks_max, Threads.atomic_add!(ACTIVE_TASKS, 1) + 1)
    return nothing
end
@inline leave_task!() = (INSTR[] && Threads.atomic_sub!(ACTIVE_TASKS, 1); nothing)

# Maxima rather than sums (e.g. the largest bucket).
function countmax!(name::Symbol, n::Integer)
    INSTR[] || return nothing
    c = acc().counters
    c[name] = max(get(c, name, typemin(Int)), n)
    return nothing
end

"""
    @phase name expr

Time `expr` under `name` when instrumentation is on.
"""
macro phase(name, ex)
    return quote
        if INSTR[]
            local t0 = time_ns()
            local r = $(esc(ex))
            record_phase!($(esc(name)), time_ns() - t0)
            r
        else
            $(esc(ex))
        end
    end
end

# Size class of a range for the counters: floor(log2(n)).
sizeclass(n::Integer) = 8sizeof(Int) - 1 - leading_zeros(max(n, 1))

"""
    instrumented(f) -> NamedTuple

Run `f()` with instrumentation on; return `(; result, wall, phases, counters)`:
`phases[name] = (seconds, calls)`, `counters[name] = count`.
"""
function instrumented(f)
    empty!(ACCS)
    append!(ACCS, [Acc() for _ in 1:Threads.maxthreadid()])
    ACTIVE_TASKS[] = 0
    INSTR[] = true
    g0 = Base.gc_num()
    t0 = time_ns()
    result = try
        f()
    finally
        INSTR[] = false
    end
    wall = (time_ns() - t0) / 1e9
    # Garbage collection during the run (stops all threads; with many threads
    # and many small allocations it can dominate).
    gd = Base.GC_Diff(Base.gc_num(), g0)
    ns = mergewith(+, Dict{Symbol,Int}(), (a.ns for a in ACCS)...)
    calls = mergewith(+, Dict{Symbol,Int}(), (a.calls for a in ACCS)...)
    # The fixed slots, under the same names as before.
    for (i, name) in enumerate(HOT_PHASES)
        c = sum(a -> a.hot_calls[i], ACCS)
        c > 0 || continue
        ns[name] = sum(a -> a.hot_ns[i], ACCS)
        calls[name] = c
    end
    for k in 1:64
        c = sum(a -> a.part_calls[k], ACCS)
        c > 0 || continue
        ns[Symbol(:partition_2e, k - 1)] = sum(a -> a.part_ns[k], ACCS)
        calls[Symbol(:partition_2e, k - 1)] = c
    end
    phases = Dict(k => (ns[k] / 1e9, calls[k]) for k in keys(ns))
    counters = Dict{Symbol,Int}()
    pc = sum(a -> sum(a.part_calls), ACCS)
    if pc > 0
        counters[:partition_calls] = pc
        counters[:threeway_partitions] = sum(a -> sum(a.part3), ACCS)
        for k in 1:64
            c = sum(a -> a.part_keys[k], ACCS)
            c > 0 && (counters[Symbol(:partition_keys_2e, k - 1)] = c)
        end
    end
    lc = sum(a -> a.leaf_calls, ACCS)
    if lc > 0
        counters[:leaf_calls] = lc
        counters[:leaf_keys] = sum(a -> a.leaf_keys, ACCS)
    end
    for a in ACCS, (k, v) in a.counters
        # Counters named *_max hold maxima; all others are sums.
        counters[k] = endswith(string(k), "_max") ? max(get(counters, k, v), v) : get(counters, k, 0) + v
    end
    counters[:gc_ns] = gd.total_time
    counters[:gc_pauses] = gd.pause
    counters[:alloc_bytes] = gd.allocd
    counters[:alloc_count] = Base.gc_alloc_count(gd)
    return (; result, wall, phases, counters)
end

"""
    print_report([io], r)

Print the phases (sorted by time) and counters of an `instrumented` result.
"""
print_report(r) = print_report(stdout, r)
function print_report(io::IO, r)
    println(io, "wall time: ", round(r.wall; digits = 4), " s; GC ",
            round(get(r.counters, :gc_ns, 0) / 1e9; digits = 4), " s in ", get(r.counters, :gc_pauses, 0),
            " pauses; ", round(get(r.counters, :alloc_bytes, 0) / 2^20; digits = 1), " MiB in ",
            get(r.counters, :alloc_count, 0), " allocations")
    println(io, "phases (task time, s; calls):")
    for (k, (t, c)) in sort(collect(r.phases); by = x -> -x[2][1])
        println(io, "  ", rpad(k, 28), lpad(string(round(t; digits = 4)), 10), "  ", c)
    end
    println(io, "counters:")
    for (k, v) in sort(collect(r.counters); by = x -> string(x[1]))
        println(io, "  ", rpad(k, 28), lpad(string(v), 14))
    end
end
