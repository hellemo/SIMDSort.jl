# Optional instrumentation: phase timers and event counters, for finding weak
# spots on machines where interactive profiling is impractical (bench/idun/).
# Off by default; then every probe costs one load and branch, and probes sit only
# at coarse steps (per pass or per partition call, never per key).
#
#   r = SIMDQuickSort.instrumented() do
#       simdsort!(v; ntasks = 8)
#   end
#   SIMDQuickSort.print_report(r)
#
# Times from concurrent tasks add up (they are CPU time, not wall time).

const INSTR = Ref(false)
# One accumulator per thread, so probes in concurrent tasks never wait on a lock
# (a shared lock inflated nested phase times at 8 threads). Recording cannot
# yield, so the thread id stays valid for the whole update.
struct Acc
    ns::Dict{Symbol,Int}
    calls::Dict{Symbol,Int}
    counters::Dict{Symbol,Int}
end
Acc() = Acc(Dict{Symbol,Int}(), Dict{Symbol,Int}(), Dict{Symbol,Int}())
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
    INSTR[] = true
    t0 = time_ns()
    result = try
        f()
    finally
        INSTR[] = false
    end
    wall = (time_ns() - t0) / 1e9
    ns = mergewith(+, Dict{Symbol,Int}(), (a.ns for a in ACCS)...)
    calls = mergewith(+, Dict{Symbol,Int}(), (a.calls for a in ACCS)...)
    phases = Dict(k => (ns[k] / 1e9, calls[k]) for k in keys(ns))
    counters = Dict{Symbol,Int}()
    for a in ACCS, (k, v) in a.counters
        # Counters named *_max hold maxima; all others are sums.
        counters[k] = endswith(string(k), "_max") ? max(get(counters, k, v), v) : get(counters, k, 0) + v
    end
    return (; result, wall, phases, counters)
end

"""
    print_report([io], r)

Print the phases (sorted by time) and counters of an `instrumented` result.
"""
print_report(r) = print_report(stdout, r)
function print_report(io::IO, r)
    println(io, "wall time: ", round(r.wall; digits = 4), " s")
    println(io, "phases (task time, s; calls):")
    for (k, (t, c)) in sort(collect(r.phases); by = x -> -x[2][1])
        println(io, "  ", rpad(k, 28), lpad(string(round(t; digits = 4)), 10), "  ", c)
    end
    println(io, "counters:")
    for (k, v) in sort(collect(r.counters); by = x -> string(x[1]))
        println(io, "  ", rpad(k, 28), lpad(string(v), 14))
    end
end
