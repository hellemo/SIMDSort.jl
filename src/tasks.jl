# Shared helpers for running work on tasks.
#
#   chunks(n, nc)                    1:n in nc contiguous ranges of near-equal size
#   partasks(n, ntasks, minlen)      the number of chunks: 1 below minlen keys
#   tforeach(f, n, ntasks; minlen)   f(r) for each chunk r of 1:n, one task each
#   tforeach_index(f, nc)            f(c) for c in 1:nc, one task each
#   tmapreduce(f, op, n, ntasks; minlen)   op over f(r) for the chunks of 1:n
#   balanced_groups(sizes, target)   consecutive items in groups of ~target size
#   run_limited(f, groups, ntasks)   f(g) for each group, at most ntasks at once
#
# The parallel quicksort (pqsort!) and ppartition! spawn their own tasks: see
# src/parallel.jl.

# Below this many keys, work runs on the calling task.
const PAR_MIN_KEYS = 1 << 14

chunks(n::Integer, nc::Integer) = collect(Iterators.partition(1:Int(n), max(1, cld(Int(n), max(1, Int(nc))))))

partasks(n::Integer, ntasks::Integer, minlen::Integer) = (ntasks <= 1 || n < minlen) ? 1 : Int(ntasks)

function tforeach(f::F, n::Integer, ntasks::Integer; minlen::Integer = PAR_MIN_KEYS) where {F}
    nc = partasks(n, ntasks, minlen)
    if nc == 1
        f(1:Int(n))
    else
        @sync for r in chunks(n, nc)
            Threads.@spawn f(r)
        end
    end
    return nothing
end

function tforeach_index(f::F, nc::Integer) where {F}
    if nc <= 1
        f(1)
    else
        @sync for c in 1:nc
            Threads.@spawn f(c)
        end
    end
    return nothing
end

# The first chunk runs on the calling task and fixes the result type, so the
# reduction is type stable for type-stable f.
function tmapreduce(f::F, op::O, n::Integer, ntasks::Integer; minlen::Integer = PAR_MIN_KEYS) where {F,O}
    nc = partasks(n, ntasks, minlen)
    nc == 1 && return f(1:Int(n))
    cs = chunks(n, nc)
    tasks = [Threads.@spawn(f(cs[c])) for c in 2:length(cs)]
    acc = f(cs[1])
    for t in tasks
        acc = op(acc, fetch(t)::typeof(acc))
    end
    return acc
end

# Consecutive items (sizes[i] for item i) in groups whose total size stays at
# or below target, except that each group has at least one item.
function balanced_groups(sizes::AbstractVector{<:Integer}, target::Integer)
    groups = UnitRange{Int}[]
    b = 1
    m = length(sizes)
    while b <= m
        g, sz = b, 0
        while g <= m && (sz == 0 || sz + sizes[g] <= target)
            sz += sizes[g]
            g += 1
        end
        push!(groups, b:g-1)
        b = g
    end
    return groups
end

# At most `ntasks` tasks of one sort run at a time, as `simdsort!` promises.
# Julia's scheduler runs a spawned task on any of its threads, so without this
# `ntasks = 2` in a 56-thread Julia used all 56 cores. A task takes a slot of
# the semaphore before it starts working; tasks waiting for a slot hold no
# thread.
spawn_limited(f, sem::Base.Semaphore) = Threads.@spawn begin
    Base.acquire(sem)
    enter_task!()
    try
        f()
    finally
        leave_task!()
        Base.release(sem)
    end
end

function run_limited(f::F, groups, ntasks::Integer) where {F}
    sem = Base.Semaphore(max(1, Int(ntasks)))
    tasks = [spawn_limited(() -> f(g), sem) for g in groups]
    foreach(wait, tasks)
    return nothing
end
