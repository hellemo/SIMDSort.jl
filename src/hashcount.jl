# Few distinct values over a wide range: count them in a hash table, after
# [CAFS] (V. S. Shlyk, "CAFS: A Cache-Aware Frequency Sort for Low-Cardinality
# Integer Data on x86-64", arXiv:2605.25040, 2026; docs/DESIGN.md §1). CAFS probes
# a cache-line bucket with AVX2 compares; this version probes linearly. The counting sort in sort.jl covers small key *ranges*; this covers few
# distinct *values* anywhere in the range.
#
# 1. A strided sample of 1024 keys estimates the number of distinct values with
#    the Chao1 estimator K̂ = u + f1²/(2(f2 + 1)) (u distinct values in the
#    sample, f1 of them seen once, f2 twice), as CAFS does.
# 2. If K̂ is small, one pass counts the keys in an open-addressing table (linear
#    probing, one table per task, merged afterwards) sized to stay in cache. If
#    a table gets more than half full, the attempt is abandoned.
# 3. The distinct keys are sorted, and each key's run is written.
#
# Keys are compared by their bits (for floats after NaNs are moved out), which
# is exact: equal bits mean equal under `isless`, and -0.0 and 0.0 stay distinct.

const HASH_SAMPLE = 1024
const HASH_MAXK = 1 << 13        # use hashing if the estimate is at most this
const HASH_MINN = 1 << 16

function chao1(s::Vector)
    sort!(s)
    u = f1 = f2 = 0
    i = 1
    while i <= length(s)
        j = i
        while j < length(s) && s[j+1] == s[i]
            j += 1
        end
        c = j - i + 1
        u += 1
        f1 += c == 1
        f2 += c == 2
        i = j + 1
    end
    return u + f1^2 / (2 * (f2 + 1))
end

# The hash key of a table key: its bits, or for a Symbol its address (symbols
# are interned, so equal names are the same object, and never freed).
@inline hashkey(x::Unsigned) = UInt64(x)
@inline hashkey(x::Symbol) = UInt64(UInt(pointer_from_objref(x)))
@inline hashslot(x, mask::Int) = ((hashkey(x) * 0x9e3779b97f4a7c15) >> 32) % Int & mask

# An open-addressing table (linear probing; capacity a power of two) from keys
# to counts; keys are compared with ===. Key and count share one slot, so a
# probe touches one cache line.
struct CountTable{K}
    slots::Vector{Tuple{K,Int}}     # (key, count); count 0 marks an empty slot
end
CountTable{K}(cap::Int) where {K} = CountTable{K}(fill((emptykey(K), 0), cap))
emptykey(::Type{U}) where {U<:Unsigned} = zero(U)
emptykey(::Type{Symbol}) = Symbol("")

capacity(t::CountTable) = length(t.slots)

# Count the keys k[r] into t; false if t got more than half full.
function count_into!(t::CountTable{K}, k::AbstractVector{K}, r::UnitRange{Int}) where {K}
    slots = t.slots
    mask = length(slots) - 1
    limit = length(slots) ÷ 2
    used = 0
    @inbounds for i in r
        x = k[i]
        s = hashslot(x, mask)
        while true
            y, c = slots[s+1]
            if c == 0
                slots[s+1] = (x, 1)
                used += 1
                used > limit && return false
                break
            elseif y === x
                slots[s+1] = (x, c + 1)
                break
            end
            s = (s + 1) & mask
        end
    end
    return true
end

# The used slots of a table, as (keys, counts).
function compact(t::CountTable{K}) where {K}
    used = filter(e -> e[2] != 0, t.slots)
    return (first.(used), last.(used))
end

# Add c to the count of key x; true if x was new. The table must have a free
# slot.
function add_count!(t::CountTable{K}, x::K, c::Int) where {K}
    slots = t.slots
    mask = length(slots) - 1
    s = hashslot(x, mask)
    @inbounds while true
        y, d = slots[s+1]
        if d == 0
            slots[s+1] = (x, c)
            return true
        elseif y === x
            slots[s+1] = (x, d + c)
            return false
        end
        s = (s + 1) & mask
    end
end

"""
    hash_count_sort!(v, order, ntasks) -> Bool

Sort `v` by counting its distinct values in a hash table if a sample suggests
there are few of them (at most `HASH_MAXK`); return `false`, leaving `v`
unchanged, otherwise.
"""
function hash_count_sort!(v::AbstractVector{T}, o::Ord, ntasks::Int) where {T<:SortTypes}
    n = length(v)
    n < HASH_MINN && return false
    U = unsigned(signedkey(T))
    k = reinterpret(U, v)
    est = chao1([k[i] for i in 1:(n ÷ HASH_SAMPLE):n])
    est > HASH_MAXK && return false
    count!(:hash_attempts)
    cap = max(1 << 10, nextpow(2, ceil(Int, 4 * est)))
    cs = chunks(n, ntasks)
    tables = [CountTable{U}(cap) for _ in cs]
    ok = Vector{Bool}(undef, length(cs))
    tforeach_index(length(cs)) do i
        ok[i] = count_into!(tables[i], k, cs[i])
    end
    all(ok) || (count!(:hash_aborts); return false)
    # Merge the tables (few entries each).
    total = Dict{U,Int}()
    for t in tables, (x, c) in t.slots
        c == 0 && continue
        total[x] = get(total, x, 0) + c
    end
    vals = T[reinterpret(T, x) for x in keys(total)]
    cnts = [total[reinterpret(U, x)] for x in vals]
    p = sortperm(vals; rev = isrev(o))
    # (New names: the closure below must not capture reassigned variables.)
    svals, scnts = vals[p], cnts[p]
    # Write the runs, by up to ntasks tasks over equal slices of v.
    count!(:hash_sorts)
    count!(:hash_distinct_keys, length(svals))
    write_runs!(v, svals, scnts, ntasks)
    return true
end

# v = vals[1] repeated cnts[1] times, then vals[2] cnts[2] times, ..., by up to
# ntasks tasks over equal slices of v.
function write_runs!(v::AbstractVector, vals::AbstractVector, cnts::AbstractVector{Int}, ntasks::Int)
    starts = cumsum([1; cnts[1:end-1]])
    tforeach(length(v), ntasks) do r
        j = searchsortedlast(starts, first(r))
        i = first(r)
        @inbounds while i <= last(r)
            e = min(last(r), starts[j] + cnts[j] - 1)
            x = vals[j]
            for q in i:e
                v[q] = x
            end
            i = e + 1
            j += 1
        end
    end
    return v
end
