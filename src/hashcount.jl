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

@inline hashslot(x::U, mask::Int) where {U<:Unsigned} =
    ((UInt64(x) * 0x9e3779b97f4a7c15) >> 32) % Int & mask

struct CountTable{U}
    keys::Vector{U}
    counts::Vector{Int}     # 0 marks an empty slot
end
CountTable{U}(cap::Int) where {U} = CountTable{U}(Vector{U}(undef, cap), zeros(Int, cap))

# Count the bit patterns of k[r] into t; false if t got more than half full.
function count_into!(t::CountTable{U}, k::AbstractVector{U}, r::UnitRange{Int}) where {U}
    keys, counts = t.keys, t.counts
    mask = length(keys) - 1
    limit = length(keys) ÷ 2
    used = 0
    @inbounds for i in r
        x = k[i]
        s = hashslot(x, mask)
        while true
            c = counts[s+1]
            if c == 0
                keys[s+1] = x
                counts[s+1] = 1
                used += 1
                used > limit && return false
                break
            elseif keys[s+1] == x
                counts[s+1] = c + 1
                break
            end
            s = (s + 1) & mask
        end
    end
    return true
end

"""
    hash_count_sort!(v, rev, ntasks) -> Bool

Sort `v` by counting its distinct values in a hash table if a sample suggests
there are few of them (at most `HASH_MAXK`); return `false`, leaving `v`
unchanged, otherwise.
"""
function hash_count_sort!(v::Vector{T}, rev::Bool, ntasks::Int) where {T<:SortTypes}
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
    for t in tables, s in eachindex(t.counts)
        c = t.counts[s]
        c == 0 && continue
        total[t.keys[s]] = get(total, t.keys[s], 0) + c
    end
    vals = T[reinterpret(T, x) for x in keys(total)]
    cnts = [total[reinterpret(U, x)] for x in vals]
    p = sortperm(vals; rev)
    # (New names: the closure below must not capture reassigned variables.)
    svals, scnts = vals[p], cnts[p]
    # Write the runs, by up to ntasks tasks over equal slices of v.
    count!(:hash_sorts)
    count!(:hash_distinct_keys, length(svals))
    starts = cumsum([1; scnts[1:end-1]])
    tforeach(n, ntasks) do r
        j = searchsortedlast(starts, first(r))
        i = first(r)
        @inbounds while i <= last(r)
            e = min(last(r), starts[j] + scnts[j] - 1)
            x = svals[j]
            for q in i:e
                v[q] = x
            end
            i = e + 1
            j += 1
        end
    end
    return true
end
