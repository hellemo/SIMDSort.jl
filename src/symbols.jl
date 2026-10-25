# Symbols (docs/src/design.md §8). Symbols are interned: equal names are the same
# object, so a symbol is identified by its address, and vectors of symbols
# usually have few distinct values. They are counted in hash tables, as the
# numeric hash counting does (hashcount.jl, [CAFS]), and only the distinct
# names are sorted by their bytes:
#
# 1. A strided sample estimates the number of distinct symbols (Chao1); above
#    SYM_MAXK, or above n/4, the byte-key path of strings is used instead.
# 2. Each task counts its chunk in its own table (no locks).
# 3. The tables are merged in parallel: task p takes the keys whose hash slots
#    fall in its share, so every symbol is merged by exactly one task.
# 4. The distinct names are ranked by the windowed byte-key sort (perm.jl).
# 5. sort!: the runs are written in parallel; sortperm: each element's rank is
#    looked up in a read-only table, then the Int32 ranks are sorted stably.

const SYM_MAXK = 1 << 20

"""
    symbol_counts(v, ntasks) -> (syms, counts) or nothing

The distinct symbols of `v` and how often each occurs, or `nothing` if there
are too many distinct symbols for counting (then the byte path is used).
"""
function symbol_counts(v::Vector{Symbol}, ntasks::Int)
    n = length(v)
    n < 1024 && return nothing
    limit = min(n ÷ 4, SYM_MAXK)
    est = chao1([hashkey(v[i]) for i in 1:(n ÷ HASH_SAMPLE):n])
    est > limit && return nothing
    cs = chunks(n, partasks(n, ntasks, PERM_PAR_MIN))
    # A chunk has at most as many distinct symbols as keys.
    cap = max(1 << 10, nextpow(2, ceil(Int, 4 * min(est, maximum(length, cs)))))
    tables = Vector{CountTable{Symbol}}(undef, length(cs))
    ok = Vector{Bool}(undef, length(cs))
    # Each task makes its own table: in parallel, and in memory near its core.
    @phase :symbol_count tforeach_index(length(cs)) do c
        t = CountTable{Symbol}(cap)
        tables[c] = t
        ok[c] = count_into!(t, v, cs[c])
    end
    all(ok) || (count!(:symbol_count_aborts); return nothing)
    return @phase :symbol_merge merge_tables(tables, cap)
end

# The keys of all tables with their summed counts. With several tables, task p
# merges the keys whose hash slot (all tables have the same capacity) falls in
# the p-th share of the slots, so the parts are disjoint. Each task reads only
# its share of each table (see merge_part), so the merge reads every table once.
function merge_tables(tables::Vector{CountTable{K}}, cap::Int) where {K}
    np = length(tables)
    np == 1 && return compact(tables[1])
    mask = cap - 1
    parts = Vector{Tuple{Vector{K},Vector{Int}}}(undef, np)
    tforeach_index(np) do q
        slots = (q - 1) * cap ÷ np : q * cap ÷ np - 1
        parts[q] = merge_part(tables, mask, slots)
    end
    return (reduce(vcat, first.(parts)), reduce(vcat, last.(parts)))
end

# The keys of the tables whose hash slot is in `slots`, with summed counts.
# With linear probing a key sits at its hash slot or after it, with no empty
# slot in between, so the keys of `slots` are in these slots and in the run of
# occupied slots after them (wrapping around; a table is at most half full).
function merge_part(tables::Vector{CountTable{K}}, mask::Int, slots::UnitRange{Int}) where {K}
    part = CountTable{K}(64; mult = HASH_MULT2)    # its keys share their top bits
    used = 0
    for t in tables
        ts = t.slots
        s = first(slots)
        @inbounds while s <= last(slots) || ts[(s&mask)+1][2] != 0
            x, c = ts[(s&mask)+1]
            s += 1
            (c == 0 || !(hashslot(x, mask, t.mult) in slots)) && continue
            used += add_count!(part, x, c)
            if 2used > capacity(part)           # keep it at most half full
                part = rehash(part)
            end
        end
    end
    return compact(part)
end

# The same entries in a table of twice the capacity.
function rehash(t::CountTable{K}) where {K}
    u = CountTable{K}(2 * capacity(t); mult = t.mult)
    for (x, c) in zip(compact(t)...)
        add_count!(u, x, c)
    end
    return u
end

# The ranks of the distinct symbols by name (1 for the smallest).
function symbol_ranks(syms::Vector{Symbol}, ntasks::Int)
    order = GC.@preserve syms begin
        ks = ByteKeys(syms, false, ntasks)
        refine!(ks, collect(1:length(syms)), 0, ntasks, prefix_cache(ks, ntasks))
    end
    return order        # syms[order] is sorted
end

"""
    symbol_sort!(v, order, ntasks) -> Bool

Sort `v` by counting its symbols; `false` (and `v` unchanged) if there are too
many distinct symbols.
"""
function symbol_sort!(v::Vector{Symbol}, o::Ord, ntasks::Int)
    r = symbol_counts(v, ntasks)
    r === nothing && return false
    syms, cnts = r
    order = @phase :symbol_rank symbol_ranks(syms, ntasks)
    isrev(o) && reverse!(order)
    count!(:symbol_sorts)
    @phase :symbol_write write_runs!(v, syms[order], cnts[order], ntasks)
    return true
end

"""
    symbol_codes(v, ntasks) -> Vector{Int32} or nothing

The rank of each element's name among the distinct symbols of `v`, or
`nothing` if there are too many distinct symbols.
"""
function symbol_codes(v::Vector{Symbol}, ntasks::Int)
    r = symbol_counts(v, ntasks)
    r === nothing && return nothing
    syms, _ = r
    order = @phase :symbol_rank symbol_ranks(syms, ntasks)
    # A read-only table from symbol to rank, looked up by all tasks.
    t = CountTable{Symbol}(nextpow(2, 4 * length(syms)))
    for (rank, i) in enumerate(order)
        add_count!(t, syms[i], rank)
    end
    codes = Vector{Int32}(undef, length(v))
    @phase :symbol_lookup tforeach(length(v), ntasks) do rg
        lookup_ranks!(codes, t, v, rg)
    end
    return codes
end

function lookup_ranks!(codes, t::CountTable{Symbol}, v, rg)
    slots = t.slots
    mask = length(slots) - 1
    @inbounds for i in rg
        x = v[i]
        s = hashslot(x, mask, t.mult)
        while slots[s+1][1] !== x
            s = (s + 1) & mask
        end
        codes[i] = slots[s+1][2] % Int32
    end
    return nothing
end
