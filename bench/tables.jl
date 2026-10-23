# Markdown result tables for the README, from the CSV files of bench.jl and
# types.jl, rendered with PrettyTables. In every row the fastest single-threaded
# and the fastest multi-threaded result are bold, each within its group.
#
#   julia --project=bench bench/tables.jl
#
# The tables replace the text between `<!-- table:NAME -->` and
# `<!-- /table:NAME -->` in README.md, and are also printed.

using CSV, DataFrames, PrettyTables, Printf

const RESULTS = joinpath(@__DIR__, "results")
const README = joinpath(@__DIR__, "..", "README.md")

# rows × columns table of best times; `groups` lists the column indices of each
# group within which the minimum is bold.
function render(rowlabels, collabels, times, groups; firstlabel = "case")
    data = Matrix{Any}(undef, length(rowlabels), length(collabels) + 1)
    for (i, l) in enumerate(rowlabels)
        data[i, 1] = l
        for j in eachindex(collabels)
            data[i, j+1] = times[i, j]
        end
    end
    isbest(i, j) = j > 1 && (g = findfirst(G -> (j - 1) in G, groups)) !== nothing &&
                   !ismissing(data[i, j]) &&
                   data[i, j] == minimum(skipmissing(data[i, k+1] for k in groups[g]))
    fmt(v, i, j) = v isa Real ? @sprintf("%.3f", v) : ismissing(v) ? "–" : v
    return pretty_table(String, data; backend = :markdown,
                        column_labels = [firstlabel; collabels],
                        alignment = [:l; fill(:r, length(collabels))],
                        highlighters = [MarkdownHighlighter(
                            (d, i, j) -> isbest(i, j), PrettyTables.MarkdownStyle(bold = true))],
                        formatters = [fmt])
end

function main_table()
    df = CSV.read(joinpath(RESULTS, "bench.csv"), DataFrame)
    nt = maximum(df.nthreads)
    cols = [("Base.sort!", 1, "Base 1 thr"), ("ChapelSort.jl psort!", 1, "ChapelSort 1 thr"),
            ("SIMDQuickSort.jl", 1, "SIMDQuickSort 1 thr"), ("x86-simd-sort", 1, "x86-simd-sort 1 thr"),
            ("Highway vqsort", 1, "vqsort 1 thr"),
            ("ChapelSort.jl psort!", nt, "ChapelSort $nt thr"), ("SIMDQuickSort.jl", nt, "SIMDQuickSort $nt thr"),
            ("x86-simd-sort", nt, "x86-simd-sort $nt thr")]
    # The case order of bench.jl (rows rerun with BENCH_ONLY end up last in the CSV).
    order = ["UInt64, 2^27", "Int64, 2^26", "Float64 (randn), 2^26", "UInt32, 2^26", "Int32, 2^26",
             "Float32 (randn), 2^26", "UInt64 in 0:1000, 2^26", "UInt64, 1000 random values, 2^26",
             "UInt64 sorted, 2^26", "UInt64 reversed, 2^26", "Int32, 2^20"]
    cases = sort(unique(df.case); by = c -> something(findfirst(==(c), order), length(order) + 1))
    times = Matrix{Union{Missing,Float64}}(missing, length(cases), length(cols))
    for (i, c) in enumerate(cases), (j, (m, t, _)) in enumerate(cols)
        r = filter(x -> x.case == c && x.method == m && x.nthreads == t, df)
        nrow(r) == 1 && (times[i, j] = r.best[1])
    end
    return render(cases, last.(cols), times, [1:5, 6:8])
end

function types_table()
    df = CSV.read(joinpath(RESULTS, "types.csv"), DataFrame)
    nt = maximum(df.nthreads)
    cols = [("Base", 1, "Base 1 thr"), ("ChapelSort.jl", 1, "ChapelSort 1 thr"),
            ("SIMDQuickSort.jl", 1, "SIMDQuickSort 1 thr"),
            ("ChapelSort.jl", nt, "ChapelSort $nt thr"), ("SIMDQuickSort.jl", nt, "SIMDQuickSort $nt thr")]
    keys_ = unique(zip(df.case, df.op))
    labels = ["$c, $op" for (c, op) in keys_]
    times = Matrix{Union{Missing,Float64}}(missing, length(keys_), length(cols))
    for (i, (c, op)) in enumerate(keys_), (j, (m, t, _)) in enumerate(cols)
        r = filter(x -> x.case == c && x.op == op && x.method == m && x.nthreads == t, df)
        nrow(r) == 1 && (times[i, j] = r.best[1])
    end
    return render(labels, last.(cols), times, [1:3, 4:5]; firstlabel = "case, operation")
end

function splice_table(text, name, table)
    a, b = "<!-- table:$name -->", "<!-- /table:$name -->"
    i, j = findfirst(a, text), findfirst(b, text)
    (i === nothing || j === nothing) && return text
    return text[1:last(i)] * "\n" * table * text[first(j):end]
end

function main()
    tm, tt = main_table(), types_table()
    println(tm, "\n", tt)
    text = read(README, String)
    text = splice_table(text, "main", tm)
    text = splice_table(text, "types", tt)
    write(README, text)
    println("updated ", README)
end

main()
