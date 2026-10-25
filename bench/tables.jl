# Result tables from the CSV files of bench.jl and types.jl, rendered with
# PrettyTables: Markdown for the README (in every row the fastest
# single-threaded and the fastest multi-threaded result bold, each within its
# group) and HTML with highlighters for the documentation (the fastest result
# bold on a green tint, results within 10% of it on a lighter one).
#
#   julia --project=bench bench/tables.jl
#   XEON_SCALING=<run dir>/scaling.csv julia --project=bench bench/tables.jl
#
# The tables replace the text between `<!-- table:NAME -->` and
# `<!-- /table:NAME -->` in README.md and docs/src/benchmarks.md. With
# XEON_SCALING (bench/measure/scaling.jl on the AVX-512 server), the docs also
# get its table at one and all threads.

using CSV, DataFrames, PrettyTables, Printf

const RESULTS = joinpath(@__DIR__, "results")
const README = joinpath(@__DIR__, "..", "README.md")
const DOCS = joinpath(@__DIR__, "..", "docs", "src", "benchmarks.md")

# HTML highlight styles (translucent, so they read on light and dark themes).
const BEST = ["font-weight" => "bold", "background-color" => "rgba(46, 160, 67, 0.25)"]
const NEAR = ["background-color" => "rgba(46, 160, 67, 0.09)"]

# rows × columns table of best times; `groups` lists the column indices of each
# group within which the minimum is highlighted. backend: :markdown or :html.
function render(rowlabels, collabels, times, groups; firstlabel = "case", backend = :markdown)
    data = Matrix{Any}(undef, length(rowlabels), length(collabels) + 1)
    for (i, l) in enumerate(rowlabels)
        data[i, 1] = l
        for j in eachindex(collabels)
            data[i, j+1] = times[i, j]
        end
    end
    # The cell's time relative to the fastest of its group (NaN for none).
    function rel(i, j)
        j > 1 && !ismissing(data[i, j]) || return NaN
        g = findfirst(G -> (j - 1) in G, groups)
        g === nothing && return NaN
        return data[i, j] / minimum(skipmissing(data[i, k+1] for k in groups[g]))
    end
    fmt(v, i, j) = v isa Real ? @sprintf("%.3f", v) : ismissing(v) ? "–" : v
    labels = [firstlabel; collabels]
    alignment = [:l; fill(:r, length(collabels))]
    if backend === :html
        return pretty_table(String, data; backend = :html, column_labels = labels, alignment,
                            highlighters = [HtmlHighlighter((d, i, j) -> rel(i, j) == 1, BEST),
                                            HtmlHighlighter((d, i, j) -> 1 < rel(i, j) <= 1.1, NEAR)],
                            formatters = [fmt], stand_alone = false)
    end
    return pretty_table(String, data; backend = :markdown, column_labels = labels, alignment,
                        highlighters = [MarkdownHighlighter(
                            (d, i, j) -> rel(i, j) == 1, PrettyTables.MarkdownStyle(bold = true))],
                        formatters = [fmt])
end

function main_table(backend = :markdown)
    df = CSV.read(joinpath(RESULTS, "bench.csv"), DataFrame)
    nt = maximum(df.nthreads)
    cols = [("Base.sort!", 1, "Base 1 thr"), ("ChapelSort.jl psort!", 1, "ChapelSort 1 thr"),
            ("SIMDSort.jl", 1, "SIMDSort 1 thr"), ("x86-simd-sort", 1, "x86-simd-sort 1 thr"),
            ("Highway vqsort", 1, "vqsort 1 thr"),
            ("ChapelSort.jl psort!", nt, "ChapelSort $nt thr"), ("SIMDSort.jl", nt, "SIMDSort $nt thr"),
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
    return render(cases, last.(cols), times, [1:5, 6:8]; backend)
end

function types_table(backend = :markdown)
    df = CSV.read(joinpath(RESULTS, "types.csv"), DataFrame)
    nt = maximum(df.nthreads)
    cols = [("Base", 1, "Base 1 thr"), ("ChapelSort.jl", 1, "ChapelSort 1 thr"),
            ("SIMDSort.jl", 1, "SIMDSort 1 thr"),
            ("ChapelSort.jl", nt, "ChapelSort $nt thr"), ("SIMDSort.jl", nt, "SIMDSort $nt thr")]
    keys_ = unique(zip(df.case, df.op))
    labels = ["$c, $op" for (c, op) in keys_]
    times = Matrix{Union{Missing,Float64}}(missing, length(keys_), length(cols))
    for (i, (c, op)) in enumerate(keys_), (j, (m, t, _)) in enumerate(cols)
        r = filter(x -> x.case == c && x.op == op && x.method == m && x.nthreads == t, df)
        nrow(r) == 1 && (times[i, j] = r.best[1])
    end
    return render(labels, last.(cols), times, [1:3, 4:5]; firstlabel = "case, operation", backend)
end

# The AVX-512 server's scaling.csv at one and all threads: SIMDSort (default),
# ChapelSort and x86-simd-sort.
function xeon_table(path)
    df = CSV.read(path, DataFrame)
    nt = maximum(df.nthreads)
    methods = ["SIMDSort", "ChapelSort", "x86-simd-sort"]
    cols = [(m, t, "$m $t thr") for t in (1, nt) for m in methods]
    cases = unique(df.case)
    times = Matrix{Union{Missing,Float64}}(missing, length(cases), length(cols))
    for (i, c) in enumerate(cases), (j, (m, t, _)) in enumerate(cols)
        r = filter(x -> x.case == c && x.method == m && x.nthreads == t, df)
        nrow(r) == 1 && (times[i, j] = r.best[1])
    end
    return render(cases, last.(cols), times, [1:3, 4:6]; backend = :html)
end

# An HTML table as a raw block of a Documenter page (scrolls when wide).
docs_block(table) = "```@raw html\n<div style=\"overflow-x: auto;\">\n" * table * "</div>\n```\n"

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
    text = read(DOCS, String)
    text = splice_table(text, "main", docs_block(main_table(:html)))
    text = splice_table(text, "types", docs_block(types_table(:html)))
    xeon = get(ENV, "XEON_SCALING", "")
    isempty(xeon) || (text = splice_table(text, "xeon", docs_block(xeon_table(xeon))))
    write(DOCS, text)
    println("updated ", DOCS)
end

main()
