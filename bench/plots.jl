# Charts for the README and docs/DESIGN.md, from the CSV files written by
# bench.jl, types.jl and crossover.jl:
#
#   julia --project=bench bench/plots.jl
#
# Writes SVG files to docs/.

using AlgebraOfGraphics, CairoMakie, CSV, DataFrames, Printf

const RESULTS = joinpath(@__DIR__, "results")
const OUT = joinpath(@__DIR__, "..", "docs")

const SURFACE = colorant"#fcfcfb"
const INK = colorant"#0b0b0b"
const INK2 = colorant"#52514e"
const GRID = colorant"#e6e5e0"
const BLUE, ORANGE, AQUA, YELLOW, MAGENTA =
    colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100", colorant"#e87ba4"
const GRAY = colorant"#8c8a83"

# One color per method, the same in every chart.
const METHODS = ["SIMDQuickSort.jl" => BLUE, "x86-simd-sort" => ORANGE,
                 "Highway vqsort" => AQUA, "ChapelSort.jl psort!" => YELLOW,
                 "ChapelSort.jl" => YELLOW, "Base.sort!" => GRAY, "Base" => GRAY]
color_of(m) = Dict(METHODS)[m]

const AXIS = (; backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
              xgridwidth = 1, ygridwidth = 1, leftspinecolor = INK2,
              bottomspinecolor = INK2, topspinevisible = false,
              rightspinevisible = false, xtickcolor = INK2, ytickcolor = INK2,
              xticklabelcolor = INK2, yticklabelcolor = INK2, xlabelcolor = INK,
              ylabelcolor = INK, titlecolor = INK, titlealign = :left,
              titlesize = 17, subtitlecolor = INK2)
const LEGEND = (; position = :bottom, framevisible = false, labelcolor = INK, tellheight = true)
const MACHINE = "Intel Core Ultra 7 268V (4P + 4E, AVX2), Julia 1.13"

# Throughput by key type for a set of (method, threads) series.
function throughput_plot(df, cases, series, title, subtitle; size = (760, 560))
    d = filter(r -> r.case in cases && (r.method, r.nthreads) in last.(series), df)
    names = Dict((m, t) => l for (l, (m, t)) in series)
    d.series = [names[(r.method, r.nthreads)] for r in eachrow(d)]
    d.melems = d.n ./ d.best ./ 1e6
    labels = first.(series)
    colors = [color_of(m) for (_, (m, _)) in series]
    plt = data(d) *
          mapping(:case => sorter(cases...) => "", :melems => "Throughput (M keys/s)";
                  color = :series => sorter(labels...) => "",
                  dodge = :series => sorter(labels...) => "") *
          visual(BarPlot; direction = :x, gap = 0.25, dodge_gap = 0.04,
                 bar_labels = :y, label_formatter = v -> string(round(Int, v)),
                 label_color = INK, label_size = 10, label_offset = 3)
    return draw(plt, scales(Color = (; palette = colors));
                axis = (; AXIS..., ygridvisible = false, title, subtitle,
                        limits = (0, maximum(d.melems) * 1.15, nothing, nothing)),
                legend = (; LEGEND..., nbanks = 2),
                figure = (; size, backgroundcolor = SURFACE, fontsize = 13))
end

const RANDOM = ["UInt64, 2^27", "Int64, 2^26", "Float64 (randn), 2^26", "UInt32, 2^26",
                "Int32, 2^26", "Float32 (randn), 2^26"]
const SPECIAL = ["UInt64 in 0:1000, 2^26", "UInt64, 1000 random values, 2^26",
                 "UInt64 sorted, 2^26", "UInt64 reversed, 2^26"]

function bench_plots()
    df = CSV.read(joinpath(RESULTS, "bench.csv"), DataFrame)
    one = ["SIMDQuickSort.jl" => ("SIMDQuickSort.jl", 1), "x86-simd-sort" => ("x86-simd-sort", 1),
           "Highway vqsort" => ("Highway vqsort", 1), "ChapelSort.jl (radix)" => ("ChapelSort.jl psort!", 1),
           "Base.sort!" => ("Base.sort!", 1)]
    save(joinpath(OUT, "random1.svg"),
         throughput_plot(df, RANDOM, one, "Random keys, one thread",
                         "Higher is better · best of 9 interleaved rounds, fresh input · $MACHINE"))
    nt = maximum(df.nthreads)
    many = ["SIMDQuickSort.jl" => ("SIMDQuickSort.jl", nt), "x86-simd-sort" => ("x86-simd-sort", nt),
            "ChapelSort.jl (radix)" => ("ChapelSort.jl psort!", nt)]
    save(joinpath(OUT, "random$nt.svg"),
         throughput_plot(df, RANDOM, many, "Random keys, $nt threads",
                         "Higher is better · best of 9 interleaved rounds, fresh input · $MACHINE";
                         size = (760, 500)))
    special = [one; "SIMDQuickSort.jl, $nt thr" => ("SIMDQuickSort.jl", nt)]
    save(joinpath(OUT, "special.svg"),
         throughput_plot(df, SPECIAL, one, "Duplicates and presorted inputs, one thread",
                         "Higher is better (log scale would hide nothing here: note the range) · $MACHINE";
                         size = (760, 460)))
end

# Speedup over Base for every type in types.csv (sortperm and sort).
function types_plot(op)
    df = CSV.read(joinpath(RESULTS, "types.csv"), DataFrame)
    d = filter(r -> r.op == op, df)
    base = Dict(r.case => r.best for r in eachrow(d) if r.method == "Base")
    d = filter(r -> r.method != "Base", d)
    d.speedup = [base[r.case] / r.best for r in eachrow(d)]
    nt = maximum(d.nthreads)
    d.series = [string(r.method, ", ", r.nthreads == 1 ? "1 thread" : "$nt threads") for r in eachrow(d)]
    labels = ["SIMDQuickSort.jl, 1 thread", "SIMDQuickSort.jl, $nt threads",
              "ChapelSort.jl, 1 thread", "ChapelSort.jl, $nt threads"]
    colors = [BLUE, colorant"#9cc2ef", YELLOW, colorant"#f5d27a"]
    cases = unique(d.case)
    plt = data(d) *
          mapping(:case => sorter(reverse(cases)...) => "", :speedup => "Speedup over Base (×, log scale)";
                  color = :series => sorter(labels...) => "",
                  dodge = :series => sorter(labels...) => "") *
          visual(BarPlot; direction = :x, gap = 0.2, dodge_gap = 0.03, fillto = 0.1,
                 bar_labels = :y, label_formatter = v -> @sprintf("%.1f×", v),
                 label_color = INK, label_size = 9, label_offset = 2)
    ref = mapping([1.0]) * visual(VLines; color = INK2, linestyle = :dash, linewidth = 1)
    fig = draw(plt + ref, scales(Color = (; palette = colors));
               axis = (; AXIS..., ygridvisible = false, xscale = log10,
                       title = "$op for other types: speedup over Base",
                       subtitle = "Higher is better · dashed line: Base · $MACHINE",
                       limits = (0.3, maximum(d.speedup) * 2.5, nothing, nothing)),
               legend = (; LEGEND..., nbanks = 2),
               figure = (; size = (780, 60 + 52 * length(cases)), backgroundcolor = SURFACE,
                         fontsize = 13))
    save(joinpath(OUT, "types_$op.svg"), fig)
end

# Radix (ChapelSort) / quicksort (SIMDQuickSort) time by size and tasks.
function crossover_plot()
    df = CSV.read(joinpath(RESULTS, "crossover.csv"), DataFrame)
    q = filter(r -> r.method == "quicksort", df)
    r = filter(r -> r.method == "radix", df)
    d = innerjoin(select(q, :case, :ntasks, :n, :best => :tq), select(r, :case, :ntasks, :n, :best => :tr);
                  on = [:case, :ntasks, :n])
    d.ratio = d.tr ./ d.tq
    d.log2n = round.(Int, log2.(d.n))
    cases = unique(d.case)
    tasks = sort(unique(d.ntasks))
    plt = data(d) *
          mapping(:log2n => "log₂ n", :ratio => "Radix time / quicksort time";
                  color = :ntasks => nonnumeric => "Tasks", layout = :case => sorter(cases...)) *
          (visual(Lines; linewidth = 2) + visual(Scatter; markersize = 8))
    ref = mapping([1.0]) * visual(HLines; color = INK2, linestyle = :dash, linewidth = 1)
    fig = draw(plt + ref, scales(Color = (; palette = [GRAY, AQUA, ORANGE, BLUE][1:length(tasks)]));
               axis = (; AXIS..., titlesize = 13, yscale = log2, yticks = [0.5, 1, 2, 4],
                       limits = (nothing, nothing, 0.6, 6)),
               legend = (; LEGEND..., nbanks = 1, orientation = :horizontal),
               figure = (; size = (860, 560), backgroundcolor = SURFACE, fontsize = 12,
                         title = "When does ChapelSort's radix sort beat the quicksort?",
                         subtitle = "Below the dashed line the radix sort is faster · $MACHINE"))
    save(joinpath(OUT, "crossover.svg"), fig)
end

function main()
    mkpath(OUT)
    bench_plots()
    types_plot("sortperm")
    types_plot("sort")
    crossover_plot()
    println("wrote ", join(filter(f -> endswith(f, ".svg"), readdir(OUT)), ", "), " to ", OUT)
end

main()
