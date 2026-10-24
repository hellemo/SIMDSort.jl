# Build the documentation:   julia --project=docs docs/make.jl
# (instantiate once: julia --project=docs -e 'using Pkg; Pkg.instantiate()')
using Documenter, SIMDSort

DocMeta.setdocmeta!(SIMDSort, :DocTestSetup, :(using SIMDSort); recursive = true)

makedocs(;
    sitename = "SIMDSort.jl",
    modules = [SIMDSort],
    authors = "Lars Hellemo",
    repo = Remotes.GitHub("hellemo", "SIMDSort.jl"),
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                             edit_link = "main", size_threshold = 400_000),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "API" => "api.md",
        "Benchmarks" => "benchmarks.md",
        "Design" => "design.md",
        "AVX-512" => "avx512.md",
        "Measuring" => "measuring.md",
        "C library" => "clib.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(; repo = "github.com/hellemo/SIMDSort.jl.git", devbranch = "main", push_preview = false)
