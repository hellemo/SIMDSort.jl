# Load a copy of a SIMDSort source tree as its own module, so that several
# versions run interleaved in one process (separate processes drift by several
# percent). Used by tune.jl (one tuning constant changed per variant) and by
# bench/ab.jl (a baseline source tree against the working tree).
#
#   M = load_variant(srcdir, "SIMDSort_base")                       # a module
#   M = load_variant(srcdir, "V1"; file = "sort.jl", name = "UNROLL", value = "4")
#
# Call at top level: methods defined while a function runs are too new for it.
# Extensions are not loaded.

function load_variant(srcdir::AbstractString, mod::AbstractString;
                      file = "", name = "", value = "")
    dir = mktempdir()
    cp(srcdir, joinpath(dir, "src"))
    main = joinpath(dir, "src", "SIMDSort.jl")
    s = read(main, String)
    occursin(r"^module SIMDSort$"m, s) || error("no `module SIMDSort` in $main")
    write(main, replace(s, r"^module SIMDSort$"m => "module $mod"))
    if !isempty(file)
        f = joinpath(dir, "src", file)
        t = read(f, String)
        re = Regex("^const $name = .*\$", "m")
        occursin(re, t) || error("no `const $name` in $file")
        write(f, replace(t, re => "const $name = $value"))
    end
    include(main)
    return Base.invokelatest(getfield, Main, Symbol(mod))
end
