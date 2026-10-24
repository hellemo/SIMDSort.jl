#!/usr/bin/env bash
# Build libsimdsort (a trimmed shared library with C entry points) with JuliaC:
#   lib/build.sh [trim mode]      (default: safe; "no" disables trimming)
# Output: lib/build/ (lib/libsimdsort.so plus the bundled Julia runtime).
# Needs Julia 1.12+, JuliaC (installed into a tools environment on first use)
# and a C compiler. The library is built for the CPU it is built on.
set -euo pipefail
cd "$(dirname "$0")"
TRIM=${1:-safe}
# The package decides its vector code paths when compiled: build for this CPU.
export JULIA_CPU_TARGET=${JULIA_CPU_TARGET:-native}
# The library's runtime threads are fixed at build time (JuliaC defaults
# libraries to 1; JULIA_NUM_THREADS does not apply): SIMDSORT_THREADS=N (default: the CPU count).
THREADS=${SIMDSORT_THREADS:-$(nproc)}
TOOLS=${JULIAC_ENV:-@juliac}
julia --project="$TOOLS" -e 'using Pkg; haskey(Pkg.project().dependencies, "JuliaC") || Pkg.add("JuliaC")' >/dev/null
julia --project=. -e 'using Pkg; Pkg.instantiate()' >/dev/null
rm -rf build
julia --project="$TOOLS" -e 'using JuliaC; JuliaC.main(ARGS)' -- \
    --output-lib libsimdsort --compile-ccallable --trim="$TRIM" --experimental \
    --jl-option threads="$THREADS" \
    --bundle build --project=. --verbose src/SIMDSortC.jl
ls -la build/lib | head
