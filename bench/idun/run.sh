#!/usr/bin/env bash
# The whole AVX-512 measurement in one go (docs/AVX512.md). Run from the
# SIMDSort.jl directory on a compute node, via job.slurm or directly:
#
#   bench/idun/run.sh [steps...]       (default: all steps, in this order)
#
# Steps: env setup test build codegen avx512 instrument profile scaling tune
#        bench types crossover
# Env:   NT       threads (default: all CPUs of the job)
#        SORTBENCH_DEPOT  Julia package depot (default /cluster/work/$USER/julia_depot,
#                 else ~/sortbench_depot; never ~/.julia)
#        OUT      results directory (default bench/idun/results/<host>-<time>)
#        ROUNDS   rounds for bench/types (default 9/3), QUICK=1 for a short run
# Every step logs to $OUT/<step>.log and the script continues after a failed
# step; $OUT/steps.tsv records status and duration of each.
set -uo pipefail
cd "$(dirname "$0")/../.."
HERE=bench/idun
NT=${NT:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$HERE/results/$(hostname -s)-$STAMP}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export OUT NT BENCH_RESULTS="$OUT" JULIA_NUM_THREADS=$NT
export SIMDSORT_LIB=${SIMDSORT_LIB:-$(cd ../ChapelSort.jl/bench/simd 2>/dev/null && pwd)/libsimdsort.so}
# A dedicated package depot, the same for setup and the job: not ~/.julia,
# whose extra (e.g. private, ssh-only) registries Pkg would try to update,
# prompting for credentials. On /cluster/work if it exists (home quota).
# SORTBENCH_DEPOT overrides it.
if [ -n "${SORTBENCH_DEPOT:-}" ]; then
    export JULIA_DEPOT_PATH="$SORTBENCH_DEPOT"
elif [ -d "/cluster/work/$USER" ]; then
    export JULIA_DEPOT_PATH="/cluster/work/$USER/julia_depot"
else
    export JULIA_DEPOT_PATH="$HOME/sortbench_depot"
fi
mkdir -p "$JULIA_DEPOT_PATH"
echo "julia: $(command -v julia || echo 'NOT FOUND'), depot: $JULIA_DEPOT_PATH, NT=$NT"
QUICK=${QUICK:-0}
if [ "$QUICK" = 1 ]; then R1=3; R2=1; XMAX=26; else R1=${ROUNDS:-9}; R2=${ROUNDS:-3}; XMAX=28; fi
JL="julia --startup-file=no"
ENVDIR=bench/idun          # lean environment: no plotting packages
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(env setup test build codegen avx512 instrument profile scaling tune bench types crossover)

step_env() {
    { echo "date: $(date -Is)"; echo "host: $(hostname)"; uname -a
      echo "slurm job: ${SLURM_JOB_ID:-none}, cpus: ${SLURM_CPUS_ON_NODE:-?}, NT=$NT"
      echo; lscpu; echo; lscpu -C 2>/dev/null; echo; free -g
      echo; numactl -H 2>/dev/null || true
      echo; cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null || true
      echo; grep -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
      echo; $JL --version; gcc --version | head -1; cmake --version | head -1
      for r in . ../ChapelSort.jl ../SIMD.jl; do echo "$r: $(git -C $r rev-parse --short HEAD 2>/dev/null || echo 'no git')"; done
    } > "$OUT/system.txt" 2>&1
    $JL --project=$ENVDIR $HERE/env.jl > "$OUT/julia_env.txt" 2>&1
}
step_setup() {
    $JL --version
    # stdin from /dev/null: a credentials prompt fails at once instead of hanging.
    $JL --project=$ENVDIR -e 'using Pkg; println("depot: ", DEPOT_PATH[1]);
        println("registries: ", join([r.name for r in Pkg.Registry.reachable_registries()], ", "));
        Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' < /dev/null
}
# The test suite in the lean environment (it has all test dependencies and
# uses the synced SIMD.jl fork; Pkg.test() would resolve the package's own
# environment and fetch the fork from GitHub).
step_test()  { JULIA_NUM_THREADS=8 $JL --project=$ENVDIR -e 'include("test/runtests.jl")'; }
step_build() { ../ChapelSort.jl/bench/simd/build.sh; }
step_codegen()    { $JL --project=$ENVDIR $HERE/codegen.jl; }
step_avx512()     { $JL -t $NT --project=$ENVDIR bench/avx512.jl all; }
step_instrument() { $JL -t $NT --project=$ENVDIR $HERE/instrument.jl; }
step_profile()    { $JL --project=$ENVDIR bench/profile.jl; }
step_scaling()    { $JL -t $NT --project=$ENVDIR $HERE/scaling.jl; }
step_tune()       { $JL -t $NT --project=$ENVDIR $HERE/tune.jl; }
step_bench()      { $JL -t $NT --project=$ENVDIR bench/bench.jl $R1; }
step_types()      { $JL -t $NT --project=$ENVDIR bench/types.jl $R2; }
step_crossover()  { $JL -t $NT --project=$ENVDIR bench/crossover.jl 3 $XMAX; }

printf "step\tstatus\tseconds\n" >> "$OUT/steps.tsv"
for s in "${STEPS[@]}"; do
    echo "== $s ($(date +%T)), log: $OUT/$s.log"
    t0=$(date +%s)
    if [ "$s" = setup ] || [ "$s" = build ]; then
        "step_$s" 2>&1 | tee "$OUT/$s.log"          # interactive steps: show progress
        st=${PIPESTATUS[0]}
    else
        "step_$s" > "$OUT/$s.log" 2>&1
        st=$?
    fi
    printf "%s\t%s\t%s\n" "$s" "$st" "$(( $(date +%s) - t0 ))" >> "$OUT/steps.tsv"
    [ $st -ne 0 ] && echo "   step $s failed ($st), see $OUT/$s.log"
done
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT (and $OUT.tar.gz)"
column -t "$OUT/steps.tsv" 2>/dev/null || cat "$OUT/steps.tsv"
