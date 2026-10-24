#!/usr/bin/env bash
# The whole AVX-512 measurement in one go (docs/AVX512.md). Run from the
# SIMDSort.jl directory on a compute node, via job.slurm or directly:
#
#   bench/idun/run.sh [steps...]       (default: all steps, in this order)
#
# Steps: env setup test build codegen avx512 instrument profile scaling tune
#        bench types crossover
# Env:   NT       threads (default: all CPUs of the job)
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
QUICK=${QUICK:-0}
if [ "$QUICK" = 1 ]; then R1=3; R2=1; XMAX=26; else R1=${ROUNDS:-9}; R2=${ROUNDS:-3}; XMAX=28; fi
JL="julia --startup-file=no"
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
    $JL --project=. $HERE/env.jl > "$OUT/julia_env.txt" 2>&1
}
step_setup() { $JL --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; }
step_test()  { $JL --project=. -e 'using Pkg; Pkg.test()'; }
step_build() { ../ChapelSort.jl/bench/simd/build.sh; }
step_codegen()    { $JL --project=bench $HERE/codegen.jl; }
step_avx512()     { $JL -t $NT --project=bench bench/avx512.jl all; }
step_instrument() { $JL -t $NT --project=bench $HERE/instrument.jl; }
step_profile()    { $JL --project=bench bench/profile.jl; }
step_scaling()    { $JL -t $NT --project=bench $HERE/scaling.jl; }
step_tune()       { $JL -t $NT --project=bench $HERE/tune.jl; }
step_bench()      { $JL -t $NT --project=bench bench/bench.jl $R1; }
step_types()      { $JL -t $NT --project=bench bench/types.jl $R2; }
step_crossover()  { $JL -t $NT --project=bench bench/crossover.jl 3 $XMAX; }

printf "step\tstatus\tseconds\n" >> "$OUT/steps.tsv"
for s in "${STEPS[@]}"; do
    echo "== $s ($(date +%T)), log: $OUT/$s.log"
    t0=$(date +%s)
    "step_$s" > "$OUT/$s.log" 2>&1
    st=$?
    printf "%s\t%s\t%s\n" "$s" "$st" "$(( $(date +%s) - t0 ))" >> "$OUT/steps.tsv"
    [ $st -ne 0 ] && echo "   step $s failed ($st), see $OUT/$s.log"
done
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT (and $OUT.tar.gz)"
column -t "$OUT/steps.tsv" 2>/dev/null || cat "$OUT/steps.tsv"
