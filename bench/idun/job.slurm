#!/bin/bash
# Slurm job for the AVX-512 measurements on Idun (docs/AVX512.md).
#   cd sortbench/SIMDQuickSort.jl && sbatch bench/idun/job.slurm
# Set the account and partition; ask for a whole node so the thread scaling
# and the memory bandwidth are not shared with other jobs.
#SBATCH --job-name=simdsort-avx512
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=bench/idun/results/slurm-%j.out
# An AVX-512 node, e.g. Intel Xeon (Skylake-SP or later) or AMD Zen 4. Use the
# feature names `sinfo -o "%N %f"` shows for such nodes, e.g.:
##SBATCH --constraint=<feature>

set -u
mkdir -p bench/idun/results
# Julia: a module if available, else juliaup in $HOME (curl -fsSL https://install.julialang.org | sh)
module load Julia/1.13 2>/dev/null || true
module load GCC CMake 2>/dev/null || true
export PATH="$HOME/.juliaup/bin:$PATH"
# Keep the package depot off the small home quota if /cluster/work exists.
[ -d "/cluster/work/$USER" ] && export JULIA_DEPOT_PATH="/cluster/work/$USER/julia_depot${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
grep -q avx512f /proc/cpuinfo || echo "WARNING: no AVX-512 on $(hostname); results are still useful as a second x86 machine"
export NT=${SLURM_CPUS_ON_NODE:-$(nproc)}
bash bench/idun/run.sh "$@"
