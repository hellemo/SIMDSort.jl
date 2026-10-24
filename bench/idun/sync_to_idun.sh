#!/usr/bin/env bash
# Copy everything the Idun benchmark needs to the cluster (see docs/AVX512.md):
# this repository, ChapelSort.jl (with the sources of the C++ sorts it compares
# against) and the SIMD.jl fork, side by side as in development.
#
#   bench/idun/sync_to_idun.sh [remote-dir]        (default: sortbench)
#
# Env: IDUN_HOST (ssh alias, default idunpw). Sends sources only: no .git,
# Manifests, build products, data files or local results. Nothing is deleted
# on the remote, so results there are safe. Uses rsync if it is installed on
# both ends, else tar over ssh.
set -euo pipefail
HOST=${IDUN_HOST:-idunpw}
DEST=${1:-sortbench}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(dirname "$REPO")                      # the directory holding the repos
REPOS=("$(basename "$REPO")" ChapelSort.jl SIMD.jl)
for r in "${REPOS[@]}"; do
    [ -d "$ROOT/$r" ] || { echo "missing $ROOT/$r" >&2; exit 1; }
done
EXCLUDES=(.git 'Manifest*.toml' '*.so' '*.o' '*.a' '*.csv' '*.parquet' '*.arrow' '*.jld2'
          '*.bin' '*.pdf' '*.log' '*.tar.gz' bench/results bench/idun/results
          bench/chapel/sortbench bench/simd/deps/highway-build)

ssh "$HOST" "mkdir -p '$DEST'"
if command -v rsync > /dev/null && ssh "$HOST" "command -v rsync" > /dev/null; then
    args=()
    for e in "${EXCLUDES[@]}"; do
        case $e in */*) args+=("--exclude=/*/$e") ;; *) args+=("--exclude=$e") ;; esac
    done
    (cd "$ROOT" && rsync -az --info=stats1 "${args[@]}" "${REPOS[@]}" "$HOST:$DEST/")
else
    echo "(rsync not available on both ends: using tar over ssh)"
    args=()
    for e in "${EXCLUDES[@]}"; do args+=("--exclude=$e"); done
    tar -C "$ROOT" -czf - "${args[@]}" "${REPOS[@]}" | ssh "$HOST" "tar -C '$DEST' -xzf -"
fi
echo "synced ${REPOS[*]} to $HOST:$DEST"
echo "next, on the login node (needs internet for packages and the C++ sources):"
echo "  ssh $HOST; cd $DEST/$(basename "$REPO"); bash bench/idun/run.sh setup build"
echo "then: sbatch bench/idun/job.slurm   (set account and partition in it first)"
