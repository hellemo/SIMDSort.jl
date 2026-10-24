#!/usr/bin/env bash
# Copy the results of Idun runs back (see docs/AVX512.md):
#
#   bench/idun/fetch_results.sh [remote-dir]       (default: sortbench)
#
# Env: IDUN_HOST (ssh alias, default idunpw). Copies
# <remote-dir>/<this repo>/bench/idun/results/ (one directory and tarball per
# run, plus the slurm logs) into bench/idun/results/ here; local runs with
# other names are kept. The results are git-ignored. Uses rsync if installed
# on both ends, else tar over ssh.
set -euo pipefail
HOST=${IDUN_HOST:-idunpw}
DEST=${1:-sortbench}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$DEST/$(basename "$REPO")/bench/idun/results"
LOCAL="$REPO/bench/idun/results"
mkdir -p "$LOCAL"
if command -v rsync > /dev/null && ssh "$HOST" "command -v rsync" > /dev/null; then
    rsync -az --info=stats1 "$HOST:$SRC/" "$LOCAL/"
else
    echo "(rsync not available on both ends: using tar over ssh)"
    ssh "$HOST" "tar -C '$SRC' -czf - ." | tar -C "$LOCAL" -xzf -
fi
echo "runs in $LOCAL:"
ls -1 "$LOCAL"
for d in "$LOCAL"/*/; do
    [ -f "$d/steps.tsv" ] && { echo "-- $(basename "$d")"; cat "$d/steps.tsv"; }
done
exit 0
