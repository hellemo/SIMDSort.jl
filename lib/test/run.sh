#!/usr/bin/env bash
# Compile test.c against the bundle in lib/build and run it:
#   lib/test/run.sh [n] [nthreads]
set -euo pipefail
cd "$(dirname "$0")/.."
cc -O2 -Iinclude test/test.c -Lbuild/lib -lsimdsort -Wl,-rpath,"$PWD/build/lib" -o build/simdsort_test
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-${2:-1}} build/simdsort_test "$@"
