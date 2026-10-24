# Measuring

## Comparing versions: `bench/ab.jl`

Every refactoring and optimization step was checked by an interleaved A/B
run of two source trees in one process (separate processes drift by several
percent on a laptop):

```sh
git worktree add /tmp/base <commit>
julia -t 8 --project=bench/measure bench/ab.jl /tmp/base/src [rounds]
```

It loads both trees as module copies (`bench/measure/variants.jl`), runs
primitive, `sortperm`, string, wide-key, pair, Symbol and forced sample-sort
cases at 1 and `NT` threads with fresh inputs, preallocated buffers and
enough repetitions per timing, checks that both give identical results, and
reports best and median. A case counts as slower only if both are worse by
more than 2%, and it is rerun alone (`AB_ONLY`) before acting on it.
`AB_NEW_EDIT="file.jl:NAME=value"` compares a variant of one constant, e.g.
`keyref.jl:SAFE_KEYS=true`.

## On any machine: `bench/measure/`

`env.jl` (features and code paths), `codegen.jl` (machine code of the hot
kernels), `instrument.jl` (phase times and counters), `scaling.jl` (thread
scaling, with the sample-sort top level forced on and off) and `tune.jl`
(each tuning constant changed alone), in a lean environment without plotting
packages. See [AVX-512](avx512.md) for what each answers.

## Test modes

Besides the test suite on the current release, the test matrix runs it on
Julia 1.10, with SIMD.jl's generic lowering (`SIMD_JL_GENERIC=1`), and with
bounds-checked kernels (`CHECK_KEYS = true` in `src/keyref.jl`), which checks
that the pointer kernels never read or write outside their arrays.
