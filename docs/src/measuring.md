# Measuring

## Tuning data from your machine

SIMDSort's kernels and thresholds were tuned on two machines: an Intel Core
Ultra 7 268V laptop (AVX2, 4 performance and 4 efficiency cores) and a
2 × 28-core Intel Xeon Gold 6348 (Ice Lake-SP, AVX-512). Other processors may
want other choices. Data from these would help most:

| Processor | Why |
|---|---|
| AMD Zen 4 and Zen 5 (Ryzen 7000/9000, EPYC 9004/9005) | AVX-512 without the Intel clock reduction, but the compress-to-memory instruction that the 512-bit partition step uses for its right side is reported to be slow there; the 512-bit path may need a different store. |
| Intel Sapphire/Emerald/Granite Rapids | Newer AVX-512 cores with a smaller clock penalty: does the 512-bit path gain more, and do floats still lag (docs: [AVX-512](avx512.md))? |
| Intel client hybrids (Alder/Raptor/Arrow Lake, Meteor Lake) | AVX2 with performance and efficiency cores; the thread scaling and the task grain were tuned on one such part only. |
| AMD Zen 2 and Zen 3 | AVX2 with different shuffle and gather costs (the permutation-table partition step uses `vpermd`). |
| Older AVX-512 (Skylake-SP, Cascade Lake) | The strongest clock reduction for 512-bit instructions: may be better off at 256 bits. |
| ARM (Apple M-series, AWS Graviton, Ampere) | No x86 permutation tables: SIMD.jl's generic lowering, so far only checked for correctness. |
| Machines with more than 56 cores or more NUMA nodes | The sample-sort threshold and the many-thread behaviour. |

To collect the data, run in a Julia started with all threads:

```julia
julia> using SIMDSort

julia> SIMDSort.tuning_report()     # a minute or two
"simdsort-tuning-myhost.txt"
```

The file holds the machine's description (CPU, caches, SIMD features, the
vector width chosen), best and median times of a fixed set of sorts with one
and with all threads (for the primitive types also with each top level
forced), and the phase breakdown of each. Please send it with an issue on
GitHub. Nothing else is collected.

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
scaling, with the sample-sort top level forced on and off), `distcross.jl`
(the crossover of the two top levels over size and tasks) and `tune.jl`
(each tuning constant changed alone), in a lean environment without plotting
packages. See [AVX-512](avx512.md) for what each answers.

## Test modes

Besides the test suite on the current release, the test matrix runs it on
Julia 1.10, with SIMD.jl's generic lowering (`SIMD_JL_GENERIC=1`), and with
bounds-checked kernels (`CHECK_KEYS = true` in `src/keyref.jl`), which checks
that the pointer kernels never read or write outside their arrays.
