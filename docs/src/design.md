# SIMDSort design

SIMDSort is a vectorized quicksort written from the published descriptions
below. It uses no code from Google Highway (vqsort) or Intel's x86-simd-sort. The
SIMD primitives come from SIMD.jl, extended with runtime permutes and compress
(`vpartition`, `vpermute`), which lower to `vpermd`/`pshufb` with a
permutation table on AVX2.

## Sources

- **[WBGS22]** J. Wassenberg, M. Blacher, J. Giesen, P. Sanders. *Vectorized and
  performance-portable quicksort.* Software: Practice and Experience 52(12), 2022.
  arXiv:2205.05982.
- **[Bra17]** B. Bramas. *A Novel Hybrid Quicksort Algorithm Vectorized using
  AVX-512 on Intel Skylake.* International Journal of Advanced Computer Science and
  Applications 8(10), 2017. arXiv:1704.08579.
- **[Bat68]** K. E. Batcher. *Sorting networks and their applications.* AFIPS
  Spring Joint Computer Conference, 1968.
- **[Hoa62]** C. A. R. Hoare. *Quicksort.* The Computer Journal 5(1), 1962.
- **[Mus97]** D. R. Musser. *Introspective sorting and selection algorithms.*
  Software: Practice and Experience 27(8), 1997.
- **[F5]** A. Arman, D. Loguinov. *F5: A Robust SIMD-Accelerated MSD Radix Sort.*
  Texas A&M University; listed by citation databases as ICDE 2026. Used:
  - zigzag-merge sorting networks (§III, Algorithm 3);
  - double min/max for 64-bit integer keys (§III-C);
  - skipping common key bits (§I).
- **[MBM93]** P. M. McIlroy, K. Bostic, M. D. McIlroy. *Engineering radix sort.*
  Computing Systems 6(1), 1993. MSD refinement of string keys (§8).

- **[CAFS]** V. S. Shlyk. *CAFS: A Cache-Aware Frequency Sort for Low-Cardinality
  Integer Data on x86-64.* arXiv:2605.25040, 2026. Used: the Chao1 estimate of
  the number of distinct keys from 1024 samples, and hash counting (§1).

- **[IPS4o]** M. Axtmann, S. Witt, D. Ferizovic, P. Sanders. *Engineering
  in-place (shared-memory) sorting algorithms.* ACM TOPC, 2022. arXiv:2009.13569.
  Used: a sample-sort distribution step with splitters from a sorted random
  sample and branch-free bucket search (§8, §7.j). Its in-place block
  permutation is not used: the distribution here is out of place.

Notation: ``N`` is the number of lanes per vector (8 for 32-bit and 4 for 64-bit keys
with 256-bit AVX2 vectors), ``n`` the length of the range being sorted.

## 1. Keys

**Integers** are mapped in place to signed integers of the same width, so that
one signed vector compare implements `isless`:

| type | map ``k(x)`` on the bit pattern |
|---|---|
| `Int32`, `Int64` | ``x`` |
| `UInt64` | ``x \oplus 2^{w-1}`` (flip the sign bit) |
| `UInt32` | ``x`` (sorted as unsigned) |
| `rev = true` | additionally ``\lnot k`` |

Every map is an involution, so the same function restores the original bits.
- AVX2 has no unsigned 64-bit compare or min/max, so `UInt64` is mapped. Sorting
  it natively (an extra xor per compare, and an in-register sign flip in the base
  case) measured 1.4% slower than paying the two map passes.
- AVX2 does have unsigned 32-bit min/max (`vpminud`/`vpmaxud`; a compare is
  `vpminud` + `vpcmpeqd`). Sorting `UInt32` natively saves the two passes:
  0.353 → 0.318 s at ``2^{26}`` keys, the same as `Int32`.
[WBGS22, §2.3] handles key types and sort order through traits instead; the map
is our choice.

**Small key ranges.** Before the map, integer inputs whose values span at most
``2^{16}`` values, and fewer than ``n/4``, are sorted by counting. This is not in the
papers.
1. A 1024-key sample rejects wide ranges at once: a sample's range can only be
   smaller than the full range.
2. The sample also sets up the count table: its range plus a margin of 32 on each
   side, clamped to the type's range.
3. Each key goes to slot ``\min((x - \mathit{base}) \bmod 2^w, W)``. This is exact
   because the table interval never wraps, and slot ``W`` collects keys outside the
   table.
4. Only if that overflow slot is non-zero is the exact min/max scanned and the
   counting repeated.
5. Counting uses four interleaved tables per task, so that runs of equal keys do
   not serialize on one counter.
6. The runs are written by up to `ntasks` tasks, each filling an equal slice of
   the output (in descending order for `rev`).

The fused pass replaced a separate min/max scan (38% of the time). At ``2^{26}``
keys in 0:1000 this takes 0.100 → 0.067 s on one thread and 0.074 → 0.023 s on
eight.

**Few distinct values in a wide range [CAFS]** (`src/hashcount.jl`). The steps:
1. A strided sample of 1024 keys estimates the number of distinct values ``K`` with
   the Chao1 estimator ``\hat K = u + f_1^2 / (2(f_2 + 1))``: ``u`` distinct values in
   the sample, ``f_1`` of them seen once, ``f_2`` twice.
2. If ``\hat K \le 2^{13}``, one pass counts the keys' bit patterns in an
   open-addressing table with ``4\hat K`` slots (linear probing, one table per
   task, merged afterwards). The table stays in the L1/L2 cache.
3. The distinct keys are sorted with `isless`, and each key's run is written by
   several tasks.

Floats are counted after the NaN partition. Equal bits mean equal keys there,
and ``-0.0`` and ``0.0`` are distinct keys. If the sample underestimates (many rare
values) and a table gets more than half full, the attempt is abandoned and the
quicksort runs. This is tested with 99% repeated values and 1% unique ones.
CAFS probes a cache-line bucket with AVX2 compares; here the probing is linear
and scalar.

**Floats** are sorted as floats:
1. One vectorized, task-parallel scan counts NaNs, `-0.0`s and `+0.0`s from the
   bit patterns.
2. If there are NaNs, a linear pass moves them to the end (the front for `rev`),
   where `isless` places them.
3. `rev` negates the values before and after sorting (an exact involution).
4. Vector float compares, and the x86 `vminps`/`vmaxps` used by the networks
   (§5), treat ``-0.0 = +0.0``, and `vmin`/`vmax` may even change the sign of equal
   zeros. So after sorting, the run of zeros (found by binary search) is
   rewritten as the counted number of ``-0.0``s followed by ``+0.0``s. All other
   values pass through compares and min/max bit-exactly.

An earlier version mapped floats to signed integers as well (sign-magnitude to
two's complement, ``x \oplus ((x \gg_a (w-1)) \gg_l 1)``). Sorting them natively
removes two passes over the data and, for `Float64`, makes the base case 25%
cheaper: AVX2 has `vminpd`/`vmaxpd`, but no 64-bit integer min/max (it takes a
compare and two blends).

## 2. Recursion [Hoa62, WBGS22 Alg. 1, Mus97]

```
sort(a[lo:hi], depth):
    while n > NBASE:
        if depth == 0: heapsort fallback (Base.sort!), return      [Mus97]
        p = choose_pivot(a[lo:hi])            # always one of the keys
        if dup:                               # many keys equal p in the sample
            (a, m) = partition3(a[lo:hi], p)  # < p | = p (done) | > p
        else:
            m = partition(a[lo:hi], p)        # a[lo:m-1] ≤ p < a[m:hi]
        if m > hi:                            # right side empty
            (mn, mx) = extrema(a[lo:hi])
            if mn == mx: return               # all keys equal
            m = partition(a[lo:hi], mn)       # left side = all copies of mn
            lo = m; continue
        recurse on the smaller side, loop on the larger one
    base_case(a[lo:hi])
```

Keys equal to the pivot go left. Since the pivot is a key of the range, the left
side is never empty; the right side is empty only when the pivot is the maximum.
That case is handled as in [WBGS22, Alg. 1]: a vector min/max scan detects ranges
of equal keys, which are done, and otherwise the minimum is used as the pivot, which
guarantees progress.

**Blocks of equal keys** are split off when the pivot sample suggests them: if at
least ``N`` of the ``9N`` sampled keys equal the pivot, a three-way partition (§3)
puts the keys below the pivot left and those above it right. The block of keys
equal to the pivot is done and excluded from recursion. This is our addition.

- For ranges large enough for the parallel partition (§6), two parallel two-way
  passes are used instead, because the three-way pass is serial: one around the
  pivot, then one of the left side around the next smaller key (`x - 1`, or
  `prevfloat`).
- Triggering it on fewer equal samples (2 or 3 of ``9N``) cost random inputs up to
  5%.

Recursing on the smaller side bounds the stack at
``\log_2 n`` frames. The depth limit is ``2\log_2 n + 4`` [WBGS22, §2.2]; past it the
range is sorted by `Base.sort!` (introsort fallback [Mus97]).

## 3. Partition [Bra17 Alg. 3, WBGS22 §2.1]

In place and bidirectional, as in [Bra17]:

1. Copy ``U`` vectors from each end of the range into a small buffer. This creates
   ``UN`` free slots at each end. [Bra17] keeps these vectors in registers; a buffer
   leaves the registers free for a larger unroll factor.
2. Main loop, while at least ``UN`` unread keys remain: load the next ``U`` vectors from
   the side with **less free space** (left free space ``=`` read pointer ``-`` write
   pointer) [Bra17, line 18]. For each vector ``x``:
   `(y, c) = vpartition(x, x ≤ p)` puts the ``c`` keys ``\le p`` first, in order, and
   the others after them. Two full-width stores follow: ``y`` at the left write
   position ``\ell`` (advance ``\ell`` by ``c``), and ``y`` ending at the right write position
   ``r`` (move ``r`` back by ``N - c``). The lanes beyond each side's valid part are
   garbage and get overwritten later.
3. Single-vector loop for the remaining whole vectors, then the held vectors and the
   tail, as described below.

**Why full-width stores are safe.** Let ``g_L, g_R`` be the free space at each end.
While ``2U`` vectors are set aside, ``g_L + g_R = 2UN``, so the side we load from has
``g \le UN`` and has ``g + UN \ge UN`` after the load; the other side has at least ``UN``
too. A full-width store at either end therefore only writes free slots.
[Bra17] uses AVX-512 masked compress-stores (`vpcompress`) instead. On AVX2 those
are emulated slowly, while `vpartition` + `vstore` costs one table lookup, one
`vpermd` and two plain stores.

**Tail.** When fewer than ``N`` unread keys remain:
1. append them to the buffer;
2. the free region ``[\ell, r)`` is now contiguous and exactly as large as the
   buffer contents;
3. partition buffer vectors with full-width stores while at least ``2N`` keys
   remain, since then ``r - \ell \ge 2N`` and the two store windows are disjoint;
4. place the last ``< 2N`` keys one by one, without branches: each key is written to
   both ``\ell`` and ``r-1``, and only the end it belongs to advances. A write to the
   wrong end only touches a free slot, which a later key overwrites.

[WBGS22, §2.1] also uses a buffer for the final keys so that no load or store goes
past the range.

**Three-way partition.** Used when the pivot sample shows many duplicates (§2).
It follows the same scheme as the two-way partition, but keys equal to the pivot
are never stored. Each vector is written twice:
- keys ``< p``, compressed to the front, at ``\ell``;
- keys ``> p``, moved to the back, ending at ``r``.

Dropped keys only add free space, so the argument for full-width stores still
holds. At the end the region between the two sides, exactly as long as the
number of dropped keys, is filled with the pivot value. Floats lose the sign of
equal zeros there, which `order_zeros!` repairs (§1).

For 32-bit keys, the two stores need two `vpartition`s. For 4-lane (64-bit)
keys, one 256-entry table indexed by both masks (4 bits each) gives a
permutation that orders the vector as ``[< p \mid = p \mid > p]``, so a single
`vpermd` serves both stores. The cost relative to the two-way partition:

| | in cache (``2^{14}`` keys) | out of cache (``2^{22}`` keys) |
|---|---:|---:|
| `Int32` | 1.16× | 1.35× |
| `Int64` | 1.13× | 1.40× |

It replaces a two-way pass plus a second pass over the left side, about 1.5
passes.

**Prefetch.** For ranges larger than 2 MB (about the L2 size), each iteration
issues a software prefetch (`llvm.prefetch`) for the cache lines 512 keys ahead
of the vectors just read, on the side read.
- On ``2^{26}`` keys this cuts the partition cost from 0.25 to 0.17 ns/key
  (`Int32`) and from 0.50 to 0.33 ns/key (`Int64`).
- In smaller ranges it costs 20–40%, because the hardware prefetchers keep up
  there. Hence the size gate (§7).
- The papers do not use software prefetch.

**Unrolling and the side choice.** [WBGS22] unrolls by 4. Here ``U = 8`` measured
best on AVX2 (in cache: 0.137, 0.124 and 0.119 ns/key for ``U`` = 4, 8, 16 with 32-bit
keys; 0.27 ns/key for all ``U`` at ``2^{26}`` keys, where the loop is
bandwidth-bound). The side to load from is data dependent and mispredicts often.
A branch-free version (`ifelse` on the read positions) was measured and rejected:
it chains each load address to the previous vectors' popcounts and was slower,
while with the branch the loads can run ahead speculatively.

## 4. Pivot [WBGS22 §2.2]

[WBGS22] samples nine 64-byte chunks at random positions, takes lane-wise medians
of three, and reduces them. Here:
- 9 vectors are loaded at pseudo-random offsets, from an xorshift generator seeded
  per sort. Each offset in ``[0, n-N]`` is ``\lfloor s \cdot (n-N+1) / 2^{64} \rfloor``, a
  multiply-high instead of a 64-bit division;
- lane-wise median of three gives 3 vectors, and a further median of three gives
  1 vector of ``N`` lane medians;
- its ``N`` lanes are sorted and the middle one is the pivot.

Ranges above ``2^{18}`` keys instead take the exact median of a larger sample:
``256`` 32-bit or ``128`` 64-bit keys (one base case), loaded as vectors from
pseudo-random offsets and sorted with the base-case network. There, `dup` is set
when at least 1/9 of the sample equals the pivot.
- This improves the split only slightly: 20.8 → 20.1 partition passes for
  `Float64` at ``2^{26}`` keys, with no measurable change in time.
- It detects duplicates more reliably: 1000 random `UInt64` values sort 3–9%
  faster.

The pivot is always an input key, as the recursion requires. Random offsets
protect against inputs constructed to defeat fixed sampling positions; the depth
limit bounds the remaining risk.

## 5. Base case [WBGS22 §2.3 and §3, Bat68, F5 §III]

**Shape.** Ranges of at most ``R \cdot N`` keys are sorted with a vectorized sorting
network over an ``R \times N`` matrix: ``R`` registers (rows) of ``N`` lanes (columns).
- ``R \in \{N, 2N, 4N\}``, at most 32, is the smallest that fits: up to 256
  32-bit or 128 64-bit keys.
- A 128/64-key base case measured 2–6% slower overall, even though it fits in
  the registers. This holds for `Int64` too.
- As in [WBGS22, §2.3], the keys are copied into a buffer, padded with `typemax`,
  sorted in registers, and the first ``n`` are copied back.

**Networks of 16 rows or fewer: zigzag merges [F5 §III, Algorithm 3].**
1. **Column sort.** Batcher's odd-even merge sort [Bat68] across the ``R``
   registers: every compare-exchange is one vector `min` and one `max` and sorts
   all ``N`` columns at once [WBGS22 §3].
2. **Zigzag merges** for ``k = 1, 2, \dots, N/2``. A sorted run of ``rk`` keys in
   ``r`` rows and ``k`` lanes is in *``k``-zigzag* order: rows ``2t`` and ``2t+1`` hold the
   elements ``2j`` and ``2j+1`` of each block of ``2k`` consecutive keys. Two runs side
   by side (``2k`` lanes) are merged by:
   1. reversing every odd row within the ``2k`` lanes (one shuffle);
   2. merging every column with Batcher's odd-even merge across the registers.
      After the reversal, the even rows of a column come from one run and the odd
      rows from the other, both sorted. These are plain vector `min`/`max`, with
      no shuffles;
   3. for exchange distances ``d = 1, 2, \dots, k``, in every pair of rows,
      swapping lane block ``t`` of the upper row with block ``t-1`` of the lower row
      (odd ``t``, blocks of ``d`` lanes; two two-input shuffles), then
      compare-exchanging the two rows.

   Each merge leaves a ``2k``-zigzag run whose columns are in a fixed permutation;
   for 8 lanes it ends in bit-reversed order ``(0, 4, 2, 6, 1, 5, 3, 7)``. [F5]
   tracks these permutations analytically. Our generator instead simulates the
   network on ranks when it generates the code (`zigzag_plan`), finds the
   permutation, and folds it into the next level's reversal masks. The increasing
   order of exchange distances was found the same way, and agrees with [F5]'s
   rotation of jump vectors.
3. **Store.** A final two-row transpose with the permutation folded in gives
   row-major output.

For the 16-row Float64 network (64 keys) this needs 274 `min`/`max`, 80 shuffles
and no blends, against 318, 80 and 40 for the earlier design. That design merged
columns with bitonic merges, done without transposing [WBGS22 §3]: a flip step
plus half-cleaners, and in-register lane steps that use only half of each
`min`/`max`. The earlier design is kept as `network_body`, used when `ZIGZAG[]` is set to
`false` (for comparisons).

**32-row networks: two halves.** With 16 AVX2 registers, a 32-row network keeps
32 registers' worth of keys live in every stage. In the old single network, 45%
of the instructions (946 of 2093 for Float64) were spill loads and stores.
- Each half is now sorted with the 16-row network.
- The halves are merged with one step comparing A[i] with B[last − i]. This is
  done register pair by register pair, so only two registers are live.
- Each half then finishes with its own 16-register half-cleaners.

That gives 798 instructions, of which 202 are spill moves. With AVX-512 (32
registers) the single network is used instead (avx512.md).

**Inlining.** The networks are inlined into their callers: as separate calls, all
``R`` vectors went through memory. Together with the split this made Float64 sorts
5–9% faster, and the zigzag merges another 0–8% (§7.g).

**Floats.** For floats, `min`/`max` are the x86 `vminps`/`vminpd` and
`vmaxps`/`vmaxpd` intrinsics. Julia's `min`/`max` add NaN handling (an extra
compare and blend), and LLVM adds the same to a compare-and-select; the inputs
never contain NaNs (§1).

All networks are generated at compile time (`@generated`), so the base case is
straight-line code with constant shuffles. Ranges of at most 8 keys use insertion
sort.

## 6. Deviations and choices not taken from the papers

- Before sorting, a scan that stops at the first key out of order detects sorted
  input, and non-increasing input, which is then reversed.

- Keys are mapped to signed integers once per sort (§1), instead of per-type
  compare traits.
- The partition step uses `vpartition` followed by two full-width stores, instead of
  two compress-stores. This is the AVX2-friendly form of [Bra17, `partition_vec`].
- The base case ends with a transpose (§5.3).
- Batcher's network sorts the columns instead of Green's.
- Parallelism (`ntasks > 1`) is our own addition. The papers sort with one thread;
  [WBGS22 §4] discusses multi-core scaling.
  - Ranges hand their smaller side to a new task down to a grain of
    ``\max(2^{16}, n/(16\cdot\mathit{ntasks}))`` keys. That is many more tasks than
    threads, so Julia's scheduler balances the fast and slow cores of a hybrid
    CPU. On 4 P + 4 E cores, 64 dynamic tasks of ``n/64`` keys finished in 53 ms,
    where 8 tasks of ``n/8`` took 78 ms.
  - Ranges above ``2^{20}`` keys are partitioned in parallel, by a number of tasks
    proportional to their size, because the serial top partitions are otherwise
    the critical path. Each task partitions one chunk in place. Then the keys
    ``> p`` that lie below the global boundary ``m = \sum_i m_i`` are swapped with the
    keys ``\le p`` above it. The two sets have the same size ``X``, and each task
    swaps an equal slice of ranks ``[jX/k, (j+1)X/k)``.
  - The key maps and the float scan run in parallel chunks.

## 7. Performance analysis

Measured on an Intel Core Ultra 7 268V (Lunar Lake: 4 P + 4 E cores, AVX2,
LPDDR5X), Julia 1.13.2. `bench/profile.jl` reproduces sections a–c below.

### a. Where the time goes (one thread, ``2^{26}`` keys, share of profile samples)

| input | partition | base case | pivot | other |
|---|---:|---:|---:|---|
| `Int32` | 66% | 27% | 6% | |
| `Int64` | 66% | 30% | 4% | |
| `Float32` (randn) | 60% | 29% | 5% | float scan 6% |
| `Float64` (randn) | 68% | 23% | 4% | float scan 4% |
| `UInt64` in 0:1000 | | | | counting sort 100% (min/max scan 38%) |
| `UInt64`, 1000 random values | 85% | | | key maps 15% |

### b. Partition cost per key by range size (ns/key, without / with prefetch)

| keys | ``2^{10}`` | ``2^{14}`` | ``2^{18}`` | ``2^{20}`` | ``2^{22}`` | ``2^{24}`` | ``2^{26}`` |
|---|---|---|---|---|---|---|---|
| `Int32` | 0.137 / 0.186 | 0.105 / 0.140 | 0.109 / 0.131 | 0.114 / 0.131 | 0.223 / 0.170 | 0.252 / 0.167 | 0.249 / 0.171 |
| `Int64` | 0.236 / 0.340 | 0.208 / 0.279 | 0.233 / 0.307 | 0.346 / 0.284 | 0.565 / 0.345 | 0.533 / 0.333 | 0.501 / 0.333 |

- **In cache:** the loop runs at about 0.11 ns per 32-bit key (about 4 cycles
  per vector of 8). Most of that is the mispredicted load-side branch (§3).
- **Out of cache:** without prefetch, the loop reached only about 30 GB/s, where
  `copyto!` reaches about 50 GB/s. Prefetch closes most of that gap.
- Below about ``2^{10}`` keys, the fixed cost per call shows up: the set-aside
  keys and the scalar tail.

### c. Base case, ns per call

| keys | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| `Int32` | 56 | 48 | 51 | 93 | 236 |
| `Float32` | 35 | 53 | 59 | 85 | 284 |
| `Int64` | 19 | 47 | 116 | 431 | |
| `Float64` | 11 | 23 | 62 | 202 | |

### d. Effect of each change (``2^{26}`` keys, median of 7 runs, seconds)

| change | `Int32` 1 thr | `Float32` 1 thr | `Int64` 1 thr | `Float64` 1 thr | `UInt64` 0:1000 1 thr | `Int32` 8 thr | `Float64` 8 thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 0.360 | 0.415 | 0.829 | 0.924 | 0.454 | 0.144 | 0.306 |
| prefetch above 2 MB (§3) | 0.326 | 0.355 | 0.686 | 0.757 | 0.364 | 0.121 | 0.302 |
| native floats, x86 float min/max (§1, §5) | 0.305 | 0.333 | 0.650 | 0.626 | | | |
| counting sort before the key map (§1) | | | | | 0.110 | | |
| finer tasks, parallel float scan (§6) | | | | | | 0.10 | 0.205 |

The machine drifts by about ±5% between runs (thermal and power limits), so each
change was compared against the previous commit in alternating runs.

### e. Tried and rejected

- **Branch-free load-side choice** (§3): slower, because each load address
  depends on the previous popcounts.
- **Radix scatter for the top levels** (replacing the out-of-cache partitions, as
  in the hybrid of [WBGS22 §4.2]): a plain 8-bit scatter costs 1.8 ns/key
  (`Int32`), more than all six out-of-cache partition levels together
  (≈ 1.6 ns/key before prefetch, ≈ 1.0 after).
- **Prefetch in all partitions:** 20–40% slower for ranges that fit in L2.
- **Unroll 4 for ranges below 1024 keys:** no gain.
- **A smaller base case** (16 rows; or for `Int64` only): 3–6% slower overall,
  even though the 32-row networks spill.
- **Green's 60-comparator column network:** at most about 1% to gain; not
  attempted.

### f. Remaining gaps

- **Unsigned 32-bit keys** (`UInt32`) pay two key-map passes, about 0.03 s at
  ``2^{26}`` keys. They could be avoided with unsigned compares (an extra xor per
  compare) in the kernels.
- **Few distinct values spread over a wide range** (1000 random `UInt64` values):
  the duplicate peeling only triggers once a range holds few distinct values. A
  radix or sample-sort top level would do better here, but measured too slow for
  random keys (e).
- **8 threads:** the top-level parallel partitions are bandwidth-bound (a
  ``2^{26}``-key `Int32` partition takes as long as `copyto!`).

### g. The final gaps

A second round targeted the gaps left after §7.d. All comparisons below were run
interleaved in one process, alternating the two versions, with fresh data;
separate processes on this laptop drift by ±5–10%.

| gap | result |
|---|---|
| `UInt32` (two key-map passes) | closed: native unsigned kernels, 0.353 → 0.318 s, equal to `Int32` (§1) |
| Counting path vs ChapelSort's | closed: fused counting and parallel fill (§1) |
| `Float64` vs x86-simd-sort, 1 thread | open, about 7% (best 0.566 vs 0.607 s, median 0.590 vs 0.615 s) |
| `UInt64`, 1000 random values over the whole range, vs vqsort | about half closed by the three-way partition (§3): −5% on 1 thread, −7% on 8 |
| 64-bit integers, 8 threads, vs ChapelSort's radix sort | open, structural (below) |
| Sorted / reverse-sorted integer inputs | closed: detected before the key map, so as fast as Base |

**`Float64`.** An earlier comparison that re-sorted one input repeatedly showed a
tie (0.620 vs 0.621 s). With the fresh-input protocol below, x86-simd-sort is
consistently about 7% faster, in two full benchmark runs. Our partition kernel
compiles to the minimal sequence (compare, movemask, table load, `vpermps`, two
stores) and runs near the copy bandwidth out of cache. The difference is
therefore elsewhere. What was checked:
- **Per vector:** the partition costs the same for all key types, about 0.83 ns
  per vector in cache. 64-bit keys simply have half as many keys per vector.
- **Small ranges:** close to their ideal cost (150 keys: 39 ns against 32 ns,
  plus 13 ns for the pivot).
- **Unroll factor:** changing it moves the total by less than 1%.
- **Base case:** almost all of its cost is the network itself (189 ns for 128
  keys). Copying, padding and the transpose are small. The 32-row network spills
  registers: it costs 3.3× the 16-row one, for twice the keys. Splitting it into
  two 64-key networks plus a bitonic merge was estimated at about 220 ns, so no
  gain.

- **Pivot quality:** an exact median of a 128-key sample for large ranges (§4)
  cut the partition passes from 20.8 to 20.1 but left the number of base cases
  unchanged (802k → 808k, mean 84 keys). The mean leaf size follows from the
  cutoff of 128 keys: ranges just above it split into two halves of 65–100 keys.
  So pivots account for at most about 3%, below the noise of the measurements.

About a quarter of the `Float64` time is spent in the base case, and the 128-key
network spills registers. A register-efficient 64-bit network, for example
following [WBGS22 §3] without the final transpose, is the remaining candidate.
The remaining 7% is probably a different small-range sort in x86-simd-sort. That
cannot be confirmed under this package's clean-room rule.

**Wide-range duplicates.**
- **Pass count:** the sort made 12.4 partition passes over the data, against
  ``\log_2 1000 \approx 10`` ideal. Of these, 7.15 were ordinary passes, 3.05
  passes where duplicates were detected, and 2.16 second passes splitting off the
  equal block.
- **The three-way partition (§3)** removes those second passes: −5% (1 thread)
  and −7% (8 threads), about half the gap to vqsort.
- **Fixes tried before it, none better than ±1%:**
  - a lower duplicate threshold (2 or 1 equal keys in ``9N`` samples): up to 35%
    slower everywhere;
  - one partition with the next smaller key when the pivot is the maximum,
    instead of a min/max scan plus a partition;
  - choosing the side to peel the equal block from by the sample counts;
  - a read-only check for all-equal ranges when every sampled key equals the
    pivot. It looked 5.6% faster when one input was re-sorted repeatedly, but
    gave nothing with fresh inputs.
- **Native `UInt64`** removes the map passes but costs as much in the kernels.

**64-bit integers at 8 threads, against a radix sort.**
- **Where the time goes:** for `UInt64` at ``2^{27}`` keys (1 GB), 0.448 of the
  0.495 s is the sort itself (the key maps take 2 × 24 ms). One parallel
  top-level partition takes 34 ms, about 60 GB/s, and about 6.5 levels run out
  of cache.
- **What a radix sort does instead:** it makes a few counting and scatter
  passes, and much less work per key, which wins once 8 threads share the memory
  bandwidth.
- **Prototype:** a parallel 8-bit radix distribution into a second array, then
  this quicksort on each bucket, then a copy back. With a preallocated buffer it
  was 12–16% faster: 0.484 against 0.549 s for `UInt64` at ``2^{27}``, and 0.197
  against 0.235 s for `Int64` at ``2^{26}``. Allocating the 1 GB buffer on every
  call (page faults) made it slower than the plain quicksort (0.632 s).
- **Not adopted.** Even the best case stays well behind ChapelSort's 0.32 s.

**Benchmark protocol.** `bench/bench.jl` draws a fresh input for every round (the
same input for all methods within a round), runs the methods in a shuffled order
each round, and reports the best round. Timing repeated sorts of one input could
let the branch predictor learn its pattern, especially for inputs that fit in
cache. Interleaving spreads thermal and power drift evenly over the methods.

### h. Quicksort or radix sort, by thread count and size

`bench/crossover.jl` compares this quicksort with ChapelSort.jl's parallel MSD
radix sort over ``n = 2^{20} \dots 2^{28}`` keys and 1, 2, 4 and 8 tasks. Like
`bench.jl`, it uses fresh inputs and interleaved runs, best of 5. The table
gives radix time / quicksort time; below 1, the radix sort is faster.

| keys | 1 task | 2 tasks | 4 tasks | 8 tasks |
|---|---|---|---|---|
| `Int64` | 1.4–2.8 | 2.0–5.2 | 1.3–2.5 | 0.94–0.99 at ≥ ``2^{26}`` (and ``2^{20}``), 1.3–1.9 at ``2^{22}``–``2^{24}`` |
| `UInt64` | 1.1–2.4 | 1.6–3.7 | 1.0–2.3 | 0.75–0.82 at ≥ ``2^{26}``, 0.91 at ``2^{20}`` |
| `UInt64`, 1000 values | 3.2–4.0 | 3.2–4.0 | 1.5–2.1 | 0.81–0.98 at ``2^{20}`` and ≥ ``2^{26}`` |
| `Float64` (randn) | 1.7–3.2 | 2.7–4.3 | 1.7–2.8 | 1.36–2.04 |
| `Int32` | 1.5–2.3 | 2.3–3.2 | 1.3–2.0 | 1.04–1.39 |

- **Where the radix sort wins:** only with 8 tasks (on 4 P + 4 E cores), and only
  for 64-bit integer keys, by 2–6% (`Int64`) and 18–25% (`UInt64`) at
  ``\ge 2^{26}`` keys.
- **Where it never wins:** for floats and 32-bit keys, at any size or task count.
- **The 2-task column:** the radix sort is relatively slower there than with 1
  task. That is a scaling issue of ChapelSort's 2-task runs.
- **A dispatch rule** ("64-bit integers, ``n \ge 2^{26}``, at least 8 tasks") would
  be specific to this machine's memory bandwidth per core. It is not
  implemented. Measurements on a server or AVX-512 node would be needed first.

### i. Final round: base case and few distinct values

The base case was the remaining lever for `Float64` (§7.g). All comparisons were
interleaved, with fresh inputs:

| change | effect (2^26 keys, 1 thread unless noted) |
|---|---|
| 32-row networks split into two halves (§5) | spill moves 946 → 202 (Float64 network); Float64 −5 to −9% (median) |
| networks inlined (no vectors passed through memory) | Float64 −4 to −5.5%; Int32 best 0.29 → 0.27 s |
| zigzag merges after F5 (§5) | 16-row Float64 network 565 → 445 instructions; Float32/UInt32 −3 to −10%, Float64 best ≈ 0.56 s |
| hash counting after CAFS (§1) | 1000 random UInt64 values 0.275 → 0.126 s; 8 threads 0.139 → 0.036 s |
| leaf-size-pruned networks (variants per register count) | rejected: −6 to −13% per isolated leaf, but +0.2 to +3.3% in full sorts (dispatch, code size) |

Final: `Float64` 0.542 s, against x86-simd-sort's 0.579 s; SIMDSort is the
fastest one-thread sort for every numeric type measured (README).

### j. A sample-sort top level for many threads (on from 16 tasks)

At 8 threads ChapelSort's radix sort beat the quicksort on random `UInt64`
(2^27 keys: 0.33 s against 0.43 s, §7.h). The idea was to replace the
memory-bound top levels of the quicksort with one distribution pass [IPS4o]:
- sample the keys and choose splitters;
- count (keeping each key's bucket in a `UInt16` array) and scatter into a
  scratch array, in parallel;
- quicksort every bucket on one thread, then copy it back while it is in cache.

`distribute_sort!` (`src/msd.jl`) does this, interleaved against the quicksort
(8 threads, best of 7):

| | quicksort | 256 buckets | 1024 | 4096 | ChapelSort |
|---|---:|---:|---:|---:|---:|
| `UInt64` 2^27 | 0.470 | 0.514 | 0.565 | 0.646 | 0.343 |
| `Int64` 2^26 | 0.186 | 0.233 | 0.268 | 0.305 | 0.213 |
| `Int64` normal 2^26 | 0.182 | 0.224 | 0.241 | 0.267 | 0.372 |

Instrumented (`UInt64` 2^27, 256 buckets): count 0.12 s, scatter 0.15 s,
bucket sorts about 0.3 s. Of the scatter, 0.09 s are first-touch page faults on
the fresh 1 GB scratch array: with the arrays kept between calls, the whole
sort takes 0.419 s, 5% under the quicksort. But keeping 1 GB per sort is not
acceptable. The rest of the quicksort's time at 8 threads goes to partitions
of 2^7 to 2^9 keys and to leaves (instrumented: about 60% of task time). That
work is compute-bound and happens in cache, so a distribution pass can't remove
it. On the 4P+4E laptop the E-cores make it worse.

Two lessons for the code:
- With a run-time bucket count, the branch-free search compiled to branches:
  29 ns/key, 8 mispredictions per key. With a compile-time `Val(nb)` it takes
  3.6 ns/key.
- `foreach_range` runs serially below `PERM_PAR_MIN` elements, so it must not
  be used to spread a handful of chunks over tasks (`foreach_chunk_index`).

**On a many-core node it wins.** On a 2 × 28-core Xeon Gold 6348 (Ice
Lake-SP), with `ntasks` correctly bounding the number of running tasks (it did
not before: `ntasks` only set the task grain), the parallel quicksort stops
scaling at about 16 threads, while the distribution step keeps going:

| seconds | 8 thr | 16 thr | 32 thr | 56 thr |
|---|---:|---:|---:|---:|
| `UInt64` 2^27, quicksort | 0.360 | 0.292 | 0.289 | 0.290 |
| `UInt64` 2^27, distribution | 0.475 | 0.276 | 0.175 | 0.147 |
| `Int64` 2^26, quicksort | 0.156 | 0.118 | 0.118 | 0.110 |
| `Int64` 2^26, distribution | 0.213 | 0.115 | 0.074 | 0.059 |
| ChapelSort `Int64` 2^26 | 0.275 | 0.153 | 0.103 | 0.078 |

**When it wins: a measured rule.** `bench/measure/distcross.jl` times both
top levels, forced, for Int64, UInt64, Float64 and Int32 at 2^20–2^28 keys
and 4–56 tasks on the Xeon. The crossover size falls as the number of tasks
grows, because the quicksort's partition passes before its tasks are
independent grow with ``\log_2`` of the task count while the distribution's
are fixed:

| crossover (keys) | 16 tasks | 20 | 24 | 32 | 40 | 48 | 56 |
|---|---|---|---|---|---|---|---|
| Int64 | 2^26 | 2^25 | 2^25 | 2^25 | 2^24 | 2^24 | 2^24 |
| UInt64 | 2^26 | 2^25 | 2^25 | 2^25 | 2^25 | 2^24 | 2^25 |
| Float64 | – | 2^26 | 2^25 | 2^25 | 2^24 | 2^25 | 2^25 |
| Int32 | – | – | 2^27 | 2^26 | 2^26 | 2^25 | 2^26 |

In bytes the 32- and 64-bit crossovers agree (Int32 2^26 keys and Int64 2^25
are both 256 MB at 32 tasks), so `use_distribution` takes the sample-sort
step from 16 tasks for 64-bit keys (20 for 32-bit keys, whose quicksort has
twice the lanes and still wins at 16) and when keys × bytes per key × tasks
≥ 2^33: 512 MB at 16 tasks, 256 MB at 32, about 150 MB at 56. Against always
taking the faster of the two, it loses 0.5% on average over the grid; the
earlier rule (from 24 tasks and 2^22 keys) lost 6.6%, up to 2.4× at 2^22
keys, where the distribution is much slower; never using the step loses
9.8% (27% for 2^24 keys and more at 16 tasks or more). A grid search over
the rule's constants (the script's fit) gives 24 tasks for 32-bit keys and
2^32.5, at 0.3%: within the noise of one run, so the round values stay.
`DIST_MODE[]` (`:on`, `:off`) forces the choice for measurements.

## 8. Other key types: windowed keys

Everything that is not a plain 32/64-bit number goes through one mechanism
(`src/perm.jl`). Every key is viewed as a big-endian bit string:
- numbers (`ukey`): the order-preserving unsigned form, with NaNs last;
- strings: their bytes followed by zeros.

**Windowed sort (`refine!`).**
1. Take a window of ``K = 64 - b`` bits of every key, where ``b`` is the number of
   bits needed for the element's position, and pack it with the position into a
   `UInt64`.
2. Sort the `UInt64`s with the quicksort above.
3. Refine runs whose windows are equal with the next window, as an MSD radix
   sort refines buckets [MBM93].
4. Finish runs of up to 64 keys with an insertion sort using a full comparison.

The positions in the low bits make every step stable, so this also gives a
`sortperm` equal to Base's (`simdsortperm`). Windows that all keys of a run share
are skipped without sorting; F5's prefix extractor does the same [F5 §I].

**Strings.**
- `String`, `SubString{String}`, and on Julia 1.14 contiguous `StringView`s are
  read in place, through pointer and length.
- The first 16 bytes of every key are cached contiguously, and each run gets its
  slice of the cache in its new order. Windows and small-run comparisons then
  rarely touch the string bytes, which removes most cache misses.
- Byte-equal strings of different length (`"a"` and `"a\0"`) are ordered by
  length at the end, as `isless` does.

**Symbols** (`src/symbols.jl`). Symbols are interned: equal names are the same
object, so a symbol is identified by its address, and vectors of symbols
usually have few distinct values. They are counted like the numeric hash
counting [CAFS]:
1. A Chao1 estimate from a sample decides; above ``n/4`` or ``2^{20}``
   distinct symbols they take the byte path.
2. Each task counts its chunk in its own table (key and count in one slot).
3. The tables are merged in parallel, each task taking the keys whose hash
   slots fall in its share. With linear probing those keys lie in that share
   of each table and the run of occupied slots after it, so each task reads
   only its share and the merge reads every table once (reading all tables
   in every task made it quadratic in the task count: 0.14 s at 56 threads).
   Each task allocates its own table, in parallel and near its core.
4. The distinct names are ranked by the byte-key sort.
5. `sort!` writes the runs in parallel. `sortperm` looks up every element's
   rank in a read-only table and sorts the `Int32` ranks stably.

The slot is the top bits of the address times an odd constant (Fibonacci
hashing). Taking middle bits instead clustered the addresses, which are
regularly spaced: 5 probes per insert at 2^18 and 2^19 slots for 10^5
symbols, where linear probing needs 1.3–1.8, so the speed depended on the
table size and with it on the sampled estimate (rounds of the same benchmark
took 0.18 or 0.27 s on one thread). The merge tables of step 3 hold keys
that share their top bits, so they hash with a second multiplier.

10^7 symbols with 10^5 distinct: on 8 threads 0.025 s (0.068 s with the
middle bits, 0.118 s with the earlier per-element dictionary lookups); on the
Xeon at 56 threads 0.022 s before the hash change, against 0.028 s for
ChapelSort.

*Possible follow-up: GC at many tasks.* After the hash change the Xeon takes
0.018 s at 56 threads (best), but rounds with a garbage collection take up to
0.059 s: the 56 count tables are 4–8 MB each (300–450 MB per sort) and hold
`Symbol` references, so the GC counts and scans them. Smaller tables were
measured and dropped (a third or half full: 13–35% slower with 1000
distinct, no gain with 10^5). The fix would be tables of addresses
(`UInt64`) from the malloc arena, reused across sorts, with
`unsafe_pointer_to_objref` turning the distinct addresses back into symbols:
safe in practice (symbols are never freed, as the hashing already assumes),
but one more unsafe spot for §10.

**Other numbers and composite types.**

| Type | Key |
|---|---|
| `Int128`, `UInt128`, BitIntegers.jl types | their bits, in 63-bit windows |
| `BigInt` | 8-byte header ``2^{63} \pm`` limb count, then the magnitude limbs (complemented if negative) |
| `BigFloat` | class/exponent header, then the significand in a common number of limbs (complemented if negative); significands via `mpfr_get_z_2exp` |
| `Float128` (Quadmath.jl) | order-preserving `UInt128` |
| `MultiFloat{T,N}` (MultiFloats.jl), `Double64`/`Double32` (DoubleFloats.jl) | leading word's key, then the lower words' keys, zeros canonicalized |
| `Date`, `DateTime`, `Time`, Periods (Dates) | their `Int64`, sorted in place |
| Tuples, Pairs | fixed-width components packed big-endian into multi-word keys; otherwise one stable `sortperm` per component, last first |
| `VersionNumber` | packed (major, minor, patch); ties with prerelease or build tags finished with `isless` |

**`sort!` for wide fixed-width keys: one sample-sort step [IPS4o]**
(`src/msd.jl`). Int128, UInt256 and other BitIntegers, Float128, Double64,
MultiFloats, and packed tuples and pairs used to be sorted by computing the
full permutation and then gathering `v[p]`. That is a random 16–32 byte access
per element, about half the time. Now, above 2^16 elements:
1. Sample 32 × 256 leading order words and sort them; every 32nd is a splitter.
   Radix digits of the leading word would not do: random floats span the sign
   bit and a few exponents, which put 10^7 `Float128`s into 4 non-empty buckets.
   If the sample has fewer than 64 distinct leading words, the old path is used.
2. Count, in parallel chunks, with a branch-free search over the 255 splitters,
   storing each element's bucket as one byte.
3. Scatter the values (only the values, not their keys) stably into a scratch
   array, in parallel.
4. Sort each bucket (about 40 000 elements, in cache) into `v`:
   - one SIMD sort of 64-bit keys, made of the order-word bits just below the
     bucket's common prefix plus the local position;
   - a full comparison in the rare runs of equal windows, with the position as
     tie-break (stable);
   - scratch buffers reused across the buckets of a task, since fresh 0.6 MB
     buffers were malloc'ed and page-faulted per bucket.

Packed tuple keys of up to 128 bits are built with straight-line `UInt128`
shifts. The general `putbits` loop indexes the word tuple at run time and was
7× slower; that mattered here because the keys are computed three times.

10^7 elements, seconds (old path → new; ChapelSort), interleaved, best of 3:

| type | 1 thread | ChapelSort | 8 threads | ChapelSort |
|---|---:|---:|---:|---:|
| `Int128` | 0.42 → 0.18 | 0.27 | 0.18 → 0.049 | 0.068 |
| `UInt256` | 0.52 → 0.28 | 0.34 | 0.21 → 0.081 | 0.093 |
| `Float128` | 0.55 → 0.22 | 0.37 | 0.29 → 0.063 | 0.088 |
| `Double64` | 0.54 → 0.27 | 0.75 | 0.33 → 0.075 | 0.185 |
| `Float64x2` | 0.52 → 0.24 | 0.49 | 0.28 → 0.064 | 0.133 |
| `Pair{Int32,Int64}` | 0.53 → 0.21 | 0.26 | 0.24 → 0.050 | 0.073 |
| `Tuple{Int64,Float64}` | 0.54 → 0.23 | 0.51 | 0.24 → 0.059 | 0.126 |

Getting there, step by step:
- First version (radix digits, values and keys both scattered, 2048 buckets):
  slower than the old path.
- Scatter the values only: Int128 faster, floats not.
- Splitters from a sample, and parallel count and scatter: faster than the old
  path for every type.
- Lean bucket sort with reused scratch: 1.5–2× faster again.

The package types come through extensions (weak dependencies). DoubleFloats'
`isless` is no total order with NaNs, so Base has no defined result there; NaNs
go last.

## 9. Instrumentation

`src/instrument.jl` has optional phase timers and event counters. They are
for machines where interactive profiling is impractical, such as cluster
nodes (`bench/measure/`).

```julia
r = SIMDSort.instrumented(() -> simdsort!(v; ntasks = 8))
SIMDSort.print_report(r)    # or r.phases, r.counters
```

When it is off (the default), a probe costs one load and a branch. Probes sit
on coarse steps only: per pass, per partition call or leaf, never per key.

What is recorded:
- **Phases:** the key map, the counting, presorted and hash checks, the
  quicksort; per partition size class `partition_2e<k>`; `pivot` and
  `leaf_sort`; the MSD and distribution steps (`msd_*`, `dist_*`).
- **Counters:** partition calls and keys per size class, three-way partitions,
  depth-limit fallbacks, equal ranges, leaf calls and keys, hash attempts and
  aborts, window sorts, spawned tasks, bucket counts and the largest bucket.

How it is kept cheap and correct:
- Accumulators are per thread, so concurrent tasks never wait on a lock. A
  shared lock had inflated nested phase times at 8 threads.
- The per-partition probes use fixed array slots, not Dict entries keyed by
  constructed symbols. That version doubled the run time of a one-thread sort.
- Below 2^12 keys only every 16th call is timed, and the time is scaled up.
  Calls and keys are always counted exactly.

The overhead is now under 10% of the wall time, and at one thread the phase
times add up to the wall time. Phase times are task times summed over threads.

## 10. Code structure and the remaining unsafe code

**Files.** `SIMDSort.jl` (public entry points, `sort_primitive!` per key
class), `keys.jl` (orderings, key maps), `keyref.jl` (key references), `partition.jl`, `network.jl`, `sort.jl` (quicksort, pivots,
counting sort), `parallel.jl` (parallel quicksort and partition), `tasks.jl`
(task helpers), `scratchmem.jl` (`ScratchArena`), `instrument.jl`, `perm.jl`
(key sources and the windowed sort), `hashcount.jl`, `symbols.jl`, `msd.jl`
(distribution passes), `wide.jl`, `compound.jl`, and the extensions.

**Types for control flow.**
- The sort direction is Base's `Forward`/`Reverse` ordering, converted once
  from `rev` at the public functions.
- The three primitive key classes (`MappedKeys`, `NativeKeys`, `FloatKeys`)
  select a `sort_primitive!` method, each with its own measured order of checks.
- The CPU target is a trait (`target()`).
- Leaf keys enter the networks through a transform (`PlainKeys`,
  `OffsetKeys`).
- Distribution passes classify through functors (`WordBucket`, `KeyBucket`)
  and build a `Distribution`.
- Key sources share one documented interface (`keybits`, `keywidth` or
  `exhausted`, `length_tiebreak`, `finish_ties!`), with byte keys behind
  `AbstractByteKeys`.
- Extensions only define `orderwords`/`has_orderwords`.

**Shared helpers.**
- Task helpers: `tforeach`, `tforeach_index`, `tmapreduce`,
  `balanced_groups` with `run_limited`, and `tworkers` (workers with their own
  scratch state).
- One partition main loop for the two- and three-way kernels.
- One sample-and-scatter path for both distribution passes.
- `@phase` for every timed region.

**Safe or raw memory access in the kernels.** The kernels are written against
key references (`keyref.jl`): `vld`/`vst`, `ld`/`st!`, `shift`, `copykeys!`,
`prefetch`. There are two backings, `ArrRange` (a `Vector` plus offset, read
with SIMD.jl's array `vload`/`vstore` under `@inbounds`) and `Ptr`. With
identical kernel code, interleaved A/B on the AVX2 laptop:

| | arrays (`SAFE_KEYS = true`) | pointers |
|---|---:|---:|
| partition kernel, 2^14 keys (in cache) | 1.22–1.25× | 1 |
| partition kernel, 2^22 keys | 1.08–1.12× | 1 |
| small sorts (24–128 keys) | 1.00–1.06× | 1 |
| small sort, 256 Int32 keys | 1.18× | 1 |
| full sort, 2^24 keys, 1 thread | 1.17–1.19× | 1.00 |
| full sort, 2^24 keys, 8 threads | 1.07–1.10× | 1.00 |

With arrays, every vector load and store recomputes the array's data address:
SIMD.jl's `llvmcall` stores carry no alias information, so LLVM cannot keep the
address in a register across them. `copyto!` also keeps its bounds checks. The
partition kernel has 579 instructions against 350. So `SAFE_KEYS = false`: the
kernels use pointers, and everything around them works on arrays and views.
`sort_auto!` takes the key vector, or the NaN-free view of a float vector.
Pointers are made only in `sort_auto!`/`sort_keys!` and for the scratch
buffers (`scratchref` follows the kind of the keys), each under one
`GC.@preserve`. The switch stays for re-measuring, e.g. if SIMD.jl gains
alias-annotated stores.

**What remains unsafe, and why.**
- `keyref.jl`: the pointer backing of the kernel primitives (above). Pointers
  are made only at the entry points, and the test matrix runs the whole suite
  with `CHECK_KEYS = true`: the kernels then work on arrays with every access
  bounds-checked, so out-of-range reads or writes would fail the tests.
- `scratchmem.jl`: `ScratchArena` takes buffers of 1 MB or more from
  `malloc`, which the GC does not count. Large temporaries had triggered full
  collections that mark every live object: 0.4 s of a 0.7 s `sortperm` of 10^7
  strings, and slower than one thread on 56 threads.
- `perm.jl`: a `Symbol`'s name is read through a pointer (it has no code
  units; symbols are never freed), and its length with `strlen`.
- `wide.jl`: BigInt/BigFloat limbs through GMP's documented `mpz_export`.

Removed:
- the array of raw string pointers;
- the bignum pointer array;
- `unsafe_load` of GMP internals;
- `unsafe_store!` of encoded words;
- every `unsafe_wrap` of the sort's data (float subranges, `as_keys`, the
  Dates extension);
- pointer arithmetic in all kernels;
- raw loads of string and encoding bytes.

**Strategies looked at for reducing it further:**

| Strategy | For | Result |
|---|---|---|
| Arrays in the kernels (`SAFE_KEYS = true`) | kernels | 17–19% slower on one thread (above); kept as an option |
| `Base.Experimental.@aliasscope` + `Const` | kernels | not applicable: `Const` promises that the array is not written within the scope, and the in-place partition reads and writes the same array |
| Alias metadata on SIMD.jl's `llvmcall` stores | kernels | no supported way to attach Julia's alias information to `llvmcall` IR; would be fragile |
| Bounds-checked test mode (`CHECK_KEYS`) | kernels | adopted: the test suite runs on bounds-checked kernels, at no production cost |
| Code-unit loads, assembled little-endian and byte-swapped | string bytes | adopted: LLVM fuses them into one `movbe`, the same as the pointer load (assembled big-endian: 8 byte loads, ~24 instructions); String `sortperm` unchanged |
| `reinterpret` instead of `unsafe_wrap` | key maps, Dates | adopted: the key-map loops still vectorize; A/B equal |
| GC-uncounted allocation without `malloc` | arena | none found: Julia has no way to keep an allocation's size out of the GC's accounting (an `unsafe_free!` was only proposed); `GC.enable(false)` is discouraged. `malloc` + `unsafe_wrap(...; own = false)` under `GC.@preserve` is the established pattern |

Not consolidated: `qsort!` and `pqsort!` remain separate. They differ in the
partition (serial, or parallel with a special case for duplicates), in
tracking the leaf bounds, and in spawning tasks, so a shared step function
would add hooks to the hottest loop for little gain.

**The regression gate.** Every refactoring step was checked with
`bench/ab.jl`: the previous source tree and the working tree as module copies
in one process, interleaved, fresh inputs, at 1 and 8 threads, including
forced sample-sort cases. A case counts as slower only if its median and best
are both worse by more than 2% and it stays so when rerun alone (`AB_ONLY`).
Each step also passed the tests on Julia 1.13, 1.10 and with generic SIMD
lowering.
