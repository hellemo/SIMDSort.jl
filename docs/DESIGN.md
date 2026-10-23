# SIMDQuickSort design

SIMDQuickSort is a vectorized quicksort written from the published descriptions
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

Notation: $N$ is the number of lanes per vector (8 for 32-bit and 4 for 64-bit keys
with 256-bit AVX2 vectors), $n$ the length of the range being sorted.

## 1. Keys

**Integers** are mapped in place to signed integers of the same width, so that
one signed vector compare implements `isless`:

| type | map $k(x)$ on the bit pattern |
|---|---|
| `Int32`, `Int64` | $x$ |
| `UInt64` | $x \oplus 2^{w-1}$ (flip the sign bit) |
| `UInt32` | $x$ (sorted as unsigned) |
| `rev = true` | additionally $\lnot k$ |

Every map is an involution, so the same function restores the original bits.
- AVX2 has no unsigned 64-bit compare or min/max, so `UInt64` is mapped. Sorting
  it natively (an extra xor per compare, and an in-register sign flip in the base
  case) measured 1.4% slower than paying the two map passes.
- AVX2 does have unsigned 32-bit min/max (`vpminud`/`vpmaxud`; a compare is
  `vpminud` + `vpcmpeqd`). Sorting `UInt32` natively saves the two passes:
  0.353 → 0.318 s at $2^{26}$ keys, the same as `Int32`.
[WBGS22, §2.3] handles key types and sort order through traits instead; the map
is our choice.

**Small key ranges.** Before the map, integer inputs whose values span at most
$2^{16}$ values, and fewer than $n/4$, are sorted by counting. This is not in the
papers.
1. A 1024-key sample rejects wide ranges at once: a sample's range can only be
   smaller than the full range.
2. The sample also sets up the count table: its range plus a margin of 32 on each
   side, clamped to the type's range.
3. Each key goes to slot $\min((x - \mathit{base}) \bmod 2^w, W)$. This is exact
   because the table interval never wraps, and slot $W$ collects keys outside the
   table.
4. Only if that overflow slot is non-zero is the exact min/max scanned and the
   counting repeated.
5. Counting uses four interleaved tables per task, so that runs of equal keys do
   not serialize on one counter.
6. The runs are written by up to `ntasks` tasks, each filling an equal slice of
   the output (in descending order for `rev`).

The fused pass replaced a separate min/max scan (38% of the time). At $2^{26}$
keys in 0:1000 this takes 0.100 → 0.067 s on one thread and 0.074 → 0.023 s on
eight.

**Floats** are sorted as floats:
1. One vectorized, task-parallel scan counts NaNs, `-0.0`s and `+0.0`s from the
   bit patterns.
2. If there are NaNs, a linear pass moves them to the end (the front for `rev`),
   where `isless` places them.
3. `rev` negates the values before and after sorting (an exact involution).
4. Vector float compares, and the x86 `vminps`/`vmaxps` used by the networks
   (§5), treat $-0.0 = +0.0$, and `vmin`/`vmax` may even change the sign of equal
   zeros. So after sorting, the run of zeros (found by binary search) is
   rewritten as the counted number of $-0.0$s followed by $+0.0$s. All other
   values pass through compares and min/max bit-exactly.

An earlier version mapped floats to signed integers as well (sign-magnitude to
two's complement, $x \oplus ((x \gg_a (w-1)) \gg_l 1)$). Sorting them natively
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
least $N$ of the $9N$ sampled keys equal the pivot, a three-way partition (§3)
puts the keys below the pivot left and those above it right. The block of keys
equal to the pivot is done and excluded from recursion. This is our addition.

- For ranges large enough for the parallel partition (§6), two parallel two-way
  passes are used instead, because the three-way pass is serial: one around the
  pivot, then one of the left side around the next smaller key (`x - 1`, or
  `prevfloat`).
- Triggering it on fewer equal samples (2 or 3 of $9N$) cost random inputs up to
  5%.

Recursing on the smaller side bounds the stack at
$\log_2 n$ frames. The depth limit is $2\log_2 n + 4$ [WBGS22, §2.2]; past it the
range is sorted by `Base.sort!` (introsort fallback [Mus97]).

## 3. Partition [Bra17 Alg. 3, WBGS22 §2.1]

In place and bidirectional, as in [Bra17]:

1. Copy $U$ vectors from each end of the range into a small buffer. This creates
   $UN$ free slots at each end. [Bra17] keeps these vectors in registers; a buffer
   leaves the registers free for a larger unroll factor.
2. Main loop, while at least $UN$ unread keys remain: load the next $U$ vectors from
   the side with **less free space** (left free space $=$ read pointer $-$ write
   pointer) [Bra17, line 18]. For each vector $x$:
   `(y, c) = vpartition(x, x ≤ p)` puts the $c$ keys $\le p$ first, in order, and
   the others after them. Two full-width stores follow: $y$ at the left write
   position $\ell$ (advance $\ell$ by $c$), and $y$ ending at the right write position
   $r$ (move $r$ back by $N - c$). The lanes beyond each side's valid part are
   garbage and get overwritten later.
3. Single-vector loop for the remaining whole vectors, then the held vectors and the
   tail, as described below.

**Why full-width stores are safe.** Let $g_L, g_R$ be the free space at each end.
While $2U$ vectors are set aside, $g_L + g_R = 2UN$, so the side we load from has
$g \le UN$ and has $g + UN \ge UN$ after the load; the other side has at least $UN$
too. A full-width store at either end therefore only writes free slots.
[Bra17] uses AVX-512 masked compress-stores (`vpcompress`) instead. On AVX2 those
are emulated slowly, while `vpartition` + `vstore` costs one table lookup, one
`vpermd` and two plain stores.

**Tail.** When fewer than $N$ unread keys remain:
1. append them to the buffer;
2. the free region $[\ell, r)$ is now contiguous and exactly as large as the
   buffer contents;
3. partition buffer vectors with full-width stores while at least $2N$ keys
   remain, since then $r - \ell \ge 2N$ and the two store windows are disjoint;
4. place the last $< 2N$ keys one by one, without branches: each key is written to
   both $\ell$ and $r-1$, and only the end it belongs to advances. A write to the
   wrong end only touches a free slot, which a later key overwrites.

[WBGS22, §2.1] also uses a buffer for the final keys so that no load or store goes
past the range.

**Three-way partition.** Used when the pivot sample shows many duplicates (§2).
It follows the same scheme as the two-way partition, but keys equal to the pivot
are never stored. Each vector is written twice:
- keys $< p$, compressed to the front, at $\ell$;
- keys $> p$, moved to the back, ending at $r$.

Dropped keys only add free space, so the argument for full-width stores still
holds. At the end the region between the two sides, exactly as long as the
number of dropped keys, is filled with the pivot value. Floats lose the sign of
equal zeros there, which `order_zeros!` repairs (§1).

For 32-bit keys, the two stores need two `vpartition`s. For 4-lane (64-bit)
keys, one 256-entry table indexed by both masks (4 bits each) gives a
permutation that orders the vector as $[< p \mid = p \mid > p]$, so a single
`vpermd` serves both stores. The cost relative to the two-way partition:

| | in cache ($2^{14}$ keys) | out of cache ($2^{22}$ keys) |
|---|---:|---:|
| `Int32` | 1.16× | 1.35× |
| `Int64` | 1.13× | 1.40× |

It replaces a two-way pass plus a second pass over the left side, about 1.5
passes.

**Prefetch.** For ranges larger than 2 MB (about the L2 size), each iteration
issues a software prefetch (`llvm.prefetch`) for the cache lines 512 keys ahead
of the vectors just read, on the side read.
- On $2^{26}$ keys this cuts the partition cost from 0.25 to 0.17 ns/key
  (`Int32`) and from 0.50 to 0.33 ns/key (`Int64`).
- In smaller ranges it costs 20–40%, because the hardware prefetchers keep up
  there. Hence the size gate (§7).
- The papers do not use software prefetch.

**Unrolling and the side choice.** [WBGS22] unrolls by 4. Here $U = 8$ measured
best on AVX2 (in cache: 0.137, 0.124 and 0.119 ns/key for $U$ = 4, 8, 16 with 32-bit
keys; 0.27 ns/key for all $U$ at $2^{26}$ keys, where the loop is
bandwidth-bound). The side to load from is data dependent and mispredicts often.
A branch-free version (`ifelse` on the read positions) was measured and rejected:
it chains each load address to the previous vectors' popcounts and was slower,
while with the branch the loads can run ahead speculatively.

## 4. Pivot [WBGS22 §2.2]

[WBGS22] samples nine 64-byte chunks at random positions, takes lane-wise medians
of three, and reduces them. Here:
- 9 vectors are loaded at pseudo-random offsets, from an xorshift generator seeded
  per sort. Each offset in $[0, n-N]$ is $\lfloor s \cdot (n-N+1) / 2^{64} \rfloor$, a
  multiply-high instead of a 64-bit division;
- lane-wise median of three gives 3 vectors, and a further median of three gives
  1 vector of $N$ lane medians;
- its $N$ lanes are sorted and the middle one is the pivot.

The pivot is always an input key, as the recursion requires. Random offsets
protect against inputs constructed to defeat fixed sampling positions; the depth
limit bounds the remaining risk.

## 5. Base case [WBGS22 §2.3 and §3, Bat68]

Ranges of at most $R \cdot N$ keys are sorted with a vectorized sorting network
over an $R \times N$ matrix: $R$ registers (rows) of $N$ lanes (columns), with
$R \in \{N, 2N, 4N\}$ (at most 32) chosen as the smallest that fits: up to 256
32-bit or 128 64-bit keys. With 16 AVX2 registers the 32-row networks spill, but
they still measured 2–6% faster overall than a 128/64-key base case. This holds
for `Int64` too: its 128-key network costs 3.7× the 64-key one, but capping it at
64 keys measured 2–6% slower overall. As in [WBGS22, §2.3], the
keys are copied into a buffer, padded with `typemax`, sorted in registers, and the
first $n$ are copied back.

1. **Column sort.** A sorting network on $R$ inputs is applied across the $R$
   registers. Every compare-exchange is one vector `min` and one `max` and sorts
   all $N$ columns at once [WBGS22 §3]. We use Batcher's odd-even merge sort
   [Bat68] (63 compare-exchanges for $R = 16$; [WBGS22] uses Green's 60). For
   floats, `min`/`max` are the x86 `vminps`/`vminpd` and `vmaxps`/`vmaxpd`
   intrinsics on AVX. Julia's `min`/`max` add NaN handling (an extra compare and
   blend), and LLVM adds the same to a compare-and-select. The inputs never
   contain NaNs (§1), and doubling the instruction count made the float networks
   twice as slow as the integer ones.
   Sorting 128 32-bit keys takes about 80 ns (0.6 ns/key), against 15 ns/key for
   insertion sort.
2. **Merging columns without transposing** [WBGS22 §3]. Number the elements
   column-major, $i = c \cdot R + \rho$ (column $c$, row $\rho$), so each column
   is a sorted run of length $R$. Bitonic merges [Bat68] combine runs of length
   $s = R, 2R, \dots, RN/2$:
   - **Flip step:** $i$ is compared with $i \oplus (2s-1)$. In matrix terms that is
     row $\rho$ against row $R-1-\rho$, lane $c$ against lane
     $c \oplus (2s/R - 1)$: one fixed lane shuffle, `min`, `max`, two blends and
     the shuffle back, per pair of rows.
   - **Half-cleaner steps** with distance $d = s/2, \dots, 1$: for $d \ge R$ the
     partner is lane $c \oplus d/R$ in the same register (shuffle, `min`, `max`,
     blend). For $d < R$ the partner is row $\rho \oplus d$ in the same lane: a
     plain vector `min`/`max` between two registers.

   The element with the lower index always receives the minimum.
3. **Store.** The result is sorted in column-major order. Each $N \times N$ block is
   transposed with $\log_2 N$ stages of two-input shuffles (swapping the
   off-diagonal $h \times h$ sub-blocks for $h = N/2, \dots, 1$). This gives
   registers in output order, which are stored contiguously.

   [WBGS22] avoid even this final transpose by choosing the permutations in their
   merges. We accept one transpose per base case for simpler code generation.

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
    $\max(2^{16}, n/(16\cdot\mathit{ntasks}))$ keys. That is many more tasks than
    threads, so Julia's scheduler balances the fast and slow cores of a hybrid
    CPU. On 4 P + 4 E cores, 64 dynamic tasks of $n/64$ keys finished in 53 ms,
    where 8 tasks of $n/8$ took 78 ms.
  - Ranges above $2^{20}$ keys are partitioned in parallel, by a number of tasks
    proportional to their size, because the serial top partitions are otherwise
    the critical path. Each task partitions one chunk in place. Then the keys
    $> p$ that lie below the global boundary $m = \sum_i m_i$ are swapped with the
    keys $\le p$ above it. The two sets have the same size $X$, and each task
    swaps an equal slice of ranks $[jX/k, (j+1)X/k)$.
  - The key maps and the float scan run in parallel chunks.

## 7. Performance analysis

Measured on an Intel Core Ultra 7 268V (Lunar Lake: 4 P + 4 E cores, AVX2,
LPDDR5X), Julia 1.13.2. `bench/profile.jl` reproduces sections a–c below.

### a. Where the time goes (one thread, $2^{26}$ keys, share of profile samples)

| input | partition | base case | pivot | other |
|---|---:|---:|---:|---|
| `Int32` | 66% | 27% | 6% | |
| `Int64` | 66% | 30% | 4% | |
| `Float32` (randn) | 60% | 29% | 5% | float scan 6% |
| `Float64` (randn) | 68% | 23% | 4% | float scan 4% |
| `UInt64` in 0:1000 | | | | counting sort 100% (min/max scan 38%) |
| `UInt64`, 1000 random values | 85% | | | key maps 15% |

### b. Partition cost per key by range size (ns/key, without / with prefetch)

| keys | $2^{10}$ | $2^{14}$ | $2^{18}$ | $2^{20}$ | $2^{22}$ | $2^{24}$ | $2^{26}$ |
|---|---|---|---|---|---|---|---|
| `Int32` | 0.137 / 0.186 | 0.105 / 0.140 | 0.109 / 0.131 | 0.114 / 0.131 | 0.223 / 0.170 | 0.252 / 0.167 | 0.249 / 0.171 |
| `Int64` | 0.236 / 0.340 | 0.208 / 0.279 | 0.233 / 0.307 | 0.346 / 0.284 | 0.565 / 0.345 | 0.533 / 0.333 | 0.501 / 0.333 |

- **In cache:** the loop runs at about 0.11 ns per 32-bit key (about 4 cycles
  per vector of 8). Most of that is the mispredicted load-side branch (§3).
- **Out of cache:** without prefetch, the loop reached only about 30 GB/s, where
  `copyto!` reaches about 50 GB/s. Prefetch closes most of that gap.
- Below about $2^{10}$ keys, the fixed cost per call shows up: the set-aside
  keys and the scalar tail.

### c. Base case, ns per call

| keys | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| `Int32` | 56 | 48 | 51 | 93 | 236 |
| `Float32` | 35 | 53 | 59 | 85 | 284 |
| `Int64` | 19 | 47 | 116 | 431 | |
| `Float64` | 11 | 23 | 62 | 202 | |

### d. Effect of each change ($2^{26}$ keys, median of 7 runs, seconds)

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
  $2^{26}$ keys. They could be avoided with unsigned compares (an extra xor per
  compare) in the kernels.
- **Few distinct values spread over a wide range** (1000 random `UInt64` values):
  the duplicate peeling only triggers once a range holds few distinct values. A
  radix or sample-sort top level would do better here, but measured too slow for
  random keys (e).
- **8 threads:** the top-level parallel partitions are bandwidth-bound (a
  $2^{26}$-key `Int32` partition takes as long as `copyto!`).

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

The remaining 7% is probably a different small-range sort in x86-simd-sort. That
cannot be confirmed under this package's clean-room rule.

**Wide-range duplicates.**
- **Pass count:** the sort made 12.4 partition passes over the data, against
  $\log_2 1000 \approx 10$ ideal. Of these, 7.15 were ordinary passes, 3.05
  passes where duplicates were detected, and 2.16 second passes splitting off the
  equal block.
- **The three-way partition (§3)** removes those second passes: −5% (1 thread)
  and −7% (8 threads), about half the gap to vqsort.
- **Fixes tried before it, none better than ±1%:**
  - a lower duplicate threshold (2 or 1 equal keys in $9N$ samples): up to 35%
    slower everywhere;
  - one partition with the next smaller key when the pivot is the maximum,
    instead of a min/max scan plus a partition;
  - choosing the side to peel the equal block from by the sample counts;
  - a read-only check for all-equal ranges when every sampled key equals the
    pivot. It looked 5.6% faster when one input was re-sorted repeatedly, but
    gave nothing with fresh inputs.
- **Native `UInt64`** removes the map passes but costs as much in the kernels.

**64-bit integers at 8 threads, against a radix sort.**
- **Where the time goes:** for `UInt64` at $2^{27}$ keys (1 GB), 0.448 of the
  0.495 s is the sort itself (the key maps take 2 × 24 ms). One parallel
  top-level partition takes 34 ms, about 60 GB/s, and about 6.5 levels run out
  of cache.
- **What a radix sort does instead:** it makes a few counting and scatter
  passes, and much less work per key, which wins once 8 threads share the memory
  bandwidth.
- **Prototype:** a parallel 8-bit radix distribution into a second array, then
  this quicksort on each bucket, then a copy back. With a preallocated buffer it
  was 12–16% faster: 0.484 against 0.549 s for `UInt64` at $2^{27}$, and 0.197
  against 0.235 s for `Int64` at $2^{26}$. Allocating the 1 GB buffer on every
  call (page faults) made it slower than the plain quicksort (0.632 s).
- **Not adopted.** Even the best case stays well behind ChapelSort's 0.32 s.

**Benchmark protocol.** `bench/bench.jl` draws a fresh input for every round (the
same input for all methods within a round), runs the methods in a shuffled order
each round, and reports the best round. Timing repeated sorts of one input could
let the branch predictor learn its pattern, especially for inputs that fit in
cache. Interleaving spreads thermal and power drift evenly over the methods.

