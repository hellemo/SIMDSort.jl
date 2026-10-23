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

All supported element types are mapped in place to signed integers of the same
width, so that one signed vector compare implements `isless`:

| type | map $k(x)$ on the bit pattern |
|---|---|
| `Int32`, `Int64` | $x$ |
| `UInt32`, `UInt64` | $x \oplus 2^{w-1}$ (flip the sign bit) |
| `Float32`, `Float64` | $x \oplus ((x \gg_a (w-1)) \gg_l 1)$ |
| `rev = true` | additionally $\lnot k$ |

The float map turns sign-magnitude into two's complement: for negative values the
magnitude bits are flipped, so larger magnitudes compare smaller, and $-0.0$ sorts
before $+0.0$ as in `isless`. Every map is an involution (it leaves the sign bit
alone), so the same function restores the original bits. NaNs are first moved to
the end (the front for `rev`) by a linear pass, where `isless` places them.

AVX2 has no unsigned or 64-bit unsigned compares; this map is why the kernels only
need signed compares. [WBGS22, §2.3] handles key types and sort order through
traits instead; the map is our choice.

## 2. Recursion [Hoa62, WBGS22 Alg. 1, Mus97]

```
sort(a[lo:hi], depth):
    while n > NBASE:
        if depth == 0: heapsort fallback (Base.sort!), return      [Mus97]
        p = choose_pivot(a[lo:hi])            # always one of the keys
        m = partition(a[lo:hi], p)            # a[lo:m-1] ≤ p < a[m:hi]
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
guarantees progress. Recursing on the smaller side bounds the stack at
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
they still measured 2–6% faster overall than a 128/64-key base case. As in [WBGS22, §2.3], the
keys are copied into a buffer, padded with `typemax`, sorted in registers, and the
first $n$ are copied back.

1. **Column sort.** A sorting network on $R$ inputs is applied across the $R$
   registers. Every compare-exchange is one vector `min` and one `max` and sorts
   all $N$ columns at once [WBGS22 §3]. We use Batcher's odd-even merge sort
   [Bat68] (63 compare-exchanges for $R = 16$; [WBGS22] uses Green's 60).
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
  - Ranges above $2^{16}$ keys hand their smaller side to a new task, with a share
    of `ntasks` proportional to its size.
  - Ranges above $2^{20}$ keys are partitioned in parallel, because the serial top
    partitions are otherwise the critical path. Each task partitions one chunk in
    place. Then the keys $> p$ that lie below the global boundary $m = \sum_i m_i$
    are swapped with the keys $\le p$ above it. The two sets have the same size
    $X$, and each task swaps an equal slice of ranks $[jX/k, (j+1)X/k)$.
  - The key maps run in parallel chunks.
