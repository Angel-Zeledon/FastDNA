# FastDNA benchmarks: methodology, full results, and history

This document holds the complete benchmark methodology behind the summary
table in the [README](../README.md#benchmarks): the test machine, how the
datasets are generated, the exact commands, per-run numbers, the correctness
cross-check, and the measurement history (including numbers that were later
found to be wrong and corrected). Everything here was measured on the machine
described below, with the scripts committed under
[`scripts/bench/`](../scripts/bench/). No numbers are copied from a paper or
a vendor claim; run the scripts yourself and you should land within noise of
the same figures.

All FastDNA figures in this document were measured with the **in-memory
counting strategy** (the default whenever the memory estimate fits the
budget -- see the README's
[Command-line interface](../README.md#command-line-interface) section for the
`--strategy`/`--max-ram` flags added since these runs).

## Test machine and dataset

| | |
|---|---|
| CPU | Intel Core i7-1165G7 @ 2.80GHz (11th Gen), 4 physical cores / 8 logical threads |
| RAM | 16 GB |
| OS | Windows 11 Pro |
| Rust | rustc/cargo 1.91.0, `cargo build --release` |
| Python | 3.12.10, NumPy 2.5.2, Biopython 1.88 |

The dataset is synthetic but realistic, not uniformly-random noise:
[`scripts/bench/generate_reads.py`](../scripts/bench/generate_reads.py)
builds a 1 Mbp random reference genome, then samples 150 bp reads across it
to 30x coverage (both strands), applying an Illumina-like quality curve
(~Q38 decaying to ~Q25 towards the read's 3' end, with occasional dips) and
injecting substitution errors at the rate that quality score implies
(`p_error = 10^(-Q/10)`, the same relationship `fastq.rs`'s own quality
trimming assumes). This gives the file a genuine two-peak k-mer frequency
spectrum -- a large peak of once/twice-seen erroneous k-mers, and a real
coverage peak around depth 30 -- instead of the flat, uninformative spectrum
uniformly-random bases would produce.

```bash
python scripts/bench/generate_reads.py 1000000 30 150 bench_small.fastq 1337
# wrote 200000 reads, 30000000 bases, genome=1000000bp, cov=30.0x
# -> a 65.8 MB FASTQ file
```

## Correctness cross-check

`k=31`, canonical k-mers. This is asserted by a script,
[`scripts/bench/crosscheck.py`](../scripts/bench/crosscheck.py), so "fast
because it does less work" has something checking for it every time these
numbers get re-published, not just the one time someone eyeballed a table.

```bash
python scripts/bench/crosscheck.py
```

This regenerates the same dataset the benchmark tables below were measured
against (same generator, same seed) and runs all four implementations --
FastDNA plus the three Python baselines -- against it, printing each one's
distinct/total counts and asserting the spread between them stays within a
stated tolerance (0.05% by default), exiting non-zero if it doesn't. Its
actual output, from this machine:

```
  fastdna       distinct=   1423950  total=    23999892
  naive_python  distinct=   1423964  total=    24000000
  biopython     distinct=   1423964  total=    24000000
  numpy         distinct=   1423964  total=    24000000

distinct spread: 0.0010%  |  total spread: 0.0004%  (tolerance: 0.05%)

OK: all four implementations agree within tolerance.
```

The 14-k-mer, 0.001% difference between FastDNA and the three Python
baselines (which do not apply quality trimming -- see their docstrings) is
exactly what FastDNA's quality trimming removing a handful of low-quality
read tails would produce, not a counting discrepancy. Four
independently-written implementations landing within 0.001% of each other is
itself a correctness check worth having, and one that runs on demand rather
than one that was merely true once. The full run takes on the order of
minutes (the two slower Python baselines are O(reads x k) pure-Python loops
over the full 200,000-read file); pass `--quick` for a small generated file
and a sanity check in seconds instead -- see the script's own `--help` for
that and its other options (`--file`/`--k`/`--tolerance-pct`).

## Counting speed: FastDNA vs. three ways to write this in Python

The comparison is against three real, runnable, correctness-checked
implementations, not a strawman:

- **`naive_python.py`** -- pure standard library, `collections.Counter`,
  string slicing. What most people write first.
- **`biopython_baseline.py`** -- same algorithm, but parses the FASTQ with
  `Bio.SeqIO` instead of hand-rolled parsing, because that's what most
  bioinformaticians reach for next.
- **`numpy_baseline.py`** -- a genuinely vectorized attempt: bases are
  2-bit-packed and k-mers built with array shifts rather than a Python loop
  per k-mer, and the reverse complement is computed with the *same*
  bit-trick FastDNA's Rust core uses, ported to NumPy.

All three count **canonical** k-mers (`min(kmer, reverse_complement(kmer))`),
same as FastDNA. None of them write an output file -- this table is the
counting step alone, isolated from I/O, on both sides: for FastDNA, the
`fastdna.count()` Python binding, not the CLI.

```bash
python scripts/bench/naive_python.py bench_small.fastq 31
python scripts/bench/biopython_baseline.py bench_small.fastq 31
python scripts/bench/numpy_baseline.py bench_small.fastq 31
python scripts/bench/fastdna_bench.py bench_small.fastq 31 <threads> <repeats>
```

200,000 reads / 30,000,000 bases, k=31, two independent runs of each Python
baseline and five of each FastDNA thread count (median reported; full
per-run times are in the script output, reproduced in the PR/commit this
table shipped with):

| Implementation | Run 1 | Run 2 | Notes |
|---|---:|---:|---|
| Pure Python (`Counter`) | 88.13 s | 113.41 s | stdlib only, no quality trimming |
| Biopython (`Bio.SeqIO`) | 121.17 s | 125.57 s | real FASTQ parser, no quality trimming |
| NumPy (vectorized) | 123.51 s | 107.11 s | 2-bit packing + array ops, no quality trimming |
| **FastDNA, 1 thread** | **4.94 s** (median of 5) | | full pipeline incl. quality trimming |
| **FastDNA, 2 threads** | **4.70 s** (median of 5) | | |
| **FastDNA, 4 threads** | **2.54 s** (median of 5) | | |
| **FastDNA, 8 threads (default)** | **1.99 s** (median of 5) | | |

Against the **fastest** Python baseline in this table (`Counter`, 88.13 s --
comparing against the slowest baseline would flatter the result), that puts
FastDNA at roughly **18-25x faster on a single thread** (88.13 / 4.94 s =
17.8x, up to 125.57 / 4.94 s = 25.4x against the slowest run recorded, the
Biopython run at 125.57 s), and **44-63x faster using the 8 threads this
laptop has** (88.13 / 1.99 s = 44.3x, up to 125.57 / 1.99 s = 63.1x).

The reason the NumPy baseline doesn't win is worth stating plainly: it still
runs a Python-level loop once per *read* (200,000 iterations), and inside
each iteration a k=31-deep chain of small array operations to build that
read's k-mer windows. NumPy's per-call overhead (tens of microseconds)
dominates when the arrays involved are only ~150 elements long; it never
amortizes the way it would over one huge array. This is a genuine, useful
data point: "vectorize it in NumPy" is not automatically a free win at this
granularity.

**Scaling is sub-linear** (1.99 s at 8 threads vs. 4.94 s at 1 thread is
~2.5x from 8x the threads, not 8x) because this machine has 4 physical cores
(8 is hyperthreaded) and, at this dataset size, opening and reading the same
65 MB file from disk on every run is a fixed cost the thread count doesn't
reduce. The architecture still scales further on larger files, where that
fixed cost amortizes away.

**Run-to-run variance**: this machine is a laptop under normal desktop load
(IDE, language servers, browser), not a dedicated benchmark rig, and the
numbers above show it -- e.g. the two `Counter` runs differ by 29% (88.13 s
vs. 113.41 s). That variance is disclosed rather than hidden: re-run the
scripts and expect figures in the same range, not bit-identical ones.

## End-to-end CLI (counting + Parquet export + QC report)

```bash
fastdna --input bench_small.fastq --output counts.parquet -k 31
```

Six runs, default 8 threads: 3.07, 3.14, 4.24, 5.00, 6.01, 9.50 s
(min 3.07 s, median 4.62 s) -- this is the number that matters if you only
care about "how long until `counts.parquet` exists on disk," and it includes
writing all 1,423,950 rows through Snappy-compressed Parquet plus the QC
JSON report, not just the in-memory count.

## Large-scale comparison: FastDNA vs. KMC3 vs. FASTK

KMC3 and FASTK are the field's own dedicated k-mer counters, not something
people write themselves -- this is the comparison that matters at scale.
Measured in WSL2 (Ubuntu 26.04) for KMC3 and FASTK, and the native Windows
release binary for FastDNA, all three against the *same* 2.14 GB synthetic
FASTQ file (35 Mbp genome, 30x coverage, 150 bp reads, same generator and
quality/error model as above, scaled up --
[`scripts/bench/generate_reads_large.py`](../scripts/bench/generate_reads_large.py),
seed 9001; exact install and run commands in
[`scripts/bench/kmc3_fastk_comparison.sh`](../scripts/bench/kmc3_fastk_comparison.sh)),
`k=31`, singleton k-mers included on all three (`kmc -ci1`, `FastK -t1`,
FastDNA's own `min_count=1` default):

**Re-measured 2026-08-25, and the conclusion inverted. See
[Measurement history and corrections](#measurement-history-and-corrections)
for the retracted table.** Two methodology changes make this run stricter
than the previous one: FastDNA is run with `-q 0` so all three tools count
exactly the same k-mers (previously FastDNA quality-trimmed while KMC3 and
FASTK did not, leaving a 2,244-k-mer discrepancy that had to be explained
away), and FastDNA was additionally compiled for Linux and run inside the
same WSL2 environment as the other two, removing the native-Windows vs.
WSL confound entirely.

All three in the **same environment** (WSL2 Ubuntu, ext4, 8 threads,
`k=31`, singletons kept -- `kmc -ci1`, `FastK -t1`, `fastdna -m 1 -q 0`):

| Tool | Time | Peak RAM | Distinct k-mers |
|---|---:|---:|---:|
| **FASTK** | **38.3 s** | 2.86 GB | 53,776,394 |
| KMC3 | 110.4 s | 9.06 GB | 53,776,394 |
| FastDNA (in-memory strategy) | 388.2 s | 10.34 GB | 53,776,394 |
| FastDNA (disk strategy) | 553.6 s | **1.21 GB** | 53,776,394 |

The same FastDNA source built natively for Windows, on the same machine
and the same input file:

| Configuration | Time | Peak RAM | Distinct k-mers |
|---|---:|---:|---:|
| FastDNA (in-memory) | 133.1 s | 8.34 GB | 53,776,394 |
| FastDNA (disk) | 281.8 s | 1.10 GB | 53,776,394 |
| FastDNA (auto -> disk) | 284.7 s | 1.09 GB | 53,776,394 |

All three FastDNA outputs are byte-identical to each other.

**All four counts agree exactly**: 53,776,394 distinct k-mers out of
840,000,000 total, across three independently written tools. Correctness is
not in question here; speed is.

**FastDNA is substantially slower than both dedicated counters, and
substantially more frugal than both with memory.** FASTK is 10x faster than
FastDNA's in-memory strategy in the shared environment (3.5x faster than the
Windows build); KMC3 is 3.5x faster (1.2x). Against that, FastDNA's disk
strategy peaks at 1.21 GB -- less than half FASTK's 2.86 GB and a seventh of
KMC3's 9.06 GB.

**Retracted 2026-09-07**: this paragraph also claimed FastDNA was the only
one of the three whose peak memory is configurable against a budget. That
is false, and was never checked before being written. `kmc --help` (3.2.4)
lists `-m<size>` -- max RAM in GB, default 12 -- and `-sm`, a strict mode
documented as "memory limit from `-m<n>` switch will not be exceeded". The
run in the table above was given `-m8`. What FastDNA actually adds over
that is not the budget but the estimator: `mem_estimate.rs` predicts peak
RSS from input size and thread count and picks a strategy against the
budget without being told which, and reports which it picked. That is a
narrower claim, and it is the true one.

One caveat that does not rescue the result but does explain part of it: the
WSL2 VM has 12 GB of RAM and FastDNA's in-memory strategy peaked at
10.34 GB, so that run was memory-constrained in a way the other two were
not. The Windows figures, where more memory was available, are the fairer
reading of the in-memory strategy -- and it is still 3.5x behind FASTK there.

### Why

Diagnosed by measurement, not inspection. Counting with near-zero export
(`-m 100`, so the table is built and traversed but almost nothing is
written) takes 110.1 s of the Windows in-memory run's 133.1 s, so Parquet
export costs about 23 s and is not the bottleneck.

The bottleneck is the volume of data sorted. FastDNA materializes **every
k-mer occurrence** as an independent 8-byte `u64` and sorts all 840,000,000
of them. KMC3's own run output reports what it does instead:

```
Total no. of k-mers        : 840,000,000
Total no. of super-k-mers  :  70,635,757
```

An 11.9x reduction in what is sorted and moved. Consecutive k-mers in a read
overlap by k-1 bases -- a 150 bp read yields 120 k-mers at `k=31`, but those
are 120 sliding windows over the same 150 bases, not 120 independent values.
A *super-k-mer* is a run of consecutive k-mers sharing a minimizer, stored
once. 110 s over 840M k-mers is 131 ns per k-mer, which is far too slow for
the handful of bit operations the extraction itself costs; the time is in
the sort, and the sort is over 12x more items than it needs to be.

A second, compounding factor: each worker builds its own private
`KmerCounter`, so peak memory scales with threads x occurrences. The crate's
own calibrated model (`src/mem_estimate.rs`) predicts
`1.168 bytes x 8 threads x 840M occurrences` = 7.85 GB against 8.34 GB
measured -- the model is sound, and the scaling with thread count is real.

See [`design-minimizer-counting.md`](design-minimizer-counting.md) for the
super-k-mer design this motivates.

### Measurement history and corrections

An earlier version of this table reported 194.5 s and 2.00 GB for FastDNA.
Both figures were wrong: re-measured on an otherwise-idle machine the time
is roughly half that and the peak memory four times it. The originals were
taken while other heavy work shared the CPU, and the memory figure never
matched any subsequent measurement. They are corrected here rather than
quietly replaced, because a published memory figure four times under the
real one is how someone plans a 16 GB run that dies twenty minutes in.

This table also did not always look like this in a second way. The first
version of this comparison had FastDNA at 920.7 s and 6.96 GB peak RAM on
the same file -- about 3x slower than KMC3 and 7.7x slower than FASTK.
`KmerCounter` was, at that point, an in-memory `HashMap<u64, u32>`: every
insertion is effectively-random bucket placement, an L3 cache miss once the
table outgrows a few tens of MB, which happens well before a real sample
finishes counting -- exactly the failure mode the field moved away from
after Jellyfish (2011), which is why KMC3 partitions k-mers into
disk-resident bins by minimizer signature before sorting each one, and why
FASTK does not hash at all, sorting 2-bit-packed k-mers into disk
partitions instead. Both are sequential-memory-access strategies.
`KmerCounter` now uses the same strategy, in memory rather than on disk:
insertion appends to a plain `Vec<u64>` (sequential, cache-friendly), and
counting happens as a one-time sort-and-compact pass on first read (see
`src/counter.rs`'s own doc comments for the detail). That change alone --
not new hardware, not a smaller test file -- is the entire difference
between 920.7 s / 6.96 GB and ~101 s / 8.02 GB.

## The head-to-head against KMC3, native on both sides (2026-09-08)

The first FastDNA-vs-KMC3 measurement since the 2026-08-25 retraction that
this project is willing to stand behind, and the first that FastDNA wins.
It replaces the "attempted" section that follows it, which recorded the run
that failed to settle anything and why.

### Conditions

| | |
|---|---|
| Host | Apple M3 Pro, Linux arm64 container, 11 cores, 7.7 GB, **not idle** (host load 6-11 from unrelated processes) |
| Both tools | native `aarch64`, same container, same compiler toolchain generation |
| KMC3 | 3.2.4, **built from source** at the `v3.2.4` tag (its Makefile handles `aarch64`); not the x86-64 release binary, which cannot run here |
| FastDNA | `cargo build --release` from this tree |
| Input | 6 Mbp genome at 30x, 150 bp reads -> 1,200,000 reads, 180,000,000 bases, ~390 MB |
| Settings | k=31, singletons kept (`-ci1` / `-m 1`), no quality trimming (`-q 0`), 11 threads, **4 GB budget to each** (`-m4` / `--max-ram 4G`) |
| Method | 9 repeats per tool, interleaved round-robin, median with full range |

### Result

| tool | median | min | max | peak RSS | distinct k-mers |
|---|---:|---:|---:|---:|---:|
| **FastDNA `binned`** | **1.99 s** | 1.90 s | 2.19 s | **0.95 GB** | 9,213,849 |
| **FastDNA `auto`** | **2.20 s** | 1.97 s | 2.38 s | **0.94 GB** | 9,213,849 |
| KMC3 3.2.4 | 3.04 s | 2.92 s | 3.19 s | 2.02 GB | 9,213,849 |
| FastDNA `memory` | 6.45 s | 6.00 s | 7.19 s | 1.91 GB | 9,213,849 |

All four agree exactly on 9,213,849 distinct k-mers.

**FastDNA's default is 1.38x faster than KMC3 here, in less than half the
memory** (0.94 GB against 2.02 GB). Forcing `binned` -- what `auto` picks
anyway on a larger input -- makes it 1.53x.

### Why this survives a noisy host

The medians alone would not be worth much: the machine was carrying an
unrelated load average of 6-11 throughout, and an earlier attempt on the
2.14 GB file saw the *same* tool vary by 7x between consecutive runs. What
makes this reportable is that **the ranges do not overlap**. Every one of
the nine FastDNA runs was faster than every one of the nine KMC3 runs:

```text
fastdna (binned) < kmc3:  [1.90, 2.19] vs [2.92, 3.19]   1.53x on medians
fastdna (auto)   < kmc3:  [1.97, 2.38] vs [2.92, 3.19]   1.38x on medians
kmc3 < fastdna (memory):  [2.92, 3.19] vs [6.00, 7.19]   2.12x on medians
```

Noise widens a range; it does not separate two of them. `head_to_head.py`
now reports exactly these non-overlapping pairs and refuses to rank
anything else, which is the only ordering a busy machine can support.

Three independent runs, one of 5 repeats and two of 9, all put the same
three tools in the same order with non-overlapping or near-non-overlapping
ranges.

### What this does *not* say

- **It does not overturn the 2.14 GB x86-64 table below.** Different
  architecture, a file 5.5x smaller, and that table measured a default path
  FastDNA no longer has. The two are not comparable and neither supersedes
  the other.
- **It is arm64, not x86-64.** KMC3's published binaries and most of its
  reported numbers are x86-64; a source build for a different ISA is a fair
  comparison on *this* machine and not necessarily elsewhere.
- **FastDNA's `memory` strategy is 2.12x slower than KMC3** on the same
  input, and is listed above so that stays visible. What is fast is the
  minimizer-partitioned path, which has been the default since 2026-09-05.
- **It says nothing about out-of-core scale.** KMC3's published result is
  729 gigabases in 33-34 GB. FastDNA's disk strategy has never been
  benchmarked anywhere near that, and this 390 MB file does not begin to
  probe it.

### The fairness bug this run also fixed

`head_to_head.py` used to hand KMC3 a hardcoded `-m4` while giving FastDNA
no budget at all. That is not a comparison: FastDNA's estimator reads what
the machine has free and picks a strategy against it, so on a small VM it
would correctly choose `disk` -- its slowest path -- while the competitor
ran with an explicit 4 GB allowance. Measured on this machine: default
budget 1,009 MB -> `auto` chose `disk`; `--max-ram 5G` -> `auto` chose
`binned`. Both tools now take the same `--max-ram`, and the historical WSL
table below has the same shape of problem (FastDNA peaked at 10.34 GB in a
12 GB VM while the other two did not).

## The head-to-head against KMC3, attempted natively (2026-09-08)

`README.md` has carried "the speed comparison is not settled" since the
default became super-k-mer partitioned. This is the attempt to settle it,
and it did not -- but it removed one of the two blockers permanently and
identified the other precisely.

**The architecture blocker is gone.** KMC3 ships x86-64 Linux binaries, so
on the Apple silicon this project is developed on it could only run
emulated, and `scripts/bench/head_to_head.py` refused to report timings for
that reason. But **KMC 3.2.4's own Makefile handles `aarch64`**
(`D_ARCH=ARM64`, `-march=armv8.4-a`). Built from source in a Linux arm64
container it runs native, alongside a native FastDNA, on one machine. The
script now does that automatically off x86-64 instead of refusing.

**Correctness, on a second dataset and both tools native.** On the 2.14 GB
synthetic file (840,000,000 occurrences, k=31, singletons kept):

| | distinct | total |
|---|---:|---:|
| KMC3 3.2.4 (native arm64) | 53,776,394 | 840,000,000 |
| FastDNA | 53,776,394 | 840,000,000 |

**Speed: still unresolved, and now for a reason that is measured.** The
Docker VM available here has 11 cores and 7.7 GB, and the host carries
unrelated load. Consecutive runs of the *same* tool on the *same* file
varied by up to 7x:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| KMC3 `-m5` | 19.95 s | **150.92 s** | 30.06 s |
| FastDNA `--max-ram 5G` | 22.49 s | 27.52 s | 19.17 s |
| FastDNA `--strategy binned` | 28.76 s | 18.01 s | **105.91 s** |

No median over a handful of runs survives that spread, and quoting one
would repeat the mistake the 2026-08-25 retraction was about. What the
numbers *do* show is that the two are in the same range once FastDNA is
given a comparable memory budget -- which is more than the old table said,
and less than a result.

**One thing the run did settle**, and it is about the chooser rather than
the speed: with the VM's default budget (1,009 MB, derived from what was
actually free) `auto` estimated an 11.40 GB peak for the in-memory path and
selected `disk`, FastDNA's slowest strategy, while KMC3 was explicitly
given `-m5`. Handed the same 5 GB with `--max-ram 5G`, `auto` estimated
3.49 GB and selected `binned`. The chooser is behaving correctly; a
benchmark that gives one tool a budget and lets the other discover a
starved one is not a fair comparison, and the earlier WSL table has the
same shape of problem.

**What is still owed**: a quiet x86-64 host with room for the working set.
`.github/workflows/validation.yml`'s `benchmark` job is exactly that, and
has never run.

## The per-bin sort: MSD partition instead of `sort_unstable` (2026-09-07)

`binned.rs` sorted each bin with a plain `sort_unstable`, while
`counter.rs`'s own buffer had gone through an MSD partition (1024 ascending
buckets, then `sort_unstable` per bucket) since it measured 1.23-1.37x
faster there. The default path never got it.

`DEFAULT_NUM_BINS`'s own doc comment gives the reason to expect it would
not help: a bin's expansion buffer is ~12.5 MiB, "L3-resident ... which
makes the per-bin sort cache-local". A partition over data already in cache
buys nothing and costs a scatter pass.

**That argument is single-threaded.** Phase 2 runs one worker per thread,
each holding its own bin: at 11 threads that is ~137 MiB of live buffers
against one shared cache -- the regime where `counter.rs`'s own thread
scaling showed the advantage holding. The two arguments point opposite
ways, so it was measured (`binned::tests::per_bin_sort_ab`, run with
`cargo test --release per_bin_sort_ab -- --ignored --nocapture`).

Method copied from `counter::msd_partition`'s: 1,600,000 keys (one bin's
buffer), real canonical k-mers rather than uniform-random `u64` (canonical
form skews the high bits an MSD partition buckets on), three coverage
shapes, 15 interleaved repeats, scratch reused. Three runs, medians:

| | `sort_unstable` | MSD | |
|---|---:|---:|---:|
| 1 thread, high coverage | 0.0169 s | 0.0124 s | **1.37x** |
| 1 thread, medium | 0.0175 s | 0.0129 s | **1.35x** |
| 1 thread, low (near-unique) | 0.0174 s | 0.0131 s | **1.32x** |
| 11 threads, high coverage | 0.0440 s | 0.0312 s | **1.41x** |
| 11 threads, medium | 0.0478 s | 0.0353 s | **1.35x** |
| 11 threads, low | 0.0448 s | 0.0341 s | **1.31x** |

The advantage holds at 11 threads rather than eroding, which is what
settled it.

### End to end it is worth much less, and that is not fully explained

Two release binaries differing only in that one line, run interleaved on
the same 2.14 GB file (`-k 31 -m 1 -q 0`, default strategy), 9 repeats
each:

| | median time | min | median peak RSS |
|---|---:|---:|---:|
| `sort_unstable` | 9.23 s | 8.84 s | 2.99 GB |
| MSD | 8.79 s | 8.48 s | 3.04 GB |
| | **1.050x** | 1.042x | 1.016x |

Output is byte-identical (SHA-256 checked against the in-memory strategy on
real reads, and `tests/dual_strategy.rs` asserts it on every run).

A 1.35x speedup on a step an earlier profile put at **39.5%** of the run
should have been worth about 1.11x overall. It is worth 1.05x. The
discrepancy was chased rather than left standing, and it resolves cleanly:
**sorting is no longer 39.5% of a binned run. It is 22.6%.**

The first attempt at that profile was discarded as unusable --
`/usr/bin/sample` emits a call tree with *cumulative* counts, and summing
those lines double-counts every parent, which attributed 95% of the run to
"sort". The usable numbers come from the same output's "Sort by top of
stack, same collapsed" section, which is leaf-attributed self time. Five
seconds sampled from a ~9 s run, waiting frames (`__psynch_cvwait`,
`semaphore_wait_trap`, ...) excluded so the shares are of work rather than
of wall time:

| | share of work |
|---|---:|
| `minimizer::SignatureScanner::push` | **31.8%** |
| `binned::BinWriter::push_sequence` | 19.0% |
| `binned::BinStore::finish` (expand + compact) | 18.5% |
| `core::slice::sort` (quicksort) | 16.3% |
| `counter::sort_keys_msd` (the partition itself) | 5.4% |
| pipeline phase-1 driver | 2.6% |
| `__bzero` / `memmove` / pivot selection | 4.1% |

Sorting -- quicksort, pivot selection and the MSD partition together --
is 22.6%. Amdahl on that fraction with a 1.35x local speedup predicts
**1.062x**, against 1.050x measured, which closes the question: the
isolated benchmark overstates the win slightly, and the older 39.5% figure
simply no longer describes this pipeline (the cross-bin merge became
parallel after it was taken).

**The finding that matters more than the 5%**: the largest single consumer
of a binned run is now the minimizer signature scanner, at 31.8% -- a
sliding-window minimum over a `VecDeque`, run once per base. That, not
sorting, is where the next speed work belongs.

The change shipped anyway because the trade has no losing side: strictly
faster, identical output, memory unchanged within the spread, and it reuses
a routine already tested and measured for `counter.rs`.

**Caveat on the absolute numbers**: the host was not idle (load average
5.7-10.3 from unrelated processes) and both arms are inflated by it. The
interleaving is what makes the *ratio* usable; the wall times are not
comparable to the 9.88 s recorded below, which was taken separately.

## The minimizer window: an inline ring buffer (2026-09-07)

The profile above put `SignatureScanner::push` at **31.8%** of a binned
run, the largest single item. The part of it that was a heap-allocated
`VecDeque<(u64, usize)>` is the sliding-window minimum, and the deque
never holds more than `k - m + 1` entries -- at most 32, since `k <= 32` on
this path. That fits inline in the scanner with a power-of-two mask instead
of a heap pointer and wrap arithmetic.

**Isolated** (`minimizer::tests::window_min_ab`, 4,000,000 bases, k=31
m=7, 15 interleaved repeats, two runs):

| | |
|---|---:|
| `VecDeque` | 0.0306 s / 0.0307 s |
| inline ring buffer | 0.0283 s / 0.0281 s |
| | **1.08x / 1.09x** |

**End to end it could not be measured on this host, and the attempt is
worth recording as a negative result.** Two release binaries differing only
in this, 9 and 11 interleaved runs each on the 2.14 GB file:

| run | host load | median ratio | min-to-min |
|---|---:|---:|---:|
| first | 6.3 | 1.104x (ring faster) | 1.164x |
| second | 13.2 | 0.939x (ring *slower*) | 1.038x |

The two disagree in direction. A ~9 s run on a host carrying unrelated
load of 6-13 has a spread (8.5-12.9 s) far wider than an effect this size,
and no number from it is usable. The isolated benchmark is the only
measurement here that means anything, and what it measures is the window,
not the 31.8%: the m-mer roll and the eligibility key are the rest of
`push` and are untouched.

Kept anyway, on three grounds and not on a speed claim: it is consistently
faster in isolation, its output is byte-identical (SHA-256 checked against
the previous binary on real reads), and it arrived with an equivalence
test the `VecDeque` version never had --
`ring_and_deque_windows_agree_on_real_sequence` drives both implementations
through the same push/evict/min sequence across five `(k, m)` pairs and
requires identical answers at every step. Two monotonic deques can agree on
most inputs and still differ on the tie rule, and the tie rule here decides
where super-k-mers break.

## Strategy comparison on Apple silicon (2026-09-05)

A second machine, and the first measurement of all three strategies against
each other on the same input in the same environment.

| | |
|---|---|
| CPU | Apple M3 Pro (`Mac15,6`), 11 threads |
| RAM | 19 GB (18 GiB) |
| OS | macOS (Darwin 25.6) |
| Build | `cargo build --release` |

Input: the same generator and seed as the WSL comparison above
(`scripts/bench/generate_reads_large.py 35000000 30 150 large.fastq 9001`),
2.14 GB, **840,000,000 k-mer occurrences**, `k=31`, `-m 1 -q 0`, default
thread count.

| Strategy | Time | Peak RSS | Distinct k-mers |
|---|---:|---:|---:|
| `memory` (what `auto` selects) | 24.10 s | 7.31 GB | 53,776,394 |
| **`binned`** | **8.41 s** | **3.59 GB** | 53,776,394 |
| `disk` | 40.67 s | 2.13 GB | 53,776,394 |

All three Parquet outputs are **byte-identical** (SHA-256
`5c1a51924554a883...`), which is the property `tests/dual_strategy.rs`
asserts on synthetic input, confirmed here at 840M occurrences.

**The minimizer-partitioned strategy is 2.9x faster than the default and
uses half its memory, and `auto` never selects it.** That is the largest
single speed result in this document, and it comes from code that has been
in the tree, correct and opt-in, since 2026-08-25.

### Scaling in threads and input size

Both strategies, both files, four thread counts. `mid` is the same generator
at seed 4242 (6 Mbp genome, 144,000,000 occurrences).

| File | Strategy | 1 thread | 4 | 8 | 11 |
|---|---|---:|---:|---:|---:|
| mid | binned | 4.26 s / 891 MiB | 1.91 s / 915 | 1.55 s / 923 | 1.48 s / 951 |
| mid | memory | 5.15 s / 785 MiB | 2.16 s / 1657 | 2.22 s / 2320 | 3.61 s / 2216 |
| large | binned | 26.47 s / 3009 MiB | 11.28 s / 3372 | 8.99 s / 3407 | 9.56 s / 3287 |
| large | memory | 48.71 s / 4744 MiB | 20.36 s / 5030 | 19.40 s / 6073 | 21.13 s / 6513 |

Two things this shows that the single-configuration table cannot:

- **`binned`'s memory is flat in threads and the in-memory strategy's is
  not** -- 3,009 -> 3,287 MiB across 1 to 11 threads against 4,744 -> 6,513.
  That is `docs/design-minimizer-counting.md` 3.6's central structural claim
  (the store is a partition, not a per-worker replica), measured.
- **The design's predicted ~3x memory cut did not survive.** The measured
  ratio is **1.78x at 8 threads and 1.98x at 11**, not 3x. The advantage is
  real and large; the published multiple was optimistic, and
  `src/mem_estimate.rs`'s own test now pins the measured figure instead.

These eight runs are what `estimate_binned_peak_bytes` was calibrated
against on the same date -- it had been an explicitly structural,
never-measured model until then, and it turned out to under-predict the
840M-occurrence runs by 26-29% while over-predicting the 144M ones by
11-19%. Both errors are gone; see that function's doc comment and
`binned_estimate_matches_the_measured_runs_it_was_fit_to`.

### Where `binned` loses, and why it is not promoted blindly

The same three strategies on a **low-complexity** input: 1.5M reads of a
single 250 bp conserved sequence with ~1% substitutions (734 MB,
330,000,000 occurrences, only 1,469,132 distinct k-mers) -- an amplicon
panel's shape, the case `docs/design-minimizer-counting.md` 6's R3 risk
named and the reason promotion was declined in the first place.

| Strategy | Time | Peak RSS |
|---|---:|---:|
| `memory` | **0.78 s** | **759 MB** |
| `binned` | 2.67 s | 2,090 MB |

**3.4x slower and 2.8x more memory** -- the exact inversion of the shotgun
result, on an input that is not small. Counts are identical (1,469,132
distinct; byte-identical Parquet).

This is why input *size* cannot gate the promotion, and why the adaptive bin
map (`src/adaptive_bins.rs`) does not by itself resolve it: when the input's
whole signature space is narrower than the bin count, there is nothing for
any packing to spread. What can gate it is the balance the packing actually
achieves on the warm-up sample, which is measurable before counting starts.

### After the promotion: what `auto` does now

`auto` measures that balance on the first 20,000 records and refuses binned
above `pipeline::MAX_ACCEPTABLE_BIN_SKEW` (3.0). The two populations are a
factor of 27 apart, so the threshold is not a fine judgement:

| input | measured balance | strategy `auto` picks | time | peak RSS |
|---|---:|---|---:|---:|
| 2.14 GB shotgun | 1.20x | `binned` | **9.88 s** | 3.43 GB |
| 392 MB shotgun | 1.21x | `binned` | 1.55 s | 978 MB |
| 734 MB amplicon-shaped | 32.71x | `in-memory` | 0.85 s | 693 MB |

Against the same runs before the promotion, on the same machine and files:
the 2.14 GB shotgun took **24.10 s at 7.31 GB**, so the default path is now
**2.4x faster at 47% of the memory**, with byte-identical Parquet output.
The amplicon case is faster too (0.85 s against 2.20 s), for a different
reason: `available_system_memory_bytes` had no macOS implementation, so
every Mac was budgeted at a flat 4 GiB regardless of its real memory, and
that budget alone was routing a run that fits comfortably in RAM to the disk
strategy. It now reads `hw.memsize`.

Both numbers below the headline are worth keeping in view: the sampling
costs one bounded prefix read (20,000 records, replayed into the run rather
than re-read), and the balance it measured is printed on the `Strategy Used`
line so a run that was announced as `binned` and finished as something else
says why.

## Where a binned run's time actually goes (2026-09-05)

Profiled rather than reasoned about, because two reasoned guesses had
already been wrong. `/usr/bin/sample` over a live run on `large.fastq`
(840,000,000 k-mer occurrences), leaves grouped by what they do, as a share
of non-idle samples:

| | share of work |
|---|---:|
| sorting (phase 2's per-bin sort) | **39.5%** |
| the rest of `BinStore::finish` | 22.2% |
| minimizer computation (phase 1) | 19.6% |
| writing super-k-mers (phase 1) | 11.5% |
| FASTQ parsing | 1.6% |
| `memcpy` | 1.2% |
| allocator | 0.8% |

And the same run split by phase, timed directly rather than sampled:

| threads | total | phase 1 | per-bin counting | cross-bin merge |
|---:|---:|---:|---:|---:|
| 1 | 26.33 s | 21.63 s | 2.94 s | 1.76 s |
| 4 | 12.18 s | 7.55 s | 2.92 s | 1.71 s |
| 8 | 10.24 s | 5.65 s | 2.76 s | 1.83 s |
| 11 | 10.22 s | 5.53 s | 2.81 s | 1.88 s |

Three findings, two of them things this document previously got wrong:

- **Parsing is not the bottleneck, and a guess said it would be.** After
  the binned promotion made counting 2.4x faster, the single-threaded
  producer *should* have grown to a large share of the run --
  `design-minimizer-counting.md`'s own step-0 gate puts the threshold for
  "build parallel parsing instead" at ~36%. Measured with
  `examples/parse_only_ceiling.rs`: **0.76 s for 2.3 GB, 3,030 MB/s, 7.7%**
  of a 9.9 s run. Not close.
- **Parquet export costs nothing measurable.** Writing 478 MB against
  writing 456 bytes (`-m 100`) is the same wall clock. The 17% figure this
  document reports for the in-memory strategy on the older machine does not
  carry over.
- **Per-bin counting did not respond to `--threads` at all** (2.94 s at
  one thread, 2.81 s at eleven). `BinStore::finish` parallelised over the
  *global* rayon pool, sized by core count, while `--threads` sized only
  phase 1's worker pool. **Fixed 2026-09-06**, and it was worse than a flag
  that under-delivers: `mem_estimate::estimate_binned_peak_bytes`'s phase-2
  term is `threads * per-bin transients`, so with `-t 1` the model counted
  one bin in flight while eleven ran -- an **under-prediction**, the one
  failure that model is calibrated never to commit. Phase 2 now runs in a
  pool sized to `--threads`, built only when that differs from the global
  pool so the default run pays nothing for it.

  Peak RSS moved as a result, and **the first attempt to record how much
  was wrong.** Those figures were one run per configuration. Repeating each
  three times on 2026-09-07 showed every one of them had been 20-55% low:

  | occurrences | threads | single run (2026-09-06) | worst of three (2026-09-07) |
  |---:|---:|---:|---:|
  | 144M | 1 | 602 MiB | 785 MiB |
  | 144M | 8 | 827 MiB | 985 MiB |
  | 144M | 11 | 858 MiB | 1,008 MiB |
  | 840M | 1 | 2,772 MiB | 3,271 MiB |
  | 840M | 4 | 2,222 MiB | 2,707 MiB |
  | 840M | 11 | 1,769 MiB | 3,239 MiB |

  That mattered, and not only for tidiness: against the honest numbers the
  calibration factor of 1.195 **under-predicted at 144M occurrences on 8
  and 11 threads** (by 1.6% and 1.3%) -- the one failure that model exists
  to make impossible. It is now 1.24, the smallest factor that leaves no
  measured *run* above the prediction, fit against the worst of three at
  every point rather than the median. A bound fit to a median is exceeded
  by half the runs it bounds.

  Two claims made here on 2026-09-06 are withdrawn with it. "Peak memory no
  longer rises with thread count" was drawn from single runs: at 144M
  occurrences it plainly does (785 -> 1,008 MiB from 1 to 11 threads), and
  at 840M it falls by ~15% rather than the 36% one pair of outliers
  suggested. And "over-predicts by up to 94%" was arithmetic on the low
  figure; the real ceiling is 33%.

  Run-to-run spread, for whoever re-measures: six of the eight points vary
  by under 2% between runs, and two vary by 12% and 23%. Repeats are not
  ceremony here -- they are the difference between the two points that move
  and the six that do not.

### The cross-bin merge, parallelised

At 1.88 s on one core against 2.76 s for all of phase 2's counting across
eleven, the merge was the largest serial stretch left. It is now split by
*key range* rather than by source -- each range picks one contiguous slice
out of every sorted bin table (binary search, no scanning), the ranges
merge independently, and they concatenate in order. That keeps the
single-pass property `k_way_merge_sorted_counts`'s doc comment defends
against a pairwise reduction tree, which would rewrite every entry once per
level.

Split points are quantiles of a sample of the real keys, not equal
divisions of the `u64` range: canonical k-mers are not uniformly
distributed, and a low-complexity sample would hand one worker most of the
data -- the same failure mode the bin balance itself had to be measured for.

| | time |
|---|---:|
| sequential merge | 1.88 s |
| parallel merge (11 parts) | 1.06 s |
| + concatenating the parts | 0.40 s |
| **total** | **1.48 s** |

**It parallelises badly, and that is the finding.** Eleven-way splitting
bought 1.8x, not 11x, because the merge reads 645 MB and writes 645 MB and
is bound by memory bandwidth rather than by comparisons. An attempt to
remove the 0.40 s concatenation by splitting the destination into disjoint
slices and copying on every core was **no faster at all**: allocating that
destination with `vec![_; n]` zero-initialises 645 MB, which costs about
what the copy it replaced did. The simpler sequential copy was kept.

End to end the 0.4 s is inside run-to-run variance (runs on this machine
span 9.3-11.0 s), so no end-to-end claim is made from it. The merge's own
before/after is what was measured.

## The `k > 32` engine, and what it costs (2026-09-05)

`src/wide_kmer.rs` + `src/wide_counter.rs` count `33 <= k <= 64` in `u128`
instead of `u64`. Same machine and file as the section above
(`mid.fastq`: 144,000,000 k-mer occurrences, 9,213,849 distinct at k=31).

**Correctness first.** At `k = 31` both engines are defined, and they
return the same count -- 9,213,849 distinct, byte for byte the same answer
-- which is what `--engine wide` exists to let anyone check on their own
data rather than trusting the test suite.

| engine | k | time | peak RSS |
|---|---:|---:|---:|
| narrow (`u64`) | 31 | **1.87 s** | **750 MiB** |
| wide (`u128`), 1 thread | 31 | 5.62 s | 793 MiB |
| wide, 4 threads | 31 | 2.54 s | 2,002 MiB |
| wide, 11 threads | 31 | 3.36 s | 4,152 MiB |
| wide, 11 threads | 41 | 4.57 s | 4,130 MiB |
| wide, 11 threads | 63 | 3.94 s | 4,298 MiB |

Three things worth reading off it:

- **The width costs about 3x in time** at one thread, which is `u128`
  arithmetic and is the price of the capability, not a defect.
- **Larger k is not slower.** k=63 finishes faster than k=31 because a
  150-base read yields 88 k-mers instead of 120 -- fewer instances to sort.
- **Four threads beats eleven**, on both axes (2.54 s / 2,002 MiB against
  3.36 s / 4,152 MiB). Peak memory scales with `threads x distinct` because
  every worker holds its own table, and the wide engine has no `binned`
  equivalent to share one. The time inversion was not predicted; the likely
  cause is the sequential fold in `merge_all`, which was not measured in
  isolation and so is not claimed.

### Two guesses the measurements killed

The first working version of the wide counter cost **4,438 MiB at one
thread**. Two hypotheses about why were wrong before one was right, and
both were cheap to test:

1. *"Memory scales with `threads x distinct`, as it does for the narrow
   in-memory strategy."* Falsified in one run: one thread cost 4,438 MiB
   and eleven cost 4,713 MiB.
2. *"The raw instance buffer is not actually being bounded."* Falsified by
   a test asserting its capacity stays near its threshold across 40M
   insertions -- which is now a permanent regression test, since a buffer
   that silently grows would put peak memory back on input size.

What it actually was: memory scaled with *occurrences* at ~26 bytes each,
the signature of repeatedly allocating and freeing a large buffer rather
than of holding one. `compact` allocated a fresh merge target on each of
~72 compactions, and the allocator does not hand freed pages back promptly.
Reusing two buffers that alternate took it to 1,187 MiB; storing the table
as parallel arrays rather than `Vec<(u128, u32)>` -- 32 bytes of which 12
are alignment padding -- took it to 793 MiB, at which point the wide engine
costs about what the narrow one does per thread.

## Worker-buffer bound: measured effect by input shape

`KmerCounter` bounds each worker's raw k-mer buffer at 2,000,000 buffered
instances (`RAW_FINALIZE_THRESHOLD` in `src/counter.rs`); crossing the
threshold triggers an eager sort-and-compact. The bound is not a flat win
across every input shape, and both directions were measured:

**Where it helps -- high coverage of a small, low-diversity template.**
Measured on three FASTQ files built by literally repeating one file's
content 1x / 4x / 16x (so the distinct-k-mer set -- 1,175,574 -- is
identical across all three by construction, and only occurrence count
changes), at the default 8 worker threads on this machine:

| Occurrences | Peak RSS, unbounded buffer | Peak RSS, current (bounded) |
|---:|---:|---:|
| 7,999,788 | 122.7 MB | 111.8 MB |
| 31,999,152 | 250.6 MB | 272.7 MB |
| 127,996,608 | 1.02 GB | 353.4 MB |

The first two rows still show real occurrence-driven growth in the current
code (each worker's ~1.0M / ~4.0M-occurrence share brackets the 2,000,000
threshold, so the buffer behaves exactly as it always did in that range);
only the third row, where each worker's ~16M-occurrence share is well past
the threshold, shows the flattening the bound exists to produce -- 4x the
occurrences for 30% more RAM, not roughly 4x more RAM.

**Where it does not help -- a large, genuinely diverse sample** (new
distinct k-mers still arriving steadily throughout the run, not a small set
repeating). Measured on the 2.14 GB / 53,774,150-distinct- /
839,987,618-occurrence benchmark file from the KMC3/FASTK comparison above,
two runs each, isolated (nothing else competing for the machine): unbounded
buffer 6.49 GB / 109.9 s and 8.81 GB / 87.6 s; current (bounded) 7.10 GB /
173.1 s and 7.66 GB / 125.3 s. The peak-RSS ranges overlap -- this
machine's run-to-run variance at this scale is wide enough that no clean
before/after memory delta can be claimed either way -- but every
bounded-buffer run was slower than every unbounded one. The reason is
structural, not noise: at ~105M occurrences per worker, the bounded buffer
triggers roughly 52 eager compactions per worker instead of one, and each
one re-merges the *entire* running table built so far, not just the newly
arrived slice -- an O(running table size) copy, repeated every time,
because the table itself keeps growing across the whole run when the input
is this diverse. The fix trades well when a worker's finalized table stays
small and stable; it trades poorly when that table keeps growing for the
whole run, which is exactly what a high-diversity sample does. The
threshold is a starting point rather than a value tuned across input
shapes.

## QC report on the benchmark dataset

What FastDNA's own QC report looks like on the small benchmark dataset:

```json
{
  "total_reads": 200000,
  "total_bases": 30000000,
  "q20_pct": 99.63,
  "q30_pct": 81.98,
  "gc_content_pct": 50.11
}
```

## Finding your machine's in-memory ceiling

[`scripts/bench/memory_ceiling.py`](../scripts/bench/memory_ceiling.py)
exists to find, on your machine, the input size past which the in-memory
strategy no longer fits -- the point where `--strategy auto` (with an
honest `--max-ram`) starts choosing the disk strategy instead. On the 16 GB
test machine above, the forced-in-memory process died somewhere between
2.14 GB and 3.98 GB of input.
