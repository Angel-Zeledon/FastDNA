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
KMC3's 9.06 GB -- and is the only one of the three whose peak memory is
configurable against a budget.

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
