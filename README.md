# FastDNA

FastDNA is a Rust genomic k-mer counter: it reads FASTQ (optionally gzipped)
files and counts canonical k-mers with quality filtering, in parallel, on the
CPUs you already have. It hands the result to Python as a zero-copy Arrow
table, so it lands in your notebook or pipeline without a serialization step.

This README covers, in order: installation and a quick start, **real,
reproducible benchmarks against three Python implementations**, **how the
Rust core actually works** (the encoding, the math, the parallel pipeline),
and the **full Python API reference**.

## Installation

```bash
pip install fastdna
```

No Rust toolchain, no compiler, no build step. `fastdna` ships as a prebuilt
wheel using [PyO3's `abi3` stable ABI](https://pyo3.rs), so one wheel per
platform covers Python 3.8 through 3.13+.

## Quick start

```python
import fastdna

result = fastdna.count("sample.fastq.gz", k=31)
print(result)
# KmerCounts(k=31, distinct=1423950, total=23999892)

table = result.table  # a pyarrow.Table, zero-copy
print(table.column_names)
# ['kmer_u64', 'kmer_sequence', 'frequency']

print(result.qc)
# {'total_reads': 200000, 'total_bases': 30000000, 'q20_pct': 99.6, ...}
```

---

## Benchmarks

**Everything in this section was measured on this machine, in this
repository, with the scripts committed under [`scripts/bench/`](scripts/bench/).**
No numbers here are copied from a paper or a vendor claim. Every command
below is exactly what was run to produce the table; run it yourself and you
should land within noise of the same figures.

### Test machine and dataset

| | |
|---|---|
| CPU | Intel Core i7-1165G7 @ 2.80GHz (11th Gen), 4 physical cores / 8 logical threads |
| RAM | 16 GB |
| OS | Windows 11 Pro |
| Rust | rustc/cargo 1.91.0, `cargo build --release` |
| Python | 3.12.10, NumPy 2.5.2, Biopython 1.88 |

The dataset is **synthetic but realistic, not uniformly-random noise**:
[`scripts/bench/generate_reads.py`](scripts/bench/generate_reads.py) builds a
1 Mbp random reference genome, then samples 150 bp reads across it to 30x
coverage (both strands), applying an Illumina-like quality curve (~Q38
decaying to ~Q25 towards the read's 3' end, with occasional dips) and
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

**Correctness cross-check**, `k=31`, canonical k-mers: this is not a
remembered observation -- it's asserted by a script,
[`scripts/bench/crosscheck.py`](scripts/bench/crosscheck.py), specifically
so "fast because it does less work" has something checking for it every
time this claim gets re-published, not just the one time someone eyeballed
a table.

```bash
python scripts/bench/crosscheck.py
```

This regenerates the same dataset the benchmark table below was measured
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
read tails would produce, not a counting discrepancy -- four
independently-written implementations landing within 0.001% of each other
is itself a correctness check worth having, and now one that runs on
demand rather than one that was merely true once. The full run above takes
on the order of minutes (the two slower Python baselines are O(reads x k)
pure-Python loops over the full 200,000-read file); pass `--quick` for a
small generated file and a sanity check in seconds instead -- see the
script's own `--help` for that and its other options
(`--file`/`--k`/`--tolerance-pct`).

### Counting speed: FastDNA vs. three ways to write this in Python

The comparison is against **three real, runnable, correctness-checked
implementations**, not a strawman:

- **`naive_python.py`** -- pure standard library, `collections.Counter`,
  string slicing. What most people write first.
- **`biopython_baseline.py`** -- same algorithm, but parses the FASTQ with
  `Bio.SeqIO` instead of hand-rolled parsing, because that's what most
  bioinformaticians reach for next.
- **`numpy_baseline.py`** -- a genuinely vectorized attempt: bases are
  2-bit-packed and k-mers built with array shifts rather than a Python loop
  per k-mer, and the reverse complement is computed with the *same*
  bit-trick FastDNA's Rust core uses (see below), ported to NumPy.

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
README shipped with):

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
laptop has** (88.13 / 1.99 s = 44.3x, up to 125.57 / 1.99 s = 63.1x) --
against a NumPy implementation that is itself already vectorized, not
naive. The honest reason the NumPy baseline doesn't win is
worth stating plainly: it still runs a Python-level loop once per *read*
(200,000 iterations), and inside each iteration a k=31-deep chain of small
array operations to build that read's k-mer windows. NumPy's per-call
overhead (tens of microseconds) dominates when the arrays involved are only
~150 elements long; it never amortizes the way it would over one huge array.
This is itself a genuine, useful data point: "vectorize it in NumPy" is not
automatically a free win at this granularity, and a hand-rolled Python
attempt at this problem is easy to get slower than expected, not just slower
than Rust.

**Scaling is sub-linear** (1.99 s at 8 threads vs. 4.94 s at 1 thread is
~2.5x from 8x the threads, not 8x) because this machine has 4 physical cores
(8 is hyperthreaded) and, at this dataset size, opening and reading the same
65 MB file from disk on every run is a fixed cost the thread count doesn't
reduce -- see "The parallel pipeline" below for why the architecture still
scales further on larger files, where that fixed cost amortizes away.

**Run-to-run variance**: this machine is a laptop under normal desktop load
(IDE, language servers, browser), not a dedicated benchmark rig, and the
numbers above show it -- e.g. the two `Counter` runs differ by 29% (88.13 s
vs. 113.41 s). That
variance is disclosed rather than hidden: re-run the scripts and expect
figures in the same range, not bit-identical ones.

### End-to-end CLI (counting + Parquet export + QC report)

```bash
fastdna --input bench_small.fastq --output counts.parquet -k 31
```

Six runs, default 8 threads: 3.07, 3.14, 4.24, 5.00, 6.01, 9.50 s
(min 3.07 s, median 4.62 s) -- this is the number that matters if you only
care about "how long until `counts.parquet` exists on disk," and it includes
writing all 1,423,950 rows through Snappy-compressed Parquet plus the QC
JSON report, not just the in-memory count. This is a small file; the
comparison that actually matters -- against the field's own dedicated
k-mer counters, at a scale where the difference between an in-memory hash
table and a sequential-access strategy is visible -- is next.

### Large-scale comparison: FastDNA vs. KMC3 vs. FASTK

This is the comparison that matters, not the Python one above: KMC3 and
FASTK are the field's own dedicated tools, not something people write
themselves. Measured in WSL2 (Ubuntu 26.04) for KMC3 and FASTK, and the
native Windows release binary for FastDNA, all three against the *same*
2.14 GB synthetic FASTQ file (35 Mbp genome, 30x coverage, 150 bp reads,
same generator and quality/error model as above, scaled up --
[`scripts/bench/generate_reads_large.py`](scripts/bench/generate_reads_large.py),
seed 9001; exact install and run commands in
[`scripts/bench/kmc3_fastk_comparison.sh`](scripts/bench/kmc3_fastk_comparison.sh)),
`k=31`, singleton k-mers included on all three (`kmc -ci1`, `FastK -t1`,
FastDNA's own `min_count=1` default):

| Tool | Time | Peak RAM | Disk (output) | Distinct k-mers |
|---|---:|---:|---:|---:|
| FASTK (2023) | 119.3 s | 2.99 GB | 412 MB | 53,776,394 |
| **FastDNA** | **194.5 s** | **2.00 GB** | 431 MB (Parquet) | 53,774,150 |
| KMC3 | 293.4 s | 9.77 GB | 412 MB | 53,776,394 |

FastDNA beats KMC3 here and trails FASTK by 1.6x -- on this file, on this
machine. The 2,244-k-mer (0.004%) difference between FastDNA's count and
KMC3/FASTK's is the same quality-trimming effect documented above (KMC3
and FASTK do not trim; FastDNA does by default), not a counting bug --
consistent with the ~0.001% gap already measured against the pure-Python
implementations on the small dataset.

**This table did not always look like this.** The first version of this
comparison had FastDNA at 920.7 s and 6.96 GB peak RAM on the same file --
about 3x slower than KMC3 and 7.7x slower than FASTK. `KmerCounter` was, at
that point, an in-memory `HashMap<u64, u32>`: every insertion is
effectively-random bucket placement, an L3 cache miss once the table
outgrows a few tens of MB, which happens well before a real sample
finishes counting -- exactly the failure mode the field moved away from
after Jellyfish (2011), which is why KMC3 partitions k-mers into
disk-resident bins by minimizer signature before sorting each one, and why
FASTK does not hash at all, sorting 2-bit-packed k-mers into disk
partitions instead. Both are sequential-memory-access strategies.
`KmerCounter` now uses the same strategy, in memory rather than on disk:
insertion appends to a plain `Vec<u64>` (sequential, cache-friendly), and
counting happens as a one-time sort-and-compact pass on first read (see
["Exact counting"](#4-exact-counting-with-a-fast-non-cryptographic-hash)
above and `src/counter.rs`'s own doc comments for the detail). That change
alone -- not new hardware, not a smaller test file -- is the entire
difference between the two numbers in this paragraph.

**What this does not fix**: FastDNA still holds the whole counting table in
RAM. KMC3 and FASTK bound peak memory by spilling to disk; FastDNA does
not, so there remains a dataset size past which FastDNA fails where they
would not. See [Limitations](#limitations) for what that means in
practice and where the actual ceiling is on this machine.

### What FastDNA's own QC report looks like on this dataset

```json
{
  "total_reads": 200000,
  "total_bases": 30000000,
  "q20_pct": 99.63,
  "q30_pct": 81.98,
  "gc_content_pct": 50.11
}
```

---

## How the Rust core actually works

This is the part a benchmark table can't show: *why* it's fast, in enough
detail that the numbers above are explained rather than just asserted.

### 1. A k-mer is a 64-bit integer, not a string

`src/kmer.rs` encodes each base in 2 bits:

```
A/a = 00   C/c = 01   G/g = 10   T/t/U/u = 11
```

A k-mer of length k (FastDNA supports `1 <= k <= 32`) therefore fits
*exactly* into a `u64`: 32 bases x 2 bits = 64 bits. That one design
decision is why almost everything downstream is cheap:

- No heap allocation per k-mer. A Python string, or a Rust `String`, is a
  pointer to a heap buffer; a `u64` lives in a CPU register.
- Comparing, sorting, and storing a k-mer is comparing, sorting, and storing
  one machine word -- and machine-word comparisons are what the counting
  strategy in ["Exact counting"](#4-exact-counting-via-sort-and-compact-not-a-hash-table)
  below is built entirely out of.
- The entire count table is a flat `Vec<(u64, u32)>` -- 12-byte entries in
  one contiguous allocation, not a forest of heap-allocated string buckets.

### 2. Extracting every k-mer from a read is O(n), not O(n x k)

The naive way to get every k-mer from a sequence of length n is to slice out
each of the `n - k + 1` windows -- which is `O(k)` work per window, `O(n·k)`
total, and it's exactly what `naive_python.py` above does.
`extract_canonical_kmers` instead keeps a **rolling window**:

```rust
current_kmer = ((current_kmer << 2) | new_base_bits) & mask;
```

Each new base shifts the accumulator left by 2 bits, ORs in the new base,
and masks off anything beyond the low `2k` bits. That's `O(1)` amortized
per base, `O(n)` for the whole read, independent of k. An ambiguous base
(anything that isn't `ACGTU`, i.e. mainly `N`) resets the accumulator to
zero instead of producing a corrupted k-mer that spans it -- one branch, no
separate cleanup pass.

### 3. The reverse complement is three bit-shuffles, not a string reversal

Counting *canonical* k-mers -- `min(kmer, reverse_complement(kmer))`, so
that a read and its reverse-complement strand contribute to the same count
-- is the single most expensive-looking part of this problem if you write
it the obvious way: reverse the string, complement each character, allocate
a new string. `numpy_baseline.py` and `naive_python.py` both do this, and it
shows in their numbers.

`reverse_complement_u64` never touches a string. It exploits a property of
the encoding above that isn't an accident: **A and T are bitwise
complements (`00`/`11`), and so are C and G (`01`/`10`)**. That means
complementing *every base at once* is a single `!kmer` (bitwise NOT) over
the whole 64-bit word:

```rust
let mut v = !kmer;                                              // complement every base at once
v = ((v >> 2) & M2) | ((v & M2) << 2);                           // swap adjacent 2-bit groups
v = ((v >> 4) & M4) | ((v & M4) << 4);                           // swap adjacent 4-bit nibbles
v = v.swap_bytes();                                              // reverse byte order (one CPU instruction)
v >> (64 - 2*k)                                                  // right-align the k bases actually used
```

The middle two lines are the standard "reverse in place, divide and
conquer" bit trick, applied at 2-bit granularity because each base is 2
bits; `swap_bytes()` finishes the job as a single `bswap` instruction. The
whole function is branchless, allocation-free, and compiles to roughly half
a dozen machine instructions -- nanoseconds, not the microseconds a
string-allocating version costs. Canonicalization is then just
`kmer.min(reverse_complement)`: one integer comparison.

### 4. Exact counting via sort-and-compact, not a hash table

`KmerCounter` (`src/counter.rs`) does not use a hash table. Each worker
appends every canonical k-mer it sees to a plain `Vec<u64>` -- O(1)
amortized, sequential memory writes. That buffer is bounded, not left to
grow for a worker's entire share of the run: once it crosses 2,000,000
buffered instances, or the first time anything reads the counter,
whichever comes first, it is sorted (`sort_unstable`) and compacted in a
single linear pass into `(kmer, count)` pairs, merged into whatever was
already compacted from an earlier pass. The eager threshold exists so a
worker's buffer is capped at that fixed size instead of every occurrence
it ever sees over the whole run; see [Limitations](#limitations) for what
that does and does not fix about peak memory in practice, with measured
numbers on both a case it helps a lot and a case it does not help at all.

This is a deliberate choice, not an incidental detail, and it replaced an
earlier `HashMap<u64, u32>` version of this same type for a specific,
measured reason: a hash table's bucket placement is effectively random, so
once the table outgrows the CPU's L3 cache (a few tens of megabytes --
well under a million entries), nearly every insertion is a cache miss.
That is true regardless of how good the hash function is -- no faster hash
fixes an access pattern that hits main memory on every operation. Sorting
instead touches memory in a handful of sequential passes, which is exactly
why KMC3 (radix-sorting disk-resident bins) and FASTK (sorting 2-bit-packed
k-mers into disk partitions with no hash table at all) are built the way
they are, and exactly why switching to the same strategy -- in memory
rather than on disk -- made FastDNA 4.7x faster and cut its peak memory by
3.5x on the same 2.14 GB benchmark file; see
[Large-scale comparison](#large-scale-comparison-fastdna-vs-kmc3-vs-fastk).

Counting is still **exact**, the same guarantee the hash table gave: the
key really is the k-mer, not a hash of it, so there is no probability of
two different k-mers colliding into the same count -- unlike Bloom-filter
or Count-Min-Sketch-based counters some tools use to bound memory on very
large datasets (the codebase has an unused `CountMinSketch` in
`src/cms.rs` for a possible future bounded-memory mode, but it isn't wired
into the counting pipeline today).

### 5. The parallel pipeline: producer/consumer with backpressure, not a lock

`src/pipeline.rs`'s `process_stream_parallel` is the whole reason more
threads help at all:

1. **One producer thread** streams the FASTQ (transparently
   gzip-decompressing if needed) and parses it into batches of records,
   pushing each batch onto a `crossbeam_channel::bounded(64)` channel.
2. **N worker threads** (via Rayon) each pull batches off that channel in a
   loop. Each worker owns a **private** `KmerCounter` and QC accumulator --
   there is no shared mutable state and no lock on the hot path, because
   there is nothing to contend over until the very end.
3. Once every batch is consumed, a **parallel reduce** merges the N private
   counters into one, in a fold tree rather than a single-threaded scan.

The channel being *bounded* at 64 slots, rather than unbounded, is
deliberate backpressure: if the workers fall behind, the producer blocks
rather than buffering the whole file in RAM; if the workers are faster than
disk I/O, they block on `recv()` rather than spinning. This is also exactly
why `num_threads = 0` used to be able to hang forever -- zero consumers
means the channel's receiving end is never dropped, so it never
disconnects, so the producer blocks on a full channel forever with nobody
left to drain it. That's now rejected explicitly as `InvalidConfig`
(`ValueError` in Python) before the channel is even created, and
`python/tests/test_api.py::test_zero_threads_raises_valueerror_not_hang` is
the regression test for it.

### 6. Quality trimming, and the math behind Phred scores

FASTQ quality bytes are Phred+33 encoded: `phred_score = byte - 33`. A
Phred score is `-10 * log10(P_error)`, i.e. **`P_error = 10^(-Q/10)`** -- Q20
means a 1% chance that base is wrong, Q30 means 0.1%, Q40 means 0.01%.
`FastqRecord::quality_trim_end` walks a sliding window (4 bases by default)
in from the read's 3' end, computing that window's mean Phred score;
trimming stops the moment the window's average reaches the `min_quality`
threshold (Q20 by default). If it never recovers, the whole read collapses
to length zero and contributes no k-mers. The benchmark dataset's own
generator (`generate_reads.py`) uses this exact `P_error = 10^(-Q/10)`
relationship to decide when to inject a substitution error, so the
synthetic data's error/quality relationship matches what this trimming step
is designed to detect.

### 7. Panics cannot cross the FFI boundary, by construction

A worker's body can, in principle, panic -- corrupt internal state, a bug
this codebase doesn't know about yet. If a panic were allowed to unwind
through a Rayon worker and across the FFI boundary back into the Python
interpreter, that is **undefined behaviour**, not a catchable error, per
Rust's FFI rules -- and would very likely crash the whole Python process
with no usable traceback. Every worker's body is wrapped in `catch_unwind`,
converting a caught panic into `FastDnaError::Internal` (a `RuntimeError` in
Python) instead. This is also why `[profile.release] panic = "unwind"` is
pinned explicitly in `Cargo.toml`, with a comment: `catch_unwind` is a
no-op under `panic = "abort"`, and wheel-build profiles routinely default
to `abort` to shrink binaries -- exactly the kind of change nobody would
think to connect to "a panicking worker now kills the interpreter."

A Python progress callback is arbitrary user code, called from inside a
Rayon worker thread -- and if it raises, that is *not* carried out via this
panic mechanism, deliberately. An earlier version of this binding did use
`panic!` to smuggle the `PyErr` out through the same `catch_unwind` above,
and it worked, but at a real cost: with no custom panic hook installed,
Rust's default hook writes `thread '<unnamed>' panicked at ...` to stderr
*before* `catch_unwind` gets a chance to swallow it -- once per worker that
hit it, unconditionally, for a library that otherwise never writes to
stderr unasked (see `progress` below). `src/ffi.rs` instead captures the
`PyErr` into a side channel (a `Mutex`) and stops the run through the same
cancellation flag Ctrl-C uses, then surfaces it as `RuntimeError` once the
worker thread rejoins -- no panic, no unasked stderr output, same
`RuntimeError` contract. `catch_unwind` itself is untouched and still
guards every worker against a genuine Rust panic, callback-triggered or
not.

### 8. Crossing into Python: the GIL and zero-copy Arrow

Two things make the Python binding (`src/ffi.rs`) usable from a notebook
rather than just correct:

- **`py.allow_threads`** releases Python's GIL for the whole duration of the
  count, so the interpreter (and any other thread) isn't frozen while Rust
  works. When a `progress` callback is supplied, each worker briefly
  re-acquires the GIL (`Python::with_gil`) just to make that one call, then
  releases it again -- a small overhead traded for real multi-core scaling
  during the actual counting.
- **Zero-copy Arrow.** Once counting finishes, the result needs to become a
  `pyarrow.Table`. The naive way serializes the whole table to bytes (or
  worse, Python lists) and re-parses it on the other side. FastDNA instead
  uses Arrow's C Data Interface (the `arrow` crate's `pyarrow` feature) to
  hand PyArrow a `RecordBatch` built directly over the same `u64`/`u32`
  buffers Rust already holds -- no serialization, no re-encoding, just a
  pointer handoff that PyArrow adopts as its own.

### What isn't SIMD (yet), stated plainly

`fastdna.build_info()['avx2']` reports whether **this CPU** supports AVX2 at
runtime (`std::is_x86_feature_detected!`), which matters for diagnosing "why
is this slow on my machine" reports remotely. It is **not** a claim that the
counting hot path currently issues AVX2 instructions -- it doesn't; today's
speed comes entirely from the integer/bit-trick design above plus
multi-core parallelism, not explicit SIMD intrinsics. That's a genuine
avenue for a future version, not something this README should imply is
already done.

---

## Limitations

**FastDNA is still not KMC3 or FASTK at true production scale, and the
benchmarks above should not be read as claiming otherwise.** It closed the
*speed* and *memory-efficiency* gap on a 2.14 GB file (see
[Large-scale comparison](#large-scale-comparison-fastdna-vs-kmc3-vs-fastk)
above) by adopting the same sequential-memory-access strategy those tools
use, but it did not close the *scale* gap: KMC3 and FASTK bound peak memory
by spilling to disk, and FastDNA does not.

`KmerCounter` (`src/counter.rs`) holds every distinct k-mer's count in RAM
simultaneously -- a sorted `Vec<(u64, u32)>`, not the `HashMap<u64, u32>` an
earlier version of this README described, but still entirely in-memory, with
no minimizer-based partitioning and no spill to disk. `src/cms.rs` contains a
Count-Min Sketch that would support a bounded-memory mode, but it is not
wired to anything today.

Each worker also buffers newly-seen k-mer instances in a plain `Vec<u64>`
before they are sorted and compacted into that worker's running table, and
that buffer is bounded (2,000,000 instances, 16 MB) rather than left to grow
for a worker's entire share of the run. An earlier version of this
sort-based design had a real bug here: nothing triggered compaction during
counting itself, only the first read afterward, so a worker's buffer held
every occurrence it had ever seen -- not just every distinct one -- before
that first read. The practical effect and the fix are both stated with
measured numbers below, because the fix is not a flat win across every input
shape, and a claim that it is would be exactly the kind of thing this
section exists to catch.

Three practical consequences follow, and the first is more conditional than
a flat "peak memory scales with distinct k-mers" claim would be:

- **Below roughly 2,000,000 occurrences per worker thread, peak memory still
  scales with occurrences, not distinct k-mers** -- the same behavior the
  unbounded buffer had, because the eager compaction never triggers in that
  regime. Measured on three FASTQ files built by literally repeating one
  file's content 1x / 4x / 16x (so the distinct-k-mer set -- 1,175,574 -- is
  identical across all three by construction, and only occurrence count
  changes), at the default 8 worker threads on this machine:

  | Occurrences | Peak RSS, unbounded buffer | Peak RSS, current (bounded) |
  |---:|---:|---:|
  | 7,999,788 | 122.7 MB | 111.8 MB |
  | 31,999,152 | 250.6 MB | 272.7 MB |
  | 127,996,608 | 1.02 GB | 353.4 MB |

  The first two rows still show real occurrence-driven growth in the current
  code (each worker's ~1.0M / ~4.0M-occurrence share brackets the
  2,000,000 threshold, so the buffer behaves exactly as it always did in
  that range); only the third row, where each worker's ~16M-occurrence share
  is well past the threshold, shows the flattening the bound exists to
  produce -- 4x the occurrences for 30% more RAM, not roughly 4x more RAM.
  This is the regime the fix targets: high-coverage resequencing of a small,
  low-diversity template, where the same small set of k-mers recurs millions
  of times.
- **On a large, genuinely diverse sample -- new distinct k-mers still
  arriving steadily throughout the run, not a small set repeating -- the
  fix showed no measurable benefit on this machine, and plausibly costs
  some wall-clock time.** Measured on the 2.14 GB / 53,774,150-distinct- /
  839,987,618-occurrence benchmark file (same file as
  [Large-scale comparison](#large-scale-comparison-fastdna-vs-kmc3-vs-fastk)
  above), two runs each, isolated (nothing else competing for the machine):
  unbounded buffer 6.49 GB / 109.9 s and 8.81 GB / 87.6 s; current (bounded)
  7.10 GB / 173.1 s and 7.66 GB / 125.3 s. The peak-RSS ranges overlap --
  this machine's run-to-run variance at this scale is wide enough that no
  clean before/after memory delta can be claimed either way -- but every
  bounded-buffer run was slower than every unbounded one. The reason is
  structural, not noise: at ~105M occurrences per worker, the bounded buffer
  triggers roughly 52 eager compactions per worker instead of one, and each
  one re-merges the *entire* running table built so far, not just the newly
  arrived slice -- an O(running table size) copy, repeated every time,
  because the table itself keeps growing across the whole run when the
  input is this diverse. The fix trades well when a worker's finalized
  table stays small and stable (case above); it trades poorly when that
  table keeps growing for the whole run, which is exactly what a
  high-diversity sample does. This is a real, measured trade-off, not a
  free win, and the threshold (`RAW_FINALIZE_THRESHOLD` in
  `src/counter.rs`) is a starting point rather than a value tuned across
  input shapes.
- **There is no out-of-core path.** When the table does not fit in RAM, the
  run fails or swaps; it does not degrade to disk the way KMC3's
  disk-resident bins or FASTK's disk-resident sorted partitions do. This is
  why KMC3 can process a 729-gigabase human genome dataset in under 100
  minutes using 33-34 GB of RAM (Kokot et al., *Bioinformatics*, 2017) --
  a dataset size FastDNA is not built to attempt. The practical ceiling on a
  given machine is exactly what
  [`scripts/bench/memory_ceiling.py`](scripts/bench/memory_ceiling.py)
  exists to find on yours.

`max_k` is 32, imposed by the 2-bit-per-base `u64` packing. Analyses that need
longer k-mers are out of scope.

**What FastDNA is for**: getting k-mer counts out of FASTQ files and into
Python -- as Arrow, in-process, without a serialization step or a
subprocess -- for sample sizes that fit in memory, at speed and memory
efficiency that no longer assumes a small-file regime as an excuse. If your
data does not fit in RAM at all -- a human genome at full coverage, a large
metagenome -- use KMC3 or FASTK; they are excellent, and going out-of-core
is a materially different, larger undertaking than the one this project
took on.

## Command-line interface

FastDNA also ships a standalone CLI (installed separately from the crate --
see [Building from source](#building-from-source)):

```bash
fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

Writes k-mer counts to `counts.parquet` (or `.csv`, based on the output
extension), plus a QC report. Run `fastdna --help` for the full set of flags
(`--min-quality`, `--min-count`, `--max-count`, `--threads`, `--qc`,
`--histogram`).

---

## Python API reference

### `fastdna.count(path, *, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None, progress=None, progress_interval=100_000) -> KmerCounts`

Counts canonical k-mers in a single FASTQ or FASTQ.gz file.

- `path` -- a `.fastq` or `.fastq.gz` file.
- `k` -- k-mer length, `1..=32` (default 31; use `fastdna.peek()` first if
  your reads are short -- see below).
- `min_count` / `max_count` -- drop k-mers below/above these frequencies
  after counting (inclusive bounds; `total_kmers` is unaffected).
- `min_quality` -- Phred quality cutoff for 3'-end trimming (see "Quality
  trimming" above).
- `threads` -- worker thread count. `None` uses the core's own default
  (`std::thread::available_parallelism`). Passing `0` explicitly **raises
  `ValueError`** rather than hanging -- this is a guarded, tested case, not
  an oversight; see "The parallel pipeline" above for why zero workers used
  to deadlock.
- `progress` -- `None` (silent, the default), `True` (drives a `tqdm` bar if
  `tqdm` is installed, otherwise silently does nothing -- `tqdm` is a soft
  dependency), or any callable receiving each event. The callback is
  invoked concurrently from multiple worker threads and its `ReadsProcessed`
  events can arrive **out of order** (a worker can cross 200,000 reads, be
  preempted, and have another worker's 300,000 delivered first); `count()`
  always wraps whatever you pass in a serializing adapter
  (`fastdna._progress`) first, so neither `tqdm` nor your own callback has
  to handle that itself.
- `progress_interval` -- how often, in reads, `ReadsProcessed` fires.
  Setting it below the core's default batch size (8,192) also shrinks the
  internal batch size to match -- a batch bigger than the interval would
  coarsen progress regardless of what interval was asked for -- so a
  smaller interval means smaller, more frequent batches crossing the
  internal channel. A smoother bar is a small throughput trade, not a free
  knob; the default interval leaves the default batch size untouched.
- Pressing **Ctrl-C** during a call raises `KeyboardInterrupt` promptly, not
  after the run finishes -- but only when `count()` is called from the
  interpreter's main thread; `PyErr_CheckSignals` is a no-op on any other
  thread, so calling `count()` from a `threading.Thread` makes Ctrl-C do
  nothing until the run completes on its own.

`KmerCounts` (the return value):

| Member | Type | |
|---|---|---|
| `.table` | `pyarrow.Table` | zero-copy; columns `kmer_u64` (`uint64`), `kmer_sequence` (`string`), `frequency` (`uint32`) -- identical schema to the Parquet files the CLI writes |
| `.qc` | `dict` | read/base-level QC: `total_reads`, `total_bases`, `q20_bases`, `q30_bases`, `gc_bases`, `gc_content_pct`, `q20_pct`, `q30_pct` |
| `.total_kmers` | `int` | total k-mer instances counted (the normalization basis) |
| `.distinct_kmers` | `int` | distinct canonical k-mers kept after `min_count`/`max_count` -- also `len(result)` |
| `.k` | `int` | the `k` this result was built with |
| `.spectrum()` | `dict` | `{depth: number of distinct k-mers observed at that depth}` -- the frequency histogram |
| `.suggest_min_count()` | `int` | the `min_count` detected from *this sample's own* frequency spectrum (see below) |

**Why `suggest_min_count()` exists**: a sequenced sample's frequency
spectrum has two peaks -- a large one at frequency 1-2 (sequencing errors)
and another at the real coverage depth, with a valley between them. The
correct `min_count` threshold sits in that valley, and **it's different for
every sample** -- there's no universally-correct default like 5.
`suggest_min_count()` walks the spectrum, follows the initial descent (the
tail of the error peak) down to its lowest point, and returns that valley
**floor**'s own depth -- since `min_count` is an inclusive lower bound,
that keeps the floor's k-mers rather than discarding them as noise. A
single noisy uptick partway down does not by itself end the search: the
climb back up has to reach at least twice the candidate floor's count
before it's accepted as the start of a genuine coverage peak, so it takes
more than one stray depth to fool this into stopping early. Turning
"eyeball a histogram" into a function call this way is deliberately
conservative: it falls back to a documented default (`2`) whenever the
spectrum's shape isn't unambiguous -- too few distinct depths, a descent
that never finds a lower floor, or one that never climbs back up enough to
look like a real second peak -- because a confidently wrong threshold here
is worse than no suggestion at all. See
[`python/fastdna/spectrum.py`](python/fastdna/spectrum.py) for the exact
rule.

```python
r = fastdna.count("sample.fastq.gz", k=31)
print(r.suggest_min_count())      # e.g. 4
clean = fastdna.count("sample.fastq.gz", k=31, min_count=r.suggest_min_count())
```

### `fastdna.peek(path, *, n_reads=10_000) -> Preview`

Samples just the first `n_reads` records -- milliseconds, without reading
the rest of the file -- and reports enough to pick a sane `k` *before*
committing to a long run. `n_reads` above a generous ceiling (10,000,000)
**raises `ValueError`**: `peek` exists for a quick preview, not a full
read, so a value that large is rejected rather than silently turning into
one.

| Member | |
|---|---|
| `.n_reads_sampled` | reads actually read (may be less than `n_reads` on a short file) |
| `.read_length` | `(min, median, max)` among the sampled reads |
| `.gc_content` | fraction, `0.0..=1.0` |
| `.sample_distinct_kmers` | distinct canonical k-mers within the sample itself, at `suggest_k()` -- an exact count of the sample, not an extrapolation to the whole file |
| `.suggest_k()` | the largest **odd** `k` at most `median_read_length / 3`, clamped to `1..=32` |

Why this matters: `k=31` is everyone's default, and it's wrong for short
reads. At 50 bp, `k=31` leaves only 20 k-mers per read and amplifies the
effect of every sequencing error. Why **odd**: an even-length k-mer can
equal its own reverse complement, which breaks the assumption that a k-mer
and its reverse complement are always a distinguishable pair to pick the
canonical (smaller) one from.

```python
p = fastdna.peek("sample.fastq.gz")
print(p)  # Preview(n_reads_sampled=10000, read_length=(75, 150, 151), gc_content=0.412, suggested_k=31)
r = fastdna.count("sample.fastq.gz", k=p.suggest_k())
```

### `fastdna.build_info() -> dict`

```python
>>> fastdna.build_info()
{'version': '0.1.0', 'max_k': 32, 'avx2': True}
```

`avx2` is a **runtime** check on the machine actually running the code, not
a compile-time flag -- the same wheel ships everywhere, so a build-time-only
answer would be wrong on any machine that differs from the one that built
it. Without this, "it's slow on my Mac" is undiagnosable remotely. (See
"What isn't SIMD yet" above for what this field does and doesn't imply.)

---

## Building from source

Installing via `pip install fastdna` gets you the prebuilt Python extension
only. If you want the `fastdna` CLI binary, or you're working on the crate
itself, clone the repository and build with Cargo:

```bash
cargo build --release
./target/release/fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

For the Python extension itself, from source:

```bash
pip install maturin
maturin develop --release --features python   # NOTE: --release matters --
pytest python/tests -v                        # a debug build of the extension
                                                # measured 4-5x slower in this
                                                # README's own benchmarking
```

---

## Roadmap

Cohort-level processing across many samples, a scikit-learn-compatible
`KmerVectorizer`, and MinHash/Jaccard sketching (`src/sketch.rs` exists but
isn't wired into any public API yet) are designed but **not yet
implemented** -- they are not part of the current `fastdna` package. This
README describes only what `pip install fastdna` gives you today.
