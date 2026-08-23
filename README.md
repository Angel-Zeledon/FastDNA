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

**Correctness cross-check**, `k=31`, canonical k-mers, before any timing was
recorded: FastDNA reports **1,423,950** distinct canonical 31-mers across
**23,999,892** total; all three independent Python implementations below
(which do not apply quality trimming -- see their docstrings) agree at
**1,423,964** distinct across **24,000,000** total. The 14-k-mer, 0.001%
difference is exactly what FastDNA's quality trimming removing a handful of
low-quality read tails would produce, not a counting discrepancy -- four
independently-written implementations landing within 0.001% of each other is
itself a correctness check worth having.

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

That puts FastDNA at roughly **20-25x faster than any of the three Python
approaches on a single thread**, and **45-60x faster using the 8 threads
this laptop has** -- against a NumPy implementation that is itself already
vectorized, not naive. The honest reason the NumPy baseline doesn't win is
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
numbers above show it -- e.g. the two `Counter` runs differ by 22%. That
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
JSON report, not just the in-memory count. **We did not benchmark against
established C/C++ k-mer counters** (KMC3, Jellyfish, DSK): they require a
Linux/conda environment this sandboxed Windows session didn't have, and we
would rather say that plainly than paste in numbers from their papers and
imply they were measured here. If you have access to one, the same
`bench_small.fastq` file (or your own run of `generate_reads.py`) is a fair
input to point it at.

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
- Comparing, hashing, and storing a k-mer is comparing, hashing, and storing
  one machine word.
- The entire count table is `HashMap<u64, u32>` -- a flat array of 12-byte
  entries, not a forest of heap-allocated string buckets.

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

### 4. Exact counting with a fast, non-cryptographic hash

Because the key *is* the k-mer (not a hash of it), counting is **exact** --
there is no probability of two different k-mers colliding into the same
count, unlike Bloom-filter or Count-Min-Sketch-based counters some tools use
to bound memory on very large datasets (the codebase has an unused
`CountMinSketch` in `src/cms.rs` for a possible future bounded-memory mode,
but it isn't wired into the counting pipeline today -- worth being explicit
about, since it would be easy to imply otherwise). The hash map itself uses
`rustc-hash`'s `FxHashMap` instead of Rust's default `SipHash`: `SipHash` is
cryptographically strong (resistant to hash-flooding attacks, which matters
for a public web server's hash maps) at the cost of more CPU cycles per
hash; `FxHash` is a simple multiply-rotate hash that is 3-10x cheaper and
unsuitable for adversarial input. FASTQ files aren't adversarial input --
the caller already chose to trust them -- so this is the right trade.

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

A Python progress callback is arbitrary user code, called from inside a
Rayon worker thread. If it raises, PyO3 turns that into a Rust panic; if
that panic were allowed to unwind through the worker and across the FFI
boundary back into the Python interpreter, that is **undefined behaviour**,
not a catchable error, per Rust's FFI rules -- and would very likely crash
the whole Python process with no usable traceback. Every worker's body is
wrapped in `catch_unwind`, converting a caught panic into
`FastDnaError::Internal` (a `RuntimeError` in Python) instead. This is also
why `[profile.release] panic = "unwind"` is pinned explicitly in
`Cargo.toml`, with a comment: `catch_unwind` is a no-op under
`panic = "abort"`, and wheel-build profiles routinely default to `abort` to
shrink binaries -- exactly the kind of change nobody would think to connect
to "a panicking Python callback now kills the interpreter."

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
- Pressing **Ctrl-C** during a call raises `KeyboardInterrupt` promptly, not
  after the run finishes.

`KmerCounts` (the return value):

| Member | Type | |
|---|---|---|
| `.table` | `pyarrow.Table` | zero-copy; columns `kmer_u64` (`uint64`), `kmer_sequence` (`string`), `frequency` (`uint32`) -- identical schema to the Parquet files the CLI writes |
| `.qc` | `dict` | read/base-level QC: `total_reads`, `total_bases`, `q20_bases`, `q30_bases`, `gc_bases`, `gc_content_pct`, `q20_pct`, `q30_pct` |
| `.total_kmers` | `int` | total k-mer instances counted (the normalization basis) |
| `.distinct_kmers` | `int` | distinct canonical k-mers kept after `min_count`/`max_count` |
| `.k` | `int` | the `k` this result was built with |
| `.spectrum()` | `dict` | `{depth: number of distinct k-mers observed at that depth}` -- the frequency histogram |
| `.suggest_min_count()` | `int` | the `min_count` detected from *this sample's own* frequency spectrum (see below) -- also `len(result)` for `.distinct_kmers` |

**Why `suggest_min_count()` exists**: a sequenced sample's frequency
spectrum has two peaks -- a large one at frequency 1-2 (sequencing errors)
and another at the real coverage depth, with a valley between them. The
correct `min_count` threshold sits in that valley, and **it's different for
every sample** -- there's no universally-correct default like 5.
`suggest_min_count()` walks the spectrum, follows the initial descent (the
tail of the error peak) until it reverses, and returns that reversal
point's depth -- turning "eyeball a histogram" into a function call. It
falls back to a conservative default (`2`) when the spectrum doesn't carry
enough signal to find a real valley (too few distinct depths, or a shape
that never stops/starts decreasing).

```python
r = fastdna.count("sample.fastq.gz", k=31)
print(r.suggest_min_count())      # e.g. 4
clean = fastdna.count("sample.fastq.gz", k=31, min_count=r.suggest_min_count())
```

### `fastdna.peek(path, *, n_reads=10_000) -> Preview`

Samples just the first `n_reads` records -- milliseconds, without reading
the rest of the file -- and reports enough to pick a sane `k` *before*
committing to a long run:

| Member | |
|---|---|
| `.n_reads_sampled` | reads actually read (may be less than `n_reads` on a short file) |
| `.read_length` | `(min, median, max)` among the sampled reads |
| `.gc_content` | fraction, `0.0..=1.0` |
| `.estimated_distinct_kmers` | distinct canonical k-mers within the sample itself, at `suggest_k()` |
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
