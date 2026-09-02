# FastDNA

FastDNA is a Rust genomic k-mer counter: it reads FASTQ (optionally gzipped)
files and counts canonical k-mers with quality filtering, in parallel, on the
CPUs you already have. It hands the result to Python as a zero-copy Arrow
table, so it lands in your notebook or pipeline without a serialization step.
When a run is predicted not to fit in RAM, the CLI can switch to a
disk-partitioned counting strategy instead of failing -- see
[Memory use and limitations](#memory-use-and-limitations).

## Installation

**FastDNA has not been published yet.** There is no release on PyPI, no
crate on crates.io, and no Bioconda package, so `pip install fastdna` fails
today with `No matching distribution found`. Building from source is the
only way to install it right now, and it is the first path below; the
`pip install` route is documented as what will happen once the first release
lands, not as something that works today. What is still ahead is tracked in
[Roadmap](#roadmap).

### From source (works today)

```bash
git clone https://github.com/Angel-Zeledon/FastDNA
cd FastDNA
pip install maturin
maturin develop --release --features python
```

That produces exactly the extension module a wheel would install, so every
example in this README runs against it. It does need a Rust toolchain, which
a published wheel will not. For the `fastdna` CLI binary, and for running
the test suite against the extension you just built, see
[Building from source](#building-from-source).

### From a wheel (once published)

```bash
pip install fastdna
```

No Rust toolchain, no compiler, no build step. `fastdna` will ship as a
prebuilt wheel using [PyO3's `abi3` stable ABI](https://pyo3.rs), so one
wheel per platform covers CPython 3.8 through 3.13+. CI
([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) already
builds and tests wheels for five platforms: manylinux x86_64, manylinux
aarch64 (cross-compiled, built but not test-executed in CI), macOS x86_64,
macOS arm64, and Windows x86_64 -- it deliberately has no publish step yet,
which is why there is nothing on PyPI to install. The wheels are tagged
`cp38-abi3` (`requires-python = ">=3.8"`), though the CI test matrix
currently runs on Python 3.11, so 3.8 support is declared but not exercised
by CI. A Bioconda recipe is drafted but not yet submitted -- see
[Roadmap](#roadmap).

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

The Arrow table drops straight into pandas -- no copy, no export step:

```python
df = result.table.to_pandas()        # pyarrow.Table -> pandas.DataFrame
print(df.nlargest(5, "frequency"))   # the five most frequent k-mers
```

(`result.to_pandas()` is a shortcut for the same thing, and
`result.to_polars()` does the equivalent for Polars.)

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Benchmarks](#benchmarks)
- [How the Rust core actually works](#how-the-rust-core-actually-works)
- [Command-line interface](#command-line-interface)
- [Memory use and limitations](#memory-use-and-limitations)
- [Python API reference](#python-api-reference)
- [Building from source](#building-from-source)
- [Roadmap](#roadmap)

---

## Benchmarks

**Summary.** All numbers below were measured on a 4-core/8-thread laptop
(i7-1165G7, 16 GB RAM, Windows 11) with the scripts committed under
[`scripts/bench/`](scripts/bench/), using FastDNA's in-memory counting
strategy. Full methodology -- machine details, dataset generation, exact
commands, per-run times, the four-way correctness cross-check, and the
measurement history (including corrections) -- lives in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

**Against three Python implementations** (200,000 reads / 30,000,000 bases,
k=31; all four count canonical k-mers and agree within 0.001%):

| Implementation | Time | Notes |
|---|---:|---|
| Pure Python (`Counter`) | 88.13 s / 113.41 s (two runs) | stdlib only, no quality trimming |
| Biopython (`Bio.SeqIO`) | 121.17 s / 125.57 s (two runs) | real FASTQ parser, no quality trimming |
| NumPy (vectorized) | 123.51 s / 107.11 s (two runs) | 2-bit packing + array ops, no quality trimming |
| **FastDNA, 1 thread** | **4.94 s** (median of 5) | full pipeline incl. quality trimming |
| **FastDNA, 8 threads (default)** | **1.99 s** (median of 5) | |

That is roughly 18-25x faster than the fastest Python baseline on a single
thread, and 44-63x faster on 8 threads.

**Against the field's dedicated k-mer counters** (same 2.14 GB synthetic
FASTQ, k=31, singletons included on all three):

| Tool | Time | Peak RAM | Distinct k-mers |
|---|---:|---:|---:|
| FASTK (2023) | 119.3 s | 2.99 GB | 53,776,394 |
| **FastDNA** (in-memory strategy) | **~101 s** (median of 3) | **8.02 GB** | 53,774,150 |
| KMC3 | 293.4 s | 9.77 GB | 53,776,394 |

FastDNA's in-memory strategy is the fastest of the three on this file and
the most memory-hungry -- that trade, and what the disk strategy changes
about it, is covered in
[Memory use and limitations](#memory-use-and-limitations). The tiny count
differences are FastDNA's quality trimming (the others don't trim), not a
counting discrepancy; [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) has the
cross-check that asserts this on demand.

### Correctness, checked against the field's own tools

Speed only matters if the numbers are right. FastDNA is validated against
the established implementation of each algorithm it borrows, **on real
sequencing data rather than generated input**, by scripts committed under
[`scripts/validation/`](scripts/validation/) and re-run on a schedule by
[`.github/workflows/validation.yml`](.github/workflows/validation.yml):

| what | against | result |
|---|---|---|
| k-mer counting | KMC3 3.2.4 | **exact match** at k=31, 21 and 15 |
| MinHash distances | Mash 2.3 | r = 0.997, no systematic bias |
| genome size | GenomeScope2 2.0.1 | within 0.29% |
| HyperLogLog cardinality | the exact count | −0.26% (bound: ~0.8%) |
| ntCard spectrum | the exact histogram | −0.6% / +2.0% / −1.5% at f1/f2/f3 |

Measured on ENA `DRR002015` (*E. coli*, 2,343,637 reads) and 8 BV-BRC
assemblies; reference tools run from pinned biocontainers, so a rerun later
compares against the same versions. The counting comparison allows **no
tolerance at all** -- with trimming off and singletons kept, both tools solve
the identical problem, so any difference would be a bug. Full methodology,
including what each comparison can and cannot establish, is in
[`docs/validation-real-data.md`](docs/validation-real-data.md).

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
total, and it's exactly what the pure-Python baseline does.
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
a new string.

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
amortized, sequential memory writes. That buffer is bounded: once it
crosses 2,000,000 buffered instances, or the first time anything reads the
counter, whichever comes first, it is sorted (`sort_unstable`) and
compacted in a single linear pass into `(kmer, count)` pairs, merged into
whatever was already compacted from an earlier pass. The bound caps a
worker's buffer at a fixed size instead of every occurrence it ever sees;
its measured effect -- a large win on one input shape, a wash-to-loss on
another -- is documented with numbers in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#worker-buffer-bound-measured-effect-by-input-shape).

This replaced an earlier `HashMap<u64, u32>` version of the same type for a
specific, measured reason: a hash table's bucket placement is effectively
random, so once the table outgrows the CPU's L3 cache (a few tens of
megabytes -- well under a million entries), nearly every insertion is a
cache miss, regardless of how good the hash function is. Sorting instead
touches memory in a handful of sequential passes, which is exactly why KMC3
(radix-sorting disk-resident bins) and FASTK (sorting 2-bit-packed k-mers
into disk partitions with no hash table at all) are built the way they are.
Switching to the same strategy made FastDNA 9x faster on the 2.14 GB
benchmark file; the before/after numbers are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#measurement-history-and-corrections).

Counting is **exact** in both strategies: the key really is the k-mer, not
a hash of it, so there is no probability of two different k-mers colliding
into the same count -- unlike Bloom-filter or Count-Min-Sketch-based
counters some tools use.

### 5. Two counting strategies, and an estimator that chooses between them

The sort-and-compact counter above is the **in-memory strategy**: fastest,
but it holds every worker's table in RAM. FastDNA also has a
**disk-partitioned strategy** (`src/disk_spill.rs`): k-mers are bucketed by
their high bits, each worker spills sorted, deduplicated runs to per-bucket
scratch files, and a final pass merges bucket by bucket -- so only one
bucket's worth of data is resident in memory at a time. Because canonical
k-mers are compared as plain integers and buckets are high-bit ranges,
concatenating the buckets' merge output in bucket order *is* the final
globally sorted table, with no extra sort. (This is deliberately simpler
than KMC3's minimizer-signature partitioning, which balances bucket sizes
better; the header comment in `src/disk_spill.rs` explains the trade.) The
result is identical to the in-memory strategy's -- same exact counts --
with bounded peak memory, at the cost of touching disk and some raw speed.

Which strategy runs is decided *before* the run starts, by
`pipeline::resolve_strategy`, using a calibrated peak-memory estimator
(`src/mem_estimate.rs`): input size predicts total k-mer occurrences, and
occurrences times thread count predicts peak RSS (the model and its
calibration runs are in that module's doc comments, and pinned by its own
tests). If the predicted peak exceeds the memory budget (`--max-ram`, or
half of currently available RAM by default), the disk strategy is chosen;
otherwise in-memory. The CLI prints the decision, the estimate, and the
budget in its startup banner, and `--strategy memory|disk` overrides the
chooser outright -- see
[Command-line interface](#command-line-interface).

### 6. The parallel pipeline: producer/consumer with backpressure, not a lock

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
the regression test for it. The disk strategy runs the same
producer/channel/worker shape, with spill writers in place of private
counters.

### 7. Quality trimming, and the math behind Phred scores

FASTQ quality bytes are Phred+33 encoded: `phred_score = byte - 33`. A
Phred score is `-10 * log10(P_error)`, i.e. **`P_error = 10^(-Q/10)`** -- Q20
means a 1% chance that base is wrong, Q30 means 0.1%, Q40 means 0.01%.
`FastqRecord::quality_trim_end` walks a sliding window (4 bases by default)
in from the read's 3' end, computing that window's mean Phred score;
trimming stops the moment the window's average reaches the `min_quality`
threshold (Q20 by default). If it never recovers, the whole read collapses
to length zero and contributes no k-mers. The benchmark dataset's own
generator uses this exact `P_error = 10^(-Q/10)` relationship to decide
when to inject a substitution error, so the synthetic data's error/quality
relationship matches what this trimming step is designed to detect.

### 8. Panics cannot cross the FFI boundary, by construction

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

### 9. Crossing into Python: the GIL and zero-copy Arrow

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

The `fastdna` CLI is installed separately from the Python package -- see
[Building from source](#building-from-source). Its default mode counts
k-mers in one or more FASTQ(.gz) file(s) and writes the frequency table plus
a QC report:

```bash
fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

`.gz` input is detected by file extension and decoded with a multi-member
gzip decoder (as required for real SRA/ENA downloads). The output format is
chosen by extension: `.parquet` writes Arrow/Snappy Parquet, anything else
writes CSV. Every exporter writes the same three columns: `kmer_u64`
(`uint64`), `kmer_sequence` (string), `frequency` (`uint32`).

### Subcommands

Counting is also reachable as an explicit `fastdna count ...` subcommand,
identical in every way to giving no subcommand at all -- both forms are
covered by this project's compatibility contract (see `CHANGELOG.md`'s "Qué
cubre este contrato"), so existing scripts that call `fastdna --input ...`
with no subcommand word keep working unchanged. Four more subcommands expose
Rust-core functionality that, before this section, was reachable only from
the Python binding (`fastdna.sketch()`, `.compare_all()`,
`.estimate_cardinality()`, `.peek()`) -- pipeline users who never leave the
shell can now sketch, compare, estimate cardinality, and preview a file
without writing Python:

| Subcommand | What it does | Equivalent Python call |
|---|---|---|
| `fastdna count ...` | Count k-mers (the default; see [Flags](#flags) below) | `fastdna.count(...)` |
| `fastdna sketch --input FILE -k 21 --sketch-size 1000 -o out.json` | Build a MinHash sketch of one file and save it as JSON | `fastdna.sketch(...)` + `Sketch.save(...)` |
| `fastdna dist --input FILE... [--metric jaccard\|containment\|mash]` | Pairwise comparison across two or more sketches and/or files | `fastdna.compare_all(...)` |
| `fastdna card --input FILE -k 31 [--precision 14]` | HyperLogLog estimate of the number of distinct k-mers | `fastdna.estimate_cardinality(...)` |
| `fastdna peek --input FILE [--n-reads 10000]` | Quick preview: read length stats, GC content, a suggested k | `fastdna.peek(...)` |

`fastdna <subcommand> --help` prints each one's full flag list.

`fastdna sketch` sketches a single file (multiple lanes of one sample should
be concatenated first, e.g. via `cat`, the same expectation the Python
binding's `GenomeSketch::from_path` already has). `fastdna dist` accepts any
mix of previously saved sketches (`.json`, as written by `fastdna sketch`)
and raw FASTQ/FASTA files, sketching the latter on the fly with its own
`-k`/`--sketch-size`; `--metric containment` prints both directions of every
pair (containment is asymmetric -- `A.containment(B) != B.containment(A)` in
general), while `jaccard` and `mash` print one row per unordered pair. Without
`--output`, `fastdna dist` prints its CSV table to stdout, so it composes
directly with a shell pipe.

```bash
fastdna sketch --input sampleA.fastq.gz -k 21 --sketch-size 1000 -o a.json
fastdna sketch --input sampleB.fastq.gz -k 21 --sketch-size 1000 -o b.json
fastdna dist --input a.json b.json --metric mash
```

### Flags

`fastdna count`'s flags (identical whether or not the `count` word is given
explicitly -- see [Subcommands](#subcommands) above). `sketch`/`dist`/
`card`/`peek` each have their own, smaller flag set, listed in the
[Subcommands](#subcommands) table and in `fastdna <subcommand> --help`.

| Flag | Default | Description |
|---|---|---|
| `-i, --input <FILE>` | (required unless `--paired-dir` is given) | Input FASTQ file (`.fastq` or `.fastq.gz`); several may be given (R1/R2, extra lanes) and are aggregated into one run |
| `-o, --output <FILE>` | `kmer_counts.parquet` | Output path for k-mer frequencies (`.csv` or `.parquet`). Ignored with `--paired-dir` (see `--paired-output`) |
| `-k, --kmer-size <N>` | `31` | Length of k-mers (`1 <= k <= 32`) |
| `-q, --min-quality <Q>` | `20` | Minimum Phred quality score cutoff (0-40) for 3'-end trimming |
| `-m, --min-count <COUNT>` | `1` | Filter out k-mers with frequency below this cutoff |
| `-M, --max-count <COUNT>` | unset | Filter out k-mers with frequency above this cutoff (repetitive regions) |
| `-t, --threads <N>` | all logical CPUs | Number of worker threads |
| `--qc <FILE>` | `qc_report.json` | Path to export the quality-control summary JSON |
| `--histogram <FILE>` | unset | Optional path to export the frequency spectrum (histogram CSV) |
| `--strategy <auto\|memory\|disk>` | `auto` | Counting strategy: `auto` picks based on the estimated peak memory versus `--max-ram`; `memory` and `disk` force one strategy outright |
| `--max-ram <SIZE>` | half of available RAM; 4 GB if undetectable | Memory budget the automatic chooser targets before switching to the disk strategy. Plain byte count or `K`/`M`/`G`/`T` suffix (binary, 1024-based units; `4G`, `4GB`, `4gb` are equivalent) |
| `--hpc` | off | Collapse homopolymer runs (e.g. `AAAAAA` -> `A`) before k-mer extraction. With the flag absent, output is byte-for-byte identical to today's. Meant for long-read input (Oxford Nanopore, PacBio), where an indel inside a homopolymer run -- not a substitution -- is the dominant sequencing error and, left uncompressed, shifts every k-mer downstream of it; short-read Illumina data has no need for it. Trades exact base-level positional correspondence with the original read for that robustness: a downstream tool mapping a k-mer back to a reference coordinate (e.g. `fastdna.annotate`) is working with compressed-sequence offsets, not the original read's |
| `--paired-dir <DIR>` | unset | Directory of FASTQ files to discover and count as samples, one counting run per sample, instead of naming files by hand with `--input`. See [Paired-end batch counting](#paired-end-batch-counting) below. Requires `--paired-output`; mutually exclusive with `--input` |
| `--paired-output <DIR>` | unset | Output directory for `--paired-dir`: each discovered sample writes its own `<sample_id>.<parquet\|csv>` file here. Created if it does not exist. Required together with `--paired-dir` |
| `--paired-format <parquet\|csv>` | `parquet` | File format for `--paired-dir`'s per-sample outputs |

At startup the CLI prints the strategy decision it will act on, alongside
the estimated peak memory and the budget it was compared against:

```
Strategy:       in-memory (estimated peak 2.28GB, budget 3.35GB)
```

### Environment variables

Two environment variables reach the same strategy chooser, mainly as an
escape hatch for callers with no flag surface of their own (notably the
Python binding, whose `count()` takes no strategy argument yet):

| Variable | Effect |
|---|---|
| `FASTDNA_STRATEGY` | `disk`, or `memory`/`in-memory`: forces that strategy (unless `--strategy` was given explicitly, which wins) |
| `FASTDNA_MAX_RAM_BYTES` | Memory budget as a plain byte count (no suffixes); `--max-ram` wins over it |

Precedence, highest first: `--strategy memory|disk`, then
`FASTDNA_STRATEGY`, then the automatic estimate compared against the budget
(`--max-ram`, then `FASTDNA_MAX_RAM_BYTES`, then the half-of-available-RAM
default).

### Paired-end batch counting

A real paired-end sample is two files -- R1 and R2 -- that must be counted
*together*, not as two separate samples. Doing that for one sample is just
`--input r1.fastq.gz --input r2.fastq.gz` (FastDNA already merges k-mers
across multiple `--input` files). Doing it for a whole directory of
patients by hand means grouping mates yourself for every one of them.
`--paired-dir` automates that grouping:

```bash
fastdna --paired-dir samples/ --paired-output counts/ -k 31
```

This discovers every sample under `samples/` using the same R1/R2 pairing
logic (`cohort::discover_samples` in the Rust core) already used internally
for cohort listing (suffixes `_R1`/`_R2`, `_1`/`_2`, either `_` or `.` as
the separator, plus Illumina's demultiplexed `..._R1_001.fastq.gz` form --
matched case-insensitively), then runs one counting pass per sample, each
mate file fed in together exactly like two `--input` files would be, and
writes that sample's counts to `counts/<sample_id>.parquet` (or `.csv` with
`--paired-format csv`).

**Unlike ad hoc cohort listing, an unpaired file is a hard error here, not
a warning.** `discover_samples` treats a lone `_R1` file with no `_R2` mate
as single-end and keeps going, recording a warning for a human to read
later. `--paired-dir` exists to run counting unattended across many
samples, so the same situation stops the whole run instead: silently
falling back to a single-end sample would produce a normal-looking output
for a directory that was actually half-paired. An empty (or entirely
non-FASTQ) directory is rejected the same way `discover_samples` already
rejects it.

`--paired-dir` is deliberately narrower than a single `--input` run: every
sample uses the same `-k`/`-q`/`-m`/`-M`/`--hpc`/`--threads` settings, and
there is no per-sample QC JSON or histogram (`--qc`/`--histogram` keep their
existing single-file meaning and are not part of this mode).

---

## Memory use and limitations

**The in-memory strategy is the fastest way FastDNA can count, and its peak
memory scales with the input.** Every worker's private counter converges
toward holding the entire distinct-k-mer table (batches are distributed
across workers essentially at random, and the same k-mers recur throughout
a real FASTQ file), so peak memory scales with `threads x distinct_kmers`,
not `distinct_kmers` alone.

### Rule of thumb

The calibrated model in `src/mem_estimate.rs` (fit against five real
release-build runs; the calibration data and residuals are in that module's
doc comments and pinned by its tests) predicts peak RSS as:

```
peak ~= 777 MB (fixed) + 48 MB x threads + 1.168 bytes x threads x total_kmer_occurrences
```

where `1.168` is `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD` -- an
empirical constant folding "distinct k-mers as a fraction of occurrences,
for FASTQ-shaped coverage data" and "bytes per distinct entry" into one
per-occurrence-per-thread rate. Occurrences are in turn estimated from
input size at ~0.366 per uncompressed FASTQ byte. In practical terms:

- **~3.4 GB of peak RAM per GB of uncompressed FASTQ at 8 threads** (plus
  the ~0.8 GB fixed base). Halve the threads, roughly halve it.
- In distinct-k-mer terms, the measured anchor point (8.02 GB peak for
  53.8M distinct k-mers at 8 threads) works out to **roughly 1 GB per ~7
  million distinct k-mers at 8 threads**.
- Gzipped input is assumed to expand ~3.5x for estimation purposes
  (`GZIP_FASTQ_EXPANSION_FACTOR` in `src/main.rs`).

The model is calibrated on Illumina-like, moderate-coverage data; unusually
low-coverage input (mostly-new k-mers) under-predicts, very high coverage
of a small genome over-predicts. It lands within ~2-8% of measured peaks at
realistic scale on the calibration machine.

### When the estimate exceeds the budget: the disk strategy

The CLI no longer simply fails when a run will not fit. With the default
`--strategy auto`, the run is predicted up front against the `--max-ram`
budget (half of currently available RAM by default), and if it does not
fit, FastDNA switches to the **disk-partitioned strategy**
(`src/disk_spill.rs`): exact counts, identical output, peak memory bounded
by one bucket plus the final table instead of the whole per-worker tables
-- at the cost of scratch-disk I/O and some speed. `--strategy disk`
forces it regardless of the estimate; `--strategy memory` forces the
in-memory path (and reintroduces the old failure mode if the input really
doesn't fit -- on the 16 GB benchmark machine, forced-in-memory runs died
somewhere between 2.14 GB and 3.98 GB of input;
[`scripts/bench/memory_ceiling.py`](scripts/bench/memory_ceiling.py) finds
the ceiling on yours).

Two honest caveats:

- The disk strategy's memory bound is only as tight as its largest bucket:
  buckets are fixed high-bit ranges, not KMC3-style minimizer partitions,
  so heavily skewed base composition can leave some buckets much larger
  than others (`src/disk_spill.rs` documents the trade).
- The **Python binding does not engage the automatic chooser**: `count()`
  supplies no input-size estimate, so it always counts in memory unless
  you set `FASTDNA_STRATEGY=disk` in the environment (see
  [Command-line interface](#command-line-interface)). A `strategy=`
  parameter on `count()` is future work.

For truly production-scale out-of-core counting -- a 729-gigabase human
dataset in 33-34 GB of RAM (Kokot et al., *Bioinformatics*, 2017) -- KMC3
and FASTK remain the mature tools; FastDNA's disk strategy is newer, less
tuned, and not yet benchmarked against them at that scale.

### Other limits

- `max_k` is 32, imposed by the 2-bit-per-base `u64` packing. Analyses that
  need longer k-mers are out of scope.
- Each worker's raw buffer is bounded at 2,000,000 buffered instances
  before an eager compaction; the measured effect of that bound -- helpful
  on low-diversity input, a wash-to-loss on high-diversity input -- is
  documented with numbers in
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#worker-buffer-bound-measured-effect-by-input-shape).

**What FastDNA is for**: getting k-mer counts out of FASTQ files and into
Python -- as Arrow, in-process, without a serialization step or a
subprocess -- fastest when the sample fits in memory, and degrading to an
exact disk-partitioned mode rather than failing when it does not.

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
- `count()` always uses the in-memory counting strategy unless the
  `FASTDNA_STRATEGY=disk` environment variable is set -- see
  [Memory use and limitations](#memory-use-and-limitations).

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

`.filter()`, `.sort_by()` and `.top()` are chainable -- each returns a new
`KmerCounts` holding a derived view (`pyarrow.compute` under the hood, no
Rust involved), the same immutable-chaining convention pandas/polars use:

```python
fastdna.count("sample.fastq.gz", k=31)     .filter(min_count=5)     .sort_by("frequency")     .top(20)     .to_pandas()   # or .to_polars()
```

`.total_kmers` and `.spectrum()`/`.suggest_min_count()` always read the
*original* counts regardless of how many `.filter()`/`.top()` calls
preceded them -- the normalization basis, and the spectrum
`suggest_min_count()` needs the error peak in, must not silently change
because a view was filtered or truncated. `.distinct_kmers`/`len()` do
track the current view: `len(result.top(20)) == 20`.

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

### `fastdna.sketch(path, *, k=21, sketch_size=1000) -> Sketch`

Builds a MinHash fingerprint of a FASTQ(.gz) file by streaming it --
memory stays bounded by `sketch_size` regardless of file size, unlike
`count()`'s in-memory strategy, which holds every distinct k-mer at once.
`k=21` (not `count()`'s `k=31`) matches the shorter k typical of
sketching/comparison work in the literature (Mash's own default).

```python
s1 = fastdna.sketch("virus1.fastq", k=21)
s2 = fastdna.sketch("virus2.fastq", k=21)

s1.jaccard(s2)         # symmetric similarity, penalizes genome-size mismatch
s1.containment(s2)     # asymmetric: what fraction of s1 is inside s2
s1.mash_distance(s2)   # Poisson-model evolutionary distance (Ondov et al., 2016);
                        # D = -(1/k)*ln(2J/(1+J)) -- 0 for identical, 1 for disjoint

s1.save("virus1.sketch.json")
fastdna.load_sketch("virus1.sketch.json")

fastdna.compare("a.fastq", "b.fastq", k=21)              # sugar for sketch()+jaccard()
fastdna.compare_all(["a.fastq", "b.fastq", "c.fastq"])   # N-choose-2 pairwise table
```

`.jaccard()` penalizes genome-size differences -- two sketches from very
differently-sized genomes report low Jaccard even if the smaller is
entirely contained in the larger; `.containment()` is the question that
does not ("is this small pathogen present in this large metagenomic
sample" is a containment question). `.mash_distance()` turns the same
overlap into an evolutionary-distance estimate instead of a raw
similarity score -- what "Mash-style" comparison actually promises, not
just a Jaccard number. It does not include Mash's own p-value against a
null hypothesis, which needs a genome-length estimate this method does
not have; that gap is stated, not silently papered over.

`fastdna.compare_all(paths, *, metric="jaccard"|"mash_distance")` builds
each sketch once (`O(N)` FASTQ reads), then compares every pair (`O(N^2)`
cheap sketch comparisons, not `O(N^2)` FASTQ reads -- the exact cost
sketching exists to avoid), returning a plain `pyarrow.Table` in long
format (`sample_a`, `sample_b`, the metric column) that composes directly
with `.sort_by()`/DuckDB/Polars.

### `fastdna.estimate_cardinality(path, *, k=31, precision=14) -> float`

Estimates the number of *distinct* canonical k-mers across an **entire**
file using HyperLogLog (Flajolet et al., 2007), in `2**precision` bytes
(16 KB at the default) regardless of file size -- a different,
complementary question from `peek().sample_distinct_kmers`, which is
exact but covers only a sampled prefix. This covers the whole file, at
the cost of a full streaming pass (the same I/O `count()` itself pays),
trading exactness for a small, known error bound (~0.8% standard error at
the default `precision=14`) instead. Useful for deciding whether a file's
exact count will fit in memory before committing to a run -- see
[Memory use and limitations](#memory-use-and-limitations).

```python
>>> fastdna.estimate_cardinality("huge_sample.fastq.gz", k=31)
53812004.2
```

---

## Building from source

A wheel -- once there is one to install, see
[Installation](#installation) -- gets you the prebuilt Python extension
only, and so does `maturin develop`. If you want the `fastdna` CLI binary,
or you're working on the crate itself, clone the repository and build with
Cargo:

```bash
cargo build --release
./target/release/fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

For the Python extension itself, the same two commands
[Installation](#installation) gives, plus the test suite:

```bash
pip install maturin
maturin develop --release --features python   # NOTE: --release matters --
pytest python/tests -v                        # a debug build of the extension
                                                # measured 4-5x slower in this
                                                # project's own benchmarking
```

`maturin build --release --features python` produces a wheel for the host
platform; the CI workflow builds the same way for the five platforms listed
under [Installation](#installation).

---

## Roadmap

Implemented and documented above: counting with automatic memory/disk
strategy selection (CLI), MinHash sketching
(`fastdna.sketch`/`compare`/`compare_all`), and HyperLogLog cardinality
estimation (`fastdna.estimate_cardinality`). The Python package also ships
newer modules not yet covered by this README's API reference -- among them
a scikit-learn-compatible `KmerVectorizer` (`fastdna.sklearn`) and cohort
embedding (`fastdna.embed.embed_cohort`); see their module docstrings until
this README catches up.

Still ahead:

- A `strategy=`/`max_ram=` parameter on `fastdna.count()`, so the Python
  binding can use the disk strategy and automatic chooser without the
  `FASTDNA_STRATEGY` environment variable.
- **Bioconda packaging.** A recipe skeleton exists at
  [`recipe/meta.yaml`](recipe/meta.yaml), but it is unverified: it has not
  been built with `conda build` or submitted to bioconda-recipes, and its
  source URL and sha256 are placeholders. The recipe's own header comments
  document the submission steps.
- Benchmarking the disk strategy against KMC3/FASTK at out-of-RAM scale.
- Explicit SIMD in the counting hot path (see "What isn't SIMD yet").
