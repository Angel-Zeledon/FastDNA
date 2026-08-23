# FastDNA

FastDNA is a Rust genomic k-mer counter: it reads FASTQ (optionally gzipped)
files and counts canonical k-mers with quality filtering, in parallel, on the
CPUs you already have. It hands the result to Python as an Arrow table, so it
lands in your notebook or pipeline without a serialization step.

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

# Any .fastq or .fastq.gz file; this example uses the small fixture
# checked into the repository at test.fastq.
result = fastdna.count("test.fastq", k=21)
print(result)
# KmerCounts(distinct=19, total=196)

table = result.table  # a pyarrow.Table
print(table.column_names)
# ['kmer_u64', 'kmer_sequence', 'frequency']

print(result.qc)
# {'total_reads': 5, 'total_bases': 300, 'q20_bases': 294, 'q30_bases': 294,
#  'gc_bases': 126, 'gc_content_pct': 42.0, 'q20_pct': 98.0, 'q30_pct': 98.0}
```

`fastdna.count(path, *, k=31, min_count=1, max_count=None, min_quality=20.0,
threads=None)` counts canonical k-mers in a single FASTQ(.gz) file:

- `path` -- a `.fastq` or `.fastq.gz` file.
- `k` -- k-mer length (default 31).
- `min_count` / `max_count` -- drop k-mers below/above these frequencies
  after counting.
- `min_quality` -- Phred quality cutoff applied during counting.
- `threads` -- worker thread count; `None` uses the core's own default.
  Passing `0` explicitly raises `ValueError` rather than hanging.

The returned `KmerCounts` object exposes:

- `.table` -- a zero-copy `pyarrow.Table` with columns `kmer_u64` (the k-mer
  packed as a `uint64`), `kmer_sequence` (its decoded string form), and
  `frequency`.
- `.qc` -- a `dict` of read/base-level quality-control summary statistics.
- `.total_kmers`, `.distinct_kmers` -- also available via `len(result)`.

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

## Why Rust

Counting k-mers over millions of reads is CPU- and memory-bound work that
Python's interpreter loop is not well suited to. FastDNA does that work in
Rust -- parallelized across cores, with the counting table and I/O living
outside the GIL -- and only crosses into Python once, to hand back an Arrow
table. We haven't published benchmark numbers; what we can say is that a
release build correctly processes this repository's own `test.fastq` fixture
end to end, from raw reads to the k-mer table above.

## Building from source

Installing via `pip install fastdna` gets you the prebuilt Python extension
only. If you want the `fastdna` CLI binary, or you're working on the crate
itself, clone the repository and build with Cargo:

```bash
cargo build --release
./target/release/fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

## Roadmap

Cohort-level processing across many samples, a scikit-learn-compatible
`KmerVectorizer`, and MinHash/Jaccard sketching are designed but **not yet
implemented** -- they are not part of the current `fastdna` package. This
README describes only what `pip install fastdna` gives you today.
