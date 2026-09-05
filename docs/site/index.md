# FastDNA

FastDNA is a Rust genomic k-mer counter with a Python API. It reads FASTQ
(optionally gzipped) files, counts **canonical** k-mers with quality
trimming, in parallel, on the CPUs you already have, and hands the result to
Python as a zero-copy [Arrow](https://arrow.apache.org/) table — so the
counts land in a notebook or a pipeline without a serialization step.

```python
import fastdna

result = fastdna.count("sample.fastq.gz", k=31)
print(result)
# KmerCounts(k=31, distinct=1423950, total=23999892)

df = result.to_pandas()          # zero-copy through pyarrow
print(df.nlargest(5, "frequency"))
```

!!! warning "FastDNA is not published yet"

    There is no release on PyPI, no crate on crates.io and no Bioconda
    package: `pip install fastdna` fails today. Building from source works
    and takes one command — see [Installation](installation.md).

## Two layers, one engine

**The counting engine (Rust).** A k-mer is a 64-bit integer, not a string:
each base is 2 bits, so a k-mer of `k <= 32` fits exactly in a `u64`.
Extraction is a rolling window (`O(1)` per base, independent of `k`), the
reverse complement is a handful of branchless bit operations rather than a
string reversal, and counting is sort-and-compact over a flat `Vec<(u64,
u32)>` rather than a hash table whose bucket placement guarantees a cache
miss per insert. When the run is predicted not to fit in the memory budget,
a disk-partitioned strategy is chosen instead of failing — the decision is
made before the run starts, by a calibrated peak-memory estimator, and
printed. Counting is **exact** under both strategies: the key is the k-mer
itself, not a hash of it, so no two k-mers can collide into one count.

**The Python API is the same engine, not a second one.** `count()`,
`count_cohort()`, sketching and comparison, k-mer tables with set
operations, and read filtering against a reference table all hand back
Arrow, zero-copy, from the Rust core. Nothing in the Python layer adds a
file format or a Rust surface of its own. The full list is in the
[API reference](api/counting.md).

Between 2026-08-24 and 2026-09-05 this package also carried a machine
learning layer — a scikit-learn vectorizer, lineage-aware cross-validation,
a leakage audit, association screening, metagenomic classification and
more. It was removed deliberately, and the reasoning is in
[`docs/goal-fast-kmer-counter.md`](https://github.com/Angel-Zeledon/FastDNA/blob/master/docs/goal-fast-kmer-counter.md).

## How fast, concretely

Measured on a 4-core/8-thread laptop (i7-1165G7, 16 GB RAM, Windows 11) over
200,000 reads / 30,000,000 bases at k=31. All four implementations count
canonical k-mers and agree within 0.001%:

| Implementation | Time |
|---|---:|
| Pure Python (`collections.Counter`) | 88.13 s |
| Biopython (`Bio.SeqIO`) | 121.17 s |
| NumPy (vectorized 2-bit packing) | 107.11 s |
| **FastDNA, 1 thread** | **4.94 s** |
| **FastDNA, 8 threads (default)** | **1.99 s** |

Against the field's dedicated counters on a 2.14 GB synthetic FASTQ at k=31,
FastDNA's in-memory strategy is the fastest of the three tested (~101 s
against FASTK's 119.3 s and KMC3's 293.4 s) and the most memory-hungry of
them. The full methodology — machine details, dataset generation, exact
commands, per-run times and the four-way correctness cross-check — is in
[`docs/BENCHMARKS.md`](https://github.com/Angel-Zeledon/FastDNA/blob/master/docs/BENCHMARKS.md)
in the repository.

## What FastDNA deliberately is not

FastDNA does not do alignment, pangenome graphs, genomic interval algebra or
tensor-framework bridges, and it will not grow them. That is a recorded
decision, not an oversight: each of those has a decade-entrenched incumbent
(`minimap2`, `vg`/`pggb`, `bedtools`), each would reuse approximately none of
this project's k-mer engine, and a published API surface is close to
permanent for a project maintained by one person. The evidence behind the
decision — including the comparison of narrow tools against broad ones in
this exact ecosystem — is in
[`docs/philosophy-narrow-not-broad.md`](https://github.com/Angel-Zeledon/FastDNA/blob/master/docs/philosophy-narrow-not-broad.md).

"Complete" here means complete *within* counting territory. It also means
this site documents what the package does today, and says so plainly when
something is planned rather than shipped.

## Where to go next

- **[Installation](installation.md)** — the real state of packaging, and the
  build-from-source path that works right now.
- **[Getting started](getting-started.md)** — counting a file, filtering and
  chaining results, comparing a cohort, and the command-line interface.
- **[API reference](api/counting.md)** — every public function, class and
  submodule, generated from the docstrings in the source.

For the engine internals — the bit tricks, the parallel pipeline, the
panic-safety boundary across FFI, the memory model and the full CLI flag
table — the
[repository README](https://github.com/Angel-Zeledon/FastDNA/blob/master/README.md)
remains the long-form reference and is not duplicated here.
