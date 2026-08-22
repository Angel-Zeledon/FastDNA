# FastDNA — Rust core distributed as a Python library

**Date:** 2026-08-22
**Status:** Design approved, implementation plan pending

---

## 1. Goal

Turn FastDNA into a Python library with a Rust engine that a bioinformatician can
install with `pip install fastdna` on Linux, Windows, or macOS **without having
Rust installed**, and use from a notebook to go from raw FASTQ files to a matrix
ready for XGBoost.

### Success criteria

1. `pip install fastdna` works on all 5 target platforms with no compiler.
2. `import fastdna; fastdna.count(...)` returns Arrow data without touching disk.
3. A 500-sample cohort produces a single matrix in one call.
4. The same binary runs on CPUs with and without AVX2, no recompilation.
5. The existing CLI keeps working with no changes to the scripts that invoke it.

---

## 2. Current state of the code (audit)

Findings from reviewing the crate as it stands:

| Finding | Detail |
|---|---|
| **The crate does not compile** | `pipeline.rs:109` passes `&Vec<Vec<u8>>` to `insert_batch(&[u64])` → `error[E0308]` |
| **Duplicated canonical logic** | `kmer.rs` has the correct version (O(1) bit-twiddling, handles `N`, tested). `bio.rs` has a slow copy over `Vec<u8>` that heap-allocates per window and does not handle `N` |
| **The pipeline uses the bad one** | `pipeline.rs` calls `bio::canonical_kmer`, not `kmer::extract_canonical_kmers` |
| **WASM is correct, native is not** | `wasm.rs:26` already uses `kmer::extract_canonical_kmers`. The divergence exists because the biological logic was embedded in `pipeline.rs` instead of living in a shared core |
| **4 dead modules** | `bio.rs` (1 consumer: the broken line), `cms.rs`, `sketch.rs`, `simd.rs` (0 consumers) |
| **Inert SIMD** | The runtime check is nested inside the compile-time gate `target_feature = "avx2"`, which is false without a `.cargo/config.toml`. The block does not exist in the compiled binary |
| **Repository has no commits** | `git log` reported that branch `master` had no commits yet. All source was unversioned |

On the positive side: `fastq.rs` already normalizes `\r\n` correctly, and
`Cargo.toml` already declares `crate-type = ["cdylib", "rlib"]` — exactly what
PyO3 needs.

---

## 3. Settled decisions

| Decision | Choice | Reason |
|---|---|---|
| Python binding | **Native PyO3, in-process** | Zero-copy via Arrow, no subprocess, no intermediate files |
| Target scale | **Virus → human WGS** | Output format is chosen explicitly, never inferred |
| Sample identity | **R1/R2 pair detection** | Counting R1 and R2 as separate patients is a silent error |
| Sequencing depth | **Raw counts + metadata** | Normalization belongs in the cross-validation loop, not in the binary |
| Cohort engine | **Disk spill + in-RAM fast path** | Bounded RAM at any scale; no spill when it isn't needed |
| CLI | **Kept** as a thin client | Preserves the benchmark against Python and the existing `.bat` files |

### Why raw counts and not normalized

With uneven depth (50M reads vs 5M), raw counts make XGBoost learn lab logistics
instead of biology. Rust does **not** solve this by normalizing: it emits raw
counts plus per-sample depth, so normalization happens inside the CV loop, which
is where it belongs statistically and where it does not leak information from the
test set.

### Why feature selection is unsupervised

K-mers are ranked by **prevalence** (how many samples contain them), never by
correlation with the label. Rust never receives labels, so it naturally falls on
the safe side. Selecting features by looking at the target variable across the
full cohort inflates the validation metric; if that is ever done, it goes in
Python and inside the CV loop.

---

## 4. Architecture

One core, three clients:

```
                 pure Rust core
    kmer · counter · cms · sketch · fastq
    qc · pipeline · cohort · select · error
                     │
      ┌──────────────┼──────────────┐
      │              │              │
  ffi/python     bin/main.rs     wasm.rs
   (PyO3)      (CLI+indicatif)   (existing)
```

**Hard rules for the core:**

- No `println!`, no `eprintln!`, no `ProgressBar`, no `.expect()` or `.unwrap()`.
- Every public function returns `Result<T, FastDnaError>`.
- Progress is emitted through a `Fn(Progress)` callback; each client decides what
  to do with it (CLI → indicatif, Python → tqdm or nothing, WASM → ignore).

This separation is what prevents the current bug from recurring: the biological
logic lives in exactly one place and all three clients share it.

---

## 5. Phase 0 — Fix the build

Minimal change, goes first and alone.

1. In `pipeline.rs`, replace the manual loop (lines ~98-111) with:
   ```rust
   let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
   local_counter.insert_batch(&canon_kmers);
   ```
2. Delete `src/bio.rs` and its `pub mod bio;` in `lib.rs`. `hash_kmer` has no
   consumers; it is not preserved.
3. Remove the unused `use crate::kmer;` that triggers the warning.

**Effect:** it compiles, stops heap-allocating a `Vec` per window, and starts
handling `N` bases correctly — the `bio.rs` version treated them as a literal
character, producing corrupt k-mers in the counts.

**Verification:** `cargo test` (the `kmer.rs` tests already cover canonicals,
reverse-complement symmetry, and the reset on `N`).

---

## 6. Feature 2 — Frequency filters

### Semantics

`min_count` and `max_count` apply **per sample**, against that sample's own count
table, never against the cohort aggregate.

### Implementation

New method on `KmerCounter`:

```rust
pub struct PruneStats { pub dropped_min: u64, pub dropped_max: u64, pub kept: u64 }

pub fn prune(&mut self, min: u32, max: Option<u32>) -> PruneStats
```

Called when each sample finishes, freeing RAM before the merge or the spill. The
export-time filter is kept as a safety net.

### Known and accepted limitation

Pruning afterwards **does not lower peak RAM for an individual sample** — the
table already grew to its maximum. Actually lowering the peak would require a
Count-Min Sketch pre-pass (approximate counting, then retain only what clears the
threshold), the way KMC and Jellyfish do it. **Not built now.** The stated goal is
to avoid blowing up the Parquet file and Python's memory, and the filter is
sufficient for that.

### CLI

`--min-count` (already exists, default 1) and `--max-count` (new, default no limit).

---

## 7. Feature 3 — Cohort engine

### 7.1 Sample discovery and grouping

The directory is scanned for `*.fastq`, `*.fq`, `*.fastq.gz`, `*.fq.gz`.

Grouping by `sample_id`: strip the extension, then strip the pair suffix
(`_R1`/`_R2`/`_1`/`_2`, also with `.` as the separator).

| Situation | Behavior |
|---|---|
| `pat_001_R1.fastq.gz` + `pat_001_R2.fastq.gz` | One sample, two files, counts merged |
| `pat_001.fastq.gz` with no suffix | One single-end sample |
| `pat_001_R1.fastq.gz` with no `_R2` | **Explicit orphan warning**; processed as single-end |
| Directory with no recognizable FASTQ | Error, not an empty matrix |

`sample_id`s are sorted lexicographically so the matrix row order is
**reproducible** across runs and platforms.

### 7.2 Parallelism strategy

Parallelism **across samples**, not within each one. Each sample is processed on a
sequential path and rayon distributes samples across threads.

Reason: 500 samples are 500 independent units of work; parallelizing inside each
one on top of the outer distribution oversubscribes the pool and adds
synchronization for no gain. Scaling stays near-linear in core count.

`process_stream_parallel` is kept for single-sample mode (`count`), where it is
the right strategy.

### 7.3 Pass 1 — Counting and prevalence

For each sample, in parallel:

1. Stream its FASTQ file(s) → `KmerCounter` (canonical, quality trimming).
2. `prune(min_count, max_count)`.
3. Record `SampleMeta`.
4. Update the global prevalence table: `prevalence[kmer] += 1` for each surviving
   k-mer (once per sample, not per occurrence).
5. Keep the counter in RAM, or spill it.

```rust
pub struct SampleMeta {
    pub sample_id: String,
    pub files: Vec<PathBuf>,
    pub total_reads: u64,
    pub total_kmers: u64,          // after trim, before prune — normalization basis
    pub distinct_raw: u64,
    pub distinct_pruned: u64,
    pub dropped_min: u64,
    pub dropped_max: u64,
}
```

`total_kmers` is the column Python uses to normalize (CPM, CLR, log).

### 7.4 Fast path vs spill

The RAM needed to keep counters alive is estimated
(`Σ distinct_pruned × 12 bytes`). Below `--max-ram` (default 4 GB) everything
stays in memory and **disk is never touched**. Above it, counters spill.

**Temp format:** raw binary, `(u64 kmer, u32 count)` pairs, little-endian. Not
Parquet: temp files are never inspected, and schema plus compression overhead is
not justified for data read exactly once.

**Location:** `--temp-dir`, defaulting to `std::env::temp_dir()`.

**Cleanup:** a `Drop` guard removes temp files on error too, including when an
exception propagates into Python. No residue is left behind.

### 7.5 Prevalence table and its guard

v1 implementation: exact `FxHashMap<u64, u32>`. Memory ≈ `distinct_union × 12 B`.

If the table exceeds `--max-vocab-ram` (default 2 GB), the program **aborts with
an actionable message**, suggesting a higher `--min-count` or `--approx-vocab`.

`--approx-vocab` is the escape hatch: it uses the `CountMinSketch` in `cms.rs` for
prevalence in fixed memory. It overestimates, so it may let a spurious k-mer into
the ranking — acceptable for ordering by prevalence, not for counts. **Implemented
only if the guard trips in real use.**

### 7.6 Feature selection and output formats

**`format = "wide"`** — dense `n_samples × top_features` matrix.

Ranked by descending prevalence; ties broken by total count, then by `kmer_u64`,
so selection is deterministic.

Size guard: `n_samples × n_features × 4 bytes`. If it exceeds
`--max-matrix-bytes` (default 4 GB), it aborts with a message stating the
estimated size and suggesting a lower `--top-features` or switching to `sparse`.
**A matrix that does not fit is never attempted.**

```
sample_id | ACGT…(1) | ACGT…(2) | … | ACGT…(N)
pat_001   |   142    |    0     | … |    87
```

**Semantics of zero.** A cell is `0` in two distinct cases the matrix does not
separate: the k-mer did not appear in that sample, or it appeared below
`min_count` and was pruned. Per-sample pruning turns sub-threshold counts into
explicit zeros.

This is the right behavior for the goal — those low counts are sequencing noise,
which is exactly what we wanted gone — but it is worth remembering when
interpreting sparsity: some zeros mean "filtered", not "absent". Anyone who needs
the distinction should run with `min_count=1` and filter in Python.

**`format = "sparse"`** — two Arrow tables, no top-N selection:

```
triplets:     sample_idx (u32) | kmer_idx (u32) | count (u32)
vocabulary:   kmer_idx (u32)   | kmer_u64 (u64) | kmer_sequence (str)
```

Python reconstructs in one line:
```python
csr = scipy.sparse.csr_matrix((t.count, (t.sample_idx, t.kmer_idx)))
```
XGBoost and scikit-learn consume `csr_matrix` directly, with no extra
reconstruction step.

### 7.7 Note on Parquet vs Arrow

The practical ~10–20k column ceiling is a **Parquet file** problem (footer
metadata grows per column and per row-group), not a data problem. When delivery
is in-memory Arrow to Python, that ceiling does not apply. Parquet becomes an
optional export format, not the transport mechanism.

### 7.8 Projection onto a fixed vocabulary

Additional primitive, required by `KmerVectorizer.transform` (§9.6):

```rust
pub fn count_projected(
    samples: &[SampleFiles],
    vocabulary: &[u64],     // sorted, defines column order
    opts: &CountOpts,
) -> Result<CohortMatrix, FastDnaError>
```

Counts new samples and projects them onto an **externally supplied** vocabulary
instead of deriving one from the samples themselves. K-mers absent from the
vocabulary are discarded; vocabulary columns the sample lacks become 0.

It is simpler than `count_cohort`: no prevalence pass, no feature selection, and
therefore **no spill** — each sample is counted, projected, and released. One
pass, memory bounded by vocabulary size.

Without this primitive there is no `transform`, and without `transform` there is
no scikit-learn integration and no structural prevention of data leakage. It is
the enabler for all of tier 2.

---

## 8. Feature 4 — MinHash / Jaccard comparison

### API

```python
fastdna.compare("virus1.fastq", "virus2.fastq", k=21, sketch_size=1000)  # -> 0.998
```

### Changes to `sketch.rs`

`sketch.rs` is reusable but needs two adjustments:

1. **Streaming construction.** `from_kmers(&[u64])` requires every k-mer in
   memory. Add `from_stream`, which consumes the `FastqReader` incrementally and
   keeps only the bottom-k. Required for large files.
2. **Better hash finalizer.** The current hash is a bare multiplication
   (`wrapping_mul`). Replace it with a splitmix64-style finalizer: bottom-k
   selects by numeric value, so mixing quality directly affects the accuracy of
   the Jaccard estimate.

`containment` is added alongside `jaccard`: Jaccard penalizes size differences
between genomes, and the common clinical question — *is this pathogen present
inside this metagenomic sample?* — is containment, not similarity. It is ~5 lines
over the same sketch.

The assertion that `k` matches between sketches is kept — comparing sketches with
different `k` has no biological meaning.

### Sketch persistence

`Sketch` serializes to disk (it already derives `Serialize`/`Deserialize`). This
eliminates recomputation: pairwise comparison of N samples via `compare(a, b)`
would do O(N²) FASTQ reads; with persistent sketches it is N.

As a consequence, **N×N comparison moves out of "out of scope"**: it was excluded
because it implied a new engine in Rust, and with persistent sketches it is five
lines of Python over existing primitives (`fastdna.compare_all`).

---

## 9. Public Python surface

### 9.0 Package structure: mixed Rust + Python

```
fastdna/
├── src/                       # Rust core
├── python/fastdna/
│   ├── __init__.py            # public API
│   ├── _core.{pyd,so}         # PyO3 extension (private)
│   ├── sklearn.py             # KmerVectorizer
│   ├── normalize.py           # CPM, CLR, log1p
│   └── spectrum.py            # valley detection
└── pyproject.toml             # maturin, python-source = "python"
```

**The FFI boundary is kept deliberately small.** Everything that is ergonomics —
scikit-learn integration, normalization, pandas conversion — lives in pure Python
on top of `_core`. Reasons:

1. Every function crossing the boundary must be compiled and tested on 5
   platforms. Every function living in Python is tested once.
2. Iterating on the Python API requires no recompilation.
3. The leading underscore in `_core` marks it private: if the boundary changes
   later, the public API does not move.

Maturin supports this layout natively via `python-source`.

---

### 9.1 Tier 1 — Core (Rust via PyO3)

```python
import fastdna

fastdna.count(paths, *, k=31, min_count=1, max_count=None,
              min_quality=20.0, threads=None, progress=None)  -> KmerCounts
fastdna.count_cohort(directory, *, k=31, min_count=1, max_count=None,
                     format="wide", top_features=10_000,
                     threads=None, progress=None)             -> Cohort
fastdna.sketch(path, *, k=21, sketch_size=1000)               -> Sketch
fastdna.peek(path, *, n_reads=10_000)                         -> Preview
fastdna.build_info()                                          -> dict
```

`progress` accepts `None` (silent), `True` (tqdm if available), or a custom
callable. The default is silence: a library does not write to stdout unasked.

`build_info()` reports version, maximum `k`, and **whether AVX2 is active on this
CPU**. Without that, "it's slow on my Mac" is undiagnosable remotely.

---

### 9.2 `KmerCounts`

```python
r = fastdna.count("sample.fastq.gz", k=31, min_count=5)

r.table                  # pyarrow.Table (kmer_u64, kmer_sequence, frequency)
r.to_pandas()            # DataFrame
r.qc                     # dict of quality metrics
r.total_kmers            # normalization basis
r.distinct_kmers
r.top(20)                # 20 most frequent k-mers

r.spectrum()             # {depth: number of distinct k-mers}
r.suggest_min_count()    # -> int, detected from the spectrum

r.save("sample.counts.parquet")
fastdna.load_counts("sample.counts.parquet")

len(r)                   # distinct k-mers
repr(r)                  # KmerCounts(k=31, distinct=104_882, total=8_931_204)
```

**`suggest_min_count()` deserves explanation because it prevents the most pain.**
A sequenced sample's frequency spectrum has two peaks: a huge one at frequency
1-2 (machine errors) and another at the real coverage depth. Between them is a
valley. The correct threshold sits in that valley, and **it differs per sample** —
there is no universal 5.

Today the user would pick `min_count` by eye. This derives it from their data,
which is the difference between discarding noise and discarding signal. The
engine already computes the histogram (`KmerCounter::generate_histogram`); it only
needed exposing, with local-minimum detection on top, in Python.

---

### 9.3 `Cohort`

```python
c = fastdna.count_cohort("./patients/", k=31, min_count=5, top_features=10_000)

c.matrix                 # pyarrow.Table
c.samples                # pyarrow.Table of SampleMeta — per-sample depth
c.vocabulary             # the k-mers chosen as columns
c.dropped                # what was filtered and why

c.to_pandas()            # DataFrame (format="wide")
c.to_numpy()             # ndarray (format="wide")
c.to_scipy()             # csr_matrix (any format)

c.normalize("cpm")       # -> normalized Cohort; also "clr", "log1p", "relative"

c.save("cohort.parquet")
fastdna.load_cohort("cohort.parquet")

repr(c)                  # Cohort(500 samples × 10_000 k-mers, k=31, format='wide')
```

`c.samples` travels attached to `c.matrix` rather than as a loose file: it is
harder to normalize incorrectly when depth sits in the same object as the counts.
That is why `c.normalize()` needs no extra arguments — it already has what it
needs.

`save`/`load_cohort` exist because a 500-sample cohort takes tens of minutes.
Losing that to a Jupyter kernel restart is unacceptable.

`c.vocabulary` is not decorative: it is what allows projecting new samples onto
the same feature basis (§7.8), and it is what makes tier 2 possible.

---

### 9.4 `Sketch` — comparison as an object, not a function

```python
s1 = fastdna.sketch("virus1.fastq", k=21)
s2 = fastdna.sketch("virus2.fastq", k=21)

s1.jaccard(s2)           # 0.998 — symmetric similarity
s1.containment(s2)       # how much of s1 is inside s2 — asymmetric

s1.save("virus1.sig")
fastdna.load_sketch("virus1.sig")

fastdna.compare("a.fastq", "b.fastq", k=21)             # sugar over the above
fastdna.compare_all(["a.fastq", "b.fastq", "c.fastq"])  # N×N matrix
```

Making the sketch a first-class object solves two things at once:

**No recomputation.** `compare(a, b)` recomputes both sketches every call. With N
samples compared pairwise that is O(N²) FASTQ reads when N suffice. Persistent
sketches mean each file is read once.

**N×N comparison stops being Rust work.** It was out of scope because it implied a
new engine; with persistent sketches it is five lines of Python over primitives
that already exist. It becomes in-scope for free.

`containment` answers a different question from Jaccard, and one asked constantly
in clinical work: *is this virus present inside this metagenomic sample?* Jaccard
penalizes genome size differences; containment does not.

---

### 9.5 `Preview` — inspect before committing

```python
p = fastdna.peek("sample.fastq.gz", n_reads=10_000)

p.n_reads_sampled
p.read_length            # (min, median, max)
p.mean_quality_by_position
p.gc_content
p.estimated_distinct_kmers
p.suggest_k()
```

Reads only the first N reads: a matter of milliseconds.

Its reason to exist: `k=31` is everyone's default, and it is wrong for short
reads — with 50 bp reads, `k=31` leaves 20 k-mers per read and amplifies the
effect of every error. `peek` answers "what `k` makes sense for *these* data?"
before launching a 40-minute job, not after.

---

### 9.6 Tier 2 — scikit-learn integration (pure Python)

```python
from fastdna.sklearn import KmerVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

pipe = Pipeline([
    ("kmers", KmerVectorizer(k=31, min_count=5, top_features=10_000)),
    ("clf",   XGBClassifier()),
])

scores = cross_val_score(pipe, fastq_paths, y, cv=5)
```

`KmerVectorizer` implements the scikit-learn estimator API:

- `fit(paths, y=None)` — counts the training samples, selects the vocabulary by
  prevalence, stores it in `self.vocabulary_`.
- `transform(paths)` — projects samples onto the **already-learned** vocabulary
  (§7.8). K-mers unseen during `fit` are ignored; missing ones become 0.
- `get_feature_names_out()` — returns the k-mer sequences.

**This is the most important piece of the whole API**, for a reason that is not
convenience but statistical correctness.

§3 noted that selecting features by looking at the full cohort inflates the
validation metric, and that care was needed. With `KmerVectorizer` that care stops
depending on the user's discipline: inside a `Pipeline`, scikit-learn calls `fit`
**only on the training fold** at each cross-validation iteration. The vocabulary
never sees the test fold. **Leakage goes from a documented risk to a structural
impossibility.**

And `get_feature_names_out()` carries a biological payoff: paired with
`model.feature_importances_`, it gives the concrete DNA sequences driving the
prediction. Those can go into BLAST to ask which gene they belong to. Without it,
the model is a black box that happens to be right; with it, it is a publishable
result.

---

### 9.7 Tier 3 — Helpers (pure Python)

```python
fastdna.normalize(matrix, samples, method="cpm")   # "cpm" | "clr" | "log1p" | "relative"
fastdna.spectrum_valley(hist)                      # local-minimum detection
```

These live in Python because they are array arithmetic that numpy already does
well, and because freezing them into Rust would be exactly the mistake avoided by
deciding to emit raw counts.

---

### 9.8 What is excluded

- CSV manifest support (R1/R2 detection was chosen instead).
- Lazy per-sample iteration over a cohort.
- Resuming interrupted cohorts.
- Alignment, assembly, or anything that is not k-mer counting.

---

### PyO3 boundary

- **The GIL is released** with `py.allow_threads` for all heavy work. Without it,
  a 4-minute cohort freezes the notebook: no progress, no Ctrl-C.
- **Progress via optional callback**; without one, total silence. `indicatif` does
  not cross the boundary — its ANSI codes are visual garbage in a Jupyter cell.
- **Zero-copy Arrow** via the `arrow` crate's `pyarrow` feature (C Data Interface).

---

## 10. CLI

Three subcommands, with `count` as the **default**:

```
fastdna count    --input sample.fastq.gz --output counts.parquet
fastdna cohort   --input-dir ./patients/ --output matrix.parquet
fastdna compare  virus1.fastq virus2.fastq
```

If no subcommand is given, clap falls back to `count`. This preserves the 3
existing invocations of the form `fastdna --input X --output Y` in
`experiments/run_covid.bat` and `ml_pipeline/run_benchmark.sh`, which are left
untouched.

Implementation: `Option<Commands>` plus flattened args; if `command` is `None`,
the flattened args are used as `count`.

---

## 11. Cross-platform distribution

### Wheels with abi3

`pyo3/abi3-py38` compiles against CPython's stable ABI: **one wheel per platform
covers Python 3.8+**, instead of one per platform × version combination
(≈5 artifacts instead of ≈30).

```
manylinux x86_64 · manylinux aarch64 · macOS x86_64 · macOS arm64 · Windows x86_64
```

Built in GitHub Actions with `maturin-action` (which brings manylinux containers
and ARM cross-compilation). `compile.bat` and the manual copying of the `.exe`
into `bin/` go away.

### PyO3 behind an optional feature

`pyo3` goes behind a `python` feature, enabled only by maturin — the same pattern
as the existing `wasm` feature.

Reason: `pyo3/extension-module` tells the crate **not** to link against
`libpython`, which is correct for an extension module but breaks compilation of
the `[[bin]]`. Isolating it behind a feature avoids that conflict.

### Fixing `simd.rs` (blocking for "any platform")

Current structure, with the runtime check nested inside the compile-time gate:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]   // false by default
{
    if is_x86_feature_detected!("avx2") { ... }                 // never compiled
}
```

Consequences: today the block does not exist in the binary. And enabling
`-C target-feature=+avx2` to "fix" it would be worse — the Linux wheel would die
with `SIGILL` on CPUs without AVX2, and on Apple Silicon (aarch64) it would not
compile at all.

**Fix:** compile-time gate on `target_arch` only; runtime detection *outside* it;
`#[target_feature(enable = "avx2")]` on the `unsafe` function. The same wheel uses
AVX2 where it exists and falls back to scalar where it does not.

Since `simd.rs` has no consumers, the valid alternative is to **delete it**. To be
decided at implementation time, depending on whether it gets wired into `kmer.rs`.

---

## 12. Error handling

```rust
pub enum FastDnaError {
    Io { path: PathBuf, source: std::io::Error },
    MalformedFastq { path: PathBuf, record: u64, reason: String },
    InvalidK { k: usize },
    NoSamplesFound { dir: PathBuf },
    MatrixTooLarge { estimated_bytes: u64, limit: u64 },
    VocabTooLarge { estimated_bytes: u64, limit: u64 },
    MismatchedK { left: usize, right: usize },
}
```

Translation per client:

| Variant | Python | CLI |
|---|---|---|
| `Io` | `FileNotFoundError` / `OSError` | message + nonzero exit |
| `MalformedFastq` | `ValueError` with path and record number | same |
| `InvalidK`, `MismatchedK` | `ValueError` | same |
| `MatrixTooLarge`, `VocabTooLarge` | `MemoryError` with size and suggestion | same |

Malformed-FASTQ errors include path and record number: in a 500-file cohort,
"parse error" with no location is useless.

---

## 13. Testing

**Unit**
- Canonicals: already covered in `kmer.rs`; add the property
  `canonical(revcomp(x)) == canonical(x)` over random inputs.
- R1/R2 grouping: pairs, single-end, orphans, names containing dots and dashes.
- `prune`: dropped counts by min and by max at the exact boundaries.
- Feature selection: determinism of tie-breaking with deliberately induced ties.
- Jaccard: sets with known overlap (identical → 1.0, disjoint → 0.0).

**Integration with fixtures**
- Tiny FASTQ files with hand-computed counts.
- A synthetic ~6-sample cohort: verify matrix shape, row order, and that `wide`
  and `sparse` encode the same data.
- Path equivalence: the in-RAM fast path and the spill path must produce
  **byte-identical** output for the same input.

**Cross-platform**
- The suite runs in CI on Linux, Windows, and macOS.
- FASTQ tests with both `\r\n` and `\n` line endings.

**Python**
- pytest against the built wheel: Arrow column types, round-trip to pandas,
  exceptions being the correct classes, and that the GIL is released (a Python
  thread keeps making progress during a long call).
- Persistence round-trip: `save` → `load` returns identical data for
  `KmerCounts`, `Cohort`, and `Sketch`.
- `suggest_min_count()` against a synthetic spectrum with a known valley.
- Asymmetric `containment`: with A ⊂ B, `A.containment(B) ≈ 1.0` while
  `A.jaccard(B)` is low. This is the test that demonstrates why both exist.

**`KmerVectorizer` — the anti-leakage guarantee**

This is the most important testable property, and it can be tested directly:

- scikit-learn's `check_estimator` against the transformer.
- `fit` on samples A,B,C and `transform` on D: k-mers unique to D **do not**
  appear in the output, and columns are exactly `vocabulary_`, in the same order.
- Inside a `Pipeline` with `cross_val_score`, record which paths `fit` receives on
  each fold and assert that **none** belong to the test fold. This turns the §9.6
  guarantee into an executable test rather than a footnote.
- `get_feature_names_out()` returns valid ACGT sequences of length `k`.

---

## 14. Out of scope (YAGNI)

- Count-Min Sketch pre-pass to lower per-sample peak RAM.
- Normalization **inside Rust** (it does exist as a Python helper, §9.7).
- Supervised feature selection.
- Resume / checkpointing of interrupted cohorts.
- CSV sample manifest.
- Alignment, assembly, or any analysis that is not k-mer counting.

*Moved into scope:* N×N comparison — it stopped requiring Rust work once `Sketch`
became persistent (§8).

---

## 15. Implementation order

| Phase | Content | Depends on |
|---|---|---|
| **0** | Fix the build; delete `bio.rs` | — |
| **A** | Extract the core; `Result` everywhere; progress callback | 0 |
| **B** | Frequency filters (`prune`, `--max-count`) | A |
| **C** | PyO3 + maturin + CI wheels; `count` + `peek` exposed | A |
| **D** | `KmerCounts`, spectrum, `suggest_min_count`, save/load | C |
| **E** | Cohort engine + `Cohort` | A, B, C |
| **F** | `count_projected` + `KmerVectorizer` (scikit-learn) | E |
| **G** | Streaming `Sketch`, persistence, `compare_all` | C |
| **H** | Fix or delete `simd.rs` | A |

Phase C comes deliberately early: it is the primary success criterion, and it is
better to discover cross-platform packaging problems with two functions exposed
than with nine.

Phase F depends on E because `KmerVectorizer` needs the prevalence-based feature
selection that lives in the cohort engine. It is the phase with the most value per
line of code: it turns the library from "generates matrices" into "is a
scikit-learn component".

---

## 16. Language convention

**English only, project-wide**: code, comments, doc comments, CLI and log
strings, commit messages, and documentation. No exceptions.

The pre-existing mixed Spanish/English content was translated in a single sweep
on 2026-08-22, before implementation started, covering `kmer.rs`, `pipeline.rs`,
`main.rs`, `bio.rs`, `Cargo.toml`, `download_covid.py`, `baseline_python.py`,
`Dockerfile`, both `.bat` entry points, and `CLAUDE.md`. `cms.rs`, `counter.rs`,
`cli.rs`, `export.rs`, `fastq.rs`, `qc.rs`, `simd.rs`, and `sketch.rs` were
already English.

The rule is recorded in `CLAUDE.md` under Conventions, where it previously said
the opposite — that comments were "a mix of Spanish and English" and new code
should follow the surrounding file. Leaving that line in place would have kept
reintroducing Spanish.

### Missing: README

The repository has **no README at any level**. This is a gap for a library meant
to be distributed on PyPI, where the README becomes the package's project
description page. It should be written in English and cover: what FastDNA is,
installation, a minimal `count` example, the cohort workflow, and the
scikit-learn integration. Not yet written.
