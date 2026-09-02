# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FastDNA is a Rust genomic k-mer counter (`fastdna_core`) exposed three ways:
a CLI binary (`fastdna`), a Python extension module (`fastdna._core`, via
PyO3/maturin, wrapped by the pure-Python package under `python/fastdna/`),
and WASM (`src/wasm.rs`). The Rust core counts canonical k-mers from FASTQ
files with quality trimming, in parallel, and hands results to Python as a
zero-copy Arrow table.

**Current mission** (`docs/goal-most-complete-genomics-ml-library.md`,
supersedes `docs/philosophy-narrow-not-broad.md`): become the most complete
library for genomics ML — closing every gap in
`docs/feature-gap-analysis.md` and `docs/ml-differentiation-roadmap.md`.
Explicitly out of scope regardless of that expansion: sequence alignment,
pangenome graphs, general genomic interval algebra (`minimap2`/`vg`/
`bedtools`'s territory — no code reuse with the k-mer engine, unrelated to
the ML mission).

## Build & test commands

CI (`.github/workflows/ci.yml`) is the source of truth for exact invocations.

**Rust:**
```bash
cargo build --all-targets          # lib + bin + tests/ + examples
cargo test --all-targets           # run all Rust tests
cargo test --test cli_args         # a single integration test file (tests/cli_args.rs)
cargo test some_test_name          # a single test by name, any target
cargo clippy --all-targets         # lint (see "Clippy" below before treating warnings as blocking)
```
**Never build with `--features python` or `--all-features` via plain
`cargo build`/`test`.** PyO3's `extension-module` feature doesn't link
libpython, which is correct for the cdylib maturin produces and fails to
link on Linux/macOS for anything else (the `[[bin]]` target, or a bare
`cargo build`). Reach the `python` feature only through maturin.

**Python:**
```bash
pip install maturin
maturin develop --release --features python   # --release matters: debug build measured 4-5x slower
pytest python/tests -v
pytest python/tests/test_api.py -v            # a single test file
pytest python/tests/test_api.py::test_name -v # a single test
```
Rebuild with `maturin develop` after any change under `src/` — the Python
tests import the compiled `fastdna._core` extension, not source.

**Clippy warning ratchet:** `Cargo.toml` denies `unwrap_used`, `expect_used`,
`print_stdout`, `print_stderr` outright — those must stay clean. Beyond
that, CI doesn't require zero warnings; it fails only if the count of
distinct warning *sites* exceeds a hardcoded baseline (currently 8) recorded
in `ci.yml`. Don't add `#[allow(...)]` to silence a preexisting one; either
fix it or leave it as counted debt.

**Network-dependent tests:** `python/tests/test_datasets.py` hits real
BV-BRC/HIVDB endpoints and is marked `@pytest.mark.network`. Deselect with
`pytest -m "not network"` when offline.

## Architecture

### Crate layout (`src/lib.rs`)

`src/lib.rs` is the compatibility boundary, enforced by `pub` vs
`pub(crate)`, not convention — `tests/public_surface.rs` compiles against
the `pub mod` list and breaks the build if one of them stops being
reachable. When adding a new public capability, decide deliberately whether
its module belongs in the `pub mod` block (compatibility contract) or the
`pub(crate)` block (internal strategy/detail, e.g. `disk_spill`, `binned`,
`minimizer`, `superkmer`, `adaptive_bins` — each swappable without breaking
callers).

The **compatibility contract** (`CHANGELOG.md`, "Qué cubre este contrato")
is: Python's `fastdna.__all__` and each documented submodule's own
`__all__`; Rust's `pub mod` list in `src/lib.rs`; the CLI flags/subcommands
documented in `fastdna --help`. Changes to any of these are breaking-change
territory even pre-1.0.

### The counting pipeline, end to end

1. **K-mer encoding** (`src/kmer.rs`): each base is 2 bits
   (`A=00 C=01 G=10 T=11`), so a k-mer (`1 <= k <= 32`) fits in a `u64`.
   Extraction is a rolling bit-shift window (O(n), not O(n·k)); an ambiguous
   base (non-ACGTU, mainly `N`) resets the accumulator rather than
   corrupting a spanning k-mer. Reverse-complement is branchless bit-twiddling
   (`!kmer`, swap 2-bit/4-bit groups, `swap_bytes()`), not string reversal.
   Canonical k-mer = `min(kmer, revcomp(kmer))`.
2. **Counting** (`src/counter.rs`): `KmerCounter` is sort-and-compact, not a
   hash table — each worker appends to a flat `Vec<u64>`, compacts via
   `sort_unstable` + linear scan once a 2,000,000-instance buffer bound is
   hit or the counter is read. This replaced a `HashMap` for a measured
   cache-locality reason documented in `docs/BENCHMARKS.md`.
3. **Strategy selection** (`src/pipeline.rs::resolve_strategy`,
   `src/mem_estimate.rs`): a calibrated model predicts peak RSS from input
   size and thread count, then picks the **in-memory** strategy (fast,
   memory scales with `threads × distinct_kmers`) or the **disk-partitioned**
   strategy (`src/disk_spill.rs`: buckets by high bits, spills sorted runs,
   merges bucket-by-bucket — bounded memory, same exact counts). Overridable
   via `--strategy`/`FASTDNA_STRATEGY` and `--max-ram`/`FASTDNA_MAX_RAM_BYTES`.
   Precedence: explicit CLI flag > env var > automatic estimate vs. budget.
   **The Python binding does not engage the automatic chooser** — `count()`
   always counts in memory unless `FASTDNA_STRATEGY=disk` is set.
4. **Parallel pipeline** (`src/pipeline.rs::process_stream_parallel`):
   producer/consumer, no locks on the hot path. One producer thread streams
   + parses FASTQ into batches over a `crossbeam_channel::bounded(64)`
   (deliberate backpressure); N Rayon workers each own a private
   `KmerCounter` + QC accumulator; a parallel-reduce fold merges them at the
   end. `num_threads = 0` is rejected explicitly as `InvalidConfig`/
   `ValueError` before the channel is built — it used to deadlock (no
   consumer ever drops the receiver, so the producer blocks on a full
   channel forever).
5. **FFI boundary** (`src/ffi.rs`): every worker body is wrapped in
   `catch_unwind` (panic → `FastDnaError::Internal` → Python `RuntimeError`,
   never an unwind across the FFI boundary — that's UB). This is why
   `[profile.release] panic = "unwind"` is pinned in `Cargo.toml`;
   `catch_unwind` is a no-op under `panic = "abort"`. A Python progress
   callback's exception is *not* carried via this panic path — it's
   captured into a side-channel `Mutex` and surfaces via the same
   cancellation flag Ctrl-C uses, specifically to avoid `panic!`'s unasked
   stderr output. `py.allow_threads` releases the GIL for the whole count;
   result delivery to Python is zero-copy via Arrow's C Data Interface
   (no serialize/re-parse round trip).

### Module map (`src/`)

- `kmer.rs`, `counter.rs`, `fastq.rs`, `pipeline.rs` — the core counting
  path described above.
- `mem_estimate.rs` — the calibrated peak-RSS model (calibration data and
  residuals live in its own doc comments, pinned by its tests).
- `disk_spill.rs`, `binned.rs`, `adaptive_bins.rs`, `minimizer.rs`,
  `superkmer.rs` — internal strategies/storage details, `pub(crate)` only.
- `sketch.rs` — MinHash sketching (`GenomeSketch`), backs
  `fastdna.sketch`/`compare`/`compare_all` and the `sketch`/`dist` CLI
  subcommands.
- `hll.rs` — HyperLogLog cardinality estimation.
- `ntcard.rs` — ntCard-style streaming frequency-spectrum estimator.
- `ktab.rs` — the on-disk k-mer table format (Parquet-backed), with
  `SORTED_BY_KEY`/`SORTED_BY_VALUE` metadata contracts other code (setops,
  read_filter) relies on being true.
- `setops.rs` — set operations (union/intersect/diff) across k-mer tables.
- `read_filter.rs` — filtering FASTQ reads by a reference k-mer table.
- `cohort/` (`discovery.rs`, `batch.rs`, `matrix.rs`) — multi-sample
  workflows: R1/R2 pairing (`discover_samples`, matched case-insensitively
  against `_R1`/`_R2`, `_1`/`_2`, and Illumina's `..._R1_001` form), batch
  counting, and cohort k-mer matrices. `--paired-dir`/`--paired-output`
  reuse `discover_samples` but escalate an unpaired file to a hard error
  (vs. cohort listing's warn-and-continue), since `--paired-dir` runs
  unattended across many samples.
- `metagenomics.rs`, `chimera_scan.rs`, `translate.rs`, `read_profile.rs`,
  `qc.rs`, `preview.rs` (backs `peek`), `export.rs` (Parquet/CSV writers),
  `atomic.rs` (write-then-rename; deliberately left `pub` because an
  existing integration test names it directly — see the comment in
  `src/lib.rs`), `error.rs` (`FastDnaError` ⇄ Python exception mapping),
  `cli.rs`/`main.rs` (CLI arg parsing and subcommand dispatch).

### Python package (`python/fastdna/`)

Thin, feature-specific modules layered over the compiled core — each is
independently import-guarded (`pytest.importorskip` in tests; optional
extras like `shap`, `umap-learn`, `biopython`, `matplotlib` are soft
dependencies, never required to import `fastdna` itself). Notable ones
beyond core counting: `sklearn.py` (`KmerVectorizer`, scikit-learn
compatible), `embed.py` (`embed_cohort`), `cv.py` (leakage-safe
cross-validation / lineage grouping — several of its functions accept
either raw FASTQ paths or a `CohortCounts` to skip re-reading files),
`audit.py` (`fastdna.audit()` — leakage-curve + model-differentiation
auditing), `cohort_counts.py` (`CohortCounts`, the in-memory counted-cohort
type multiple modules now accept in place of paths), `mic.py`
(`MicRegressor`), `datasets.py` (real labeled cohorts — BV-BRC AMR, Stanford
HIVDB — cached under `~/.fastdna/datasets`), `genomescope.py`,
`taxonomy.py`, `gwas.py`, `multiomics.py`, `explain.py`/`interpret.py`,
`spectrum.py` (backs `KmerCounts.suggest_min_count()`'s valley-finding
logic — see its module docstring for the exact rule before changing it).

`python/tests/conftest.py` registers a stub `fastdna._core` when the
compiled extension is missing, so pure-Python tests (e.g.
`active_learning`) can run without a Rust build — but this means a broken
`maturin develop` can look like a passing (partial) suite locally; CI
explicitly asserts the real compiled extension is imported before running
tests (see the `Assert the real compiled extension is importable` step in
`ci.yml`) — do the same sanity check locally if tests are unexpectedly
skipping/passing.

### Tests with special conventions

- `tests/public_surface.rs` — compiles against exactly the modules/items
  documented as public; don't "fix" its imports without checking whether
  that's actually loosening or tightening the compatibility contract.
- `tests/review_findings.rs` / `python/tests/test_review_findings.py` —
  regression tests pinning specific defects found during a dated
  correctness review; each is expected to fail until the defect is fixed,
  and each docstring explains the finding it pins. Follow the same pattern
  (dated header, one finding per test, explicit rationale) if adding more.

## Design conventions worth preserving

- **Decisions are justified in comments where they were made**, often at
  length, including alternatives considered and rejected and the measured
  numbers behind a choice (see `Cargo.toml`'s `flate2`/`lto` comments,
  `src/lib.rs`'s module-visibility comments, `ci.yml`'s clippy-ratchet
  block). When changing something with a comment like this, update or
  address the reasoning, don't just delete it.
- **Measured, not assumed.** Performance and memory claims in this repo are
  backed by numbers in `docs/BENCHMARKS.md` or a module's own doc comments,
  with methodology stated. Prefer citing/adding a measurement over an
  unverified performance claim.
- **Exactness is a stated property.** Counting is exact (real keys, not
  hashes) in both strategies; sketching/cardinality estimation is
  explicitly probabilistic with a stated error bound. Don't blur that line
  in docs or code comments.
- `min_count`/`max_count` filtering never changes `total_kmers` — it's the
  normalization basis and must reflect the original counts regardless of
  how many `.filter()`/`.top()` views were derived from them.
