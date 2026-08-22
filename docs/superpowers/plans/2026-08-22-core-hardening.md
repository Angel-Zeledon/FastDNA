# FastDNA Core Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Rust core into a library-grade module that never writes to stdout, never panics, returns `Result` everywhere, reports progress through a caller-supplied callback, and supports per-sample frequency filtering.

**Architecture:** Introduce a single error type (`FastDnaError`) and a progress channel (`Progress` + callback) in the core. Convert `pipeline` and `export` to return `Result`. Move all console output and `indicatif` rendering into `main.rs`, which becomes a thin client that translates core errors into messages and exit codes. Add `KmerCounter::prune` for the frequency filters. Delete the dead `simd.rs`.

**Tech Stack:** Rust 2021, clap 4.5 (derive), rayon 1.10, crossbeam-channel 0.5, rustc-hash 2.0, arrow/parquet 53.4, indicatif 0.17 (binary only). No new dependencies are added by this plan.

**Spec:** `docs/superpowers/specs/2026-08-22-fastdna-python-design.md` (phases A, B, H of §15)

## Global Constraints

- **English only** — all code, comments, doc comments, CLI and log strings, and commit messages. No exceptions (spec §16).
- **No `println!`, `eprintln!`, `ProgressBar`, `.unwrap()`, or `.expect()` in any module under `src/` except `src/main.rs`** (spec §4). `unwrap` is permitted inside `#[cfg(test)]` blocks and integration tests.
- **Every public core function returns `Result<T, FastDnaError>`** (spec §4).
- **`k` is constrained to `1..=32`** because of the 2-bit `u64` packing.
- **Progress is emitted through a callback**, never rendered by the core (spec §4).
- **`total_kmers` is the count of k-mer occurrences after quality trimming and before pruning.** Pruning must never change it — it is the normalization basis (spec §7.3).
- Phase 0 is already complete and committed as `c5fdc4b`. The crate compiles and `cargo test` passes 7 tests.

---

## File Structure

**Created:**
- `src/error.rs` — `FastDnaError` enum, `Display`, `Error`, and the crate-wide `Result<T>` alias. Sole owner of error taxonomy.
- `src/progress.rs` — `Progress` event enum and the `ProgressFn` callback alias. Sole owner of progress reporting types.
- `tests/error_handling.rs` — integration tests for malformed input and error propagation.
- `tests/cli_args.rs` — integration tests for argument parsing.

**Modified:**
- `src/lib.rs` — register `error` and `progress`; deregister `simd`.
- `src/counter.rs` — add `PruneStats` and `KmerCounter::prune`.
- `src/cli.rs` — add `--max-count`.
- `src/fastq.rs` — no signature change; `next_record` keeps returning `io::Result`, and `pipeline` adds path/record context.
- `src/pipeline.rs` — return `Result`, accept a source path and progress callback, remove `eprintln!` and `unwrap_or`.
- `src/export.rs` — return `crate::error::Result` instead of `Box<dyn Error>`.
- `src/main.rs` — handle `Result`, own all console output, wire `indicatif` to the progress callback, honour `--qc` and `--histogram`, exit non-zero on failure.
- `tests/pipeline_integration.rs` — update call sites for the new signature; add progress coverage.

**Deleted:**
- `src/simd.rs` — dead code with an inverted feature gate (spec §11).

---

### Task 1: Error type

Everything downstream returns this type, so it lands first.

**Files:**
- Create: `src/error.rs`
- Modify: `src/lib.rs`
- Test: inline `#[cfg(test)] mod tests` in `src/error.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub enum FastDnaError` with variants `Io { path: PathBuf, source: std::io::Error }`, `MalformedFastq { path: PathBuf, record: u64, reason: String }`, `InvalidK { k: usize }`, `NoSamplesFound { dir: PathBuf }`, `MatrixTooLarge { estimated_bytes: u64, limit: u64 }`, `VocabTooLarge { estimated_bytes: u64, limit: u64 }`, `MismatchedK { left: usize, right: usize }`, `Internal { detail: String }`
  - `pub type Result<T> = std::result::Result<T, FastDnaError>;`

**Note on `Internal`:** spec §12 does not list this variant. It is added here for one purpose only — a worker thread that panics and fails `JoinHandle::join`. Without it, Task 6 would need `.unwrap()` on the join, violating the global constraint. It maps to a generic exception in later phases.

Variants `NoSamplesFound`, `MatrixTooLarge`, and `VocabTooLarge` are unused by this plan; they are consumed by Plan 3 (cohort engine). They are defined now so the enum is not churned later. Add `#[allow(dead_code)]` is **not** needed — public enum variants do not trigger dead-code warnings.

- [ ] **Step 1: Write the failing test**

Create `src/error.rs` containing only the test module for now:

```rust
// src/error.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn malformed_fastq_message_locates_the_failure() {
        let err = FastDnaError::MalformedFastq {
            path: PathBuf::from("patient_007.fastq.gz"),
            record: 41_337,
            reason: "expected '@' at record start".to_string(),
        };

        let msg = err.to_string();
        assert!(msg.contains("patient_007.fastq.gz"), "message must name the file: {msg}");
        assert!(msg.contains("41337"), "message must give the record number: {msg}");
        assert!(msg.contains("expected '@' at record start"), "message must give the reason: {msg}");
    }

    #[test]
    fn invalid_k_message_states_the_valid_range() {
        let msg = FastDnaError::InvalidK { k: 99 }.to_string();
        assert!(msg.contains("99"));
        assert!(msg.contains("1") && msg.contains("32"), "must state the 1..=32 range: {msg}");
    }

    #[test]
    fn io_error_exposes_its_source() {
        use std::error::Error;
        let err = FastDnaError::Io {
            path: PathBuf::from("missing.fastq"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        assert!(err.source().is_some(), "Io must expose the underlying io::Error");
        assert!(err.to_string().contains("missing.fastq"));
    }

    #[test]
    fn matrix_too_large_reports_both_numbers() {
        let msg = FastDnaError::MatrixTooLarge { estimated_bytes: 8_000_000_000, limit: 4_000_000_000 }.to_string();
        assert!(msg.contains("8000000000"));
        assert!(msg.contains("4000000000"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib error`
Expected: FAIL to compile — `cannot find type FastDnaError in this scope`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `src/error.rs`, above the test module:

```rust
// src/error.rs

use std::fmt;
use std::path::PathBuf;

/// Every fallible operation in the FastDNA core returns this type.
#[derive(Debug)]
pub enum FastDnaError {
    /// An underlying I/O failure, carrying the path that caused it.
    Io { path: PathBuf, source: std::io::Error },
    /// A FASTQ record that could not be parsed, located precisely.
    MalformedFastq { path: PathBuf, record: u64, reason: String },
    /// `k` outside the 1..=32 range imposed by 2-bit packing.
    InvalidK { k: usize },
    /// A cohort directory containing no recognizable FASTQ files.
    NoSamplesFound { dir: PathBuf },
    /// A dense matrix that would exceed the configured byte limit.
    MatrixTooLarge { estimated_bytes: u64, limit: u64 },
    /// A prevalence table that would exceed the configured byte limit.
    VocabTooLarge { estimated_bytes: u64, limit: u64 },
    /// Two sketches built with different `k` cannot be compared.
    MismatchedK { left: usize, right: usize },
    /// A worker thread panicked. This indicates a bug in FastDNA.
    Internal { detail: String },
}

impl fmt::Display for FastDnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastDnaError::Io { path, source } => {
                write!(f, "I/O error on {}: {}", path.display(), source)
            }
            FastDnaError::MalformedFastq { path, record, reason } => write!(
                f,
                "malformed FASTQ in {} at record {}: {}",
                path.display(),
                record,
                reason
            ),
            FastDnaError::InvalidK { k } => {
                write!(f, "invalid k-mer size {k}: k must be between 1 and 32 inclusive")
            }
            FastDnaError::NoSamplesFound { dir } => write!(
                f,
                "no FASTQ files found in {} (expected .fastq, .fq, .fastq.gz or .fq.gz)",
                dir.display()
            ),
            FastDnaError::MatrixTooLarge { estimated_bytes, limit } => write!(
                f,
                "dense matrix would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 lower --top-features or use --format sparse"
            ),
            FastDnaError::VocabTooLarge { estimated_bytes, limit } => write!(
                f,
                "vocabulary table would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 raise --min-count or use --approx-vocab"
            ),
            FastDnaError::MismatchedK { left, right } => {
                write!(f, "cannot compare sketches built with different k: {left} and {right}")
            }
            FastDnaError::Internal { detail } => {
                write!(f, "internal error (this is a bug in FastDNA): {detail}")
            }
        }
    }
}

impl std::error::Error for FastDnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastDnaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, FastDnaError>;
```

Then register the module in `src/lib.rs` by adding this line so the list stays alphabetical, directly after `pub mod counter;`:

```rust
pub mod error;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib error`
Expected: PASS, 4 tests.

Run: `cargo check --all-targets`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/error.rs src/lib.rs
git commit -m "Add FastDnaError as the core error type

Every fallible core operation will return this. Variants carry enough
context to locate a failure: paths on I/O errors, path plus record number
on malformed FASTQ, and both numbers on the size guards.

NoSamplesFound, MatrixTooLarge and VocabTooLarge are defined now but first
used by the cohort engine, so the enum is not churned later."
```

---

### Task 2: Delete the dead SIMD module

**Files:**
- Delete: `src/simd.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. This task only removes code.

**Rationale (spec §11):** `simd.rs` has zero consumers. Its AVX2 path is unreachable because the runtime check `is_x86_feature_detected!` sits *inside* the compile-time gate `#[cfg(target_feature = "avx2")]`, which is false without a `.cargo/config.toml`. Enabling that gate would make Linux wheels die with `SIGILL` on CPUs without AVX2 and would not compile on aarch64 at all. Separately, vectorizing would buy little: `extract_canonical_kmers` builds each k-mer from the previous one (`current_kmer << 2 | bits`), a serial dependency chain that SIMD cannot parallelize. If profiling later shows base encoding is hot, it can return written correctly.

- [ ] **Step 1: Confirm there are no consumers**

Run: `grep -rn "simd\|encode_32_bytes" src/ tests/`
Expected: matches only in `src/simd.rs` itself and the `pub mod simd;` line in `src/lib.rs`. If anything else matches, stop and report — the premise of this task is wrong.

- [ ] **Step 2: Delete the module**

```bash
git rm src/simd.rs
```

Remove this line from `src/lib.rs`:

```rust
pub mod simd;
```

- [ ] **Step 3: Verify the build is clean**

Run: `cargo test`
Expected: PASS, 7 tests, no warnings.

Run: `cargo check --all-targets`
Expected: no output beyond the `Finished` line.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "Delete the dead SIMD module

simd.rs had no consumers and its AVX2 path was unreachable: the runtime
is_x86_feature_detected! check sat inside a compile-time target_feature
gate that is false by default, so the block never compiled in.

Enabling that gate would be worse than leaving it dead -- the Linux wheel
would SIGILL on CPUs without AVX2 and aarch64 would not compile at all.
The rolling k-mer window is also a serial dependency chain, so vectorizing
base encoding would buy little. Removed rather than fixed."
```

---

### Task 3: Frequency filtering on the counter

**Files:**
- Modify: `src/counter.rs`
- Test: inline `#[cfg(test)] mod tests` in `src/counter.rs` (this file currently has no tests)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct PruneStats { pub dropped_min: u64, pub dropped_max: u64, pub kept: u64 }` — derives `Debug, Default, Clone, Copy, PartialEq, Eq`
  - `pub fn KmerCounter::prune(&mut self, min: u32, max: Option<u32>) -> PruneStats`

**Semantics:** `min` is inclusive — a k-mer with `count == min` is **kept**. `max` is inclusive — a k-mer with `count == max` is **kept**. `prune` must not change `total_kmers()`.

- [ ] **Step 1: Write the failing test**

Append to `src/counter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a counter where k-mer `i` appears `i` times, for i in 1..=5.
    fn counter_with_graded_counts() -> KmerCounter {
        let mut c = KmerCounter::new();
        for kmer in 1u64..=5 {
            for _ in 0..kmer {
                c.insert(kmer);
            }
        }
        c
    }

    #[test]
    fn prune_drops_counts_below_min_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(3, None);

        assert_eq!(stats.dropped_min, 2, "k-mers seen 1 and 2 times must go");
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 3, "k-mers seen 3, 4 and 5 times must stay");
        assert_eq!(c.distinct_kmers(), 3);
        assert_eq!(c.get_count(3), 3, "count == min is kept");
        assert_eq!(c.get_count(2), 0, "count < min is gone");
    }

    #[test]
    fn prune_drops_counts_above_max_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, Some(4));

        assert_eq!(stats.dropped_max, 1, "only the k-mer seen 5 times is over the cap");
        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.kept, 4);
        assert_eq!(c.get_count(4), 4, "count == max is kept");
        assert_eq!(c.get_count(5), 0);
    }

    #[test]
    fn prune_applies_both_bounds_at_once() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(2, Some(4));

        assert_eq!(stats.dropped_min, 1);
        assert_eq!(stats.dropped_max, 1);
        assert_eq!(stats.kept, 3);
        assert_eq!(c.distinct_kmers(), 3);
    }

    #[test]
    fn prune_leaves_total_kmers_untouched() {
        let mut c = counter_with_graded_counts();
        let before = c.total_kmers();
        assert_eq!(before, 15, "1+2+3+4+5 occurrences");

        c.prune(4, None);

        assert_eq!(
            c.total_kmers(),
            before,
            "total_kmers is the normalization basis and must survive pruning"
        );
    }

    #[test]
    fn prune_with_permissive_bounds_drops_nothing() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, None);

        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 5);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib counter`
Expected: FAIL to compile — `no method named prune found for struct KmerCounter`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/counter.rs`, immediately after the `use rustc_hash::FxHashMap;` line:

```rust
/// Outcome of a `prune` call, for reporting what a filter removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub dropped_min: u64,
    pub dropped_max: u64,
    pub kept: u64,
}
```

Add this method inside `impl KmerCounter`, after `merge`:

```rust
    /// Removes k-mers outside the inclusive `[min, max]` frequency band.
    ///
    /// Both bounds are inclusive: a k-mer whose count equals `min` or `max` is
    /// kept. `total_kmers` is deliberately left unchanged — it is the
    /// normalization basis and must reflect the sample's true depth.
    pub fn prune(&mut self, min: u32, max: Option<u32>) -> PruneStats {
        let mut stats = PruneStats::default();

        self.table.retain(|_, count| {
            if *count < min {
                stats.dropped_min += 1;
                false
            } else if max.is_some_and(|cap| *count > cap) {
                stats.dropped_max += 1;
                false
            } else {
                stats.kept += 1;
                true
            }
        });

        stats
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib counter`
Expected: PASS, 5 tests.

Run: `cargo test`
Expected: PASS, 12 tests total, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/counter.rs
git commit -m "Add KmerCounter::prune for frequency filtering

Both bounds are inclusive. total_kmers is deliberately not adjusted: it is
the per-sample normalization basis and must keep reflecting true sequencing
depth after noise k-mers are dropped.

First tests for counter.rs."
```

---

### Task 4: Expose `--max-count` on the CLI

**Files:**
- Modify: `src/cli.rs`
- Create: `tests/cli_args.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Cli::max_count: Option<u32>` — `None` means no upper bound.

- [ ] **Step 1: Write the failing test**

Create `tests/cli_args.rs`:

```rust
//! Argument-parsing contract. These run clap directly rather than spawning the
//! binary, so they stay fast and need no extra dev-dependencies.

use clap::Parser;
use fastdna::cli::Cli;

#[test]
fn max_count_defaults_to_no_upper_bound() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq"]);
    assert_eq!(cli.max_count, None, "absence of the flag must mean no cap");
}

#[test]
fn max_count_is_parsed_when_supplied() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq", "--max-count", "10000"]);
    assert_eq!(cli.max_count, Some(10_000));
}

#[test]
fn min_count_still_defaults_to_one() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq"]);
    assert_eq!(cli.min_count, 1);
}

#[test]
fn existing_short_flags_are_unchanged() {
    let cli = Cli::parse_from([
        "fastdna", "--input", "s.fastq", "-k", "21", "-q", "30", "-m", "5",
    ]);
    assert_eq!(cli.kmer_size, 21);
    assert_eq!(cli.min_quality, 30.0);
    assert_eq!(cli.min_count, 5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test cli_args`
Expected: FAIL to compile — `no field max_count on type Cli`.

- [ ] **Step 3: Write minimal implementation**

In `src/cli.rs`, add this field immediately after the existing `min_count` field:

```rust
    /// Filter out k-mers with frequency above this cutoff (repetitive regions)
    #[arg(short = 'M', long, value_name = "COUNT")]
    pub max_count: Option<u32>,
```

The short flag is capital `-M`; lowercase `-m` is already taken by `min_count`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test cli_args`
Expected: PASS, 4 tests.

Run: `cargo test`
Expected: PASS, 16 tests total.

- [ ] **Step 5: Commit**

```bash
git add src/cli.rs tests/cli_args.rs
git commit -m "Add --max-count to filter hyper-abundant k-mers

Short flag is -M; -m was already min_count. None means no upper bound.

First argument-parsing tests. They drive clap directly instead of spawning
the binary, so they need no extra dev-dependencies."
```

---

### Task 5: Progress reporting types

**Files:**
- Create: `src/progress.rs`
- Modify: `src/lib.rs`
- Test: inline `#[cfg(test)] mod tests` in `src/progress.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub enum Progress` with variants `ReadsProcessed(u64)`, `SampleStarted { index: usize, total: usize }`, `SampleFinished { index: usize, total: usize }`, `Finished { reads: u64 }` — derives `Debug, Clone, Copy, PartialEq, Eq`
  - `pub type ProgressFn<'a> = Option<&'a (dyn Fn(Progress) + Send + Sync)>;`
  - `pub const PROGRESS_INTERVAL: u64 = 100_000;`

`SampleStarted` and `SampleFinished` are unused by this plan; the cohort engine in Plan 3 emits them. They are defined now so the enum is not churned later.

- [ ] **Step 1: Write the failing test**

Create `src/progress.rs` containing only the test module for now:

```rust
// src/progress.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_closure_can_be_used_as_a_progress_callback() {
        let seen: Arc<Mutex<Vec<Progress>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();

        let callback = move |event: Progress| {
            sink.lock().unwrap().push(event);
        };
        let progress: ProgressFn = Some(&callback);

        if let Some(f) = progress {
            f(Progress::ReadsProcessed(100_000));
            f(Progress::Finished { reads: 250_000 });
        }

        let events = seen.lock().unwrap();
        assert_eq!(
            *events,
            vec![
                Progress::ReadsProcessed(100_000),
                Progress::Finished { reads: 250_000 },
            ]
        );
    }

    #[test]
    fn absent_callback_is_representable_and_costs_nothing() {
        let progress: ProgressFn = None;
        assert!(progress.is_none(), "silence must be the representable default");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib progress`
Expected: FAIL to compile — `cannot find type Progress in this scope`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `src/progress.rs`, above the test module:

```rust
// src/progress.rs

/// How often the pipeline emits `ReadsProcessed`, in reads.
pub const PROGRESS_INTERVAL: u64 = 100_000;

/// A progress event emitted by the core.
///
/// The core never renders progress. It hands these to a caller-supplied
/// callback, and each client decides what to do: the CLI drives `indicatif`,
/// Python forwards to `tqdm` or discards, WASM ignores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// Cumulative reads consumed so far.
    ReadsProcessed(u64),
    /// A cohort sample has started processing (0-based index).
    SampleStarted { index: usize, total: usize },
    /// A cohort sample has finished processing (0-based index).
    SampleFinished { index: usize, total: usize },
    /// Work is complete; carries the final read count.
    Finished { reads: u64 },
}

/// An optional progress callback.
///
/// `None` means silence, which is the default for library use. The callback is
/// invoked from worker threads, hence `Send + Sync`.
pub type ProgressFn<'a> = Option<&'a (dyn Fn(Progress) + Send + Sync)>;
```

Register the module in `src/lib.rs`, after `pub mod pipeline;`:

```rust
pub mod progress;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib progress`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/progress.rs src/lib.rs
git commit -m "Add Progress events and the ProgressFn callback alias

The core must not render progress -- indicatif's ANSI codes are garbage in
a Jupyter cell. It emits typed events and each client decides how to show
them. None means silence, which is the right default for library use.

SampleStarted and SampleFinished are defined now but first emitted by the
cohort engine, so the enum is not churned later."
```

---

### Task 6: Make the pipeline fallible and silent

This is the largest task. It changes the pipeline's public signature, so `main.rs` and the existing integration tests are updated in the same commit to keep the tree building.

**Files:**
- Modify: `src/pipeline.rs`
- Modify: `src/main.rs` (call site only; full rework is Task 8)
- Modify: `tests/pipeline_integration.rs`
- Test: `tests/error_handling.rs` (create)

**Interfaces:**
- Consumes: `FastDnaError`, `Result` (Task 1); `Progress`, `ProgressFn`, `PROGRESS_INTERVAL` (Task 5).
- Produces:
  - `pub fn process_stream_parallel<R: BufRead + Send + 'static>(reader: FastqReader<R>, config: PipelineConfig, source: &Path, progress: ProgressFn<'_>) -> Result<(KmerCounter, QcSummary, u64)>`

**Behavioural contract:**
- `k` outside `1..=32` returns `InvalidK` **before** any reading happens.
- A read error from `FastqReader::next_record` returns `MalformedFastq` carrying `source` and the 1-based record number that failed.
- A panicking worker or producer returns `Internal`.
- No `eprintln!` and no `unwrap`/`expect` remain in the file.
- With `progress: None`, behaviour is identical to before.
- `source` is the path used for error messages. In-memory callers pass `Path::new("<memory>")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/error_handling.rs`:

```rust
//! The pipeline must fail loudly and locate the failure, never print and continue.

use std::io::Cursor;
use std::path::Path;

use fastdna::error::FastDnaError;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
}

fn config(k: usize) -> PipelineConfig {
    PipelineConfig { k, min_quality: 20.0, quality_window: 4, batch_size: 8, num_threads: 2 }
}

#[test]
fn rejects_k_above_the_packing_limit_before_reading() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(33),
        Path::new("sample.fastq"),
        None,
    );

    match result {
        Err(FastDnaError::InvalidK { k }) => assert_eq!(k, 33),
        other => panic!("expected InvalidK, got {other:?}"),
    }
}

#[test]
fn rejects_k_of_zero() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(0),
        Path::new("sample.fastq"),
        None,
    );
    assert!(matches!(result, Err(FastDnaError::InvalidK { k: 0 })));
}

#[test]
fn a_valid_stream_still_succeeds() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGTACGT\n+\nIIIIIIII\n"),
        config(4),
        Path::new("sample.fastq"),
        None,
    );

    let (counter, _qc, reads) = result.expect("valid input must succeed");
    assert_eq!(reads, 1);
    assert_eq!(counter.total_kmers(), 5);
}
```

Add to `tests/pipeline_integration.rs` a progress test, and update every existing call site to the new four-argument signature:

```rust
#[test]
fn progress_callback_receives_a_final_event() {
    use std::sync::{Arc, Mutex};
    use fastdna::progress::Progress;

    let mut fastq = String::new();
    for i in 0..50 {
        fastq.push_str(&format!("@r{}\nACGTACGTAC\n+\nIIIIIIIIII\n", i));
    }

    let seen: Arc<Mutex<Vec<Progress>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let callback = move |e: Progress| sink.lock().unwrap().push(e);

    let (_counter, _qc, reads) = process_stream_parallel(
        reader_for(&fastq),
        config(5),
        std::path::Path::new("<memory>"),
        Some(&callback),
    )
    .expect("valid input");

    let events = seen.lock().unwrap();
    assert_eq!(
        events.last(),
        Some(&Progress::Finished { reads }),
        "the last event must report the final read count"
    );
}

#[test]
fn silent_and_observed_runs_agree() {
    let fastq = "@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nTTGCAACGTT\n+\nIIIIIIIIII\n";
    let noop = |_: fastdna::progress::Progress| {};

    let (silent, _, _) =
        process_stream_parallel(reader_for(fastq), config(5), std::path::Path::new("<memory>"), None)
            .expect("valid");
    let (observed, _, _) = process_stream_parallel(
        reader_for(fastq),
        config(5),
        std::path::Path::new("<memory>"),
        Some(&noop),
    )
    .expect("valid");

    assert_eq!(silent.total_kmers(), observed.total_kmers());
    assert_eq!(silent.distinct_kmers(), observed.distinct_kmers());
}
```

Update the four existing tests in `tests/pipeline_integration.rs` by replacing each call of the form

```rust
process_stream_parallel(reader_for(fastq), config(4))
```

with

```rust
process_stream_parallel(reader_for(fastq), config(4), std::path::Path::new("<memory>"), None)
    .expect("valid input")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test error_handling`
Expected: FAIL to compile — `this function takes 2 arguments but 4 arguments were supplied`.

- [ ] **Step 3: Write the implementation**

Replace the whole body of `src/pipeline.rs` with:

```rust
// src/pipeline.rs

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::thread;
use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReader, FastqRecord};
use crate::kmer;
use crate::progress::{Progress, ProgressFn, PROGRESS_INTERVAL};
use crate::qc::QcSummary;

pub struct PipelineConfig {
    pub k: usize,
    pub min_quality: f64,
    pub quality_window: usize,
    pub batch_size: usize,
    pub num_threads: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            k: 31,
            min_quality: 20.0,
            quality_window: 4,
            batch_size: 8192,
            num_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        }
    }
}

type RecordBatch = Vec<FastqRecord>;

/// Streams a FASTQ source and returns its canonical k-mer counts.
///
/// `source` names the input for error messages only; in-memory callers pass
/// `Path::new("<memory>")`. `progress` is optional; `None` means silence.
pub fn process_stream_parallel<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    if config.k == 0 || config.k > 32 {
        return Err(FastDnaError::InvalidK { k: config.k });
    }

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(64);

    let batch_size = config.batch_size;
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let source_owned: PathBuf = source.to_path_buf();

    // 1. Producer thread. Returns the read count, or the record it choked on.
    let reader_handle = thread::spawn(move || -> Result<u64> {
        let mut reader = reader;
        let mut current_batch = Vec::with_capacity(batch_size);
        let mut total_reads: u64 = 0;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    current_batch.push(record);
                    total_reads += 1;

                    if current_batch.len() >= batch_size {
                        let batch_to_send =
                            std::mem::replace(&mut current_batch, Vec::with_capacity(batch_size));
                        if sender.send(batch_to_send).is_err() {
                            break;
                        }
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    return Err(FastDnaError::MalformedFastq {
                        path: source_owned,
                        record: total_reads + 1,
                        reason: err.to_string(),
                    });
                }
            }
        }

        if !current_batch.is_empty() {
            let _ = sender.send(current_batch);
        }

        Ok(total_reads)
    });

    // 2. Parallel consumer pool. Each worker owns private state, so the hot
    //    path has no locks and no shared hashmap.
    let results: Vec<(KmerCounter, QcSummary, u64)> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let mut local_counter = KmerCounter::with_capacity(131_072);
            let mut local_qc = QcSummary::default();
            let mut local_reads: u64 = 0;

            while let Ok(mut batch) = receiver.recv() {
                for record in &mut batch {
                    local_qc.observe_record(record);
                    record.quality_trim_end(min_qual, qual_win);

                    // Canonical k-mers: 2-bit packed, O(1) rolling window, and
                    // ambiguous bases ('N') reset the window rather than
                    // producing corrupt k-mers.
                    let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
                    local_counter.insert_batch(&canon_kmers);

                    local_reads += 1;
                    if let Some(emit) = progress {
                        if local_reads % PROGRESS_INTERVAL == 0 {
                            emit(Progress::ReadsProcessed(local_reads));
                        }
                    }
                }
            }

            (local_counter, local_qc, local_reads)
        })
        .collect();

    let total_reads = reader_handle
        .join()
        .map_err(|_| FastDnaError::Internal { detail: "FASTQ reader thread panicked".to_string() })??;

    // 3. Map-reduce combine phase.
    let (master_counter, mut master_qc, _) = results.into_par_iter().reduce(
        || (KmerCounter::new(), QcSummary::default(), 0u64),
        |(mut acc_cnt, mut acc_qc, acc_reads), (local_cnt, local_qc, local_reads)| {
            acc_cnt.merge(local_cnt);
            acc_qc.merge(&local_qc);
            (acc_cnt, acc_qc, acc_reads + local_reads)
        },
    );

    master_qc.finalize();

    if let Some(emit) = progress {
        emit(Progress::Finished { reads: total_reads });
    }

    Ok((master_counter, master_qc, total_reads))
}
```

Note the double `??` on the join: the outer unwraps the `JoinHandle` result, the inner unwraps the producer's `Result<u64>`.

In `src/main.rs`, update the single call site so the tree builds. Replace:

```rust
    let (counter, qc, _total_reads) = process_stream_parallel(fastq_reader, config);
```

with:

```rust
    let (counter, qc, _total_reads) =
        match process_stream_parallel(fastq_reader, config, &args.input, None) {
            Ok(v) => v,
            Err(e) => {
                pb.abandon();
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
```

This is a placeholder; Task 8 replaces it with proper structure.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test error_handling`
Expected: PASS, 3 tests.

Run: `cargo test`
Expected: PASS, 25 tests total.

Run: `grep -n "eprintln!\|println!\|unwrap()\|expect(" src/pipeline.rs`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.rs src/main.rs tests/pipeline_integration.rs tests/error_handling.rs
git commit -m "Make the pipeline fallible and silent

process_stream_parallel now returns Result and takes a source path plus an
optional progress callback. A malformed record returns MalformedFastq with
the path and 1-based record number instead of printing to stderr and
truncating the stream silently -- in a 500-file cohort, a parse error with
no location is useless.

k is validated before any reading happens. The reader thread's panic maps
to Internal rather than being swallowed by unwrap_or(0), which previously
turned a crashed producer into a successful run reporting zero reads."
```

---

### Task 7: Convert the exporters to the core error type

**Files:**
- Modify: `src/export.rs`
- Modify: `src/main.rs` (call sites only)
- Test: `tests/export_errors.rs` (create)

**Interfaces:**
- Consumes: `FastDnaError`, `Result` (Task 1).
- Produces (all signatures change return type from `std::result::Result<_, Box<dyn Error>>` to `crate::error::Result<_>`):
  - `pub fn export_counts_parquet<P: AsRef<Path>>(counter: &KmerCounter, output_path: P, k: usize, min_count: u32) -> Result<usize>`
  - `pub fn export_parquet<P: AsRef<Path>>(...) -> Result<usize>` (same params)
  - `pub fn export_counts_csv<P: AsRef<Path>>(...) -> Result<usize>` (same params)
  - `pub fn export_csv<P: AsRef<Path>>(...) -> Result<usize>` (same params)
  - `pub fn export_histogram_csv<P: AsRef<Path>>(counter: &KmerCounter, output_path: P) -> Result<()>`

Parquet and Arrow errors are mapped to `FastDnaError::Io` carrying the output path, since from the caller's perspective they are write failures.

- [ ] **Step 1: Write the failing test**

Create `tests/export_errors.rs`:

```rust
//! Export failures must name the file that could not be written.

use fastdna::counter::KmerCounter;
use fastdna::error::FastDnaError;
use fastdna::export;

fn small_counter() -> KmerCounter {
    let mut c = KmerCounter::new();
    c.insert(0);
    c.insert(1);
    c
}

#[test]
fn csv_export_to_an_unwritable_path_names_the_path() {
    let bad = std::path::Path::new("no_such_directory_xyz").join("out.csv");

    let result = export::export_csv(&small_counter(), &bad, 4, 1);

    match result {
        Err(FastDnaError::Io { path, .. }) => {
            assert!(path.to_string_lossy().contains("out.csv"), "got {path:?}");
        }
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn parquet_export_to_an_unwritable_path_names_the_path() {
    let bad = std::path::Path::new("no_such_directory_xyz").join("out.parquet");

    let result = export::export_parquet(&small_counter(), &bad, 4, 1);

    assert!(matches!(result, Err(FastDnaError::Io { .. })), "got {result:?}");
}

#[test]
fn successful_csv_export_reports_rows_written() {
    let dir = std::env::temp_dir().join("fastdna_export_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("ok.csv");

    let written = export::export_csv(&small_counter(), &out, 4, 1).expect("must succeed");

    assert_eq!(written, 2);
    let _ = std::fs::remove_file(&out);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test export_errors`
Expected: FAIL to compile — the exporters return `Box<dyn Error>`, which cannot be matched against `FastDnaError`.

- [ ] **Step 3: Write the implementation**

In `src/export.rs`, replace the imports block header:

```rust
use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::kmer;
```

Add this helper directly below the imports:

```rust
/// Wraps any writer failure as an I/O error naming the destination.
fn io_err<E: std::fmt::Display>(path: &Path, err: E) -> FastDnaError {
    FastDnaError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(err.to_string()),
    }
}
```

Then, for each of the five public functions:

1. Change the return type from `-> std::result::Result<T, Box<dyn std::error::Error>>` to `-> Result<T>`.
2. Bind the path once at the top so it can be named in errors, e.g. in `export_counts_parquet`:

```rust
    let path = output_path.as_ref();
    let file = File::create(path).map_err(|e| FastDnaError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
```

3. Replace every remaining `?` on a parquet/arrow/io call with `.map_err(|e| io_err(path, e))?`. Specifically, in `export_counts_parquet`: `ArrowWriter::try_new(...)`, each `write_chunk(...)` call, and `writer.close()`. In `export_counts_csv` and `export_histogram_csv`: each `writeln!` and the final `writer.flush()`.

4. Change the private helper's signature and threading of the path:

```rust
fn write_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    u64s: &[u64],
    seqs: &[String],
    freqs: &[u32],
    path: &Path,
) -> Result<()> {
    let u64_arr: ArrayRef = Arc::new(UInt64Array::from(u64s.to_vec()));
    let seq_arr: ArrayRef = Arc::new(StringArray::from_iter_values(seqs.iter().map(|s| s.as_str())));
    let freq_arr: ArrayRef = Arc::new(UInt32Array::from(freqs.to_vec()));

    let batch = RecordBatch::try_new(schema.clone(), vec![u64_arr, seq_arr, freq_arr])
        .map_err(|e| io_err(path, e))?;
    writer.write(&batch).map_err(|e| io_err(path, e))?;
    Ok(())
}
```

5. The thin aliases `export_parquet` and `export_csv` only need their return type changed; their bodies already delegate.

In `src/main.rs`, the two export call sites currently end in `.unwrap()`. Replace:

```rust
    let records_written = if output_str.ends_with(".parquet") {
        export::export_parquet(&counter, &args.output, args.kmer_size, args.min_count).unwrap()
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count).unwrap()
    };
```

with:

```rust
    let export_result = if output_str.ends_with(".parquet") {
        export::export_parquet(&counter, &args.output, args.kmer_size, args.min_count)
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count)
    };
    let records_written = match export_result {
        Ok(n) => n,
        Err(e) => {
            pb_export.abandon();
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test export_errors`
Expected: PASS, 3 tests.

Run: `cargo test`
Expected: PASS, 28 tests total.

Run: `grep -n "Box<dyn" src/export.rs`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/export.rs src/main.rs tests/export_errors.rs
git commit -m "Return FastDnaError from the exporters

Box<dyn Error> erased which file failed to write. Parquet and Arrow errors
now map to Io carrying the output path, so a full disk or a bad directory
says which destination it was."
```

---

### Task 8: Rework `main.rs` as a thin client

**Files:**
- Modify: `src/main.rs`

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 4, 5, 6, 7.
- Produces: no library surface. This task only restructures the binary.

**What changes:**
1. All logic moves into `fn run(args: Cli) -> Result<()>`; `fn main()` only maps errors to a message and exit code 1.
2. `indicatif` is driven by the progress callback rather than a bare spinner.
3. `--max-count` is applied via `counter.prune` before export.
4. `--qc` and `--histogram` are honoured. Both flags are currently parsed and then ignored — the QC summary is only printed to stdout and the histogram is never written at all. This is a pre-existing bug in flags that silently do nothing.

- [ ] **Step 1: Write the failing test**

Create the test as a shell check, since this task's deliverable is binary behaviour. Add to `tests/cli_args.rs`:

```rust
#[test]
fn qc_and_histogram_paths_are_available_to_the_binary() {
    let cli = Cli::parse_from([
        "fastdna", "--input", "s.fastq",
        "--qc", "my_qc.json",
        "--histogram", "my_hist.csv",
    ]);
    assert_eq!(cli.qc.to_string_lossy(), "my_qc.json");
    assert_eq!(
        cli.histogram.as_ref().map(|p| p.to_string_lossy().to_string()),
        Some("my_hist.csv".to_string())
    );
}
```

- [ ] **Step 2: Run the test and the manual check to verify current behaviour is broken**

Run: `cargo test --test cli_args`
Expected: PASS (the flags parse fine; the bug is that nothing consumes them).

Run:
```bash
rm -f /tmp/check_hist.csv
cargo run --quiet --release -- --input test.fastq --output /tmp/check.parquet --histogram /tmp/check_hist.csv
ls /tmp/check_hist.csv
```
Expected: `ls` reports **no such file** — this is the bug being fixed.

- [ ] **Step 3: Write the implementation**

Replace the whole body of `src/main.rs` with:

```rust
// src/main.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use std::time::Instant;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use flate2::read::MultiGzDecoder;

use fastdna::cli::Cli;
use fastdna::error::{FastDnaError, Result};
use fastdna::export;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};
use fastdna::progress::Progress;

fn main() -> ExitCode {
    let args = Cli::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn spinner(message: &str) -> ProgressBar {
    let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");

    let pb = ProgressBar::new_spinner();
    pb.set_style(style);
    pb.set_message(message.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

fn run(args: Cli) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: High-Performance Genomic Kernel (Rust)  ");
    println!("==================================================");
    println!("Input:          {}", args.input.display());
    println!("Output:         {}", args.output.display());
    println!("k-mer Size:     {}", args.kmer_size);
    println!("Quality Cutoff: Q >= {}", args.min_quality);

    let threads = args.threads.unwrap_or(8);
    println!("Worker Threads: {threads}");
    println!("--------------------------------------------------");

    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4,
        min_quality: args.min_quality,
        batch_size: 10_000,
        num_threads: threads,
    };

    let start_time = Instant::now();
    let pb = spinner("Analyzing genomic reads in streaming...");

    let is_gz = args.input.extension().is_some_and(|ext| ext == "gz");
    let file = File::open(&args.input).map_err(|e| FastDnaError::Io {
        path: args.input.clone(),
        source: e,
    })?;

    let buf_reader: Box<dyn BufRead + Send + 'static> = if is_gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let fastq_reader = FastqReader::new(buf_reader);

    // The core stays silent; this closure is what turns events into a spinner.
    let bar = pb.clone();
    let on_progress = move |event: Progress| {
        if let Progress::ReadsProcessed(n) = event {
            bar.set_message(format!("Analyzing genomic reads... {n} processed"));
        }
    };

    let (mut counter, qc, total_reads) = process_stream_parallel(
        fastq_reader,
        config,
        &args.input,
        Some(&on_progress),
    )?;

    let elapsed = start_time.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Processing completed in {elapsed:.2}s"));

    // Frequency filters, applied in RAM before anything is written.
    let prune_stats = counter.prune(args.min_count, args.max_count);
    if prune_stats.dropped_min > 0 || prune_stats.dropped_max > 0 {
        println!(
            "Filtered: {} k-mers below --min-count, {} above --max-count",
            prune_stats.dropped_min, prune_stats.dropped_max
        );
    }

    let pb_export = spinner("Compressing and writing to disk (Parquet/CSV)...");
    let export_start = Instant::now();

    let output_str = args.output.to_string_lossy();
    let records_written = if output_str.ends_with(".parquet") {
        export::export_parquet(&counter, &args.output, args.kmer_size, args.min_count)?
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count)?
    };

    let export_elapsed = export_start.elapsed().as_secs_f64();
    pb_export.finish_with_message(format!("Export completed in {export_elapsed:.2}s"));

    // Honour --qc and --histogram, which were previously parsed and ignored.
    qc.export_json(&args.qc).map_err(|e| FastDnaError::Io {
        path: args.qc.clone(),
        source: e,
    })?;
    println!("QC report written to: {}", args.qc.display());

    if let Some(histogram_path) = &args.histogram {
        export::export_histogram_csv(&counter, histogram_path)?;
        println!("Histogram written to: {}", histogram_path.display());
    }

    println!("--------------------------------------------------");
    println!("Total Reads: {total_reads}");
    println!(
        "Total k-mers Indexed: {} | Distinct k-mers: {}",
        counter.total_kmers(),
        counter.distinct_kmers()
    );
    println!("Records Written to Disk: {records_written}");

    Ok(())
}
```

- [ ] **Step 4: Verify the fix**

Run: `cargo test`
Expected: PASS, 29 tests total, no warnings.

Run:
```bash
rm -f /tmp/check_hist.csv /tmp/check_qc.json
cargo run --quiet --release -- --input test.fastq --output /tmp/check.parquet \
  --qc /tmp/check_qc.json --histogram /tmp/check_hist.csv -k 21
head -3 /tmp/check_hist.csv
head -c 120 /tmp/check_qc.json
```
Expected: both files exist; the histogram starts with the header `coverage_depth,kmer_distinct_count`.

Run, to confirm the failure path:
```bash
cargo run --quiet --release -- --input does_not_exist.fastq --output /tmp/x.parquet; echo "exit=$?"
```
Expected: a message on stderr naming `does_not_exist.fastq`, and `exit=1`.

Run: `cargo run --quiet --release -- --input test.fastq --output /tmp/y.parquet -k 99; echo "exit=$?"`
Expected: `error: invalid k-mer size 99: k must be between 1 and 32 inclusive`, and `exit=1`.

- [ ] **Step 5: Commit**

```bash
git add src/main.rs tests/cli_args.rs
git commit -m "Rework main.rs as a thin client over the core

All logic moves into run() -> Result<()>; main() only maps errors to a
message and a non-zero exit code. Previously a failure could exit 0.

indicatif is now driven by the core's progress callback instead of a blind
spinner, which is the arrangement that lets Python swap in tqdm or silence.

Also honour --qc and --histogram. Both were parsed and then ignored: the QC
summary was only printed to stdout and the histogram was never written at
all, so the flags silently did nothing."
```

---

## Verification

After all eight tasks:

- [ ] `cargo test` — 29 tests pass
- [ ] `cargo check --all-targets` — no warnings
- [ ] `grep -rn "println!\|eprintln!\|unwrap()\|expect(" src/ --include=*.rs | grep -v "^src/main.rs" | grep -v "#\[cfg(test)\]"` — the only remaining hits should be inside `#[cfg(test)]` modules and `ProgressStyle::with_template(...).unwrap_or_else(...)` which is not an unwrap
- [ ] `ls src/` — no `bio.rs`, no `simd.rs`
- [ ] The end-to-end run writes the Parquet, the QC JSON, and the histogram CSV
- [ ] A missing input file exits 1 with a message naming the file

---

## Self-Review Notes

**Spec coverage for phases A, B, H:**

| Spec requirement | Task |
|---|---|
| §4 core returns `Result` everywhere | 1, 6, 7 |
| §4 no console output in core | 6, 7, 8 |
| §4 progress via callback | 5, 6, 8 |
| §6 `prune` with inclusive bounds | 3 |
| §6 `total_kmers` unchanged by pruning | 3 |
| §6 `--max-count` flag | 4 |
| §11 fix or delete `simd.rs` | 2 |
| §12 `FastDnaError` variants | 1 |
| §16 English only | all |

**Deferred to later plans:** §7 cohort engine (Plan 3), §8 sketch changes (Plan 4), §9 Python surface (Plans 2–4), §10 CLI subcommands (Plan 2 — the subcommand split lands with the Python packaging work, since both restructure the entry points), §11 wheels and abi3 (Plan 2).

**Known deviation from spec:** `FastDnaError::Internal` is added beyond the variants listed in §12, to avoid an `.unwrap()` on `JoinHandle::join`. Documented in Task 1.

**Carried forward to Plan 2:** the `output filename collision` warning between the `fastdna` bin target and the `fastdna` cdylib lib target. Harmless today, but Plan 2 adds PyO3 to that same cdylib, so it should be resolved there by renaming the lib target.
