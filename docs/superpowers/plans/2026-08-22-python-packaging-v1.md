# FastDNA v1 — Python Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the packaging and the FFI boundary on five platforms. This phase is NOT the public v1 release — it is the risk-retirement step that must succeed before the cohort engine and KmerVectorizer are built on top of it. Nothing here is published to PyPI.

**Architecture:** Add PyO3 behind an optional `python` Cargo feature (mirroring the existing `wasm` feature), expose a deliberately small FFI surface in `src/ffi.rs`, and put all ergonomics in a thin pure-Python package under `python/fastdna/`. Build abi3 wheels in GitHub Actions with `maturin-action`.

**Tech Stack:** Rust 2021, PyO3 with `abi3-py38` + `extension-module`, maturin as the build backend, arrow 53.4 with its `pyarrow` feature, pytest. No change to the existing core.

**Spec:** `docs/superpowers/specs/2026-08-22-fastdna-python-design.md` — §9 (public Python surface), §11 (cross-platform distribution). This plan implements phases C and D of §15.

## Global Constraints

- **English only** — code, comments, doc comments, CLI/log strings, commit messages, and the README. No exceptions.
- **No `println!`, `eprintln!`, `.unwrap()`, or `.expect()` under `src/` except `src/main.rs`.** Enforced by the clippy lints already in `Cargo.toml`; a violation fails the build.
- **`pyo3` must be an optional dependency behind a `python` feature.** `pyo3/extension-module` tells the crate not to link `libpython`, which breaks the `[[bin]]` target. The existing `wasm` feature is the pattern to copy.
- **abi3-py38** — one wheel per platform covering Python 3.8+, not one per (platform × version).
- **`[profile.release] panic = "unwind"` must remain.** `catch_unwind` is a no-op under abort, and the pipeline relies on it to stop a panicking Python callback from unwinding across the FFI boundary.
- **The GIL is released** with `py.allow_threads` around every long-running call.
- **The FFI surface stays minimal.** Anything that can live in pure Python does.
- Existing behaviour must not regress: `cargo test` and `cargo clippy --all-targets` stay green, and the CLI keeps working.

---

## What v1 Is, and Is Not

**In:** `count`, `peek`, `build_info`, the `KmerCounts` result object, progress, cancellation, and wheels for five targets.

**Out, deliberately:** `count_cohort` (needs the cohort engine), `KmerVectorizer` (needs cohort feature selection), `Sketch`/`compare`. Each is its own plan. Shipping a working single-sample library on five platforms is what proves the architecture; adding the cohort engine first would delay that proof without reducing its risk.

---

## File Structure

**Created:**
- `src/ffi.rs` — PyO3 module. The entire FFI surface, feature-gated. Sole owner of GIL handling and Rust↔Python type conversion.
- `pyproject.toml` — maturin build backend and project metadata.
- `python/fastdna/__init__.py` — public Python API; re-exports from `_core`, adds nothing heavy.
- `python/fastdna/_progress.py` — progress adapter (tqdm or callable or silence) and event serialization.
- `python/tests/test_api.py` — pytest suite run against the built wheel.
- `.github/workflows/wheels.yml` — abi3 wheel builds for five targets.
- `README.md` — the crate has none; PyPI renders it as the project page.

**Modified:**
- `Cargo.toml` — optional `pyo3`, `python` feature, `arrow/pyarrow`, lib name.
- `src/lib.rs` — register `ffi` behind the feature.

---

### Task 1: Cargo wiring and the `python` feature

Nothing can be built until the crate can produce an extension module without breaking the binary.

**Files:**
- Modify: `Cargo.toml`, `src/lib.rs`
- Create: `src/ffi.rs` (skeleton only)

**Interfaces:**
- Consumes: nothing.
- Produces: a `python` Cargo feature; `src/ffi.rs` exposing `#[pymodule] fn _core`.

**The bin/lib name collision must be resolved here.** `cargo test` currently warns that the `fastdna` bin and the `fastdna` lib produce the same `fastdna.pdb`. Adding a cdylib for PyO3 makes that worse. Rename the lib target to `fastdna_core` via `[lib] name = "fastdna_core"`, keeping the binary as `fastdna`. Every `use fastdna::` in `tests/` becomes `use fastdna_core::`.

- [ ] **Step 1: Verify the collision exists**

Run: `cargo test 2>&1 | grep -A3 "filename collision"`
Expected: the warning naming `fastdna.pdb`. This is the defect being fixed; record the output.

- [ ] **Step 2: Add the feature and rename the lib target**

In `Cargo.toml`:

```toml
[lib]
name = "fastdna_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["abi3-py38", "extension-module"], optional = true }

[features]
default = []
python = ["dep:pyo3", "arrow/pyarrow"]
wasm = ["dep:wasm-bindgen", "dep:serde-wasm-bindgen"]
```

Note `extension-module` is listed unconditionally inside the optional dependency, so it only applies when the `python` feature pulls `pyo3` in. The `[[bin]]` target never enables that feature, so it still links normally.

- [ ] **Step 3: Create the module skeleton**

`src/ffi.rs`:

```rust
// src/ffi.rs
//! PyO3 bindings. The entire FFI surface lives here and is deliberately small:
//! everything that can be expressed in pure Python lives in `python/fastdna/`
//! instead, because each function crossing this boundary must be compiled and
//! tested on five platforms.

#![cfg(feature = "python")]

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

Register in `src/lib.rs`:

```rust
#[cfg(feature = "python")]
pub mod ffi;
```

- [ ] **Step 4: Update test imports and verify both targets build**

Replace `use fastdna::` with `use fastdna_core::` across `tests/*.rs`.

Run: `cargo test`
Expected: all tests pass, and the `filename collision` warning is gone.

Run: `cargo build --features python`
Expected: compiles.

Run: `cargo clippy --all-targets`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/lib.rs src/ffi.rs tests/
git commit -m "Add the python feature and resolve the bin/lib name collision

pyo3 goes behind an optional feature, mirroring the existing wasm feature:
extension-module tells the crate not to link libpython, which breaks the
[[bin]] target unless it is feature-gated.

Renames the lib target to fastdna_core so the binary and library stop
producing the same fastdna.pdb -- a pre-existing warning that adding a
second cdylib consumer would have made worse."
```

---

### Task 2: `count` across the boundary

**Files:**
- Modify: `src/ffi.rs`
- Create: `python/fastdna/__init__.py`, `pyproject.toml`

**Interfaces:**
- Consumes: `pipeline::process_stream_parallel`, `PipelineConfig`, `FastDnaError` (Task 1's feature gate).
- Produces: `_core.count(path, k, min_count, max_count, min_quality, threads) -> PyKmerCounts` with `.table()`, `.qc()`, `.total_kmers`, `.distinct_kmers`.

**Error mapping** — implement exactly the table in spec §12:

| Variant | Python exception |
|---|---|
| `Io` | `OSError` (`FileNotFoundError` when the source kind is `NotFound`) |
| `MalformedFastq` | `ValueError`, message carrying path and record number |
| `InvalidK`, `MismatchedK`, `InvalidConfig` | `ValueError` |
| `MatrixTooLarge`, `VocabTooLarge` | `MemoryError` |
| `Export` | `RuntimeError` |
| `Cancelled` | `KeyboardInterrupt` |
| `Internal` | `RuntimeError` |

Write this as one `impl From<FastDnaError> for PyErr` so no call site can diverge from it.

- [ ] **Step 1: Write the failing test**

`python/tests/test_api.py`:

```python
import pathlib
import pytest
import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def test_count_returns_arrow_table(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)
    r = fastdna.count(str(path), k=5)
    tbl = r.table
    assert tbl.num_rows > 0
    assert set(tbl.column_names) == {"kmer_u64", "kmer_sequence", "frequency"}


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        fastdna.count(str(tmp_path / "nope.fastq"), k=5)


def test_invalid_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGT"])
    with pytest.raises(ValueError):
        fastdna.count(str(path), k=99)


def test_zero_threads_raises_valueerror_not_hang(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 100)
    with pytest.raises(ValueError):
        fastdna.count(str(path), k=5, threads=0)
```

The last test is the Python-side regression guard for the `num_threads == 0` hang: `threads=0` must raise, never hang.

- [ ] **Step 2: Run to verify it fails**

Run: `maturin develop --features python && pytest python/tests -v`
Expected: `ModuleNotFoundError` or `AttributeError: module 'fastdna' has no attribute 'count'`.

- [ ] **Step 3: Implement**

In `src/ffi.rs`, add the error conversion, a `PyKmerCounts` class holding the `KmerCounter` plus `QcSummary`, and a `count` function that:
1. validates and builds `PipelineConfig`,
2. opens the file (gzip by extension, as `main.rs` does),
3. calls `process_stream_parallel` **inside `py.allow_threads`**,
4. applies `prune(min_count, max_count)`,
5. returns `PyKmerCounts`.

`table()` converts to Arrow via the `pyarrow` feature's `IntoPyArrow`; reuse the schema from `export.rs` (`kmer_u64: u64`, `kmer_sequence: utf8`, `frequency: u32`) so the Parquet files and the in-memory table stay identical.

`python/fastdna/__init__.py` re-exports `count` and `__version__` from `._core`.

`pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "fastdna"
requires-python = ">=3.8"
dependencies = ["pyarrow>=14"]

[tool.maturin]
features = ["python"]
module-name = "fastdna._core"
python-source = "python"
```

- [ ] **Step 4: Verify**

Run: `maturin develop --features python && pytest python/tests -v`
Expected: 4 passed.

Run: `cargo test && cargo clippy --all-targets`
Expected: still green — the feature gate must not affect the default build.

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs python/ pyproject.toml
git commit -m "Expose count() to Python over the FFI boundary

Returns Arrow zero-copy rather than serializing through a file. The GIL is
released for the duration of the count, so a long call does not freeze the
calling notebook.

One From<FastDnaError> for PyErr implements the spec's exception mapping in
a single place, so no call site can diverge from it. Includes a regression
test that threads=0 raises ValueError rather than hanging."
```

---

### Task 3: Progress and cancellation

The two things that make a long call usable from a notebook, and the two most likely to be got wrong.

**Files:**
- Modify: `src/ffi.rs`
- Create: `python/fastdna/_progress.py`
- Modify: `python/tests/test_api.py`

**Interfaces:**
- Consumes: `Progress`, `ProgressFn`, the `Option<Arc<AtomicBool>>` cancel token.
- Produces: `count(..., progress=None)`; `KeyboardInterrupt` on Ctrl-C.

**Three properties the core's contract requires** (spec §PyO3 boundary — read it before implementing):

1. The callback is invoked **concurrently and re-entrantly from several rayon workers at once**. Each invocation must re-acquire the GIL with `Python::with_gil`, inside a call that released it with `allow_threads`. Forgetting the release deadlocks every worker on the GIL.
2. `ReadsProcessed` values **arrive out of order** — a worker can cross 200_000, be preempted, and have another's 300_000 delivered first. `tqdm` is not thread-safe. The Python adapter serializes events with a lock and ignores any count lower than the highest seen.
3. A panic in the callback is caught by the worker's `catch_unwind` and returned as `Internal`.

**Cancellation** — spawn a watcher that periodically calls `PyErr_CheckSignals` (via `Python::check_signals`) while the GIL is released, and sets the `AtomicBool` when it returns an error. The core returns `Cancelled`, which maps to `KeyboardInterrupt`.

- [ ] **Step 1: Write the failing tests**

```python
def test_progress_receives_events(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 500)
    seen = []
    fastdna.count(str(path), k=5, progress=seen.append, progress_interval=10)
    assert len(seen) > 1, "expected several progress events"


def test_progress_counts_are_monotonic_for_the_consumer(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 2000)
    seen = []
    fastdna.count(str(path), k=5, progress=seen.append, progress_interval=10, threads=4)
    reads = [e for e in seen if isinstance(e, int)]
    assert reads == sorted(reads), "adapter must serialize out-of-order worker events"


def test_a_panicking_callback_becomes_runtimeerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 500)

    def boom(_):
        raise ValueError("callback exploded")

    with pytest.raises(RuntimeError):
        fastdna.count(str(path), k=5, progress=boom, progress_interval=10)
```

The second test is the one that matters: it runs four threads specifically so out-of-order delivery is likely, and fails if the adapter does not serialize.

- [ ] **Step 2: Run to verify they fail**

Run: `maturin develop --features python && pytest python/tests -v -k progress`
Expected: `TypeError: count() got an unexpected keyword argument 'progress'`.

- [ ] **Step 3: Implement**

Rust side: accept an optional `PyObject`, wrap it in a closure that does `Python::with_gil(|py| callback.call1(py, (event,)))`, pass it as `ProgressFn`. Build the cancel token as `Arc<AtomicBool>` and hand a clone to the signal watcher.

Python side (`_progress.py`): a small adapter holding a `threading.Lock` and the highest count seen, forwarding to `tqdm` when `progress=True` and tqdm is importable, to a user callable when one is given, and dropping everything when `None`.

- [ ] **Step 4: Verify**

Run: `pytest python/tests -v`
Expected: 7 passed.

Manual check, recorded in the report: start a long count in a REPL, press Ctrl-C, confirm `KeyboardInterrupt` is raised within a second or two rather than at the end of the run.

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs python/
git commit -m "Wire progress and cancellation through the FFI boundary

The callback is invoked concurrently from rayon workers, so each invocation
re-acquires the GIL and the Python adapter serializes events -- worker
counts arrive out of order and tqdm is not thread-safe.

Ctrl-C now works: a watcher calls PyErr_CheckSignals while the GIL is
released and sets the cancellation token, which surfaces as
KeyboardInterrupt instead of a four-minute wait."
```

---

### Task 4: `peek` and `build_info`

**Files:**
- Modify: `src/ffi.rs`, `python/fastdna/__init__.py`, `python/tests/test_api.py`
- Create: `src/preview.rs`

**Interfaces:**
- Produces: `fastdna.peek(path, n_reads=10_000) -> Preview` with `n_reads_sampled`, `read_length` (min/median/max), `gc_content`, `suggest_k()`; `fastdna.build_info() -> dict`.

**Why `peek` earns its place in v1:** `k=31` is everyone's default and it is wrong for short reads — with 50 bp reads it leaves 20 k-mers per read and amplifies every sequencing error. `peek` answers "what `k` suits *these* data" in milliseconds, before a long run rather than after.

`build_info()` reports version, max `k`, and **whether AVX2 is active on this CPU**. Without it, a "it's slow on my Mac" report is undiagnosable remotely.

`suggest_k()` rule: the largest odd `k` at most `median_read_length / 3`, clamped to `1..=32`. Odd avoids palindromic k-mers being their own reverse complement.

- [ ] **Step 1: Write the failing test**

```python
def test_peek_reports_read_geometry(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 50)
    p = fastdna.peek(str(path))
    assert p.n_reads_sampled == 50
    assert p.read_length == (20, 20, 20)
    assert 0.0 <= p.gc_content <= 1.0
    assert 1 <= p.suggest_k() <= 32
    assert p.suggest_k() % 2 == 1


def test_build_info_reports_avx2(tmp_path):
    info = fastdna.build_info()
    assert "version" in info and "avx2" in info
    assert isinstance(info["avx2"], bool)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest python/tests -v -k "peek or build_info"`
Expected: `AttributeError: module 'fastdna' has no attribute 'peek'`.

- [ ] **Step 3: Implement**

`src/preview.rs` holds the sampling logic in pure Rust (no PyO3) so it is unit-testable with `cargo test`; `ffi.rs` only wraps it. Read at most `n_reads` records and stop — do not read the whole file.

- [ ] **Step 4: Verify**

Run: `cargo test && pytest python/tests -v`
Expected: all green, including new Rust unit tests for `suggest_k` boundaries (median 3 → k=1, median 96 → k=31, median 300 → clamped to 31).

- [ ] **Step 5: Commit**

```bash
git add src/preview.rs src/ffi.rs python/
git commit -m "Add peek() and build_info()

peek samples the first N reads to answer what k suits the data before a long
run commits to a wrong one. build_info reports whether AVX2 is live on this
CPU, without which a remote performance report is undiagnosable."
```

---

### Task 5: Wheels for five platforms

The deliverable is not a `.so` on your machine — it is a wheel a bioinformatician installs without a compiler.

**Files:**
- Create: `.github/workflows/wheels.yml`, `README.md`

**Interfaces:**
- Produces: abi3 wheels for `manylinux x86_64`, `manylinux aarch64`, `macOS x86_64`, `macOS arm64`, `Windows x86_64`.

**Why abi3 matters:** without it you build one wheel per (platform × Python version) — roughly 30 artifacts. `abi3-py38` compiles against CPython's stable ABI, so one wheel per platform covers 3.8 through 3.13+. Five artifacts instead of thirty.

- [ ] **Step 1: Write the workflow**

Use `PyO3/maturin-action`, which brings the manylinux containers and the aarch64 cross-compilation. Matrix over the five targets. Run `pytest` against the built wheel on the native runners. Upload artifacts.

- [ ] **Step 2: Write the README**

The crate has none, and PyPI renders it as the project page. Cover: what FastDNA is in two sentences, `pip install fastdna`, a minimal `count` example with its output, the CLI, and a short "why Rust" note with the benchmark figure. English.

- [ ] **Step 3: Verify locally what can be verified locally**

Run: `maturin build --release --features python`
Expected: a wheel in `target/wheels/`. Confirm its filename contains `abi3` — if it says `cp311` instead, the abi3 feature is not active and every Python version would need its own wheel.

Run: `pip install --force-reinstall target/wheels/*.whl && python -c "import fastdna; print(fastdna.build_info())"`
Expected: prints the dict, from an installed wheel rather than a dev build.

- [ ] **Step 4: Commit**

```bash
git add .github/ README.md
git commit -m "Build abi3 wheels for five platforms

abi3-py38 means one wheel per platform covers Python 3.8+, five artifacts
instead of about thirty. Adds the README, which PyPI renders as the project
page and which the repository has never had."
```

---

## Verification

- [ ] `cargo test` — green, and the filename-collision warning is gone
- [ ] `cargo clippy --all-targets` — clean
- [ ] `cargo build` with no features — the CLI still builds and runs
- [ ] `pytest python/tests` — all green
- [ ] `maturin build --release --features python` — produces an `abi3` wheel
- [ ] The wheel installs into a clean venv and `import fastdna` works with no Rust toolchain present
- [ ] Ctrl-C during a long `count` raises `KeyboardInterrupt` promptly
- [ ] `threads=0` raises `ValueError` rather than hanging

## Out of Scope

`count_cohort`, `KmerVectorizer`, `Sketch`/`compare`, incremental cohorts, and replacing `fastq.rs` with needletail. Each is its own plan. v1 exists to prove the packaging and the boundary on five platforms; the cohort engine adds risk to that proof without reducing it.
