// src/ffi.rs
//! PyO3 bindings. The entire FFI surface lives here and is deliberately small:
//! everything that can be expressed in pure Python lives in `python/fastdna/`
//! instead, because each function crossing this boundary must be compiled and
//! tested on five platforms.
//!
//! The module itself is gated in `lib.rs` via `#[cfg(feature = "python")]` on
//! the `pub mod ffi;` declaration, so no inner `#![cfg(...)]` is needed here
//! -- adding one produces a `duplicated_attributes` clippy warning.

// The `#[pyfunction]`/`#[pymethods]` macros (pyo3 0.22) generate a
// trampoline item around any non-getter function/method returning
// `PyResult<_>` that ends in an identity `.into::<PyErr>()`; clippy's
// useless_conversion lint fires on that generated code, but *reports* the
// warning at the original item's return-type span for readability. That
// span-borrowing is exactly why a per-function `#[allow(clippy::
// useless_conversion)]` does not suppress it: tried directly on `spectrum`,
// `count`, `peek` and `build_info` (the four functions that trigger this),
// and all four warnings remained, because the lint-triggering code lives in
// a separate generated item the attribute never reaches. Scoped to the
// module instead -- narrow enough, since the module exists solely to hold
// `#[pyfunction]`/`#[pymethods]` items that all share this pattern.
#![allow(clippy::useless_conversion)]

// `pyo3::create_exception!` (used below for `FastDnaErrorBase`) expands to
// code that checks `cfg(feature = "gil-refs")` -- a cfg pyo3 declares for
// *itself*, not one this crate's own Cargo.toml lists, so Cargo's
// check-cfg linting reports it as unexpected. A per-item `#[allow(...)]`
// placed directly on the macro invocation does not suppress this (rustc
// says so explicitly: the attribute is "ignored, since it's applied to the
// macro invocation" rather than to what it expands into) -- module-level
// is the only scope that reaches generated code here too.
#![allow(unexpected_cfgs)]

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

// The counts table builds Arrow's layout directly (see `build_record_batch`);
// the translation tables use builders, because their row count is not known
// before streaming (see `translate_file`). Both forms are needed here.
use arrow::array::{
    ArrayRef, Int8Builder, StringArray, StringBuilder, UInt32Array, UInt32Builder, UInt64Array,
};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::ToPyArrow;
use arrow::record_batch::RecordBatch;
use flate2::read::MultiGzDecoder;
use pyo3::exceptions::{PyKeyError, PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::cohort;
use crate::cohort_vocab;
use crate::counter::KmerCounter;
use crate::error::FastDnaError;
use crate::export;
use crate::fastq::FastqReader;
use crate::kmer;
use crate::ktab::{self, KmerTable};
use crate::pipeline::{process_stream_parallel_with_policy, MemoryPolicy, PipelineConfig};
use crate::preview;
use crate::progress::Progress;
use crate::qc::QcSummary;
use crate::read_filter;
use crate::read_profile;
use crate::setops;
use crate::sketch::{FracSketch, GenomeSketch};
use crate::translate::{self, Frame, StopHandling, TranslationTable};
use crate::hll;
use crate::metagenomics;
use crate::ntcard;

// The base every FastDNA-specific exception inherits from, alongside
// whichever ordinary Python builtin it already behaved as before this
// hierarchy existed (see the per-variant mapping in `From<FastDnaError> for
// PyErr` below). Catching `FastDnaError` catches any FastDNA-specific
// error; catching `ValueError`/`OSError`/`MemoryError`/`RuntimeError`/
// `FileNotFoundError` keeps working exactly as before, because every leaf
// class genuinely inherits from both.
//
// H-10: `pyo3::create_exception!` only supports a single base class, so the
// leaves below (which must inherit from *two* classes) cannot be declared
// with it. This was tried the other way first: a `create_exception!` per
// leaf naming `FastDnaError` as its one base, with a comment claiming it
// "also" behaved as the matching builtin -- that compiled clean but only
// ever produced single inheritance, silently breaking every existing
// `pytest.raises(ValueError)` / `except OSError:` call site (30 test
// failures, caught in this session before landing; see CHANGELOG.md). Real
// multiple inheritance needs actual Python multiple inheritance, which
// means an actual Python `class` statement -- `register_exception_hierarchy`
// below builds the thirteen leaves with a short embedded-Python snippet,
// run once at import time, inside `_core`'s own `#[pymodule]` function.
// This is a documented, ordinary way to work around
// `create_exception!`'s single-base limitation, not a hack.
//
// Named `FastDnaErrorBase`, not `FastDnaError`, purely to avoid shadowing
// `crate::error::FastDnaError` (this file's `use crate::error::FastDnaError;`
// above) inside this module's namespace -- the macro below defines a new
// Rust item named exactly what its second argument says, and a same-named
// local item wins over the `use` import, which broke every other use of
// `error::FastDnaError` in this file the first time this was tried (54
// compile errors, from `?` no longer finding a matching `From` impl). The
// PYTHON-visible name is still plain `FastDnaError` -- that's the string
// key `register_exception_hierarchy` registers it under below, independent
// of this Rust identifier.
//
// (Both comment blocks above use `//`, not `///`: a doc comment on a macro
// invocation itself is a no-op -- rustdoc has nothing to attach it to and
// warns `unused_doc_comment` -- so this follows the plain-`//` convention
// this file already used for the block this replaces.)
//
// The `cfg(feature = "gil-refs")` warning this macro's expansion would
// otherwise print here (pyo3's own internal cfg, not this crate's -- see
// the module-level `#![allow(unexpected_cfgs)]` near the top of this file
// for why the allow has to live there instead of on this invocation) is
// suppressed at the module level, not here.
pyo3::create_exception!(
    _core,
    FastDnaErrorBase,
    pyo3::exceptions::PyException,
    "Base class for every exception FastDNA's Rust core raises. Catching \
     this catches any FastDNA-specific error, regardless of which ordinary \
     Python exception type (ValueError, OSError, ...) it also is."
);

/// One `class Leaf(FastDnaError, Builtin): pass` per error kind that needs
/// its own catchable identity. `FastDnaError` is injected into the exec
/// globals by `register_exception_hierarchy`; the builtins (`ValueError`,
/// `OSError`, `FileNotFoundError`, `MemoryError`, `RuntimeError`) come from
/// Python's own `__builtins__`, which `Python::run_bound` supplies to exec'd
/// code automatically, the same as a plain `exec()` call would.
///
/// Order matches `EXCEPTION_LEAF_NAMES` below and the match arms in
/// `From<FastDnaError> for PyErr`; keep all three in step when adding a
/// variant.
const EXCEPTION_HIERARCHY_SOURCE: &str = "\
class MalformedFastqError(FastDnaError, ValueError):
    '''A FASTQ record that could not be parsed.'''
class InvalidKError(FastDnaError, ValueError):
    '''k is outside the 1..=32 range 2-bit packing allows.'''
class MismatchedKError(FastDnaError, ValueError):
    '''Two sketches built with different k cannot be compared.'''
class MismatchedScaleError(FastDnaError, ValueError):
    '''Two FracSketches built with different scale cannot be compared.'''
class InvalidConfigError(FastDnaError, ValueError):
    '''A configuration value supplied by the caller is not usable.'''
class NoSamplesFoundError(FastDnaError, ValueError):
    '''A cohort directory contains no recognizable FASTQ files.'''
class LoadError(FastDnaError, ValueError):
    '''A previously-saved artifact (e.g. a GenomeSketch) could not be read back.'''
class IoNotFoundError(FastDnaError, FileNotFoundError):
    '''The path FastDNA was asked to read does not exist.'''
class IoError(FastDnaError, OSError):
    '''An I/O failure other than the path simply not existing.'''
class MatrixTooLargeError(FastDnaError, MemoryError):
    '''The requested dense matrix would exceed the configured byte limit.'''
class VocabTooLargeError(FastDnaError, MemoryError):
    '''The requested vocabulary table would exceed the configured byte limit.'''
class ExportError(FastDnaError, RuntimeError):
    '''Serialization or writer failure while exporting results.'''
class InternalError(FastDnaError, RuntimeError):
    '''A worker thread panicked. This indicates a bug in FastDNA.'''
";

/// Every leaf name declared in `EXCEPTION_HIERARCHY_SOURCE`, in the same
/// order, so `register_exception_hierarchy` can pull each one out of the
/// exec globals afterward.
const EXCEPTION_LEAF_NAMES: &[&str] = &[
    "MalformedFastqError",
    "InvalidKError",
    "MismatchedKError",
    "MismatchedScaleError",
    "InvalidConfigError",
    "NoSamplesFoundError",
    "LoadError",
    "IoNotFoundError",
    "IoError",
    "MatrixTooLargeError",
    "VocabTooLargeError",
    "ExportError",
    "InternalError",
];

/// Populated once, from inside `_core`'s `#[pymodule]` function, before any
/// `#[pyfunction]` becomes callable. Python guarantees a module's own
/// top-level init code finishes before its functions can be invoked from
/// the importing side, so every read from `From<FastDnaError> for PyErr`
/// sees this already filled in -- there is no ordering race to guard here.
static EXCEPTION_CLASSES: OnceLock<HashMap<&'static str, Py<PyAny>>> = OnceLock::new();

/// Builds the thirteen dual-inheriting leaf classes from
/// `EXCEPTION_HIERARCHY_SOURCE` and registers all fourteen names (the base
/// plus the thirteen leaves) on `m`, so Python can also
/// `from fastdna._core import InvalidKError` directly.
fn register_exception_hierarchy(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FastDnaError", py.get_type_bound::<FastDnaErrorBase>())?;

    let globals = PyDict::new_bound(py);
    globals.set_item("FastDnaError", py.get_type_bound::<FastDnaErrorBase>())?;
    py.run_bound(EXCEPTION_HIERARCHY_SOURCE, Some(&globals), None)?;

    let mut classes = HashMap::with_capacity(EXCEPTION_LEAF_NAMES.len());
    for &name in EXCEPTION_LEAF_NAMES {
        let cls = globals.get_item(name)?.unwrap_or_else(|| {
            panic!(
                "{name} is declared in EXCEPTION_HIERARCHY_SOURCE but missing from \
                 the exec globals afterward -- EXCEPTION_LEAF_NAMES has drifted out \
                 of step with the source"
            )
        });
        m.add(name, cls.clone())?;
        classes.insert(name, cls.unbind());
    }
    EXCEPTION_CLASSES
        .set(classes)
        .map_err(|_| PyRuntimeError::new_err("fastdna._core was initialized more than once"))?;
    Ok(())
}

/// Builds a `PyErr` for one of the thirteen leaf classes registered by
/// `register_exception_hierarchy`, by calling the class the way Python
/// itself would (`SomeError(message)`) rather than constructing an
/// instance by hand.
fn leaf_exception(py: Python<'_>, name: &'static str, message: String) -> PyErr {
    let classes = EXCEPTION_CLASSES.get().unwrap_or_else(|| {
        panic!("fastdna._core must finish importing before any FastDnaError can convert to PyErr")
    });
    let cls = classes
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not a registered exception leaf"))
        .bind(py);
    match cls.call1((message,)) {
        Ok(instance) => PyErr::from_value_bound(instance),
        // Constructing the exception instance itself failed -- surface
        // that failure rather than the original error it was standing in
        // for, since it is the more actionable problem at this point.
        Err(err) => err,
    }
}

/// The single place the spec's error-to-exception table (design doc §12) is
/// implemented. No call site maps a `FastDnaError` to a `PyErr` directly, so
/// no call site can diverge from this table.
impl From<FastDnaError> for PyErr {
    fn from(err: FastDnaError) -> PyErr {
        // `Cancelled` deliberately does not join the hierarchy above:
        // Python's own `KeyboardInterrupt` inherits from `BaseException`,
        // not `Exception`, specifically so a broad `except Exception:` does
        // not swallow it. Folding it into `FastDnaError` (an `Exception`
        // subclass) would defeat that on purpose, so it stays a plain,
        // undecorated `KeyboardInterrupt`.
        if matches!(err, FastDnaError::Cancelled) {
            return PyKeyboardInterrupt::new_err(err.to_string());
        }

        let message = err.to_string();
        let name: &'static str = match &err {
            // `NotFound` gets the more specific exception; every other kind
            // of I/O failure (permission denied, a bad gzip stream, ...) is
            // a generic OSError.
            FastDnaError::Io { source, .. } => {
                if source.kind() == std::io::ErrorKind::NotFound {
                    "IoNotFoundError"
                } else {
                    "IoError"
                }
            }
            FastDnaError::MalformedFastq { .. } => "MalformedFastqError",
            FastDnaError::InvalidK { .. } => "InvalidKError",
            FastDnaError::MismatchedK { .. } => "MismatchedKError",
            FastDnaError::MismatchedScale { .. } => "MismatchedScaleError",
            FastDnaError::InvalidConfig { .. } => "InvalidConfigError",
            // Not in the spec's table (which predates the cohort engine),
            // but a caller-supplied bad directory is the same kind of
            // mistake as InvalidConfig, so it gets the same treatment
            // rather than being left unclassified.
            FastDnaError::NoSamplesFound { .. } => "NoSamplesFoundError",
            // Also not in the spec's table (predates GenomeSketch
            // persistence). A corrupt/foreign sketch file is bad input
            // data, the same kind of mistake as a malformed FASTQ record,
            // not an internal failure -- ValueError-family, not RuntimeError.
            FastDnaError::Load { .. } => "LoadError",
            FastDnaError::MatrixTooLarge { .. } => "MatrixTooLargeError",
            FastDnaError::VocabTooLarge { .. } => "VocabTooLargeError",
            FastDnaError::Export { .. } => "ExportError",
            FastDnaError::Internal { .. } => "InternalError",
            FastDnaError::Cancelled => unreachable!("handled above, before the message is built"),
            // No wildcard arm: `FastDnaError` (src/error.rs) is
            // `#[non_exhaustive]` for callers *outside* this crate only --
            // inside it, matching stays genuinely exhaustive (see
            // error.rs's own `non_exhaustive_does_not_block_construction_
            // inside_the_crate` test), so a `_ =>` arm here would be dead
            // code today (confirmed by `unreachable_patterns` when this was
            // tried) and would silently swallow a real classification
            // decision the next time a variant is added. Adding a variant
            // to `FastDnaError` must be a compile error here until someone
            // decides which leaf class it maps to.
        };
        Python::with_gil(|py| leaf_exception(py, name, message))
    }
}

/// `count()`'s own polling loop result: the pipeline's outcome (counter,
/// QC summary, total reads read) or the error that ended it.
type CountOutcome = Result<(KmerCounter, QcSummary, u64), FastDnaError>;

/// Wraps an Arrow failure from `build_record_batch`. Mirrors
/// `export.rs::export_err`, but the "path" is a placeholder: nothing here
/// touches a disk, and `FastDnaError::Export` maps to `RuntimeError` on the
/// Python side either way.
fn in_memory_export_err<E: std::fmt::Display>(err: E) -> FastDnaError {
    FastDnaError::Export {
        path: PathBuf::from("<in-memory Arrow table>"),
        reason: err.to_string(),
        // No typed cause: this function's only caller builds the message
        // with `format!`, not from a real error object.
        source: None,
    }
}

/// Builds the single in-memory Arrow `RecordBatch` backing `KmerCounts.table`,
/// using the exact schema `export.rs` writes to Parquet with (§9.2 of the
/// design doc: the in-memory table and the Parquet files must have identical
/// columns).
///
/// Assembles Arrow's own storage layout directly -- a `u64` values buffer, a
/// `u32` values buffer, and one contiguous value buffer plus an `i32` offsets
/// buffer for the sequence column -- and hands each `Vec` over by move
/// (`UInt64Array::from(Vec)` and `Buffer::from_vec` adopt the allocation
/// rather than copying it). This is the same treatment `export.rs` applies to
/// its Parquet chunks, and for the same reason: on the benchmark file this
/// table is 53,776,394 rows, so every per-row cost is paid 53.8 million times.
///
/// Against the previous `StringBuilder` version, per row:
///
/// 1. one heap allocation for the `String` `decode_kmer` returns -- gone; the
///    bases are written straight into the value buffer by `decode_kmer_into`,
/// 2. one `String::from_utf8` scan of `k` bytes that cannot fail (every byte
///    comes from the `b"ACGT"` literal) -- gone; what remains is a single
///    `std::str::from_utf8` over the whole buffer inside `StringArray::
///    try_new`, the same bytes scanned once contiguously instead of in 53.8
///    million separate calls, plus one `is_char_boundary` per row (a load and
///    a compare) against the allocate/free pair it replaces,
/// 3. one `memcpy` of those `k` bytes from the `String` into the builder's
///    value buffer -- gone; the bases are written once instead of twice
///    (~1.67 GB not copied at benchmark scale),
/// 4. one `free` when the `String` is dropped -- gone,
/// 5. for each of the three columns, one `NullBufferBuilder::append_non_null`
///    -- an `Option` check plus a counter increment that can never produce a
///    null here, since no column is nullable -- gone; `Vec::push` does the
///    capacity check alone. 3 x 53.8 million branches removed.
///
/// The `i32` offset cast is proved sound once per call rather than checked
/// once per row (which is what `StringBuilder`'s internal `expect("offset
/// overflow")` does): each row contributes exactly `k` bytes, so `n * k` is
/// the exact final buffer length and bounding it up front bounds every
/// intermediate offset. That also turns an unwind at the FFI boundary into an
/// ordinary `Export` error.
/// `with_sequence` mirrors `export::counts_schema`'s own flag: the decoded
/// `kmer_sequence` column costs one `decode_kmer_into` call and one offset
/// push per row, on top of everything else this function already does, and
/// the majority of this crate's own Python-side ML surface never reads it.
/// Off by default (see `count()`'s `with_sequence=false`), reconstructible
/// without rereading the FASTQ via `KmerCounts.with_sequence()`.
fn build_record_batch(
    counter: &KmerCounter,
    k: usize,
    with_sequence: bool,
) -> Result<RecordBatch, FastDnaError> {
    let schema = export::counts_schema(with_sequence);

    // One lock acquisition on the counter rather than two: `Iter` reports its
    // own exact remaining length, so the row count comes from the same guard
    // that is about to be iterated, instead of a separate `distinct_kmers()`
    // call that locks, re-runs `finalize_inner`'s validity check and unlocks
    // again. It is exact rather than a hint, which is what lets every buffer
    // below be sized once and never regrow.
    let rows = counter.iter();
    let n = rows.size_hint().0;

    let mut kmers: Vec<u64> = Vec::with_capacity(n);
    let mut freqs: Vec<u32> = Vec::with_capacity(n);

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(3);

    if with_sequence {
        // Exact, not an estimate: `decode_kmer_into` appends exactly `k`
        // bytes per k-mer. `checked_mul` because `usize` is 32 bits on some
        // wheel targets; the `i32::MAX` bound is Arrow's, for the offsets
        // of a `Utf8` column.
        let total_seq_bytes = n
            .checked_mul(k)
            .filter(|&bytes| bytes <= i32::MAX as usize)
            .ok_or_else(|| {
                in_memory_export_err(format!(
                    "{n} k-mers of {k} bases exceed the {} byte limit of an Arrow Utf8 column",
                    i32::MAX
                ))
            })?;

        let mut seq_bytes: Vec<u8> = Vec::with_capacity(total_seq_bytes);
        // An offsets buffer has one more entry than it has values: the
        // leading 0 that opens the first string.
        let mut seq_offsets: Vec<i32> = Vec::with_capacity(n + 1);
        seq_offsets.push(0);

        for (kmer_bits, count) in rows {
            kmers.push(kmer_bits);
            kmer::decode_kmer_into(kmer_bits, k, &mut seq_bytes);
            // Bounded by `total_seq_bytes <= i32::MAX` above, so this cast
            // is value-preserving for every row.
            seq_offsets.push(seq_bytes.len() as i32);
            freqs.push(count);
        }

        columns.push(Arc::new(UInt64Array::from(kmers)));
        let offsets = OffsetBuffer::new(ScalarBuffer::from(seq_offsets));
        let seq_arr: ArrayRef = Arc::new(
            StringArray::try_new(offsets, Buffer::from_vec(seq_bytes), None)
                .map_err(in_memory_export_err)?,
        );
        columns.push(seq_arr);
    } else {
        for (kmer_bits, count) in rows {
            kmers.push(kmer_bits);
            freqs.push(count);
        }
        columns.push(Arc::new(UInt64Array::from(kmers)));
    }

    columns.push(Arc::new(UInt32Array::from(freqs)));

    RecordBatch::try_new(schema, columns).map_err(in_memory_export_err)
}

/// The Python-visible result of `count()`. Holds the counter and QC summary
/// so `.qc`, `.total_kmers` and `.distinct_kmers` can be computed lazily
/// rather than all up front. `table_cache` holds the one Arrow batch
/// `.table` may need to build -- see that getter for why.
#[pyclass(name = "KmerCounts", module = "fastdna._core")]
struct PyKmerCounts {
    counter: KmerCounter,
    qc: QcSummary,
    k: usize,
    // Whether `.table` includes `kmer_sequence`. Fixed at construction time
    // by `count(with_sequence=...)`: the cache below holds at most one
    // batch, so this cannot be a per-call choice without invalidating it.
    with_sequence: bool,
    table_cache: OnceLock<RecordBatch>,
}

#[pymethods]
impl PyKmerCounts {
    /// A `pyarrow.RecordBatch` with columns `kmer_u64`, `kmer_sequence`,
    /// `frequency`. `python/fastdna/__init__.py` wraps this in
    /// `pyarrow.Table.from_batches([...])`, itself a cheap wrap rather than
    /// a copy, to present the `pyarrow.Table` the public API promises.
    ///
    /// The hand-off to pyarrow itself is zero-copy, via Arrow's C Data
    /// Interface (`to_pyarrow`, which shares `batch`'s existing buffers
    /// rather than duplicating them) -- but *building* that batch from
    /// `self.counter` is not: decoding each k-mer back to a string is a
    /// copy no matter how it is arranged. That build is therefore released
    /// under `py.allow_threads` (it walks every distinct k-mer, the same
    /// order of cost as the count itself, so holding the GIL for it would
    /// freeze the notebook for a second time right after the first),
    /// cached, and only cloned -- cheap, a handful of `Arc` bumps over the
    /// underlying buffers -- on repeat access, since `r.table.num_rows`
    /// followed by `r.table.to_pandas()` is an entirely natural thing for a
    /// caller to write and must not rebuild the whole table twice.
    #[getter]
    fn table(&self, py: Python<'_>) -> PyResult<PyObject> {
        let batch = match self.table_cache.get() {
            Some(cached) => cached.clone(),
            None => {
                let built = py.allow_threads(|| {
                    build_record_batch(&self.counter, self.k, self.with_sequence)
                })?;
                // `OnceLock::set` can in general lose a race to a
                // concurrent initializer, and that race is real here, not
                // hypothetical: `py.allow_threads` above releases the GIL
                // for the whole build, which is exactly what lets a second
                // Python thread call `.table` on this same object and enter
                // this method concurrently -- the GIL alone does not
                // serialize callers the way it would if it stayed held for
                // this method's whole body. Using `built` regardless of
                // whether `set` won is what makes this correct on its own
                // terms even so: whichever build actually landed in the
                // cache, this call still returns the (equal) batch it just
                // computed rather than trusting that it must have won.
                let _ = self.table_cache.set(built.clone());
                built
            }
        };
        batch.to_pyarrow(py)
    }

    #[getter]
    fn qc(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_reads", self.qc.total_reads)?;
        dict.set_item("total_bases", self.qc.total_bases)?;
        dict.set_item("q20_bases", self.qc.q20_bases)?;
        dict.set_item("q30_bases", self.qc.q30_bases)?;
        dict.set_item("gc_bases", self.qc.gc_bases)?;
        dict.set_item("gc_content_pct", self.qc.gc_content_pct)?;
        dict.set_item("q20_pct", self.qc.q20_pct)?;
        dict.set_item("q30_pct", self.qc.q30_pct)?;
        Ok(dict.into())
    }

    #[getter]
    fn total_kmers(&self) -> u64 {
        self.counter.total_kmers()
    }

    #[getter]
    fn distinct_kmers(&self) -> usize {
        self.counter.distinct_kmers()
    }

    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// `{depth: number of distinct k-mers observed at that depth}`.
    ///
    /// A method rather than a property, matching the design doc's
    /// `r.spectrum()` (§9.2) -- unlike `.table`/`.qc`, which are cheap
    /// accessors, this rebuilds the histogram on every call, which reads
    /// as a call rather than a stored field.
    ///
    /// Exposes `KmerCounter::generate_histogram` so `suggest_min_count()`
    /// (pure Python, `python/fastdna/spectrum.py`) can do local-minimum
    /// detection over it: the correct `min_count` threshold sits in the
    /// valley between the error peak (frequency 1-2) and the true coverage
    /// peak, and differs per sample -- there is no universal default.
    fn spectrum(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (depth, count) in self.counter.generate_histogram() {
            dict.set_item(depth, count)?;
        }
        Ok(dict.into())
    }
}

/// Converts a core `Progress` event into the Python object handed to a
/// user's callback. `ReadsProcessed` becomes a plain Python `int` (so
/// `isinstance(event, int)` identifies it on the Python side); the other
/// variants become a small dict tagged by `"event"`. `count()` only ever
/// emits `ReadsProcessed` and `Finished` today -- the cohort-only variants
/// are handled here too so this stays correct once `count_cohort` exists.
fn progress_event_into_py(py: Python<'_>, event: Progress) -> PyObject {
    match event {
        Progress::ReadsProcessed(n) => n.into_py(py),
        Progress::Finished { reads } => {
            let dict = PyDict::new_bound(py);
            // `PyDict::set_item` only fails on unhashable keys or a
            // conversion failure; string keys and integer values can do
            // neither, so discarding the `Result` here does not hide a
            // reachable error.
            let _ = dict.set_item("event", "finished");
            let _ = dict.set_item("reads", reads);
            dict.into_py(py)
        }
        Progress::SampleStarted { index, total } => {
            let dict = PyDict::new_bound(py);
            let _ = dict.set_item("event", "sample_started");
            let _ = dict.set_item("index", index);
            let _ = dict.set_item("total", total);
            dict.into_py(py)
        }
        Progress::SampleFinished { index, total } => {
            let dict = PyDict::new_bound(py);
            let _ = dict.set_item("event", "sample_finished");
            let _ = dict.set_item("index", index);
            let _ = dict.set_item("total", total);
            dict.into_py(py)
        }
    }
}

/// Opens `path` for streaming, transparently decompressing `.gz` inputs --
/// the same rule `main.rs` applies for the CLI.
fn open_fastq_reader(path: &PathBuf) -> Result<FastqReader<Box<dyn BufRead + Send + 'static>>, FastDnaError> {
    let is_gz = path.extension().is_some_and(|ext| ext == "gz");
    let file = File::open(path).map_err(|e| FastDnaError::Io { path: path.clone(), source: e })?;

    let buf_reader: Box<dyn BufRead + Send + 'static> = if is_gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    Ok(FastqReader::new(buf_reader))
}

/// Counts canonical k-mers in a single FASTQ(.gz) file.
///
/// `threads=None` uses the core's own default (`PipelineConfig::default`),
/// not a hardcoded number duplicated here. `threads=0` is passed straight
/// through to `process_stream_parallel`, which rejects it as
/// `InvalidConfig` -- it must never be silently folded into the `None`
/// default, since that would let the documented "zero worker tasks hangs
/// forever" bug back in through this exact door.
///
/// `progress`, if given, is a Python callable invoked concurrently and
/// re-entrantly from several rayon worker threads at once (design doc,
/// "Callback contract"). Each invocation re-acquires the GIL with
/// `Python::with_gil`, inside this function's single `py.allow_threads`
/// call -- forgetting that release would deadlock every worker on the GIL.
/// A Python exception raised inside `progress` is captured into
/// `callback_error` (a side channel, not a panic -- see that variable's own
/// comment below for why) and the run is stopped via the same cancellation
/// flag Ctrl-C uses; once the worker thread rejoins, `callback_error` is
/// checked first and, if set, turned into `RuntimeError`, matching the
/// documented callback contract without letting the exception silently
/// vanish at the FFI boundary.
///
/// `progress_interval` defaults to the core's own default (100,000 reads)
/// and is exposed as a keyword argument because the default makes progress
/// untestable against small inputs: a run shorter than the interval never
/// emits a single `ReadsProcessed` event.
///
/// Cancellation: the actual counting runs on its own worker thread; this
/// function's thread polls that worker for a result roughly every 50ms
/// and, while waiting, polls `Python::check_signals` (i.e.
/// `PyErr_CheckSignals`) itself, setting the shared cancellation flag the
/// moment a signal (Ctrl-C) is pending. That two-thread split is required,
/// not incidental: `PyErr_CheckSignals` only has an effect when called
/// from the OS thread the interpreter recognizes as its main thread, and a
/// call blocked directly inside `process_stream_parallel` would occupy
/// that thread for the whole run, leaving nothing free to poll. See the
/// comment at the worker's `thread::spawn` call below for how this was
/// verified. The core notices the flag at its next batch boundary and
/// returns `FastDnaError::Cancelled`, which `impl From<..> for PyErr`
/// above maps to `KeyboardInterrupt`.
#[pyfunction]
#[pyo3(signature = (path, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None, progress=None, progress_interval=None, hpc=false, with_sequence=false))]
#[allow(clippy::too_many_arguments)]
fn count(
    py: Python<'_>,
    path: String,
    k: usize,
    min_count: u32,
    max_count: Option<u32>,
    min_quality: f64,
    threads: Option<usize>,
    progress: Option<PyObject>,
    progress_interval: Option<u64>,
    // Collapse homopolymer runs before k-mer extraction. See
    // `pipeline::PipelineConfig::hpc` and the CLI's `--hpc` for the full
    // rationale: off by default (byte-identical to today's output), meant
    // for long-read (Nanopore/PacBio) input where indels inside homopolymer
    // runs, not substitutions, are the dominant sequencing error.
    hpc: bool,
    // Include the decoded `kmer_sequence` column in `.table`. Off by
    // default -- see `export::counts_schema`'s doc comment for the full
    // rationale (it is derivable from `kmer_u64` and costs ~17% of a run's
    // wall time on the benchmark file). `KmerCounts.with_sequence()`
    // reconstructs it locally when this was left off.
    with_sequence: bool,
) -> PyResult<PyKmerCounts> {
    let path_buf = PathBuf::from(path);
    let reader = open_fastq_reader(&path_buf)?;

    // Built once and read four times. `PipelineConfig::default()` is not a
    // constant: its `num_threads` calls `std::thread::available_parallelism`,
    // which is a syscall on Windows and a cgroup-quota file read on Linux, so
    // the previous four separate `default()` calls per `count()` did that work
    // four times over to answer the same question. Scalar field reads below do
    // not move it, so it is still whole for the `..defaults` fill-in.
    let defaults = PipelineConfig::default();

    let num_threads = threads.unwrap_or(defaults.num_threads);
    let progress_interval = progress_interval.unwrap_or(defaults.progress_interval);
    // Batches are the unit progress is accounted in (pipeline.rs emits at
    // most once per batch received, on interval crossing), so a batch far
    // larger than `progress_interval` would silently coarsen progress no
    // matter how small the caller asks for it. Capped at the core's own
    // default batch size so the common case (no explicit interval) is
    // unaffected.
    let batch_size = (progress_interval as usize).clamp(1, defaults.batch_size);
    let config = PipelineConfig { k, min_quality, num_threads, batch_size, progress_interval, hpc, ..defaults };

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_worker = cancel.clone();
    let cancel_for_progress = cancel.clone();
    // Side channel for a Python exception raised inside `progress`, instead
    // of carrying it out via `panic!`. The panic route works (pipeline.rs's
    // `catch_unwind` does convert it into `FastDnaError::Internal`), but it
    // has a real cost: with no custom panic hook installed, Rust's default
    // one writes `thread '<unnamed>' panicked at ...` to stderr *before*
    // `catch_unwind` ever gets a chance to swallow it -- once per worker
    // that hit the panic, unconditionally, for a library that otherwise
    // never writes to stderr unasked. A process-global hook could suppress
    // that, but installing one here would also silence panic output for the
    // *host* application's own unrelated threads for as long as the hook is
    // installed -- not this crate's stderr to give up. `Mutex`, not
    // `OnceLock`: `OnceLock<T>: Sync` requires `T: Sync`, which `PyErr` is
    // not guaranteed to be, so `Arc<OnceLock<PyErr>>` would not necessarily
    // be shareable across the threads this needs to cross; `Mutex<T>: Sync`
    // only requires `T: Send`, which `PyErr` is.
    let callback_error: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let callback_error_for_progress = callback_error.clone();
    let (result_tx, result_rx) = mpsc::channel::<Result<(KmerCounter, QcSummary, u64), FastDnaError>>();

    // The actual counting runs on its own OS thread, not on the thread that
    // entered this function. This inversion exists entirely because of a
    // `PyErr_CheckSignals` constraint verified empirically while building
    // this (not merely assumed from documentation): CPython only acts on a
    // pending signal when `PyErr_CheckSignals` is called *from the OS
    // thread the interpreter recognizes as its main thread* -- calling it
    // from any other thread, including a dedicated "watcher" thread spawned
    // solely to poll it, is a silent no-op. An earlier version of this
    // function did exactly that (a separate watcher thread polling
    // `check_signals` while this thread blocked inside `process_stream_
    // parallel`), and Ctrl-C during a real long run was not caught until
    // the call finished on its own -- confirmed with an instrumented
    // build, not merely reasoned about. Moving the heavy work to a worker
    // thread frees up *this* thread -- the one Python actually considers
    // main, in the ordinary case of `count()` being called from a
    // notebook's or script's main thread -- to keep polling `check_signals`
    // itself while the worker runs.
    //
    // `progress_closure` is moved into the worker thread whole (rather than
    // built and borrowed from out here, as a plain watcher-thread design
    // would do) because `ProgressFn<'_>` is a borrow and `thread::spawn`
    // requires `'static`; taking the reference inside the worker's own
    // closure, after the move, satisfies both.
    let worker_handle = thread::spawn(move || {
        let progress_closure = progress.map(|cb| {
            move |event: Progress| {
                Python::with_gil(|py| {
                    let py_event = progress_event_into_py(py, event);
                    if let Err(e) = cb.call1(py, (py_event,)) {
                        // Record the first callback error (matching
                        // pipeline.rs's own "first panic wins" priority for
                        // worker panics) and ask the whole pipeline to stop
                        // via the same cancellation flag Ctrl-C uses, rather
                        // than panicking -- see `callback_error`'s
                        // declaration above for why. `count()` checks this
                        // channel, once the worker thread has rejoined,
                        // ahead of anything the pipeline itself returned.
                        let mut slot = match callback_error_for_progress.lock() {
                            Ok(guard) => guard,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                        drop(slot);
                        cancel_for_progress.store(true, Ordering::Relaxed);
                    }
                });
            }
        });
        let progress_ref: crate::progress::ProgressFn<'_> =
            progress_closure.as_ref().map(|f| f as &(dyn Fn(Progress) + Send + Sync));

        // `MemoryPolicy::default()` reproduces this function's previous
        // in-memory-only behavior exactly (no `estimated_input_bytes`, so
        // `resolve_strategy` has no basis to choose the disk strategy on
        // its own) unless a caller explicitly opts in via the
        // `FASTDNA_STRATEGY` / `FASTDNA_MAX_RAM_BYTES` environment
        // variables `resolve_strategy` reads as a fallback -- see that
        // function's doc comment. Those exist specifically because this
        // binding has no dedicated `strategy=`/`max_ram=` keyword argument
        // of its own yet (adding one is out of scope here); they let
        // `scripts/bench/crosscheck.py` and similar tooling exercise both
        // counting strategies through the same public `fastdna.count()`
        // Python API without needing one.
        let outcome = process_stream_parallel_with_policy(
            reader,
            config,
            &path_buf,
            progress_ref,
            Some(cancel_for_worker),
            MemoryPolicy::default(),
        )
        .map(|(counter, qc, reads, _decision)| (counter, qc, reads));
        // The receiving end only ever stops listening after it has already
        // gotten a result (see the loop below), so a failed send here is
        // unreachable; there is nothing useful to do with that error even
        // if it somehow occurred.
        let _ = result_tx.send(outcome);
    });

    // Released for the duration of the count: without this, a long-running
    // call freezes the calling notebook -- no progress, no Ctrl-C -- for as
    // long as the count takes. This closure runs on the thread that called
    // `count()`, polling for the worker's result without blocking on it
    // indefinitely, so it can also poll `check_signals` on the one thread
    // where doing so actually has an effect (see the comment above).
    let (outcome, signal_error): (CountOutcome, Option<PyErr>) =
        py.allow_threads(move || {
            // `check_signals` returning `Err` means CPython consumed the
            // pending signal (its flag is cleared) and handed us the
            // exception to raise. It must be kept: if the worker finishes
            // successfully inside the same 50ms window, discarding it
            // would swallow the user's Ctrl-C entirely -- a completed
            // result would come back as if nothing had been pressed.
            let mut signal_error: Option<PyErr> = None;
            let outcome = loop {
                match result_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(outcome) => break outcome,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Err(err) = Python::with_gil(|py| py.check_signals()) {
                            signal_error = Some(err);
                            // Setting the flag does not itself produce a
                            // result: the worker notices it at its next
                            // batch boundary and returns `Cancelled` on its
                            // own, which the next loop iteration's
                            // `recv_timeout` picks up like any other
                            // outcome.
                            cancel.store(true, Ordering::Relaxed);
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break Err(FastDnaError::Internal {
                            detail: "count() worker thread ended without sending a result"
                                .to_string(),
                        });
                    }
                }
            };
            (outcome, signal_error)
        });

    // The worker has already sent its result by the time the loop above
    // observes it, so this returns almost immediately; joining still
    // matters so the thread is not left detached and running past the end
    // of this function.
    let _ = worker_handle.join();

    // A raising progress callback takes priority over whatever the pipeline
    // itself returned: cancelling via the shared flag (see `callback_error`
    // above) can just as easily leave `outcome` looking like an ordinary
    // `Cancelled` or even a completed `Ok` (e.g. if the exception was raised
    // from the final `Finished` event, after all counting work was already
    // done) -- neither of those is the error the caller's callback actually
    // raised, so this is checked unconditionally, before `outcome` is
    // trusted either way.
    let callback_error = match callback_error.lock() {
        Ok(mut guard) => guard.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    };
    if let Some(e) = callback_error {
        return Err(PyRuntimeError::new_err(format!("progress callback raised a Python exception: {e}")));
    }

    // Second priority: a consumed interrupt. The exception CPython handed
    // over (typically KeyboardInterrupt) outranks whatever the worker
    // returned -- including a successful count that slipped in during the
    // signal race window -- because the user's Ctrl-C was already eaten
    // from the interpreter's pending-signal state and this is the only
    // place left that can honour it.
    if let Some(err) = signal_error {
        return Err(err);
    }

    // Prune stays under the same released-GIL window as the count: it is
    // an O(distinct k-mers) `retain` over the whole table, the same order
    // of cost as the count itself, so applying it after the GIL is
    // reacquired would freeze the notebook a second time immediately after
    // the first. Matches what `main.rs` does: frequency filters are
    // applied in RAM, after the pipeline and before the result is handed
    // back.
    let outcome = py.allow_threads(|| {
        outcome.map(|(mut counter, qc, total_reads)| {
            counter.prune(min_count, max_count);
            (counter, qc, total_reads)
        })
    });

    let (counter, qc, _total_reads) = outcome?;

    Ok(PyKmerCounts { counter, qc, k, with_sequence, table_cache: OnceLock::new() })
}

/// The Python-visible result of `peek()`. Wraps `preview::PreviewStats`
/// directly -- the sampling logic itself lives in `src/preview.rs`, with no
/// PyO3 dependency, so it stays unit-testable via `cargo test` alone.
#[pyclass(name = "Preview", module = "fastdna._core")]
struct PyPreview {
    inner: preview::PreviewStats,
}

#[pymethods]
impl PyPreview {
    #[getter]
    fn n_reads_sampled(&self) -> usize {
        self.inner.n_reads_sampled
    }

    /// `(min, median, max)` read length among the sampled reads.
    #[getter]
    fn read_length(&self) -> (usize, usize, usize) {
        self.inner.read_length
    }

    #[getter]
    fn gc_content(&self) -> f64 {
        self.inner.gc_content
    }

    #[getter]
    fn sample_distinct_kmers(&self) -> usize {
        self.inner.sample_distinct_kmers
    }

    /// The largest odd `k` at most `median_read_length / 3`, clamped to
    /// `1..=32`. See `preview::PreviewStats::suggest_k` for why odd matters.
    fn suggest_k(&self) -> usize {
        self.inner.suggest_k()
    }

    fn __repr__(&self) -> String {
        format!(
            "Preview(n_reads_sampled={}, read_length={:?}, gc_content={:.3}, suggested_k={})",
            self.inner.n_reads_sampled,
            self.inner.read_length,
            self.inner.gc_content,
            self.inner.suggest_k()
        )
    }
}

/// Samples the first `n_reads` records of a FASTQ(.gz) file and reports its
/// read-length geometry, GC content, and a suggested `k` -- in milliseconds,
/// without reading the rest of the file. Its reason to exist: `k=31` is
/// everyone's default and it is wrong for short reads (design doc §9.5).
///
/// Released under `py.allow_threads` like `count()`'s own I/O-bound work:
/// `n_reads` is caller-controlled (`preview::MAX_N_READS` rejects only the
/// clearly unreasonable values), and an ordinary-looking but large request
/// against a large file could otherwise run for a while holding the GIL,
/// freezing the calling interpreter with no way to interrupt it -- the same
/// problem `count()`'s own GIL release and Ctrl-C polling exist to avoid.
#[pyfunction]
#[pyo3(signature = (path, n_reads=10_000))]
fn peek(py: Python<'_>, path: String, n_reads: usize) -> PyResult<PyPreview> {
    let inner = py.allow_threads(|| preview::peek(PathBuf::from(path), n_reads))?;
    Ok(PyPreview { inner })
}

/// Reports facts a remote bug report cannot otherwise supply: the installed
/// version, the maximum supported `k` (fixed by the 2-bit packing in
/// `kmer.rs`), and whether AVX2 is live on *this* CPU right now. The last of
/// these is a runtime check, not a compile-time one: the same wheel ships to
/// AVX2 and non-AVX2 machines, so a compile-time-only answer would be wrong
/// wherever it runs that differs from the build machine. Without it, "it's
/// slow on my Mac" is undiagnosable remotely (design doc §9.1).
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item("max_k", 32usize)?;
    dict.set_item("avx2", avx2_is_live())?;
    Ok(dict.into())
}

/// Runtime AVX2 detection, gated at compile time on the architecture only
/// (never on `target_feature`, which would bake a single build's CPU into
/// every wheel -- see design doc §11, "Fixing simd.rs").
fn avx2_is_live() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The Python-visible result of `sketch()` and `load_sketch()`. Wraps
/// `sketch::GenomeSketch` (MinHash bottom-k sketching, design doc §9.4) --
/// a fingerprint that estimates similarity between two samples without
/// ever materializing their full k-mer sets side by side.
#[pyclass(name = "Sketch", module = "fastdna._core")]
struct PySketch {
    inner: GenomeSketch,
}

#[pymethods]
impl PySketch {
    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }

    #[getter]
    fn sketch_size(&self) -> usize {
        self.inner.sketch_size
    }

    /// Symmetric similarity: the fraction of the union of both sketches'
    /// k-mer sets that is shared, estimated from the bottom-k overlap.
    /// Penalizes genome-size differences -- two sketches from genomes of
    /// very different sizes report a low Jaccard even if the smaller one
    /// is entirely contained in the larger. Raises `ValueError` if the two
    /// sketches were built with different `k` (comparing them has no
    /// biological meaning).
    fn jaccard(&self, other: &PySketch) -> PyResult<f64> {
        Ok(self.inner.jaccard(&other.inner)?)
    }

    /// Asymmetric containment: what fraction of *this* sketch's k-mers
    /// also appear in `other`. Unlike `jaccard`, does not penalize a size
    /// mismatch -- the question this answers is "is this (small) pathogen
    /// present in this (large) metagenomic sample", not "how similar are
    /// these two genomes overall". `self.containment(other)` and
    /// `other.containment(self)` are different questions with different
    /// answers. Same `ValueError` behaviour as `jaccard` for mismatched
    /// `k`.
    fn containment(&self, other: &PySketch) -> PyResult<f64> {
        Ok(self.inner.containment(&other.inner)?)
    }

    /// Estimates the per-base mutation rate implied by `.jaccard()`, under
    /// the Poisson mutation model Mash itself uses. `.jaccard()` alone is
    /// a similarity score; this is the same overlap turned into an
    /// evolutionary-distance estimate, which is what "Mash-style"
    /// comparison actually promises. See `GenomeSketch::mash_distance`'s
    /// own doc comment for the formula and what this does not give (a
    /// p-value against a null hypothesis, which needs a genome-length
    /// estimate this type does not have).
    fn mash_distance(&self, other: &PySketch) -> PyResult<f64> {
        Ok(self.inner.mash_distance(&other.inner)?)
    }

    /// Persists the sketch as JSON. Computing a sketch is the expensive
    /// part (a full pass over the FASTQ file); comparing saved sketches
    /// afterwards is what keeps an N-sample comparison from re-reading N
    /// large files on every later query.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(path)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Sketch(k={}, sketch_size={}, hashes={})", self.inner.k, self.inner.sketch_size, self.inner.hashes.len())
    }
}

/// Builds a MinHash sketch of a single FASTQ(.gz) file by streaming it --
/// memory stays bounded by `sketch_size` regardless of file size, unlike
/// `count()`, which must hold every distinct k-mer.
///
/// Released under `py.allow_threads` like `count()`'s and `peek()`'s own
/// I/O-bound work, for the same reason: a large file could otherwise run
/// for a while holding the GIL, freezing the calling interpreter with no
/// way to interrupt it.
#[pyfunction]
#[pyo3(signature = (path, k=21, sketch_size=1000))]
fn sketch(py: Python<'_>, path: String, k: usize, sketch_size: usize) -> PyResult<PySketch> {
    let inner = py.allow_threads(|| GenomeSketch::from_path(path, sketch_size, k))?;
    Ok(PySketch { inner })
}

/// Loads a sketch previously written by `Sketch.save`.
#[pyfunction]
fn load_sketch(path: String) -> PyResult<PySketch> {
    let inner = GenomeSketch::load(path)?;
    Ok(PySketch { inner })
}

/// The Python-visible result of `frac_sketch()` and `load_frac_sketch()`.
/// Wraps `sketch::FracSketch` -- a FracMinHash ("scaled MinHash")
/// fingerprint whose size scales with the underlying k-mer set's true
/// cardinality instead of being clamped to a fixed count the way
/// `Sketch`'s bottom-k is. Use this instead of `Sketch` for containment
/// queries where the two sides being compared differ a lot in size (e.g. a
/// small pathogen sketch against a large metagenomic sample) -- see
/// `FracSketch`'s own doc comment in `sketch.rs` for why bottom-k is biased
/// there and this is not.
#[pyclass(name = "FracSketch", module = "fastdna._core")]
struct PyFracSketch {
    inner: FracSketch,
}

#[pymethods]
impl PyFracSketch {
    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }

    #[getter]
    fn scale(&self) -> u64 {
        self.inner.scale
    }

    /// Asymmetric containment: what fraction of *this* sketch's k-mers also
    /// appear in `other`. Unlike `Sketch.containment`, the estimate does
    /// not shrink or lose resolution as `other`'s underlying set grows --
    /// see `FracSketch::containment`'s doc comment. Raises `ValueError` if
    /// the two sketches were built with a different `k` or a different
    /// `scale`.
    fn containment(&self, other: &PyFracSketch) -> PyResult<f64> {
        Ok(self.inner.containment(&other.inner)?)
    }

    /// Symmetric similarity: the fraction of the union of both sketches'
    /// k-mer sets that is shared. Same `ValueError` behaviour as
    /// `containment` for a mismatched `k` or `scale`.
    fn jaccard(&self, other: &PyFracSketch) -> PyResult<f64> {
        Ok(self.inner.jaccard(&other.inner)?)
    }

    /// Persists the sketch as JSON, for :func:`load_frac_sketch` later.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(path)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("FracSketch(k={}, scale={}, hashes={})", self.inner.k, self.inner.scale, self.inner.hashes.len())
    }
}

/// Builds a FracMinHash ("scaled MinHash") sketch of a single FASTQ(.gz)
/// file by streaming it. Unlike `sketch()`'s bottom-k, memory is bounded by
/// `~|distinct k-mers| / scale`, not by a fixed constant -- the sketch's
/// size is allowed to track the file's true k-mer cardinality, which is
/// what makes its containment estimate unbiased against `sketch()`'s when
/// comparing sets of very different sizes (see `FracSketch`'s doc comment
/// in `sketch.rs`).
///
/// `scale=1000` by default (a k-mer's hash is kept with probability
/// 1/1000), the same order of magnitude as sourmash's own common default.
#[pyfunction]
#[pyo3(signature = (path, k=21, scale=1000))]
fn frac_sketch(py: Python<'_>, path: String, k: usize, scale: u64) -> PyResult<PyFracSketch> {
    let inner = py.allow_threads(|| FracSketch::from_path(path, scale, k))?;
    Ok(PyFracSketch { inner })
}

/// Loads a FracSketch previously written by `FracSketch.save`.
#[pyfunction]
fn load_frac_sketch(path: String) -> PyResult<PyFracSketch> {
    let inner = FracSketch::load(path)?;
    Ok(PyFracSketch { inner })
}

/// The Python-visible random-access query handle over a sorted k-mer table
/// (`ktab::KmerTable`) -- the query-side counterpart to `count()`'s
/// write-only Parquet output (`docs/feature-gap-analysis.md`'s S1). Every
/// counting run's default `.parquet` output already qualifies: no separate
/// conversion step exists, see `ktab.rs`'s module doc comment.
#[pyclass(name = "KmerTable", module = "fastdna._core")]
struct PyKmerTable {
    inner: KmerTable,
}

/// Accepts either a Python `int` (the table's raw `kmer_u64` encoding,
/// taken as-is) or a `str` (a DNA sequence of exactly the table's `k`,
/// canonicalized the same way counting does) -- see `ktab::
/// encode_query_kmer`'s doc comment for the same contract on the Rust/CLI
/// side. Kept as a free function rather than inlined into every
/// `#[pymethods]` call site so `get` and `__getitem__` cannot drift on what
/// counts as a valid argument.
fn extract_kmer_arg(kmer: &Bound<'_, PyAny>, k: usize) -> PyResult<u64> {
    if let Ok(raw) = kmer.extract::<u64>() {
        return Ok(raw);
    }
    let sequence: String = kmer.extract().map_err(|_| {
        PyValueError::new_err(
            "kmer must be a DNA sequence (str) of length k, or the table's raw kmer_u64 encoding (a non-negative int)",
        )
    })?;
    Ok(ktab::encode_query_kmer(&sequence, k)?)
}

#[pymethods]
impl PyKmerTable {
    /// Opens `path` and validates it as a queryable k-mer table (footer
    /// metadata and per-row-group statistics only -- see `KmerTable::open`'s
    /// doc comment for exactly what is and is not checked). Raises
    /// `LoadError` for a file that is not a FastDNA k-mer table.
    #[staticmethod]
    fn open(py: Python<'_>, path: String) -> PyResult<Self> {
        let inner = py.allow_threads(|| KmerTable::open(path))?;
        Ok(Self { inner })
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// Total row count (the table's distinct-k-mer count), read once from
    /// Parquet metadata at `open` time -- no row was ever decoded to answer
    /// this.
    fn __len__(&self) -> usize {
        self.inner.len() as usize
    }

    /// The frequency recorded for `kmer`, or `None` if it is absent from the
    /// table. See `extract_kmer_arg` for what `kmer` may be.
    fn get(&self, py: Python<'_>, kmer: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
        let encoded = extract_kmer_arg(kmer, self.inner.k())?;
        Ok(py.allow_threads(|| self.inner.get(encoded))?)
    }

    /// Same lookup as `get`, but raises `KeyError` instead of returning
    /// `None` for an absent k-mer -- the usual Python mapping convention
    /// (`table[kmer]` vs. `table.get(kmer)`), mirroring `dict`.
    fn __getitem__(&self, py: Python<'_>, kmer: &Bound<'_, PyAny>) -> PyResult<u32> {
        match self.get(py, kmer)? {
            Some(count) => Ok(count),
            None => Err(PyKeyError::new_err(kmer.repr()?.to_string())),
        }
    }

    /// `kmer in table` -- `True` iff `get(kmer)` would not be `None`.
    fn __contains__(&self, py: Python<'_>, kmer: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.get(py, kmer)?.is_some())
    }

    fn __repr__(&self) -> String {
        format!("KmerTable(k={}, len={})", self.inner.k(), self.inner.len())
    }
}

/// The Python-visible outcome of a `filter_reads` call: how many reads were
/// read in total, and how many were written to the output under the
/// chosen mode -- the Python-visible counterpart of `read_filter::
/// FilterStats`.
#[pyclass(name = "FilterStats", module = "fastdna._core")]
struct PyFilterStats {
    #[pyo3(get)]
    reads_total: u64,
    #[pyo3(get)]
    reads_written: u64,
}

#[pymethods]
impl PyFilterStats {
    fn __repr__(&self) -> String {
        format!("FilterStats(reads_total={}, reads_written={})", self.reads_total, self.reads_written)
    }
}

/// Parses `filter_reads`'s `mode` string argument into `read_filter::
/// FilterMode`. A plain string rather than a Python enum, for the same
/// reason `parse_combine_op` above is: nothing else in this FFI surface
/// exposes a Python-level enum type.
fn parse_filter_mode(mode: &str) -> PyResult<read_filter::FilterMode> {
    match mode {
        "keep" => Ok(read_filter::FilterMode::Keep),
        "discard" => Ok(read_filter::FilterMode::Discard),
        other => Err(PyValueError::new_err(format!("mode must be 'keep' or 'discard' -- got '{other}'"))),
    }
}

/// Filters one or more FASTQ/FASTA files against `table`, writing reads
/// that should be kept to `output` -- the Python-visible counterpart of
/// `fastdna filter` / `read_filter::run_filter`
/// (`docs/feature-gap-analysis.md`'s S4). See `read_filter.rs`'s module doc
/// comment for the full design: threshold semantics (`min_fraction`,
/// inclusive `>=`), what `mode` ("keep"/"discard") means, and the
/// documented single-end-only scope (each of `inputs` is filtered
/// independently; paired-end R1/R2 synchronization is not implemented).
///
/// `table` is taken as an already-open `KmerTable` (`python/fastdna/
/// __init__.py`'s `KmerTable.filter_reads` calls this with `self._raw`),
/// the same "the Python wrapper builds the call, the FFI function takes
/// already-open handles" shape `ktab_union`/`ktab_intersect`/`ktab_diff`
/// already use.
///
/// Released under `py.allow_threads` like every other bulk I/O operation in
/// this module: filtering a real FASTQ file is I/O- and CPU-bound work with
/// nothing Python-specific in it, and holding the GIL for it would freeze
/// the calling interpreter for the duration.
#[pyfunction]
#[pyo3(signature = (table, inputs, mode, output, min_fraction=0.1))]
fn filter_reads(
    py: Python<'_>,
    table: PyRef<'_, PyKmerTable>,
    inputs: Vec<String>,
    mode: &str,
    output: String,
    min_fraction: f64,
) -> PyResult<PyFilterStats> {
    let mode = parse_filter_mode(mode)?;
    let table_inner = table.inner.clone();
    let input_specs: Vec<crate::fastq::InputSpec> =
        inputs.iter().map(|p| crate::fastq::InputSpec::from_arg(std::path::Path::new(p))).collect();
    let output_path = PathBuf::from(output);

    let stats = py.allow_threads(|| -> Result<read_filter::FilterStats, FastDnaError> {
        let index = read_filter::ReferenceIndex::from_table(&table_inner)?;
        read_filter::run_filter(input_specs, &index, mode, min_fraction, &output_path)
    })?;

    Ok(PyFilterStats { reads_total: stats.reads_total, reads_written: stats.reads_written })
}

/// The Python-visible outcome of a `filter_reads_paired` call: how many
/// *pairs* were read in total, and how many pairs were written -- the
/// Python-visible counterpart of `read_filter::PairedFilterStats`.
#[pyclass(name = "PairedFilterStats", module = "fastdna._core")]
struct PyPairedFilterStats {
    #[pyo3(get)]
    pairs_total: u64,
    #[pyo3(get)]
    pairs_written: u64,
}

#[pymethods]
impl PyPairedFilterStats {
    fn __repr__(&self) -> String {
        format!("PairedFilterStats(pairs_total={}, pairs_written={})", self.pairs_total, self.pairs_written)
    }
}

/// Filters synchronized R1/R2 FASTQ/FASTA file pairs against `table`,
/// writing pairs that should be kept to `output`/`output2` -- the
/// Python-visible counterpart of `fastdna filter --input2 ... --output2
/// ...` / `read_filter::run_filter_paired`. See `read_filter.rs`'s module
/// doc comment ("Paired-end (R1/R2) synchronized filtering") for the full
/// design: a pair is kept or discarded as *one unit* if either mate
/// matches the reference, never independently per mate, and both mates of
/// a kept pair are always written together so the two output streams can
/// never drift out of sync with each other.
///
/// `inputs`/`inputs2` are each concatenated into one stream first (the
/// same multi-file convention `filter_reads`'s own `inputs` already uses),
/// and it is those two streams that are paired, record by record; they do
/// not need to hold the same number of *files*, only the same *total*
/// record count -- see `fastq::PairedSourceReader`'s doc comment. A
/// genuine mismatch there raises rather than silently truncating either
/// side.
///
/// `table` is taken as an already-open `KmerTable` (`python/fastdna/
/// __init__.py`'s `KmerTable.filter_reads_paired` calls this with
/// `self._raw`), the same "the Python wrapper builds the call, the FFI
/// function takes already-open handles" shape `filter_reads` already uses.
///
/// Released under `py.allow_threads` for the same reason `filter_reads`
/// already is: filtering real FASTQ files is I/O- and CPU-bound work with
/// nothing Python-specific in it.
#[pyfunction]
#[pyo3(signature = (table, inputs, inputs2, mode, output, output2, min_fraction=0.1))]
#[allow(clippy::too_many_arguments)]
fn filter_reads_paired(
    py: Python<'_>,
    table: PyRef<'_, PyKmerTable>,
    inputs: Vec<String>,
    inputs2: Vec<String>,
    mode: &str,
    output: String,
    output2: String,
    min_fraction: f64,
) -> PyResult<PyPairedFilterStats> {
    let mode = parse_filter_mode(mode)?;
    let table_inner = table.inner.clone();
    let inputs_r1: Vec<crate::fastq::InputSpec> =
        inputs.iter().map(|p| crate::fastq::InputSpec::from_arg(std::path::Path::new(p))).collect();
    let inputs_r2: Vec<crate::fastq::InputSpec> =
        inputs2.iter().map(|p| crate::fastq::InputSpec::from_arg(std::path::Path::new(p))).collect();
    let output_path = PathBuf::from(output);
    let output2_path = PathBuf::from(output2);

    let stats = py.allow_threads(|| -> Result<read_filter::PairedFilterStats, FastDnaError> {
        let index = read_filter::ReferenceIndex::from_table(&table_inner)?;
        read_filter::run_filter_paired(
            inputs_r1, inputs_r2, &index, mode, min_fraction, &output_path, &output2_path,
        )
    })?;

    Ok(PyPairedFilterStats { pairs_total: stats.pairs_total, pairs_written: stats.pairs_written })
}

/// The Python-visible outcome of a `profile_reads` call: how many reads were
/// read in total, and how many of those yielded at least one k-mer to
/// profile -- the Python-visible counterpart of `read_profile::
/// ProfileStats`.
#[pyclass(name = "ProfileStats", module = "fastdna._core")]
struct PyProfileStats {
    #[pyo3(get)]
    reads_total: u64,
    #[pyo3(get)]
    reads_profiled: u64,
}

#[pymethods]
impl PyProfileStats {
    fn __repr__(&self) -> String {
        format!("ProfileStats(reads_total={}, reads_profiled={})", self.reads_total, self.reads_profiled)
    }
}

/// Profiles one or more FASTQ/FASTA files against `table`, writing an
/// RLE-compressed per-read profile to `output` and a per-read summary table
/// to `summary` -- the Python-visible counterpart of `fastdna profile` /
/// `read_profile::run_profile` (`docs/feature-gap-analysis.md`'s S3). See
/// `read_profile.rs`'s module doc comment for the full design: why the
/// profile is RLE-compressed rather than one row per base, the summary
/// table's column naming, and the single-end-only scope shared with
/// `filter_reads`.
///
/// `table` is taken as an already-open `KmerTable`, the same "the Python
/// wrapper builds the call, the FFI function takes already-open handles"
/// shape `filter_reads`/`ktab_union`/`ktab_intersect`/`ktab_diff` already
/// use.
///
/// Released under `py.allow_threads` like every other bulk I/O operation in
/// this module: profiling a real FASTQ file is I/O- and CPU-bound work with
/// nothing Python-specific in it, and holding the GIL for it would freeze
/// the calling interpreter for the duration.
#[pyfunction]
#[pyo3(signature = (table, inputs, output, summary))]
fn profile_reads(
    py: Python<'_>,
    table: PyRef<'_, PyKmerTable>,
    inputs: Vec<String>,
    output: String,
    summary: String,
) -> PyResult<PyProfileStats> {
    let table_inner = table.inner.clone();
    let input_specs: Vec<crate::fastq::InputSpec> =
        inputs.iter().map(|p| crate::fastq::InputSpec::from_arg(std::path::Path::new(p))).collect();
    let output_path = PathBuf::from(output);
    let summary_path = PathBuf::from(summary);

    let stats = py.allow_threads(|| -> Result<read_profile::ProfileStats, FastDnaError> {
        let index = read_profile::ProfileIndex::from_table(&table_inner)?;
        read_profile::run_profile(input_specs, &index, &output_path, &summary_path)
    })?;

    Ok(PyProfileStats { reads_total: stats.reads_total, reads_profiled: stats.reads_profiled })
}

/// Parses the `combine` string argument `ktab_union`/`ktab_intersect`
/// accept into `setops::CombineOp`. A plain string rather than a Python
/// enum: nothing else in this FFI surface (`ffi.rs`) exposes a Python-level
/// enum type, and a mistyped value is exactly as actionable as a `ValueError`
/// naming the three valid spellings.
fn parse_combine_op(combine: &str) -> PyResult<setops::CombineOp> {
    match combine {
        "sum" => Ok(setops::CombineOp::Sum),
        "min" => Ok(setops::CombineOp::Min),
        "max" => Ok(setops::CombineOp::Max),
        other => Err(PyValueError::new_err(format!(
            "combine must be one of 'sum', 'min', 'max' -- got '{other}'"
        ))),
    }
}

/// Union of several k-mer tables, written to `output` and reopened as a new
/// `KmerTable` -- the Python-visible counterpart of `fastdna union` /
/// `setops::union` (`docs/feature-gap-analysis.md`'s S2). See `setops::
/// union`'s doc comment for the streaming merge and what `combine` ("sum",
/// "min" or "max") means.
///
/// Takes a `Vec` of already-open tables rather than a variadic `*args`
/// spelling: `python/fastdna/__init__.py`'s `KmerTable.union(*others,
/// output=...)` builds this list on the Python side, keeping the choice of
/// call-site ergonomics (a method taking `*others`) separate from this
/// FFI function's own shape.
///
/// Released under `py.allow_threads` like every other bulk operation in
/// this module (`count`, `sketch`, ...): merging and re-exporting a real
/// k-mer table is I/O- and CPU-bound work with nothing Python-specific in
/// it, and holding the GIL for it would freeze the calling interpreter for
/// the duration.
#[pyfunction]
#[pyo3(signature = (tables, output, combine="sum"))]
fn ktab_union(py: Python<'_>, tables: Vec<PyRef<'_, PyKmerTable>>, output: String, combine: &str) -> PyResult<PyKmerTable> {
    let combine = parse_combine_op(combine)?;
    let inner: Vec<KmerTable> = tables.iter().map(|t| t.inner.clone()).collect();
    let opened = py.allow_threads(|| -> Result<KmerTable, FastDnaError> {
        let input_paths: Vec<&Path> = inner.iter().map(KmerTable::path).collect();
        setops::guard_against_output_overwrite(&input_paths, Path::new(&output))?;
        let rows = setops::union(&inner, combine)?;
        let k = inner.first().map(KmerTable::k).unwrap_or(0);
        export::export_pairs_parquet(rows, &output, k)?;
        KmerTable::open(&output)
    })?;
    Ok(PyKmerTable { inner: opened })
}

/// Intersection of several k-mer tables, written to `output` and reopened
/// as a new `KmerTable` -- the Python-visible counterpart of `fastdna
/// intersect` / `setops::intersect`. See `ktab_union`'s doc comment for why
/// this takes a `Vec` of already-open tables and runs under `py.
/// allow_threads`.
#[pyfunction]
#[pyo3(signature = (tables, output, combine="min"))]
fn ktab_intersect(
    py: Python<'_>,
    tables: Vec<PyRef<'_, PyKmerTable>>,
    output: String,
    combine: &str,
) -> PyResult<PyKmerTable> {
    let combine = parse_combine_op(combine)?;
    let inner: Vec<KmerTable> = tables.iter().map(|t| t.inner.clone()).collect();
    let opened = py.allow_threads(|| -> Result<KmerTable, FastDnaError> {
        let input_paths: Vec<&Path> = inner.iter().map(KmerTable::path).collect();
        setops::guard_against_output_overwrite(&input_paths, Path::new(&output))?;
        let rows = setops::intersect(&inner, combine)?;
        let k = inner.first().map(KmerTable::k).unwrap_or(0);
        export::export_pairs_parquet(rows, &output, k)?;
        KmerTable::open(&output)
    })?;
    Ok(PyKmerTable { inner: opened })
}

/// Asymmetric difference (`a` minus `subtract`), written to `output` and
/// reopened as a new `KmerTable` -- the Python-visible counterpart of
/// `fastdna diff` / `setops::diff` (the reference-subtraction/host-removal
/// use case). See `setops::diff`'s doc comment for what
/// `max_subtract_count` means and `ktab_union`'s for why this runs under
/// `py.allow_threads`.
#[pyfunction]
#[pyo3(signature = (a, subtract, output, max_subtract_count=0))]
fn ktab_diff(
    py: Python<'_>,
    a: PyRef<'_, PyKmerTable>,
    subtract: Vec<PyRef<'_, PyKmerTable>>,
    output: String,
    max_subtract_count: u32,
) -> PyResult<PyKmerTable> {
    let a_inner = a.inner.clone();
    let subtract_inner: Vec<KmerTable> = subtract.iter().map(|t| t.inner.clone()).collect();
    let opened = py.allow_threads(|| -> Result<KmerTable, FastDnaError> {
        let input_paths: Vec<&Path> =
            std::iter::once(a_inner.path()).chain(subtract_inner.iter().map(KmerTable::path)).collect();
        setops::guard_against_output_overwrite(&input_paths, Path::new(&output))?;
        let rows = setops::diff(&a_inner, &subtract_inner, max_subtract_count)?;
        export::export_pairs_parquet(rows, &output, a_inner.k())?;
        KmerTable::open(&output)
    })?;
    Ok(PyKmerTable { inner: opened })
}

/// Estimates the number of distinct canonical k-mers across an *entire*
/// FASTQ(.gz) file using HyperLogLog, in a fixed, small amount of memory
/// (`2^precision` bytes) regardless of file size -- unlike `peek()`'s
/// `sample_distinct_kmers`, which is exact but only over a sampled
/// prefix, this covers the whole file at the cost of a full streaming
/// pass (the same I/O `count()` itself pays) rather than a sample's worth.
///
/// Released under `py.allow_threads` like `count()`/`peek()`/`sketch()`'s
/// own I/O-bound work, for the same reason: a large file would otherwise
/// hold the GIL for the whole pass with no way to interrupt it.
#[pyfunction]
#[pyo3(signature = (path, k=31, precision=hll::DEFAULT_PRECISION))]
fn estimate_cardinality(py: Python<'_>, path: String, k: usize, precision: u32) -> PyResult<f64> {
    Ok(py.allow_threads(|| hll::estimate_cardinality(path, k, precision))?)
}

/// Estimates the k-mer frequency spectrum -- how many distinct canonical
/// k-mers occur exactly once, twice, ... -- across one or more FASTQ(.gz)/
/// FASTA(.gz) files, in one streaming pass and a fixed, small amount of
/// memory (`2^precision * 16` bytes) regardless of input size, via an
/// ntCard-style sketch (`ntcard.rs`; `docs/feature-gap-analysis.md`'s
/// S7(a)).
///
/// Returns a plain `{depth: distinct_kmers}` dict, the same shape
/// `KmerCounts.spectrum()` returns from an *exact* count -- so this can be
/// handed straight to `fastdna.genomescope.profile_genome(spectrum, k=k)`
/// wherever an exact spectrum is too expensive to compute first. A
/// module-level function, not a method, matching `estimate_cardinality`'s
/// own precedent: there is no persistent object here for a method to hang
/// off, only a streaming computation over file paths.
///
/// See `ntcard.rs`'s module doc comment for the algorithm and its
/// measured accuracy (f1, the error/noise class, is the hardest to
/// estimate and is characterized separately from every other class).
/// Released under `py.allow_threads` like `estimate_cardinality`/`count`/
/// `peek`'s own I/O-bound work.
#[pyfunction]
#[pyo3(signature = (paths, k=31, precision=hll::DEFAULT_PRECISION, max_frequency=None))]
fn estimate_spectrum(
    py: Python<'_>,
    paths: Vec<String>,
    k: usize,
    precision: u32,
    max_frequency: Option<u32>,
) -> PyResult<PyObject> {
    let estimate =
        py.allow_threads(|| ntcard::estimate_spectrum(&paths, k, precision, max_frequency))?;

    let dict = PyDict::new_bound(py);
    for (depth, count) in estimate.spectrum {
        // Rounded to the nearest non-negative integer, matching
        // `KmerCounts.spectrum()`'s all-integer contract -- see
        // `ntcard::write_spectrum`'s own doc comment for the same
        // rounding rule applied to the CLI's file output.
        dict.set_item(depth, count.round().max(0.0) as u64)?;
    }
    Ok(dict.into())
}

/// The schema of the frame-translation table `translate_sequences()` and
/// `translate_file()` hand back.
///
/// **Column contract** (stated here for the same reason
/// `export::counts_schema` states its own: everything downstream reads
/// these names, so they are an API):
///
/// - `sequence_id`: the record's identifier. For `translate_file` this is
///   the FASTA/FASTQ header with its `>`/`@` marker stripped and truncated
///   at the first whitespace -- the accession, the way BLAST and SAM define
///   it, not the whole description line. For `translate_sequences` it is
///   whatever id the caller supplied.
/// - `frame`: `+1`, `+2`, `+3`, `-1`, `-2`, `-3`. Signed, so `Int8` rather
///   than the `UInt8` the other integer columns in this crate use; negative
///   means the reverse-complement strand.
/// - `protein`: the translated amino-acid sequence, `*` for a stop and `X`
///   for an untranslatable codon (see `translate::AMBIGUOUS_AA`). May be
///   empty -- a sequence shorter than one codon in that frame translates to
///   nothing, and that is a row with an empty string, not a missing row,
///   so a caller can always find every (sequence, frame) pair they asked
///   for.
///
/// Rows are emitted sequence-major: every requested frame of the first
/// sequence, then every frame of the second, and so on. Unlike
/// `counts_schema`, this schema lives here rather than in `export.rs`
/// because nothing writes it to Parquet -- it exists only as an in-memory
/// Arrow table crossing this boundary, so putting it in `export.rs` would
/// imply a file format that does not exist.
fn proteins_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("sequence_id", DataType::Utf8, false),
        Field::new("frame", DataType::Int8, false),
        Field::new("protein", DataType::Utf8, false),
    ]))
}

/// The schema of the amino-acid k-mer table `protein_kmers()` hands back.
///
/// **Column contract:**
///
/// - `sequence_id`: which protein the k-mer came from. Counts are *per
///   protein*, not pooled across the input -- pooling is a one-line
///   group-by on the caller's side, whereas un-pooling is impossible once
///   done here.
/// - `aa_kmer`: the k-mer as literal amino-acid letters. **Not canonical**
///   -- see `translate::amino_acid_kmers` for why proteins cannot be
///   canonicalized the way DNA k-mers are. `MA` and `AM` are distinct rows.
/// - `count`: occurrences of that k-mer within that protein.
///
/// Rows are protein-major and, within a protein, sorted by `aa_kmer`, so
/// the table is byte-identical across runs on the same input.
fn aa_kmers_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("sequence_id", DataType::Utf8, false),
        Field::new("aa_kmer", DataType::Utf8, false),
        Field::new("count", DataType::UInt32, false),
    ]))
}

/// Wraps an Arrow `RecordBatch` construction failure the same way
/// `build_record_batch` above does: a schema/length mismatch is an `Export`
/// failure against a synthetic path, not an I/O error.
fn in_memory_batch(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<RecordBatch, FastDnaError> {
    RecordBatch::try_new(schema, columns).map_err(|e| FastDnaError::Export {
        path: PathBuf::from("<in-memory Arrow table>"),
        reason: e.to_string(),
        source: Some(Box::new(e)),
    })
}

/// Resolves the caller's `table`/`frames`/`to_stop` arguments into the
/// core's own types once, up front, so an invalid id or frame is a
/// `ValueError` before any sequence is read rather than partway through a
/// large file.
fn resolve_translation_args(
    frames: &[i8],
    table: u8,
    to_stop: bool,
) -> Result<(&'static TranslationTable, Vec<Frame>, StopHandling), FastDnaError> {
    let table = TranslationTable::from_id(table)?;
    let resolved: Vec<Frame> =
        frames.iter().map(|&f| Frame::from_i8(f)).collect::<Result<Vec<_>, _>>()?;
    if resolved.is_empty() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "frames",
            reason: "at least one reading frame must be requested".to_string(),
        });
    }
    let stop_handling =
        if to_stop { StopHandling::StopAtFirst } else { StopHandling::Translate };
    Ok((table, resolved, stop_handling))
}

/// Builds a `proteins_schema` batch one translated frame at a time.
///
/// Shared by `translate_sequences` and `translate_file` so the two cannot
/// drift into producing differently-shaped tables --
/// `python/tests/test_translate.py` asserts they agree, and this is what
/// makes that hold structurally rather than by coincidence.
///
/// **What this replaced, and what it costs instead.** The previous form
/// collected `Vec<(String, i8, String)>` and then walked it to fill the
/// Arrow builders. Per row that was two heap allocations and two frees --
/// the sequence id, cloned once for *each* requested frame, and the protein
/// `String` -- plus a 56-byte tuple moved into a vector that existed only to
/// be walked once. Rows now go straight into the builders, so per row: no
/// allocation, and the id and protein bytes are copied once (they were
/// copied once anyway, out of the temporaries). For a file of N records in
/// six frames that is `12 * N` allocations and `12 * N` frees removed.
///
/// Arrow's `StringBuilder` is already the "one contiguous value buffer plus
/// offsets" representation, so nothing needs to change there. What does
/// change is who grows: the intermediate row vector used to grow by
/// doubling and the builders were then sized exactly, whereas the builders
/// now do the growing. That is the one part of this that is a swap rather
/// than a removal, and it is a favourable one -- growth copies plain bytes,
/// where the row vector's growth copied 56-byte tuples *and* every row cost
/// two `malloc`/`free` pairs on top.
struct ProteinsBatchBuilder {
    ids: StringBuilder,
    frames: Int8Builder,
    proteins: StringBuilder,
    /// Reused by every `push`: `translate_into` clears and refills it, so
    /// translating N sequences in F frames allocates this buffer once
    /// instead of `N * F` times.
    scratch: Vec<u8>,
}

impl ProteinsBatchBuilder {
    /// For a caller that knows its row count up front (`translate_sequences`
    /// does: ids times frames).
    fn with_capacity(rows: usize) -> Self {
        ProteinsBatchBuilder {
            ids: StringBuilder::with_capacity(rows, rows * 16),
            frames: Int8Builder::with_capacity(rows),
            proteins: StringBuilder::with_capacity(rows, rows * 64),
            scratch: Vec::new(),
        }
    }

    /// For a caller streaming an input of unknown length (`translate_file`).
    fn new() -> Self {
        ProteinsBatchBuilder {
            ids: StringBuilder::new(),
            frames: Int8Builder::new(),
            proteins: StringBuilder::new(),
            scratch: Vec::new(),
        }
    }

    /// Translates one (sequence, frame) pair and appends it as a row.
    fn push(
        &mut self,
        id: &str,
        sequence: &[u8],
        frame: Frame,
        table: &TranslationTable,
        stop_handling: StopHandling,
    ) {
        translate::translate_into(sequence, frame, table, stop_handling, &mut self.scratch);
        self.ids.append_value(id);
        self.frames.append_value(frame.as_i8());
        // Every byte in `scratch` came from an NCBI `AAs` row or from
        // `translate::AMBIGUOUS_AA`, so this always succeeds. It is the same
        // check `translate` runs internally via `String::from_utf8`, moved
        // here rather than added: what is saved is the `String` that used to
        // carry the result the two feet from there to `append_value`.
        self.proteins.append_value(std::str::from_utf8(&self.scratch).unwrap_or_default());
    }

    fn finish(mut self) -> Result<RecordBatch, FastDnaError> {
        in_memory_batch(
            proteins_schema(),
            vec![
                Arc::new(self.ids.finish()) as ArrayRef,
                Arc::new(self.frames.finish()) as ArrayRef,
                Arc::new(self.proteins.finish()) as ArrayRef,
            ],
        )
    }
}

/// The identifier part of a FASTA/FASTQ header: the marker byte (`>` or
/// `@`) dropped, then everything up to the first whitespace.
///
/// A header line is `>accession free text description`, and the accession
/// is the part every other tool keys on. Keeping the whole line would make
/// `sequence_id` unjoinable against anything else the user has, and would
/// put arbitrary text into a column callers group by.
///
/// Borrows from `header` whenever it is valid UTF-8, which every real
/// header is; only genuinely malformed bytes take the owned branch that
/// `from_utf8_lossy` allocates for its replacement characters. Returning
/// `String` meant `into_owned()` on an otherwise-`Borrowed` `Cow`: one
/// allocation, one copy and one free per record, for bytes that were about
/// to be copied into an Arrow buffer anyway.
fn header_to_sequence_id(header: &[u8]) -> Cow<'_, str> {
    let without_marker = match header.first() {
        Some(b'>') | Some(b'@') => &header[1..],
        _ => header,
    };
    let end = without_marker
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(without_marker.len());
    String::from_utf8_lossy(&without_marker[..end])
}

/// Translates in-memory sequences in the requested reading frames,
/// returning a `pyarrow.RecordBatch` under `proteins_schema` (see there for
/// the column contract).
///
/// `ids` and `sequences` are parallel lists; a length mismatch is a
/// `ValueError` rather than a `zip()` that silently truncates to the
/// shorter one and mislabels every protein after the first divergence.
///
/// Released under `py.allow_threads` like the other bulk work in this
/// module: translating a few million bases holds no Python state and would
/// otherwise freeze the calling interpreter for its duration.
#[pyfunction]
#[pyo3(signature = (ids, sequences, frames, table=1, to_stop=false))]
fn translate_sequences(
    py: Python<'_>,
    ids: Vec<String>,
    sequences: Vec<String>,
    frames: Vec<i8>,
    table: u8,
    to_stop: bool,
) -> PyResult<PyObject> {
    if ids.len() != sequences.len() {
        return Err(PyValueError::new_err(format!(
            "ids and sequences must have the same length -- got {} ids but {} sequences. \
             Zipping them short would attach the wrong id to every protein after the \
             mismatch, so this is refused rather than truncated.",
            ids.len(),
            sequences.len()
        )));
    }

    let batch = py.allow_threads(move || {
        let (table, frames, stop_handling) = resolve_translation_args(&frames, table, to_stop)?;
        let mut builder = ProteinsBatchBuilder::with_capacity(ids.len() * frames.len());
        // `iter()`, not `into_iter()`: the id is appended straight into the
        // Arrow buffer, so it no longer has to be cloned once per requested
        // frame -- six sequences' worth of `String` allocation per sequence
        // in the common six-frame call.
        for (id, sequence) in ids.iter().zip(sequences.iter()) {
            for frame in &frames {
                builder.push(id, sequence.as_bytes(), *frame, table, stop_handling);
            }
        }
        builder.finish()
    })?;

    batch.to_pyarrow(py)
}

/// Streams a FASTA/FASTQ(.gz) file through the same reader the counting
/// pipeline uses and translates every record in the requested frames,
/// returning a `pyarrow.RecordBatch` under `proteins_schema`.
///
/// Reuses `open_fastq_reader` (and therefore `FastqReader`'s own
/// content-based FASTA/FASTQ sniffing and gzip handling) rather than
/// growing a second file-reading path: a file that `count()` can read must
/// be a file this can read, and the only way to guarantee that is for both
/// to go through the same reader.
///
/// Unlike `count()`, the whole result is materialized in memory -- a
/// protein is a third the length of its DNA, but six frames of it is twice
/// the input size, so this is for genes, contigs and modest read sets
/// rather than for a whole sequencing run.
#[pyfunction]
#[pyo3(signature = (path, frames, table=1, to_stop=false))]
fn translate_file(
    py: Python<'_>,
    path: String,
    frames: Vec<i8>,
    table: u8,
    to_stop: bool,
) -> PyResult<PyObject> {
    let path_buf = PathBuf::from(path);

    let batch = py.allow_threads(move || {
        let (table, frames, stop_handling) = resolve_translation_args(&frames, table, to_stop)?;
        let mut reader = open_fastq_reader(&path_buf)?;

        let mut builder = ProteinsBatchBuilder::new();
        let mut record_number: u64 = 0;
        loop {
            let record = reader.next_record().map_err(|e| FastDnaError::MalformedFastq {
                path: path_buf.clone(),
                record: record_number + 1,
                reason: e.to_string(),
            })?;
            let Some(record) = record else { break };
            record_number += 1;

            let id = header_to_sequence_id(&record.id);
            for frame in &frames {
                builder.push(id.as_ref(), &record.seq, *frame, table, stop_handling);
            }
        }
        builder.finish()
    })?;

    batch.to_pyarrow(py)
}

/// Counts amino-acid k-mers in each of `proteins`, returning a
/// `pyarrow.RecordBatch` under `aa_kmers_schema` (see there for the column
/// contract, and `translate::amino_acid_kmers` for why these k-mers are not
/// canonical and do not share `kmer.rs`'s packed representation).
#[pyfunction]
#[pyo3(signature = (ids, proteins, k=3))]
fn protein_kmers(
    py: Python<'_>,
    ids: Vec<String>,
    proteins: Vec<String>,
    k: usize,
) -> PyResult<PyObject> {
    if ids.len() != proteins.len() {
        return Err(PyValueError::new_err(format!(
            "ids and proteins must have the same length -- got {} ids but {} proteins. \
             Zipping them short would attach the wrong id to every k-mer after the \
             mismatch, so this is refused rather than truncated.",
            ids.len(),
            proteins.len()
        )));
    }

    let batch = py.allow_threads(move || {
        // Straight into the builders. The previous form collected
        // `Vec<(String, String, u32)>` first, which cost per row: one
        // `String` for the id (cloned once per k-mer of the protein), one
        // `String` for the k-mer itself, and a 56-byte tuple pushed into a
        // vector that was then walked once to fill these very builders. All
        // three are gone -- `count_amino_acid_kmers_borrowed` hands back
        // k-mers that borrow from the protein, and both strings are copied
        // exactly once, into the Arrow value buffers they were destined for.
        let mut id_builder = StringBuilder::new();
        let mut kmer_builder = StringBuilder::new();
        let mut count_builder = UInt32Builder::new();
        for (id, protein) in ids.iter().zip(proteins.iter()) {
            for (aa_kmer, count) in translate::count_amino_acid_kmers_borrowed(protein, k)? {
                id_builder.append_value(id);
                kmer_builder.append_value(aa_kmer);
                count_builder.append_value(count);
            }
        }

        in_memory_batch(
            aa_kmers_schema(),
            vec![
                Arc::new(id_builder.finish()) as ArrayRef,
                Arc::new(kmer_builder.finish()) as ArrayRef,
                Arc::new(count_builder.finish()) as ArrayRef,
            ],
        )
    })?;

    batch.to_pyarrow(py)
}

/// The Python-visible k-mer -> lowest-common-ancestor database behind
/// `fastdna.metagenomics`. Wraps `metagenomics::KmerDatabase`; every
/// algorithm, format and validation decision lives there, with no PyO3
/// dependency, so all of it stays testable under `cargo test` alone.
#[pyclass(name = "KmerDatabase", module = "fastdna._core")]
struct PyKmerDatabase {
    inner: metagenomics::KmerDatabase,
}

#[pymethods]
impl PyKmerDatabase {
    /// Loads a database written by `save`.
    ///
    /// Released under `py.allow_threads` like every other bulk operation in
    /// this module: a database is hundreds of megabytes of file to read and
    /// validate, and holding the GIL for it would freeze the calling
    /// interpreter with no way to interrupt it.
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<PyKmerDatabase> {
        let inner = py.allow_threads(|| metagenomics::KmerDatabase::load(path))?;
        Ok(PyKmerDatabase { inner })
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.inner.save(path))?;
        Ok(())
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// The number of distinct canonical k-mers in the table.
    #[getter]
    fn n_kmers(&self) -> usize {
        self.inner.len()
    }

    /// Resident bytes. Exposed rather than left to be discovered by OOM:
    /// this representation costs 12 bytes per k-mer and does not scale to
    /// RefSeq, and a user planning a reference set needs to be able to see
    /// the number before committing to one.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Classifies every read of a FASTA/FASTQ(.gz) file, returning a
    /// `pyarrow.RecordBatch` with columns `read_id`, `tax_id`,
    /// `confidence`, `n_kmers`, `n_classified_kmers`.
    #[pyo3(signature = (reads, confidence_threshold=0.0))]
    fn classify(&self, py: Python<'_>, reads: String, confidence_threshold: f64) -> PyResult<PyObject> {
        let batch = py.allow_threads(|| {
            let rows = self.inner.classify_path(&reads, confidence_threshold)?;
            metagenomics::classification_batch(&rows)
        })?;
        batch.to_pyarrow(py)
    }

    /// Aggregates a list of per-read taxon ids into the abundance report.
    ///
    /// Takes the ids rather than the classification table itself: pulling
    /// one column out of an Arrow table is a line of pyarrow on the Python
    /// side, and keeping it there rather than teaching this binding to
    /// consume an Arrow table keeps the FFI surface to what genuinely needs
    /// compiling on five platforms.
    fn abundance(&self, py: Python<'_>, tax_ids: Vec<u32>) -> PyResult<PyObject> {
        let batch = py.allow_threads(|| self.inner.abundance_batch(&tax_ids))?;
        batch.to_pyarrow(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "KmerDatabase(k={}, n_kmers={}, taxa={}, memory_bytes={})",
            self.inner.k(),
            self.inner.len(),
            self.inner.taxonomy().len(),
            self.inner.memory_bytes()
        )
    }
}

/// Builds a k-mer -> lowest-common-ancestor database from a reference
/// FASTA and a taxonomy TSV, optionally saving it to `output`.
///
/// The database is returned whether or not `output` is given, so a caller
/// that wants to build and classify in one session never pays a save and a
/// reload for it.
#[pyfunction]
#[pyo3(signature = (reference, taxonomy, k=31, output=None))]
fn build_database(
    py: Python<'_>,
    reference: String,
    taxonomy: String,
    k: usize,
    output: Option<String>,
) -> PyResult<PyKmerDatabase> {
    let inner = py.allow_threads(|| -> Result<metagenomics::KmerDatabase, FastDnaError> {
        let db = metagenomics::KmerDatabase::build(&reference, &taxonomy, k)?;
        if let Some(output) = &output {
            db.save(output)?;
        }
        Ok(db)
    })?;
    Ok(PyKmerDatabase { inner })
}

/// The schema of `cohort_presence_matrix`'s `"triples"` batch: one row per
/// nonzero `(sample, k-mer)` entry, in COO form -- `scipy.sparse.csr_matrix`
/// accepts `(data, (row, col))` triples in any order, so no particular
/// ordering is promised here.
fn cohort_triples_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("row", DataType::UInt32, false),
        Field::new("col", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]))
}

/// The schema of `cohort_presence_matrix`'s `"kmers"` batch: the decoded
/// sequence of every surviving column, ascending by k-mer value -- see
/// `cohort::matrix`'s module doc comment for why that is already the
/// lexicographic order `gwas.py` promises.
fn cohort_kmers_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("kmer_sequence", DataType::Utf8, false)]))
}

/// Streams every file in `paths` through the same per-sample pipeline
/// `count()` uses, one at a time, and folds the results directly into a
/// cohort-wide k-mer presence/count matrix (`cohort::matrix::
/// build_cohort_matrix`) -- without ever building a per-file Arrow
/// `RecordBatch` the way `python/fastdna/gwas.py::cohort_presence_matrix`
/// used to by calling `count()` once per file. See `cohort::matrix`'s module
/// doc comment for exactly what that saved and why: in short, `count()`'s
/// `.table` decodes *every* distinct k-mer of *every* sample to an ASCII
/// string before this module's cohort-level `min_samples` filter ever runs,
/// and the Python fold that followed it copied that same data a second time
/// via `pyarrow.concat_arrays`. This function decodes a k-mer's sequence
/// exactly once, only for k-mers that end up as a column of the returned
/// matrix.
///
/// Returns a `dict` with:
/// - `"triples"`: a `pyarrow.RecordBatch` under `cohort_triples_schema`
///   (`row`/`col`/`value`), one row per nonzero matrix entry.
/// - `"kmers"`: a `pyarrow.RecordBatch` under `cohort_kmers_schema`
///   (`kmer_sequence`), one row per matrix column, already in the
///   lexicographic order `gwas.py` promises.
/// - `"n_samples"`, `"n_kmers"`: the matrix shape.
/// - `"n_candidates"`: how many k-mers passed `min_samples` before any
///   `max_kmers` truncation -- equal to `n_kmers` unless truncation
///   happened.
/// - `"truncation_cutoff"`: the minor-sample-count of the highest-ranked
///   k-mer `max_kmers` still dropped, or `None` if nothing was truncated.
///   `python/fastdna/gwas.py` uses these last two to reproduce its own
///   truncation `UserWarning` without recomputing anything.
///
/// No progress callback and no fine-grained cancellation, unlike `count()`:
/// this is a new, additive entry point rather than a replacement for
/// `count()`'s own Ctrl-C/progress ergonomics. `py.check_signals()` is
/// polled once between files (not mid-file), which is a real cancellation
/// point, not a decoration -- a large cohort is exactly the shape of call
/// where a user given no way to interrupt it would notice.
#[pyfunction]
#[pyo3(signature = (paths, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None, min_samples=2, max_kmers=None))]
#[allow(clippy::too_many_arguments)]
fn cohort_presence_matrix(
    py: Python<'_>,
    paths: Vec<String>,
    k: usize,
    min_count: u32,
    max_count: Option<u32>,
    min_quality: f64,
    threads: Option<usize>,
    min_samples: u32,
    max_kmers: Option<usize>,
) -> PyResult<PyObject> {
    // Read once: `PipelineConfig::default()` calls
    // `std::thread::available_parallelism` (see `count()`'s own comment on
    // this), so it is computed once here rather than once per file.
    let defaults = PipelineConfig::default();
    let num_threads = threads.unwrap_or(defaults.num_threads);
    let quality_window = defaults.quality_window;
    let batch_size = defaults.batch_size;
    let progress_interval = defaults.progress_interval;

    let mut counters: Vec<KmerCounter> = Vec::with_capacity(paths.len());
    for path in &paths {
        // A real cancellation point: without this, a cohort of hundreds of
        // files gives a user no way to interrupt the run between files. Not
        // mid-file -- see this function's own doc comment for why that is a
        // deliberate, documented simplification rather than an oversight.
        py.check_signals()?;

        let path_buf = PathBuf::from(path);
        let reader = open_fastq_reader(&path_buf)?;
        let config = PipelineConfig {
            k,
            min_quality,
            quality_window,
            batch_size,
            num_threads,
            progress_interval,
            hpc: defaults.hpc,
        };

        let mut counter = py
            .allow_threads(|| {
                process_stream_parallel_with_policy(
                    reader,
                    config,
                    &path_buf,
                    None,
                    None,
                    MemoryPolicy::default(),
                )
            })
            .map(|(counter, _qc, _reads, _decision)| counter)?;

        // Matches `count()`: per-sample frequency filtering happens in RAM,
        // after the pipeline, before this sample's table is used for
        // anything else.
        py.allow_threads(|| counter.prune(min_count, max_count));
        counters.push(counter);
    }

    // The merge itself walks every sample's table (already fully in memory,
    // no further I/O), so it is released the same way the per-file counting
    // above is.
    let built = py.allow_threads(|| cohort::build_cohort_matrix(&counters, min_samples, max_kmers, k));
    drop(counters);

    let triples_batch = in_memory_batch(
        cohort_triples_schema(),
        vec![
            Arc::new(UInt32Array::from(built.row)) as ArrayRef,
            Arc::new(UInt32Array::from(built.col)) as ArrayRef,
            Arc::new(UInt32Array::from(built.value)) as ArrayRef,
        ],
    )?;

    let mut seq_builder = StringBuilder::with_capacity(built.kmer_sequences.len(), built.kmer_sequences.len() * k);
    for sequence in &built.kmer_sequences {
        seq_builder.append_value(sequence);
    }
    let kmers_batch =
        in_memory_batch(cohort_kmers_schema(), vec![Arc::new(seq_builder.finish()) as ArrayRef])?;

    let dict = PyDict::new_bound(py);
    dict.set_item("n_samples", built.n_samples)?;
    dict.set_item("n_kmers", built.n_kmers)?;
    dict.set_item("n_candidates", built.n_candidates)?;
    dict.set_item("truncation_cutoff", built.truncation_cutoff)?;
    dict.set_item("triples", triples_batch.to_pyarrow(py)?)?;
    dict.set_item("kmers", kmers_batch.to_pyarrow(py)?)?;
    Ok(dict.into())
}

/// The schema of `rank_cohort_vocabulary`'s returned `pyarrow.RecordBatch`:
/// one row per selected k-mer, already in ranked (best-first) order -- see
/// `cohort_vocab::rank_vocabulary`'s own doc comment for the ranking rule
/// (`prevalence` desc, `total_freq` desc, `kmer_u64` asc).
fn cohort_vocab_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("kmer_u64", DataType::UInt64, false),
        Field::new("prevalence", DataType::UInt32, false),
        Field::new("total_freq", DataType::UInt32, false),
    ]))
}

/// K-way merges every sample table named in `table_paths` (each a sorted
/// `(kmer_u64, frequency)` Parquet table -- see `ktab.rs`) and ranks the
/// resulting distinct k-mers, returning a `pyarrow.RecordBatch` under
/// `cohort_vocab_schema` (`kmer_u64`/`prevalence`/`total_freq`), already in
/// ranked (best-first) order -- nothing left for the caller to sort. The
/// Python-visible counterpart of `cohort_vocab::rank_vocabulary`; see that
/// function's own doc comment for the streaming merge, the tie-break rule,
/// and the bounded-memory guarantee `top_n` provides.
///
/// `python/fastdna/sklearn.py::KmerVectorizer`'s `disk_backed=True`
/// vocabulary-learning path calls this directly on each training sample's
/// own per-sample Parquet table, instead of `_learn_vocabulary`/`_learn_
/// vocabulary_streaming`'s approach of holding the cohort's rows (fully, or
/// `chunk_size` at a time) in Python-side Arrow arrays. See that module's
/// own docstring for why `disk_backed` exists and how it interacts with
/// `chunk_size`.
///
/// Released under `py.allow_threads` like every other bulk operation in
/// this module (`ktab_union` and friends): merging several real k-mer
/// tables is I/O- and CPU-bound work with nothing Python-specific in it,
/// and holding the GIL for it would freeze the calling interpreter for a
/// cohort-sized run.
#[pyfunction]
#[pyo3(signature = (table_paths, top_n=None))]
fn rank_cohort_vocabulary(py: Python<'_>, table_paths: Vec<String>, top_n: Option<usize>) -> PyResult<PyObject> {
    let paths: Vec<PathBuf> = table_paths.into_iter().map(PathBuf::from).collect();
    let (kmers, prevalence, total_freq) =
        py.allow_threads(|| cohort_vocab::rank_vocabulary(&paths, top_n))?;

    let batch = in_memory_batch(
        cohort_vocab_schema(),
        vec![
            Arc::new(UInt64Array::from(kmers)) as ArrayRef,
            Arc::new(UInt32Array::from(prevalence)) as ArrayRef,
            Arc::new(UInt32Array::from(total_freq)) as ArrayRef,
        ],
    )?;
    batch.to_pyarrow(py)
}

/// The schema of `project_cohort_onto_vocabulary`'s returned
/// `pyarrow.RecordBatch`: one row per nonzero `(sample, vocabulary k-mer)`
/// entry, in COO form -- `scipy.sparse.csr_matrix` accepts `(data, (row,
/// col))` triples in any order, so no particular ordering is promised here
/// (matches `cohort_triples_schema`'s own contract, above).
fn cohort_projection_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("row", DataType::UInt32, false),
        Field::new("col", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]))
}

/// Projects every sample table in `table_paths` onto the fixed
/// `vocabulary` (a list of `kmer_u64` codes, typically `rank_cohort_
/// vocabulary`'s own `kmer_u64` column), returning a `pyarrow.RecordBatch`
/// under `cohort_projection_schema` (`row`/`col`/`value`) -- one COO entry
/// per `(sample, vocabulary k-mer)` pair actually present in that sample's
/// table. The Python-visible counterpart of `cohort_vocab::project_onto_
/// vocabulary`; see that function's own doc comment for the memory bound
/// this exists to provide (one sample's own table plus the vocabulary
/// lookup resident at a time, never the whole cohort at once) and for why
/// `col` matches `vocabulary`'s own given order rather than a re-sorted
/// one.
///
/// Released under `py.allow_threads` for the same reason as `rank_cohort_
/// vocabulary` and every other bulk table operation in this module.
#[pyfunction]
fn project_cohort_onto_vocabulary(
    py: Python<'_>,
    table_paths: Vec<String>,
    vocabulary: Vec<u64>,
) -> PyResult<PyObject> {
    let paths: Vec<PathBuf> = table_paths.into_iter().map(PathBuf::from).collect();
    let (rows, cols, values) =
        py.allow_threads(|| cohort_vocab::project_onto_vocabulary(&paths, &vocabulary))?;

    let batch = in_memory_batch(
        cohort_projection_schema(),
        vec![
            Arc::new(UInt32Array::from(rows)) as ArrayRef,
            Arc::new(UInt32Array::from(cols)) as ArrayRef,
            Arc::new(UInt32Array::from(values)) as ArrayRef,
        ],
    )?;
    batch.to_pyarrow(py)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    register_exception_hierarchy(py, m)?;
    m.add_class::<PyKmerCounts>()?;
    m.add_class::<PyPreview>()?;
    m.add_class::<PySketch>()?;
    m.add_class::<PyFracSketch>()?;
    m.add_class::<PyKmerTable>()?;
    m.add_class::<PyFilterStats>()?;
    m.add_class::<PyPairedFilterStats>()?;
    m.add_class::<PyProfileStats>()?;
    m.add_class::<PyKmerDatabase>()?;
    m.add_function(wrap_pyfunction!(count, m)?)?;
    m.add_function(wrap_pyfunction!(peek, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(sketch, m)?)?;
    m.add_function(wrap_pyfunction!(load_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(frac_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(load_frac_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_cardinality, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(translate_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(translate_file, m)?)?;
    m.add_function(wrap_pyfunction!(protein_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(build_database, m)?)?;
    m.add_function(wrap_pyfunction!(cohort_presence_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rank_cohort_vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(project_cohort_onto_vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(ktab_union, m)?)?;
    m.add_function(wrap_pyfunction!(ktab_intersect, m)?)?;
    m.add_function(wrap_pyfunction!(ktab_diff, m)?)?;
    m.add_function(wrap_pyfunction!(filter_reads, m)?)?;
    m.add_function(wrap_pyfunction!(filter_reads_paired, m)?)?;
    m.add_function(wrap_pyfunction!(profile_reads, m)?)?;
    Ok(())
}
