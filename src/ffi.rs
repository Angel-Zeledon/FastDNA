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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use arrow::array::{ArrayRef, Int8Builder, StringBuilder, UInt32Builder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::ToPyArrow;
use arrow::record_batch::RecordBatch;
use flate2::read::MultiGzDecoder;
use pyo3::exceptions::{
    PyFileNotFoundError, PyKeyboardInterrupt, PyMemoryError, PyOSError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::counter::KmerCounter;
use crate::error::FastDnaError;
use crate::export;
use crate::fastq::FastqReader;
use crate::kmer;
use crate::pipeline::{process_stream_parallel_with_policy, MemoryPolicy, PipelineConfig};
use crate::preview;
use crate::progress::Progress;
use crate::qc::QcSummary;
use crate::sketch::GenomeSketch;
use crate::translate::{self, Frame, StopHandling, TranslationTable};
use crate::hll;

/// The single place the spec's error-to-exception table (design doc §12) is
/// implemented. No call site maps a `FastDnaError` to a `PyErr` directly, so
/// no call site can diverge from this table.
impl From<FastDnaError> for PyErr {
    fn from(err: FastDnaError) -> PyErr {
        match &err {
            // `NotFound` gets the more specific exception; every other kind
            // of I/O failure (permission denied, a bad gzip stream, ...) is
            // a generic OSError.
            FastDnaError::Io { source, .. } => {
                if source.kind() == std::io::ErrorKind::NotFound {
                    PyFileNotFoundError::new_err(err.to_string())
                } else {
                    PyOSError::new_err(err.to_string())
                }
            }
            FastDnaError::MalformedFastq { .. }
            | FastDnaError::InvalidK { .. }
            | FastDnaError::MismatchedK { .. }
            | FastDnaError::InvalidConfig { .. }
            // Not in the spec's table (which predates the cohort engine),
            // but a caller-supplied bad directory is the same kind of
            // mistake as InvalidConfig, so it gets the same treatment
            // rather than being left an unclassified RuntimeError.
            | FastDnaError::NoSamplesFound { .. }
            // Also not in the spec's table (predates GenomeSketch
            // persistence). A corrupt/foreign sketch file is bad input
            // data, the same kind of mistake as a malformed FASTQ record,
            // not an internal failure -- ValueError, not RuntimeError.
            | FastDnaError::Load { .. } => PyValueError::new_err(err.to_string()),
            FastDnaError::MatrixTooLarge { .. } | FastDnaError::VocabTooLarge { .. } => {
                PyMemoryError::new_err(err.to_string())
            }
            FastDnaError::Export { .. } => PyRuntimeError::new_err(err.to_string()),
            FastDnaError::Cancelled => PyKeyboardInterrupt::new_err(err.to_string()),
            FastDnaError::Internal { .. } => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Builds the single in-memory Arrow `RecordBatch` backing `KmerCounts.table`,
/// using the exact schema `export.rs` writes to Parquet with (§9.2 of the
/// design doc: the in-memory table and the Parquet files must have identical
/// columns).
///
/// Builds directly into Arrow's own builders rather than collecting into
/// intermediate `Vec<u64>`/`Vec<String>`/`Vec<u32>` first: at a realistic
/// count of distinct k-mers, retaining a second full copy of every decoded
/// k-mer string just to hand it to `StringArray::from_iter_values` a moment
/// later is measurable transient memory for no benefit.
fn build_record_batch(counter: &KmerCounter, k: usize) -> Result<RecordBatch, FastDnaError> {
    let schema = export::counts_schema();
    let n = counter.distinct_kmers();

    let mut u64_builder = UInt64Builder::with_capacity(n);
    // `k + 1` is a rough per-string byte estimate (the alphabet is ASCII,
    // so bytes == characters); a data-capacity hint that undershoots costs
    // reallocations, not correctness, so it does not need to be exact.
    let mut seq_builder = StringBuilder::with_capacity(n, n * (k + 1));
    let mut freq_builder = UInt32Builder::with_capacity(n);

    for (kmer_bits, count) in counter.iter() {
        u64_builder.append_value(kmer_bits);
        seq_builder.append_value(kmer::decode_kmer(kmer_bits, k));
        freq_builder.append_value(count);
    }

    let u64_arr: ArrayRef = Arc::new(u64_builder.finish());
    let seq_arr: ArrayRef = Arc::new(seq_builder.finish());
    let freq_arr: ArrayRef = Arc::new(freq_builder.finish());

    RecordBatch::try_new(schema, vec![u64_arr, seq_arr, freq_arr]).map_err(|e| FastDnaError::Export {
        path: PathBuf::from("<in-memory Arrow table>"),
        reason: e.to_string(),
    })
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
                let built = py.allow_threads(|| build_record_batch(&self.counter, self.k))?;
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
#[pyo3(signature = (path, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None, progress=None, progress_interval=None))]
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
) -> PyResult<PyKmerCounts> {
    let path_buf = PathBuf::from(path);
    let reader = open_fastq_reader(&path_buf)?;

    let num_threads = threads.unwrap_or_else(|| PipelineConfig::default().num_threads);
    let progress_interval = progress_interval.unwrap_or_else(|| PipelineConfig::default().progress_interval);
    // Batches are the unit progress is accounted in (pipeline.rs emits at
    // most once per batch received, on interval crossing), so a batch far
    // larger than `progress_interval` would silently coarsen progress no
    // matter how small the caller asks for it. Capped at the core's own
    // default batch size so the common case (no explicit interval) is
    // unaffected.
    let batch_size = (progress_interval as usize).clamp(1, PipelineConfig::default().batch_size);
    let config = PipelineConfig { k, min_quality, num_threads, batch_size, progress_interval, ..PipelineConfig::default() };

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
    let (outcome, signal_error): (Result<(KmerCounter, QcSummary, u64), FastDnaError>, Option<PyErr>) =
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

    Ok(PyKmerCounts { counter, qc, k, table_cache: OnceLock::new() })
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

/// Collects `(sequence_id, frame, protein)` rows into an Arrow batch under
/// `proteins_schema`. Shared by `translate_sequences` and `translate_file`
/// so the two cannot drift into producing differently-shaped tables --
/// `python/tests/test_translate.py` asserts they agree, and this is what
/// makes that hold structurally rather than by coincidence.
fn build_proteins_batch(rows: Vec<(String, i8, String)>) -> Result<RecordBatch, FastDnaError> {
    let n = rows.len();
    let mut id_builder = StringBuilder::with_capacity(n, n * 16);
    let mut frame_builder = Int8Builder::with_capacity(n);
    let mut protein_builder = StringBuilder::with_capacity(n, n * 64);

    for (id, frame, protein) in rows {
        id_builder.append_value(id);
        frame_builder.append_value(frame);
        protein_builder.append_value(protein);
    }

    in_memory_batch(
        proteins_schema(),
        vec![
            Arc::new(id_builder.finish()) as ArrayRef,
            Arc::new(frame_builder.finish()) as ArrayRef,
            Arc::new(protein_builder.finish()) as ArrayRef,
        ],
    )
}

/// The identifier part of a FASTA/FASTQ header: the marker byte (`>` or
/// `@`) dropped, then everything up to the first whitespace.
///
/// A header line is `>accession free text description`, and the accession
/// is the part every other tool keys on. Keeping the whole line would make
/// `sequence_id` unjoinable against anything else the user has, and would
/// put arbitrary text into a column callers group by.
fn header_to_sequence_id(header: &[u8]) -> String {
    let without_marker = match header.first() {
        Some(b'>') | Some(b'@') => &header[1..],
        _ => header,
    };
    let end = without_marker
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(without_marker.len());
    String::from_utf8_lossy(&without_marker[..end]).into_owned()
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
        let mut rows = Vec::with_capacity(ids.len() * frames.len());
        for (id, sequence) in ids.into_iter().zip(sequences.into_iter()) {
            for frame in &frames {
                rows.push((
                    id.clone(),
                    frame.as_i8(),
                    translate::translate(sequence.as_bytes(), *frame, table, stop_handling),
                ));
            }
        }
        build_proteins_batch(rows)
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

        let mut rows = Vec::new();
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
                rows.push((
                    id.clone(),
                    frame.as_i8(),
                    translate::translate(&record.seq, *frame, table, stop_handling),
                ));
            }
        }
        build_proteins_batch(rows)
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
        let mut rows: Vec<(String, String, u32)> = Vec::new();
        for (id, protein) in ids.iter().zip(proteins.iter()) {
            for (aa_kmer, count) in translate::count_amino_acid_kmers(protein, k)? {
                rows.push((id.clone(), aa_kmer, count));
            }
        }

        let n = rows.len();
        let mut id_builder = StringBuilder::with_capacity(n, n * 16);
        let mut kmer_builder = StringBuilder::with_capacity(n, n * (k + 1));
        let mut count_builder = UInt32Builder::with_capacity(n);
        for (id, aa_kmer, count) in rows {
            id_builder.append_value(id);
            kmer_builder.append_value(aa_kmer);
            count_builder.append_value(count);
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyKmerCounts>()?;
    m.add_class::<PyPreview>()?;
    m.add_class::<PySketch>()?;
    m.add_function(wrap_pyfunction!(count, m)?)?;
    m.add_function(wrap_pyfunction!(peek, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(sketch, m)?)?;
    m.add_function(wrap_pyfunction!(load_sketch, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_cardinality, m)?)?;
    m.add_function(wrap_pyfunction!(translate_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(translate_file, m)?)?;
    m.add_function(wrap_pyfunction!(protein_kmers, m)?)?;
    Ok(())
}
