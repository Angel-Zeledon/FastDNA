// src/ffi.rs
//! PyO3 bindings. The entire FFI surface lives here and is deliberately small:
//! everything that can be expressed in pure Python lives in `python/fastdna/`
//! instead, because each function crossing this boundary must be compiled and
//! tested on five platforms.
//!
//! The module itself is gated in `lib.rs` via `#[cfg(feature = "python")]` on
//! the `pub mod ffi;` declaration, so no inner `#![cfg(...)]` is needed here
//! -- adding one produces a `duplicated_attributes` clippy warning.

// The `#[pyfunction]` macro (pyo3 0.22) generates a wrapper item around any
// function whose body uses `?` to convert a crate error into `PyErr` while
// itself returning `PyResult`; clippy's useless_conversion lint fires on
// that macro-generated code, attributing the warning back to the original
// function's source span. An `#[allow]` on the function itself does not
// reach the separate generated item, so this is scoped to the module
// instead -- narrow enough, since the module exists solely to hold
// `#[pyfunction]`/`#[pymethods]` items that all share this pattern.
#![allow(clippy::useless_conversion)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray, UInt32Array, UInt64Array};
use arrow::pyarrow::PyArrowType;
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
use crate::pipeline::{process_stream_parallel, PipelineConfig};
use crate::qc::QcSummary;

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
            | FastDnaError::NoSamplesFound { .. } => PyValueError::new_err(err.to_string()),
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
fn build_record_batch(counter: &KmerCounter, k: usize) -> Result<RecordBatch, FastDnaError> {
    let schema = export::counts_schema();
    let n = counter.distinct_kmers();
    let mut u64s = Vec::with_capacity(n);
    let mut seqs = Vec::with_capacity(n);
    let mut freqs = Vec::with_capacity(n);

    for (&kmer_bits, &count) in counter.iter() {
        u64s.push(kmer_bits);
        seqs.push(kmer::decode_kmer(kmer_bits, k));
        freqs.push(count);
    }

    let u64_arr: ArrayRef = Arc::new(UInt64Array::from(u64s));
    let seq_arr: ArrayRef = Arc::new(StringArray::from_iter_values(seqs.iter().map(|s| s.as_str())));
    let freq_arr: ArrayRef = Arc::new(UInt32Array::from(freqs));

    RecordBatch::try_new(schema, vec![u64_arr, seq_arr, freq_arr]).map_err(|e| FastDnaError::Export {
        path: PathBuf::from("<in-memory Arrow table>"),
        reason: e.to_string(),
    })
}

/// The Python-visible result of `count()`. Holds the counter and QC summary
/// so `.table`, `.qc`, `.total_kmers` and `.distinct_kmers` can be computed
/// lazily rather than all up front.
#[pyclass(name = "KmerCounts")]
struct PyKmerCounts {
    counter: KmerCounter,
    qc: QcSummary,
    k: usize,
}

#[pymethods]
impl PyKmerCounts {
    /// A zero-copy `pyarrow.RecordBatch` with columns `kmer_u64`,
    /// `kmer_sequence`, `frequency`. `python/fastdna/__init__.py` wraps this
    /// in `pyarrow.Table.from_batches([...])`, itself a cheap wrap rather
    /// than a copy, to present the `pyarrow.Table` the public API promises.
    #[getter]
    fn table(&self, py: Python<'_>) -> PyResult<PyObject> {
        let batch = build_record_batch(&self.counter, self.k)?;
        Ok(PyArrowType(batch).into_py(py))
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
#[pyfunction]
#[pyo3(signature = (path, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None))]
#[allow(clippy::too_many_arguments)]
fn count(
    py: Python<'_>,
    path: String,
    k: usize,
    min_count: u32,
    max_count: Option<u32>,
    min_quality: f64,
    threads: Option<usize>,
) -> PyResult<PyKmerCounts> {
    let path_buf = PathBuf::from(path);
    let reader = open_fastq_reader(&path_buf)?;

    let num_threads = threads.unwrap_or_else(|| PipelineConfig::default().num_threads);
    let config = PipelineConfig {
        k,
        min_quality,
        quality_window: 4,
        batch_size: 10_000,
        num_threads,
        progress_interval: PipelineConfig::default().progress_interval,
    };

    // Released for the duration of the count: without this, a long-running
    // call freezes the calling notebook -- no progress, no Ctrl-C -- for as
    // long as the count takes.
    let outcome: Result<(KmerCounter, QcSummary, u64), FastDnaError> =
        py.allow_threads(|| process_stream_parallel(reader, config, &path_buf, None, None));
    let (mut counter, qc, _total_reads) = outcome?;

    // Matches what `main.rs` does: frequency filters are applied in RAM,
    // after the pipeline and before the result is handed back.
    counter.prune(min_count, max_count);

    Ok(PyKmerCounts { counter, qc, k })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyKmerCounts>()?;
    m.add_function(wrap_pyfunction!(count, m)?)?;
    Ok(())
}
