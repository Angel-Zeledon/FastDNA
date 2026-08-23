// src/pipeline.rs

use std::io::BufRead;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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

/// A worker's result: its private counter and QC state, or the message from
/// a panic that occurred while it was processing (see `catch_unwind` below).
type WorkerOutcome = std::result::Result<(KmerCounter, QcSummary), String>;

/// Extracts a human-readable message from a caught panic payload.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "worker thread panicked".to_string()
    }
}

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

    // With zero consumer tasks, `receiver` is never dropped, the channel never
    // disconnects, and the producer thread blocks forever once the bounded
    // channel fills up -- this must be caught before the channel even exists.
    if config.num_threads == 0 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "num_threads",
            reason: "must be at least 1".to_string(),
        });
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

    // Shared across workers so `ReadsProcessed` is genuinely cumulative.
    // A per-worker counter would report roughly reads/num_threads and jump
    // around non-monotonically. This cannot live in the producer thread
    // instead: `thread::spawn` demands `'static` and `ProgressFn<'a>` is a
    // borrow, so the callback can only be used from the rayon closures, which
    // borrow rather than move.
    let reads_seen = AtomicU64::new(0);

    // 2. Parallel consumer pool. Each worker owns private state, so the hot
    //    path has no locks and no shared hashmap.
    //
    //    The worker body is wrapped in `catch_unwind`: a panic here (e.g.
    //    `extract_canonical_kmers` on a corrupt record, `quality_trim_end`
    //    indexing past a quality line shorter than its sequence, or a
    //    caller-supplied `emit` panicking) must become `Err(Internal)`
    //    rather than unwind out of `process_stream_parallel`. That matters
    //    doubly once a PyO3 binding exists: `emit` will be Python code, and
    //    an unwind across the FFI boundary is undefined behaviour, not a
    //    catchable error.
    //
    //    If a worker panics, its `catch_unwind` still returns (the panic
    //    doesn't propagate), so it stops pulling from `receiver` only
    //    because its own loop already exited. To guarantee the producer
    //    never blocks on a full channel with no one left draining it -- the
    //    degenerate case is `num_threads == 1` and the sole worker panics --
    //    the panicked worker drains and discards whatever remains until the
    //    channel disconnects (which happens once the producer thread
    //    returns and drops its `Sender`). This keeps throughput flowing for
    //    any surviving workers and guarantees the function returns instead
    //    of hanging, regardless of how many workers panicked or when.
    let results: Vec<WorkerOutcome> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let mut local_counter = KmerCounter::with_capacity(131_072);
                let mut local_qc = QcSummary::default();

                while let Ok(mut batch) = receiver.recv() {
                    // Computed before the record loop below consumes the
                    // batch, per the batch-granularity progress accounting.
                    let n = batch.len() as u64;

                    for record in &mut batch {
                        local_qc.observe_record(record);
                        record.quality_trim_end(min_qual, qual_win);

                        // Canonical k-mers: 2-bit packed, O(1) rolling window, and
                        // ambiguous bases ('N') reset the window rather than
                        // producing corrupt k-mers.
                        let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
                        local_counter.insert_batch(&canon_kmers);
                    }

                    // Accounted once per batch, not once per record: at
                    // production batch sizes (~10k) against a 100k
                    // PROGRESS_INTERVAL, per-record fetch_add would hammer a
                    // shared cache line on every worker for no observable
                    // benefit. Firing on interval *crossing* (rather than
                    // exact modulo) is required because a batched counter
                    // will rarely land exactly on a multiple.
                    if let Some(emit) = progress {
                        let prev = reads_seen.fetch_add(n, Ordering::Relaxed);
                        if prev / PROGRESS_INTERVAL != (prev + n) / PROGRESS_INTERVAL {
                            emit(Progress::ReadsProcessed(prev + n));
                        }
                    }
                }

                (local_counter, local_qc)
            }));

            match outcome {
                Ok(v) => Ok(v),
                Err(payload) => {
                    // Drain (and discard) whatever is left so the bounded
                    // channel never backs up and blocks the producer, even
                    // if this was the only worker still consuming.
                    while receiver.recv().is_ok() {}
                    Err(panic_message(payload))
                }
            }
        })
        .collect();

    let total_reads = reader_handle
        .join()
        .map_err(|_| FastDnaError::Internal { detail: "FASTQ reader thread panicked".to_string() })??;

    let mut worker_panic: Option<String> = None;
    let mut outcomes: Vec<(KmerCounter, QcSummary)> = Vec::with_capacity(results.len());
    for outcome in results {
        match outcome {
            Ok(v) => outcomes.push(v),
            Err(detail) => {
                if worker_panic.is_none() {
                    worker_panic = Some(detail);
                }
            }
        }
    }
    if let Some(detail) = worker_panic {
        return Err(FastDnaError::Internal { detail });
    }

    // 3. Map-reduce combine phase.
    let (master_counter, mut master_qc) = outcomes.into_par_iter().reduce(
        || (KmerCounter::new(), QcSummary::default()),
        |(mut acc_cnt, mut acc_qc), (local_cnt, local_qc)| {
            acc_cnt.merge(local_cnt);
            acc_qc.merge(&local_qc);
            (acc_cnt, acc_qc)
        },
    );

    master_qc.finalize();

    if let Some(emit) = progress {
        // Also guarded: this runs on the main thread, outside any worker's
        // catch_unwind, so an unwind here needs its own net.
        if catch_unwind(AssertUnwindSafe(|| emit(Progress::Finished { reads: total_reads }))).is_err() {
            return Err(FastDnaError::Internal { detail: "progress callback panicked".to_string() });
        }
    }

    Ok((master_counter, master_qc, total_reads))
}
