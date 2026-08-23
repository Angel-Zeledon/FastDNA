// src/pipeline.rs

use std::io::BufRead;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqReader, FastqRecord};
use crate::kmer;
use crate::progress::{Progress, ProgressFn, PROGRESS_INTERVAL};
use crate::qc::QcSummary;

pub struct PipelineConfig {
    pub k: usize,
    pub min_quality: f64,
    pub quality_window: usize,
    pub batch_size: usize,
    pub num_threads: usize,
    /// How often `ReadsProcessed` is emitted, in reads. Defaults to
    /// `PROGRESS_INTERVAL`. Small samples (or tests) should lower this --
    /// otherwise a run of fewer than the interval never emits a single
    /// `ReadsProcessed` event, only `Finished` at the end.
    pub progress_interval: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            k: 31,
            min_quality: 20.0,
            quality_window: 4,
            batch_size: 8192,
            num_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            progress_interval: PROGRESS_INTERVAL,
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
///
/// `cancel`, if given, lets a caller interrupt a long-running call: setting
/// the flag causes this function to return `Err(FastDnaError::Cancelled)`
/// rather than partial counts. It is `Arc`, not a borrow like `ProgressFn`,
/// because the producer thread -- which is `'static` (see below) -- must be
/// able to see it too: if only the worker pool checked it, cancelling would
/// leave the producer blocked forever on a full channel with no one left to
/// drain it, the same deadlock item A guards against for `num_threads == 0`.
pub fn process_stream_parallel<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
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

    // `prev / progress_interval` below divides by this value on every batch
    // a progress callback is supplied; zero panics unconditionally the first
    // time that division runs. Rejected here, at the same entry-point
    // validation point as `num_threads`, rather than only when `progress`
    // is `Some`, so a caller cannot leave a latent panic in a config value
    // that merely isn't exercised by today's call but might be by tomorrow's.
    if config.progress_interval == 0 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "progress_interval",
            reason: "must be at least 1".to_string(),
        });
    }

    // `quality_trim_end` bounds its trim window by `window_size` but never
    // validates it: with `window_size == 0` the loop condition
    // `end_pos >= window_size` is always true, the empty window makes
    // `sum / 0` produce `NaN` (so the `>= min_qual` break never fires), and
    // `end_pos -= 1` underflows once `end_pos` reaches zero. Rejected here
    // rather than inside `quality_trim_end` itself, which runs once per read
    // and must stay branch-free for this.
    if config.quality_window == 0 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "quality_window",
            reason: "must be at least 1".to_string(),
        });
    }

    // The producer loop only ships a batch once `current_batch.len() >=
    // batch_size`; with `batch_size == 0` that is trivially true after every
    // single record, so each read crosses the `bounded(64)` channel in its
    // own one-element `Vec`. That is not a hang or a wrong answer, but a
    // severe throughput cliff plus per-record allocation churn from a
    // plausible caller mistake (e.g. an off-by-one default), so it is
    // rejected here rather than left to silently degrade every run.
    if config.batch_size == 0 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "batch_size",
            reason: "must be at least 1".to_string(),
        });
    }

    // `quality_trim_end` decides where to stop trimming with
    // `avg_qual >= min_qual`. Every comparison against `NaN` is `false`, so a
    // `NaN` `min_quality` never breaks the loop early and silently trims
    // every read down to `window_size - 1` bases -- the run still completes
    // and returns near-zero counts, with no panic and no error, which is
    // exactly the silent-wrong-answer class this validation block exists to
    // rule out. `+-inf` are rejected too: `+inf` behaves like `NaN` here
    // (every average is `< +inf`, so nothing ever breaks the loop), while
    // `-inf` makes every comparison `true` and trims nothing at all --
    // a different wrong answer, not a safe one.
    //
    // This guard and the `quality_window` guard above protect each other and
    // neither should be removed on its own: this NaN/`+-inf` loop only
    // terminates safely today *because* `quality_window >= 1` is guaranteed.
    // With `window_size == 0` the same non-finite comparison instead loops
    // forever (see the `quality_window` guard's comment for why). Do not
    // drop either guard as "defensive noise" without re-checking the other.
    if !config.min_quality.is_finite() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "min_quality",
            reason: "must be a finite number (not NaN or infinite)".to_string(),
        });
    }

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(64);

    let batch_size = config.batch_size;
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let progress_interval = config.progress_interval;
    let source_owned: PathBuf = source.to_path_buf();
    // Cloned before the move below: the reader thread needs its own 'static
    // owned handle to the flag, distinct from the one the rayon workers
    // check (see the function doc comment for why this must be Arc).
    let cancel_for_reader = cancel.clone();

    // 1. Producer thread. Returns the read count, or the record it choked on.
    let reader_handle = thread::spawn(move || -> Result<u64> {
        let mut reader = reader;
        let mut current_batch = Vec::with_capacity(batch_size);
        let mut total_reads: u64 = 0;
        let mut cancelled = false;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    current_batch.push(record);
                    total_reads += 1;

                    if current_batch.len() >= batch_size {
                        // Checked once per batch dispatch, not per record: an
                        // atomic load in the per-record hot path would cost
                        // measurable throughput for no benefit -- a human
                        // pressing Ctrl-C does not need sub-batch latency.
                        if let Some(tok) = &cancel_for_reader {
                            if tok.load(Ordering::Relaxed) {
                                cancelled = true;
                                break;
                            }
                        }

                        let batch_to_send =
                            std::mem::replace(&mut current_batch, Vec::with_capacity(batch_size));
                        if sender.send(batch_to_send).is_err() {
                            break;
                        }
                    }
                }
                Ok(None) => break,
                // A genuine I/O failure (a corrupt gzip member, an NFS read
                // error) is not a data problem, so it must not be reported
                // as MalformedFastq -- that would blame the bytes for a
                // hardware/transport fault and attach a record number that
                // means nothing for it.
                Err(FastqReadError::Io(source_err)) => {
                    return Err(FastDnaError::Io { path: source_owned, source: source_err });
                }
                // A structural violation in the bytes themselves: attach the
                // path and the 1-based index of the record that failed.
                Err(FastqReadError::Malformed(reason)) => {
                    return Err(FastDnaError::MalformedFastq {
                        path: source_owned,
                        record: total_reads + 1,
                        reason,
                    });
                }
            }
        }

        // On cancellation, drop the leftover batch instead of sending it:
        // the whole result is discarded by the Cancelled check below, so
        // there is no point handing workers more to chew through. `sender`
        // is dropped when this closure returns either way, which disconnects
        // the channel and lets any worker still consuming exit its loop.
        if !cancelled && !current_batch.is_empty() {
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
    //
    //    Cancellation is checked once per batch received (not per record,
    //    same reasoning as the producer side). The same drain-and-discard
    //    applies here as in the panic case, and for the same reason: the
    //    producer checks its own copy of the flag only once per batch
    //    dispatch, so it may already be blocked trying to send into a full
    //    channel (or about to be) by the time a worker notices cancellation.
    //    If every worker simply stopped consuming, that send -- and the
    //    producer thread it lives on -- would block forever, the same
    //    deadlock item A guards against. Draining keeps the channel moving
    //    until the producer itself notices the flag and drops `Sender`.
    let results: Vec<WorkerOutcome> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let mut local_counter = KmerCounter::with_capacity(131_072);
                let mut local_qc = QcSummary::default();

                while let Ok(mut batch) = receiver.recv() {
                    if let Some(tok) = &cancel {
                        if tok.load(Ordering::Relaxed) {
                            break;
                        }
                    }

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
                    // production batch sizes (~10k) against a 100k default
                    // progress_interval, per-record fetch_add would hammer a
                    // shared cache line on every worker for no observable
                    // benefit. Firing on interval *crossing* (rather than
                    // exact modulo) is required because a batched counter
                    // will rarely land exactly on a multiple.
                    if let Some(emit) = progress {
                        let prev = reads_seen.fetch_add(n, Ordering::Relaxed);
                        if prev / progress_interval != (prev + n) / progress_interval {
                            emit(Progress::ReadsProcessed(prev + n));
                        }
                    }
                }

                (local_counter, local_qc)
            }));

            match outcome {
                Ok(v) => {
                    // If cancelled, this worker's loop above may have broken
                    // out with batches still in flight; drain them so the
                    // producer (or another worker mid-send) never blocks.
                    // Cheap when there is nothing to do: if the run was not
                    // cancelled the flag check below is false and this is
                    // skipped entirely; if it was cancelled but this worker's
                    // loop instead ended because the channel had already
                    // disconnected, the drain call returns immediately.
                    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
                        while receiver.recv().is_ok() {}
                    }
                    Ok(v)
                }
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
    // A worker panic is an actual bug and takes priority over reporting
    // cancellation: if one worker panicked while others simply noticed the
    // cancel flag and stopped, the panic is the thing the caller needs to
    // know about, not that the run also happened to be cancelled.
    if let Some(detail) = worker_panic {
        return Err(FastDnaError::Internal { detail });
    }

    // Checked after the join and after the panic check: a cancelled run
    // must not look like a successful one with fewer reads, so this takes
    // priority over returning partial counts.
    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
        return Err(FastDnaError::Cancelled);
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
