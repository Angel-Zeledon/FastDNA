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
use crate::disk_spill::{self, ScratchDir, SpillWriter};
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqReader, FastqRecord};
use crate::kmer;
use crate::mem_estimate;
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
/// Validates the subset of `PipelineConfig` that both counting strategies
/// share, before either commits to any work. Extracted so
/// `process_stream_parallel` (the in-memory strategy) and
/// `process_stream_parallel_disk` (the disk strategy, see `disk_spill.rs`)
/// reject the same bad config the same way instead of maintaining two
/// copies of these checks that could drift apart. This is a pure
/// extraction of what used to be inline at the top of
/// `process_stream_parallel` -- same checks, same order, same error
/// values -- not a behavior change.
fn validate_config(config: &PipelineConfig) -> Result<()> {
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

    Ok(())
}

pub fn process_stream_parallel<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    validate_config(&config)?;

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

/// Which of FastDNA's two counting strategies produced a result.
///
/// `InMemory` (`process_stream_parallel`) sorts and compacts entirely in
/// RAM and is the faster of the two whenever the input fits; `Disk` (see
/// `disk_spill.rs`) partitions k-mers into buckets, spills each to scratch
/// files, and merges bucket by bucket so only one bucket is resident at
/// once, trading speed for a peak memory footprint that does not scale
/// with `threads * distinct_kmers` the way the in-memory strategy's does
/// (see `mem_estimate.rs` for the measured reason that scaling exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountStrategy {
    InMemory,
    Disk,
}

impl CountStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CountStrategy::InMemory => "in-memory",
            CountStrategy::Disk => "disk",
        }
    }
}

/// Caller-supplied knobs for `process_stream_parallel_with_policy`'s
/// automatic strategy choice. Every field defaults to "figure it out" --
/// `MemoryPolicy::default()` reproduces today's in-memory-only behavior for
/// any caller that does not know or care about the disk strategy.
#[derive(Debug, Clone, Default)]
pub struct MemoryPolicy {
    /// Forces a specific strategy, bypassing the estimate entirely. `None`
    /// lets `resolve_strategy` decide.
    pub strategy: Option<CountStrategy>,
    /// The memory budget the automatic chooser compares its estimate
    /// against. `None` uses `mem_estimate::default_max_ram_bytes()` (half
    /// of currently available system memory, or a fixed fallback -- see
    /// that function's doc comment).
    pub max_ram_bytes: Option<u64>,
    /// Best-effort decompressed input size in bytes, used to derive an
    /// occurrence estimate (see
    /// `mem_estimate::estimate_occurrences_from_bytes`). `None` disables
    /// size-based estimation: the chooser then has no basis to predict a
    /// nonzero peak, so it defaults to `InMemory` rather than guessing.
    /// Callers that know their input size (a real file on disk) should
    /// always supply it; callers that do not (an in-memory buffer, as most
    /// of this crate's own tests use) get today's behavior unchanged.
    pub estimated_input_bytes: Option<u64>,
}

/// What the automatic chooser decided and why, returned alongside the
/// counting result so a caller can report it -- "record which one was
/// used so it is visible rather than mysterious" is a stated requirement,
/// not an afterthought.
#[derive(Debug, Clone, Copy)]
pub struct StrategyDecision {
    pub strategy: CountStrategy,
    pub estimated_occurrences: u64,
    pub estimated_peak_bytes: u64,
    pub budget_bytes: u64,
    /// True when a `FASTDNA_STRATEGY`/`FASTDNA_MAX_RAM_BYTES` environment
    /// variable (see `resolve_strategy`) contributed to this decision,
    /// rather than `policy` and the estimate alone -- surfaced so a report
    /// built from this struct can say so, instead of silently attributing
    /// an environment-driven choice to the estimator.
    pub env_override_applied: bool,
}

/// Decides which counting strategy a run should use, without running
/// anything. Pure and side-effect-free except for reading two environment
/// variables (see below), so a caller can call this once to report the
/// decision (e.g. the CLI, before it prints its banner) and
/// `process_stream_parallel_with_policy` can call it again internally to
/// act on it, with no risk of the two disagreeing.
///
/// Resolution order: an explicit `policy.strategy` wins outright. Failing
/// that, `FASTDNA_STRATEGY` (`disk`, or `memory`/`in-memory`) is checked --
/// an escape hatch for forcing a strategy through callers that have no
/// dedicated API surface for it yet, most notably the Python bindings
/// (`ffi.rs`'s `count()` takes no strategy argument, and adding one is out
/// of scope here; see the CLI's `--strategy` flag for the intended primary
/// interface). Failing that, the estimate decides:
/// `mem_estimate::estimate_peak_bytes` against a budget that is
/// `policy.max_ram_bytes`, then `FASTDNA_MAX_RAM_BYTES` (parsed as a plain
/// byte count) if set, then `mem_estimate::default_max_ram_bytes()`. With
/// no `estimated_input_bytes` at all, the estimate has nothing to work
/// from and this always resolves to `InMemory` -- the conservative choice
/// for library callers who have not told this function enough to justify
/// spilling to disk.
pub fn resolve_strategy(policy: &MemoryPolicy, config: &PipelineConfig) -> StrategyDecision {
    let env_max_ram = std::env::var("FASTDNA_MAX_RAM_BYTES").ok().and_then(|s| s.trim().parse::<u64>().ok());
    let budget_bytes = policy
        .max_ram_bytes
        .or(env_max_ram)
        .unwrap_or_else(mem_estimate::default_max_ram_bytes);

    let estimated_occurrences =
        policy.estimated_input_bytes.map(mem_estimate::estimate_occurrences_from_bytes).unwrap_or(0);
    let estimated_peak_bytes = mem_estimate::estimate_peak_bytes(estimated_occurrences, config.num_threads);

    let env_strategy = std::env::var("FASTDNA_STRATEGY").ok().and_then(|s| match s.trim() {
        "disk" => Some(CountStrategy::Disk),
        "memory" | "in-memory" => Some(CountStrategy::InMemory),
        _ => None,
    });

    let (strategy, env_override_applied) = match policy.strategy {
        Some(s) => (s, false),
        None => match env_strategy {
            Some(s) => (s, true),
            None => {
                let auto = if policy.estimated_input_bytes.is_some() && estimated_peak_bytes > budget_bytes {
                    CountStrategy::Disk
                } else {
                    CountStrategy::InMemory
                };
                // The env var only "applied" if it actually supplied the
                // budget -- when `policy.max_ram_bytes` is set it wins at
                // the `.or(env_max_ram)` above and the env value had no
                // effect on the decision.
                (auto, policy.max_ram_bytes.is_none() && env_max_ram.is_some())
            }
        },
    };

    StrategyDecision { strategy, estimated_occurrences, estimated_peak_bytes, budget_bytes, env_override_applied }
}

/// How a disk-strategy worker failed. A real `FastDnaError` (a spill-file
/// I/O failure, most commonly disk-full on the temp drive) must survive the
/// reduce phase *typed*: stringifying it and rewrapping as
/// `FastDnaError::Internal` -- the old behaviour -- blamed an environmental
/// failure on "a bug in FastDNA" and stripped the path and `source()` the
/// caller needs to diagnose it. Only a genuine panic belongs in `Internal`.
enum DiskWorkerFailure {
    Error(FastDnaError),
    Panic(String),
}

impl DiskWorkerFailure {
    fn into_error(self) -> FastDnaError {
        match self {
            DiskWorkerFailure::Error(err) => err,
            DiskWorkerFailure::Panic(detail) => FastDnaError::Internal { detail },
        }
    }
}

/// The disk strategy's per-worker outcome: a manifest of spilled run files
/// per bucket (see `disk_spill::SpillWriter::finish`) instead of a private
/// `KmerCounter` -- the whole point of this strategy is that no worker
/// builds one of those.
type DiskWorkerOutcome = std::result::Result<(Vec<Vec<PathBuf>>, QcSummary), DiskWorkerFailure>;

/// The disk-partitioned counting strategy. Structurally a sibling of
/// `process_stream_parallel`, not a variant of it: the producer thread and
/// channel setup below are intentionally re-implemented rather than
/// factored out from that function, so this strategy's own bugs cannot
/// reach back into the in-memory path's already-tested behavior, and vice
/// versa. `config` must already be valid
/// (`process_stream_parallel_with_policy` validates it before choosing a
/// strategy).
fn process_stream_parallel_disk<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    let scratch = ScratchDir::new()?;
    let bucket_bits = disk_spill::DEFAULT_BUCKET_BITS;

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(64);

    let batch_size = config.batch_size;
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let progress_interval = config.progress_interval;
    let source_owned: PathBuf = source.to_path_buf();
    let cancel_for_reader = cancel.clone();

    // 1. Producer thread -- same shape and same reasoning as
    // `process_stream_parallel`'s (see that function's inline comments for
    // why each piece is there); duplicated rather than shared, see this
    // function's own doc comment for why.
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
                Err(FastqReadError::Io(source_err)) => {
                    return Err(FastDnaError::Io { path: source_owned, source: source_err });
                }
                Err(FastqReadError::Malformed(reason)) => {
                    return Err(FastDnaError::MalformedFastq {
                        path: source_owned,
                        record: total_reads + 1,
                        reason,
                    });
                }
            }
        }

        if !cancelled && !current_batch.is_empty() {
            let _ = sender.send(current_batch);
        }

        Ok(total_reads)
    });

    let reads_seen = AtomicU64::new(0);
    // Tracks total k-mer occurrences the same way `KmerCounter::insert_batch`
    // does internally (an exact running count of every instance handed to
    // it, independent of any later per-key saturation) -- required for the
    // disk strategy's result to be bit-identical to the in-memory
    // strategy's, including `total_kmers()`, not merely its distinct-kmer
    // table.
    let total_occurrences = AtomicU64::new(0);

    // 2. Parallel consumer pool. Each worker spills bucketed, sorted runs
    // to its own scratch files instead of building a private KmerCounter --
    // see `disk_spill.rs` for why that is what keeps this strategy's peak
    // memory from scaling with `threads * distinct_kmers`.
    let results: Vec<DiskWorkerOutcome> = (0..config.num_threads)
        .into_par_iter()
        .map(|worker_idx| {
            let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<(Vec<Vec<PathBuf>>, QcSummary)> {
                let mut spill = SpillWriter::new(&scratch, worker_idx, k, bucket_bits);
                let mut local_qc = QcSummary::default();

                while let Ok(mut batch) = receiver.recv() {
                    if let Some(tok) = &cancel {
                        if tok.load(Ordering::Relaxed) {
                            break;
                        }
                    }

                    let n = batch.len() as u64;

                    for record in &mut batch {
                        local_qc.observe_record(record);
                        record.quality_trim_end(min_qual, qual_win);

                        let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
                        total_occurrences.fetch_add(canon_kmers.len() as u64, Ordering::Relaxed);
                        spill.insert_batch(&canon_kmers)?;
                    }

                    if let Some(emit) = progress {
                        let prev = reads_seen.fetch_add(n, Ordering::Relaxed);
                        if prev / progress_interval != (prev + n) / progress_interval {
                            emit(Progress::ReadsProcessed(prev + n));
                        }
                    }
                }

                let manifest = spill.finish()?;
                Ok((manifest, local_qc))
            }));

            match outcome {
                Ok(Ok(v)) => {
                    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
                        while receiver.recv().is_ok() {}
                    }
                    Ok(v)
                }
                Ok(Err(fastdna_err)) => {
                    // A spill I/O failure is a real error, not a panic --
                    // still drain so the producer never blocks on a full
                    // channel with this worker no longer consuming.
                    while receiver.recv().is_ok() {}
                    Err(DiskWorkerFailure::Error(fastdna_err))
                }
                Err(payload) => {
                    while receiver.recv().is_ok() {}
                    Err(DiskWorkerFailure::Panic(panic_message(payload)))
                }
            }
        })
        .collect();

    let total_reads = reader_handle
        .join()
        .map_err(|_| FastDnaError::Internal { detail: "FASTQ reader thread panicked".to_string() })??;

    let mut worker_failure: Option<DiskWorkerFailure> = None;
    let mut manifests: Vec<Vec<Vec<PathBuf>>> = Vec::with_capacity(results.len());
    let mut master_qc = QcSummary::default();
    for outcome in results {
        match outcome {
            Ok((manifest, qc)) => {
                manifests.push(manifest);
                master_qc.merge(&qc);
            }
            Err(failure) => {
                if worker_failure.is_none() {
                    worker_failure = Some(failure);
                }
            }
        }
    }
    if let Some(failure) = worker_failure {
        return Err(failure.into_error());
    }

    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
        return Err(FastDnaError::Cancelled);
    }

    master_qc.finalize();

    // 3. Merge phase: bucket by bucket, so only one bucket's worth of
    // spilled data is resident at a time (see `disk_spill::merge_buckets`).
    let num_buckets = 1usize << bucket_bits;
    let merged = disk_spill::merge_buckets(&manifests, num_buckets)?;
    let counter = KmerCounter::from_sorted_entries(merged, total_occurrences.load(Ordering::Relaxed));

    if let Some(emit) = progress {
        if catch_unwind(AssertUnwindSafe(|| emit(Progress::Finished { reads: total_reads }))).is_err() {
            return Err(FastDnaError::Internal { detail: "progress callback panicked".to_string() });
        }
    }

    // `scratch` drops here, recursively removing every spill file this run
    // created -- including on the early returns above (a worker panic, a
    // cancellation, an I/O failure): `Drop` runs during unwinding as well
    // as ordinary scope exit, per this crate's `panic = "unwind"` profile
    // setting (see `Cargo.toml`).
    Ok((counter, master_qc, total_reads))
}

/// Streams a FASTQ source and returns its canonical k-mer counts, choosing
/// between FastDNA's two counting strategies automatically (or as forced
/// by `policy`) and reporting which one ran.
///
/// This is `process_stream_parallel` plus strategy selection, not a
/// replacement for it: `process_stream_parallel` itself is untouched and
/// keeps behaving exactly as it always has for any caller that does not
/// need the disk strategy (every one of this crate's existing tests calls
/// it directly, unmodified, for exactly that reason). Pass
/// `MemoryPolicy::default()` here to reproduce that same in-memory-only
/// behavior while additionally getting a `CountStrategy` back.
pub fn process_stream_parallel_with_policy<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
    policy: MemoryPolicy,
) -> Result<(KmerCounter, QcSummary, u64, StrategyDecision)> {
    validate_config(&config)?;
    let decision = resolve_strategy(&policy, &config);

    let (counter, qc, total_reads) = match decision.strategy {
        CountStrategy::InMemory => process_stream_parallel(reader, config, source, progress, cancel)?,
        CountStrategy::Disk => process_stream_parallel_disk(reader, config, source, progress, cancel)?,
    };

    Ok((counter, qc, total_reads, decision))
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: unwrap/expect denial
// is about production paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A disk-strategy worker's real `FastDnaError` (e.g. disk-full while
    /// spilling) must reach the caller typed, with its path and source
    /// intact -- not stringified into `Internal`, which blames the
    /// environment's failure on "a bug in FastDNA".
    #[test]
    fn a_worker_io_failure_stays_io_instead_of_becoming_internal() {
        let failure = DiskWorkerFailure::Error(FastDnaError::Io {
            path: std::path::PathBuf::from("spill_run_3.bin"),
            source: std::io::Error::new(std::io::ErrorKind::StorageFull, "disk full"),
        });
        match failure.into_error() {
            FastDnaError::Io { path, source } => {
                assert_eq!(path, std::path::PathBuf::from("spill_run_3.bin"));
                assert_eq!(source.kind(), std::io::ErrorKind::StorageFull);
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn a_worker_panic_still_becomes_internal() {
        let failure = DiskWorkerFailure::Panic("index out of bounds".to_string());
        match failure.into_error() {
            FastDnaError::Internal { detail } => assert!(detail.contains("index out of bounds")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    /// `env_override_applied` must be false when `--max-ram` shadowed the
    /// env var: the decision then owes nothing to the environment.
    #[test]
    fn policy_max_ram_reports_no_env_override_even_if_env_var_is_set() {
        // Set-and-remove of a process-global var: no other test in this
        // crate reads FASTDNA_MAX_RAM_BYTES (grep-verified), so the brief
        // window cannot race a concurrent reader.
        std::env::set_var("FASTDNA_MAX_RAM_BYTES", "123456789");
        let policy = MemoryPolicy {
            strategy: None,
            max_ram_bytes: Some(1_000_000_000),
            estimated_input_bytes: Some(10_000_000),
        };
        let decision = resolve_strategy(&policy, &PipelineConfig::default());
        std::env::remove_var("FASTDNA_MAX_RAM_BYTES");

        assert_eq!(decision.budget_bytes, 1_000_000_000, "--max-ram must win the budget");
        assert!(
            !decision.env_override_applied,
            "the env var did not shape this decision and must not claim credit"
        );
    }
}
