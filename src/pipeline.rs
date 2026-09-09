// src/pipeline.rs

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

use crate::adaptive_bins::SignatureHistogram;
use crate::binned::{BinStore, BinnedConfig};
use crate::counter::KmerCounter;
use crate::disk_spill::{self, ScratchDir, SpillWriter};
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqRecord, RecordSource};
use crate::kmer;
use crate::mem_estimate;
use crate::progress::{Progress, ProgressFn, PROGRESS_INTERVAL};
use crate::qc::QcSummary;

// `Clone`: every field is `Copy`, so this is a plain field-for-field copy.
// Added for `cohort::batch::count_paired_samples`, which builds one
// `PipelineConfig` per discovered sample from a single caller-supplied
// template rather than requiring the caller to reconstruct it by hand for
// every sample.
#[derive(Clone)]
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
    /// Collapse homopolymer runs (`kmer::homopolymer_compress_into`) before
    /// k-mer extraction. `false` by default: every existing caller keeps
    /// today's output byte-for-byte. Opt in for long-read (Nanopore/PacBio)
    /// input, where indels inside homopolymer runs -- not substitutions --
    /// are the dominant error mode; short-read Illumina data has no need
    /// for this and should leave it off.
    pub hpc: bool,
    /// Count each k-mer as it reads forward, instead of folding it together
    /// with its reverse complement. `true` (canonical) by default: every
    /// existing caller keeps today's output byte-for-byte, and canonical is
    /// the right answer for ordinary shotgun data, where a fragment is
    /// sequenced from an arbitrary end.
    ///
    /// `false` is `--no-canonical` on the CLI and KMC3's `-b`, and is for
    /// strand-specific input, where a k-mer and its reverse complement are
    /// two different observations rather than two views of one.
    pub canonical: bool,
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
            hpc: false,
            canonical: true,
        }
    }
}

type RecordBatch = Vec<FastqRecord>;

/// The depth of the producer -> worker channel. Bounded on purpose: the
/// backpressure is what keeps memory flat on multi-GB inputs.
const CHANNEL_DEPTH: usize = 64;

/// Capacity of the worker -> producer batch-recycling channel (see
/// `spawn_producer`).
///
/// Sized so `try_send` can never fail for lack of room, which is what keeps
/// the buffer pool from shrinking under load: the pool can never hold more
/// than `CHANNEL_DEPTH + num_threads + 1` batches (see the argument in
/// `spawn_producer`), so a channel of that size can always take one back.
fn recycle_depth(num_threads: usize) -> usize {
    CHANNEL_DEPTH + num_threads + 1
}

/// A worker's result: its private counter and QC state, or the message from
/// a panic that occurred while it was processing (see `catch_unwind` below).
/// Generic over the sink so the same worker body serves both k-mer
/// widths; see `CountSink`.
type WorkerOutcome<C> = std::result::Result<(C, QcSummary), String>;

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

/// Where to say a producer-side failure happened.
///
/// A single-stream source (`FastqReader`) has no path of its own, so the
/// answer is the `source` the caller named plus the producer's own running
/// read count -- exactly what this was before multi-file input existed. A
/// source spanning several files answers for itself, naming the file it was
/// actually reading and numbering the record within *that* file: told
/// "record 3 of lane4.fastq.gz", a user can find it; told "record 4,000,003
/// of <inputs>", they cannot.
fn failing_location<S: RecordSource>(
    reader: &S,
    fallback_path: &Path,
    reads_so_far: u64,
) -> (PathBuf, u64) {
    match reader.current_source() {
        Some((path, records_in_file)) => (path, records_in_file + 1),
        None => (fallback_path.to_path_buf(), reads_so_far + 1),
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
    validate_config_for(config, 32)
}

/// `validate_config`, told which engine is asking.
///
/// `max_k` is the calling sink's own `MAX_K` rather than a constant: the
/// narrow engine tops out at 32 because two bits per base fill a `u64`
/// there, and the wide one at 64 for the same reason in a `u128`. A single
/// hard-coded 32 here would have made the wide path reject every input it
/// exists to accept.
fn validate_config_for(config: &PipelineConfig, max_k: usize) -> Result<()> {
    if config.k == 0 || config.k > max_k {
        return Err(FastDnaError::InvalidK { k: config.k, max: max_k });
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
    // single record, so each read crosses the `CHANNEL_DEPTH` channel in its
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

/// Spawns the producer thread that both counting strategies share.
///
/// Each strategy used to carry its own byte-identical copy of this loop --
/// see `process_stream_parallel_disk`'s doc comment for the reasoning, which
/// held while the loop was six lines of `next_record` and `push`. It is no
/// longer six lines: it refills record buffers in place and takes drained
/// batches back over a recycling channel, and two copies of that would be
/// two chances for the strategies to drift on something
/// `tests/dual_strategy.rs` requires to be bit-identical. What stays
/// per-strategy is what genuinely differs -- what a worker does with a batch
/// -- not how batches are produced.
///
/// # Allocation
///
/// `next_record` allocates `id`, `seq` and `qual` fresh for every record and
/// the consuming worker frees them: 3 x 7 million = 21 million
/// allocate/free pairs on the 7-million-record benchmark file. This loop
/// instead writes into a record slot the batch already owns, via
/// `RecordSource::next_record_into`, and gets whole batches back from the
/// workers over `recycle_rx`. Each slot's three `Vec`s reach their
/// high-water mark within the first batch and never reallocate after that,
/// so the steady-state cost is zero allocations per record. What remains is
/// the one-time fill of the pool: at most
/// `CHANNEL_DEPTH + num_threads + 1` batches x `batch_size` records x 3
/// `Vec`s -- ~2.2 million for the default 8 threads and a 10 000-record
/// batch, against 21 million repeated, i.e. ~19 million allocate/free pairs
/// removed.
///
/// # Why the recycling channel cannot deadlock
///
/// Every use of it is non-blocking: `try_recv` here, `try_send` in the
/// workers. No thread ever waits on it, so it adds no edge to the wait-for
/// graph and cannot participate in a cycle -- the forward channel's
/// discipline (bounded, workers drain to disconnect on every exit path) is
/// untouched and remains the only place anything blocks. A worker that dies
/// holding a batch costs this loop one allocation, not a hang; once this
/// thread returns and drops `recycle_rx`, every worker's `try_send` reports
/// `Disconnected` and the batch is simply dropped.
///
/// # Why it cannot inflate peak memory
///
/// A new batch is allocated only when `try_recv` finds the pool empty, and
/// at that instant every other batch is either in the forward channel
/// (<= `CHANNEL_DEPTH`) or in a worker's hand (<= `num_threads`). So the
/// pool never exceeds `CHANNEL_DEPTH + num_threads + 1` batches -- the same
/// number that are live simultaneously today -- and the recycling channel,
/// sized to hold all of them, is drawing from that same fixed pool rather
/// than adding to it.
fn spawn_producer<S: RecordSource>(
    reader: S,
    sender: Sender<RecordBatch>,
    recycle_rx: Receiver<RecordBatch>,
    batch_size: usize,
    source_owned: PathBuf,
    cancel_for_reader: Option<Arc<AtomicBool>>,
) -> thread::JoinHandle<Result<u64>> {
    thread::spawn(move || -> Result<u64> {
        let mut reader = reader;
        // Nothing downstream of this producer reads `FastqRecord::id`:
        // grep-verified across the crate -- `qc::observe_record`, both
        // worker loops, `kmer` and `export` touch only `seq` and `qual`.
        // So the reader is told it may consume each FASTQ header line
        // without copying it: headers are ~40-60 bytes across 7 million
        // records, i.e. ~350 MB memcpy'd and immediately discarded. Set
        // here, on the reader this run owns, so every other caller of
        // `FastqReader` keeps the default and still gets ids.
        reader.set_keep_ids(false);

        let mut current_batch: RecordBatch = Vec::with_capacity(batch_size);
        // Slots `0..filled` of `current_batch` hold this batch's reads.
        // Anything past that is a recycled slot still waiting to be
        // refilled -- present on every batch after the first, which is
        // exactly what makes the refill allocation-free.
        let mut filled: usize = 0;
        let mut total_reads: u64 = 0;
        let mut cancelled = false;

        loop {
            if filled == current_batch.len() {
                // Grows the batch by one slot. This fires only while a
                // freshly allocated batch is filling for the first time: a
                // recycled batch always comes back with exactly
                // `batch_size` slots, because the only batch ever sent
                // shorter than that is the final partial one, which is sent
                // at end of stream and so is never refilled. Every record
                // after the pool has warmed up therefore skips this and
                // reuses the three `Vec`s the slot already owns.
                current_batch.push(FastqRecord::default());
            }
            let Some(slot) = current_batch.get_mut(filled) else {
                // Unreachable: the push above guarantees
                // `filled < current_batch.len()`. Reached fallibly rather
                // than by `unwrap` (denied crate-wide) or `unsafe`.
                return Err(FastDnaError::Internal {
                    detail: "producer record slot missing".to_string(),
                });
            };

            match reader.next_record_into(slot) {
                Ok(true) => {
                    filled += 1;
                    total_reads += 1;

                    if filled >= batch_size {
                        // Checked once per batch dispatch, not per record:
                        // an atomic load in the per-record hot path would
                        // cost measurable throughput for no benefit -- a
                        // human pressing Ctrl-C does not need sub-batch
                        // latency.
                        if let Some(tok) = &cancel_for_reader {
                            if tok.load(Ordering::Relaxed) {
                                cancelled = true;
                                break;
                            }
                        }

                        // `filled == batch_size` and no batch ever grows
                        // past `batch_size`, so the batch is exactly full
                        // here and needs no truncation.
                        debug_assert_eq!(current_batch.len(), filled);
                        let next = recycle_rx
                            .try_recv()
                            .unwrap_or_else(|_| Vec::with_capacity(batch_size));
                        let batch_to_send = std::mem::replace(&mut current_batch, next);
                        filled = 0;
                        if sender.send(batch_to_send).is_err() {
                            break;
                        }
                    }
                }
                Ok(false) => break,
                // A genuine I/O failure (a corrupt gzip member, an NFS read
                // error) is not a data problem, so it must not be reported
                // as MalformedFastq -- that would blame the bytes for a
                // hardware/transport fault and attach a record number that
                // means nothing for it.
                Err(FastqReadError::Io(source_err)) => {
                    let (path, _) = failing_location(&reader, &source_owned, total_reads);
                    return Err(FastDnaError::Io { path, source: source_err });
                }
                // A structural violation in the bytes themselves: attach the
                // path and the 1-based index of the record that failed.
                Err(FastqReadError::Malformed(reason)) => {
                    let (path, record) = failing_location(&reader, &source_owned, total_reads);
                    return Err(FastDnaError::MalformedFastq { path, record, reason });
                }
            }
        }

        // Drops the slot pushed but never filled by the final iteration,
        // plus any recycled slots this last batch did not reach. Without
        // it a worker would count blank records that were never read.
        current_batch.truncate(filled);

        // On cancellation, drop the leftover batch instead of sending it:
        // the whole result is discarded by the Cancelled check in the
        // caller, so there is no point handing workers more to chew
        // through. `sender` is dropped when this closure returns either
        // way, which disconnects the channel and lets any worker still
        // consuming exit its loop.
        if !cancelled && !current_batch.is_empty() {
            let _ = sender.send(current_batch);
        }

        Ok(total_reads)
    })
}

/// Which k-mer encoding a run should use.
///
/// Two engines, chosen by `k`: `kmer.rs`/`counter.rs` pack two bits per
/// base into a `u64` and stop at 32 bases; `wide_kmer.rs`/`wide_counter.rs`
/// do the same in a `u128` and stop at 64. They are separate rather than
/// one generic engine for the reasons `wide_kmer.rs`'s module doc gives --
/// chiefly that the narrow path is the one measured exactly equal to KMC3,
/// and re-validating a generic rewrite of it is a larger job than writing
/// a second engine beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountEngine {
    /// `1 <= k <= 32`.
    Narrow,
    /// `1 <= k <= 64`. Accepts the narrow range too, which is what makes a
    /// differential check on real data possible: counting one file both
    /// ways at `k = 31` must produce the same k-mers, and
    /// `wide_kmer.rs`'s own test asserts that on synthetic input.
    Wide,
}

impl CountEngine {
    pub fn as_str(self) -> &'static str {
        match self {
            CountEngine::Narrow => "narrow",
            CountEngine::Wide => "wide",
        }
    }
}

/// What a caller asked for, before `k` is taken into account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineChoice {
    /// Pick by `k`. The default, and what every existing caller gets.
    #[default]
    Auto,
    Narrow,
    Wide,
}

/// Resolves the engine a run will use, or explains why the request cannot
/// be honoured.
///
/// `Auto` routes by `k` alone: at or below 32 the narrow engine is both
/// faster and the one with external validation behind it, so it wins;
/// above that only the wide engine can represent the k-mer at all.
///
/// Forcing is honoured where it is possible and refused where it is not,
/// with the refusal naming the flag that fixes it:
///
/// * `Narrow` with `k > 32` cannot work -- 33 bases do not fit in a `u64`
///   -- so it is an error rather than a silent upgrade. A caller who named
///   an engine gets that engine or an explanation, never a different one.
/// * `Wide` with `k <= 32` is allowed. It is slower and has no external
///   validation of its own, and it is exactly what a differential check
///   needs, so it is available rather than second-guessed.
/// * Any `k` above 64 is outside both engines, and is `InvalidK` rather
///   than an engine complaint: no flag fixes it.
pub fn resolve_engine(choice: EngineChoice, k: usize) -> Result<CountEngine> {
    if k == 0 || k > crate::wide_kmer::MAX_WIDE_K {
        return Err(FastDnaError::InvalidK { k, max: crate::wide_kmer::MAX_WIDE_K });
    }
    match choice {
        EngineChoice::Auto => {
            Ok(if k <= 32 { CountEngine::Narrow } else { CountEngine::Wide })
        }
        EngineChoice::Wide => Ok(CountEngine::Wide),
        EngineChoice::Narrow if k <= 32 => Ok(CountEngine::Narrow),
        EngineChoice::Narrow => Err(FastDnaError::InvalidConfig {
            parameter: "engine",
            reason: format!(
                "k={k} needs more than the 64 bits the narrow engine packs a k-mer into \
                 (2 bits per base, so 32 bases at most). Use --engine wide (or --engine auto, \
                 which picks it for you above k=32), or lower k to 32 or less."
            ),
        }),
    }
}

/// The per-worker counting state, abstracted over k-mer width.
///
/// # Why this trait exists, and why it is this small
///
/// The in-memory pipeline's worker body is about ninety lines, and almost
/// all of it is the argument for why it cannot deadlock: `catch_unwind`
/// around the body, drain-and-discard on panic *and* on cancellation so a
/// producer blocked on a full channel always has someone reading, batch
/// recycling, and progress accounted per batch rather than per record.
/// Exactly three lines of it depend on how wide a k-mer is.
///
/// Supporting `k > 32` (`wide_kmer.rs`) meant either duplicating those
/// ninety lines -- two copies of a deadlock argument, which is how one of
/// them eventually stops being true -- or abstracting the three. This is
/// the three.
///
/// The narrow path's behaviour is unchanged by construction:
/// `process_stream_parallel` still takes and returns exactly what it did,
/// and monomorphisation gives `NarrowSink` the same code the concrete
/// version had. `docs/BENCHMARKS.md` records a measurement confirming that
/// rather than assuming it.
trait CountSink: Send {
    /// The largest `k` this sink's encoding can hold. `validate_config`
    /// checks against this rather than against a constant, so the two
    /// engines cannot disagree with the checker about their own range.
    const MAX_K: usize;

    /// What the worker pool folds down to. The narrow sink hands back a
    /// `KmerCounter` (still lazily finalizable, which `export.rs` relies
    /// on); the wide one hands back an already-finished table.
    type Output: Send;

    /// A fresh sink for one worker.
    fn worker() -> Self;

    /// Extracts every canonical k-mer of `seq` at width `k` and counts it.
    ///
    /// The scratch buffer lives inside the sink rather than in the worker
    /// body: it is allocated once per worker and refilled per record,
    /// which is the allocation the `_into` extraction forms exist for, and
    /// its element type is the one thing that differs between widths.
    /// Extracts `seq`'s k-mers and folds them in. `canonical` picks which
    /// extraction: `true` folds each k-mer with its reverse complement (the
    /// default and what shotgun data wants), `false` counts it as it reads
    /// forward (`--no-canonical`, KMC3's `-b`).
    fn absorb(&mut self, seq: &[u8], k: usize, canonical: bool);

    /// Folds every worker's sink into the run's answer.
    fn merge_all(sinks: Vec<Self>) -> Self::Output
    where
        Self: Sized;
}

/// `k <= 32`: `kmer.rs` and `counter.rs`, the path validated exactly
/// against KMC3.
struct NarrowSink {
    counter: KmerCounter,
    scratch: Vec<u64>,
}

impl CountSink for NarrowSink {
    const MAX_K: usize = 32;
    type Output = KmerCounter;

    fn worker() -> Self {
        Self { counter: KmerCounter::with_capacity(131_072), scratch: Vec::new() }
    }

    fn absorb(&mut self, seq: &[u8], k: usize, canonical: bool) {
        if canonical {
            kmer::extract_canonical_kmers_into(seq, k, &mut self.scratch);
        } else {
            kmer::extract_forward_kmers_into(seq, k, &mut self.scratch);
        }
        self.counter.insert_batch(&self.scratch);
    }

    fn merge_all(sinks: Vec<Self>) -> KmerCounter {
        KmerCounter::merge_all(sinks.into_iter().map(|sink| sink.counter).collect())
    }
}

/// `33 <= k <= 64`: `wide_kmer.rs` and `wide_counter.rs`.
struct WideSink {
    counter: crate::wide_counter::WideKmerCounter,
    scratch: Vec<u128>,
}

impl CountSink for WideSink {
    const MAX_K: usize = crate::wide_kmer::MAX_WIDE_K;
    type Output = crate::wide_counter::WideCounts;

    fn worker() -> Self {
        Self { counter: crate::wide_counter::WideKmerCounter::new(), scratch: Vec::new() }
    }

    fn absorb(&mut self, seq: &[u8], k: usize, canonical: bool) {
        if canonical {
            crate::wide_kmer::extract_canonical_kmers_into(seq, k, &mut self.scratch);
        } else {
            crate::wide_kmer::extract_forward_kmers_into(seq, k, &mut self.scratch);
        }
        self.counter.insert_batch(&self.scratch);
    }

    fn merge_all(sinks: Vec<Self>) -> crate::wide_counter::WideCounts {
        crate::wide_counter::WideKmerCounter::merge_all(
            sinks.into_iter().map(|sink| sink.counter).collect(),
        )
        .finish()
    }
}

pub fn process_stream_parallel<S: RecordSource>(
    reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    process_stream_parallel_with_sink::<S, NarrowSink>(reader, config, source, progress, cancel)
}

/// The same streaming count, for `33 <= k <= 64`, through `wide_kmer.rs`
/// and `wide_counter.rs`.
///
/// Identical in every respect except the width of a k-mer: same producer,
/// same channel and backpressure, same cancellation and panic handling,
/// same quality trimming and homopolymer compression. It returns an
/// already-finished `WideCounts` rather than a `KmerCounter` because the
/// wide counter has no lazy finalization to preserve (`wide_counter.rs`
/// explains why it does not need any).
///
/// `pipeline`'s automatic strategy chooser does not reach this: the disk
/// and binned strategies are `u64`-shaped throughout (`disk_spill.rs`
/// spills 8-byte keys, `binned.rs` packs super-k-mers two bits per base
/// into a store sized for them), so wide counting is in-memory only. That
/// is a real limitation and `resolve_engine` states it where a caller will
/// see it rather than leaving it to be discovered.
pub fn process_stream_parallel_wide<S: RecordSource>(
    reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(crate::wide_counter::WideCounts, QcSummary, u64)> {
    process_stream_parallel_with_sink::<S, WideSink>(reader, config, source, progress, cancel)
}

fn process_stream_parallel_with_sink<S: RecordSource, C: CountSink>(
    reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(C::Output, QcSummary, u64)> {
    validate_config_for(&config, C::MAX_K)?;
    // The source gets a say too: an input list that can produce nothing at
    // all is a caller mistake, and a run that counted zero reads in silence
    // is indistinguishable from a real sample that happened to be empty.
    reader.validate()?;

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(CHANNEL_DEPTH);
    // Drained batches travel back to the producer here so their record
    // buffers can be refilled in place instead of reallocated per record.
    // See `spawn_producer` for the allocation count, the deadlock argument,
    // and why this cannot raise peak memory.
    let (recycle_tx, recycle_rx): (Sender<RecordBatch>, Receiver<RecordBatch>) =
        bounded(recycle_depth(config.num_threads));

    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let progress_interval = config.progress_interval;
    let hpc = config.hpc;
    let canonical = config.canonical;

    // 1. Producer thread. Returns the read count, or the record it choked on.
    //    `cancel` is cloned rather than borrowed: the reader thread is
    //    'static and needs its own owned handle to the flag, distinct from
    //    the one the rayon workers check (see the function doc comment for
    //    why this must be Arc).
    let reader_handle = spawn_producer(
        reader,
        sender,
        recycle_rx,
        config.batch_size,
        source.to_path_buf(),
        cancel.clone(),
    );

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
    let results: Vec<WorkerOutcome<C>> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                // The sink owns both the counting table and the k-mer
                // scratch buffer, allocated once per worker instead of
                // once per record. `extract_canonical_kmers` allocates a
                // fresh `Vec` on every call, so the per-record form costs
                // one malloc/free pair per read -- ~7 million of them on a
                // 2.14 GB FASTQ. The `_into` variant the sink calls clears
                // and refills that buffer, which reaches its high-water
                // mark within the first few reads and never allocates
                // again. See `CountSink` for why this is behind a trait.
                let mut sink = C::worker();
                let mut local_qc = QcSummary::default();
                // Only ever populated when `hpc` is set; otherwise k-mer
                // extraction reads `record.seq` directly, so a run with the
                // flag off pays no allocation and no extra pass at all.
                let mut hpc_buf: Vec<u8> = Vec::new();

                while let Ok(mut batch) = receiver.recv() {
                    if let Some(tok) = &cancel {
                        if tok.load(Ordering::Relaxed) {
                            break;
                        }
                    }

                    // Computed before the record loop below, per the
                    // batch-granularity progress accounting, and before the
                    // batch is handed back to the producer.
                    let n = batch.len() as u64;

                    for record in &mut batch {
                        local_qc.observe_record(record);
                        record.quality_trim_end(min_qual, qual_win);

                        // Canonical k-mers: 2-bit packed, O(1) rolling window
                        // on both strands, and ambiguous bases ('N') reset the
                        // window rather than producing corrupt k-mers.
                        let seq: &[u8] = if hpc {
                            kmer::homopolymer_compress_into(&record.seq, &mut hpc_buf);
                            &hpc_buf
                        } else {
                            &record.seq
                        };
                        sink.absorb(seq, k, canonical);
                    }

                    // Hand the record buffers back so the producer can refill
                    // them in place. Non-blocking and best-effort by design:
                    // a disconnected pool (the producer has finished) just
                    // drops the batch, exactly as this loop did before the
                    // pool existed, and the pool is sized so a full one is
                    // impossible. See `spawn_producer`.
                    let _ = recycle_tx.try_send(batch);

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

                (sink, local_qc)
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
    let mut counters: Vec<C> = Vec::with_capacity(results.len());
    // `QcSummary::merge` is O(1) -- five `u64` additions -- so it is folded
    // in right here, sequentially, exactly as the disk strategy already
    // does. A k-way form of it would buy nothing, and doing it here rather
    // than in a second pass keeps the counters unpaired for `merge_all`
    // below. `merge` deliberately touches only the raw counters, never the
    // percentages, which is what makes folding in this order equivalent to
    // any other; `finalize` runs once at the end.
    let mut master_qc = QcSummary::default();
    for outcome in results {
        match outcome {
            Ok((sink, qc)) => {
                counters.push(sink);
                master_qc.merge(&qc);
            }
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

    // 3. Combine phase: one k-way merge, not a pairwise reduction tree.
    //
    // `reduce` folded two counters at a time, so every entry was rewritten
    // once per level of the tree. At benchmark scale (8 workers, 53.8
    // million distinct k-mers, 16 bytes per entry) that is 4 merges
    // producing 13.4M entries, 2 producing 26.9M and 1 producing 53.8M:
    // 161 million entries written, ~2.6 GB of `memcpy`, in 7 allocations
    // the largest of which is 860 MB. `merge_all` writes each of the 53.8
    // million final entries exactly once -- 860 MB in one allocation, so
    // ~1.7 GB of copying and 6 large allocations removed.
    //
    // Giving up the tree's parallelism costs nothing even on the critical
    // path: the tree's longest chain is 13.4M + 26.9M + 53.8M = 94.1M
    // entries written even if every level ran perfectly in parallel, and
    // the levels compete for the same memory bandwidth rather than
    // multiplying it. The per-entry comparison count is unchanged too --
    // an entry crossing `log2(workers)` merge levels at one compare each
    // is the same `log(sources)` the k-way heap charges it.
    let master_counter = C::merge_all(counters);

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

/// Which of FastDNA's counting strategies produced a result.
///
/// `InMemory` (`process_stream_parallel`) sorts and compacts entirely in
/// RAM and is the fastest whenever the input fits; `Disk` (see
/// `disk_spill.rs`) partitions k-mers into buckets, spills each to scratch
/// files, and merges bucket by bucket so only one bucket is resident at
/// once, trading speed for a peak memory footprint that does not scale
/// with `threads * distinct_kmers` the way the in-memory strategy's does
/// (see `mem_estimate.rs` for the measured reason that scaling exists).
///
/// `Binned` (see `binned.rs`) partitions by canonical minimizer instead,
/// storing runs of consecutive k-mers that share a bin as 2-bit packed
/// super-k-mers rather than one `u64` per occurrence, and counting bin by
/// bin in RAM. It is **opt-in only**: `resolve_strategy`'s automatic
/// chooser never selects it, and `--strategy binned` /
/// `FASTDNA_STRATEGY=binned` are the only ways to reach it. That is
/// deliberate -- `docs/design-minimizer-counting.md` §5 step 6 makes
/// promoting it to the automatic chooser a separate, later decision, to be
/// taken only once a real (non-synthetic) sample has been counted correctly
/// and a per-bin occupancy report examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountStrategy {
    InMemory,
    Disk,
    Binned,
}

impl CountStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CountStrategy::InMemory => "in-memory",
            CountStrategy::Disk => "disk",
            CountStrategy::Binned => "binned",
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
    /// True when neither `policy.strategy` nor `FASTDNA_STRATEGY` named a
    /// strategy and the estimate chose on its own. Only an automatic choice
    /// is second-guessed by the bin-balance check in
    /// `process_stream_parallel_with_policy`: a caller who asked for a
    /// strategy by name gets the one they asked for.
    pub auto_selected: bool,
    /// How lopsided the binned strategy's bins would be on this input's
    /// first `BIN_BALANCE_SAMPLE_RECORDS` records
    /// (`DynamicBinMap::predicted_skew`), when that was measured. `None`
    /// when nothing measured it -- which is every run that did not have
    /// binned as its automatic choice.
    pub bin_balance: Option<f64>,
}

/// Decides which counting strategy a run should use, without running
/// anything. Pure and side-effect-free except for reading two environment
/// variables (see below), so a caller can call this once to report the
/// decision (e.g. the CLI, before it prints its banner) and
/// `process_stream_parallel_with_policy` can call it again internally to
/// act on it, with no risk of the two disagreeing.
///
/// Resolution order: an explicit `policy.strategy` wins outright. Failing
/// that, `FASTDNA_STRATEGY` (`disk`, `memory`/`in-memory`, or `binned`) is
/// checked --
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
///
/// The automatic arm chooses between `InMemory` and `Disk` and **nothing
/// else**: `CountStrategy::Binned` is reachable only by being named, in
/// `policy.strategy` or in `FASTDNA_STRATEGY`. `the_automatic_chooser_never_
/// selects_the_binned_strategy` pins that, because promoting it is §5 step 6
/// -- a separate decision that is explicitly out of scope until a real
/// sample has been counted correctly.
pub fn resolve_strategy(policy: &MemoryPolicy, config: &PipelineConfig) -> StrategyDecision {
    let env_max_ram = std::env::var("FASTDNA_MAX_RAM_BYTES").ok().and_then(|s| s.trim().parse::<u64>().ok());
    let budget_bytes = policy
        .max_ram_bytes
        .or(env_max_ram)
        .unwrap_or_else(mem_estimate::default_max_ram_bytes);

    let estimated_occurrences =
        policy.estimated_input_bytes.map(mem_estimate::estimate_occurrences_from_bytes).unwrap_or(0);
    // The in-memory strategy's predicted peak. This is the number the
    // automatic chooser needs and the only one it may use, because the
    // question it asks is precisely "would an in-memory run fit?" -- the
    // fallback to `Disk` below is what happens when the answer is no. It is
    // *not* necessarily the number reported, which is the peak of whichever
    // strategy actually got chosen; see `estimated_peak_bytes` at the end.
    let in_memory_peak_bytes =
        mem_estimate::estimate_peak_bytes(estimated_occurrences, config.num_threads);

    let env_strategy = std::env::var("FASTDNA_STRATEGY").ok().and_then(|s| match s.trim() {
        "disk" => Some(CountStrategy::Disk),
        "memory" | "in-memory" => Some(CountStrategy::InMemory),
        // Reachable only by asking for it by name, exactly like the two
        // above -- and unlike them, it is never reachable any other way.
        // The `auto` arm below deliberately does not know this variant
        // exists; see `CountStrategy::Binned`.
        "binned" => Some(CountStrategy::Binned),
        _ => None,
    });

    // The binned strategy's predicted peak, for the same question asked of
    // the in-memory one above. Evaluated at the defaults the binned path
    // actually runs with (`BinnedConfig::new(k).sanitized()`), so the
    // prediction is of the run that would happen, not of a hypothetical
    // configuration.
    let binned_defaults = BinnedConfig::new(config.k).sanitized();
    let binned_peak_bytes = mem_estimate::estimate_binned_peak_bytes(
        estimated_occurrences,
        config.num_threads,
        binned_defaults.num_bins,
        binned_defaults.chunk_bytes,
    );

    let (strategy, env_override_applied, auto_selected) = match policy.strategy {
        Some(s) => (s, false, false),
        None => match env_strategy {
            Some(s) => (s, true, false),
            None => {
                // Binned first, when it fits: measured at 2.9x faster than
                // the in-memory strategy and at half its peak on 840M
                // occurrences (`docs/BENCHMARKS.md`), with byte-identical
                // output. It is chosen only for an input whose size is
                // known, exactly as `Disk` is -- a stream (`-`) gives the
                // estimator nothing to work with, and the strategy that
                // needs no estimate to be safe is the one that should run
                // blind.
                //
                // This choice is provisional in a way the other two are
                // not: `process_stream_parallel_with_policy` measures the
                // input's actual bin balance before committing, and
                // downgrades if the bins cannot spread. See
                // `MAX_ACCEPTABLE_BIN_SKEW`.
                let sized_input = policy.estimated_input_bytes.is_some();
                let auto = if sized_input && binned_peak_bytes <= budget_bytes {
                    CountStrategy::Binned
                } else if sized_input && in_memory_peak_bytes > budget_bytes {
                    CountStrategy::Disk
                } else if !sized_input && config.num_threads >= BINNED_BLIND_MIN_THREADS {
                    // An unsized input (a stream: `-`, a pipe) used to land
                    // on `InMemory` here, on the reasoning that the strategy
                    // needing no estimate to be safe should run blind. That
                    // reasoning was right and the conclusion was backwards:
                    // it is `Binned` that needs no estimate, above a thread
                    // count -- see `BINNED_BLIND_MIN_THREADS`.
                    CountStrategy::Binned
                } else {
                    CountStrategy::InMemory
                };
                // The env var only "applied" if it actually supplied the
                // budget -- when `policy.max_ram_bytes` is set it wins at
                // the `.or(env_max_ram)` above and the env value had no
                // effect on the decision.
                (auto, policy.max_ram_bytes.is_none() && env_max_ram.is_some(), true)
            }
        },
    };

    // Report the predicted peak of the strategy that was actually chosen,
    // not of the one the chooser happened to evaluate. Before
    // `mem_estimate::estimate_binned_peak_bytes` existed there was only one
    // model, so `--strategy binned` was reported with the in-memory
    // strategy's prediction -- on the benchmark input that is 8.34 GB
    // announced for a run the design says costs 2.36 GiB, i.e. an error
    // larger than the entire footprint being predicted. A user picks
    // `binned` precisely to fit a run into memory it otherwise would not,
    // which makes that specific line the one they are most likely to be
    // reading, and the worst one to leave wrong.
    //
    // The automatic decision above used to be untouched by this -- it
    // branched on `in_memory_peak_bytes` alone and `auto` could not select
    // binned at all. That changed on 2026-09-05, once the binned model was
    // calibrated against real runs and the bin-balance guard existed to
    // catch the input shape it loses on; `auto` now evaluates both models
    // and this line reports whichever strategy that produced.
    //
    // `Disk` keeps the in-memory model's figure. That is also not the right
    // number for it -- the whole point of spilling is that peak RSS stops
    // tracking `threads * distinct_kmers` -- but building and justifying a
    // disk model is not this change's subject, and substituting a guess
    // would be exactly the invented number `docs/PERFORMANCE_PLAN.md`
    // forbids. It is left visibly as it was rather than quietly improved.
    let estimated_peak_bytes = match strategy {
        CountStrategy::Binned => {
            // The same config `process_stream_parallel_binned` builds, so
            // the prediction describes the run that will actually happen.
            let binned = BinnedConfig::new(config.k).sanitized();
            mem_estimate::estimate_binned_peak_bytes(
                estimated_occurrences,
                config.num_threads,
                binned.num_bins,
                binned.chunk_bytes,
            )
        }
        CountStrategy::InMemory | CountStrategy::Disk => in_memory_peak_bytes,
    };

    StrategyDecision {
        strategy,
        estimated_occurrences,
        estimated_peak_bytes,
        budget_bytes,
        env_override_applied,
        auto_selected,
        // Never measured here: `resolve_strategy` is pure and reads no
        // input. `process_stream_parallel_with_policy` fills this in when
        // it samples.
        bin_balance: None,
    }
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
/// `process_stream_parallel`, not a variant of it: the consumer pool and
/// reduce phase below are intentionally kept separate rather than factored
/// out from that function, so this strategy's own bugs cannot reach back
/// into the in-memory path's already-tested behavior, and vice versa.
///
/// The one exception is the producer thread (`spawn_producer`), which the
/// two strategies now share. It was duplicated while it was a six-line
/// `next_record`/`push` loop; it is now a buffer-recycling loop whose
/// behaviour `tests/dual_strategy.rs` requires to be bit-identical across
/// the two strategies, and one shared implementation is the safer way to
/// guarantee that than two copies. Nothing strategy-specific lives in it.
///
/// `config` must already be valid (`process_stream_parallel_with_policy`
/// validates it before choosing a strategy).
fn process_stream_parallel_disk<S: RecordSource>(
    reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    let scratch = ScratchDir::new()?;
    let bucket_bits = disk_spill::DEFAULT_BUCKET_BITS;

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(CHANNEL_DEPTH);
    let (recycle_tx, recycle_rx): (Sender<RecordBatch>, Receiver<RecordBatch>) =
        bounded(recycle_depth(config.num_threads));

    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let progress_interval = config.progress_interval;
    let hpc = config.hpc;
    let canonical = config.canonical;

    // 1. Producer thread -- the same one the in-memory strategy uses. This
    // is the one piece the two strategies share, because it is the one
    // piece that must behave identically for `tests/dual_strategy.rs` to
    // hold; see `spawn_producer` for why it is no longer duplicated.
    let reader_handle = spawn_producer(
        reader,
        sender,
        recycle_rx,
        config.batch_size,
        source.to_path_buf(),
        cancel.clone(),
    );

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
                // Worker-owned scratch; same reasoning as the in-memory
                // strategy above -- one allocation per worker instead of one
                // per record.
                let mut canon_kmers: Vec<u64> = Vec::new();
                // Same "only populated when `hpc` is set" rationale as the
                // in-memory strategy's worker loop.
                let mut hpc_buf: Vec<u8> = Vec::new();

                while let Ok(mut batch) = receiver.recv() {
                    if let Some(tok) = &cancel {
                        if tok.load(Ordering::Relaxed) {
                            break;
                        }
                    }

                    let n = batch.len() as u64;

                    // Accumulated in a local and published once per batch
                    // rather than once per record. `fetch_add` on a shared
                    // counter is a contended atomic read-modify-write on one
                    // cache line, and every worker was hitting it on every
                    // read: 7 million on the benchmark file, versus one per
                    // batch (~700 for the whole run at a 10 000-record
                    // batch). The total is unchanged -- addition is
                    // associative and commutative, so the order the
                    // per-batch sums land in does not matter -- and the
                    // value is still only read after every worker has
                    // finished and `results` has been collected, which is
                    // what makes `Relaxed` sound here as it was before. A
                    // worker that panics or hits a spill error loses its
                    // current batch's partial sum, which is unobservable:
                    // both paths make the whole call return `Err`.
                    let mut batch_occurrences: u64 = 0;

                    for record in &mut batch {
                        local_qc.observe_record(record);
                        record.quality_trim_end(min_qual, qual_win);

                        let seq: &[u8] = if hpc {
                            kmer::homopolymer_compress_into(&record.seq, &mut hpc_buf);
                            &hpc_buf
                        } else {
                            &record.seq
                        };
                        if canonical {
                            kmer::extract_canonical_kmers_into(seq, k, &mut canon_kmers);
                        } else {
                            kmer::extract_forward_kmers_into(seq, k, &mut canon_kmers);
                        }
                        batch_occurrences += canon_kmers.len() as u64;
                        spill.insert_batch(&canon_kmers)?;
                    }

                    total_occurrences.fetch_add(batch_occurrences, Ordering::Relaxed);

                    // See the in-memory strategy's worker loop: non-blocking
                    // and best-effort, so this can neither block nor fail in
                    // a way that matters.
                    let _ = recycle_tx.try_send(batch);

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

/// The binned strategy's per-worker outcome. Only a `QcSummary`: the
/// counting result itself never passes through here, because the whole
/// point is that a worker writes into the shared `BinStore` instead of
/// building anything of its own.
type BinnedWorkerOutcome = std::result::Result<QcSummary, String>;

/// How many leading records of a `binned` run are read directly (not via
/// the producer/worker pool) to build a data-adaptive bin map before the
/// main pipeline starts -- the fix for the R3 bin-skew finding in
/// `docs/design-minimizer-counting.md` (see `adaptive_bins.rs`). Sized as a
/// compromise: large enough that a real amplicon panel's dominant,
/// near-identical sequence is measured across many reads rather than
/// guessed from one or two, small enough that this single-threaded warm-up
/// pass -- run before any worker starts -- is not where a run's wall clock
/// goes. Not calibrated by a sweep, the same honest caveat
/// `DEFAULT_NUM_BINS` and `DEFAULT_CHUNK_BYTES` carry.
///
/// These records are never discarded: they are counted for real, through
/// the very map they were sampled to build, immediately after it exists
/// (see the warm-up block in `process_stream_parallel_binned`).
const ADAPTIVE_SAMPLE_RECORDS: usize = 20_000;

/// The minimizer-partitioned counting strategy
/// (`docs/design-minimizer-counting.md`, and see `binned.rs` for the
/// algorithm).
///
/// Structurally a sibling of `process_stream_parallel` and
/// `process_stream_parallel_disk`, not a variant of either, for the same
/// reason those two are siblings of each other: this strategy's own bugs
/// must not be able to reach back into the two already-tested paths. The
/// producer (`spawn_producer`) is shared, as it is between those two, and
/// nothing strategy-specific lives in it.
///
/// What a worker does per record is deliberately the same shape as the other
/// two -- `qc.observe_record` before trimming, then `quality_trim_end`, then
/// the k-mer work -- so `tests/dual_strategy.rs` can require all three to be
/// indistinguishable from their counts alone. The one difference is the last
/// step: instead of extracting a `Vec<u64>` of occurrences, the worker cuts
/// the sequence into super-k-mers and appends each to its bin's open chunk.
///
/// `config` must already be valid (`process_stream_parallel_with_policy`
/// validates it before choosing a strategy).
fn process_stream_parallel_binned<S: RecordSource>(
    mut reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let progress_interval = config.progress_interval;
    let hpc = config.hpc;
    let canonical = config.canonical;
    let binned_config = BinnedConfig { canonical, ..BinnedConfig::new(k) }.sanitized();

    // Nothing downstream of the warm-up or the main pipeline reads
    // `FastqRecord::id` (see `spawn_producer`'s doc comment for the
    // grep-verified claim); set once, here, before either consumes a
    // record, rather than only inside `spawn_producer`.
    reader.set_keep_ids(false);

    // 0. Warm-up: sample a bounded prefix directly off `reader`, before the
    // producer/worker pool exists, and use it to build a data-adaptive bin
    // map instead of the static hash-to-bin one -- the fix for the R3
    // bin-skew finding (`adaptive_bins.rs`). The sampled records are read
    // via the same `RecordSource` the producer would otherwise have read
    // them from, so nothing is duplicated or skipped: the producer below
    // simply continues from wherever this loop stopped.
    //
    // These sequences are counted for real a few lines down, through the
    // very map they were sampled to build, so a run that happens to be
    // entirely warm-up (an input smaller than `ADAPTIVE_SAMPLE_RECORDS`)
    // still counts every record -- it just never reaches the producer.
    let mut warmup_seqs: Vec<Vec<u8>> = Vec::new();
    let mut warmup_qc = QcSummary::default();
    let mut histogram = SignatureHistogram::new();
    let mut hpc_buf: Vec<u8> = Vec::new();

    for _ in 0..ADAPTIVE_SAMPLE_RECORDS {
        let mut record = FastqRecord::default();
        match reader.next_record_into(&mut record) {
            Ok(true) => {
                warmup_qc.observe_record(&record);
                record.quality_trim_end(min_qual, qual_win);
                let seq: Vec<u8> = if hpc {
                    kmer::homopolymer_compress_into(&record.seq, &mut hpc_buf);
                    hpc_buf.clone()
                } else {
                    record.seq
                };
                histogram.observe_sequence(&seq, k, binned_config.m);
                warmup_seqs.push(seq);
            }
            Ok(false) => break,
            Err(FastqReadError::Io(source_err)) => {
                let (path, _) = failing_location(&reader, source, warmup_seqs.len() as u64);
                return Err(FastDnaError::Io { path, source: source_err });
            }
            Err(FastqReadError::Malformed(reason)) => {
                let (path, record) = failing_location(&reader, source, warmup_seqs.len() as u64);
                return Err(FastDnaError::MalformedFastq { path, record, reason });
            }
        }
    }

    let bin_map = histogram.build_bin_map(binned_config.num_bins);

    // The one piece of shared state, and it is shared by design: worker 3's
    // and worker 5's super-k-mers for bin 17 are disjoint pieces of one
    // store rather than two private copies of the same summary. See
    // `binned.rs` for why that is what removes the `threads * occurrences`
    // term from the memory model.
    let store = BinStore::with_bin_map(binned_config, bin_map);

    // Counts the warm-up sample for real, through the adaptive map it was
    // just used to build, before the main pipeline starts on the rest of
    // the stream.
    let mut warmup_occurrences: u64 = 0;
    {
        let mut warmup_writer = store.writer();
        for seq in &warmup_seqs {
            warmup_occurrences += warmup_writer.push_sequence(&store, seq) as u64;
        }
        warmup_writer.finish(&store);
    }
    let warmup_reads = warmup_seqs.len() as u64;
    drop(warmup_seqs);

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(CHANNEL_DEPTH);
    let (recycle_tx, recycle_rx): (Sender<RecordBatch>, Receiver<RecordBatch>) =
        bounded(recycle_depth(config.num_threads));

    // 1. Producer thread -- the same one the other two strategies use. It
    // continues reading `reader` exactly where the warm-up loop left off.
    let reader_handle = spawn_producer(
        reader,
        sender,
        recycle_rx,
        config.batch_size,
        source.to_path_buf(),
        cancel.clone(),
    );

    let reads_seen = AtomicU64::new(warmup_reads);
    // Counted exactly the way `KmerCounter::insert_batch` counts its own --
    // every instance handed over, before any per-key saturation -- so that
    // `total_kmers()` matches the other two strategies bit for bit and not
    // merely the distinct-k-mer table. Seeded with the warm-up's own
    // occurrences for the same reason `reads_seen` is seeded above.
    let total_occurrences = AtomicU64::new(warmup_occurrences);

    // 2. Parallel consumer pool.
    let results: Vec<BinnedWorkerOutcome> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let mut writer = store.writer();
                let mut local_qc = QcSummary::default();
                // Same "only populated when `hpc` is set" rationale as the
                // other two strategies' worker loops.
                let mut hpc_buf: Vec<u8> = Vec::new();

                while let Ok(mut batch) = receiver.recv() {
                    if let Some(tok) = &cancel {
                        if tok.load(Ordering::Relaxed) {
                            break;
                        }
                    }

                    let n = batch.len() as u64;
                    // Accumulated locally and published once per batch, not
                    // once per record: same reasoning as the disk strategy's
                    // worker loop.
                    let mut batch_occurrences: u64 = 0;

                    for record in &mut batch {
                        local_qc.observe_record(record);
                        record.quality_trim_end(min_qual, qual_win);

                        let seq: &[u8] = if hpc {
                            kmer::homopolymer_compress_into(&record.seq, &mut hpc_buf);
                            &hpc_buf
                        } else {
                            &record.seq
                        };

                        // The binned counterpart of
                        // `extract_canonical_kmers_into` + `insert_batch`.
                        // The returned occurrence count is exactly
                        // `extract_canonical_kmers(seq, k).len()` -- that
                        // equality is the multiset invariant `superkmer.rs`
                        // proves, and it is what keeps `total_kmers()`
                        // identical across strategies (hpc on or off).
                        batch_occurrences += writer.push_sequence(&store, seq) as u64;
                    }

                    total_occurrences.fetch_add(batch_occurrences, Ordering::Relaxed);

                    let _ = recycle_tx.try_send(batch);

                    if let Some(emit) = progress {
                        let prev = reads_seen.fetch_add(n, Ordering::Relaxed);
                        if prev / progress_interval != (prev + n) / progress_interval {
                            emit(Progress::ReadsProcessed(prev + n));
                        }
                    }
                }

                // Publishes this worker's partially filled chunks. Skipping
                // it would silently drop up to `bins * chunk_bytes` of
                // super-k-mers per worker -- the tail of the input -- which
                // is exactly the R2 class of failure, so it is not left to
                // a `Drop` impl that a panic or an early return could make
                // conditional.
                writer.finish(&store);
                local_qc
            }));

            match outcome {
                Ok(qc) => {
                    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
                        while receiver.recv().is_ok() {}
                    }
                    Ok(qc)
                }
                Err(payload) => {
                    // Drain so the bounded channel never backs up and blocks
                    // the producer, even if this was the only worker still
                    // consuming.
                    while receiver.recv().is_ok() {}
                    Err(panic_message(payload))
                }
            }
        })
        .collect();

    let total_reads = warmup_reads
        + reader_handle
            .join()
            .map_err(|_| FastDnaError::Internal { detail: "FASTQ reader thread panicked".to_string() })??;

    let mut worker_panic: Option<String> = None;
    // Starts from the warm-up's own QC rather than `QcSummary::default()`,
    // so its reads are not silently missing from the totals a report
    // shows. `merge` is a plain field-wise sum (`qc.rs`), so starting from
    // any base and merging the rest in is equivalent to any other order.
    let mut master_qc = warmup_qc;
    for outcome in results {
        match outcome {
            Ok(qc) => master_qc.merge(&qc),
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

    if cancel.as_ref().is_some_and(|tok| tok.load(Ordering::Relaxed)) {
        return Err(FastDnaError::Cancelled);
    }

    master_qc.finalize();

    // 3. Phase 2 plus the cross-bin merge. A bin's chunks are freed as soon
    // as they have been expanded, and the merge is streaming, so the only
    // thing that grows here is the final table.
    //
    // Run inside a pool sized to `config.num_threads`, not on rayon's
    // global one. Both halves of `finish` parallelise -- over bins, then
    // over key ranges -- and the global pool is sized by core count, so
    // until this existed `--threads 1` still counted bins on every core.
    // That was not only a flag that did not mean what it said: it made
    // `mem_estimate::estimate_binned_peak_bytes` **under-predict**, because
    // its phase-2 term is `threads * per-bin transients` and the real
    // number of bins in flight ignored `threads` entirely. A memory model
    // that says a run fits when it does not is the one failure that model
    // is calibrated never to commit.
    //
    // Only when the two differ. Building a pool spawns that many threads,
    // and on the default run -- where `num_threads` already *is* the core
    // count -- that is a set of threads created to do exactly what the
    // existing ones would have done. No timing is quoted for the saving:
    // the runs that would have measured it shared the machine with
    // unrelated load, and `docs/BENCHMARKS.md` already records one
    // retracted table taken that way. Falls back to the global pool if a
    // pool cannot be built (thread limits, a sandbox): the run is then no
    // worse than it was before this paragraph existed, which is better than
    // failing a count over it.
    let merged = if config.num_threads == rayon::current_num_threads() {
        store.finish()
    } else {
        match rayon::ThreadPoolBuilder::new().num_threads(config.num_threads).build() {
            Ok(pool) => pool.install(|| store.finish()),
            Err(_) => store.finish(),
        }
    };
    let counter = KmerCounter::from_sorted_entries(merged, total_occurrences.load(Ordering::Relaxed));

    if let Some(emit) = progress {
        if catch_unwind(AssertUnwindSafe(|| emit(Progress::Finished { reads: total_reads }))).is_err() {
            return Err(FastDnaError::Internal { detail: "progress callback panicked".to_string() });
        }
    }

    Ok((counter, master_qc, total_reads))
}

/// Reads a bounded prefix of `reader` and reports how well the binned
/// strategy's bins would balance on it, returning the records so the run can
/// replay them (see [`Prefixed`]).
///
/// The measurement is `DynamicBinMap::predicted_skew` over exactly the
/// histogram `process_stream_parallel_binned`'s own warm-up would build, so
/// this cannot disagree with the map the run then uses. `None` means the
/// sample was empty or had nothing to pack -- an input too small to say
/// anything about, which the caller treats as "no objection".
///
/// Quality trimming and homopolymer compression are applied to the sampled
/// sequences exactly as the counting path would, because both change which
/// minimizers appear; the *records* are returned untouched, so replaying
/// them through the real path trims them once, not twice.
fn sample_bin_balance<S: RecordSource>(
    reader: &mut S,
    config: &PipelineConfig,
) -> Result<(Vec<FastqRecord>, Option<f64>)> {
    let binned_config = BinnedConfig::new(config.k).sanitized();
    let mut histogram = SignatureHistogram::new();
    let mut sampled: Vec<FastqRecord> = Vec::with_capacity(64);
    let mut hpc_buf: Vec<u8> = Vec::new();
    let mut trim_buf = FastqRecord::default();

    for _ in 0..BIN_BALANCE_SAMPLE_RECORDS {
        let mut record = FastqRecord::default();
        match reader.next_record_into(&mut record) {
            Ok(true) => {
                trim_buf.clone_from(&record);
                trim_buf.quality_trim_end(config.min_quality, config.quality_window);
                let seq: &[u8] = if config.hpc {
                    kmer::homopolymer_compress_into(&trim_buf.seq, &mut hpc_buf);
                    &hpc_buf
                } else {
                    &trim_buf.seq
                };
                histogram.observe_sequence(seq, config.k, binned_config.m);
                sampled.push(record);
            }
            Ok(false) => break,
            // A read error here is the run's error: the same record would
            // fail a moment later in the producer, and reporting it now
            // keeps the failure attributable to the file rather than to the
            // sampling.
            Err(FastqReadError::Io(source_err)) => {
                return Err(FastDnaError::Io { path: PathBuf::new(), source: source_err })
            }
            Err(FastqReadError::Malformed(reason)) => {
                return Err(FastDnaError::MalformedFastq {
                    path: PathBuf::new(),
                    record: sampled.len() as u64 + 1,
                    reason,
                })
            }
        }
    }

    let balance = histogram.build_bin_map(binned_config.num_bins).predicted_skew();
    Ok((sampled, balance))
}

/// Where a run goes when the binned strategy is refused: the same choice the
/// automatic chooser would have made if binned had never been a candidate.
fn fallback_from_binned(
    policy: &MemoryPolicy,
    decision: &StrategyDecision,
    threads: usize,
) -> CountStrategy {
    let in_memory_peak =
        mem_estimate::estimate_peak_bytes(decision.estimated_occurrences, threads);
    if policy.estimated_input_bytes.is_some() && in_memory_peak > decision.budget_bytes {
        CountStrategy::Disk
    } else {
        CountStrategy::InMemory
    }
}

/// A record source that yields a buffer of already-read records first, then
/// delegates to the source they came from.
///
/// Exists so the automatic chooser can *look* at the input before committing
/// to a strategy: `process_stream_parallel_with_policy` reads a bounded
/// prefix to measure how well the binned strategy's bins would balance
/// (`bin_balance_of`), and whichever strategy then runs has to see those
/// records too. Buffering them and replaying them is what makes the check
/// free of a second pass -- the alternative, re-opening the input, does not
/// work for a stream (`-`, stdin) at all.
///
/// `current_source` reports the *inner* source's position, which is ahead of
/// the record being replayed while the buffer drains. That only affects
/// which record number an I/O error names, and only for the first
/// `BIN_BALANCE_SAMPLE_RECORDS` records of a run.
struct Prefixed<S> {
    buffered: std::vec::IntoIter<FastqRecord>,
    inner: S,
}

impl<S: RecordSource> RecordSource for Prefixed<S> {
    fn next_record(&mut self) -> crate::fastq::ReadResult<Option<FastqRecord>> {
        match self.buffered.next() {
            Some(record) => Ok(Some(record)),
            None => self.inner.next_record(),
        }
    }

    fn next_record_into(&mut self, record: &mut FastqRecord) -> crate::fastq::ReadResult<bool> {
        match self.buffered.next() {
            Some(buffered) => {
                *record = buffered;
                Ok(true)
            }
            None => self.inner.next_record_into(record),
        }
    }

    fn set_keep_ids(&mut self, keep: bool) {
        self.inner.set_keep_ids(keep);
    }

    fn current_source(&self) -> Option<(PathBuf, u64)> {
        self.inner.current_source()
    }

    fn validate(&self) -> Result<()> {
        self.inner.validate()
    }
}

/// How many records the automatic chooser reads to measure bin balance.
///
/// The same bound `ADAPTIVE_SAMPLE_RECORDS` uses for building the bin map,
/// and for the same reason: it is enough records to see an input's signature
/// composition and small enough that buffering them costs a few megabytes.
/// Deliberately the same constant rather than a second knob -- the sample
/// this measures balance on is the sample the map would be built from.
const BIN_BALANCE_SAMPLE_RECORDS: usize = ADAPTIVE_SAMPLE_RECORDS;

/// The worst bin balance the automatic chooser will accept before refusing
/// the binned strategy.
///
/// Measured, not chosen for roundness. On 2026-09-05, over the first 20,000
/// records of each input (`docs/BENCHMARKS.md`):
///
/// | input | measured balance | binned vs in-memory |
/// |---|---:|---|
/// | 2.14 GB shotgun, 840M occurrences | **1.20x** | 2.9x faster, half the memory |
/// | 392 MB shotgun, 144M occurrences | **1.21x** | 1.4x faster |
/// | 734 MB amplicon-shaped, 330M occurrences | **32.71x** | 3.4x slower, 2.8x more memory |
///
/// The two populations are a factor of 27 apart, so the threshold is not a
/// fine judgement -- anything between about 2 and 25 separates them. 3.0 is
/// close to the good end on purpose: the cost of refusing binned on an
/// input that would have been fine is one run at the old speed, and the
/// cost of accepting it on an input like the amplicon is a run that is
/// slower *and* heavier than every alternative.
/// Thread count from which `Binned` is the right blind choice for an input
/// whose size is unknown -- a stream, a pipe, `--input -`.
///
/// The chooser normally compares both strategies' predicted peaks against
/// the budget, which needs the input's size. A stream has none, so before
/// this constant existed the fallback was `InMemory`: "the strategy that
/// needs no estimate to be safe is the one that should run blind."
///
/// That is exactly backwards, and the models say so. `estimate_peak_bytes`
/// scales its dominant term with **occurrences x threads**;
/// `estimate_binned_peak_bytes` scales its with occurrences alone (only the
/// open-chunk term touches threads, and it is bounded). Sweeping both from
/// 1,000 to 4,000,000,000 occurrences at the crate's own constants, the
/// input size at which `Binned` first becomes the *heavier* of the two is:
///
/// ```text
///  1 thread    191,712,358 occurrences  (~0.5 GB of FASTQ)
///  2 threads   431,352,805 occurrences  (~1.0 GB)
///  4 threads   never, anywhere in the swept range
///  8 threads   never
/// 11 threads   never
/// 16 threads   never
/// ```
///
/// So at four threads or more, `Binned` is predicted lighter than
/// `InMemory` at *every* input size, which is what makes it safe to choose
/// without knowing the size. It is also the faster of the two by a
/// measured margin -- 3.2x on the 390 MB head-to-head, 2.9x on 840M
/// occurrences (`docs/BENCHMARKS.md`) -- so the old fallback was picking
/// the slower and heavier path for every piped run.
///
/// Below four threads the crossover is real and this rule declines to
/// apply, leaving those runs on `InMemory` exactly as before.
///
/// The bin-balance guard is unaffected and still runs: `sample_bin_balance`
/// buffers the records it samples and replays them, so a stream gets the
/// same skew check and the same downgrade a file does.
const BINNED_BLIND_MIN_THREADS: usize = 4;

const MAX_ACCEPTABLE_BIN_SKEW: f64 = 3.0;

/// Streams a FASTQ source and returns its canonical k-mer counts, choosing
/// between FastDNA's counting strategies automatically (or as forced
/// by `policy`) and reporting which one ran.
///
/// This is `process_stream_parallel` plus strategy selection, not a
/// replacement for it: `process_stream_parallel` itself is untouched and
/// keeps behaving exactly as it always has for any caller that does not
/// need the disk strategy (every one of this crate's existing tests calls
/// it directly, unmodified, for exactly that reason). Pass
/// `MemoryPolicy::default()` here to reproduce that same in-memory-only
/// behavior while additionally getting a `CountStrategy` back.
pub fn process_stream_parallel_with_policy<S: RecordSource>(
    mut reader: S,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
    cancel: Option<Arc<AtomicBool>>,
    policy: MemoryPolicy,
) -> Result<(KmerCounter, QcSummary, u64, StrategyDecision)> {
    validate_config(&config)?;
    reader.validate()?;
    let mut decision = resolve_strategy(&policy, &config);

    // The binned strategy is the fastest one this crate has on
    // high-diversity input and the slowest on low-diversity input, and the
    // difference is not visible in anything the chooser above can see:
    // `docs/BENCHMARKS.md` measures 2.9x faster on a shotgun file and 3.4x
    // slower on an amplicon-shaped one that is not small. What separates
    // them is whether the minimizer signatures spread across the bins, so
    // that is measured here -- on a bounded prefix of the real input, before
    // committing -- rather than guessed from byte counts.
    //
    // Only when the chooser picked binned on its own: an explicit
    // `--strategy binned` or `FASTDNA_STRATEGY=binned` means the caller has
    // asked for it by name and is owed the strategy they asked for, not a
    // second opinion.
    let mut prefix: Vec<FastqRecord> = Vec::new();
    if decision.strategy == CountStrategy::Binned && decision.auto_selected {
        let (sampled, balance) = sample_bin_balance(&mut reader, &config)?;
        prefix = sampled;
        match balance {
            Some(skew) if skew > MAX_ACCEPTABLE_BIN_SKEW => {
                decision.strategy = fallback_from_binned(&policy, &decision, config.num_threads);
                decision.bin_balance = Some(skew);
            }
            other => decision.bin_balance = other,
        }
    }

    let reader = Prefixed { buffered: prefix.into_iter(), inner: reader };

    let (counter, qc, total_reads) = match decision.strategy {
        CountStrategy::InMemory => process_stream_parallel(reader, config, source, progress, cancel)?,
        CountStrategy::Disk => process_stream_parallel_disk(reader, config, source, progress, cancel)?,
        CountStrategy::Binned => process_stream_parallel_binned(reader, config, source, progress, cancel)?,
    };

    Ok((counter, qc, total_reads, decision))
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: unwrap/expect denial
// is about production paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::fastq::FastqReader;
    use std::io::Cursor;

    /// The producer refills record slots a recycled batch already owns, so
    /// the failure modes it can have and an allocating one cannot are: a
    /// slot keeping the tail of a longer predecessor, and a final partial
    /// batch shipping the recycled slots it never reached as if they were
    /// real reads. Both would show up as a wrong read count or wrong k-mer
    /// total, so this walks every relationship between the record count and
    /// the batch size with alternating record lengths.
    #[test]
    fn recycled_batches_count_exactly_the_records_that_were_read() {
        const BATCH: usize = 8;
        const K: usize = 5;

        for n_records in [1usize, BATCH - 1, BATCH, BATCH + 1, 2 * BATCH, 3 * BATCH + 3] {
            let mut text = String::new();
            let mut expected_kmers: u64 = 0;
            for i in 0..n_records {
                // A long record followed by a short one: the arrangement
                // that catches a slot whose buffers were not cleared.
                let len = if i % 2 == 0 { 40 } else { 8 };
                let seq = "ACGT".repeat(len / 4);
                // Q40 throughout, so quality trimming is a no-op and the
                // k-mer total stays analytic.
                text.push_str(&format!("@r{i}\n{seq}\n+\n{}\n", "I".repeat(len)));
                expected_kmers += (len - K + 1) as u64;
            }

            let config = PipelineConfig {
                k: K,
                min_quality: 20.0,
                quality_window: 4,
                batch_size: BATCH,
                num_threads: 3,
                progress_interval: 1,
                hpc: false,
                canonical: true,
            };
            let reader = FastqReader::new(Cursor::new(text.into_bytes()));
            let (counter, qc, total_reads) =
                process_stream_parallel(reader, config, Path::new("<memory>"), None, None)
                    .expect("valid input must succeed");

            assert_eq!(total_reads, n_records as u64, "producer read count, n={n_records}");
            assert_eq!(qc.total_reads, n_records as u64, "QC read count, n={n_records}");
            assert_eq!(
                counter.total_kmers(),
                expected_kmers,
                "k-mer total, n={n_records}: a recycled slot leaked or a blank slot was counted"
            );
        }
    }

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

    /// Every variant must have a distinct, stable name: it is what the CLI
    /// banner prints and what a report attributes a run to.
    #[test]
    fn count_strategy_as_str_names_every_variant_distinctly() {
        let names = [
            CountStrategy::InMemory.as_str(),
            CountStrategy::Disk.as_str(),
            CountStrategy::Binned.as_str(),
        ];
        assert_eq!(names, ["in-memory", "disk", "binned"]);
        let unique: std::collections::HashSet<&str> = names.into_iter().collect();
        assert_eq!(unique.len(), names.len(), "two strategies share a name");
    }

    /// The automatic arm's rules, after binned was promoted on 2026-09-05
    /// (`docs/design-minimizer-counting.md` 5 step 6's decision, reversed
    /// once the memory model was calibrated and the bin-balance guard
    /// existed):
    ///
    /// - An input of **unknown size** never gets binned. A stream gives the
    ///   estimator nothing, and the strategy that is safe without an
    ///   estimate is the one that should run blind.
    /// - A sized input gets **binned when its predicted peak fits the
    ///   budget**, because it is measured at 2.9x the in-memory strategy's
    ///   speed and half its peak, with byte-identical output.
    /// - Otherwise the old rule stands: disk if in-memory would not fit,
    ///   in-memory if it would.
    ///
    /// The guard that makes this safe is not in `resolve_strategy` and
    /// cannot be: it needs to look at the input. See
    /// `process_stream_parallel_with_policy` and `MAX_ACCEPTABLE_BIN_SKEW`.
    #[test]
    fn the_automatic_chooser_selects_binned_only_for_a_sized_input_that_fits() {
        for threads in [1usize, 8, 64] {
            let config = PipelineConfig { num_threads: threads, ..PipelineConfig::default() };

            // Unknown size: binned from `BINNED_BLIND_MIN_THREADS` up,
            // in-memory below it, whatever the budget says -- the budget
            // cannot be compared against a peak nothing can estimate.
            //
            // This assertion used to read "never binned, whatever the
            // budget says", which pinned the old fallback. It is inverted
            // rather than deleted because the case still needs pinning:
            // see `BINNED_BLIND_MIN_THREADS` for the sweep showing binned
            // is the lighter of the two at every input size from four
            // threads up, which is what makes the blind choice safe.
            let expected_blind = if threads >= BINNED_BLIND_MIN_THREADS {
                CountStrategy::Binned
            } else {
                CountStrategy::InMemory
            };
            for max_ram in [None, Some(0u64), Some(1), Some(1 << 40)] {
                let policy =
                    MemoryPolicy { strategy: None, max_ram_bytes: max_ram, estimated_input_bytes: None };
                let decision = resolve_strategy(&policy, &config);
                assert_eq!(
                    decision.strategy, expected_blind,
                    "wrong blind choice for an unsized input ({threads} threads, \
                     budget {max_ram:?})"
                );
            }

            // A sized input with room: binned.
            let policy = MemoryPolicy {
                strategy: None,
                max_ram_bytes: Some(64 << 30),
                estimated_input_bytes: Some(2_140_000_000),
            };
            assert_eq!(
                resolve_strategy(&policy, &config).strategy,
                CountStrategy::Binned,
                "auto passed over binned for a sized input with a 64 GiB budget \
                 ({threads} threads)"
            );

            // A sized input with no room for either in-memory or binned:
            // disk, exactly as before.
            let policy = MemoryPolicy {
                strategy: None,
                max_ram_bytes: Some(1),
                estimated_input_bytes: Some(100 << 30),
            };
            assert_eq!(
                resolve_strategy(&policy, &config).strategy,
                CountStrategy::Disk,
                "auto must still fall back to disk when nothing fits ({threads} threads)"
            );
        }
    }

    /// `auto_selected` is what tells
    /// `process_stream_parallel_with_policy` whether it may second-guess the
    /// strategy with the bin-balance check. It must be true only for a
    /// choice the estimate made on its own -- a caller who named a strategy
    /// gets the one they named.
    #[test]
    fn auto_selected_is_true_only_when_nothing_forced_the_strategy() {
        let config = PipelineConfig::default();
        let sized = Some(2_140_000_000u64);

        let automatic = MemoryPolicy {
            strategy: None,
            max_ram_bytes: Some(64 << 30),
            estimated_input_bytes: sized,
        };
        assert!(resolve_strategy(&automatic, &config).auto_selected);

        let forced = MemoryPolicy {
            strategy: Some(CountStrategy::Binned),
            max_ram_bytes: Some(64 << 30),
            estimated_input_bytes: sized,
        };
        assert!(!resolve_strategy(&forced, &config).auto_selected);
    }

    /// Forcing the binned strategy through `policy` must be honoured and
    /// must not be reported as an environment override -- the caller asked
    /// for it directly.
    #[test]
    fn an_explicit_binned_policy_is_honoured_without_claiming_an_env_override() {
        let policy = MemoryPolicy {
            strategy: Some(CountStrategy::Binned),
            max_ram_bytes: Some(1),
            estimated_input_bytes: Some(100 << 30),
        };
        let decision = resolve_strategy(&policy, &PipelineConfig::default());
        assert_eq!(decision.strategy, CountStrategy::Binned);
        assert!(!decision.env_override_applied);
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

    /// A run forced onto `binned` must be reported with the *binned* memory
    /// model's prediction. Reporting the in-memory model's figure -- what
    /// this did before `mem_estimate::estimate_binned_peak_bytes` existed --
    /// told a user who chose `binned` specifically to fit a constrained
    /// machine that the run would cost several times what the design says it
    /// costs.
    ///
    /// Asserted against `estimate_binned_peak_bytes` itself rather than a
    /// hard-coded byte count, so re-calibrating that model (which is
    /// structural today and explicitly wants replacing with a measured one)
    /// does not require editing this test to keep it honest.
    #[test]
    fn a_binned_decision_reports_the_binned_memory_model_not_the_in_memory_one() {
        let input_bytes = 2_299_666_912u64;
        let threads = 8usize;
        let policy = MemoryPolicy {
            strategy: Some(CountStrategy::Binned),
            max_ram_bytes: Some(64 << 30),
            estimated_input_bytes: Some(input_bytes),
        };
        let config = PipelineConfig { num_threads: threads, ..PipelineConfig::default() };
        let decision = resolve_strategy(&policy, &config);

        let binned = BinnedConfig::new(config.k).sanitized();
        let expected = mem_estimate::estimate_binned_peak_bytes(
            decision.estimated_occurrences,
            threads,
            binned.num_bins,
            binned.chunk_bytes,
        );
        assert_eq!(decision.estimated_peak_bytes, expected);

        // And it must actually be the smaller number -- otherwise the whole
        // reason to reach for this strategy would be missing, and the change
        // above would be cosmetic.
        let in_memory = mem_estimate::estimate_peak_bytes(decision.estimated_occurrences, threads);
        assert!(
            decision.estimated_peak_bytes < in_memory,
            "binned predicted {} bytes against in-memory's {in_memory}",
            decision.estimated_peak_bytes
        );
    }

    /// The counterpart guarantee, restated for what it is now: whichever
    /// strategy `auto` reaches, the reported peak is **that strategy's own
    /// model**, never another's.
    ///
    /// Until 2026-09-05 this test asserted something stronger and simpler --
    /// that every automatic decision reported `estimate_peak_bytes`, because
    /// `auto` could only reach the two strategies that model covers. Binned
    /// being promoted is exactly what makes that no longer the right
    /// assertion: a run reported with the in-memory model's figure while the
    /// binned strategy runs would be off by more than the footprint it is
    /// predicting, which is the mistake the `estimated_peak_bytes` match
    /// arm was written to prevent in the first place.
    #[test]
    fn every_automatic_decision_reports_its_own_strategys_model() {
        for input_bytes in [None, Some(0u64), Some(4096), Some(2_140_000_000), Some(100 << 30)] {
            for threads in [1usize, 8, 64] {
                for max_ram in [None, Some(0u64), Some(1 << 40)] {
                    let policy = MemoryPolicy {
                        strategy: None,
                        max_ram_bytes: max_ram,
                        estimated_input_bytes: input_bytes,
                    };
                    let config = PipelineConfig { num_threads: threads, ..PipelineConfig::default() };
                    let decision = resolve_strategy(&policy, &config);
                    let binned_defaults = BinnedConfig::new(config.k).sanitized();
                    let expected = match decision.strategy {
                        CountStrategy::Binned => mem_estimate::estimate_binned_peak_bytes(
                            decision.estimated_occurrences,
                            threads,
                            binned_defaults.num_bins,
                            binned_defaults.chunk_bytes,
                        ),
                        _ => mem_estimate::estimate_peak_bytes(decision.estimated_occurrences, threads),
                    };
                    assert_eq!(
                        decision.estimated_peak_bytes, expected,
                        "the reported peak is not {}'s own model at {input_bytes:?} bytes, \
                         {threads} threads, budget {max_ram:?}",
                        decision.strategy.as_str()
                    );
                }
            }
        }
    }
}
