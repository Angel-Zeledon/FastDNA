// src/disk_spill.rs
//
// The disk-partitioned counting strategy: bucket k-mers by their high bits,
// spill each worker's sorted, deduplicated runs to per-bucket scratch
// files, then merge bucket by bucket so only one bucket's worth of data is
// resident in memory at a time.
//
// This exists because the in-memory strategy (`counter.rs`'s `KmerCounter`)
// has a structural memory ceiling: every worker's private counter converges
// toward holding the *entire* distinct-k-mer table once the input is large
// relative to its cardinality (see `mem_estimate.rs` for the measured
// consequence), so peak memory scales with `threads * distinct_kmers`, not
// `distinct_kmers` alone. This module trades that away for a strategy whose
// peak memory is bounded by one bucket plus the final output table, at the
// cost of touching disk and giving up the in-memory strategy's raw speed.
//
// # Why bucket by high bits, not by a minimizer signature
//
// KMC3 partitions by a minimizer (the lexicographically smallest substring
// of a small window within each k-mer), which balances bucket sizes far
// better than a fixed bit-range split: minimizer distribution reflects
// actual k-mer content, so it adapts to whatever compositional bias a real
// genome has. High-bit bucketing does not adapt -- a genome or amplicon set
// with skewed base composition (extreme GC bias, low-complexity repeats)
// can leave some buckets far larger than others, and this module's "one
// bucket resident at a time" memory bound is only as tight as its largest
// bucket.
//
// High-bit bucketing is used here anyway, deliberately, for two reasons.
// First, it is *trivial to prove correct*: `bucket_of` is a shift and a
// mask, and because canonical k-mers are compared as plain integers, high-
// bit bucketing has the property that bucket 0 through bucket `2^bits - 1`
// covers the k-mer space in ascending order -- concatenating each bucket's
// sorted merge output in bucket order is *already* the final globally
// sorted table, with no additional sort needed. A minimizer scheme loses
// that property (a minimizer's bucket id is not monotonic in the k-mer's
// integer value), so it would need either a final sort pass over the whole
// output (undermining the "only one bucket resident" bound this module
// exists to provide) or a second, separate index to recover order. Second,
// per this task's explicit direction: land a correct, well-tested disk path
// that is slower or less balanced than ideal, rather than a cleverer one
// that cannot be shown correct in the time available. A minimizer-bucketed
// version, with a real balance improvement, is a reasonable future
// iteration on top of this one -- not a rewrite of it, since the spill/merge
// machinery below does not care how `bucket_of` computes its answer.
//
// # On-disk format
//
// Each spilled run is its own file: a flat sequence of `(u64 kmer, u32
// count)` pairs, little-endian, back to back, with no header or length
// prefix -- EOF marks the end of the run. One file per run (rather than
// concatenating a worker's runs for a bucket into a single growing file)
// keeps the merge phase's file handling trivial: open, read sequentially,
// done. It costs more open file descriptors and directory entries during a
// run with many eager flushes; see `merge_buckets` for why that cost is
// judged acceptable here.
//
// The format is fixed and load-bearing: `save`/`load` round-trips through
// it and the cascade's intermediate files are written in it, so the bytes
// may not change even though nothing about them is self-describing.
// `encode_entry`/`decode_entry` are the single definition of those 12
// bytes; every writer and reader in this module goes through them, and
// `run_file_bytes_are_the_flat_little_endian_pairs_the_format_promises`
// pins the exact byte sequence.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{FastDnaError, Result};

/// Bytes one `(u64 kmer, u32 count)` entry occupies on disk: 8 + 4,
/// little-endian, back to back with no padding. Deliberately *not*
/// `size_of::<(u64, u32)>()`, which is 16 because of tuple alignment -- the
/// on-disk format is 12 and must stay 12.
const ENTRY_BYTES: usize = 12;

/// Entries staged in memory between `write` calls on a run file. A fixed
/// byte budget (`8192 * 12` = 96 KiB), not a value tuned to any machine's
/// block size or cache: the point is only that one `write` carries many
/// entries instead of one, and the count it removes is arithmetic, not
/// hardware-dependent. The previous `BufWriter` default (8 KiB) issued one
/// `write` per 682 entries and two `write_all` calls *per entry* on top of
/// it; this issues one `write` per 8192 entries and one `extend_from_slice`
/// per entry. The buffer is allocated once per `SpillWriter` and reused for
/// every run of every flush, so its cost is one allocation per worker.
const WRITE_SLAB_ENTRIES: usize = 8192;
const WRITE_SLAB_BYTES: usize = WRITE_SLAB_ENTRIES * ENTRY_BYTES;

/// `BufReader` capacity for a run file, in whole entries (`4096 * 12` =
/// 48 KiB). Six times the `BufReader` default, which is a deliberate,
/// counted trade rather than a free win: it divides the `read` syscalls a
/// run costs by 6 (a 12 MB run drops from 1536 reads to 256), at the price
/// of 40 KiB more resident per open reader -- at most `MAX_OPEN_RUNS` (64)
/// are open at once, so 2.5 MB extra against a merge that already holds a
/// whole bucket's output table in memory.
const READ_BUF_BYTES: usize = ENTRY_BYTES * 4096;

/// `bucket_bits` used when the caller does not override it. 32 buckets: a
/// starting point, not a value proven optimal by a sweep -- enough that a
/// single bucket typically holds a small fraction of the whole distinct-
/// k-mer table (bounding the "one bucket resident" memory claim to
/// something meaningfully smaller than the in-memory strategy's per-worker
/// table), while keeping the open-file-descriptor and scratch-file count
/// this module produces (`threads * buckets * flushes-per-worker`) well
/// within any real OS's default limits.
pub const DEFAULT_BUCKET_BITS: u32 = 5;

/// Raw k-mers buffered per worker before an eager flush to disk. Same
/// value as `counter.rs::RAW_FINALIZE_THRESHOLD` and for the same reason
/// (see that constant's doc comment): bounds a worker's memory footprint
/// to a fixed multiple of this, independent of how large the input is,
/// rather than growing for the whole run.
pub const SPILL_RAW_THRESHOLD: usize = 2_000_000;

/// Maps a canonical k-mer to one of `2^bucket_bits` buckets, using its high
/// bits within the `2*k`-bit range 2-bit packing actually uses (not the
/// full 64-bit word, most of which is always zero for `k < 32`). See the
/// module doc comment for why high bits, not a minimizer, and why this
/// specific choice keeps bucket order and k-mer order aligned.
///
/// Handles `k` small enough that `2*k < bucket_bits` (every valid `k` in
/// `1..=32}` against `bucket_bits` up to 32) without panicking: the shift
/// saturates to zero rather than underflowing, and the result is still
/// masked to `bucket_bits` width, so a tiny `k` simply uses fewer of the
/// available buckets rather than causing a shift-amount overflow.
pub fn bucket_of(kmer: u64, k: usize, bucket_bits: u32) -> usize {
    let used_bits = (2 * k) as u32;
    let shift = used_bits.saturating_sub(bucket_bits);
    let mask: u64 = if bucket_bits >= 64 { u64::MAX } else { (1u64 << bucket_bits) - 1 };
    ((kmer >> shift) & mask) as usize
}

/// A uniquely-named scratch directory that is recursively removed when it
/// is dropped -- including when a worker thread panics (this crate builds
/// with `panic = "unwind"` specifically so destructors like this one still
/// run during unwind; see `Cargo.toml`'s `[profile.release]` comment) and
/// when a caller's `?` returns early from any function still holding this
/// value. Cleanup is best-effort: `Drop` has no error channel, and a
/// failure to remove scratch files (permissions, another process holding a
/// handle) is not worth panicking over during an unwind that may already be
/// in progress.
pub struct ScratchDir {
    root: PathBuf,
}

/// Disambiguates scratch directories created within the same process
/// (`std::process::id()` alone is not unique across, e.g., parallel test
/// threads in one `cargo test` binary, or two `ScratchDir`s created in
/// quick succession within one run).
static SCRATCH_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

impl ScratchDir {
    /// Creates a new, empty scratch directory. Location: the
    /// `FASTDNA_SPILL_DIR` environment variable if set, else the OS temp
    /// directory. The override exists because the OS temp dir is often a
    /// small system drive (especially on Windows), while the disk strategy
    /// spills exactly when the input is large -- the user needs a way to
    /// point scratch space at the drive that actually has room.
    pub fn new() -> Result<Self> {
        let unique = SCRATCH_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let base = std::env::var_os("FASTDNA_SPILL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let root = base.join(format!("fastdna-spill-{}-{unique}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&root).map_err(|e| FastDnaError::Io { path: root.clone(), source: e })?;
        Ok(Self { root })
    }

    /// Path for one worker's one run within one bucket. `run_index` is that
    /// (worker, bucket) pair's own monotonically increasing counter (see
    /// `SpillWriter`), not a global one -- it only needs to be unique
    /// within this one directory for this one (worker, bucket) pair.
    fn run_path(&self, worker: usize, bucket: usize, run_index: usize) -> PathBuf {
        self.root.join(format!("w{worker}_b{bucket}_r{run_index}.bin"))
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn io_err(path: &Path, source: io::Error) -> FastDnaError {
    FastDnaError::Io { path: path.to_path_buf(), source }
}

/// Serializes one entry into the exact 12 bytes the on-disk format calls
/// for -- byte-for-byte what `kmer.to_le_bytes()` followed by
/// `count.to_le_bytes()` produced before, just staged in one array so the
/// caller appends once instead of twice.
///
/// The `to_le_bytes`/`from_le_bytes` conversions are kept rather than
/// replaced by a `#[repr(C, packed)]` struct copy: on a little-endian
/// target they already compile to a plain store with no byte shuffling, so
/// the operation count removed would be zero, while a struct copy would
/// need `unsafe` and would silently write the wrong bytes on a big-endian
/// target. That is a trade with no counted benefit, so it is not taken.
#[inline]
fn encode_entry(kmer: u64, count: u32) -> [u8; ENTRY_BYTES] {
    let mut entry = [0u8; ENTRY_BYTES];
    let (kmer_bytes, count_bytes) = entry.split_at_mut(8);
    kmer_bytes.copy_from_slice(&kmer.to_le_bytes());
    count_bytes.copy_from_slice(&count.to_le_bytes());
    entry
}

#[inline]
fn decode_entry(entry: &[u8; ENTRY_BYTES]) -> (u64, u32) {
    let mut kmer_bytes = [0u8; 8];
    let mut count_bytes = [0u8; 4];
    kmer_bytes.copy_from_slice(&entry[..8]);
    count_bytes.copy_from_slice(&entry[8..]);
    (u64::from_le_bytes(kmer_bytes), u32::from_le_bytes(count_bytes))
}

/// A run file being written, fed one entry at a time through a caller-owned
/// staging slab.
///
/// The slab is passed in rather than owned so a single buffer serves every
/// run a `SpillWriter` produces (and every cascade intermediate one merge
/// produces): the write path allocates once per worker, not once per run
/// per flush.
///
/// This replaces a `BufWriter` fed two `write_all` calls per entry. Per
/// entry that was: two `Result`-returning calls, two remaining-capacity
/// comparisons, two error-mapping closures and two copies into the
/// `BufWriter`'s own buffer, which was then copied again to the file. Now
/// it is one 12-byte `extend_from_slice` into the slab and one length
/// comparison, with the slab handed to `write_all` directly -- one copy of
/// each byte instead of two.
struct RunSink {
    file: File,
    path: PathBuf,
}

impl RunSink {
    fn create(path: PathBuf) -> Result<Self> {
        let file = File::create(&path).map_err(|e| io_err(&path, e))?;
        Ok(Self { file, path })
    }

    #[inline]
    fn push(&mut self, slab: &mut Vec<u8>, kmer: u64, count: u32) -> Result<()> {
        slab.extend_from_slice(&encode_entry(kmer, count));
        if slab.len() >= WRITE_SLAB_BYTES {
            self.drain(slab)?;
        }
        Ok(())
    }

    fn drain(&mut self, slab: &mut Vec<u8>) -> Result<()> {
        if !slab.is_empty() {
            self.file.write_all(slab).map_err(|e| io_err(&self.path, e))?;
            slab.clear();
        }
        Ok(())
    }

    /// Pushes the tail of the slab out. There is no `flush` to follow:
    /// `write_all` on a `File` goes straight to the OS, so the explicit
    /// `BufWriter::flush` this replaces is gone along with the buffer.
    fn finish(&mut self, slab: &mut Vec<u8>) -> Result<()> {
        self.drain(slab)
    }
}

/// Writes one already-sorted, already-deduplicated run to `path` as flat
/// little-endian `(u64, u32)` pairs.
///
/// Test-only: the production write path (`SpillWriter::flush_sorted` and
/// the cascade in `merge_bucket_bounded`) streams straight into a `RunSink`
/// without ever materializing a run as a slice. This is expressed in terms
/// of the same `RunSink`, so the two cannot drift on the byte format.
#[cfg(test)]
fn write_run(path: &Path, run: &[(u64, u32)]) -> Result<()> {
    let mut slab: Vec<u8> = Vec::with_capacity(WRITE_SLAB_BYTES);
    let mut sink = RunSink::create(path.to_path_buf())?;
    for &(kmer, count) in run {
        sink.push(&mut slab, kmer, count)?;
    }
    sink.finish(&mut slab)
}

/// Streams `(u64, u32)` pairs out of one run file, one at a time, so a
/// merge over many runs holds only a small buffered read window per run in
/// memory rather than each run's full contents.
struct RunReader {
    reader: BufReader<File>,
    path: PathBuf,
}

impl RunReader {
    fn open(path: PathBuf) -> Result<Self> {
        let file = File::open(&path).map_err(|e| io_err(&path, e))?;
        Ok(Self { reader: BufReader::with_capacity(READ_BUF_BYTES, file), path })
    }

    /// Returns the next entry, or `None` at a clean end-of-run. A partial
    /// read (EOF strictly inside one pair) can only mean this module's own
    /// writer produced a truncated file -- a bug, not a data problem -- so
    /// it is reported as `FastDnaError::Internal` rather than silently
    /// treated as end-of-run.
    ///
    /// One `fill_buf` + `consume` pair per entry, replacing two `read_exact`
    /// calls (each of which is itself a loop over `read`, with its own
    /// `UnexpectedEof` branch and, for the second, an error-mapping closure
    /// built per entry). The entry is copied once, out of the reader's own
    /// buffer; the two-`read_exact` form copied 8 bytes and then 4 through
    /// the same path. Per merge that is `2N` calls down to `N` over every
    /// entry in every run file -- and every k-mer occurrence passes through
    /// a run file at least once, twice more for each cascade pass.
    fn next(&mut self) -> Result<Option<(u64, u32)>> {
        let mut entry = [0u8; ENTRY_BYTES];
        let filled;
        {
            let available = self.reader.fill_buf().map_err(|e| io_err(&self.path, e))?;
            filled = available.len().min(ENTRY_BYTES);
            entry[..filled].copy_from_slice(&available[..filled]);
        }
        self.reader.consume(filled);

        if filled == ENTRY_BYTES {
            return Ok(Some(decode_entry(&entry)));
        }
        if filled == 0 {
            return Ok(None);
        }
        // An entry straddling the read buffer's boundary. With a capacity
        // that is a whole number of entries this only happens when the
        // underlying `read` returned a short count, so it is rare enough to
        // keep off the hot path entirely.
        self.finish_straddling_entry(entry, filled)
    }

    #[cold]
    fn finish_straddling_entry(
        &mut self,
        mut entry: [u8; ENTRY_BYTES],
        mut filled: usize,
    ) -> Result<Option<(u64, u32)>> {
        while filled < ENTRY_BYTES {
            let taken;
            {
                let available = self.reader.fill_buf().map_err(|e| io_err(&self.path, e))?;
                if available.is_empty() {
                    return Err(FastDnaError::Internal {
                        detail: format!(
                            "spill run file {} ended mid-entry -- truncated scratch file",
                            self.path.display()
                        ),
                    });
                }
                taken = available.len().min(ENTRY_BYTES - filled);
                entry[filled..filled + taken].copy_from_slice(&available[..taken]);
            }
            self.reader.consume(taken);
            filled += taken;
        }
        Ok(Some(decode_entry(&entry)))
    }
}

/// Per-worker accumulator for the disk strategy: buffers raw k-mer
/// instances (bounded by `SPILL_RAW_THRESHOLD`, exactly like `counter.rs`'s
/// `KmerCounter` bounds its own raw buffer, and for the same reason -- see
/// that module's `RAW_FINALIZE_THRESHOLD` doc comment), and on crossing the
/// threshold, sorts the whole buffer once and does two things in the same
/// linear scan: collapses runs of equal k-mers into `(kmer, count)` pairs
/// (identical algorithm to `counter.rs::compact_raw`, reimplemented here
/// rather than shared because it operates on this module's per-bucket
/// output instead of `counter.rs`'s private `Inner`), and routes each pair
/// into its bucket's next run file. Sorting the whole buffer once, rather
/// than bucketing first and sorting each bucket's slice separately, works
/// because ascending k-mer order and ascending bucket order coincide for
/// high-bit bucketing (see the module doc comment) -- a single sorted scan
/// visits buckets in order for free.
pub struct SpillWriter<'a> {
    scratch: &'a ScratchDir,
    worker: usize,
    k: usize,
    bucket_bits: u32,
    raw: Vec<u64>,
    /// Staging buffer for the write path, allocated once here and reused by
    /// every run of every flush -- see `WRITE_SLAB_BYTES`.
    slab: Vec<u8>,
    next_run_index: Vec<usize>,
    manifest: Vec<Vec<PathBuf>>,
}

impl<'a> SpillWriter<'a> {
    pub fn new(scratch: &'a ScratchDir, worker: usize, k: usize, bucket_bits: u32) -> Self {
        let num_buckets = 1usize << bucket_bits;
        Self {
            scratch,
            worker,
            k,
            bucket_bits,
            raw: Vec::with_capacity(SPILL_RAW_THRESHOLD),
            slab: Vec::with_capacity(WRITE_SLAB_BYTES),
            next_run_index: vec![0; num_buckets],
            manifest: vec![Vec::new(); num_buckets],
        }
    }

    /// Buffers `kmers`, eagerly flushing to disk once the buffer crosses
    /// `SPILL_RAW_THRESHOLD` -- mirrors `KmerCounter::insert_batch`'s own
    /// eager-flush cadence.
    pub fn insert_batch(&mut self, kmers: &[u64]) -> Result<()> {
        self.raw.extend_from_slice(kmers);
        if self.raw.len() >= SPILL_RAW_THRESHOLD {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.raw.is_empty() {
            return Ok(());
        }

        self.raw.sort_unstable();

        // Both buffers are moved out of `self` so `flush_sorted` can borrow
        // `self.next_run_index` and `self.manifest` mutably while it scans,
        // then moved back with their allocations intact -- `mem::take`
        // leaves an empty `Vec` behind, and the originals are restored
        // (cleared, capacity kept) on the error path as well as the success
        // path, so a failed flush costs no reallocation on the next one.
        let mut raw = std::mem::take(&mut self.raw);
        let mut slab = std::mem::take(&mut self.slab);

        let result = self.flush_sorted(&raw, &mut slab);

        raw.clear();
        self.raw = raw;
        slab.clear();
        self.slab = slab;
        result
    }

    /// Writes the already-sorted `raw` buffer out as one run file per
    /// bucket it touches.
    ///
    /// This streams: no `Vec<Vec<(u64, u32)>>` staging area is built. The
    /// previous form allocated `num_buckets` fresh `Vec`s per flush (32 with
    /// the default `bucket_bits`, plus every doubling realloc as each grew),
    /// pushed one 16-byte tuple per distinct k-mer into them, then read all
    /// of that back out to serialize it -- so every distinct k-mer's bytes
    /// were written to memory twice before reaching the file, and each
    /// flush's per-bucket vectors were dropped immediately afterwards.
    /// Streaming removes, per flush: `num_buckets` allocations, every
    /// growth realloc (a `Vec` doubling to `n` entries copies ~`n` entries
    /// in total, so ~24 MB of `memcpy` per full 2,000,000-k-mer flush), and
    /// one whole pass over the flush's distinct entries. With one flush per
    /// `SPILL_RAW_THRESHOLD` k-mers, a run over ~1.9e9 k-mer occurrences
    /// does ~950 flushes: ~30,000 vector allocations and ~23 GB of buffer
    /// copying that no longer happen.
    ///
    /// Correctness rests on the module's bucket-order invariant: `raw` is
    /// sorted and `bucket_of` never decreases as the k-mer value rises (see
    /// the module doc comment and
    /// `bucket_of_is_monotonic_in_kmer_value_for_a_fixed_k`), so an
    /// ascending scan enters each bucket once and leaves it for good. Note
    /// that a violation would cost extra run files, not wrong counts: each
    /// file is still written in ascending k-mer order, and `merge_buckets`
    /// already folds a key appearing in several runs.
    fn flush_sorted(&mut self, raw: &[u64], slab: &mut Vec<u8>) -> Result<()> {
        // `bucket_of`'s shift and mask depend only on `k` and `bucket_bits`,
        // both fixed for this writer's entire lifetime. Hoisting them out
        // removes a multiply, a saturating subtract, a compare and a shifted
        // mask construction from every distinct k-mer -- ~53.8 million per
        // full-size flush's worth of output, times every flush in the run.
        let shift = ((2 * self.k) as u32).saturating_sub(self.bucket_bits);
        let mask: u64 =
            if self.bucket_bits >= 64 { u64::MAX } else { (1u64 << self.bucket_bits) - 1 };

        // `usize::MAX` cannot be a real bucket id (`num_buckets` is
        // `1 << bucket_bits` with `bucket_bits <= 32`), so it is a safe
        // "nothing open yet" sentinel and the bucket-change test stays a
        // single integer compare.
        let mut open_bucket = usize::MAX;
        let mut sink: Option<RunSink> = None;

        // `chunk_by` over the sorted buffer yields each run of equal k-mers
        // in one step. The previous `drain(..).peekable()` form cost, per
        // raw k-mer, a `peek` (an `Option<&u64>` build and compare) *plus* a
        // `next`, and a `saturating_add` per duplicate; this costs one
        // comparison per raw k-mer and one length read per distinct k-mer.
        // Raw k-mers are the largest count in this module -- one per k-mer
        // occurrence in the input, ~1.9e9 on a 2 GB FASTQ.
        for group in raw.chunk_by(|a, b| a == b) {
            let Some(&kmer) = group.first() else { continue };
            let count = u32::try_from(group.len()).unwrap_or(u32::MAX);
            let bucket = ((kmer >> shift) & mask) as usize;

            if bucket != open_bucket {
                if let Some(mut done) = sink.take() {
                    done.finish(slab)?;
                }
                let run_index = self.next_run_index[bucket];
                self.next_run_index[bucket] = run_index + 1;
                let path = self.scratch.run_path(self.worker, bucket, run_index);
                sink = Some(RunSink::create(path.clone())?);
                self.manifest[bucket].push(path);
                open_bucket = bucket;
            }

            if let Some(active) = sink.as_mut() {
                active.push(slab, kmer, count)?;
            }
        }

        if let Some(mut done) = sink {
            done.finish(slab)?;
        }

        Ok(())
    }

    /// Flushes any remaining buffered k-mers and returns the manifest of
    /// every run file this worker wrote, indexed by bucket -- what
    /// `merge_buckets` needs to find this worker's contribution to each
    /// bucket.
    pub fn finish(mut self) -> Result<Vec<Vec<PathBuf>>> {
        self.flush()?;
        Ok(self.manifest)
    }
}

/// Maximum run files opened simultaneously during any one merge pass. A
/// bucket can accumulate one run per worker per eager flush --
/// `occurrences / SPILL_RAW_THRESHOLD` in total for real data, over a
/// thousand for a single-digit-GB FASTQ -- so opening every run at once
/// blows through `ulimit -n` (1024 by default on Linux) on exactly the
/// inputs the disk strategy exists for. Merges over more runs than this
/// cascade: groups of at most this many runs are streamed into
/// intermediate run files, repeatedly, until one final pass fits. 64 keeps
/// pass count low (a 4096-run bucket needs just two passes) while staying
/// far under any real descriptor limit even with the input FASTQ, the
/// output file, and a handful of library descriptors also open.
pub const MAX_OPEN_RUNS: usize = 64;

/// Merges one bucket's run files (across every worker that wrote to it)
/// into `out`, in ascending k-mer order, via a streaming k-way merge over a
/// binary min-heap -- the same algorithm as
/// `counter.rs::k_way_merge_sorted_counts`, adapted to pull from
/// `RunReader`s instead of in-memory slices so no single run (let alone the
/// whole bucket) needs to be fully resident to be merged.
fn merge_bucket_sources(sources: &mut [RunReader], out: &mut Vec<(u64, u32)>) -> Result<()> {
    merge_sources_into(sources, |kmer, count| {
        out.push((kmer, count));
        Ok(())
    })
}

/// The streaming k-way merge itself, emitting each merged `(kmer, count)`
/// pair to `emit` instead of materializing anywhere -- the cascade path
/// writes pairs straight to an intermediate run file, and buffering a
/// 64-run group in memory first would reintroduce the unbounded-memory
/// problem this module exists to avoid.
fn merge_sources_into(
    sources: &mut [RunReader],
    mut emit: impl FnMut(u64, u32) -> Result<()>,
) -> Result<()> {
    // The count belonging to whichever entry each source currently has *in
    // the heap*. Exhaustion is already encoded by a source's absence from
    // the heap, so this needs neither an `Option` nor a copy of the k-mer
    // (the heap holds it): 4 bytes per source instead of the 24 an
    // `Option<(u64, u32)>` occupies, and one `u32` load per popped entry
    // instead of a discriminant test, a 16-byte tuple read, a `map` and an
    // `unwrap_or`. That is per emitted entry *and* per folded duplicate --
    // once for every entry in every run file the merge touches.
    let mut pending: Vec<u32> = vec![0; sources.len()];
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(sources.len());

    for (idx, source) in sources.iter_mut().enumerate() {
        if let Some((kmer, count)) = source.next()? {
            pending[idx] = count;
            heap.push(Reverse((kmer, idx)));
        }
    }

    while let Some(Reverse((kmer, idx))) = heap.pop() {
        let mut count = pending[idx];

        if let Some((next_kmer, next_count)) = sources[idx].next()? {
            pending[idx] = next_count;
            heap.push(Reverse((next_kmer, idx)));
        }

        // Fold in every other source currently sitting at the same key --
        // sources can share keys (a k-mer seen by two different workers, or
        // by the same worker across two different flushes), but never
        // within one source, since each run is already deduplicated. (That
        // is also why the entry just pushed for `idx` can never match
        // `kmer` and does not need re-testing.)
        //
        // The peek destructures the heap's root in one step and the pop
        // that follows is unconditional. The previous form tested the root
        // with `matches!` and then re-tested the `Option` that `pop`
        // returned, spending two `Option` matches per folded duplicate and
        // one wasted `matches!` per emitted entry where one test suffices.
        while let Some(&Reverse((peek_kmer, other_idx))) = heap.peek() {
            if peek_kmer != kmer {
                break;
            }
            let _ = heap.pop();
            count = count.saturating_add(pending[other_idx]);
            if let Some((next_kmer, next_count)) = sources[other_idx].next()? {
                pending[other_idx] = next_count;
                heap.push(Reverse((next_kmer, other_idx)));
            }
        }

        emit(kmer, count)?;
    }

    Ok(())
}

/// Merges every worker's spilled runs into one globally sorted,
/// deduplicated `(kmer, count)` table -- the disk strategy's counterpart to
/// `KmerCounter::finalized`.
///
/// `manifests[worker][bucket]` is that worker's run files for that bucket,
/// as returned by `SpillWriter::finish`. Buckets are processed strictly in
/// ascending order, one at a time: only the current bucket's run readers
/// (each holding a small buffered read window, not the whole run -- see
/// `RunReader`) and the already-accumulated output vector are alive at any
/// point, which is the "only one bucket resident" memory bound this whole
/// module exists to provide. Because high-bit bucketing keeps ascending
/// bucket order aligned with ascending k-mer order (see the module doc
/// comment), appending each bucket's merged output to `out` in bucket order
/// produces a globally sorted table with no separate final sort pass.
pub fn merge_buckets(manifests: &[Vec<Vec<PathBuf>>], num_buckets: usize) -> Result<Vec<(u64, u32)>> {
    let mut out = Vec::new();

    for bucket in 0..num_buckets {
        // Sized in one pass over the (per-worker, so single-digit-length)
        // manifest list before collecting, so the collect below never
        // reallocates. A bucket holds one run per worker per flush --
        // ~950 paths on a 2 GB input -- which the previous `Vec::new()`
        // reached through ~10 doublings, each moving every `PathBuf` header
        // already in the vector.
        let total_paths: usize =
            manifests.iter().map(|m| m.get(bucket).map_or(0, Vec::len)).sum();
        if total_paths == 0 {
            continue;
        }
        let mut paths: Vec<PathBuf> = Vec::with_capacity(total_paths);
        for worker_manifest in manifests {
            if let Some(worker_paths) = worker_manifest.get(bucket) {
                paths.extend(worker_paths.iter().cloned());
            }
        }
        merge_bucket_bounded(paths, bucket, &mut out)?;
    }

    Ok(out)
}

/// Merges one bucket's runs into `out` while never holding more than
/// `MAX_OPEN_RUNS` files open at once. Oversized merges cascade: each pass
/// streams groups of at most `MAX_OPEN_RUNS` runs into intermediate run
/// files (written beside the originals, inside the same scratch directory,
/// so `ScratchDir`'s drop still cleans them up on any failure), until one
/// final pass fits. Intermediates from a finished pass are deleted eagerly
/// -- a cascade would otherwise briefly double the bucket's disk footprint
/// pass after pass; the original run files are left for `ScratchDir`.
fn merge_bucket_bounded(paths: Vec<PathBuf>, bucket: usize, out: &mut Vec<(u64, u32)>) -> Result<()> {
    let mut paths = paths;
    let mut pass = 0usize;
    // One staging slab for every intermediate this cascade writes, on the
    // same terms as `SpillWriter`'s: allocated at most once per bucket
    // instead of once per group per pass, and it replaces a `BufWriter` fed
    // two `write_all` calls per entry (see `RunSink`).
    let mut slab: Vec<u8> = Vec::new();

    while paths.len() > MAX_OPEN_RUNS {
        if slab.capacity() == 0 {
            slab.reserve_exact(WRITE_SLAB_BYTES);
        }
        let mut next_paths: Vec<PathBuf> = Vec::with_capacity(paths.len() / MAX_OPEN_RUNS + 1);
        for (group_index, group) in paths.chunks(MAX_OPEN_RUNS).enumerate() {
            // A one-file group needs no merging; pass it through untouched.
            if group.len() == 1 {
                next_paths.push(group[0].clone());
                continue;
            }
            let parent = group[0].parent().unwrap_or_else(|| Path::new("."));
            let intermediate = parent.join(format!("cascade_b{bucket}_p{pass}_g{group_index}.bin"));

            let mut sources = open_sources(group)?;
            let mut sink = RunSink::create(intermediate.clone())?;
            merge_sources_into(&mut sources, |kmer, count| sink.push(&mut slab, kmer, count))?;
            sink.finish(&mut slab)?;
            drop(sink);
            drop(sources);

            // Inputs that were themselves intermediates are consumed now.
            if pass > 0 {
                for path in group {
                    let _ = std::fs::remove_file(path);
                }
            }
            next_paths.push(intermediate);
        }
        paths = next_paths;
        pass += 1;
    }

    let mut sources = open_sources(&paths)?;
    merge_bucket_sources(&mut sources, out)?;
    drop(sources);
    if pass > 0 {
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(())
}

fn open_sources(paths: &[PathBuf]) -> Result<Vec<RunReader>> {
    paths.iter().map(|path| RunReader::open(path.clone())).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bucket_of_is_monotonic_in_kmer_value_for_a_fixed_k() {
        // The whole point of high-bit bucketing (see the module doc
        // comment): ascending k-mer value must never produce a *smaller*
        // bucket id, or bucket-order concatenation would not be sorted.
        let k = 16;
        let bits = 5;
        let mut prev_bucket = 0usize;
        let mut prev_kmer = 0u64;
        for kmer in (0u64..(1u64 << (2 * k))).step_by(997) {
            let bucket = bucket_of(kmer, k, bits);
            assert!(
                bucket >= prev_bucket,
                "kmer {kmer} (bucket {bucket}) came after {prev_kmer} (bucket {prev_bucket}) but sorted lower"
            );
            prev_bucket = bucket;
            prev_kmer = kmer;
        }
    }

    #[test]
    fn bucket_of_never_panics_for_tiny_k_against_default_bucket_bits() {
        for k in 1..=32usize {
            for bits in [1u32, 5, 6, 32] {
                let _ = bucket_of(0, k, bits);
                let _ = bucket_of(u64::MAX, k, bits);
            }
        }
    }

    #[test]
    fn scratch_dir_is_created_and_removed_on_drop() {
        let root;
        {
            let scratch = ScratchDir::new().expect("scratch dir creation must succeed");
            root = scratch.root.clone();
            assert!(root.is_dir(), "scratch directory must exist while the guard is alive");
        }
        assert!(!root.exists(), "scratch directory must be gone once the guard drops");
    }

    #[test]
    fn two_scratch_dirs_created_back_to_back_do_not_collide() {
        let a = ScratchDir::new().expect("first scratch dir");
        let b = ScratchDir::new().expect("second scratch dir");
        assert_ne!(a.root, b.root, "concurrent scratch dirs must not share a path");
    }

    #[test]
    fn spill_writer_roundtrips_counts_through_a_single_flush() {
        let scratch = ScratchDir::new().unwrap();
        let k = 8;
        let mut writer = SpillWriter::new(&scratch, 0, k, 4);

        // Below `SPILL_RAW_THRESHOLD`, so `finish` is what triggers the
        // only flush -- this exercises the "flush on finish" path
        // specifically, not the eager mid-run one.
        let kmers = [1u64, 2, 2, 3, 3, 3, 100, 100];
        writer.insert_batch(&kmers).unwrap();
        let manifest = writer.finish().unwrap();

        let merged = merge_buckets(&[manifest], 1usize << 4).unwrap();
        assert_eq!(merged, vec![(1, 1), (2, 2), (3, 3), (100, 2)]);
    }

    #[test]
    fn spill_writer_across_multiple_eager_flushes_still_merges_correctly() {
        let scratch = ScratchDir::new().unwrap();
        let k = 10;
        let mut writer = SpillWriter::new(&scratch, 0, k, 5);

        // Three times over the raw threshold, so multiple eager flushes
        // fire before `finish`'s own trailing flush -- the seam where a
        // bug in per-flush run boundaries (rather than a single in-memory
        // buffer) would surface. Small alphabet so counts recur across
        // flush boundaries, the same trick `counter.rs`'s own tests use.
        let total = SPILL_RAW_THRESHOLD * 3;
        let kmers: Vec<u64> = (0..total as u64).map(|i| i % 5).collect();
        for chunk in kmers.chunks(97_331) {
            writer.insert_batch(chunk).unwrap();
        }
        let manifest = writer.finish().unwrap();

        let merged = merge_buckets(&[manifest], 1usize << 5).unwrap();
        assert_eq!(merged.len(), 5, "exactly 5 distinct kmers (0..5) across the whole run");
        let expected_count = (total / 5) as u32;
        for &(kmer, count) in &merged {
            assert!(kmer < 5);
            assert_eq!(count, expected_count, "kmer {kmer} count across flush boundaries");
        }
        // Bucket-order concatenation must already be globally sorted.
        let mut sorted = merged.clone();
        sorted.sort_unstable_by_key(|&(k, _)| k);
        assert_eq!(merged, sorted, "merge_buckets output must already be in ascending kmer order");
    }

    #[test]
    fn merge_buckets_combines_counts_across_multiple_workers() {
        let scratch = ScratchDir::new().unwrap();
        let k = 6;
        let bits = 3;

        let mut w0 = SpillWriter::new(&scratch, 0, k, bits);
        w0.insert_batch(&[5u64, 5, 9]).unwrap();
        let m0 = w0.finish().unwrap();

        let mut w1 = SpillWriter::new(&scratch, 1, k, bits);
        w1.insert_batch(&[5u64, 20, 20, 20]).unwrap();
        let m1 = w1.finish().unwrap();

        let merged = merge_buckets(&[m0, m1], 1usize << bits).unwrap();
        assert_eq!(merged, vec![(5, 3), (9, 1), (20, 3)]);
    }

    #[test]
    fn merge_buckets_of_no_manifests_is_empty() {
        let merged = merge_buckets(&[], 1usize << 5).unwrap();
        assert!(merged.is_empty());
    }

    /// One bucket accumulating more run files than `MAX_OPEN_RUNS` must
    /// still merge correctly -- via cascaded passes -- instead of opening
    /// every run simultaneously, which blows through `ulimit -n` on the
    /// very inputs the disk strategy exists for. 200 runs each holding
    /// overlapping keys checks both the cascade plumbing and that counts
    /// survive being merged twice.
    #[test]
    fn a_bucket_with_more_runs_than_the_open_file_bound_merges_correctly() {
        let scratch = ScratchDir::new().unwrap();
        let num_runs = MAX_OPEN_RUNS * 3 + 7;
        let mut manifest: Vec<Vec<PathBuf>> = vec![Vec::new(); 1];

        for run_index in 0..num_runs {
            let path = scratch.run_path(0, 0, run_index);
            // Every run holds kmers 0..10; run `i` also holds `1000 + i`.
            let mut run: Vec<(u64, u32)> = (0u64..10).map(|kmer| (kmer, 2)).collect();
            run.push((1_000 + run_index as u64, 5));
            write_run(&path, &run).unwrap();
            manifest[0].push(path);
        }

        let merged = merge_buckets(&[manifest], 1).unwrap();

        assert_eq!(merged.len(), 10 + num_runs);
        for &(kmer, count) in merged.iter().take(10) {
            assert!(kmer < 10);
            assert_eq!(count, 2 * num_runs as u32, "kmer {kmer} appears twice in every run");
        }
        for (offset, &(kmer, count)) in merged.iter().skip(10).enumerate() {
            assert_eq!(kmer, 1_000 + offset as u64);
            assert_eq!(count, 5);
        }
        let mut sorted = merged.clone();
        sorted.sort_unstable_by_key(|&(k, _)| k);
        assert_eq!(merged, sorted, "cascaded output must still be globally sorted");
    }

    /// `FASTDNA_SPILL_DIR` redirects scratch space away from the OS temp
    /// directory -- often a small system drive on Windows -- to wherever
    /// the user actually has room.
    #[test]
    fn spill_dir_env_var_overrides_the_os_temp_directory() {
        let base = std::env::temp_dir().join("fastdna_custom_spill_test");
        std::fs::create_dir_all(&base).unwrap();
        // No other test in this crate reads FASTDNA_SPILL_DIR concurrently;
        // ScratchDir::new snapshots it synchronously before this removes it.
        std::env::set_var("FASTDNA_SPILL_DIR", &base);
        let scratch = ScratchDir::new();
        std::env::remove_var("FASTDNA_SPILL_DIR");

        let scratch = scratch.expect("scratch dir under the override must be created");
        assert!(
            scratch.root.starts_with(&base),
            "scratch root {} must live under the override {}",
            scratch.root.display(),
            base.display()
        );
    }

    /// The on-disk format is load-bearing outside this module's own
    /// round-trip (`save`/`load` and the cascade's intermediates all read
    /// bytes some other code path wrote), so pin the exact bytes rather
    /// than only that a write followed by a read agrees with itself.
    #[test]
    fn run_file_bytes_are_the_flat_little_endian_pairs_the_format_promises() {
        let scratch = ScratchDir::new().unwrap();
        let path = scratch.root.join("format.bin");
        write_run(&path, &[(0x0102_0304_0506_0708u64, 0x0a0b_0c0du32), (1, 2)]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 2 * ENTRY_BYTES, "12 bytes per entry, no header, no padding");
        assert_eq!(
            bytes,
            vec![
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // kmer, little-endian
                0x0d, 0x0c, 0x0b, 0x0a, // count, little-endian
                1, 0, 0, 0, 0, 0, 0, 0, // kmer 1
                2, 0, 0, 0, // count 2
            ]
        );
    }

    /// A run longer than both the write slab and the read buffer, so the
    /// mid-run `write` and the reader's buffer refill are each crossed
    /// several times -- the seam where an off-by-one in the batched write
    /// or the `fill_buf`/`consume` read would surface.
    #[test]
    fn a_run_longer_than_the_write_slab_and_read_buffer_roundtrips_exactly() {
        let scratch = ScratchDir::new().unwrap();
        let path = scratch.root.join("big.bin");
        let run: Vec<(u64, u32)> =
            (0u64..30_000).map(|i| (i, u32::try_from(i).unwrap_or(u32::MAX) + 1)).collect();
        write_run(&path, &run).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().len() as usize,
            run.len() * ENTRY_BYTES,
            "batched writes must not change the file's length"
        );

        let mut reader = RunReader::open(path).unwrap();
        let mut read_back: Vec<(u64, u32)> = Vec::with_capacity(run.len());
        while let Some(entry) = reader.next().unwrap() {
            read_back.push(entry);
        }
        assert_eq!(read_back, run);
    }

    #[test]
    fn run_reader_reports_truncated_file_as_internal_error_not_silent_none() {
        let scratch = ScratchDir::new().unwrap();
        let path = scratch.root.join("truncated.bin");
        // A whole kmer (8 bytes) but only 2 of the count's 4 bytes -- a
        // clean EOF could never happen here from this module's own writer.
        std::fs::write(&path, [0u8; 10]).unwrap();

        let mut reader = RunReader::open(path).unwrap();
        let err = reader.next().unwrap_err();
        assert!(matches!(err, FastDnaError::Internal { .. }));
    }
}
