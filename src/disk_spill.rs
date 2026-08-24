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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{FastDnaError, Result};

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

/// Writes one already-sorted, already-deduplicated run to `path` as flat
/// little-endian `(u64, u32)` pairs.
fn write_run(path: &Path, run: &[(u64, u32)]) -> Result<()> {
    let file = File::create(path).map_err(|e| io_err(path, e))?;
    let mut writer = BufWriter::new(file);
    for &(kmer, count) in run {
        writer.write_all(&kmer.to_le_bytes()).map_err(|e| io_err(path, e))?;
        writer.write_all(&count.to_le_bytes()).map_err(|e| io_err(path, e))?;
    }
    writer.flush().map_err(|e| io_err(path, e))?;
    Ok(())
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
        Ok(Self { reader: BufReader::new(file), path })
    }

    /// Returns the next entry, or `None` at a clean end-of-run. A partial
    /// read (EOF strictly between the two halves of one pair) can only mean
    /// this module's own writer produced a truncated file -- a bug, not a
    /// data problem -- so it is reported as `FastDnaError::Internal` rather
    /// than silently treated as end-of-run.
    fn next(&mut self) -> Result<Option<(u64, u32)>> {
        let mut kmer_buf = [0u8; 8];
        match self.reader.read_exact(&mut kmer_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(io_err(&self.path, e)),
        }
        let mut count_buf = [0u8; 4];
        self.reader.read_exact(&mut count_buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                FastDnaError::Internal {
                    detail: format!(
                        "spill run file {} ended mid-entry -- truncated scratch file",
                        self.path.display()
                    ),
                }
            } else {
                io_err(&self.path, e)
            }
        })?;
        Ok(Some((u64::from_le_bytes(kmer_buf), u32::from_le_bytes(count_buf))))
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
    num_buckets: usize,
    raw: Vec<u64>,
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
            num_buckets,
            raw: Vec::with_capacity(SPILL_RAW_THRESHOLD),
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

        let mut per_bucket: Vec<Vec<(u64, u32)>> = (0..self.num_buckets).map(|_| Vec::new()).collect();
        {
            let mut iter = self.raw.drain(..).peekable();
            while let Some(kmer) = iter.next() {
                let mut count: u32 = 1;
                while iter.peek() == Some(&kmer) {
                    iter.next();
                    count = count.saturating_add(1);
                }
                let bucket = bucket_of(kmer, self.k, self.bucket_bits);
                per_bucket[bucket].push((kmer, count));
            }
        }

        for (bucket, run) in per_bucket.into_iter().enumerate() {
            if run.is_empty() {
                continue;
            }
            let run_index = self.next_run_index[bucket];
            self.next_run_index[bucket] += 1;
            let path = self.scratch.run_path(self.worker, bucket, run_index);
            write_run(&path, &run)?;
            self.manifest[bucket].push(path);
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
    let mut fronts: Vec<Option<(u64, u32)>> = Vec::with_capacity(sources.len());
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(sources.len());

    for (idx, source) in sources.iter_mut().enumerate() {
        let front = source.next()?;
        if let Some((kmer, _)) = front {
            heap.push(Reverse((kmer, idx)));
        }
        fronts.push(front);
    }

    while let Some(Reverse((kmer, idx))) = heap.pop() {
        let mut count = fronts[idx].map(|(_, c)| c).unwrap_or(0);

        fronts[idx] = sources[idx].next()?;
        if let Some((next_kmer, _)) = fronts[idx] {
            heap.push(Reverse((next_kmer, idx)));
        }

        // Fold in every other source currently sitting at the same key --
        // sources can share keys (a k-mer seen by two different workers, or
        // by the same worker across two different flushes), but never
        // within one source, since each run is already deduplicated.
        loop {
            let same_key = matches!(heap.peek(), Some(&Reverse((peek_kmer, _))) if peek_kmer == kmer);
            if !same_key {
                break;
            }
            if let Some(Reverse((_, other_idx))) = heap.pop() {
                if let Some((_, other_count)) = fronts[other_idx] {
                    count = count.saturating_add(other_count);
                }
                fronts[other_idx] = sources[other_idx].next()?;
                if let Some((next_kmer, _)) = fronts[other_idx] {
                    heap.push(Reverse((next_kmer, other_idx)));
                }
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
        let mut paths: Vec<PathBuf> = Vec::new();
        for worker_manifest in manifests {
            if let Some(worker_paths) = worker_manifest.get(bucket) {
                paths.extend(worker_paths.iter().cloned());
            }
        }
        if paths.is_empty() {
            continue;
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

    while paths.len() > MAX_OPEN_RUNS {
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
            let file = File::create(&intermediate).map_err(|e| io_err(&intermediate, e))?;
            let mut writer = BufWriter::new(file);
            merge_sources_into(&mut sources, |kmer, count| {
                writer.write_all(&kmer.to_le_bytes()).map_err(|e| io_err(&intermediate, e))?;
                writer.write_all(&count.to_le_bytes()).map_err(|e| io_err(&intermediate, e))?;
                Ok(())
            })?;
            writer.flush().map_err(|e| io_err(&intermediate, e))?;
            drop(writer);
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
