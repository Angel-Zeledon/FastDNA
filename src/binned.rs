// src/binned.rs

//! Minimizer-partitioned counting: phase 1 accumulates super-k-mer bytes
//! into `NUM_BINS` append-only chunk lists, phase 2 counts one bin at a
//! time, and a final k-way merge puts the bins back into one globally
//! sorted table (`docs/design-minimizer-counting.md` §3.2, §3.5, §3.6).
//!
//! Standalone: `pipeline.rs` does not reach this module until step 4, and
//! neither the in-memory nor the disk strategy is touched by it.
//!
//! # What changes, and why it is the larger half of the win
//!
//! Today every worker builds a private `KmerCounter`. Batches reach workers
//! essentially at random and the same k-mers recur throughout a real FASTQ,
//! so **every worker's private table converges toward the entire
//! distinct-k-mer set**, not `1/N` of it (`mem_estimate.rs` documents the
//! measured reason). The calibrated model is
//! `777 MiB + threads x 48 MiB + 1.168 x threads x occurrences`, and the
//! `threads x occurrences` term is what makes an 8-thread run cost 8.34 GB.
//!
//! Here a worker accumulates *super-k-mer bytes* instead, and those are a
//! **partition** of the input rather than a replicated summary: worker 3's
//! and worker 5's contributions to bin 17 are disjoint pieces of one store.
//! Eight workers hold that store *between* them, not each. The
//! `threads x occurrences` term does not shrink -- it disappears.
//!
//! Two thread-proportional terms remain, both bounded and both independent
//! of input size: the open chunks (`threads x bins x chunk_bytes`, 64 MiB at
//! 8 threads and the defaults) and phase 2's per-worker expansion buffers.
//! That is the design's falsifiable structural claim, and it is what
//! `tests/dual_strategy.rs` and the §4.1 RSS measurement exist to check.
//!
//! # Why a lock per bin is not a bottleneck
//!
//! The hot path is lock-free: a worker appends to *its own* open [`Chunk`]
//! for the target bin -- a bounds check and a ~12-byte copy. Only when a
//! chunk fills does it publish, taking one `Mutex<Vec<Chunk>>` per bin per
//! 16 KiB. Over a benchmark-scale run that is `829 MiB / 16 KiB` ~ 53,000
//! acquisitions spread across 8 threads and 512 independent mutexes.
//!
//! # Why the final merge is needed at all
//!
//! `disk_spill.rs` gets globally sorted output for free, because high-bit
//! bucketing keeps ascending bucket order aligned with ascending k-mer
//! order. A minimizer bin holds k-mers scattered across the whole `u64`
//! range by construction, so that property is gone -- irrecoverably. KMC
//! simply accepts this and ships a database grouped by signature; FastDNA
//! cannot, because `export.rs`, the GenomeScope histogram path and the
//! planned binary table all assume ascending `kmer_u64`. §3.5 resolves it as
//! option A: one final streaming k-way merge over the `NUM_BINS` sorted bin
//! tables, `log2(512) = 9` comparisons per entry over the **compacted**
//! table rather than over the occurrences. The machinery already exists --
//! `counter.rs::k_way_merge_sorted_counts` is exactly this algorithm over
//! in-memory slices -- so this module reuses it rather than adding a third
//! implementation of a k-way merge to the crate.
//!
//! Note that the bins are not merely sorted, they are **disjoint**: the
//! signature is a function of the canonical k-mer (`sig(x) == sig(rc(x))`,
//! so `sig(canon(x)) == sig(x)`), which means every occurrence of one
//! canonical k-mer reaches exactly one bin. The merge folds equal keys with
//! `saturating_add` anyway, so a violation would be summed rather than
//! duplicated -- but `bins_hold_disjoint_kmer_sets` asserts the property
//! directly, because it is the invariant the whole architecture rests on.

use std::sync::Mutex;

use rayon::prelude::*;

use crate::counter::k_way_merge_sorted_counts;
use crate::fastq::FastqRecord;
use crate::minimizer::{bin_of, DEFAULT_M, DEFAULT_NUM_BINS};
use crate::superkmer::{encode_record_into, for_each_superkmer, records, MAX_SUPER_KMER_BASES};

/// Size of one bin chunk, per §3.4.
///
/// 16 KiB keeps `threads x bins x chunk_bytes` at 64 MiB for 8 threads and
/// 512 bins, and makes a publish happen once per 16 KiB of super-k-mer bytes
/// rather than once per record.
pub const DEFAULT_CHUNK_BYTES: usize = 16 * 1024;

/// The smallest chunk that can hold any single record: a 255-base
/// super-k-mer is `1 + 64 = 65` bytes. A chunk below this could reject a
/// record no matter how empty it was, which would drop k-mers silently.
pub const MIN_CHUNK_BYTES: usize = 1 + MAX_SUPER_KMER_BASES.div_ceil(4);

/// Everything tunable about the binned strategy, in one place.
///
/// §3.3 and §3.2 both call for `m` and the bin count to be configurable
/// rather than baked in: the defaults were sized the same way
/// `DEFAULT_BUCKET_BITS` and `RAW_FINALIZE_THRESHOLD` were, not proven by a
/// sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinnedConfig {
    /// k-mer length, `1..=32`.
    pub k: usize,
    /// Minimizer length, `1..=k`. Odd is preferable -- see
    /// [`crate::minimizer::DEFAULT_M`].
    pub m: usize,
    /// Number of bins; a power of two.
    pub num_bins: usize,
    /// Bytes per chunk; at least [`MIN_CHUNK_BYTES`].
    pub chunk_bytes: usize,
}

impl BinnedConfig {
    /// The defaults, for a given `k`.
    pub fn new(k: usize) -> Self {
        Self { k, m: DEFAULT_M, num_bins: DEFAULT_NUM_BINS, chunk_bytes: DEFAULT_CHUNK_BYTES }
    }

    /// Clamps the fields whose violation would be a silent wrong answer
    /// rather than a slow one, so a caller cannot construct a store that
    /// drops records.
    ///
    /// `m` is clamped into `1..=k` (a minimizer longer than the k-mer has no
    /// window), `num_bins` is rounded up to a power of two (`bin_of` masks
    /// rather than divides) and `chunk_bytes` up to [`MIN_CHUNK_BYTES`].
    /// `k` itself is *not* clamped: an out-of-range `k` makes
    /// `for_each_superkmer` emit nothing, exactly as
    /// `extract_canonical_kmers` returns nothing, and silently changing it
    /// would count something the caller did not ask for.
    pub fn sanitized(mut self) -> Self {
        self.num_bins = self.num_bins.max(1).next_power_of_two();
        self.chunk_bytes = self.chunk_bytes.max(MIN_CHUNK_BYTES);
        self.m = self.m.clamp(1, self.k.max(1));
        self
    }
}

/// A fixed-capacity byte buffer of back-to-back super-k-mer records.
///
/// Zero-initialised, so the byte just past the last record is the `0` length
/// prefix that §3.4's layout uses to terminate a chunk -- a reader needs no
/// separate length to stop in the right place, and [`Chunk::filled`] is
/// belt-and-braces rather than the only thing standing between a decoder and
/// the uninitialised tail.
///
/// `Box<[u8]>` rather than the `Box<[u8; CHUNK_BYTES]>` §3.4 sketches,
/// because the chunk size is a [`BinnedConfig`] field: a const-generic array
/// would push that tuning knob into the type system for the sake of one
/// `usize` in a fat pointer.
#[derive(Debug)]
pub struct Chunk {
    bytes: Box<[u8]>,
    len: usize,
}

impl Chunk {
    pub fn new(capacity: usize) -> Self {
        Self { bytes: vec![0u8; capacity].into_boxed_slice(), len: 0 }
    }

    /// The written prefix.
    #[inline(always)]
    pub fn filled(&self) -> &[u8] {
        // `len` never exceeds the allocation: it only ever grows by exactly
        // what `encode_record_into` reported writing into the tail.
        self.bytes.get(..self.len).unwrap_or(&[])
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends one super-k-mer, or returns `false` if it does not fit --
    /// leaving the chunk untouched, so the caller can publish it and retry
    /// against a fresh one.
    #[inline]
    fn try_push(&mut self, bases: &[u8]) -> bool {
        let Some(tail) = self.bytes.get_mut(self.len..) else {
            return false;
        };
        match encode_record_into(bases, tail) {
            Some(written) => {
                self.len += written;
                true
            }
            None => false,
        }
    }
}

/// One bin's accumulated super-k-mers, plus the whole set of bins.
///
/// Chunks are handed over whole, so the hot path never holds a lock while it
/// is writing bases.
#[derive(Debug)]
pub struct BinStore {
    bins: Vec<Mutex<Vec<Chunk>>>,
    config: BinnedConfig,
}

impl BinStore {
    pub fn new(config: BinnedConfig) -> Self {
        let config = config.sanitized();
        let bins = (0..config.num_bins).map(|_| Mutex::new(Vec::new())).collect();
        Self { bins, config }
    }

    pub fn config(&self) -> BinnedConfig {
        self.config
    }

    pub fn num_bins(&self) -> usize {
        self.config.num_bins
    }

    /// A fresh per-worker writer. Every worker needs its own; sharing one
    /// would reintroduce the lock this design exists to avoid.
    pub fn writer(&self) -> BinWriter {
        BinWriter::new(self.config)
    }

    /// Hands a full chunk to its bin. Called once per `chunk_bytes` of
    /// super-k-mer payload, not once per record.
    ///
    /// A poisoned mutex is recovered rather than propagated, for the same
    /// reason `counter.rs` recovers one: a worker that panicked mid-publish
    /// has already had its whole result discarded by the pipeline's panic
    /// check, and turning that into a second panic here would replace a
    /// diagnosable failure with an unrelated one.
    pub fn publish(&self, bin: usize, chunk: Chunk) {
        if chunk.is_empty() {
            return;
        }
        if let Some(slot) = self.bins.get(bin) {
            let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.push(chunk);
        }
    }

    /// Takes one bin's chunks, leaving it empty. Phase 2 calls this so the
    /// chunks are dropped as soon as they have been expanded, rather than
    /// staying alive alongside the growing output.
    fn take_bin(&self, bin: usize) -> Vec<Chunk> {
        match self.bins.get(bin) {
            Some(slot) => {
                let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                std::mem::take(&mut *guard)
            }
            None => Vec::new(),
        }
    }

    /// How many bytes of super-k-mer payload the store currently holds.
    /// Diagnostic only -- §6's R3 asks for a per-bin occupancy report before
    /// this strategy is ever made the default.
    pub fn occupancy(&self) -> Vec<usize> {
        self.bins
            .iter()
            .map(|slot| {
                let guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.iter().map(|c| c.len).sum()
            })
            .collect()
    }

    /// Phase 2 plus the cross-bin merge: expands each bin's super-k-mers,
    /// sorts and compacts it, then merges the bin tables into one globally
    /// sorted, deduplicated table.
    ///
    /// Bins are processed in parallel; the merge is sequential and streaming.
    /// The expansion buffer is created per rayon worker, not per bin, so a
    /// 512-bin run does not pay 512 allocations of ~12.5 MiB.
    pub fn finish(&self) -> Vec<(u64, u32)> {
        let k = self.config.k;
        let tables: Vec<Vec<(u64, u32)>> = (0..self.config.num_bins)
            .into_par_iter()
            .map_init(Vec::<u64>::new, |expanded, bin| self.count_bin(bin, k, expanded))
            .collect();

        k_way_merge_sorted_counts(tables)
    }

    /// The sequential form of [`finish`](Self::finish), for callers that are
    /// already inside a rayon scope or want a deterministic single-threaded
    /// run. Produces a bit-identical table: a bin's chunks are expanded into
    /// an unsorted buffer that is then sorted, so the order chunks arrived in
    /// cannot reach the result.
    pub fn finish_sequential(&self) -> Vec<(u64, u32)> {
        let k = self.config.k;
        let mut expanded: Vec<u64> = Vec::new();
        let tables: Vec<Vec<(u64, u32)>> =
            (0..self.config.num_bins).map(|bin| self.count_bin(bin, k, &mut expanded)).collect();

        k_way_merge_sorted_counts(tables)
    }

    /// Expands, sorts and compacts one bin, freeing its chunks before the
    /// sort so the two never coexist.
    fn count_bin(&self, bin: usize, k: usize, expanded: &mut Vec<u64>) -> Vec<(u64, u32)> {
        let chunks = self.take_bin(bin);
        if chunks.is_empty() {
            return Vec::new();
        }

        expanded.clear();
        for chunk in &chunks {
            for record in records(chunk.filled()) {
                record.expand_canonical_into(k, expanded);
            }
        }
        // The chunks are dead the moment they have been expanded, and a
        // bin's `Vec<u64>` is 8x its packed form -- holding both across the
        // sort would be the single largest avoidable transient in phase 2.
        drop(chunks);

        expanded.sort_unstable();
        compact_sorted(expanded)
    }
}

/// Compacts a sorted `Vec<u64>` of occurrences into a sorted, deduplicated
/// `(kmer, count)` table.
///
/// The same run-length pass `counter.rs::compact_raw` performs, with the
/// same saturation rule (see [`run_count`]).
fn compact_sorted(sorted: &[u64]) -> Vec<(u64, u32)> {
    let mut out: Vec<(u64, u32)> = Vec::new();
    let len = sorted.len();
    let mut i = 0usize;
    while i < len {
        let kmer = sorted[i];
        let run_start = i;
        i += 1;
        while i < len && sorted[i] == kmer {
            i += 1;
        }
        out.push((kmer, run_count(i - run_start)));
    }
    out
}

/// A run of `len` identical occurrences as a `u32` count, saturating rather
/// than wrapping or panicking.
///
/// The §5.1 invariant, pulled out of [`compact_sorted`] so it can be tested
/// at the boundary without materialising four billion `u64`s: `counter.rs`
/// applies exactly this clamp at every compaction and every merge, and the
/// binned path has to agree with it or one pathological k-mer would make the
/// two strategies disagree.
#[inline(always)]
fn run_count(len: usize) -> u32 {
    len.min(u32::MAX as usize) as u32
}

/// A worker's private set of open chunks, one per bin.
///
/// Every append lands in this worker's own chunk for the target bin, so the
/// per-record path never touches a lock. A chunk is published to the shared
/// [`BinStore`] only when the next record does not fit into it.
#[derive(Debug)]
pub struct BinWriter {
    open: Vec<Chunk>,
    config: BinnedConfig,
    /// Total k-mer occurrences routed through this writer, counted exactly
    /// the way `KmerCounter::insert_batch` counts its own -- every instance
    /// handed over, independent of any later per-key saturation. Required
    /// for `total_kmers()` to match the other strategies bit for bit.
    occurrences: u64,
}

impl BinWriter {
    fn new(config: BinnedConfig) -> Self {
        let config = config.sanitized();
        // Allocated up front rather than lazily: every bin is touched within
        // the first few thousand reads anyway, and `Vec<Option<Chunk>>`
        // would put a branch on the per-super-k-mer path to save memory only
        // for inputs too small for any of this to matter. This is the
        // `threads x bins x chunk_bytes` term §3.6 budgets at 64 MiB.
        let open = (0..config.num_bins).map(|_| Chunk::new(config.chunk_bytes)).collect();
        Self { open, config, occurrences: 0 }
    }

    /// How many k-mer occurrences this writer has routed.
    pub fn occurrences(&self) -> u64 {
        self.occurrences
    }

    /// Cuts one sequence into super-k-mers and routes each to its bin.
    ///
    /// Returns the number of k-mer occurrences added, which is exactly
    /// `kmer::extract_canonical_kmers(seq, k).len()` -- the multiset
    /// invariant proved in `superkmer.rs` is what makes that equality hold.
    pub fn push_sequence(&mut self, store: &BinStore, seq: &[u8]) -> usize {
        let BinnedConfig { k, m, num_bins, chunk_bytes } = self.config;
        let mut added = 0usize;

        for_each_superkmer(seq, k, m, |_start, bases, signature| {
            added += bases.len() - k + 1;
            let bin = bin_of(signature, num_bins);
            let Some(chunk) = self.open.get_mut(bin) else {
                // Unreachable: `bin_of` is total into `0..num_bins` and
                // `open` has exactly that length. Reached fallibly rather
                // than by an index that could panic on a hot path.
                return;
            };

            if chunk.try_push(bases) {
                return;
            }

            // Full: hand it over whole and start a fresh one. `std::mem::
            // replace` publishes the filled chunk without ever leaving the
            // slot empty, so a record can never be dropped between the two.
            let full = std::mem::replace(chunk, Chunk::new(chunk_bytes));
            store.publish(bin, full);
            let pushed = chunk.try_push(bases);
            debug_assert!(pushed, "a record must fit into an empty chunk: MIN_CHUNK_BYTES guarantees it");
        });

        self.occurrences += added as u64;
        added
    }

    /// Publishes every partially filled chunk. Must be called, or a whole
    /// worker's tail of super-k-mers -- up to `bins x chunk_bytes` of them --
    /// is silently dropped.
    pub fn finish(self, store: &BinStore) {
        for (bin, chunk) in self.open.into_iter().enumerate() {
            store.publish(bin, chunk);
        }
    }

    /// Publishes every partially filled chunk and keeps the writer usable,
    /// for a caller that wants to drain without giving up its allocations.
    pub fn flush(&mut self, store: &BinStore) {
        for (bin, chunk) in self.open.iter_mut().enumerate() {
            if chunk.is_empty() {
                continue;
            }
            let full = std::mem::replace(chunk, Chunk::new(self.config.chunk_bytes));
            store.publish(bin, full);
        }
    }
}

/// What a binned count produced: the same pair `KmerCounter` exposes, so a
/// caller can hand both straight to `KmerCounter::from_sorted_entries`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinnedCounts {
    /// Sorted ascending by k-mer, deduplicated.
    pub entries: Vec<(u64, u32)>,
    /// Every k-mer instance seen, before any per-key saturation.
    pub total_occurrences: u64,
}

/// Counts a slice of records with the binned strategy, single-threaded.
///
/// The §5 step 3 entry point. `pipeline.rs` does not call this -- it drives
/// [`BinStore`] and [`BinWriter`] directly, one writer per rayon worker --
/// but this is the same code path with one writer, which is what makes the
/// differential tests below a test of the real thing.
pub fn count_records(records: &[FastqRecord], config: BinnedConfig) -> BinnedCounts {
    let store = BinStore::new(config);
    let mut writer = store.writer();
    for record in records {
        writer.push_sequence(&store, &record.seq);
    }
    let total_occurrences = writer.occurrences();
    writer.finish(&store);

    BinnedCounts { entries: store.finish_sequential(), total_occurrences }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::fastq::FastqReader;
    use crate::kmer::extract_canonical_kmers;
    use std::io::Cursor;

    /// The same deterministic FASTQ generator `tests/dual_strategy.rs`
    /// uses -- a fixed-seed xorshift64, no RNG crate -- so a failure here is
    /// exactly reproducible and the two test suites exercise the same
    /// shapes. Draws `num_reads` reads of `read_len` bases from a
    /// `genome_len`-base synthetic genome; with `num_reads * read_len` far
    /// exceeding `genome_len` the same k-mers necessarily recur across many
    /// reads, and therefore across many chunk and bin boundaries, which is
    /// exactly where a seam bug would surface. A small fraction of reads
    /// carry one ambiguous base, so the window-reset behaviour around `N` is
    /// compared too, not just plain counting.
    fn synthetic_fastq(num_reads: usize, read_len: usize, genome_len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut next_u64 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let bases = [b'A', b'C', b'G', b'T'];
        let genome: Vec<u8> = (0..genome_len).map(|_| bases[(next_u64() % 4) as usize]).collect();

        let mut out = Vec::with_capacity(num_reads * (read_len * 2 + 24));
        for i in 0..num_reads {
            let start = (next_u64() as usize) % (genome_len - read_len + 1);
            let mut seq: Vec<u8> = genome[start..start + read_len].to_vec();
            if next_u64() % 37 == 0 {
                let pos = (next_u64() as usize) % read_len;
                seq[pos] = b'N';
            }
            out.extend_from_slice(format!("@read{i}\n").as_bytes());
            out.extend_from_slice(&seq);
            out.extend_from_slice(b"\n+\n");
            out.extend(std::iter::repeat_n(b'I', read_len));
            out.push(b'\n');
        }
        out
    }

    fn parse(fastq: &[u8]) -> Vec<FastqRecord> {
        let mut reader = FastqReader::new(Cursor::new(fastq.to_vec()));
        let mut out = Vec::new();
        while let Some(record) = reader.next_record().expect("the generated FASTQ must parse") {
            out.push(record);
        }
        out
    }

    /// The reference: exactly what the in-memory strategy does -- extract
    /// canonical k-mers per record and feed them to a `KmerCounter`.
    fn reference(records: &[FastqRecord], k: usize) -> (Vec<(u64, u32)>, u64) {
        let mut counter = KmerCounter::new();
        for record in records {
            counter.insert_batch(&extract_canonical_kmers(&record.seq, k));
        }
        (counter.iter().collect(), counter.total_kmers())
    }

    // ---------------------------------------------------------------
    // The differential test: binned vs KmerCounter.
    // ---------------------------------------------------------------

    /// §5 step 3's acceptance criterion, over many synthetic inputs rather
    /// than a golden file: the binned strategy's `(kmer, count)` table and
    /// total occurrence count must equal `KmerCounter`'s **exactly**, not
    /// approximately, on every shape.
    ///
    /// The parameter sweep is deliberately hostile to the design's own
    /// assumptions: bin counts down to 1 (every k-mer in one bin, so the
    /// cross-bin merge degenerates) and up to 1024, chunks at the 65-byte
    /// minimum (a publish per record, so every chunk boundary is exercised)
    /// as well as the 16 KiB default, `m` from 1 (a two-base alphabet of
    /// signatures) to 11, and `k` from 5 to 32.
    #[test]
    fn binned_counts_match_kmer_counter_exactly_across_many_synthetic_inputs() {
        let mut cases = 0usize;
        for (reads, read_len, genome_len, seed) in [
            (2_000usize, 60usize, 500usize, 0x5EED_C0FF_EE42u64),
            (500, 40, 3_000, 0xC0DE_1234_5678),
            (200, 150, 1_000, 0xABCD_EF01_2345),
            (50, 250, 400, 0x1357_9BDF_2468),
            (1_000, 31, 200, 0xFEED_FACE_BEEF),
            (3, 300, 350, 0x0BAD_F00D_0BAD),
        ] {
            let fastq = synthetic_fastq(reads, read_len, genome_len, seed);
            let parsed = parse(&fastq);

            for k in [5usize, 15, 21, 31, 32] {
                if k > read_len {
                    continue;
                }
                let (expected_entries, expected_total) = reference(&parsed, k);

                for (m, num_bins, chunk_bytes) in [
                    (DEFAULT_M, DEFAULT_NUM_BINS, DEFAULT_CHUNK_BYTES),
                    (DEFAULT_M, 1, DEFAULT_CHUNK_BYTES),
                    (DEFAULT_M, 4, MIN_CHUNK_BYTES),
                    (1, 64, DEFAULT_CHUNK_BYTES),
                    (3, 1024, MIN_CHUNK_BYTES),
                    (11, 256, 512),
                ] {
                    if m > k {
                        continue;
                    }
                    let config = BinnedConfig { k, m, num_bins, chunk_bytes };
                    let got = count_records(&parsed, config);

                    assert_eq!(
                        got.total_occurrences, expected_total,
                        "total occurrences differ at k={k} m={m} bins={num_bins} chunk={chunk_bytes} seed={seed:#x}"
                    );
                    assert_eq!(
                        got.entries, expected_entries,
                        "count table differs at k={k} m={m} bins={num_bins} chunk={chunk_bytes} seed={seed:#x}"
                    );
                    cases += 1;
                }
            }
        }
        println!("binned vs KmerCounter: {cases} (input, k, m, bins, chunk) combinations agreed exactly");
        assert!(cases > 100, "the sweep must actually have run: {cases} cases");
    }

    /// The same equality with the work spread across rayon workers and
    /// several writers, which is what `pipeline.rs` will do at step 4.
    /// Chunk *arrival* order into a bin is nondeterministic across threads;
    /// the result must not be.
    #[test]
    fn many_writers_and_a_parallel_finish_produce_the_same_table() {
        let fastq = synthetic_fastq(4_000, 80, 1_500, 0x2026_0825_1234_5678);
        let parsed = parse(&fastq);
        let k = 21usize;
        let (expected_entries, expected_total) = reference(&parsed, k);

        let config = BinnedConfig::new(k);
        let store = BinStore::new(config);

        // Four independent writers over interleaved slices of the input,
        // the same way four rayon workers receive interleaved batches.
        let totals: Vec<u64> = (0..4usize)
            .into_par_iter()
            .map(|worker| {
                let mut writer = store.writer();
                for record in parsed.iter().skip(worker).step_by(4) {
                    writer.push_sequence(&store, &record.seq);
                }
                let total = writer.occurrences();
                writer.finish(&store);
                total
            })
            .collect();

        assert_eq!(totals.iter().sum::<u64>(), expected_total);
        assert_eq!(store.finish(), expected_entries);
    }

    /// Sequential and parallel phase 2 must be bit-identical: a strategy
    /// whose answer depends on how many threads ran it is not a strategy.
    #[test]
    fn sequential_and_parallel_finish_agree() {
        let fastq = synthetic_fastq(1_500, 70, 900, 0x4242_4242_4242_4242);
        let parsed = parse(&fastq);
        let config = BinnedConfig::new(25);

        let one = {
            let store = BinStore::new(config);
            let mut writer = store.writer();
            for record in &parsed {
                writer.push_sequence(&store, &record.seq);
            }
            writer.finish(&store);
            store.finish_sequential()
        };
        let many = {
            let store = BinStore::new(config);
            let mut writer = store.writer();
            for record in &parsed {
                writer.push_sequence(&store, &record.seq);
            }
            writer.finish(&store);
            store.finish()
        };
        assert_eq!(one, many);
    }

    // ---------------------------------------------------------------
    // The architectural invariants.
    // ---------------------------------------------------------------

    /// Every occurrence of one canonical k-mer must reach exactly one bin.
    /// This is R2 -- the risk that produces a well-formed sorted table with
    /// wrong numbers -- checked directly rather than only through the
    /// aggregate count, so a violation is reported as what it is instead of
    /// as an off-by-one somewhere in a total.
    #[test]
    fn bins_hold_disjoint_kmer_sets() {
        // A genome large enough that the distinct-k-mer set can actually
        // reach every bin: 800 bases would only have 770 distinct 31-mers to
        // spread over 512 bins, and a mostly empty partition would prove
        // nothing about disjointness at scale.
        let fastq = synthetic_fastq(6_000, 90, 40_000, 0x9999_8888_7777_6666);
        let parsed = parse(&fastq);
        let k = 31usize;

        let store = BinStore::new(BinnedConfig::new(k));
        let mut writer = store.writer();
        for record in &parsed {
            writer.push_sequence(&store, &record.seq);
        }
        writer.finish(&store);

        let mut owner: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut nonempty = 0usize;
        let mut expanded = Vec::new();
        for bin in 0..store.num_bins() {
            let table = store.count_bin(bin, k, &mut expanded);
            if table.is_empty() {
                continue;
            }
            nonempty += 1;
            assert!(table.windows(2).all(|w| w[0].0 < w[1].0), "bin {bin} is not sorted and deduplicated");
            for (kmer, _) in table {
                if let Some(previous) = owner.insert(kmer, bin) {
                    panic!("k-mer {kmer:#x} reached both bin {previous} and bin {bin}");
                }
            }
        }
        println!("{nonempty}/{} bins occupied, {} distinct k-mers", store.num_bins(), owner.len());
        assert!(nonempty > 400, "the partition must actually spread the input across bins");
        assert!(owner.len() > 30_000, "the input must actually be diverse: {} distinct k-mers", owner.len());
    }

    /// A per-bin occupancy report is what §6's R3 asks for before this
    /// strategy could ever be promoted. This pins that the report exists and
    /// that a synthetic input does not concentrate in one bin -- the skew
    /// question itself is answered by real data, not here.
    #[test]
    fn per_bin_occupancy_is_reported_and_is_not_wildly_skewed() {
        let fastq = synthetic_fastq(3_000, 100, 2_000, 0x1122_3344_5566_7788);
        let parsed = parse(&fastq);
        let store = BinStore::new(BinnedConfig::new(31));
        let mut writer = store.writer();
        for record in &parsed {
            writer.push_sequence(&store, &record.seq);
        }
        writer.finish(&store);

        let occupancy = store.occupancy();
        let total: usize = occupancy.iter().sum();
        let max = occupancy.iter().copied().max().unwrap_or(0);
        let mean = total as f64 / occupancy.len() as f64;
        println!(
            "occupancy: {total} bytes over {} bins, mean {mean:.0}, max {max} ({:.1}x mean)",
            occupancy.len(),
            max as f64 / mean
        );
        assert!(total > 0);
        assert!(
            (max as f64) < 20.0 * mean,
            "one bin holds {max} bytes against a {mean:.0}-byte mean; the static hash-to-bin map is not spreading synthetic input"
        );
    }

    /// A chunk must publish and continue rather than drop the record that
    /// did not fit. Driven at the minimum chunk size, where a publish
    /// happens on nearly every record.
    #[test]
    fn a_full_chunk_publishes_and_keeps_every_record() {
        let fastq = synthetic_fastq(600, 120, 700, 0xAAAA_BBBB_CCCC_DDDD);
        let parsed = parse(&fastq);
        let k = 21usize;
        let (expected_entries, expected_total) = reference(&parsed, k);

        let config = BinnedConfig { k, m: DEFAULT_M, num_bins: 8, chunk_bytes: MIN_CHUNK_BYTES };
        let got = count_records(&parsed, config);
        assert_eq!(got.total_occurrences, expected_total);
        assert_eq!(got.entries, expected_entries);
    }

    /// Forgetting `finish` must not be the difference between a right and a
    /// wrong answer *silently*: this pins that the tail really is in the
    /// open chunks, so the requirement is real and documented rather than
    /// folklore.
    #[test]
    fn unpublished_open_chunks_hold_the_tail_of_the_input() {
        let fastq = synthetic_fastq(50, 60, 300, 0xFACE_FACE_FACE_FACE);
        let parsed = parse(&fastq);
        let k = 21usize;

        let store = BinStore::new(BinnedConfig::new(k));
        let mut writer = store.writer();
        for record in &parsed {
            writer.push_sequence(&store, &record.seq);
        }
        // Small input, default chunk size: nothing has overflowed yet, so
        // everything is still in the writer.
        assert_eq!(store.occupancy().iter().sum::<usize>(), 0);
        writer.finish(&store);
        assert!(store.occupancy().iter().sum::<usize>() > 0);
    }

    /// `flush` must be equivalent to `finish` for everything already
    /// written, and must leave the writer usable.
    #[test]
    fn flush_is_equivalent_to_finishing_and_starting_again() {
        let fastq = synthetic_fastq(400, 70, 600, 0x0F0F_0F0F_0F0F_0F0F);
        let parsed = parse(&fastq);
        let k = 21usize;
        let (expected_entries, expected_total) = reference(&parsed, k);

        let store = BinStore::new(BinnedConfig::new(k));
        let mut writer = store.writer();
        for (i, record) in parsed.iter().enumerate() {
            writer.push_sequence(&store, &record.seq);
            if i % 50 == 49 {
                writer.flush(&store);
            }
        }
        assert_eq!(writer.occurrences(), expected_total);
        writer.finish(&store);
        assert_eq!(store.finish(), expected_entries);
    }

    // ---------------------------------------------------------------
    // Edge cases and the pinned behaviours from §5.1.
    // ---------------------------------------------------------------

    #[test]
    fn an_empty_input_produces_an_empty_table() {
        let got = count_records(&[], BinnedConfig::new(31));
        assert!(got.entries.is_empty());
        assert_eq!(got.total_occurrences, 0);
    }

    #[test]
    fn reads_that_yield_no_kmers_are_counted_as_zero_not_as_an_error() {
        let records: Vec<FastqRecord> = [b"ACGT".to_vec(), vec![b'N'; 100], Vec::new()]
            .into_iter()
            .map(|seq| FastqRecord { id: Vec::new(), qual: vec![b'I'; seq.len()], seq })
            .collect();
        let got = count_records(&records, BinnedConfig::new(31));
        assert!(got.entries.is_empty());
        assert_eq!(got.total_occurrences, 0);
    }

    /// The §5.1 saturation invariant: a count must stop at `u32::MAX` rather
    /// than wrap or panic. Checked at the boundary on `run_count`, the clamp
    /// `compact_sorted` applies, because materialising four billion
    /// occurrences to reach it through the full path would cost 34 GB and
    /// prove nothing extra.
    #[test]
    fn counts_saturate_at_u32_max_rather_than_wrapping() {
        assert_eq!(run_count(0), 0);
        assert_eq!(run_count(1), 1);
        assert_eq!(run_count(u32::MAX as usize - 1), u32::MAX - 1);
        assert_eq!(run_count(u32::MAX as usize), u32::MAX);
        assert_eq!(run_count(u32::MAX as usize + 1), u32::MAX, "must saturate, not wrap to 0");
        assert_eq!(run_count(usize::MAX), u32::MAX);
        // And the clamp is genuinely the one the compaction uses.
        assert_eq!(compact_sorted(&[5, 5, 5]), vec![(5, run_count(3))]);
    }

    #[test]
    fn compaction_matches_a_naive_count_on_small_inputs() {
        let mut raw = vec![7u64, 1, 7, 3, 3, 3, 9, 1, 1, 1, 1];
        raw.sort_unstable();
        assert_eq!(compact_sorted(&raw), vec![(1, 5), (3, 3), (7, 2), (9, 1)]);
        assert!(compact_sorted(&[]).is_empty());
        assert_eq!(compact_sorted(&[42]), vec![(42, 1)]);
    }

    /// A degenerate configuration must be clamped where a silent wrong
    /// answer would otherwise follow, and left alone where the caller's
    /// intent is unambiguous.
    #[test]
    fn the_config_is_sanitized_where_a_bad_value_would_be_silent() {
        let c = BinnedConfig { k: 31, m: 0, num_bins: 0, chunk_bytes: 1 }.sanitized();
        assert_eq!(c.m, 1);
        assert_eq!(c.num_bins, 1);
        assert_eq!(c.chunk_bytes, MIN_CHUNK_BYTES);

        // A non-power-of-two bin count rounds up, because `bin_of` masks.
        assert_eq!(BinnedConfig { num_bins: 300, ..BinnedConfig::new(31) }.sanitized().num_bins, 512);
        // `m > k` has no window; clamped to k.
        assert_eq!(BinnedConfig { k: 5, m: 11, ..BinnedConfig::new(5) }.sanitized().m, 5);
        // `k` is never rewritten: an out-of-range k must count nothing, not
        // something else.
        assert_eq!(BinnedConfig { k: 99, ..BinnedConfig::new(99) }.sanitized().k, 99);
        assert!(count_records(&[], BinnedConfig::new(99)).entries.is_empty());
    }

    #[test]
    fn min_chunk_bytes_holds_the_largest_possible_record() {
        assert_eq!(MIN_CHUNK_BYTES, crate::superkmer::record_len(MAX_SUPER_KMER_BASES));
        let mut chunk = Chunk::new(MIN_CHUNK_BYTES);
        assert!(chunk.is_empty());
        assert!(chunk.filled().is_empty());
        assert!(chunk.try_push(&vec![b'A'; MAX_SUPER_KMER_BASES]));
        assert!(!chunk.try_push(b"ACGT"), "a full chunk must refuse rather than overflow");
        assert_eq!(chunk.filled().len(), MIN_CHUNK_BYTES);
    }
}
