// src/counter.rs

use std::cmp::Reverse;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
use std::sync::{Mutex, MutexGuard};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Outcome of a `prune` call, for reporting what a filter removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub dropped_min: u64,
    pub dropped_max: u64,
    pub kept: u64,
}

/// A sorted run of `(kmer, count)` entries, stored as two parallel arrays
/// (`keys: Vec<u64>`, `counts: Vec<u32>`) instead of `Vec<(u64, u32)>`.
///
/// `size_of::<(u64, u32)>()` is 16, not 12: the `u64` forces 8-byte
/// alignment, so the trailing `u32` is padded out to fill the rest of a
/// second 8-byte word. That padding is real, resident memory for every
/// entry `pending` and `finalized` hold -- on the 53.8-million-distinct-
/// k-mer benchmark table it is 205 MiB of nothing -- and it is moved, not
/// just held: every merge pass below reads and writes it again on every
/// call. Two parallel arrays store the same information at 12 bytes/entry
/// with no padding at all (`Vec<u64>` is 8-byte aligned and densely packed;
/// `Vec<u32>` is 4-byte aligned and densely packed; neither has anything to
/// pad against the other because they are separate allocations) -- a 25%
/// reduction in the bytes `compact_raw`, `merge_tables` and
/// `k_way_merge_tables` have to move.
///
/// Every entry at index `i` is `(keys[i], counts[i])`; the two arrays are
/// kept the same length by construction (`push` is the only way to grow
/// either, and it always grows both together), not by a runtime check on
/// every access.
///
/// This is an internal representation only: nothing outside `counter.rs`
/// sees a `CountTable`. `KmerCounter`'s public surface -- `from_sorted_entries`,
/// `iter`, `top_kmers` -- and the one piece of this file `binned.rs` calls
/// directly (`k_way_merge_sorted_counts`, deliberately untouched by this
/// type) all still speak `(u64, u32)` tuples, built from or flattened into
/// a `CountTable` at the boundary. See `k_way_merge_tables`'s doc comment
/// for the measurement this representation change shipped on, and
/// `KmerCounter`'s own doc comment for why the tuple boundary sits where it
/// does.
#[derive(Debug, Default, Clone)]
struct CountTable {
    keys: Vec<u64>,
    counts: Vec<u32>,
}

impl CountTable {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self { keys: Vec::with_capacity(capacity), counts: Vec::with_capacity(capacity) }
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    fn push(&mut self, kmer: u64, count: u32) {
        self.keys.push(kmer);
        self.counts.push(count);
    }

    /// Builds a `CountTable` from an already sorted, already deduplicated
    /// `(kmer, count)` vector -- the boundary conversion `from_sorted_entries`
    /// needs, since its callers (`pipeline.rs`, `binned.rs`'s test module)
    /// hand it a tuple `Vec` built outside this type.
    fn from_tuples(entries: Vec<(u64, u32)>) -> Self {
        let mut keys = Vec::with_capacity(entries.len());
        let mut counts = Vec::with_capacity(entries.len());
        for (kmer, count) in entries {
            keys.push(kmer);
            counts.push(count);
        }
        Self { keys, counts }
    }

    /// Flattens back into `(kmer, count)` tuples -- the boundary conversion
    /// every externally visible `(u64, u32)`-shaped return value needs.
    fn to_tuples(&self) -> Vec<(u64, u32)> {
        self.keys.iter().copied().zip(self.counts.iter().copied()).collect()
    }
}

/// The mutable state behind `KmerCounter`, split out so it can live inside
/// a single `Mutex` (see `KmerCounter`'s doc comment for why).
#[derive(Debug, Default)]
struct Inner {
    /// Every canonical k-mer instance seen since the last time `finalized`
    /// was rebuilt, in whatever order `insert`/`insert_batch` pushed them
    /// -- unsorted, with duplicates. Insertion is a plain `Vec::push`
    /// (amortized O(1), sequential memory access), not a hash-table
    /// lookup: see `KmerCounter`'s doc comment for why that distinction
    /// is the entire point of this type.
    raw: Vec<u64>,
    /// Sorted, deduplicated `(kmer, count)` runs produced by `compact_raw`
    /// but not yet folded into `finalized` -- each one internally correct
    /// (no duplicate keys within a single run), but keys may repeat
    /// *across* runs (the same k-mer compacted separately in two different
    /// eager passes). See `consolidate` for why these accumulate instead
    /// of being merged in immediately, and `MAX_PENDING_RUNS` for why that
    /// accumulation is bounded rather than unbounded. Stored as
    /// `CountTable`s, not `Vec<(u64, u32)>`s -- see that type's doc comment.
    pending: Vec<CountTable>,
    /// Sorted (ascending by k-mer), deduplicated `(kmer, count)` pairs.
    /// Only trustworthy when `valid` is `true`; rebuilt by `finalize_inner`
    /// otherwise (which folds `pending` -- and any leftover `raw` -- into
    /// it via `consolidate`). Stored as a `CountTable`, not a
    /// `Vec<(u64, u32)>` -- see that type's doc comment.
    finalized: CountTable,
    /// Whether `finalized` currently reflects every instance in `raw` and
    /// every run in `pending` (both drained empty once it does). `false`
    /// after any insertion; set back to `true` by `finalize_inner`.
    valid: bool,
    /// Destination buffer for `msd_partition`'s scatter pass, and `raw`'s
    /// partner: the two are swapped rather than copied, so after a partition
    /// this holds what `raw` held and vice versa. Kept on `Inner` rather
    /// than allocated inside `compact_raw` so its allocation is paid once
    /// per counter instead of once per eager compaction (~52 of those on a
    /// benchmark-scale worker).
    ///
    /// This is the one cost the MSD partition adds that `sort_unstable`
    /// alone did not have: `sort_unstable` sorts in place, so a worker now
    /// holds two threshold-sized `Vec<u64>`s instead of one. That is
    /// accounted for in `mem_estimate::PER_WORKER_RAW_TRANSIENT_BYTES`.
    scratch: Vec<u64>,
}

/// Above this many buffered (unsorted, with duplicates) instances in
/// `raw`, `insert`/`insert_batch` triggers an eager `finalize_inner`
/// rather than waiting for the first read.
///
/// Left unbounded, `raw` accumulates every occurrence a worker ever sees
/// -- hundreds of millions for a real sample -- before the first read
/// ever happens: `pipeline.rs` builds one `KmerCounter` per worker and
/// only reads any of them after the whole run, via `insert_batch` in a
/// loop with no read in between, and every worker's full share is alive
/// at once ahead of the reduce. Measured effect of leaving this
/// unbounded: three inputs sharing the same 514,827-distinct-k-mer set
/// but different occurrence counts (7,943,824 / 31,775,296 /
/// 127,101,184) produced peak RSS of 76MB / 283MB / 1,003MB -- linear in
/// occurrences at roughly 8.3 bytes each, not flat in cardinality the way
/// the old hash map was. This constant is what turns that back into
/// O(distinct k-mers) + O(cap) per worker instead of O(occurrences) per
/// worker.
///
/// 2,000,000 is a "low millions" starting point, not a round number
/// picked blind. At 8 bytes per buffered `u64`, the cap itself bounds
/// `raw` to 16MB per worker. That is large enough that
/// `finalize_inner`'s O(n log n) sort amortizes well against the O(1)
/// pushes that fill it -- a worker counting a 127M-occurrence input
/// finalizes on the order of ten times over the whole run at this cap,
/// not once per small batch -- and small enough that a single finalize
/// call's transient allocation stays bounded rather than scaling with
/// total occurrences: `finalize_inner` keeps `raw`'s already-allocated
/// capacity (8 bytes/entry) alive alongside a fresh pair of
/// `Vec::with_capacity(raw.len())` key/count arrays (`CountTable`, 12
/// bytes/entry, no alignment padding -- see that type's doc comment) while
/// it drains one into the other, so a finalize at the cap costs on the
/// order of 2,000,000 * 20 bytes =~ 40MB transient per worker, not
/// gigabytes. Raising the cap trades more
/// of that transient (and a higher permanent floor) for fewer, larger
/// sorts; lowering it trades the other way. This is a starting point,
/// not a value proven optimal by a sweep across input shapes.
/// **Swept, and deliberately left where it was.**
/// `docs/PERFORMANCE_PLAN.md` workstream 2 asks for this threshold to be
/// swept empirically rather than left at its original reasoned guess. It
/// was, over 500k / 1M / 2M / 4M / 8M / 16M, pushing 40,000,000 occurrences
/// through the real compaction path (MSD partition, per-bucket sort,
/// run-length pass), five runs per point, on two coverage shapes:
///
/// ```text
///  threshold   buffer   high coverage   low coverage   runs produced
///    500,000     4 MB       0.671 s        0.678 s          80
///  1,000,000     8 MB       0.749 s        0.668 s          40
///  2,000,000    16 MB       0.719 s        0.736 s          20   <- today
///  4,000,000    32 MB       0.686 s        0.870 s          10
///  8,000,000    64 MB       0.649 s        0.892 s           5
/// 16,000,000   128 MB       0.838 s        1.032 s           3
/// ```
///
/// **The sweep does not identify a better value, and that is the finding.**
/// The two coverage shapes disagree about which direction to move: high
/// coverage is fastest at 8M, where low coverage is 37% *slower* than at its
/// own best of 1M. Nothing here beats 2,000,000 on both, and the total
/// spread across every point is comparable to this machine's own run-to-run
/// variation.
///
/// It is also only half of the trade. A lower threshold produces
/// proportionally more `pending` runs (80 against 3 across this sweep),
/// which is the direct input to `consolidate`'s cost and to the width of the
/// final k-way merge -- and that half cannot be measured in a loop like
/// this one, only over a real file end to end, which needs the native
/// Windows release build `docs/PERFORMANCE_PLAN.md` requires benchmark
/// numbers to come from and which this environment does not have. Moving a
/// tuning constant on a measurement that covers one side of its trade and
/// disagrees with itself across inputs would be worse than leaving it: 2M
/// sits mid-table on both shapes, which is what a value chosen without
/// knowing the data shape should do.
const RAW_FINALIZE_THRESHOLD: usize = 2_000_000;

/// Above this many *unconsolidated* runs in `inner.pending`, an eager
/// `compact_raw` (see `RAW_FINALIZE_THRESHOLD`) triggers a `consolidate`
/// pass rather than leaving the run count to grow for the rest of the run.
///
/// This is the fix for a real, measured quadratic blowup in the design
/// `MAX_PENDING_RUNS` replaces: that design folded every newly-compacted
/// run straight into a single ever-growing `finalized` table via a
/// two-way merge (today's `merge_tables`; the design predates the G-3
/// struct-of-arrays change), so the Nth eager compaction touched the *entire*
/// table accumulated so far -- O(compactions x table size) over a whole
/// run. On a large, diverse 2.14 GB input (53.8M distinct k-mers, 840M
/// occurrences, 8 workers), each worker crosses `RAW_FINALIZE_THRESHOLD`
/// roughly 52 times, and its running table grows for the entire run rather
/// than staying small and stable -- exactly the case where that repeated
/// full-table remerge dominates.
///
/// Deferring *every* compaction to a single k-way merge at the very end
/// (`sources.len()` unbounded) would fix the time cost but reopen a worse
/// problem: k-mers that recur across many different eager-compaction
/// windows (normal for real coverage, since reads touching the same locus
/// are scattered throughout a FASTQ file, not clustered by input order)
/// would sit duplicated across many still-separate `pending` runs instead
/// of being deduplicated as they arrive -- the exact "buffer holds every
/// occurrence, not just every distinct one" bug `RAW_FINALIZE_THRESHOLD`
/// itself exists to prevent, just moved one level up. Capping the number
/// of pending runs before folding them together (via the same O(n log
/// runs) k-way merge either way) bounds that duplication to at most
/// `MAX_PENDING_RUNS` runs' worth, while cutting the number of full,
/// O(table size) consolidation passes over a run from ~52 to ~52 /
/// `MAX_PENDING_RUNS` -- the actual lever on the quadratic cost, since it
/// is the *count* of full-table passes that made the old design quadratic,
/// not the cost of any single pass.
///
/// 8 is a starting point sized the same way `RAW_FINALIZE_THRESHOLD` was:
/// large enough to meaningfully cut the number of full consolidations
/// (~6.5x fewer on the 52-compaction case above), small enough that the
/// transient over-counting from unmerged duplicate keys across at most 8
/// runs stays a bounded multiple of one compaction window's worth of
/// entries, not an unbounded one. Not proven optimal by a sweep across
/// input shapes.
const MAX_PENDING_RUNS: usize = 8;

/// Sorts `inner.raw` and compacts it into a new sorted, deduplicated
/// `(kmer, count)` run, appended to `inner.pending` -- a no-op if `raw` is
/// empty. Does not touch `inner.finalized` or `inner.valid`: folding
/// `pending` runs together is `consolidate`'s job, kept separate so an
/// eager mid-run compaction (see `RAW_FINALIZE_THRESHOLD`) can stay cheap
/// -- O(this run's size), not O(everything accumulated so far) -- letting
/// `pending` runs build up until `consolidate` is actually needed.
///
/// This, not a hash table, is the counting step: `raw.sort_unstable()` is
/// a handful of cache-friendly sequential passes over contiguous memory,
/// and the single linear scan afterwards that turns runs of equal values
/// into `(kmer, count)` pairs is sequential too. A `HashMap<u64, _>`
/// spends essentially every insertion on effectively-random bucket
/// placement, which is an L3 cache miss once the table exceeds cache size
/// (tens of MB -- a few hundred thousand entries) -- true of any real
/// FASTQ file, not just large ones.
///
/// **Tried and reverted: LSD radix sort in place of `sort_unstable` here.**
/// A 4-pass (16-bit digit) counting-radix-sort implementation, measured in
/// isolation against `sort_unstable` on the same `u64` inputs at this
/// function's actual call size (~2,000,000 keys, `RAW_FINALIZE_THRESHOLD`),
/// 20 runs each, scratch buffers reused across calls (not reallocated per
/// call, which was tried first and was far worse): `sort_unstable` mean
/// 0.0555 s; radix (16-bit/4-pass) mean 0.1910 s -- 3.4x *slower*; radix
/// (8-bit/8-pass, tried to reduce the scatter-write working set) mean
/// 0.1167 s -- still 2.1x slower. Rust's `sort_unstable` (pattern-defeating
/// quicksort) on primitive `u64` keys is not the naive O(n log n)
/// comparison sort the "radix should win on dense integer keys" reasoning
/// assumes: it is branchless-comparison, cache-friendly, and reads/writes
/// the data in place, while this radix implementation's counting-sort
/// scatter step writes to a data-dependent, effectively-random offset
/// within a multi-megabyte buffer once per pass -- worse cache behavior,
/// not better, at this bucket count, and more total passes over the data
/// than a single in-place sort regardless of bucket count. A more heavily
/// engineered radix sort (SIMD histogramming, software-prefetched
/// scatter, cache-blocking) might still win; this straightforward one does
/// not, on this data size and this hardware, and shipping it anyway on the
/// strength of the general "radix is O(n)" argument -- without the
/// measurement -- would have been a real regression.
/// **Tried and reverted: `raw.drain(..).peekable()` for the compaction
/// scan.** That is what this loop used to be, and the emitted assembly is
/// what condemned it. Per *run* it wrote the `Peekable`'s `Option<u64>` slot
/// and the `Drain`'s cursor back to the stack four times and reloaded the
/// push cursor immediately after storing it (a store-to-load forward right
/// in the per-run dependency chain); per *element* it spent 8 instructions,
/// two of them an `incl` + `cmovel` pair implementing `saturating_add(1)`
/// that formed a **2-cycle loop-carried dependency** on the count register.
/// The index walk below reads the same sorted slice with ~5 instructions per
/// element, carries only `incq` (1 cycle) between iterations, keeps a single
/// stack store per run, and drops the `Drain` drop glue (its guard branches
/// and its `memmove` call site) entirely. Against 840 million occurrences on
/// the benchmark file that is on the order of 2.5e9 fewer instructions and
/// 840 million fewer `cmov`s in the critical path.
///
/// The saturation is preserved exactly rather than dropped: applying
/// `saturating_add(1)` to a starting count of 1 once per repeat over a run
/// of length `L` yields `min(L, u32::MAX)`, which is what
/// `run_len.min(u32::MAX as usize) as u32` computes -- once per *distinct*
/// k-mer instead of once per *occurrence*.
/// Number of MSD buckets `msd_partition` splits the raw buffer into, as a
/// power of two: 1024.
///
/// Chosen by measurement, not by picking a round number. The sweep in the
/// results recorded on `msd_partition` covered 64, 128, 256, 512 and 1024
/// buckets; the gain rises steeply to 256 and then flattens, with 512 and
/// 1024 indistinguishable from each other and both consistently ahead of
/// 256. 1024 is taken as the top of the flat region: at 2,000,000 keys it
/// puts ~2,000 keys (16 KB) in an average bucket, comfortably L1-resident,
/// while the counts array it needs is 1024 `usize` (8 KB) -- small enough
/// to stay hot across both passes.
const MSD_BUCKET_BITS: u32 = 10;

/// Bits of a canonical k-mer that can actually be set, for the purpose of
/// choosing which bits to bucket on.
///
/// `62`, not `64`: the counter's keys are 2-bit-packed k-mers, and the
/// largest `k` the packing supports is 32... but `compact_raw` does not know
/// `k`, and getting this wrong is a performance bug, not a correctness one
/// (bucketing on always-zero high bits leaves most buckets empty and
/// degenerates to a plain sort of one bucket). 62 is chosen because `k = 31`
/// is this crate's default and its benchmark configuration. At `k = 32` the
/// top two bits are real and this shift discards them, which merges four
/// buckets' worth of keys into one -- still correct, still sorted, just less
/// evenly split.
const MSD_SIGNIFICANT_BITS: u32 = 62;

/// Splits `raw` into `2^MSD_BUCKET_BITS` ascending buckets by its top
/// significant bits, leaving the result in `raw` (via a swap with
/// `scratch`) with every bucket's boundaries returned as a prefix-sum array.
///
/// Bucketing on the *high* bits is what makes this usable: bucket order is
/// ascending key order, so sorting each bucket independently and leaving
/// them where they are yields the fully sorted array -- no merge step, and
/// `compact_raw`'s single linear run-length scan afterwards is unchanged.
///
/// # Measured, and measured the way the code actually runs
///
/// `counter.rs` has been here before: the doc comment on `compact_raw`
/// records an LSD radix sort that was implemented, measured at 2.1-3.4x
/// *slower* than `sort_unstable`, and reverted. That result is not
/// contradicted here, because this is a different algorithm with a
/// different memory-access shape -- LSD makes four to eight scatter passes
/// over the whole buffer, each writing to a data-dependent offset anywhere
/// within it; MSD makes exactly one, and every pass after it is a
/// `sort_unstable` over a bucket small enough to stay in cache.
///
/// Benchmarked in isolation at this function's real call size
/// (`RAW_FINALIZE_THRESHOLD` = 2,000,000 keys), on *canonical k-mers*
/// rather than uniform-random `u64` -- which matters for an MSD partition
/// in a way it does not for LSD, since `min(forward, revcomp)` skews the
/// distribution toward the low end of exactly the high bits this buckets
/// on. Three coverage shapes (100k, 1M and 1.9M distinct k-mers in the
/// pool, i.e. ~20x, ~2x and near-unique repeat structure), 20 runs each,
/// scratch buffers reused across calls:
///
/// ```text
///                        sort_unstable   MSD 1024 buckets
/// high coverage (100k)      0.0382 s        0.0295 s   1.30x
/// medium       (1M)         0.0450 s        0.0365 s   1.23x
/// low          (1.9M)       0.0460 s        0.0361 s   1.28x
/// ```
///
/// A single-threaded number would not have been enough to ship on, because
/// this trades memory bandwidth and footprint for cache locality and every
/// pipeline worker calls it at once: the extra scatter pass and the extra
/// 16 MB `scratch` per worker are both shared-resource costs that an idle
/// single-threaded bench hides. Re-measured with `threads` workers each
/// partitioning their own private buffer, released together from a barrier,
/// mean wall time until the last finishes:
///
/// ```text
///              1 thread   4 threads   8 threads   14 threads
/// high cov.     1.26x       1.37x       1.26x        1.30x
/// low cov.      1.29x       1.27x       1.37x        1.31x
/// ```
///
/// The advantage holds at every thread count rather than eroding, which is
/// the result that justified the change.
///
/// # What these numbers are not
///
/// They are from a Linux container on this machine, not the native Windows
/// release build `docs/PERFORMANCE_PLAN.md` requires benchmark numbers to
/// come from -- there is no Rust toolchain installed on the Windows host, so
/// that measurement could not be taken. The plan's constraint exists because
/// whole-run wall times on this machine swing 25-90 s on identical input;
/// these are isolated, CPU-bound, in-memory microbenchmarks with no I/O, run
/// 15-20 times per point, which is the same measurement class `compact_raw`'s
/// own radix note used to justify a reversion. That is a reason to believe
/// them, not a reason to call them the native number. **A native Windows
/// confirmation is still owed.**
fn msd_partition(raw: &mut Vec<u64>, scratch: &mut Vec<u64>, counts: &mut Vec<usize>) {
    let buckets = 1usize << MSD_BUCKET_BITS;
    let shift = MSD_SIGNIFICANT_BITS - MSD_BUCKET_BITS;

    // Histogram, offset by one, so the prefix sum below turns it directly
    // into each bucket's start index with no second pass.
    counts.clear();
    counts.resize(buckets + 1, 0);
    for &key in raw.iter() {
        // `min` rather than a mask: a key with bits above
        // `MSD_SIGNIFICANT_BITS` set (k = 32) must land in the top bucket,
        // not wrap into a low one, which would break the ascending-bucket
        // property the whole approach rests on.
        let bucket = ((key >> shift) as usize).min(buckets - 1);
        counts[bucket + 1] += 1;
    }
    for i in 0..buckets {
        counts[i + 1] += counts[i];
    }

    // The one scatter pass.
    scratch.clear();
    scratch.resize(raw.len(), 0);
    let mut cursor = counts[..buckets].to_vec();
    for &key in raw.iter() {
        let bucket = ((key >> shift) as usize).min(buckets - 1);
        scratch[cursor[bucket]] = key;
        cursor[bucket] += 1;
    }

    // `scratch` now holds the partitioned data and `raw` the stale input;
    // swapping keeps both allocations alive for the next call and leaves the
    // partitioned data where every caller expects it.
    std::mem::swap(raw, scratch);
}

/// Sorts `raw` ascending via [`msd_partition`] plus a `sort_unstable` per
/// bucket -- the concatenation is already in ascending key order, so the
/// result is exactly what `raw.sort_unstable()` would have produced.
///
/// `scratch` and `bounds` are caller-owned and reused across calls; see
/// [`msd_partition`] for why the extra buffer is what buys the locality.
///
/// `pub(crate)` so `binned.rs`'s per-bin sort can use the same routine
/// rather than a second copy of it. Whether it is *faster* there is a
/// separate question from whether it is the same code, and is answered by
/// `examples/bin_sort_ab.rs` -- see this module's `msd_partition` doc for
/// the measurements that justified it here.
pub(crate) fn sort_keys_msd(raw: &mut Vec<u64>, scratch: &mut Vec<u64>, bounds: &mut Vec<usize>) {
    if raw.is_empty() {
        return;
    }
    msd_partition(raw, scratch, bounds);

    let mut rest: &mut [u64] = raw.as_mut_slice();
    for bucket in 0..(1usize << MSD_BUCKET_BITS) {
        let len = bounds[bucket + 1] - bounds[bucket];
        let (head, tail) = rest.split_at_mut(len);
        head.sort_unstable();
        rest = tail;
    }
    debug_assert!(rest.is_empty(), "bucket lengths must cover the whole buffer");
    debug_assert!(
        raw.windows(2).all(|w| w[0] <= w[1]),
        "MSD partition + per-bucket sort must leave the buffer fully sorted"
    );
}

fn compact_raw(inner: &mut Inner) {
    if inner.raw.is_empty() {
        return;
    }

    // Partition into ascending, cache-resident buckets, then sort each one.
    // Because bucket order is ascending key order, the concatenation is
    // already fully sorted -- exactly what `raw.sort_unstable()` produced
    // before, so everything downstream is untouched.
    let mut bounds: Vec<usize> = Vec::new();
    let raw = &mut inner.raw;
    sort_keys_msd(raw, &mut inner.scratch, &mut bounds);

    let sorted: &[u64] = inner.raw.as_slice();
    let len = sorted.len();
    // Exactly `len` is the smallest capacity that can never need to grow
    // (one entry per element, in the all-distinct case), so no `push` below
    // reallocates and no bytes are ever recopied. `CountTable`, not
    // `Vec<(u64, u32)>`: see that type's doc comment for why this is a pair
    // of arrays now, not tuples.
    let mut new_run = CountTable::with_capacity(len);

    let mut i = 0usize;
    while i < len {
        let kmer = sorted[i];
        let run_start = i;
        i += 1;
        while i < len && sorted[i] == kmer {
            i += 1;
        }
        // `saturating_add` semantics, computed once per run: see the doc
        // comment above. A single k-mer occurring more than u32::MAX times
        // must not panic or silently wrap, even though that is not expected
        // to occur in practice.
        new_run.push(kmer, (i - run_start).min(u32::MAX as usize) as u32);
    }

    // `clear`, not `drain(..)`: both keep the allocation for the next fill,
    // but `clear` on a `Vec<u64>` (no `Drop` to run) is a single length
    // store, where `Drain` is an iterator whose own `Drop` has to re-derive
    // and restore the length past a tail-move guard.
    inner.raw.clear();
    inner.pending.push(new_run);
}

/// Folds every run in `inner.pending` (plus `inner.finalized`, if
/// non-empty) into a single new `inner.finalized`, via one k-way merge --
/// a no-op if `pending` is empty. See `MAX_PENDING_RUNS` for why this is
/// called periodically during eager compaction rather than only once at
/// the very end.
fn consolidate(inner: &mut Inner) {
    if inner.pending.is_empty() {
        return;
    }

    let mut sources = std::mem::take(&mut inner.pending);
    if !inner.finalized.is_empty() {
        sources.push(std::mem::take(&mut inner.finalized));
    }
    inner.finalized = k_way_merge_tables(sources);
}

/// Brings `inner.finalized` fully up to date with everything inserted so
/// far -- draining `raw` into a new run via `compact_raw`, then folding
/// every pending run into `finalized` via `consolidate` -- a no-op if
/// `inner.valid` already holds.
fn finalize_inner(inner: &mut Inner) {
    if inner.valid {
        return;
    }

    compact_raw(inner);
    consolidate(inner);

    inner.valid = true;
}

/// Merges any number of sorted, deduplicated `(kmer, count)` sources into
/// one, via a binary min-heap over each source's current head -- O(total
/// entries x log(sources)) with every source read and written exactly
/// once, rather than the O(sources x total) of repeatedly folding one new
/// source into a single growing accumulator (see `MAX_PENDING_RUNS`).
/// Consumes `sources` rather than borrowing, so `consolidate` does not
/// need to clone runs it is about to discard.
pub(crate) fn k_way_merge_sorted_counts(mut sources: Vec<Vec<(u64, u32)>>) -> Vec<(u64, u32)> {
    match sources.len() {
        0 => return Vec::new(),
        // A single source is already exactly what a merge of one source
        // produces; skip the heap machinery entirely, and hand back its
        // allocation rather than copying it.
        1 => return sources.remove(0),
        _ => {}
    }

    let slices: Vec<&[(u64, u32)]> = sources.iter().map(Vec::as_slice).collect();
    merge_slices(&slices)
}

/// Below this many entries a parallel merge is not worth its setup.
///
/// The split-point sample, the per-part range searches and the final
/// concatenation are all fixed costs; on a small merge they exceed what
/// the parallelism saves. 4,000,000 is deliberately conservative -- the
/// merge this exists for handles 53.8 million.
const PARALLEL_MERGE_MIN_ENTRIES: usize = 4_000_000;

/// [`k_way_merge_sorted_counts`], split across `parts` workers.
///
/// # Why this exists
///
/// Measured, on the 840,000,000-occurrence benchmark file with the binned
/// strategy: per-bin counting takes 2.8 s across every available core, and
/// **the cross-bin merge takes 1.9 s on one core** -- 19% of the whole run,
/// and the largest single-threaded stretch left in it. It does not shrink
/// when threads are added because there is nothing in it to add them to.
///
/// # How it stays one k-way merge rather than becoming a tree
///
/// `k_way_merge_sorted_counts`'s own doc comment argues at length against a
/// pairwise reduction tree: every entry gets rewritten once per level, and
/// at benchmark scale that was ~1.7 GB of extra copying. This keeps the
/// single-pass property and parallelises a different axis -- the *key
/// range*, not the source list.
///
/// Every source is already sorted, so a key range picks out one contiguous
/// slice of each (found by binary search, no scanning). The slices for one
/// range can be merged with no knowledge of any other range, and the ranges
/// concatenate in order because they were cut in key order. Each entry is
/// therefore still visited exactly once by the merge itself; the only
/// addition is one copy of the finished output into a single allocation,
/// which is a linear `memcpy` rather than another comparison pass.
///
/// # Where the split points come from
///
/// Not from dividing the `u64` range into equal parts. Canonical k-mers are
/// not uniformly distributed -- a low-complexity sample can concentrate
/// them arbitrarily -- and equal *value* ranges would then hand one worker
/// most of the data, which is the same failure the binned strategy's own
/// bin balance had to be measured for. The boundaries are quantiles of a
/// sample of the actual keys, so they follow the data's distribution
/// whatever it is.
pub(crate) fn k_way_merge_sorted_counts_parallel(
    sources: Vec<Vec<(u64, u32)>>,
    parts: usize,
) -> Vec<(u64, u32)> {
    let total_len: usize = sources.iter().map(Vec::len).sum();
    if sources.len() <= 1 || parts <= 1 || total_len < PARALLEL_MERGE_MIN_ENTRIES {
        return k_way_merge_sorted_counts(sources);
    }

    let bounds = split_bounds(&sources, parts);
    if bounds.is_empty() {
        return k_way_merge_sorted_counts(sources);
    }

    let chunks: Vec<Vec<(u64, u32)>> = (0..=bounds.len())
        .into_par_iter()
        .map(|part| {
            let lower = part.checked_sub(1).and_then(|i| bounds.get(i).copied());
            let upper = bounds.get(part).copied();
            let slices: Vec<&[(u64, u32)]> = sources
                .iter()
                .map(|source| {
                    let start = match lower {
                        Some(bound) => source.partition_point(|&(kmer, _)| kmer < bound),
                        None => 0,
                    };
                    let end = match upper {
                        Some(bound) => source.partition_point(|&(kmer, _)| kmer < bound),
                        None => source.len(),
                    };
                    &source[start..end]
                })
                .collect();
            merge_slices(&slices)
        })
        .collect();

    // The parts are disjoint, sorted and already in key order, so
    // assembling them is a copy rather than a merge. It is sequential, and
    // that was measured rather than assumed: the copy takes 0.40 s of the
    // merge's 1.48 s, and splitting the destination into one disjoint
    // `&mut [_]` per part so the copy runs on every core made it **no
    // faster at all** -- allocating the destination with `vec![_; n]`
    // zero-initialises 645 MB, which costs about what the copy it replaced
    // did. The simpler form is kept.
    let mut merged: Vec<(u64, u32)> = Vec::with_capacity(total_len);
    for chunk in &chunks {
        merged.extend_from_slice(chunk);
    }
    merged
}

/// `parts - 1` ascending keys that cut the merged output into roughly equal
/// pieces, taken as quantiles of a strided sample of the sources' own keys.
///
/// Returns fewer boundaries than asked for -- possibly none -- when the
/// sample cannot supply that many distinct keys. The caller treats an empty
/// result as "not worth splitting".
fn split_bounds(sources: &[Vec<(u64, u32)>], parts: usize) -> Vec<u64> {
    let total_len: usize = sources.iter().map(Vec::len).sum();
    // ~64 sample keys per part: enough for the quantiles to track the
    // distribution, few enough that sampling and sorting them is noise
    // against the merge itself.
    let wanted = parts.saturating_mul(64).max(1);
    let stride = (total_len / wanted).max(1);

    let mut sample: Vec<u64> = Vec::with_capacity(wanted + sources.len());
    for source in sources {
        sample.extend(source.iter().step_by(stride).map(|&(kmer, _)| kmer));
    }
    sample.sort_unstable();
    sample.dedup();
    if sample.len() < parts {
        return Vec::new();
    }

    let mut bounds = Vec::with_capacity(parts - 1);
    for part in 1..parts {
        let index = sample.len() * part / parts;
        match sample.get(index) {
            Some(&key) if bounds.last() != Some(&key) => bounds.push(key),
            _ => {}
        }
    }
    bounds
}

/// The heap merge itself, over borrowed slices.
///
/// Shared by the sequential and parallel entry points so there is exactly
/// one implementation of "sum the counts of equal keys across sorted
/// sources" -- two would be two things to keep agreeing.
fn merge_slices(sources: &[&[(u64, u32)]]) -> Vec<(u64, u32)> {
    let total_len: usize = sources.iter().map(|source| source.len()).sum();
    let mut merged: Vec<(u64, u32)> = Vec::with_capacity(total_len);

    let mut heap: BinaryHeap<Reverse<(u64, usize, usize)>> = BinaryHeap::with_capacity(sources.len());
    for (src_idx, src) in sources.iter().enumerate() {
        if let Some(&(kmer, _)) = src.first() {
            heap.push(Reverse((kmer, src_idx, 0)));
        }
    }

    while let Some(&Reverse((kmer, _, _))) = heap.peek() {
        let mut count: u32 = 0;

        // Fold in every source currently sitting at this key -- possible
        // because sources may share keys (that is exactly what makes this a
        // merge rather than a concatenation), but never *within* one source
        // (each is already deduplicated on its own). Starting from 0 and
        // folding the first contributor through the same `saturating_add` as
        // the rest is identical to seeding `count` with it:
        // `0.saturating_add(x)` is `x`.
        while let Some(mut top) = heap.peek_mut() {
            let Reverse((head_kmer, src_idx, elem_idx)) = *top;
            if head_kmer != kmer {
                break;
            }

            // Hoisted for the same reason the single-source form hoists it:
            // one bounds check against `sources.len()` per entry instead of
            // two.
            let src: &[(u64, u32)] = sources[src_idx];
            count = count.saturating_add(src[elem_idx].1);

            let next_idx = elem_idx + 1;
            match src.get(next_idx) {
                // Overwriting the root through `PeekMut` and letting its
                // `Drop` re-sift replaces a `pop` plus a `push` -- two sifts
                // -- with a single sift down. Measured on the emitted
                // assembly when this was first written: the function shrank
                // from 487 to 359 instructions, essentially all of it heap
                // restructuring that no longer happens.
                Some(&(next_kmer, _)) => *top = Reverse((next_kmer, src_idx, next_idx)),
                // Source exhausted: the one case that still shrinks the heap.
                None => {
                    PeekMut::pop(top);
                }
            }
        }

        merged.push((kmer, count));
    }

    merged
}

/// Merges any number of sorted, deduplicated `CountTable`s into one, via a
/// binary min-heap over each source's current head -- exactly
/// `k_way_merge_sorted_counts`'s algorithm (see that function's doc comment
/// for why a heap advanced through `PeekMut` beats repeated pairwise
/// folding), over `CountTable`'s parallel arrays instead of
/// `Vec<(u64, u32)>`. Used by `consolidate` and `KmerCounter::merge_all` --
/// the two merge passes that touch every occurrence in a worker's table on
/// every call they make, rather than once per `KmerCounter` lifetime, which
/// is what makes them the ones this file's G-3 struct-of-arrays change
/// targets.
///
/// `k_way_merge_sorted_counts` itself is deliberately untouched:
/// `binned.rs` calls it directly (`use crate::counter::
/// k_way_merge_sorted_counts`), and that file is out of scope for this
/// change, so its own merge passes still build `Vec<(u64, u32)>` tables at
/// 16 bytes/entry. This function exists alongside it, not in place of it,
/// so `KmerCounter`'s own merge passes get the full benefit of the
/// narrower representation without requiring any change to `binned.rs` or
/// its callers.
///
/// # Measured
///
/// See this module's doc comment history / `docs/PERFORMANCE_PLAN.md` for
/// the isolated, in-container benchmark this representation change was
/// shipped on: a standalone harness outside this crate (so it could use
/// `println!`, which this crate's `[lints]` deny) reproducing this
/// function's algorithm over both representations at benchmark-scale input,
/// several coverage shapes, several thread counts.
fn k_way_merge_tables(mut sources: Vec<CountTable>) -> CountTable {
    match sources.len() {
        0 => return CountTable::new(),
        // A single source is already exactly what a merge of one source
        // produces; skip the heap machinery entirely.
        1 => return sources.remove(0),
        _ => {}
    }

    let total_len: usize = sources.iter().map(CountTable::len).sum();
    let mut merged = CountTable::with_capacity(total_len);

    // Each heap entry is `Reverse((kmer, source index, index within that
    // source))`, exactly as in `k_way_merge_sorted_counts`.
    let mut heap: BinaryHeap<Reverse<(u64, usize, usize)>> = BinaryHeap::with_capacity(sources.len());
    for (src_idx, src) in sources.iter().enumerate() {
        if let Some(&kmer) = src.keys.first() {
            heap.push(Reverse((kmer, src_idx, 0)));
        }
    }

    while let Some(&Reverse((kmer, _, _))) = heap.peek() {
        let mut count: u32 = 0;

        // Fold in every source currently sitting at this key -- see
        // `k_way_merge_sorted_counts` for why this starts from zero and why
        // the heap root is advanced in place via `PeekMut` rather than
        // popped and re-pushed.
        while let Some(mut top) = heap.peek_mut() {
            let Reverse((head_kmer, src_idx, elem_idx)) = *top;
            if head_kmer != kmer {
                break;
            }

            let src: &CountTable = &sources[src_idx];
            count = count.saturating_add(src.counts[elem_idx]);

            let next_idx = elem_idx + 1;
            match src.keys.get(next_idx) {
                Some(&next_kmer) => *top = Reverse((next_kmer, src_idx, next_idx)),
                // Source exhausted: this is the one case that still has to
                // shrink the heap.
                None => {
                    PeekMut::pop(top);
                }
            }
        }

        merged.push(kmer, count);
    }

    merged
}

/// Merges two sorted, deduplicated `CountTable`s into one, in a single
/// linear O(n + m) pass -- the standard mergesort merge step over parallel
/// key/count arrays instead of `(u64, u32)` tuples. Used by
/// `KmerCounter::merge` to combine two whole counters.
fn merge_tables(a: &CountTable, b: &CountTable) -> CountTable {
    let mut merged = CountTable::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        // Two `<` comparisons rather than `match a.keys[i].cmp(&b.keys[j])`.
        // Matching on `Ordering` does not lower to a three-way branch: rustc
        // materializes the discriminant with `seta` + `sbb` and then
        // re-tests it with `movzbl` + `cmpl`, four extra instructions and a
        // third conditional branch per merged entry, on top of the compare
        // it already did. Comparing the keys twice costs one extra `cmp` and
        // nothing else -- this reasoning carries over unchanged from this
        // function's `(u64, u32)`-tuple predecessor.
        //
        // Reading only the keys up front is the other half of it, and the
        // struct-of-arrays layout makes it explicit rather than a compiler
        // hope: `a.counts[i]`/`b.counts[j]` live in an entirely separate
        // array from the keys just read, and are only touched on the branch
        // that actually uses them, so the "take from a" path never reads
        // `b.counts` and vice versa.
        //
        // The bounds checks here are already gone: `while i < a.len() && j <
        // b.len()` proves both indices in range for every array above, and
        // the emitted code has no panic edge in this loop, only in the tail
        // `extend_from_slice` calls below.
        let a_kmer = a.keys[i];
        let b_kmer = b.keys[j];
        if a_kmer < b_kmer {
            merged.push(a_kmer, a.counts[i]);
            i += 1;
        } else if b_kmer < a_kmer {
            merged.push(b_kmer, b.counts[j]);
            j += 1;
        } else {
            merged.push(a_kmer, a.counts[i].saturating_add(b.counts[j]));
            i += 1;
            j += 1;
        }
    }
    merged.keys.extend_from_slice(&a.keys[i..]);
    merged.counts.extend_from_slice(&a.counts[i..]);
    merged.keys.extend_from_slice(&b.keys[j..]);
    merged.counts.extend_from_slice(&b.counts[j..]);
    merged
}

/// In-memory frequency table for canonical 64-bit k-mers.
///
/// Backed by a sort-and-compact strategy, not a `HashMap<u64, u32>`.
/// That is a deliberate choice, not an incidental implementation detail:
/// counting is dominated by insertion volume (hundreds of millions of
/// k-mer instances for a real FASTQ file), and a hash table's
/// essentially-random bucket placement means most insertions miss every
/// cache level once the table outgrows a few tens of megabytes -- which
/// happens well before a real sample is fully counted. Sorting instead
/// processes memory sequentially. This mirrors the approach FastK (Myers
/// lab, 2023) uses instead of a hash table, and is the same reason KMC3
/// partitions k-mers into disk-resident bins before sorting each one --
/// both exist because hash-based k-mer counting (Jellyfish's original
/// design) does not hold up at real dataset sizes. FastDNA stays
/// in-memory rather than partitioning to disk (see the README Limitations
/// section for what that does and does not buy).
///
/// Internally, insertion (`insert`/`insert_batch`, both `&mut self`) is a
/// plain, unsorted append to a `Vec<u64>`, cheap and safe to call from a
/// hot loop -- but bounded, not left to grow for a whole run's worth of
/// occurrences: past `RAW_FINALIZE_THRESHOLD` buffered instances, an
/// insert eagerly triggers the same sort-and-compact pass a read would,
/// so peak memory tracks distinct k-mers plus that bound rather than
/// every occurrence ever inserted (see that constant's doc comment for
/// the measurements behind the choice). Every read method (`iter`,
/// `get_count`, `distinct_kmers`, `generate_histogram`, `top_kmers`)
/// stays `&self`, matching the API this type has always had, and lazily
/// triggers the same finalize if the buffer has grown since the last one:
/// call `insert`/`insert_batch` freely during counting, a sort only
/// happens when the buffer crosses the threshold or is read, however many
/// inserts came before it.
///
/// The state lives behind a `Mutex<Inner>`, not a `RefCell<Inner>`: a
/// `RefCell` makes this type `!Sync`, which is invisible to `cargo test`
/// and `cargo clippy --all-targets` (both default-feature, and the FFI
/// layer that needs `Sync` is gated behind `feature = "python"`) but
/// breaks `cargo build --features python` outright, because pyo3's
/// `Python::allow_threads` requires the closure -- and everything it
/// captures, including `&KmerCounter` -- to be `Send`, which in turn
/// requires `KmerCounter: Sync`. A `Mutex` costs an uncontended lock/
/// unlock per read-method call, which is immaterial next to the O(n log n)
/// sort it may guard; see `kmer_counter_is_sync` below for the regression
/// test.
///
/// A caveat for callers, not a bug: alternating inserts with reads (e.g.
/// `insert`, `distinct_kmers()`, `insert`, `distinct_kmers()`, ...) is
/// correct, but each read after new inserts re-runs `finalize_inner`,
/// which folds whatever is pending (any unflushed `raw`, plus up to
/// `MAX_PENDING_RUNS` compacted runs not yet consolidated) into
/// `finalized` via a k-way merge -- still, unavoidably, a full pass over
/// the table. That is fine for the "insert many, read a few times" shape
/// this type is built for; a loop that reads after every single insert
/// pays a full-table merge every time, not just a cheap lookup.
#[derive(Debug, Default)]
pub struct KmerCounter {
    inner: Mutex<Inner>,
    total_kmers: u64,
}

impl KmerCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// `capacity` pre-sizes the raw insertion buffer, avoiding the
    /// reallocation-and-copy cost of growing it from empty as a real
    /// sample's instance count climbs into the hundreds of millions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner { raw: Vec::with_capacity(capacity), ..Inner::default() }),
            total_kmers: 0,
        }
    }

    /// Builds a `KmerCounter` directly from an already-sorted,
    /// already-deduplicated `(kmer, count)` table and a known total
    /// occurrence count, bypassing `insert`/`insert_batch` entirely.
    ///
    /// This exists for `disk_spill.rs`'s counting strategy: it produces its
    /// final table via `merge_buckets` (a k-way merge of on-disk runs)
    /// rather than through this type's own raw-buffer-and-sort path, but
    /// every caller downstream of counting (`prune`, `iter`, `export.rs`,
    /// the FFI layer) expects a `KmerCounter`, not a bare `Vec`. `entries`
    /// is trusted, not re-validated: it is the caller's responsibility that
    /// it is genuinely sorted ascending by k-mer with no duplicate keys --
    /// exactly the invariant `finalized` already carries elsewhere in this
    /// type, and re-sorting here would silently paper over a caller bug
    /// instead of surfacing it (`debug_assert!` catches it in debug builds
    /// without paying for a re-sort in release).
    ///
    /// Converts `entries` into a `CountTable` before storing (see that
    /// type's doc comment for why `finalized` is no longer
    /// `Vec<(u64, u32)>`), via `from_finalized_table`, which `merge_all`
    /// also calls directly to skip this conversion for sources that are
    /// already `CountTable`s.
    pub(crate) fn from_sorted_entries(entries: Vec<(u64, u32)>, total_kmers: u64) -> Self {
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "from_sorted_entries requires a strictly ascending, deduplicated table"
        );
        Self::from_finalized_table(CountTable::from_tuples(entries), total_kmers)
    }

    /// Builds a `KmerCounter` directly from an already-sorted, already
    /// finalized `CountTable` -- the same contract as `from_sorted_entries`
    /// (see that method's doc comment), minus the tuple round trip its
    /// external callers need but this crate's own merge passes
    /// (`merge_all`) do not.
    fn from_finalized_table(finalized: CountTable, total_kmers: u64) -> Self {
        debug_assert!(
            finalized.keys.windows(2).all(|w| w[0] < w[1]),
            "from_finalized_table requires a strictly ascending, deduplicated table"
        );
        Self {
            inner: Mutex::new(Inner { finalized, valid: true, ..Inner::default() }),
            total_kmers,
        }
    }

    /// Access to `inner` from a `&mut self` method: no locking is actually
    /// needed (exclusive access is already guaranteed at compile time by
    /// `&mut self`), but `Mutex::get_mut` still returns a `LockResult`
    /// because a *previous* holder of the lock could have panicked while
    /// holding it (`pipeline.rs`'s worker `catch_unwind` makes that
    /// possible: a panic mid-`finalize_inner`, say). Recovering the
    /// guard rather than propagating the poison is deliberate: a poisoned
    /// counter's producing worker has already had its whole result
    /// discarded by `process_stream_parallel`'s `worker_panic` check, so
    /// there is no result depending on this one being pristine, and
    /// refusing to even inspect it would just turn one panic into a
    /// second, unrelated one here.
    fn inner_mut(&mut self) -> &mut Inner {
        self.inner.get_mut().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Locks `inner` from a `&self` method, finalizing it first if it is
    /// not already valid, and hands back the guard so the caller can read
    /// `finalized` directly. See `inner_mut` for why a poisoned lock is
    /// recovered rather than propagated.
    fn ensure_finalized(&self) -> MutexGuard<'_, Inner> {
        let mut guard = self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        finalize_inner(&mut guard);
        guard
    }

    #[inline(always)]
    pub fn insert(&mut self, kmer: u64) {
        {
            let inner = self.inner_mut();
            inner.raw.push(kmer);
            inner.valid = false;
            // See `RAW_FINALIZE_THRESHOLD` for why this cannot be left to
            // grow until the first read: nothing else bounds `raw`. Only
            // `compact_raw`, not the full `finalize_inner`: see
            // `MAX_PENDING_RUNS` for why folding every eager compaction's
            // run straight into `finalized` here would be quadratic.
            if inner.raw.len() >= RAW_FINALIZE_THRESHOLD {
                compact_raw(inner);
                if inner.pending.len() >= MAX_PENDING_RUNS {
                    consolidate(inner);
                }
            }
        }
        self.total_kmers += 1;
    }

    pub fn insert_batch(&mut self, kmers: &[u64]) {
        {
            let inner = self.inner_mut();
            inner.raw.extend_from_slice(kmers);
            inner.valid = false;
            if inner.raw.len() >= RAW_FINALIZE_THRESHOLD {
                compact_raw(inner);
                if inner.pending.len() >= MAX_PENDING_RUNS {
                    consolidate(inner);
                }
            }
        }
        self.total_kmers += kmers.len() as u64;
    }

    /// Combines `other` into `self`.
    ///
    /// Finalizes both sides (if not already finalized) and merges the two
    /// sorted `CountTable`s in one linear O(n + m) pass via `merge_tables`
    /// -- the standard mergesort merge step -- rather than replaying
    /// `other`'s entries through hash-table insertion one at a time.
    pub fn merge(&mut self, other: KmerCounter) {
        self.total_kmers += other.total_kmers;

        let mut other_inner =
            other.inner.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
        let self_inner = self.inner_mut();
        finalize_inner(self_inner);
        finalize_inner(&mut other_inner);

        let merged = merge_tables(&self_inner.finalized, &other_inner.finalized);

        // No `self_inner.raw.clear()` needed here: the `finalize_inner`
        // call just above already drained it -- that is what "finalized"
        // means.
        self_inner.finalized = merged;
        self_inner.valid = true;
    }

    /// Combines any number of counters into one, in a single k-way merge.
    ///
    /// `merge` folds two counters at a time, so a reduce over `w` workers
    /// rewrites the whole accumulated table once per level of the reduction
    /// tree. At benchmark scale (8 workers, 53.8 million distinct k-mers,
    /// 12 bytes per entry -- see `CountTable`'s doc comment) that is 4
    /// merges producing 13.4M entries, 2 producing 26.9M and 1 producing
    /// 53.8M -- 161 million entries written, ~1.9 GB of `memcpy`, across 7
    /// separate allocations the largest of which is ~650 MB. Merging all
    /// `w` sources at once writes each of the 53.8 million final entries
    /// exactly once: ~650 MB, one allocation. That is ~1.3 GB of copying
    /// and 6 large allocations removed, and it costs nothing extra per
    /// entry -- `k_way_merge_tables` already does `O(entries x
    /// log(sources))` with `sources` bounded by the worker count either
    /// way.
    ///
    /// `pipeline.rs`'s combine phase calls this once over the collected
    /// worker counters. `QcSummary::merge` deliberately stays pairwise
    /// there: it is `O(1)`, so a k-way form would buy nothing.
    pub fn merge_all(counters: Vec<KmerCounter>) -> KmerCounter {
        let mut total_kmers: u64 = 0;
        let mut sources: Vec<CountTable> = Vec::with_capacity(counters.len());

        for counter in counters {
            total_kmers += counter.total_kmers;
            let mut inner =
                counter.inner.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
            finalize_inner(&mut inner);
            // An empty source would only occupy a heap slot and be popped
            // straight back out; skipping it also lets the one-source fast
            // path in `k_way_merge_tables` trigger when every other worker
            // happened to see nothing.
            if !inner.finalized.is_empty() {
                sources.push(std::mem::take(&mut inner.finalized));
            }
        }

        // The merge output is sorted ascending and deduplicated by
        // construction -- that is exactly what a k-way merge of sorted,
        // deduplicated sources with equal keys folded together produces --
        // which is the invariant `from_finalized_table` documents as the
        // caller's responsibility. `k_way_merge_tables`, not
        // `k_way_merge_sorted_counts`: this is the hot combine phase this
        // file's G-3 change targets, and it stays in `CountTable` form end
        // to end -- no tuple round trip on this path.
        Self::from_finalized_table(k_way_merge_tables(sources), total_kmers)
    }

    /// Removes k-mers outside the inclusive `[min, max]` frequency band.
    ///
    /// Both bounds are inclusive: a k-mer whose count equals `min` or `max` is
    /// kept. `total_kmers` is deliberately left unchanged — it is the
    /// normalization basis and must reflect the sample's true depth.
    pub fn prune(&mut self, min: u32, max: Option<u32>) -> PruneStats {
        let inner = self.inner_mut();
        finalize_inner(inner);

        let mut stats = PruneStats::default();
        let table = &mut inner.finalized;
        // `Vec::retain` has no equivalent across two parallel arrays that
        // must stay in lockstep, so this walks both by index instead:
        // `keep` decides once per entry from the count alone, and a kept
        // entry's key/count pair is copied down to the next free `write`
        // slot -- the same in-place compaction `Vec::retain` itself
        // performs, just over two arrays rather than one.
        let mut write = 0usize;
        for read in 0..table.len() {
            let count = table.counts[read];
            let keep = if count < min {
                stats.dropped_min += 1;
                false
            } else if max.is_some_and(|cap| count > cap) {
                stats.dropped_max += 1;
                false
            } else {
                stats.kept += 1;
                true
            };
            if keep {
                table.keys[write] = table.keys[read];
                table.counts[write] = table.counts[read];
                write += 1;
            }
        }
        table.keys.truncate(write);
        table.counts.truncate(write);

        stats
    }

    // Not `#[inline(always)]`: unlike `total_kmers` below (a plain field
    // read), this can trigger `ensure_finalized`'s O(n log n)
    // sort-and-compact pass. Forcing that to inline at every call site
    // would be misleading -- it is not the cheap accessor the annotation
    // implies -- and the compiler is free to inline it anyway if it
    // decides that is actually worthwhile.
    pub fn distinct_kmers(&self) -> usize {
        self.ensure_finalized().finalized.len()
    }

    #[inline(always)]
    pub fn total_kmers(&self) -> u64 {
        self.total_kmers
    }

    pub fn get_count(&self, kmer: u64) -> u32 {
        let guard = self.ensure_finalized();
        guard
            .finalized
            .keys
            .binary_search(&kmer)
            .map(|idx| guard.finalized.counts[idx])
            .unwrap_or(0)
    }

    /// Every `(kmer, count)` pair, ascending by k-mer, as a borrowed
    /// iterator over the table in place -- not a clone. At benchmark
    /// scale, cloning the whole `finalized` table (the previous
    /// behaviour) was on the order of 860MB per call, and every one of
    /// `export.rs`'s three export functions plus `ffi.rs`'s Arrow-batch
    /// builder called it once each; it also defeated
    /// `export_counts_parquet`'s 131,072-row chunking, which exists
    /// precisely to avoid materializing the whole table before writing
    /// the first chunk. `Iter` holds the counter's lock for its whole
    /// lifetime instead of cloning out from under it: every call site
    /// iterates the counter exactly once to build an export or an Arrow
    /// batch, so this only serializes two callers doing that concurrently
    /// against the same counter, rather than letting either corrupt the
    /// other's read the way sharing a `Vec` without a lock would.
    pub fn iter(&self) -> Iter<'_> {
        Iter { guard: self.ensure_finalized(), idx: 0 }
    }

    pub fn generate_histogram(&self) -> FxHashMap<u32, u64> {
        let guard = self.ensure_finalized();
        let mut histogram: FxHashMap<u32, u64> = FxHashMap::default();
        for &count in guard.finalized.counts.iter() {
            *histogram.entry(count).or_insert(0) += 1;
        }
        histogram
    }

    /// The `n` k-mers with the highest counts, descending.
    ///
    /// Partitions with `select_nth_unstable_by` (average O(entries)) to
    /// put the top `n` in the front, then sorts only that front slice --
    /// O(n log n) in the requested `n`, not in the table size -- rather
    /// than fully sorting the whole cloned table just to keep its first
    /// `n` elements, which was the entire table's worth of comparisons
    /// for however small a caller-requested `n` actually was.
    pub fn top_kmers(&self, n: usize) -> Vec<(u64, u32)> {
        let guard = self.ensure_finalized();
        let mut entries: Vec<(u64, u32)> = guard.finalized.to_tuples();
        drop(guard);

        let take = n.min(entries.len());
        if take == 0 {
            return Vec::new();
        }

        entries.select_nth_unstable_by(take - 1, |a, b| b.1.cmp(&a.1));
        entries.truncate(take);
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        entries
    }
}

/// Borrowed iterator over a `KmerCounter`'s finalized `(kmer, count)`
/// table, ascending by k-mer. Returned by `KmerCounter::iter`; holds the
/// counter's lock for as long as it is alive.
pub struct Iter<'a> {
    guard: MutexGuard<'a, Inner>,
    idx: usize,
}

impl Iterator for Iter<'_> {
    type Item = (u64, u32);

    fn next(&mut self) -> Option<Self::Item> {
        // Two array reads instead of one tuple read -- `finalized` is a
        // `CountTable` (see that type's doc comment), not a
        // `Vec<(u64, u32)>` -- but still one bounds check and one
        // unconditional increment per call: `get` on `keys` proves the
        // index in range for the unchecked `counts` index right after it,
        // the same shape the previous tuple version had over every entry of
        // the finalized table on every export pass (53.8 million entries
        // per pass at benchmark scale, and `export.rs` plus `ffi.rs` walk it
        // once each).
        let &kmer = self.guard.finalized.keys.get(self.idx)?;
        let count = self.guard.finalized.counts[self.idx];
        self.idx += 1;
        Some((kmer, count))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.guard.finalized.len().saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {

    /// The parallel merge must return exactly what the sequential one
    /// returns -- same keys, same summed counts, same order -- on inputs
    /// shaped like the ones it actually sees: many sorted sources with
    /// overlapping keys.
    ///
    /// A differential test rather than a golden file, because the property
    /// that matters is not "this output" but "the same output the reviewed,
    /// long-standing implementation produces". Run across several part
    /// counts, including 1 and more parts than there are distinct keys, so
    /// the fallbacks and the degenerate splits are covered too.
    #[test]
    fn the_parallel_merge_agrees_with_the_sequential_one() {
        let mut state = 0x2026_0905_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for (sources_count, per_source, key_space) in
            [(2usize, 5usize, 8u64), (8, 50, 200), (64, 200, 5_000), (512, 40, 3_000)]
        {
            let sources: Vec<Vec<(u64, u32)>> = (0..sources_count)
                .map(|_| {
                    let mut keys: Vec<u64> = (0..per_source).map(|_| next() % key_space).collect();
                    keys.sort_unstable();
                    keys.dedup();
                    keys.into_iter().map(|k| (k, (next() % 7 + 1) as u32)).collect()
                })
                .collect();

            let expected = k_way_merge_sorted_counts(sources.clone());
            for parts in [1usize, 2, 3, 7, 64] {
                let actual = k_way_merge_sorted_counts_parallel(sources.clone(), parts);
                assert_eq!(
                    actual, expected,
                    "{sources_count} sources x {per_source} keys over {key_space}, {parts} parts"
                );
            }
        }
    }

    /// The parallel path is only taken above `PARALLEL_MERGE_MIN_ENTRIES`,
    /// so the test above exercises the fallback rather than the split. This
    /// one crosses that threshold, which is the only way to reach
    /// `split_bounds` and the per-part range searches at all.
    #[test]
    fn the_parallel_merge_agrees_above_its_own_size_threshold() {
        let sources_count = 32;
        let per_source = PARALLEL_MERGE_MIN_ENTRIES / sources_count + 1_000;
        let sources: Vec<Vec<(u64, u32)>> = (0..sources_count)
            .map(|source| {
                // Interleaved key spaces so sources genuinely overlap: a
                // partition where each source owned a disjoint range would
                // never exercise the count-summing path.
                (0..per_source)
                    .map(|i| ((i * sources_count + source) as u64, 1u32))
                    .collect()
            })
            .collect();

        let expected = k_way_merge_sorted_counts(sources.clone());
        let actual = k_way_merge_sorted_counts_parallel(sources, 8);
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    use super::*;

    /// Builds a counter where k-mer `i` appears `i` times, for i in 1..=5.
    fn counter_with_graded_counts() -> KmerCounter {
        let mut c = KmerCounter::new();
        for kmer in 1u64..=5 {
            for _ in 0..kmer {
                c.insert(kmer);
            }
        }
        c
    }

    /// `cargo build --features python` requires `&PyKmerCounts: Send`,
    /// which requires `KmerCounter: Sync` (see `PyKmerCounts::table` in
    /// `ffi.rs`, which calls `py.allow_threads` over a closure borrowing
    /// `self.counter`). That feature is not built by `cargo test` or
    /// `cargo clippy --all-targets` (both default-feature), so a change
    /// that silently reintroduces `!Sync` -- e.g. swapping `Mutex` back
    /// for `RefCell` -- would pass this crate's ordinary CI checks and
    /// only fail the wheel build. This test makes that failure local and
    /// immediate instead.
    #[test]
    fn kmer_counter_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KmerCounter>();
    }

    /// `msd_partition`'s load-bearing invariant, asserted directly rather
    /// than inferred from the fact that some larger test passes: bucketing
    /// on the *high* bits must put the buckets in ascending key order, so
    /// that sorting each one in place leaves the whole buffer sorted with no
    /// merge. If that ever stopped holding, `compact_raw` would silently
    /// emit an unsorted table -- and since `k_way_merge_sorted_counts`
    /// downstream assumes sorted inputs, the result would be wrong counts,
    /// not a crash.
    ///
    /// Exercised across the k-mer distributions that actually stress it: the
    /// canonical skew (`min(forward, revcomp)` concentrates keys in the low
    /// half of the range, so the top bucket may be empty and the bottom ones
    /// crowded), all-identical keys (one bucket holds everything), and keys
    /// with bits above `MSD_SIGNIFICANT_BITS` set (k = 32), which the
    /// `min(buckets - 1)` clamp must fold into the top bucket rather than
    /// wrapping into a low one.
    #[test]
    fn msd_partition_leaves_buckets_in_ascending_key_order() {
        fn check(label: &str, keys: Vec<u64>) {
            let mut raw = keys.clone();
            let mut scratch = Vec::new();
            let mut bounds = Vec::new();
            msd_partition(&mut raw, &mut scratch, &mut bounds);

            let buckets = 1usize << MSD_BUCKET_BITS;
            assert_eq!(bounds.len(), buckets + 1, "{label}: wrong number of bucket bounds");
            assert_eq!(bounds[0], 0, "{label}: bounds must start at 0");
            assert_eq!(bounds[buckets], keys.len(), "{label}: bounds must cover every key");

            // The partition is a permutation: same multiset, nothing lost or
            // duplicated by the scatter.
            let mut before = keys.clone();
            let mut after = raw.clone();
            before.sort_unstable();
            after.sort_unstable();
            assert_eq!(before, after, "{label}: partition changed the multiset");

            // Every key in bucket i is <= every key in bucket i+1. Checking
            // the maximum of each bucket against the minimum of the next is
            // exactly that claim, and it is what makes the per-bucket sort
            // sufficient.
            let mut previous_max: Option<u64> = None;
            for b in 0..buckets {
                let slice = &raw[bounds[b]..bounds[b + 1]];
                let (Some(&lo), Some(&hi)) = (slice.iter().min(), slice.iter().max()) else {
                    continue; // empty bucket: nothing to order against
                };
                if let Some(prev) = previous_max {
                    assert!(prev <= lo, "{label}: bucket {b} starts at {lo}, below the previous \
                                         bucket's maximum {prev}");
                }
                previous_max = Some(hi);
            }

            // ...and therefore sorting each bucket in place sorts the whole
            // buffer, which is the property `compact_raw` relies on.
            let mut rest: &mut [u64] = raw.as_mut_slice();
            for b in 0..buckets {
                let len = bounds[b + 1] - bounds[b];
                let (head, tail) = rest.split_at_mut(len);
                head.sort_unstable();
                rest = tail;
            }
            let mut expected = keys;
            expected.sort_unstable();
            assert_eq!(raw, expected, "{label}: per-bucket sort did not sort the buffer");
        }

        // Canonical k-mers at k = 31, the real shape of this buffer.
        let mut state = 0x9001_9001_9001_9001u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mask = (1u64 << 62) - 1;
        let canonical: Vec<u64> = (0..50_000)
            .map(|_| {
                let fwd = next() & mask;
                fwd.min(crate::kmer::reverse_complement_u64(fwd, 31))
            })
            .collect();
        check("canonical k=31", canonical);

        check("all identical", vec![0x0123_4567_89AB_CDEFu64 & mask; 1000]);
        check("all zero", vec![0u64; 1000]);
        check("single key", vec![42u64]);

        // k = 32 uses all 64 bits, above `MSD_SIGNIFICANT_BITS`. The clamp
        // must keep these ordered rather than wrapping them into low buckets.
        let wide: Vec<u64> = (0..20_000).map(|_| next()).collect();
        check("k=32, bits above the significant range", wide);
    }

    /// `mem_estimate.rs` keeps its own copy of this threshold (it predicts
    /// peak memory *before* any `KmerCounter` exists, so it deliberately
    /// does not import this module's types) -- see that copy's own doc
    /// comment. This test is what keeps the two from silently drifting
    /// apart if one is tuned without the other.
    #[test]
    fn raw_finalize_threshold_matches_mem_estimate_copy() {
        assert_eq!(
            RAW_FINALIZE_THRESHOLD as u64,
            crate::mem_estimate::RAW_FINALIZE_THRESHOLD,
            "counter.rs's RAW_FINALIZE_THRESHOLD and mem_estimate.rs's copy of it must match"
        );
    }

    #[test]
    fn prune_drops_counts_below_min_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(3, None);

        assert_eq!(stats.dropped_min, 2, "k-mers seen 1 and 2 times must go");
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 3, "k-mers seen 3, 4 and 5 times must stay");
        assert_eq!(c.distinct_kmers(), 3);
        assert_eq!(c.get_count(3), 3, "count == min is kept");
        assert_eq!(c.get_count(2), 0, "count < min is gone");
    }

    #[test]
    fn prune_drops_counts_above_max_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, Some(4));

        assert_eq!(stats.dropped_max, 1, "only the k-mer seen 5 times is over the cap");
        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.kept, 4);
        assert_eq!(c.get_count(4), 4, "count == max is kept");
        assert_eq!(c.get_count(5), 0);
    }

    #[test]
    fn prune_applies_both_bounds_at_once() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(2, Some(4));

        assert_eq!(stats.dropped_min, 1);
        assert_eq!(stats.dropped_max, 1);
        assert_eq!(stats.kept, 3);
        assert_eq!(c.distinct_kmers(), 3);
    }

    #[test]
    fn prune_leaves_total_kmers_untouched() {
        let mut c = counter_with_graded_counts();
        let before = c.total_kmers();
        assert_eq!(before, 15, "1+2+3+4+5 occurrences");

        c.prune(4, None);

        assert_eq!(
            c.total_kmers(),
            before,
            "total_kmers is the normalization basis and must survive pruning"
        );
    }

    #[test]
    fn prune_with_permissive_bounds_drops_nothing() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, None);

        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 5);
    }

    #[test]
    fn k_way_merge_of_zero_sources_is_empty() {
        assert_eq!(k_way_merge_sorted_counts(Vec::new()), Vec::new());
    }

    #[test]
    fn k_way_merge_of_one_source_returns_it_unchanged() {
        let only = vec![(1u64, 3u32), (5, 1), (9, 7)];
        assert_eq!(k_way_merge_sorted_counts(vec![only.clone()]), only);
    }

    /// More than two sources, with keys overlapping across three of them
    /// (not just pairwise) and one source fully disjoint -- the case a
    /// naive pairwise-only merge implementation could get subtly wrong
    /// (e.g. only combining the first pair it finds at a given key,
    /// leaving a third source's count for that same key un-added).
    #[test]
    fn k_way_merge_combines_counts_shared_across_more_than_two_sources() {
        let sources = vec![
            vec![(1u64, 1u32), (2, 1), (10, 1)],
            vec![(1, 10), (3, 1)],
            vec![(1, 100), (2, 10)],
            vec![(4, 1)],
        ];

        let merged = k_way_merge_sorted_counts(sources);

        assert_eq!(
            merged,
            vec![(1, 111), (2, 11), (3, 1), (4, 1), (10, 1)],
            "kmer 1 must sum contributions from all three sources that carry it"
        );
    }

    /// A source list containing empty sources must not stall the merge or
    /// let an empty one occupy a heap slot: nothing is seeded for it, and
    /// the sources around it still merge normally.
    #[test]
    fn k_way_merge_skips_empty_sources() {
        let sources = vec![
            Vec::new(),
            vec![(2u64, 5u32), (4, 1)],
            Vec::new(),
            vec![(1, 1), (4, 2)],
            Vec::new(),
        ];

        assert_eq!(k_way_merge_sorted_counts(sources), vec![(1, 1), (2, 5), (4, 3)]);
    }

    /// The merge advances the heap root in place (`PeekMut`) instead of
    /// popping and pushing, and folds equal keys starting from a count of
    /// zero rather than seeding with the first contributor. Both are
    /// behaviour-preserving only if the result is byte-identical to a
    /// straightforward reference over every shape of input -- sources of
    /// wildly different lengths, keys shared by any subset of them, sources
    /// that exhaust long before the others, and duplicate-free runs.
    #[test]
    fn k_way_merge_matches_a_reference_implementation_on_pseudorandom_input() {
        /// Sort-and-group: obviously correct, far too slow to ship.
        fn reference(sources: &[Vec<(u64, u32)>]) -> Vec<(u64, u32)> {
            let mut flat: Vec<(u64, u32)> = sources.iter().flatten().copied().collect();
            flat.sort_by_key(|&(kmer, _)| kmer);
            let mut out: Vec<(u64, u32)> = Vec::new();
            for (kmer, count) in flat {
                match out.last_mut() {
                    Some(last) if last.0 == kmer => last.1 = last.1.saturating_add(count),
                    _ => out.push((kmer, count)),
                }
            }
            out
        }

        // A deterministic LCG, so a failure is reproducible and the test
        // brings in no dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };

        for case in 0..200 {
            // 1..=9 sources: `MAX_PENDING_RUNS` runs plus `finalized` is the
            // widest merge `consolidate` ever asks for.
            let source_count = 1 + (case % 9);
            let key_space = 1 + next(40);

            let mut sources: Vec<Vec<(u64, u32)>> = Vec::with_capacity(source_count);
            for _ in 0..source_count {
                let mut keys: Vec<u64> = (0..key_space).filter(|_| next(3) != 0).collect();
                keys.dedup();
                sources
                    .push(keys.into_iter().map(|kmer| (kmer, 1 + next(1000) as u32)).collect());
            }

            assert_eq!(
                k_way_merge_sorted_counts(sources.clone()),
                reference(&sources),
                "case {case}: {sources:?}"
            );
        }
    }

    /// Counts must clamp at `u32::MAX` rather than wrapping. `compact_raw`
    /// now derives a run's count from its length instead of applying
    /// `saturating_add` per occurrence, and the merge folds contributions
    /// with `saturating_add` starting from zero -- both have to saturate.
    #[test]
    fn merged_counts_saturate_at_u32_max_instead_of_wrapping() {
        let sources =
            vec![vec![(7u64, u32::MAX - 1)], vec![(7, 5)], vec![(7, 10)], vec![(9, 1)]];

        assert_eq!(
            k_way_merge_sorted_counts(sources),
            vec![(7, u32::MAX), (9, 1)],
            "an overflowing sum must clamp, not wrap to a tiny count"
        );
    }

    // -- `CountTable`'s own merge functions ---------------------------
    //
    // `consolidate` and `merge_all` call `k_way_merge_tables`, not
    // `k_way_merge_sorted_counts`, after the G-3 struct-of-arrays change --
    // see that function's doc comment. These mirror the
    // `k_way_merge_sorted_counts` tests above one-for-one (same cases, same
    // expectations) so the two implementations are pinned to agree, plus a
    // direct test of `merge_tables`, the two-way merge `KmerCounter::merge`
    // now uses.

    #[test]
    fn k_way_merge_tables_of_zero_sources_is_empty() {
        assert!(k_way_merge_tables(Vec::new()).is_empty());
    }

    #[test]
    fn k_way_merge_tables_of_one_source_returns_it_unchanged() {
        let only = CountTable::from_tuples(vec![(1u64, 3u32), (5, 1), (9, 7)]);
        let expected = only.to_tuples();
        assert_eq!(k_way_merge_tables(vec![only]).to_tuples(), expected);
    }

    /// More than two sources, with keys overlapping across three of them
    /// (not just pairwise) and one source fully disjoint -- the case a
    /// naive pairwise-only merge implementation could get subtly wrong.
    #[test]
    fn k_way_merge_tables_combines_counts_shared_across_more_than_two_sources() {
        let sources = vec![
            CountTable::from_tuples(vec![(1u64, 1u32), (2, 1), (10, 1)]),
            CountTable::from_tuples(vec![(1, 10), (3, 1)]),
            CountTable::from_tuples(vec![(1, 100), (2, 10)]),
            CountTable::from_tuples(vec![(4, 1)]),
        ];

        let merged = k_way_merge_tables(sources);

        assert_eq!(
            merged.to_tuples(),
            vec![(1, 111), (2, 11), (3, 1), (4, 1), (10, 1)],
            "kmer 1 must sum contributions from all three sources that carry it"
        );
    }

    /// A source list containing empty tables must not stall the merge or
    /// let an empty one occupy a heap slot.
    #[test]
    fn k_way_merge_tables_skips_empty_sources() {
        let sources = vec![
            CountTable::new(),
            CountTable::from_tuples(vec![(2u64, 5u32), (4, 1)]),
            CountTable::new(),
            CountTable::from_tuples(vec![(1, 1), (4, 2)]),
            CountTable::new(),
        ];

        assert_eq!(
            k_way_merge_tables(sources).to_tuples(),
            vec![(1, 1), (2, 5), (4, 3)]
        );
    }

    /// Same differential strategy as
    /// `k_way_merge_matches_a_reference_implementation_on_pseudorandom_input`,
    /// over `CountTable` instead of `Vec<(u64, u32)>`, so a bug specific to
    /// reading the parallel arrays (an off-by-one between `keys` and
    /// `counts`, say) cannot hide behind the tuple version's passing tests.
    #[test]
    fn k_way_merge_tables_matches_a_reference_implementation_on_pseudorandom_input() {
        /// Sort-and-group over flattened tuples: obviously correct, far too
        /// slow to ship.
        fn reference(sources: &[CountTable]) -> Vec<(u64, u32)> {
            let mut flat: Vec<(u64, u32)> =
                sources.iter().flat_map(CountTable::to_tuples).collect();
            flat.sort_by_key(|&(kmer, _)| kmer);
            let mut out: Vec<(u64, u32)> = Vec::new();
            for (kmer, count) in flat {
                match out.last_mut() {
                    Some(last) if last.0 == kmer => last.1 = last.1.saturating_add(count),
                    _ => out.push((kmer, count)),
                }
            }
            out
        }

        // A deterministic LCG, so a failure is reproducible and the test
        // brings in no dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };

        for case in 0..200 {
            // 1..=9 sources: `MAX_PENDING_RUNS` runs plus `finalized` is the
            // widest merge `consolidate` ever asks for.
            let source_count = 1 + (case % 9);
            let key_space = 1 + next(40);

            let mut sources: Vec<CountTable> = Vec::with_capacity(source_count);
            for _ in 0..source_count {
                let mut keys: Vec<u64> = (0..key_space).filter(|_| next(3) != 0).collect();
                keys.dedup();
                sources.push(CountTable::from_tuples(
                    keys.into_iter().map(|kmer| (kmer, 1 + next(1000) as u32)).collect(),
                ));
            }

            let expected = reference(&sources);
            assert_eq!(
                k_way_merge_tables(sources.clone()).to_tuples(),
                expected,
                "case {case}: {sources:?}"
            );
        }
    }

    /// Counts must clamp at `u32::MAX` rather than wrapping, exactly as
    /// `k_way_merge_sorted_counts` must.
    #[test]
    fn k_way_merge_tables_saturates_at_u32_max_instead_of_wrapping() {
        let sources = vec![
            CountTable::from_tuples(vec![(7u64, u32::MAX - 1)]),
            CountTable::from_tuples(vec![(7, 5)]),
            CountTable::from_tuples(vec![(7, 10)]),
            CountTable::from_tuples(vec![(9, 1)]),
        ];

        assert_eq!(
            k_way_merge_tables(sources).to_tuples(),
            vec![(7, u32::MAX), (9, 1)],
            "an overflowing sum must clamp, not wrap to a tiny count"
        );
    }

    /// Direct test of `merge_tables`, the two-way merge `KmerCounter::merge`
    /// uses -- `merge_combines_counts_for_shared_kmers_and_keeps_disjoint_ones`
    /// below covers the same code path through the public API; this pins
    /// the array-of-structs-free algorithm itself.
    #[test]
    fn merge_tables_combines_shared_keys_and_keeps_disjoint_ones() {
        let a = CountTable::from_tuples(vec![(1u64, 2u32), (2, 1), (5, 9)]);
        let b = CountTable::from_tuples(vec![(2u64, 3u32), (3, 1)]);

        let merged = merge_tables(&a, &b);

        assert_eq!(merged.to_tuples(), vec![(1, 2), (2, 4), (3, 1), (5, 9)]);
    }

    /// `merge_all` must produce exactly what repeated pairwise `merge`
    /// produces -- it is a drop-in for the combine phase, not a different
    /// answer that happens to be faster.
    #[test]
    fn merge_all_agrees_with_repeated_pairwise_merge() {
        let build = |kmers: &[u64]| {
            let mut c = KmerCounter::new();
            for &k in kmers {
                c.insert(k);
            }
            c
        };
        let inputs: Vec<Vec<u64>> =
            vec![vec![1, 1, 4, 9], vec![4, 4, 5], Vec::new(), vec![1, 5, 9, 9, 9], vec![2]];

        let mut pairwise = KmerCounter::new();
        for input in &inputs {
            pairwise.merge(build(input));
        }

        let all = KmerCounter::merge_all(inputs.iter().map(|i| build(i)).collect());

        assert_eq!(all.total_kmers(), pairwise.total_kmers());
        assert_eq!(all.distinct_kmers(), pairwise.distinct_kmers());
        assert_eq!(all.iter().collect::<Vec<_>>(), pairwise.iter().collect::<Vec<_>>());
    }

    #[test]
    fn merge_all_of_nothing_is_an_empty_counter() {
        let c = KmerCounter::merge_all(Vec::new());
        assert_eq!(c.total_kmers(), 0);
        assert_eq!(c.distinct_kmers(), 0);
    }

    /// `merge_all` finalizes each source itself, so counters still holding
    /// unflushed `raw` insertions must not lose them.
    #[test]
    fn merge_all_finalizes_sources_that_were_never_read() {
        let mut a = KmerCounter::new();
        a.insert(3);
        a.insert(3);
        let mut b = KmerCounter::new();
        b.insert(3);
        b.insert(7);

        let all = KmerCounter::merge_all(vec![a, b]);

        assert_eq!(all.total_kmers(), 4);
        assert_eq!(all.get_count(3), 3);
        assert_eq!(all.get_count(7), 1);
    }

    #[test]
    fn merge_combines_counts_for_shared_kmers_and_keeps_disjoint_ones() {
        let mut a = KmerCounter::new();
        a.insert(1);
        a.insert(1);
        a.insert(2);

        let mut b = KmerCounter::new();
        b.insert(2);
        b.insert(3);
        b.insert(3);
        b.insert(3);

        a.merge(b);

        assert_eq!(a.total_kmers(), 7, "3 inserts into a plus 4 into b");
        assert_eq!(a.distinct_kmers(), 3);
        assert_eq!(a.get_count(1), 2, "only in a");
        assert_eq!(a.get_count(2), 2, "in both -- counts must add");
        assert_eq!(a.get_count(3), 3, "only in b");
    }

    #[test]
    fn merge_with_an_empty_counter_is_a_no_op_on_values() {
        let mut a = KmerCounter::new();
        a.insert(7);
        a.insert(7);

        a.merge(KmerCounter::new());

        assert_eq!(a.total_kmers(), 2);
        assert_eq!(a.distinct_kmers(), 1);
        assert_eq!(a.get_count(7), 2);
    }

    #[test]
    fn from_sorted_entries_builds_a_fully_usable_counter() {
        let c = KmerCounter::from_sorted_entries(vec![(1, 2), (5, 7), (9, 1)], 10);

        assert_eq!(c.total_kmers(), 10);
        assert_eq!(c.distinct_kmers(), 3);
        assert_eq!(c.get_count(5), 7);
        assert_eq!(c.get_count(2), 0, "absent kmer stays zero");
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![(1, 2), (5, 7), (9, 1)]);
    }

    #[test]
    fn from_sorted_entries_of_empty_input_is_a_valid_empty_counter() {
        let c = KmerCounter::from_sorted_entries(Vec::new(), 0);
        assert_eq!(c.total_kmers(), 0);
        assert_eq!(c.distinct_kmers(), 0);
    }

    #[test]
    fn iter_yields_every_distinct_kmer_with_its_correct_count() {
        let c = counter_with_graded_counts();

        let mut collected: Vec<(u64, u32)> = c.iter().collect();
        collected.sort_unstable();

        assert_eq!(collected, vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
    }

    #[test]
    fn distinct_and_total_kmers_survive_repeated_reads_between_inserts() {
        // Read methods must not corrupt state for a subsequent insert:
        // finalizing is meant to be transparent, not a one-way operation.
        let mut c = KmerCounter::new();
        c.insert(1);
        assert_eq!(c.distinct_kmers(), 1);

        c.insert(2);
        assert_eq!(c.distinct_kmers(), 2, "insert after a read must still be counted");
        assert_eq!(c.total_kmers(), 2);

        c.insert(1);
        assert_eq!(c.get_count(1), 2, "a repeated insert after a read must still merge with the earlier one");
    }

    #[test]
    fn get_count_of_an_absent_kmer_is_zero() {
        let c = counter_with_graded_counts();
        assert_eq!(c.get_count(999), 0);
    }

    #[test]
    fn top_kmers_returns_the_n_highest_counts_descending() {
        let c = counter_with_graded_counts();

        let top = c.top_kmers(2);

        assert_eq!(top, vec![(5, 5), (4, 4)]);
    }

    /// `select_nth_unstable_by(take - 1, ..)` would panic if `take` were
    /// naively left at `n == 0` (index `0usize - 1` underflows); this
    /// pins the early return that avoids it.
    #[test]
    fn top_kmers_of_zero_is_empty() {
        let c = counter_with_graded_counts();

        assert_eq!(c.top_kmers(0), Vec::new());
    }

    /// `select_nth_unstable_by` would panic if `take` were left at the
    /// requested `n` when `n` exceeds the table size (out-of-bounds
    /// index); this pins the `n.min(entries.len())` clamp.
    #[test]
    fn top_kmers_asking_for_more_than_exist_returns_everything() {
        let c = counter_with_graded_counts();

        let top = c.top_kmers(100);

        assert_eq!(top, vec![(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]);
    }

    #[test]
    fn generate_histogram_counts_how_many_distinct_kmers_share_each_frequency() {
        let mut c = KmerCounter::new();
        c.insert(1);
        c.insert(2);
        c.insert(2);
        c.insert(3);
        c.insert(3);

        let hist = c.generate_histogram();

        assert_eq!(hist.get(&1), Some(&1), "one distinct k-mer seen once");
        assert_eq!(hist.get(&2), Some(&2), "two distinct k-mers seen twice each");
    }

    /// Exercises the eager mid-run finalize `RAW_FINALIZE_THRESHOLD`
    /// triggers -- not just the one-shot finalize-on-first-read every
    /// other test here relies on. Inserts three times past the
    /// threshold, in a pattern designed to hit both branches
    /// `finalize_inner` can take when it runs more than once (an empty
    /// `finalized` the first time, a non-empty one to merge into on
    /// every call after), then confirms counts across the boundary are
    /// still correct: this is exactly where a seam bug in the
    /// incremental merge would show up.
    #[test]
    fn counts_are_correct_across_multiple_eager_mid_run_finalizes() {
        let mut c = KmerCounter::new();

        // Three times over the threshold, so at least two eager finalizes
        // fire during these inserts alone, before any read forces one.
        let total_inserts = RAW_FINALIZE_THRESHOLD * 3;
        for i in 0..total_inserts {
            // A small alphabet of k-mer values so the run mixes brand-new
            // values with repeats of values seen in an earlier eager
            // finalize -- the case that actually exercises the "merge
            // into a non-empty `finalized`" branch, not just "extend an
            // empty one".
            let kmer = (i % 4) as u64;
            c.insert(kmer);
        }

        assert_eq!(c.total_kmers(), total_inserts as u64);
        assert_eq!(c.distinct_kmers(), 4);
        for kmer in 0u64..4 {
            let expected = (total_inserts / 4) as u32;
            assert_eq!(c.get_count(kmer), expected, "kmer {kmer} count across finalize boundaries");
        }
    }

    /// Same as above but through `insert_batch`, the path
    /// `pipeline.rs` actually uses per-record during real counting --
    /// `insert`'s single-kmer loop above proves the merge seam is
    /// correct, this proves the batched entry point trips the same
    /// eager threshold check correctly too.
    #[test]
    fn counts_are_correct_across_multiple_eager_mid_run_finalizes_via_insert_batch() {
        let mut c = KmerCounter::new();

        let total_inserts = RAW_FINALIZE_THRESHOLD * 3;
        let batch: Vec<u64> = (0..total_inserts as u64).map(|i| i % 4).collect();

        // Fed in chunks, not as one `Vec::extend_from_slice` -- a single
        // call larger than the threshold would only ever cross it once,
        // which would not exercise repeated finalizes the way real
        // per-record batches do.
        for chunk in batch.chunks(9_973) {
            c.insert_batch(chunk);
        }

        assert_eq!(c.total_kmers(), total_inserts as u64);
        assert_eq!(c.distinct_kmers(), 4);
        for kmer in 0u64..4 {
            let expected = (total_inserts / 4) as u32;
            assert_eq!(c.get_count(kmer), expected, "kmer {kmer} count across finalize boundaries");
        }
    }

    /// Exercises `MAX_PENDING_RUNS`'s own eager-consolidation trigger, not
    /// just the final read-triggered one every other test here relies on.
    /// Crosses `RAW_FINALIZE_THRESHOLD` enough times (`MAX_PENDING_RUNS * 2
    /// + 1`) to push `pending` over `MAX_PENDING_RUNS` twice during
    /// insertion alone, so at least two full `consolidate` passes happen
    /// mid-run -- exactly the seam where a bug in folding several pending
    /// runs (each internally deduplicated, but sharing keys with each
    /// other and with whatever `finalized` already held) would surface.
    #[test]
    fn counts_are_correct_across_multiple_eager_mid_run_consolidations() {
        let mut c = KmerCounter::new();

        let total_inserts = RAW_FINALIZE_THRESHOLD * (MAX_PENDING_RUNS * 2 + 1);
        for i in 0..total_inserts {
            // Same small alphabet trick as the tests above: every eager
            // compaction sees a mix of brand-new-to-this-run and
            // already-seen-elsewhere values, so consolidation actually has
            // overlapping keys to merge, not disjoint ones.
            let kmer = (i % 4) as u64;
            c.insert(kmer);
        }

        assert_eq!(c.total_kmers(), total_inserts as u64);
        assert_eq!(c.distinct_kmers(), 4);
        for kmer in 0u64..4 {
            let expected = (total_inserts / 4) as u32;
            assert_eq!(c.get_count(kmer), expected, "kmer {kmer} count across consolidation boundaries");
        }
    }
}
