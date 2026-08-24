// src/counter.rs

use std::sync::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;

/// Outcome of a `prune` call, for reporting what a filter removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub dropped_min: u64,
    pub dropped_max: u64,
    pub kept: u64,
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
    /// Sorted (ascending by k-mer), deduplicated `(kmer, count)` pairs.
    /// Only trustworthy when `valid` is `true`; rebuilt from `raw` by
    /// `finalize_inner` otherwise.
    finalized: Vec<(u64, u32)>,
    /// Whether `finalized` currently reflects every instance in `raw`
    /// (which is drained empty once it does). `false` after any
    /// insertion; set back to `true` by `finalize_inner`.
    valid: bool,
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
/// capacity (8 bytes/entry) alive alongside a fresh
/// `Vec::with_capacity(raw.len())` of `(u64, u32)` pairs (16 bytes/entry
/// after alignment padding) while it drains one into the other, so a
/// finalize at the cap costs on the order of 2,000,000 * 24 bytes =~
/// 48MB transient per worker, not gigabytes. Raising the cap trades more
/// of that transient (and a higher permanent floor) for fewer, larger
/// sorts; lowering it trades the other way. This is a starting point,
/// not a value proven optimal by a sweep across input shapes.
const RAW_FINALIZE_THRESHOLD: usize = 2_000_000;

/// Sorts `inner.raw` and compacts it into `inner.finalized` as sorted,
/// deduplicated `(kmer, count)` pairs -- a no-op if `inner.valid` already
/// holds.
///
/// This, not a hash table, is the counting step: `raw.sort_unstable()` is
/// a handful of cache-friendly sequential passes over contiguous memory,
/// and the single linear scan afterwards that turns runs of equal values
/// into `(kmer, count)` pairs is sequential too. A `HashMap<u64, _>`
/// spends essentially every insertion on effectively-random bucket
/// placement, which is an L3 cache miss once the table exceeds cache size
/// (tens of MB -- a few hundred thousand entries) -- true of any real
/// FASTQ file, not just large ones. Measured effect on a 2.14 GB / 53.7M
/// distinct k-mer benchmark file is documented in the README.
fn finalize_inner(inner: &mut Inner) {
    if inner.valid {
        return;
    }

    if inner.raw.is_empty() {
        // Nothing new since the last finalize (or ever): `finalized`, if
        // non-empty, already reflects everything -- e.g. a read followed
        // by no further inserts before another read.
        inner.valid = true;
        return;
    }

    inner.raw.sort_unstable();

    // Compact the newly-sorted instances into (kmer, count) pairs. This
    // is *only* the new arrivals since the last finalize, not the whole
    // table -- `inner.finalized` may already hold entries from an earlier
    // finalize (a read followed by more inserts is a normal sequence, not
    // a one-way transition, and so is the eager finalize
    // `RAW_FINALIZE_THRESHOLD` triggers mid-run, well before any read),
    // so it must be merged into, never discarded.
    let mut new_entries: Vec<(u64, u32)> = Vec::with_capacity(inner.raw.len());
    let mut iter = inner.raw.drain(..).peekable();
    while let Some(kmer) = iter.next() {
        let mut count: u32 = 1;
        while iter.peek() == Some(&kmer) {
            iter.next();
            // `saturating_add`, not `+=`: a single k-mer occurring more
            // than u32::MAX times should not panic or silently wrap, even
            // though that is not expected to occur in practice.
            count = count.saturating_add(1);
        }
        new_entries.push((kmer, count));
    }

    if inner.finalized.is_empty() {
        inner.finalized = new_entries;
    } else {
        let previous = std::mem::take(&mut inner.finalized);
        inner.finalized = merge_sorted_counts(&previous, &new_entries);
    }

    inner.valid = true;
}

/// Merges two sorted, deduplicated `(kmer, count)` sequences into one, in
/// a single linear O(n + m) pass -- the standard mergesort merge step.
/// Shared by `finalize_inner` (merging newly-finalized entries into
/// whatever was already finalized) and `KmerCounter::merge` (combining
/// two whole counters), so the two never diverge.
fn merge_sorted_counts(a: &[(u64, u32)], b: &[(u64, u32)]) -> Vec<(u64, u32)> {
    let mut merged: Vec<(u64, u32)> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                merged.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push((a[i].0, a[i].1.saturating_add(b[j].1)));
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&a[i..]);
    merged.extend_from_slice(&b[j..]);
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
            // grow until the first read: nothing else bounds `raw`.
            if inner.raw.len() >= RAW_FINALIZE_THRESHOLD {
                finalize_inner(inner);
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
                finalize_inner(inner);
            }
        }
        self.total_kmers += kmers.len() as u64;
    }

    /// Combines `other` into `self`.
    ///
    /// Finalizes both sides (if not already finalized) and merges the two
    /// sorted `(kmer, count)` sequences in one linear O(n + m) pass -- the
    /// standard mergesort merge step -- rather than replaying `other`'s
    /// entries through hash-table insertion one at a time.
    pub fn merge(&mut self, other: KmerCounter) {
        self.total_kmers += other.total_kmers;

        let mut other_inner =
            other.inner.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
        let self_inner = self.inner_mut();
        finalize_inner(self_inner);
        finalize_inner(&mut other_inner);

        let merged = merge_sorted_counts(&self_inner.finalized, &other_inner.finalized);

        self_inner.raw.clear();
        self_inner.finalized = merged;
        self_inner.valid = true;
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
        inner.finalized.retain(|&(_, count)| {
            if count < min {
                stats.dropped_min += 1;
                false
            } else if max.is_some_and(|cap| count > cap) {
                stats.dropped_max += 1;
                false
            } else {
                stats.kept += 1;
                true
            }
        });

        stats
    }

    #[inline(always)]
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
            .binary_search_by_key(&kmer, |&(k, _)| k)
            .map(|idx| guard.finalized[idx].1)
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
        for &(_, count) in guard.finalized.iter() {
            *histogram.entry(count).or_insert(0) += 1;
        }
        histogram
    }

    pub fn top_kmers(&self, n: usize) -> Vec<(u64, u32)> {
        let guard = self.ensure_finalized();
        let mut entries: Vec<(u64, u32)> = guard.finalized.clone();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
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
        let item = self.guard.finalized.get(self.idx).copied();
        if item.is_some() {
            self.idx += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.guard.finalized.len().saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
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
}
