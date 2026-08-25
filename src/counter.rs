// src/counter.rs

use std::cmp::Reverse;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
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
    /// Sorted, deduplicated `(kmer, count)` runs produced by `compact_raw`
    /// but not yet folded into `finalized` -- each one internally correct
    /// (no duplicate keys within a single run), but keys may repeat
    /// *across* runs (the same k-mer compacted separately in two different
    /// eager passes). See `consolidate` for why these accumulate instead
    /// of being merged in immediately, and `MAX_PENDING_RUNS` for why that
    /// accumulation is bounded rather than unbounded.
    pending: Vec<Vec<(u64, u32)>>,
    /// Sorted (ascending by k-mer), deduplicated `(kmer, count)` pairs.
    /// Only trustworthy when `valid` is `true`; rebuilt by `finalize_inner`
    /// otherwise (which folds `pending` -- and any leftover `raw` -- into
    /// it via `consolidate`).
    finalized: Vec<(u64, u32)>,
    /// Whether `finalized` currently reflects every instance in `raw` and
    /// every run in `pending` (both drained empty once it does). `false`
    /// after any insertion; set back to `true` by `finalize_inner`.
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

/// Above this many *unconsolidated* runs in `inner.pending`, an eager
/// `compact_raw` (see `RAW_FINALIZE_THRESHOLD`) triggers a `consolidate`
/// pass rather than leaving the run count to grow for the rest of the run.
///
/// This is the fix for a real, measured quadratic blowup in the design
/// `MAX_PENDING_RUNS` replaces: that design folded every newly-compacted
/// run straight into a single ever-growing `finalized` table via
/// `merge_sorted_counts`, so the Nth eager compaction touched the *entire*
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
fn compact_raw(inner: &mut Inner) {
    if inner.raw.is_empty() {
        return;
    }

    inner.raw.sort_unstable();

    let sorted: &[u64] = inner.raw.as_slice();
    let len = sorted.len();
    // Exactly `len` is the smallest capacity that can never need to grow
    // (one entry per element, in the all-distinct case), so no `push` below
    // reallocates and no bytes are ever recopied.
    let mut new_run: Vec<(u64, u32)> = Vec::with_capacity(len);

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
        new_run.push((kmer, (i - run_start).min(u32::MAX as usize) as u32));
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
    inner.finalized = k_way_merge_sorted_counts(sources);
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
        // produces; skip the heap machinery entirely.
        1 => return sources.remove(0),
        _ => {}
    }

    let total_len: usize = sources.iter().map(Vec::len).sum();
    let mut merged: Vec<(u64, u32)> = Vec::with_capacity(total_len);

    // Each heap entry is `Reverse((kmer, source index, index within that
    // source))` -- `Reverse` turns `BinaryHeap`'s natural max-heap into the
    // min-heap a merge needs, and the two indices are enough to advance
    // exactly the source a popped entry came from.
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

            // Hoisted: the previous form indexed `sources[src_idx]` twice
            // per entry (once to read the count, once to look up the next
            // element), which is two bounds checks against `sources.len()`
            // where one suffices -- one compare-and-branch pair removed per
            // merged entry, over every entry of every source on every
            // consolidation pass.
            let src: &[(u64, u32)] = &sources[src_idx];
            count = count.saturating_add(src[elem_idx].1);

            let next_idx = elem_idx + 1;
            match src.get(next_idx) {
                // Overwriting the root through `PeekMut` and letting its
                // `Drop` re-sift replaces a `pop` (which sifts the hole all
                // the way down to a leaf and then sifts the moved-in element
                // back up) *plus* a `push` (another sift up) with a single
                // sift down -- roughly half the heap element moves and
                // comparisons, on every one of the tens of millions of
                // entries a consolidation pass merges. Counted on the
                // emitted assembly, the whole function shrank from 487 to
                // 359 instructions, essentially all of it inlined heap
                // restructuring that no longer happens.
                Some(&(next_kmer, _)) => *top = Reverse((next_kmer, src_idx, next_idx)),
                // Source exhausted: this is the one case that still has to
                // shrink the heap.
                None => {
                    PeekMut::pop(top);
                }
            }
        }

        merged.push((kmer, count));
    }

    merged
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
        // Two `<` comparisons rather than `match a[i].0.cmp(&b[j].0)`.
        // Matching on `Ordering` does not lower to a three-way branch: rustc
        // materializes the discriminant with `seta` + `sbb` and then
        // re-tests it with `movzbl` + `cmpl`, four extra instructions and a
        // third conditional branch per merged entry, on top of the compare
        // it already did. Comparing the keys twice costs one extra `cmp` and
        // nothing else.
        //
        // Reading only the keys up front is the other half of it: the counts
        // are then loaded on the branch that actually uses them, so the
        // "take from a" path never loads b's count and vice versa. Loading
        // the whole pair up front (the obvious way to write this) is what
        // makes an eager, sometimes-dead load appear in the loop.
        //
        // Net per merged entry: 4 fewer ALU instructions and one fewer
        // conditional branch, with no added load -- over 161 million entries
        // written by the combine phase at benchmark scale (see `merge_all`
        // for that count), or 53.8 million once that path is adopted.
        //
        // The bounds checks here are already gone: `while i < a.len() && j <
        // b.len()` proves both indices in range and the emitted code has no
        // panic edge in this loop, only in the tail slicing below.
        let a_kmer = a[i].0;
        let b_kmer = b[j].0;
        if a_kmer < b_kmer {
            merged.push(a[i]);
            i += 1;
        } else if b_kmer < a_kmer {
            merged.push(b[j]);
            j += 1;
        } else {
            merged.push((a_kmer, a[i].1.saturating_add(b[j].1)));
            i += 1;
            j += 1;
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
    pub(crate) fn from_sorted_entries(entries: Vec<(u64, u32)>, total_kmers: u64) -> Self {
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "from_sorted_entries requires a strictly ascending, deduplicated table"
        );
        Self {
            inner: Mutex::new(Inner { finalized: entries, valid: true, ..Inner::default() }),
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
    /// 16 bytes per entry) that is 4 merges producing 13.4M entries, 2
    /// producing 26.9M and 1 producing 53.8M -- 161 million entries written,
    /// 2.6 GB of `memcpy`, across 7 separate allocations the largest of
    /// which is 860 MB. Merging all `w` sources at once writes each of the
    /// 53.8 million final entries exactly once: 860 MB, one allocation.
    /// That is ~1.7 GB of copying and 6 large allocations removed, and it
    /// costs nothing extra per entry -- `k_way_merge_sorted_counts` already
    /// does `O(entries x log(sources))` with `sources` bounded by the worker
    /// count either way.
    ///
    /// `pipeline.rs`'s combine phase calls this once over the collected
    /// worker counters. `QcSummary::merge` deliberately stays pairwise
    /// there: it is `O(1)`, so a k-way form would buy nothing.
    pub fn merge_all(counters: Vec<KmerCounter>) -> KmerCounter {
        let mut total_kmers: u64 = 0;
        let mut sources: Vec<Vec<(u64, u32)>> = Vec::with_capacity(counters.len());

        for counter in counters {
            total_kmers += counter.total_kmers;
            let mut inner =
                counter.inner.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
            finalize_inner(&mut inner);
            // An empty source would only occupy a heap slot and be popped
            // straight back out; skipping it also lets the one-source fast
            // path in `k_way_merge_sorted_counts` trigger when every other
            // worker happened to see nothing.
            if !inner.finalized.is_empty() {
                sources.push(std::mem::take(&mut inner.finalized));
            }
        }

        // The merge output is sorted ascending and deduplicated by
        // construction -- that is exactly what a k-way merge of sorted,
        // deduplicated sources with equal keys folded together produces --
        // which is the invariant `from_sorted_entries` documents as the
        // caller's responsibility.
        Self::from_sorted_entries(k_way_merge_sorted_counts(sources), total_kmers)
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
        let mut entries: Vec<(u64, u32)> = guard.finalized.clone();
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
        // `?` on the borrow, rather than `.copied()` followed by
        // `if item.is_some()`: the previous form tested the same
        // discriminant twice -- once inside `get` to build the `Option`, and
        // again to decide whether to advance -- and made the advance itself
        // conditional. One test and one unconditional increment here, over
        // every entry of the finalized table on every export pass (53.8
        // million entries per pass at benchmark scale, and `export.rs` plus
        // `ffi.rs` walk it once each).
        let &item = self.guard.finalized.get(self.idx)?;
        self.idx += 1;
        Some(item)
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
