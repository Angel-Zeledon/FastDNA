// src/wide_counter.rs
//! Sort-and-compact counting for `u128` k-mers (`33 <= k <= 64`).
//!
//! # Why this is not `counter.rs`
//!
//! `counter.rs` is the narrow engine's counting table, and it is not a
//! simple one: an MSD radix partition ahead of the sort, a lazily
//! finalized table behind a `Mutex`, a buffered-instance bound tuned
//! against measurements, and a documented history of alternatives that
//! were tried and reverted *with the numbers that killed them*. Every one
//! of those decisions was made against `u64` inputs on a real benchmark
//! file.
//!
//! None of that carries over for free. A `u128` comparison is not one
//! instruction, the radix pass's bucket count was chosen for 8-byte keys,
//! and the buffer bound was fit to how many 8-byte instances fit a cache
//! level. Porting that machinery and *assuming* the tuning still holds
//! would be exactly the kind of unmeasured claim this repository keeps
//! finding in itself. So this counter does the simple thing -- append,
//! sort, compact -- and everything it does beyond that was put there by a
//! measurement rather than by analogy.
//!
//! # What was measured, and what it cost
//!
//! On `mid.fastq` (144,000,000 k-mer occurrences, 9,213,849 distinct,
//! `k = 31` so the narrow engine can be compared against directly), peak
//! RSS and wall clock as this file changed:
//!
//! | | 1 thread | 11 threads |
//! |---|---:|---:|
//! | first working version | 7.98 s / 4,438 MiB | 4.11 s / 4,713 MiB |
//! | reusing the merge buffer | 5.58 s / 1,187 MiB | 4.60 s / 4,402 MiB |
//! | + parallel arrays | **5.62 s / 793 MiB** | **3.36 s / 4,152 MiB** |
//! | *the narrow engine, same file* | *1.87 s / 750 MiB* | *1.87 s / 750 MiB* |
//!
//! Both changes came from measurements that contradicted a guess. The
//! first version's memory did not scale with threads (1 thread cost as
//! much as 11) and did not scale with distinct k-mers either -- it scaled
//! with *occurrences*, at ~26 bytes each, which is the signature of
//! repeatedly allocating and freeing a large buffer rather than of holding
//! one. See `scratch_kmers` and `table_kmers` for the two fixes.
//!
//! # What is still true, and is a real limit
//!
//! At one thread this counter now costs about what the narrow one costs
//! (793 MiB against 750 MiB) and about three times the time -- which is
//! the `u128` arithmetic, and is the price of the capability.
//!
//! At eleven it costs 4,152 MiB, because peak memory here scales with
//! `threads * distinct_kmers`: every worker holds its own table. That is
//! the same term the narrow *in-memory* strategy has, and the narrow
//! engine has a way out of it (`binned.rs` partitions by minimizer so the
//! store is shared rather than replicated). **The wide engine has no such
//! strategy.**
//!
//! Which makes thread count the only lever, and it does not behave the way
//! a thread count usually does. Measured on the same file:
//!
//! | threads | time | peak RSS |
//! |---:|---:|---:|
//! | 1 | 5.62 s | 793 MiB |
//! | 4 | **2.54 s** | **2,002 MiB** |
//! | 11 | 3.36 s | 4,152 MiB |
//!
//! Four threads is *both* faster than eleven and half its memory. The
//! first draft of this paragraph predicted the opposite -- fewer threads
//! trading wall clock for memory -- and the measurement contradicted it,
//! so this is what the table says rather than what the reasoning did. The
//! likely cause is the sequential fold in `merge_all`: eleven tables cost
//! more to merge than four, and past some point that outweighs what the
//! extra workers won. Likely, not established; nothing here measured the
//! fold in isolation.
//!
//! # Shape
//!
//! Each worker owns one [`WideKmerCounter`], appends into it, and the
//! pipeline folds them together at the end. There is no interior
//! mutability and no lazy finalization: [`WideKmerCounter::finish`]
//! consumes the counter and returns an immutable [`WideCounts`], which is
//! the only thing readers ever see. That is a smaller contract than
//! `KmerCounter`'s, and it is smaller on purpose -- the narrow counter's
//! `Mutex` exists to let `&self` methods finalize on demand, a
//! convenience nothing here needs.

use crate::wide_kmer;

/// Buffered k-mer instances after which a counter compacts eagerly.
///
/// The same 2,000,000 `counter.rs` uses, and for the same reason -- an
/// unbounded raw buffer makes peak memory a function of input size rather
/// than of the bound. The *value* is inherited rather than re-derived:
/// `counter.rs` fit it against 8-byte instances and these are 16-byte, so
/// the same count is twice the bytes. That is a deliberate conservatism
/// (compacting sooner, not later) and not a measurement of the right
/// threshold for this width.
const RAW_FINALIZE_THRESHOLD: usize = 2_000_000;

/// A worker's private, still-uncompacted count table.
#[derive(Debug, Default)]
pub struct WideKmerCounter {
    /// Every k-mer instance seen since the last compaction, in arrival
    /// order. Sequential appends, which is the whole reason this is a
    /// `Vec` and not a hash map (`docs/BENCHMARKS.md` records the 9x this
    /// choice was worth for the narrow engine).
    raw: Vec<u128>,
    /// Compacted counts, sorted by k-mer, as two parallel arrays rather
    /// than one `Vec<(u128, u32)>`.
    ///
    /// Not a style preference: `(u128, u32)` has 16-byte alignment, so the
    /// tuple occupies **32 bytes to carry 20** -- 37% of the table is
    /// padding. At 11 workers over 9.2M distinct k-mers that padding alone
    /// is 1.2 GiB. `counter.rs`'s finalized table is split for the same
    /// reason, and `WideCounts` already was, so this also removes the
    /// `unzip` that `finish` used to pay.
    table_kmers: Vec<u128>,
    table_counts: Vec<u32>,
    /// The buffers [`Self::compact`] merges into, swapped with the table
    /// rather than allocated fresh.
    ///
    /// Measured, not guessed. Allocating a new merge target on every
    /// compaction meant ~72 allocate-grow-free cycles on a 144M-occurrence
    /// file, each up to the full table size, and peak RSS came out at
    /// **4,497 MiB against a table that holds 294 MiB** -- the allocator
    /// does not hand freed pages back promptly (the same effect
    /// `mem_estimate.rs`'s binned calibration factor absorbs), so the
    /// high-water marks accumulate instead of overlapping. Swapping two
    /// buffers that each grow once to their final size turns 72 large
    /// allocations into 2.
    scratch_kmers: Vec<u128>,
    scratch_counts: Vec<u32>,
    total: u64,
}

impl WideKmerCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a batch of already-canonical k-mers.
    ///
    /// Compacts when the raw buffer crosses [`RAW_FINALIZE_THRESHOLD`], so
    /// a worker's footprint is bounded by that threshold plus its share of
    /// the distinct k-mers rather than by the input's length.
    pub fn insert_batch(&mut self, kmers: &[u128]) {
        self.raw.extend_from_slice(kmers);
        self.total += kmers.len() as u64;
        if self.raw.len() >= RAW_FINALIZE_THRESHOLD {
            self.compact();
        }
    }

    /// Folds the raw buffer into the sorted table.
    ///
    /// Sorts the buffer, run-length encodes it, and merges that against
    /// the existing table in one linear pass -- both sides are sorted, so
    /// the merge never re-sorts what was already compacted.
    fn compact(&mut self) {
        if self.raw.is_empty() {
            return;
        }
        self.raw.sort_unstable();

        // Reused, not reallocated: see `scratch_kmers`' own doc comment for
        // the measurement that made this worth doing. Both sides are read
        // by index rather than consumed, because consuming the table would
        // drop the very allocation this is trying to keep alive.
        let mut merged_kmers = std::mem::take(&mut self.scratch_kmers);
        let mut merged_counts = std::mem::take(&mut self.scratch_counts);
        merged_kmers.clear();
        merged_counts.clear();
        merged_kmers.reserve(self.table_kmers.len() + self.raw.len() / 2);
        merged_counts.reserve(self.table_kmers.len() + self.raw.len() / 2);

        let (kmers, counts, raw) = (&self.table_kmers, &self.table_counts, &self.raw);
        let (mut ti, mut ri) = (0usize, 0usize);

        // Consumes the run of `value` at `ri`, returning its length.
        let run_from = |raw: &[u128], ri: &mut usize, value: u128| -> u32 {
            let mut run = 0u32;
            while *ri < raw.len() && raw[*ri] == value {
                *ri += 1;
                run = run.saturating_add(1);
            }
            run
        };

        loop {
            match (kmers.get(ti).copied(), raw.get(ri).copied()) {
                (None, None) => break,
                (Some(tk), Some(rk)) if tk < rk => {
                    merged_kmers.push(tk);
                    merged_counts.push(counts[ti]);
                    ti += 1;
                }
                (Some(tk), Some(rk)) if rk < tk => {
                    merged_kmers.push(rk);
                    merged_counts.push(run_from(raw, &mut ri, rk));
                }
                (Some(tk), Some(_)) => {
                    // Equal: the table's existing count plus this buffer's
                    // run of the same k-mer.
                    let existing = counts[ti];
                    ti += 1;
                    let run = run_from(raw, &mut ri, tk);
                    merged_kmers.push(tk);
                    merged_counts.push(existing.saturating_add(run));
                }
                (Some(tk), None) => {
                    merged_kmers.push(tk);
                    merged_counts.push(counts[ti]);
                    ti += 1;
                }
                (None, Some(rk)) => {
                    merged_kmers.push(rk);
                    merged_counts.push(run_from(raw, &mut ri, rk));
                }
            }
        }

        // `clear`, not a reallocation: the raw buffer's capacity is the
        // bound `RAW_FINALIZE_THRESHOLD` names, and giving it back would
        // mean re-growing it on the next batch.
        self.raw.clear();
        // The finished merge becomes the table; the table's own allocations
        // become the next compaction's merge targets. Two of each,
        // alternating, each grown once.
        std::mem::swap(&mut self.table_kmers, &mut merged_kmers);
        std::mem::swap(&mut self.table_counts, &mut merged_counts);
        self.scratch_kmers = merged_kmers;
        self.scratch_counts = merged_counts;
    }

    /// Folds several workers' counters into one.
    ///
    /// A parallel reduce would be the obvious thing here, and `counter.rs`
    /// does exactly that. This is a sequential fold because the wide path
    /// has no measurements behind it either way, and a sequential fold
    /// cannot be wrong about a merge order -- see the module doc comment
    /// on what is deliberately not claimed.
    pub fn merge_all(counters: Vec<WideKmerCounter>) -> WideKmerCounter {
        let mut out = WideKmerCounter::new();
        for mut counter in counters {
            counter.compact();
            out.total += counter.total;
            if out.table_kmers.is_empty() {
                out.table_kmers = counter.table_kmers;
                out.table_counts = counter.table_counts;
                continue;
            }
            let (kmers, counts) = merge_sorted_tables(
                std::mem::take(&mut out.table_kmers),
                std::mem::take(&mut out.table_counts),
                counter.table_kmers,
                counter.table_counts,
            );
            out.table_kmers = kmers;
            out.table_counts = counts;
        }
        out
    }

    /// Consumes the counter and returns its finished table.
    pub fn finish(mut self) -> WideCounts {
        self.compact();
        WideCounts { kmers: self.table_kmers, counts: self.table_counts, total: self.total }
    }
}

/// Merges two sorted count tables, summing counts for k-mers in both.
///
/// Takes and returns parallel arrays for the same reason the table is
/// stored that way -- see `WideKmerCounter::table_kmers`.
fn merge_sorted_tables(
    left_kmers: Vec<u128>,
    left_counts: Vec<u32>,
    right_kmers: Vec<u128>,
    right_counts: Vec<u32>,
) -> (Vec<u128>, Vec<u32>) {
    let capacity = left_kmers.len() + right_kmers.len();
    let mut out_kmers = Vec::with_capacity(capacity);
    let mut out_counts = Vec::with_capacity(capacity);
    let (mut l, mut r) = (0usize, 0usize);

    loop {
        match (left_kmers.get(l).copied(), right_kmers.get(r).copied()) {
            (None, None) => break,
            (Some(lk), Some(rk)) if lk < rk => {
                out_kmers.push(lk);
                out_counts.push(left_counts[l]);
                l += 1;
            }
            (Some(lk), Some(rk)) if rk < lk => {
                let _ = lk;
                out_kmers.push(rk);
                out_counts.push(right_counts[r]);
                r += 1;
            }
            (Some(lk), Some(_)) => {
                out_kmers.push(lk);
                out_counts.push(left_counts[l].saturating_add(right_counts[r]));
                l += 1;
                r += 1;
            }
            (Some(lk), None) => {
                out_kmers.push(lk);
                out_counts.push(left_counts[l]);
                l += 1;
            }
            (None, Some(rk)) => {
                out_kmers.push(rk);
                out_counts.push(right_counts[r]);
                r += 1;
            }
        }
    }

    (out_kmers, out_counts)
}

/// A finished wide count table: distinct k-mers in ascending order, with
/// their occurrence counts.
///
/// Struct-of-arrays rather than `Vec<(u128, u32)>` for the same reason
/// `counter.rs`'s finalized table is: the export path writes one column at
/// a time, and a tuple layout would pad each entry to 32 bytes for a
/// 16-byte key and a 4-byte count.
#[derive(Debug, Default)]
pub struct WideCounts {
    kmers: Vec<u128>,
    counts: Vec<u32>,
    total: u64,
}

impl WideCounts {
    /// Distinct canonical k-mers.
    pub fn distinct_kmers(&self) -> usize {
        self.kmers.len()
    }

    /// Total k-mer instances counted, before deduplication. Unaffected by
    /// any downstream `min_count` filter, exactly as `KmerCounts`'
    /// `total_kmers` is -- it is the normalization basis.
    pub fn total_kmers(&self) -> u64 {
        self.total
    }

    /// `(kmer, count)` pairs in ascending k-mer order.
    pub fn iter(&self) -> impl Iterator<Item = (u128, u32)> + '_ {
        self.kmers.iter().copied().zip(self.counts.iter().copied())
    }

    /// The 16-byte big-endian keys, in the same order as [`Self::iter`] --
    /// what the Parquet writer stores. See
    /// [`crate::wide_kmer::to_key_bytes`] for why big-endian keeps the
    /// table sorted for a reader that never decodes it.
    pub fn key_bytes(&self) -> impl Iterator<Item = [u8; 16]> + '_ {
        self.kmers.iter().copied().map(wide_kmer::to_key_bytes)
    }

    /// Occurrence counts, aligned with [`Self::key_bytes`].
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Drops k-mers outside `[min, max]`, reporting how many went each
    /// way -- the wide twin of `KmerCounter::prune`.
    ///
    /// `total_kmers` is deliberately **not** adjusted, exactly as the
    /// narrow counter leaves it alone: it is the normalization basis and
    /// must keep reflecting what was counted, however many rows a filter
    /// later removed. `CLAUDE.md` names this as a convention worth
    /// preserving, and it would be easy to "fix" here by accident.
    pub fn prune(&mut self, min: u32, max: Option<u32>) -> (u64, u64) {
        let mut dropped_min = 0u64;
        let mut dropped_max = 0u64;
        let mut kept_kmers = Vec::with_capacity(self.kmers.len());
        let mut kept_counts = Vec::with_capacity(self.counts.len());

        for (kmer, count) in self.kmers.iter().copied().zip(self.counts.iter().copied()) {
            if count < min {
                dropped_min += 1;
            } else if max.is_some_and(|ceiling| count > ceiling) {
                dropped_max += 1;
            } else {
                kept_kmers.push(kmer);
                kept_counts.push(count);
            }
        }

        self.kmers = kept_kmers;
        self.counts = kept_counts;
        (dropped_min, dropped_max)
    }

    /// The frequency spectrum: how many distinct k-mers occur exactly
    /// once, twice, and so on. Same shape `KmerCounter::generate_histogram`
    /// returns for the narrow engine.
    pub fn histogram(&self) -> std::collections::BTreeMap<u32, u64> {
        let mut histogram = std::collections::BTreeMap::new();
        for &count in &self.counts {
            *histogram.entry(count).or_insert(0) += 1;
        }
        histogram
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn counted(batches: &[&[u128]]) -> WideCounts {
        let mut counter = WideKmerCounter::new();
        for batch in batches {
            counter.insert_batch(batch);
        }
        counter.finish()
    }

    #[test]
    fn counts_and_sorts_a_single_batch() {
        let counts = counted(&[&[5, 1, 5, 3, 1, 5]]);
        assert_eq!(counts.iter().collect::<Vec<_>>(), vec![(1, 2), (3, 1), (5, 3)]);
        assert_eq!(counts.distinct_kmers(), 3);
        assert_eq!(counts.total_kmers(), 6);
    }

    #[test]
    fn accumulates_across_batches() {
        let counts = counted(&[&[1, 2], &[2, 3], &[3, 3]]);
        assert_eq!(counts.iter().collect::<Vec<_>>(), vec![(1, 1), (2, 2), (3, 3)]);
        assert_eq!(counts.total_kmers(), 6);
    }

    /// The compaction path is only reached above the threshold, so this is
    /// the test that exercises the sorted merge against an existing table
    /// rather than the single-shot sort.
    #[test]
    fn compacting_mid_stream_gives_the_same_answer_as_one_shot() {
        let batch: Vec<u128> = (0..RAW_FINALIZE_THRESHOLD as u128 + 1_000).map(|i| i % 977).collect();
        let mut chunked = WideKmerCounter::new();
        for slice in batch.chunks(10_000) {
            chunked.insert_batch(slice);
        }
        let chunked = chunked.finish();

        let mut one_shot = WideKmerCounter::new();
        one_shot.insert_batch(&batch);
        let one_shot = one_shot.finish();

        assert_eq!(chunked.iter().collect::<Vec<_>>(), one_shot.iter().collect::<Vec<_>>());
        assert_eq!(chunked.total_kmers(), one_shot.total_kmers());
    }

    #[test]
    fn merge_all_is_the_same_as_counting_everything_in_one_counter() {
        let a: Vec<u128> = vec![1, 2, 2, 7];
        let b: Vec<u128> = vec![2, 3, 7, 7];
        let c: Vec<u128> = vec![9];

        let mut counters = Vec::new();
        for batch in [&a, &b, &c] {
            let mut counter = WideKmerCounter::new();
            counter.insert_batch(batch);
            counters.push(counter);
        }
        let merged = WideKmerCounter::merge_all(counters).finish();

        let mut all = a.clone();
        all.extend(&b);
        all.extend(&c);
        let single = counted(&[&all]);

        assert_eq!(merged.iter().collect::<Vec<_>>(), single.iter().collect::<Vec<_>>());
        assert_eq!(merged.total_kmers(), single.total_kmers());
    }

    #[test]
    fn an_empty_counter_finishes_to_an_empty_table() {
        let counts = WideKmerCounter::new().finish();
        assert_eq!(counts.distinct_kmers(), 0);
        assert_eq!(counts.total_kmers(), 0);
        assert!(counts.iter().next().is_none());
        assert!(counts.histogram().is_empty());
    }

    #[test]
    fn merging_empty_counters_is_not_a_special_case() {
        let mut one = WideKmerCounter::new();
        one.insert_batch(&[4, 4, 8]);
        let merged =
            WideKmerCounter::merge_all(vec![WideKmerCounter::new(), one, WideKmerCounter::new()])
                .finish();
        assert_eq!(merged.iter().collect::<Vec<_>>(), vec![(4, 2), (8, 1)]);
    }

    /// The bound the whole design rests on: a worker's raw buffer must
    /// stay near `RAW_FINALIZE_THRESHOLD` however many occurrences pass
    /// through it, or peak memory becomes a function of input size and the
    /// counter is unusable on a real file. Asserted on the *capacity*, not
    /// the length -- a `Vec` that is drained but keeps a gigabyte of
    /// capacity costs exactly as much as a full one.
    #[test]
    fn the_raw_buffer_stays_bounded_however_many_occurrences_arrive() {
        let mut counter = WideKmerCounter::new();
        // 20x the threshold, in realistic per-record batches, over a
        // distinct set far larger than any batch so compaction cannot
        // collapse it into nothing.
        let distinct = 5_000_000u128;
        for batch_start in (0..40_000_000u128).step_by(120) {
            let batch: Vec<u128> = (0..120).map(|i| (batch_start + i) % distinct).collect();
            counter.insert_batch(&batch);
        }
        assert!(
            counter.raw.capacity() <= RAW_FINALIZE_THRESHOLD * 2,
            "raw buffer grew to {} entries ({} MiB) against a {RAW_FINALIZE_THRESHOLD}-entry \
             bound -- peak memory now scales with input size",
            counter.raw.capacity(),
            counter.raw.capacity() * std::mem::size_of::<u128>() / (1024 * 1024)
        );
    }

    #[test]
    fn prune_drops_outside_the_window_and_reports_both_directions() {
        let mut counts = counted(&[&[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]);
        let total_before = counts.total_kmers();
        let (dropped_min, dropped_max) = counts.prune(2, Some(3));
        assert_eq!(dropped_min, 1, "kmer 1 occurs once");
        assert_eq!(dropped_max, 1, "kmer 4 occurs four times");
        assert_eq!(counts.iter().collect::<Vec<_>>(), vec![(2, 2), (3, 3)]);
        assert_eq!(
            counts.total_kmers(),
            total_before,
            "total_kmers is the normalization basis and must survive filtering"
        );
    }

    #[test]
    fn histogram_matches_the_counts_it_summarizes() {
        let counts = counted(&[&[1, 1, 1, 2, 2, 3]]);
        let histogram = counts.histogram();
        assert_eq!(histogram.get(&1), Some(&1));
        assert_eq!(histogram.get(&2), Some(&1));
        assert_eq!(histogram.get(&3), Some(&1));
    }

    /// Keys leave in ascending order and round-trip through the byte form
    /// the Parquet writer stores.
    #[test]
    fn key_bytes_are_ascending_and_round_trip() {
        let counts = counted(&[&[u128::MAX, 0, 1 << 100, 42]]);
        let keys: Vec<[u8; 16]> = counts.key_bytes().collect();
        for window in keys.windows(2) {
            assert!(window[0] < window[1], "stored keys are not ascending");
        }
        let decoded: Vec<u128> = keys.iter().map(|&k| wide_kmer::from_key_bytes(k)).collect();
        assert_eq!(decoded, counts.kmers);
    }
}
