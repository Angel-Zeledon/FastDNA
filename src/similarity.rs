// src/similarity.rs
//! Exact pairwise similarity across two or more already-counted k-mer
//! tables (`ktab::KmerTable`) -- the differentiator `fastdna similarity`
//! offers over `fastdna dist`: `dist` compares MinHash *sketches*
//! (`sketch.rs`), which are fast but only approximate the metrics they
//! report and, being presence/absence hashes, cannot use abundance at all.
//! This module reads the real, exact counts a `fastdna count` run already
//! wrote and computes both the set-theoretic metrics a sketch already
//! approximates (Jaccard, containment) *exactly*, and one a sketch cannot
//! produce even approximately: Bray-Curtis dissimilarity, which needs the
//! actual per-k-mer counts, not just which k-mers are present. KMC3's
//! `kmc_tools` can build the union/intersection tables that would let a
//! caller compute these by hand, but reports no similarity metric of its
//! own at all -- this is squarely "an operation on counted k-mers" (this
//! crate's mission rule), not a reimplementation of anything `kmc_tools`
//! already does.
//!
//! # Why `pub`, not `pub(crate)`
//!
//! `src/lib.rs`'s module-visibility comment draws the line at whether a
//! module is "a strategy or a storage detail whose shape is documented as
//! subject to change" (`pub(crate)`: `binned`, `disk_spill`, `minimizer`,
//! ...) versus a capability a consumer of `fastdna_core` is expected to
//! name directly (`pub`: `setops`, `sketch`, `ktab`, ...). `pairwise_
//! similarity` is the latter: it is the library entry point the `fastdna
//! similarity` CLI subcommand is a thin wrapper over (the same
//! relationship `setops::union`/`intersect`/`diff` have with their own CLI
//! subcommands), and nothing about its output shape (`PairSimilarity`) is
//! an internal implementation detail -- it is the answer this module
//! exists to compute. `PairAccum` and `triangular_index`, its own
//! internal bookkeeping, stay private to this module for exactly the
//! opposite reason, the same split `setops.rs` draws between its public
//! `union`/`intersect`/`diff` and its crate-private `MultiTableMerge`/
//! `MergedRow`.
//!
//! # Why one merge pass, not one pass per pair
//!
//! `setops::MultiTableMerge` already streams, for every distinct k-mer
//! across N input tables, a `MergedRow` carrying each table's own count
//! for it (or `None`). Every statistic below -- `shared`/`only_a`/
//! `only_b`, the sum of per-k-mer minimums Bray-Curtis needs, and each
//! table's total occurrence count -- is a running sum over that same
//! stream: nothing here needs a second look at any row once it has been
//! read. `pairwise_similarity` therefore builds exactly one
//! `MultiTableMerge` regardless of how many tables or pairs are being
//! compared, the same "one merge, many derived answers" shape `setops::
//! union`/`intersect`/`diff` already use, rather than looping
//! `MultiTableMerge::new` once per pair (`n*(n-1)/2` merges, each re-doing
//! the I/O and the k-way heap merge of every *other* pair's merge from
//! scratch). The only per-row cost this adds over a plain union is
//! updating every pair's accumulator from the row already sitting in
//! memory, and only for the pairs that actually share it: a k-mer present
//! in 2 of 200 tables costs one accumulator update, not the 19,900 a full
//! pair sweep would cost. That is possible because the exclusive counts
//! are *derived* rather than accumulated (`only_a == |A| - shared`), so
//! the pairs where neither or only one table has the k-mer never need to
//! be visited at all.
//!
//! Nothing beyond one Arrow batch per input table (`RangeIter`'s own
//! bound, inherited from `MultiTableMerge`) and one `PairAccum` per pair
//! is held in memory at a time, however many distinct k-mers the tables
//! hold in total.
//!
//! # Why containment is reported in both directions
//!
//! `containment_ab = |A ∩ B| / |A|` and `containment_ba = |A ∩ B| / |B|`
//! answer different questions ("what fraction of A's k-mers are also in
//! B" vs. the reverse) and are only equal when `|A| == |B|`. This is the
//! same asymmetry `cli::CliDistMetric`'s own doc comment already draws for
//! `fastdna dist --metric containment`, and the same reason that
//! subcommand reports both directions rather than picking one arbitrarily
//! or averaging them into a single, less informative number.
//!
//! # The degenerate cases, decided and tested rather than left to produce
//! `NaN`
//!
//! Every denominator below is zero in exactly one situation: the table(s)
//! it divides by are empty. A `KmerTable`'s rows always carry a count
//! `>= 1` (a k-mer that was counted at all was counted at least once), so
//! "this table's total occurrence count is zero" and "this table has zero
//! distinct k-mers" are the same condition -- there is no case where one
//! is zero and the other is not.
//!
//! - **Jaccard** (`shared / (shared + only_a + only_b)`): zero only when
//!   *both* tables are empty (nothing shared, nothing exclusive to
//!   either). Defined as **1.0** -- two empty tables have no k-mer on
//!   which they disagree, so they are read as identical, the same
//!   "vacuously equal" reading two empty sets get in set theory. One empty
//!   and one non-empty table never hits this: `only_b` (or `only_a`) alone
//!   already makes the denominator positive, and the ordinary formula
//!   correctly returns `0.0`.
//! - **Containment** (`shared / |A|`, and the mirror `shared / |B|`): zero
//!   only when the table in the denominator is itself empty. Defined as
//!   **1.0** -- the empty set is a subset of every set, so "what fraction
//!   of A's (zero) k-mers are also in B" is vacuously "all of them",
//!   mirroring the same convention Jaccard's empty/empty case uses. This
//!   holds regardless of whether the *other* table is empty too.
//! - **Bray-Curtis** (`1 - 2*sum(min(count_a, count_b)) / (total_a +
//!   total_b)`): zero only when *both* tables are empty (`total_a ==
//!   total_b == 0`). Defined as **0.0** -- no dissimilarity, matching
//!   Jaccard's "two empty tables are identical" reading rather than the
//!   opposite ("maximally dissimilar") reading a `1.0` fallback would
//!   otherwise suggest for the same input. One empty and one non-empty
//!   table never hits this: `total_a + total_b` is already positive from
//!   the non-empty side, and the ordinary formula correctly returns `1.0`
//!   (every occurrence in the non-empty table is unmatched, the maximum
//!   possible dissimilarity).
//!
//! Two disjoint, non-empty tables need no special case at all: Jaccard and
//! containment both fall out of the ordinary formula as `0.0` (no shared
//! k-mers, positive denominators), and Bray-Curtis falls out as `1.0`
//! (`sum(min(...))` is `0` because nothing is shared, positive
//! denominator).

use crate::error::{FastDnaError, Result};
use crate::ktab::KmerTable;
use crate::setops::{check_same_k, MergedRow, MultiTableMerge};

/// One unordered pair's similarity, as returned by `pairwise_similarity`.
/// `index_a`/`index_b` index into the same `&[KmerTable]` slice
/// `pairwise_similarity` was called with (always `index_a < index_b`, one
/// entry per unordered pair) -- callers that need a label for each side
/// (the CLI, naming each table by the path it was opened from) zip these
/// back against their own list of inputs rather than this module trying to
/// invent a naming scheme a `KmerTable` itself has no notion of.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairSimilarity {
    /// Index of the first table in this pair, into the slice
    /// `pairwise_similarity` was given.
    pub index_a: usize,
    /// Index of the second table in this pair. Always `> index_a`.
    pub index_b: usize,
    /// `|A ∩ B|`: distinct canonical k-mers present in both tables.
    pub shared: u64,
    /// `|A \ B|`: distinct k-mers present in `index_a`'s table only.
    pub only_a: u64,
    /// `|B \ A|`: distinct k-mers present in `index_b`'s table only.
    pub only_b: u64,
    /// `shared / (shared + only_a + only_b)`, or `1.0` if both tables are
    /// empty -- see the module doc comment's "degenerate cases" section.
    pub jaccard: f64,
    /// `shared / |A|` (`|A|` = `index_a`'s table's own distinct k-mer
    /// count), or `1.0` if `A` is empty. Not symmetric with
    /// `containment_ba` in general -- see the module doc comment.
    pub containment_ab: f64,
    /// `shared / |B|`, or `1.0` if `B` is empty. The mirror of
    /// `containment_ab`.
    pub containment_ba: f64,
    /// `1 - 2*sum(min(count_a, count_b)) / (total_a + total_b)`, summed
    /// over every k-mer in the union of the two tables (a k-mer present in
    /// only one side contributes `min(count, 0) == 0` to the numerator),
    /// or `0.0` if both tables are empty -- see the module doc comment.
    pub bray_curtis: f64,
}

/// Per-pair running totals accumulated while `pairwise_similarity` streams
/// `MultiTableMerge`'s rows -- everything needed to compute one
/// `PairSimilarity` once the merge is exhausted, except each table's own
/// distinct-k-mer count (`KmerTable::len`, already free from Parquet
/// footer metadata -- see `ktab.rs`'s own doc comment -- so there is
/// nothing to gain by re-deriving it from the merge as well) and each
/// table's total occurrence count (`total_count` below, tracked
/// separately from any one pair since it does not depend on which other
/// table it is being compared against).
#[derive(Debug, Clone, Copy, Default)]
struct PairAccum {
    shared: u64,
    /// `sum(min(count_a, count_b))` over k-mers this pair shares. Only
    /// incremented when both sides are `Some` -- a k-mer exclusive to one
    /// side contributes `min(count, 0) == 0`, so skipping it here rather
    /// than adding a literal zero is the same value, cheaper.
    sum_min: u64,
}

/// Maps an unordered pair `(i, j)` with `i < j < n` to a dense index into a
/// `Vec` of length `n*(n-1)/2` -- the standard "strictly upper triangle of
/// an `n x n` matrix, row-major, diagonal and lower triangle omitted"
/// packing. Chosen over a `HashMap<(usize, usize), PairAccum>` for the
/// same reason `counter.rs` chose a flat sorted `Vec` over a `HashMap` in
/// the first place (`docs/BENCHMARKS.md`'s cache-locality argument): this
/// is looked up twice per row of the merge (once for the totals loop's
/// nested pair loop, `n*(n-1)/2` times per row), so a hash and a possible
/// collision chain on the hot path is strictly worse than a multiply and
/// an add into a flat array with no fallible lookup at all.
fn triangular_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j && j < n, "triangular_index requires i < j < n (i={i}, j={j}, n={n})");
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Exact pairwise similarity across every unordered pair in `tables`, in
/// one streaming pass over `setops::MultiTableMerge` -- see the module doc
/// comment for the metrics, why one merge suffices for every pair, why
/// containment is reported in both directions, and how the degenerate
/// (empty-table) cases are defined.
///
/// Rejects `tables` built at different `k` (`setops::check_same_k` --
/// their `kmer_u64` encodings would not even mean the same thing) and
/// fewer than two tables (a similarity between one table and nothing is
/// not a comparison).
///
/// # Cost
///
/// I/O is exactly one streaming pass over every input table combined --
/// the same `O(total distinct rows across all tables)` bound
/// `MultiTableMerge`'s own doc comment states, not `O(n)` separate passes.
/// Per row, this does `O(n^2)` work (updating every pair's accumulator),
/// but that work touches only the `Vec<Option<u32>>` `MergedRow` already
/// decoded into memory -- it reads nothing further from disk. For the
/// table counts this command is meant for (`--input` names a handful of
/// samples, not thousands), that `O(n^2)` in-memory bookkeeping is
/// negligible next to the I/O it shares with a plain union; it would only
/// dominate at an `n` this crate's cohort tooling (`cohort/`) already
/// handles with sketches, not exact tables, for exactly that reason.
pub fn pairwise_similarity(tables: &[KmerTable]) -> Result<Vec<PairSimilarity>> {
    check_same_k(tables)?;
    if tables.len() < 2 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "input tables",
            reason: format!(
                "similarity needs at least two tables to compare; got {} (a single table has \
                 nothing to compare itself against)",
                tables.len()
            ),
        });
    }

    let n = tables.len();
    let mut distinct = vec![0u64; n];
    let mut total_count = vec![0u64; n];
    let mut accum = vec![PairAccum::default(); n * (n - 1) / 2];
    // Reused across rows so the hot loop allocates nothing: which tables
    // hold the k-mer this row is about, and with what count.
    let mut present: Vec<(usize, u32)> = Vec::with_capacity(n);

    for row in MultiTableMerge::new(tables)? {
        let MergedRow { per_table, .. } = row?;

        present.clear();
        for (i, count) in per_table.iter().enumerate() {
            if let Some(c) = count {
                // Each table's own tallies, independent of any pairing:
                // how many distinct k-mers it holds, and its grand total
                // occurrence count (Bray-Curtis's denominator).
                distinct[i] += 1;
                total_count[i] += u64::from(*c);
                present.push((i, *c));
            }
        }

        // Only pairs that *share* this k-mer are touched. The exclusive
        // counts are not accumulated at all: `only_a` is exactly
        // `|A| - shared`, which the tallies above already know, so a k-mer
        // sitting in 2 of 200 tables costs one accumulator update here
        // instead of the 19,900 a full pair sweep would cost. Cohort-scale
        // input is precisely where that difference decides whether this
        // command is usable.
        for (slot, &(i, ci)) in present.iter().enumerate() {
            for &(j, cj) in &present[slot + 1..] {
                let entry = &mut accum[triangular_index(i, j, n)];
                entry.shared += 1;
                entry.sum_min += u64::from(ci.min(cj));
            }
        }
    }

    let mut results = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &accum[triangular_index(i, j, n)];

            // Derived, not accumulated: every k-mer of table `i` is either
            // shared with `j` or exclusive to `i`, so the exclusive counts
            // follow from tallies the same pass already produced. Deriving
            // them also removes the second source of truth an earlier
            // draft had -- it took `|A|` from Parquet row-group metadata
            // while taking `shared` from the merge, so a table whose
            // footer disagreed with its rows would have reported a
            // containment above 1.0 rather than failing.
            let only_a = distinct[i] - a.shared;
            let only_b = distinct[j] - a.shared;

            let jaccard_denom = a.shared + only_a + only_b;
            let jaccard = if jaccard_denom == 0 {
                // Both tables empty -- see the module doc comment.
                1.0
            } else {
                a.shared as f64 / jaccard_denom as f64
            };

            let containment_ab = if distinct[i] == 0 { 1.0 } else { a.shared as f64 / distinct[i] as f64 };
            let containment_ba = if distinct[j] == 0 { 1.0 } else { a.shared as f64 / distinct[j] as f64 };

            let bc_denom = total_count[i] + total_count[j];
            let bray_curtis = if bc_denom == 0 {
                // Both tables empty -- see the module doc comment.
                0.0
            } else {
                1.0 - 2.0 * a.sum_min as f64 / bc_denom as f64
            };

            results.push(PairSimilarity {
                index_a: i,
                index_b: j,
                shared: a.shared,
                only_a,
                only_b,
                jaccard,
                containment_ab,
                containment_ba,
                bray_curtis,
            });
        }
    }

    Ok(results)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;
    use std::path::PathBuf;

    const EPS: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!((actual - expected).abs() < EPS, "{what}: expected {expected}, got {actual}");
    }

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_similarity_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a table through the real counting + export path (not a
    /// hand-built Parquet file), matching `setops.rs`'s own test
    /// convention -- these tests exercise exactly the files `fastdna
    /// count` produces.
    fn build_table(name: &str, k: usize, entries: &[u64]) -> (KmerTable, PathBuf) {
        let path = temp_path(name);
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, &path, k, 1, false).unwrap();
        (KmerTable::open(&path).unwrap(), path)
    }

    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn identical_tables_are_maximally_similar() {
        let (a, pa) = build_table("identical_a", 4, &[1, 1, 1, 2, 3, 3]);
        let (b, pb) = build_table("identical_b", 4, &[1, 1, 1, 2, 3, 3]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];

        assert_eq!(p.shared, 3, "kmers 1, 2, 3 all shared");
        assert_eq!(p.only_a, 0);
        assert_eq!(p.only_b, 0);
        assert_close(p.jaccard, 1.0, "jaccard");
        assert_close(p.containment_ab, 1.0, "containment_ab");
        assert_close(p.containment_ba, 1.0, "containment_ba");
        assert_close(p.bray_curtis, 0.0, "bray_curtis");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn disjoint_tables_have_zero_jaccard_and_maximal_bray_curtis() {
        let (a, pa) = build_table("disjoint_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("disjoint_b", 4, &[3, 3, 3]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        let p = &pairs[0];

        assert_eq!(p.shared, 0);
        assert_eq!(p.only_a, 2, "kmers 1 and 2");
        assert_eq!(p.only_b, 1, "kmer 3");
        assert_close(p.jaccard, 0.0, "jaccard");
        assert_close(p.containment_ab, 0.0, "containment_ab");
        assert_close(p.containment_ba, 0.0, "containment_ba");
        assert_close(p.bray_curtis, 1.0, "bray_curtis: nothing shared, so maximally dissimilar");

        cleanup(&[pa, pb]);
    }

    /// Hand-computed partial overlap: A = {1: 3, 2: 1}, B = {1: 2, 3: 1}.
    /// shared = {1} (min(3,2)=2), only_a = {2}, only_b = {3}.
    /// jaccard = 1 / (1+1+1) = 1/3.
    /// containment_ab = 1/2 (|A|=2), containment_ba = 1/2 (|B|=2).
    /// total_a = 3+1 = 4, total_b = 2+1 = 3.
    /// bray_curtis = 1 - 2*2/(4+3) = 1 - 4/7 = 3/7.
    #[test]
    fn partial_overlap_matches_hand_computed_metrics() {
        let (a, pa) = build_table("partial_a", 4, &[1, 1, 1, 2]);
        let (b, pb) = build_table("partial_b", 4, &[1, 1, 3]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        let p = &pairs[0];

        assert_eq!(p.shared, 1);
        assert_eq!(p.only_a, 1);
        assert_eq!(p.only_b, 1);
        assert_close(p.jaccard, 1.0 / 3.0, "jaccard");
        assert_close(p.containment_ab, 0.5, "containment_ab");
        assert_close(p.containment_ba, 0.5, "containment_ba");
        assert_close(p.bray_curtis, 3.0 / 7.0, "bray_curtis");

        cleanup(&[pa, pb]);
    }

    /// `|A| != |B|` makes `containment_ab != containment_ba` explicit: A is
    /// fully contained in B (every one of A's 2 distinct k-mers is also in
    /// B), but B has 4 distinct k-mers of which only 2 are in A.
    #[test]
    fn containment_is_asymmetric_when_table_sizes_differ() {
        let (a, pa) = build_table("asym_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("asym_b", 4, &[1, 2, 3, 4]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        let p = &pairs[0];

        assert_eq!(p.shared, 2);
        assert_close(p.containment_ab, 1.0, "A fully contained in B: 2/2");
        assert_close(p.containment_ba, 0.5, "only half of B's kmers are in A: 2/4");
        assert!(
            (p.containment_ab - p.containment_ba).abs() > EPS,
            "containment must actually differ by direction here"
        );

        cleanup(&[pa, pb]);
    }

    #[test]
    fn mismatched_k_is_rejected() {
        let (a, pa) = build_table("mismatched_k_a", 4, &[1]);
        let (b, pb) = build_table("mismatched_k_b", 6, &[1]);

        match pairwise_similarity(&[a, b]) {
            Err(FastDnaError::InvalidConfig { reason, .. }) => assert!(reason.contains('k'), "{reason}"),
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }

        cleanup(&[pa, pb]);
    }

    #[test]
    fn requires_at_least_two_tables() {
        let (a, pa) = build_table("min_tables_a", 4, &[1]);
        match pairwise_similarity(std::slice::from_ref(&a)) {
            Err(FastDnaError::InvalidConfig { .. }) => {}
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[pa]);
    }

    #[test]
    fn two_empty_tables_are_defined_as_identical() {
        let (a, pa) = build_table("empty_both_a", 4, &[]);
        let (b, pb) = build_table("empty_both_b", 4, &[]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        let p = &pairs[0];

        assert_eq!(p.shared, 0);
        assert_eq!(p.only_a, 0);
        assert_eq!(p.only_b, 0);
        assert_close(p.jaccard, 1.0, "two empty tables: defined as identical, not NaN");
        assert_close(p.containment_ab, 1.0, "empty A vacuously contained in empty B");
        assert_close(p.containment_ba, 1.0, "empty B vacuously contained in empty A");
        assert_close(p.bray_curtis, 0.0, "two empty tables: defined as no dissimilarity, not NaN");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn one_empty_table_against_a_non_empty_one() {
        let (a, pa) = build_table("empty_one_a", 4, &[]);
        let (b, pb) = build_table("empty_one_b", 4, &[1, 1, 2]);

        let pairs = pairwise_similarity(&[a, b]).unwrap();
        let p = &pairs[0];

        assert_eq!(p.shared, 0);
        assert_eq!(p.only_a, 0);
        assert_eq!(p.only_b, 2);
        assert_close(p.jaccard, 0.0, "no shared kmers, and the denominator is not zero here");
        assert_close(p.containment_ab, 1.0, "empty A is vacuously contained in B");
        assert_close(p.containment_ba, 0.0, "none of B's kmers are in empty A");
        assert_close(p.bray_curtis, 1.0, "every occurrence in B is unmatched: maximal dissimilarity");

        cleanup(&[pa, pb]);
    }

    /// The one-pass requirement's own regression test: three tables with a
    /// different overlap pattern for each of the three pairs, verified
    /// against hand-computed expectations from a *single*
    /// `pairwise_similarity` call -- proving the shared merge keeps every
    /// pair's accumulator independent of the others, not just that the
    /// pairwise formulas are correct in isolation.
    ///
    /// A = {1: 2, 2: 1}, B = {1: 1, 3: 3}, C = {2: 4, 3: 1}.
    /// (0,1) A-B: shared={1} min(2,1)=1, only_a={2}, only_b={3}.
    /// (0,2) A-C: shared={2} min(1,4)=1, only_a={1}, only_b={3}.
    /// (1,2) B-C: shared={3} min(3,1)=1, only_a={1}, only_b={2}.
    #[test]
    fn three_table_run_computes_every_pair_correctly_in_one_pass() {
        let (a, pa) = build_table("three_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("three_b", 4, &[1, 3, 3, 3]);
        let (c, pc) = build_table("three_c", 4, &[2, 2, 2, 2, 3]);

        let pairs = pairwise_similarity(&[a, b, c]).unwrap();
        assert_eq!(pairs.len(), 3);

        let ab = pairs.iter().find(|p| p.index_a == 0 && p.index_b == 1).unwrap();
        assert_eq!(ab.shared, 1);
        assert_eq!(ab.only_a, 1);
        assert_eq!(ab.only_b, 1);
        assert_close(ab.jaccard, 1.0 / 3.0, "A-B jaccard");
        // total_a = 2+1 = 3, total_b = 1+3 = 4, sum_min = min(2,1) = 1.
        assert_close(ab.bray_curtis, 1.0 - 2.0 * 1.0 / 7.0, "A-B bray_curtis");

        let ac = pairs.iter().find(|p| p.index_a == 0 && p.index_b == 2).unwrap();
        assert_eq!(ac.shared, 1);
        assert_eq!(ac.only_a, 1);
        assert_eq!(ac.only_b, 1);
        assert_close(ac.jaccard, 1.0 / 3.0, "A-C jaccard");
        // total_a = 3, total_c = 4+1 = 5, sum_min = min(1,4) = 1.
        assert_close(ac.bray_curtis, 1.0 - 2.0 * 1.0 / 8.0, "A-C bray_curtis");

        let bc = pairs.iter().find(|p| p.index_a == 1 && p.index_b == 2).unwrap();
        assert_eq!(bc.shared, 1);
        assert_eq!(bc.only_a, 1);
        assert_eq!(bc.only_b, 1);
        assert_close(bc.jaccard, 1.0 / 3.0, "B-C jaccard");
        // total_b = 4, total_c = 5, sum_min = min(3,1) = 1.
        assert_close(bc.bray_curtis, 1.0 - 2.0 * 1.0 / 9.0, "B-C bray_curtis");

        cleanup(&[pa, pb, pc]);
    }
}
