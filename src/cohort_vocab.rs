// src/cohort_vocab.rs
//! Cohort-scale vocabulary selection and projection for
//! `python/fastdna/sklearn.py::KmerVectorizer`'s `disk_backed=True` path
//! (`docs/audit/ml-gaps.md` G-9's remaining half: `chunk_size` already
//! bounds memory for *learning* the vocabulary -- see that module's own
//! `_learn_vocabulary_streaming` -- but the final *projection* step still
//! took every sample's full, unfiltered k-mer table into memory at once via
//! `_CohortCounts`/`_project`, which does not fit for a cohort of thousands
//! of samples at a few million distinct k-mers each even though the final
//! vocabulary is typically a few thousand columns).
//!
//! # Why this is a new module, not folded into `setops.rs` or `ktab.rs`
//!
//! Both functions here are built entirely on `ktab::KmerTable`'s public
//! `open`/`iter` contract and, for `rank_vocabulary`, on `setops::
//! MultiTableMerge` -- the same binary-heap k-way merge `setops::union`
//! already runs across several tables. Neither reads a table's internals
//! directly, mirroring `setops.rs`'s own arm's-length relationship with
//! `ktab.rs` (that module's doc comment). What is different here, and what
//! justifies a separate module rather than two more functions in
//! `setops.rs`, is the *question* being answered: `setops.rs` combines
//! several tables into one same-shaped `(kmer_u64, frequency)` table;
//! this module answers two questions specific to a fixed, ranked feature
//! set -- "which k-mers make the cut" (`rank_vocabulary`) and "where does
//! each sample land in that fixed coordinate system" (`project_onto_
//! vocabulary`) -- neither of which produces a `KmerTable`-shaped result at
//! all (a COO triple list, or a ranking, not a sorted `(u64, u32)` stream).
//!
//! # Why `rank_vocabulary` reuses `setops::MultiTableMerge` directly
//!
//! `MultiTableMerge::next` already computes, for every distinct k-mer
//! across every input table, exactly one `MergedRow { kmer, per_table }`
//! where `per_table[i]` is `Some(count)` if table `i` has this k-mer and
//! `None` otherwise. Prevalence -- "how many of these training samples
//! contain this k-mer" -- is precisely `per_table.iter().filter(|c|
//! c.is_some()).count()`, and total_freq is the sum of the `Some` entries:
//! both of the tallies `python/fastdna/sklearn.py::_select_vocabulary`
//! already ranks by. There is nothing to reimplement -- only a ranking and
//! (optionally) a bounded top-`n` selection to add on top of a merge this
//! crate already has, tested, and streaming.
//!
//! # Why `project_onto_vocabulary` does not reuse `MultiTableMerge`
//!
//! `MultiTableMerge` is a k-way merge *across* tables, useful when a
//! question needs every table's view of the same k-mer at once (union,
//! intersect, diff, and this module's own `rank_vocabulary`). Projection
//! needs the opposite shape: one sample's rows, checked one at a time
//! against an already-fixed, already-small vocabulary -- there is no
//! "advance whichever source is behind" step, because the vocabulary is a
//! static lookup table, not another stream to merge against. A resident
//! `HashMap<u64, u32>` built once from the (small, ranked) vocabulary and
//! reused across every sample's own `KmerTable::iter()` pass answers this
//! in one linear scan per sample, at `O(top_n)` fixed memory for the
//! lookup table itself -- the same "resident structure built once, queried
//! per row" shape `read_filter::ReferenceIndex` uses for reference
//! filtering, except keyed by column index instead of by presence alone.

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::Result;
use crate::ktab::KmerTable;
use crate::setops::{self, MultiTableMerge};

/// How a cohort's distinct k-mers are ordered when only `top_n` of them can
/// be kept. Both variants are implemented identically by
/// `python/fastdna/sklearn.py::_select_vocabulary`, which is the reason
/// this is an explicit parameter rather than a constant: the two code paths
/// (in-memory NumPy and this one) must not be able to drift apart, and
/// `python/tests/test_sklearn.py::test_disk_backed_fit_matches_in_memory_
/// fit_exactly` fails if they do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabularyRanking {
    /// `(prevalence desc, total_freq desc, kmer_u64 asc)`. The right rule
    /// when the feature value is a *count*: a k-mer present in every sample
    /// still varies in depth between them, so universal presence is a mark
    /// of a trustworthy, sample-general feature rather than a useless one.
    PrevalenceThenFrequency,
    /// `(min(prevalence, n_samples - prevalence) desc, kmer_u64 asc)`. The
    /// right rule when the feature value is *binary presence*, where that
    /// quantity is a monotone function of the feature's variance: a k-mer
    /// present in every sample encodes to 1 everywhere and carries exactly
    /// no information, and one present in half the samples carries the most
    /// any binary feature can.
    ///
    /// Ties are broken by `kmer_u64` alone, deliberately **not** by
    /// `total_freq`: depth is the quantity presence encoding exists to
    /// ignore (see `KmerVectorizer.__init__`'s comment on
    /// `representation`), and using it here would concentrate the
    /// vocabulary in whichever locus happened to be sequenced deepest.
    /// Ordering by the k-mer's own encoding instead scatters the selection
    /// across the genome and stays deterministic.
    BinaryInformativeness,
}

/// K-way merges every sample table's sorted `(kmer_u64, frequency)` stream
/// (`setops::MultiTableMerge` -- see the module doc comment) and ranks the
/// resulting distinct k-mers by `ranking`, the exact tie-break order
/// `python/fastdna/sklearn.py::_select_vocabulary` already implements for
/// the corresponding `representation` -- this is a drop-in replacement for
/// that ranking (run without ever materializing the whole cohort's rows in
/// memory at once), not a new ranking policy.
///
/// Returns `(kmer_u64, prevalence, total_freq)` as three parallel vectors,
/// already in ranked (best-first) order -- there is nothing left for a
/// caller to sort. `prevalence` and `total_freq` are reported for every
/// returned k-mer whichever ranking chose it, so a caller can always see
/// the tallies behind the order it got.
///
/// Under `BinaryInformativeness`, a k-mer present in *every* sample is
/// dropped outright rather than ranked last: its column is constant, so it
/// cannot inform any model, and returning it would let a `top_n` larger
/// than the informative set quietly refill the vocabulary with the exact
/// features that rule exists to exclude. The returned vocabulary is
/// therefore sometimes shorter than `top_n`, which is the honest answer.
///
/// `top_n`: `None` returns every distinct k-mer across `table_paths` (after
/// `min_count`/other per-sample filtering, which already happened when
/// each table was counted and written). `Some(n)` returns at most the top
/// `n`, selected with a bounded min-heap of size `n`: at any point during
/// the merge, the heap holds only the best `n` candidates seen so far, so
/// peak memory for the ranked-selection path is `O(n)`, never `O(distinct
/// k-mers across the cohort)` -- unlike the `None` path, which necessarily
/// holds every distinct k-mer's tally at once because every one of them is
/// part of the answer.
///
/// `table_paths` may be empty (returns three empty vectors -- there is
/// nothing to rank) or name a single table (ranks that one sample's own
/// k-mers; `setops::union`/`intersect`'s "at least two tables" requirement
/// does not apply here, since ranking is a well-defined question for one
/// sample too). Every table must share the same `k`
/// (`setops::check_same_k`, the identical guard `union`/`intersect`/`diff`
/// already enforce) -- their `kmer_u64` values are otherwise not even
/// comparable.
pub fn rank_vocabulary(
    table_paths: &[PathBuf],
    top_n: Option<usize>,
    ranking: VocabularyRanking,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u32>)> {
    if table_paths.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let tables: Vec<KmerTable> = table_paths.iter().map(KmerTable::open).collect::<Result<_>>()?;
    setops::check_same_k(&tables)?;

    // "Constant across the cohort" is a statement about a cohort. With a
    // single table every k-mer is present in every sample, so
    // `BinaryInformativeness` would score all of them zero and drop the
    // lot, answering "which of this sample's k-mers matter" with nothing
    // -- when that question does still have a good answer. Variance needs
    // two samples to exist at all, so below two the count-shaped rule is
    // the only one that means anything, and it is what runs.
    // `python/fastdna/sklearn.py::_select_vocabulary` makes the identical
    // exception, for the identical reason -- the equivalence tests in
    // `python/tests/test_sklearn.py` cover one-sample cohorts too.
    let n_samples = table_paths.len() as u32;
    let ranking = if n_samples < 2 {
        VocabularyRanking::PrevalenceThenFrequency
    } else {
        ranking
    };

    let merge = MultiTableMerge::new(&tables)?;

    match top_n {
        None => {
            // Every distinct k-mer is part of the answer, so nothing can be
            // dropped as the merge streams -- this path's memory really is
            // O(distinct k-mers), which is the honest cost of "return
            // everything" (see the doc comment above).
            let mut ranked: Vec<Candidate> = Vec::new();
            for row in merge {
                let row = row?;
                if let Some(candidate) = rank_candidate(&row, ranking, n_samples) {
                    ranked.push(candidate);
                }
            }
            // Descending by key: the "greater is better" order
            // `rank_candidate` encodes for whichever ranking is in force.
            ranked.sort_unstable_by_key(|candidate| std::cmp::Reverse(candidate.0));
            Ok(unzip_ranked(ranked))
        }
        Some(top_n) => {
            // A bounded min-heap of size `top_n`: `BinaryHeap` is a
            // max-heap, so wrapping each candidate's key in `Reverse`
            // turns "pop the largest `Reverse(key)`" into "pop the
            // smallest `key`" -- i.e. the current worst-ranked kept
            // candidate. Pushing every new candidate and then popping
            // whenever the heap grows past `top_n` keeps exactly the best
            // `top_n` candidates seen so far, at any point in the merge --
            // the same online top-k selection `_select_vocabulary`'s own
            // `np.lexsort` cannot do without the whole array resident.
            let mut heap: BinaryHeap<std::cmp::Reverse<Candidate>> =
                BinaryHeap::with_capacity(top_n.saturating_add(1));
            for row in merge {
                let row = row?;
                let Some(candidate) = rank_candidate(&row, ranking, n_samples) else {
                    continue;
                };
                heap.push(std::cmp::Reverse(candidate));
                if heap.len() > top_n {
                    heap.pop();
                }
            }
            let mut ranked: Vec<Candidate> = heap.into_iter().map(|std::cmp::Reverse(item)| item).collect();
            ranked.sort_unstable_by_key(|candidate| std::cmp::Reverse(candidate.0));
            Ok(unzip_ranked(ranked))
        }
    }
}

/// `(primary, secondary, Reverse(kmer))`, with "greater is better"
/// semantics throughout, so a plain `Ord`/`cmp` on the triple answers
/// "which candidate ranks first" directly and neither call site above
/// needs a bespoke comparator. What the two leading fields *hold* depends
/// on the `VocabularyRanking` in force (see `rank_candidate`); the last
/// field never does: `Reverse(kmer)` turns "smaller `kmer_u64` is better"
/// into "bigger `Reverse(kmer)` is better", which is the deterministic
/// final tie-break both rankings share -- and the one
/// `_select_vocabulary` itself spells `distinct` (ascending) in its
/// `np.lexsort`.
type RankKey = (u32, u32, std::cmp::Reverse<u64>);

/// A candidate k-mer: the key that decides where it ranks, followed by the
/// three values `rank_vocabulary` returns for it.
///
/// The tallies ride alongside the key rather than being read back out of
/// it, because under `BinaryInformativeness` the key no longer *contains*
/// them (its primary field is `min(prevalence, n_samples - prevalence)`),
/// and the returned `prevalence`/`total_freq` must be the real tallies
/// under either ranking. Ordering still only ever consults the key: it
/// ends in `Reverse(kmer)`, which is distinct for every candidate, so no
/// comparison reaches the trailing fields.
type Candidate = (RankKey, u64, u32, u32);

/// Scores one merged row under `ranking`, or `None` if that ranking
/// excludes it from the vocabulary outright.
///
/// The only exclusion is `BinaryInformativeness`'s: a k-mer present in all
/// `n_samples` samples encodes to 1 in every one of them, so its column is
/// constant and cannot inform any model. Ranking it last would not be
/// enough -- a `top_n` larger than the informative set would refill the
/// vocabulary with exactly those k-mers -- so it is dropped here, before
/// the top-`n` selection ever sees it. See `VocabularyRanking`'s own doc
/// comment for why prevalence n/2 is the maximum and why the tie-break
/// deliberately skips `total_freq`.
fn rank_candidate(
    row: &setops::MergedRow,
    ranking: VocabularyRanking,
    n_samples: u32,
) -> Option<Candidate> {
    let prevalence = row.per_table.iter().filter(|c| c.is_some()).count() as u32;
    let total_freq = row.per_table.iter().flatten().fold(0u32, |acc, &c| acc.saturating_add(c));

    let key = match ranking {
        VocabularyRanking::PrevalenceThenFrequency => {
            (prevalence, total_freq, std::cmp::Reverse(row.kmer))
        }
        VocabularyRanking::BinaryInformativeness => {
            // A merged row exists only because some table has this k-mer,
            // so `prevalence >= 1` and this saturating difference is a
            // real subtraction; informativeness reaches 0 only at
            // `prevalence == n_samples`, the constant column dropped
            // below.
            let informativeness = prevalence.min(n_samples.saturating_sub(prevalence));
            if informativeness == 0 {
                return None;
            }
            (informativeness, 0, std::cmp::Reverse(row.kmer))
        }
    };

    Some((key, row.kmer, prevalence, total_freq))
}

/// Splits a ranked `Candidate` list (already in best-first order) back
/// into the three parallel vectors `rank_vocabulary` returns -- so there
/// is exactly one place (`rank_candidate`) that computes a k-mer's
/// prevalence and total_freq and exactly one (this function) that unpacks
/// them, instead of two call sites separately reaching into a `MergedRow`
/// and risking disagreement about what "prevalence" means.
fn unzip_ranked(ranked: Vec<Candidate>) -> (Vec<u64>, Vec<u32>, Vec<u32>) {
    let mut kmers = Vec::with_capacity(ranked.len());
    let mut prevalence = Vec::with_capacity(ranked.len());
    let mut total_freq = Vec::with_capacity(ranked.len());
    for (_, kmer, prev, freq) in ranked {
        kmers.push(kmer);
        prevalence.push(prev);
        total_freq.push(freq);
    }
    (kmers, prevalence, total_freq)
}

/// Projects every sample table in `table_paths` onto the fixed `vocabulary`
/// (typically `rank_vocabulary`'s own `kmer_u64` output, or any other
/// caller-supplied list of `kmer_u64` codes -- the caller does not need to
/// pre-sort or de-duplicate it), returning a COO triple `(sample_row_index,
/// vocabulary_col_index, count)` as three parallel vectors: one entry per
/// `(sample, k-mer)` pair where the sample's table actually contains that
/// vocabulary k-mer. A vocabulary k-mer absent from every sample table
/// simply never appears in the output -- there is no "emit a zero row" step
/// (`scipy.sparse.csr_matrix` already treats an absent entry as zero).
///
/// `vocabulary_col_index` is the position of that k-mer in the *caller's
/// own* `vocabulary` slice (not a re-sorted position): the whole point of
/// this function is that its output can be assembled into a
/// `(len(table_paths), len(vocabulary))` sparse matrix whose columns line
/// up with `vocabulary` exactly as given -- `rank_vocabulary`'s own output
/// is already in ranked order, not `kmer_u64` order, and re-sorting it here
/// would silently break that alignment for any caller relying on it (which
/// `python/fastdna/sklearn.py::KmerVectorizer` does, via `self.
/// vocabulary_`). A resident `HashMap<u64, u32>` built once from
/// `vocabulary` (kmer -> its original index) answers "is this sample's
/// k-mer one of ours, and which column" in O(1) per row, with no sort
/// needed for correctness at all -- unlike `read_filter::ReferenceIndex`'s
/// sorted array plus `binary_search`, a hash map does not care what order
/// its keys were inserted in, which is the concrete mechanism behind "the
/// caller does not need to pre-sort": there is no comparison-based
/// structure here for input order to violate. If `vocabulary` contains a
/// duplicate `kmer_u64`, the later occurrence's index wins (in practice
/// this never happens with `rank_vocabulary`'s own output, which ranks
/// each *distinct* k-mer exactly once).
///
/// Streaming, per sample: each sample's `KmerTable` is opened, iterated
/// once via `KmerTable::iter()`, and dropped before the next sample's table
/// is opened -- never more than one sample's own table plus the vocabulary
/// map resident at once. That, together with the vocabulary map's own
/// `O(vocabulary_size)` footprint, is the whole memory bound this function
/// exists to provide over `python/fastdna/sklearn.py::_project`'s previous
/// approach of concatenating every sample's *entire, unfiltered* k-mer
/// table into one `_CohortCounts` before projecting.
///
/// Every sample table must share the same `k` (checked incrementally,
/// against the first table's own `k`, as each subsequent table is opened --
/// the same requirement `setops::check_same_k` enforces for `rank_
/// vocabulary` and every `setops.rs` operation, applied here one table at a
/// time rather than to a `Vec` held open all at once, to preserve the "one
/// table resident" memory bound above).
pub fn project_onto_vocabulary(
    table_paths: &[PathBuf],
    vocabulary: &[u64],
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let mut vocab_index: HashMap<u64, u32> = HashMap::with_capacity(vocabulary.len());
    for (index, &kmer) in vocabulary.iter().enumerate() {
        vocab_index.insert(kmer, index as u32);
    }

    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();

    let mut expected_k: Option<usize> = None;
    for (sample_index, path) in table_paths.iter().enumerate() {
        let table = KmerTable::open(path)?;
        match expected_k {
            None => expected_k = Some(table.k()),
            Some(k) if k == table.k() => {}
            Some(k) => {
                return Err(crate::error::FastDnaError::InvalidConfig {
                    parameter: "table_paths",
                    reason: format!(
                        "table {sample_index} ({}) has k={}, but table 0 has k={k} -- projection \
                         requires every sample table to share the same k, the same requirement \
                         setops.rs's operations enforce",
                        path.display(),
                        table.k()
                    ),
                });
            }
        }

        if vocab_index.is_empty() {
            // Nothing in the vocabulary can ever match -- skip straight to
            // the next sample rather than paying for a full table scan
            // that is guaranteed to produce zero rows. `expected_k` above
            // still ran first, so a k mismatch is still caught even when
            // the vocabulary is empty.
            continue;
        }

        for row in table.iter()? {
            let (kmer, count) = row?;
            if let Some(&col) = vocab_index.get(&kmer) {
                rows.push(sample_index as u32);
                cols.push(col);
                values.push(count);
            }
        }
    }

    Ok((rows, cols, values))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_cohort_vocab_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a table through the real counting + export path (not a
    /// hand-built Parquet file), matching `setops.rs`'s/`ktab.rs`'s own
    /// test convention -- these tests exercise exactly the files `fastdna
    /// count` produces.
    fn build_table(name: &str, k: usize, entries: &[u64]) -> PathBuf {
        let path = temp_path(name);
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, &path, k, 1, false).unwrap();
        path
    }

    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    // -- rank_vocabulary --------------------------------------------------

    #[test]
    fn rank_vocabulary_of_empty_table_paths_is_three_empty_vectors() {
        let (kmers, prevalence, total_freq) = rank_vocabulary(&[], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert!(kmers.is_empty());
        assert!(prevalence.is_empty());
        assert!(total_freq.is_empty());
    }

    #[test]
    fn rank_vocabulary_of_a_single_table_ranks_its_own_kmers() {
        let a = build_table("rank_single_a", 4, &[1, 1, 1, 2]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(std::slice::from_ref(&a), None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        // Both k-mers are present in exactly one (the only) sample, so
        // prevalence ties at 1 and total_freq breaks the tie: kmer 1 (3
        // occurrences) ranks before kmer 2 (1 occurrence).
        assert_eq!(kmers, vec![1, 2]);
        assert_eq!(prevalence, vec![1, 1]);
        assert_eq!(total_freq, vec![3, 1]);

        cleanup(&[a]);
    }

    #[test]
    fn rank_vocabulary_ranks_by_prevalence_first_not_total_freq() {
        // kmer 1: present in both samples (prevalence 2), total_freq 1 + 1 = 2.
        // kmer 2: present in only sample a, but with enormous depth there
        // (total_freq 100) -- must still rank behind kmer 1, since
        // prevalence is the primary key, not total_freq.
        let a = build_table("rank_prevalence_a", 4, &{
            let mut v = vec![1];
            v.extend(std::iter::repeat_n(2, 100));
            v
        });
        let b = build_table("rank_prevalence_b", 4, &[1]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers, vec![1, 2], "prevalence must dominate total_freq in the ranking");
        assert_eq!(prevalence, vec![2, 1]);
        assert_eq!(total_freq, vec![2, 100]);

        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_breaks_remaining_ties_by_ascending_kmer() {
        // Both kmers 1 and 2 appear once in each of two samples: prevalence
        // 2, total_freq 2, tied on both -- must fall back to kmer_u64
        // ascending, i.e. 1 before 2.
        let a = build_table("rank_tiebreak_a", 4, &[1, 2]);
        let b = build_table("rank_tiebreak_b", 4, &[2, 1]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers, vec![1, 2]);
        assert_eq!(prevalence, vec![2, 2]);
        assert_eq!(total_freq, vec![2, 2]);

        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_top_n_none_returns_every_distinct_kmer() {
        let a = build_table("rank_topn_none_a", 4, &[1, 2, 3, 4, 5]);

        let (kmers, ..) = rank_vocabulary(std::slice::from_ref(&a), None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers.len(), 5);

        cleanup(&[a]);
    }

    #[test]
    fn rank_vocabulary_top_n_truncates_to_the_best_n_in_ranked_order() {
        // Five distinct kmers, one sample: ranking falls back to
        // total_freq (all tied at prevalence 1), so higher total_freq
        // ranks first, ties (there are none here) broken by ascending
        // kmer.
        let entries: Vec<u64> = vec![
            1, // total_freq 1
            2, 2, // total_freq 2
            3, 3, 3, // total_freq 3
            4, 4, 4, 4, // total_freq 4
            5, 5, 5, 5, 5, // total_freq 5
        ];
        let a = build_table("rank_topn_truncate_a", 4, &entries);

        let (kmers, prevalence, total_freq) = rank_vocabulary(std::slice::from_ref(&a), Some(3), VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers, vec![5, 4, 3], "top 3 by total_freq, best first");
        assert_eq!(prevalence, vec![1, 1, 1]);
        assert_eq!(total_freq, vec![5, 4, 3]);

        cleanup(&[a]);
    }

    #[test]
    fn rank_vocabulary_top_n_larger_than_distinct_count_returns_everything() {
        let a = build_table("rank_topn_oversized_a", 4, &[1, 2, 3]);

        let (kmers, ..) = rank_vocabulary(std::slice::from_ref(&a), Some(1000), VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers.len(), 3);

        cleanup(&[a]);
    }

    #[test]
    fn rank_vocabulary_top_n_respects_the_exact_tie_break_order_under_truncation() {
        // Three kmers tied on prevalence and total_freq (1 and 2 both in
        // both samples with the same counts; 3 only in one) -- exercise
        // that truncation to top_n=2 keeps the two SMALLEST kmers among a
        // tie, matching `_select_vocabulary`'s ascending-kmer tie-break,
        // not an arbitrary or insertion-order-dependent subset.
        let a = build_table("rank_topn_tiebreak_a", 4, &[1, 2, 3]);
        let b = build_table("rank_topn_tiebreak_b", 4, &[2, 1]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone()], Some(2), VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers, vec![1, 2], "kmer 3 (prevalence 1) must lose to the tied pair (prevalence 2)");
        assert_eq!(prevalence, vec![2, 2]);
        assert_eq!(total_freq, vec![2, 2]);

        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_with_an_empty_table_among_the_inputs_ranks_the_rest() {
        let a = build_table("rank_empty_member_a", 4, &[]);
        let b = build_table("rank_empty_member_b", 4, &[7, 7]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert_eq!(kmers, vec![7]);
        assert_eq!(prevalence, vec![1]);
        assert_eq!(total_freq, vec![2]);

        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_of_all_empty_tables_is_empty() {
        let a = build_table("rank_all_empty_a", 4, &[]);
        let b = build_table("rank_all_empty_b", 4, &[]);

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        assert!(kmers.is_empty());
        assert!(prevalence.is_empty());
        assert!(total_freq.is_empty());

        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_rejects_mismatched_k_across_tables() {
        let a = build_table("rank_mismatched_k_a", 4, &[1]);
        let b = build_table("rank_mismatched_k_b", 6, &[1]);

        match rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::PrevalenceThenFrequency) {
            Err(crate::error::FastDnaError::InvalidConfig { reason, .. }) => {
                assert!(reason.contains('k'), "{reason}");
            }
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[a, b]);
    }

    #[test]
    fn rank_vocabulary_matches_the_documented_selection_rule_end_to_end() {
        // A slightly larger synthetic cohort, ranked once via
        // rank_vocabulary and once by re-implementing the same rule
        // (prevalence desc, total_freq desc, kmer asc) directly over the
        // same tables' rows read back with KmerTable::iter -- an
        // independent check that the streaming merge-based ranking above
        // really does agree with a straightforward reference
        // implementation, not just with itself.
        let a = build_table("rank_e2e_a", 4, &[1, 1, 2, 3, 3, 3]);
        let b = build_table("rank_e2e_b", 4, &[1, 2, 2, 4]);
        let c = build_table("rank_e2e_c", 4, &[3, 4, 4, 4]);

        let tables = [
            KmerTable::open(&a).unwrap(),
            KmerTable::open(&b).unwrap(),
            KmerTable::open(&c).unwrap(),
        ];
        let mut reference: HashMap<u64, (u32, u32)> = HashMap::new();
        for table in &tables {
            for row in table.iter().unwrap() {
                let (kmer, freq) = row.unwrap();
                let entry = reference.entry(kmer).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += freq;
            }
        }
        let mut expected: Vec<(u64, u32, u32)> =
            reference.into_iter().map(|(k, (prev, freq))| (k, prev, freq)).collect();
        expected.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(y.2.cmp(&x.2)).then(x.0.cmp(&y.0)));

        let (kmers, prevalence, total_freq) = rank_vocabulary(&[a.clone(), b.clone(), c.clone()], None, VocabularyRanking::PrevalenceThenFrequency).unwrap();
        let actual: Vec<(u64, u32, u32)> =
            kmers.iter().zip(&prevalence).zip(&total_freq).map(|((&k, &p), &f)| (k, p, f)).collect();
        assert_eq!(actual, expected);

        cleanup(&[a, b, c]);
    }

    // -- rank_vocabulary, BinaryInformativeness ----------------------------
    //
    // The ranking a presence/absence matrix needs (see
    // `VocabularyRanking`'s doc comment). Its Python counterpart is
    // `python/fastdna/sklearn.py::_select_vocabulary`'s `representation ==
    // "presence"` branch, and `python/tests/test_review_findings_2026_09_
    // 02.py` pins the defect that made the split necessary.

    /// A cohort of `n_samples` tables where `kmer` is present in exactly
    /// `prevalence` of them -- the shape every test below needs, with the
    /// sample tables built through the real counting + export path like
    /// every other test here.
    fn cohort_with_prevalences(name: &str, n_samples: usize, kmer_prevalences: &[(u64, usize)]) -> Vec<PathBuf> {
        (0..n_samples)
            .map(|sample| {
                let entries: Vec<u64> = kmer_prevalences
                    .iter()
                    .filter(|(_, prevalence)| sample < *prevalence)
                    .map(|(kmer, _)| *kmer)
                    .collect();
                build_table(&format!("{name}_{sample}"), 4, &entries)
            })
            .collect()
    }

    #[test]
    fn binary_ranking_drops_kmers_present_in_every_sample() {
        // kmer 1 is in all 4 samples: presence-encoded it is 1 in every
        // row, variance zero. It must not appear at all -- not even last.
        let paths = cohort_with_prevalences("binary_drop", 4, &[(1, 4), (2, 2), (3, 1)]);

        let (kmers, prevalence, _) =
            rank_vocabulary(&paths, None, VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![2, 3], "the all-present k-mer must be dropped, not ranked last");
        assert_eq!(prevalence, vec![2, 1]);

        cleanup(&paths);
    }

    #[test]
    fn binary_ranking_prefers_half_present_over_nearly_universal() {
        // Informativeness is min(prevalence, n - prevalence) over n = 4:
        // kmer 2 (prevalence 2) scores 2; kmer 1 (prevalence 3) and kmer 3
        // (prevalence 1) both score 1 and tie, broken by ascending kmer.
        // Note kmer 1 would have ranked FIRST under
        // PrevalenceThenFrequency -- that inversion is the whole point.
        let paths = cohort_with_prevalences("binary_half", 4, &[(1, 3), (2, 2), (3, 1)]);

        let (kmers, prevalence, _) =
            rank_vocabulary(&paths, None, VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![2, 1, 3]);
        assert_eq!(prevalence, vec![2, 3, 1], "the reported tallies stay the real ones");

        cleanup(&paths);
    }

    #[test]
    fn binary_ranking_ignores_total_freq_when_breaking_ties() {
        // Both k-mers sit at prevalence 1 of 2, so informativeness ties.
        // kmer 7 carries 100x the depth of kmer 3 and still ranks behind
        // it: depth is exactly what presence encoding exists to ignore,
        // so the tie-break is the k-mer's own encoding, ascending.
        let a = build_table("binary_tie_a", 4, &{
            let mut v = vec![3];
            v.extend(std::iter::repeat_n(7, 100));
            v
        });
        let b = build_table("binary_tie_b", 4, &[9, 9]);

        let (kmers, _, total_freq) =
            rank_vocabulary(&[a.clone(), b.clone()], None, VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![3, 7, 9]);
        assert_eq!(total_freq, vec![1, 100, 2], "depth is reported, just not ranked on");

        cleanup(&[a, b]);
    }

    #[test]
    fn binary_ranking_returns_fewer_than_top_n_when_the_informative_set_is_smaller() {
        // The reason the constants are dropped rather than ranked last: a
        // generous top_n would otherwise refill the vocabulary with
        // exactly the all-ones columns this ranking exists to exclude.
        let paths = cohort_with_prevalences("binary_short", 3, &[(1, 3), (2, 3), (3, 1)]);

        let (kmers, ..) =
            rank_vocabulary(&paths, Some(50), VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![3], "a short vocabulary is the honest answer here");

        cleanup(&paths);
    }

    #[test]
    fn binary_ranking_top_n_keeps_the_most_informative_n() {
        // The bounded-heap path (Some(top_n)) must select on the same key
        // the full-sort path does -- n = 4, so kmer 2 (prevalence 2)
        // scores 2 and outranks both prevalence-3 and prevalence-1 k-mers.
        let paths = cohort_with_prevalences("binary_topn", 4, &[(1, 3), (2, 2), (3, 1)]);

        let (kmers, ..) =
            rank_vocabulary(&paths, Some(1), VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![2]);

        cleanup(&paths);
    }

    #[test]
    fn binary_ranking_of_a_single_sample_falls_back_to_prevalence_ranking() {
        // Every k-mer in a one-sample cohort is trivially "present in
        // every sample", so applying the binary rule literally would
        // return nothing. Below two samples there is no variance to rank
        // by, so the count-shaped rule runs instead -- the identical
        // exception `_select_vocabulary` makes.
        let a = build_table("binary_single", 4, &[1, 1, 1, 2]);

        let (kmers, prevalence, total_freq) =
            rank_vocabulary(std::slice::from_ref(&a), None, VocabularyRanking::BinaryInformativeness).unwrap();

        assert_eq!(kmers, vec![1, 2], "ranked by depth, exactly as PrevalenceThenFrequency would");
        assert_eq!(prevalence, vec![1, 1]);
        assert_eq!(total_freq, vec![3, 1]);

        cleanup(&[a]);
    }

    // -- project_onto_vocabulary -------------------------------------------

    #[test]
    fn project_onto_vocabulary_of_empty_table_paths_is_three_empty_vectors() {
        let (rows, cols, values) = project_onto_vocabulary(&[], &[1, 2, 3]).unwrap();
        assert!(rows.is_empty());
        assert!(cols.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn project_onto_vocabulary_of_an_empty_vocabulary_is_three_empty_vectors() {
        let a = build_table("project_empty_vocab_a", 4, &[1, 2, 3]);
        let (rows, cols, values) = project_onto_vocabulary(std::slice::from_ref(&a), &[]).unwrap();
        assert!(rows.is_empty());
        assert!(cols.is_empty());
        assert!(values.is_empty());
        cleanup(&[a]);
    }

    #[test]
    fn project_onto_vocabulary_keeps_only_vocabulary_kmers_and_reports_their_true_count() {
        let a = build_table("project_basic_a", 4, &[1, 1, 1, 2, 5]);
        // Vocabulary has 3 columns; only kmers 1 and 2 from this sample's
        // table are in it (5 is not).
        let (rows, cols, values) = project_onto_vocabulary(std::slice::from_ref(&a), &[10, 1, 2]).unwrap();

        let mut got: Vec<(u32, u32, u32)> =
            rows.into_iter().zip(cols).zip(values).map(|((r, c), v)| (r, c, v)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(0, 1, 3), (0, 2, 1)], "kmer 1 -> col 1 (count 3), kmer 2 -> col 2 (count 1)");

        cleanup(&[a]);
    }

    #[test]
    fn project_onto_vocabulary_column_index_matches_the_callers_own_vocabulary_order_not_sorted_order() {
        let a = build_table("project_order_a", 4, &[5, 9]);
        // Deliberately unsorted vocabulary: kmer 9 is column 0, kmer 5 is
        // column 1 -- the caller's own order must be preserved, not
        // re-sorted ascending internally.
        let (rows, cols, values) = project_onto_vocabulary(std::slice::from_ref(&a), &[9, 5]).unwrap();

        let mut got: Vec<(u32, u32, u32)> =
            rows.into_iter().zip(cols).zip(values).map(|((r, c), v)| (r, c, v)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(0, 0, 1), (0, 1, 1)], "kmer 9 -> col 0, kmer 5 -> col 1, per the given order");

        cleanup(&[a]);
    }

    #[test]
    fn project_onto_vocabulary_a_vocabulary_kmer_absent_from_every_sample_produces_no_rows() {
        let a = build_table("project_absent_a", 4, &[1, 2]);
        let b = build_table("project_absent_b", 4, &[1]);

        let (rows, cols, values) = project_onto_vocabulary(&[a.clone(), b.clone()], &[99]).unwrap();
        assert!(rows.is_empty());
        assert!(cols.is_empty());
        assert!(values.is_empty());

        cleanup(&[a, b]);
    }

    #[test]
    fn project_onto_vocabulary_indexes_sample_rows_by_position_in_table_paths() {
        let a = build_table("project_rowindex_a", 4, &[1]);
        let b = build_table("project_rowindex_b", 4, &[1]);
        let c = build_table("project_rowindex_c", 4, &[1]);

        let (rows, cols, values) = project_onto_vocabulary(&[a.clone(), b.clone(), c.clone()], &[1]).unwrap();
        let mut got: Vec<(u32, u32, u32)> =
            rows.into_iter().zip(cols).zip(values).map(|((r, c), v)| (r, c, v)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(0, 0, 1), (1, 0, 1), (2, 0, 1)]);

        cleanup(&[a, b, c]);
    }

    #[test]
    fn project_onto_vocabulary_with_a_single_table_projects_just_that_one_sample() {
        let a = build_table("project_single_a", 4, &[3, 3, 4]);
        let (rows, cols, values) = project_onto_vocabulary(std::slice::from_ref(&a), &[3, 4]).unwrap();

        let mut got: Vec<(u32, u32, u32)> =
            rows.into_iter().zip(cols).zip(values).map(|((r, c), v)| (r, c, v)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(0, 0, 2), (0, 1, 1)]);

        cleanup(&[a]);
    }

    #[test]
    fn project_onto_vocabulary_rejects_mismatched_k_across_sample_tables() {
        let a = build_table("project_mismatched_k_a", 4, &[1]);
        let b = build_table("project_mismatched_k_b", 6, &[1]);

        match project_onto_vocabulary(&[a.clone(), b.clone()], &[1]) {
            Err(crate::error::FastDnaError::InvalidConfig { reason, .. }) => {
                assert!(reason.contains('k'), "{reason}");
            }
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[a, b]);
    }

    #[test]
    fn project_onto_vocabulary_with_an_empty_sample_table_contributes_no_rows() {
        let a = build_table("project_empty_sample_a", 4, &[]);
        let b = build_table("project_empty_sample_b", 4, &[1, 1]);

        let (rows, cols, values) = project_onto_vocabulary(&[a.clone(), b.clone()], &[1]).unwrap();
        assert_eq!(rows, vec![1]);
        assert_eq!(cols, vec![0]);
        assert_eq!(values, vec![2]);

        cleanup(&[a, b]);
    }

    /// A composability smoke test bridging both functions: a vocabulary
    /// ranked by `rank_vocabulary` is immediately usable as `project_onto_
    /// vocabulary`'s input, with columns landing exactly where `rank_
    /// vocabulary`'s own output order says they should -- the intended
    /// `disk_backed=True` pipeline end to end, entirely in Rust.
    #[test]
    fn rank_then_project_round_trips_a_small_synthetic_cohort() {
        let a = build_table("roundtrip_a", 4, &[1, 1, 2, 3]);
        let b = build_table("roundtrip_b", 4, &[1, 2, 2, 4]);

        let (vocabulary, ..) = rank_vocabulary(&[a.clone(), b.clone()], Some(2), VocabularyRanking::PrevalenceThenFrequency).unwrap();
        // kmer 1: prevalence 2, total_freq 3. kmer 2: prevalence 2,
        // total_freq 3. Tied -> ascending kmer -> [1, 2].
        assert_eq!(vocabulary, vec![1, 2]);

        let (rows, cols, values) = project_onto_vocabulary(&[a.clone(), b.clone()], &vocabulary).unwrap();
        let mut got: Vec<(u32, u32, u32)> =
            rows.into_iter().zip(cols).zip(values).map(|((r, c), v)| (r, c, v)).collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 2)],
            "sample 0: kmer1 x2, kmer2 x1; sample 1: kmer1 x1, kmer2 x2"
        );

        cleanup(&[a, b]);
    }
}
