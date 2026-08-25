// src/cohort/matrix.rs
//! Builds a cohort-wide k-mer presence/count matrix directly from a set of
//! already-counted samples, without ever materializing a per-sample Arrow
//! `RecordBatch` (`ffi.rs::build_record_batch`'s `kmer_u64`/`kmer_sequence`/
//! `frequency` columns).
//!
//! # Why this exists
//!
//! `python/fastdna/gwas.py::cohort_presence_matrix` used to call
//! `fastdna.count()` once per sample -- which decodes *every* distinct
//! canonical k-mer that survived that sample's own `min_count` into an ASCII
//! string and writes it into an Arrow value buffer (`kmer::decode_kmer_into`,
//! `k` bytes per k-mer) -- and then immediately `pyarrow.concat_arrays`'d all
//! N samples' `kmer_u64`/`kmer_sequence`/`frequency` columns into one array
//! before it ever asked which k-mers actually survive the cohort-level
//! `min_samples` filter. Concretely, for `n` distinct k-mers in one sample
//! and k-mer length `k`, that per-sample build allocates and writes
//! `n * (8 [u64] + 4 [i32 offset] + k [ASCII bases] + 4 [u32 frequency])`
//! bytes, and `concat_arrays` then copies essentially the same
//! `n * (16 + k)` bytes a second time across the whole cohort -- almost
//! entirely for k-mers that, at typical `min_samples >= 2` thresholds, are
//! private to one sample and get thrown away moments later. At `k = 31` the
//! sequence bytes alone are 31 of those 47 bytes per row, i.e. the majority
//! of the traffic is a string nobody ends up keeping.
//!
//! This module instead works directly from each sample's already-sorted,
//! already-deduplicated `KmerCounter` table (`counter.rs`'s `finalized`
//! `Vec<(u64, u32)>`, which the pipeline produces regardless of whether this
//! module exists) and only ever decodes a k-mer to a string once it has
//! already been confirmed to survive both the `min_samples` filter and any
//! `max_kmers` truncation -- i.e. exactly the k-mers that end up as columns
//! of the returned matrix, not every k-mer of every sample.
//!
//! # Column order comes for free
//!
//! `kmer.rs` packs a k-mer's bases into a `u64` most-significant-base-first
//! (`decode_kmer_into` peels the *lowest* bits off first into the *last*
//! output byte), and `A`/`C`/`G`/`T` map to bit patterns `0`/`1`/`2`/`3` in
//! that same order. For two k-mers of equal length that makes ascending
//! `u64` order and ASCII-lexicographic order of the decoded string exactly
//! the same order -- so sorting candidate columns by `u64` (which
//! `KmerCounter`'s own sort-and-compact strategy already does, for free) is
//! sorting them the way `gwas.py`'s docstring promises `kmer_sequences` are
//! ordered, with no separate string sort required.
//!
//! # What this module does not do
//!
//! It does not run the counting pipeline itself (`process_stream_parallel`
//! and friends do that, unchanged) and it does not decide per-sample ids,
//! path validation, or FASTQ I/O -- callers (`ffi.rs::cohort_presence_matrix`)
//! hand it already-built `KmerCounter`s, one per sample, in row order.

use rustc_hash::FxHashMap;

use crate::counter::KmerCounter;
use crate::kmer::decode_kmer;

/// A cohort-wide k-mer matrix in COO (row, col, value) form, plus enough
/// bookkeeping for the caller to reproduce `gwas.py`'s truncation warning
/// without recomputing anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohortMatrix {
    /// Number of samples (matrix rows) this was built from.
    pub n_samples: usize,
    /// Number of columns in the returned matrix -- k-mers that passed
    /// `min_samples` and, if `max_kmers` was set and exceeded, survived the
    /// minor-sample-count ranking too.
    pub n_kmers: usize,
    /// How many k-mers passed `min_samples`, before any `max_kmers`
    /// truncation. Equal to `n_kmers` unless truncation happened.
    pub n_candidates: usize,
    /// The minor-sample-count (`min(present, absent)` across the cohort) of
    /// the highest-ranked k-mer that `max_kmers` still dropped. `None` when
    /// no truncation happened (either `max_kmers` was `None`, or every
    /// candidate fit).
    pub truncation_cutoff: Option<u32>,
    /// Sample index (row) of each nonzero entry. Same length as `col` and
    /// `value`.
    pub row: Vec<u32>,
    /// Column index of each nonzero entry, into `kmer_sequences`.
    pub col: Vec<u32>,
    /// That (sample, k-mer) pair's count.
    pub value: Vec<u32>,
    /// Decoded canonical k-mer of every column, ascending -- see the module
    /// doc comment for why this is already lexicographic.
    pub kmer_sequences: Vec<String>,
}

/// Builds a `CohortMatrix` from `counters` (one per sample, in row order).
///
/// `min_samples`: a k-mer must appear in at least this many samples to
/// become a column (the same semantics as `gwas.py::cohort_presence_matrix`'s
/// parameter of the same name).
///
/// `max_kmers`: if `Some` and more than `max_kmers` candidates survive
/// `min_samples`, only the `max_kmers` with the highest minor-sample-count
/// (`min(present, absent)`, the minor-allele-count analogue) are kept, ties
/// broken by prevalence, then total depth, then k-mer value -- the exact
/// ranking `gwas.py`'s own (now-superseded) `np.lexsort` used, reproduced
/// here so a caller cannot tell the two apart from their output.
///
/// `k`: the k-mer length, needed only to decode the final surviving columns
/// back into ASCII sequences.
pub fn build_cohort_matrix(
    counters: &[KmerCounter],
    min_samples: u32,
    max_kmers: Option<usize>,
    k: usize,
) -> CohortMatrix {
    let n_samples = counters.len();

    // Pass 1: one hash-map pass over every (sample, k-mer) occurrence,
    // folding each distinct k-mer's prevalence (how many samples carry it)
    // and total depth across the cohort. This is the direct analogue of
    // `gwas.py`'s `pc.dictionary_encode` + `np.bincount` pair, done without
    // ever materializing a k-mer's decoded sequence.
    let mut stats: FxHashMap<u64, (u32, u64)> = FxHashMap::default();
    for counter in counters {
        for (kmer, count) in counter.iter() {
            let entry = stats.entry(kmer).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += u64::from(count);
        }
    }

    // Pass 2: keep only k-mers observed in enough samples, then (if
    // `max_kmers` bites) rank and truncate exactly as `gwas.py` used to.
    let mut candidates: Vec<(u64, u32, u64)> = stats
        .into_iter()
        .filter(|&(_, (prevalence, _))| prevalence >= min_samples)
        .map(|(kmer, (prevalence, total_freq))| (kmer, prevalence, total_freq))
        .collect();
    let n_candidates = candidates.len();

    let mut truncation_cutoff = None;
    if let Some(cap) = max_kmers {
        if candidates.len() > cap {
            let n_samples_u32 = n_samples as u32;
            let minor_of = |prevalence: u32| prevalence.min(n_samples_u32 - prevalence);
            // Ranked by minor-sample-count descending (primary), then
            // prevalence descending, then total depth descending, then
            // k-mer value ascending -- the same tie-break chain
            // `np.lexsort((distinct, -total_freq, -prevalence,
            // -minor_sample_count))` applied, primary key last.
            candidates.sort_unstable_by(|&(a_kmer, a_prev, a_freq), &(b_kmer, b_prev, b_freq)| {
                minor_of(b_prev)
                    .cmp(&minor_of(a_prev))
                    .then_with(|| b_prev.cmp(&a_prev))
                    .then_with(|| b_freq.cmp(&a_freq))
                    .then_with(|| a_kmer.cmp(&b_kmer))
            });
            // The highest-ranked entry among the ones about to be dropped --
            // `gwas.py`'s `dropped[0]`.
            let (_, cutoff_prevalence, _) = candidates[cap];
            truncation_cutoff = Some(minor_of(cutoff_prevalence));
            candidates.truncate(cap);
        }
    }

    // Final column order: ascending k-mer value, i.e. lexicographic by
    // decoded sequence (see the module doc comment). A no-op pass when
    // `max_kmers` never re-sorted this away from the order `stats.into_iter()`
    // happened to yield.
    candidates.sort_unstable_by_key(|&(kmer, _, _)| kmer);

    let n_kmers = candidates.len();
    let nnz: usize = candidates.iter().map(|&(_, prevalence, _)| prevalence as usize).sum();

    let column_of: FxHashMap<u64, u32> = candidates
        .iter()
        .enumerate()
        .map(|(col, &(kmer, _, _))| (kmer, col as u32))
        .collect();

    // Pass 3: walk every sample's table once more, emitting a (row, col,
    // value) triple for every entry whose k-mer survived. `nnz` was computed
    // exactly above, so these three buffers are sized once and never
    // regrow.
    let mut row = Vec::with_capacity(nnz);
    let mut col = Vec::with_capacity(nnz);
    let mut value = Vec::with_capacity(nnz);
    for (sample_idx, counter) in counters.iter().enumerate() {
        // `n_samples` is bounded by how many `KmerCounts` a caller can hold
        // in memory at once, nowhere near `u32::MAX`.
        let sample_idx = sample_idx as u32;
        for (kmer, count) in counter.iter() {
            if let Some(&column) = column_of.get(&kmer) {
                row.push(sample_idx);
                col.push(column);
                value.push(count);
            }
        }
    }

    let kmer_sequences: Vec<String> =
        candidates.iter().map(|&(kmer, _, _)| decode_kmer(kmer, k)).collect();

    CohortMatrix {
        n_samples,
        n_kmers,
        n_candidates,
        truncation_cutoff,
        row,
        col,
        value,
        kmer_sequences,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter_from(kmers_with_counts: &[(u64, u32)]) -> KmerCounter {
        let mut c = KmerCounter::new();
        for &(kmer, count) in kmers_with_counts {
            for _ in 0..count {
                c.insert(kmer);
            }
        }
        c
    }

    #[test]
    fn two_samples_sharing_one_kmer_and_disjoint_on_another() {
        // Sample 0 has kmers {1: 3, 5: 1}; sample 1 has kmers {1: 2, 9: 4}.
        let counters = vec![counter_from(&[(1, 3), (5, 1)]), counter_from(&[(1, 2), (9, 4)])];

        let m = build_cohort_matrix(&counters, 1, None, 2);

        assert_eq!(m.n_samples, 2);
        assert_eq!(m.n_candidates, 3, "kmers 1, 5, 9 all pass min_samples=1");
        assert_eq!(m.n_kmers, 3);
        assert_eq!(m.truncation_cutoff, None);
        assert_eq!(m.row.len(), 4, "4 nonzero (sample, kmer) pairs total");

        // Columns are ascending by kmer value: 1 -> 0, 5 -> 1, 9 -> 2.
        let mut triples: Vec<(u32, u32, u32)> =
            m.row.iter().zip(&m.col).zip(&m.value).map(|((&r, &c), &v)| (r, c, v)).collect();
        triples.sort_unstable();
        assert_eq!(triples, vec![(0, 0, 3), (0, 1, 1), (1, 0, 2), (1, 2, 4)]);
    }

    #[test]
    fn min_samples_drops_kmers_private_to_one_sample() {
        let counters = vec![counter_from(&[(1, 3), (5, 1)]), counter_from(&[(1, 2), (9, 4)])];

        let m = build_cohort_matrix(&counters, 2, None, 2);

        assert_eq!(m.n_candidates, 1, "only kmer 1 is present in both samples");
        assert_eq!(m.n_kmers, 1);
        assert_eq!(m.kmer_sequences.len(), 1);
        assert_eq!(m.row, vec![0, 1]);
        assert_eq!(m.col, vec![0, 0]);
        assert_eq!(m.value, vec![3, 2]);
    }

    #[test]
    fn min_samples_above_every_prevalence_yields_an_empty_matrix() {
        let counters = vec![counter_from(&[(1, 1)]), counter_from(&[(2, 1)])];

        let m = build_cohort_matrix(&counters, 2, None, 2);

        assert_eq!(m.n_kmers, 0);
        assert!(m.row.is_empty());
        assert!(m.col.is_empty());
        assert!(m.value.is_empty());
        assert!(m.kmer_sequences.is_empty());
    }

    #[test]
    fn columns_are_ordered_ascending_by_kmer_value() {
        // Inserted out of order; the merge/hash pass must not leak that
        // order into the output.
        let counters =
            vec![counter_from(&[(9, 1), (1, 1), (5, 1)]), counter_from(&[(9, 1), (1, 1), (5, 1)])];

        let m = build_cohort_matrix(&counters, 1, None, 4);

        // decode_kmer(1, 4) < decode_kmer(5, 4) < decode_kmer(9, 4) is
        // exactly what ascending kmer order gives (see the module doc
        // comment); confirm both agree.
        let mut sorted_sequences = m.kmer_sequences.clone();
        sorted_sequences.sort();
        assert_eq!(m.kmer_sequences, sorted_sequences, "columns must already be lexicographic");
    }

    #[test]
    fn max_kmers_keeps_the_highest_minor_sample_count_and_reports_the_cutoff() {
        // 3 samples. kmer A is in all 3 (minor count 0, untestable). kmer B
        // is in 2 of 3 (minor count 1). kmer C is in 1 of 3 (minor count 1,
        // tied with B on minor count but lower prevalence so ranked below
        // it). max_kmers=1 must keep only B.
        let a = 100u64;
        let b = 200u64;
        let c = 300u64;
        let counters = vec![
            counter_from(&[(a, 1), (b, 1), (c, 1)]),
            counter_from(&[(a, 1), (b, 1)]),
            counter_from(&[(a, 1)]),
        ];

        let m = build_cohort_matrix(&counters, 1, Some(1), 8);

        assert_eq!(m.n_candidates, 3);
        assert_eq!(m.n_kmers, 1);
        assert_eq!(m.row.len(), 2, "kmer B is present in exactly 2 samples");
        assert_eq!(m.truncation_cutoff, Some(1), "the highest-ranked dropped kmer (C) has minor count 1");
    }

    #[test]
    fn max_kmers_larger_than_the_candidate_set_truncates_nothing() {
        let counters = vec![counter_from(&[(1, 1), (2, 1)]), counter_from(&[(1, 1), (2, 1)])];

        let m = build_cohort_matrix(&counters, 1, Some(100), 4);

        assert_eq!(m.n_candidates, 2);
        assert_eq!(m.n_kmers, 2);
        assert_eq!(m.truncation_cutoff, None);
    }

    #[test]
    fn ties_in_minor_sample_count_break_by_prevalence_then_depth_then_kmer_value() {
        // Two samples. kmer 10 and kmer 20 both have minor count 0
        // (present in both). kmer 10 has higher total depth (5 vs 3), so it
        // must rank above kmer 20 when max_kmers=1 keeps only one of them.
        let counters = vec![counter_from(&[(10, 3), (20, 2)]), counter_from(&[(10, 2), (20, 1)])];

        let m = build_cohort_matrix(&counters, 1, Some(1), 4);

        assert_eq!(m.n_kmers, 1);
        assert_eq!(m.col, vec![0, 0], "kmer 10 is present in both samples, so 2 nonzero entries");
        // kmer 10's total depth is 5 (3+2), kmer 20's is 3 (2+1): 10 must
        // win the tie-break and be the surviving column.
        assert_eq!(m.value.iter().sum::<u32>(), 5, "kmer 10 (higher depth) must be the kept column");
    }

    #[test]
    fn no_samples_is_an_empty_matrix_not_a_panic() {
        let m = build_cohort_matrix(&[], 1, None, 4);
        assert_eq!(m.n_samples, 0);
        assert_eq!(m.n_kmers, 0);
    }
}
