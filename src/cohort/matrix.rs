// src/cohort/matrix.rs
//! Builds a cohort-wide k-mer presence/count matrix directly from a set of
//! already-counted samples, without ever materializing a per-sample Arrow
//! `RecordBatch` (`ffi.rs::build_record_batch`'s `kmer_u64`/`kmer_sequence`/
//! `frequency` columns).
//!
//! # A note on the `gwas.py` references below
//!
//! `python/fastdna/gwas.py` was removed on 2026-09-05 with the rest of the
//! ML layer (`docs/goal-fast-kmer-counter.md`). Every reference to it in
//! this module is left standing on purpose: this code exists *because* of
//! what that module did, and its ranking rule, its `min_count`/`min_samples`
//! defaults and its truncation warning were all chosen to match it exactly.
//! Rewriting the comments to hide that would delete the reasoning and keep
//! only the conclusion. Read them as "the Python module this replaced".
//! The `fastdna matrix` CLI verb is what this backs today, and it never
//! depended on the Python side.
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
//! # What `build_cohort_matrix` itself does not do
//!
//! The core function does not run the counting pipeline itself
//! (`process_stream_parallel` and friends do that, unchanged) and it does
//! not decide per-sample ids, path validation, or FASTQ I/O -- callers
//! (`ffi.rs::cohort_presence_matrix`) hand it already-built `KmerCounter`s,
//! one per sample, in row order.
//!
//! `build_cohort_matrix_from_directory`/`build_cohort_matrix_from_files`,
//! further down in this module, are the exception: they exist for
//! `cli::MatrixArgs`/`main.rs::run_matrix` (`fastdna matrix`,
//! `docs/feature-gap-analysis.md`'s S6) and do own sample
//! discovery/counting, wiring `cohort::discovery`/`pipeline::
//! process_stream_parallel` into `build_cohort_matrix` the same way
//! `cohort::batch::count_paired_samples` already wires discovery into
//! per-sample counting for `--paired-dir`. `build_cohort_matrix` itself is
//! unchanged by their existence -- they are callers of it, not a
//! replacement for the pure, already-counted-samples-in path above.

use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::cohort::discovery::{discover_samples, SampleFiles};
use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::fastq::MultiSourceReader;
use crate::kmer::decode_kmer;
use crate::pipeline::{process_stream_parallel, PipelineConfig};

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
    /// The raw 2-bit-packed encoding of every column's k-mer, parallel to
    /// `kmer_sequences` (`kmer_u64[col]` is what `kmer_sequences[col]`
    /// decodes from). Kept alongside the already-decoded sequence rather
    /// than making a caller re-encode it from ASCII: it is exactly the
    /// compact column `export.rs`'s own default (`kmer_u64`, sequence
    /// opt-in) already prefers for on-disk storage, and every candidate's
    /// `u64` value is already in hand when `kmer_sequences` is built --
    /// keeping it costs one more `Vec` of values already computed, not a
    /// second pass.
    pub kmer_u64: Vec<u64>,
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
    let kmer_u64: Vec<u64> = candidates.iter().map(|&(kmer, _, _)| kmer).collect();

    CohortMatrix {
        n_samples,
        n_kmers,
        n_candidates,
        truncation_cutoff,
        row,
        col,
        value,
        kmer_sequences,
        kmer_u64,
    }
}

/// Counts every sample in `samples` independently, returning each sample's
/// finalized `KmerCounter` in the same order as `samples` -- the shared
/// counting loop behind both `build_cohort_matrix_from_directory` and
/// `build_cohort_matrix_from_files`.
///
/// Mirrors `cohort::batch::count_paired_samples`'s own per-sample counting
/// loop, minus that function's per-sample export step: this loop only needs
/// each sample's counter in memory long enough to hand the whole set to
/// `build_cohort_matrix`. `pipeline_config` is a template, cloned once per
/// sample -- see `count_paired_samples`'s doc comment for why (every sample
/// gets the same `k`, quality cutoff, thread count, ...). `min_count` is
/// applied to each sample's counter before it is returned, the same
/// per-sample depth filter `gwas.py::cohort_presence_matrix` forwards to
/// `fastdna.count()` for every sample: at realistic coverage a depth-1
/// k-mer is overwhelmingly likely a sequencing error, and left unfiltered
/// each one becomes a private, cohort-meaningless column that the
/// cohort-level `min_samples` filter cannot see coming (it only ever
/// observes counts, not their reliability).
fn count_samples(
    samples: &[SampleFiles],
    pipeline_config: &PipelineConfig,
    min_count: u32,
) -> Result<Vec<KmerCounter>> {
    let mut counters = Vec::with_capacity(samples.len());
    for sample in samples {
        let reader = MultiSourceReader::from_paths(sample.files.clone());
        // `sample.files` is never empty for a `SampleFiles` built by
        // `discover_samples` (it only creates an entry once it has seen a
        // file), and the `--sample` path below always supplies exactly one
        // file per entry -- but `unwrap_or_default` (an empty path used only
        // as an error-message label) is used instead of `.expect(...)` to
        // keep this function panic-free regardless of how a `SampleFiles`
        // was constructed, per this crate's "no unwrap/expect in src/"
        // convention (`Cargo.toml`'s `[lints.clippy]`).
        let source_label = sample.files.first().cloned().unwrap_or_default();

        let (mut counter, _qc, _total_reads) =
            process_stream_parallel(reader, pipeline_config.clone(), &source_label, None, None)?;
        counter.prune(min_count, None);
        counters.push(counter);
    }
    Ok(counters)
}

/// Builds a `CohortMatrix` from every FASTQ sample discovered in `dir` --
/// the library half of `fastdna matrix --input DIR` (`cli::MatrixArgs`).
///
/// Reuses `discover_samples`'s ad hoc R1/R2 pairing (the lenient variant,
/// not `cohort::batch::discover_paired_samples`'s strict unattended-pipeline
/// one): an orphaned mate still becomes a usable, single-end sample instead
/// of aborting the whole cohort, and every such sample's warning is
/// returned in `orphan_warnings` for the caller to print, exactly as ad hoc
/// cohort listing is documented to behave (`discovery.rs`'s module doc
/// comment). A directory-of-samples matrix build is closer to that use case
/// than to `--paired-dir`'s unattended-pipeline one: a human is looking at
/// the matrix this produces, and a warning they can read is more useful
/// here than a hard stop over one imperfectly-paired sample in an otherwise
/// large cohort.
///
/// `min_samples` is validated against the discovered sample count up front,
/// before any counting happens -- the same "fails fast, before any file is
/// touched" discipline `cli::CountArgs::validate` already follows, and the
/// same check `gwas.py::cohort_presence_matrix` makes in Python: a
/// `min_samples` above the cohort size can never be satisfied by any k-mer,
/// so the matrix would silently come back empty rather than erroring.
///
/// Returns `(matrix, sample_ids, orphan_warnings)`: `sample_ids[i]` names
/// row `i` of `matrix` (`discover_samples` already returns samples in
/// lexicographic, reproducible order).
pub fn build_cohort_matrix_from_directory(
    dir: &Path,
    pipeline_config: &PipelineConfig,
    min_count: u32,
    min_samples: u32,
    max_kmers: Option<usize>,
) -> Result<(CohortMatrix, Vec<String>, Vec<String>)> {
    let samples = discover_samples(dir)?;
    if min_samples as usize > samples.len() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--min-samples",
            reason: format!(
                "min_samples={min_samples} exceeds the cohort size ({} samples discovered in {}): \
                 no k-mer can be present in more samples than exist, so the matrix would come back \
                 empty. Use --min-samples <= {}.",
                samples.len(),
                dir.display(),
                samples.len()
            ),
        });
    }

    let orphan_warnings: Vec<String> = samples
        .iter()
        .filter(|s| !s.orphan_warning.is_empty())
        .map(|s| s.orphan_warning.clone())
        .collect();
    let sample_ids: Vec<String> = samples.iter().map(|s| s.sample_id.clone()).collect();

    let counters = count_samples(&samples, pipeline_config, min_count)?;
    let matrix = build_cohort_matrix(&counters, min_samples, max_kmers, pipeline_config.k);

    Ok((matrix, sample_ids, orphan_warnings))
}

/// Builds a `CohortMatrix` from an explicit, caller-named list of sample
/// files (`fastdna matrix --sample FILE...`) -- one whole file per sample,
/// with no automatic R1/R2 pairing, for cohorts whose files are not laid
/// out in one directory or do not follow a naming convention
/// `discover_samples` recognizes. `sample_ids[i]` names `sample_files[i]`
/// and becomes row `i` of the returned matrix; the two must be the same
/// length.
///
/// Same `min_samples` upfront validation as `build_cohort_matrix_from_
/// directory`, for the same reason.
pub fn build_cohort_matrix_from_files(
    sample_ids: &[String],
    sample_files: &[PathBuf],
    pipeline_config: &PipelineConfig,
    min_count: u32,
    min_samples: u32,
    max_kmers: Option<usize>,
) -> Result<CohortMatrix> {
    if sample_ids.len() != sample_files.len() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "sample_ids",
            reason: format!(
                "sample_ids has {} entries but sample_files has {}: they must be the same length, \
                 one id per file",
                sample_ids.len(),
                sample_files.len()
            ),
        });
    }
    if min_samples as usize > sample_files.len() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--min-samples",
            reason: format!(
                "min_samples={min_samples} exceeds the cohort size ({} samples given): no k-mer can \
                 be present in more samples than exist, so the matrix would come back empty. Use \
                 --min-samples <= {}.",
                sample_files.len(),
                sample_files.len()
            ),
        });
    }

    let samples: Vec<SampleFiles> = sample_ids
        .iter()
        .cloned()
        .zip(sample_files.iter().cloned())
        .map(|(sample_id, path)| SampleFiles { sample_id, files: vec![path], orphan_warning: String::new() })
        .collect();

    let counters = count_samples(&samples, pipeline_config, min_count)?;
    Ok(build_cohort_matrix(&counters, min_samples, max_kmers, pipeline_config.k))
}

#[cfg(test)]
// `unwrap`/`expect` are denied under `src/` for production code (`Cargo.
// toml`'s `[lints.clippy]`), not for test assertions -- same rationale
// `discovery.rs`/`batch.rs` already state on their own test modules.
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    /// `kmer_u64` must stay parallel to `kmer_sequences` -- same column
    /// order, and `kmer_u64[col]` must decode to exactly `kmer_sequences
    /// [col]` -- since `export_cohort_matrix_parquet` (`export.rs`) writes
    /// the former without re-deriving it from the latter.
    #[test]
    fn kmer_u64_is_parallel_to_kmer_sequences_and_decodes_to_match() {
        let counters =
            vec![counter_from(&[(9, 1), (1, 1), (5, 1)]), counter_from(&[(9, 1), (1, 1), (5, 1)])];

        let m = build_cohort_matrix(&counters, 1, None, 4);

        assert_eq!(m.kmer_u64.len(), m.kmer_sequences.len());
        assert_eq!(m.kmer_u64, vec![1, 5, 9], "ascending, same order as kmer_sequences");
        for (bits, seq) in m.kmer_u64.iter().zip(&m.kmer_sequences) {
            assert_eq!(&decode_kmer(*bits, 4), seq);
        }
    }

    // ---------------------------------------------------------------------
    // build_cohort_matrix_from_directory / build_cohort_matrix_from_files
    // ---------------------------------------------------------------------

    fn config(k: usize) -> PipelineConfig {
        PipelineConfig {
            k,
            min_quality: 0.0,
            quality_window: 4,
            batch_size: 4,
            num_threads: 2,
            progress_interval: 100_000,
            hpc: false,
            canonical: true,
        }
    }

    fn write_fixture(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).expect("failed to write fixture file");
    }

    // k=4 fixtures: "AACG" (canonical: its reverse complement "CGTT" sorts
    // higher) appears in both files below; "TTTT" (its own reverse
    // complement -- a palindrome, canonical by construction) only in one.
    const SAMPLE_A: &str = "@r1\nAACGAACG\n+\nIIIIIIII\n";
    const SAMPLE_B: &str = "@r1\nAACGTTTT\n+\nIIIIIIII\n";

    #[test]
    fn from_directory_discovers_counts_and_builds_a_matrix() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_fixture(dir.path(), "pat_a.fastq", SAMPLE_A);
        write_fixture(dir.path(), "pat_b.fastq", SAMPLE_B);

        let (matrix, sample_ids, orphan_warnings) =
            build_cohort_matrix_from_directory(dir.path(), &config(4), 1, 1, None)
                .expect("a clean two-sample directory must succeed");

        assert_eq!(sample_ids, vec!["pat_a", "pat_b"]);
        assert!(orphan_warnings.is_empty(), "no pair-suffixed files here, nothing to warn about");
        assert_eq!(matrix.n_samples, 2);
        assert!(matrix.kmer_sequences.contains(&"AACG".to_string()));
    }

    #[test]
    fn from_directory_rejects_min_samples_above_the_cohort_size_before_counting() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_fixture(dir.path(), "pat_a.fastq", SAMPLE_A);

        let result = build_cohort_matrix_from_directory(dir.path(), &config(4), 1, 2, None);
        match result {
            Err(FastDnaError::InvalidConfig { parameter, reason }) => {
                assert_eq!(parameter, "--min-samples");
                assert!(reason.contains("2") && reason.contains('1'), "{reason}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn from_directory_surfaces_an_orphan_warning_rather_than_erroring() {
        // Directory discovery (unlike `--paired-dir`) is the lenient path:
        // an orphaned R1 becomes a single-end sample plus a warning the
        // caller can print, not a hard failure.
        let dir = tempfile::tempdir().expect("temp dir");
        write_fixture(dir.path(), "pat_a_R1.fastq", SAMPLE_A);

        let (matrix, sample_ids, orphan_warnings) =
            build_cohort_matrix_from_directory(dir.path(), &config(4), 1, 1, None)
                .expect("an orphaned mate must still succeed, as single-end");

        assert_eq!(sample_ids, vec!["pat_a"]);
        assert_eq!(matrix.n_samples, 1);
        assert_eq!(orphan_warnings.len(), 1, "the orphan must be reported, not silently dropped");
    }

    #[test]
    fn from_files_builds_the_same_shape_matrix_as_from_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path_a = dir.path().join("a.fastq");
        let path_b = dir.path().join("b.fastq");
        std::fs::write(&path_a, SAMPLE_A).expect("write fixture");
        std::fs::write(&path_b, SAMPLE_B).expect("write fixture");

        let sample_ids = vec!["sample_a".to_string(), "sample_b".to_string()];
        let matrix =
            build_cohort_matrix_from_files(&sample_ids, &[path_a, path_b], &config(4), 1, 1, None)
                .expect("two explicitly named files must succeed");

        assert_eq!(matrix.n_samples, 2);
        assert!(matrix.kmer_sequences.contains(&"AACG".to_string()));
    }

    #[test]
    fn from_files_rejects_mismatched_id_and_file_counts() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path_a = dir.path().join("a.fastq");
        std::fs::write(&path_a, SAMPLE_A).expect("write fixture");

        let sample_ids = vec!["a".to_string(), "b".to_string()];
        let result = build_cohort_matrix_from_files(&sample_ids, &[path_a], &config(4), 1, 1, None);
        match result {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "sample_ids"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn from_files_rejects_min_samples_above_the_cohort_size() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path_a = dir.path().join("a.fastq");
        std::fs::write(&path_a, SAMPLE_A).expect("write fixture");

        let sample_ids = vec!["a".to_string()];
        let result = build_cohort_matrix_from_files(&sample_ids, &[path_a], &config(4), 1, 5, None);
        assert!(matches!(result, Err(FastDnaError::InvalidConfig { parameter: "--min-samples", .. })));
    }
}
