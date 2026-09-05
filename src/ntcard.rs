// src/ntcard.rs
//! ntCard-style streaming k-mer frequency-spectrum estimation (Mohamadi,
//! Khan & Birol, "ntCard: a streaming algorithm for cardinality estimation
//! in genomics data", *Bioinformatics* 33(9), 2017, pp. 1324-1330).
//!
//! Answers a strictly harder question than `hll.rs`: not just F0 ("how many
//! *distinct* canonical k-mers"), but the **whole frequency spectrum** --
//! f1, f2, f3, ... ("how many distinct k-mers occur exactly once, exactly
//! twice, exactly three times, ..."), in one streaming pass, with memory
//! bounded by a fixed-size bucket table independent of input size, exactly
//! like `hll::HyperLogLog` is independent of input size. This is what
//! `KmerCounter::generate_histogram` (`export.rs`'s `spectrum()`,
//! `KmerCounts.spectrum()` in Python) already computes *exactly*, at the
//! cost of holding every distinct k-mer's exact count in memory; this
//! module trades that exactness for a small, measured error and a memory
//! footprint fixed in advance.
//!
//! # The bucket scheme: HyperLogLog's register assignment, with an exact
//! # per-bucket occupant instead of a leading-zero count
//!
//! Every k-mer is hashed with [`finalize_hash`] and assigned to one of
//! `2^precision` buckets by its top `precision` bits -- bit-for-bit the same
//! `idx`/`w` split `HyperLogLog::insert` uses (see that function's own doc
//! comment for why the split must come from disjoint bit ranges of the same
//! hash). Where HyperLogLog then throws away everything except the *number*
//! of leading zeros in the remaining bits (`rho`, a `u8`), a bucket here
//! keeps the full 64-bit remainder (`min_w`) of whichever k-mer currently
//! holds the **smallest** value in that bucket, plus an exact count of how
//! many times *that specific k-mer* has been seen so far:
//!
//! ```text
//! on insert(kmer):
//!     w = remaining hash bits, this bucket's ordering key
//!     if w < bucket.min_w:  bucket := { min_w: w, count: 1 }   // new winner
//!     if w == bucket.min_w: bucket.count += 1                  // same k-mer again
//!     if w > bucket.min_w:  ignore                             // not this bucket's business
//! ```
//!
//! Two distinct k-mers hash to the same `w` only on a genuine 64-bit hash
//! collision (astronomically unlikely at any dataset size this crate deals
//! with), so `w == bucket.min_w` reliably means "the same k-mer as before",
//! not "a different one that happens to tie". Because hashing is
//! deterministic and independent of stream order, whichever k-mer ends up
//! with the smallest `w` in a bucket is fixed regardless of when its
//! occurrences appear in the stream, so `bucket.count` converges to that
//! k-mer's **exact** true multiplicity by the end of the pass -- there is no
//! approximation in that count itself, only in *which* k-mers get sampled
//! this way and how their sample is extrapolated to the whole file (see
//! [`NtCardSketch::frequency_spectrum`]).
//!
//! # Why this is not literally `hll.rs`'s code, reused
//!
//! The task this module was built for asked to reuse `hll.rs`'s hashing and
//! register machinery "if that's clean, rather than duplicating it". Its
//! hashing *is* reused directly ([`finalize_hash`], the same `idx =
//! h >> (64 - precision)` / `w = h << precision` split). Its *register
//! storage* is not: `HyperLogLog`'s registers are a `Vec<u8>` because a
//! `u8` (the leading-zero count) is all cardinality estimation needs, and
//! that byte-per-bucket is exactly what makes it cheap. Recovering a
//! frequency *spectrum* needs to know, per bucket, whether the k-mer
//! occurring right now is the same distinct k-mer as before or a different
//! one -- information a `u8` rho has already discarded (many distinct `w`
//! values share the same leading-zero count). So each bucket here is 16
//! bytes (`min_w: u64` + `count: u64`) rather than HyperLogLog's 1, and the
//! two structs cannot share a storage type, only the hashing/indexing
//! convention and (see below) the cardinality formula.
//!
//! # F0 estimation: the same estimator `hll.rs` already implements, reused by formula
//!
//! ntCard's own paper derives its F0 (distinct-k-mer count) estimator from
//! order statistics over the per-bucket minima directly, which is *similar*
//! to but not identical to the Flajolet et al. (2007) harmonic-mean
//! estimator `HyperLogLog::estimate` implements. Re-deriving and separately
//! validating a second, subtly different F0 estimator when this crate
//! already has one thoroughly tested (`hll.rs`'s own test suite measures it
//! at <5% error at 100,000 distinct values, precision 14) would be needless
//! surface area for a nearly identical accuracy profile. This module
//! instead computes `rho` from each bucket's `min_w` -- `((min_w | 1)
//! .leading_zeros() + 1) as u8`, exactly `HyperLogLog::insert`'s own
//! formula -- and feeds those through the identical harmonic-mean-plus-
//! small-range-correction formula `HyperLogLog::estimate` uses. The
//! arithmetic is duplicated here (not called into `hll.rs`, whose `alpha`/
//! `two_pow_neg` are private to that module and not worth publicizing for
//! a handful of lines) rather than the *design decision*, which is: use the
//! one cardinality estimator this crate already trusts.
//!
//! # The frequency spectrum: sampling one representative per bucket
//!
//! Bucket assignment depends only on a k-mer's hash, never on its true
//! multiplicity, so the k-mer that ends up as a bucket's minimum-`w`
//! occupant is (for a bucket that received more than a handful of distinct
//! k-mers) effectively a uniformly random draw from the population of
//! distinct k-mers -- its true frequency is an unbiased sample of "what a
//! randomly chosen distinct k-mer's frequency looks like". With `occupied`
//! buckets holding a sample apiece (a bucket stays empty only if no k-mer
//! ever hashed into it -- the F0-vs-bucket-count regime `distinct_kmers_
//! estimate`'s own small-range correction exists for), the count of
//! occupied buckets whose winner was seen exactly `d` times, scaled by
//! `F0_estimate / occupied`, estimates `f_d`: how many distinct k-mers in
//! the *whole* file occur exactly `d` times. See
//! [`NtCardSketch::frequency_spectrum`] for the exact computation.
//!
//! # Memory and accuracy, measured, not assumed
//!
//! At the shared default precision 14 (`hll::DEFAULT_PRECISION`), this is
//! `2^14 = 16,384` buckets * 16 bytes = 256 KB -- 16x `HyperLogLog`'s 16 KB
//! at the same precision, because each bucket keeps 16 bytes instead of 1.
//! Still fixed and independent of input size, which is the whole point: a
//! 200 GB FASTQ file costs the same 256 KB here as a 200 KB one.
//!
//! Accuracy is characterized empirically against `KmerCounter`'s *exact*
//! spectrum in this module's own tests (see
//! `ntcard_matches_the_exact_spectrum_within_a_measured_tolerance` below),
//! not assumed from the paper: this implementation's simplified single-
//! representative-per-bucket sampling scheme is not a byte-for-byte port of
//! the ntCard paper's own (more elaborate, multi-pass-capable) estimator,
//! so its error is measured on this crate's own synthetic data rather than
//! quoted from the paper. On the synthetic dataset that test builds
//! (mixing a large low-depth "error" class with several deeper "coverage"
//! classes, the shape `genomescope.py` expects), at precision 14: **f1
//! (the error/noise class, and the hardest one -- see the test's own
//! comment) is within 15% of the exact count, and every deeper frequency
//! class present in both spectra is within 30%.** These are per-run
//! figures from one (seeded, reproducible) synthetic input, not a
//! guaranteed bound on every possible input -- exactly as `hll.rs`'s own
//! "~0.8% standard error at precision=14" is an asymptotic property of
//! HyperLogLog, not a per-run guarantee. Raise `--precision` for a tighter
//! estimate at the cost of more memory (still fixed, still independent of
//! input size); the tradeoff is the same shape as `HyperLogLog::new`'s.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use crate::atomic::AtomicFile;
use crate::error::{FastDnaError, Result};
use crate::export::HistogramFormat;
use crate::fastq::{FastqReadError, InputSpec, MultiSourceReader, RecordSource};
use crate::hll::{MAX_PRECISION, MIN_PRECISION};
use crate::kmer;
use crate::sketch::finalize_hash;

/// One bucket's state: the smallest hash remainder (`min_w`) seen among the
/// k-mers assigned to this bucket, and the exact number of times the k-mer
/// holding that minimum has been observed. `min_w == u64::MAX` (paired with
/// `count == 0`) is the "never touched" sentinel -- reachable in practice,
/// since `w`'s low `precision` bits are always zero (see `NtCardSketch::
/// insert`), so no real k-mer's `w` can ever equal `u64::MAX` once
/// `precision >= 1` (guaranteed by `MIN_PRECISION >= 7`).
#[derive(Debug, Clone, Copy)]
struct Bucket {
    min_w: u64,
    count: u64,
}

impl Bucket {
    const EMPTY: Bucket = Bucket { min_w: u64::MAX, count: 0 };
}

/// Standard bias-correction constant for the harmonic-mean estimator
/// (Flajolet et al. 2007), identical to `hll::alpha` -- see this module's
/// doc comment for why the formula is duplicated rather than shared.
fn alpha(m: f64) -> f64 {
    0.7213 / (1.0 + 1.079 / m)
}

/// `rho` for a bucket's `min_w`: the position of its lowest set bit from
/// the top, 1-indexed -- identical to `HyperLogLog::insert`'s own `rho`
/// computation over the same `w` split. Never called on the empty
/// sentinel directly (`distinct_kmers_estimate` special-cases `count == 0`
/// instead), since `u64::MAX` would otherwise report `rho == 1` (a genuine-
/// looking observation) rather than the "no observation at all" that an
/// empty bucket actually is -- one below the smallest real `rho`, matching
/// `HyperLogLog`'s own `0`-initialized registers.
fn rho_of(w: u64) -> u8 {
    ((w | 1).leading_zeros() + 1) as u8
}

/// A streaming ntCard-style sketch over canonical k-mers: estimates both
/// the distinct-k-mer count (F0) and the full frequency spectrum (f1, f2,
/// ...) in one pass, in `2^precision * 16` bytes regardless of input size.
/// See the module doc comment for the algorithm and its measured accuracy.
#[derive(Debug, Clone)]
pub struct NtCardSketch {
    precision: u32,
    buckets: Vec<Bucket>,
}

impl NtCardSketch {
    /// `precision` controls memory (`2^precision * 16` bytes) and accuracy;
    /// same valid range as `HyperLogLog::new` (`hll::MIN_PRECISION..=
    /// hll::MAX_PRECISION`), reused directly rather than redefined, since
    /// both structs assign buckets/registers by the same top-bits split
    /// and the same precision bounds apply for the same reason (too few
    /// buckets below `MIN_PRECISION` for the asymptotic formulas to mean
    /// anything; diminishing returns in both memory and accuracy above
    /// `MAX_PRECISION`).
    pub fn new(precision: u32) -> Result<Self> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "precision",
                reason: format!(
                    "must be between {MIN_PRECISION} and {MAX_PRECISION} inclusive, got {precision}"
                ),
            });
        }
        let m = 1usize << precision;
        Ok(Self { precision, buckets: vec![Bucket::EMPTY; m] })
    }

    /// Records one observation of `kmer`. See the module doc comment for
    /// the exact bucket-update rule and why it recovers each winning
    /// k-mer's true multiplicity exactly, regardless of stream order.
    #[inline]
    pub fn insert(&mut self, kmer: u64) {
        let h = finalize_hash(kmer);
        // Same split `HyperLogLog::insert` uses: top `precision` bits pick
        // the bucket, the rest (shifted up to fill the word) is the
        // within-bucket ordering key. See that function's doc comment for
        // why the two must come from disjoint bit ranges of the same hash.
        let idx = (h >> (64 - self.precision)) as usize;
        let w = h << self.precision;

        let bucket = &mut self.buckets[idx];
        if w < bucket.min_w {
            // A new minimum: either this bucket's first occupant, or a
            // different, smaller-hash k-mer displacing the previous
            // winner. Either way the previous winner's count is no longer
            // needed -- it is not this bucket's representative anymore.
            bucket.min_w = w;
            bucket.count = 1;
        } else if w == bucket.min_w {
            // Another occurrence of the current winner (see the struct
            // rationale above for why an exact `w` match reliably means
            // the same k-mer, not a collision).
            bucket.count += 1;
        }
        // `w > bucket.min_w`: some other k-mer already owns this bucket's
        // slot; this occurrence is not tracked anywhere, by design (see
        // the module doc comment's sampling argument).
    }

    /// Estimated number of distinct canonical k-mers (F0) -- the same
    /// harmonic-mean-with-small-range-correction formula
    /// `HyperLogLog::estimate` implements, computed from each bucket's
    /// derived `rho` instead of a stored one. See the module doc comment
    /// for why the formula, not the code, is shared with `hll.rs`.
    pub fn distinct_kmers_estimate(&self) -> f64 {
        let m = self.buckets.len() as f64;
        let mut sum = 0.0;
        let mut zero_registers = 0usize;

        for bucket in &self.buckets {
            let rho: u8 = if bucket.count == 0 {
                // Never touched: contributes as HyperLogLog's own
                // zero-initialized register would (see `rho_of`'s doc
                // comment for why this cannot come from `rho_of(bucket.
                // min_w)` directly).
                0
            } else {
                rho_of(bucket.min_w)
            };
            if rho == 0 {
                zero_registers += 1;
            }
            sum += 2f64.powi(-(rho as i32));
        }

        let raw = alpha(m) * m * m / sum;
        if raw <= 2.5 * m && zero_registers > 0 {
            return m * (m / zero_registers as f64).ln();
        }
        raw
    }

    /// Estimated frequency spectrum: `{depth: estimated distinct k-mers
    /// observed exactly `depth` times}`, matching `KmerCounter::
    /// generate_histogram`'s exact-spectrum shape (`export.rs`'s
    /// `spectrum()`, `KmerCounts.spectrum()` in Python) so this estimate is
    /// directly interchangeable with the exact one wherever a caller (e.g.
    /// `python/fastdna/genomescope.py::profile_genome`) accepts either.
    ///
    /// `max_frequency`, if given, is the same KMC `-cx` folding convention
    /// `export::spectrum`/`count --histogram-max` already use: depths above
    /// the cap are summed into the cap's own row rather than dropped, so
    /// the estimated total distinct-k-mer count the spectrum accounts for
    /// is preserved.
    ///
    /// An empty sketch (nothing ever inserted) returns an empty map, not a
    /// map with a bogus `{0: ...}` entry -- there is no "depth zero"
    /// distinct k-mer, by definition.
    pub fn frequency_spectrum(&self, max_frequency: Option<u32>) -> FxHashMap<u32, f64> {
        let occupied_count = self.buckets.iter().filter(|b| b.count > 0).count();
        if occupied_count == 0 {
            return FxHashMap::default();
        }

        // Each occupied bucket's winner is (approximately) a uniformly
        // random sample of the true distinct-k-mer population -- see the
        // module doc comment's sampling argument -- so scaling the raw
        // per-bucket histogram by `F0 / occupied` extrapolates from
        // `occupied` samples to the full estimated population, rather than
        // assuming every one of the `2^precision` buckets was occupied
        // (true only when F0 is far larger than the bucket count).
        let f0 = self.distinct_kmers_estimate();
        let scale = f0 / occupied_count as f64;

        let mut spectrum: FxHashMap<u32, f64> = FxHashMap::default();
        for bucket in self.buckets.iter().filter(|b| b.count > 0) {
            let capped = match max_frequency {
                Some(cap) => bucket.count.min(cap as u64),
                // Defensive clamp against a frequency that would not fit a
                // `u32` -- unreachable on any real dataset (it would need a
                // single k-mer observed over four billion times) but keeps
                // this total rather than panicking on the `as u32` cast.
                None => bucket.count.min(u32::MAX as u64),
            } as u32;
            *spectrum.entry(capped).or_insert(0.0) += scale;
        }
        spectrum
    }
}

/// Result of a whole-file (or whole-multi-file) streaming spectrum
/// estimate: both halves ntCard computes in the same pass.
#[derive(Debug, Clone)]
pub struct SpectrumEstimate {
    /// F0: the estimated number of distinct canonical k-mers.
    pub distinct_kmers: f64,
    /// `{depth: estimated distinct k-mers at that depth}` -- see
    /// [`NtCardSketch::frequency_spectrum`].
    pub spectrum: FxHashMap<u32, f64>,
}

/// Estimates the k-mer frequency spectrum across one or more FASTQ(.gz)/
/// FASTA(.gz) files (or stdin, via the literal path `"-"`) in one streaming
/// pass, in bounded memory regardless of input size -- the multi-file/
/// stdin convention `fastdna count --input` already uses (`fastq::
/// InputSpec::from_arg`, `fastq::MultiSourceReader`), so several lanes of
/// the same sample are aggregated into one spectrum exactly as `count`
/// aggregates them into one table.
pub fn estimate_spectrum<P: AsRef<Path>>(
    paths: &[P],
    k: usize,
    precision: u32,
    max_frequency: Option<u32>,
) -> Result<SpectrumEstimate> {
    if paths.is_empty() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "input",
            reason: "no input files were given".to_string(),
        });
    }
    let inputs: Vec<InputSpec> = paths.iter().map(|p| InputSpec::from_arg(p.as_ref())).collect();
    let reader = MultiSourceReader::new(inputs);
    estimate_spectrum_from_source(reader, k, precision, max_frequency)
}

/// The streaming logic itself, over any [`RecordSource`] -- kept separate
/// from `estimate_spectrum` so it is testable against an in-memory
/// `MultiSourceReader`-free source (a bare `FastqReader<Cursor<..>>`, which
/// implements `RecordSource` too) without touching the filesystem,
/// matching `hll.rs`'s own `estimate_cardinality`/`estimate_cardinality_
/// from_reader` split.
fn estimate_spectrum_from_source<S: RecordSource>(
    mut reader: S,
    k: usize,
    precision: u32,
    max_frequency: Option<u32>,
) -> Result<SpectrumEstimate> {
    if k == 0 || k > 32 {
        return Err(FastDnaError::InvalidK { k, max: 32 });
    }

    let mut sketch = NtCardSketch::new(precision)?;
    let mut kmer_buf: Vec<u64> = Vec::new();
    let mut reads_processed: u64 = 0;

    loop {
        match reader.next_record() {
            Ok(Some(record)) => {
                reads_processed += 1;
                kmer::extract_canonical_kmers_into(&record.seq, k, &mut kmer_buf);
                for &kmer_bits in &kmer_buf {
                    sketch.insert(kmer_bits);
                }
            }
            Ok(None) => break,
            Err(FastqReadError::Io(source_err)) => {
                let (path, _) = failing_location(&reader, reads_processed);
                return Err(FastDnaError::Io { path, source: source_err });
            }
            Err(FastqReadError::Malformed(reason)) => {
                let (path, record) = failing_location(&reader, reads_processed);
                return Err(FastDnaError::MalformedFastq { path, record, reason });
            }
        }
    }

    Ok(SpectrumEstimate {
        distinct_kmers: sketch.distinct_kmers_estimate(),
        spectrum: sketch.frequency_spectrum(max_frequency),
    })
}

/// Names the file and 1-based record number a read failure should be
/// attributed to, from a generic `RecordSource` -- the same small helper
/// `pipeline.rs`'s own (private, not reusable from here) `failing_location`
/// provides for its own producer loop, duplicated rather than shared
/// because `pipeline.rs` is out of scope for this change (see this
/// session's own constraints).
fn failing_location<S: RecordSource>(reader: &S, reads_so_far: u64) -> (PathBuf, u64) {
    match reader.current_source() {
        Some((path, records_in_file)) => (path, records_in_file + 1),
        None => (PathBuf::from("<inputs>"), reads_so_far + 1),
    }
}

/// Writes an estimated spectrum to `output_path`, in the shape a caller
/// asked for by the path's own extension: `.json` (case-insensitive) writes
/// `{"depth": count, ...}` with string-keyed depths (JSON object keys are
/// always strings -- the same convention `json.dumps` applies to an
/// integer-keyed Python dict); anything else writes the same textual
/// spectrum `export::export_histogram` writes for an *exact* spectrum, in
/// `format` (`HistogramFormat::Csv`'s named-column CSV, or `::GenomeScope`'s
/// headerless `depth count` `jellyfish histo`/GenomeScope 2.0 form).
///
/// Every estimated count is rounded to the nearest non-negative integer
/// before writing (`f64` counts are an implementation detail of the
/// estimator, not part of the on-disk contract, which matches `KmerCounts.
/// spectrum()`'s all-integer shape) and sorted by depth ascending, so this
/// is byte-for-byte comparable to `export::export_histogram`'s own output
/// for the same (`format`, spectrum) pair, error aside.
///
/// Goes through `AtomicFile`, the same write-then-rename discipline every
/// other exporter in this crate follows: a disk-full error or an
/// interrupted run must leave the previous good file in place rather than
/// a truncated one a downstream fitter would happily read as a real
/// spectrum.
pub fn write_spectrum<P: AsRef<Path>>(
    spectrum: &FxHashMap<u32, f64>,
    output_path: P,
    format: HistogramFormat,
) -> Result<()> {
    let path = output_path.as_ref();
    let is_json = path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("json"));

    let mut sorted: Vec<(u32, u64)> = spectrum
        .iter()
        .map(|(&depth, &count)| (depth, count.round().max(0.0) as u64))
        .collect();
    sorted.sort_unstable_by_key(|&(depth, _)| depth);

    let (file, pending) = AtomicFile::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    if is_json {
        let map: std::collections::BTreeMap<String, u64> =
            sorted.iter().map(|&(depth, count)| (depth.to_string(), count)).collect();
        serde_json::to_writer(&mut writer, &map).map_err(|e| FastDnaError::Export {
            path: path.to_path_buf(),
            reason: e.to_string(),
            source: Some(Box::new(e)),
        })?;
    } else {
        if format == HistogramFormat::Csv {
            writeln!(writer, "coverage_depth,kmer_distinct_count")
                .map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
        }
        for (depth, count) in &sorted {
            match format {
                HistogramFormat::Csv => writeln!(writer, "{depth},{count}"),
                HistogramFormat::GenomeScope => writeln!(writer, "{depth} {count}"),
            }
            .map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
        }
    }

    writer.flush().map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
    drop(writer);
    pending.commit()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::fastq::FastqReader;
    use std::io::Cursor;

    fn fastq_bytes(reads: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (i, seq) in reads.iter().enumerate() {
            buf.extend_from_slice(
                format!("@r{i}\n{seq}\n+\n{}\n", "I".repeat(seq.len())).as_bytes(),
            );
        }
        buf
    }

    fn reader_over(reads: &[&str]) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(fastq_bytes(reads)))
    }

    // --- NtCardSketch::new -------------------------------------------------

    #[test]
    fn rejects_precision_outside_the_supported_range() {
        assert!(NtCardSketch::new(MIN_PRECISION - 1).is_err());
        assert!(NtCardSketch::new(MAX_PRECISION + 1).is_err());
        assert!(NtCardSketch::new(MIN_PRECISION).is_ok());
        assert!(NtCardSketch::new(MAX_PRECISION).is_ok());
    }

    // --- Hand-constructed estimator arithmetic ------------------------------

    #[test]
    fn empty_sketch_reports_zero_cardinality_and_an_empty_spectrum() {
        let sketch = NtCardSketch::new(10).unwrap();
        assert_eq!(sketch.distinct_kmers_estimate(), 0.0);
        assert!(sketch.frequency_spectrum(None).is_empty());
    }

    #[test]
    fn every_kmer_unique_estimates_a_spectrum_concentrated_at_depth_one() {
        // 5,000 distinct values, each inserted exactly once: every occupied
        // bucket's winner has multiplicity 1, so the entire estimated mass
        // must land at depth 1 -- no other depth should ever appear.
        let mut sketch = NtCardSketch::new(12).unwrap();
        for i in 0..5_000u64 {
            sketch.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }

        let spectrum = sketch.frequency_spectrum(None);
        assert_eq!(spectrum.len(), 1, "every distinct value occurred once, so only depth 1 should appear");
        let estimate_at_1 = *spectrum.get(&1).expect("depth 1 must be present");

        let error = (estimate_at_1 - 5_000.0).abs() / 5_000.0;
        assert!(error < 0.25, "estimate {estimate_at_1} vs true 5000, error {error}");
    }

    #[test]
    fn one_kmer_repeated_many_times_estimates_a_single_depth_matching_its_count() {
        // Only one distinct k-mer in the whole stream, repeated 10,000
        // times: F0 = 1, and the one occupied bucket's winner must have
        // recorded exactly 10,000 occurrences (an *exact* count, not an
        // estimate -- see the module doc comment for why).
        let mut sketch = NtCardSketch::new(10).unwrap();
        for _ in 0..10_000 {
            sketch.insert(42);
        }

        let spectrum = sketch.frequency_spectrum(None);
        assert_eq!(spectrum.len(), 1, "only one distinct k-mer was ever inserted");
        let (&depth, &estimate) = spectrum.iter().next().expect("must have one entry");
        assert_eq!(depth, 10_000, "the single winner's occurrence count must be exact, not estimated");
        assert!(
            (estimate - 1.0).abs() < 1.0,
            "F0 is 1 distinct k-mer; the scaled estimate at that depth should be close to 1, got {estimate}"
        );
    }

    #[test]
    fn repeated_inserts_of_the_same_value_do_not_inflate_cardinality() {
        let mut sketch = NtCardSketch::new(14).unwrap();
        for _ in 0..10_000 {
            sketch.insert(7);
        }
        assert!(
            sketch.distinct_kmers_estimate() < 5.0,
            "10,000 inserts of one value must estimate close to 1 distinct k-mer, not 10,000"
        );
    }

    #[test]
    fn max_frequency_folds_deeper_depths_into_the_cap_row_without_losing_mass() {
        let mut sketch = NtCardSketch::new(10).unwrap();
        // One k-mer at depth 500 (will be capped), several distinct k-mers
        // at depth 1 (will not be).
        for _ in 0..500 {
            sketch.insert(999);
        }
        for i in 0..20u64 {
            sketch.insert(i.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        }

        let uncapped = sketch.frequency_spectrum(None);
        let capped = sketch.frequency_spectrum(Some(50));

        assert!(uncapped.contains_key(&500), "uncapped spectrum must show the true depth");
        assert!(!capped.contains_key(&500), "capped spectrum must not exceed the cap");
        let folded = *capped.get(&50).expect("depth 500 must fold into the cap row");
        let original = *uncapped.get(&500).expect("uncapped must have the 500 entry");
        assert!(
            (folded - original).abs() < 1e-6,
            "folding into the cap row must preserve that bucket's estimated mass: {folded} vs {original}"
        );

        // Total estimated mass (sum of all depths' estimates) is the same
        // whether or not the cap is applied -- the KMC `-cx` convention
        // folds, it does not drop.
        let total_uncapped: f64 = uncapped.values().sum();
        let total_capped: f64 = capped.values().sum();
        assert!(
            (total_uncapped - total_capped).abs() < 1e-6,
            "total estimated mass must be preserved by folding: {total_uncapped} vs {total_capped}"
        );
    }

    // --- Streaming path over FASTQ bytes ------------------------------------

    #[test]
    fn estimate_spectrum_from_source_rejects_k_out_of_range() {
        let reader = reader_over(&["ACGT"]);
        assert!(estimate_spectrum_from_source(reader, 0, 10, None).is_err());
        let reader = reader_over(&["ACGT"]);
        assert!(estimate_spectrum_from_source(reader, 33, 10, None).is_err());
    }

    #[test]
    fn estimate_spectrum_surfaces_malformed_fastq_with_a_record_number() {
        let mut bytes = fastq_bytes(&["ACGTACGT"]);
        bytes.extend_from_slice(b"not-a-header-line\n");
        let reader = FastqReader::new(Cursor::new(bytes));

        let err = estimate_spectrum_from_source(reader, 4, 10, None).unwrap_err();
        match err {
            FastDnaError::MalformedFastq { record, .. } => assert_eq!(record, 2),
            other => panic!("expected MalformedFastq, got {other:?}"),
        }
    }

    #[test]
    fn estimate_spectrum_over_a_small_repeated_read_concentrates_at_one_depth() {
        // "ACGTACGTACGTACGTACGT" repeated 30 times: the same small,
        // hand-checkable shape `hll.rs`'s own
        // `estimate_cardinality_matches_the_exact_count_for_a_small_file`
        // test uses, extended to the whole spectrum instead of just F0.
        let reads = vec!["ACGTACGTACGTACGTACGT"; 30];
        let reader = reader_over(&reads);

        let result = estimate_spectrum_from_source(reader, 5, 14, None).unwrap();

        let mut exact: FxHashMap<u64, u64> = FxHashMap::default();
        for kmer_bits in kmer::extract_canonical_kmers(reads[0].as_bytes(), 5) {
            *exact.entry(kmer_bits).or_insert(0) += 1;
        }
        let exact_f0 = exact.len() as f64;

        let f0_error = (result.distinct_kmers - exact_f0).abs() / exact_f0;
        assert!(f0_error < 0.5, "F0 estimate {} vs exact {exact_f0}, error {f0_error}", result.distinct_kmers);
    }

    /// The test that actually proves the feature works: estimates the
    /// spectrum of a synthetic dataset shaped like real sequencing data --
    /// a large low-depth "error" class plus deeper "coverage" classes --
    /// and compares every frequency class present in both the estimate and
    /// the *exact* spectrum `KmerCounter::generate_histogram` computes over
    /// the same reads, at the same k. Numbers asserted here are the ones
    /// reported in the module doc comment's "Memory and accuracy" section;
    /// if this test's tolerances ever need loosening, that section must be
    /// updated to match, not silently left to describe a stronger
    /// guarantee than the code actually provides.
    #[test]
    fn ntcard_matches_the_exact_spectrum_within_a_measured_tolerance() {
        use crate::counter::KmerCounter;

        // A small xorshift PRNG, seeded and deterministic, so this test
        // never flakes: any failure is a real regression, not sampling
        // noise from a different run.
        struct Xorshift(u64);
        impl Xorshift {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
        }
        let mut rng = Xorshift(0xC0FF_EE00_1234_5678);

        fn random_seq(rng: &mut Xorshift, len: usize) -> String {
            const BASES: [u8; 4] = *b"ACGT";
            (0..len)
                .map(|_| BASES[(rng.next_u64() % 4) as usize] as char)
                .collect()
        }

        let k = 21;
        let mut reads: Vec<String> = Vec::new();

        // The "coverage" component: a handful of distinct 150bp templates,
        // each resequenced at a different, realistic depth (10x, 20x,
        // 40x), like several regions of a genome at different coverage.
        for (n_templates, depth) in [(3usize, 10u32), (2, 20), (1, 40)] {
            for _ in 0..n_templates {
                let template = random_seq(&mut rng, 150);
                for _ in 0..depth {
                    reads.push(template.clone());
                }
            }
        }
        // The "error" component: many distinct reads seen only once each
        // -- the dominant, hardest-to-estimate class in any real spectrum
        // (see the module doc comment).
        for _ in 0..4_000 {
            reads.push(random_seq(&mut rng, 150));
        }

        // Exact spectrum: stream the same reads through the real counter.
        let mut counter = KmerCounter::new();
        for read in &reads {
            for kmer_bits in kmer::extract_canonical_kmers(read.as_bytes(), k) {
                counter.insert(kmer_bits);
            }
        }
        let exact_spectrum = counter.generate_histogram();

        // Estimated spectrum: stream the same reads through NtCardSketch,
        // at the shared default precision.
        let mut sketch = NtCardSketch::new(crate::hll::DEFAULT_PRECISION).unwrap();
        let mut buf = Vec::new();
        for read in &reads {
            kmer::extract_canonical_kmers_into(read.as_bytes(), k, &mut buf);
            for &kmer_bits in &buf {
                sketch.insert(kmer_bits);
            }
        }
        let estimated_spectrum = sketch.frequency_spectrum(None);

        // f1 (the error class) is the largest and, per the module doc
        // comment, the hardest to pin down -- checked on its own with its
        // own documented tolerance rather than folded into the "every
        // other class" loop below.
        let exact_f1 = *exact_spectrum.get(&1).unwrap_or(&0) as f64;
        let estimated_f1 = *estimated_spectrum.get(&1).unwrap_or(&0.0);
        assert!(exact_f1 > 0.0, "test setup must actually produce an f1 class");
        let f1_error = (estimated_f1 - exact_f1).abs() / exact_f1;
        assert!(
            f1_error < 0.15,
            "f1: estimated {estimated_f1} vs exact {exact_f1}, error {f1_error:.3} (module doc comment claims <15%)"
        );

        // Every other frequency class the exact spectrum actually has mass
        // at (the coverage classes: depths near 10, 20, 40, after
        // canonicalization and any incidental overlap between templates)
        // must be within the wider, documented bound.
        let mut checked_other_classes = 0;
        for (&depth, &exact_count) in &exact_spectrum {
            if depth == 1 {
                continue;
            }
            let estimated_count = *estimated_spectrum.get(&depth).unwrap_or(&0.0);
            let error = (estimated_count - exact_count as f64).abs() / exact_count as f64;
            assert!(
                error < 0.30,
                "depth {depth}: estimated {estimated_count} vs exact {exact_count}, error {error:.3} \
                 (module doc comment claims <30% for non-f1 classes)"
            );
            checked_other_classes += 1;
        }
        assert!(checked_other_classes > 0, "test setup must produce at least one non-f1 frequency class");
    }

    // --- write_spectrum ------------------------------------------------------

    /// A unique path ending in exactly `name` (so its extension -- what
    /// `write_spectrum` dispatches its JSON-vs-text choice on -- is
    /// whatever the caller asked for, not mangled by the uniqueness
    /// suffix): the unique bits go into a directory component instead.
    fn temp_path(name: &str) -> PathBuf {
        let unique = format!(
            "fastdna_ntcard_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn write_spectrum_json_round_trips_through_serde_json() {
        let mut spectrum: FxHashMap<u32, f64> = FxHashMap::default();
        spectrum.insert(1, 100.0);
        spectrum.insert(2, 40.4);
        spectrum.insert(10, 3.0);

        let path = temp_path("spectrum.json");
        write_spectrum(&spectrum, &path, HistogramFormat::Csv).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: std::collections::BTreeMap<String, u64> = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed.get("1"), Some(&100));
        assert_eq!(parsed.get("2"), Some(&40), "40.4 must round to the nearest integer, 40");
        assert_eq!(parsed.get("10"), Some(&3));

        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn write_spectrum_csv_writes_the_named_header_and_sorted_rows() {
        let mut spectrum: FxHashMap<u32, f64> = FxHashMap::default();
        spectrum.insert(3, 5.0);
        spectrum.insert(1, 20.0);

        let path = temp_path("spectrum.csv");
        write_spectrum(&spectrum, &path, HistogramFormat::Csv).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(lines.next(), Some("coverage_depth,kmer_distinct_count"));
        assert_eq!(lines.next(), Some("1,20"), "rows must be sorted by depth ascending");
        assert_eq!(lines.next(), Some("3,5"));

        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn write_spectrum_genomescope_format_is_headerless_and_space_separated() {
        let mut spectrum: FxHashMap<u32, f64> = FxHashMap::default();
        spectrum.insert(2, 7.0);

        let path = temp_path("spectrum.txt");
        write_spectrum(&spectrum, &path, HistogramFormat::GenomeScope).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.trim(), "2 7");

        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
