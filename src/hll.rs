// src/hll.rs
//! HyperLogLog cardinality estimation (Flajolet, Fusy, Gandouet & Meunier,
//! 2007) -- answers "how many *distinct* canonical k-mers are in this
//! file" for files too large to hold every distinct k-mer in memory to
//! count exactly, in a fixed, small amount of memory regardless of file
//! size: `2^precision` single-byte registers, so 16,384 bytes at the
//! default precision of 14, whether the file has a thousand distinct
//! k-mers or a billion.
//!
//! This is a different question from `KmerCounter`'s: it answers "how
//! many distinct k-mers" with a small, known error bound, never "what is
//! each one's count" (there is no way to recover that, or even to
//! enumerate the k-mers themselves, from the registers alone -- the whole
//! memory saving comes from deliberately not retaining that information).
//! `peek()`'s own `sample_distinct_kmers` is an *exact* count, but only
//! over a sampled prefix; this is an *approximate* count, but over the
//! whole file.

use std::io::BufRead;
use std::path::Path;

use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqReader};
use crate::kmer;
use crate::sketch::finalize_hash;

/// Standard bias-correction constant for the harmonic-mean estimator,
/// Flajolet et al.'s own asymptotic formula (valid for `m >= 128`, i.e.
/// `precision >= 7`, which `MIN_PRECISION` already enforces).
fn alpha(m: f64) -> f64 {
    0.7213 / (1.0 + 1.079 / m)
}

/// Below this precision the register count is too small for the
/// asymptotic `alpha_m` formula (and the standard error, `~1.04/sqrt(m)`,
/// would already be a coarse 13% at `m=64`) to be a meaningful estimate.
pub const MIN_PRECISION: u32 = 7;
/// Above this precision, registers stop being cheap: `2^18` bytes is
/// already 256 KB for one estimator, and accuracy past `~0.4%` standard
/// error is rarely useful for "roughly how many distinct k-mers".
pub const MAX_PRECISION: u32 = 18;

/// A HyperLogLog cardinality estimator over `u64` keys (canonical k-mers).
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    precision: u32,
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// `precision` controls both memory (`2^precision` bytes) and
    /// accuracy (standard error `~1.04 / sqrt(2^precision)`): the default
    /// of 14 is 16,384 bytes for ~0.8% standard error, the same tradeoff
    /// point most production HyperLogLog implementations default to.
    pub fn new(precision: u32) -> Result<Self> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "precision",
                reason: format!("must be between {MIN_PRECISION} and {MAX_PRECISION} inclusive, got {precision}"),
            });
        }
        let m = 1usize << precision;
        Ok(Self { precision, registers: vec![0u8; m] })
    }

    /// Records one observation of `kmer`. Idempotent in the sense that
    /// matters here: inserting the same k-mer any number of times affects
    /// the estimate the same as inserting it once, which is the entire
    /// point of a cardinality (distinct-count) estimator.
    #[inline]
    pub fn insert(&mut self, kmer: u64) {
        let h = finalize_hash(kmer);
        // The top `precision` bits select a register; the remaining
        // `64 - precision` bits (shifted up to occupy the full word, zero-
        // padded at the bottom) provide the leading-zero count. These must
        // come from disjoint bit ranges of the same hash so the two are
        // statistically independent draws, which is why the index uses the
        // top bits and the count uses everything below them, not the same
        // bits read two ways.
        //
        // The shift matters, not just the mask: `leading_zeros()` counts
        // across the full 64-bit width, so an unshifted `h >> precision`
        // would always carry `precision` extra leading zeros contributed
        // by the shift itself, not by the actual bit pattern -- silently
        // inflating every `rho`, and with it the whole estimate (caught by
        // `estimate_of_100_000_distinct_values_is_within_a_generous_error_
        // bound` reporting ~70x the true count before this was fixed to
        // shift left instead).
        let idx = (h >> (64 - self.precision)) as usize;
        let w = h << self.precision;
        // `| 1` guards against `w == 0` (every remainder bit zero), which
        // would otherwise report 64 leading zeros -- not wrong in the
        // sense of overflowing (`rho` still fits in a `u8`), but an
        // unbounded outlier for an event `2^-(64-precision)` likely,
        // capped instead at the same value one bit set at position 0
        // would give.
        let rho = ((w | 1).leading_zeros() + 1) as u8;
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// The cardinality estimate: Flajolet et al.'s harmonic-mean
    /// estimator, with small-range correction (linear counting) when the
    /// raw estimate falls in the region where it is known to be biased.
    /// Large-range correction (relevant only near `2^32` for a 32-bit
    /// hash) is not implemented: this estimator's hash is 64 bits wide,
    /// and no realistic FASTQ file approaches the cardinality where that
    /// correction would matter.
    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw = alpha(m) * m * m / sum;

        if raw <= 2.5 * m {
            let zero_registers = self.registers.iter().filter(|&&r| r == 0).count();
            if zero_registers > 0 {
                return m * (m / zero_registers as f64).ln();
            }
        }
        raw
    }

    /// Combines `other`'s observations into `self`: register-wise
    /// maximum, the standard HyperLogLog merge. Requires equal
    /// `precision` (mismatched register-array sizes have no sensible
    /// merge) -- returns `InvalidConfig` rather than panicking, since this
    /// is caller-suppliable data (two estimators built at different
    /// precisions), not a programming error.
    pub fn merge(&mut self, other: &HyperLogLog) -> Result<()> {
        if self.precision != other.precision {
            return Err(FastDnaError::InvalidConfig {
                parameter: "precision",
                reason: format!(
                    "cannot merge HyperLogLog estimators with different precision ({} and {})",
                    self.precision, other.precision
                ),
            });
        }
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if b > *a {
                *a = b;
            }
        }
        Ok(())
    }
}

/// The default register precision `estimate_cardinality` uses when the
/// caller does not ask for a different one: 16,384 bytes, ~0.8% standard
/// error -- see `HyperLogLog::new`'s own doc comment for the tradeoff.
pub const DEFAULT_PRECISION: u32 = 14;

/// Estimates the number of distinct canonical k-mers across an *entire*
/// FASTQ(.gz) file in bounded memory (`2^precision` bytes for the
/// estimator itself, regardless of file size), by streaming it once.
///
/// Unlike `peek()`'s `sample_distinct_kmers` (exact, but only over a
/// sampled prefix), this reads the whole file -- the cost this pays for
/// covering the whole file rather than a sample is time (one full
/// streaming pass, the same I/O `count()` itself pays), not memory.
pub fn estimate_cardinality<P: AsRef<Path>>(path: P, k: usize, precision: u32) -> Result<f64> {
    let path_ref = path.as_ref();
    let reader = FastqReader::from_path(path_ref)
        .map_err(|e| FastDnaError::Io { path: path_ref.to_path_buf(), source: e })?;
    estimate_cardinality_from_reader(reader, k, precision, path_ref)
}

/// The streaming logic itself, over any `BufRead` -- kept separate from
/// `estimate_cardinality` so it is testable against an in-memory buffer
/// without touching the filesystem, matching `preview.rs`'s own
/// `peek`/`peek_from_reader` split.
fn estimate_cardinality_from_reader<R: BufRead>(
    mut reader: FastqReader<R>,
    k: usize,
    precision: u32,
    source: &Path,
) -> Result<f64> {
    if k == 0 || k > 32 {
        return Err(FastDnaError::InvalidK { k });
    }

    let mut hll = HyperLogLog::new(precision)?;
    let mut record_number: u64 = 0;

    loop {
        match reader.next_record() {
            Ok(Some(record)) => {
                record_number += 1;
                for kmer_bits in kmer::extract_canonical_kmers(&record.seq, k) {
                    hll.insert(kmer_bits);
                }
            }
            Ok(None) => break,
            Err(FastqReadError::Io(source_err)) => {
                return Err(FastDnaError::Io { path: source.to_path_buf(), source: source_err })
            }
            Err(FastqReadError::Malformed(reason)) => {
                return Err(FastDnaError::MalformedFastq {
                    path: source.to_path_buf(),
                    record: record_number + 1,
                    reason,
                })
            }
        }
    }

    Ok(hll.estimate())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fastq_bytes(reads: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (i, seq) in reads.iter().enumerate() {
            buf.extend_from_slice(format!("@r{i}
{seq}
+
{}
", "I".repeat(seq.len())).as_bytes());
        }
        buf
    }

    fn reader_over(reads: &[&str]) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(fastq_bytes(reads)))
    }

    #[test]
    fn estimate_cardinality_matches_the_exact_count_for_a_small_file() {
        // Small enough that HLL's small-range linear-counting correction
        // applies, and small enough to also count exactly by hand via a
        // HashSet, so this checks the whole streaming path end to end,
        // not just the HyperLogLog struct in isolation.
        let reads = vec!["ACGTACGTACGTACGTACGT"; 30];
        let reader = reader_over(&reads);

        let estimate =
            estimate_cardinality_from_reader(reader, 5, DEFAULT_PRECISION, Path::new("<memory>")).unwrap();

        let mut exact: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for kmer_bits in kmer::extract_canonical_kmers(reads[0].as_bytes(), 5) {
            exact.insert(kmer_bits);
        }

        let error = (estimate - exact.len() as f64).abs() / exact.len() as f64;
        assert!(error < 0.5, "estimate {estimate} vs exact {}, error {error}", exact.len());
    }

    #[test]
    fn estimate_cardinality_rejects_k_out_of_range() {
        let reader = reader_over(&["ACGT"]);
        assert!(estimate_cardinality_from_reader(reader, 0, DEFAULT_PRECISION, Path::new("<memory>")).is_err());
    }

    #[test]
    fn estimate_cardinality_surfaces_malformed_fastq_with_a_record_number() {
        let mut bytes = fastq_bytes(&["ACGT"]);
        bytes.extend_from_slice(b"not-a-header-line
");
        let reader = FastqReader::new(Cursor::new(bytes));

        let err = estimate_cardinality_from_reader(reader, 4, DEFAULT_PRECISION, Path::new("bad.fastq")).unwrap_err();

        match err {
            FastDnaError::MalformedFastq { record, .. } => assert_eq!(record, 2),
            other => panic!("expected MalformedFastq, got {other:?}"),
        }
    }


    #[test]
    fn rejects_precision_outside_the_supported_range() {
        assert!(HyperLogLog::new(MIN_PRECISION - 1).is_err());
        assert!(HyperLogLog::new(MAX_PRECISION + 1).is_err());
        assert!(HyperLogLog::new(MIN_PRECISION).is_ok());
        assert!(HyperLogLog::new(MAX_PRECISION).is_ok());
    }

    #[test]
    fn empty_estimator_reports_zero() {
        let hll = HyperLogLog::new(14).unwrap();
        assert_eq!(hll.estimate(), 0.0);
    }

    #[test]
    fn estimate_of_100_000_distinct_values_is_within_a_generous_error_bound() {
        // Standard error at precision 14 is ~1.04/sqrt(16384) =~ 0.81%;
        // three standard errors (~2.4%) already covers >99.7% of runs
        // under the model, so 5% is a deliberately generous bound to
        // avoid a flaky test while still catching a badly broken
        // estimator (one off by 2x, or stuck at a wrong constant).
        let mut hll = HyperLogLog::new(14).unwrap();
        let true_count = 100_000u64;
        for i in 0..true_count {
            // Multiplied by a large odd constant so consecutive integers
            // -- which would otherwise cluster in the low bits `insert`
            // uses for register selection -- spread across the hash
            // input the way real (already 2-bit-packed, non-sequential)
            // canonical k-mers would.
            hll.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }

        let estimate = hll.estimate();
        let error = (estimate - true_count as f64).abs() / true_count as f64;
        assert!(error < 0.05, "estimate {estimate} vs true {true_count}, error {error}");
    }

    #[test]
    fn repeated_inserts_of_the_same_value_do_not_inflate_the_estimate() {
        let mut hll = HyperLogLog::new(14).unwrap();
        for _ in 0..10_000 {
            hll.insert(42);
        }
        assert!(hll.estimate() < 5.0, "10,000 inserts of one value must estimate close to 1, not 10,000");
    }

    #[test]
    fn small_range_linear_counting_kicks_in_for_low_cardinalities() {
        // Deliberately small relative to m=16384, exercising the linear-
        // counting branch (raw <= 2.5*m) rather than the plain harmonic
        // estimator -- a bug in that branch would not show up in the
        // 100,000-value test above, which stays in the harmonic regime.
        let mut hll = HyperLogLog::new(14).unwrap();
        let true_count = 200u64;
        for i in 0..true_count {
            hll.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }

        let estimate = hll.estimate();
        let error = (estimate - true_count as f64).abs() / true_count as f64;
        assert!(error < 0.25, "estimate {estimate} vs true {true_count}, error {error}");
    }

    #[test]
    fn merge_of_disjoint_ranges_approximates_the_combined_cardinality() {
        let mut a = HyperLogLog::new(14).unwrap();
        let mut b = HyperLogLog::new(14).unwrap();
        for i in 0..50_000u64 {
            a.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }
        for i in 50_000..100_000u64 {
            b.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }

        a.merge(&b).expect("equal precision must merge");

        let estimate = a.estimate();
        let error = (estimate - 100_000.0).abs() / 100_000.0;
        assert!(error < 0.05, "merged estimate {estimate} vs true 100000, error {error}");
    }

    #[test]
    fn merge_of_overlapping_ranges_does_not_double_count() {
        let mut a = HyperLogLog::new(14).unwrap();
        let mut b = HyperLogLog::new(14).unwrap();
        // b's range is entirely inside a's: the union is just a's own
        // 100,000, not 100,000 + 30,000.
        for i in 0..100_000u64 {
            a.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }
        for i in 0..30_000u64 {
            b.insert(i.wrapping_mul(0x9E3779B97F4A7C15));
        }

        a.merge(&b).expect("equal precision must merge");

        let estimate = a.estimate();
        let error = (estimate - 100_000.0).abs() / 100_000.0;
        assert!(error < 0.05, "merged estimate {estimate} vs true 100000 (not 130000), error {error}");
    }

    #[test]
    fn merge_rejects_mismatched_precision() {
        let mut a = HyperLogLog::new(10).unwrap();
        let b = HyperLogLog::new(14).unwrap();
        assert!(a.merge(&b).is_err());
    }
}
