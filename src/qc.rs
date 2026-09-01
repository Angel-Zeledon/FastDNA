// src/qc.rs

use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::atomic::AtomicFile;
use crate::error::{FastDnaError, Result};
use crate::fastq::FastqRecord;

/// Quality Control (QC) summary report for sequencing health.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QcSummary {
    pub total_reads: u64,
    pub total_bases: u64,
    pub q20_bases: u64,
    pub q30_bases: u64,
    pub gc_bases: u64,
    pub gc_content_pct: f64,
    pub q20_pct: f64,
    pub q30_pct: f64,
}

/// The smallest quality byte whose Phred+33 score is at least 20, and the
/// same for 30.
///
/// `FastqRecord::phred_score(q)` is `q.saturating_sub(33)`, so
/// `phred_score(q) >= 20` holds exactly when `q >= 53`: for `q < 33` the
/// saturation floor is 0, which fails both tests, and `q >= 33` gives
/// `q - 33 >= 20` iff `q >= 53`. Same argument at 30 / 63. Comparing the raw
/// byte is therefore bit-identical to scoring it first, and removes the
/// subtraction from the per-base loop -- see `count_bases`.
///
/// `phred_thresholds_agree_with_phred_score` below pins that equivalence
/// over all 256 byte values, so a change to `phred_score`'s offset cannot
/// silently desynchronize these constants.
const Q20_MIN_QUAL_BYTE: u8 = 33 + 20;
const Q30_MIN_QUAL_BYTE: u8 = 33 + 30;

/// The value `(b | 0x20) & !0x04` takes for exactly `C`, `c`, `G` and `g`.
///
/// ASCII upper- and lowercase differ only in bit 5, so `| 0x20` folds case;
/// `'c'` (0x63) and `'g'` (0x67) then differ only in bit 2, so clearing it
/// collapses the pair. The test is exact, not approximate: requiring
/// `(b | 0x20) & 0xFB == 0x63` forces bit 5 set and leaves bit 2 free, so
/// `b | 0x20` is in `{0x63, 0x67}` and `b` is in `{'C','c','G','g'}` and
/// nothing else. `is_gc_base_matches_the_explicit_four_way_test` below pins
/// that over all 256 byte values.
const GC_FOLDED: u8 = b'c' & 0xFB;

/// Whether a base counts towards GC, in one `or`, one `and` and one compare
/// instead of the four `==` the explicit form used.
///
/// The four-way `==` chain did not lower to four compares: rustc turned it
/// into a range check plus a bit test, which is fewer instructions but costs
/// **two data-dependent conditional branches per base**. Both are close to a
/// coin flip on real DNA (the range check separates A/T from C/G, the bit
/// test separates C/G from the rest of that range), so at ~40-60% GC they
/// mispredict on the order of half the time -- 2 x 1.05e9 unpredictable
/// branches on the benchmark file. This form has none: it lowers to
/// `and`/`cmp`/`sete` and feeds an unconditional `add`, so the per-base
/// control flow is the loop back-edge and nothing else. (rustc folds the
/// `| 0x20` and the `& 0xFB` into a single `and` against a shifted-down
/// constant, so it really is one `and` and one `cmp` in the loop.)
#[inline(always)]
fn is_gc_base(b: u8) -> bool {
    ((b | 0x20) & 0xFB) == GC_FOLDED
}

/// GC bases in `seq`. Used only when `seq` and `qual` disagree in length
/// (a hand-built, structurally invalid record); the equal-length path
/// fuses this into `count_bases`.
#[inline]
fn count_gc(seq: &[u8]) -> u64 {
    let mut gc: u64 = 0;
    for &b in seq {
        gc += is_gc_base(b) as u64;
    }
    gc
}

/// `(q20, q30)` bases in `qual`. Companion to `count_gc`; see `count_bases`.
#[inline]
fn count_quality(qual: &[u8]) -> (u64, u64) {
    let (mut q20, mut q30) = (0u64, 0u64);
    for &q in qual {
        q20 += (q >= Q20_MIN_QUAL_BYTE) as u64;
        q30 += (q >= Q30_MIN_QUAL_BYTE) as u64;
    }
    (q20, q30)
}

/// `(gc, q20, q30)` for a record whose `seq` and `qual` are the same length,
/// in one pass over the pair rather than one pass each.
///
/// Three things are removed per base relative to the two-loop form, all of
/// them multiplied by 1.05e9 bases on the benchmark file:
///
/// * **The second loop's bookkeeping.** Two traversals need two induction
///   variables, two bound compares and two back-edges; one fused traversal
///   needs one of each. Counted on the emitted assembly: 17 instructions per
///   base across the two loops, 14 in the fused one -- 3.15e9 fewer.
/// * **The `saturating_sub`.** `phred_score` lowered to `sub` + `movzbl` +
///   `cmovb` (3 instructions) before either comparison; comparing the raw
///   byte against `Q20_MIN_QUAL_BYTE` / `Q30_MIN_QUAL_BYTE` needs none. That
///   is 3.15e9 instructions, bit-identically (see those constants).
/// * **Four data-dependent branches.** The old GC test branched twice per
///   base (see `is_gc_base`) and the old quality test branched twice more,
///   each guarding an increment. All four are gone: the comparisons now feed
///   `sete`/`sbb` into unconditional adds.
///
/// Accumulating into locals rather than straight into `self` also removes
/// the store per hit that the old form emitted (`movq %r9, 32(%rcx)` after
/// every GC base, and the same for q20/q30) -- up to 2.6e9 stores replaced
/// by three at the end of the record.
///
/// `zip` is what proves both slices in bounds to the optimizer without
/// `unsafe`: it advances two cursors against one precomputed length, so
/// there is no per-base bound compare to elide in the first place.
///
/// Rejected here, having checked the assembly rather than assumed: a
/// 256-byte GC lookup table (same instruction count, but a second load whose
/// address depends on the first -- an L1 latency chain per base where the
/// bitwise form is three 1-cycle ALU ops -- plus 4 cache lines of pressure),
/// and narrowing the accumulators to `u32` or `u8` in the hope of unlocking
/// auto-vectorization (rustc vectorizes none of these shapes; the emitted
/// loop is byte-for-byte the same count, so it would buy an overflow
/// argument for nothing).
#[inline]
fn count_bases(seq: &[u8], qual: &[u8]) -> (u64, u64, u64) {
    let (mut gc, mut q20, mut q30) = (0u64, 0u64, 0u64);
    for (&b, &q) in seq.iter().zip(qual.iter()) {
        gc += is_gc_base(b) as u64;
        q20 += (q >= Q20_MIN_QUAL_BYTE) as u64;
        q30 += (q >= Q30_MIN_QUAL_BYTE) as u64;
    }
    (gc, q20, q30)
}

impl QcSummary {
    pub fn observe_record(&mut self, record: &FastqRecord) {
        self.total_reads += 1;
        self.total_bases += record.seq.len() as u64;

        let (seq, qual) = (record.seq.as_slice(), record.qual.as_slice());
        // One well-predicted branch per record (7 million on the benchmark
        // file, always the same way) buys the fused per-base loop for the
        // 1.05e9 bases behind it. The reader guarantees the two are the same
        // length -- `next_fastq_record_into` rejects a record where they are
        // not, and the FASTA path resizes `qual` to `seq.len()` -- but
        // `FastqRecord`'s fields are public, so a hand-built record with
        // mismatched lengths must still be counted the way it was before
        // (all of `seq` for GC, all of `qual` for Q20/Q30) rather than
        // silently clipped to the shorter of the two by `zip`.
        let (gc, q20, q30) = if seq.len() == qual.len() {
            count_bases(seq, qual)
        } else {
            let (q20, q30) = count_quality(qual);
            (count_gc(seq), q20, q30)
        };

        self.gc_bases += gc;
        self.q20_bases += q20;
        self.q30_bases += q30;
    }

    pub fn merge(&mut self, other: &QcSummary) {
        self.total_reads += other.total_reads;
        self.total_bases += other.total_bases;
        self.q20_bases += other.q20_bases;
        self.q30_bases += other.q30_bases;
        self.gc_bases += other.gc_bases;
    }

    pub fn finalize(&mut self) {
        if self.total_bases > 0 {
            let total = self.total_bases as f64;
            self.gc_content_pct = (self.gc_bases as f64 / total) * 100.0;
            self.q20_pct = (self.q20_bases as f64 / total) * 100.0;
            self.q30_pct = (self.q30_bases as f64 / total) * 100.0;
        }
    }

    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let to_err = |e: std::io::Error| FastDnaError::Io { path: path.to_path_buf(), source: e };

        let (file, pending) = AtomicFile::create(path)?;
        let mut writer = BufWriter::new(file);
        // `serde_json::Error` conflates two distinct failure kinds: an
        // underlying I/O error propagated up from the writer (`is_io()`
        // true), and a genuine serialization failure (a type that cannot be
        // represented, `is_io()` false). Only the former belongs in `Io`,
        // which promises `source()` returns the actual `std::io::Error`;
        // wrapping a serialization failure in a synthetic `io::Error::other`
        // is exactly the laundering `FastDnaError::Export` exists to avoid.
        serde_json::to_writer_pretty(&mut writer, self).map_err(|e| {
            if e.is_io() {
                FastDnaError::Io { path: path.to_path_buf(), source: e.into() }
            } else {
                FastDnaError::Export {
                    path: path.to_path_buf(),
                    reason: e.to_string(),
                    source: Some(Box::new(e)),
                }
            }
        })?;
        writer.flush().map_err(to_err)?;
        drop(writer);
        pending.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(seq: &[u8], qual: &[u8]) -> FastqRecord {
        FastqRecord { id: b"@r".to_vec(), seq: seq.to_vec(), qual: qual.to_vec() }
    }

    /// `is_gc_base` replaced an explicit four-way `==` chain with a bitwise
    /// fold. Exhaustive over the byte, because "exact, not approximate" is
    /// the whole claim: a single extra byte counted as GC would silently
    /// shift `gc_content_pct` on every run.
    #[test]
    fn is_gc_base_matches_the_explicit_four_way_test() {
        for b in 0u8..=255 {
            let explicit = b == b'G' || b == b'g' || b == b'C' || b == b'c';
            assert_eq!(is_gc_base(b), explicit, "byte {b} ({:?})", b as char);
        }
    }

    /// The Q20/Q30 loops compare the raw quality byte instead of scoring it
    /// first. Exhaustive over the byte, and stated against
    /// `FastqRecord::phred_score` itself rather than a copy of its formula,
    /// so changing the Phred offset there fails here instead of quietly
    /// moving every Q20/Q30 figure.
    #[test]
    fn phred_thresholds_agree_with_phred_score() {
        for q in 0u8..=255 {
            let score = FastqRecord::phred_score(q);
            assert_eq!(q >= Q20_MIN_QUAL_BYTE, score >= 20, "byte {q}");
            assert_eq!(q >= Q30_MIN_QUAL_BYTE, score >= 30, "byte {q}");
        }
    }

    #[test]
    fn observe_record_counts_gc_case_insensitively_and_ignores_other_bases() {
        let mut qc = QcSummary::default();
        qc.observe_record(&record(b"ACGTacgtNNnn", &[b'I'; 12]));

        assert_eq!(qc.total_reads, 1);
        assert_eq!(qc.total_bases, 12);
        assert_eq!(qc.gc_bases, 4, "C, G, c and g only -- not N, n, A, T");
    }

    #[test]
    fn observe_record_counts_q20_and_q30_at_and_around_the_boundaries() {
        // Phred+33: '5' = 53 = Q20, '?' = 63 = Q30.
        let qual = [52u8, 53, 62, 63, 33, 255];
        let mut qc = QcSummary::default();
        qc.observe_record(&record(&[b'A'; 6], &qual));

        assert_eq!(qc.q20_bases, 4, "Q20, Q29, Q30 and Q222 clear 20; Q19 and Q0 do not");
        assert_eq!(qc.q30_bases, 2, "only Q30 and Q222 clear 30");
    }

    /// The equal-length fast path and the mismatched-length fallback must
    /// agree wherever both are defined, and the fallback must keep the
    /// pre-fusion behaviour (all of `seq` for GC, all of `qual` for quality)
    /// rather than clipping to the shorter slice.
    #[test]
    fn a_structurally_invalid_record_is_counted_the_way_it_always_was() {
        let mut qc = QcSummary::default();
        qc.observe_record(&record(b"GGGGGG", b"II"));

        assert_eq!(qc.total_bases, 6, "total_bases has always followed seq");
        assert_eq!(qc.gc_bases, 6, "every base of seq is scanned, not just the first two");
        assert_eq!(qc.q20_bases, 2, "every byte of qual is scanned, and there are only two");
        assert_eq!(qc.q30_bases, 2);
    }

    /// `merge` must touch only the raw counters -- the parallel reduce in
    /// `pipeline.rs` merges per-worker summaries in an arbitrary order and
    /// calls `finalize` exactly once at the end, which is only correct if
    /// merging is associative and the percentages are derived, never merged.
    #[test]
    fn merge_is_associative_and_leaves_percentages_to_finalize() {
        let build = |reads, bases, q20, q30, gc| QcSummary {
            total_reads: reads,
            total_bases: bases,
            q20_bases: q20,
            q30_bases: q30,
            gc_bases: gc,
            ..QcSummary::default()
        };
        let (a, b, c) = (build(1, 10, 9, 8, 5), build(2, 20, 18, 16, 10), build(4, 40, 36, 32, 20));

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        let mut right = b.clone();
        right.merge(&c);
        let mut right_all = a.clone();
        right_all.merge(&right);

        assert_eq!(left.total_bases, right_all.total_bases);
        assert_eq!(left.gc_bases, right_all.gc_bases);
        assert_eq!(left.q20_bases, right_all.q20_bases);
        assert_eq!(left.q30_bases, right_all.q30_bases);
        assert_eq!(left.gc_content_pct, 0.0, "merge must not compute percentages");

        left.finalize();
        assert!((left.gc_content_pct - 50.0).abs() < 1e-9);
        assert!((left.q20_pct - 90.0).abs() < 1e-9);
        assert!((left.q30_pct - 80.0).abs() < 1e-9);
    }
}
