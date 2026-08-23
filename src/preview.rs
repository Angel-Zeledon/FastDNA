// src/preview.rs
//! Sampling logic for `fastdna.peek()` -- pure Rust, no PyO3, so it is
//! unit-testable with `cargo test` alone. `src/ffi.rs` only wraps
//! `preview::peek` and converts `PreviewStats` to a Python object.
//!
//! Reads at most `n_reads` records from the source and stops; it never
//! reads the whole file, however large it is (design doc §9.5).

use std::collections::HashSet;
use std::io::BufRead;
use std::path::Path;

use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqReader, FastqRecord};
use crate::kmer;

/// A quick summary of a FASTQ file's read geometry and base composition,
/// computed from at most the first `n_reads` records.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewStats {
    pub n_reads_sampled: usize,
    /// (min, median, max) read length among the sampled reads.
    pub read_length: (usize, usize, usize),
    pub gc_content: f64,
    /// Distinct canonical k-mers observed within the sample itself, at
    /// `suggest_k()`'s recommended k. This is a property of the sample, not
    /// an extrapolation to the whole file -- extrapolating would need
    /// assumptions `peek` has no basis for, since it deliberately never
    /// learns the file's true size. Named `sample_distinct_kmers`, not
    /// `estimated_distinct_kmers`: it is an exact count of the sample's own
    /// k-mers, not an estimate of anything, and "estimated" reads as "an
    /// extrapolation to the whole file" -- exactly the thing this field
    /// does not do.
    pub sample_distinct_kmers: usize,
}

impl PreviewStats {
    /// The largest odd `k` at most `median_read_length / 3`, clamped to
    /// `1..=32`.
    ///
    /// Odd matters: an even-length k-mer can equal its own reverse
    /// complement, which breaks the canonical form's assumption that a
    /// k-mer and its reverse complement are always distinguishable choices.
    pub fn suggest_k(&self) -> usize {
        suggest_k_for_median(self.read_length.1)
    }
}

/// The `suggest_k` rule in isolation, so it can be unit-tested directly
/// against the exact boundary values the plan calls out, without building a
/// `PreviewStats` first.
fn suggest_k_for_median(median_read_length: usize) -> usize {
    let raw = median_read_length / 3;
    // Clamp before enforcing oddness, not after: clamping an odd candidate
    // down to the even boundary 32 would leave an even k, and 31 (not 32) is
    // what the design doc's boundary case (median 300) requires.
    let upper = raw.clamp(1, 32);
    if upper % 2 == 1 {
        upper
    } else {
        (upper - 1).max(1)
    }
}

/// The largest `n_reads` this function accepts. `peek`'s entire reason to
/// exist is a "milliseconds, without reading the rest of the file" preview
/// (see the module doc comment); a caller-supplied value far beyond any
/// legitimate preview size (a typo, a value meant for a different
/// parameter) would silently turn that into a long, ordinary full read
/// instead -- and, called from Python, without releasing the GIL for
/// something that size, would freeze the interpreter for the duration with
/// no way to interrupt it. Rejected outright instead, the same way
/// `process_stream_parallel` rejects `num_threads == 0` rather than letting
/// a bad value degrade silently.
pub const MAX_N_READS: usize = 10_000_000;

/// Reads at most `n_reads` records from `path`, transparently
/// decompressing `.gz` inputs (`FastqReader::from_path` already applies the
/// same extension rule the CLI and `ffi.rs` use).
pub fn peek<P: AsRef<Path>>(path: P, n_reads: usize) -> Result<PreviewStats> {
    if n_reads > MAX_N_READS {
        return Err(FastDnaError::InvalidConfig {
            parameter: "n_reads",
            reason: format!("must be at most {MAX_N_READS} (got {n_reads}); peek() is meant for a quick preview, not a full read"),
        });
    }

    let path_ref = path.as_ref();
    let reader = FastqReader::from_path(path_ref)
        .map_err(|e| FastDnaError::Io { path: path_ref.to_path_buf(), source: e })?;
    peek_from_reader(reader, n_reads, path_ref)
}

/// The sampling logic itself, over any `BufRead` -- kept separate from
/// `peek` so it is testable against an in-memory buffer without touching
/// the filesystem.
fn peek_from_reader<R: BufRead>(mut reader: FastqReader<R>, n_reads: usize, source: &Path) -> Result<PreviewStats> {
    let mut records: Vec<FastqRecord> = Vec::with_capacity(n_reads.min(10_000));

    for _ in 0..n_reads {
        match reader.next_record() {
            Ok(Some(record)) => records.push(record),
            Ok(None) => break,
            Err(FastqReadError::Io(source_err)) => {
                return Err(FastDnaError::Io { path: source.to_path_buf(), source: source_err })
            }
            Err(FastqReadError::Malformed(reason)) => {
                return Err(FastDnaError::MalformedFastq {
                    path: source.to_path_buf(),
                    record: (records.len() as u64) + 1,
                    reason,
                })
            }
        }
    }

    let mut lengths: Vec<usize> = Vec::with_capacity(records.len());
    let mut gc_bases: u64 = 0;
    let mut total_bases: u64 = 0;

    for record in &records {
        lengths.push(record.seq.len());
        total_bases += record.seq.len() as u64;
        for &b in &record.seq {
            if matches!(b, b'G' | b'g' | b'C' | b'c') {
                gc_bases += 1;
            }
        }
    }

    lengths.sort_unstable();
    let (min_len, median_len, max_len) = if lengths.is_empty() {
        (0, 0, 0)
    } else {
        (lengths[0], lengths[(lengths.len() - 1) / 2], lengths[lengths.len() - 1])
    };

    // The k-mer pass uses the median computed above, so it happens after
    // the length pass rather than being fused into the same loop.
    let k = suggest_k_for_median(median_len);
    let mut kmer_seen: HashSet<u64> = HashSet::new();
    for record in &records {
        for kmer_bits in kmer::extract_canonical_kmers(&record.seq, k) {
            kmer_seen.insert(kmer_bits);
        }
    }

    let gc_content = if total_bases > 0 { gc_bases as f64 / total_bases as f64 } else { 0.0 };

    Ok(PreviewStats {
        n_reads_sampled: records.len(),
        read_length: (min_len, median_len, max_len),
        gc_content,
        sample_distinct_kmers: kmer_seen.len(),
    })
}

#[cfg(test)]
// Matches the established pattern in `progress.rs`: `unwrap`/`expect` are
// denied under `src/` because a production code path must never panic on a
// caller's bad input, but test assertions are not that code path, and
// spelling every assertion as a `match` would obscure what each test checks.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fastq_bytes(reads: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (i, seq) in reads.iter().enumerate() {
            buf.extend_from_slice(format!("@r{i}\n{seq}\n+\n{}\n", "I".repeat(seq.len())).as_bytes());
        }
        buf
    }

    fn reader_over(reads: &[&str]) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(fastq_bytes(reads)))
    }

    #[test]
    fn suggest_k_boundary_median_3_gives_k_1() {
        assert_eq!(suggest_k_for_median(3), 1);
    }

    #[test]
    fn suggest_k_boundary_median_96_gives_k_31() {
        // 96 / 3 == 32, which is even; the largest odd k at most 32 is 31.
        assert_eq!(suggest_k_for_median(96), 31);
    }

    #[test]
    fn suggest_k_boundary_median_300_is_clamped_to_31() {
        // 300 / 3 == 100, clamped to the 1..=32 range first (giving 32,
        // still even), then adjusted down to the largest odd value, 31.
        assert_eq!(suggest_k_for_median(300), 31);
    }

    #[test]
    fn suggest_k_never_goes_below_1() {
        assert_eq!(suggest_k_for_median(0), 1);
        assert_eq!(suggest_k_for_median(2), 1);
    }

    #[test]
    fn peek_reports_read_geometry_and_stops_at_n_reads() {
        let reads = vec!["ACGTACGTACGTACGTACGT"; 50];
        let reader = reader_over(&reads);

        let stats = peek_from_reader(reader, 10_000, Path::new("<memory>")).expect("peek must succeed");

        assert_eq!(stats.n_reads_sampled, 50);
        assert_eq!(stats.read_length, (20, 20, 20));
        assert!((0.0..=1.0).contains(&stats.gc_content));
        assert_eq!(stats.gc_content, 0.5, "ACGT is 50% G/C");
        let k = stats.suggest_k();
        assert!((1..=32).contains(&k));
        assert_eq!(k % 2, 1, "suggest_k must always return an odd value");
    }

    #[test]
    fn peek_stops_reading_before_the_end_of_a_larger_file() {
        let reads = vec!["ACGTACGTAC"; 100];
        let reader = reader_over(&reads);

        let stats = peek_from_reader(reader, 10, Path::new("<memory>")).expect("peek must succeed");

        assert_eq!(stats.n_reads_sampled, 10, "must stop at n_reads, not read all 100 records");
    }

    #[test]
    fn peek_on_an_empty_file_does_not_panic() {
        let reader = reader_over(&[]);

        let stats = peek_from_reader(reader, 10, Path::new("<memory>")).expect("peek must succeed on empty input");

        assert_eq!(stats.n_reads_sampled, 0);
        assert_eq!(stats.read_length, (0, 0, 0));
        assert_eq!(stats.gc_content, 0.0);
        assert_eq!(stats.suggest_k(), 1);
    }

    #[test]
    fn peek_surfaces_malformed_fastq_with_a_record_number() {
        let mut bytes = fastq_bytes(&["ACGT"]);
        bytes.extend_from_slice(b"not-a-header-line\n");
        let reader = FastqReader::new(Cursor::new(bytes));

        let err = peek_from_reader(reader, 10, Path::new("bad.fastq")).unwrap_err();

        match err {
            FastDnaError::MalformedFastq { record, .. } => assert_eq!(record, 2),
            other => panic!("expected MalformedFastq, got {other:?}"),
        }
    }
}
