//! Task 8's spec calls out two FASTQ edge cases that had no coverage: CRLF
//! line endings (the likeliest silent-corruption path on the Windows wheel)
//! and zero-byte input. Both are exercised through the public pipeline API.
//!
//! Structural FASTQ validation itself (missing `@`/`+` markers, a
//! sequence/quality length mismatch, a file truncated mid-record) is
//! covered below too, now that `FastqReader::next_record` performs it.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;
use std::path::Path;

use fastdna::error::FastDnaError;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &[u8]) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.to_vec()))
}

fn config(k: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
    }
}

#[test]
fn crlf_and_lf_line_endings_produce_identical_counts() {
    let records = [
        ("r1", "ACGTACGTAC", "IIIIIIIIII"),
        ("r2", "TTGCAACGTT", "IIIIIIIIII"),
        ("r3", "GGGGCCCCAA", "IIIIIIIIII"),
    ];

    let mut lf = String::new();
    let mut crlf = String::new();
    for (id, seq, qual) in records {
        lf.push_str(&format!("@{id}\n{seq}\n+\n{qual}\n"));
        crlf.push_str(&format!("@{id}\r\n{seq}\r\n+\r\n{qual}\r\n"));
    }

    let (lf_counter, _lf_qc, lf_reads) =
        process_stream_parallel(reader_for(lf.as_bytes()), config(5), std::path::Path::new("<memory>"), None, None)
            .expect("LF input must parse");
    let (crlf_counter, _crlf_qc, crlf_reads) =
        process_stream_parallel(reader_for(crlf.as_bytes()), config(5), std::path::Path::new("<memory>"), None, None)
            .expect("CRLF input must parse");

    assert_eq!(lf_reads, crlf_reads, "both encodings must yield the same read count");
    assert_eq!(
        lf_counter.total_kmers(),
        crlf_counter.total_kmers(),
        "a stray \r left in the sequence would corrupt k-mer extraction and change this count"
    );
    assert_eq!(lf_counter.distinct_kmers(), crlf_counter.distinct_kmers());
}

#[test]
fn empty_input_yields_a_clean_zero_count_result_not_an_error_or_hang() {
    let result = process_stream_parallel(reader_for(b""), config(4), std::path::Path::new("<memory>"), None, None);

    let (counter, qc, reads) = result.expect("zero bytes of input must not be an error");
    assert_eq!(reads, 0);
    assert_eq!(counter.total_kmers(), 0);
    assert_eq!(counter.distinct_kmers(), 0);
    assert_eq!(qc.total_reads, 0);
}

#[test]
fn header_not_starting_with_at_sign_is_malformed_fastq() {
    let fastq = b"r1\nACGTACGT\n+\nIIIIIIII\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { reason, .. }) => {
            assert!(
                reason.contains('@'),
                "reason must name what was expected and missing: {reason}"
            );
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

#[test]
fn separator_not_starting_with_plus_is_malformed_fastq() {
    let fastq = b"@r1\nACGTACGT\n*\nIIIIIIII\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { reason, .. }) => {
            assert!(reason.contains('+'), "reason must name what was expected and missing: {reason}");
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// A quality line shorter than its sequence line must be caught as
/// structurally invalid input, not left to reach `quality_trim_end`, which
/// indexes `qual[start..end_pos]` bounded only by `seq.len()` -- a
/// guaranteed index-out-of-bounds panic inside a rayon worker if this ever
/// got past `next_record`.
#[test]
fn quality_shorter_than_sequence_is_malformed_fastq_not_a_panic() {
    let fastq = b"@r1\nACGTACGT\n+\nIII\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { reason, .. }) => {
            assert!(
                reason.contains('8') && reason.contains('3'),
                "reason should state both lengths: {reason}"
            );
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// A file that ends before a record's four lines are all present must be an
/// error, not a silently short record built from whatever lines happened to
/// exist.
#[test]
fn file_truncated_mid_record_is_malformed_fastq_not_a_silent_short_record() {
    // A complete first record, then a second record that stops after its
    // header -- no sequence, separator, or quality line follows.
    let fastq = b"@r1\nACGTACGT\n+\nIIIIIIII\n@r2\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { record, .. }) => {
            assert_eq!(record, 2, "must report the record that was cut short");
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// IUPAC ambiguity codes beyond 'N' (R, Y, S, W, K, M, ...) are legitimate
/// FASTQ content and must not be rejected by structural validation, which
/// checks only the four-line shape -- never the sequence alphabet.
/// `extract_canonical_kmers` already handles any non-ACGT byte by resetting
/// its rolling window, so a read containing them simply yields no k-mers
/// that span the ambiguous positions.
#[test]
fn iupac_ambiguity_codes_parse_fine_and_yield_no_kmers_spanning_them() {
    // "ACGT" + all six non-N IUPAC codes + "ACGT": 8 raw 4-base windows,
    // but only the two windows fully inside a run of unambiguous bases
    // ("ACGT" at each end) produce a k-mer.
    let fastq = b"@r1\nACGTRYSWKMACGT\n+\nIIIIIIIIIIIIII\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    let (counter, _qc, reads) = result.expect("IUPAC ambiguity codes must not be rejected");
    assert_eq!(reads, 1);
    assert_eq!(
        counter.total_kmers(),
        2,
        "only the two 4-base windows entirely within the leading/trailing ACGT runs should count"
    );
}
