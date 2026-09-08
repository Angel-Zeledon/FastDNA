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

use fastdna_core::error::FastDnaError;
use fastdna_core::fastq::FastqReader;
use fastdna_core::pipeline::{process_stream_parallel, PipelineConfig};

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
        canonical: true,
        hpc: false,
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
                reason.contains("length 8") && reason.contains("length 3"),
                "reason should state both full lengths, not just a digit that could match anywhere: {reason}"
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

/// The same truncation check as above, but caught at the second of the
/// three EOF points `next_record` guards: the file ends after the sequence
/// line, with no separator line following.
#[test]
fn file_truncated_after_sequence_line_is_malformed_fastq_not_a_silent_short_record() {
    // A complete first record, then a second record that stops after its
    // sequence line -- no separator or quality line follows.
    let fastq = b"@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nACGT\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { record, reason, .. }) => {
            assert_eq!(record, 2, "must report the record that was cut short");
            assert!(
                reason.contains("separator"),
                "reason must name what line was missing: {reason}"
            );
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// The same truncation check again, at the third and final EOF point: the
/// file ends after the separator line, with no quality line following.
#[test]
fn file_truncated_after_separator_line_is_malformed_fastq_not_a_silent_short_record() {
    // A complete first record, then a second record that stops after its
    // separator line -- no quality line follows.
    let fastq = b"@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nACGT\n+\n";
    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { record, reason, .. }) => {
            assert_eq!(record, 2, "must report the record that was cut short");
            assert!(
                reason.contains("quality"),
                "reason must name what line was missing: {reason}"
            );
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
    // "ACGT" + all six non-N IUPAC codes + "ACGT": a 14-base read has 11
    // raw 4-base windows, but only the two windows fully inside a run of
    // unambiguous bases ("ACGT" at each end) produce a k-mer.
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

/// `read_until(b'\n', ..)` has no length cap, so a file with no early
/// newline -- a mangled download, a binary handed over by mistake -- can
/// have a first "line" that is the entire file. The error message must not
/// become a second full copy of that content.
#[test]
fn oversized_malformed_fastq_header_produces_a_bounded_error_message() {
    let mut fastq = Vec::new();
    fastq.push(b'A'); // neither '@' (FASTQ) nor '>' (FASTA)
    fastq.extend(std::iter::repeat_n(b'A', 5_000));
    fastq.push(b'\n');

    let result = process_stream_parallel(reader_for(&fastq), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { reason, .. }) => {
            assert!(
                reason.len() < 200,
                "reason must be bounded regardless of input size, got {} bytes",
                reason.len()
            );
            assert!(reason.contains('@'), "reason must still say what was expected: {reason}");
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// The same cap applies to the FASTA parser, which quotes the offending
/// header back: a 5,000-character description line (not unusual in a
/// reference assembly) must not be reproduced whole in the error.
#[test]
fn oversized_fasta_header_produces_a_bounded_error_message() {
    let mut fasta = Vec::new();
    fasta.push(b'>');
    fasta.extend(std::iter::repeat_n(b'A', 5_000));
    fasta.push(b'\n'); // header with no sequence under it

    let result = process_stream_parallel(reader_for(&fasta), config(4), Path::new("<memory>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { reason, .. }) => {
            assert!(
                reason.len() < 200,
                "reason must be bounded regardless of input size, got {} bytes",
                reason.len()
            );
            assert!(
                reason.contains("no sequence"),
                "reason must still say what was wrong: {reason}"
            );
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}
