//! Task 8's spec calls out two FASTQ edge cases that had no coverage: CRLF
//! line endings (the likeliest silent-corruption path on the Windows wheel)
//! and zero-byte input. Both are exercised through the public pipeline API
//! rather than by reaching into `fastq.rs` internals, since structural
//! FASTQ validation is scheduled as its own later task.

// Integration tests legitimately use `.expect()` to fail fast on setup
// errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;

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
        process_stream_parallel(reader_for(lf.as_bytes()), config(5), std::path::Path::new("<memory>"), None)
            .expect("LF input must parse");
    let (crlf_counter, _crlf_qc, crlf_reads) =
        process_stream_parallel(reader_for(crlf.as_bytes()), config(5), std::path::Path::new("<memory>"), None)
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
    let result = process_stream_parallel(reader_for(b""), config(4), std::path::Path::new("<memory>"), None);

    let (counter, qc, reads) = result.expect("zero bytes of input must not be an error");
    assert_eq!(reads, 0);
    assert_eq!(counter.total_kmers(), 0);
    assert_eq!(counter.distinct_kmers(), 0);
    assert_eq!(qc.total_reads, 0);
}
