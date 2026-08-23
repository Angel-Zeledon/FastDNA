//! The pipeline must fail loudly and locate the failure, never print and continue.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, Cursor, Read};
use std::path::Path;

use fastdna::error::FastDnaError;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
}

/// A `BufRead` shim that behaves like a normal in-memory reader for a fixed
/// number of `fill_buf` calls, then fails every call after that. Since a
/// `Cursor`'s `fill_buf` always hands back the whole remaining buffer in one
/// call, and `next_record` performs exactly one `read_until` per FASTQ line,
/// each successful call here corresponds to exactly one line. Three full
/// records is 12 successful calls; the 13th (the header line of record 4)
/// fails, which is what pins the 1-based record arithmetic in
/// `malformed_record_reports_the_1_based_record_number`.
struct FailAfterN {
    inner: Cursor<Vec<u8>>,
    remaining_ok_calls: usize,
}

impl Read for FailAfterN {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl BufRead for FailAfterN {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.remaining_ok_calls == 0 {
            return Err(std::io::Error::other("simulated read failure"));
        }
        self.remaining_ok_calls -= 1;
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
    }
}

fn config(k: usize) -> PipelineConfig {
    PipelineConfig { k, min_quality: 20.0, quality_window: 4, batch_size: 8, num_threads: 2 }
}

#[test]
fn rejects_k_above_the_packing_limit_before_reading() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(33),
        Path::new("sample.fastq"),
        None,
    );

    match result {
        Err(FastDnaError::InvalidK { k }) => assert_eq!(k, 33),
        other => panic!("expected InvalidK, got {other:?}"),
    }
}

#[test]
fn rejects_k_of_zero() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(0),
        Path::new("sample.fastq"),
        None,
    );
    assert!(matches!(result, Err(FastDnaError::InvalidK { k: 0 })));
}

#[test]
fn a_valid_stream_still_succeeds() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGTACGT\n+\nIIIIIIII\n"),
        config(4),
        Path::new("sample.fastq"),
        None,
    );

    let (counter, _qc, reads) = result.expect("valid input must succeed");
    assert_eq!(reads, 1);
    assert_eq!(counter.total_kmers(), 5);
}

#[test]
fn malformed_record_reports_the_1_based_record_number() {
    // Three good records (12 successful lines), then the reader fails on the
    // very first line of what would be record 4.
    let mut fastq = String::new();
    for i in 0..3 {
        fastq.push_str(&format!("@r{i}\nACGT\n+\nIIII\n"));
    }

    let shim = FailAfterN { inner: Cursor::new(fastq.into_bytes()), remaining_ok_calls: 12 };
    let reader = FastqReader::new(shim);

    let result = process_stream_parallel(reader, config(4), Path::new("cohort/sample.fastq"), None);

    match result {
        Err(FastDnaError::MalformedFastq { path, record, .. }) => {
            assert_eq!(record, 4, "must report the 1-based index of the record that failed");
            assert_eq!(path, Path::new("cohort/sample.fastq"));
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// `num_threads: 0` must be rejected before the channel is created. If the
/// guard were removed, zero consumer tasks would spawn, the bounded(64)
/// channel would fill up, and the producer thread would block forever with
/// no one left to drain it -- this test would hang rather than fail.
#[test]
fn zero_threads_is_rejected_instead_of_hanging() {
    let mut fastq = String::new();
    for i in 0..5_000 {
        fastq.push_str(&format!("@r{i}\nACGTACGT\n+\nIIIIIIII\n"));
    }

    let bad_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 0,
    };

    let result = process_stream_parallel(reader_for(&fastq), bad_config, Path::new("sample.fastq"), None);

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "num_threads"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn a_panicking_progress_callback_becomes_an_internal_error() {
    let fastq = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTGCAACG\n+\nIIIIIIII\n";
    let panics = |_: fastdna::progress::Progress| panic!("progress callback exploded");

    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("sample.fastq"), Some(&panics));

    match result {
        Err(FastDnaError::Internal { .. }) => {}
        other => panic!("expected Internal, got {other:?}"),
    }
}
