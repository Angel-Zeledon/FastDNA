//! The pipeline must fail loudly and locate the failure, never print and continue.

use std::io::Cursor;
use std::path::Path;

use fastdna::error::FastDnaError;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
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
