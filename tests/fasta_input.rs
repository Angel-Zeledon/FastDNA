//! FASTA input, end to end through the counting pipeline.
//!
//! The reader-level parsing rules live in `src/fastq.rs`'s own test module.
//! What this file pins down is the property that makes FASTA support
//! trustworthy rather than merely present: a FASTA file and a FASTQ file
//! holding the same sequences must produce byte-identical k-mer counts.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;

use fastdna_core::counter::KmerCounter;
use fastdna_core::fastq::FastqReader;
use fastdna_core::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(text: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(text.as_bytes().to_vec()))
}

/// `min_quality: 0.0` so the FASTQ side is trimmed exactly as little as the
/// FASTA side, whose synthetic Q40 quality never triggers trimming either.
fn config(k: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 0.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    }
}

fn count(text: &str, k: usize) -> KmerCounter {
    let (counter, _qc, _reads) = process_stream_parallel(
        reader_for(text),
        config(k),
        std::path::Path::new("<memory>"),
        None,
        None,
    )
    .expect("valid input");
    counter
}

fn table(counter: &KmerCounter) -> Vec<(u64, u32)> {
    let mut entries: Vec<(u64, u32)> = counter.iter().collect();
    entries.sort_unstable();
    entries
}

#[test]
fn a_fasta_file_counts_identically_to_the_equivalent_fastq() {
    let sequences = ["ACGTACGTTGCA", "GGGGCCCCAAAA", "ACGTNNNNACGT"];

    let fasta: String = sequences
        .iter()
        .enumerate()
        .map(|(i, s)| format!(">read{i}\n{s}\n"))
        .collect();
    let fastq: String = sequences
        .iter()
        .enumerate()
        .map(|(i, s)| format!("@read{i}\n{s}\n+\n{}\n", "I".repeat(s.len())))
        .collect();

    let from_fasta = count(&fasta, 5);
    let from_fastq = count(&fastq, 5);

    assert_eq!(from_fasta.total_kmers(), from_fastq.total_kmers());
    assert_eq!(from_fasta.distinct_kmers(), from_fastq.distinct_kmers());
    assert_eq!(table(&from_fasta), table(&from_fastq));
    assert!(from_fasta.distinct_kmers() > 0, "the fixture must count something");
}

/// Line wrapping is a formatting detail of the file, not of the sequence: a
/// wrapped genome must yield the same k-mers as the same genome on one line,
/// including the k-mers that straddle a wrap boundary.
#[test]
fn wrapping_a_fasta_sequence_does_not_change_its_kmers() {
    let flat = ">chr1\nACGTTGCAACGTTGCAACGT\n";
    let wrapped = ">chr1\nACGTT\nGCAAC\nGTTGC\nAACGT\n";

    assert_eq!(table(&count(flat, 7)), table(&count(wrapped, 7)));
}

/// The pipeline reports total reads from the producer; a FASTA record is one
/// "read" for that purpose, so a wrapped genome must not be counted as one
/// read per line.
#[test]
fn each_fasta_record_counts_as_exactly_one_read() {
    let (_counter, qc, total_reads) = process_stream_parallel(
        reader_for(">a\nAC\nGT\nAC\n>b\nTTTT\n"),
        config(4),
        std::path::Path::new("<memory>"),
        None,
        None,
    )
    .expect("valid input");

    assert_eq!(total_reads, 2);
    assert_eq!(qc.total_reads, 2);
    assert_eq!(qc.total_bases, 10, "6 bases in the wrapped record plus 4");
}
