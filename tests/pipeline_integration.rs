//! End-to-end tests for the parallel pipeline.
//!
//! These cover the contract that `pipeline` must extract k-mers using the same
//! canonical, ambiguity-aware logic as `kmer::extract_canonical_kmers`. The
//! pipeline previously hand-rolled its own sliding window over a byte-string
//! helper, which silently disagreed with that logic on `N` bases.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;

use fastdna_core::fastq::FastqReader;
use fastdna_core::kmer;
use fastdna_core::pipeline::{process_stream_parallel, PipelineConfig};

/// Builds a reader over an in-memory FASTQ payload.
fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
}

/// All-'I' quality is Phred 40, so quality trimming never truncates the reads
/// in these fixtures and the k-mer assertions stay exact.
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
fn pipeline_resets_kmer_window_on_ambiguous_base() {
    // "ACGTNACGT" has 6 raw 4-base windows, but only 2 contain no 'N'.
    // An implementation that does not reset on 'N' reports 6.
    let fastq = "@r1\nACGTNACGT\n+\nIIIIIIIII\n";

    let (counter, _qc, total_reads) =
        process_stream_parallel(reader_for(fastq), config(4), std::path::Path::new("<memory>"), None, None)
            .expect("valid input");

    assert_eq!(total_reads, 1);
    assert_eq!(
        counter.total_kmers(),
        2,
        "expected only the two N-free windows to yield k-mers"
    );
    assert_eq!(counter.distinct_kmers(), 1, "both windows are ACGT");
}

#[test]
fn pipeline_merges_forward_and_reverse_complement_strands() {
    // AACG and CGTT are reverse complements, so both must collapse onto the
    // single canonical k-mer AACG (the lexicographic minimum of the pair).
    let fastq = "@r1\nAACG\n+\nIIII\n@r2\nCGTT\n+\nIIII\n";

    let (counter, _qc, total_reads) =
        process_stream_parallel(reader_for(fastq), config(4), std::path::Path::new("<memory>"), None, None)
            .expect("valid input");

    assert_eq!(total_reads, 2);
    assert_eq!(counter.total_kmers(), 2);
    assert_eq!(
        counter.distinct_kmers(),
        1,
        "reverse-complement reads must share one canonical k-mer"
    );

    let (kmer_bits, count) = counter.iter().next().expect("one entry");
    assert_eq!(kmer::decode_kmer(kmer_bits, 4), "AACG");
    assert_eq!(count, 2);
}

#[test]
fn pipeline_yields_nothing_for_reads_shorter_than_k() {
    let fastq = "@r1\nACG\n+\nIII\n";

    let (counter, _qc, total_reads) =
        process_stream_parallel(reader_for(fastq), config(4), std::path::Path::new("<memory>"), None, None)
            .expect("valid input");

    assert_eq!(total_reads, 1);
    assert_eq!(counter.total_kmers(), 0);
    assert_eq!(counter.distinct_kmers(), 0);
}

#[test]
fn pipeline_agrees_with_kmer_module_across_a_batch() {
    // Whatever the pipeline does internally, its totals must match the
    // reference implementation applied read by read.
    let reads = ["ACGTACGTAC", "TTGCANNACGT", "GGGGCCCCAAAA", "AC"];
    let mut fastq = String::new();
    for (i, r) in reads.iter().enumerate() {
        fastq.push_str(&format!("@r{}\n{}\n+\n{}\n", i, r, "I".repeat(r.len())));
    }

    let k = 5;
    let expected_total: usize = reads
        .iter()
        .map(|r| kmer::extract_canonical_kmers(r.as_bytes(), k).len())
        .sum();

    let (counter, _qc, _) =
        process_stream_parallel(reader_for(&fastq), config(k), std::path::Path::new("<memory>"), None, None)
            .expect("valid input");

    assert_eq!(counter.total_kmers(), expected_total as u64);
}

#[test]
fn progress_callback_receives_a_final_event() {
    use std::sync::{Arc, Mutex};
    use fastdna_core::progress::Progress;
    use fastdna_core::pipeline::PipelineConfig;

    // Exactly 250 reads, asserted below as a literal -- not derived from the
    // pipeline's own report of itself, which would make the assertion
    // tautological (it previously ran 50 reads against a 100,000 interval,
    // so no ReadsProcessed ever fired and this test only proved the
    // pipeline agrees with itself).
    const READ_COUNT: usize = 250;
    let mut fastq = String::new();
    for i in 0..READ_COUNT {
        fastq.push_str(&format!("@r{}\nACGTACGTAC\n+\nIIIIIIIIII\n", i));
    }

    let seen: Arc<Mutex<Vec<Progress>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let callback = move |e: Progress| sink.lock().unwrap().push(e);

    let low_interval_config = PipelineConfig {
        k: 5,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 20,
    };

    let (_counter, _qc, reads) = process_stream_parallel(
        reader_for(&fastq),
        low_interval_config,
        std::path::Path::new("<memory>"),
        Some(&callback),
        None,
    )
    .expect("valid input");

    assert_eq!(reads, READ_COUNT as u64, "the pipeline must report the true read count");

    let events = seen.lock().unwrap();
    let reads_processed_count = events
        .iter()
        .filter(|e| matches!(e, Progress::ReadsProcessed(_)))
        .count();
    assert!(
        reads_processed_count > 0,
        "with 250 reads against a progress_interval of 20, at least one \
         ReadsProcessed event must have fired: {events:?}"
    );

    assert_eq!(
        events.last(),
        Some(&Progress::Finished { reads: READ_COUNT as u64 }),
        "the last event must report the true, literal read count"
    );
}

#[test]
fn silent_and_observed_runs_agree() {
    let fastq = "@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nTTGCAACGTT\n+\nIIIIIIIIII\n";
    let noop = |_: fastdna_core::progress::Progress| {};

    let (silent, _, _) =
        process_stream_parallel(reader_for(fastq), config(5), std::path::Path::new("<memory>"), None, None)
            .expect("valid");
    let (observed, _, _) = process_stream_parallel(
        reader_for(fastq),
        config(5),
        std::path::Path::new("<memory>"),
        Some(&noop),
        None,
    )
    .expect("valid");

    assert_eq!(silent.total_kmers(), observed.total_kmers());
    assert_eq!(silent.distinct_kmers(), observed.distinct_kmers());
}
