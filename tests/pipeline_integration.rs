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
        hpc: false,
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

/// End-to-end version of `kmer::homopolymer_compress_into`'s own
/// `a_deletion_inside_a_homopolymer_run_is_absorbed_by_compression` unit
/// test: run the same clean-vs-deleted pair of reads through the actual
/// pipeline, once with `hpc` off and once with it on, and confirm the flag
/// changes the counted output exactly the way the unit test predicts.
#[test]
fn pipeline_hpc_flag_absorbs_a_run_internal_indel_end_to_end() {
    let k = 4;
    let clean = "GATCAAAAAATCG"; // run of 6 A's
    let with_deletion = "GATCAAAAATCG"; // run of 5 A's: one deleted
    let fastq = format!(
        "@clean\n{clean}\n+\n{}\n@deleted\n{with_deletion}\n+\n{}\n",
        "I".repeat(clean.len()),
        "I".repeat(with_deletion.len())
    );

    let without_hpc_config = config(k);
    let (without_hpc, _qc, _reads) = process_stream_parallel(
        reader_for(&fastq),
        without_hpc_config,
        std::path::Path::new("<memory>"),
        None,
        None,
    )
    .expect("valid input");

    let with_hpc_config = PipelineConfig { hpc: true, ..config(k) };
    let (with_hpc, _qc, _reads) = process_stream_parallel(
        reader_for(&fastq),
        with_hpc_config,
        std::path::Path::new("<memory>"),
        None,
        None,
    )
    .expect("valid input");

    // Without --hpc, the two reads keep their raw (different) lengths --
    // 13 and 12 bases -- so extraction yields (13-4+1) + (12-4+1) = 19 total
    // occurrences.
    assert_eq!(without_hpc.total_kmers(), 19, "raw read lengths must be untouched without --hpc");

    // With --hpc, both reads compress to the identical sequence "GATCATCG"
    // (the run of A's collapses to one, regardless of whether it started as
    // 6 or 5 long), so the run's total occurrence count is exactly double
    // what a single compressed read alone produces: (8-4+1) * 2 = 10.
    let mut compressed_once = Vec::new();
    kmer::homopolymer_compress_into(clean.as_bytes(), &mut compressed_once);
    assert_eq!(compressed_once, b"GATCATCG");
    let single_read_kmers = kmer::extract_canonical_kmers(&compressed_once, k);
    let single_read_distinct: std::collections::HashSet<u64> = single_read_kmers.iter().copied().collect();
    assert_eq!(with_hpc.total_kmers(), single_read_kmers.len() as u64 * 2);
    assert_eq!(
        with_hpc.distinct_kmers(),
        single_read_distinct.len(),
        "the two reads are byte-identical after compression, so together they must contribute no \
         distinct k-mers beyond a single compressed read's own set"
    );

    assert_ne!(
        with_hpc.total_kmers(),
        without_hpc.total_kmers(),
        "the --hpc flag must actually change the counted output on this fixture, or the test \
         proves nothing"
    );
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
        hpc: false,
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
