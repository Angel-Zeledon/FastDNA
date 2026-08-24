//! Verifies FastDNA's two counting strategies produce bit-identical
//! results.
//!
//! This is the disk-partitioned strategy's acceptance criterion (see
//! `src/disk_spill.rs`'s module doc comment): a caller must never be able
//! to tell, from the counts alone, which strategy actually ran. This test
//! forces each strategy explicitly (via `MemoryPolicy::strategy`, bypassing
//! the automatic estimate-vs-budget chooser entirely) over the exact same
//! input and asserts every observable count matches exactly -- not merely
//! within a tolerance.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;
use std::path::Path;
use std::sync::Mutex;

/// Serializes every test in this file against every other one. Needed
/// specifically for `disk_strategy_cleans_up_its_scratch_directory_after_a_
/// successful_run` below, which scans the shared OS temp directory for
/// `fastdna-spill-*` entries: `cargo test` runs the tests within one binary
/// concurrently on separate threads of the *same process*, so without this,
/// that scan can observe another of this file's own tests' scratch
/// directory mid-flight and misreport it as a leak. Cross-process
/// interference (other test binaries, run as separate processes with
/// different PIDs baked into the directory name) is not a concern here.
static SERIALIZE_TESTS: Mutex<()> = Mutex::new(());

use fastdna_core::fastq::FastqReader;
use fastdna_core::pipeline::{process_stream_parallel_with_policy, CountStrategy, MemoryPolicy, PipelineConfig};

fn reader_for(fastq: &[u8]) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.to_vec()))
}

/// Deterministic FASTQ generator (no RNG crate: a fixed-seed xorshift64),
/// so this test is exactly reproducible. Draws `num_reads` reads of
/// `read_len` bases from a `genome_len`-base synthetic genome; with
/// `num_reads * read_len` far exceeding `genome_len`, the same k-mers
/// necessarily recur across many different reads -- and therefore across
/// many different eager flush/spill boundaries in both strategies -- which
/// is exactly the case where a seam bug in either strategy's incremental
/// merge would surface. A small fraction of reads carry one ambiguous
/// base, so both strategies are also compared on window-reset behavior
/// around 'N', not just on plain counting.
fn synthetic_fastq(num_reads: usize, read_len: usize, genome_len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let bases = [b'A', b'C', b'G', b'T'];
    let genome: Vec<u8> = (0..genome_len).map(|_| bases[(next_u64() % 4) as usize]).collect();

    let mut out = Vec::with_capacity(num_reads * (read_len * 2 + 24));
    for i in 0..num_reads {
        let start = (next_u64() as usize) % (genome_len - read_len + 1);
        let mut seq: Vec<u8> = genome[start..start + read_len].to_vec();
        if next_u64() % 37 == 0 {
            let pos = (next_u64() as usize) % read_len;
            seq[pos] = b'N';
        }
        out.extend_from_slice(format!("@read{i}\n").as_bytes());
        out.extend_from_slice(&seq);
        out.extend_from_slice(b"\n+\n");
        out.extend(std::iter::repeat_n(b'I', read_len));
        out.push(b'\n');
    }
    out
}

fn config(k: usize, threads: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 512,
        num_threads: threads,
        progress_interval: 1_000_000,
    }
}

/// Runs `fastq` through the given forced strategy and returns
/// `(total_reads, total_kmers, sorted (kmer, count) table)`.
fn run_forced(fastq: &[u8], strategy: CountStrategy, threads: usize, k: usize) -> (u64, u64, Vec<(u64, u32)>) {
    let policy = MemoryPolicy { strategy: Some(strategy), ..MemoryPolicy::default() };
    let (counter, _qc, total_reads, decision) = process_stream_parallel_with_policy(
        reader_for(fastq),
        config(k, threads),
        Path::new("<memory>"),
        None,
        None,
        policy,
    )
    .expect("run must succeed");

    assert_eq!(decision.strategy, strategy, "the forced strategy must be the one that actually ran");

    let entries: Vec<(u64, u32)> = counter.iter().collect();
    (total_reads, counter.total_kmers(), entries)
}

#[test]
fn both_strategies_agree_bit_for_bit_on_a_multi_flush_input() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    // Sized to comfortably exceed both strategies' internal eager-flush
    // thresholds (2,000,000 raw k-mer instances) more than once across 4
    // workers, not just cross it barely -- see this file's own doc comment
    // for why that matters.
    let fastq = synthetic_fastq(200_000, 40, 3_000, 0x5EED_C0FF_EE42);
    let k = 21;
    let threads = 4;

    let (mem_reads, mem_total, mem_entries) = run_forced(&fastq, CountStrategy::InMemory, threads, k);
    let (disk_reads, disk_total, disk_entries) = run_forced(&fastq, CountStrategy::Disk, threads, k);

    assert_eq!(mem_reads, disk_reads, "both strategies must report the same read count");
    assert!(mem_total > 2_000_000, "sanity check: this input must actually exceed the eager-flush threshold");
    assert_eq!(mem_total, disk_total, "both strategies must report the same total k-mer occurrence count");
    assert_eq!(
        mem_entries, disk_entries,
        "both strategies must produce an identical (kmer, count) table, not merely a similar one"
    );
}

#[test]
fn both_strategies_agree_bit_for_bit_on_a_small_single_flush_input() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    // The small-input edge case: well under either threshold, single
    // worker, at most one flush -- the opposite corner from the test
    // above, so a bug that only shows up with zero/one flushes (e.g. an
    // empty final bucket, or a manifest with no runs at all) is covered
    // too.
    let fastq = synthetic_fastq(200, 30, 400, 0xC0DE_1234_5678);
    let k = 15;
    let threads = 1;

    let (mem_reads, mem_total, mem_entries) = run_forced(&fastq, CountStrategy::InMemory, threads, k);
    let (disk_reads, disk_total, disk_entries) = run_forced(&fastq, CountStrategy::Disk, threads, k);

    assert_eq!(mem_reads, disk_reads);
    assert_eq!(mem_total, disk_total);
    assert_eq!(mem_entries, disk_entries);
}

#[test]
fn disk_strategy_cleans_up_its_scratch_directory_after_a_successful_run() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    let fastq = synthetic_fastq(5_000, 40, 2_000, 0xABCD_EF01_2345);
    let before: Vec<std::path::PathBuf> = std::fs::read_dir(std::env::temp_dir())
        .expect("read temp dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("fastdna-spill-")))
        .collect();

    let _ = run_forced(&fastq, CountStrategy::Disk, 2, 17);

    let after: Vec<std::path::PathBuf> = std::fs::read_dir(std::env::temp_dir())
        .expect("read temp dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("fastdna-spill-")))
        .collect();

    assert_eq!(
        before.len(),
        after.len(),
        "no fastdna-spill-* scratch directory should remain after a successful disk-strategy run \
         (before: {before:?}, after: {after:?})"
    );
}
