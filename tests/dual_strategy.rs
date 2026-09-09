//! Verifies FastDNA's three counting strategies produce bit-identical
//! results.
//!
//! This is the disk-partitioned strategy's acceptance criterion (see
//! `src/disk_spill.rs`'s module doc comment): a caller must never be able
//! to tell, from the counts alone, which strategy actually ran. This test
//! forces each strategy explicitly (via `MemoryPolicy::strategy`, bypassing
//! the automatic estimate-vs-budget chooser entirely) over the exact same
//! input and asserts every observable count matches exactly -- not merely
//! within a tolerance.
//!
//! The same criterion now covers the minimizer-partitioned `Binned`
//! strategy (`src/binned.rs`), which reaches its answer by a route that
//! shares almost nothing with the other two: it never materialises a k-mer
//! occurrence as a `u64` during phase 1, it partitions k-mer space by
//! canonical minimizer rather than by high bits, and it has to put the
//! partition back into ascending k-mer order with a final k-way merge
//! because -- unlike high-bit bucketing -- bin order says nothing about
//! k-mer order. Three independent routes to one table is exactly the point:
//! the failure mode this suite exists to catch is a *silent* wrong count,
//! and a silent wrong count that reproduced identically down all three
//! routes would have to be a coincidence.

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
use fastdna_core::pipeline::{
    process_stream_parallel_with_policy, resolve_strategy, CountStrategy, MemoryPolicy, PipelineConfig,
};

/// Every strategy a caller can force. Adding one here is what makes the
/// comparisons below cover it -- there is no per-strategy test to forget to
/// write.
const ALL_STRATEGIES: [CountStrategy; 3] =
    [CountStrategy::InMemory, CountStrategy::Disk, CountStrategy::Binned];

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
        canonical: true,
        hpc: false,
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

/// Runs `fastq` through every strategy and asserts all of them produced
/// bit-identical observable results, naming the pair that disagreed if any
/// does.
fn assert_all_strategies_agree(fastq: &[u8], threads: usize, k: usize, min_total: u64) {
    let baseline = CountStrategy::InMemory;
    let (base_reads, base_total, base_entries) = run_forced(fastq, baseline, threads, k);
    assert!(
        base_total >= min_total,
        "sanity check: this input must actually reach the regime it was written for ({base_total} < {min_total})"
    );

    for strategy in ALL_STRATEGIES {
        if strategy == baseline {
            continue;
        }
        let (reads, total, entries) = run_forced(fastq, strategy, threads, k);
        assert_eq!(
            reads,
            base_reads,
            "{} and {} report different read counts",
            baseline.as_str(),
            strategy.as_str()
        );
        assert_eq!(
            total,
            base_total,
            "{} and {} report different total k-mer occurrence counts",
            baseline.as_str(),
            strategy.as_str()
        );
        assert_eq!(
            entries.len(),
            base_entries.len(),
            "{} and {} disagree on the number of distinct k-mers",
            baseline.as_str(),
            strategy.as_str()
        );
        assert_eq!(
            entries,
            base_entries,
            "{} and {} produce different (kmer, count) tables, not merely similar ones",
            baseline.as_str(),
            strategy.as_str()
        );
        assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "{}'s output must stay strictly ascending by kmer_u64",
            strategy.as_str()
        );
    }
}

#[test]
fn all_strategies_agree_bit_for_bit_on_a_multi_flush_input() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    // Sized to comfortably exceed the in-memory and disk strategies'
    // internal eager-flush thresholds (2,000,000 raw k-mer instances) more
    // than once across 4 workers, not just cross it barely -- see this
    // file's own doc comment for why that matters. For the binned strategy
    // the same size crosses a different seam: it fills and publishes
    // thousands of 16 KiB chunks, so chunk boundaries and cross-worker
    // interleaving within one bin are exercised too.
    let fastq = synthetic_fastq(200_000, 40, 3_000, 0x5EED_C0FF_EE42);
    assert_all_strategies_agree(&fastq, 4, 21, 2_000_001);
}

#[test]
fn all_strategies_agree_bit_for_bit_on_a_small_single_flush_input() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    // The small-input edge case: well under any threshold, single worker,
    // at most one flush -- the opposite corner from the test above, so a
    // bug that only shows up with zero/one flushes (an empty final bucket,
    // a manifest with no runs at all, a bin store whose every chunk is
    // still open and unpublished) is covered too.
    let fastq = synthetic_fastq(200, 30, 400, 0xC0DE_1234_5678);
    assert_all_strategies_agree(&fastq, 1, 15, 1);
}

/// Thread count must not be observable in the answer either. The binned
/// strategy is the one where that could newly break: its workers share one
/// `BinStore`, so the order chunks arrive in a bin depends on scheduling,
/// and phase 2 runs in parallel over bins.
#[test]
fn all_strategies_agree_across_thread_counts() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    let fastq = synthetic_fastq(20_000, 60, 2_500, 0x2026_0825_0000_0001);
    let k = 25;

    for strategy in ALL_STRATEGIES {
        let (_, one_total, one_entries) = run_forced(&fastq, strategy, 1, k);
        let (_, many_total, many_entries) = run_forced(&fastq, strategy, 8, k);
        assert_eq!(one_total, many_total, "{}: total differs between 1 and 8 threads", strategy.as_str());
        assert_eq!(one_entries, many_entries, "{}: table differs between 1 and 8 threads", strategy.as_str());
    }

    assert_all_strategies_agree(&fastq, 8, k, 1);
}

/// A read shorter than `k`, an all-`N` read, a pure homopolymer read (the
/// one whose every window routes to the binned strategy's fallback bin 0)
/// and a poly-T read (which canonicalises onto the same fallback) all have
/// to be counted the same way by every strategy. A strategy that dropped a
/// whole record here would still produce a well-formed table.
#[test]
fn all_strategies_agree_on_reads_that_yield_no_kmers() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    let mut fastq = synthetic_fastq(500, 40, 600, 0x7777_6666_5555_4444);
    let quals = "I".repeat(40);
    fastq.extend_from_slice(b"@short\nACGT\n+\nIIII\n");
    for (name, base) in [("ambiguous", "N"), ("homopolymer", "A"), ("polyt", "T")] {
        fastq.extend_from_slice(format!("@{name}\n{}\n+\n{quals}\n", base.repeat(40)).as_bytes());
    }

    assert_all_strategies_agree(&fastq, 2, 31, 1);
}

/// What `auto` may and may not reach, checked at the seam a user actually
/// crosses as well as in `pipeline.rs`'s own unit tests.
///
/// Binned was promoted on 2026-09-05 (measured at 2.9x the in-memory
/// strategy's speed and half its peak, `docs/BENCHMARKS.md`), and promoted
/// again on 2026-09-08 to cover streams, so this pins two rules: an
/// unsized input gets binned from `BINNED_BLIND_MIN_THREADS` threads up and
/// in-memory below, and an input that fits nothing still falls back to
/// disk.
///
/// The unsized case used to assert the opposite ("never binned"), which was
/// correct while the fallback was in-memory. It is inverted rather than
/// deleted: the case still needs pinning, and inverting it here is what
/// makes the change visible at the seam a user crosses rather than only in
/// `pipeline.rs`'s own tests.
#[test]
fn the_automatic_chooser_respects_the_promotion_rules_for_binned() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    for max_ram in [None, Some(1u64), Some(1 << 40)] {
        let policy =
            MemoryPolicy { strategy: None, max_ram_bytes: max_ram, estimated_input_bytes: None };
        // 8 threads is above the blind threshold; 2 is below it.
        assert_eq!(
            resolve_strategy(&policy, &config(31, 8)).strategy,
            CountStrategy::Binned,
            "auto should pick binned for an unsized input at 8 threads, budget {max_ram:?}"
        );
        assert_eq!(
            resolve_strategy(&policy, &config(31, 2)).strategy,
            CountStrategy::InMemory,
            "auto should stay in-memory for an unsized input at 2 threads, budget {max_ram:?}"
        );
    }

    let starved = MemoryPolicy {
        strategy: None,
        max_ram_bytes: Some(1),
        estimated_input_bytes: Some(100 << 30),
    };
    assert_eq!(
        resolve_strategy(&starved, &config(31, 8)).strategy,
        CountStrategy::Disk,
        "auto must still reach disk when neither in-memory nor binned fits"
    );

    let roomy = MemoryPolicy {
        strategy: None,
        max_ram_bytes: Some(64 << 30),
        estimated_input_bytes: Some(1_000_000),
    };
    assert_eq!(
        resolve_strategy(&roomy, &config(31, 8)).strategy,
        CountStrategy::Binned,
        "auto passed over binned for a sized input with room to spare"
    );
}

/// `FASTDNA_STRATEGY=binned` is the second of the two ways in, and the only
/// one available to callers with no strategy argument of their own -- the
/// Python bindings' `count()` still takes none. It must resolve to the
/// binned strategy and be reported as an environment override, so a report
/// built from the decision cannot attribute it to the estimator.
///
/// The env var is process-global; every test in this file holds
/// `SERIALIZE_TESTS` while it runs, and the other test binaries are separate
/// processes with their own environment, so the set-and-remove window here
/// cannot race a concurrent reader.
#[test]
fn the_environment_variable_reaches_the_binned_strategy_and_says_so() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());

    std::env::set_var("FASTDNA_STRATEGY", "binned");
    let decision = resolve_strategy(&MemoryPolicy::default(), &config(31, 4));
    let with_explicit_policy = resolve_strategy(
        &MemoryPolicy { strategy: Some(CountStrategy::InMemory), ..MemoryPolicy::default() },
        &config(31, 4),
    );
    std::env::remove_var("FASTDNA_STRATEGY");

    assert_eq!(decision.strategy, CountStrategy::Binned);
    assert!(decision.env_override_applied, "an env-driven choice must not be attributed to the estimate");
    assert_eq!(
        with_explicit_policy.strategy,
        CountStrategy::InMemory,
        "an explicit policy must still win outright over the environment"
    );
}

/// And the whole way through: an env-selected binned run must produce the
/// same table as an explicitly forced in-memory one.
#[test]
fn an_env_selected_binned_run_agrees_with_an_in_memory_run() {
    let _guard = SERIALIZE_TESTS.lock().unwrap_or_else(|p| p.into_inner());
    let fastq = synthetic_fastq(5_000, 50, 1_200, 0x1234_5678_9ABC_DEF0);
    let k = 21;

    let (mem_reads, mem_total, mem_entries) = run_forced(&fastq, CountStrategy::InMemory, 2, k);

    std::env::set_var("FASTDNA_STRATEGY", "binned");
    let outcome = process_stream_parallel_with_policy(
        reader_for(&fastq),
        config(k, 2),
        Path::new("<memory>"),
        None,
        None,
        MemoryPolicy::default(),
    );
    std::env::remove_var("FASTDNA_STRATEGY");

    let (counter, _qc, reads, decision) = outcome.expect("run must succeed");
    assert_eq!(decision.strategy, CountStrategy::Binned, "the env var must be what actually ran");
    assert_eq!(reads, mem_reads);
    assert_eq!(counter.total_kmers(), mem_total);
    assert_eq!(counter.iter().collect::<Vec<(u64, u32)>>(), mem_entries);
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
