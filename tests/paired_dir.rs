//! `--paired-dir`: the CLI's automatic paired-end batch-counting mode.
//!
//! `cohort::discover_samples` already groups R1/R2 files into samples for
//! ad hoc cohort listing, but nothing ran a counting pass from that
//! grouping -- a user pointing FastDNA at a directory of paired FASTQ files
//! had to build the two-file `--input` list per sample by hand. These tests
//! exercise the wiring end to end through the public library API
//! (`cli::Cli` parsing/validation plus `cohort::count_paired_samples`,
//! exactly what `main.rs::run_paired_dir` calls), the same pattern
//! `tests/multi_input.rs` and `tests/cli_args.rs` already use to test the
//! CLI surface without spawning the compiled binary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use clap::Parser;

use fastdna_core::cli::Cli;
use fastdna_core::cohort::{self, PairedOutputFormat};
use fastdna_core::error::FastDnaError;
use fastdna_core::pipeline::PipelineConfig;

/// A throwaway directory under the system temp dir, removed on drop -- same
/// pattern as `tests/multi_input.rs::Fixture`.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_paired_dir_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn config(k: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 0.0,
        quality_window: 4,
        batch_size: 4,
        num_threads: 2,
        progress_interval: 100_000,
        hpc: false,
    }
}

const R1: &str = "@r1a\nACGTACGTTG\n+\nIIIIIIIIII\n@r1b\nGGGGCCCCAA\n+\nIIIIIIIIII\n";
const R2: &str = "@r2a\nTTGCAACGTT\n+\nIIIIIIIIII\n@r2b\nACGTACGTTG\n+\nIIIIIIIIII\n";

// ---------------------------------------------------------------------
// CLI parsing and cross-flag validation
// ---------------------------------------------------------------------

#[test]
fn paired_dir_alone_parses_and_validates() {
    let cli = Cli::parse_from(["fastdna", "--paired-dir", "samples", "--paired-output", "out"]);
    assert_eq!(cli.paired_dir.as_deref(), Some(Path::new("samples")));
    assert_eq!(cli.paired_output.as_deref(), Some(Path::new("out")));
    assert!(cli.validate().is_ok());
}

#[test]
fn paired_dir_without_paired_output_is_rejected() {
    let cli = Cli::parse_from(["fastdna", "--paired-dir", "samples"]);
    let err = cli.validate().expect_err("--paired-dir needs somewhere to write");
    assert!(err.contains("--paired-output"), "message must name the missing flag: {err}");
}

#[test]
fn paired_output_without_paired_dir_is_rejected() {
    let cli = Cli::parse_from(["fastdna", "--input", "s.fastq", "--paired-output", "out"]);
    let err = cli.validate().expect_err("--paired-output alone is meaningless");
    assert!(err.contains("--paired-dir"), "message must name the flag it depends on: {err}");
}

#[test]
fn combining_input_and_paired_dir_is_rejected() {
    let cli =
        Cli::parse_from(["fastdna", "--input", "s.fastq", "--paired-dir", "d", "--paired-output", "o"]);
    let err = cli.validate().expect_err("--input and --paired-dir must not both be given");
    assert!(err.contains("--input") && err.contains("--paired-dir"), "message: {err}");
}

/// The one existing behavior this feature must not silently change: a plain
/// `--input`-only invocation, with neither paired flag touched, keeps
/// validating exactly as it always has.
#[test]
fn plain_input_only_invocation_is_unaffected() {
    let cli = Cli::parse_from(["fastdna", "--input", "s.fastq"]);
    assert!(cli.paired_dir.is_none());
    assert!(cli.paired_output.is_none());
    assert!(cli.validate().is_ok());
}

/// Giving neither `--input` nor `--paired-dir` used to only fail deep inside
/// the counting run (`MultiSourceReader::validate`'s "no input files were
/// given"), after the startup banner had already printed. `Cli::validate`
/// now catches it immediately, which is what "fails fast, loudly, and
/// before the counting run" (`CLAUDE.md`) requires.
#[test]
fn neither_input_nor_paired_dir_is_rejected_by_validate_before_any_run_starts() {
    let cli = Cli::parse_from(["fastdna"]);
    let err = cli.validate().expect_err("one of --input or --paired-dir is required");
    assert!(err.contains("--input") && err.contains("--paired-dir"), "message: {err}");
}

// ---------------------------------------------------------------------
// End-to-end: discovery -> counting -> per-sample export
// ---------------------------------------------------------------------

#[test]
fn a_paired_directory_is_discovered_and_counted_into_one_output_per_sample() {
    let src = Fixture::new("ok_src");
    src.write("pat_001_R1.fastq", R1);
    src.write("pat_001_R2.fastq", R2);
    src.write("pat_002_R1.fastq", R2);
    src.write("pat_002_R2.fastq", R1);
    let out = Fixture::new("ok_out");

    let results = cohort::count_paired_samples(
        &src.dir,
        &out.dir,
        PairedOutputFormat::Csv,
        &config(5),
        1,
        None,
    )
    .expect("a cleanly paired two-sample directory must succeed");

    assert_eq!(results.len(), 2);
    let ids: Vec<&str> = results.iter().map(|r| r.sample_id.as_str()).collect();
    assert_eq!(ids, vec!["pat_001", "pat_002"], "sample order must be reproducible");
    for r in &results {
        assert_eq!(r.total_reads, 4, "both mates of each sample must be counted together");
        assert!(r.output_path.exists(), "{} must exist", r.output_path.display());
        assert_eq!(r.output_path, out.dir.join(format!("{}.csv", r.sample_id)));
    }
}

/// The failure mode this feature exists to prevent: an R1 file with no R2
/// mate must stop the run before any counting happens, rather than quietly
/// running that sample as single-end and producing a plausible-looking
/// output alongside the properly paired ones.
#[test]
fn an_unpaired_file_alongside_a_paired_sample_fails_the_whole_run() {
    let src = Fixture::new("orphan_src");
    src.write("pat_001_R1.fastq", R1);
    src.write("pat_001_R2.fastq", R2);
    src.write("pat_002_R1.fastq", R1); // no R2 mate for pat_002
    let out = Fixture::new("orphan_out");

    let result = cohort::count_paired_samples(
        &src.dir,
        &out.dir,
        PairedOutputFormat::Csv,
        &config(5),
        1,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, reason }) => {
            assert_eq!(parameter, "--paired-dir");
            assert!(reason.contains("pat_002"), "message must name the orphaned sample: {reason}");
        }
        other => panic!("expected InvalidConfig naming the orphan, got {other:?}"),
    }
    // Fails before any counting starts: the properly paired sample must not
    // have been counted and written either -- a partial cohort output would
    // look complete until something downstream found the missing sample.
    assert!(
        std::fs::read_dir(&out.dir).unwrap().next().is_none(),
        "no output must be written when discovery itself rejects the directory"
    );
}

#[test]
fn an_empty_directory_is_rejected_rather_than_producing_an_empty_cohort() {
    let src = Fixture::new("empty_src");
    let out = Fixture::new("empty_out");

    let result = cohort::count_paired_samples(
        &src.dir,
        &out.dir,
        PairedOutputFormat::Csv,
        &config(5),
        1,
        None,
    );

    assert!(
        matches!(result, Err(FastDnaError::NoSamplesFound { .. })),
        "an empty sample directory must be an error, got {result:?}"
    );
}
