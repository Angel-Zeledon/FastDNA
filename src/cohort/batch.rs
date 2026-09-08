// src/cohort/batch.rs
//! Wires `discover_samples`'s R1/R2 pairing into an actual counting run --
//! the library half of the CLI's `--paired-dir` mode.
//!
//! `discovery.rs` only groups files into samples; it runs nothing. Before
//! this module existed, turning "a directory of paired FASTQ files" into
//! counts still meant a user (or a wrapper script) manually building the
//! two-file `--input` list for every sample by hand, even though the
//! pairing logic to do that automatically already existed and was tested.
//! This module is the missing wiring: discover, then run one counting pass
//! per sample, each writing its own output -- nothing about the pairing
//! heuristics themselves changes.

use std::fs;
use std::path::{Path, PathBuf};

use crate::atomic;
use crate::cohort::discovery::{discover_samples, SampleFiles};
use crate::error::{FastDnaError, Result};
use crate::export;
use crate::fastq::MultiSourceReader;
use crate::pipeline::{process_stream_parallel, PipelineConfig};

/// Output file format for `count_paired_samples`'s per-sample exports.
/// Mirrors `cli::CliOutputFormat` one-for-one; kept as a separate type so
/// this module has no dependency on `clap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedOutputFormat {
    Parquet,
    Csv,
}

impl PairedOutputFormat {
    fn extension(self) -> &'static str {
        match self {
            PairedOutputFormat::Parquet => "parquet",
            PairedOutputFormat::Csv => "csv",
        }
    }
}

/// Same discovery as `discover_samples`, but strict about pairing.
///
/// `discover_samples` treats a pair-suffixed file with no mate as a
/// *recorded warning* and falls back to counting it as single-end --
/// correct for ad hoc cohort listing, where a human reads the warning
/// before deciding what to do. `--paired-dir` exists to run counting
/// unattended across many samples with nobody reading anything until it is
/// done, so the same situation has to stop the whole run instead: a
/// half-paired directory silently producing a normal-looking, single-end
/// output for one sample is exactly the "silent empty output is a bug
/// class" failure mode this crate's CLI validation philosophy rules out
/// (see `CLAUDE.md`). This does not change `discover_samples` or its
/// pairing heuristics at all -- it only decides, on top of its result,
/// that an orphan here is fatal rather than advisory.
pub fn discover_paired_samples(dir: &Path) -> Result<Vec<SampleFiles>> {
    let samples = discover_samples(dir)?;
    if let Some(bad) = samples.iter().find(|s| !s.orphan_warning.is_empty()) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--paired-dir",
            reason: bad.orphan_warning.clone(),
        });
    }
    Ok(samples)
}

/// The output path `count_paired_samples` writes one sample's counts to:
/// `<output_dir>/<sample_id>.<extension>`. `sample_id` is a single path
/// component (it is derived from one filename's stem in `discovery.rs` and
/// can therefore never contain a path separator), so this can never resolve
/// outside `output_dir`.
pub fn sample_output_path(output_dir: &Path, sample_id: &str, format: PairedOutputFormat) -> PathBuf {
    output_dir.join(format!("{sample_id}.{}", format.extension()))
}

/// One sample's outcome from `count_paired_samples`, reported so a caller
/// (the CLI) can print per-sample progress without this function doing any
/// I/O to stdout itself -- the library core stays silent, per this crate's
/// convention (`CLAUDE.md`: production code returns `Result`, `println!` is
/// denied outside `main.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleRunResult {
    pub sample_id: String,
    pub output_path: PathBuf,
    pub total_reads: u64,
    pub records_written: usize,
}

/// Discovers, counts, and exports every sample in `dir`, one independent
/// counting run per sample (a discovered sample's mate files are fed into
/// that run together, exactly as multiple `--input` files already are),
/// each writing its own `<sample_id>.<format>` file into `output_dir`.
///
/// `pipeline_config` is a template: every field is `Copy`, and one clone of
/// it is consumed per sample by `process_stream_parallel`, so the same `k`,
/// quality cutoff, thread count, etc. apply to every sample in the cohort.
/// `min_count`/`max_count` are applied to each sample's counter before that
/// sample's export, exactly like the single-run CLI path.
///
/// Fails fast, before any counting starts: `discover_paired_samples` rejects
/// an unpaired file or an empty/all-non-FASTQ directory, `output_dir` is
/// created if missing, and every sample's output path is preflighted for
/// writability up front (`atomic::preflight_writable`) -- so a typo'd or
/// read-only output directory is reported in milliseconds, not after some
/// (possibly large) prefix of the cohort has already been counted.
/// A failure counting or exporting any one sample aborts the whole run
/// rather than silently skipping that sample: a cohort directory missing
/// one output would look complete until something downstream tried to read
/// it and found a gap.
///
/// Deliberately narrower than the single-sample CLI path: no QC JSON and no
/// histogram per sample. Those two flags keep their existing single-file
/// semantics; extending them to a per-sample naming scheme is future work,
/// not required for `--paired-dir` to be useful on its own.
pub fn count_paired_samples(
    dir: &Path,
    output_dir: &Path,
    format: PairedOutputFormat,
    pipeline_config: &PipelineConfig,
    min_count: u32,
    max_count: Option<u32>,
    with_sequence: bool,
) -> Result<Vec<SampleRunResult>> {
    let samples = discover_paired_samples(dir)?;

    fs::create_dir_all(output_dir)
        .map_err(|e| FastDnaError::Io { path: output_dir.to_path_buf(), source: e })?;

    let output_paths: Vec<PathBuf> =
        samples.iter().map(|s| sample_output_path(output_dir, &s.sample_id, format)).collect();
    for path in &output_paths {
        atomic::preflight_writable(path)?;
    }

    let mut results = Vec::with_capacity(samples.len());
    for (sample, output_path) in samples.iter().zip(output_paths) {
        let reader = MultiSourceReader::from_paths(sample.files.clone());
        // `sample.files` is never empty: `discover_samples` only creates a
        // `SampleFiles` entry once it has seen at least one file for that
        // sample id.
        let source_label =
            sample.files.first().cloned().unwrap_or_else(|| dir.to_path_buf());

        let (mut counter, _qc, total_reads) = process_stream_parallel(
            reader,
            pipeline_config.clone(),
            &source_label,
            None,
            None,
        )?;

        counter.prune(min_count, max_count);

        let records_written = match format {
            PairedOutputFormat::Parquet => export::export_parquet(
                &counter,
                &output_path,
                pipeline_config.k,
                min_count,
                with_sequence,
            )?,
            PairedOutputFormat::Csv => {
                export::export_csv(&counter, &output_path, pipeline_config.k, min_count, with_sequence)?
            }
        };

        results.push(SampleRunResult {
            sample_id: sample.sample_id.clone(),
            output_path,
            total_reads,
            records_written,
        });
    }

    Ok(results)
}

#[cfg(test)]
// Same rationale as `discovery.rs`'s test module: `unwrap`/`expect` are
// denied under `src/` for production code, not for test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::error::FastDnaError;

    fn fixture() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir for fixture")
    }

    fn write(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).expect("failed to write fixture file");
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
            canonical: true,
        }
    }

    const R1: &str = "@r1a\nACGTACGTTG\n+\nIIIIIIIIII\n@r1b\nGGGGCCCCAA\n+\nIIIIIIIIII\n";
    const R2: &str = "@r2a\nTTGCAACGTT\n+\nIIIIIIIIII\n@r2b\nACGTACGTTG\n+\nIIIIIIIIII\n";

    #[test]
    fn a_properly_paired_sample_is_counted_and_exported() {
        let src = fixture();
        write(src.path(), "pat_001_R1.fastq", R1);
        write(src.path(), "pat_001_R2.fastq", R2);
        let out = fixture();

        let results = count_paired_samples(
            src.path(),
            out.path(),
            PairedOutputFormat::Csv,
            &config(5),
            1,
            None,
            false,
        )
        .expect("a cleanly paired sample must succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].sample_id, "pat_001");
        assert_eq!(results[0].total_reads, 4, "both mates' reads must be counted together");
        assert!(results[0].records_written > 0);
        assert_eq!(results[0].output_path, out.path().join("pat_001.csv"));
        assert!(results[0].output_path.exists(), "the per-sample file must actually be written");
    }

    #[test]
    fn two_samples_in_one_directory_each_get_their_own_output() {
        let src = fixture();
        write(src.path(), "pat_a_R1.fastq", R1);
        write(src.path(), "pat_a_R2.fastq", R2);
        write(src.path(), "pat_b_R1.fastq", R2);
        write(src.path(), "pat_b_R2.fastq", R1);
        let out = fixture();

        let results = count_paired_samples(
            src.path(),
            out.path(),
            PairedOutputFormat::Parquet,
            &config(5),
            1,
            None,
            false,
        )
        .expect("two cleanly paired samples must succeed");

        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.sample_id.as_str()).collect();
        assert_eq!(ids, vec!["pat_a", "pat_b"], "sample order must be reproducible");
        for r in &results {
            assert!(r.output_path.exists());
            assert_eq!(r.output_path.extension().and_then(|e| e.to_str()), Some("parquet"));
        }
    }

    /// The failure case at the center of this feature: an R1 file with no R2
    /// mate must stop the whole run, not silently become a single-end
    /// sample with a normal-looking output.
    #[test]
    fn an_unpaired_r1_file_is_a_hard_error_not_a_silent_single_end_run() {
        let src = fixture();
        write(src.path(), "pat_001_R1.fastq", R1);
        let out = fixture();

        let result =
            count_paired_samples(src.path(), out.path(), PairedOutputFormat::Csv, &config(5), 1, None, false);

        match result {
            Err(FastDnaError::InvalidConfig { parameter, reason }) => {
                assert_eq!(parameter, "--paired-dir");
                assert!(reason.contains("R1"), "message must explain the orphan: {reason}");
            }
            other => panic!("expected InvalidConfig naming the orphan, got {other:?}"),
        }
        assert!(
            std::fs::read_dir(out.path()).expect("output dir must still exist").next().is_none(),
            "no partial output must be written when discovery itself fails"
        );
    }

    #[test]
    fn an_empty_directory_is_an_error_not_an_empty_cohort() {
        let src = fixture();
        let out = fixture();

        let result =
            count_paired_samples(src.path(), out.path(), PairedOutputFormat::Csv, &config(5), 1, None, false);

        assert!(
            matches!(result, Err(FastDnaError::NoSamplesFound { .. })),
            "an empty sample directory must be an error, got {result:?}"
        );
    }

    #[test]
    fn the_output_directory_is_created_if_it_does_not_exist_yet() {
        let src = fixture();
        write(src.path(), "pat_001_R1.fastq", R1);
        write(src.path(), "pat_001_R2.fastq", R2);
        let out_parent = fixture();
        let out = out_parent.path().join("does_not_exist_yet");
        assert!(!out.exists());

        count_paired_samples(src.path(), &out, PairedOutputFormat::Csv, &config(5), 1, None, false)
            .expect("a missing output directory must be created, not rejected");

        assert!(out.join("pat_001.csv").exists());
    }
}
