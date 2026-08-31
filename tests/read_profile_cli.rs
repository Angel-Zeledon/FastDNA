//! End-to-end "count -> profile" test for `docs/feature-gap-analysis.md`'s
//! S3 (`fastdna profile`, `src/read_profile.rs`): proves the real compiled
//! binary wires `cli::ProfileArgs` into `read_profile::run_profile`, and
//! that both output files it writes (the RLE profile and the per-read
//! summary) contain the expected rows -- the same `CARGO_BIN_EXE_fastdna`
//! convention `tests/read_filter_cli.rs` and `tests/ktab_cli.rs` already
//! use, for the same reason: only the real binary proves the whole
//! wire-up, not just that clap accepts the flags.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use arrow::array::{Array, Float64Array, StringArray, UInt32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_read_profile_cli_{name}_{unique}"));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_fastq(path: &std::path::Path, reads: &[(&str, &str)]) {
    let mut f = std::fs::File::create(path).expect("create fastq fixture");
    for (id, seq) in reads {
        writeln!(f, "@{id}\n{seq}\n+\n{}", "I".repeat(seq.len())).expect("write fixture");
    }
}

fn run_fastdna(args: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin).args(args).output().expect("run fastdna binary");
    assert!(
        output.status.success(),
        "fastdna {args:?} exited with {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn run_fastdna_in_dir(dir: &std::path::Path, args: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin).args(args).current_dir(dir).output().expect("run fastdna binary");
    assert!(
        output.status.success(),
        "fastdna {args:?} (cwd={}) exited with {:?}\nstdout: {}\nstderr: {}",
        dir.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn run_fastdna_expect_failure(args: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin).args(args).output().expect("run fastdna binary");
    assert!(
        !output.status.success(),
        "fastdna {args:?} was expected to fail but exited successfully\nstdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// One row of the RLE profile table, read back directly with the `parquet`
/// crate (a normal dependency of this crate, not merely a dev-dependency) --
/// the same "inspect the real output file" approach `src/export.rs`'s own
/// tests use, applied here to the compiled binary's actual output.
struct ProfileRow {
    read_id: String,
    start: u32,
    run_length: u32,
    count: u32,
}

fn read_profile_rows(path: &std::path::Path) -> Vec<ProfileRow> {
    let file = std::fs::File::open(path).expect("open profile output");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).expect("open profile as parquet").build().expect("build profile reader");
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.expect("read profile batch");
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().expect("read_id column");
        let starts = batch.column(1).as_any().downcast_ref::<UInt32Array>().expect("start column");
        let run_lengths = batch.column(2).as_any().downcast_ref::<UInt32Array>().expect("run_length column");
        let counts = batch.column(3).as_any().downcast_ref::<UInt32Array>().expect("count column");
        for i in 0..batch.num_rows() {
            rows.push(ProfileRow {
                read_id: ids.value(i).to_string(),
                start: starts.value(i),
                run_length: run_lengths.value(i),
                count: counts.value(i),
            });
        }
    }
    rows
}

/// One row of the per-read summary table.
struct SummaryRow {
    read_id: String,
    n_kmers: u32,
    n_present_kmers: u32,
    min_count: Option<u32>,
    median_count: Option<f64>,
    max_count: Option<u32>,
}

fn read_summary_rows(path: &std::path::Path) -> Vec<SummaryRow> {
    let file = std::fs::File::open(path).expect("open summary output");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).expect("open summary as parquet").build().expect("build summary reader");
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.expect("read summary batch");
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().expect("read_id column");
        let n_kmers = batch.column(1).as_any().downcast_ref::<UInt32Array>().expect("n_kmers column");
        let n_present = batch.column(2).as_any().downcast_ref::<UInt32Array>().expect("n_present_kmers column");
        let min_counts = batch.column(3).as_any().downcast_ref::<UInt32Array>().expect("min_count column");
        let median_counts = batch.column(4).as_any().downcast_ref::<Float64Array>().expect("median_count column");
        let max_counts = batch.column(5).as_any().downcast_ref::<UInt32Array>().expect("max_count column");
        for i in 0..batch.num_rows() {
            rows.push(SummaryRow {
                read_id: ids.value(i).to_string(),
                n_kmers: n_kmers.value(i),
                n_present_kmers: n_present.value(i),
                min_count: (!min_counts.is_null(i)).then(|| min_counts.value(i)),
                median_count: (!median_counts.is_null(i)).then(|| median_counts.value(i)),
                max_count: (!max_counts.is_null(i)).then(|| max_counts.value(i)),
            });
        }
    }
    rows
}

/// Builds a reference table at k=4 from a genome made entirely of "AAAA"
/// repeats (canonical: "A" is `0b00` in every base slot, so "AAAA" packs to
/// `0`, smaller than its own reverse complement "TTTT"'s `255` -- the same
/// fixture convention `tests/read_filter_cli.rs` and `python/tests/
/// test_read_filter.py` already use), then a sample with one read fully
/// covered by that reference and one read sharing nothing with it.
fn build_reference_and_sample(scratch: &ScratchDir) -> (PathBuf, PathBuf) {
    let reference_fastq = scratch.path("reference.fastq");
    write_fastq(&reference_fastq, &[("ref1", "AAAAAAAAAAAA")]);
    let reference_table = scratch.path("reference.parquet");
    run_fastdna(&[
        "count",
        "--input",
        reference_fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        reference_table.to_str().unwrap(),
    ]);

    let sample_fastq = scratch.path("sample.fastq");
    write_fastq(
        &sample_fastq,
        &[("matching", "AAAAAAAAAAAA"), ("non_matching", "GCGCGCGCGCGC"), ("short", "AC")],
    );

    (reference_table, sample_fastq)
}

#[test]
fn profile_writes_the_expected_rle_rows_and_summary_rows() {
    let scratch = ScratchDir::new("basic");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let profile_out = scratch.path("profile.parquet");
    let summary_out = scratch.path("summary.parquet");
    let stdout = run_fastdna(&[
        "profile",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "-o",
        profile_out.to_str().unwrap(),
        "--summary",
        summary_out.to_str().unwrap(),
    ]);
    assert!(stdout.contains("Reads read:     3"));
    assert!(stdout.contains("Reads profiled: 2"), "the too-short read must not count as profiled: {stdout}");

    let profile_rows = read_profile_rows(&profile_out);
    let matching: Vec<&ProfileRow> = profile_rows.iter().filter(|r| r.read_id == "matching").collect();
    assert_eq!(matching.len(), 1, "a uniformly-covered read must collapse to exactly one RLE run");
    assert_eq!(matching[0].start, 0);
    assert_eq!(matching[0].run_length, 9, "\"AAAAAAAAAAAA\" at k=4 has 9 canonical k-mer positions");
    assert!(matching[0].count > 0, "every one of this read's k-mers is the reference's own repeated k-mer");

    let non_matching: Vec<&ProfileRow> = profile_rows.iter().filter(|r| r.read_id == "non_matching").collect();
    assert_eq!(non_matching.len(), 1);
    assert_eq!(non_matching[0].count, 0, "none of this read's k-mers appear in the reference");

    let short_rows: Vec<&ProfileRow> = profile_rows.iter().filter(|r| r.read_id == "short").collect();
    assert!(short_rows.is_empty(), "a read shorter than k must produce no profile rows at all");

    let summary_rows = read_summary_rows(&summary_out);
    assert_eq!(summary_rows.len(), 3, "every read gets a summary row, even the too-short one");

    let matching_summary = summary_rows.iter().find(|r| r.read_id == "matching").expect("matching summary row");
    assert_eq!(matching_summary.n_kmers, 9);
    assert_eq!(matching_summary.n_present_kmers, 9);
    assert!(matching_summary.min_count.unwrap() > 0);

    let short_summary = summary_rows.iter().find(|r| r.read_id == "short").expect("short summary row");
    assert_eq!(short_summary.n_kmers, 0);
    assert_eq!(short_summary.n_present_kmers, 0);
    assert_eq!(short_summary.min_count, None, "a read with no k-mers has no distribution to summarize");
    assert_eq!(short_summary.median_count, None);
    assert_eq!(short_summary.max_count, None);
}

/// `--summary` defaults to `read_profile_summary.parquet`, resolved
/// relative to the process's current directory -- so running from a chosen
/// scratch directory with no `--summary` flag must still produce that file
/// there.
#[test]
fn profile_uses_the_default_summary_path_when_not_given() {
    let scratch = ScratchDir::new("default_summary");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let profile_out = scratch.path("profile.parquet");
    run_fastdna_in_dir(
        &scratch.0,
        &[
            "profile",
            "--input",
            sample_fastq.to_str().unwrap(),
            "--table",
            reference_table.to_str().unwrap(),
            "-o",
            profile_out.to_str().unwrap(),
        ],
    );

    let default_summary = scratch.path("read_profile_summary.parquet");
    assert!(default_summary.exists(), "the default --summary path must be used when the flag is omitted");
    assert_eq!(read_summary_rows(&default_summary).len(), 3);
}

#[test]
fn profile_requires_a_table() {
    let scratch = ScratchDir::new("no_table");
    let (_reference_table, sample_fastq) = build_reference_and_sample(&scratch);
    let profile_out = scratch.path("profile.parquet");

    let stderr = run_fastdna_expect_failure(&[
        "profile",
        "--input",
        sample_fastq.to_str().unwrap(),
        "-o",
        profile_out.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

#[test]
fn profile_rejects_a_file_that_is_not_a_kmer_table() {
    let scratch = ScratchDir::new("not_a_table");
    let (_reference_table, sample_fastq) = build_reference_and_sample(&scratch);
    let profile_out = scratch.path("profile.parquet");

    let stderr = run_fastdna_expect_failure(&[
        "profile",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        sample_fastq.to_str().unwrap(),
        "-o",
        profile_out.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}
