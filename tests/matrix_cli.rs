//! End-to-end "count several samples -> cohort matrix -> Parquet" test for
//! `docs/feature-gap-analysis.md`'s S6 (`fastdna matrix`, `src/cohort/
//! matrix.rs`, `src/export.rs::export_cohort_matrix_parquet`): proves the
//! real compiled binary discovers/counts a directory of samples (or an
//! explicit `--sample` list), builds a `CohortMatrix`, and writes it as a
//! Parquet file whose rows are readable with no FastDNA-specific tooling --
//! the same `CARGO_BIN_EXE_fastdna` convention `tests/setops_cli.rs` and
//! `tests/ktab_cli.rs` already use, for the same reason: only the real
//! binary proves the whole wire-up, not just that clap accepts the flags.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use arrow::array::{Array, StringArray, UInt32Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_matrix_cli_{name}_{unique}"));
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

fn write_fastq(path: &std::path::Path, reads: &[&str]) {
    let mut f = std::fs::File::create(path).expect("create fastq fixture");
    for (i, seq) in reads.iter().enumerate() {
        writeln!(f, "@r{i}\n{seq}\n+\n{}", "I".repeat(seq.len())).expect("write fixture");
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

/// Reads every `(sample_id, kmer_u64, count)` row out of a cohort-matrix
/// Parquet file -- the "no FastDNA-specific tooling needed" claim
/// `export_cohort_matrix_parquet`'s doc comment makes, exercised here with
/// nothing but `arrow`/`parquet` directly, the same crates a DuckDB/pandas/
/// polars consumer would reach for.
fn read_cohort_rows(path: &std::path::Path) -> (Vec<String>, HashSet<(String, u64, u32)>) {
    let file = std::fs::File::open(path).expect("open cohort matrix parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("valid parquet file");
    let schema = builder.schema().clone();
    let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let reader = builder.build().expect("build parquet reader");
    let mut rows = HashSet::new();
    for batch in reader {
        let batch = batch.expect("read record batch");
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().expect("sample_id column");
        let kmers = batch.column(1).as_any().downcast_ref::<UInt64Array>().expect("kmer_u64 column");
        let counts = batch
            .column(batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("count column");
        for row in 0..batch.num_rows() {
            rows.insert((ids.value(row).to_string(), kmers.value(row), counts.value(row)));
        }
    }
    (column_names, rows)
}

/// Two samples, each written as a single-end FASTQ file directly into a
/// directory: `--input DIR` must discover both via `cohort::
/// discover_samples` and count them into the matrix. k=4; "ACGT" (a
/// palindrome, canonical by construction -- see `tests/setops_cli.rs`'s own
/// comment for why this avoids reasoning about which strand counting
/// picks) is shared by both samples so it survives `--min-samples 2`.
#[test]
fn input_directory_discovers_and_counts_samples_into_a_cohort_matrix() {
    let scratch = ScratchDir::new("dir");
    let dir = scratch.path("samples");
    std::fs::create_dir_all(&dir).expect("create sample dir");

    write_fastq(&dir.join("pat_a.fastq"), &["ACGT", "ACGT", "GGCC"]);
    write_fastq(&dir.join("pat_b.fastq"), &["ACGT", "AAAA", "AAAA"]);

    let out = scratch.path("cohort.parquet");
    run_fastdna(&[
        "matrix",
        "--input",
        dir.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "-k",
        "4",
        "--min-count",
        "1",
        "--min-samples",
        "1",
    ]);
    assert!(out.exists());

    let (columns, rows) = read_cohort_rows(&out);
    assert_eq!(columns, vec!["sample_id", "kmer_u64", "count"], "no --with-sequence: lean schema");

    let sample_ids: HashSet<&str> = rows.iter().map(|(id, _, _)| id.as_str()).collect();
    assert_eq!(sample_ids, HashSet::from(["pat_a", "pat_b"]));

    // "ACGT" is the one k-mer shared between the two samples, so exactly
    // one kmer_u64 value must appear in two rows (once per sample); every
    // other k-mer is private to one sample and appears in exactly one row.
    let mut per_kmer: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
    for (_, kmer, count) in &rows {
        per_kmer.entry(*kmer).or_default().push(*count);
    }
    let shared: Vec<&Vec<u32>> = per_kmer.values().filter(|counts| counts.len() == 2).collect();
    assert_eq!(shared.len(), 1, "exactly one kmer (ACGT) must be shared between the two samples");
    let mut shared_counts = shared[0].clone();
    shared_counts.sort_unstable();
    assert_eq!(shared_counts, vec![1, 2], "pat_a's count (2) and pat_b's count (1) of ACGT must both appear");
}

/// `--sample FILE...` names each sample explicitly, deriving its id from
/// the file name the same way `gwas.py::_sample_id_from_path` does.
#[test]
fn explicit_sample_files_are_counted_with_ids_derived_from_their_names() {
    let scratch = ScratchDir::new("files");
    let file_a = scratch.path("cohortA.fastq");
    let file_b = scratch.path("cohortB.fastq");
    write_fastq(&file_a, &["ACGT", "ACGT"]);
    write_fastq(&file_b, &["ACGT", "GGCC"]);

    let out = scratch.path("cohort.parquet");
    run_fastdna(&[
        "matrix",
        "--sample",
        file_a.to_str().unwrap(),
        file_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "-k",
        "4",
        "--min-count",
        "1",
        "--min-samples",
        "1",
    ]);

    let (_columns, rows) = read_cohort_rows(&out);
    let sample_ids: HashSet<&str> = rows.iter().map(|(id, _, _)| id.as_str()).collect();
    assert_eq!(sample_ids, HashSet::from(["cohortA", "cohortB"]));
}

/// `--with-sequence` adds the decoded `kmer_sequence` column between
/// `kmer_u64` and `count`.
#[test]
fn with_sequence_flag_adds_the_decoded_column() {
    let scratch = ScratchDir::new("with_seq");
    let dir = scratch.path("samples");
    std::fs::create_dir_all(&dir).expect("create sample dir");
    write_fastq(&dir.join("pat_a.fastq"), &["ACGT", "ACGT"]);
    write_fastq(&dir.join("pat_b.fastq"), &["ACGT", "ACGT"]);

    let out = scratch.path("cohort.parquet");
    run_fastdna(&[
        "matrix",
        "--input",
        dir.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "-k",
        "4",
        "--min-count",
        "1",
        "--min-samples",
        "1",
        "--with-sequence",
    ]);

    let (columns, _rows) = read_cohort_rows(&out);
    assert_eq!(columns, vec!["sample_id", "kmer_u64", "kmer_sequence", "count"]);
}

#[test]
fn min_samples_above_the_cohort_size_is_rejected() {
    let scratch = ScratchDir::new("min_samples_too_high");
    let dir = scratch.path("samples");
    std::fs::create_dir_all(&dir).expect("create sample dir");
    write_fastq(&dir.join("pat_a.fastq"), &["ACGT"]);

    let out = scratch.path("cohort.parquet");
    let stderr = run_fastdna_expect_failure(&[
        "matrix",
        "--input",
        dir.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "-k",
        "4",
        "--min-samples",
        "5",
    ]);
    assert!(stderr.contains("min-samples") || stderr.contains("min_samples"), "stderr: {stderr}");
    assert!(!out.exists(), "no file must be written when the run is rejected");
}

#[test]
fn input_and_sample_together_are_rejected() {
    let scratch = ScratchDir::new("both_given");
    let dir = scratch.path("samples");
    std::fs::create_dir_all(&dir).expect("create sample dir");
    write_fastq(&dir.join("pat_a.fastq"), &["ACGT"]);
    let file = scratch.path("extra.fastq");
    write_fastq(&file, &["ACGT"]);

    let out = scratch.path("cohort.parquet");
    let stderr = run_fastdna_expect_failure(&[
        "matrix",
        "--input",
        dir.to_str().unwrap(),
        "--sample",
        file.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(stderr.contains("--input") && stderr.contains("--sample"), "stderr: {stderr}");
}

#[test]
fn neither_input_nor_sample_is_rejected() {
    let scratch = ScratchDir::new("neither_given");
    let out = scratch.path("cohort.parquet");
    let stderr = run_fastdna_expect_failure(&["matrix", "-o", out.to_str().unwrap()]);
    assert!(stderr.contains("--input") && stderr.contains("--sample"), "stderr: {stderr}");
}
