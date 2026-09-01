//! End-to-end "count -> query" test for `docs/feature-gap-analysis.md`'s
//! S1 (`fastdna query`, `src/ktab.rs`): proves that `fastdna count`'s
//! ordinary Parquet output is directly queryable with no extra conversion
//! step, by spawning the real compiled binary via `CARGO_BIN_EXE_fastdna`
//! -- the same convention `tests/cli_subcommands.rs` uses, and for the same
//! reason: only the real binary proves `main.rs` actually wires `cli::
//! QueryArgs` into `ktab::KmerTable`, not just that clap accepts the flags.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_ktab_cli_{name}_{unique}"));
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

/// Runs the binary and asserts it fails, returning stderr -- the mirror of
/// `run_fastdna` for the negative-path tests below.
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

/// "ACGT" is its own reverse complement (a palindromic 4-mer: A<->T, C<->G),
/// so its canonical encoding is itself -- chosen so this test does not need
/// to reason about which strand k-mer counting picked. Three single-record
/// reads of exactly this one 4-mer produce a table with exactly one row:
/// `(encode("ACGT"), 3)`.
#[test]
fn count_then_query_finds_the_exact_recorded_frequency() {
    let scratch = ScratchDir::new("count_query_hit");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGT", "ACGT", "ACGT"]);

    let table_path = scratch.path("counts.parquet");
    run_fastdna(&[
        "count",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        table_path.to_str().unwrap(),
    ]);
    assert!(table_path.exists(), "count must write the Parquet table");

    let stdout = run_fastdna(&["query", "--table", table_path.to_str().unwrap(), "--kmer", "ACGT"]);
    assert!(stdout.contains("found, count = 3"), "expected count 3 for ACGT: {stdout}");

    // The same table's `k` and row count are reported before the lookup
    // result, so a caller can sanity-check the table they opened.
    assert!(stdout.contains("k:     4"), "must report the table's k: {stdout}");
    assert!(stdout.contains("Rows:  1"), "must report the table's row count: {stdout}");
}

/// "GGCC" is also a palindrome (G<->C), distinct from "ACGT", and never
/// appears in the fixture -- a genuine miss, not a malformed query.
#[test]
fn count_then_query_reports_an_absent_kmer_as_not_found() {
    let scratch = ScratchDir::new("count_query_miss");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGT", "ACGT", "ACGT"]);

    let table_path = scratch.path("counts.parquet");
    run_fastdna(&["count", "--input", fastq.to_str().unwrap(), "-k", "4", "-o", table_path.to_str().unwrap()]);

    let stdout = run_fastdna(&["query", "--table", table_path.to_str().unwrap(), "--kmer", "GGCC"]);
    assert!(stdout.contains("not found in table"), "GGCC must be absent: {stdout}");
}

/// `--kmer` also accepts the table's raw `kmer_u64` encoding directly, as a
/// plain decimal integer: "ACGT" packs to `0b00_01_10_11` = 27
/// (A=00, C=01, G=10, T=11).
#[test]
fn query_accepts_the_raw_numeric_kmer_encoding() {
    let scratch = ScratchDir::new("count_query_numeric");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGT", "ACGT", "ACGT"]);

    let table_path = scratch.path("counts.parquet");
    run_fastdna(&["count", "--input", fastq.to_str().unwrap(), "-k", "4", "-o", table_path.to_str().unwrap()]);

    let stdout = run_fastdna(&["query", "--table", table_path.to_str().unwrap(), "--kmer", "27"]);
    assert!(stdout.contains("found, count = 3"), "numeric 27 must resolve to the same row as ACGT: {stdout}");
}

/// A k-mer of the wrong length is a malformed query, not a miss: it must
/// fail loudly rather than silently reporting "not found".
#[test]
fn query_rejects_a_kmer_of_the_wrong_length() {
    let scratch = ScratchDir::new("count_query_wrong_length");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGT", "ACGT", "ACGT"]);

    let table_path = scratch.path("counts.parquet");
    run_fastdna(&["count", "--input", fastq.to_str().unwrap(), "-k", "4", "-o", table_path.to_str().unwrap()]);

    let stderr = run_fastdna_expect_failure(&["query", "--table", table_path.to_str().unwrap(), "--kmer", "AC"]);
    assert!(stderr.contains("length"), "error must explain the length mismatch: {stderr}");
}

/// A file that is not a FastDNA k-mer table (here: the FASTQ input itself)
/// must be rejected with an actionable error, not a panic or a silent
/// wrong answer.
#[test]
fn query_rejects_a_file_that_is_not_a_kmer_table() {
    let scratch = ScratchDir::new("count_query_not_a_table");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGT"]);

    let stderr = run_fastdna_expect_failure(&["query", "--table", fastq.to_str().unwrap(), "--kmer", "ACGT"]);
    assert!(!stderr.is_empty(), "must report an error, not silently exit");
}
