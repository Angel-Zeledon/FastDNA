//! End-to-end "count -> setops -> query" test for
//! `docs/feature-gap-analysis.md`'s S2 (`fastdna union|intersect|diff`,
//! `src/setops.rs`): proves the real compiled binary wires `cli::
//! UnionArgs`/`IntersectArgs`/`DiffArgs` into `setops::union`/`intersect`/
//! `diff` and `export::export_pairs_parquet`, and that the result is
//! immediately queryable with `fastdna query` -- the same
//! `CARGO_BIN_EXE_fastdna` convention `tests/ktab_cli.rs` and
//! `tests/cli_subcommands.rs` already use, for the same reason: only the
//! real binary proves the whole wire-up, not just that clap accepts the
//! flags.

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
        let dir = std::env::temp_dir().join(format!("fastdna_setops_cli_{name}_{unique}"));
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

fn query_count(table: &std::path::Path, kmer: &str) -> Option<u32> {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin)
        .args(["query", "--table", table.to_str().unwrap(), "--kmer", kmer])
        .output()
        .expect("run fastdna query");
    assert!(output.status.success(), "query failed: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if stdout.contains("not found in table") {
        return None;
    }
    // "<kmer>: found, count = <n>"
    let marker = "count = ";
    let idx = stdout.find(marker).expect("found line must report a count");
    let rest = &stdout[idx + marker.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    Some(digits.parse().expect("count must be a valid integer"))
}

/// Builds two k=4 sample tables sharing "ACGT" (a palindrome, canonical by
/// construction -- see `tests/ktab_cli.rs`'s own comment for why this
/// fixture choice avoids reasoning about which strand counting picks):
/// table A additionally has "GGCC" (also a palindrome), table B
/// additionally has "AAAA". `ACGT` is 3x in A, 1x in B; `GGCC` is 2x, only
/// in A; `AAAA` is 2x, only in B.
fn build_sample_tables(scratch: &ScratchDir) -> (PathBuf, PathBuf) {
    let fastq_a = scratch.path("a.fastq");
    write_fastq(&fastq_a, &["ACGT", "ACGT", "ACGT", "GGCC", "GGCC"]);
    let table_a = scratch.path("a.parquet");
    run_fastdna(&["count", "--input", fastq_a.to_str().unwrap(), "-k", "4", "-o", table_a.to_str().unwrap()]);

    let fastq_b = scratch.path("b.fastq");
    write_fastq(&fastq_b, &["ACGT", "AAAA", "AAAA"]);
    let table_b = scratch.path("b.parquet");
    run_fastdna(&["count", "--input", fastq_b.to_str().unwrap(), "-k", "4", "-o", table_b.to_str().unwrap()]);

    (table_a, table_b)
}

#[test]
fn union_sums_the_shared_kmer_and_keeps_each_table_own_kmer() {
    let scratch = ScratchDir::new("union");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let out = scratch.path("union.parquet");
    run_fastdna(&[
        "union",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(out.exists());

    assert_eq!(query_count(&out, "ACGT"), Some(4), "3 (A) + 1 (B) = 4");
    assert_eq!(query_count(&out, "GGCC"), Some(2), "only in A, unchanged");
    assert_eq!(query_count(&out, "AAAA"), Some(2), "only in B, unchanged");
}

#[test]
fn union_combine_max_keeps_the_larger_side_instead_of_summing() {
    let scratch = ScratchDir::new("union_max");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let out = scratch.path("union_max.parquet");
    run_fastdna(&[
        "union",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--combine",
        "max",
    ]);

    assert_eq!(query_count(&out, "ACGT"), Some(3), "max(3, 1) = 3");
}

#[test]
fn intersect_keeps_only_the_shared_kmer_with_min_by_default() {
    let scratch = ScratchDir::new("intersect");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let out = scratch.path("intersect.parquet");
    run_fastdna(&[
        "intersect",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);

    assert_eq!(query_count(&out, "ACGT"), Some(1), "min(3, 1) = 1");
    assert_eq!(query_count(&out, "GGCC"), None, "not shared -- must be dropped");
    assert_eq!(query_count(&out, "AAAA"), None, "not shared -- must be dropped");
}

#[test]
fn diff_removes_the_shared_kmer_and_keeps_a_own_count_for_the_rest() {
    let scratch = ScratchDir::new("diff");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let out = scratch.path("diff.parquet");
    run_fastdna(&[
        "diff",
        "--input",
        table_a.to_str().unwrap(),
        "--subtract",
        table_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);

    assert_eq!(query_count(&out, "ACGT"), None, "present in b at all -- dropped at the default threshold");
    assert_eq!(query_count(&out, "GGCC"), Some(2), "only in a, kept with a's own count");
    assert_eq!(query_count(&out, "AAAA"), None, "never in a to begin with");
}

#[test]
fn diff_max_subtract_count_tolerates_a_low_count_in_the_reference() {
    let scratch = ScratchDir::new("diff_threshold");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let out = scratch.path("diff_tolerant.parquet");
    run_fastdna(&[
        "diff",
        "--input",
        table_a.to_str().unwrap(),
        "--subtract",
        table_b.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--max-subtract-count",
        "1",
    ]);

    // "ACGT" is at count 1 in b, at or under the threshold, so it survives
    // with a's own count (3) rather than being treated as contamination.
    assert_eq!(query_count(&out, "ACGT"), Some(3));
    assert_eq!(query_count(&out, "GGCC"), Some(2));
}

#[test]
fn union_requires_at_least_two_input_tables() {
    let scratch = ScratchDir::new("union_arity");
    let (table_a, _table_b) = build_sample_tables(&scratch);

    let out = scratch.path("union_single.parquet");
    let stderr = run_fastdna_expect_failure(&[
        "union",
        "--input",
        table_a.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

#[test]
fn setops_output_composes_as_a_valid_input_to_a_further_setop() {
    let scratch = ScratchDir::new("compose");
    let (table_a, table_b) = build_sample_tables(&scratch);

    let union_path = scratch.path("union.parquet");
    run_fastdna(&[
        "union",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        "-o",
        union_path.to_str().unwrap(),
    ]);

    // A third table subtracted from the union: only "GGCC"'s reference
    // (built fresh here, containing only "AAAA") removes AAAA from the
    // union, leaving ACGT (4) and GGCC (2).
    let fastq_c = scratch.path("c.fastq");
    write_fastq(&fastq_c, &["AAAA"]);
    let table_c = scratch.path("c.parquet");
    run_fastdna(&["count", "--input", fastq_c.to_str().unwrap(), "-k", "4", "-o", table_c.to_str().unwrap()]);

    let final_path = scratch.path("final.parquet");
    run_fastdna(&[
        "diff",
        "--input",
        union_path.to_str().unwrap(),
        "--subtract",
        table_c.to_str().unwrap(),
        "-o",
        final_path.to_str().unwrap(),
    ]);

    assert_eq!(query_count(&final_path, "ACGT"), Some(4));
    assert_eq!(query_count(&final_path, "GGCC"), Some(2));
    assert_eq!(query_count(&final_path, "AAAA"), None, "removed by the second diff");
}
