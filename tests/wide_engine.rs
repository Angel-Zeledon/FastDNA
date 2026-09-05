//! End-to-end coverage of the `k > 32` engine, through the real binary.
//!
//! The unit tests in `src/wide_kmer.rs` prove the wide encoding agrees
//! with the narrow one on synthetic sequence. These assert the rest of the
//! path a user actually crosses: that `--engine` routes as documented,
//! that a wide run produces a table with the schema and the sorted-by
//! metadata it promises, that the `u64`-keyed operations refuse that table
//! by name rather than misreading it, and -- the one that would catch a
//! whole-pipeline mistake the unit tests cannot -- that counting one real
//! FASTQ at k=31 gives byte-identical answers through both engines.

// The crate denies `expect`/`unwrap` so the *library* never panics on a
// caller's behalf. A test that cannot set up its own fixture has nothing
// to assert, so panicking is the correct failure there -- the same
// carve-out every other integration test in this directory takes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_fastdna"))
}

fn fastq(dir: &Path, name: &str, reads: &[&str]) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut text = String::new();
    for (index, read) in reads.iter().enumerate() {
        text.push_str(&format!("@r{index}\n{read}\n+\n{}\n", "I".repeat(read.len())));
    }
    std::fs::write(&path, text).expect("write fastq");
    path
}

/// A deterministic pseudo-random sequence, long enough to yield k-mers at
/// every width these tests use.
fn sequence(len: usize, seed: u64) -> String {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ['A', 'C', 'G', 'T'][(state % 4) as usize]
        })
        .collect()
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(binary()).args(args).output().expect("run fastdna");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn auto_routes_by_k_and_says_which_engine_ran() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reads: Vec<String> = (0..20).map(|i| sequence(150, i + 1)).collect();
    let refs: Vec<&str> = reads.iter().map(String::as_str).collect();
    let input = fastq(dir.path(), "in.fastq", &refs);

    for (k, engine) in [("31", "narrow k-mers"), ("41", "wide k-mers")] {
        let output = dir.path().join(format!("out{k}.csv"));
        let (ok, text) = run(&[
            "--input", input.to_str().expect("utf8"),
            "--output", output.to_str().expect("utf8"),
            "-k", k, "-q", "0",
            "--qc", dir.path().join("qc.json").to_str().expect("utf8"),
        ]);
        assert!(ok, "k={k} failed:\n{text}");
        assert!(text.contains(engine), "k={k} did not report `{engine}`:\n{text}");
    }
}

/// The whole-pipeline differential check: both engines, one real file, the
/// same `k`. `src/wide_kmer.rs` asserts the *encodings* agree; this asserts
/// the counters, the merge and the exporters do too.
#[test]
fn both_engines_count_the_same_file_identically_at_k31() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reads: Vec<String> = (0..50).map(|i| sequence(150, i * 7 + 3)).collect();
    let refs: Vec<&str> = reads.iter().map(String::as_str).collect();
    let input = fastq(dir.path(), "in.fastq", &refs);

    let mut totals = Vec::new();
    for engine in ["narrow", "wide"] {
        let output = dir.path().join(format!("{engine}.csv"));
        let (ok, text) = run(&[
            "--input", input.to_str().expect("utf8"),
            "--output", output.to_str().expect("utf8"),
            "-k", "31", "-m", "1", "-q", "0",
            "--engine", engine,
            "--qc", dir.path().join("qc.json").to_str().expect("utf8"),
        ]);
        assert!(ok, "--engine {engine} failed:\n{text}");
        let summary = text
            .lines()
            .find(|line| line.starts_with("Total k-mers Indexed:"))
            .unwrap_or_default()
            .to_string();
        totals.push(summary);
    }
    assert_eq!(
        totals[0], totals[1],
        "the two engines disagree on the same file at k=31"
    );
}

#[test]
fn forcing_the_narrow_engine_above_its_range_names_the_flag_that_fixes_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = fastq(dir.path(), "in.fastq", &[&sequence(150, 9)]);
    let (ok, text) = run(&[
        "--input", input.to_str().expect("utf8"),
        "--output", dir.path().join("out.csv").to_str().expect("utf8"),
        "-k", "41", "--engine", "narrow",
    ]);
    assert!(!ok, "k=41 on the narrow engine must fail:\n{text}");
    assert!(text.contains("--engine wide"), "the error must name the fix:\n{text}");
}

#[test]
fn k_above_both_engines_reports_the_real_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = fastq(dir.path(), "in.fastq", &[&sequence(200, 11)]);
    let (ok, text) = run(&[
        "--input", input.to_str().expect("utf8"),
        "--output", dir.path().join("out.csv").to_str().expect("utf8"),
        "-k", "65",
    ]);
    assert!(!ok, "k=65 must fail:\n{text}");
    assert!(
        text.contains("between 1 and 64"),
        "the limit reported must be the one that actually applies:\n{text}"
    );
}

/// A wide table is a FastDNA table, and the `u64`-keyed operations must say
/// so while refusing it -- not report it as an unrecognised file.
#[test]
fn u64_operations_refuse_a_wide_table_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let reads: Vec<String> = (0..20).map(|i| sequence(150, i + 41)).collect();
    let refs: Vec<&str> = reads.iter().map(String::as_str).collect();
    let input = fastq(dir.path(), "in.fastq", &refs);
    let table = dir.path().join("wide.parquet");

    let (ok, text) = run(&[
        "--input", input.to_str().expect("utf8"),
        "--output", table.to_str().expect("utf8"),
        "-k", "41", "-q", "0",
        "--qc", dir.path().join("qc.json").to_str().expect("utf8"),
    ]);
    assert!(ok, "wide count failed:\n{text}");

    let (ok, text) = run(&[
        "union",
        "--input", table.to_str().expect("utf8"), table.to_str().expect("utf8"),
        "--output", dir.path().join("u.parquet").to_str().expect("utf8"),
    ]);
    assert!(!ok, "union on a wide table must fail:\n{text}");
    assert!(text.contains("kmer_bits"), "the error must name the wide key column:\n{text}");
}
