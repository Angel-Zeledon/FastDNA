//! End-to-end "count -> similarity" test for `fastdna similarity`
//! (`src/similarity.rs`): proves the real compiled binary wires `cli::
//! SimilarityArgs` into `similarity::pairwise_similarity` and its CSV
//! output, the same `CARGO_BIN_EXE_fastdna` convention `tests/setops_cli.rs`
//! and `tests/ktab_cli.rs` already use, for the same reason: only the real
//! binary proves the whole wire-up, not just that clap accepts the flags.

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
        let dir = std::env::temp_dir().join(format!("fastdna_similarity_cli_{name}_{unique}"));
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

fn count_table(scratch: &ScratchDir, name: &str, k: &str, reads: &[&str]) -> PathBuf {
    let fastq = scratch.path(&format!("{name}.fastq"));
    write_fastq(&fastq, reads);
    let table = scratch.path(&format!("{name}.parquet"));
    run_fastdna(&["count", "--input", fastq.to_str().unwrap(), "-k", k, "-o", table.to_str().unwrap()]);
    table
}

/// One data row of the `sample_a,sample_b,shared,only_a,only_b,jaccard,
/// containment_ab,containment_ba,bray_curtis` CSV `write_similarity_table`
/// writes -- parsed by column position rather than re-deriving a CSV
/// parser, matching the `query_count` string-slicing convention
/// `tests/setops_cli.rs` already uses for the same reason (no CSV crate is
/// a dependency of this test).
struct Row {
    shared: u64,
    only_a: u64,
    only_b: u64,
    jaccard: f64,
    containment_ab: f64,
    containment_ba: f64,
    bray_curtis: f64,
}

fn parse_rows(csv: &str) -> Vec<Row> {
    let mut lines = csv.lines();
    let header = lines.next().expect("csv must have a header line");
    assert_eq!(
        header,
        "sample_a,sample_b,shared,only_a,only_b,jaccard,containment_ab,containment_ba,bray_curtis"
    );
    lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 9, "row must have 9 columns: {line}");
            Row {
                shared: fields[2].parse().expect("shared must be an integer"),
                only_a: fields[3].parse().expect("only_a must be an integer"),
                only_b: fields[4].parse().expect("only_b must be an integer"),
                jaccard: fields[5].parse().expect("jaccard must be a float"),
                containment_ab: fields[6].parse().expect("containment_ab must be a float"),
                containment_ba: fields[7].parse().expect("containment_ba must be a float"),
                bray_curtis: fields[8].parse().expect("bray_curtis must be a float"),
            }
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() < 1e-9, "{what}: expected {expected}, got {actual}");
}

#[test]
fn identical_tables_report_perfect_similarity_to_stdout() {
    let scratch = ScratchDir::new("identical");
    let table_a = count_table(&scratch, "a", "4", &["ACGT", "ACGT", "GGCC"]);
    let table_b = count_table(&scratch, "b", "4", &["ACGT", "ACGT", "GGCC"]);

    let stdout =
        run_fastdna(&["similarity", "--input", table_a.to_str().unwrap(), table_b.to_str().unwrap()]);
    let rows = parse_rows(&stdout);
    assert_eq!(rows.len(), 1);

    assert_eq!(rows[0].shared, 2);
    assert_eq!(rows[0].only_a, 0);
    assert_eq!(rows[0].only_b, 0);
    assert_close(rows[0].jaccard, 1.0, "jaccard");
    assert_close(rows[0].containment_ab, 1.0, "containment_ab");
    assert_close(rows[0].containment_ba, 1.0, "containment_ba");
    assert_close(rows[0].bray_curtis, 0.0, "bray_curtis");
}

/// Table A: ACGT x3, GGCC x2 (distinct = {ACGT, GGCC}).
/// Table B: ACGT x1, AAAA x2 (distinct = {ACGT, AAAA}).
/// shared = {ACGT} (min(3,1)=1), only_a = {GGCC}, only_b = {AAAA}.
/// jaccard = 1/3. containment_ab = containment_ba = 1/2 (both |A|=|B|=2).
/// total_a = 3+2 = 5, total_b = 1+2 = 3; bray_curtis = 1 - 2*1/8 = 0.75.
#[test]
fn partial_overlap_matches_hand_computed_metrics() {
    let scratch = ScratchDir::new("partial");
    let table_a = count_table(&scratch, "a", "4", &["ACGT", "ACGT", "ACGT", "GGCC", "GGCC"]);
    let table_b = count_table(&scratch, "b", "4", &["ACGT", "AAAA", "AAAA"]);

    let stdout =
        run_fastdna(&["similarity", "--input", table_a.to_str().unwrap(), table_b.to_str().unwrap()]);
    let rows = parse_rows(&stdout);
    assert_eq!(rows.len(), 1);

    assert_eq!(rows[0].shared, 1);
    assert_eq!(rows[0].only_a, 1);
    assert_eq!(rows[0].only_b, 1);
    assert_close(rows[0].jaccard, 1.0 / 3.0, "jaccard");
    assert_close(rows[0].containment_ab, 0.5, "containment_ab");
    assert_close(rows[0].containment_ba, 0.5, "containment_ba");
    assert_close(rows[0].bray_curtis, 0.75, "bray_curtis");
}

/// Table X: ACGT x2, GGCC x1 (distinct = {ACGT, GGCC}, both shared with Y).
/// Table Y: ACGT, GGCC, TTAA, CCGG once each (distinct = 4, two exclusive).
/// containment_ab (X in Y) = 2/2 = 1.0 (X fully contained in Y);
/// containment_ba (Y in X) = 2/4 = 0.5 -- the two must differ.
#[test]
fn containment_is_reported_in_both_directions_and_can_differ() {
    let scratch = ScratchDir::new("containment");
    let table_x = count_table(&scratch, "x", "4", &["ACGT", "ACGT", "GGCC"]);
    let table_y = count_table(&scratch, "y", "4", &["ACGT", "GGCC", "TTAA", "CCGG"]);

    let stdout =
        run_fastdna(&["similarity", "--input", table_x.to_str().unwrap(), table_y.to_str().unwrap()]);
    let rows = parse_rows(&stdout);
    assert_eq!(rows.len(), 1);

    assert_eq!(rows[0].shared, 2);
    assert_close(rows[0].containment_ab, 1.0, "X fully contained in Y");
    assert_close(rows[0].containment_ba, 0.5, "only half of Y's kmers are in X");
    assert!(
        (rows[0].containment_ab - rows[0].containment_ba).abs() > 1e-9,
        "containment must actually differ by direction"
    );
}

#[test]
fn output_flag_writes_the_same_csv_to_a_file_instead_of_stdout() {
    let scratch = ScratchDir::new("output_file");
    let table_a = count_table(&scratch, "a", "4", &["ACGT", "ACGT", "GGCC"]);
    let table_b = count_table(&scratch, "b", "4", &["ACGT", "ACGT", "GGCC"]);
    let out = scratch.path("similarity.csv");

    let stdout = run_fastdna(&[
        "similarity",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ]);
    assert!(stdout.is_empty(), "data must go to the file, not stdout, when --output is given");
    assert!(out.exists());

    let contents = std::fs::read_to_string(&out).expect("read output csv");
    let rows = parse_rows(&contents);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].shared, 2);
}

#[test]
fn three_tables_report_one_row_per_unordered_pair() {
    let scratch = ScratchDir::new("three");
    let table_a = count_table(&scratch, "a", "4", &["ACGT", "ACGT", "GGCC"]);
    let table_b = count_table(&scratch, "b", "4", &["ACGT", "TTAA", "TTAA", "TTAA"]);
    let table_c = count_table(&scratch, "c", "4", &["GGCC", "GGCC", "GGCC", "GGCC", "TTAA"]);

    let stdout = run_fastdna(&[
        "similarity",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
        table_c.to_str().unwrap(),
    ]);
    let rows = parse_rows(&stdout);
    assert_eq!(rows.len(), 3, "3 tables must produce 3*(3-1)/2 = 3 pairs");
}

#[test]
fn a_single_table_is_rejected() {
    let scratch = ScratchDir::new("single");
    let table_a = count_table(&scratch, "a", "4", &["ACGT"]);

    let stderr = run_fastdna_expect_failure(&["similarity", "--input", table_a.to_str().unwrap()]);
    assert!(!stderr.is_empty());
}

#[test]
fn mismatched_k_across_tables_is_rejected() {
    let scratch = ScratchDir::new("mismatched_k");
    let table_a = count_table(&scratch, "a", "4", &["ACGT"]);
    let table_b = count_table(&scratch, "b", "6", &["ACGTAC"]);

    let stderr = run_fastdna_expect_failure(&[
        "similarity",
        "--input",
        table_a.to_str().unwrap(),
        table_b.to_str().unwrap(),
    ]);
    assert!(stderr.contains('k'), "error must mention k: {stderr}");
}
