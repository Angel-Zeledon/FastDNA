//! CLI paths that coverage showed no test ever executed.
//!
//! `cargo llvm-cov` put `main.rs` at 81.7% of lines, and the gap was not
//! spread evenly: 42 uncovered lines in `run_paired_dir`, 19 in
//! `run_count_wide`, 14 in `run_dist`. `tests/paired_dir.rs` already
//! exercises the *library* side of paired discovery (config validation,
//! `discover_samples`) but never the binary, which is why every line of
//! the subcommand that uses it was unexecuted.
//!
//! These drive the real binary and check the artifacts it leaves behind,
//! not just its exit status: a subcommand that returns 0 and writes
//! nothing passes an exit-code test and fails a user.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_cli_uncovered_{name}_{unique}"));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn subdir(&self, name: &str) -> PathBuf {
        let d = self.0.join(name);
        std::fs::create_dir_all(&d).expect("create subdir");
        d
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> (bool, String) {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let out = Command::new(bin).args(args).output().expect("run fastdna");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// A deterministic sequence: these tests assert exact counts, so nothing
/// may vary between runs.
fn sequence(len: usize, seed: u64) -> String {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            b"ACGT"[(state >> 33) as usize % 4] as char
        })
        .collect()
}

fn write_fastq(path: &Path, reads: &[String]) {
    let mut f = std::fs::File::create(path).expect("create fastq");
    for (i, seq) in reads.iter().enumerate() {
        writeln!(f, "@r{i}\n{seq}\n+\n{}", "I".repeat(seq.len())).expect("write");
    }
}

/// Two samples, each with an R1/R2 pair, plus the `--qc` redirection every
/// test here needs (it defaults to `qc_report.json` in the *current*
/// directory and is written unconditionally).
fn paired_input_dir(dir: &ScratchDir) -> PathBuf {
    let input = dir.subdir("reads");
    for (sample, seed) in [("alpha", 11u64), ("beta", 77)] {
        for (mate, offset) in [("R1", 0u64), ("R2", 1)] {
            let reads: Vec<String> =
                (0..12).map(|i| sequence(80, seed + offset * 1000 + i)).collect();
            write_fastq(&input.join(format!("{sample}_{mate}.fastq")), &reads);
        }
    }
    input
}

#[test]
fn paired_dir_writes_one_table_per_discovered_sample() {
    let dir = ScratchDir::new("paired_ok");
    let input = paired_input_dir(&dir);
    let output = dir.path("counts");

    let (ok, text) = run(&[
        "count",
        "--paired-dir", input.to_str().unwrap(),
        "--paired-output", output.to_str().unwrap(),
        "-k", "21", "-q", "0", "-m", "1",
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(ok, "paired-dir run failed:\n{text}");

    // One file per sample, named from the sample id, and each a real
    // k-mer table rather than an empty placeholder.
    let mut written: Vec<String> = std::fs::read_dir(&output)
        .expect("output dir exists")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    written.sort();
    assert_eq!(written, vec!["alpha.parquet", "beta.parquet"], "got {written:?}");

    for name in &written {
        let table = output.join(name);
        let (ok, out) = run(&["query", "--table", table.to_str().unwrap(), "--kmer", "0"]);
        assert!(ok, "{name} is not a readable k-mer table:\n{out}");
        let rows: u64 = out
            .lines()
            .find_map(|l| l.strip_prefix("Rows:").and_then(|n| n.trim().parse().ok()))
            .unwrap_or_else(|| panic!("no Rows: line for {name}:\n{out}"));
        assert!(rows > 0, "{name} has no k-mers");
        assert!(out.contains("k:     21"), "{name} was counted at the wrong k:\n{out}");
    }
}

#[test]
fn paired_dir_counts_both_mates_into_one_sample() {
    // The property that makes pairing worth doing: a sample's two mate
    // files go into a *single* counting run, so its table holds strictly
    // more k-mers than either mate alone would.
    let dir = ScratchDir::new("paired_merges");
    let input = paired_input_dir(&dir);
    let output = dir.path("counts");

    let (ok, text) = run(&[
        "count",
        "--paired-dir", input.to_str().unwrap(),
        "--paired-output", output.to_str().unwrap(),
        "-k", "21", "-q", "0", "-m", "1",
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(ok, "{text}");

    let rows_of = |table: &Path| -> u64 {
        let (ok, out) = run(&["query", "--table", table.to_str().unwrap(), "--kmer", "0"]);
        assert!(ok, "{out}");
        out.lines()
            .find_map(|l| l.strip_prefix("Rows:").and_then(|n| n.trim().parse().ok()))
            .expect("Rows: line")
    };

    let paired = rows_of(&output.join("alpha.parquet"));

    let r1_only = dir.path("r1_only.parquet");
    let (ok, text) = run(&[
        "count",
        "--input", input.join("alpha_R1.fastq").to_str().unwrap(),
        "-k", "21", "-q", "0", "-m", "1",
        "-o", r1_only.to_str().unwrap(),
        "--qc", dir.path("qc2.json").to_str().unwrap(),
    ]);
    assert!(ok, "{text}");

    assert!(
        paired > rows_of(&r1_only),
        "the paired sample must hold more k-mers than one mate alone: {paired}"
    );
}

#[test]
fn paired_dir_writes_csv_when_asked() {
    let dir = ScratchDir::new("paired_csv");
    let input = paired_input_dir(&dir);
    let output = dir.path("counts_csv");

    let (ok, text) = run(&[
        "count",
        "--paired-dir", input.to_str().unwrap(),
        "--paired-output", output.to_str().unwrap(),
        "--paired-format", "csv",
        "-k", "21", "-q", "0", "-m", "1",
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(ok, "csv paired run failed:\n{text}");

    let csv = output.join("alpha.csv");
    assert!(csv.is_file(), "expected {csv:?}");
    let body = std::fs::read_to_string(&csv).expect("read csv");
    let header = body.lines().next().expect("a header line");
    assert!(header.contains("kmer_u64"), "unexpected header: {header}");
    assert!(body.lines().count() > 1, "csv has a header and no rows");
}

#[test]
fn paired_dir_rejects_a_file_with_no_mate() {
    // Documented as a hard error rather than a warning: this mode runs
    // unattended over many samples, and silently demoting a half-paired
    // sample to single-end produces a normal-looking run over corrupted
    // grouping.
    let dir = ScratchDir::new("paired_orphan");
    let input = paired_input_dir(&dir);
    write_fastq(&input.join("gamma_R1.fastq"), &[sequence(80, 5)]);

    let (ok, text) = run(&[
        "count",
        "--paired-dir", input.to_str().unwrap(),
        "--paired-output", dir.path("out").to_str().unwrap(),
        "-k", "21",
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "an orphaned mate must fail the run:\n{text}");
    assert!(text.contains("gamma"), "the error must name the offending file:\n{text}");
}

#[test]
fn paired_dir_rejects_an_empty_directory() {
    let dir = ScratchDir::new("paired_empty");
    let (ok, text) = run(&[
        "count",
        "--paired-dir", dir.subdir("nothing").to_str().unwrap(),
        "--paired-output", dir.path("out").to_str().unwrap(),
        "-k", "21",
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "an empty directory must fail rather than count nothing:\n{text}");
}

#[test]
fn dist_reports_every_pair_and_honours_the_metric() {
    let dir = ScratchDir::new("dist");
    let a = dir.path("a.fastq");
    let b = dir.path("b.fastq");
    let genome = sequence(2_000, 4242);
    write_fastq(&a, &(0..30).map(|i| genome[i * 20..i * 20 + 200].to_string()).collect::<Vec<_>>());
    write_fastq(&b, &(0..30).map(|i| genome[600 + i * 20..800 + i * 20].to_string()).collect::<Vec<_>>());

    // Default metric, to stdout.
    let (ok, out) = run(&["dist", "--input", a.to_str().unwrap(), b.to_str().unwrap(), "-k", "21"]);
    assert!(ok, "{out}");
    assert!(out.contains("jaccard"), "the header should name the metric:\n{out}");
    let rows = out.lines().filter(|l| l.contains(".fastq")).count();
    assert_eq!(rows, 1, "two symmetric inputs give exactly one unordered pair:\n{out}");

    // Containment is asymmetric, so the same two inputs give both
    // directions -- two rows, not one.
    let table = dir.path("dist.csv");
    let (ok, out) = run(&[
        "dist", "--input", a.to_str().unwrap(), b.to_str().unwrap(),
        "-k", "21", "--metric", "containment", "-o", table.to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    let body = std::fs::read_to_string(&table).expect("read dist csv");
    let data_rows = body.lines().skip(1).filter(|l| !l.trim().is_empty()).count();
    assert_eq!(data_rows, 2, "containment must report both directions:\n{body}");
    assert!(body.lines().next().unwrap().contains("containment"), "{body}");
}

#[test]
fn count_above_k32_writes_csv_and_a_histogram() {
    // `run_count_wide`'s CSV and histogram branches: the wide engine has
    // its own export path, separate from the narrow one.
    let dir = ScratchDir::new("wide_csv");
    let reads: Vec<String> = (0..20).map(|i| sequence(150, i + 900)).collect();
    let input = dir.path("in.fastq");
    write_fastq(&input, &reads);

    let csv = dir.path("counts.csv");
    let hist = dir.path("hist.csv");
    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "41", "-q", "0", "-m", "1",
        "-o", csv.to_str().unwrap(),
        "--histogram", hist.to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(ok, "wide csv run failed:\n{text}");

    let body = std::fs::read_to_string(&csv).expect("read csv");
    let header = body.lines().next().expect("header");
    assert!(header.contains("kmer_sequence"), "wide CSV keys on the decoded sequence: {header}");
    let first = body.lines().nth(1).expect("a data row");
    assert_eq!(first.split(',').next().unwrap().len(), 41, "row is not a 41-base k-mer: {first}");

    let hist_body = std::fs::read_to_string(&hist).expect("read histogram");
    assert!(hist_body.lines().count() > 1, "empty histogram");

    // The QC report is written for the wide path too, not only the narrow.
    let qc = std::fs::read_to_string(dir.path("qc.json")).expect("read qc");
    assert!(qc.contains("total_reads"), "qc report is not a QC report: {qc}");
}
