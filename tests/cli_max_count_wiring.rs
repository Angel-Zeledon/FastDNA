//! `cli_args.rs` proves clap parses `--max-count`, and `counter.rs` proves
//! `KmerCounter::prune` filters by it -- but nothing proved the joint: that
//! the CLI flag is actually wired through `main.rs` into `prune`, which was
//! Task 4's entire deliverable. That wiring lives in `main.rs::run`, which
//! is not itself exposed as a testable function, so this test invokes the
//! compiled binary directly (via `CARGO_BIN_EXE_fastdna`, which Cargo sets
//! automatically for integration tests in a crate with a `[[bin]]` target)
//! and inspects the CSV it writes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::process::Command;

/// A fresh scratch directory per test run, cleaned up on drop.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_cli_wiring_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self, file: &str) -> std::path::PathBuf {
        self.0.join(file)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Builds a FASTQ file where the 4-mer "AAAA" occurs `high` times (one read
/// each, read length == k, so exactly one k-mer per read) and "CCCC" occurs
/// `low` times.
fn write_fastq(path: &std::path::Path, high: usize, low: usize) {
    let mut f = std::fs::File::create(path).expect("create fastq fixture");
    for i in 0..high {
        writeln!(f, "@a{i}\nAAAA\n+\nIIII").expect("write fixture");
    }
    for i in 0..low {
        writeln!(f, "@c{i}\nCCCC\n+\nIIII").expect("write fixture");
    }
}

/// Runs the fastdna binary, asserting it exits successfully.
fn run_fastdna(args: &[&str]) {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin).args(args).output().expect("run fastdna binary");
    assert!(
        output.status.success(),
        "fastdna exited with {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn max_count_flag_is_wired_through_to_prune() {
    let scratch = ScratchDir::new("max_count");
    let fastq_path = scratch.path("input.fastq");
    write_fastq(&fastq_path, 10, 2); // AAAA x10, CCCC x2

    // Run 1: no --max-count. Both k-mers must survive.
    let out_uncapped = scratch.path("uncapped.csv");
    let qc_uncapped = scratch.path("uncapped_qc.json");
    run_fastdna(&[
        "--input",
        fastq_path.to_str().unwrap(),
        "--output",
        out_uncapped.to_str().unwrap(),
        "--kmer-size",
        "4",
        "--min-count",
        "1",
        "--qc",
        qc_uncapped.to_str().unwrap(),
        "--threads",
        "2",
    ]);
    let uncapped_csv = std::fs::read_to_string(&out_uncapped).expect("read uncapped csv");
    assert!(uncapped_csv.contains("AAAA"), "without --max-count, AAAA (count 10) must survive:\n{uncapped_csv}");
    assert!(uncapped_csv.contains("CCCC"), "without --max-count, CCCC (count 2) must survive:\n{uncapped_csv}");

    // Run 2: --max-count 5. AAAA (count 10) must be pruned; CCCC (count 2)
    // must survive. If the flag were parsed but not wired to `prune`, this
    // run would produce the same output as run 1.
    let out_capped = scratch.path("capped.csv");
    let qc_capped = scratch.path("capped_qc.json");
    run_fastdna(&[
        "--input",
        fastq_path.to_str().unwrap(),
        "--output",
        out_capped.to_str().unwrap(),
        "--kmer-size",
        "4",
        "--min-count",
        "1",
        "--max-count",
        "5",
        "--qc",
        qc_capped.to_str().unwrap(),
        "--threads",
        "2",
    ]);
    let capped_csv = std::fs::read_to_string(&out_capped).expect("read capped csv");
    assert!(
        !capped_csv.contains("AAAA"),
        "--max-count 5 must prune AAAA (count 10) via KmerCounter::prune:\n{capped_csv}"
    );
    assert!(
        capped_csv.contains("CCCC"),
        "--max-count 5 must keep CCCC (count 2, under the cap):\n{capped_csv}"
    );
}
