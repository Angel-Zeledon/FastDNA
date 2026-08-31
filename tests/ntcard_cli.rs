//! End-to-end test for `fastdna spectrum` (`docs/feature-gap-analysis.md`'s
//! S7(a): ntCard-style streaming k-mer frequency-spectrum estimation,
//! `src/ntcard.rs`), spawning the real compiled binary via
//! `CARGO_BIN_EXE_fastdna` -- the same convention `tests/ktab_cli.rs` uses,
//! and for the same reason: only the real binary proves `main.rs` actually
//! wires `cli::SpectrumArgs` into `ntcard::estimate_spectrum`/`ntcard::
//! write_spectrum`, not just that clap accepts the flags.

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
        let dir = std::env::temp_dir().join(format!("fastdna_ntcard_cli_{name}_{unique}"));
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

/// "AAAA" is exactly `k` bases long (k=4), so each read yields exactly one
/// window and therefore exactly one canonical k-mer occurrence -- 200
/// identical reads is 200 *exact* occurrences of that one distinct k-mer
/// (`NtCardSketch::insert`'s bucket-winner count is exact regardless of
/// estimation error elsewhere; see `src/ntcard.rs`'s module doc comment),
/// so the reported depth itself is hand-checkable even though the
/// distinct-k-mer *count* at that depth is an estimate.
#[test]
fn spectrum_csv_reports_the_exact_depth_for_a_repeated_single_window_read() {
    let scratch = ScratchDir::new("csv");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["AAAA"; 200]);

    let out = scratch.path("spectrum.csv");
    let stdout = run_fastdna(&[
        "spectrum",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(stdout.contains("Estimated distinct k-mers:"), "banner must report F0: {stdout}");
    assert!(out.exists(), "spectrum must write the CSV file");

    let contents = std::fs::read_to_string(&out).unwrap();
    let mut lines = contents.lines();
    assert_eq!(lines.next(), Some("coverage_depth,kmer_distinct_count"), "must use the shared CSV header");
    let row = lines.next().expect("must have at least one data row");
    assert!(row.starts_with("200,"), "expected a single row at the exact depth 200, got: {row}");
}

/// The same run, written as JSON via the `.json` extension: proves the
/// extension-sniffed output format switch actually reaches the CLI, not
/// just `ntcard::write_spectrum`'s own unit tests.
#[test]
fn spectrum_json_extension_writes_a_valid_json_object() {
    let scratch = ScratchDir::new("json");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["AAAA"; 50]);

    let out = scratch.path("spectrum.json");
    run_fastdna(&["spectrum", "--input", fastq.to_str().unwrap(), "-k", "4", "-o", out.to_str().unwrap()]);

    let contents = std::fs::read_to_string(&out).unwrap();
    assert!(contents.trim_start().starts_with('{'), "must be a JSON object: {contents}");
    assert!(contents.contains("\"50\""), "must report the depth-50 class: {contents}");
}

/// `--max-frequency` folds anything deeper into the cap's own row -- the
/// same KMC `-cx` convention `count --histogram-max` uses -- rather than
/// dropping it, so the reported depths never exceed the cap.
#[test]
fn spectrum_max_frequency_caps_the_reported_depth() {
    let scratch = ScratchDir::new("cap");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["AAAA"; 200]);

    let out = scratch.path("spectrum.csv");
    run_fastdna(&[
        "spectrum",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "4",
        "--max-frequency",
        "10",
        "-o",
        out.to_str().unwrap(),
    ]);

    let contents = std::fs::read_to_string(&out).unwrap();
    for line in contents.lines().skip(1) {
        let depth: u32 = line.split(',').next().unwrap().parse().unwrap();
        assert!(depth <= 10, "depth {depth} exceeds --max-frequency 10: {contents}");
    }
}

/// `--format genomescope` writes the headerless, space-separated form,
/// matching `count --histogram-format genomescope`.
#[test]
fn spectrum_genomescope_format_is_headerless() {
    let scratch = ScratchDir::new("genomescope");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["AAAA"; 30]);

    let out = scratch.path("spectrum.txt");
    run_fastdna(&[
        "spectrum",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "4",
        "--format",
        "genomescope",
        "-o",
        out.to_str().unwrap(),
    ]);

    let contents = std::fs::read_to_string(&out).unwrap();
    assert!(!contents.contains("coverage_depth"), "genomescope format must not carry the CSV header");
    let row = contents.lines().next().expect("must have a data row");
    assert!(row.contains(' ') && !row.contains(','), "expected a space-separated row, got: {row}");
}

/// A k-mer size outside `1..=32` must fail loudly with an actionable
/// message, not silently produce an empty or wrong spectrum.
#[test]
fn spectrum_rejects_an_out_of_range_kmer_size() {
    let scratch = ScratchDir::new("bad_k");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGTACGT"]);

    let out = scratch.path("spectrum.csv");
    let stderr = run_fastdna_expect_failure(&[
        "spectrum",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "99",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(stderr.contains("99"), "error must name the invalid k: {stderr}");
    assert!(!out.exists(), "no output should be written on a rejected run");
}
