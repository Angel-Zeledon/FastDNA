//! What happens when the filesystem says no.
//!
//! Coverage put 37 uncovered lines in `export.rs` and 8 in `disk_spill.rs`
//! on `Io`/`export_err` branches -- the code that turns a failed write or
//! a corrupt read into a `FastDnaError` rather than a panic. Every one of
//! them was unexecuted, which is the same problem the wide reader's
//! rejection paths had: the handling exists, and nothing had ever proved
//! it runs.
//!
//! These provoke real failures -- a read-only directory, a corrupt gzip
//! member, an output path that is a directory -- and require the process
//! to exit non-zero with a message naming the file. A panic here would be
//! a defect even though the run "failed", because it crosses the FFI
//! boundary in the Python binding as an unrecoverable `PanicException`.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(unix)] // the read-only-directory cases need POSIX permission bits

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_io_fail_{name}_{unique}"));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Restore write permission first, or the removal itself fails.
        if let Ok(entries) = std::fs::read_dir(&self.0) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    let _ = std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o755));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(args: &[&str]) -> (bool, String) {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let out = Command::new(bin).args(args).output().expect("run fastdna");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    // A panic is a distinct failure from an error, and the whole point of
    // these tests is that this code takes the second path.
    assert!(
        !text.contains("panicked at"),
        "the process panicked instead of reporting an error:\n{text}"
    );
    (out.status.success(), text)
}

fn write_reads(path: &Path, n: usize) {
    let mut f = std::fs::File::create(path).expect("create fastq");
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for i in 0..n {
        let seq: String = (0..120)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                b"ACGT"[(state >> 33) as usize % 4] as char
            })
            .collect();
        writeln!(f, "@r{i}\n{seq}\n+\n{}", "I".repeat(seq.len())).expect("write");
    }
}

fn read_only_dir(dir: &ScratchDir, name: &str) -> PathBuf {
    let d = dir.path(name);
    std::fs::create_dir_all(&d).expect("create dir");
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o555)).expect("chmod");
    d
}

#[test]
fn a_parquet_output_that_cannot_be_written_is_an_error_not_a_panic() {
    let dir = ScratchDir::new("parquet_ro");
    let input = dir.path("in.fastq");
    write_reads(&input, 20);
    let locked = read_only_dir(&dir, "locked");

    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "21", "-q", "0",
        "-o", locked.join("counts.parquet").to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "writing into a read-only directory must fail:\n{text}");
    assert!(text.contains("counts.parquet"), "the error must name the file:\n{text}");
}

#[test]
fn a_csv_output_that_cannot_be_written_is_an_error_not_a_panic() {
    // A different export path from Parquet's, with its own `io_err` calls.
    let dir = ScratchDir::new("csv_ro");
    let input = dir.path("in.fastq");
    write_reads(&input, 20);
    let locked = read_only_dir(&dir, "locked");

    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "21", "-q", "0",
        "-o", locked.join("counts.csv").to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "{text}");
    assert!(text.contains("counts.csv"), "the error must name the file:\n{text}");
}

#[test]
fn an_unwritable_qc_report_fails_before_the_count_starts() {
    // `--qc` is preflighted deliberately: the report is written *after*
    // counting, so discovering it is unwritable at the end would throw
    // away the whole run's work.
    let dir = ScratchDir::new("qc_ro");
    let input = dir.path("in.fastq");
    write_reads(&input, 20);
    let locked = read_only_dir(&dir, "locked");

    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "21",
        "-o", dir.path("counts.parquet").to_str().unwrap(),
        "--qc", locked.join("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "{text}");
    assert!(text.contains("qc.json"), "the error must name the report:\n{text}");
    assert!(
        !dir.path("counts.parquet").exists(),
        "the run should have stopped before counting, not after writing output"
    );
}

#[test]
fn an_unwritable_histogram_is_an_error_not_a_panic() {
    let dir = ScratchDir::new("hist_ro");
    let input = dir.path("in.fastq");
    write_reads(&input, 20);
    let locked = read_only_dir(&dir, "locked");

    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "21", "-q", "0",
        "-o", dir.path("counts.parquet").to_str().unwrap(),
        "--histogram", locked.join("hist.csv").to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "{text}");
    assert!(text.contains("hist.csv"), "the error must name the histogram:\n{text}");
}

#[test]
fn an_output_path_that_is_a_directory_is_an_error_not_a_panic() {
    let dir = ScratchDir::new("out_is_dir");
    let input = dir.path("in.fastq");
    write_reads(&input, 20);
    let as_dir = dir.path("counts.parquet");
    std::fs::create_dir_all(&as_dir).expect("create dir");

    let (ok, text) = run(&[
        "count",
        "--input", input.to_str().unwrap(),
        "-k", "21", "-q", "0",
        "-o", as_dir.to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "an output path that is a directory must fail:\n{text}");
}

#[test]
fn a_corrupt_gzip_member_fails_mid_stream_with_the_file_named() {
    // A valid gzip header followed by garbage: the failure happens *during*
    // reading, after the pipeline has started, which is a different code
    // path from "this file is not a FASTQ at all".
    let dir = ScratchDir::new("bad_gz");
    let path = dir.path("broken.fastq.gz");
    let mut bytes = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03];
    bytes.extend(std::iter::repeat_n(0xA5, 4096));
    std::fs::write(&path, &bytes).expect("write broken gzip");

    let (ok, text) = run(&[
        "count",
        "--input", path.to_str().unwrap(),
        "-k", "21",
        "-o", dir.path("counts.parquet").to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "a corrupt gzip must fail:\n{text}");
    assert!(text.contains("broken.fastq.gz"), "the error must name the file:\n{text}");
}

#[test]
fn a_truncated_gzip_is_an_error_not_a_silent_short_read() {
    // The dangerous case: a gzip cut off part-way decompresses cleanly up
    // to the cut and then fails. Silently returning the prefix would be a
    // *wrong count* that looks entirely normal.
    let dir = ScratchDir::new("short_gz");
    let plain = dir.path("full.fastq");
    write_reads(&plain, 200);

    let gz = dir.path("full.fastq.gz");
    let status = Command::new("gzip")
        .args(["-c", plain.to_str().unwrap()])
        .stdout(std::fs::File::create(&gz).expect("create gz"))
        .status()
        .expect("run gzip");
    assert!(status.success(), "gzip failed");

    let whole = std::fs::read(&gz).expect("read gz");
    assert!(whole.len() > 200, "the fixture must be big enough to truncate");
    let truncated = dir.path("truncated.fastq.gz");
    std::fs::write(&truncated, &whole[..whole.len() * 2 / 3]).expect("write truncated");

    let (ok, text) = run(&[
        "count",
        "--input", truncated.to_str().unwrap(),
        "-k", "21", "-q", "0",
        "-o", dir.path("counts.parquet").to_str().unwrap(),
        "--qc", dir.path("qc.json").to_str().unwrap(),
    ]);
    assert!(!ok, "a truncated gzip must fail rather than count the prefix:\n{text}");
}
