//! `fastdna -i -`: reading the sample from a pipe.
//!
//! `fasterq-dump SRR... | fastdna -i -` is standard HPC practice, and the
//! only honest way to test a pipe is to actually build one, so these spawn
//! the real binary (`CARGO_BIN_EXE_fastdna`, which cargo guarantees is
//! built before this test runs) and write to its stdin.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const FASTQ: &str = "@r1\nACGTACGTTG\n+\nIIIIIIIIII\n@r2\nACGTACGTTG\n+\nIIIIIIIIII\n";

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_stdin_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The flags every run here shares: small k, no quality trimming, and CSV
/// output so the result can be compared as text.
fn args_for(out: &Path, qc: &Path, input: &str) -> Vec<String> {
    ["-i", input, "-o", &out.to_string_lossy(), "--qc", &qc.to_string_lossy(), "-k", "5", "-q", "0"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Runs the binary with `stdin_bytes` on stdin and returns the counts CSV.
fn run_with_stdin(fx: &Fixture, stdin_bytes: &[u8]) -> String {
    let out = fx.dir.join("stdin_counts.csv");
    let qc = fx.dir.join("stdin_qc.json");

    let mut child = Command::new(env!("CARGO_BIN_EXE_fastdna"))
        .args(args_for(&out, &qc, "-"))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary must be runnable");

    child
        .stdin
        .as_mut()
        .expect("stdin was piped")
        .write_all(stdin_bytes)
        .unwrap();
    // Dropped so the child sees EOF rather than blocking forever.
    drop(child.stdin.take());

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read_to_string(&out).expect("the counts file must have been written")
}

/// Runs the binary over the same bytes written to a named file, so the two
/// results can be compared.
fn run_with_file(fx: &Fixture, name: &str, bytes: &[u8]) -> String {
    let input = fx.dir.join(name);
    std::fs::write(&input, bytes).unwrap();
    let out = fx.dir.join("file_counts.csv");
    let qc = fx.dir.join("file_qc.json");

    let output = Command::new(env!("CARGO_BIN_EXE_fastdna"))
        .args(args_for(&out, &qc, &input.to_string_lossy()))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("the binary must be runnable");
    assert!(
        output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read_to_string(&out).expect("the counts file must have been written")
}

/// The counts CSV a run produced, as rows, with a sanity check that it is
/// the schema every exporter writes.
fn rows(csv: &str) -> Vec<String> {
    let mut lines = csv.lines();
    assert_eq!(
        lines.next(),
        Some("kmer_u64,kmer_sequence,frequency"),
        "the exporter schema must be untouched"
    );
    let rows: Vec<String> = lines.filter(|l| !l.is_empty()).map(|l| l.to_string()).collect();
    assert!(!rows.is_empty(), "the run produced no counts at all");
    rows
}

/// A pipe must be indistinguishable from the same bytes in a file. Asserting
/// equality against a file run, rather than against hand-computed
/// frequencies, is what makes this a test of the plumbing rather than of my
/// arithmetic.
#[test]
fn plain_fastq_on_stdin_counts_exactly_as_the_same_file_does() {
    let fx = Fixture::new("plain");
    assert_eq!(
        rows(&run_with_stdin(&fx, FASTQ.as_bytes())),
        rows(&run_with_file(&fx, "reads.fastq", FASTQ.as_bytes()))
    );
}

/// A pipe has no filename, so gzip cannot be detected by extension. It must
/// be detected by the stream's own magic bytes instead -- otherwise
/// `... | gzip -c | fastdna -i -` reads binary as FASTQ and fails with a
/// baffling malformed-record error.
#[test]
fn gzipped_fastq_on_stdin_is_detected_by_its_magic_bytes() {
    let fx = Fixture::new("gz");
    let gz = gzip(FASTQ.as_bytes());
    assert_eq!(&gz[..2], &[0x1f, 0x8b], "the fixture must really be gzip");

    assert_eq!(
        rows(&run_with_stdin(&fx, &gz)),
        rows(&run_with_file(&fx, "reads.fastq", FASTQ.as_bytes())),
        "a gzipped pipe must count exactly as the same reads uncompressed"
    );
}

/// Format detection is by content, and a pipe is where that matters most:
/// there is not even a misleading extension to fall back on.
#[test]
fn fasta_on_stdin_is_detected_by_content() {
    let fx = Fixture::new("fasta");
    // The same two sequences as FASTQ, one of them line-wrapped.
    let fasta = b">r1\nACGTACGTTG\n>r2\nACGTA\nCGTTG\n";

    assert_eq!(
        rows(&run_with_stdin(&fx, fasta)),
        rows(&run_with_file(&fx, "reads.fastq", FASTQ.as_bytes())),
        "FASTA on a pipe must count as the equivalent FASTQ"
    );
}

/// A minimal, dependency-free gzip member: stored (uncompressed) deflate
/// blocks wrapped in a gzip header and trailer. Integration tests only see
/// the library and dev-dependencies, and `flate2` is neither, so the fixture
/// is built by hand rather than by adding a dependency for four bytes of
/// header.
fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];

    // Stored deflate blocks: [final?][LEN][NLEN][raw bytes], max 65535 each.
    let mut chunks = data.chunks(65_535).peekable();
    if data.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    }
    while let Some(chunk) = chunks.next() {
        let is_last = chunks.peek().is_none();
        out.push(if is_last { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }

    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}
