//! End-to-end "count at k>32 -> query" through the real binary.
//!
//! Until now the wide engine was a dead end: `fastdna count --engine wide`
//! wrote a `kmer_bits` Parquet that nothing in this crate could read back,
//! and `KmerTable::open` rejected it by name. `src/wide_ktab.rs` closes
//! that for `query`, and these tests are what says so -- spawned against
//! the compiled binary (`CARGO_BIN_EXE_fastdna`) for the same reason
//! `tests/ktab_cli.rs` is: only the real binary proves `main.rs` routes a
//! wide table to the wide reader instead of the narrow one.
//!
//! The counts these assert against are not this crate's own opinion of
//! itself: each expected frequency is derived from the fixture sequence by
//! the test, and the narrow/wide agreement below is checked against the
//! `u64` engine that `scripts/validation/` measured exactly equal to KMC3.

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
        let dir = std::env::temp_dir().join(format!("fastdna_wide_query_{name}_{unique}"));
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

fn write_fastq(path: &Path, reads: &[String]) {
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

/// A deterministic sequence with no ambiguous bases, long enough that a
/// k=41 window has somewhere to slide.
fn fixture_sequence() -> String {
    // A fixed literal, not an RNG: the expected counts below are derived
    // from this string by the test itself, and a seeded RNG would only add
    // a layer between the assertion and the thing it asserts about.
    const UNIT: &str = "ACGTTGCAAGGCTTACCGATCGATTACAGCATCGGATCCAT";
    UNIT.repeat(6)
}

/// Counts `kmer`'s occurrences in `seq` the obvious way -- one string
/// compare per window, canonicalized by taking the smaller of the k-mer
/// and its reverse complement as text. Slow and completely independent of
/// `src/wide_kmer.rs`, which is the point: it is the outside answer the
/// engine is checked against.
fn count_in_sequence(seq: &str, kmer: &str) -> u32 {
    let k = kmer.len();
    let canonical = |s: &str| -> String {
        let rc: String = s
            .chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'C' => 'G',
                'G' => 'C',
                'T' => 'A',
                other => other,
            })
            .collect();
        if s <= rc.as_str() {
            s.to_string()
        } else {
            rc
        }
    };
    let target = canonical(kmer);
    let bytes = seq.as_bytes();
    (0..=bytes.len().saturating_sub(k))
        .filter(|&i| canonical(&seq[i..i + k]) == target)
        .count() as u32
}

/// `count` at `k`, into `dir/counts.parquet`. `--qc` is redirected into
/// the scratch directory on purpose: it defaults to `qc_report.json`
/// relative to the *current* directory, and a test that leaves one in the
/// crate root -- or fails outright when that directory is read-only, as it
/// is inside a container mounting the source read-only -- is testing the
/// harness, not the code.
fn count_to_parquet(dir: &ScratchDir, seq: &str, k: usize) -> PathBuf {
    let fastq = dir.path("sample.fastq");
    let reads: Vec<String> = (0..4).map(|_| seq.to_string()).collect();
    write_fastq(&fastq, &reads);

    let out = dir.path("counts.parquet");
    let qc = dir.path("qc.json");
    run_fastdna(&[
        "count",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        &k.to_string(),
        "-o",
        out.to_str().unwrap(),
        "--qc",
        qc.to_str().unwrap(),
    ]);
    out
}

fn query(table: &Path, kmer: &str) -> String {
    run_fastdna(&["query", "--table", table.to_str().unwrap(), "--kmer", kmer])
}

#[test]
fn a_wide_table_answers_a_point_query_with_the_recorded_frequency() {
    let dir = ScratchDir::new("hit");
    let seq = fixture_sequence();
    let table = count_to_parquet(&dir, &seq, 41);

    let kmer = &seq[10..51];
    assert_eq!(kmer.len(), 41);
    // Four identical reads, so the k-mer's frequency is four times what it
    // occurs in one copy of the sequence.
    let expected = 4 * count_in_sequence(&seq, kmer);
    assert!(expected > 0, "the fixture should contain this k-mer");

    let stdout = query(&table, kmer);
    assert!(stdout.contains("k:     41"), "expected k=41 in:\n{stdout}");
    assert!(
        stdout.contains(&format!("found, count = {expected}")),
        "expected count {expected} in:\n{stdout}"
    );
}

#[test]
fn a_wide_table_reports_an_absent_kmer_as_not_found() {
    let dir = ScratchDir::new("miss");
    let seq = fixture_sequence();
    let table = count_to_parquet(&dir, &seq, 41);

    // Not in the fixture: a homopolymer run the fixture unit never
    // produces, and its own reverse complement's canonical form is a
    // different homopolymer, so neither strand can match.
    let absent = "A".repeat(41);
    assert_eq!(count_in_sequence(&seq, &absent), 0);

    let stdout = query(&table, &absent);
    assert!(stdout.contains("not found in table"), "expected a miss in:\n{stdout}");
}

#[test]
fn the_query_answer_is_the_same_whichever_engine_wrote_the_table() {
    // The load-bearing check: at k=31 both engines can count the same
    // input, so the wide reader's answer must equal the narrow reader's on
    // the same k-mer. If they ever differ, one of the two paths -- the
    // encoding, the big-endian key, the statistics pruning -- is wrong.
    let dir = ScratchDir::new("agree");
    let seq = fixture_sequence();
    let fastq = dir.path("sample.fastq");
    let reads: Vec<String> = (0..4).map(|_| seq.clone()).collect();
    write_fastq(&fastq, &reads);

    let mut answers = Vec::new();
    for engine in ["narrow", "wide"] {
        let out = dir.path(&format!("counts_{engine}.parquet"));
        let qc = dir.path(&format!("qc_{engine}.json"));
        run_fastdna(&[
            "count",
            "--input",
            fastq.to_str().unwrap(),
            "-k",
            "31",
            "--engine",
            engine,
            "-o",
            out.to_str().unwrap(),
            "--qc",
            qc.to_str().unwrap(),
        ]);
        answers.push(query(&out, &seq[5..36]));
    }

    let narrow = &answers[0];
    let wide = &answers[1];
    let line = |s: &str| {
        s.lines()
            .find(|l| l.contains("found"))
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(line(narrow), line(wide), "narrow:\n{narrow}\nwide:\n{wide}");
    assert!(line(narrow).contains("count = "), "expected a hit, got:\n{narrow}");
}

#[test]
fn a_wrong_length_kmer_is_rejected_rather_than_reported_absent() {
    // "You typed it wrong" and "it is not in the table" are different
    // answers; a query too short for the table's k must not come back as
    // an ordinary miss.
    let dir = ScratchDir::new("wrong_length");
    let table = count_to_parquet(&dir, &fixture_sequence(), 41);

    let stderr = run_fastdna_expect_failure(&[
        "query",
        "--table",
        table.to_str().unwrap(),
        "--kmer",
        "ACGTACGTACGT",
    ]);
    assert!(stderr.contains("41"), "the error should name the table's k:\n{stderr}");
}

#[test]
fn an_ambiguous_base_in_a_wide_query_is_rejected() {
    let dir = ScratchDir::new("ambiguous");
    let table = count_to_parquet(&dir, &fixture_sequence(), 41);

    let mut kmer: Vec<u8> = fixture_sequence().as_bytes()[0..41].to_vec();
    kmer[20] = b'N';
    let kmer = String::from_utf8(kmer).unwrap();

    let stderr =
        run_fastdna_expect_failure(&["query", "--table", table.to_str().unwrap(), "--kmer", &kmer]);
    assert!(
        stderr.contains("nucleotide"),
        "the error should say the base is ambiguous:\n{stderr}"
    );
}

#[test]
fn a_narrow_table_still_goes_to_the_narrow_reader() {
    // The routing has to keep working in both directions: the wide reader
    // rejects a `kmer_u64` table, so a k<=32 table reaching it would fail
    // rather than answer.
    let dir = ScratchDir::new("narrow_still_works");
    let seq = fixture_sequence();
    let table = count_to_parquet(&dir, &seq, 21);

    let kmer = &seq[3..24];
    let expected = 4 * count_in_sequence(&seq, kmer);
    let stdout = query(&table, kmer);
    assert!(stdout.contains("k:     21"), "expected k=21 in:\n{stdout}");
    assert!(
        stdout.contains(&format!("found, count = {expected}")),
        "expected count {expected} in:\n{stdout}"
    );
}

#[test]
fn a_file_that_is_not_a_kmer_table_is_rejected_by_the_router() {
    // `table_key` reads the footer before either reader is chosen, so a
    // file with no `fastdna.sorted_by` fails there with one clear message
    // rather than with whichever of the two readers happened to be tried.
    let dir = ScratchDir::new("not_a_table");
    let bogus = dir.path("notes.parquet");
    std::fs::write(&bogus, b"this is not parquet at all").unwrap();

    let stderr =
        run_fastdna_expect_failure(&["query", "--table", bogus.to_str().unwrap(), "--kmer", "ACGT"]);
    assert!(!stderr.is_empty(), "a failure should explain itself");
}
