//! `--no-canonical`: counting each k-mer as it reads forward instead of
//! folding it with its reverse complement (KMC3's `-b`).
//!
//! The properties here are definitional rather than numbers this crate
//! produced. Canonicalising *merges* a k-mer with its reverse complement,
//! so turning it off can only split rows, never create or destroy
//! occurrences: distinct counts go up or stay equal, and the total is
//! unchanged. And a non-canonical table must answer a reverse-complement
//! query with a miss, because that key genuinely is not in it.
//!
//! `scripts/validation/kmc3_equivalence.py --no-canonical` is the outside
//! check: KMC3's `-b` on real reads agrees exactly.

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
        let dir = std::env::temp_dir().join(format!("fastdna_nocanon_{name}_{unique}"));
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

fn run(args: &[&str]) -> (bool, String) {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let out = Command::new(bin).args(args).output().expect("run fastdna");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' => 'T',
            'C' => 'G',
            'G' => 'C',
            'T' => 'A',
            other => other,
        })
        .collect()
}

/// A deterministic pseudo-random genome, and reads tiled across it. Not a
/// repeated motif: a motif shorter than k makes every window a function of
/// position, which would make the canonical/non-canonical difference an
/// artifact of the fixture rather than of the counting.
/// `both_strands` decides what the fixture can demonstrate, and the two
/// tests below need opposite things:
///
/// - `true` (what a real library gives, a fragment sequenced from an
///   arbitrary end) is required to show canonicalisation *merging*
///   anything. With one strand only there is nothing to merge and the two
///   distinct counts come out equal.
/// - `false` is required to show a non-canonical table *missing* a reverse
///   complement. With both strands, both orientations genuinely occur, and
///   a miss would be wrong.
fn write_reads(dir: &ScratchDir, name: &str, both_strands: bool) -> (PathBuf, String) {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let genome: String = (0..1_500)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            b"ACGT"[(state >> 33) as usize % 4] as char
        })
        .collect();

    let path = dir.path(name);
    let mut file = std::fs::File::create(&path).expect("create fastq");
    let mut start = 0;
    let mut i = 0;
    while start + 120 <= genome.len() {
        let read = &genome[start..start + 120];
        let read =
            if !both_strands || i % 2 == 0 { read.to_string() } else { reverse_complement(read) };
        writeln!(file, "@r{i}\n{read}\n+\n{}", "I".repeat(read.len())).expect("write");
        start += 29;
        i += 1;
    }
    (path, genome)
}

fn count(dir: &ScratchDir, reads: &Path, name: &str, extra: &[&str]) -> (PathBuf, u64, u64) {
    let out = dir.path(name);
    let qc = dir.path(&format!("{name}.qc.json"));
    let mut args: Vec<&str> = vec![
        "count",
        "--input", reads.to_str().unwrap(),
        "-k", "21",
        "-q", "0",
        "-m", "1",
        "-o", out.to_str().unwrap(),
        "--qc", qc.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    let (ok, text) = run(&args);
    assert!(ok, "count failed:\n{text}");

    let grab = |needle: &str| -> u64 {
        text.split(needle)
            .nth(1)
            .and_then(|rest| rest.trim_start().split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no {needle} in:\n{text}"))
    };
    (out, grab("Distinct k-mers:"), grab("Total k-mers Indexed:"))
}

#[test]
fn turning_off_canonicalisation_splits_rows_without_changing_occurrences() {
    let dir = ScratchDir::new("counts");
    // Both strands: without them canonicalisation has nothing to merge.
    let (reads, _) = write_reads(&dir, "in.fastq", true);

    let (_, canon_distinct, canon_total) = count(&dir, &reads, "canon.parquet", &[]);
    let (_, forward_distinct, forward_total) =
        count(&dir, &reads, "forward.parquet", &["--no-canonical"]);

    // Canonicalising merges a k-mer with its reverse complement. Turning it
    // off can only split rows -- never invent or lose an occurrence.
    assert_eq!(
        canon_total, forward_total,
        "the same reads contain the same number of k-mer occurrences either way"
    );
    assert!(
        forward_distinct > canon_distinct,
        "non-canonical must split at least one pair: {forward_distinct} vs {canon_distinct}"
    );
    // And it can at most double: every canonical row holds one or two
    // forward k-mers.
    assert!(
        forward_distinct <= canon_distinct * 2,
        "non-canonical cannot more than double the rows: {forward_distinct} vs {canon_distinct}"
    );
}

#[test]
fn a_non_canonical_table_answers_a_reverse_complement_query_with_a_miss() {
    let dir = ScratchDir::new("query");
    // One strand only: the reverse complement of a k-mer of this fixture
    // genuinely never occurs, which is what makes the miss below meaningful.
    let (reads, genome) = write_reads(&dir, "in.fastq", false);
    let kmer = &genome[40..61];
    assert_eq!(kmer.len(), 21);
    let rc = reverse_complement(kmer);
    assert_ne!(kmer, rc.as_str(), "the fixture k-mer must not be its own reverse complement");

    let (canon, _, _) = count(&dir, &reads, "canon.parquet", &[]);
    let (forward, _, _) = count(&dir, &reads, "forward.parquet", &["--no-canonical"]);

    // Canonical: both orientations are the same row.
    for q in [kmer, rc.as_str()] {
        let (ok, text) = run(&["query", "--table", canon.to_str().unwrap(), "--kmer", q]);
        assert!(ok, "{text}");
        assert!(text.contains("found, count ="), "canonical must find {q}:\n{text}");
    }

    // Non-canonical: only the orientation that actually occurs. This is the
    // case that would fail silently if `query` canonicalised the query --
    // it would look up a key the table cannot contain and report a miss
    // indistinguishable from a real absence.
    let (ok, text) = run(&["query", "--table", forward.to_str().unwrap(), "--kmer", kmer]);
    assert!(ok, "{text}");
    assert!(text.contains("found, count ="), "the forward k-mer must be present:\n{text}");
    assert!(text.contains("non-canonical"), "the banner should say which convention:\n{text}");

    let (ok, text) = run(&["query", "--table", forward.to_str().unwrap(), "--kmer", &rc]);
    assert!(ok, "{text}");
    assert!(text.contains("not found"), "the reverse complement must be absent:\n{text}");
}

#[test]
fn mixing_conventions_in_a_set_operation_is_refused_by_name() {
    let dir = ScratchDir::new("mixed");
    let (reads, _) = write_reads(&dir, "in.fastq", true);
    let (canon, _, _) = count(&dir, &reads, "canon.parquet", &[]);
    let (forward, _, _) = count(&dir, &reads, "forward.parquet", &["--no-canonical"]);

    let (ok, text) = run(&[
        "union",
        "--input", canon.to_str().unwrap(), forward.to_str().unwrap(),
        "--output", dir.path("u.parquet").to_str().unwrap(),
    ]);
    assert!(!ok, "mixing conventions must fail:\n{text}");
    assert!(text.contains("non-canonical"), "the error should name the mismatch:\n{text}");

    // Two of the same kind is fine, and the result keeps the convention.
    let out = dir.path("uu.parquet");
    let (ok, text) = run(&[
        "union",
        "--input", forward.to_str().unwrap(), forward.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
    ]);
    assert!(ok, "a same-convention union must succeed:\n{text}");

    let (ok, text) = run(&["query", "--table", out.to_str().unwrap(), "--kmer", "0"]);
    assert!(ok, "{text}");
    assert!(
        text.contains("non-canonical"),
        "the output must inherit the inputs' convention:\n{text}"
    );
}

#[test]
fn filtering_against_a_non_canonical_reference_uses_the_same_convention() {
    // Every read is drawn from the file the reference was counted from, so
    // at --min-fraction 1.0 keep-mode keeps all of them -- but only if the
    // reads are extracted the same way the reference was. A filter that
    // canonicalised against a non-canonical index would match almost
    // nothing.
    let dir = ScratchDir::new("filter");
    let (reads, _) = write_reads(&dir, "in.fastq", true);
    let (forward, _, _) = count(&dir, &reads, "forward.parquet", &["--no-canonical"]);

    let kept = dir.path("kept.fastq");
    let (ok, text) = run(&[
        "filter",
        "--table", forward.to_str().unwrap(),
        "--input", reads.to_str().unwrap(),
        "--mode", "keep",
        "--min-fraction", "1.0",
        "--output", kept.to_str().unwrap(),
    ]);
    assert!(ok, "filter failed:\n{text}");

    let n_in = std::fs::read_to_string(&reads).unwrap().lines().count() / 4;
    let n_out = std::fs::read_to_string(&kept).unwrap().lines().count() / 4;
    assert!(n_in > 0);
    assert_eq!(n_out, n_in, "every read came from the reference itself");
}
