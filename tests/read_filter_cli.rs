//! End-to-end "count -> filter" test for `docs/feature-gap-analysis.md`'s S4
//! (`fastdna filter`, `src/read_filter.rs`): proves the real compiled binary
//! wires `cli::FilterArgs` into `read_filter::run_filter`, and that both
//! `--mode keep`/`--mode discard` and gzip output actually work end to end --
//! the same `CARGO_BIN_EXE_fastdna` convention `tests/setops_cli.rs` and
//! `tests/ktab_cli.rs` already use, for the same reason: only the real
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
        let dir = std::env::temp_dir().join(format!("fastdna_read_filter_cli_{name}_{unique}"));
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

fn write_fastq(path: &std::path::Path, reads: &[(&str, &str)]) {
    let mut f = std::fs::File::create(path).expect("create fastq fixture");
    for (id, seq) in reads {
        writeln!(f, "@{id}\n{seq}\n+\n{}", "I".repeat(seq.len())).expect("write fixture");
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

fn read_fastq_ids(path: &std::path::Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect("read filtered output");
    text.lines().filter(|l| l.starts_with('@')).map(|l| l[1..].to_string()).collect()
}

fn read_gzipped_fastq_ids(path: &std::path::Path) -> Vec<String> {
    use flate2::read::MultiGzDecoder;
    use std::io::Read;

    let compressed = std::fs::read(path).expect("read gzip output");
    let mut decoder = MultiGzDecoder::new(compressed.as_slice());
    let mut text = String::new();
    decoder.read_to_string(&mut text).expect("decompress gzip output");
    text.lines().filter(|l| l.starts_with('@')).map(|l| l[1..].to_string()).collect()
}

/// Builds a reference table at k=4 from a genome-like sequence entirely made
/// of "GGGG" repeats, then a sample with one read that is a full match to
/// that reference and one read that shares nothing with it.
fn build_reference_and_sample(scratch: &ScratchDir) -> (PathBuf, PathBuf) {
    let reference_fastq = scratch.path("reference.fastq");
    write_fastq(&reference_fastq, &[("ref1", "GGGGGGGGGGGG")]);
    let reference_table = scratch.path("reference.parquet");
    run_fastdna(&[
        "count",
        "--input",
        reference_fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        reference_table.to_str().unwrap(),
    ]);

    let sample_fastq = scratch.path("sample.fastq");
    write_fastq(
        &sample_fastq,
        &[
            ("matching", "GGGGGGGGGGGG"),
            ("non_matching", "ACATACATACAT"),
        ],
    );

    (reference_table, sample_fastq)
}

#[test]
fn discard_mode_removes_the_matching_read_and_keeps_the_rest() {
    let scratch = ScratchDir::new("discard");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let out = scratch.path("filtered.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out.to_str().unwrap(),
    ]);

    assert!(out.exists());
    let ids = read_fastq_ids(&out);
    assert_eq!(ids, vec!["non_matching"], "the matching read must be removed under --mode discard");
}

#[test]
fn keep_mode_writes_only_the_matching_read() {
    let scratch = ScratchDir::new("keep");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let out = scratch.path("filtered.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "--min-fraction",
        "0.5",
        "-o",
        out.to_str().unwrap(),
    ]);

    let ids = read_fastq_ids(&out);
    assert_eq!(ids, vec!["matching"], "only the matching read should survive --mode keep");
}

#[test]
fn gzip_output_is_a_real_gzip_stream_readable_back() {
    let scratch = ScratchDir::new("gzip");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let out = scratch.path("filtered.fastq.gz");
    run_fastdna(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out.to_str().unwrap(),
    ]);

    let ids = read_gzipped_fastq_ids(&out);
    assert_eq!(ids, vec!["non_matching"]);
}

#[test]
fn fasta_input_is_written_as_fastq_output() {
    let scratch = ScratchDir::new("fasta_input");
    let (reference_table, _sample_fastq) = build_reference_and_sample(&scratch);

    let fasta_input = scratch.path("sample.fasta");
    std::fs::write(&fasta_input, b">contig_a\nACATACATACAT\n>contig_b\nGGGGGGGGGGGG\n")
        .expect("write fasta fixture");

    let out = scratch.path("filtered.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        fasta_input.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out.to_str().unwrap(),
    ]);

    let text = std::fs::read_to_string(&out).unwrap();
    // FASTA input must come out as valid FASTQ: four lines per kept record,
    // a synthetic quality line of the same length as the sequence.
    assert!(text.contains("@contig_a\nACATACATACAT\n+\nIIIIIIIIIIII\n"));
    assert!(!text.contains("contig_b"), "the matching contig must be discarded");
}

#[test]
fn filter_requires_a_mode() {
    let scratch = ScratchDir::new("no_mode");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);
    let out = scratch.path("filtered.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

#[test]
fn filter_rejects_an_out_of_range_min_fraction() {
    let scratch = ScratchDir::new("bad_fraction");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);
    let out = scratch.path("filtered.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "--min-fraction",
        "1.5",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

// ---------------------------------------------------------------------
// Paired-end (--input2/--output2) filtering
// ---------------------------------------------------------------------

fn build_reference_and_paired_sample(scratch: &ScratchDir) -> (PathBuf, PathBuf, PathBuf) {
    let reference_fastq = scratch.path("reference.fastq");
    write_fastq(&reference_fastq, &[("ref1", "GGGGGGGGGGGG")]);
    let reference_table = scratch.path("reference.parquet");
    run_fastdna(&[
        "count",
        "--input",
        reference_fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        reference_table.to_str().unwrap(),
    ]);

    // pair "a": only R1 matches the reference. pair "b": only R2 matches.
    // pair "c": neither mate matches.
    let r1 = scratch.path("r1.fastq");
    write_fastq(
        &r1,
        &[
            ("a/1", "GGGGGGGGGGGG"),
            ("b/1", "ACATACATACAT"),
            ("c/1", "ACATACATACAT"),
        ],
    );
    let r2 = scratch.path("r2.fastq");
    write_fastq(
        &r2,
        &[
            ("a/2", "ACATACATACAT"),
            ("b/2", "GGGGGGGGGGGG"),
            ("c/2", "ACATACATACAT"),
        ],
    );

    (reference_table, r1, r2)
}

#[test]
fn paired_keep_mode_writes_a_pair_if_either_mate_matches() {
    let scratch = ScratchDir::new("paired_keep");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);

    let out1 = scratch.path("out1.fastq");
    let out2 = scratch.path("out2.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "--min-fraction",
        "0.5",
        "-o",
        out1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);

    let ids1 = read_fastq_ids(&out1);
    let ids2 = read_fastq_ids(&out2);
    assert_eq!(ids1, vec!["a/1", "b/1"], "pairs a and b each have a matching mate");
    assert_eq!(ids2, vec!["a/2", "b/2"], "R2 output must stay synchronized with R1's kept pairs");
}

#[test]
fn paired_discard_mode_writes_a_pair_only_if_neither_mate_matches() {
    let scratch = ScratchDir::new("paired_discard");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);

    let out1 = scratch.path("out1.fastq");
    let out2 = scratch.path("out2.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);

    let ids1 = read_fastq_ids(&out1);
    let ids2 = read_fastq_ids(&out2);
    assert_eq!(ids1, vec!["c/1"], "only pair c has no matching mate on either side");
    assert_eq!(ids2, vec!["c/2"]);
}

#[test]
fn paired_gzip_outputs_are_real_gzip_streams_readable_back() {
    let scratch = ScratchDir::new("paired_gzip");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);

    let out1 = scratch.path("out1.fastq.gz");
    let out2 = scratch.path("out2.fastq.gz");
    run_fastdna(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);

    assert_eq!(read_gzipped_fastq_ids(&out1), vec!["c/1"]);
    assert_eq!(read_gzipped_fastq_ids(&out2), vec!["c/2"]);
}

#[test]
fn input2_without_output2_is_rejected() {
    let scratch = ScratchDir::new("input2_no_output2");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);
    let out1 = scratch.path("out1.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "-o",
        out1.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
    assert!(!out1.exists(), "no output should be written before the argument-shape error");
}

#[test]
fn output2_without_input2_is_rejected() {
    let scratch = ScratchDir::new("output2_no_input2");
    let (reference_table, r1, _r2) = build_reference_and_paired_sample(&scratch);
    let out1 = scratch.path("out1.fastq");
    let out2 = scratch.path("out2.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "-o",
        out1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

#[test]
fn output_and_output2_pointing_at_the_same_file_are_rejected() {
    let scratch = ScratchDir::new("paired_same_output");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);
    let shared = scratch.path("shared.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "-o",
        shared.to_str().unwrap(),
        "--output2",
        shared.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
    assert!(!shared.exists(), "neither mate should be written when both outputs collide");
}

#[test]
fn paired_output_must_not_overwrite_an_r1_or_r2_input_file() {
    let scratch = ScratchDir::new("paired_overwrite_input");
    let (reference_table, r1, r2) = build_reference_and_paired_sample(&scratch);
    let original_r1 = std::fs::read(&r1).unwrap();
    let out2 = scratch.path("out2.fastq");

    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        // --output points back at the R1 input file itself.
        "-o",
        r1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
    assert_eq!(std::fs::read(&r1).unwrap(), original_r1, "the R1 input file must survive untouched");
}

#[test]
fn paired_desync_between_r1_and_r2_is_reported_not_silently_truncated() {
    let scratch = ScratchDir::new("paired_desync");
    let reference_fastq = scratch.path("reference.fastq");
    write_fastq(&reference_fastq, &[("ref1", "GGGGGGGGGGGG")]);
    let reference_table = scratch.path("reference.parquet");
    run_fastdna(&[
        "count",
        "--input",
        reference_fastq.to_str().unwrap(),
        "-k",
        "4",
        "-o",
        reference_table.to_str().unwrap(),
    ]);

    // R1 has two reads, R2 only one.
    let r1 = scratch.path("r1.fastq");
    write_fastq(&r1, &[("a/1", "ACATACATACAT"), ("b/1", "ACATACATACAT")]);
    let r2 = scratch.path("r2.fastq");
    write_fastq(&r2, &[("a/2", "ACATACATACAT")]);

    let out1 = scratch.path("out1.fastq");
    let out2 = scratch.path("out2.fastq");
    let stderr = run_fastdna_expect_failure(&[
        "filter",
        "--input",
        r1.to_str().unwrap(),
        "--input2",
        r2.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "keep",
        "-o",
        out1.to_str().unwrap(),
        "--output2",
        out2.to_str().unwrap(),
    ]);
    assert!(!stderr.is_empty());
}

#[test]
fn filter_output_composes_as_a_valid_count_input() {
    // Not a strict requirement of S4, but a useful smoke test: filtered
    // FASTQ output is itself valid FASTQ that `fastdna count` can consume.
    let scratch = ScratchDir::new("compose");
    let (reference_table, sample_fastq) = build_reference_and_sample(&scratch);

    let out = scratch.path("filtered.fastq");
    run_fastdna(&[
        "filter",
        "--input",
        sample_fastq.to_str().unwrap(),
        "--table",
        reference_table.to_str().unwrap(),
        "--mode",
        "discard",
        "--min-fraction",
        "0.5",
        "-o",
        out.to_str().unwrap(),
    ]);

    let recounted = scratch.path("recounted.parquet");
    run_fastdna(&["count", "--input", out.to_str().unwrap(), "-k", "4", "-o", recounted.to_str().unwrap()]);
    assert!(recounted.exists());
}
