//! The four new `fastdna` subcommands (`sketch`, `dist`, `card`, `peek`)
//! plus the explicit `count` subcommand and its equivalence with giving no
//! subcommand at all (Q4, `docs/feature-gap-analysis.md`).
//!
//! Split the same way the rest of this crate's CLI tests are split:
//! argument-*parsing* tests run clap directly (`Cli::parse_from`/
//! `try_parse_from`, matching `tests/cli_args.rs`'s own doc comment on why
//! -- fast, no extra dev-dependencies), and a smaller set of end-to-end
//! tests spawn the real compiled binary via `CARGO_BIN_EXE_fastdna`
//! (matching `tests/cli_max_count_wiring.rs`) for the parts that only the
//! binary itself proves: that `main.rs` actually wires each subcommand's
//! parsed `Args` into `sketch.rs`/`hll.rs`/`preview.rs` and produces real
//! output, not just that clap accepts the flags.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use clap::Parser;
use fastdna_core::cli::{Cli, CliDistMetric, Command as CliCommand};

// ---------------------------------------------------------------------
// Parsing: dispatch and defaults
// ---------------------------------------------------------------------

#[test]
fn no_subcommand_word_still_dispatches_to_count() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq"]);
    assert!(cli.command.is_none(), "no subcommand word must leave `command` empty");
    assert_eq!(
        cli.count.input.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
        vec!["sample.fastq".to_string()],
        "the implicit count path must still populate `count` exactly as before subcommands existed"
    );
}

#[test]
fn explicit_count_subcommand_parses_the_same_flags_as_the_implicit_default() {
    let explicit = Cli::parse_from(["fastdna", "count", "--input", "sample.fastq", "-k", "25"]);
    let implicit = Cli::parse_from(["fastdna", "--input", "sample.fastq", "-k", "25"]);

    let CliCommand::Count(explicit_args) = explicit.command.expect("count subcommand must be present") else {
        panic!("expected Command::Count");
    };

    assert_eq!(explicit_args.input, implicit.count.input);
    assert_eq!(explicit_args.kmer_size, implicit.count.kmer_size);
    assert_eq!(explicit_args.kmer_size, 25);
}

#[test]
fn sketch_subcommand_parses_with_its_own_defaults() {
    let cli = Cli::parse_from(["fastdna", "sketch", "--input", "s.fastq", "-o", "s.json"]);
    match cli.command.expect("sketch subcommand must be present") {
        CliCommand::Sketch(args) => {
            assert_eq!(args.input, PathBuf::from("s.fastq"));
            assert_eq!(args.output, PathBuf::from("s.json"));
            assert_eq!(args.kmer_size, 21, "sketch defaults k to 21, matching fastdna.sketch()'s Python default");
            assert_eq!(args.sketch_size, 1000, "matching fastdna.sketch()'s Python default");
        }
        other => panic!("expected Command::Sketch, got {other:?}"),
    }
}

#[test]
fn sketch_subcommand_accepts_explicit_kmer_size_and_sketch_size() {
    let cli = Cli::parse_from([
        "fastdna", "sketch", "--input", "s.fastq", "-k", "17", "--sketch-size", "200", "-o", "s.json",
    ]);
    match cli.command.expect("sketch subcommand must be present") {
        CliCommand::Sketch(args) => {
            assert_eq!(args.kmer_size, 17);
            assert_eq!(args.sketch_size, 200);
        }
        other => panic!("expected Command::Sketch, got {other:?}"),
    }
}

#[test]
fn sketch_without_an_output_path_is_rejected() {
    assert!(
        Cli::try_parse_from(["fastdna", "sketch", "--input", "s.fastq"]).is_err(),
        "-o is required for sketch: without it there is nowhere to save the fingerprint"
    );
}

#[test]
fn sketch_without_an_input_path_is_rejected() {
    assert!(
        Cli::try_parse_from(["fastdna", "sketch", "-o", "s.json"]).is_err(),
        "--input is required for sketch"
    );
}

#[test]
fn dist_defaults_to_the_jaccard_metric() {
    let cli = Cli::parse_from(["fastdna", "dist", "--input", "a.json", "b.json"]);
    match cli.command.expect("dist subcommand must be present") {
        CliCommand::Dist(args) => {
            assert_eq!(args.input.len(), 2);
            assert_eq!(args.metric, CliDistMetric::Jaccard);
            assert!(args.output.is_none(), "no --output means the table prints to stdout");
        }
        other => panic!("expected Command::Dist, got {other:?}"),
    }
}

#[test]
fn dist_metric_containment_and_mash_are_selectable() {
    let containment = Cli::parse_from(["fastdna", "dist", "--input", "a.json", "b.json", "--metric", "containment"]);
    let mash = Cli::parse_from(["fastdna", "dist", "--input", "a.json", "b.json", "--metric", "mash"]);

    match containment.command.expect("dist subcommand must be present") {
        CliCommand::Dist(args) => assert_eq!(args.metric, CliDistMetric::Containment),
        other => panic!("expected Command::Dist, got {other:?}"),
    }
    match mash.command.expect("dist subcommand must be present") {
        CliCommand::Dist(args) => assert_eq!(args.metric, CliDistMetric::MashDistance),
        other => panic!("expected Command::Dist, got {other:?}"),
    }
}

#[test]
fn dist_rejects_an_unknown_metric() {
    assert!(
        Cli::try_parse_from(["fastdna", "dist", "--input", "a.json", "b.json", "--metric", "cosine"])
            .is_err()
    );
}

#[test]
fn dist_requires_at_least_two_inputs() {
    assert!(
        Cli::try_parse_from(["fastdna", "dist", "--input", "a.json"]).is_err(),
        "a pairwise comparison of one thing is not a comparison"
    );
    assert!(
        Cli::try_parse_from(["fastdna", "dist", "--input", "a.json", "b.json", "c.json"]).is_ok(),
        "more than two inputs must still be accepted"
    );
}

#[test]
fn card_subcommand_parses_with_its_own_defaults() {
    let cli = Cli::parse_from(["fastdna", "card", "--input", "s.fastq"]);
    match cli.command.expect("card subcommand must be present") {
        CliCommand::Card(args) => {
            assert_eq!(args.input, PathBuf::from("s.fastq"));
            assert_eq!(args.kmer_size, 31, "card defaults k to 31, matching count's default");
            assert_eq!(args.precision, 14, "matching fastdna.estimate_cardinality()'s Python default");
        }
        other => panic!("expected Command::Card, got {other:?}"),
    }
}

#[test]
fn peek_subcommand_parses_with_its_own_defaults() {
    let cli = Cli::parse_from(["fastdna", "peek", "--input", "s.fastq"]);
    match cli.command.expect("peek subcommand must be present") {
        CliCommand::Peek(args) => {
            assert_eq!(args.input, PathBuf::from("s.fastq"));
            assert_eq!(args.n_reads, 10_000, "matching fastdna.peek()'s Python default");
        }
        other => panic!("expected Command::Peek, got {other:?}"),
    }
}

#[test]
fn peek_accepts_an_explicit_n_reads() {
    let cli = Cli::parse_from(["fastdna", "peek", "--input", "s.fastq", "--n-reads", "5"]);
    match cli.command.expect("peek subcommand must be present") {
        CliCommand::Peek(args) => assert_eq!(args.n_reads, 5),
        other => panic!("expected Command::Peek, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// End-to-end: real files through the compiled binary
// ---------------------------------------------------------------------

/// Mirrors `tests/cli_max_count_wiring.rs::ScratchDir`: a unique-per-run
/// scratch directory so concurrent test runs never collide, cleaned up on
/// drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_cli_subcommands_{name}_{unique}"));
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

/// Runs the fastdna binary, returning stdout on success and panicking with
/// both streams on failure -- same contract as `cli_max_count_wiring.rs::
/// run_fastdna`, but returning stdout since several of these tests assert
/// on the console summary rather than a written file.
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

#[test]
fn sketch_then_dist_round_trip_reports_perfect_self_similarity() {
    let scratch = ScratchDir::new("sketch_dist");
    let fastq = scratch.path("sample.fastq");
    // Long enough, and varied enough, to give more than a handful of
    // distinct 4-mers -- a self-comparison of the resulting sketch should
    // land at exactly 1.0 for every symmetric metric regardless of how
    // many k-mers that turns out to be.
    write_fastq(&fastq, &["ACGTACGTTGCA", "GGGGCCCCAATT", "TTGCAACGTTGA"]);

    let sketch_path = scratch.path("sample.sketch.json");
    run_fastdna(&[
        "sketch",
        "--input",
        fastq.to_str().unwrap(),
        "-k",
        "4",
        "--sketch-size",
        "100",
        "-o",
        sketch_path.to_str().unwrap(),
    ]);
    assert!(sketch_path.exists(), "fastdna sketch must write the saved sketch file");
    let saved = std::fs::read_to_string(&sketch_path).expect("read saved sketch");
    // `serde_json::to_writer_pretty` (what `GenomeSketch::save` uses) puts a
    // space after the colon.
    assert!(saved.contains("\"k\": 4"), "saved sketch must record k=4: {saved}");

    // Comparing the saved sketch against itself: jaccard/mash must both
    // report perfect similarity, proving `fastdna dist` actually loads the
    // `.json` sketch (rather than silently re-sketching a FASTQ named
    // ".json") and wires it into `GenomeSketch::jaccard`/`mash_distance`.
    let jaccard_out = run_fastdna(&[
        "dist",
        "--input",
        sketch_path.to_str().unwrap(),
        sketch_path.to_str().unwrap(),
        "--metric",
        "jaccard",
    ]);
    let jaccard_lines: Vec<&str> = jaccard_out.lines().collect();
    assert_eq!(jaccard_lines[0], "sample_a,sample_b,jaccard");
    assert!(
        jaccard_lines.get(1).is_some_and(|l| l.ends_with(",1")),
        "self-comparison must report jaccard 1.0:\n{jaccard_out}"
    );

    let mash_out = run_fastdna(&[
        "dist",
        "--input",
        sketch_path.to_str().unwrap(),
        sketch_path.to_str().unwrap(),
        "--metric",
        "mash",
    ]);
    let mash_lines: Vec<&str> = mash_out.lines().collect();
    assert_eq!(mash_lines[0], "sample_a,sample_b,mash_distance");
    assert!(
        mash_lines.get(1).is_some_and(|l| l.ends_with(",0")),
        "self-comparison must report mash_distance 0.0:\n{mash_out}"
    );
}

#[test]
fn dist_containment_reports_both_directions_of_a_pair() {
    let scratch = ScratchDir::new("dist_containment");
    // `small` is 40 identical bases of a 20-base repeat -- every one of its
    // 4-mers also occurs in `large`, which repeats that same 20-base motif
    // plus extra unrelated bases, so small-in-large containment must be
    // 1.0 while large-in-small is strictly less than 1.0 -- the asymmetry
    // `--metric containment` exists to show.
    let small = scratch.path("small.fastq");
    let large = scratch.path("large.fastq");
    write_fastq(&small, &["ACGTACGTACGTACGTACGT"]);
    write_fastq(&large, &["ACGTACGTACGTACGTACGTTTTTTTTTTGGGGGGGGGGCCCCCCCCCC"]);

    let out = run_fastdna(&[
        "dist",
        "--input",
        small.to_str().unwrap(),
        large.to_str().unwrap(),
        "--metric",
        "containment",
        "-k",
        "4",
        "--sketch-size",
        "1000",
    ]);

    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "sample_a,sample_b,containment");
    // Both orderings must appear -- containment is asymmetric, unlike
    // jaccard/mash, which only ever print one row per pair.
    assert_eq!(lines.len(), 3, "two directed rows expected for a 2-input containment run:\n{out}");
    let small_in_large = lines.iter().find(|l| l.starts_with(&format!("{},{}", small.display(), large.display())));
    assert!(small_in_large.is_some(), "expected a small-in-large row:\n{out}");
    assert!(
        small_in_large.unwrap().ends_with(",1"),
        "every 4-mer of `small` occurs in `large`, so containment(small, large) must be 1.0: {}",
        small_in_large.unwrap()
    );
}

#[test]
fn card_reports_a_plausible_distinct_kmer_estimate() {
    let scratch = ScratchDir::new("card");
    let fastq = scratch.path("sample.fastq");
    // 50 reads of the same short sequence: only a handful of distinct
    // canonical 4-mers regardless of the repeat count, small enough that
    // HyperLogLog's linear-counting branch applies and the estimate stays
    // close to the true (small) value rather than blowing up with the read
    // count.
    write_fastq(&fastq, &vec!["ACGTACGTAC"; 50]);

    let out = run_fastdna(&["card", "--input", fastq.to_str().unwrap(), "-k", "4"]);
    let estimate: f64 = out
        .lines()
        .find_map(|l| l.strip_prefix("Estimated distinct k-mers: "))
        .unwrap_or_else(|| panic!("missing estimate line:\n{out}"))
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("estimate line was not a number:\n{out}"));
    assert!(
        (1.0..20.0).contains(&estimate),
        "50 reads of one short repeated sequence must estimate a small distinct-kmer count, got {estimate}:\n{out}"
    );
}

#[test]
fn peek_reports_read_geometry_and_a_suggested_k() {
    let scratch = ScratchDir::new("peek");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["ACGTACGTAC", "GGGGCCCCAA", "TTGCAACGTT"]);

    let out = run_fastdna(&["peek", "--input", fastq.to_str().unwrap()]);
    assert!(out.contains("Reads sampled:                 3"), "missing/wrong read count:\n{out}");
    assert!(out.contains("Read length (min/median/max):  10/10/10"), "missing/wrong length stats:\n{out}");
    assert!(out.contains("Suggested k:"), "missing suggested-k line:\n{out}");
}

#[test]
fn explicit_count_subcommand_produces_identical_output_to_the_implicit_default() {
    let scratch = ScratchDir::new("count_equivalence");
    let fastq = scratch.path("sample.fastq");
    write_fastq(&fastq, &["AAAA", "AAAA", "CCCC"]);

    let implicit_out = scratch.path("implicit.csv");
    run_fastdna(&[
        "--input",
        fastq.to_str().unwrap(),
        "--output",
        implicit_out.to_str().unwrap(),
        "-k",
        "4",
        "--qc",
        scratch.path("implicit_qc.json").to_str().unwrap(),
    ]);

    let explicit_out = scratch.path("explicit.csv");
    run_fastdna(&[
        "count",
        "--input",
        fastq.to_str().unwrap(),
        "--output",
        explicit_out.to_str().unwrap(),
        "-k",
        "4",
        "--qc",
        scratch.path("explicit_qc.json").to_str().unwrap(),
    ]);

    let implicit_csv = std::fs::read_to_string(&implicit_out).expect("read implicit output");
    let explicit_csv = std::fs::read_to_string(&explicit_out).expect("read explicit output");
    assert_eq!(
        implicit_csv, explicit_csv,
        "`fastdna count ...` and `fastdna ...` (no subcommand) must produce byte-identical output"
    );
}
