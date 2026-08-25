//! Metagenomic classification, end to end through real files.
//!
//! The algorithm itself -- LCA assignment, root-to-leaf scoring, Kraken 2's
//! confidence arithmetic -- is pinned by the unit tests in
//! `src/metagenomics.rs`, against hand-counted k-mers. What this file pins
//! down is everything those cannot reach: that a database built from a
//! FASTA on disk agrees with one built in memory, that it survives a
//! save/load round trip byte for byte, that the reader's own FASTA/FASTQ/
//! gzip handling reaches the classifier unchanged, and that a failure in
//! any of those names the file it happened in.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::{Path, PathBuf};

use fastdna_core::error::FastDnaError;
use fastdna_core::metagenomics::{KmerDatabase, UNCLASSIFIED_TAX_ID};

/// The toy tree: root 1 > genus 10 > species 100 and 200.
const TAXONOMY: &str = "\
tax_id\tparent_tax_id\trank\tname\tsequence_ids
1\t1\tno rank\troot\t
10\t1\tgenus\tToyella\t
100\t10\tspecies\tToyella alpha\tspecies_a
200\t10\tspecies\tToyella beta\tspecies_b
";

const K: usize = 11;
const SHARED: &str = "GATTACAGATTACAGGCC";
const ONLY_A: &str = "TTGCACCGTAAGCTATCG";
const ONLY_B: &str = "ACGCGTTAACCGGATCAT";

fn reference() -> String {
    format!(">species_a Toyella alpha chromosome\n{ONLY_A}{SHARED}\n>species_b\n{ONLY_B}{SHARED}\n")
}

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_metagenomics_it_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn write_gz(&self, name: &str, contents: &str) -> PathBuf {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let path = self.dir.join(name);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(contents.as_bytes()).unwrap();
        std::fs::write(&path, encoder.finish().unwrap()).unwrap();
        path
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// The reference and taxonomy every test here builds from.
    fn standard_inputs(&self) -> (PathBuf, PathBuf) {
        (self.write("reference.fasta", &reference()), self.write("taxonomy.tsv", TAXONOMY))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fastq(reads: &[(&str, &str)]) -> String {
    reads
        .iter()
        .map(|(id, seq)| format!("@{id}\n{seq}\n+\n{}\n", "I".repeat(seq.len())))
        .collect()
}

fn tax_ids(db: &KmerDatabase, reads: &Path, threshold: f64) -> Vec<u32> {
    db.classify_path(reads, threshold).expect("valid reads").iter().map(|r| r.tax_id).collect()
}

#[test]
fn a_database_built_from_files_classifies_reads_to_the_expected_taxa() {
    let fx = Fixture::new("end_to_end");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).expect("reference and taxonomy agree");

    let reads = fx.write(
        "reads.fastq",
        &fastq(&[
            ("only_a", ONLY_A),
            ("only_b", ONLY_B),
            ("shared", SHARED),
            ("chimera", &format!("{ONLY_A}{ONLY_B}")),
            ("nothing", "CTCTAGGACTGACTCTAGGACTGA"),
        ]),
    );

    assert_eq!(
        tax_ids(&db, &reads, 0.0),
        vec![100, 200, 10, 10, UNCLASSIFIED_TAX_ID],
        "species-specific reads to their species, shared and chimeric reads to the genus"
    );
}

/// FASTA and FASTQ reach the classifier through the same reader, so the
/// same reads in either container must classify identically -- the same
/// property `tests/fasta_input.rs` pins for counting.
#[test]
fn fasta_and_fastq_reads_classify_identically() {
    let fx = Fixture::new("fasta_reads");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    let as_fastq = fx.write("reads.fastq", &fastq(&[("r1", ONLY_A), ("r2", SHARED)]));
    let as_fasta = fx.write("reads.fasta", &format!(">r1\n{ONLY_A}\n>r2\n{SHARED}\n"));

    assert_eq!(tax_ids(&db, &as_fastq, 0.0), tax_ids(&db, &as_fasta, 0.0));
    assert_eq!(tax_ids(&db, &as_fasta, 0.0), vec![100, 10]);
}

#[test]
fn gzipped_reads_and_a_gzipped_reference_both_work() {
    let fx = Fixture::new("gzipped");
    let reference = fx.write_gz("reference.fasta.gz", &reference());
    let taxonomy = fx.write("taxonomy.tsv", TAXONOMY);
    let db = KmerDatabase::build(&reference, &taxonomy, K).expect("a gzipped reference builds");

    let reads = fx.write_gz("reads.fastq.gz", &fastq(&[("r1", ONLY_A), ("r2", ONLY_B)]));
    assert_eq!(tax_ids(&db, &reads, 0.0), vec![100, 200]);
}

#[test]
fn a_saved_database_classifies_exactly_as_the_one_it_was_saved_from() {
    let fx = Fixture::new("round_trip");
    let (reference, taxonomy) = fx.standard_inputs();
    let built = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    let db_path = fx.path("toy.fdb");
    built.save(&db_path).unwrap();
    let loaded = KmerDatabase::load(&db_path).unwrap();

    let reads = fx.write(
        "reads.fastq",
        &fastq(&[("a", ONLY_A), ("s", SHARED), ("c", &format!("{ONLY_A}{ONLY_B}"))]),
    );

    let from_built = built.classify_path(&reads, 0.0).unwrap();
    let from_loaded = loaded.classify_path(&reads, 0.0).unwrap();
    assert_eq!(from_built, from_loaded, "a round trip must not change a single call");
}

#[test]
fn the_abundance_report_sums_to_one_over_every_read() {
    let fx = Fixture::new("abundance");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    let reads = fx.write(
        "reads.fastq",
        &fastq(&[
            ("a1", ONLY_A),
            ("a2", ONLY_A),
            ("b1", ONLY_B),
            ("none", "CTCTAGGACTGACTCTAGGACTGA"),
        ]),
    );

    let rows = db.classify_path(&reads, 0.0).unwrap();
    let assigned: Vec<u32> = rows.iter().map(|r| r.tax_id).collect();
    let batch = db.abundance_batch(&assigned).unwrap();

    assert_eq!(batch.num_rows(), 3, "species 100, species 200, unclassified");

    let column = batch.column_by_name("relative_abundance").unwrap();
    let values = column
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("relative_abundance is Float64");
    let total: f64 = (0..values.len()).map(|i| values.value(i)).sum();
    assert!((total - 1.0).abs() < 1e-12, "relative abundance summed to {total}, not 1.0");
}

/// An empty read file is a well-formed empty answer, not an error and not a
/// hang -- the same guarantee `tests/fastq_robustness.rs` pins for counting.
#[test]
fn an_empty_read_file_classifies_to_an_empty_result() {
    let fx = Fixture::new("empty_reads");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    let reads = fx.write("empty.fastq", "");
    assert!(db.classify_path(&reads, 0.0).unwrap().is_empty());
    assert_eq!(db.abundance_batch(&[]).unwrap().num_rows(), 0);
}

#[test]
fn a_malformed_read_file_names_the_file_and_the_record() {
    let fx = Fixture::new("malformed_reads");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    // Second record's quality line is shorter than its sequence.
    let reads = fx.write(
        "broken.fastq",
        &format!("@ok\n{ONLY_A}\n+\n{}\n@bad\n{ONLY_B}\n+\nIII\n", "I".repeat(ONLY_A.len())),
    );

    match db.classify_path(&reads, 0.0) {
        Err(FastDnaError::MalformedFastq { path, record, .. }) => {
            assert_eq!(path, reads);
            assert_eq!(record, 2, "the second record is the broken one");
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

#[test]
fn a_reference_sequence_missing_from_the_taxonomy_names_it() {
    let fx = Fixture::new("unmapped_sequence");
    let reference = fx.write("reference.fasta", &format!(">species_a\n{ONLY_A}\n>surprise\n{ONLY_B}\n"));
    let taxonomy = fx.write("taxonomy.tsv", TAXONOMY);

    match KmerDatabase::build(&reference, &taxonomy, K) {
        Err(FastDnaError::Load { path, reason }) => {
            assert_eq!(path, reference);
            assert!(reason.contains("surprise"), "reason: {reason}");
            assert!(reason.contains("taxonomy.tsv"), "reason must point at the fix: {reason}");
        }
        other => panic!("expected Load, got {other:?}"),
    }
}

#[test]
fn a_taxonomy_file_that_does_not_exist_is_an_io_error() {
    let fx = Fixture::new("missing_taxonomy");
    let reference = fx.write("reference.fasta", &reference());
    match KmerDatabase::build(&reference, fx.path("nope.tsv"), K) {
        Err(FastDnaError::Io { .. }) => {}
        other => panic!("expected Io, got {other:?}"),
    }
}

/// Raising the threshold may only make a call less specific or drop it --
/// it must never move a call sideways to a different branch of the tree.
/// That is what makes the knob safe to turn without re-reading the docs.
#[test]
fn raising_the_threshold_only_ever_generalizes_a_call() {
    let fx = Fixture::new("threshold_monotonic");
    let (reference, taxonomy) = fx.standard_inputs();
    let db = KmerDatabase::build(&reference, &taxonomy, K).unwrap();

    let reads = fx.write(
        "reads.fastq",
        &fastq(&[
            ("species_a_genome", &format!("{ONLY_A}{SHARED}")),
            ("partial", &format!("{ONLY_A}CTCTAGGACTGA")),
            ("shared", SHARED),
        ]),
    );

    let mut previous = tax_ids(&db, &reads, 0.0);
    for step in 1..=10 {
        let threshold = f64::from(step) / 10.0;
        let current = tax_ids(&db, &reads, threshold);
        for (before, after) in previous.iter().zip(current.iter()) {
            if before == after {
                continue;
            }
            assert!(
                *after == UNCLASSIFIED_TAX_ID
                    || db.taxonomy().is_ancestor_or_self(*after, *before),
                "at threshold {threshold}, a call moved from {before} to {after}, which is \
                 neither an ancestor of it nor unclassified"
            );
        }
        previous = current;
    }
}
