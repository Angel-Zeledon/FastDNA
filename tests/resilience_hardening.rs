//! Resilience contract for the I/O layer: atomic output replacement, the
//! input-overwrite guard, multi-member gzip support in `from_path`, and
//! trim safety on malformed records. Each test pins a defect found by the
//! 2026-08-24 full review (see docs/superpowers/plans/2026-08-24-*.md).

#![allow(clippy::unwrap_used)]

use std::fs;
use std::io::Write as _;

use flate2::write::GzEncoder;
use flate2::Compression;
use tempfile::TempDir;

use fastdna_core::atomic::{same_file, AtomicFile};
use fastdna_core::counter::KmerCounter;
use fastdna_core::export;
use fastdna_core::fastq::{FastqReader, FastqRecord};

const READ: &str = "ACGTACGTGGCCAATTACGTACGTGGCCAATT";

fn fastq_text(reads: &[&str]) -> String {
    reads
        .iter()
        .enumerate()
        .map(|(i, seq)| format!("@r{}\n{}\n+\n{}\n", i, seq, "I".repeat(seq.len())))
        .collect()
}

fn gz_member(text: &str) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(text.as_bytes()).unwrap();
    enc.finish().unwrap()
}

fn count_records(mut reader: FastqReader<Box<dyn std::io::BufRead + Send>>) -> usize {
    let mut n = 0;
    while let Some(_record) = reader.next_record().unwrap() {
        n += 1;
    }
    n
}

// ---- same_file: the input-overwrite guard's foundation ----

#[test]
fn same_file_recognizes_an_identical_path() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.fastq");
    fs::write(&path, "x").unwrap();
    assert!(same_file(&path, &path));
}

#[test]
fn same_file_sees_through_a_relative_vs_absolute_spelling() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.fastq");
    fs::write(&path, "x").unwrap();
    let dotted = dir.path().join(".").join("data.fastq");
    assert!(same_file(&path, &dotted));
}

#[cfg(windows)]
#[test]
fn same_file_is_case_insensitive_on_windows() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.fastq");
    fs::write(&path, "x").unwrap();
    let upper = dir.path().join("DATA.FASTQ");
    assert!(same_file(&path, &upper));
}

#[test]
fn same_file_distinguishes_two_real_files() {
    let dir = TempDir::new().unwrap();
    let a = dir.path().join("a.fastq");
    let b = dir.path().join("b.csv");
    fs::write(&a, "x").unwrap();
    assert!(!same_file(&a, &b), "a planned output beside the input is not the input");
}

#[test]
fn same_file_never_matches_a_path_in_a_missing_directory() {
    let dir = TempDir::new().unwrap();
    let a = dir.path().join("a.fastq");
    fs::write(&a, "x").unwrap();
    let ghost = dir.path().join("no_such_dir").join("a.fastq");
    assert!(!same_file(&a, &ghost));
}

// ---- AtomicFile: no truncate-in-place, no partial output ----

#[test]
fn dropping_an_uncommitted_atomic_file_leaves_nothing_behind() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("out.csv");
    {
        let (mut file, _pending) = AtomicFile::create(&dest).unwrap();
        file.write_all(b"half-written").unwrap();
        // `_pending` dropped without commit: the write is abandoned.
    }
    assert!(!dest.exists(), "an abandoned write must not surface at the destination");
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        0,
        "no temp file may be left behind"
    );
}

#[test]
fn an_abandoned_write_preserves_the_previous_good_output() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("out.csv");
    fs::write(&dest, "previous good run").unwrap();
    {
        let (mut file, _pending) = AtomicFile::create(&dest).unwrap();
        file.write_all(b"half-written garbage").unwrap();
    }
    assert_eq!(
        fs::read_to_string(&dest).unwrap(),
        "previous good run",
        "a failed export must not clobber the last good file"
    );
}

#[test]
fn a_committed_atomic_file_replaces_the_destination() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("out.csv");
    fs::write(&dest, "old").unwrap();
    let (mut file, pending) = AtomicFile::create(&dest).unwrap();
    file.write_all(b"new content").unwrap();
    drop(file);
    pending.commit().unwrap();
    assert_eq!(fs::read_to_string(&dest).unwrap(), "new content");
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        1,
        "only the destination may remain after commit"
    );
}

// ---- exporters ride on AtomicFile ----

fn counter_with_data() -> KmerCounter {
    let mut counter = KmerCounter::new();
    let kmers = fastdna_core::kmer::extract_canonical_kmers(READ.as_bytes(), 4);
    counter.insert_batch(&kmers);
    counter
}

#[test]
fn csv_export_leaves_no_temp_sibling_on_success() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("counts.csv");
    export::export_counts_csv(&counter_with_data(), &dest, 4, 1, false).unwrap();
    assert!(dest.exists());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn parquet_export_leaves_no_temp_sibling_on_success() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("counts.parquet");
    export::export_counts_parquet(&counter_with_data(), &dest, 4, 1, false, true).unwrap();
    assert!(dest.exists());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn failed_csv_export_to_a_missing_directory_creates_nothing() {
    let dir = TempDir::new().unwrap();
    let dest = dir.path().join("ghost").join("counts.csv");
    let err = export::export_counts_csv(&counter_with_data(), &dest, 4, 1, false).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("counts.csv"), "error must name the destination: {msg}");
    assert!(!dest.exists());
}

// ---- from_path gzip semantics ----

#[test]
fn from_path_reads_every_member_of_a_multi_member_gz() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("two_members.fastq.gz");
    let mut bytes = gz_member(&fastq_text(&[READ]));
    bytes.extend(gz_member(&fastq_text(&[READ, READ])));
    fs::write(&path, bytes).unwrap();

    let reader = FastqReader::from_path(&path).unwrap();
    assert_eq!(
        count_records(reader),
        3,
        "a concatenated (SRA-style) gz must yield the records of every member"
    );
}

#[test]
fn from_path_honours_an_uppercase_gz_extension() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("SAMPLE.FASTQ.GZ");
    fs::write(&path, gz_member(&fastq_text(&[READ]))).unwrap();

    let reader = FastqReader::from_path(&path).unwrap();
    assert_eq!(count_records(reader), 1, ".GZ must be recognized as gzip");
}

// ---- quality_trim_end on a malformed record ----

#[test]
fn quality_trim_end_survives_a_qual_shorter_than_seq() {
    let mut record = FastqRecord {
        id: b"r1".to_vec(),
        seq: b"ACGTACGTACGT".to_vec(),
        qual: b"II".to_vec(),
    };
    // Must not panic; a structurally invalid record can reach this method
    // through the public API even though the pipeline's reader rejects it.
    record.quality_trim_end(30.0, 4);
    assert!(record.seq.len() <= 12);
    assert_eq!(
        record.seq.len(),
        record.qual.len(),
        "trim must restore the seq/qual length invariant it depends on"
    );
}
