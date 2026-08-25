//! Multiple `--input` files, and `-` for stdin.
//!
//! Real samples arrive as R1/R2 across several lanes, and the standard HPC
//! idiom is `fasterq-dump ... | fastdna -i -`. Both funnel into the same
//! bounded channel the single-file producer already uses, so what these
//! tests pin down is that the aggregate is exactly the concatenation, and
//! that an error still names the file it actually came from.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;
use std::path::{Path, PathBuf};

use fastdna_core::counter::KmerCounter;
use fastdna_core::error::FastDnaError;
use fastdna_core::fastq::{FastqReader, MultiSourceReader};
use fastdna_core::pipeline::{process_stream_parallel, PipelineConfig};

fn config(k: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 0.0,
        quality_window: 4,
        batch_size: 4,
        num_threads: 2,
        progress_interval: 100_000,
    }
}

fn table(counter: &KmerCounter) -> Vec<(u64, u32)> {
    let mut entries: Vec<(u64, u32)> = counter.iter().collect();
    entries.sort_unstable();
    entries
}

/// A throwaway directory under the system temp dir, removed on drop.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_multi_input_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn count_files(paths: Vec<PathBuf>, k: usize) -> (KmerCounter, u64) {
    let reader = MultiSourceReader::from_paths(paths);
    let (counter, _qc, reads) =
        process_stream_parallel(reader, config(k), Path::new("<inputs>"), None, None)
            .expect("valid input");
    (counter, reads)
}

fn count_text(text: &str, k: usize) -> KmerCounter {
    let reader = FastqReader::new(Cursor::new(text.as_bytes().to_vec()));
    let (counter, _qc, _reads) =
        process_stream_parallel(reader, config(k), Path::new("<memory>"), None, None)
            .expect("valid input");
    counter
}

const A: &str = "@a1\nACGTACGTTG\n+\nIIIIIIIIII\n@a2\nGGGGCCCCAA\n+\nIIIIIIIIII\n";
const B: &str = "@b1\nTTGCAACGTT\n+\nIIIIIIIIII\n@b2\nACGTACGTTG\n+\nIIIIIIIIII\n";

#[test]
fn two_files_count_exactly_as_their_concatenation() {
    let fx = Fixture::new("concat");
    let a = fx.write("a.fastq", A);
    let b = fx.write("b.fastq", B);

    let (multi, reads) = count_files(vec![a, b], 5);
    let concatenated = count_text(&format!("{A}{B}"), 5);

    assert_eq!(reads, 4, "read counts must aggregate across files");
    assert_eq!(multi.total_kmers(), concatenated.total_kmers());
    assert_eq!(table(&multi), table(&concatenated));
    assert!(multi.distinct_kmers() > 0, "the fixture must count something");
}

/// A single path through the multi-file reader must be indistinguishable
/// from the single-file path it replaces -- `-i sample.fastq` is by far the
/// most common invocation and must not change at all.
#[test]
fn one_file_still_counts_exactly_as_before() {
    let fx = Fixture::new("single");
    let a = fx.write("a.fastq", A);

    let (multi, reads) = count_files(vec![a], 5);
    assert_eq!(reads, 2);
    assert_eq!(table(&multi), table(&count_text(A, 5)));
}

/// Order must not matter to the counts (k-mer counting is commutative), but
/// the reads must all be there.
#[test]
fn file_order_does_not_change_the_counts() {
    let fx = Fixture::new("order");
    let a = fx.write("a.fastq", A);
    let b = fx.write("b.fastq", B);

    let (forward, _) = count_files(vec![a.clone(), b.clone()], 5);
    let (reverse, _) = count_files(vec![b, a], 5);
    assert_eq!(table(&forward), table(&reverse));
}

/// The error must name the file that actually failed, not the first input
/// and not a generic "<inputs>" label, and the record number must be the
/// one *within that file* -- a user asked to look at "record 2" of a
/// four-record run cannot find it otherwise.
#[test]
fn a_malformed_second_file_is_named_in_the_error() {
    let fx = Fixture::new("bad_second");
    let good = fx.write("good.fastq", A);
    // Two well-formed records, then a header with no '@'.
    let bad = fx.write("broken.fastq", &format!("{B}NOT_A_HEADER\nACGT\n+\nIIII\n"));

    let reader = MultiSourceReader::from_paths(vec![good, bad.clone()]);
    let result = process_stream_parallel(reader, config(5), Path::new("<inputs>"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { path, record, reason }) => {
            assert_eq!(path, bad, "the failing file must be named, not the first input");
            assert_eq!(record, 3, "record numbers restart per file: this is broken.fastq's 3rd");
            assert!(reason.contains('@'), "reason: {reason}");
        }
        other => panic!("expected MalformedFastq naming broken.fastq, got {other:?}"),
    }
}

/// A file that cannot be opened at all (the second one, so the first has
/// already been read successfully) must surface as an I/O error naming that
/// file -- not as a malformed-record error blaming the bytes.
#[test]
fn a_missing_second_file_is_reported_as_io_naming_that_file() {
    let fx = Fixture::new("missing_second");
    let good = fx.write("good.fastq", A);
    let missing = fx.dir.join("does_not_exist.fastq");

    let reader = MultiSourceReader::from_paths(vec![good, missing.clone()]);
    let result = process_stream_parallel(reader, config(5), Path::new("<inputs>"), None, None);

    match result {
        Err(FastDnaError::Io { path, .. }) => {
            assert_eq!(path, missing, "the unopenable file must be named");
        }
        other => panic!("expected Io naming the missing file, got {other:?}"),
    }
}

/// Formats are per-file, decided by each file's own first byte: a FASTA and
/// a FASTQ in the same run are both read correctly.
#[test]
fn fasta_and_fastq_files_can_be_mixed_in_one_run() {
    let fx = Fixture::new("mixed_format");
    let fastq = fx.write("reads.fastq", A);
    let fasta = fx.write("genome.fasta", ">g1\nTTGCAA\nCGTT\n>g2\nACGTACGTTG\n");

    let (multi, reads) = count_files(vec![fastq, fasta], 5);
    let equivalent = count_text(&format!("{A}{B}"), 5);

    assert_eq!(reads, 4);
    assert_eq!(table(&multi), table(&equivalent));
}

/// An empty input list is a caller mistake, not an empty result set. The
/// CLI can never produce one (clap requires at least one `--input`), but a
/// library caller can, and silently counting nothing would look exactly
/// like a successful run over a sample that happened to have no reads.
#[test]
fn an_empty_input_list_is_rejected_rather_than_counted_as_zero_reads() {
    let reader = MultiSourceReader::from_paths(Vec::<PathBuf>::new());
    let result = process_stream_parallel(reader, config(5), Path::new("<inputs>"), None, None);
    assert!(
        matches!(result, Err(FastDnaError::InvalidConfig { .. })),
        "an empty input list must be an error, got {result:?}"
    );
}

/// The pipeline's own single-reader contract is untouched: a plain
/// `FastqReader` still counts, and still attributes errors to the `source`
/// path the caller passed, exactly as before this file existed.
#[test]
fn a_plain_fastq_reader_still_attributes_errors_to_the_caller_supplied_source() {
    let reader = FastqReader::new(Cursor::new(b"@ok\nACGT\n+\nIIII\nBROKEN\n".to_vec()));
    let result = process_stream_parallel(reader, config(4), Path::new("caller/named.fastq"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { path, record, .. }) => {
            assert_eq!(path, Path::new("caller/named.fastq"));
            assert_eq!(record, 2);
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}
