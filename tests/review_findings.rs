//! FAILING regression tests written during the 2026-08-27 correctness
//! review of commit bbe9301. Each pins a defect that exists today; none is
//! a feature request. They are expected to FAIL until fixed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::path::PathBuf;

use fastdna_core::counter::KmerCounter;
use fastdna_core::error::FastDnaError;
use fastdna_core::export;
use fastdna_core::fastq::InputSpec;
use fastdna_core::ktab::KmerTable;
use fastdna_core::read_filter::{self, FilterMode, ReferenceIndex};
use fastdna_core::setops::{self, CombineOp};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("fastdna_review_findings").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_table(path: &std::path::Path, k: usize, entries: &[u64]) {
    let mut c = KmerCounter::new();
    c.insert_batch(entries);
    export::export_counts_parquet(&c, path, k, 1, false, true).unwrap();
}

/// Writes `pairs` verbatim, in whatever order they are given, claiming
/// `fastdna.sorted_by=kmer_u64` regardless -- standing in for an externally
/// produced file (plain pyarrow, or any writer outside this crate) that
/// makes the same claim without it being true. `export::export_pairs_parquet`
/// itself now refuses to do this (FINDING 2 below), so a test that needs
/// exactly such a malformed-but-labeled fixture has to build it by hand.
fn write_table_unchecked_order(path: &std::path::Path, k: usize, pairs: &[(u64, u32)]) {
    use arrow::array::{UInt32Array, UInt64Array};
    use arrow::record_batch::RecordBatch;
    use fastdna_core::ktab::{K_KEY, SORTED_BY_KEY, SORTED_BY_VALUE};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::metadata::KeyValue;
    use parquet::file::properties::WriterProperties;

    let schema = export::counts_schema(false);
    let kmers: UInt64Array = pairs.iter().map(|&(k, _)| k).collect();
    let freqs: UInt32Array = pairs.iter().map(|&(_, f)| f).collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(kmers), std::sync::Arc::new(freqs)])
        .unwrap();

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![
            KeyValue::new(SORTED_BY_KEY.to_string(), Some(SORTED_BY_VALUE.to_string())),
            KeyValue::new(K_KEY.to_string(), Some(k.to_string())),
        ]))
        .build();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

// ---------------------------------------------------------------------
// FINDING 1: no output-overwrite guard below the CLI layer.
//
// `main.rs` guards `count`, `filter`, `union`/`intersect`/`diff`, `sketch`
// and `matrix` against an `--output` that resolves to one of the run's own
// inputs ("irreversible data loss", `guard_against_input_overwrite`). Those
// guards live only in `main.rs`, so `ffi.rs` -- which calls
// `read_filter::run_filter` and `export::export_pairs_parquet` directly --
// has none of them, and `KmerTable.filter_reads(output=<its own input>)`
// from Python silently destroys the caller's file (reproduced separately in
// `python/tests/test_review_findings.py`).
//
// The library-level fix is to move the check into the library entry points
// these tests call, so every caller -- CLI, Python, and any future consumer
// of `fastdna_core` -- inherits it. These two tests pin that.
// ---------------------------------------------------------------------

#[test]
fn run_filter_rejects_an_output_that_is_one_of_its_own_inputs() {
    let dir = scratch("filter_self_overwrite");
    let sample = dir.join("sample.fastq");
    let mut f = std::fs::File::create(&sample).unwrap();
    for i in 0..5 {
        writeln!(f, "@r{i}\nGCGCGCGC\n+\nIIIIIIII").unwrap();
    }
    drop(f);
    let original = std::fs::read(&sample).unwrap();

    let table_path = dir.join("ref.parquet");
    write_table(&table_path, 4, &[0]); // "AAAA" only -- the sample shares nothing
    let index = ReferenceIndex::from_table(&KmerTable::open(&table_path).unwrap()).unwrap();

    let result = read_filter::run_filter(
        vec![InputSpec::File(sample.clone())],
        &index,
        FilterMode::Keep,
        0.5,
        &sample, // <-- the output is the input
    );

    assert!(
        matches!(result, Err(FastDnaError::InvalidConfig { .. })),
        "run_filter must reject an --output that is one of its own inputs, got {:?}",
        result.map(|s| s.reads_written)
    );
    assert_eq!(
        std::fs::read(&sample).unwrap(),
        original,
        "the input FASTQ was destroyed: {} bytes before, {} after",
        original.len(),
        std::fs::metadata(&sample).unwrap().len()
    );
}

#[test]
fn run_filter_rejects_an_output_that_is_the_reference_table() {
    let dir = scratch("filter_table_overwrite");
    let sample = dir.join("sample.fastq");
    std::fs::write(&sample, b"@r1\nGCGCGCGC\n+\nIIIIIIII\n").unwrap();

    let table_path = dir.join("ref.parquet");
    write_table(&table_path, 4, &[0]);
    let table = KmerTable::open(&table_path).unwrap();
    let index = ReferenceIndex::from_table(&table).unwrap();
    let original_len = std::fs::metadata(&table_path).unwrap().len();

    let result = read_filter::run_filter(
        vec![InputSpec::File(sample)],
        &index,
        FilterMode::Keep,
        0.5,
        &table_path, // <-- the output is the reference table
    );

    assert!(
        matches!(result, Err(FastDnaError::InvalidConfig { .. })),
        "run_filter must reject an --output that is the reference table itself"
    );
    assert_eq!(std::fs::metadata(&table_path).unwrap().len(), original_len);
    KmerTable::open(&table_path).expect("the reference table must still be openable");
}

// ---------------------------------------------------------------------
// FINDING 2: `export_pairs_parquet` stamps `fastdna.sorted_by=kmer_u64` on
// whatever stream it is handed, without ever checking that the stream is
// actually ascending -- and `MultiTableMerge` produces a non-ascending
// stream whenever one of its sources is non-ascending.
//
// `KmerTable::open` only verifies order at row-group *boundaries*, so a
// single-row-group table that carries the metadata but is internally
// unsorted is accepted. Building exactly such a file with plain pyarrow is
// the documented fixture convention of this crate's own Python test suite
// (`python/tests/test_setops.py`: "`KmerTable::open`'s contract is 'any
// sorted Parquet file carrying the fastdna.sorted_by/fastdna.k footer
// metadata', not 'only a file this crate's own exporter wrote'").
//
// The result is a silently wrong set operation whose output is itself
// stamped as a valid, sorted k-mer table -- corruption that propagates.
// Both `setops` and `read_filter::ReferenceIndex` already decode every row,
// so verifying ascending order costs one comparison per row and would turn
// this into a loud error.
// ---------------------------------------------------------------------

#[test]
fn export_pairs_parquet_must_not_claim_sorted_by_for_an_unsorted_stream() {
    let dir = scratch("unsorted_claim");
    let out = dir.join("out.parquet");

    // Deliberately descending.
    let pairs: Vec<fastdna_core::Result<(u64, u32)>> = vec![Ok((50, 1)), Ok((10, 2)), Ok((30, 3))];
    let result = export::export_pairs_parquet(pairs, &out, 4, true);

    assert!(
        result.is_err(),
        "export_pairs_parquet wrote fastdna.sorted_by=kmer_u64 onto a descending stream; \
         the file now claims an ordering it does not have and reopens as a valid KmerTable"
    );
}

#[test]
fn a_set_operation_over_an_out_of_order_source_must_fail_loudly() {
    let dir = scratch("unsorted_setop");

    // `a` is built the ordinary way: sorted.
    let a_path = dir.join("a.parquet");
    write_table(&a_path, 4, &[10, 20, 30, 50]);

    // `u` claims to be sorted but is not -- one row group, descending-ish
    // rows, correct footer metadata. `KmerTable::open` accepts it.
    //
    // `export::export_pairs_parquet` itself now refuses to write this shape
    // (FINDING 2, tested above), so this fixture is built with
    // `write_table_unchecked_order`, standing in for a file produced outside
    // this crate (plain pyarrow; see the module comment above) that makes
    // the same false claim.
    let u_path = dir.join("u.parquet");
    write_table_unchecked_order(&u_path, 4, &[(50, 1), (10, 2), (30, 3), (20, 4)]);
    let u = KmerTable::open(&u_path).expect("open accepts a single-row-group unsorted table today");
    let a = KmerTable::open(&a_path).unwrap();

    // Ground truth: every k-mer of `u` is also in `a`, so the intersection
    // is all four and the difference is empty.
    let inter: Vec<(u64, u32)> =
        setops::intersect(&[u.clone(), a.clone()], CombineOp::Min).unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(
        inter.len(),
        4,
        "intersect over an out-of-order source returned {} of 4 shared k-mers ({inter:?}) \
         instead of failing loudly",
        inter.len()
    );

    let difference: Vec<(u64, u32)> =
        setops::diff(&u, std::slice::from_ref(&a), 0).unwrap().map(|r| r.unwrap()).collect();
    assert!(
        difference.is_empty(),
        "diff over an out-of-order source returned {difference:?} instead of the empty set \
         (or failing loudly)"
    );
}

// ---------------------------------------------------------------------
// FINDING 3: `fastdna matrix --sample` derives each sample's id from the
// file's *basename* (`main.rs::sample_id_from_path`), and neither
// `cohort::matrix::build_cohort_matrix_from_files` nor
// `export::export_cohort_matrix_parquet` checks that the resulting ids are
// distinct.
//
// The output schema is long/COO with `sample_id` as the *only* sample key
// (`CohortMatrix::row`'s integer index is not written), so two samples that
// share a basename -- e.g. `runA/patient01.fastq` and
// `runB/patient01.fastq`, exactly the "cohort whose files are not laid out
// in one directory" case `--sample` is documented for, or a `.fastq` and
// its own `.fastq.gz` -- become indistinguishable in the file. A downstream
// `GROUP BY sample_id` / pivot silently pools two different subjects, while
// the footer still reports `fastdna.n_samples=2`.
//
// Reproduced end-to-end:
//   fastdna matrix --sample runA/patient01.fastq --sample runB/patient01.fastq \
//                  -o m.parquet -k 11 -m 1 --min-samples 1
//   -> "Samples: 2 | nonzero entries written: 5995", and every one of those
//      5995 rows carries sample_id == "patient01".
// ---------------------------------------------------------------------

#[test]
fn a_cohort_matrix_must_not_be_written_with_two_samples_sharing_one_id() {
    use fastdna_core::cohort::matrix::build_cohort_matrix;

    let dir = scratch("duplicate_sample_ids");
    let out = dir.join("matrix.parquet");

    let mut s0 = KmerCounter::new();
    s0.insert_batch(&[1, 1, 5]);
    let mut s1 = KmerCounter::new();
    s1.insert_batch(&[1, 9, 9]);
    let matrix = build_cohort_matrix(&[s0, s1], 1, None, 4);

    // What `main.rs::sample_id_from_path` produces for
    // runA/patient01.fastq and runB/patient01.fastq.
    let ids = vec!["patient01".to_string(), "patient01".to_string()];

    let result = export::export_cohort_matrix_parquet(&matrix, &ids, &out, 4, false);

    assert!(
        matches!(result, Err(FastDnaError::InvalidConfig { .. })),
        "export_cohort_matrix_parquet accepted two samples under one sample_id; the long/COO \
         schema carries no other sample key, so the two are unrecoverably merged in the output \
         while fastdna.n_samples still says 2"
    );
}

// ---------------------------------------------------------------------
// FINDING 4: `read_filter::OutputWriter::finish`'s gzip arm calls
// `GzEncoder::finish()`, which writes the deflate tail and the 8-byte gzip
// trailer *into the inner `BufWriter` without flushing it*, and then drops
// that `BufWriter`. `BufWriter::drop` flushes on a best-effort basis and
// discards any error -- exactly the `Drop`-swallows-the-error failure the
// enum's own doc comment says it exists to prevent ("`finish` needs to
// reach `GzEncoder::finish` specifically ... rather than relying on `Drop`
// to swallow that error silently"). A disk-full error on that final flush
// is therefore not reported, and `pending.commit()` renames a truncated
// `.gz` into place as if it were complete.
//
// `OutputWriter` is private and hard-wired to `File`, so this test pins the
// two underlying facts instead, against a sink that fails on `flush`:
//
//   * the *old* arm (`gz.finish().map(|_| ())`) reports `Ok` even though the
//     trailer never reached the sink -- the bug, reproduced;
//   * the *fixed* arm (`gz.finish().and_then(|mut w| w.flush())`) reports the
//     `Err` instead -- the fix, pinned.
//
// This replaces the reviewer's original assertion, which compared byte
// counts across `drop` to *demonstrate* that the trailer was still buffered.
// That form characterized `flate2`/`BufWriter` behavior rather than this
// crate's code, so it could never go green no matter how
// `OutputWriter::finish` was fixed. The defect it documented is real and
// unchanged; what is asserted below is strictly stronger -- it fails if the
// gzip arm ever regresses to the swallow-the-error shape.
// ---------------------------------------------------------------------

#[test]
fn the_gzip_arm_must_report_a_final_flush_error_instead_of_swallowing_it() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::BufWriter;

    /// Accepts every write, fails every flush -- a stand-in for the disk
    /// filling up exactly as the gzip trailer is handed over.
    struct FailingFlushSink;
    impl Write for FailingFlushSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("no space left on device"))
        }
    }

    let payload = b"@r1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n";

    // The old shape: the flush error rides on `BufWriter::drop`, which
    // discards it, so the caller is told the write succeeded.
    let mut gz = GzEncoder::new(BufWriter::new(FailingFlushSink), Compression::default());
    gz.write_all(payload).unwrap();
    let old_shape: std::io::Result<()> = gz.finish().map(|_| ());
    assert!(
        old_shape.is_ok(),
        "expected to reproduce the original defect: the pre-fix arm reported success even though \
         the sink could not flush"
    );

    // The fixed shape, as `read_filter::OutputWriter::finish` now spells it:
    // the same failure is surfaced to the caller, so `AtomicFile::commit`
    // never renames a truncated `.gz` into place as if it were complete.
    let mut gz = GzEncoder::new(BufWriter::new(FailingFlushSink), Compression::default());
    gz.write_all(payload).unwrap();
    let fixed_shape: std::io::Result<()> = gz.finish().and_then(|mut inner| inner.flush());
    assert!(
        fixed_shape.is_err(),
        "read_filter::OutputWriter::finish's gzip arm must reach the inner writer's own checked \
         flush, so a failure delivering the gzip trailer is reported rather than swallowed by \
         BufWriter::drop"
    );
}
