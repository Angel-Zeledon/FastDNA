//! Export failures must name the file that could not be written.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use fastdna::counter::KmerCounter;
use fastdna::error::FastDnaError;
use fastdna::export;

fn small_counter() -> KmerCounter {
    let mut c = KmerCounter::new();
    c.insert(0);
    c.insert(1);
    c
}

#[test]
fn csv_export_to_an_unwritable_path_names_the_path() {
    let bad = std::path::Path::new("no_such_directory_xyz").join("out.csv");

    let result = export::export_csv(&small_counter(), &bad, 4, 1);

    match result {
        Err(FastDnaError::Io { path, .. }) => {
            assert!(path.to_string_lossy().contains("out.csv"), "got {path:?}");
        }
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn parquet_export_to_an_unwritable_path_names_the_path() {
    let bad = std::path::Path::new("no_such_directory_xyz").join("out.parquet");

    let result = export::export_parquet(&small_counter(), &bad, 4, 1);

    assert!(matches!(result, Err(FastDnaError::Io { .. })), "got {result:?}");
}

#[test]
fn successful_csv_export_reports_rows_written() {
    let dir = std::env::temp_dir().join("fastdna_export_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("ok.csv");

    let written = export::export_csv(&small_counter(), &out, 4, 1).expect("must succeed");

    assert_eq!(written, 2);
    let _ = std::fs::remove_file(&out);
}
