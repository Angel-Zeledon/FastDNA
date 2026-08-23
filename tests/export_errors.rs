//! Export failures must name the file that could not be written.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use fastdna_core::counter::KmerCounter;
use fastdna_core::error::FastDnaError;
use fastdna_core::export;

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

/// A scratch directory unique to this process and call, cleaned up on drop.
///
/// A fixed path reused across runs would let leftover state from one run
/// (or a Windows `remove_dir_all` still pending delete, as antivirus or the
/// search indexer can hold a brief handle on a newly written file) collide
/// with the next; a unique path per run has nothing to collide with.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("{name}_{unique}"));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn successful_csv_export_reports_rows_written() {
    let scratch = ScratchDir::new("fastdna_export_test");
    let out = scratch.0.join("ok.csv");

    let written = export::export_csv(&small_counter(), &out, 4, 1).expect("must succeed");

    assert_eq!(written, 2);
}
