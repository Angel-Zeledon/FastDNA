//! `QcSummary::export_json` had no test coverage at all. The recent fix
//! split `serde_json` failures into `Io` (the writer itself failed) versus
//! `Export` (the value could not be serialized) -- see the comment on
//! `QcSummary::export_json` in `src/qc.rs`.
//!
//! Only the `Io` path is exercised below. The `Export` branch is not: every
//! `QcSummary` field is a `u64` or `f64`, and `serde_json` never fails to
//! serialize a bare `f64` -- non-finite values (NaN, infinity, reachable
//! here since the fields are `pub` and unconstrained by any invariant) are
//! written out as JSON `null` rather than rejected (verified directly: a
//! `QcSummary` with `gc_content_pct: f64::NAN` exported successfully, with
//! `"gc_content_pct": null` in the output). So for this specific struct,
//! `FastDnaError::Export` is dead code reachable only if a future field
//! changes its type to something `serde_json` can genuinely reject (e.g. a
//! non-string map key). Contriving a test around today's fields would just
//! assert on serde_json's NaN-handling implementation detail, not on
//! anything this crate controls.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use fastdna_core::error::FastDnaError;
use fastdna_core::qc::QcSummary;

#[test]
fn export_json_to_an_unwritable_path_is_io_and_names_the_path() {
    let bad = std::path::Path::new("no_such_directory_xyz").join("qc.json");

    let result = QcSummary::default().export_json(&bad);

    match result {
        Err(FastDnaError::Io { path, source }) => {
            assert!(path.to_string_lossy().contains("qc.json"), "got {path:?}");
            // The whole point of the Io/Export split is that Io must carry
            // the genuine io::Error from the failed write, not a laundered
            // stand-in -- File::create on a missing parent directory fails
            // with NotFound.
            assert_eq!(
                source.kind(),
                std::io::ErrorKind::NotFound,
                "source must be the real io::Error from File::create, got {source:?}"
            );
        }
        other => panic!("expected Io error, got {other:?}"),
    }
}
