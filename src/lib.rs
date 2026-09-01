// src/lib.rs

// The public surface. These are the modules a consumer of `fastdna_core`
// is expected to name, and the ones the `[[bin]]`, the `python` feature's
// `ffi` and the `wasm` feature all reach through.
pub mod chimera_scan;
pub mod cli;
pub mod cohort;
pub mod cohort_vocab;
pub mod counter;
pub mod error;
pub mod export;
pub mod fastq;
pub mod hll;
pub mod kmer;
pub mod ktab;
pub mod mem_estimate;
pub mod metagenomics;
pub mod ntcard;
pub mod pipeline;
pub mod preview;
pub mod progress;
pub mod qc;
pub mod read_filter;
pub mod read_profile;
pub mod setops;
pub mod sketch;
pub mod translate;

// Implementation. Reachable from anywhere inside the crate, from nowhere
// outside it.
//
// Every one of these is a strategy or a storage detail whose shape is
// documented as subject to change: `adaptive_bins` is the KMC2-style bin
// map (`docs/design-minimizer-counting.md` step 6), `superkmer` is the
// 2-bit packing, `binned` is the opt-in strategy that
// `pipeline::resolve_strategy` deliberately never selects automatically,
// `disk_spill` owns temporary files, `minimizer` owns the signature
// function. Leaving them `pub` made all of them part of the crate's
// compatibility contract; each is exercised by its own tests and by
// `pipeline`, not by outside callers.
//
// `#[allow(dead_code)]`: `pub` at the crate root made every `pub` item in
// these modules reachable-from-outside as far as rustc's dead_code
// analysis is concerned, so none of it was ever checked. `pub(crate)`
// removes that exemption and immediately surfaces several functions and
// methods (e.g. `binned::count_records_adaptive`, `superkmer::SuperKmer`)
// that only their own module's tests call -- genuine dead code, but a
// separate cleanup from the visibility change itself (H-08) and outside
// this change's file scope (their source lives in `adaptive_bins.rs`,
// `binned.rs`, `disk_spill.rs`, `minimizer.rs`, `superkmer.rs`, none of
// which this change is permitted to edit beyond `binned.rs`'s test
// module). Silenced at the module boundary rather than left to inflate
// the warning count; removing the genuinely-unused items is future work.
#[allow(dead_code)]
pub(crate) mod adaptive_bins;
#[allow(dead_code)]
pub(crate) mod binned;
#[allow(dead_code)]
pub(crate) mod disk_spill;
#[allow(dead_code)]
pub(crate) mod minimizer;
#[allow(dead_code)]
pub(crate) mod superkmer;

// `atomic` owns write-then-rename and, per H-08, belongs in the block
// above -- but `tests/resilience_hardening.rs` (pre-existing at HEAD,
// outside this change's scope) names `fastdna_core::atomic::{same_file,
// AtomicFile}` directly, so narrowing this one module's visibility would
// break an already-green integration test that this change is not
// permitted to touch. Left `pub` as a deliberate, documented exception
// until that test is updated to stop reaching into it.
pub mod atomic;

// Entry points for the optional targets, public by definition.
#[cfg(feature = "python")]
pub mod ffi;
pub mod wasm;

pub use error::{FastDnaError, Result};
pub use progress::{Progress, ProgressFn};
