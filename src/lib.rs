// src/lib.rs

pub mod atomic;
pub mod cli;
pub mod cms;
pub mod cohort;
pub mod counter;
pub mod disk_spill;
pub mod error;
pub mod export;
pub mod fastq;
pub mod hll;
#[cfg(feature = "python")]
pub mod ffi;
pub mod kmer;
pub mod mem_estimate;
pub mod metagenomics;
pub mod minimizer;
pub mod pipeline;
pub mod preview;
pub mod progress;
pub mod qc;
pub mod sketch;
pub mod translate;
pub mod wasm;

pub use error::{FastDnaError, Result};
pub use progress::{Progress, ProgressFn};
