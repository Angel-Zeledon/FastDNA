// src/lib.rs

pub mod cli;
pub mod cms;
pub mod counter;
pub mod error;
pub mod export;
pub mod fastq;
#[cfg(feature = "python")]
pub mod ffi;
pub mod kmer;
pub mod pipeline;
pub mod progress;
pub mod qc;
pub mod sketch;
pub mod wasm;

pub use error::{FastDnaError, Result};
pub use progress::{Progress, ProgressFn};
