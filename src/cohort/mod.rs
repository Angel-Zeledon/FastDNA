// src/cohort/mod.rs
//! Cohort-level processing: given a directory of many patients' FASTQ
//! files, figure out which files belong to which sample before anything
//! downstream (counting, vectorizing) ever runs.

pub mod batch;
pub mod discovery;
pub mod matrix;

pub use batch::{count_paired_samples, discover_paired_samples, sample_output_path, PairedOutputFormat, SampleRunResult};
pub use discovery::{discover_samples, SampleFiles};
pub use matrix::{build_cohort_matrix, CohortMatrix};
