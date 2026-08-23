// src/cohort/mod.rs
//! Cohort-level processing: given a directory of many patients' FASTQ
//! files, figure out which files belong to which sample before anything
//! downstream (counting, vectorizing) ever runs.

pub mod discovery;

pub use discovery::{discover_samples, SampleFiles};
