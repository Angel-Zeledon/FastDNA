// src/error.rs

use std::fmt;
use std::path::PathBuf;

/// Every fallible operation in the FastDNA core returns this type.
#[derive(Debug)]
pub enum FastDnaError {
    /// An underlying I/O failure, carrying the path that caused it.
    Io { path: PathBuf, source: std::io::Error },
    /// A FASTQ record that could not be parsed, located precisely.
    MalformedFastq { path: PathBuf, record: u64, reason: String },
    /// `k` outside the 1..=32 range imposed by 2-bit packing.
    InvalidK { k: usize },
    /// A cohort directory containing no recognizable FASTQ files.
    NoSamplesFound { dir: PathBuf },
    /// A dense matrix that would exceed the configured byte limit.
    MatrixTooLarge { estimated_bytes: u64, limit: u64 },
    /// A prevalence table that would exceed the configured byte limit.
    VocabTooLarge { estimated_bytes: u64, limit: u64 },
    /// Two sketches built with different `k` cannot be compared.
    MismatchedK { left: usize, right: usize },
    /// Two `FracSketch`es built with different `scale` cannot be compared:
    /// each hash is kept independently of set size at probability `1/scale`,
    /// so comparing sketches built at different scales would compare sets
    /// sampled at different rates, silently distorting the ratio.
    MismatchedScale { left: u64, right: u64 },
    /// A worker thread panicked. This indicates a bug in FastDNA.
    Internal { detail: String },
    /// A configuration value supplied by the caller is not usable.
    InvalidConfig { parameter: &'static str, reason: String },
    /// Serialization or writer failure while exporting results.
    Export { path: PathBuf, reason: String },
    /// A file that was supposed to be a previously-saved artifact (e.g. a
    /// `GenomeSketch` written by `save`) could not be read back -- corrupt
    /// JSON, a foreign file, or data that fails the loader's own
    /// consistency checks. Deliberately distinct from `Export`: an error
    /// while reading must never claim to be an error while writing, which
    /// is what reusing `Export` for both would tell a caller.
    Load { path: PathBuf, reason: String },
    /// The caller asked to stop before the work finished.
    ///
    /// Not an error in the "something broke" sense -- it is a normal
    /// outcome -- but it belongs in this type because it is how a
    /// cancelled call returns early instead of completing.
    Cancelled,
}

impl fmt::Display for FastDnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastDnaError::Io { path, source } => {
                write!(f, "I/O error on {}: {}", path.display(), source)
            }
            FastDnaError::MalformedFastq { path, record, reason } => write!(
                f,
                "malformed FASTQ in {} at record {}: {}",
                path.display(),
                record,
                reason
            ),
            FastDnaError::InvalidK { k } => {
                write!(f, "invalid k-mer size {k}: k must be between 1 and 32 inclusive")
            }
            FastDnaError::NoSamplesFound { dir } => write!(
                f,
                "no FASTQ files found in {} (expected .fastq, .fq, .fastq.gz or .fq.gz)",
                dir.display()
            ),
            FastDnaError::MatrixTooLarge { estimated_bytes, limit } => write!(
                f,
                "dense matrix would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 lower --top-features or use --format sparse"
            ),
            FastDnaError::VocabTooLarge { estimated_bytes, limit } => write!(
                f,
                "vocabulary table would need {estimated_bytes} bytes, over the {limit} byte limit; \
                 raise --min-count or use --approx-vocab"
            ),
            FastDnaError::MismatchedK { left, right } => {
                write!(f, "cannot compare sketches built with different k: {left} and {right}")
            }
            FastDnaError::MismatchedScale { left, right } => write!(
                f,
                "cannot compare FracSketches built with different scale: {left} and {right}"
            ),
            FastDnaError::Internal { detail } => {
                write!(f, "internal error (this is a bug in FastDNA): {detail}")
            }
            FastDnaError::InvalidConfig { parameter, reason } => {
                write!(f, "invalid configuration for {parameter}: {reason}")
            }
            FastDnaError::Export { path, reason } => {
                write!(f, "export failed for {}: {reason}", path.display())
            }
            FastDnaError::Load { path, reason } => {
                write!(f, "load failed for {}: {reason}", path.display())
            }
            FastDnaError::Cancelled => write!(f, "operation cancelled by caller"),
        }
    }
}

impl std::error::Error for FastDnaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastDnaError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, FastDnaError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn malformed_fastq_message_locates_the_failure() {
        let err = FastDnaError::MalformedFastq {
            path: PathBuf::from("patient_007.fastq.gz"),
            record: 41_337,
            reason: "expected '@' at record start".to_string(),
        };

        let msg = err.to_string();
        assert!(msg.contains("patient_007.fastq.gz"), "message must name the file: {msg}");
        assert!(msg.contains("41337"), "message must give the record number: {msg}");
        assert!(msg.contains("expected '@' at record start"), "message must give the reason: {msg}");
    }

    #[test]
    fn invalid_k_message_states_the_valid_range() {
        let msg = FastDnaError::InvalidK { k: 99 }.to_string();
        assert!(msg.contains("99"));
        assert!(msg.contains("1") && msg.contains("32"), "must state the 1..=32 range: {msg}");
    }

    #[test]
    fn io_error_exposes_its_source() {
        use std::error::Error;
        let err = FastDnaError::Io {
            path: PathBuf::from("missing.fastq"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        assert!(err.source().is_some(), "Io must expose the underlying io::Error");
        assert!(err.to_string().contains("missing.fastq"));
    }

    #[test]
    fn matrix_too_large_reports_both_numbers() {
        let msg = FastDnaError::MatrixTooLarge { estimated_bytes: 8_000_000_000, limit: 4_000_000_000 }.to_string();
        assert!(msg.contains("8000000000"));
        assert!(msg.contains("4000000000"));
    }
}
