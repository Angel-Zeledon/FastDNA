// src/fastq.rs

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use flate2::read::MultiGzDecoder;

/// Represents a single FASTQ sequencing record.
pub struct FastqRecord {
    pub id: Vec<u8>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl FastqRecord {
    #[inline(always)]
    pub fn phred_score(qual_byte: u8) -> u8 {
        qual_byte.saturating_sub(33)
    }

    pub fn quality_trim_end(&mut self, min_qual: f64, window_size: usize) {
        // The reader guarantees seq and qual are the same length, but this
        // is a public method on a struct with public fields: a structurally
        // invalid record must clamp to the shared prefix, not index past
        // the shorter buffer and panic.
        let mut end_pos = self.seq.len().min(self.qual.len());
        if end_pos < window_size {
            self.seq.truncate(end_pos);
            self.qual.truncate(end_pos);
            return;
        }
        let qual_slice = &self.qual;

        while end_pos >= window_size {
            let start = end_pos - window_size;
            let window = &qual_slice[start..end_pos];

            let sum: u64 = window.iter().map(|&q| Self::phred_score(q) as u64).sum();
            let avg_qual = sum as f64 / window_size as f64;

            if avg_qual >= min_qual {
                break;
            }
            end_pos -= 1;
        }

        self.seq.truncate(end_pos);
        self.qual.truncate(end_pos);
    }
}

/// The two ways `next_record` can fail: the underlying source could not be
/// read, or bytes were read successfully but do not form a structurally
/// valid FASTQ record.
///
/// Kept as its own small type rather than `crate::error::Result` directly:
/// the reader has no path of its own (that lives with the caller -- see
/// `process_stream_parallel`'s `source` parameter) and does not track a
/// record number across calls (the caller already does, via its own read
/// count, so re-deriving one here would just duplicate that state). The
/// caller attaches both when it converts this into a `FastDnaError`.
#[derive(Debug)]
pub enum FastqReadError {
    /// Reading from the underlying source failed. Not a data problem.
    Io(io::Error),
    /// The bytes read do not form a structurally valid FASTQ record: a
    /// missing `@`/`+` marker, a sequence/quality length mismatch, or a
    /// file that ends before a record is complete.
    Malformed(String),
}

impl fmt::Display for FastqReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FastqReadError::Io(e) => write!(f, "{e}"),
            FastqReadError::Malformed(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for FastqReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FastqReadError::Io(e) => Some(e),
            FastqReadError::Malformed(_) => None,
        }
    }
}

impl From<io::Error> for FastqReadError {
    fn from(e: io::Error) -> Self {
        FastqReadError::Io(e)
    }
}

/// Crate-local result alias for `FastqReader::next_record`.
pub(crate) type ReadResult<T> = std::result::Result<T, FastqReadError>;

/// Strips a single trailing `\n`, and a preceding `\r` if present, from a
/// line buffer read by `read_until(b'\n', ..)`.
fn strip_newline(buf: &mut Vec<u8>) {
    if buf.ends_with(b"\n") {
        buf.pop();
    }
    if buf.ends_with(b"\r") {
        buf.pop();
    }
}

/// Maximum number of characters of a malformed line quoted into an error
/// message. `read_until(b'\n', ..)` has no length cap, and a FASTA file
/// handed to this FASTQ reader by mistake -- a routine user error -- has no
/// early newline at all: its first "line" can be the entire file. Without a
/// cap, the error message becomes a second full copy of that content,
/// propagated through `FastDnaError::MalformedFastq` and across the FFI
/// boundary into a Python exception string.
const MAX_ERROR_PREVIEW_CHARS: usize = 80;

/// Renders a line buffer for embedding in an error message, truncated to at
/// most `MAX_ERROR_PREVIEW_CHARS` characters (not bytes, so a multi-byte
/// UTF-8 sequence is never split) with a trailing ellipsis when truncated.
fn preview_for_error(buf: &[u8]) -> String {
    let lossy = String::from_utf8_lossy(buf);
    let mut chars = lossy.chars();
    let preview: String = chars.by_ref().take(MAX_ERROR_PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

/// Streaming FASTQ reader supporting both plain-text and gzip files.
pub struct FastqReader<R: BufRead> {
    reader: R,
    line_buf: Vec<u8>,
}

impl FastqReader<Box<dyn BufRead + Send>> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let is_gzipped = path_ref
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("gz"));

        // MultiGzDecoder, not GzDecoder: real SRA/ENA downloads are
        // multi-member gzip, and a single-member decoder silently stops at
        // the first member boundary -- the counting pipeline (`main.rs`,
        // `ffi.rs`) already decodes all members, and `preview`/`sketch`/
        // `hll` reading a truncated prefix of the same file must not
        // disagree with it.
        let reader: Box<dyn BufRead + Send> = if is_gzipped {
            Box::new(BufReader::with_capacity(128 * 1024, MultiGzDecoder::new(file)))
        } else {
            Box::new(BufReader::with_capacity(128 * 1024, file))
        };

        Ok(FastqReader {
            reader,
            line_buf: Vec::with_capacity(512),
        })
    }
}

impl<R: BufRead> FastqReader<R> {
    pub fn new(reader: R) -> Self {
        FastqReader {
            reader,
            line_buf: Vec::with_capacity(512),
        }
    }

    /// Reads the next four-line FASTQ record, validating its structure.
    ///
    /// Checks performed: the header line starts with `@`, the separator
    /// line starts with `+`, and the sequence and quality lines are the
    /// same length. A file that ends before all four lines of a record are
    /// present is reported as malformed rather than silently returning a
    /// short record. The sequence alphabet itself is not validated here --
    /// IUPAC ambiguity codes beyond `N` are legitimate FASTQ content, and
    /// `kmer::extract_canonical_kmers` already treats any non-ACGT byte as
    /// a window reset.
    pub fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        self.line_buf.clear();
        if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
            return Ok(None);
        }
        strip_newline(&mut self.line_buf);
        if !self.line_buf.starts_with(b"@") {
            return Err(FastqReadError::Malformed(format!(
                "header line must start with '@', got {:?}",
                preview_for_error(&self.line_buf)
            )));
        }
        let id = self.line_buf.clone();

        self.line_buf.clear();
        if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
            return Err(FastqReadError::Malformed(
                "file ends mid-record: missing sequence line after header".to_string(),
            ));
        }
        strip_newline(&mut self.line_buf);
        let seq = self.line_buf.clone();

        self.line_buf.clear();
        if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
            return Err(FastqReadError::Malformed(
                "file ends mid-record: missing separator line after sequence".to_string(),
            ));
        }
        strip_newline(&mut self.line_buf);
        if !self.line_buf.starts_with(b"+") {
            return Err(FastqReadError::Malformed(format!(
                "separator line must start with '+', got {:?}",
                preview_for_error(&self.line_buf)
            )));
        }

        self.line_buf.clear();
        if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
            return Err(FastqReadError::Malformed(
                "file ends mid-record: missing quality line after separator".to_string(),
            ));
        }
        strip_newline(&mut self.line_buf);
        let qual = self.line_buf.clone();

        if seq.len() != qual.len() {
            return Err(FastqReadError::Malformed(format!(
                "sequence length {} does not match quality length {}",
                seq.len(),
                qual.len()
            )));
        }

        Ok(Some(FastqRecord { id, seq, qual }))
    }
}
