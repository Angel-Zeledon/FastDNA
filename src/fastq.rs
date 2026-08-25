// src/fastq.rs

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use flate2::read::MultiGzDecoder;

use crate::error::{FastDnaError, Result};

/// Represents a single sequencing record.
///
/// Named for FASTQ because that is where it came from and what the rest of
/// the pipeline is written against, but it is also what a FASTA record is
/// parsed into: the reader synthesizes a `qual` string for FASTA input (see
/// `SYNTHETIC_FASTA_QUALITY`) so that every stage downstream -- quality
/// trimming, QC, k-mer extraction -- stays exactly as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// message. `read_until(b'\n', ..)` has no length cap, and a file with no
/// early newline -- a mangled download, a binary handed over by mistake, or
/// simply a reference assembly's very long description line -- can have a
/// first "line" that is the entire file. Without a cap, the error message
/// becomes a second full copy of that content, propagated through
/// `FastDnaError::MalformedFastq` and across the FFI boundary into a Python
/// exception string.
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

/// The quality byte given to every base of a FASTA record: `'I'` is Phred+33
/// for Q40. A FASTA file carries no per-base qualities, and the value must be
/// high enough that `quality_trim_end` at the CLI's default `-q 20` is a
/// no-op -- an assembly silently truncated from its 3' end would be a wrong
/// answer with no error to point at. Q40 (1 error in 10,000) is the
/// conventional "as good as it gets" Illumina score and comfortably clears
/// every cutoff a user is likely to pass.
const SYNTHETIC_FASTA_QUALITY: u8 = b'I';

/// Which parser `next_record` is running. Decided once, from the first
/// non-blank byte of the stream (see `sniff_format`), and never revisited:
/// a file is FASTA or FASTQ, not both, and re-sniffing per record would turn
/// a corrupt record in the middle of a large file into a silent format
/// switch instead of the error it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Not yet determined -- the stream has not been read from.
    Unknown,
    Fastq,
    Fasta,
}

/// Bytes that may legally precede the first record marker and carry no
/// content of their own. Skipped when deciding the format so that a file
/// with a blank first line is still recognized by its first *real* byte.
fn is_blank(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r' | b' ' | b'\t')
}

/// Decides the format from a peeked prefix of the stream: the first
/// non-blank byte is `>` for FASTA, and anything else (including `@`) is
/// handled by the FASTQ parser, which keeps its existing diagnostics for
/// genuinely malformed input.
///
/// Content, not extension: `.fa`, `.fasta`, `.fna`, `.fastq`, `.txt`, no
/// extension at all, and a pipe with no name whatsoever all reach this
/// function the same way, and a mislabelled file is read for what it
/// actually contains.
fn sniff_format(prefix: &[u8]) -> Format {
    match prefix.iter().copied().find(|&b| !is_blank(b)) {
        Some(b'>') => Format::Fasta,
        // Nothing but blanks in the peeked prefix means an empty (or
        // whitespace-only) file: either parser yields no records, so the
        // choice does not matter and FASTQ keeps today's behavior.
        _ => Format::Fastq,
    }
}

/// A boxed, thread-movable byte stream. The pipeline's producer runs on its
/// own `'static` thread, which is why `Send + 'static` is part of the type
/// rather than a bound applied at each call site.
pub type BoxedStream = Box<dyn BufRead + Send + 'static>;

/// The size of the read buffer every input stream gets, gzip or plain.
const STREAM_BUFFER_BYTES: usize = 128 * 1024;

/// Whether a stream's first bytes are the gzip magic number (`1f 8b`).
///
/// Extension-based detection is fine for a named file and impossible for a
/// pipe, which is what `fastdna -i -` is: `fasterq-dump ... | gzip -c |
/// fastdna -i -` has no filename at all. Without this, those bytes reach the
/// FASTQ parser and fail with a malformed-record error that says nothing
/// about the real problem.
pub fn looks_gzipped(prefix: &[u8]) -> bool {
    prefix.starts_with(&[0x1f, 0x8b])
}

/// Wraps `stream` in a gzip decoder if -- and only if -- its own first bytes
/// say it is gzip. The peek goes through `BufRead::fill_buf`, which consumes
/// nothing, so a plain stream is handed back with every byte still in front
/// of it.
///
/// `MultiGzDecoder`, not `GzDecoder`: see `FastqReader::from_path` for why
/// multi-member support is required rather than merely nice.
pub fn decode_sniffing_gzip(mut stream: BoxedStream) -> io::Result<BoxedStream> {
    let gzipped = looks_gzipped(stream.fill_buf()?);
    if gzipped {
        Ok(Box::new(BufReader::with_capacity(
            STREAM_BUFFER_BYTES,
            MultiGzDecoder::new(stream),
        )))
    } else {
        Ok(stream)
    }
}

/// One input in a run: a file on disk, or the process's standard input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSpec {
    File(PathBuf),
    /// Written `-` on the command line. Exclusive: a run reads stdin or
    /// files, never both (see `Cli::validate`).
    Stdin,
}

/// The `-` that means "read standard input" on the command line.
pub const STDIN_ARG: &str = "-";

impl InputSpec {
    /// Interprets one `--input` value: the literal `-` is stdin, anything
    /// else is a path. A file genuinely named `-` is unreachable this way,
    /// which is the same trade every tool that supports pipes makes.
    pub fn from_arg(path: &Path) -> Self {
        if path.as_os_str() == STDIN_ARG {
            InputSpec::Stdin
        } else {
            InputSpec::File(path.to_path_buf())
        }
    }

    /// The name to put in an error message about this input.
    pub fn display_path(&self) -> PathBuf {
        match self {
            InputSpec::File(path) => path.clone(),
            InputSpec::Stdin => PathBuf::from("<stdin>"),
        }
    }

    /// Opens the stream, decompressing gzip. A named file is decided by
    /// extension (unchanged from `FastqReader::from_path`); stdin has no
    /// extension to consult, so it is decided by magic bytes.
    fn open(&self) -> io::Result<BoxedStream> {
        match self {
            InputSpec::File(path) => {
                let file = File::open(path)?;
                let is_gzipped = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.eq_ignore_ascii_case("gz"));
                if is_gzipped {
                    Ok(Box::new(BufReader::with_capacity(
                        STREAM_BUFFER_BYTES,
                        MultiGzDecoder::new(file),
                    )))
                } else {
                    Ok(Box::new(BufReader::with_capacity(STREAM_BUFFER_BYTES, file)))
                }
            }
            // `io::stdin()`, not `stdin().lock()`: the lock guard is not
            // `Send`, and this stream is moved onto the producer thread.
            InputSpec::Stdin => decode_sniffing_gzip(Box::new(BufReader::with_capacity(
                STREAM_BUFFER_BYTES,
                io::stdin(),
            ))),
        }
    }
}

/// Streaming sequence reader supporting FASTQ and FASTA, plain-text and
/// gzip. The format is sniffed from the stream's own first non-blank byte on
/// the first `next_record` call, so every existing constructor -- `new`,
/// `from_path` -- gains FASTA support without a signature change and without
/// callers having to say which format they have.
pub struct FastqReader<R: BufRead> {
    reader: R,
    line_buf: Vec<u8>,
    format: Format,
    /// FASTA only: the `>` header line already read past the end of the
    /// previous record. A FASTA sequence is terminated by the *next* header,
    /// which can only be discovered by reading it, so it is parked here for
    /// the following call instead of being pushed back into the stream.
    pending_header: Option<Vec<u8>>,
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
            format: Format::Unknown,
            pending_header: None,
        })
    }
}

impl<R: BufRead> FastqReader<R> {
    pub fn new(reader: R) -> Self {
        FastqReader {
            reader,
            line_buf: Vec::with_capacity(512),
            format: Format::Unknown,
            pending_header: None,
        }
    }

    /// Reads the next record, in whichever format this stream turned out to
    /// hold. See `sniff_format` for how that is decided and `next_fastq_record`
    /// / `next_fasta_record` for the two parsers.
    pub fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        if self.format == Format::Unknown {
            // `fill_buf` peeks without consuming, so whichever parser runs
            // next still sees the very first byte of the stream. A prefix
            // shorter than the whole leading run of blanks is not a real
            // case (the first fill is at least a page, and no file starts
            // with kilobytes of whitespace), and `sniff_format` degrades to
            // today's FASTQ behavior if it ever were.
            let prefix = self.reader.fill_buf()?;
            self.format = sniff_format(prefix);
        }

        match self.format {
            Format::Fasta => self.next_fasta_record(),
            // `Unknown` cannot survive the block above; treating it as FASTQ
            // keeps this match exhaustive without an unreachable panic.
            Format::Fastq | Format::Unknown => self.next_fastq_record(),
        }
    }

    /// Reads the next FASTA record: a `>` header followed by one or more
    /// sequence lines, which are concatenated (wrapped FASTA is the norm for
    /// assemblies -- 60 or 80 columns per line). Blank lines are skipped.
    /// The sequence alphabet is passed through verbatim, lowercase and
    /// ambiguity codes included, for the same reason the FASTQ parser does
    /// not validate it: `kmer::extract_canonical_kmers` already handles both.
    fn next_fasta_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        // The header is either one left over from the previous call (the
        // line that ended that record's sequence) or the next non-blank
        // line of the stream.
        let id = match self.pending_header.take() {
            Some(header) => header,
            None => loop {
                self.line_buf.clear();
                if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
                    return Ok(None);
                }
                strip_newline(&mut self.line_buf);
                if self.line_buf.is_empty() {
                    continue;
                }
                if !self.line_buf.starts_with(b">") {
                    return Err(FastqReadError::Malformed(format!(
                        "FASTA header line must start with '>', got {:?}",
                        preview_for_error(&self.line_buf)
                    )));
                }
                break self.line_buf.clone();
            },
        };

        let mut seq: Vec<u8> = Vec::with_capacity(256);
        loop {
            self.line_buf.clear();
            if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
                break;
            }
            strip_newline(&mut self.line_buf);
            if self.line_buf.is_empty() {
                continue;
            }
            if self.line_buf.starts_with(b">") {
                self.pending_header = Some(self.line_buf.clone());
                break;
            }
            seq.extend_from_slice(&self.line_buf);
        }

        if seq.is_empty() {
            // A header with nothing under it is not a zero-length sequence
            // to count silently: it means the file is truncated or was
            // concatenated wrongly, and the caller needs to know which
            // record so it can be found.
            return Err(FastqReadError::Malformed(format!(
                "FASTA record {:?} has no sequence",
                preview_for_error(&id)
            )));
        }

        let qual = vec![SYNTHETIC_FASTA_QUALITY; seq.len()];
        Ok(Some(FastqRecord { id, seq, qual }))
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
    fn next_fastq_record(&mut self) -> ReadResult<Option<FastqRecord>> {
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

/// What the pipeline's producer thread needs from an input: records, one at
/// a time, and enough context to blame the right file when one is bad.
///
/// A trait rather than a concrete type because the two implementations are
/// genuinely different shapes -- `FastqReader<R>` is one stream with no path
/// of its own (in-memory callers, the FFI, wasm), while `MultiSourceReader`
/// spans several named files -- and because it keeps the producer's error
/// handling in one place instead of forking it per input kind.
pub trait RecordSource: Send + 'static {
    fn next_record(&mut self) -> ReadResult<Option<FastqRecord>>;

    /// The file currently being read, and how many records have been read
    /// *from that file* so far -- so the record that failed is this count
    /// plus one, counted per file rather than across the whole run. `None`
    /// means the source has no path of its own, and the pipeline falls back
    /// to the `source` path its caller supplied plus its own running read
    /// count -- today's behavior, unchanged.
    fn current_source(&self) -> Option<(PathBuf, u64)> {
        None
    }

    /// Rejects a source that cannot produce anything, before any work is
    /// scheduled. The default accepts everything; `MultiSourceReader`
    /// overrides it so an empty input list is an error rather than a
    /// successful run that silently counted zero reads.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl<R: BufRead + Send + 'static> RecordSource for FastqReader<R> {
    fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        FastqReader::next_record(self)
    }
}

/// Reads several inputs back to back as one record stream.
///
/// Files are opened lazily, one at a time, and closed as soon as they are
/// exhausted: a run over 200 lanes holds one file handle, not 200. Each
/// file's format (FASTA or FASTQ) and compression are decided
/// independently, so a run may mix them freely.
pub struct MultiSourceReader {
    remaining: std::vec::IntoIter<InputSpec>,
    /// The input being read now, if one is open. Kept separately from
    /// `current_path` so that a file which failed to *open* can still be
    /// named in the resulting error.
    current: Option<FastqReader<BoxedStream>>,
    current_path: Option<PathBuf>,
    records_in_current: u64,
    total_inputs: usize,
}

impl MultiSourceReader {
    pub fn new(inputs: Vec<InputSpec>) -> Self {
        MultiSourceReader {
            total_inputs: inputs.len(),
            remaining: inputs.into_iter(),
            current: None,
            current_path: None,
            records_in_current: 0,
        }
    }

    /// Convenience constructor for a list of plain paths, applying the `-`
    /// convention (see `InputSpec::from_arg`).
    pub fn from_paths<P: AsRef<Path>>(paths: Vec<P>) -> Self {
        Self::new(paths.iter().map(|p| InputSpec::from_arg(p.as_ref())).collect())
    }

    pub fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        loop {
            if self.current.is_none() {
                let Some(spec) = self.remaining.next() else {
                    return Ok(None);
                };
                // Recorded *before* the open attempt so that a failure to
                // open is still attributable to this file.
                self.current_path = Some(spec.display_path());
                self.records_in_current = 0;
                self.current = Some(FastqReader::new(spec.open()?));
            }

            // `current` was just set, or was already `Some` -- but reach for
            // it fallibly rather than unwrapping, since `unwrap` is denied
            // crate-wide and a panic here would cross the FFI boundary.
            let Some(reader) = self.current.as_mut() else {
                return Ok(None);
            };

            match reader.next_record()? {
                Some(record) => {
                    self.records_in_current += 1;
                    return Ok(Some(record));
                }
                // This input is exhausted; drop it (closing the file) and
                // move to the next one. `current_path` is deliberately left
                // pointing at it so the last file read stays nameable.
                None => self.current = None,
            }
        }
    }
}

impl RecordSource for MultiSourceReader {
    fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        MultiSourceReader::next_record(self)
    }

    fn current_source(&self) -> Option<(PathBuf, u64)> {
        self.current_path.clone().map(|path| (path, self.records_in_current))
    }

    fn validate(&self) -> Result<()> {
        if self.total_inputs == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "input",
                reason: "no input files were given".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: unwrap/expect denial is
// about production paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn reader_over(text: &str) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(text.as_bytes().to_vec()))
    }

    fn collect(mut reader: FastqReader<impl BufRead>) -> Vec<FastqRecord> {
        let mut out = Vec::new();
        while let Some(record) = reader.next_record().expect("valid input") {
            out.push(record);
        }
        out
    }

    #[test]
    fn fasta_single_line_records_are_read_with_synthetic_q40_quality() {
        let records = collect(reader_over(">chr1 first\nACGTACGT\n>chr2\nTTTT\n"));

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, b">chr1 first");
        assert_eq!(records[0].seq, b"ACGTACGT");
        assert_eq!(
            records[0].qual, b"IIIIIIII",
            "a FASTA record must carry a synthetic Q40 quality string of its own length"
        );
        assert_eq!(records[1].seq, b"TTTT");
        assert_eq!(records[1].qual, b"IIII");
    }

    /// A synthetic Q40 quality string exists so that quality trimming is a
    /// no-op on FASTA input: an assembly has no per-base qualities and must
    /// not be silently truncated by the default `-q 20`.
    #[test]
    fn quality_trimming_is_a_no_op_on_a_fasta_record() {
        let mut records = collect(reader_over(">g\nACGTACGTACGT\n"));
        let record = &mut records[0];
        record.quality_trim_end(20.0, 4);
        assert_eq!(record.seq, b"ACGTACGTACGT");
    }

    #[test]
    fn fasta_wrapped_sequence_lines_are_joined_into_one_record() {
        let records = collect(reader_over(">chr1\nACGT\nACGT\nAC\n>chr2\nGG\n"));

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGTACGTAC");
        assert_eq!(records[0].qual.len(), 10);
        assert_eq!(records[1].seq, b"GG");
    }

    #[test]
    fn fasta_crlf_line_endings_are_stripped() {
        let records = collect(reader_over(">chr1\r\nACGT\r\nACGT\r\n>chr2\r\nTT\r\n"));

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGTACGT");
        assert_eq!(records[1].seq, b"TT");
    }

    #[test]
    fn an_empty_fasta_file_yields_no_records_and_no_error() {
        assert!(collect(reader_over("")).is_empty());
    }

    /// Blank lines between records are legal FASTA and must not end a run or
    /// be mistaken for a record boundary.
    #[test]
    fn blank_lines_between_fasta_records_are_ignored() {
        let records = collect(reader_over(">a\nACGT\n\n\n>b\nTTTT\n\n"));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGT");
        assert_eq!(records[1].seq, b"TTTT");
    }

    /// The reader must not normalize the alphabet: `kmer::encode_base`
    /// already accepts lowercase and treats `N` as a window reset, and QC
    /// counts GC case-insensitively. Rewriting bases here would hide soft-
    /// masked regions from anything that later wants to see them.
    #[test]
    fn fasta_lowercase_and_ambiguous_bases_are_passed_through_verbatim() {
        let records = collect(reader_over(">masked\nacgtNNNNacgt\n"));
        assert_eq!(records[0].seq, b"acgtNNNNacgt");
        assert_eq!(records[0].qual.len(), 12);
    }

    #[test]
    fn a_fasta_header_with_no_sequence_is_malformed() {
        let mut reader = reader_over(">empty\n>next\nACGT\n");
        match reader.next_record() {
            Err(FastqReadError::Malformed(reason)) => {
                assert!(
                    reason.contains("empty"),
                    "the reason must quote the offending header: {reason}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn a_trailing_fasta_header_with_no_sequence_is_malformed() {
        let mut reader = reader_over(">a\nACGT\n>dangling\n");
        assert!(reader.next_record().expect("first record is fine").is_some());
        match reader.next_record() {
            Err(FastqReadError::Malformed(reason)) => {
                assert!(reason.contains("dangling"), "reason: {reason}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    /// The format is decided by content, not by extension: a `.fastq` file
    /// holding FASTA (and vice versa) must still be read correctly.
    #[test]
    fn fastq_input_is_still_parsed_as_fastq() {
        let records = collect(reader_over("@r1\nACGT\n+\n!!!!\n@r2\nTTTT\n+\nIIII\n"));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, b"@r1");
        assert_eq!(records[0].qual, b"!!!!");
    }

    /// Sniffing must not swallow the FASTQ reader's existing diagnostics: a
    /// file starting with neither `>` nor `@` is still a malformed FASTQ.
    #[test]
    fn a_file_starting_with_neither_marker_is_still_a_malformed_fastq() {
        let mut reader = reader_over("ACGT\nACGT\n");
        match reader.next_record() {
            Err(FastqReadError::Malformed(reason)) => {
                assert!(reason.contains('@'), "reason: {reason}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    fn gzip_bytes(plain: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as _;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(plain).unwrap();
        encoder.finish().unwrap()
    }

    /// A throwaway directory under the system temp dir, removed on drop.
    struct Fixture {
        dir: std::path::PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("fastdna_fastq_rs_{name}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Fixture { dir }
        }

        fn write(&self, name: &str, bytes: &[u8]) -> std::path::PathBuf {
            let path = self.dir.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn gzipped_fasta_is_detected_through_the_decompressed_bytes() {
        let fx = Fixture::new("fasta_gz");
        let path = fx.write("genome.fasta.gz", &gzip_bytes(b">chr1\nACGT\nACGT\n>chr2\nGGGG\n"));

        let records = collect(FastqReader::from_path(&path).unwrap());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGTACGT");
        assert_eq!(records[1].seq, b"GGGG");
    }

    // -- gzip magic-byte sniffing -------------------------------------

    /// stdin has no extension, so the only way to know it carries gzip is
    /// the stream's own two magic bytes.
    #[test]
    fn the_gzip_magic_bytes_are_recognized() {
        assert!(looks_gzipped(&[0x1f, 0x8b, 0x08, 0x00]));
        assert!(looks_gzipped(&gzip_bytes(b"@r\nACGT\n+\nIIII\n")));
    }

    #[test]
    fn plain_text_is_not_mistaken_for_gzip() {
        for plain in [&b"@r1\nACGT\n+\nIIII\n"[..], b">chr1\nACGT\n", b"", b"\x1f", b"\x8b\x1f"] {
            assert!(!looks_gzipped(plain), "misdetected as gzip: {plain:?}");
        }
    }

    #[test]
    fn a_gzip_stream_with_no_filename_is_decoded_by_its_magic_bytes() {
        let compressed = gzip_bytes(b"@r1\nACGTACGT\n+\nIIIIIIII\n");
        let reader = decode_sniffing_gzip(Box::new(Cursor::new(compressed))).unwrap();
        let records = collect(FastqReader::new(reader));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, b"ACGTACGT");
    }

    #[test]
    fn a_plain_stream_with_no_filename_is_passed_through_untouched() {
        let reader = decode_sniffing_gzip(Box::new(Cursor::new(b">g\nACGT\n".to_vec()))).unwrap();
        let records = collect(FastqReader::new(reader));
        assert_eq!(records[0].seq, b"ACGT");
    }

    // -- multiple inputs ----------------------------------------------

    fn collect_multi(mut reader: MultiSourceReader) -> Vec<FastqRecord> {
        let mut out = Vec::new();
        while let Some(record) = reader.next_record().expect("valid input") {
            out.push(record);
        }
        out
    }

    /// Compression is decided per file, by that file's own extension, so a
    /// run may mix `.fastq.gz` lanes with a plain `.fastq` one -- which is
    /// exactly what a partially-decompressed download looks like.
    #[test]
    fn gzipped_and_plain_files_can_be_mixed_in_one_input_list() {
        let fx = Fixture::new("mixed_gz");
        let plain = fx.write("a.fastq", b"@a1\nACGT\n+\nIIII\n");
        let gz = fx.write("b.fastq.gz", &gzip_bytes(b"@b1\nTTTT\n+\nIIII\n@b2\nGGGG\n+\nIIII\n"));
        let fasta_gz = fx.write("c.fa.gz", &gzip_bytes(b">c1\nCCCC\n"));

        let records = collect_multi(MultiSourceReader::from_paths(vec![plain, gz, fasta_gz]));

        let seqs: Vec<&[u8]> = records.iter().map(|r| r.seq.as_slice()).collect();
        assert_eq!(seqs, vec![&b"ACGT"[..], b"TTTT", b"GGGG", b"CCCC"]);
    }

    /// Record numbers restart at 1 for each file: a user told to look at
    /// "record 2" needs that to mean the second record of the named file.
    #[test]
    fn the_current_source_tracks_the_file_and_its_own_record_number() {
        let fx = Fixture::new("source_tracking");
        let a = fx.write("a.fastq", b"@a1\nACGT\n+\nIIII\n@a2\nTTTT\n+\nIIII\n");
        let b = fx.write("b.fastq", b"@b1\nGGGG\n+\nIIII\n");

        let mut reader = MultiSourceReader::from_paths(vec![a.clone(), b.clone()]);

        reader.next_record().unwrap().unwrap();
        assert_eq!(reader.current_source(), Some((a.clone(), 1)));
        reader.next_record().unwrap().unwrap();
        assert_eq!(reader.current_source(), Some((a, 2)));
        reader.next_record().unwrap().unwrap();
        assert_eq!(reader.current_source(), Some((b.clone(), 1)));

        assert!(reader.next_record().unwrap().is_none(), "both files are exhausted");
        assert_eq!(reader.current_source(), Some((b, 1)), "the last file stays named");
    }

    /// An unopenable file must surface as an I/O failure naming that file --
    /// not as a malformed-record error blaming the bytes it never read.
    #[test]
    fn an_unopenable_file_reports_io_and_names_itself() {
        let fx = Fixture::new("unopenable");
        let good = fx.write("good.fastq", b"@a1\nACGT\n+\nIIII\n");
        let missing = fx.dir.join("nope.fastq");

        let mut reader = MultiSourceReader::from_paths(vec![good, missing.clone()]);
        reader.next_record().unwrap().unwrap();

        match reader.next_record() {
            Err(FastqReadError::Io(_)) => {}
            other => panic!("expected Io, got {other:?}"),
        }
        assert_eq!(
            reader.current_source().map(|(p, _)| p),
            Some(missing),
            "the error must be attributable to the file that could not be opened"
        );
    }

    #[test]
    fn an_empty_input_list_yields_no_records() {
        assert!(collect_multi(MultiSourceReader::new(Vec::new())).is_empty());
    }
}
