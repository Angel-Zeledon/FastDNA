// src/fastq.rs

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use flate2::read::MultiGzDecoder;

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

    #[test]
    fn gzipped_fasta_is_detected_through_the_decompressed_bytes() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as _;

        let dir = std::env::temp_dir().join("fastdna_fasta_gz_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("genome.fasta.gz");

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b">chr1\nACGT\nACGT\n>chr2\nGGGG\n").unwrap();
        let compressed = encoder.finish().unwrap();
        std::fs::write(&path, compressed).unwrap();

        let records = collect(FastqReader::from_path(&path).unwrap());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGTACGT");
        assert_eq!(records[1].seq, b"GGGG");

        let _ = std::fs::remove_file(&path);
    }
}
