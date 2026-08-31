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
///
/// `Default` is derived so a caller can hold one record across a whole
/// stream and refill it through `FastqReader::next_record_into`, which
/// reuses the three buffers instead of allocating three fresh `Vec`s per
/// record. See that method for the counts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

    /// Trims low-quality bases from the 3' end: the read is shortened until
    /// the trailing `window_size` bases average at least `min_qual`.
    ///
    /// The window sum is carried between steps instead of being recomputed.
    /// Sliding the window one base towards the 5' end drops exactly one base
    /// and admits exactly one, so a step costs one subtraction plus one
    /// addition rather than `window_size` additions. A 150 bp read whose
    /// whole tail has to be trimmed at the CLI defaults (`-q 20`, window 4)
    /// costs 4 + 2*146 = 296 phred lookups instead of 147*4 = 588, and the
    /// saving grows linearly with `window_size`. The first window still costs
    /// `window_size` additions and the loop still breaks on the first passing
    /// window, so a read that needs no trimming -- the common case -- does
    /// exactly the work it did before, never more.
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

        // Hoisted out of the loop because it is loop-invariant: one
        // usize -> f64 conversion per record instead of one per step. The
        // division itself stays -- rewriting `sum / window_len >= min_qual`
        // as `sum >= min_qual * window_len` rounds differently and could
        // move a trim boundary by a base, which is a behaviour change, not
        // an optimization.
        let window_len = window_size as f64;
        let mut sum: u64 = self.qual[end_pos - window_size..end_pos]
            .iter()
            .map(|&q| Self::phred_score(q) as u64)
            .sum();

        loop {
            // Invariant: `end_pos >= window_size`, and `sum` is the phred sum
            // of `qual[end_pos - window_size..end_pos]`.
            if sum as f64 / window_len >= min_qual {
                break;
            }
            end_pos -= 1;
            if end_pos < window_size {
                break;
            }
            // The base leaving the window is the one now just past its end
            // (`end_pos`); the base entering is its new first
            // (`end_pos - window_size`). Both indices are < the original
            // `end_pos`, which was clamped to `qual.len()` above.
            sum -= Self::phred_score(self.qual[end_pos]) as u64;
            sum += Self::phred_score(self.qual[end_pos - window_size]) as u64;
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

/// The outcome of `FastqReader::skip_marked_line`: a line consumed without
/// being copied, a line whose first byte was not the expected marker (and
/// which is therefore left *unconsumed* for the caller's error path), or
/// end of stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkedLine {
    Eof,
    Mismatch,
    Skipped,
}

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

// This crate's gzip handling -- sniffing (`looks_gzipped`), stream-wrapping
// (`decode_sniffing_gzip`), and the two call sites below that build a
// `MultiGzDecoder` directly (`from_path`, `InputSpec::open`) -- all go
// through `flate2`. `Cargo.toml` pins `flate2 = "1.0"` with no backend
// feature enabled by default, so `flate2` uses its default backend,
// `miniz_oxide`: pure Rust, no C toolchain needed to build this crate, and
// measurably slower at inflate than the C implementations `flate2` can
// also use.
//
// Decompression happens on the pipeline's producer thread, in series with
// reading -- see the comment inside `from_path` below -- so for real gzip
// input this is on the critical path of every run. It does not show up in
// `docs/BENCHMARKS.md` because that file's reference input is plain FASTQ;
// real SRA/ENA downloads are gzip, so a real user pays this cost on every
// run that number does not.
//
// `docs/audit/performance-v2.md` (G-4) proposed making flate2's `zlib-ng`
// feature unconditional, citing a claimed 2-3x inflate speedup, but said
// in its own words to measure before shipping it ("hay que medirlo antes
// de darlo por bueno"). Measured, not assumed: built this crate twice in
// the same `rust:1-slim-bookworm` container this project's Docker-only
// verification already uses (there is no local Rust toolchain to measure
// natively), once with default features and once with `--features
// zlib-ng`, and ran a harness that calls `FastqReader::from_path` and
// drains every record via `next_record_into` -- this crate's actual read
// path, not a synthetic zlib microbenchmark -- against a realistic 348 MB
// synthetic `.fastq.gz` (1,066,666 reads, 160M bases, generated with
// `scripts/bench/generate_reads_large.py`, gzip level 6, ~75 MB
// compressed). Three runs each, same container, same file, wall clock end
// to end:
//
//                run 1     run 2     run 3    median
//   miniz_oxide  8.24 s    8.25 s    6.64 s    8.24 s
//   zlib-ng      2.05 s    2.16 s    4.38 s    2.16 s
//
// zlib-ng wins every single pairing, including the most adversarial one
// (its slowest run, 4.38 s, against miniz_oxide's fastest, 6.64 s: still
// 1.5x faster). Median-to-median is 3.8x; mean-to-mean, which does not
// discard either backend's one noisy run, is 2.7x. Either way this clears
// the plan's claimed 2-3x, on this machine, on this file.
//
// The cost the plan flagged is real: `flate2/zlib-ng` pulls in
// `libz-ng-sys`, which needs a C compiler and `cmake` at build time --
// both had to be `apt-get install`ed into the container above; neither is
// there by default. Today this crate needs neither. `compile.bat`, the
// documented native-Windows build path, is a bare `cargo build --release`
// with no toolchain setup of its own, and this session has no way to
// confirm a C toolchain and cmake are present wherever that script
// actually gets run, nor to test the macOS legs of
// `.github/workflows/wheels.yml` at all -- only Linux x86_64, in Docker,
// was verified here. Making zlib-ng unconditional would trade a measured,
// real speedup for an unmeasured, real risk of breaking a build this
// session cannot see.
//
// So: opt-in, not default. `[features] zlib-ng = ["flate2/zlib-ng"]` in
// `Cargo.toml` turns it on (`cargo build --release --features zlib-ng`,
// or for the Python wheel `maturin build --release --features
// python,zlib-ng`) on a machine known to have a C toolchain and cmake --
// true of the manylinux containers and Bioconda that
// `docs/audit/performance-v2.md` itself names. `default = []` leaves
// every build that does not opt in -- `compile.bat`, plain `cargo
// build`/`test`/`clippy`, and every leg of `wheels.yml` as it stands
// today -- on miniz_oxide, unchanged.

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
/// multi-member support is required rather than merely nice. Which
/// *backend* flate2 uses to inflate (miniz_oxide vs zlib-ng) is a separate,
/// build-time choice -- see the comment above `STREAM_BUFFER_BYTES` in this
/// file for the G-4 measurement and decision.
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
    ///
    /// The gzip backend compiled in (miniz_oxide by default, zlib-ng if
    /// built with `--features zlib-ng`) is the same for every branch here;
    /// see the comment above `STREAM_BUFFER_BYTES` in this file.
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
    /// Whether the FASTQ parser fills `FastqRecord::id`. See
    /// `set_keep_ids`; `true` in every constructor, so no existing caller
    /// sees a change.
    keep_ids: bool,
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
        //
        // Which *backend* flate2 uses to inflate (miniz_oxide vs zlib-ng)
        // is a build-time choice, not a call here -- see the comment above
        // `STREAM_BUFFER_BYTES` for the G-4 measurement and decision.
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
            keep_ids: true,
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
            keep_ids: true,
        }
    }

    /// Tells this reader whether the caller will read `FastqRecord::id`.
    ///
    /// Default `true`: every existing caller keeps today's behaviour
    /// byte-for-byte, and the public field stays populated for anything
    /// that asserts on it. Set to `false` -- as the counting pipeline's
    /// producer does, since nothing downstream of it reads `id` -- the
    /// FASTQ parser consumes the header line without copying it (see
    /// `skip_marked_line` for the counts) and leaves `record.id` empty.
    ///
    /// FASTA is deliberately unaffected: a FASTA header is one per contig,
    /// not one per read, so its copy is not on any hot path, and the bytes
    /// are needed verbatim for `pending_header` and for the "record has no
    /// sequence" diagnostic.
    pub fn set_keep_ids(&mut self, keep: bool) {
        self.keep_ids = keep;
    }

    /// Consumes one line, up to and including its `\n`, without copying it,
    /// after checking that it starts with `marker`.
    ///
    /// Returns `Mismatch` *without consuming anything* when the first byte
    /// is not `marker`, so the caller's error path can still `read_until`
    /// the same line and quote it exactly as it did before.
    ///
    /// `read_until` scans for the newline and then copies the bytes into a
    /// caller buffer; `skip_until` runs the same scan and skips the copy.
    /// For the two FASTQ lines whose content is never read -- the `@`
    /// header when `keep_ids` is off, and the `+` separator always -- that
    /// copy is pure waste: on the 7-million-record benchmark file, headers
    /// are ~40-60 bytes each, so ~350 MB is memcpy'd and immediately
    /// discarded, plus one `strip_newline` (two `ends_with` and up to two
    /// `pop`s) per record. The added cost is the one `fill_buf` peek below,
    /// which for a `BufReader` holding buffered bytes -- 128 KB against
    /// ~250-byte records, so ~500 records per refill -- is an inlined
    /// `pos < cap` test returning an already-filled slice, not a read.
    fn skip_marked_line(&mut self, marker: u8) -> io::Result<MarkedLine> {
        match self.reader.fill_buf()?.first().copied() {
            None => Ok(MarkedLine::Eof),
            Some(byte) if byte != marker => Ok(MarkedLine::Mismatch),
            Some(_) => {
                self.reader.skip_until(b'\n')?;
                Ok(MarkedLine::Skipped)
            }
        }
    }

    /// Reads the line `skip_marked_line` reported as `Mismatch` (it left the
    /// line unconsumed) into the reader's own scratch buffer and strips its
    /// line ending, ready to be quoted in a malformed-record error. Cold
    /// path only -- one call per run, immediately before returning an error.
    fn read_mismatched_line(&mut self) -> io::Result<()> {
        self.line_buf.clear();
        self.reader.read_until(b'\n', &mut self.line_buf)?;
        strip_newline(&mut self.line_buf);
        Ok(())
    }

    /// Reads the next record, in whichever format this stream turned out to
    /// hold. See `sniff_format` for how that is decided and
    /// `next_fastq_record_into` / `next_fasta_record_into` for the two
    /// parsers.
    ///
    /// Allocates a fresh record. `next_record_into` is the same parse
    /// writing into buffers the caller already owns; prefer it on any path
    /// that reads more than a handful of records.
    pub fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        let mut record = FastqRecord::default();
        if self.next_record_into(&mut record)? {
            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// Reads the next record into `record`'s existing buffers, returning
    /// `false` at end of stream. The three `Vec`s are cleared and refilled,
    /// so a caller that keeps one record across a whole stream pays the
    /// allocation once rather than three times per record -- 21 million
    /// allocate/free pairs on the 7-million-record benchmark file.
    ///
    /// The record is left in an unspecified (cleared) state when this
    /// returns `false` or an error; only a `true` return promises contents.
    ///
    /// Note for callers that batch: a record moved into a batch cannot be
    /// refilled, so exploiting this needs the buffers handed back once the
    /// batch has been consumed (see `RecordSource::next_record_into`).
    pub fn next_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        if self.format == Format::Unknown {
            // Sniffed once per stream, not once per record: `format` is
            // sticky, so all that survives in the per-record path is this
            // one enum compare. `fill_buf` peeks without consuming, so
            // whichever parser runs next still sees the very first byte of
            // the stream. A prefix shorter than the whole leading run of
            // blanks is not a real case (the first fill is at least a page,
            // and no file starts with kilobytes of whitespace), and
            // `sniff_format` degrades to today's FASTQ behavior if it ever
            // were.
            let prefix = self.reader.fill_buf()?;
            self.format = sniff_format(prefix);
        }

        match self.format {
            Format::Fasta => self.next_fasta_record_into(record),
            // `Unknown` cannot survive the block above; treating it as FASTQ
            // keeps this match exhaustive without an unreachable panic.
            Format::Fastq | Format::Unknown => self.next_fastq_record_into(record),
        }
    }

    /// Reads the next FASTA record: a `>` header followed by one or more
    /// sequence lines, which are concatenated (wrapped FASTA is the norm for
    /// assemblies -- 60 or 80 columns per line). Blank lines are skipped.
    /// The sequence alphabet is passed through verbatim, lowercase and
    /// ambiguity codes included, for the same reason the FASTQ parser does
    /// not validate it: `kmer::extract_canonical_kmers` already handles both.
    fn next_fasta_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        // The header is either one left over from the previous call (the
        // line that ended that record's sequence) or the next non-blank
        // line of the stream. Either way it lands in `record.id` directly:
        // no `line_buf.clone()`, so no per-record allocation for it.
        record.id.clear();
        match self.pending_header.take() {
            Some(header) => record.id.extend_from_slice(&header),
            None => loop {
                self.line_buf.clear();
                if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
                    return Ok(false);
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
                record.id.extend_from_slice(&self.line_buf);
                break;
            },
        }

        record.seq.clear();
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
            record.seq.extend_from_slice(&self.line_buf);
        }

        if record.seq.is_empty() {
            // A header with nothing under it is not a zero-length sequence
            // to count silently: it means the file is truncated or was
            // concatenated wrongly, and the caller needs to know which
            // record so it can be found.
            return Err(FastqReadError::Malformed(format!(
                "FASTA record {:?} has no sequence",
                preview_for_error(&record.id)
            )));
        }

        // `clear` + `resize` rather than `vec![..; len]`: on a reused record
        // this reuses the buffer the previous record left behind instead of
        // allocating a new one.
        record.qual.clear();
        record.qual.resize(record.seq.len(), SYNTHETIC_FASTA_QUALITY);
        Ok(true)
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
    ///
    /// Each of the three kept lines is read straight into its field on
    /// `record`. The previous version read every line into `self.line_buf`
    /// and then `clone()`d it into the record, which copied the bytes twice
    /// and allocated once per field: on the 7-million-record benchmark file
    /// that is 21 million redundant `memcpy`s, plus 21 million
    /// allocate/free pairs that only `next_record` (which owns a throwaway
    /// record) still pays. `read_until` is kept rather than hand-rolled so
    /// the newline scan stays std's word-at-a-time `memchr` and the number
    /// of `fill_buf` calls per record is unchanged at four.
    ///
    /// When `keep_ids` is off the header line is consumed by
    /// `skip_marked_line` instead, which does the same newline scan without
    /// the copy and without the `strip_newline` that only existed to make
    /// the stored id presentable.
    fn next_fastq_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        record.id.clear();
        if self.keep_ids {
            if self.reader.read_until(b'\n', &mut record.id)? == 0 {
                return Ok(false);
            }
            strip_newline(&mut record.id);
            if !record.id.starts_with(b"@") {
                return Err(FastqReadError::Malformed(format!(
                    "header line must start with '@', got {:?}",
                    preview_for_error(&record.id)
                )));
            }
        } else {
            match self.skip_marked_line(b'@')? {
                MarkedLine::Skipped => {}
                MarkedLine::Eof => return Ok(false),
                MarkedLine::Mismatch => {
                    // Cold path: nothing was consumed, so the offending line
                    // is quoted byte-for-byte as the `keep_ids` branch above
                    // quotes it.
                    self.read_mismatched_line()?;
                    return Err(FastqReadError::Malformed(format!(
                        "header line must start with '@', got {:?}",
                        preview_for_error(&self.line_buf)
                    )));
                }
            }
        }

        record.seq.clear();
        if self.reader.read_until(b'\n', &mut record.seq)? == 0 {
            return Err(FastqReadError::Malformed(
                "file ends mid-record: missing sequence line after header".to_string(),
            ));
        }
        strip_newline(&mut record.seq);

        // The separator line is the one line whose content is *never* read,
        // in either id mode, so it is always skipped rather than copied:
        // `skip_marked_line` checks the `+` from the peeked buffer and then
        // consumes the line with no `memcpy` and no `strip_newline`. Files
        // that repeat the header after the `+` -- an older but still common
        // Illumina convention -- make this the same ~40-60 bytes per record
        // the header skip saves. The strip is still done before the error
        // message is built, so a malformed separator is quoted back
        // byte-for-byte as it was before.
        match self.skip_marked_line(b'+')? {
            MarkedLine::Skipped => {}
            MarkedLine::Eof => {
                return Err(FastqReadError::Malformed(
                    "file ends mid-record: missing separator line after sequence".to_string(),
                ))
            }
            MarkedLine::Mismatch => {
                self.read_mismatched_line()?;
                return Err(FastqReadError::Malformed(format!(
                    "separator line must start with '+', got {:?}",
                    preview_for_error(&self.line_buf)
                )));
            }
        }

        record.qual.clear();
        if self.reader.read_until(b'\n', &mut record.qual)? == 0 {
            return Err(FastqReadError::Malformed(
                "file ends mid-record: missing quality line after separator".to_string(),
            ));
        }
        strip_newline(&mut record.qual);

        if record.seq.len() != record.qual.len() {
            return Err(FastqReadError::Malformed(format!(
                "sequence length {} does not match quality length {}",
                record.seq.len(),
                record.qual.len()
            )));
        }

        Ok(true)
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

    /// Reads the next record into buffers the caller already owns,
    /// returning `false` at end of stream. Semantics are identical to
    /// `next_record`; only the allocation is different.
    ///
    /// This is what the pipeline's producer uses: `next_record` allocates
    /// `id`, `seq` and `qual` fresh for every record and the consuming
    /// worker frees them, which is 21 million allocate/free pairs on the
    /// 7-million-record benchmark file. The producer gets its record
    /// buffers back over a non-blocking recycling channel once a worker has
    /// consumed a batch (see `pipeline.rs`), because a record pushed into a
    /// batch has been moved away and cannot be refilled.
    ///
    /// The default implementation is the allocating one, so an implementor
    /// with no buffer of its own keeps working unchanged.
    fn next_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        match self.next_record()? {
            Some(fresh) => {
                *record = fresh;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Tells the source whether the caller will read `FastqRecord::id`. See
    /// `FastqReader::set_keep_ids` for what turning it off buys and for the
    /// FASTA carve-out.
    ///
    /// The default is a no-op -- ids are kept -- so an implementor with no
    /// cheaper path, and every existing caller that never calls this, keeps
    /// today's behaviour exactly.
    fn set_keep_ids(&mut self, _keep: bool) {}

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

    fn next_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        FastqReader::next_record_into(self, record)
    }

    fn set_keep_ids(&mut self, keep: bool) {
        FastqReader::set_keep_ids(self, keep);
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
    /// Applied to every file this reader opens, including the ones it has
    /// not reached yet. `true` by default, as in `FastqReader`.
    keep_ids: bool,
}

impl MultiSourceReader {
    pub fn new(inputs: Vec<InputSpec>) -> Self {
        MultiSourceReader {
            total_inputs: inputs.len(),
            remaining: inputs.into_iter(),
            current: None,
            current_path: None,
            records_in_current: 0,
            keep_ids: true,
        }
    }

    /// See `FastqReader::set_keep_ids`. Recorded here so that files opened
    /// later in the run inherit it, and forwarded to the file currently
    /// open, if any.
    pub fn set_keep_ids(&mut self, keep: bool) {
        self.keep_ids = keep;
        if let Some(reader) = self.current.as_mut() {
            reader.set_keep_ids(keep);
        }
    }

    /// Convenience constructor for a list of plain paths, applying the `-`
    /// convention (see `InputSpec::from_arg`).
    pub fn from_paths<P: AsRef<Path>>(paths: Vec<P>) -> Self {
        Self::new(paths.iter().map(|p| InputSpec::from_arg(p.as_ref())).collect())
    }

    pub fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        let mut record = FastqRecord::default();
        if self.next_record_into(&mut record)? {
            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// The buffer-reusing form of `next_record`; see
    /// `RecordSource::next_record_into` for why a caller should prefer it.
    pub fn next_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        loop {
            if self.current.is_none() {
                let Some(spec) = self.remaining.next() else {
                    return Ok(false);
                };
                // Recorded *before* the open attempt so that a failure to
                // open is still attributable to this file.
                self.current_path = Some(spec.display_path());
                self.records_in_current = 0;
                let mut opened = FastqReader::new(spec.open()?);
                opened.set_keep_ids(self.keep_ids);
                self.current = Some(opened);
            }

            // `current` was just set, or was already `Some` -- but reach for
            // it fallibly rather than unwrapping, since `unwrap` is denied
            // crate-wide and a panic here would cross the FFI boundary.
            let Some(reader) = self.current.as_mut() else {
                return Ok(false);
            };

            if reader.next_record_into(record)? {
                self.records_in_current += 1;
                return Ok(true);
            }
            // This input is exhausted; drop it (closing the file) and
            // move to the next one. `current_path` is deliberately left
            // pointing at it so the last file read stays nameable.
            self.current = None;
        }
    }
}

impl RecordSource for MultiSourceReader {
    fn next_record(&mut self) -> ReadResult<Option<FastqRecord>> {
        MultiSourceReader::next_record(self)
    }

    fn next_record_into(&mut self, record: &mut FastqRecord) -> ReadResult<bool> {
        MultiSourceReader::next_record_into(self, record)
    }

    fn set_keep_ids(&mut self, keep: bool) {
        MultiSourceReader::set_keep_ids(self, keep);
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

    // -- buffer-reusing reads ------------------------------------------

    /// `next_record_into` exists purely to avoid three allocations per
    /// record; it must parse byte-for-byte what `next_record` parses, or the
    /// counts change the moment the pipeline adopts it.
    fn collect_into(mut reader: FastqReader<impl BufRead>) -> Vec<FastqRecord> {
        let mut out = Vec::new();
        let mut scratch = FastqRecord::default();
        while reader.next_record_into(&mut scratch).expect("valid input") {
            out.push(scratch.clone());
        }
        out
    }

    #[test]
    fn reusing_one_record_parses_exactly_what_allocating_reads_parse() {
        for text in [
            "@r1\nACGTACGT\n+\n!!!!!!!!\n@r2\nTT\n+\nII\n",
            "@r1\r\nACGTACGT\r\n+r1\r\nIIIIIIII\r\n",
            ">chr1\nACGT\nACGT\n>chr2\nGG\n",
            "",
        ] {
            assert_eq!(
                collect_into(reader_over(text)),
                collect(reader_over(text)),
                "next_record_into disagreed with next_record on {text:?}"
            );
        }
    }

    /// The failure mode a `clear`-and-refill reader can have and an
    /// allocating one cannot: a short record following a long one keeping
    /// the tail of its predecessor.
    #[test]
    fn a_short_record_after_a_long_one_leaves_no_stale_bytes() {
        let mut reader = reader_over("@long\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n@s\nAC\n+\nII\n");
        let mut record = FastqRecord::default();

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.seq, b"ACGTACGTACGTACGT");

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.id, b"@s");
        assert_eq!(record.seq, b"AC");
        assert_eq!(record.qual, b"II");
    }

    #[test]
    fn a_reused_record_survives_a_fasta_record_shorter_than_its_predecessor() {
        let mut reader = reader_over(">big\nACGTACGTACGT\n>small\nTT\n");
        let mut record = FastqRecord::default();

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.seq.len(), 12);

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.id, b">small");
        assert_eq!(record.seq, b"TT");
        assert_eq!(record.qual, b"II", "the synthetic quality must be resized, not left long");
    }

    /// A separator line is no longer newline-stripped on the success path,
    /// so the malformed case must still quote the line exactly as it did.
    #[test]
    fn a_malformed_separator_is_quoted_without_its_line_ending() {
        for text in ["@r1\nACGT\n*sep\nIIII\n", "@r1\r\nACGT\r\n*sep\r\nIIII\r\n"] {
            match reader_over(text).next_record() {
                Err(FastqReadError::Malformed(reason)) => {
                    assert!(
                        reason.contains("\"*sep\""),
                        "the offending separator must be quoted verbatim and unterminated: {reason}"
                    );
                }
                other => panic!("expected Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_separator_line_is_still_malformed() {
        match reader_over("@r1\nACGT\n\nIIII\n").next_record() {
            Err(FastqReadError::Malformed(reason)) => assert!(reason.contains('+'), "{reason}"),
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    /// `MultiSourceReader` forwards the reusing read across a file boundary,
    /// including the per-file record numbering the error messages rely on.
    #[test]
    fn the_multi_source_reader_reuses_a_record_across_file_boundaries() {
        let fx = Fixture::new("multi_into");
        let a = fx.write("a.fastq", b"@a1\nACGTACGT\n+\nIIIIIIII\n");
        let b = fx.write("b.fastq", b"@b1\nGG\n+\nII\n");

        let mut reader = MultiSourceReader::from_paths(vec![a.clone(), b.clone()]);
        let mut record = FastqRecord::default();

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.seq, b"ACGTACGT");
        assert_eq!(reader.current_source(), Some((a, 1)));

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.id, b"@b1");
        assert_eq!(record.seq, b"GG");
        assert_eq!(reader.current_source(), Some((b, 1)));

        assert!(!reader.next_record_into(&mut record).unwrap());
    }

    // -- skipping the id copy -------------------------------------------

    /// `set_keep_ids(false)` may only drop the id: every other byte of
    /// every record must be exactly what the id-keeping parser produced, or
    /// the pipeline's counts change the moment it opts in.
    #[test]
    fn skipping_ids_parses_exactly_what_keeping_them_parses() {
        for text in [
            "@r1\nACGTACGT\n+\n!!!!!!!!\n@r2\nTT\n+\nII\n",
            "@r1\r\nACGTACGT\r\n+r1\r\nIIIIIIII\r\n",
            // The separator repeating the header -- the older Illumina
            // convention, and the case the separator skip saves most on.
            "@r1\nACGT\n+r1 the whole header again\nIIII\n@r2\nGG\n+r2\nII\n",
            // A file whose last record has no trailing newline.
            "@r1\nACGT\n+\nIIII",
            "",
        ] {
            let kept = collect(reader_over(text));

            let mut reader = reader_over(text);
            reader.set_keep_ids(false);
            let mut skipped = Vec::new();
            let mut scratch = FastqRecord::default();
            while reader.next_record_into(&mut scratch).expect("valid input") {
                skipped.push(scratch.clone());
            }

            assert_eq!(skipped.len(), kept.len(), "record count changed on {text:?}");
            for (with_id, without) in kept.iter().zip(skipped.iter()) {
                assert_eq!(with_id.seq, without.seq, "seq changed on {text:?}");
                assert_eq!(with_id.qual, without.qual, "qual changed on {text:?}");
                assert!(without.id.is_empty(), "id must be left empty when skipped");
            }
        }
    }

    /// The header and separator lines are consumed without being copied, so
    /// the malformed-line diagnostics have to be rebuilt from a line that
    /// was deliberately left unconsumed. They must come out identical.
    #[test]
    fn skipping_ids_keeps_the_malformed_line_diagnostics_intact() {
        for (text, expected_fragment) in [
            ("r1\nACGT\n+\nIIII\n", "\"r1\""),
            ("@r1\nACGT\n*sep\nIIII\n", "\"*sep\""),
            ("@r1\r\nACGT\r\n*sep\r\nIIII\r\n", "\"*sep\""),
            ("@r1\nACGT\n\nIIII\n", "\"\""),
        ] {
            let mut reader = reader_over(text);
            reader.set_keep_ids(false);
            match reader.next_record() {
                Err(FastqReadError::Malformed(reason)) => assert!(
                    reason.contains(expected_fragment),
                    "on {text:?} expected {expected_fragment} in: {reason}"
                ),
                other => panic!("expected Malformed on {text:?}, got {other:?}"),
            }
        }
    }

    /// Truncation must still be reported as truncation, not as a silent
    /// short record, when the lines that detect it are skipped rather than
    /// read.
    #[test]
    fn skipping_ids_still_detects_a_truncated_record() {
        for (text, fragment) in [
            ("@r1\nACGT\n+\nIIII\n@r2\n", "sequence line"),
            ("@r1\nACGT\n+\nIIII\n@r2\nGGGG\n", "separator line"),
            ("@r1\nACGT\n+\nIIII\n@r2\nGGGG\n+\n", "quality line"),
        ] {
            let mut reader = reader_over(text);
            reader.set_keep_ids(false);
            let mut record = FastqRecord::default();
            assert!(reader.next_record_into(&mut record).expect("first record is fine"));
            match reader.next_record_into(&mut record) {
                Err(FastqReadError::Malformed(reason)) => {
                    assert!(reason.contains(fragment), "on {text:?}: {reason}")
                }
                other => panic!("expected Malformed on {text:?}, got {other:?}"),
            }
        }
    }

    /// FASTA is outside the carve-out: its headers are per contig, not per
    /// read, so they stay populated even with ids switched off.
    #[test]
    fn skipping_ids_leaves_fasta_headers_alone() {
        let mut reader = reader_over(">chr1\nACGT\n>chr2\nGG\n");
        reader.set_keep_ids(false);
        let mut record = FastqRecord::default();

        assert!(reader.next_record_into(&mut record).unwrap());
        assert_eq!(record.id, b">chr1");
        assert_eq!(record.seq, b"ACGT");
    }

    /// A multi-file run must apply the setting to files it has not opened
    /// yet, not just to the one open when it was set.
    #[test]
    fn the_multi_source_reader_applies_the_id_setting_to_later_files() {
        let fx = Fixture::new("multi_skip_ids");
        let a = fx.write("a.fastq", b"@a1\nACGTACGT\n+\nIIIIIIII\n");
        let b = fx.write("b.fastq", b"@b1\nGG\n+\nII\n");

        let mut reader = MultiSourceReader::from_paths(vec![a, b]);
        reader.set_keep_ids(false);
        let mut record = FastqRecord::default();

        assert!(reader.next_record_into(&mut record).unwrap());
        assert!(record.id.is_empty());
        assert_eq!(record.seq, b"ACGTACGT");

        assert!(reader.next_record_into(&mut record).unwrap(), "second file");
        assert!(record.id.is_empty(), "the second file must inherit the setting");
        assert_eq!(record.seq, b"GG");
    }

    // -- rolling-window quality trimming --------------------------------

    /// The reference the rolling sum replaced: re-sum the whole window at
    /// every step. Kept here so the optimized loop is pinned to the exact
    /// cut point the naive one produced, for every input below.
    fn trim_end_by_resumming(qual: &[u8], min_qual: f64, window_size: usize) -> usize {
        let mut end_pos = qual.len();
        if end_pos < window_size {
            return end_pos;
        }
        while end_pos >= window_size {
            let window = &qual[end_pos - window_size..end_pos];
            let sum: u64 = window.iter().map(|&q| FastqRecord::phred_score(q) as u64).sum();
            if sum as f64 / window_size as f64 >= min_qual {
                break;
            }
            end_pos -= 1;
        }
        end_pos
    }

    #[test]
    fn the_rolling_window_cuts_where_the_resumming_window_cut() {
        // A deterministic spread of quality strings: all-good, all-bad, a
        // decaying 3' tail (the real-world shape), and a single dip that a
        // rolling sum must recover from rather than carry forward.
        let quals: Vec<Vec<u8>> = vec![
            b"IIIIIIIIIIIIIIII".to_vec(),
            b"!!!!!!!!!!!!!!!!".to_vec(),
            b"IIIIIIIIIIII####".to_vec(),
            b"IIII####IIIIIIII".to_vec(),
            b"I".to_vec(),
            Vec::new(),
            (0..64u32).map(|i| (33 + (i * 7) % 42) as u8).collect(),
        ];

        for qual in &quals {
            for window_size in 1..=8usize {
                for min_qual in [0.0, 5.0, 20.0, 30.0, 41.0] {
                    let mut record = FastqRecord {
                        id: b"@r".to_vec(),
                        seq: vec![b'A'; qual.len()],
                        qual: qual.clone(),
                    };
                    record.quality_trim_end(min_qual, window_size);

                    let expected = trim_end_by_resumming(qual, min_qual, window_size);
                    assert_eq!(
                        record.seq.len(),
                        expected,
                        "qual={:?} window={window_size} min={min_qual}",
                        String::from_utf8_lossy(qual)
                    );
                    assert_eq!(record.qual.len(), expected, "seq and qual must stay in step");
                }
            }
        }
    }
}
