// src/read_filter.rs
//! Read filtering by k-mer content (`docs/feature-gap-analysis.md`'s S4):
//! stream a FASTQ/FASTA file against a reference `ktab::KmerTable` and keep
//! or discard each *read* based on how many of its own canonical k-mers are
//! found in that reference -- `kmc_tools filter`'s and BBDuk's `ref=`/`k=`
//! filtering, the classic host/contaminant-removal or targeted-enrichment
//! workflow.
//!
//! # Why a resident sorted array of the reference, not `KmerTable::get`
//!
//! `ktab.rs`'s own module doc comment is explicit that `get`/`range` reopen
//! the Parquet file and re-decode at least one row group *per call*, and
//! recommends against calling `get` in a tight loop for exactly that reason.
//! A filtering run is the tight loop `ktab.rs` warns about taken to its
//! extreme: every k-mer of every read in the input stream is a lookup
//! against the *same* reference table, often millions of them. Reopening a
//! Parquet file per k-mer would make filtering catastrophically slow -- not
//! "slower than ideal", but unusable on any real input.
//!
//! The reference table is the smaller, static side of this operation by
//! construction (a host genome, an adapter set, a contaminant reference),
//! read once per filtering run, not once per read -- the mirror image of why
//! `setops.rs` refuses to load *either* side of a table-to-table set
//! operation into memory (there, either side could be arbitrarily large and
//! the whole point is to support tables bigger than RAM). Here, one specific
//! side is expected to be the bounded one, and it is read into memory
//! exactly once via [`KmerTable::iter`] (already a tested, streaming code
//! path) into a flat `Vec<u64>`. Because a `KmerTable`'s rows are already
//! sorted and deduplicated by construction (`ktab.rs`'s own invariant), that
//! `Vec` needs no re-sorting and no hash table: [`ReferenceIndex::contains`]
//! is a plain `binary_search`, and the resident structure is exactly 8 bytes
//! per distinct reference k-mer -- no frequency column, no per-entry hash
//! overhead, since filtering only ever asks "is this k-mer present at all".
//!
//! This does mean a filtering run's memory scales with the reference
//! table's own size, not with the input read stream's (which is streamed
//! throughout, never materialized -- see [`run_filter`]). For a very large
//! reference (e.g. a multi-gigabase genome at a small k) that resident
//! array itself could be large; no attempt is made here to page or prune it,
//! matching this task's own priority order ("correctness and streaming
//! memory-boundedness [of the read side] matter more than raw speed here").
//! Bounding the *reference* side too (e.g. an on-disk pruned index) is a
//! reasonable follow-up once a real reference size makes it necessary --
//! not built speculatively, per this crate's "measure before extending"
//! standard.
//!
//! # Threshold semantics
//!
//! A read "matches" the reference when the fraction of its own canonical
//! k-mers found in the reference is `>= min_fraction` (inclusive, so a read
//! sitting exactly on the threshold matches -- see
//! [`matching_fraction`]/[`read_is_match`]). A read that yields *no*
//! canonical k-mers at all (shorter than the table's `k`, or entirely
//! ambiguous bases) is defined to never match, regardless of
//! `min_fraction` -- including a `min_fraction` of `0.0`. The alternative
//! (treating an empty read as a vacuous 0.0-fraction match against a
//! `min_fraction` of `0.0`) would make every degenerate read a "match" for a
//! reason that has nothing to do with the reference's actual content, which
//! silently corrupts both directions: every such read would be kept under
//! `--mode keep` and discarded under `--mode discard`, regardless of the
//! sample.
//!
//! # Keep vs. discard mode
//!
//! [`FilterMode::Keep`] writes only reads that match the reference
//! (targeted enrichment: "give me only the reads that look like this
//! organism/panel"). [`FilterMode::Discard`] writes only reads that do
//! *not* match (host/contaminant removal: "give me my sample with this
//! host/contaminant's reads removed"). Both share the exact same streaming
//! pass and the exact same match decision; only which side of it is written
//! differs.
//!
//! # Output format
//!
//! Output is always written as FASTQ (`@id`/sequence/`+`/quality), even
//! when every input was FASTA: `fastq.rs`'s FASTA reader already synthesizes
//! a quality string of the right length for every FASTA record
//! (`SYNTHETIC_FASTA_QUALITY`, Phred+33 'I' = Q40) precisely so downstream
//! stages do not need a second code path, and this module reuses exactly
//! that convention rather than inventing a FASTA writer.
//!
//! # Scope: single-end only
//!
//! Each `--input`/`inputs` file is filtered independently, read by read;
//! several files are read as one concatenated stream into one output file,
//! the same convention `count`'s own multi-file `--input` already
//! establishes (`fastq::MultiSourceReader`). **Paired-end (R1/R2)
//! synchronized filtering -- where a pair is kept or discarded as one unit
//! if either mate matches, which is the standard convention real pipelines
//! rely on -- is not implemented here.** Passing a sample's R1 and R2 files
//! both via `--input` filters each mate independently against the
//! reference and writes whichever of them individually passes, in file
//! order; a mate can be written while its partner is silently dropped, and
//! the two output streams are not kept synchronized against each other by
//! anything in this module. Wiring `cohort::discovery`'s pairing logic
//! through a real paired-aware filtering pass (reading both mates in lock
//! step, deciding on the pair, writing two synchronized output streams) is
//! real, additional work beyond this module's current single-stream
//! writer, and is left as an explicitly documented follow-up rather than
//! shipped half-correct.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::atomic::AtomicFile;
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqRecord, InputSpec, MultiSourceReader, RecordSource};
use crate::kmer;
use crate::ktab::KmerTable;

/// A resident, sorted array of every k-mer in a reference `KmerTable`,
/// built once per filtering run. See the module doc comment for why this
/// (and not a per-k-mer `KmerTable::get` call) is the right structure here.
#[derive(Debug, Clone)]
pub struct ReferenceIndex {
    k: usize,
    /// Ascending, deduplicated -- inherited directly from `KmerTable::iter`'s
    /// own guarantee, never re-sorted here.
    kmers: Vec<u64>,
    /// The reference table's own file, kept so `run_filter` can reject an
    /// `--output` that points back at it (the table is fully consumed into
    /// `kmers` up front, so nothing would notice a truncated reference file
    /// mid-run -- the damage would only surface the next time anyone tried
    /// to reopen it). See `run_filter`'s own doc comment for the full guard.
    table_path: PathBuf,
}

impl ReferenceIndex {
    /// Streams `table` once (via `KmerTable::iter`, which decodes one Arrow
    /// batch at a time -- see `ktab.rs`) into a flat sorted array of just
    /// its keys. The table's own frequency column is dropped: filtering
    /// only ever asks "is this k-mer present at all", never "how often".
    pub fn from_table(table: &KmerTable) -> Result<Self> {
        let mut kmers = Vec::with_capacity(table.len() as usize);
        for row in table.iter()? {
            let (kmer, _count) = row?;
            kmers.push(kmer);
        }
        Ok(Self { k: table.k(), kmers, table_path: table.path().to_path_buf() })
    }

    /// The `k` every resident k-mer was packed with (the reference table's
    /// own `k`) -- reads are extracted at this `k`, not any `k` of their
    /// own.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The reference table's own file -- see the field's doc comment for
    /// why `run_filter` needs it.
    pub fn table_path(&self) -> &Path {
        &self.table_path
    }

    /// Number of distinct k-mers held in memory.
    pub fn len(&self) -> usize {
        self.kmers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kmers.is_empty()
    }

    /// Whether `kmer` (already canonical, `k`-bit-packed) is present in the
    /// reference. `O(log n)` over the resident array.
    pub fn contains(&self, kmer: u64) -> bool {
        self.kmers.binary_search(&kmer).is_ok()
    }
}

/// Which reads a filtering run writes to `--output`. See the module doc
/// comment's "keep vs. discard mode" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Write only reads that match the reference (targeted enrichment).
    Keep,
    /// Write only reads that do *not* match the reference (host/contaminant
    /// removal).
    Discard,
}

/// Rejects a `min_fraction` that cannot be compared meaningfully: `NaN`
/// makes every `>=` comparison `false` (so `--mode keep` would silently
/// keep nothing and `--mode discard` would silently keep everything,
/// regardless of the reference), and a value outside `0.0..=1.0` can never
/// be crossed (or is always crossed) by a genuine fraction. Called from
/// both the CLI (`cli::parse_min_fraction` mirrors this for a fast,
/// pre-run rejection) and `run_filter` directly, since a Python caller
/// reaches this module without going through clap's parser at all.
pub fn validate_min_fraction(min_fraction: f64) -> Result<()> {
    if !min_fraction.is_finite() || !(0.0..=1.0).contains(&min_fraction) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "min_fraction",
            reason: format!("{min_fraction} must be a finite number between 0.0 and 1.0 inclusive"),
        });
    }
    Ok(())
}

/// The fraction of `seq`'s own canonical k-mers found in `index`, or `None`
/// if `seq` yields no canonical k-mers at all (shorter than `index.k()`, or
/// entirely ambiguous bases) -- see the module doc comment for why an empty
/// read is a distinct case from a genuine zero-fraction read rather than
/// being silently folded into one.
///
/// `kmer_buf` is a caller-owned scratch buffer, reused across calls (the
/// same "buffers the caller already owns" shape `FastqReader::
/// next_record_into` uses) so a filtering run over millions of reads does
/// not allocate a fresh `Vec` per read.
pub fn matching_fraction(seq: &[u8], index: &ReferenceIndex, kmer_buf: &mut Vec<u64>) -> Option<f64> {
    kmer::extract_canonical_kmers_into(seq, index.k(), kmer_buf);
    if kmer_buf.is_empty() {
        return None;
    }
    let matched = kmer_buf.iter().filter(|&&km| index.contains(km)).count();
    Some(matched as f64 / kmer_buf.len() as f64)
}

/// Whether `seq` matches the reference at `min_fraction` (`>=`, inclusive --
/// see the module doc comment). A read with no k-mers of its own
/// (`matching_fraction` returning `None`) never matches, regardless of
/// `min_fraction`.
pub fn read_is_match(seq: &[u8], index: &ReferenceIndex, min_fraction: f64, kmer_buf: &mut Vec<u64>) -> bool {
    matches!(matching_fraction(seq, index, kmer_buf), Some(fraction) if fraction >= min_fraction)
}

/// Outcome of a filtering run: how many reads were read in total, and how
/// many were written to `--output` under the chosen `FilterMode`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterStats {
    pub reads_total: u64,
    pub reads_written: u64,
}

/// Writes one record in FASTQ format (`@id`, sequence, `+`, quality) --
/// used for every kept read regardless of whether it originated from FASTQ
/// or FASTA input. `record.id` carries whichever marker byte the reader
/// originally saw (`@` for FASTQ, `>` for FASTA -- see `FastqReader`'s two
/// parsers in `fastq.rs`); it is stripped and replaced with `@` here so a
/// FASTA-origin record's header is never emitted with a stray leading `>`
/// or doubled marker.
fn write_fastq_record<W: Write>(writer: &mut W, record: &FastqRecord) -> io::Result<()> {
    writer.write_all(b"@")?;
    let id = match record.id.first() {
        Some(b'@') | Some(b'>') => &record.id[1..],
        _ => &record.id[..],
    };
    writer.write_all(id)?;
    writer.write_all(b"\n")?;
    writer.write_all(&record.seq)?;
    writer.write_all(b"\n+\n")?;
    writer.write_all(&record.qual)?;
    writer.write_all(b"\n")
}

/// The output side of a filtering run: a plain buffered writer, or one
/// wrapping a gzip encoder -- decided by `--output`'s own extension (`.gz`,
/// case-insensitive), the same convention `fastq.rs`'s input side already
/// uses for compression detection. An enum rather than a boxed trait
/// object: `finish` needs to reach `GzEncoder::finish` specifically (which
/// writes the gzip trailer and reports any I/O error from doing so) rather
/// than relying on `Drop` to swallow that error silently, which a `Box<dyn
/// Write>` would force.
enum OutputWriter {
    Plain(BufWriter<File>),
    Gz(GzEncoder<BufWriter<File>>),
}

impl Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            OutputWriter::Plain(w) => w.write(buf),
            OutputWriter::Gz(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputWriter::Plain(w) => w.flush(),
            OutputWriter::Gz(w) => w.flush(),
        }
    }
}

impl OutputWriter {
    /// Flushes and, for gzip output, writes the trailer -- must be called
    /// (and its result checked) before the underlying `AtomicFile` is
    /// committed, or a disk-full/interrupted gzip stream would be silently
    /// renamed into place as if it were complete.
    fn finish(self) -> io::Result<()> {
        match self {
            OutputWriter::Plain(mut w) => w.flush(),
            // `GzEncoder::finish()` writes the deflate tail and the 8-byte
            // gzip trailer into the inner `BufWriter` -- but does not flush
            // that `BufWriter` itself. Returning right after `finish()`
            // would let those last bytes ride on `BufWriter`'s own `Drop`,
            // which flushes on a best-effort basis and discards any error
            // (a disk-full write there would be silently swallowed, and
            // `AtomicFile::commit` would then rename a truncated `.gz` into
            // place as if it were complete -- precisely what this enum's
            // own doc comment says `finish` exists to prevent). `and_then`
            // reaches the inner writer's own checked `flush` instead, so
            // that error surfaces here like any other I/O failure.
            OutputWriter::Gz(w) => w.finish().and_then(|mut inner| inner.flush()),
        }
    }
}

/// Opens `path` for atomic write-then-rename (`atomic::AtomicFile`, the
/// same discipline every other writer in this crate follows), choosing a
/// plain or gzip-wrapped writer by `path`'s own extension.
fn open_output(path: &Path) -> Result<(OutputWriter, AtomicFile)> {
    let (file, pending) = AtomicFile::create(path)?;
    let is_gz = path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gz"));
    let writer = if is_gz {
        OutputWriter::Gz(GzEncoder::new(BufWriter::new(file), Compression::default()))
    } else {
        OutputWriter::Plain(BufWriter::new(file))
    };
    Ok((writer, pending))
}

/// Where to blame a producer-side failure -- the same small helper
/// `metagenomics.rs` keeps its own copy of (`pipeline.rs`'s own version is
/// private to that module), for the same reason: a source spanning several
/// files can name the exact file and record it was reading; a bare
/// in-memory source cannot, and falls back to `fallback_path` plus this
/// call's own running count.
fn failing_location<S: RecordSource>(source: &S, fallback_path: &Path, records_so_far: u64) -> (PathBuf, u64) {
    match source.current_source() {
        Some((path, records_in_file)) => (path, records_in_file + 1),
        None => (fallback_path.to_path_buf(), records_so_far + 1),
    }
}

/// Streams every record out of `source`, deciding keep/discard per read via
/// `read_is_match`, and writes the kept ones to `writer` in FASTQ form.
/// One pass, one record buffer reused via `next_record_into`, one k-mer
/// scratch buffer reused via `matching_fraction` -- the input stream is
/// never materialized beyond the one record currently being decided.
pub fn filter_records<S: RecordSource, W: Write>(
    mut source: S,
    index: &ReferenceIndex,
    mode: FilterMode,
    min_fraction: f64,
    writer: &mut W,
    fallback_path: &Path,
) -> Result<FilterStats> {
    let mut record = FastqRecord::default();
    let mut kmer_buf: Vec<u64> = Vec::new();
    let mut stats = FilterStats::default();

    loop {
        match source.next_record_into(&mut record) {
            Ok(true) => {
                let is_match = read_is_match(&record.seq, index, min_fraction, &mut kmer_buf);
                let keep = match mode {
                    FilterMode::Keep => is_match,
                    FilterMode::Discard => !is_match,
                };
                if keep {
                    write_fastq_record(writer, &record)
                        .map_err(|e| FastDnaError::Io { path: fallback_path.to_path_buf(), source: e })?;
                    stats.reads_written += 1;
                }
                stats.reads_total += 1;
            }
            Ok(false) => break,
            Err(FastqReadError::Io(source_err)) => {
                let (path, _) = failing_location(&source, fallback_path, stats.reads_total);
                return Err(FastDnaError::Io { path, source: source_err });
            }
            Err(FastqReadError::Malformed(reason)) => {
                let (path, record_no) = failing_location(&source, fallback_path, stats.reads_total);
                return Err(FastDnaError::MalformedFastq { path, record: record_no, reason });
            }
        }
    }

    Ok(stats)
}

/// Rejects an `--output` that would overwrite the reference table or any
/// input file a filtering run still needs to read.
///
/// This lives here, in `run_filter` itself, rather than only in the CLI
/// (`main.rs::run_filter`'s own pre-check, kept for a fast, pre-header
/// rejection there): `ffi.rs::filter_reads` -- the Python binding -- calls
/// straight into this function with no CLI layer in between, so before this
/// guard existed here, `KmerTable.filter_reads(output=<its own input>)` from
/// Python silently destroyed the caller's file (`tests/review_findings.rs`/
/// `python/tests/test_review_findings.py` pin exactly this). Putting the
/// check here means the CLI, the Python binding, and any future
/// `fastdna_core` consumer all inherit it from one place, instead of each
/// needing its own copy.
///
/// A truncated output landing on top of a live input would corrupt the very
/// read still in progress; a truncated output landing on top of the
/// reference table would destroy it after it has already been fully
/// consumed into a resident `ReferenceIndex` (so the run itself would
/// "succeed", and only the next attempt to reopen the table would notice).
fn guard_against_output_overwrite(inputs: &[InputSpec], index: &ReferenceIndex, output: &Path) -> Result<()> {
    if crate::atomic::same_file(index.table_path(), output) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "output",
            reason: format!(
                "points at the reference table {} and would overwrite it",
                index.table_path().display()
            ),
        });
    }
    for input in inputs {
        if let InputSpec::File(path) = input {
            if crate::atomic::same_file(path, output) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "output",
                    reason: format!("points at the input file {} and would overwrite it", path.display()),
                });
            }
        }
    }
    Ok(())
}

/// The full filtering run: opens `inputs` as one concatenated stream (see
/// the module doc comment's single-end scope note), filters it against
/// `index` under `mode`/`min_fraction`, and writes the kept reads to
/// `output` (FASTQ, gzipped iff `output`'s extension says so),
/// atomically -- shared by the CLI (`main.rs::run_filter`) and the Python
/// binding (`ffi.rs::filter_reads`) so both go through exactly one
/// validated, tested path rather than two copies of this wiring.
pub fn run_filter(
    inputs: Vec<InputSpec>,
    index: &ReferenceIndex,
    mode: FilterMode,
    min_fraction: f64,
    output: &Path,
) -> Result<FilterStats> {
    validate_min_fraction(min_fraction)?;
    guard_against_output_overwrite(&inputs, index, output)?;

    let source = MultiSourceReader::new(inputs);
    source.validate()?;

    let (mut writer, pending) = open_output(output)?;
    let fallback_path = PathBuf::from("<inputs>");
    // On error, `pending` (the `AtomicFile` guard) is dropped by this `?`'s
    // unwind, which removes the abandoned temp file and leaves `output`
    // untouched -- nothing extra to clean up here.
    let stats = filter_records(source, index, mode, min_fraction, &mut writer, &fallback_path)?;

    writer.finish().map_err(|e| FastDnaError::Io { path: output.to_path_buf(), source: e })?;
    pending.commit()?;

    Ok(stats)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;
    use crate::fastq::FastqReader;
    use std::io::Cursor;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_read_filter_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a reference `ReferenceIndex` at k=4 from a hand-picked list of
    /// canonical-encoded k-mers, through the real counting + export path
    /// (matching `ktab.rs`'s/`setops.rs`'s own test convention: these tests
    /// exercise exactly the files `fastdna count` produces).
    fn build_index(name: &str, k: usize, entries: &[u64]) -> (ReferenceIndex, PathBuf) {
        let path = temp_path(name);
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, &path, k, 1, false).unwrap();
        let table = KmerTable::open(&path).unwrap();
        (ReferenceIndex::from_table(&table).unwrap(), path)
    }

    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    // -- ReferenceIndex --------------------------------------------------

    #[test]
    fn reference_index_reports_membership_correctly() {
        let (index, p) = build_index("membership", 4, &[6, 6, 0]);
        assert_eq!(index.len(), 2);
        assert!(index.contains(6));
        assert!(index.contains(0));
        assert!(!index.contains(3), "3 lies between the two keys but is not itself present");
        assert!(!index.contains(u64::MAX));
        cleanup(&[p]);
    }

    #[test]
    fn reference_index_from_an_empty_table_contains_nothing() {
        let (index, p) = build_index("empty_ref", 4, &[]);
        assert!(index.is_empty());
        assert!(!index.contains(0));
        cleanup(&[p]);
    }

    // -- matching_fraction / read_is_match --------------------------------

    #[test]
    fn matching_fraction_is_none_for_a_read_shorter_than_k() {
        let (index, p) = build_index("short_read", 8, &[1, 2, 3]);
        let mut buf = Vec::new();
        assert_eq!(matching_fraction(b"ACG", &index, &mut buf), None, "3 bases < k=8 yields no k-mers");
        cleanup(&[p]);
    }

    #[test]
    fn a_read_with_no_kmers_never_matches_in_either_mode_even_at_zero_threshold() {
        let (index, p) = build_index("short_read_modes", 8, &[]);
        let mut buf = Vec::new();
        // min_fraction = 0.0 would make a genuine 0/n fraction match; a read
        // with *no* k-mers at all must still not match, by design.
        assert!(!read_is_match(b"AC", &index, 0.0, &mut buf));
        cleanup(&[p]);
    }

    #[test]
    fn a_read_fully_covered_by_the_reference_matches_at_full_threshold() {
        // k=4, "ACGTACGT" -> canonical k-mers at positions 0..=4, all built
        // from the same repeating unit; put every one of those exact
        // encodings in the reference so the read is a 100% match.
        let seq = b"ACGTACGT";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        let (index, p) = build_index("full_match", 4, &kmers);

        let mut buf = Vec::new();
        let fraction = matching_fraction(seq, &index, &mut buf).unwrap();
        assert!((fraction - 1.0).abs() < 1e-9, "every k-mer of this read is in the reference: {fraction}");
        assert!(read_is_match(seq, &index, 1.0, &mut buf));
        cleanup(&[p]);
    }

    #[test]
    fn a_read_with_zero_reference_overlap_does_not_clear_a_positive_threshold_but_does_clear_zero() {
        let seq = b"ACGTACGT";
        // A disjoint reference: canonical k-mers of a completely different
        // sequence. Unlike `matching_fraction_is_none_for_a_read_shorter_
        // than_k`, this read genuinely has k-mers of its own and a
        // genuinely computed fraction (0.0, not "no data") -- so, unlike
        // that case, it *does* satisfy an inclusive `>= 0.0` threshold; it
        // is only a non-match once the threshold rises above zero.
        let unrelated = kmer::extract_canonical_kmers(b"TTTTTTTT", 4);
        let (index, p) = build_index("no_overlap", 4, &unrelated);

        let mut buf = Vec::new();
        let fraction = matching_fraction(seq, &index, &mut buf).unwrap();
        assert_eq!(fraction, 0.0);
        assert!(
            !read_is_match(seq, &index, 0.1, &mut buf),
            "a real 0.0 fraction must not clear a positive threshold"
        );
        assert!(
            read_is_match(seq, &index, 0.0, &mut buf),
            "a real 0.0 fraction does satisfy an inclusive >= 0.0 threshold"
        );
        cleanup(&[p]);
    }

    #[test]
    fn the_match_threshold_is_inclusive_at_the_boundary() {
        // k=4, seq "ACGTAC" -> 3 canonical k-mers (positions 0,1,2). Put
        // exactly the first two of the *forward* encodings in the
        // reference and check the fraction lands exactly on 2/3, then
        // confirm >= behavior at that exact value.
        let seq = b"ACGTAC";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        assert_eq!(kmers.len(), 3, "sanity: this fixture must yield exactly 3 k-mers");
        let (index, p) = build_index("boundary", 4, &kmers[..2]);

        let mut buf = Vec::new();
        let fraction = matching_fraction(seq, &index, &mut buf).unwrap();
        assert!((fraction - (2.0 / 3.0)).abs() < 1e-9, "fraction: {fraction}");
        assert!(read_is_match(seq, &index, fraction, &mut buf), "a fraction exactly at the threshold must match (>=)");
        assert!(!read_is_match(seq, &index, fraction + 0.01, &mut buf), "just above the fraction must not match");
        cleanup(&[p]);
    }

    // -- filter_records: mode semantics -----------------------------------

    fn reader_over(text: &str) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(text.as_bytes().to_vec()))
    }

    #[test]
    fn keep_mode_writes_only_matching_reads() {
        let matching_kmers = kmer::extract_canonical_kmers(b"ACGTACGT", 4);
        let (index, p) = build_index("keep_mode", 4, &matching_kmers);

        let reader = reader_over("@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTTTTT\n+\nIIIIIIII\n");
        let mut out: Vec<u8> = Vec::new();
        let stats =
            filter_records(reader, &index, FilterMode::Keep, 1.0, &mut out, Path::new("<test>")).unwrap();

        assert_eq!(stats.reads_total, 2);
        assert_eq!(stats.reads_written, 1, "only r1 fully matches the reference");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("@r1"));
        assert!(!text.contains("@r2"));
        cleanup(&[p]);
    }

    #[test]
    fn discard_mode_writes_only_non_matching_reads() {
        let matching_kmers = kmer::extract_canonical_kmers(b"ACGTACGT", 4);
        let (index, p) = build_index("discard_mode", 4, &matching_kmers);

        let reader = reader_over("@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTTTTT\n+\nIIIIIIII\n");
        let mut out: Vec<u8> = Vec::new();
        let stats =
            filter_records(reader, &index, FilterMode::Discard, 1.0, &mut out, Path::new("<test>")).unwrap();

        assert_eq!(stats.reads_total, 2);
        assert_eq!(stats.reads_written, 1, "only r2 is not a full match, so only it survives discard");
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("@r1"));
        assert!(text.contains("@r2"));
        cleanup(&[p]);
    }

    #[test]
    fn a_read_with_zero_matching_kmers_is_kept_under_discard_and_dropped_under_keep() {
        let unrelated = kmer::extract_canonical_kmers(b"GGGGGGGG", 4);
        let (index, p) = build_index("zero_match", 4, &unrelated);

        let make_reader = || reader_over("@r1\nACGTACGT\n+\nIIIIIIII\n");

        let mut kept: Vec<u8> = Vec::new();
        let stats_discard =
            filter_records(make_reader(), &index, FilterMode::Discard, 0.1, &mut kept, Path::new("<t>")).unwrap();
        assert_eq!(stats_discard.reads_written, 1);

        let mut dropped: Vec<u8> = Vec::new();
        let stats_keep =
            filter_records(make_reader(), &index, FilterMode::Keep, 0.1, &mut dropped, Path::new("<t>")).unwrap();
        assert_eq!(stats_keep.reads_written, 0);

        cleanup(&[p]);
    }

    #[test]
    fn fasta_origin_records_are_written_as_fastq_with_synthetic_quality() {
        let (index, p) = build_index("fasta_out", 4, &[]);
        let reader = reader_over(">contig1\nACGTACGT\n");
        let mut out: Vec<u8> = Vec::new();
        // Discard mode with an empty reference and a positive threshold:
        // nothing can clear it, so every read is written -- this test is
        // about output shape, not the match decision. (A threshold of
        // exactly 0.0 would trivially match every read with real k-mers --
        // see `a_read_with_zero_reference_overlap_does_not_clear_a_positive_
        // threshold_but_does_clear_zero` -- so it is deliberately not used
        // here.)
        let stats =
            filter_records(reader, &index, FilterMode::Discard, 0.5, &mut out, Path::new("<t>")).unwrap();
        assert_eq!(stats.reads_written, 1);

        let text = String::from_utf8(out).unwrap();
        assert_eq!(text, "@contig1\nACGTACGT\n+\nIIIIIIII\n", "FASTA input must be written as FASTQ verbatim");
        cleanup(&[p]);
    }

    #[test]
    fn multiple_input_files_are_filtered_as_one_concatenated_stream() {
        use crate::fastq::InputSpec;

        let dir = std::env::temp_dir().join("fastdna_read_filter_multi_input");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        let a = dir.join(format!("a_{unique}.fastq"));
        let b = dir.join(format!("b_{unique}.fastq"));
        std::fs::write(&a, b"@a1\nACGTACGT\n+\nIIIIIIII\n").unwrap();
        std::fs::write(&b, b"@b1\nTTTTTTTT\n+\nIIIIIIII\n").unwrap();

        let (index, p) = build_index("multi_input", 4, &[]);
        let source = MultiSourceReader::new(vec![InputSpec::File(a.clone()), InputSpec::File(b.clone())]);
        let mut out: Vec<u8> = Vec::new();
        // A positive threshold against an empty reference: see the comment
        // in `fasta_origin_records_are_written_as_fastq_with_synthetic_
        // quality` for why 0.0 would trivially match every read here.
        let stats =
            filter_records(source, &index, FilterMode::Discard, 0.5, &mut out, Path::new("<t>")).unwrap();

        assert_eq!(stats.reads_total, 2);
        assert_eq!(stats.reads_written, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("@a1"));
        assert!(text.contains("@b1"));

        cleanup(&[p]);
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- validate_min_fraction ---------------------------------------------

    #[test]
    fn validate_min_fraction_rejects_out_of_range_and_non_finite_values() {
        assert!(validate_min_fraction(0.0).is_ok());
        assert!(validate_min_fraction(1.0).is_ok());
        assert!(matches!(validate_min_fraction(-0.1), Err(FastDnaError::InvalidConfig { .. })));
        assert!(matches!(validate_min_fraction(1.1), Err(FastDnaError::InvalidConfig { .. })));
        assert!(matches!(validate_min_fraction(f64::NAN), Err(FastDnaError::InvalidConfig { .. })));
        assert!(matches!(validate_min_fraction(f64::INFINITY), Err(FastDnaError::InvalidConfig { .. })));
    }

    // -- run_filter: end-to-end through real files, including gzip output --

    #[test]
    fn run_filter_writes_a_real_gzip_output_file() {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;

        let dir = std::env::temp_dir().join("fastdna_read_filter_gzip");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        let input_path = dir.join(format!("in_{unique}.fastq"));
        std::fs::write(&input_path, b"@r1\nACGTACGT\n+\nIIIIIIII\n").unwrap();
        let output_path = dir.join(format!("out_{unique}.fastq.gz"));

        let (index, p) = build_index("run_filter_gzip", 4, &[]);
        // A positive threshold against an empty reference: see the comment
        // in `fasta_origin_records_are_written_as_fastq_with_synthetic_
        // quality` for why 0.0 would trivially match every read here.
        let stats = run_filter(
            vec![InputSpec::File(input_path.clone())],
            &index,
            FilterMode::Discard,
            0.5,
            &output_path,
        )
        .unwrap();
        assert_eq!(stats.reads_written, 1);

        let compressed = std::fs::read(&output_path).unwrap();
        let mut decoder = MultiGzDecoder::new(compressed.as_slice());
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed).unwrap();
        assert_eq!(decompressed, "@r1\nACGTACGT\n+\nIIIIIIII\n");

        cleanup(&[p]);
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_filter_rejects_an_invalid_min_fraction_before_touching_any_file() {
        let (index, p) = build_index("run_filter_invalid_fraction", 4, &[]);
        let result = run_filter(
            vec![InputSpec::File(PathBuf::from("does-not-exist.fastq"))],
            &index,
            FilterMode::Keep,
            2.0,
            &PathBuf::from("does-not-matter.fastq"),
        );
        assert!(matches!(result, Err(FastDnaError::InvalidConfig { .. })));
        cleanup(&[p]);
    }
}
