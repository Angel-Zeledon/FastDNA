// src/read_profile.rs
//! Per-read k-mer profiles (`docs/feature-gap-analysis.md`'s S3): for each
//! read in a FASTQ/FASTA input, the vector of counts its own constituent
//! canonical k-mers have in a reference `ktab::KmerTable` -- read position
//! `i` -> the count of the k-mer starting at position `i`. This is FastK's
//! signature feature (no KMC equivalent), and the substrate for error
//! detection (a run of count-1 positions in an otherwise high-count read is
//! a sequencing error), QV estimation, and exact Merqury-style assembly QC,
//! as opposed to `python/fastdna/assembly_qc.py::evaluate_assembly`'s
//! current set-membership approximation (see that module's own docstring).
//!
//! Two passes, exactly as this item's original design named: the reference
//! table is built first, by the existing `count` -> `export.rs` path (S1),
//! then this module opens it as a `ktab::KmerTable` and streams reads
//! against it. S1 is what unblocked this: before it existed there was
//! nothing queryable to profile a read against.
//!
//! # Output format: RLE, not a tidy per-base table
//!
//! Two shapes were on the table for the profile itself (the per-read
//! summary statistics below are a separate, small, always-tidy table
//! either way -- this decision is only about the full positional profile):
//!
//! 1. **Long/tidy**: one row per `(read_id, position, count)` -- trivially
//!    consumable from DuckDB/pandas/polars with zero FastDNA-specific
//!    tooling, the same reasoning `S6`'s `export_cohort_matrix_parquet`
//!    gives for its own long/COO shape.
//! 2. **RLE**: one row per `(read_id, start, run_length, count)`, covering a
//!    maximal run of consecutive read positions that all carry the same
//!    count.
//!
//! RLE was chosen, for a reason the tidy shape does not share: a per-base
//! row is enormous at real scale in a way a per-cohort-matrix row is not.
//! One row per base per read means a single lane of 150bp short reads at
//! modest coverage is already hundreds of millions of rows before a single
//! *sample* is profiled twice, let alone a cohort -- a 30x whole-genome
//! human run (~600 million 150bp reads) is on the order of **90 billion**
//! per-base rows even before Parquet's own encoding is applied, against
//! this crate's existing `export_counts_parquet` writing one row per
//! *distinct k-mer* (tens of millions, not tens of billions) for the same
//! input. RLE avoids materializing that per-base table at all: consecutive
//! canonical k-mers along a read overlap by `k - 1` bases and, at typical
//! sequencing depth, are overwhelmingly supported by the same set of
//! overlapping reads across a non-repetitive, error-free stretch of
//! genome -- so their reference counts are frequently *identical*, not
//! merely close. This is not a novel observation invented for this module:
//! it is the same empirical regularity FastK's own `.prof` format is built
//! to exploit, and it is why a run boundary in this profile is itself
//! informative -- it marks either a real coverage transition (entering or
//! leaving a repeat/copy-number region) or a sequencing error breaking an
//! otherwise-uniform stretch, which is exactly the signal `read_profile`
//! exists to expose. This crate has not independently re-measured a
//! compression ratio against real sequencing data in this session (unlike
//! the zlib-ng backend decision in `fastq.rs`, which cites a controlled,
//! reproduced measurement) -- the claim here is the same well-established
//! shape FastK itself ships for the same reason, not a number benchmarked
//! here. RLE degrades gracefully to (in the worst case, a maximally
//! erratic profile with no two adjacent counts ever equal) the same
//! per-base row count the tidy shape always pays, so there is no
//! correctness cost to choosing it -- only an upside that scales with how
//! much real runs actually compress.
//!
//! **Lossless round-trip**: [`expand_rle`] reconstructs the exact full
//! per-position `(position, count)` sequence from a read's RLE rows, with a
//! gap left exactly where the original per-position extraction had no
//! k-mer at all (see [`build_read_profile`]'s doc comment on ambiguous
//! bases) -- pinned by this module's own round-trip tests.
//!
//! # `ProfileIndex`: a parallel resident index, not a generalized
//! `read_filter::ReferenceIndex`
//!
//! `read_filter.rs`'s `ReferenceIndex` already solves "load a reference
//! table once into a resident, binary-searched structure instead of
//! calling `KmerTable::get` in a per-k-mer tight loop" (see that module's
//! own doc comment for why `get` is unusable here at all: it reopens the
//! Parquet file and re-decodes a row group per call). Profiling needs the
//! same trick but must keep each k-mer's *count*, not just its presence,
//! since the whole point of a profile is the count itself.
//!
//! `ReferenceIndex` was not widened to optionally carry counts (e.g. a
//! `Vec<Option<u32>>` alongside its keys). That type is directly
//! covered by `read_filter.rs`'s own tests and is reachable from a stable
//! public method (`ReferenceIndex::from_table`) used by both the CLI and
//! `ffi.rs::filter_reads`; changing its shape to serve a second, unrelated
//! caller (this module has no use for "is this k-mer present at all" as a
//! distinct question -- it always wants the count, where absence is simply
//! the count `0`) risks disturbing tested, shipped behavior for a save of
//! one small struct definition. `ProfileIndex` below is a few-line parallel
//! type instead: same construction shape (stream `KmerTable::iter` once
//! into a resident, sorted structure), same complexity characteristics
//! (`O(n)` build, `O(log n)` point lookup, no hash overhead), but an
//! explicit `Vec<u32>` of counts parallel to the sorted keys, since a
//! profile is never asked "present or not" -- only "what count".
//!
//! Both widths, like `ReferenceIndex`: `ProfileIndex::from_wide_table`
//! builds the same structure over `u128` keys. The run-length encoder in
//! `build_read_profile` sees neither -- extraction and reference lookup
//! happen first and hand it `(position, count)` pairs -- which is why the
//! part carrying the gap rules and run boundaries exists once rather than
//! once per width.
//!
//! Memory: resident on the reference table's size, exactly as
//! `ReferenceIndex` documents for filtering -- the same deliberate tradeoff,
//! for the same reason (the reference is the bounded side; the read stream
//! is fully streamed and never materialized, see [`run_profile`]).
//!
//! # Per-read summary statistics
//!
//! Alongside (not instead of) the RLE profile, this module always writes a
//! second, small, per-read summary table: `read_id`, `n_kmers`,
//! `n_present_kmers`, `min_count`, `median_count`, `max_count`. This is
//! deliberately not folded into a single output: most consumers (a QV
//! estimator scanning for outlier reads, a quick per-sample sanity check)
//! want these six small numbers per read and never need to decompress a
//! single RLE run, so making them pay for opening and scanning the (much
//! larger) profile table just to answer "how many of this read's k-mers
//! were even present in the reference" would defeat the RLE format's own
//! reason for existing.
//!
//! `n_kmers`/`n_present_kmers` follow `metagenomics.rs`'s
//! `ReadClassification` column-naming convention (`n_kmers`,
//! `n_classified_kmers`) rather than the gap-analysis document's own
//! shorthand ("n_present") verbatim: both are `n_<qualifier>_kmers`, naming
//! *what a k-mer had to do to be counted* -- "classified to a taxon" there,
//! "present in the reference table" here -- so a caller already familiar
//! with one schema recognizes the pattern in the other instead of learning
//! a second, differently-shaped convention for a structurally identical
//! idea. `n_kmers` itself matches `ReadClassification::n_kmers` exactly:
//! both count the read's own extracted canonical k-mers (`kmer::
//! extract_canonical_kmers_with_positions_into`'s output length), not
//! `seq.len() - k + 1` -- a read with an ambiguous base has genuinely fewer
//! canonical k-mers than its raw length would suggest, and both schemas
//! report the true count, not the naive one.
//!
//! `min_count`/`median_count`/`max_count` summarize the *full* per-position
//! count sequence, including positions whose k-mer was entirely absent from
//! the reference (count `0`) -- not just the present ones. A read that is
//! mostly one high, uniform count with a handful of `0`s is exactly the
//! error-detection signal this module exists to surface, and folding those
//! zeros out of the distribution would hide the very thing a QV estimator
//! is looking for. All three are `None` (a null Parquet cell) for a read
//! with `n_kmers == 0` -- there is no distribution to summarize, the same
//! "no data, not a computed zero" distinction `read_filter.rs`'s
//! `matching_fraction` already draws for an empty extraction.
//!
//! # Streaming, memory-bounded reads
//!
//! Reads are streamed one at a time via `fastq::MultiSourceReader`, exactly
//! as `read_filter::run_filter` streams its input -- never materialized as
//! a whole file in memory. Memory is bounded by the resident `ProfileIndex`
//! (the reference table) plus one read's worth of scratch buffers, the
//! same tradeoff `read_filter.rs` documents and deliberately makes for the
//! same reason: the reference is the side expected to be bounded; the read
//! stream is not, and is never asked to be.
//!
//! # Scope: single-end only
//!
//! Each `--input`/`inputs` file is profiled independently, read by read,
//! and several files are read as one concatenated stream into the same pair
//! of output files -- the same convention `count`'s own multi-file
//! `--input` and `read_filter.rs`'s own single-end scope already establish.
//! Paired-end (R1/R2) synchronized profiling is not implemented here, for
//! the same reason `read_filter.rs` leaves it as an explicit, documented
//! follow-up rather than shipping it half-correct: a profile is inherently
//! a per-read (not a per-pair) artifact, so unlike filtering there is no
//! obvious "keep/discard as a unit" decision this format is even missing --
//! but a caller wanting mate-aware read *ids* (e.g. `/1`, `/2` suffixes
//! preserved distinctly per mate) gets exactly what each file's own FASTQ
//! headers already say, with no synchronization attempted or assumed
//! between the two streams.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::atomic::AtomicFile;
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqRecord, InputSpec, MultiSourceReader, RecordSource};
use crate::kmer;
use crate::wide_kmer;
use crate::ktab::KmerTable;

/// A resident, sorted `(k-mer, count)` index built once per profiling run --
/// see this module's doc comment ("`ProfileIndex`: a parallel resident
/// index") for why this is a small parallel type rather than a widened
/// `read_filter::ReferenceIndex`.
#[derive(Debug, Clone)]
pub struct ProfileIndex {
    k: usize,
    /// Ascending, deduplicated -- inherited directly from the table
    /// iterator's own guarantee, never re-sorted here.
    kmers: ProfileKeys,
    /// `counts[i]` is the reference table's frequency for key `i`.
    counts: Vec<u32>,
}

/// The resident key array, at whichever width the reference was written in
/// -- the same shape, and the same reasoning, as
/// `read_filter::IndexKeys`: the width belongs to the file, so it is an
/// enum here rather than a type parameter threaded through every caller.
#[derive(Debug, Clone)]
enum ProfileKeys {
    Narrow(Vec<u64>),
    Wide(Vec<u128>),
}

/// Per-read scratch for [`build_read_profile`], reused across reads.
///
/// Holds one extraction buffer per key width plus the resolved
/// `(position, reference count)` pairs the run-length encoder actually
/// works over. Splitting the extraction from the encoding is what lets the
/// RLE loop -- the part with the gap handling and the run-boundary rules --
/// exist once rather than once per width: by the time it runs, the key is
/// already gone.
#[derive(Debug, Default, Clone)]
pub struct ProfileScratch {
    narrow: Vec<(u32, u64)>,
    wide: Vec<(u32, u128)>,
    /// `(read position, the reference's count for the k-mer there)`.
    resolved: Vec<(u32, u32)>,
}

impl ProfileScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ProfileIndex {
    /// Streams `table` once (via `KmerTable::iter`) into two parallel,
    /// resident `Vec`s -- the keys and their counts, in the table's own
    /// ascending order. `O(n)` to build, `O(log n)` per lookup afterward.
    pub fn from_table(table: &KmerTable) -> Result<Self> {
        let mut kmers = Vec::with_capacity(table.len() as usize);
        let mut counts = Vec::with_capacity(table.len() as usize);
        for row in table.iter()? {
            let (kmer, count) = row?;
            kmers.push(kmer);
            counts.push(count);
        }
        Ok(Self { k: table.k(), kmers: ProfileKeys::Narrow(kmers), counts })
    }

    /// `from_table` for a wide (`kmer_bits`, `33 <= k <= 64`) reference.
    /// Twice the resident bytes per k-mer, which is the price of profiling
    /// above k=32 and is stated rather than discovered.
    pub fn from_wide_table(table: &crate::wide_ktab::WideKmerTable) -> Result<Self> {
        let mut kmers = Vec::with_capacity(table.len() as usize);
        let mut counts = Vec::with_capacity(table.len() as usize);
        for row in table.iter()? {
            let (kmer, count) = row?;
            kmers.push(kmer);
            counts.push(count);
        }
        Ok(Self { k: table.k(), kmers: ProfileKeys::Wide(kmers), counts })
    }

    /// The `k` every resident k-mer was packed with (the reference table's
    /// own `k`) -- reads are extracted at this `k`, not any `k` of their
    /// own.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of distinct k-mers held in memory.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// `"narrow"` or `"wide"` -- which key width this reference was built
    /// at, which `k` alone does not answer (a table counted with
    /// `--engine wide` below k=32 is wide too).
    pub fn width(&self) -> &'static str {
        match &self.kmers {
            ProfileKeys::Narrow(_) => "narrow",
            ProfileKeys::Wide(_) => "wide",
        }
    }

    /// The reference table's frequency for `kmer` (already canonical and
    /// packed) in a **narrow** reference, or `None` if it is absent --
    /// callers building a profile treat an absent k-mer as count `0`, not as
    /// missing data (see the module doc comment's summary-statistics
    /// section for why that distinction matters and where it is drawn).
    ///
    /// Always `None` on a wide reference, which holds no `u64` keys. This
    /// was `pub fn get(u64)` before profiling learned both widths; it is
    /// crate-private now for the reason
    /// `read_filter::ReferenceIndex::contains_narrow` gives -- there is no
    /// correct public answer for a `u64` lookup against a `u128` index, and
    /// no caller chooses the width anyway.
    pub(crate) fn get_narrow(&self, kmer: u64) -> Option<u32> {
        match &self.kmers {
            ProfileKeys::Narrow(keys) => keys.binary_search(&kmer).ok().map(|i| self.counts[i]),
            ProfileKeys::Wide(_) => None,
        }
    }

    /// [`Self::get_narrow`] for a wide reference.
    pub(crate) fn get_wide(&self, kmer: u128) -> Option<u32> {
        match &self.kmers {
            ProfileKeys::Wide(keys) => keys.binary_search(&kmer).ok().map(|i| self.counts[i]),
            ProfileKeys::Narrow(_) => None,
        }
    }
}

/// One maximal run of consecutive read positions that all carry the same
/// reference count -- the unit `read_profile`'s Parquet output is written
/// in. `start` is the 0-based read position of the run's first k-mer (in
/// k-mer-start coordinates, matching `kmer::
/// extract_canonical_kmers_with_positions_into`'s own position convention);
/// the run covers positions `start..start + run_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RleRun {
    pub start: u32,
    pub run_length: u32,
    pub count: u32,
}

/// Per-read summary statistics written alongside the RLE profile -- see the
/// module doc comment's "Per-read summary statistics" section for why these
/// six fields, this naming, and why the distribution stats include absent
/// (count-0) positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileSummaryStats {
    /// The read's own extracted canonical k-mer count (not `seq.len() - k +
    /// 1`; see the module doc comment).
    pub n_kmers: u32,
    /// How many of those k-mers had a nonzero reference count.
    pub n_present_kmers: u32,
    /// `None` iff `n_kmers == 0` (no distribution to summarize).
    pub min_count: Option<u32>,
    pub median_count: Option<f64>,
    pub max_count: Option<u32>,
}

impl ProfileSummaryStats {
    /// The all-null-distribution summary for a read that yielded no k-mers
    /// at all (shorter than `k`, or entirely ambiguous bases).
    fn empty() -> Self {
        Self { n_kmers: 0, n_present_kmers: 0, min_count: None, median_count: None, max_count: None }
    }
}

/// The median of an already-ascending-sorted slice, or `None` for an empty
/// slice. The usual even-length convention (average of the two middle
/// values) is used, which is why this returns `f64` rather than `u32` even
/// though every input value is an integer count.
fn median_of_sorted(sorted: &[u32]) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        Some(sorted[n / 2] as f64)
    } else {
        Some((sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0)
    }
}

/// Builds one read's RLE profile and summary statistics against `index`.
///
/// `positions_buf` is a caller-owned scratch buffer (the same
/// "buffers the caller already owns" shape `read_filter::matching_fraction`
/// uses for its own `kmer_buf`), reused across calls so a profiling run over
/// millions of reads allocates it once, not once per read.
///
/// Rejects (as `FastDnaError::InvalidConfig`) a read whose length exceeds
/// `u32::MAX` bases, before extraction: `kmer::
/// extract_canonical_kmers_with_positions_into` packs each k-mer's start
/// position into a `u32`, and this is the one call site responsible for
/// upholding that function's own documented precondition -- see its doc
/// comment in `kmer.rs`. Not exercised by a unit test here: constructing a
/// read fixture larger than 4 billion bases to hit this path is not a
/// practical test to write (it would need to allocate several gigabytes
/// just to build the input), so this is a documented, reasoned guard
/// rather than a measured one.
///
/// A read shorter than `index.k()` (or entirely ambiguous bases) yields an
/// empty run list and `ProfileSummaryStats::empty()` -- not an error: a
/// short read is ordinary input, not a malformed one, the same treatment
/// `read_filter.rs`'s "a read with no k-mers never matches" gives it.
pub fn build_read_profile(
    seq: &[u8],
    index: &ProfileIndex,
    scratch: &mut ProfileScratch,
) -> Result<(Vec<RleRun>, ProfileSummaryStats)> {
    if seq.len() > u32::MAX as usize {
        return Err(FastDnaError::InvalidConfig {
            parameter: "read length",
            reason: format!(
                "a read of {} bases exceeds the {} bases this crate's per-read profiling can represent",
                seq.len(),
                u32::MAX
            ),
        });
    }

    // Extraction and reference lookup happen here, at the index's own
    // width; everything below works over `(position, count)` pairs and has
    // no key in it at all. That split is why the run-length encoder -- the
    // part carrying the gap rules and the run boundaries -- exists once
    // instead of once per width.
    let resolved = &mut scratch.resolved;
    resolved.clear();
    match index.width() {
        "wide" => {
            wide_kmer::extract_canonical_kmers_with_positions_into(seq, index.k(), &mut scratch.wide);
            resolved.reserve(scratch.wide.len());
            for &(pos, km) in scratch.wide.iter() {
                resolved.push((pos, index.get_wide(km).unwrap_or(0)));
            }
        }
        _ => {
            kmer::extract_canonical_kmers_with_positions_into(seq, index.k(), &mut scratch.narrow);
            resolved.reserve(scratch.narrow.len());
            for &(pos, km) in scratch.narrow.iter() {
                resolved.push((pos, index.get_narrow(km).unwrap_or(0)));
            }
        }
    }

    if resolved.is_empty() {
        return Ok((Vec::new(), ProfileSummaryStats::empty()));
    }

    let mut runs: Vec<RleRun> = Vec::new();
    let mut counts_for_stats: Vec<u32> = Vec::with_capacity(resolved.len());
    let mut n_present: u32 = 0;

    let mut idx = 0usize;
    while idx < resolved.len() {
        let (start, count) = resolved[idx];
        counts_for_stats.push(count);
        if count > 0 {
            n_present += 1;
        }

        let mut run_length: u32 = 1;
        let mut cur_pos = start;
        let mut next_idx = idx + 1;
        while next_idx < resolved.len() {
            let (next_pos, next_count) = resolved[next_idx];
            // A run never crosses a gap: two positions are only part of the
            // same run when they are numerically consecutive (`next_pos ==
            // cur_pos + 1`), not merely adjacent in `positions_buf` -- an
            // ambiguous-base gap (see `kmer.rs`'s doc comment) must not be
            // silently bridged as if the count were continuous across it.
            if next_pos != cur_pos + 1 {
                break;
            }
            if next_count != count {
                break;
            }
            counts_for_stats.push(next_count);
            if next_count > 0 {
                n_present += 1;
            }
            run_length += 1;
            cur_pos = next_pos;
            next_idx += 1;
        }

        runs.push(RleRun { start, run_length, count });
        idx = next_idx;
    }

    counts_for_stats.sort_unstable();
    let stats = ProfileSummaryStats {
        n_kmers: resolved.len() as u32,
        n_present_kmers: n_present,
        min_count: counts_for_stats.first().copied(),
        median_count: median_of_sorted(&counts_for_stats),
        max_count: counts_for_stats.last().copied(),
    };

    Ok((runs, stats))
}

/// Reconstructs the exact full per-position `(position, count)` sequence a
/// read's RLE rows encode -- the lossless-round-trip half of this module's
/// output-format decision (see the module doc comment). A position with no
/// entry (because the original read had no valid k-mer there -- see
/// `kmer::extract_canonical_kmers_with_positions_into`'s doc comment on
/// ambiguous-base gaps) is simply absent from the result, not filled with a
/// synthetic value: the gap is real information, not a compression
/// artifact to paper over.
pub fn expand_rle(runs: &[RleRun]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for run in runs {
        for offset in 0..run.run_length {
            out.push((run.start + offset, run.count));
        }
    }
    out
}

/// Outcome of a profiling run: how many reads were read in total, and how
/// many of those actually yielded at least one k-mer (see
/// `ProfileSummaryStats::empty`'s doc comment for when a read yields none).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProfileStats {
    pub reads_total: u64,
    pub reads_profiled: u64,
}

/// The Parquet schema for the RLE profile table: `read_id`, `start`,
/// `run_length`, `count`. See the module doc comment for why this shape
/// (RLE, not one row per base) was chosen.
pub fn profile_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("read_id", DataType::Utf8, false),
        Field::new("start", DataType::UInt32, false),
        Field::new("run_length", DataType::UInt32, false),
        Field::new("count", DataType::UInt32, false),
    ]))
}

/// The Parquet schema for the per-read summary table: `read_id`, `n_kmers`,
/// `n_present_kmers`, `min_count`, `median_count`, `max_count`. See the
/// module doc comment's "Per-read summary statistics" section for the
/// naming and nullability rationale.
pub fn summary_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("read_id", DataType::Utf8, false),
        Field::new("n_kmers", DataType::UInt32, false),
        Field::new("n_present_kmers", DataType::UInt32, false),
        Field::new("min_count", DataType::UInt32, true),
        Field::new("median_count", DataType::Float64, true),
        Field::new("max_count", DataType::UInt32, true),
    ]))
}

/// Wraps an Arrow/Parquet serialization or writer failure as
/// `FastDnaError::Export`, mirroring `export.rs::export_err`'s own reasoning
/// for why this must not be laundered through `FastDnaError::Io` (it is not
/// an I/O failure: the bytes may never touch a disk). Kept as a local copy
/// rather than reused from `export.rs`, whose own `export_err` is private to
/// that module.
fn export_err<E>(path: &Path, err: E) -> FastDnaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    FastDnaError::Export { path: path.to_path_buf(), reason: err.to_string(), source: Some(Box::new(err)) }
}

/// Where to blame a producer-side failure -- the same small helper
/// `read_filter.rs` and `metagenomics.rs` each keep their own copy of, for
/// the same reason: a source spanning several files can name the exact file
/// it was reading; a bare in-memory source cannot, and falls back to
/// `fallback_path` plus this call's own running count.
fn failing_location<S: RecordSource>(source: &S, fallback_path: &Path, records_so_far: u64) -> (PathBuf, u64) {
    match source.current_source() {
        Some((path, records_in_file)) => (path, records_in_file + 1),
        None => (fallback_path.to_path_buf(), records_so_far + 1),
    }
}

/// The reference sequence/read id for a FASTQ/FASTA header: the first
/// whitespace-delimited token, without the leading `>`/`@` marker --
/// mirrors `metagenomics.rs`'s private `record_id_of` exactly, so a read
/// profiled here and the same read classified there are named identically.
fn read_id_of(header: &[u8]) -> String {
    let text = String::from_utf8_lossy(header);
    let trimmed = text.trim_start_matches(['>', '@']);
    trimmed.split_whitespace().next().unwrap_or("").to_string()
}

/// Accumulates one Parquet chunk of RLE profile rows before it is handed to
/// Arrow -- the same chunk-then-flush shape `export.rs::ChunkBuffers` uses,
/// sized down to this schema's four columns.
struct ProfileChunkBuffers {
    read_ids: Vec<String>,
    starts: Vec<u32>,
    run_lengths: Vec<u32>,
    counts: Vec<u32>,
}

impl ProfileChunkBuffers {
    fn with_capacity(rows: usize) -> Self {
        Self {
            read_ids: Vec::with_capacity(rows),
            starts: Vec::with_capacity(rows),
            run_lengths: Vec::with_capacity(rows),
            counts: Vec::with_capacity(rows),
        }
    }

    fn push(&mut self, read_id: &str, run: &RleRun) {
        self.read_ids.push(read_id.to_string());
        self.starts.push(run.start);
        self.run_lengths.push(run.run_length);
        self.counts.push(run.count);
    }

    fn len(&self) -> usize {
        self.starts.len()
    }

    fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

fn write_profile_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    chunk: ProfileChunkBuffers,
    path: &Path,
) -> Result<()> {
    let ProfileChunkBuffers { read_ids, starts, run_lengths, counts } = chunk;
    let id_refs: Vec<&str> = read_ids.iter().map(String::as_str).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(id_refs)),
        Arc::new(UInt32Array::from(starts)),
        Arc::new(UInt32Array::from(run_lengths)),
        Arc::new(UInt32Array::from(counts)),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| export_err(path, e))?;
    writer.write(&batch).map_err(|e| export_err(path, e))?;
    Ok(())
}

/// Accumulates one Parquet chunk of per-read summary rows. Nullable columns
/// (`min_count`/`median_count`/`max_count`) are plain `Vec<Option<_>>`:
/// Arrow's `UInt32Array`/`Float64Array` both build directly from an iterator
/// of `Option<T>`, so no separate validity-bitmap bookkeeping is needed here.
struct SummaryChunkBuffers {
    read_ids: Vec<String>,
    n_kmers: Vec<u32>,
    n_present_kmers: Vec<u32>,
    min_counts: Vec<Option<u32>>,
    median_counts: Vec<Option<f64>>,
    max_counts: Vec<Option<u32>>,
}

impl SummaryChunkBuffers {
    fn with_capacity(rows: usize) -> Self {
        Self {
            read_ids: Vec::with_capacity(rows),
            n_kmers: Vec::with_capacity(rows),
            n_present_kmers: Vec::with_capacity(rows),
            min_counts: Vec::with_capacity(rows),
            median_counts: Vec::with_capacity(rows),
            max_counts: Vec::with_capacity(rows),
        }
    }

    fn push(&mut self, read_id: &str, stats: &ProfileSummaryStats) {
        self.read_ids.push(read_id.to_string());
        self.n_kmers.push(stats.n_kmers);
        self.n_present_kmers.push(stats.n_present_kmers);
        self.min_counts.push(stats.min_count);
        self.median_counts.push(stats.median_count);
        self.max_counts.push(stats.max_count);
    }

    fn len(&self) -> usize {
        self.read_ids.len()
    }

    fn is_empty(&self) -> bool {
        self.read_ids.is_empty()
    }
}

fn write_summary_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    chunk: SummaryChunkBuffers,
    path: &Path,
) -> Result<()> {
    let SummaryChunkBuffers { read_ids, n_kmers, n_present_kmers, min_counts, median_counts, max_counts } = chunk;
    let id_refs: Vec<&str> = read_ids.iter().map(String::as_str).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(id_refs)),
        Arc::new(UInt32Array::from(n_kmers)),
        Arc::new(UInt32Array::from(n_present_kmers)),
        Arc::new(UInt32Array::from(min_counts)),
        Arc::new(Float64Array::from(median_counts)),
        Arc::new(UInt32Array::from(max_counts)),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| export_err(path, e))?;
    writer.write(&batch).map_err(|e| export_err(path, e))?;
    Ok(())
}

/// The number of rows buffered before a chunk is flushed to Arrow -- the
/// same 131,072 `export.rs`'s own chunked writers use, kept identical so
/// this module's memory footprint per chunk is directly comparable to every
/// other exporter in this crate.
const CHUNK_SIZE: usize = 131_072;

/// The full profiling run: opens `inputs` as one concatenated stream (see
/// the module doc comment's single-end scope note), profiles each read
/// against `index`, and writes the RLE profile to `profile_output` and the
/// per-read summary to `summary_output` -- both atomically, both streamed,
/// neither ever holding more than one chunk's worth of rows in memory
/// regardless of how large the input is. Shared by the CLI
/// (`main.rs::run_profile`) and the Python binding (`ffi.rs::profile_reads`)
/// so both go through exactly one validated, tested path.
pub fn run_profile(inputs: Vec<InputSpec>, index: &ProfileIndex, profile_output: &Path, summary_output: &Path) -> Result<ProfileStats> {
    let mut source = MultiSourceReader::new(inputs);
    source.validate()?;

    let (profile_file, profile_pending) = AtomicFile::create(profile_output)?;
    let profile_schema = profile_schema();
    let profile_props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![KeyValue::new("fastdna.k".to_string(), Some(index.k().to_string()))]))
        .build();
    let mut profile_writer = ArrowWriter::try_new(profile_file, profile_schema.clone(), Some(profile_props))
        .map_err(|e| export_err(profile_output, e))?;

    let (summary_file, summary_pending) = AtomicFile::create(summary_output)?;
    let summary_schema = summary_schema();
    let summary_props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![KeyValue::new("fastdna.k".to_string(), Some(index.k().to_string()))]))
        .build();
    let mut summary_writer = ArrowWriter::try_new(summary_file, summary_schema.clone(), Some(summary_props))
        .map_err(|e| export_err(summary_output, e))?;

    let mut profile_chunk = ProfileChunkBuffers::with_capacity(CHUNK_SIZE);
    let mut summary_chunk = SummaryChunkBuffers::with_capacity(CHUNK_SIZE);

    let mut record = FastqRecord::default();
    let mut positions_buf = ProfileScratch::new();
    let mut stats = ProfileStats::default();
    let mut record_no: u64 = 0;
    let fallback_path = PathBuf::from("<inputs>");

    loop {
        match source.next_record_into(&mut record) {
            Ok(true) => {
                record_no += 1;
                stats.reads_total += 1;

                let mut read_id = read_id_of(&record.id);
                if read_id.is_empty() {
                    // A header with nothing after its marker still has to
                    // be identifiable in the output tables, and its
                    // position in the file is the only name it has -- the
                    // same fallback `metagenomics.rs::classify_source` uses.
                    read_id = format!("read_{record_no}");
                }

                let (runs, read_stats) = build_read_profile(&record.seq, index, &mut positions_buf)?;
                if read_stats.n_kmers > 0 {
                    stats.reads_profiled += 1;
                }

                for run in &runs {
                    profile_chunk.push(&read_id, run);
                    if profile_chunk.len() >= CHUNK_SIZE {
                        let full = std::mem::replace(&mut profile_chunk, ProfileChunkBuffers::with_capacity(CHUNK_SIZE));
                        write_profile_chunk(&mut profile_writer, &profile_schema, full, profile_output)?;
                    }
                }

                summary_chunk.push(&read_id, &read_stats);
                if summary_chunk.len() >= CHUNK_SIZE {
                    let full = std::mem::replace(&mut summary_chunk, SummaryChunkBuffers::with_capacity(CHUNK_SIZE));
                    write_summary_chunk(&mut summary_writer, &summary_schema, full, summary_output)?;
                }
            }
            Ok(false) => break,
            Err(FastqReadError::Io(source_err)) => {
                let (path, _) = failing_location(&source, &fallback_path, stats.reads_total);
                return Err(FastDnaError::Io { path, source: source_err });
            }
            Err(FastqReadError::Malformed(reason)) => {
                let (path, record_no) = failing_location(&source, &fallback_path, stats.reads_total);
                return Err(FastDnaError::MalformedFastq { path, record: record_no, reason });
            }
        }
    }

    if !profile_chunk.is_empty() {
        write_profile_chunk(&mut profile_writer, &profile_schema, profile_chunk, profile_output)?;
    }
    if !summary_chunk.is_empty() {
        write_summary_chunk(&mut summary_writer, &summary_schema, summary_chunk, summary_output)?;
    }

    // `close` consumes each writer and with it the last handle on its temp
    // file; only then can the rename-over-destination succeed on Windows --
    // the same ordering `export.rs`'s own writers rely on.
    profile_writer.close().map_err(|e| export_err(profile_output, e))?;
    profile_pending.commit()?;
    summary_writer.close().map_err(|e| export_err(summary_output, e))?;
    summary_pending.commit()?;

    Ok(stats)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_read_profile_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a `ProfileIndex` at k=4 from a hand-picked list of canonical
    /// k-mer encodings with known counts, through the real counting +
    /// export path -- matching `ktab.rs`'s/`read_filter.rs`'s own test
    /// convention: these tests exercise exactly the files `fastdna count`
    /// produces.
    fn build_index(name: &str, k: usize, entries: &[u64]) -> (ProfileIndex, PathBuf) {
        let path = temp_path(name);
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, &path, k, 1, false).unwrap();
        let table = KmerTable::open(&path).unwrap();
        (ProfileIndex::from_table(&table).unwrap(), path)
    }

    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    // -- ProfileIndex ------------------------------------------------------

    #[test]
    fn profile_index_reports_the_exact_recorded_count() {
        let (index, p) = build_index("index_hit", 4, &[6, 6, 6, 0]);
        assert_eq!(index.get_narrow(6), Some(3));
        assert_eq!(index.get_narrow(0), Some(1));
        assert_eq!(index.get_narrow(3), None, "3 lies between the two keys but is not itself present");
        cleanup(&[p]);
    }

    #[test]
    fn profile_index_from_an_empty_table_has_nothing() {
        let (index, p) = build_index("index_empty", 4, &[]);
        assert!(index.is_empty());
        assert_eq!(index.get_narrow(0), None);
        cleanup(&[p]);
    }

    // -- build_read_profile: the core cases the task calls for -------------

    /// A read whose k-mers are all present, all with the same known count:
    /// one single RLE run spanning every position, and a summary with that
    /// exact count as min == median == max.
    #[test]
    fn a_read_fully_covered_at_one_count_produces_one_run() {
        // k=4, "AAAACAAA" -> 5 canonical k-mers at positions 0..=4, chosen
        // (unlike a simpler repeating sequence such as "ACGTACGT") so that
        // all five are pairwise *distinct* canonical encodings -- otherwise
        // two positions sharing one reference row would silently sum their
        // insertions and break the "every position reports exactly 7" setup
        // this test relies on. Put every one of those five exact encodings
        // into the reference at count 7.
        let seq = b"AAAACAAA";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        assert_eq!(kmers.len(), 5, "sanity: this fixture must yield exactly 5 k-mers");
        assert_eq!(
            kmers.iter().collect::<std::collections::HashSet<_>>().len(),
            5,
            "sanity: this fixture's 5 k-mers must be pairwise distinct"
        );
        let mut entries = Vec::new();
        for &km in &kmers {
            for _ in 0..7 {
                entries.push(km);
            }
        }
        let (index, p) = build_index("one_run", 4, &entries);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(seq, &index, &mut buf).unwrap();

        assert_eq!(runs, vec![RleRun { start: 0, run_length: 5, count: 7 }]);
        assert_eq!(stats.n_kmers, 5);
        assert_eq!(stats.n_present_kmers, 5);
        assert_eq!(stats.min_count, Some(7));
        assert_eq!(stats.median_count, Some(7.0));
        assert_eq!(stats.max_count, Some(7));

        cleanup(&[p]);
    }

    /// A read whose k-mers are entirely absent from the reference: every
    /// position reports count 0, folded into one run, and `n_present_kmers`
    /// is 0 while `n_kmers` is not -- the case that proves "absent" and "no
    /// data" are kept distinct.
    #[test]
    fn a_read_with_no_kmers_in_the_reference_is_one_run_of_zero() {
        let seq = b"ACGTACGT";
        let unrelated = kmer::extract_canonical_kmers(b"GGGGGGGG", 4);
        let (index, p) = build_index("all_absent", 4, &unrelated);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(seq, &index, &mut buf).unwrap();

        assert_eq!(runs, vec![RleRun { start: 0, run_length: 5, count: 0 }]);
        assert_eq!(stats.n_kmers, 5);
        assert_eq!(stats.n_present_kmers, 0, "none of this read's k-mers are in the reference");
        assert_eq!(stats.min_count, Some(0));
        assert_eq!(stats.median_count, Some(0.0));
        assert_eq!(stats.max_count, Some(0));

        cleanup(&[p]);
    }

    /// A read shorter than `k` yields no k-mers at all -- an empty profile
    /// and an all-null summary, not an error and not a zero-filled run.
    #[test]
    fn a_read_shorter_than_k_yields_an_empty_profile_and_a_null_summary() {
        let (index, p) = build_index("too_short", 8, &[1, 2, 3]);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(b"ACG", &index, &mut buf).unwrap();

        assert!(runs.is_empty());
        assert_eq!(stats, ProfileSummaryStats::empty());

        cleanup(&[p]);
    }

    /// A run must not bridge the gap an ambiguous base leaves (see
    /// `kmer.rs`'s doc comment on `extract_canonical_kmers_with_positions_
    /// into`): even when the count on both sides happens to be identical,
    /// the discontinuity in read position must start a new run, not extend
    /// the first one across positions that have no k-mer at all.
    #[test]
    fn a_run_never_bridges_an_ambiguous_base_gap_even_at_equal_counts() {
        // k=4: "ACGT" (positions 0..=3 -> only position 0 is a valid
        // window), then "N", then "ACGT" again (only position 5 is valid
        // after the reset -- see kmer.rs's own positional test for why).
        let seq = b"ACGTNACGT";
        let acgt_kmer = kmer::extract_canonical_kmers(b"ACGT", 4)[0];
        // Same k-mer both sides of the gap, same count either way -- if
        // runs merged by count alone (ignoring position continuity) this
        // would wrongly collapse into a single run.
        let (index, p) = build_index("gap_same_count", 4, &[acgt_kmer, acgt_kmer]);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(seq, &index, &mut buf).unwrap();

        assert_eq!(
            runs,
            vec![RleRun { start: 0, run_length: 1, count: 2 }, RleRun { start: 5, run_length: 1, count: 2 }],
            "the gap around the ambiguous base must produce two runs, not one merged across it"
        );
        assert_eq!(stats.n_kmers, 2);

        cleanup(&[p]);
    }

    /// A run breaks exactly where the count changes, and only there --
    /// pinning the RLE construction against a read with a genuine multi-run
    /// profile (the realistic "mostly one depth, one low-count blip" shape
    /// this format exists to compress).
    #[test]
    fn a_count_change_starts_a_new_run_at_the_exact_position() {
        // k=4, "AAAACAAA" -> 5 pairwise-distinct canonical k-mers at
        // positions 0..=4 (same fixture as the "one run" test above, picked
        // for the same reason: no two positions alias to the same
        // reference row). Because every position's k-mer is unique, each
        // one's reference count can be set independently by construction,
        // with no risk of two positions silently sharing (and summing)
        // one counter entry -- give position 2 alone a distinct, lower
        // count than its four neighbours.
        let seq = b"AAAACAAA";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        assert_eq!(kmers.len(), 5);
        let mut entries = Vec::new();
        for (i, &km) in kmers.iter().enumerate() {
            let count = if i == 2 { 1 } else { 9 };
            for _ in 0..count {
                entries.push(km);
            }
        }
        let (index, p) = build_index("count_change", 4, &entries);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(seq, &index, &mut buf).unwrap();

        assert_eq!(
            runs,
            vec![
                RleRun { start: 0, run_length: 2, count: 9 },
                RleRun { start: 2, run_length: 1, count: 1 },
                RleRun { start: 3, run_length: 2, count: 9 },
            ]
        );
        assert_eq!(stats.n_kmers, 5);
        assert_eq!(stats.n_present_kmers, 5);
        assert_eq!(stats.min_count, Some(1));
        assert_eq!(stats.max_count, Some(9));
        // Sorted counts: [1, 9, 9, 9, 9] -> median is the middle (index 2) = 9.
        assert_eq!(stats.median_count, Some(9.0));

        cleanup(&[p]);
    }

    /// Ambiguous-base handling matches the shared k-mer extractor exactly
    /// -- the flattened `(position, kmer)` pairs behind a profile must be
    /// identical to what `kmer::extract_canonical_kmers_with_positions_
    /// into` itself returns, since `build_read_profile` is a thin
    /// count-lookup layer over that extraction, not a second
    /// implementation of it.
    #[test]
    fn ambiguous_base_handling_matches_the_shared_extractor() {
        let seq = b"ACGTNNNNACGTACGT";
        let (index, p) = build_index("ambiguous", 4, &[]);

        let mut buf = ProfileScratch::new();
        let (runs, stats) = build_read_profile(seq, &index, &mut buf).unwrap();

        let mut expected_positions = Vec::new();
        kmer::extract_canonical_kmers_with_positions_into(seq, 4, &mut expected_positions);

        let flattened = expand_rle(&runs);
        let flattened_positions: Vec<u32> = flattened.iter().map(|&(pos, _)| pos).collect();
        let expected_only_positions: Vec<u32> = expected_positions.iter().map(|&(pos, _)| pos).collect();
        assert_eq!(flattened_positions, expected_only_positions);
        assert_eq!(stats.n_kmers, expected_positions.len() as u32);

        cleanup(&[p]);
    }

    // -- expand_rle: lossless round trip -----------------------------------

    #[test]
    fn expand_rle_round_trips_a_multi_run_profile_exactly() {
        let runs = vec![
            RleRun { start: 0, run_length: 3, count: 5 },
            RleRun { start: 3, run_length: 1, count: 1 },
            RleRun { start: 4, run_length: 2, count: 5 },
        ];
        let expanded = expand_rle(&runs);
        assert_eq!(
            expanded,
            vec![(0, 5), (1, 5), (2, 5), (3, 1), (4, 5), (5, 5)],
            "expand_rle must reproduce the exact full per-position sequence"
        );
    }

    #[test]
    fn expand_rle_leaves_gaps_where_the_original_runs_left_gaps() {
        // Positions 5..=8 missing (an ambiguous-base gap) must stay
        // missing after expansion, not be silently filled in.
        let runs = vec![RleRun { start: 0, run_length: 5, count: 2 }, RleRun { start: 9, run_length: 2, count: 2 }];
        let expanded = expand_rle(&runs);
        let positions: Vec<u32> = expanded.iter().map(|&(pos, _)| pos).collect();
        assert_eq!(positions, vec![0, 1, 2, 3, 4, 9, 10]);
    }

    #[test]
    fn expand_rle_of_an_empty_profile_is_empty() {
        assert!(expand_rle(&[]).is_empty());
    }

    /// For every one of `build_read_profile`'s own fixtures above,
    /// `expand_rle` applied to its output must reproduce exactly the counts
    /// a direct (non-RLE) per-position lookup would have produced --
    /// pinning the round-trip property against the real construction path,
    /// not just against hand-built `RleRun` literals.
    #[test]
    fn expand_rle_round_trips_build_read_profiles_own_output() {
        let seq = b"AAAACAAA";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        let mut entries = Vec::new();
        for (i, &km) in kmers.iter().enumerate() {
            let count = if i == 2 { 1 } else { 9 };
            for _ in 0..count {
                entries.push(km);
            }
        }
        let (index, p) = build_index("round_trip_real", 4, &entries);

        let mut buf = ProfileScratch::new();
        let (runs, _stats) = build_read_profile(seq, &index, &mut buf).unwrap();
        let expanded = expand_rle(&runs);

        // Independently recompute the expected per-position counts by
        // looking each position's k-mer up directly, with no RLE involved.
        let mut positions: Vec<(u32, u64)> = Vec::new();
        kmer::extract_canonical_kmers_with_positions_into(seq, 4, &mut positions);
        let direct: Vec<(u32, u32)> =
            positions.iter().map(|&(pos, km)| (pos, index.get_narrow(km).unwrap_or(0))).collect();

        assert_eq!(expanded, direct);
        cleanup(&[p]);
    }

    // -- run_profile: end-to-end through real files -------------------------

    /// Reads a Parquet file back into `(read_id, count)` pairs from the RLE
    /// profile schema -- a minimal reader used only by this module's own
    /// tests to check what `run_profile` actually wrote.
    fn read_profile_rows(path: &Path) -> Vec<(String, u32, u32, u32)> {
        use arrow::array::{Array, UInt32Array as A32};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let starts = batch.column(1).as_any().downcast_ref::<A32>().unwrap();
            let run_lengths = batch.column(2).as_any().downcast_ref::<A32>().unwrap();
            let counts = batch.column(3).as_any().downcast_ref::<A32>().unwrap();
            for i in 0..batch.num_rows() {
                rows.push((ids.value(i).to_string(), starts.value(i), run_lengths.value(i), counts.value(i)));
            }
        }
        rows
    }

    #[test]
    fn run_profile_writes_both_output_files_with_matching_read_ids() {
        use crate::fastq::InputSpec;

        let dir = std::env::temp_dir().join("fastdna_run_profile_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        let input_path = dir.join(format!("in_{unique}.fastq"));
        // r1 = "AAAACAAA" (k=4): five pairwise-distinct canonical k-mers
        // (see `a_read_fully_covered_at_one_count_produces_one_run`'s own
        // comment for why a non-repeating fixture matters here). r2 =
        // "GGGG" canonicalizes to a value disjoint from all five of r1's,
        // so it is genuinely absent from a reference built only from r1's
        // own k-mers, not merely reported absent by coincidence.
        std::fs::write(&input_path, b"@r1\nAAAACAAA\n+\nIIIIIIII\n@r2\nGGGG\n+\nIIII\n").unwrap();

        let seq = b"AAAACAAA";
        let kmers = kmer::extract_canonical_kmers(seq, 4);
        let (index, index_path) = build_index("run_profile_e2e", 4, &kmers);

        let profile_out = dir.join(format!("profile_{unique}.parquet"));
        let summary_out = dir.join(format!("summary_{unique}.parquet"));

        let stats =
            run_profile(vec![InputSpec::File(input_path.clone())], &index, &profile_out, &summary_out).unwrap();
        assert_eq!(stats.reads_total, 2);
        assert_eq!(stats.reads_profiled, 2, "even r2 (all-absent k-mers) still counts as profiled: it has k-mers");

        let rows = read_profile_rows(&profile_out);
        let r1_rows: Vec<_> = rows.iter().filter(|r| r.0 == "r1").collect();
        assert_eq!(r1_rows.len(), 1, "r1's uniform-count profile must collapse to one run");
        assert_eq!(r1_rows[0].3, 1, "each of r1's k-mers appears once in the reference");

        let r2_rows: Vec<_> = rows.iter().filter(|r| r.0 == "r2").collect();
        assert_eq!(r2_rows.len(), 1, "r2's k-mers are entirely absent from the reference: one run of count 0");
        assert_eq!(r2_rows[0].3, 0);

        cleanup(&[index_path]);
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&profile_out);
        let _ = std::fs::remove_file(&summary_out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_profile_reports_the_correct_footer_k() {
        use crate::fastq::InputSpec;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let dir = std::env::temp_dir().join("fastdna_run_profile_footer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        let input_path = dir.join(format!("in_{unique}.fastq"));
        std::fs::write(&input_path, b"@r1\nACGTACGT\n+\nIIIIIIII\n").unwrap();

        let (index, index_path) = build_index("run_profile_footer", 4, &[]);
        let profile_out = dir.join(format!("profile_{unique}.parquet"));
        let summary_out = dir.join(format!("summary_{unique}.parquet"));

        run_profile(vec![InputSpec::File(input_path.clone())], &index, &profile_out, &summary_out).unwrap();

        for path in [&profile_out, &summary_out] {
            let file = File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let metadata = builder.metadata().clone();
            let kvs = metadata.file_metadata().key_value_metadata().unwrap();
            let k_value = kvs.iter().find(|kv| kv.key == "fastdna.k").and_then(|kv| kv.value.clone());
            assert_eq!(k_value, Some("4".to_string()));
        }

        cleanup(&[index_path]);
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&profile_out);
        let _ = std::fs::remove_file(&summary_out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_profile_rejects_an_input_list_with_no_files() {
        let (index, p) = build_index("run_profile_empty_input", 4, &[]);
        let dir = std::env::temp_dir().join("fastdna_run_profile_empty_input_test");
        std::fs::create_dir_all(&dir).unwrap();
        let profile_out = dir.join("profile.parquet");
        let summary_out = dir.join("summary.parquet");

        let result = run_profile(Vec::new(), &index, &profile_out, &summary_out);
        assert!(matches!(result, Err(FastDnaError::InvalidConfig { .. })));

        cleanup(&[p]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
