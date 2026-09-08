// src/ktab.rs
//! A random-access query layer over the sorted `(kmer_u64, frequency)`
//! Parquet table `export.rs` already writes -- KMC's `.kmc_pre`/`.kmc_suf`,
//! FastK's `.ktab` and Jellyfish's `.jf` all give competitors a binary
//! k-mer database with fast point lookup; this module is FastDNA's answer
//! to that gap (`docs/feature-gap-analysis.md`'s S1).
//!
//! # Design decision: no new file format
//!
//! Two designs were on the table: (1) a custom fixed-record binary file
//! (12-byte `(u64, u32)` records) with a hand-rolled sparse index, or (2)
//! declaring the sorted Parquet this crate already writes to be the k-mer
//! table format, and building a reader over it that uses Parquet's own
//! per-row-group column statistics for pruning. (2) was chosen, for three
//! reasons:
//!
//! 1. **The table already exists.** `counter.rs`'s `KmerCounter::iter`
//!    yields `(kmer, count)` pairs in strictly ascending `kmer_u64` order
//!    by construction (`CountTable`'s own invariant, enforced by every
//!    merge path in that module), and `export::export_counts_parquet`
//!    already writes exactly that order to Parquet, one row group per
//!    131,072-row chunk. There is nothing to invent -- only something to
//!    declare and read back with a query API, which is also why
//!    `write_table` is not a new writer at all: it is `export_counts_
//!    parquet_with_metadata` with two footer key-value pairs attached (see
//!    below), reusing the exact same tested chunked-write path every other
//!    export already goes through.
//! 2. **A format only this tool reads is the same lock-in this crate
//!    already declined elsewhere** (`docs/feature-gap-analysis.md`'s
//!    "Deliberately not copying" list rejects FastK's/KMC's own binary
//!    formats for the same reason). A `.parquet` k-mer table opens today in
//!    pandas, polars, DuckDB and Spark with zero code from this crate,
//!    which is a direct, load-bearing win for a counter whose output is
//!    meant to be read by other tools (`docs/goal-fast-kmer-counter.md`).
//!    A custom binary format would not extend that reach at all.
//! 3. **Parquet row-group statistics genuinely support pruned point
//!    lookups**, which was confirmed rather than assumed before committing
//!    to this design (per this task's own instruction): `parquet::arrow::
//!    arrow_reader::ParquetRecordBatchReaderBuilder::with_row_groups` lets a
//!    reader open only the row groups a query can possibly match, selected
//!    from `ParquetMetaData`'s per-column-chunk `Statistics` (min/max) with
//!    no row ever decoded -- see `KmerTable::range` below, which both
//!    `get` and `iter` are built from.
//!
//! # What this trades away
//!
//! A hand-rolled sparse index (option 1, or the "load the whole key column
//! into memory" variant this task's plan doc considered) answers a point
//! query with an O(log n) in-memory binary search once the index is
//! resident. This design instead re-decodes at least one row group
//! (≤131,072 rows) per `get()` call, and reopens the file for every call to
//! `range`/`get` rather than keeping a persistent reader -- correctness and
//! simplicity were prioritized over shaving that cost, and the OS page
//! cache absorbs most of it on repeated queries against the same file
//! within one process. A caller doing many lookups against the same table
//! should prefer `range`/`iter` (which decode each touched row group
//! exactly once, streaming, regardless of how many k-mers inside it are
//! asked about) over calling `get` in a tight loop. Building a persistent,
//! reusable reader/cache is a reasonable follow-up once a real access
//! pattern justifies it -- not built here on a guess, per this crate's own
//! "measure before extending" standard (`docs/superpowers/plans/
//! 2026-08-24-completeness-phase.md`, Task 1's closing bullet).
//!
//! # Scope: the in-memory counting strategy's output only
//!
//! `counter.rs`'s `KmerCounter` (in-memory strategy) and `disk_spill.rs`
//! (disk-partitioned strategy) both produce a globally sorted `(u64, u32)`
//! table with the same shape, and both eventually reach `export.rs`'s
//! Parquet writer -- so a table built from either strategy is a valid
//! `KmerTable` to `open`. What is *not* covered here: `binned.rs`'s
//! opt-in minimizer-partitioned strategy is not wired into the normal
//! `count` -> export path at all (`docs/feature-gap-analysis.md`'s S5), so
//! there is nothing extra for this module to support or exclude there.

use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, UInt32Array, UInt64Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::file::statistics::Statistics;

use crate::error::{FastDnaError, Result};
use crate::kmer;

/// Parquet key-value metadata key recording that a file's rows are sorted
/// ascending by the column named in its value (`SORTED_BY_VALUE`).
/// `KmerTable::open` requires this to be present and correct before it will
/// treat a `.parquet` file as a queryable k-mer table -- see the module doc
/// comment's design-decision section for why a missing or wrong value is
/// rejected outright rather than the file being scanned to verify sortedness
/// from scratch (that scan is exactly the cost a lazy `open` exists to
/// avoid).
pub const SORTED_BY_KEY: &str = "fastdna.sorted_by";
/// The only value `SORTED_BY_KEY` is ever written with: every k-mer table
/// this crate writes is sorted by its `kmer_u64` column.
pub const SORTED_BY_VALUE: &str = "kmer_u64";
/// The `SORTED_BY_KEY` value a **wide** table carries (`33 <= k <= 64`,
/// `wide_kmer.rs`). A separate value, not a separate contract: the key
/// column is 16 big-endian bytes rather than a `u64`, and big-endian is
/// chosen precisely so byte order is still numeric order, which is what
/// keeps "sorted by the key column" meaning the same thing for both.
///
/// `KmerTable::open` accepts only `SORTED_BY_VALUE`, so a wide table is
/// rejected by name rather than misread as a narrow one. A wide table is
/// read by [`crate::wide_ktab::WideKmerTable`] instead, which `query`, the
/// set operations and `similarity` select via [`table_key`]. The two
/// operations with no wide form are `filter` and `profile`: they index the
/// reference as a `Vec<u64>` and binary-search it once per read, so the key
/// type there is the data structure rather than an annotation.
pub const SORTED_BY_WIDE_VALUE: &str = "kmer_bits";
/// Parquet key-value metadata key recording the `k` every row's `kmer_u64`
/// was packed with. Needed because a raw `u64` cannot be decoded (or a
/// query sequence encoded) without knowing `k` -- unlike the counts schema
/// itself, `k` is not a column, since every row in one file shares the same
/// value.
pub const K_KEY: &str = "fastdna.k";

/// One row group's `kmer_u64` range, read once from `ParquetMetaData` at
/// `open` time -- cheap (footer-only, no row decoded) and enough to prune
/// every `get`/`range` call down to the row groups that can possibly
/// contain a match.
#[derive(Debug, Clone, Copy)]
struct RowGroupRange {
    /// Index into the file's row groups, as `with_row_groups` expects it.
    index: usize,
    min: u64,
    max: u64,
}

/// A handle to a sorted `(kmer_u64, frequency)` Parquet table, opened
/// without decoding any row. See the module doc comment for the design
/// this implements and what it trades away.
#[derive(Debug, Clone)]
pub struct KmerTable {
    path: PathBuf,
    k: usize,
    total_rows: u64,
    kmer_col: usize,
    freq_col: usize,
    /// Ascending, non-overlapping by construction (checked at `open` time):
    /// `row_groups[i].max <= row_groups[i + 1].min` for every adjacent pair.
    row_groups: Vec<RowGroupRange>,
}

/// Wraps a Parquet/Arrow error encountered while reading a table as
/// `FastDnaError::Load`, preserving the original error as `source` --
/// mirrors `export.rs::export_err`'s reasoning for why this must not be
/// laundered through `FastDnaError::Io` (it is not an I/O failure; the
/// bytes may already be fully read and simply not decode).
fn load_err<E>(path: &Path, err: E) -> FastDnaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    FastDnaError::Load { path: path.to_path_buf(), reason: err.to_string(), source: Some(Box::new(err)) }
}

/// A `FastDnaError::Load` with no underlying error object -- for a file
/// that opens and decodes fine but fails one of `open`'s own consistency
/// checks (wrong schema, missing metadata, an unsorted row-group boundary).
fn load_reason(path: &Path, reason: String) -> FastDnaError {
    FastDnaError::Load { path: path.to_path_buf(), reason, source: None }
}

/// Which key column a `.parquet` k-mer table is sorted by, and therefore
/// which reader can open it: [`KmerTable`] for `Narrow`,
/// [`crate::wide_ktab::WideKmerTable`] for `Wide`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKey {
    /// `kmer_u64` -- written at `k <= 32`.
    Narrow,
    /// `kmer_bits` -- written at `33 <= k <= 64`.
    Wide,
}

/// Reads only `path`'s Parquet footer and reports which key column it
/// declares.
///
/// Exists so a caller that can handle either width (`fastdna query`) picks
/// the right reader in one step, instead of opening with one and falling
/// back to the other on failure -- a fallback whose error message, when
/// *both* fail, is necessarily the wrong one of the two.
///
/// `FastDnaError::Load` if the file is not Parquet, or carries no
/// `fastdna.sorted_by` metadata naming a key column this crate writes. It
/// deliberately validates nothing else: the reader it selects does the
/// full check, and duplicating that here would be two places to keep in
/// agreement.
pub fn table_key<P: AsRef<Path>>(path: P) -> Result<TableKey> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| load_err(path, e))?;
    let metadata = builder.metadata().file_metadata().clone();

    let sorted_by = metadata
        .key_value_metadata()
        .and_then(|pairs| pairs.iter().find(|p| p.key == SORTED_BY_KEY))
        .and_then(|p| p.value.as_deref());

    match sorted_by {
        Some(SORTED_BY_VALUE) => Ok(TableKey::Narrow),
        Some(SORTED_BY_WIDE_VALUE) => Ok(TableKey::Wide),
        Some(other) => Err(load_reason(
            path,
            format!(
                "'{SORTED_BY_KEY}' says this file is sorted by '{other}', which is not a k-mer \
                 key column this crate writes ('{SORTED_BY_VALUE}' for k<=32, \
                 '{SORTED_BY_WIDE_VALUE}' above it)"
            ),
        )),
        None => Err(load_reason(
            path,
            format!(
                "missing '{SORTED_BY_KEY}' Parquet metadata -- this file was not written as a \
                 FastDNA k-mer table (run `fastdna count` with a .parquet output, which writes it \
                 automatically)"
            ),
        )),
    }
}

impl KmerTable {
    /// Opens `path` and validates it as a queryable k-mer table: the
    /// Parquet footer is read (schema, key-value metadata, one row group's
    /// worth of `Statistics` per row group), but no row is decoded --
    /// `open` costs a handful of KB of footer I/O regardless of how large
    /// the table is.
    ///
    /// Rejected as `FastDnaError::Load`, with an actionable reason, rather
    /// than silently guessing or panicking:
    /// - a schema that does not have a `kmer_u64: UInt64` column and a
    ///   `frequency: UInt32` column (an optional `kmer_sequence` column, as
    ///   `--with-sequence` adds, is allowed and simply ignored -- the query
    ///   API never needs it);
    /// - missing or mismatched `fastdna.sorted_by`/`fastdna.k` metadata
    ///   (see `SORTED_BY_KEY`'s doc comment for why this is required rather
    ///   than independently re-verified);
    /// - a row group whose `kmer_u64` column carries no statistics (nothing
    ///   to prune or verify order with);
    /// - row groups whose ascending order is violated at their boundaries
    ///   (the cheapest check that can be made without decoding a row: see
    ///   the module doc comment for what this does and does not prove).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| FastDnaError::Io { path: path.clone(), source: e })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| load_err(&path, e))?;

        let schema = builder.schema();
        // Both columns must be declared non-nullable, not merely the right
        // type: `counts_schema`/`counts_schema`-shaped writers never emit a
        // null in either column (every row is a real, counted k-mer), and
        // `get`/`RangeIter` below decode with `Array::value(i)`, which
        // silently returns the *other* row's payload for a null slot rather
        // than erroring (Arrow's null-safe `value` contract is "the bit
        // pattern is unspecified for a null index", not "zero" or "panic").
        // A table with `kmer_u64=[null, 10, 20]`, `frequency=[5, null, 7]`
        // would otherwise open as `len=3` and answer `get(10) == Some(5)` --
        // another row's count entirely. Rejecting a nullable schema here is
        // the one check that closes that door, before a single row is ever
        // decoded.
        let kmer_col = schema
            .index_of("kmer_u64")
            .ok()
            .filter(|&i| schema.field(i).data_type() == &DataType::UInt64 && !schema.field(i).is_nullable())
            .ok_or_else(|| {
                load_reason(
                    &path,
                    "missing a non-nullable kmer_u64: uint64 column. A table written at k>32 \
                     keys on a 16-byte kmer_bits column instead (see SORTED_BY_WIDE_VALUE) and \
                     is read by wide_ktab::WideKmerTable, which query, the set operations and \
                     similarity all select for themselves via `table_key`. Only read filtering \
                     and profiling have no wide form -- they index the reference as a Vec<u64> \
                     -- so reaching this message means one of those was asked for a wide \
                     table. Anything else is not a FastDNA k-mer table at all"
                        .to_string(),
                )
            })?;
        let freq_col = schema
            .index_of("frequency")
            .ok()
            .filter(|&i| schema.field(i).data_type() == &DataType::UInt32 && !schema.field(i).is_nullable())
            .ok_or_else(|| {
                load_reason(
                    &path,
                    "missing a non-nullable frequency: uint32 column -- this is not a FastDNA k-mer table"
                        .to_string(),
                )
            })?;

        let metadata = builder.metadata().clone();
        let file_meta = metadata.file_metadata();
        let kv = file_meta.key_value_metadata();

        let sorted_by_ok = kv.is_some_and(|pairs| {
            pairs.iter().any(|p| p.key == SORTED_BY_KEY && p.value.as_deref() == Some(SORTED_BY_VALUE))
        });
        if !sorted_by_ok {
            return Err(load_reason(
                &path,
                format!(
                    "missing or unexpected '{SORTED_BY_KEY}' Parquet metadata -- this file was not \
                     written as a FastDNA k-mer table (run `fastdna count` with a .parquet output, \
                     which writes this automatically)"
                ),
            ));
        }
        let k: usize = kv
            .and_then(|pairs| pairs.iter().find(|p| p.key == K_KEY))
            .and_then(|p| p.value.as_deref())
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| load_reason(&path, format!("missing or unparsable '{K_KEY}' Parquet metadata")))?;

        let mut row_groups = Vec::with_capacity(metadata.num_row_groups());
        let mut total_rows: u64 = 0;
        let mut prev_max: Option<u64> = None;

        for (index, rg) in metadata.row_groups().iter().enumerate() {
            // A row group with no rows (a table written from an empty
            // source, e.g. `pyarrow.parquet.write_table` on a zero-row
            // `Table`) carries no meaningful min/max -- there is nothing to
            // prune or to verify sort order against, and nothing for `get`/
            // `range` to ever find there, so it is simply skipped rather
            // than rejected. This is the one case `total_rows` is not
            // incremented by, since `rg.num_rows()` is already 0.
            if rg.num_rows() == 0 {
                continue;
            }
            let stats = rg.column(kmer_col).statistics().ok_or_else(|| {
                load_reason(&path, format!("row group {index} has no statistics on kmer_u64 -- cannot query it"))
            })?;
            let (min, max) = match stats {
                Statistics::Int64(v) => {
                    let min = *v.min_opt().ok_or_else(|| {
                        load_reason(&path, format!("row group {index}'s kmer_u64 statistics have no minimum"))
                    })?;
                    let max = *v.max_opt().ok_or_else(|| {
                        load_reason(&path, format!("row group {index}'s kmer_u64 statistics have no maximum"))
                    })?;
                    // `kmer_u64` is stored as Parquet's INT64 physical type
                    // with an unsigned logical annotation (Arrow's UInt64 ->
                    // Parquet mapping): the min/max are the correct values
                    // for *unsigned* comparison already, just carried in an
                    // `i64` field. `as u64` reinterprets the same bit
                    // pattern rather than performing a numeric conversion
                    // (which would corrupt any value above `i64::MAX`, real
                    // for k=32 canonical k-mers using the full 64-bit
                    // range), so this is a bit cast, not a value cast.
                    (min as u64, max as u64)
                }
                other => {
                    return Err(load_reason(
                        &path,
                        format!("row group {index}'s kmer_u64 statistics are an unexpected type: {other:?}"),
                    ))
                }
            };
            if min > max {
                return Err(load_reason(&path, format!("row group {index} has an invalid statistics range")));
            }
            if let Some(prev) = prev_max {
                if min < prev {
                    return Err(load_reason(
                        &path,
                        format!(
                            "row group {index} starts at k-mer {min}, before the previous row group's \
                             maximum {prev} -- the file is not sorted ascending by kmer_u64 and cannot \
                             be queried as a FastDNA k-mer table"
                        ),
                    ));
                }
            }
            prev_max = Some(max);
            total_rows += rg.num_rows() as u64;
            row_groups.push(RowGroupRange { index, min, max });
        }

        Ok(Self { path, k, total_rows, kmer_col, freq_col, row_groups })
    }

    /// The `k` every row's `kmer_u64` was packed with (from the table's own
    /// `fastdna.k` metadata -- never re-derived from row contents, since an
    /// empty table has no rows to derive it from).
    pub fn k(&self) -> usize {
        self.k
    }

    /// The file this table was opened from. Needed by any caller that must
    /// check a *destination* path against a table's own source before
    /// writing to it (`read_filter::run_filter`'s reference-table guard,
    /// `setops`'s "don't let a set operation's output overwrite one of its
    /// own inputs" guard) -- both need the path a `KmerTable` was built
    /// from, not just its decoded rows.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total row count, read from Parquet row-group metadata at `open` time
    /// (no row decoded) -- the table's distinct-k-mer count.
    pub fn len(&self) -> u64 {
        self.total_rows
    }

    pub fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// Point lookup: the frequency recorded for `kmer`, or `None` if it is
    /// absent from the table. `kmer` must already be the canonical,
    /// `k`-bit-packed encoding this table's rows use -- see
    /// `encode_query_kmer` for turning a DNA sequence or a decimal string
    /// into that form.
    ///
    /// Implemented as `range(kmer, kmer)`'s first (and only possible)
    /// result: both share the same row-group pruning, so a lookup for a
    /// k-mer outside every row group's `[min, max]` never opens a reader at
    /// all, and one whose range does match decodes at most that one row
    /// group.
    pub fn get(&self, kmer: u64) -> Result<Option<u32>> {
        match self.range(kmer, kmer)?.next() {
            Some(Ok((_, count))) => Ok(Some(count)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Every `(kmer, frequency)` pair with `low <= kmer <= high`, ascending,
    /// streamed row group by row group: only row groups whose statistics
    /// range overlaps `[low, high]` are ever opened
    /// (`ParquetRecordBatchReaderBuilder::with_row_groups`), and each
    /// touched row group is decoded in Arrow's own batches rather than all
    /// at once, so no call materializes more than one batch's worth of the
    /// table in memory regardless of how wide the requested range is.
    pub fn range(&self, low: u64, high: u64) -> Result<RangeIter> {
        let pruned: Vec<usize> =
            self.row_groups.iter().filter(|rg| rg.max >= low && rg.min <= high).map(|rg| rg.index).collect();

        let file = File::open(&self.path).map_err(|e| FastDnaError::Io { path: self.path.clone(), source: e })?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| load_err(&self.path, e))?
            .with_row_groups(pruned)
            .build()
            .map_err(|e| load_err(&self.path, e))?;

        Ok(RangeIter {
            path: self.path.clone(),
            reader,
            low,
            high,
            kmer_col: self.kmer_col,
            freq_col: self.freq_col,
            buffer: VecDeque::new(),
            last_yielded: None,
        })
    }

    /// Every `(kmer, frequency)` pair in the table, ascending -- `range`
    /// over the full `u64` domain.
    pub fn iter(&self) -> Result<RangeIter> {
        self.range(0, u64::MAX)
    }
}

/// Streaming iterator over a (possibly row-group-pruned) range of a
/// `KmerTable`, returned by [`KmerTable::range`]/[`KmerTable::iter`]. Each
/// `next()` call decodes at most one more Arrow batch from the underlying
/// Parquet reader; a batch already read is drained from `buffer` before the
/// reader is touched again.
pub struct RangeIter {
    path: PathBuf,
    reader: ParquetRecordBatchReader,
    low: u64,
    high: u64,
    kmer_col: usize,
    freq_col: usize,
    buffer: VecDeque<(u64, u32)>,
    /// The last `kmer_u64` this iterator has actually handed back, across
    /// every Arrow batch decoded so far (not just the current one) --
    /// `None` before the first row is yielded.
    ///
    /// `KmerTable::open` only verifies order at row-group *boundaries*
    /// (cheap, footer-statistics-only -- see its own doc comment for why);
    /// a single-row-group table that carries the right `fastdna.sorted_by`
    /// metadata but whose rows are not actually ascending is accepted by
    /// `open` by design. Both of this iterator's real consumers --
    /// `setops::MultiTableMerge`'s binary-heap merge and `read_filter::
    /// ReferenceIndex`'s `binary_search` -- silently assume this iterator's
    /// own order guarantee holds, so a corrupt or hand-built (e.g. plain
    /// pyarrow) file that violates it must not be allowed to propagate
    /// silently wrong results (a merge-join miscounting an intersection, a
    /// binary search finding the wrong row's frequency). This field lets
    /// `next` compare across batch boundaries and refuse to continue
    /// (`Err`) the moment true disorder is found, rather than only within
    /// one small Arrow batch (see the per-batch sort in `next` below for
    /// what that narrower, cheaper repair already covers on its own).
    last_yielded: Option<u64>,
}

impl Iterator for RangeIter {
    type Item = Result<(u64, u32)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((kmer, freq)) = self.buffer.pop_front() {
                if let Some(prev) = self.last_yielded {
                    if kmer < prev {
                        return Some(Err(load_reason(
                            &self.path,
                            format!(
                                "k-mer {kmer} was read after k-mer {prev}: this table's rows are not \
                                 actually sorted ascending by kmer_u64, even though its footer metadata \
                                 claims `fastdna.sorted_by=kmer_u64` -- refusing to read further rather \
                                 than silently returning results computed over an assumed order that \
                                 does not hold"
                            ),
                        )));
                    }
                }
                self.last_yielded = Some(kmer);
                return Some(Ok((kmer, freq)));
            }

            match self.reader.next() {
                Some(Ok(batch)) => {
                    let kmers = match batch.column(self.kmer_col).as_any().downcast_ref::<UInt64Array>() {
                        Some(a) => a,
                        None => {
                            return Some(Err(FastDnaError::Internal {
                                detail: "kmer_u64 column decoded as an unexpected Arrow array type".to_string(),
                            }))
                        }
                    };
                    let freqs = match batch.column(self.freq_col).as_any().downcast_ref::<UInt32Array>() {
                        Some(a) => a,
                        None => {
                            return Some(Err(FastDnaError::Internal {
                                detail: "frequency column decoded as an unexpected Arrow array type".to_string(),
                            }))
                        }
                    };
                    let mut pending: Vec<(u64, u32)> = Vec::with_capacity(batch.num_rows());
                    for i in 0..batch.num_rows() {
                        let kmer = kmers.value(i);
                        if kmer < self.low || kmer > self.high {
                            continue;
                        }
                        pending.push((kmer, freqs.value(i)));
                    }
                    // A Parquet reader hands back one Arrow batch at a time
                    // (far smaller than a whole row group), and every batch
                    // this crate's own writers (`export.rs`) ever produce is
                    // already ascending -- so for well-formed input, this
                    // sort is a single no-swap comparison pass, not real
                    // work. It only does anything against a batch built
                    // outside this crate whose actual row order does not
                    // match its declared order: repairing *within* one
                    // batch this cheaply is worth doing before falling back
                    // to the harder, cross-batch check above (which cannot
                    // repair anything -- rows from an earlier batch have
                    // already been handed to the caller).
                    pending.sort_unstable_by_key(|&(kmer, _)| kmer);
                    self.buffer.extend(pending);
                    // An empty (fully filtered-out) batch loops back around
                    // rather than returning `None` early -- more batches may
                    // still hold matching rows.
                }
                Some(Err(e)) => return Some(Err(load_err(&self.path, e))),
                None => return None,
            }
        }
    }
}

/// Parses a CLI/Python-supplied k-mer argument into this table's canonical,
/// `k`-bit-packed `u64` encoding: a plain non-negative decimal integer is
/// taken as that encoding directly (for programmatic callers passing a
/// value already read from a `kmer_u64` column); anything else is treated
/// as a DNA sequence and must be exactly `k` bases of unambiguous A/C/G/T/U
/// (case-insensitive), canonicalized the same way counting does
/// (`kmer::extract_canonical_kmers`).
///
/// A sequence of the wrong length or containing an ambiguous base (`N`) is
/// `FastDnaError::InvalidConfig`, not a silent `None` result indistinguishable
/// from "not in the table" -- those are different problems (a malformed
/// query vs. a genuine miss) and must not be reported the same way.
pub fn encode_query_kmer(input: &str, k: usize) -> Result<u64> {
    let trimmed = input.trim();
    if !trimmed.is_empty() && trimmed.bytes().all(|b| b.is_ascii_digit()) {
        return trimmed.parse::<u64>().map_err(|e| FastDnaError::InvalidConfig {
            parameter: "kmer",
            reason: format!("'{trimmed}' looks numeric but does not fit in a u64: {e}"),
        });
    }

    let bytes = trimmed.as_bytes();
    if bytes.len() != k {
        return Err(FastDnaError::InvalidConfig {
            parameter: "kmer",
            reason: format!("'{trimmed}' has length {}, but this table's k is {k}", bytes.len()),
        });
    }

    match kmer::extract_canonical_kmers(bytes, k).as_slice() {
        [only] => Ok(*only),
        _ => Err(FastDnaError::InvalidConfig {
            parameter: "kmer",
            reason: format!(
                "'{trimmed}' contains a character that is not an unambiguous nucleotide (A/C/G/T/U)"
            ),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_ktab_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a small table of k=4 canonical k-mers with distinct counts,
    /// via the real `export.rs` writer (not a hand-built Parquet file), so
    /// these tests exercise exactly the file `fastdna count` produces.
    fn write_test_table(path: &Path, k: usize, entries: &[u64]) -> KmerCounter {
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, path, k, 1, false).unwrap();
        counter
    }

    #[test]
    fn open_rejects_a_file_with_no_sorted_by_metadata() {
        // A plain Parquet file with the right columns, but not written by
        // this crate's exporter (no key-value metadata at all).
        use arrow::array::{UInt32Array as A32, UInt64Array as A64};
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let path = temp_path("no_metadata");
        let schema = Arc::new(Schema::new(vec![
            Field::new("kmer_u64", DataType::UInt64, false),
            Field::new("frequency", DataType::UInt32, false),
        ]));
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(A64::from(vec![1u64, 2, 3])), Arc::new(A32::from(vec![1u32, 2, 3]))],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        match KmerTable::open(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains(SORTED_BY_KEY), "reason must name the missing metadata: {reason}");
            }
            other => panic!("expected Load error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_a_file_with_the_wrong_columns() {
        use arrow::array::StringArray;
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let path = temp_path("wrong_columns");
        let schema = Arc::new(Schema::new(vec![Field::new("not_a_kmer_column", DataType::Utf8, false)]));
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["x"]))]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        match KmerTable::open(&path) {
            Err(FastDnaError::Load { .. }) => {}
            other => panic!("expected Load error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_of_a_missing_file_is_an_io_error() {
        let path = temp_path("does_not_exist");
        match KmerTable::open(&path) {
            Err(FastDnaError::Io { .. }) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[test]
    fn get_finds_a_present_kmer_and_reports_its_exact_count() {
        let path = temp_path("present");
        // k=4: "AAAA" = 0, "AACG" = 6 (canonical; see export.rs's own test
        // fixture for the same value). 6 appears twice, 0 once.
        write_test_table(&path, 4, &[6, 6, 0]);

        let table = KmerTable::open(&path).unwrap();
        assert_eq!(table.k(), 4);
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(6).unwrap(), Some(2));
        assert_eq!(table.get(0).unwrap(), Some(1));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn get_of_an_absent_kmer_is_none_not_an_error() {
        let path = temp_path("absent");
        write_test_table(&path, 4, &[6, 6, 0]);

        let table = KmerTable::open(&path).unwrap();
        // 3 lies strictly between the table's two distinct keys (0 and 6),
        // so it is inside at least one row group's range without actually
        // being a row -- the case that proves a range match still requires
        // an exact decoded hit, not just statistics overlap.
        assert_eq!(table.get(3).unwrap(), None);
        // Entirely outside every row group's range: must not even need to
        // decode to answer `None`.
        assert_eq!(table.get(u64::MAX).unwrap(), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn get_on_an_empty_table_is_none() {
        let path = temp_path("empty");
        write_test_table(&path, 4, &[]);

        let table = KmerTable::open(&path).unwrap();
        assert!(table.is_empty());
        assert_eq!(table.get(0).unwrap(), None);

        let _ = std::fs::remove_file(&path);
    }

    /// `export.rs`'s own writer never calls `writer.write()` for a
    /// zero-entry counter (see `write_test_table`'s doc comment), so it
    /// never actually produces a *row group* with zero rows -- but a
    /// cross-tool-written table can: `pyarrow.parquet.write_table` on a
    /// zero-row `Table` writes exactly one row group with no rows and no
    /// column statistics (nothing to compute them from). `open` must treat
    /// that row group as "nothing to prune", not reject the file for
    /// missing statistics -- this is exactly the gap the Python test suite
    /// caught (`python/tests/test_ktab.py`'s empty-table fixture, built
    /// with plain pyarrow) before this test existed to pin it on the Rust
    /// side too.
    #[test]
    fn open_and_get_tolerate_a_real_zero_row_row_group() {
        use arrow::array::{UInt32Array as A32, UInt64Array as A64};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::metadata::KeyValue;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let path = temp_path("zero_row_group");
        let schema = export::counts_schema(false);
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![
                KeyValue::new(SORTED_BY_KEY.to_string(), Some(SORTED_BY_VALUE.to_string())),
                KeyValue::new(K_KEY.to_string(), Some("4".to_string())),
            ]))
            .build();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        // An explicit zero-row batch, unlike `export_counts_parquet`'s
        // "skip the write entirely" path: this really does add one row
        // group with `num_rows() == 0` to the file's footer.
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(A64::from(Vec::<u64>::new())), Arc::new(A32::from(Vec::<u32>::new()))],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let table = KmerTable::open(&path).unwrap();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.get(0).unwrap(), None);
        assert_eq!(table.iter().unwrap().next().transpose().unwrap(), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn iter_yields_every_row_in_ascending_order_matching_get() {
        let path = temp_path("iter_order");
        let kmers: Vec<u64> = (0u64..2000).map(|i| i.wrapping_mul(0x9E37_79B9) & 0xFFFF_FFFF).collect();
        write_test_table(&path, 16, &kmers);

        let table = KmerTable::open(&path).unwrap();
        let collected: Vec<(u64, u32)> = table.iter().unwrap().collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(collected.len(), table.len() as usize);
        assert!(collected.windows(2).all(|w| w[0].0 < w[1].0), "iter() must yield strictly ascending k-mers");
        for &(kmer, count) in &collected {
            assert_eq!(table.get(kmer).unwrap(), Some(count), "iter()'s view must match get()'s for {kmer}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn iter_spans_multiple_row_groups_and_stays_sorted_and_complete() {
        let path = temp_path("multi_row_group");
        // export.rs chunks every 131_072 rows into its own row group; this
        // spans three, so pruning/streaming across row-group boundaries is
        // actually exercised, not just a single-group table.
        let rows = 131_072 * 2 + 500;
        let kmers: Vec<u64> = (0..rows as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 2).collect();
        write_test_table(&path, 31, &kmers);

        let table = KmerTable::open(&path).unwrap();
        assert_eq!(table.len(), rows as u64, "every distinct k-mer must be counted exactly once");

        let collected: Vec<(u64, u32)> = table.iter().unwrap().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(collected.len(), rows);
        assert!(collected.windows(2).all(|w| w[0].0 < w[1].0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn range_returns_only_kmers_within_bounds_inclusive() {
        let path = temp_path("range");
        write_test_table(&path, 4, &[1, 2, 3, 4, 5]);

        let table = KmerTable::open(&path).unwrap();
        let collected: Vec<u64> =
            table.range(2, 4).unwrap().collect::<Result<Vec<_>>>().unwrap().into_iter().map(|(k, _)| k).collect();
        assert_eq!(collected, vec![2, 3, 4], "bounds must be inclusive on both ends");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encode_query_kmer_accepts_a_numeric_literal_as_the_raw_encoding() {
        assert_eq!(encode_query_kmer("42", 4).unwrap(), 42);
    }

    #[test]
    fn encode_query_kmer_encodes_a_sequence_of_the_right_length_canonically() {
        // "ACGT" at k=4, same fixture kmer.rs's own tests use.
        let encoded = encode_query_kmer("ACGT", 4).unwrap();
        assert_eq!(encoded, kmer::extract_canonical_kmers(b"ACGT", 4)[0]);
    }

    #[test]
    fn encode_query_kmer_rejects_a_sequence_of_the_wrong_length() {
        match encode_query_kmer("ACG", 4) {
            Err(FastDnaError::InvalidConfig { .. }) => {}
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encode_query_kmer_rejects_an_ambiguous_base() {
        match encode_query_kmer("ACGN", 4) {
            Err(FastDnaError::InvalidConfig { .. }) => {}
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
}
