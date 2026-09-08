// src/wide_ktab.rs
//! `ktab.rs`'s reader for a **wide** k-mer table -- the sorted
//! `(kmer_bits, frequency)` Parquet that `export::export_wide_counts_parquet`
//! writes at `33 <= k <= 64`.
//!
//! # Why a second reader rather than a generic one
//!
//! The same reason `export_wide_counts_parquet` is not a generic writer,
//! and `wide_kmer.rs` is not a generic encoder: the `u64` path is the one
//! measured exactly equal to KMC3 on real reads, and every consumer of
//! `KmerTable` -- `setops`, `read_filter`, `similarity` -- is built on its
//! concrete `u64` key. Making that generic to add a second key type would
//! put those results back in question to save duplicating three hundred
//! lines. The wide reader sits beside it and the narrow one does not
//! change.
//!
//! The two are not as parallel as they look, either. Three things differ
//! in kind, not just in width:
//!
//! 1. **The key is a byte string, not an integer.** Arrow has no 128-bit
//!    integer, so the column is `FixedSizeBinary(16)` and Parquet's
//!    per-row-group statistics come back as `FixedLenByteArray`, not
//!    `Int64`. They are compared *bytewise*, which is exactly why
//!    `wide_kmer::to_key_bytes` writes big-endian: bytewise order over
//!    those 16 bytes is numeric order over the `u128` they encode, so
//!    Parquet's own min/max pruning is correct without decoding anything.
//! 2. **A row group's statistics can be the wrong length.** A `u64`
//!    statistic is a `u64`; a byte-array statistic is however many bytes
//!    the writer put there. A 15-byte or 17-byte minimum is a corrupt or
//!    foreign file, and is rejected rather than zero-padded into a
//!    plausible-looking key.
//! 3. **There is no `iter()` over "the full domain".** `KmerTable::iter`
//!    is `range(0, u64::MAX)`; the wide equivalent is `range(0,
//!    u128::MAX)`, which is fine, but the k-mers it yields are `u128` and
//!    no existing consumer can accept them. `iter`/`range` exist here
//!    because a point lookup is implemented as a degenerate range and
//!    because a future wide `union`/`intersect` needs them -- not because
//!    anything calls them today.
//!
//! # What is not here
//!
//! Set operations, read filtering and similarity. Those are `u64`-keyed
//! end to end (`setops::MultiTableMerge`'s heap, `read_filter::
//! ReferenceIndex`'s binary search over a `Vec<u64>`), and giving them a
//! wide form is a larger change than opening the file. `KmerTable::open`
//! rejects a wide table by name and says so; this module is what makes
//! `fastdna query` the one operation that does not have to.

use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, FixedSizeBinaryArray, UInt32Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::file::statistics::Statistics;

use crate::error::{FastDnaError, Result};
use crate::ktab::{K_KEY, SORTED_BY_KEY, SORTED_BY_WIDE_VALUE};
use crate::wide_kmer;

/// The width of a `kmer_bits` value, in bytes. A `u128` is 16 bytes and
/// the column is declared `FixedSizeBinary(16)`; this names the number so
/// the schema check, the statistics length check and the decode all cite
/// the same one.
const KEY_BYTES: usize = 16;

/// One row group's `kmer_bits` range, decoded from the footer at `open`
/// time. `u128` rather than the raw bytes so pruning is an integer
/// comparison; the conversion is exact because the bytes are big-endian
/// and exactly `KEY_BYTES` long (both checked at `open`).
#[derive(Debug, Clone, Copy)]
struct RowGroupRange {
    index: usize,
    min: u128,
    max: u128,
}

/// A handle to a sorted `(kmer_bits, frequency)` Parquet table, opened
/// without decoding a row -- the wide counterpart of [`crate::ktab::KmerTable`].
#[derive(Debug, Clone)]
pub struct WideKmerTable {
    path: PathBuf,
    k: usize,
    /// From the table's own `fastdna.canonical` footer key; absent means
    /// `true`. See `ktab::CANONICAL_KEY`.
    canonical: bool,
    total_rows: u64,
    kmer_col: usize,
    freq_col: usize,
    /// Ascending and non-overlapping at their boundaries, checked at
    /// `open` time exactly as the narrow reader checks its own.
    row_groups: Vec<RowGroupRange>,
}

fn load_err<E>(path: &Path, err: E) -> FastDnaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    FastDnaError::Load { path: path.to_path_buf(), reason: err.to_string(), source: Some(Box::new(err)) }
}

fn load_reason(path: &Path, reason: String) -> FastDnaError {
    FastDnaError::Load { path: path.to_path_buf(), reason, source: None }
}

/// A `kmer_bits` value as the `u128` it encodes.
///
/// Returns `None` for anything that is not exactly `KEY_BYTES` long: the
/// alternative -- padding a short value or truncating a long one -- would
/// turn a corrupt file into a table that opens and answers queries with
/// keys nobody wrote.
fn key_from_bytes(bytes: &[u8]) -> Option<u128> {
    let fixed: [u8; KEY_BYTES] = bytes.try_into().ok()?;
    Some(wide_kmer::from_key_bytes(fixed))
}

impl WideKmerTable {
    /// Opens `path` and validates it as a queryable **wide** k-mer table.
    /// Footer-only, like the narrow `open`: schema, key-value metadata and
    /// one `Statistics` per row group, with no row decoded.
    ///
    /// Rejected as `FastDnaError::Load`, with an actionable reason:
    /// - a schema without a non-nullable `kmer_bits: FixedSizeBinary(16)`
    ///   column and a non-nullable `frequency: UInt32` one. Non-nullable is
    ///   required for the same reason the narrow reader requires it: Arrow's
    ///   `Array::value(i)` returns an unspecified payload for a null slot
    ///   rather than erroring, so a nullable column would let a table
    ///   answer a lookup with another row's count;
    /// - `fastdna.sorted_by` that is missing, or that says `kmer_u64` --
    ///   i.e. a *narrow* table, which `ktab::KmerTable` reads and this does
    ///   not;
    /// - a missing or unparsable `fastdna.k`;
    /// - a row group with no statistics on the key column, statistics of
    ///   the wrong Parquet type, or a min/max that is not exactly 16 bytes;
    /// - row groups whose ascending order is violated at their boundaries.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| FastDnaError::Io { path: path.clone(), source: e })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| load_err(&path, e))?;

        let schema = builder.schema();
        let kmer_col = schema
            .index_of("kmer_bits")
            .ok()
            .filter(|&i| {
                schema.field(i).data_type() == &DataType::FixedSizeBinary(KEY_BYTES as i32)
                    && !schema.field(i).is_nullable()
            })
            .ok_or_else(|| {
                load_reason(
                    &path,
                    format!(
                        "missing a non-nullable kmer_bits: fixed_size_binary({KEY_BYTES}) column. A \
                         table written at k<=32 keys on a kmer_u64 column instead and is read by \
                         KmerTable, not this reader"
                    ),
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
        let kv = metadata.file_metadata().key_value_metadata();

        let sorted_by_ok = kv.is_some_and(|pairs| {
            pairs
                .iter()
                .any(|p| p.key == SORTED_BY_KEY && p.value.as_deref() == Some(SORTED_BY_WIDE_VALUE))
        });
        if !sorted_by_ok {
            return Err(load_reason(
                &path,
                format!(
                    "missing or unexpected '{SORTED_BY_KEY}' Parquet metadata -- expected \
                     '{SORTED_BY_WIDE_VALUE}', which `fastdna count` writes for k>32"
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
        let mut prev_max: Option<u128> = None;

        for (index, rg) in metadata.row_groups().iter().enumerate() {
            // Same as the narrow reader: an empty row group has no
            // meaningful min/max and nothing for a query to find, so it is
            // skipped rather than rejected.
            if rg.num_rows() == 0 {
                continue;
            }
            let stats = rg.column(kmer_col).statistics().ok_or_else(|| {
                load_reason(&path, format!("row group {index} has no statistics on kmer_bits -- cannot query it"))
            })?;
            let (min, max) = match stats {
                Statistics::FixedLenByteArray(v) => {
                    let min = v
                        .min_opt()
                        .and_then(|b| key_from_bytes(b.data()))
                        .ok_or_else(|| {
                            load_reason(
                                &path,
                                format!(
                                    "row group {index}'s kmer_bits statistics have no minimum, or one \
                                     that is not {KEY_BYTES} bytes"
                                ),
                            )
                        })?;
                    let max = v
                        .max_opt()
                        .and_then(|b| key_from_bytes(b.data()))
                        .ok_or_else(|| {
                            load_reason(
                                &path,
                                format!(
                                    "row group {index}'s kmer_bits statistics have no maximum, or one \
                                     that is not {KEY_BYTES} bytes"
                                ),
                            )
                        })?;
                    (min, max)
                }
                other => {
                    return Err(load_reason(
                        &path,
                        format!("row group {index}'s kmer_bits statistics are an unexpected type: {other:?}"),
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
                             maximum {prev} -- the file is not sorted ascending by kmer_bits and cannot \
                             be queried as a FastDNA k-mer table"
                        ),
                    ));
                }
            }
            prev_max = Some(max);
            total_rows += rg.num_rows() as u64;
            row_groups.push(RowGroupRange { index, min, max });
        }

        let canonical = match kv
            .and_then(|pairs| pairs.iter().find(|p| p.key == crate::ktab::CANONICAL_KEY))
            .and_then(|p| p.value.as_deref())
        {
            None | Some("true") => true,
            Some("false") => false,
            Some(other) => {
                return Err(load_reason(
                    &path,
                    format!("'fastdna.canonical' is '{other}', which is neither 'true' nor 'false'"),
                ))
            }
        };

        Ok(Self { path, k, canonical, total_rows, kmer_col, freq_col, row_groups })
    }

    /// The `k` every row's `kmer_bits` was packed with, from the table's
    /// own `fastdna.k` metadata.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether this table's k-mers were canonicalised -- see
    /// [`crate::ktab::KmerTable::canonical`].
    pub fn canonical(&self) -> bool {
        self.canonical
    }

    /// The file this table was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total rows across every non-empty row group.
    pub fn len(&self) -> u64 {
        self.total_rows
    }

    pub fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// Point lookup: the frequency recorded for `kmer`, or `None` if it is
    /// absent. `kmer` must already be the canonical packed encoding this
    /// table's rows use -- `encode_query_wide_kmer` turns a DNA sequence or
    /// a decimal string into it.
    pub fn get(&self, kmer: u128) -> Result<Option<u32>> {
        match self.range(kmer, kmer)?.next() {
            Some(Ok((_, count))) => Ok(Some(count)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Every `(kmer, frequency)` pair with `low <= kmer <= high`,
    /// ascending, streamed row group by row group -- only row groups whose
    /// statistics overlap `[low, high]` are opened.
    pub fn range(&self, low: u128, high: u128) -> Result<WideRangeIter> {
        let pruned: Vec<usize> =
            self.row_groups.iter().filter(|rg| rg.max >= low && rg.min <= high).map(|rg| rg.index).collect();

        let file = File::open(&self.path).map_err(|e| FastDnaError::Io { path: self.path.clone(), source: e })?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| load_err(&self.path, e))?
            .with_row_groups(pruned)
            .build()
            .map_err(|e| load_err(&self.path, e))?;

        Ok(WideRangeIter {
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

    /// Every `(kmer, frequency)` pair in the table, ascending.
    pub fn iter(&self) -> Result<WideRangeIter> {
        self.range(0, u128::MAX)
    }
}

/// Streaming iterator over a row-group-pruned range of a
/// [`WideKmerTable`]. Mirrors `ktab::RangeIter`, including its
/// cross-batch order check -- see that type's `last_yielded` field for why
/// a reader that assumes sortedness must verify it as it goes rather than
/// trusting footer metadata alone.
pub struct WideRangeIter {
    path: PathBuf,
    reader: ParquetRecordBatchReader,
    low: u128,
    high: u128,
    kmer_col: usize,
    freq_col: usize,
    buffer: VecDeque<(u128, u32)>,
    last_yielded: Option<u128>,
}

impl Iterator for WideRangeIter {
    type Item = Result<(u128, u32)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((kmer, freq)) = self.buffer.pop_front() {
                if let Some(prev) = self.last_yielded {
                    if kmer < prev {
                        return Some(Err(load_reason(
                            &self.path,
                            format!(
                                "k-mer {kmer} was read after k-mer {prev}: this table's rows are not \
                                 actually sorted ascending by kmer_bits, even though its footer metadata \
                                 claims `fastdna.sorted_by=kmer_bits` -- refusing to read further rather \
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
                    let kmers = match batch.column(self.kmer_col).as_any().downcast_ref::<FixedSizeBinaryArray>() {
                        Some(a) => a,
                        None => {
                            return Some(Err(FastDnaError::Internal {
                                detail: "kmer_bits column decoded as an unexpected Arrow array type".to_string(),
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
                    let mut pending: Vec<(u128, u32)> = Vec::with_capacity(batch.num_rows());
                    for i in 0..batch.num_rows() {
                        // `open` pinned the column type at
                        // `FixedSizeBinary(16)`, so every value is exactly
                        // 16 bytes; a `None` here would mean the decoded
                        // batch disagrees with the schema it was read
                        // under, which is a corrupt file rather than
                        // something to skip silently.
                        let Some(kmer) = key_from_bytes(kmers.value(i)) else {
                            return Some(Err(load_reason(
                                &self.path,
                                format!(
                                    "row {i} of a decoded batch has a kmer_bits value that is not \
                                     {KEY_BYTES} bytes, contradicting the column's own schema"
                                ),
                            )));
                        };
                        if kmer < self.low || kmer > self.high {
                            continue;
                        }
                        pending.push((kmer, freqs.value(i)));
                    }
                    // As in `ktab::RangeIter`: a no-swap pass for every
                    // batch this crate writes, and a cheap within-batch
                    // repair for one built elsewhere.
                    pending.sort_unstable_by_key(|&(kmer, _)| kmer);
                    self.buffer.extend(pending);
                }
                Some(Err(e)) => return Some(Err(load_err(&self.path, e))),
                None => return None,
            }
        }
    }
}

/// Parses a user-supplied k-mer for a wide table into its packed `u128`.
///
/// Mirrors `ktab::encode_query_kmer`: an all-digit input is taken as the
/// packed encoding directly (for a caller passing a value already read out
/// of a `kmer_bits` column), and anything else is a DNA sequence of
/// exactly `k` unambiguous bases, canonicalized the way counting does.
///
/// A malformed query is `FastDnaError::InvalidConfig`, never a silent
/// `None` -- "you typed it wrong" and "it is not in the table" are
/// different answers and must not look the same.
/// `canonical` must match the table being queried -- see
/// [`crate::ktab::encode_query_kmer`] for why a mismatch is a silent wrong
/// answer rather than an error.
pub fn encode_query_wide_kmer(input: &str, k: usize, canonical: bool) -> Result<u128> {
    let trimmed = input.trim();
    if !trimmed.is_empty() && trimmed.bytes().all(|b| b.is_ascii_digit()) {
        return trimmed.parse::<u128>().map_err(|e| FastDnaError::InvalidConfig {
            parameter: "kmer",
            reason: format!("'{trimmed}' looks numeric but does not fit in a u128: {e}"),
        });
    }

    let bytes = trimmed.as_bytes();
    if bytes.len() != k {
        return Err(FastDnaError::InvalidConfig {
            parameter: "kmer",
            reason: format!("'{trimmed}' has length {}, but this table's k is {k}", bytes.len()),
        });
    }

    let mut extracted = Vec::new();
    if canonical {
        wide_kmer::extract_canonical_kmers_into(bytes, k, &mut extracted);
    } else {
        wide_kmer::extract_forward_kmers_into(bytes, k, &mut extracted);
    }
    match extracted.as_slice() {
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
    use crate::export;
    use crate::wide_counter::WideKmerCounter;

    fn temp_path(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        std::env::temp_dir().join(format!("fastdna_wide_ktab_{name}_{unique}.parquet"))
    }

    /// Writes a wide table holding exactly `sequences`' k-mers, and hands
    /// back the path plus the `(kmer, count)` pairs it should contain --
    /// computed here from the counter, so a test asserts against the
    /// counted truth rather than against a hardcoded number that would
    /// silently stop meaning anything if the fixture changed.
    fn write_table(name: &str, k: usize, sequences: &[&str]) -> (PathBuf, Vec<(u128, u32)>) {
        let mut counter = WideKmerCounter::new();
        let mut kmers = Vec::new();
        for seq in sequences {
            kmers.clear();
            wide_kmer::extract_canonical_kmers_into(seq.as_bytes(), k, &mut kmers);
            counter.insert_batch(&kmers);
        }
        let counts = counter.finish();
        let expected: Vec<(u128, u32)> = counts.iter().collect();

        let path = temp_path(name);
        export::export_wide_counts_parquet(&counts, &path, k, false, true).expect("write wide table");
        (path, expected)
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn iter_yields_every_row_in_ascending_order() {
        let (path, expected) = write_table("iter", 35, &["ACGTTGCAAGGCTTACCGATCGATTACAGCATCGGATCCAT"]);
        let _cleanup = Cleanup(path.clone());

        let table = WideKmerTable::open(&path).unwrap();
        assert_eq!(table.k(), 35);
        assert_eq!(table.len(), expected.len() as u64);
        assert!(!table.is_empty());

        let got: Vec<(u128, u32)> = table.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(got, expected);
        assert!(got.windows(2).all(|w| w[0].0 < w[1].0), "iter must be ascending and deduplicated");
    }

    #[test]
    fn range_is_inclusive_at_both_ends() {
        let (path, expected) = write_table("range", 33, &["ACGTTGCAAGGCTTACCGATCGATTACAGCATCGGATCCAT"]);
        let _cleanup = Cleanup(path.clone());
        assert!(expected.len() >= 3, "the fixture should yield at least three distinct k-mers");

        let table = WideKmerTable::open(&path).unwrap();
        let low = expected[1].0;
        let high = expected[expected.len() - 2].0;

        let got: Vec<(u128, u32)> = table.range(low, high).unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(got, expected[1..expected.len() - 1].to_vec());
    }

    #[test]
    fn get_finds_every_kmer_the_table_holds_and_no_others() {
        let (path, expected) = write_table("get", 40, &["ACGTTGCAAGGCTTACCGATCGATTACAGCATCGGATCCATTGCA"]);
        let _cleanup = Cleanup(path.clone());

        let table = WideKmerTable::open(&path).unwrap();
        for &(kmer, count) in &expected {
            assert_eq!(table.get(kmer).unwrap(), Some(count), "k-mer {kmer} should be present");
        }

        // A value between two real keys, so it exercises the "inside a row
        // group's range but not in it" path rather than being pruned away
        // by statistics before a row is decoded.
        let absent = expected[0].0 + 1;
        if !expected.iter().any(|&(k, _)| k == absent) {
            assert_eq!(table.get(absent).unwrap(), None);
        }
    }

    #[test]
    fn a_narrow_table_is_refused_by_name() {
        // The mirror of `KmerTable::open`'s refusal of a wide table: each
        // reader must say which one the file needs, not fail obscurely.
        use crate::counter::KmerCounter;

        let mut counter = KmerCounter::new();
        counter.insert_batch(&crate::kmer::extract_canonical_kmers(b"ACGTACGTACGTACGTACGTACGT", 21));
        let path = temp_path("narrow_input");
        let _cleanup = Cleanup(path.clone());
        export::export_counts_parquet(&counter, &path, 21, 1, false, true).expect("write narrow table");

        let err = WideKmerTable::open(&path).unwrap_err().to_string();
        assert!(err.contains("kmer_bits"), "the error should name the column it wanted: {err}");
        assert!(err.contains("kmer_u64"), "and the one it found: {err}");

        // And the router sends it to the right reader in the first place.
        assert_eq!(crate::ktab::table_key(&path).unwrap(), crate::ktab::TableKey::Narrow);
    }

    #[test]
    fn the_router_recognises_a_wide_table() {
        let (path, _) = write_table("router", 33, &["ACGTTGCAAGGCTTACCGATCGATTACAGCATCGGATCCAT"]);
        let _cleanup = Cleanup(path.clone());
        assert_eq!(crate::ktab::table_key(&path).unwrap(), crate::ktab::TableKey::Wide);
    }

    #[test]
    fn a_numeric_query_is_taken_as_the_packed_encoding() {
        // For a caller passing back a value read out of a `kmer_bits`
        // column rather than a sequence -- the same affordance
        // `ktab::encode_query_kmer` has.
        let k = 33;
        let seq = "ACGTTGCAAGGCTTACCGATCGATTACAGCATC";
        assert_eq!(seq.len(), k);
        let packed = wide_kmer::extract_canonical_kmers(seq.as_bytes(), k)[0];

        assert_eq!(encode_query_wide_kmer(seq, k, true).unwrap(), packed);
        assert_eq!(encode_query_wide_kmer(&packed.to_string(), k, true).unwrap(), packed);
    }

    #[test]
    fn a_malformed_query_is_an_error_not_a_miss() {
        let k = 33;
        assert!(encode_query_wide_kmer("ACGT", k, true).is_err(), "wrong length");
        let with_n = format!("N{}", "ACGTTGCAAGGCTTACCGATCGATTACAGCAT");
        assert_eq!(with_n.len(), k);
        assert!(encode_query_wide_kmer(&with_n, k, true).is_err(), "ambiguous base");
    }
}
