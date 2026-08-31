// src/export.rs

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use arrow::array::{ArrayRef, StringArray, UInt32Array, UInt64Array};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use rustc_hash::FxHashMap;

use crate::atomic::AtomicFile;
use crate::cohort::matrix::CohortMatrix;
use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::kmer;
use crate::ktab;

/// Wraps an Arrow/Parquet serialization or writer failure. These are not I/O
/// errors: the bytes may never touch a disk (e.g. a schema mismatch building
/// a `RecordBatch` in memory), so laundering them through `FastDnaError::Io`
/// with a synthetic `std::io::Error` erases their real type and misleads
/// callers -- including a future Python binding, where this must surface as
/// a distinct exception class rather than `OSError`.
fn export_err<E>(path: &Path, err: E) -> FastDnaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    FastDnaError::Export {
        path: path.to_path_buf(),
        reason: err.to_string(),
        source: Some(Box::new(err)),
    }
}

/// Wraps a genuine `std::io::Error` from a writer call (`writeln!`, `flush`)
/// as `FastDnaError::Io`, preserving the real source error.
fn io_err(path: &Path, err: std::io::Error) -> FastDnaError {
    FastDnaError::Io { path: path.to_path_buf(), source: err }
}

/// The schema shared by every k-mer count table FastDNA produces -- the
/// Parquet files this module writes and the in-memory Arrow table the
/// Python binding hands back (`ffi.rs`). Defined once here so the two
/// never drift apart: a user who writes one and reads the other must see
/// identical columns.
///
/// `with_sequence` decides whether `kmer_sequence` is included.
///
/// That column is entirely derivable from `kmer_u64` plus `k` -- it is
/// exactly what `kmer::decode_kmer_into` computes -- and on the benchmark
/// file (53,776,394 distinct k-mers at k=31) it is 1,667 MB against 430 MB
/// for `kmer_u64` and 215 MB for `frequency`: 2.6x larger than the other
/// two columns combined. Writing it is the majority of the 23.0 s the
/// export step takes out of a 133.1 s end-to-end run.
///
/// It defaults to off (see `count()` in `ffi.rs` and `--with-sequence` in
/// `cli.rs`) because the majority consumer -- every ML path this crate
/// ships, `sklearn`, `cv`, `gwas` -- operates in `u64` space and never
/// reads it. Whoever needs it asks explicitly, or reconstructs it from
/// `kmer_u64` without rereading the FASTQ (`KmerCounts.with_sequence()` in
/// `python/fastdna/__init__.py`).
pub fn counts_schema(with_sequence: bool) -> Arc<Schema> {
    let mut fields = vec![Field::new("kmer_u64", DataType::UInt64, false)];
    if with_sequence {
        fields.push(Field::new("kmer_sequence", DataType::Utf8, false));
    }
    fields.push(Field::new("frequency", DataType::UInt32, false));
    Arc::new(Schema::new(fields))
}

/// The column buffers one Parquet chunk is assembled in, held in exactly the
/// layout Arrow stores those columns as: a `u64` values buffer, a `u32`
/// values buffer, and -- for the k-mer sequence column, when included --
/// one contiguous value buffer plus an `i32` offsets buffer.
///
/// This exists to keep the decoded bases from being copied twice. The
/// straightforward `Vec<String>` version costs, *per exported k-mer*
/// (53.8 million of them on the benchmark file):
///
/// 1. one heap allocation for the `String`,
/// 2. one `String::from_utf8` scan of `k` bytes that cannot fail, because
///    every byte came from the `b"ACGT"` literal,
/// 3. one `memcpy` of those `k` bytes into Arrow's own value buffer inside
///    `StringArray::from_iter_values`,
/// 4. one `free` when the chunk is cleared.
///
/// Writing the bases straight into `seq_bytes` removes all four: the buffer
/// *is* Arrow's value buffer (`Buffer::from_vec` takes ownership of the
/// allocation rather than copying it). Pre-sizing it also removes the
/// repeated doubling `from_iter_values` does -- it starts its value buffer
/// at capacity 0 and regrows it to ~4 MB per chunk, copying everything
/// written so far each time, roughly one extra full-buffer copy in total.
///
/// What remains is a single `std::str::from_utf8` over the whole chunk
/// buffer inside `StringArray::try_new` (the same bytes as before, scanned
/// contiguously instead of in 53.8 million separate calls) plus one
/// `is_char_boundary` per row -- a load and a compare, against the
/// allocate/free pair it replaces.
///
/// When `with_sequence` is `false`, `seq_bytes`/`seq_offsets` stay empty
/// and `push` skips `decode_kmer_into` entirely -- the point of turning the
/// column off, not merely of writing it more cheaply.
struct ChunkBuffers {
    with_sequence: bool,
    kmers: Vec<u64>,
    seq_bytes: Vec<u8>,
    seq_offsets: Vec<i32>,
    freqs: Vec<u32>,
}

impl ChunkBuffers {
    /// `rows` rows of `k`-base sequences, sized up front so nothing regrows.
    fn with_capacity(rows: usize, k: usize, with_sequence: bool) -> Self {
        let (seq_bytes_cap, seq_offsets) = if with_sequence {
            // An offsets buffer has one more entry than it has values: the
            // leading 0 that opens the first string.
            let mut offsets = Vec::with_capacity(rows + 1);
            offsets.push(0);
            (rows.saturating_mul(k), offsets)
        } else {
            (0, Vec::new())
        };
        Self {
            with_sequence,
            kmers: Vec::with_capacity(rows),
            seq_bytes: Vec::with_capacity(seq_bytes_cap),
            seq_offsets,
            freqs: Vec::with_capacity(rows),
        }
    }

    fn push(&mut self, kmer_bits: u64, k: usize, count: u32) {
        self.kmers.push(kmer_bits);
        if self.with_sequence {
            kmer::decode_kmer_into(kmer_bits, k, &mut self.seq_bytes);
            // Cast is safe for any chunk Arrow can hold in an i32-offset
            // column; `export_counts_parquet` caps a chunk at 131 072 rows
            // of at most 32 bases, i.e. 4 MiB, far below `i32::MAX`.
            self.seq_offsets.push(self.seq_bytes.len() as i32);
        }
        self.freqs.push(count);
    }

    fn len(&self) -> usize {
        self.kmers.len()
    }

    fn is_empty(&self) -> bool {
        self.kmers.is_empty()
    }
}

/// Writes `counter`'s finalized table as Parquet, plus two footer
/// key-value metadata entries -- `fastdna.sorted_by=kmer_u64` and
/// `fastdna.k=<k>` (`ktab::SORTED_BY_KEY`/`ktab::K_KEY`) -- recording that
/// this file's rows are what `ktab::KmerTable::open` requires before it
/// will treat a `.parquet` file as a queryable k-mer table.
///
/// This is not conditional on any flag: `counter.iter()` is *always*
/// globally sorted ascending by `kmer_u64` (`counter.rs`'s `CountTable`
/// invariant, upheld by every merge path that can produce a `KmerCounter`,
/// in-memory or disk-spilled), so every file this function writes already
/// satisfies the one property `ktab.rs` needs -- the metadata simply
/// records a fact that was already true, rather than changing what gets
/// written. Concretely: `fastdna count -o counts.parquet` needs no extra
/// step or flag to become queryable with `fastdna query --table
/// counts.parquet`; see `ktab.rs`'s module doc comment for the full
/// "no new file format" design this follows. Adding footer metadata is not
/// a schema change (`CHANGELOG.md`'s compatibility contract governs
/// columns, not footer key-value pairs), so this is safe for every
/// existing caller, including `--with-sequence` output and every per-sample
/// file `cohort::batch` writes.
pub fn export_counts_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
    with_sequence: bool,
) -> Result<usize> {
    let path = output_path.as_ref();
    let (file, pending) = AtomicFile::create(path)?;
    let schema = counts_schema(with_sequence);

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![
            KeyValue::new(ktab::SORTED_BY_KEY.to_string(), Some(ktab::SORTED_BY_VALUE.to_string())),
            KeyValue::new(ktab::K_KEY.to_string(), Some(k.to_string())),
        ]))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| export_err(path, e))?;
    let chunk_size = 131_072;
    let mut total_written = 0;

    let mut chunk = ChunkBuffers::with_capacity(chunk_size, k, with_sequence);

    for (kmer_bits, count) in counter.iter() {
        if count >= min_count {
            chunk.push(kmer_bits, k, count);

            if chunk.len() >= chunk_size {
                total_written += chunk.len();
                // The buffers are handed to Arrow by move, so a fresh set is
                // started here: ~410 allocations across the whole benchmark
                // export, against the 53.8 million this replaces.
                let full = std::mem::replace(
                    &mut chunk,
                    ChunkBuffers::with_capacity(chunk_size, k, with_sequence),
                );
                write_chunk(&mut writer, &schema, full, path)?;
            }
        }
    }

    if !chunk.is_empty() {
        total_written += chunk.len();
        write_chunk(&mut writer, &schema, chunk, path)?;
    }

    // `close` consumes the writer and with it the last handle on the temp
    // file; only then can the rename-over-destination succeed on Windows.
    writer.close().map_err(|e| export_err(path, e))?;
    pending.commit()?;
    Ok(total_written)
}

/// Writes an already ascending, already-deduplicated `(kmer_u64,
/// frequency)` stream to `output_path` in exactly the shape and footer
/// metadata `export_counts_parquet` writes -- so a set operation's result
/// (`setops.rs`'s `union`/`intersect`/`diff`) is immediately reopenable
/// with `ktab::KmerTable::open`, no conversion step, the same "declare it,
/// don't reinvent it" move `ktab.rs`'s own module doc comment makes for
/// `count`'s output.
///
/// Unlike `export_counts_parquet`, there is no `KmerCounter` to iterate --
/// the caller has already merged several tables into one sorted stream,
/// possibly failing partway through (a Parquet read error surfacing from
/// one of the *input* tables), so `pairs` yields `Result` rather than a bare
/// tuple and this function propagates that error exactly like one of its
/// own I/O failures.
///
/// No `kmer_sequence` column and no `min_count` filtering: both are
/// `count`-specific knobs on a raw counting run, and a set operation's
/// output is already exactly the rows its caller decided to keep -- there
/// is nothing left here to filter, and the sequence is reconstructible from
/// `kmer_u64` by any caller who wants it (`kmer::decode_kmer`), the same as
/// every other lean-by-default table this crate writes.
///
/// Sharing the chunking/writer machinery (`ChunkBuffers`, `write_chunk`)
/// with `export_counts_parquet` rather than duplicating it means the two
/// can never drift on row-group size, compression, or footer metadata.
pub fn export_pairs_parquet<P, I>(pairs: I, output_path: P, k: usize) -> Result<usize>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = Result<(u64, u32)>>,
{
    let path = output_path.as_ref();
    let (file, pending) = AtomicFile::create(path)?;
    let schema = counts_schema(false);

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![
            KeyValue::new(ktab::SORTED_BY_KEY.to_string(), Some(ktab::SORTED_BY_VALUE.to_string())),
            KeyValue::new(ktab::K_KEY.to_string(), Some(k.to_string())),
        ]))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| export_err(path, e))?;
    let chunk_size = 131_072;
    let mut total_written = 0;
    let mut chunk = ChunkBuffers::with_capacity(chunk_size, k, false);
    // The footer metadata written above claims `fastdna.sorted_by=kmer_u64`
    // unconditionally -- true for every in-crate caller (`setops::union`/
    // `intersect`/`diff`'s own merge is provably ascending), but `pairs` is
    // a bare `IntoIterator`, reachable from outside the crate with no such
    // guarantee. Writing the claim onto a stream that does not hold it would
    // make the resulting file `KmerTable::open`-valid while silently lying
    // about its own order -- exactly the failure `ktab::RangeIter`'s own
    // cross-batch order check exists to catch on read, so it is caught here
    // on write instead, before a single row leaves this function. Returning
    // early (without calling `pending.commit()`) leaves the destination
    // untouched and the temp file cleaned up by `AtomicFile`'s `Drop`, the
    // same abandoned-write path every other error in this loop already
    // takes via `?`.
    let mut last_kmer: Option<u64> = None;

    for pair in pairs {
        let (kmer_bits, count) = pair?;
        if let Some(prev) = last_kmer {
            if kmer_bits < prev {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "pairs",
                    reason: format!(
                        "k-mer {kmer_bits} was written after k-mer {prev}: pairs must be ascending \
                         by kmer_u64, since this function always records fastdna.sorted_by=kmer_u64"
                    ),
                });
            }
        }
        last_kmer = Some(kmer_bits);
        chunk.push(kmer_bits, k, count);

        if chunk.len() >= chunk_size {
            total_written += chunk.len();
            let full = std::mem::replace(&mut chunk, ChunkBuffers::with_capacity(chunk_size, k, false));
            write_chunk(&mut writer, &schema, full, path)?;
        }
    }

    if !chunk.is_empty() {
        total_written += chunk.len();
        write_chunk(&mut writer, &schema, chunk, path)?;
    }

    writer.close().map_err(|e| export_err(path, e))?;
    pending.commit()?;
    Ok(total_written)
}

/// The schema `export_cohort_matrix_parquet` writes: one row per nonzero
/// `(sample, k-mer)` entry of a `CohortMatrix`, in long/"tidy"/COO form --
/// `sample_id`, `kmer_u64`, optionally `kmer_sequence`, then `count`.
///
/// A long table, not a dense `samples x kmers` grid, because that is
/// exactly the shape `CohortMatrix` already holds in memory (`cohort::
/// matrix`'s module doc comment: it is built as `(row, col, value)` COO
/// triples specifically to avoid ever materializing the dense form). A
/// real cohort's matrix is overwhelmingly zero -- most k-mers are private
/// to a handful of samples -- so writing the dense grid would inflate a
/// file that is mostly zero bytes by orders of magnitude for no benefit;
/// this format writes exactly the `nnz` nonzero entries `CohortMatrix`
/// already counted, no more.
///
/// `sample_id` is a plain string column rather than a row index: a caller
/// opening this file in DuckDB, pandas or polars should not have to also
/// carry around a separate `row -> sample_id` lookup table just to know
/// whose k-mer a row belongs to -- the whole point of writing a generic
/// Parquet file (`docs/feature-gap-analysis.md`'s S6) is that it is usable
/// with no FastDNA-specific tooling at all. `kmer_u64` (not the decoded
/// sequence) is the default k-mer identifier for the same reason
/// `counts_schema`'s own doc comment gives for `count`'s output: it is an
/// 8-byte fixed-width column, against a variable-length ASCII string that,
/// for realistic `k`, is several times larger; it reconstructs to the
/// exact same canonical sequence via `kmer::decode_kmer` given `k` (recorded
/// in this file's own footer metadata, see below) -- so nothing is lost by
/// leaving `kmer_sequence` opt-in.
pub fn cohort_matrix_schema(with_sequence: bool) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("kmer_u64", DataType::UInt64, false),
    ];
    if with_sequence {
        fields.push(Field::new("kmer_sequence", DataType::Utf8, false));
    }
    fields.push(Field::new("count", DataType::UInt32, false));
    Arc::new(Schema::new(fields))
}

/// Writes a `CohortMatrix` (`cohort::matrix::build_cohort_matrix`, or
/// `cohort::matrix::build_cohort_matrix_from_directory`/`_from_files`) as a
/// Parquet file under `cohort_matrix_schema` -- the generic, FastDNA-tool-
/// independent file artifact `docs/feature-gap-analysis.md`'s S6 named as
/// the one piece still missing once the matrix-building engine and its
/// Python wiring (`gwas.py::cohort_presence_matrix`) had already landed.
///
/// `sample_ids[i]` names row `i` of `matrix` (i.e. every entry in
/// `matrix.row` equal to `i`); its length must equal `matrix.n_samples`.
/// This is a distinct argument, not a field of `CohortMatrix` itself,
/// because building the matrix (`build_cohort_matrix`) never needs a
/// sample's *name* -- only `count_samples`'s caller (the CLI, or a future
/// Python entry point) does, once it is time to write a human-readable
/// file.
///
/// Footer key-value metadata records `fastdna.k`, `fastdna.n_samples`,
/// `fastdna.n_kmers`, `fastdna.n_candidates` and (when truncation happened)
/// `fastdna.truncation_cutoff` -- the same numbers `gwas.py`'s truncation
/// warning is built from, so a caller reading this file back (with no
/// FastDNA library at all, just any Parquet reader that exposes footer
/// metadata) can recover whether `--max-kmers` truncated anything and by
/// how much, without recomputing it. Unlike `export_counts_parquet`'s
/// `fastdna.sorted_by=kmer_u64` metadata, this file makes no such claim:
/// rows are grouped by sample first (see `build_cohort_matrix`'s own pass
/// 3, which is where `matrix.row`/`matrix.col`/`matrix.value` come from),
/// not globally sorted by `kmer_u64`, so this file is not a valid
/// `ktab::KmerTable` and does not claim to be one.
///
/// Returns the number of rows written (`matrix.row.len()`, i.e. the
/// matrix's `nnz`). Goes through `AtomicFile` like every other exporter in
/// this module: a disk-full error or an interrupted run must leave the
/// previous good file in place rather than a truncated one a downstream
/// reader would happily treat as complete.
pub fn export_cohort_matrix_parquet<P: AsRef<Path>>(
    matrix: &CohortMatrix,
    sample_ids: &[String],
    output_path: P,
    k: usize,
    with_sequence: bool,
) -> Result<usize> {
    let path = output_path.as_ref();
    if sample_ids.len() != matrix.n_samples {
        return Err(FastDnaError::InvalidConfig {
            parameter: "sample_ids",
            reason: format!(
                "export_cohort_matrix_parquet: expected {} sample id(s) (CohortMatrix::n_samples), \
                 got {}",
                matrix.n_samples,
                sample_ids.len()
            ),
        });
    }
    // The long/COO schema this function writes carries no sample key other
    // than `sample_id` itself (see this function's own doc comment): two
    // samples sharing one id would be unrecoverably merged into a single
    // group of rows on read-back, while `fastdna.n_samples` in the footer
    // still claims the original, higher count. Rejected here, before a
    // single row is written, rather than left to silently corrupt the
    // caller's cohort.
    {
        let mut seen = rustc_hash::FxHashMap::default();
        for (row, id) in sample_ids.iter().enumerate() {
            if let Some(&first_row) = seen.get(id.as_str()) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "sample_ids",
                    reason: format!(
                        "sample id {id:?} is used for both sample {first_row} and sample {row}; \
                         every sample must have a distinct id"
                    ),
                });
            }
            seen.insert(id.as_str(), row);
        }
    }

    let (file, pending) = AtomicFile::create(path)?;
    let schema = cohort_matrix_schema(with_sequence);

    let mut metadata = vec![
        KeyValue::new("fastdna.k".to_string(), Some(k.to_string())),
        KeyValue::new("fastdna.n_samples".to_string(), Some(matrix.n_samples.to_string())),
        KeyValue::new("fastdna.n_kmers".to_string(), Some(matrix.n_kmers.to_string())),
        KeyValue::new("fastdna.n_candidates".to_string(), Some(matrix.n_candidates.to_string())),
    ];
    if let Some(cutoff) = matrix.truncation_cutoff {
        metadata.push(KeyValue::new("fastdna.truncation_cutoff".to_string(), Some(cutoff.to_string())));
    }

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(metadata))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| export_err(path, e))?;

    let nnz = matrix.row.len();
    let chunk_size = 131_072;
    let mut start = 0usize;
    while start < nnz {
        let end = (start + chunk_size).min(nnz);

        let sample_id_col: ArrayRef = Arc::new(StringArray::from_iter_values(
            matrix.row[start..end].iter().map(|&r| sample_ids[r as usize].as_str()),
        ));
        let kmer_u64_col: ArrayRef = Arc::new(UInt64Array::from_iter_values(
            matrix.col[start..end].iter().map(|&c| matrix.kmer_u64[c as usize]),
        ));
        let mut columns: Vec<ArrayRef> = vec![sample_id_col, kmer_u64_col];
        if with_sequence {
            let seq_col: ArrayRef = Arc::new(StringArray::from_iter_values(
                matrix.col[start..end].iter().map(|&c| matrix.kmer_sequences[c as usize].as_str()),
            ));
            columns.push(seq_col);
        }
        let count_col: ArrayRef = Arc::new(UInt32Array::from(matrix.value[start..end].to_vec()));
        columns.push(count_col);

        let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| export_err(path, e))?;
        writer.write(&batch).map_err(|e| export_err(path, e))?;
        start = end;
    }

    // `close` is called even for an empty (`nnz == 0`) matrix: a cohort
    // whose min_samples/max_kmers left no surviving column must still
    // produce a valid, openable, zero-row Parquet file under this schema --
    // the same "empty result is still a real file" discipline
    // `export_pairs_parquet`'s own empty-stream test pins down.
    writer.close().map_err(|e| export_err(path, e))?;
    pending.commit()?;
    Ok(nnz)
}

pub fn export_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
    with_sequence: bool,
) -> Result<usize> {
    export_counts_parquet(counter, output_path, k, min_count, with_sequence)
}

/// Takes the chunk by value so every buffer reaches Arrow as a move.
/// `UInt64Array::from(Vec<u64>)` and `Buffer::from_vec` both adopt the
/// existing allocation, so none of the three columns is copied here -- the
/// previous `&[T]` signature forced a `to_vec()` on each numeric column,
/// 12 bytes per row (8 for the k-mer, 4 for the count) memcpy'd for nothing,
/// or ~645 MB across the benchmark's 53.8 million rows.
fn write_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    chunk: ChunkBuffers,
    path: &Path,
) -> Result<()> {
    let ChunkBuffers { with_sequence, kmers, seq_bytes, seq_offsets, freqs } = chunk;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(3);
    columns.push(Arc::new(UInt64Array::from(kmers)));
    if with_sequence {
        let offsets = OffsetBuffer::new(ScalarBuffer::from(seq_offsets));
        let seq_arr: ArrayRef = Arc::new(
            StringArray::try_new(offsets, Buffer::from_vec(seq_bytes), None)
                .map_err(|e| export_err(path, e))?,
        );
        columns.push(seq_arr);
    }
    columns.push(Arc::new(UInt32Array::from(freqs)));

    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| export_err(path, e))?;
    writer.write(&batch).map_err(|e| export_err(path, e))?;
    Ok(())
}

pub fn export_counts_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
    with_sequence: bool,
) -> Result<usize> {
    let path = output_path.as_ref();
    let (file, pending) = AtomicFile::create(path)?;
    let mut writer = BufWriter::with_capacity(512 * 1024, file);
    if with_sequence {
        writeln!(writer, "kmer_u64,kmer_sequence,frequency").map_err(|e| io_err(path, e))?;
    } else {
        writeln!(writer, "kmer_u64,frequency").map_err(|e| io_err(path, e))?;
    }

    // Reused across every row. `decode_kmer` would allocate a `String` (and
    // free it, and re-validate its UTF-8) once per exported k-mer -- 53.8
    // million times on the benchmark file. The bases are copied into the
    // `BufWriter` exactly as often as before; only the allocation, the free
    // and the scan go away.
    let mut seq_buf: Vec<u8> = Vec::with_capacity(k);

    let mut written = 0;
    for (kmer_bits, count) in counter.iter() {
        if count >= min_count {
            if with_sequence {
                seq_buf.clear();
                kmer::decode_kmer_into(kmer_bits, k, &mut seq_buf);

                write!(writer, "{kmer_bits},").map_err(|e| io_err(path, e))?;
                writer.write_all(&seq_buf).map_err(|e| io_err(path, e))?;
                writeln!(writer, ",{count}").map_err(|e| io_err(path, e))?;
            } else {
                writeln!(writer, "{kmer_bits},{count}").map_err(|e| io_err(path, e))?;
            }
            written += 1;
        }
    }
    writer.flush().map_err(|e| io_err(path, e))?;
    drop(writer);
    pending.commit()?;
    Ok(written)
}

pub fn export_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
    with_sequence: bool,
) -> Result<usize> {
    export_counts_csv(counter, output_path, k, min_count, with_sequence)
}

/// How to serialize a k-mer frequency spectrum.
///
/// Two formats rather than one because the file has two audiences that
/// cannot both be served: a human or a pandas script wants a named-column
/// CSV, and GenomeScope 2.0 -- the standard route to genome size,
/// heterozygosity and ploidy -- wants precisely what `jellyfish histo`
/// emits and rejects anything else, header included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistogramFormat {
    /// `coverage_depth,kmer_distinct_count` header, then `depth,count`.
    /// The format `--histogram` has always written; the default, so no
    /// existing script changes behavior.
    #[default]
    Csv,
    /// Headerless, space-separated `depth count`, ascending. Byte-for-byte
    /// what `jellyfish histo` produces and GenomeScope 2.0 consumes; also
    /// what `kmc_tools transform histogram`, ntCard and meryl emit.
    GenomeScope,
}

/// Builds the spectrum: how many *distinct* k-mers were seen at each depth.
///
/// `max_depth`, if given, is KMC's `-cx` convention: depths above the cap
/// are summed into the cap's own row rather than dropped, so the total
/// number of distinct k-mers the spectrum accounts for is preserved. That
/// matters because GenomeScope fits a coverage model against that total; a
/// truncated tail makes the fit quietly wrong rather than visibly missing.
/// The cap row is created even when no k-mer had exactly that depth --
/// otherwise the folded k-mers would vanish, which is the very thing the
/// convention exists to prevent.
fn spectrum(counter: &KmerCounter, max_depth: Option<u32>) -> Vec<(u32, u64)> {
    let mut hist_map: FxHashMap<u32, u64> = FxHashMap::default();
    for count in counter.iter().map(|(_, c)| c) {
        let depth = match max_depth {
            Some(cap) => count.min(cap),
            None => count,
        };
        *hist_map.entry(depth).or_insert(0) += 1;
    }

    let mut sorted: Vec<(u32, u64)> = hist_map.into_iter().collect();
    sorted.sort_unstable_by_key(|&(depth, _)| depth);
    sorted
}

/// Writes the k-mer frequency spectrum in `format`, capping depths at
/// `max_depth` if given (see `spectrum`).
///
/// Goes through `AtomicFile` like every other exporter: a disk-full error
/// or a Ctrl+C partway through must leave the previous good histogram in
/// place rather than a truncated file that a downstream fitter will happily
/// read as a real spectrum.
pub fn export_histogram<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    format: HistogramFormat,
    max_depth: Option<u32>,
) -> Result<()> {
    let path = output_path.as_ref();
    let (file, pending) = AtomicFile::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    if format == HistogramFormat::Csv {
        writeln!(writer, "coverage_depth,kmer_distinct_count").map_err(|e| io_err(path, e))?;
    }

    for (depth, distinct) in spectrum(counter, max_depth) {
        match format {
            HistogramFormat::Csv => writeln!(writer, "{depth},{distinct}"),
            HistogramFormat::GenomeScope => writeln!(writer, "{depth} {distinct}"),
        }
        .map_err(|e| io_err(path, e))?;
    }

    writer.flush().map_err(|e| io_err(path, e))?;
    drop(writer);
    pending.commit()?;
    Ok(())
}

/// The uncapped CSV spectrum -- the exact bytes `--histogram` produced
/// before a format flag existed. Kept as its own entry point so callers
/// that already use it are untouched.
pub fn export_histogram_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
) -> Result<()> {
    export_histogram(counter, output_path, HistogramFormat::Csv, None)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A schema mismatch inside `write_chunk` (mismatched column lengths) is
    /// a genuine Arrow error, not an I/O failure. This exercises that path
    /// directly against the private `write_chunk` helper -- the public
    /// export functions always build correctly-shaped batches internally,
    /// so there is no way to reach this failure through the public API
    /// without corrupting a `KmerCounter` first, which would not be testing
    /// the same thing.
    #[test]
    fn arrow_schema_mismatch_becomes_an_export_error_not_io() {
        let dir = std::env::temp_dir().join("fastdna_export_err_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mismatch.parquet");

        let file = File::create(&path).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("kmer_u64", DataType::UInt64, false),
            Field::new("kmer_sequence", DataType::Utf8, false),
            Field::new("frequency", DataType::UInt32, false),
        ]));
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        // Mismatched lengths: two u64s but only one sequence/frequency.
        let mut chunk = ChunkBuffers::with_capacity(2, 4, true);
        chunk.push(0, 4, 5);
        chunk.kmers.push(2);

        let result = write_chunk(&mut writer, &schema, chunk, &path);

        match result {
            Err(FastDnaError::Export { path: p, .. }) => {
                assert_eq!(p, path, "the Export error must name the destination file");
            }
            other => panic!("expected Export, got {other:?}"),
        }

        let _ = std::fs::remove_file(&path);
    }

    /// The CSV row is now emitted as three writes into the `BufWriter`
    /// (integer, decoded bases from a reused buffer, integer) rather than one
    /// `writeln!` over a freshly allocated `String`. The bytes on disk must
    /// be identical, header included -- `train_classifier.py` parses these
    /// exact column names.
    #[test]
    fn csv_rows_keep_their_exact_bytes() {
        let k = 4;
        let mut counter = KmerCounter::new();
        // "AACG" = 00 00 01 10 = 6, canonical already (its reverse
        // complement "CGTT" = 0b01101111 = 111 is larger).
        counter.insert_batch(&[6, 6, 0]);

        let dir = std::env::temp_dir().join("fastdna_export_csv_bytes_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("counts.csv");

        let written = export_counts_csv(&counter, &path, k, 1, true).unwrap();
        assert_eq!(written, 2);

        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.remove(0), "kmer_u64,kmer_sequence,frequency");
        lines.sort_unstable();
        assert_eq!(lines, vec!["0,AAAA,1", "6,AACG,2"]);

        let _ = std::fs::remove_file(&path);
    }

    /// `with_sequence=false` is the default (see `counts_schema`'s doc
    /// comment for why): the header and every row must drop the
    /// `kmer_sequence` column entirely, not merely leave it empty.
    #[test]
    fn csv_export_without_sequence_omits_the_column_entirely() {
        let k = 4;
        let mut counter = KmerCounter::new();
        counter.insert_batch(&[6, 6, 0]);

        let dir = std::env::temp_dir().join("fastdna_export_csv_no_seq_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("counts.csv");

        let written = export_counts_csv(&counter, &path, k, 1, false).unwrap();
        assert_eq!(written, 2);

        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.remove(0), "kmer_u64,frequency");
        lines.sort_unstable();
        assert_eq!(lines, vec!["0,1", "6,2"]);

        let _ = std::fs::remove_file(&path);
    }

    /// The `kmer_sequence` column is now assembled as raw Arrow buffers --
    /// bases written straight into the value buffer, offsets pushed by hand
    /// -- instead of a `Vec<String>`. Nothing else in the tree reads a
    /// Parquet file back, so this pins the result: every row's decoded
    /// sequence must still equal `decode_kmer`, and the offsets must still
    /// cut the value buffer in the right places.
    ///
    /// Deliberately spans a chunk boundary (131 072 rows) so the
    /// buffers-are-moved-and-replaced path is exercised alongside the
    /// short final chunk.
    #[test]
    fn parquet_sequence_column_round_trips_across_a_chunk_boundary() {
        use arrow::array::{Array, StringArray as RoundTripStrings, UInt64Array as RoundTripU64s};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let k = 31;
        let rows = 131_072 + 5;

        let mut counter = KmerCounter::new();
        // Distinct, spread across the 2-bit space so the decoded bases vary.
        let kmers: Vec<u64> =
            (0..rows as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 2).collect();
        counter.insert_batch(&kmers);

        let dir = std::env::temp_dir().join("fastdna_export_roundtrip_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("counts.parquet");

        let written = export_counts_parquet(&counter, &path, k, 1, true).unwrap();
        assert_eq!(written, counter.distinct_kmers());

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();

        let mut seen = 0usize;
        for batch in reader {
            let batch = batch.unwrap();
            let bits = batch.column(0).as_any().downcast_ref::<RoundTripU64s>().unwrap();
            let seqs = batch.column(1).as_any().downcast_ref::<RoundTripStrings>().unwrap();
            assert_eq!(bits.len(), seqs.len());

            for row in 0..bits.len() {
                assert_eq!(
                    seqs.value(row),
                    kmer::decode_kmer(bits.value(row), k),
                    "sequence column diverges from decode_kmer at row {row}"
                );
            }
            seen += bits.len();
        }

        assert_eq!(seen, written, "every written row must read back");

        let _ = std::fs::remove_file(&path);
    }

    /// The headline claim behind turning the column off by default: the
    /// schema drops to two columns, and the file this produces is smaller
    /// than the one carrying `kmer_sequence` for the same data.
    #[test]
    fn parquet_without_sequence_has_two_columns_and_is_smaller() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let k = 21;
        let rows = 5_000;

        let mut counter = KmerCounter::new();
        let kmers: Vec<u64> =
            (0..rows as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 2).collect();
        counter.insert_batch(&kmers);

        let dir = std::env::temp_dir().join("fastdna_export_no_seq_size_test");
        std::fs::create_dir_all(&dir).unwrap();
        let lean_path = dir.join("lean.parquet");
        let full_path = dir.join("full.parquet");

        export_counts_parquet(&counter, &lean_path, k, 1, false).unwrap();
        export_counts_parquet(&counter, &full_path, k, 1, true).unwrap();

        let file = File::open(&lean_path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            assert_eq!(
                batch.schema().fields().len(),
                2,
                "without with_sequence the schema must be kmer_u64+frequency only"
            );
        }

        let lean_size = std::fs::metadata(&lean_path).unwrap().len();
        let full_size = std::fs::metadata(&full_path).unwrap().len();
        assert!(
            lean_size < full_size,
            "without the sequence column ({lean_size} B) must be smaller than with it \
             ({full_size} B)"
        );

        let _ = std::fs::remove_file(&lean_path);
        let _ = std::fs::remove_file(&full_path);
    }

    /// `export_pairs_parquet`'s output must be indistinguishable from
    /// `export_counts_parquet`'s: same schema, same footer metadata, so a
    /// set operation's result (`setops.rs`) reopens as a valid `KmerTable`
    /// with no special-casing anywhere in `ktab.rs`.
    #[test]
    fn export_pairs_parquet_round_trips_and_is_a_valid_kmer_table() {
        use crate::ktab::KmerTable;

        let dir = std::env::temp_dir().join("fastdna_export_pairs_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pairs.parquet");

        let pairs: Vec<Result<(u64, u32)>> = vec![Ok((1, 5)), Ok((2, 7)), Ok((100, 1))];
        let written = export_pairs_parquet(pairs, &path, 4).unwrap();
        assert_eq!(written, 3);

        let table = KmerTable::open(&path).unwrap();
        assert_eq!(table.k(), 4);
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(1).unwrap(), Some(5));
        assert_eq!(table.get(2).unwrap(), Some(7));
        assert_eq!(table.get(100).unwrap(), Some(1));

        let _ = std::fs::remove_file(&path);
    }

    /// An empty stream (e.g. an intersection with no shared k-mers) must
    /// still write a valid, openable, zero-row table -- not be rejected or
    /// skipped -- so a caller can always reopen a set operation's result
    /// without special-casing "what if there were no matches".
    #[test]
    fn export_pairs_parquet_of_an_empty_stream_is_a_valid_empty_table() {
        use crate::ktab::KmerTable;

        let dir = std::env::temp_dir().join("fastdna_export_pairs_empty_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_pairs.parquet");

        let pairs: Vec<Result<(u64, u32)>> = Vec::new();
        let written = export_pairs_parquet(pairs, &path, 4).unwrap();
        assert_eq!(written, 0);

        let table = KmerTable::open(&path).unwrap();
        assert!(table.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    /// A failure partway through the input stream (e.g. a Parquet read
    /// error surfacing from one of a set operation's own input tables) must
    /// propagate as an error from this function, not be silently swallowed
    /// or produce a truncated-but-successful file.
    #[test]
    fn export_pairs_parquet_propagates_an_error_from_the_stream() {
        let dir = std::env::temp_dir().join("fastdna_export_pairs_err_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("err_pairs.parquet");

        let pairs: Vec<Result<(u64, u32)>> =
            vec![Ok((1, 1)), Err(FastDnaError::Internal { detail: "boom".to_string() })];
        let result = export_pairs_parquet(pairs, &path, 4);
        assert!(matches!(result, Err(FastDnaError::Internal { .. })));

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------
    // export_cohort_matrix_parquet
    // -----------------------------------------------------------------

    /// Two samples sharing k-mer 1, each with one private k-mer -- the same
    /// small fixture shape `cohort/matrix.rs`'s own unit tests use, built
    /// through the real `build_cohort_matrix` rather than a hand-rolled
    /// `CohortMatrix` literal, so this test also pins down that the two
    /// modules' expectations of each other's shape stay in sync.
    fn small_cohort_matrix() -> (crate::cohort::matrix::CohortMatrix, Vec<String>) {
        use crate::cohort::matrix::build_cohort_matrix;

        let mut sample_0 = KmerCounter::new();
        sample_0.insert_batch(&[1, 1, 1, 5]);
        let mut sample_1 = KmerCounter::new();
        sample_1.insert_batch(&[1, 1, 9, 9, 9, 9]);

        let matrix = build_cohort_matrix(&[sample_0, sample_1], 1, None, 4);
        (matrix, vec!["sample_0".to_string(), "sample_1".to_string()])
    }

    /// Without `--with-sequence` the file must carry exactly `sample_id`,
    /// `kmer_u64`, `count` -- no more, no less -- mirroring `counts_schema`'s
    /// own default-lean convention.
    #[test]
    fn cohort_matrix_parquet_schema_without_sequence_has_three_columns() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let (matrix, sample_ids) = small_cohort_matrix();
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_schema_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cohort_no_seq.parquet");

        let written = export_cohort_matrix_parquet(&matrix, &sample_ids, &path, 4, false).unwrap();
        assert_eq!(written, matrix.row.len());

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let schema = batch.schema();
            let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert_eq!(names, vec!["sample_id", "kmer_u64", "count"]);
        }

        let _ = std::fs::remove_file(&path);
    }

    /// `--with-sequence` inserts `kmer_sequence` between `kmer_u64` and
    /// `count`, and every value in it must equal `decode_kmer(kmer_u64, k)`
    /// -- the same "derivable, but written when asked" contract
    /// `export_counts_parquet`'s own sequence column keeps.
    #[test]
    fn cohort_matrix_parquet_with_sequence_adds_a_matching_column() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use arrow::array::{Array, StringArray as RtStrings, UInt64Array as RtU64s};

        let (matrix, sample_ids) = small_cohort_matrix();
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_seq_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cohort_with_seq.parquet");

        export_cohort_matrix_parquet(&matrix, &sample_ids, &path, 4, true).unwrap();

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let mut seen = 0usize;
        for batch in reader {
            let batch = batch.unwrap();
            let schema = batch.schema();
            let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            assert_eq!(names, vec!["sample_id", "kmer_u64", "kmer_sequence", "count"]);

            let kmers = batch.column(1).as_any().downcast_ref::<RtU64s>().unwrap();
            let seqs = batch.column(2).as_any().downcast_ref::<RtStrings>().unwrap();
            for row in 0..batch.num_rows() {
                assert_eq!(seqs.value(row), kmer::decode_kmer(kmers.value(row), 4));
            }
            seen += batch.num_rows();
        }
        assert_eq!(seen, matrix.row.len());

        let _ = std::fs::remove_file(&path);
    }

    /// Every `(sample_id, kmer_u64, count)` row written must exactly match
    /// what `CohortMatrix`'s own `(row, col, value)` COO triples encode --
    /// the round-trip this schema exists to make possible with no
    /// FastDNA-specific reader.
    #[test]
    fn cohort_matrix_parquet_rows_round_trip_the_coo_triples() {
        use std::collections::HashSet;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use arrow::array::{Array, StringArray as RtStrings, UInt32Array as RtU32s, UInt64Array as RtU64s};

        let (matrix, sample_ids) = small_cohort_matrix();
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_roundtrip_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cohort_roundtrip.parquet");

        export_cohort_matrix_parquet(&matrix, &sample_ids, &path, 4, false).unwrap();

        let expected: HashSet<(String, u64, u32)> = matrix
            .row
            .iter()
            .zip(&matrix.col)
            .zip(&matrix.value)
            .map(|((&r, &c), &v)| (sample_ids[r as usize].clone(), matrix.kmer_u64[c as usize], v))
            .collect();

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let mut actual: HashSet<(String, u64, u32)> = HashSet::new();
        for batch in reader {
            let batch = batch.unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<RtStrings>().unwrap();
            let kmers = batch.column(1).as_any().downcast_ref::<RtU64s>().unwrap();
            let counts = batch.column(2).as_any().downcast_ref::<RtU32s>().unwrap();
            for row in 0..batch.num_rows() {
                actual.insert((ids.value(row).to_string(), kmers.value(row), counts.value(row)));
            }
        }

        assert_eq!(actual, expected);
    }

    /// A schema mismatch here is a caller bug (a `sample_ids` list built for
    /// the wrong cohort), and must be rejected before any file is written --
    /// not silently truncated or panicking on an out-of-bounds index.
    #[test]
    fn cohort_matrix_parquet_rejects_a_sample_ids_length_mismatch() {
        let (matrix, _sample_ids) = small_cohort_matrix();
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_mismatch_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("should_not_exist.parquet");

        let wrong_ids = vec!["only_one".to_string()];
        let result = export_cohort_matrix_parquet(&matrix, &wrong_ids, &path, 4, false);
        match result {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "sample_ids"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        assert!(!path.exists(), "no file must be written on a rejected call");
    }

    /// An empty matrix (e.g. `min_samples` above the cohort size, or
    /// `min_samples` filtering out every candidate) must still produce a
    /// valid, openable, zero-row file under the schema -- not be rejected --
    /// mirroring `export_pairs_parquet`'s own empty-stream guarantee.
    #[test]
    fn cohort_matrix_parquet_of_an_empty_matrix_is_a_valid_zero_row_file() {
        use crate::cohort::matrix::build_cohort_matrix;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let matrix = build_cohort_matrix(&[], 1, None, 4);
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_empty_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_cohort.parquet");

        let written = export_cohort_matrix_parquet(&matrix, &[], &path, 4, false).unwrap();
        assert_eq!(written, 0);

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let total: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(total, 0);

        let _ = std::fs::remove_file(&path);
    }

    /// The footer must carry `k`/`n_samples`/`n_kmers`/`n_candidates`, and
    /// `truncation_cutoff` specifically when `max_kmers` actually truncated
    /// something -- so a caller with no FastDNA library, just a Parquet
    /// reader that exposes footer key-value metadata, can recover the same
    /// facts `gwas.py`'s truncation warning is built from.
    #[test]
    fn cohort_matrix_parquet_footer_metadata_reports_truncation() {
        use crate::cohort::matrix::build_cohort_matrix;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let mut sample_0 = KmerCounter::new();
        sample_0.insert_batch(&[1, 2, 3]);
        let mut sample_1 = KmerCounter::new();
        sample_1.insert_batch(&[1, 2, 3]);
        let matrix = build_cohort_matrix(&[sample_0, sample_1], 1, Some(1), 4);
        assert!(matrix.truncation_cutoff.is_some(), "fixture must actually truncate");

        let sample_ids = vec!["s0".to_string(), "s1".to_string()];
        let dir = std::env::temp_dir().join("fastdna_export_cohort_matrix_footer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("footer.parquet");
        export_cohort_matrix_parquet(&matrix, &sample_ids, &path, 4, false).unwrap();

        let file = File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let metadata = builder.metadata().clone();
        let kvs = metadata.file_metadata().key_value_metadata().unwrap();
        let get = |key: &str| kvs.iter().find(|kv| kv.key == key).and_then(|kv| kv.value.clone());

        assert_eq!(get("fastdna.k"), Some("4".to_string()));
        assert_eq!(get("fastdna.n_samples"), Some("2".to_string()));
        assert_eq!(get("fastdna.n_kmers"), Some(matrix.n_kmers.to_string()));
        assert_eq!(get("fastdna.n_candidates"), Some(matrix.n_candidates.to_string()));
        assert_eq!(
            get("fastdna.truncation_cutoff"),
            matrix.truncation_cutoff.map(|c| c.to_string()),
            "truncation_cutoff must be recorded exactly when the matrix itself was truncated"
        );

        let _ = std::fs::remove_file(&path);
    }
}
