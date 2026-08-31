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
use parquet::file::properties::WriterProperties;
use rustc_hash::FxHashMap;

use crate::atomic::AtomicFile;
use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::kmer;

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
}
