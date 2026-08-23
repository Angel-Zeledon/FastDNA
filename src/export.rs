// src/export.rs

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use arrow::array::{ArrayRef, StringArray, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rustc_hash::FxHashMap;

use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::kmer;

/// Wraps an Arrow/Parquet serialization or writer failure. These are not I/O
/// errors: the bytes may never touch a disk (e.g. a schema mismatch building
/// a `RecordBatch` in memory), so laundering them through `FastDnaError::Io`
/// with a synthetic `std::io::Error` erases their real type and misleads
/// callers -- including a future Python binding, where this must surface as
/// a distinct exception class rather than `OSError`.
fn export_err<E: std::fmt::Display>(path: &Path, err: E) -> FastDnaError {
    FastDnaError::Export {
        path: path.to_path_buf(),
        reason: err.to_string(),
    }
}

/// Wraps a genuine `std::io::Error` from a writer call (`writeln!`, `flush`)
/// as `FastDnaError::Io`, preserving the real source error.
fn io_err(path: &Path, err: std::io::Error) -> FastDnaError {
    FastDnaError::Io { path: path.to_path_buf(), source: err }
}

pub fn export_counts_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize> {
    let path = output_path.as_ref();
    let file = File::create(path).map_err(|e| FastDnaError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("kmer_u64", DataType::UInt64, false),
        Field::new("kmer_sequence", DataType::Utf8, false),
        Field::new("frequency", DataType::UInt32, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| export_err(path, e))?;
    let chunk_size = 131_072;
    let mut total_written = 0;

    let mut u64_chunk = Vec::with_capacity(chunk_size);
    let mut seq_chunk = Vec::with_capacity(chunk_size);
    let mut freq_chunk = Vec::with_capacity(chunk_size);

    for (&kmer_bits, &count) in counter.iter() {
        if count >= min_count {
            u64_chunk.push(kmer_bits);
            seq_chunk.push(kmer::decode_kmer(kmer_bits, k));
            freq_chunk.push(count);

            if u64_chunk.len() >= chunk_size {
                write_chunk(&mut writer, &schema, &u64_chunk, &seq_chunk, &freq_chunk, path)?;
                total_written += u64_chunk.len();
                u64_chunk.clear();
                seq_chunk.clear();
                freq_chunk.clear();
            }
        }
    }

    if !u64_chunk.is_empty() {
        total_written += u64_chunk.len();
        write_chunk(&mut writer, &schema, &u64_chunk, &seq_chunk, &freq_chunk, path)?;
    }

    writer.close().map_err(|e| export_err(path, e))?;
    Ok(total_written)
}

pub fn export_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize> {
    export_counts_parquet(counter, output_path, k, min_count)
}

fn write_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    u64s: &[u64],
    seqs: &[String],
    freqs: &[u32],
    path: &Path,
) -> Result<()> {
    let u64_arr: ArrayRef = Arc::new(UInt64Array::from(u64s.to_vec()));
    let seq_arr: ArrayRef = Arc::new(StringArray::from_iter_values(seqs.iter().map(|s| s.as_str())));
    let freq_arr: ArrayRef = Arc::new(UInt32Array::from(freqs.to_vec()));

    let batch = RecordBatch::try_new(schema.clone(), vec![u64_arr, seq_arr, freq_arr])
        .map_err(|e| export_err(path, e))?;
    writer.write(&batch).map_err(|e| export_err(path, e))?;
    Ok(())
}

pub fn export_counts_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize> {
    let path = output_path.as_ref();
    let file = File::create(path).map_err(|e| FastDnaError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::with_capacity(512 * 1024, file);
    writeln!(writer, "kmer_u64,kmer_sequence,frequency").map_err(|e| io_err(path, e))?;

    let mut written = 0;
    for (&kmer_bits, &count) in counter.iter() {
        if count >= min_count {
            writeln!(writer, "{},{},{}", kmer_bits, kmer::decode_kmer(kmer_bits, k), count)
                .map_err(|e| io_err(path, e))?;
            written += 1;
        }
    }
    writer.flush().map_err(|e| io_err(path, e))?;
    Ok(written)
}

pub fn export_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize> {
    export_counts_csv(counter, output_path, k, min_count)
}

pub fn export_histogram_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
) -> Result<()> {
    let path = output_path.as_ref();
    let file = File::create(path).map_err(|e| FastDnaError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    writeln!(writer, "coverage_depth,kmer_distinct_count").map_err(|e| io_err(path, e))?;

    let mut hist_map: FxHashMap<u32, u64> = FxHashMap::default();
    for &count in counter.iter().map(|(_, c)| c) {
        *hist_map.entry(count).or_insert(0) += 1;
    }

    let mut sorted: Vec<(u32, u64)> = hist_map.into_iter().collect();
    sorted.sort_unstable_by_key(|&(cov, _)| cov);

    for (coverage, count) in sorted {
        writeln!(writer, "{},{}", coverage, count).map_err(|e| io_err(path, e))?;
    }
    writer.flush().map_err(|e| io_err(path, e))?;
    Ok(())
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
        let u64s = vec![1u64, 2u64];
        let seqs = vec!["AAAA".to_string()];
        let freqs = vec![5u32];

        let result = write_chunk(&mut writer, &schema, &u64s, &seqs, &freqs, &path);

        match result {
            Err(FastDnaError::Export { path: p, .. }) => {
                assert_eq!(p, path, "the Export error must name the destination file");
            }
            other => panic!("expected Export, got {other:?}"),
        }

        let _ = std::fs::remove_file(&path);
    }
}
