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
use crate::kmer;

pub fn export_counts_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("kmer_u64", DataType::UInt64, false),
        Field::new("kmer_sequence", DataType::Utf8, false),
        Field::new("frequency", DataType::UInt32, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
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
                write_chunk(&mut writer, &schema, &u64_chunk, &seq_chunk, &freq_chunk)?;
                total_written += u64_chunk.len();
                u64_chunk.clear();
                seq_chunk.clear();
                freq_chunk.clear();
            }
        }
    }

    if !u64_chunk.is_empty() {
        total_written += u64_chunk.len();
        write_chunk(&mut writer, &schema, &u64_chunk, &seq_chunk, &freq_chunk)?;
    }

    writer.close()?;
    Ok(total_written)
}

pub fn export_parquet<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    export_counts_parquet(counter, output_path, k, min_count)
}

fn write_chunk(
    writer: &mut ArrowWriter<File>,
    schema: &Arc<Schema>,
    u64s: &[u64],
    seqs: &[String],
    freqs: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let u64_arr: ArrayRef = Arc::new(UInt64Array::from(u64s.to_vec()));
    let seq_arr: ArrayRef = Arc::new(StringArray::from_iter_values(seqs.iter().map(|s| s.as_str())));
    let freq_arr: ArrayRef = Arc::new(UInt32Array::from(freqs.to_vec()));

    let batch = RecordBatch::try_new(schema.clone(), vec![u64_arr, seq_arr, freq_arr])?;
    writer.write(&batch)?;
    Ok(())
}

pub fn export_counts_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(512 * 1024, file);
    writeln!(writer, "kmer_u64,kmer_sequence,frequency")?;

    let mut written = 0;
    for (&kmer_bits, &count) in counter.iter() {
        if count >= min_count {
            writeln!(writer, "{},{},{}", kmer_bits, kmer::decode_kmer(kmer_bits, k), count)?;
            written += 1;
        }
    }
    writer.flush()?;
    Ok(written)
}

pub fn export_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
    k: usize,
    min_count: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    export_counts_csv(counter, output_path, k, min_count)
}

pub fn export_histogram_csv<P: AsRef<Path>>(
    counter: &KmerCounter,
    output_path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    writeln!(writer, "coverage_depth,kmer_distinct_count")?;

    let mut hist_map: FxHashMap<u32, u64> = FxHashMap::default();
    for &count in counter.iter().map(|(_, c)| c) {
        *hist_map.entry(count).or_insert(0) += 1;
    }

    let mut sorted: Vec<(u32, u64)> = hist_map.into_iter().collect();
    sorted.sort_unstable_by_key(|&(cov, _)| cov);

    for (coverage, count) in sorted {
        writeln!(writer, "{},{}", coverage, count)?;
    }
    writer.flush()?;
    Ok(())
}