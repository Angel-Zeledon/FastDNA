// src/main.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use flate2::read::MultiGzDecoder;

use fastdna::cli::Cli;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};
use fastdna::fastq::FastqReader;
use fastdna::export;

fn main() {
    let args = Cli::parse();
    
    println!("==================================================");
    println!(" FastDNA: High-Performance Genomic Kernel (Rust)  ");
    println!("==================================================");
    
    println!("Input:          {}", args.input.display());
    println!("Output:         {}", args.output.display());
    println!("k-mer Size:     {}", args.kmer_size);
    println!("Quality Cutoff: Q >= {}", args.min_quality);
    
    let threads = args.threads.unwrap_or(8);
    println!("Worker Threads: {}", threads);
    println!("--------------------------------------------------");

    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4, 
        min_quality: args.min_quality as f64, // coerced to f64
        batch_size: 10000,
        num_threads: threads,
    };

    let start_time = Instant::now();

    let spinner_style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style);
    pb.set_message("Analyzing genomic reads in streaming...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let is_gz = args.input.extension().map_or(false, |ext| ext == "gz");
    let file = File::open(&args.input).expect("Error opening input file");
    
    let buf_reader: Box<dyn BufRead + Send + 'static> = if is_gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let fastq_reader = FastqReader::new(buf_reader);

    let (counter, qc, _total_reads) = process_stream_parallel(fastq_reader, config);
    
    let elapsed = start_time.elapsed().as_secs_f64();
    pb.finish_with_message(format!("✔ Processing completed in {:.2}s", elapsed));

    let pb_export = ProgressBar::new_spinner();
    pb_export.set_style(
        ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] {msg}")
            .unwrap()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    pb_export.set_message("Compressing and writing to disk (Parquet/CSV)...");
    pb_export.enable_steady_tick(std::time::Duration::from_millis(100));

    let export_start = Instant::now();
    let output_str = args.output.to_string_lossy();
    let records_written = if output_str.ends_with(".parquet") {
        export::export_parquet(&counter, &args.output, args.kmer_size, args.min_count).unwrap()
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count).unwrap()
    };

    let export_elapsed = export_start.elapsed().as_secs_f64();
    pb_export.finish_with_message(format!("✔ Export completed in {:.2}s", export_elapsed));

    println!("--------------------------------------------------");
    if let Ok(qc_json) = serde_json::to_string(&qc) {
        println!("QC Summary: {}", qc_json);
    }
    println!("Total k-mers Indexed: {} | Distinct k-mers: {}", counter.total_kmers(), counter.distinct_kmers());
    println!("Records Written to Disk: {}", records_written);
}