// src/main.rs

// main.rs legitimately owns all console output for the CLI binary; the
// crate-wide clippy denials in Cargo.toml exist to keep the library core
// silent, not this entry point.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use std::time::Instant;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use flate2::read::MultiGzDecoder;

use fastdna::cli::Cli;
use fastdna::error::{FastDnaError, Result};
use fastdna::export;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};
use fastdna::progress::Progress;

fn main() -> ExitCode {
    let args = Cli::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn spinner(message: &str) -> ProgressBar {
    let style = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");

    let pb = ProgressBar::new_spinner();
    pb.set_style(style);
    pb.set_message(message.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

fn run(args: Cli) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: High-Performance Genomic Kernel (Rust)  ");
    println!("==================================================");
    println!("Input:          {}", args.input.display());
    println!("Output:         {}", args.output.display());
    println!("k-mer Size:     {}", args.kmer_size);
    println!("Quality Cutoff: Q >= {}", args.min_quality);

    // A single source of truth for the thread-count default: PipelineConfig's
    // own Default impl, not a hardcoded number duplicated here.
    let default_config = PipelineConfig::default();
    let threads = args.threads.unwrap_or(default_config.num_threads);
    println!("Worker Threads: {threads}");
    println!("--------------------------------------------------");

    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4,
        min_quality: args.min_quality,
        batch_size: 10_000,
        num_threads: threads,
        progress_interval: default_config.progress_interval,
    };

    let start_time = Instant::now();
    let pb = spinner("Analyzing genomic reads in streaming...");

    let is_gz = args.input.extension().is_some_and(|ext| ext == "gz");
    let file = File::open(&args.input)
        .map_err(|e| FastDnaError::Io {
            path: args.input.clone(),
            source: e,
        })
        .inspect_err(|_| pb.abandon())?;

    let buf_reader: Box<dyn BufRead + Send + 'static> = if is_gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let fastq_reader = FastqReader::new(buf_reader);

    // The core stays silent; this closure is what turns events into a spinner.
    let bar = pb.clone();
    let on_progress = move |event: Progress| {
        if let Progress::ReadsProcessed(n) = event {
            bar.set_message(format!("Analyzing genomic reads... {n} processed"));
        }
    };

    let (mut counter, qc, total_reads) = process_stream_parallel(
        fastq_reader,
        config,
        &args.input,
        Some(&on_progress),
        None, // the CLI has no way to cancel a running call yet
    )
    .inspect_err(|_| pb.abandon())?;

    let elapsed = start_time.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Processing completed in {elapsed:.2}s"));

    // Frequency filters, applied in RAM before anything is written.
    let prune_stats = counter.prune(args.min_count, args.max_count);
    if prune_stats.dropped_min > 0 || prune_stats.dropped_max > 0 {
        println!(
            "Filtered: {} k-mers below --min-count, {} above --max-count",
            prune_stats.dropped_min, prune_stats.dropped_max
        );
    }

    let pb_export = spinner("Compressing and writing to disk (Parquet/CSV)...");
    let export_start = Instant::now();

    let output_str = args.output.to_string_lossy();
    let records_written = if output_str.ends_with(".parquet") {
        export::export_parquet(&counter, &args.output, args.kmer_size, args.min_count)
            .inspect_err(|_| pb_export.abandon())?
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count)
            .inspect_err(|_| pb_export.abandon())?
    };

    let export_elapsed = export_start.elapsed().as_secs_f64();
    pb_export.finish_with_message(format!("Export completed in {export_elapsed:.2}s"));

    // Honour --qc and --histogram, which were previously parsed and ignored.
    qc.export_json(&args.qc)?;
    println!("QC report written to: {}", args.qc.display());

    if let Some(histogram_path) = &args.histogram {
        export::export_histogram_csv(&counter, histogram_path)?;
        println!("Histogram written to: {}", histogram_path.display());
    }

    println!("--------------------------------------------------");
    println!("Total Reads: {total_reads}");
    println!(
        "Total k-mers Indexed: {} | Distinct k-mers: {}",
        counter.total_kmers(),
        counter.distinct_kmers()
    );
    println!("Records Written to Disk: {records_written}");

    Ok(())
}
