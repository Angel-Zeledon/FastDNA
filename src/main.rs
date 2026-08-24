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

use fastdna_core::cli::{Cli, CliStrategy};
use fastdna_core::error::{FastDnaError, Result};
use fastdna_core::export;
use fastdna_core::fastq::FastqReader;
use fastdna_core::pipeline::{
    process_stream_parallel_with_policy, CountStrategy, MemoryPolicy, PipelineConfig,
};
use fastdna_core::progress::Progress;

/// A `.gz` input's on-disk size is compressed, not the decompressed size
/// the counting pipeline actually sees -- feeding the compressed size
/// straight into `mem_estimate`'s occurrence estimate would badly
/// under-predict peak memory for gzip input. This is a fixed, documented
/// approximation rather than a live measurement (e.g. decompressing a
/// prefix to measure the real ratio): FASTQ text compresses reasonably
/// consistently with gzip (repetitive quality strings, a small DNA
/// alphabet), and a fixed multiplier costs nothing on every run, unlike a
/// live sample. If this proves too far off in practice, a live-sampled
/// ratio is the natural next step -- this is a starting point, not a
/// value proven optimal by a sweep across real `.gz` inputs.
const GZIP_FASTQ_EXPANSION_FACTOR: f64 = 3.5;

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

/// Formats a byte count for the strategy banner (binary units, one decimal
/// place) -- `format_bytes(8_600_000_000)` -> `"8.01GB"`.
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0;
    while value >= 1024.0 && unit_idx < UNITS.len() - 1 {
        value /= 1024.0;
        unit_idx += 1;
    }
    format!("{value:.2}{}", UNITS[unit_idx])
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

    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4,
        min_quality: args.min_quality,
        batch_size: 10_000,
        num_threads: threads,
        progress_interval: default_config.progress_interval,
    };

    let is_gz = args.input.extension().is_some_and(|ext| ext == "gz");
    let file = File::open(&args.input).map_err(|e| FastDnaError::Io { path: args.input.clone(), source: e })?;

    // Best-effort: a size we cannot read (an unusual filesystem, a stream
    // that lies about its length) just disables size-based estimation --
    // see `MemoryPolicy::estimated_input_bytes`'s doc comment for why that
    // defaults to the conservative in-memory choice rather than failing
    // the whole run over a memory *prediction* that could not be made.
    let on_disk_bytes = file.metadata().ok().map(|m| m.len());
    let estimated_input_bytes = on_disk_bytes.map(|bytes| {
        if is_gz {
            (bytes as f64 * GZIP_FASTQ_EXPANSION_FACTOR) as u64
        } else {
            bytes
        }
    });

    let policy = MemoryPolicy {
        strategy: match args.strategy {
            CliStrategy::Auto => None,
            CliStrategy::Memory => Some(CountStrategy::InMemory),
            CliStrategy::Disk => Some(CountStrategy::Disk),
        },
        max_ram_bytes: args.max_ram,
        estimated_input_bytes,
    };

    // Printed from the *same* pure decision `process_stream_parallel_with_
    // policy` itself will act on below (not a separate ad hoc guess), so
    // this line and the run it precedes can never disagree with each
    // other -- "record which one was used so it is visible rather than
    // mysterious" means this has to be the actual decision, not a
    // approximation of it.
    let preview_decision = fastdna_core::pipeline::resolve_strategy(&policy, &config);
    println!(
        "Strategy:       {} (estimated peak {}, budget {})",
        preview_decision.strategy.as_str(),
        format_bytes(preview_decision.estimated_peak_bytes),
        format_bytes(preview_decision.budget_bytes),
    );
    println!("--------------------------------------------------");

    let start_time = Instant::now();
    let pb = spinner("Analyzing genomic reads in streaming...");

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

    let (mut counter, qc, total_reads, decision) = process_stream_parallel_with_policy(
        fastq_reader,
        config,
        &args.input,
        Some(&on_progress),
        None, // the CLI has no way to cancel a running call yet
        policy,
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
    println!("Strategy Used: {}", decision.strategy.as_str());
    println!("Total Reads: {total_reads}");
    println!(
        "Total k-mers Indexed: {} | Distinct k-mers: {}",
        counter.total_kmers(),
        counter.distinct_kmers()
    );
    println!("Records Written to Disk: {records_written}");

    Ok(())
}
