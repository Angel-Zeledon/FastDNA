// src/main.rs

// main.rs legitimately owns all console output for the CLI binary; the
// crate-wide clippy denials in Cargo.toml exist to keep the library core
// silent, not this entry point.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use fastdna_core::cli::{Cli, CliHistogramFormat, CliOutputFormat, CliStrategy};
use fastdna_core::cohort;
use fastdna_core::error::{FastDnaError, Result};
use fastdna_core::export;
use fastdna_core::fastq::{InputSpec, MultiSourceReader};
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

/// Rejects the run before any work happens when an output path would
/// overwrite the input: export runs after the input is fully read, so
/// without this guard `-o` pointed at the input silently replaces the
/// user's FASTQ with a counts table -- irreversible data loss.
fn guard_against_input_overwrite(args: &Cli) -> Result<()> {
    let outputs: [(&str, Option<&PathBuf>); 3] = [
        ("--output", Some(&args.output)),
        ("--qc", Some(&args.qc)),
        ("--histogram", args.histogram.as_ref()),
    ];
    for (flag, path) in outputs.into_iter() {
        let Some(path) = path else { continue };
        // Every input is checked, and the message names the specific one
        // that collided: with several inputs, "an input file" would leave
        // the user to work out which of eight lanes was about to be
        // destroyed.
        for input in &args.input {
            if fastdna_core::atomic::same_file(input, path) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "output paths",
                    reason: format!(
                        "{flag} points at the input file {} and would overwrite it",
                        input.display()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Probes every output path for writability before the counting run, so a
/// typo'd output directory fails in milliseconds instead of after hours.
fn preflight_outputs(args: &Cli) -> Result<()> {
    fastdna_core::atomic::preflight_writable(&args.output)?;
    fastdna_core::atomic::preflight_writable(&args.qc)?;
    if let Some(histogram) = &args.histogram {
        fastdna_core::atomic::preflight_writable(histogram)?;
    }
    Ok(())
}

/// The decompressed size of everything about to be read, for the strategy
/// estimate. `None` disables size-based estimation entirely (see
/// `MemoryPolicy::estimated_input_bytes`), which is the conservative
/// in-memory choice rather than a guess.
///
/// Sizes are summed across inputs, since that is what one run will hold.
/// Best-effort per file: a size that cannot be read (an unusual filesystem,
/// a named pipe that lies about its length) disables the estimate rather
/// than failing a whole run over a memory *prediction*. Standard input has
/// no size at all -- there is no way to know how much is coming down a pipe
/// -- so any run reading it falls back to in-memory, as documented.
fn estimate_total_input_bytes(inputs: &[InputSpec]) -> Option<u64> {
    let mut total: u64 = 0;
    for input in inputs {
        let InputSpec::File(path) = input else {
            return None;
        };
        let bytes = std::fs::metadata(path).ok()?.len();
        let is_gz = path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gz"));
        let expanded = if is_gz {
            (bytes as f64 * GZIP_FASTQ_EXPANSION_FACTOR) as u64
        } else {
            bytes
        };
        total = total.saturating_add(expanded);
    }
    Some(total)
}

/// Renders the input list for the banner: one path, or a count and the
/// paths, so a mistyped eighth lane is visible before the run starts.
fn format_inputs(inputs: &[InputSpec]) -> String {
    let names: Vec<String> = inputs.iter().map(|i| i.display_path().display().to_string()).collect();
    match names.len() {
        1 => names.join(""),
        n => format!("{n} inputs: {}", names.join(", ")),
    }
}

/// The `--paired-dir` mode: discover every sample under `dir` via
/// `cohort::discover_paired_samples`'s strict R1/R2 pairing (an unpaired
/// file or an empty directory is rejected there, loudly, before any
/// counting starts), then run one counting pass per sample, each writing
/// its own `<sample_id>.<format>` file into `--paired-output`.
///
/// Deliberately narrower than the single-run path in `run`: no QC JSON, no
/// histogram, no memory-policy auto-strategy banner per sample -- those all
/// keep their existing single-run semantics and are unused here. See
/// `cohort::batch::count_paired_samples`'s doc comment for the full
/// rationale; this function is thin CLI glue around it plus the console
/// output every other invocation already has.
fn run_paired_dir(args: &Cli, dir: &Path) -> Result<()> {
    // Guaranteed `Some` by `Cli::validate`, called by this function's only
    // caller before `paired_dir` is ever inspected.
    let output_dir = args.paired_output.as_ref().ok_or_else(|| FastDnaError::InvalidConfig {
        parameter: "--paired-output",
        reason: "required together with --paired-dir".to_string(),
    })?;

    println!("==================================================");
    println!(" FastDNA: Paired-End Cohort Counting (Rust)        ");
    println!("==================================================");
    println!("Sample directory: {}", dir.display());
    println!("Output directory: {}", output_dir.display());
    println!("k-mer Size:       {}", args.kmer_size);
    println!("Quality Cutoff:   Q >= {}", args.min_quality);
    if args.hpc {
        println!("Homopolymer Compression: on");
    }

    let default_config = PipelineConfig::default();
    let threads = args.threads.unwrap_or(default_config.num_threads);
    println!("Worker Threads:   {threads}");
    println!("--------------------------------------------------");

    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4,
        min_quality: args.min_quality,
        batch_size: 10_000,
        num_threads: threads,
        progress_interval: default_config.progress_interval,
        hpc: args.hpc,
    };

    let format = match args.paired_format {
        CliOutputFormat::Parquet => cohort::PairedOutputFormat::Parquet,
        CliOutputFormat::Csv => cohort::PairedOutputFormat::Csv,
    };

    let start_time = Instant::now();
    let results = cohort::count_paired_samples(
        dir,
        output_dir,
        format,
        &config,
        args.min_count,
        args.max_count,
        args.with_sequence,
    )?;
    let elapsed = start_time.elapsed().as_secs_f64();

    for r in &results {
        println!(
            "{}: {} reads, {} records written -> {}",
            r.sample_id,
            r.total_reads,
            r.records_written,
            r.output_path.display()
        );
    }
    println!("--------------------------------------------------");
    println!("Samples Counted: {} in {elapsed:.2}s", results.len());

    Ok(())
}

fn run(args: Cli) -> Result<()> {
    args.validate().map_err(|reason| FastDnaError::InvalidConfig {
        parameter: "count filters",
        reason,
    })?;

    // `--paired-dir` is a wholly separate mode: discover-then-loop instead
    // of a single run. `Cli::validate` above already guarantees `--input`
    // is empty and `--paired-output` is `Some` whenever `paired_dir` is
    // `Some`, so this branch owns none of the single-run guards below (they
    // read `args.input`/`args.output`, which are meaningless here).
    if let Some(dir) = args.paired_dir.clone() {
        return run_paired_dir(&args, &dir);
    }

    guard_against_input_overwrite(&args)?;
    preflight_outputs(&args)?;

    println!("==================================================");
    println!(" FastDNA: High-Performance Genomic Kernel (Rust)  ");
    println!("==================================================");
    let inputs: Vec<InputSpec> = args.input.iter().map(|p| InputSpec::from_arg(p)).collect();
    println!("Input:          {}", format_inputs(&inputs));
    println!("Output:         {}", args.output.display());
    println!("k-mer Size:     {}", args.kmer_size);
    println!("Quality Cutoff: Q >= {}", args.min_quality);
    if args.hpc {
        println!("Homopolymer Compression: on");
    }

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
        hpc: args.hpc,
    };

    let estimated_input_bytes = estimate_total_input_bytes(&inputs);

    let policy = MemoryPolicy {
        strategy: match args.strategy {
            CliStrategy::Auto => None,
            CliStrategy::Memory => Some(CountStrategy::InMemory),
            CliStrategy::Disk => Some(CountStrategy::Disk),
            CliStrategy::Binned => Some(CountStrategy::Binned),
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

    // A fallback label only: `MultiSourceReader` names the file it was
    // actually reading when something goes wrong, so this is used solely
    // for the degenerate case where it has not opened anything yet.
    let source_label = inputs
        .first()
        .map(|i| i.display_path())
        .unwrap_or_else(|| PathBuf::from("<inputs>"));

    // Files are opened lazily, one at a time, inside the producer thread:
    // a run over 200 lanes holds one file handle, not 200.
    let fastq_reader = MultiSourceReader::new(inputs);

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
        &source_label,
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

    // Case-insensitive: on Windows filesystems `OUT.PARQUET` is the same
    // file as `out.parquet`, and writing CSV bytes into it hands the
    // downstream Parquet reader a corrupt file.
    let wants_parquet = args
        .output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"));
    let records_written = if wants_parquet {
        export::export_parquet(
            &counter,
            &args.output,
            args.kmer_size,
            args.min_count,
            args.with_sequence,
        )
        .inspect_err(|_| pb_export.abandon())?
    } else {
        export::export_csv(&counter, &args.output, args.kmer_size, args.min_count, args.with_sequence)
            .inspect_err(|_| pb_export.abandon())?
    };

    let export_elapsed = export_start.elapsed().as_secs_f64();
    pb_export.finish_with_message(format!("Export completed in {export_elapsed:.2}s"));

    // Honour --qc and --histogram, which were previously parsed and ignored.
    qc.export_json(&args.qc)?;
    println!("QC report written to: {}", args.qc.display());

    if let Some(histogram_path) = &args.histogram {
        let format = match args.histogram_format {
            CliHistogramFormat::Csv => export::HistogramFormat::Csv,
            CliHistogramFormat::GenomeScope => export::HistogramFormat::GenomeScope,
        };
        export::export_histogram(&counter, histogram_path, format, args.histogram_max)?;
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
