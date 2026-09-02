// src/main.rs

// main.rs legitimately owns all console output for the CLI binary; the
// crate-wide clippy denials in Cargo.toml exist to keep the library core
// silent, not this entry point.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use fastdna_core::cli::{
    CardArgs, Cli, CliDistMetric, CliHistogramFormat, CliOutputFormat, CliStrategy, Command,
    CountArgs, DiffArgs, DistArgs, FilterArgs, IntersectArgs, MatrixArgs, PeekArgs, ProfileArgs,
    QueryArgs, SketchArgs, SpectrumArgs, UnionArgs,
};
use fastdna_core::cohort;
use fastdna_core::error::{FastDnaError, Result};
use fastdna_core::export;
use fastdna_core::fastq::{InputSpec, MultiSourceReader};
use fastdna_core::hll;
use fastdna_core::ktab::{self, KmerTable};
use fastdna_core::ntcard;
use fastdna_core::pipeline::{
    process_stream_parallel_with_policy, CountStrategy, MemoryPolicy, PipelineConfig,
};
use fastdna_core::preview;
use fastdna_core::progress::Progress;
use fastdna_core::read_filter;
use fastdna_core::read_profile;
use fastdna_core::setops;
use fastdna_core::sketch::GenomeSketch;

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
    let cli = Cli::parse();
    // No subcommand word (`cli.command == None`) and explicit `fastdna
    // count ...` both resolve to `run`, the original counting path --
    // `Cli::command`'s doc comment explains why `CountArgs` backs both
    // instead of two struct definitions that could drift apart.
    let outcome = match cli.command {
        None => run(cli.count),
        Some(Command::Count(args)) => run(args),
        Some(Command::Sketch(args)) => run_sketch(args),
        Some(Command::Dist(args)) => run_dist(args),
        Some(Command::Card(args)) => run_card(args),
        Some(Command::Peek(args)) => run_peek(args),
        Some(Command::Query(args)) => run_query(args),
        Some(Command::Union(args)) => run_union(args),
        Some(Command::Intersect(args)) => run_intersect(args),
        Some(Command::Diff(args)) => run_diff(args),
        Some(Command::Filter(args)) => run_filter(args),
        Some(Command::Matrix(args)) => run_matrix(args),
        Some(Command::Profile(args)) => run_profile(args),
        Some(Command::Spectrum(args)) => run_spectrum(args),
    };
    match outcome {
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
fn guard_against_input_overwrite(args: &CountArgs) -> Result<()> {
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
fn preflight_outputs(args: &CountArgs) -> Result<()> {
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
fn run_paired_dir(args: &CountArgs, dir: &Path) -> Result<()> {
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

fn run(args: CountArgs) -> Result<()> {
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

/// `fastdna sketch`: thin console-output wrapper around `sketch::
/// GenomeSketch::from_path` + `.save`, the same pair `fastdna.sketch()` +
/// `Sketch.save` already expose to Python (see `cli::SketchArgs`'s doc
/// comment).
fn run_sketch(args: SketchArgs) -> Result<()> {
    // Same "would this output clobber the input" guard `guard_against_
    // input_overwrite` runs for `count`, sized down to sketch's one input
    // and one output instead of a list of each.
    if fastdna_core::atomic::same_file(&args.input, &args.output) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--output",
            reason: format!(
                "points at the input file {} and would overwrite it",
                args.input.display()
            ),
        });
    }
    fastdna_core::atomic::preflight_writable(&args.output)?;

    println!("==================================================");
    println!(" FastDNA: MinHash Sketch                           ");
    println!("==================================================");
    println!("Input:       {}", args.input.display());
    println!("k-mer Size:  {}", args.kmer_size);
    println!("Sketch Size: {}", args.sketch_size);
    println!("--------------------------------------------------");

    let start = Instant::now();
    let sketch = GenomeSketch::from_path(&args.input, args.sketch_size, args.kmer_size)?;
    sketch.save(&args.output)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Distinct hashes kept: {} of {}", sketch.hashes.len(), sketch.sketch_size);
    println!("Sketch written to:    {} ({elapsed:.2}s)", args.output.display());

    Ok(())
}

/// Loads `path` as a previously saved sketch if it looks like one
/// (`.json`, case-insensitive -- the extension `GenomeSketch::save`
/// always writes), otherwise streams it as a FASTQ/FASTA file and builds a
/// fresh sketch with `k`/`sketch_size`. Shared by `run_dist` so a `fastdna
/// dist` call can mix saved sketches and raw files on the same command
/// line without the caller having to say which is which.
fn load_or_build_sketch(path: &Path, k: usize, sketch_size: usize) -> Result<GenomeSketch> {
    let looks_like_a_saved_sketch =
        path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
    if looks_like_a_saved_sketch {
        GenomeSketch::load(path)
    } else {
        GenomeSketch::from_path(path, sketch_size, k)
    }
}

/// `fastdna dist`: pairwise comparison across every input named by
/// `--input`, analogous to Python's `fastdna.compare_all` (see `cli::
/// DistArgs`'s doc comment for why `containment` reports both directions
/// of a pair while `jaccard`/`mash` report each pair once).
///
/// The status banner below goes to stderr, not stdout, unlike every other
/// subcommand's banner: `--output`-less `dist` writes its *data* (the CSV
/// table) to stdout specifically so it composes with a shell pipe, and a
/// banner mixed into that stream would corrupt it for any consumer that
/// does not already know to skip a fixed number of header lines.
fn run_dist(args: DistArgs) -> Result<()> {
    eprintln!("==================================================");
    eprintln!(" FastDNA: Pairwise Distance                        ");
    eprintln!("==================================================");
    eprintln!("Inputs: {}", args.input.len());
    eprintln!("Metric: {:?}", args.metric);
    eprintln!("--------------------------------------------------");

    // Each input is loaded or sketched exactly once here, then compared
    // in memory `n*(n-1)/2` (or `n*(n-1)` for containment) times -- the
    // same "sketch once, compare many" shape `compare_all` uses in Python,
    // for the same reason: re-reading a FASTQ file per comparison would
    // make an `n`-sample run cost O(n^2) file reads instead of O(n).
    let mut labeled_sketches: Vec<(String, GenomeSketch)> = Vec::with_capacity(args.input.len());
    for path in &args.input {
        let sketch = load_or_build_sketch(path, args.kmer_size, args.sketch_size)?;
        labeled_sketches.push((path.display().to_string(), sketch));
    }

    let metric_column = match args.metric {
        CliDistMetric::Jaccard => "jaccard",
        CliDistMetric::Containment => "containment",
        CliDistMetric::MashDistance => "mash_distance",
    };

    let mut rows: Vec<(&str, &str, f64)> = Vec::new();
    if matches!(args.metric, CliDistMetric::Containment) {
        // Asymmetric: every ordered pair (i, j) with i != j is its own
        // question ("what fraction of i's k-mers are in j"), so both
        // directions are reported rather than picking one arbitrarily.
        for (i, (name_i, sketch_i)) in labeled_sketches.iter().enumerate() {
            for (j, (name_j, sketch_j)) in labeled_sketches.iter().enumerate() {
                if i == j {
                    continue;
                }
                rows.push((name_i, name_j, sketch_i.containment(sketch_j)?));
            }
        }
    } else {
        // Symmetric: one row per unordered pair.
        for i in 0..labeled_sketches.len() {
            for j in (i + 1)..labeled_sketches.len() {
                let (name_i, sketch_i) = &labeled_sketches[i];
                let (name_j, sketch_j) = &labeled_sketches[j];
                // Only Jaccard/MashDistance reach this branch (Containment
                // was handled and `continue`d past above), so a two-way
                // check suffices without an unreachable third arm.
                let value = if matches!(args.metric, CliDistMetric::MashDistance) {
                    sketch_i.mash_distance(sketch_j)?
                } else {
                    sketch_i.jaccard(sketch_j)?
                };
                rows.push((name_i, name_j, value));
            }
        }
    }

    match &args.output {
        Some(path) => {
            // Write-to-temp-then-rename, the same discipline every other
            // writer in this crate follows (`atomic.rs`; see `sketch::
            // GenomeSketch::save` for the pattern this mirrors) -- a failed
            // write must not leave a truncated table where a good one used
            // to be.
            let (file, pending) = fastdna_core::atomic::AtomicFile::create(path)?;
            let mut writer = BufWriter::new(file);
            write_dist_table(&mut writer, metric_column, &rows)
                .map_err(|e| FastDnaError::Io { path: path.clone(), source: e })?;
            writer.flush().map_err(|e| FastDnaError::Io { path: path.clone(), source: e })?;
            drop(writer);
            pending.commit()?;
            eprintln!("Distance table written to: {}", path.display());
        }
        None => {
            // No file named: the table is the point of the call, so it
            // goes to stdout rather than being thrown away, making
            // `fastdna dist ... | column -s, -t` and similar pipes work
            // out of the box.
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            write_dist_table(&mut handle, metric_column, &rows)
                .map_err(|e| FastDnaError::Internal { detail: e.to_string() })?;
        }
    }

    Ok(())
}

/// Writes the `sample_a,sample_b,<metric>` CSV body shared by `run_dist`'s
/// file and stdout paths, so the two destinations can never format a row
/// differently from each other.
fn write_dist_table<W: Write>(w: &mut W, metric_column: &str, rows: &[(&str, &str, f64)]) -> std::io::Result<()> {
    writeln!(w, "sample_a,sample_b,{metric_column}")?;
    for (a, b, value) in rows {
        writeln!(w, "{a},{b},{value}")?;
    }
    Ok(())
}

/// `fastdna card`: thin console-output wrapper around `hll::
/// estimate_cardinality` (see `cli::CardArgs`'s doc comment).
fn run_card(args: CardArgs) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: Cardinality Estimate (HyperLogLog)       ");
    println!("==================================================");
    println!("Input:      {}", args.input.display());
    println!("k-mer Size: {}", args.kmer_size);
    println!("Precision:  {} ({} bytes)", args.precision, 1usize << args.precision);
    println!("--------------------------------------------------");

    let start = Instant::now();
    let estimate = hll::estimate_cardinality(&args.input, args.kmer_size, args.precision)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Estimated distinct k-mers: {estimate:.0}");
    println!("Elapsed: {elapsed:.2}s");

    Ok(())
}

/// `fastdna peek`: thin console-output wrapper around `preview::peek` (see
/// `cli::PeekArgs`'s doc comment).
fn run_peek(args: PeekArgs) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: Preview                                  ");
    println!("==================================================");
    println!("Input: {}", args.input.display());
    println!("--------------------------------------------------");

    let stats = preview::peek(&args.input, args.n_reads)?;
    let suggested_k = stats.suggest_k();

    println!("Reads sampled:                 {}", stats.n_reads_sampled);
    println!(
        "Read length (min/median/max):  {}/{}/{}",
        stats.read_length.0, stats.read_length.1, stats.read_length.2
    );
    println!("GC content:                    {:.2}%", stats.gc_content * 100.0);
    println!("Distinct k-mers in sample:     {}", stats.sample_distinct_kmers);
    println!("Suggested k:                   {suggested_k}");

    Ok(())
}

/// `fastdna query`: thin console-output wrapper around `ktab::KmerTable`
/// (see `cli::QueryArgs`'s doc comment). Opens the table, encodes `--kmer`
/// against the table's own `k` (`ktab::encode_query_kmer`), and reports
/// whether it was found -- exit code and stdout only, no file written.
fn run_query(args: QueryArgs) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: K-mer Table Query                        ");
    println!("==================================================");
    println!("Table: {}", args.table.display());

    let table = KmerTable::open(&args.table)?;
    println!("k:     {}", table.k());
    println!("Rows:  {}", table.len());
    println!("--------------------------------------------------");

    let encoded = ktab::encode_query_kmer(&args.kmer, table.k())?;
    match table.get(encoded)? {
        Some(count) => println!("{}: found, count = {count}", args.kmer),
        None => println!("{}: not found in table", args.kmer),
    }

    Ok(())
}

/// Rejects a set operation whose `--output` would overwrite one of its own
/// input tables -- the same "would this write clobber something the run
/// still needs to read" guard `guard_against_input_overwrite` runs for
/// `count`, and for the same reason: a set op's inputs are read lazily,
/// row group by row group, over the whole run, so a truncated output file
/// landing on top of a live input would corrupt the very read still in
/// progress.
fn guard_against_setops_output_overwrite(inputs: &[PathBuf], output: &Path) -> Result<()> {
    for input in inputs {
        if fastdna_core::atomic::same_file(input, output) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "--output",
                reason: format!("points at the input table {} and would overwrite it", input.display()),
            });
        }
    }
    Ok(())
}

/// Opens every path in `paths` as a `KmerTable`, in order -- shared by
/// `run_union`/`run_intersect`/`run_diff` so a bad path is reported the
/// same way (`ktab::KmerTable::open`'s own `Load`/`Io` errors) regardless
/// of which subcommand hit it.
fn open_tables(paths: &[PathBuf]) -> Result<Vec<KmerTable>> {
    paths.iter().map(KmerTable::open).collect()
}

/// `fastdna union`: thin console-output wrapper around `setops::union`
/// (see `cli::UnionArgs`'s doc comment).
fn run_union(args: UnionArgs) -> Result<()> {
    guard_against_setops_output_overwrite(&args.input, &args.output)?;

    println!("==================================================");
    println!(" FastDNA: K-mer Table Union                        ");
    println!("==================================================");
    println!("Inputs:  {}", args.input.len());
    println!("Combine: {:?}", args.combine);
    println!("--------------------------------------------------");

    let tables = open_tables(&args.input)?;
    // Guaranteed non-empty by `--input`'s `num_args = 2..`, so `tables[0]`
    // cannot panic here.
    let k = tables[0].k();

    let start = Instant::now();
    let rows = setops::union(&tables, args.combine.into())?;
    let written = export::export_pairs_parquet(rows, &args.output, k)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Distinct k-mers written: {written} ({elapsed:.2}s)");
    println!("Output: {}", args.output.display());

    Ok(())
}

/// `fastdna intersect`: thin console-output wrapper around `setops::
/// intersect` (see `cli::IntersectArgs`'s doc comment).
fn run_intersect(args: IntersectArgs) -> Result<()> {
    guard_against_setops_output_overwrite(&args.input, &args.output)?;

    println!("==================================================");
    println!(" FastDNA: K-mer Table Intersection                 ");
    println!("==================================================");
    println!("Inputs:  {}", args.input.len());
    println!("Combine: {:?}", args.combine);
    println!("--------------------------------------------------");

    let tables = open_tables(&args.input)?;
    let k = tables[0].k();

    let start = Instant::now();
    let rows = setops::intersect(&tables, args.combine.into())?;
    let written = export::export_pairs_parquet(rows, &args.output, k)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Distinct k-mers written: {written} ({elapsed:.2}s)");
    println!("Output: {}", args.output.display());

    Ok(())
}

/// `fastdna diff`: thin console-output wrapper around `setops::diff` (see
/// `cli::DiffArgs`'s doc comment).
fn run_diff(args: DiffArgs) -> Result<()> {
    let mut all_inputs = args.subtract.clone();
    all_inputs.push(args.input.clone());
    guard_against_setops_output_overwrite(&all_inputs, &args.output)?;

    println!("==================================================");
    println!(" FastDNA: K-mer Table Difference                   ");
    println!("==================================================");
    println!("Input:               {}", args.input.display());
    println!("Subtract:            {} table(s)", args.subtract.len());
    println!("Max subtract count:  {}", args.max_subtract_count);
    println!("--------------------------------------------------");

    let a = KmerTable::open(&args.input)?;
    let subtract = open_tables(&args.subtract)?;
    let k = a.k();

    let start = Instant::now();
    let rows = setops::diff(&a, &subtract, args.max_subtract_count)?;
    let written = export::export_pairs_parquet(rows, &args.output, k)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Distinct k-mers written: {written} ({elapsed:.2}s)");
    println!("Output: {}", args.output.display());

    Ok(())
}

/// `fastdna filter`: thin console-output wrapper around `read_filter::
/// run_filter`/`run_filter_paired` (see `cli::FilterArgs`'s doc comment for
/// the full design). Dispatches to the paired-end path when `--input2`/
/// `--output2` were given (`FilterArgs::validate` has already confirmed
/// they were given together, or not at all); otherwise this is the
/// original single-end path, unchanged.
fn run_filter(args: FilterArgs) -> Result<()> {
    args.validate().map_err(|reason| FastDnaError::InvalidConfig { parameter: "filter", reason })?;

    if args.is_paired() {
        return run_filter_paired(args);
    }

    // Same "would this write clobber something the run still needs to
    // read" guard every other subcommand runs: neither the reference table
    // nor any input file may be the output.
    if fastdna_core::atomic::same_file(&args.table, &args.output) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--output",
            reason: format!(
                "points at the reference table {} and would overwrite it",
                args.table.display()
            ),
        });
    }
    for input in &args.input {
        if fastdna_core::atomic::same_file(input, &args.output) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "--output",
                reason: format!("points at the input file {} and would overwrite it", input.display()),
            });
        }
    }
    fastdna_core::atomic::preflight_writable(&args.output)?;

    let inputs: Vec<InputSpec> = args.input.iter().map(|p| InputSpec::from_arg(p)).collect();

    println!("==================================================");
    println!(" FastDNA: Read Filtering                           ");
    println!("==================================================");
    println!("Input:        {}", format_inputs(&inputs));
    println!("Table:        {}", args.table.display());
    println!("Mode:         {:?}", args.mode);
    println!("Min fraction: {}", args.min_fraction);
    println!("--------------------------------------------------");

    let table = KmerTable::open(&args.table)?;
    println!("Reference k:  {}", table.k());
    println!("Reference kmers: {}", table.len());
    let index = read_filter::ReferenceIndex::from_table(&table)?;

    let pb = spinner("Filtering reads...");
    let start = Instant::now();
    let stats = read_filter::run_filter(inputs, &index, args.mode.into(), args.min_fraction, &args.output)
        .inspect_err(|_| pb.abandon())?;
    let elapsed = start.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Filtering completed in {elapsed:.2}s"));

    println!("--------------------------------------------------");
    println!("Reads read:    {}", stats.reads_total);
    println!("Reads written: {}", stats.reads_written);
    println!("Output: {}", args.output.display());

    Ok(())
}

/// The paired-end branch of `fastdna filter` (`--input2`/`--output2` both
/// given). Mirrors `run_filter`'s single-end body: the same fast,
/// pre-header overwrite guard (extended to both R1/R2 inputs and both
/// outputs, plus the pair-specific "`--output` and `--output2` must not be
/// the same file" check `read_filter::run_filter_paired` also makes
/// authoritatively -- see that function's own doc comment for why both
/// copies exist), then `read_filter::run_filter_paired` itself.
fn run_filter_paired(args: FilterArgs) -> Result<()> {
    // `FilterArgs::validate` (already run by the caller) guarantees
    // `--output2` is `Some` whenever `is_paired()` is true; this match
    // keeps that invariant enforced without panicking if it is ever
    // violated by a caller that skips `validate` (e.g. a future direct
    // construction of `FilterArgs`).
    let output2 = match &args.output2 {
        Some(path) => path.clone(),
        None => {
            return Err(FastDnaError::InvalidConfig {
                parameter: "--output2",
                reason: "required for paired-end filtering".to_string(),
            })
        }
    };

    if fastdna_core::atomic::same_file(&args.output, &output2) {
        return Err(FastDnaError::InvalidConfig {
            parameter: "--output2",
            reason: format!(
                "output ({}) and output2 ({}) resolve to the same file: writing both mates of a \
                 pair to one destination would leave it holding only whichever mate's write \
                 committed last",
                args.output.display(),
                output2.display()
            ),
        });
    }
    for output in [&args.output, &output2] {
        if fastdna_core::atomic::same_file(&args.table, output) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "--output",
                reason: format!(
                    "points at the reference table {} and would overwrite it",
                    args.table.display()
                ),
            });
        }
        for input in args.input.iter().chain(args.input2.iter()) {
            if fastdna_core::atomic::same_file(input, output) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "--output",
                    reason: format!(
                        "points at the input file {} and would overwrite it",
                        input.display()
                    ),
                });
            }
        }
    }
    fastdna_core::atomic::preflight_writable(&args.output)?;
    fastdna_core::atomic::preflight_writable(&output2)?;

    let inputs_r1: Vec<InputSpec> = args.input.iter().map(|p| InputSpec::from_arg(p)).collect();
    let inputs_r2: Vec<InputSpec> = args.input2.iter().map(|p| InputSpec::from_arg(p)).collect();

    println!("==================================================");
    println!(" FastDNA: Paired-End Read Filtering                 ");
    println!("==================================================");
    println!("R1 input:     {}", format_inputs(&inputs_r1));
    println!("R2 input:     {}", format_inputs(&inputs_r2));
    println!("Table:        {}", args.table.display());
    println!("Mode:         {:?}", args.mode);
    println!("Min fraction: {}", args.min_fraction);
    println!("--------------------------------------------------");

    let table = KmerTable::open(&args.table)?;
    println!("Reference k:  {}", table.k());
    println!("Reference kmers: {}", table.len());
    let index = read_filter::ReferenceIndex::from_table(&table)?;

    let pb = spinner("Filtering read pairs...");
    let start = Instant::now();
    let stats = read_filter::run_filter_paired(
        inputs_r1,
        inputs_r2,
        &index,
        args.mode.into(),
        args.min_fraction,
        &args.output,
        &output2,
    )
    .inspect_err(|_| pb.abandon())?;
    let elapsed = start.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Filtering completed in {elapsed:.2}s"));

    println!("--------------------------------------------------");
    println!("Pairs read:    {}", stats.pairs_total);
    println!("Pairs written: {}", stats.pairs_written);
    println!("Output R1: {}", args.output.display());
    println!("Output R2: {}", output2.display());

    Ok(())
}

/// Derives a sample id from a file name for `fastdna matrix --sample`,
/// matching `python/fastdna/gwas.py::_sample_id_from_path`'s convention
/// exactly (a trailing `.gz` stripped first, then the remaining extension)
/// so a sample id computed here and one computed there agree for the same
/// file -- e.g. a cohort built once with this CLI verb and cross-checked
/// once against `gwas.py` names the same sample the same way.
fn sample_id_from_path(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let without_gz = if name.to_ascii_lowercase().ends_with(".gz") {
        name[..name.len() - 3].to_string()
    } else {
        name
    };
    match Path::new(&without_gz).file_stem() {
        Some(stem) => stem.to_string_lossy().into_owned(),
        None => without_gz,
    }
}

/// `fastdna matrix`: thin console-output wrapper around `cohort::matrix::
/// build_cohort_matrix_from_directory`/`build_cohort_matrix_from_files` plus
/// `export::export_cohort_matrix_parquet` (see `cli::MatrixArgs`'s doc
/// comment for the two ways to name a cohort).
fn run_matrix(args: MatrixArgs) -> Result<()> {
    args.validate().map_err(|reason| FastDnaError::InvalidConfig { parameter: "matrix samples", reason })?;

    // Same "would this write clobber something the run still needs to
    // read" guard every other subcommand runs.
    for input in &args.sample {
        if fastdna_core::atomic::same_file(input, &args.output) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "--output",
                reason: format!(
                    "points at the sample file {} and would overwrite it",
                    input.display()
                ),
            });
        }
    }
    fastdna_core::atomic::preflight_writable(&args.output)?;

    println!("==================================================");
    println!(" FastDNA: Cohort K-mer Matrix                      ");
    println!("==================================================");

    let default_config = PipelineConfig::default();
    let threads = args.threads.unwrap_or(default_config.num_threads);
    let config = PipelineConfig {
        k: args.kmer_size,
        quality_window: 4,
        min_quality: args.min_quality,
        batch_size: 10_000,
        num_threads: threads,
        progress_interval: default_config.progress_interval,
        hpc: false,
    };

    let pb = spinner("Counting cohort samples...");
    let start = Instant::now();

    let (matrix, sample_ids) = if let Some(dir) = &args.input {
        pb.set_message(format!("Discovering and counting samples in {}...", dir.display()));
        let (matrix, sample_ids, orphan_warnings) = cohort::build_cohort_matrix_from_directory(
            dir,
            &config,
            args.min_count,
            args.min_samples,
            args.max_kmers,
        )
        .inspect_err(|_| pb.abandon())?;
        for warning in &orphan_warnings {
            pb.println(format!("warning: {warning}"));
        }
        (matrix, sample_ids)
    } else {
        pb.set_message(format!("Counting {} explicitly named sample(s)...", args.sample.len()));
        let sample_ids: Vec<String> = args.sample.iter().map(|p| sample_id_from_path(p)).collect();
        let matrix = cohort::build_cohort_matrix_from_files(
            &sample_ids,
            &args.sample,
            &config,
            args.min_count,
            args.min_samples,
            args.max_kmers,
        )
        .inspect_err(|_| pb.abandon())?;
        (matrix, sample_ids)
    };

    let elapsed = start.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Counted {} sample(s) in {elapsed:.2}s", matrix.n_samples));

    if matrix.n_candidates > matrix.n_kmers {
        println!(
            "Truncated: {} k-mers passed --min-samples, {} dropped by --max-kmers (every dropped \
             k-mer had a minor-sample-count of {} or lower)",
            matrix.n_candidates,
            matrix.n_candidates - matrix.n_kmers,
            matrix.truncation_cutoff.unwrap_or(0)
        );
    }

    let written = export::export_cohort_matrix_parquet(
        &matrix,
        &sample_ids,
        &args.output,
        args.kmer_size,
        args.with_sequence,
    )?;

    println!("--------------------------------------------------");
    println!(
        "Samples: {} | k-mers (columns): {} | nonzero entries written: {written}",
        matrix.n_samples, matrix.n_kmers
    );
    println!("Output: {}", args.output.display());

    Ok(())
}

/// `fastdna spectrum`: thin console-output wrapper around `ntcard::
/// estimate_spectrum` + `ntcard::write_spectrum` (see `cli::SpectrumArgs`'s
/// doc comment, and `src/ntcard.rs`'s module doc comment for the algorithm
/// and its measured accuracy).
fn run_spectrum(args: SpectrumArgs) -> Result<()> {
    println!("==================================================");
    println!(" FastDNA: Streaming K-mer Spectrum (ntCard)        ");
    println!("==================================================");
    let inputs: Vec<InputSpec> = args.input.iter().map(|p| InputSpec::from_arg(p)).collect();
    println!("Input:      {}", format_inputs(&inputs));
    println!("k-mer Size: {}", args.kmer_size);
    println!("Precision:  {} ({} buckets)", args.precision, 1usize << args.precision);
    println!("--------------------------------------------------");

    let start = Instant::now();
    let estimate = ntcard::estimate_spectrum(&args.input, args.kmer_size, args.precision, args.max_frequency)?;
    let format = match args.format {
        CliHistogramFormat::Csv => export::HistogramFormat::Csv,
        CliHistogramFormat::GenomeScope => export::HistogramFormat::GenomeScope,
    };
    ntcard::write_spectrum(&estimate.spectrum, &args.output, format)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Estimated distinct k-mers: {:.0}", estimate.distinct_kmers);
    println!("Spectrum written to: {} ({elapsed:.2}s)", args.output.display());

    Ok(())
}

/// `fastdna profile`: thin console-output wrapper around `read_profile::
/// run_profile` (see `cli::ProfileArgs`'s doc comment for the full design).
fn run_profile(args: ProfileArgs) -> Result<()> {
    // Resolved once, before the guards: `--summary`'s default is a file
    // NAME placed beside `--output`, not a path relative to the working
    // directory (see `ProfileArgs::resolve_summary_path`). Everything
    // below -- the clobber guards, the preflight, the banner -- must see
    // the path that will actually be written.
    let summary = args.resolve_summary_path();

    // Same "would this write clobber something the run still needs to
    // read" guard every other subcommand runs: neither the reference table
    // nor any input file may be one of this run's own outputs.
    for guarded in [&args.output, &summary] {
        if fastdna_core::atomic::same_file(&args.table, guarded) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "output paths",
                reason: format!("points at the reference table {} and would overwrite it", args.table.display()),
            });
        }
        for input in &args.input {
            if fastdna_core::atomic::same_file(input, guarded) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "output paths",
                    reason: format!("points at the input file {} and would overwrite it", input.display()),
                });
            }
        }
    }
    fastdna_core::atomic::preflight_writable(&args.output)?;
    fastdna_core::atomic::preflight_writable(&summary)?;

    let inputs: Vec<InputSpec> = args.input.iter().map(|p| InputSpec::from_arg(p)).collect();

    println!("==================================================");
    println!(" FastDNA: Per-Read K-mer Profiles                  ");
    println!("==================================================");
    println!("Input:   {}", format_inputs(&inputs));
    println!("Table:   {}", args.table.display());
    println!("Profile: {}", args.output.display());
    println!("Summary: {}", summary.display());
    println!("--------------------------------------------------");

    let table = KmerTable::open(&args.table)?;
    println!("Reference k:     {}", table.k());
    println!("Reference kmers: {}", table.len());
    let index = read_profile::ProfileIndex::from_table(&table)?;

    let pb = spinner("Profiling reads...");
    let start = Instant::now();
    let stats = read_profile::run_profile(inputs, &index, &args.output, &summary).inspect_err(|_| pb.abandon())?;
    let elapsed = start.elapsed().as_secs_f64();
    pb.finish_with_message(format!("Profiling completed in {elapsed:.2}s"));

    println!("--------------------------------------------------");
    println!("Reads read:     {}", stats.reads_total);
    println!("Reads profiled: {}", stats.reads_profiled);
    println!("Profile output: {}", args.output.display());
    println!("Summary output: {}", summary.display());

    Ok(())
}
