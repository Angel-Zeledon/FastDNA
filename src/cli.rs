// src/cli.rs

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// User-facing choice of counting strategy, forwarded into
/// `pipeline::MemoryPolicy::strategy`. `Auto` (the default) leaves the
/// choice to `pipeline::resolve_strategy`'s memory estimate; `Memory` and
/// `Disk` force one strategy outright, for benchmarking and debugging (see
/// `pipeline::resolve_strategy`'s doc comment).
///
/// `Binned` is the minimizer-partitioned super-k-mer strategy
/// (`binned.rs`). Unlike the other two it is **not** something `Auto` can
/// ever pick: naming it here, or setting `FASTDNA_STRATEGY=binned`, is the
/// only way to run it. Promoting it is a separate, later decision
/// (`docs/design-minimizer-counting.md` §5 step 6).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliStrategy {
    Auto,
    Memory,
    Disk,
    Binned,
}

/// User-facing choice of spectrum file format, forwarded into
/// `export::HistogramFormat`. A separate enum from that one so the CLI's
/// spelling of the values (`csv`, `genomescope`) is a CLI decision and the
/// library type stays free to be named for what it is.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliHistogramFormat {
    #[default]
    Csv,
    #[value(name = "genomescope")]
    GenomeScope,
}

/// Output file format for `--paired-dir`'s per-sample exports, forwarded
/// into `cohort::batch::PairedOutputFormat`. `--output`'s format is chosen
/// by the extension of its single file path (see `main.rs`'s `wants_parquet`);
/// a directory of per-sample files has no such extension to read, so
/// `--paired-dir` mode needs this explicit enum instead.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliOutputFormat {
    #[default]
    Parquet,
    Csv,
}

/// Parses a `--max-ram` value: a plain byte count, or a number with a
/// `K`/`M`/`G`/`T` suffix (case-insensitive, an optional trailing `B`
/// ignored -- `4G`, `4GB`, and `4gb` all mean the same 4*1024^3 bytes).
/// Binary (1024-based) units, matching `mem_estimate.rs`'s own units and
/// what `free -h`/Task Manager report, not decimal (1000-based) ones.
fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("must not be empty".to_string());
    }

    let upper = trimmed.to_ascii_uppercase();
    let without_b = upper.strip_suffix('B').unwrap_or(&upper);
    let (digits, multiplier): (&str, u64) = if let Some(d) = without_b.strip_suffix('K') {
        (d, 1024)
    } else if let Some(d) = without_b.strip_suffix('M') {
        (d, 1024 * 1024)
    } else if let Some(d) = without_b.strip_suffix('G') {
        (d, 1024 * 1024 * 1024)
    } else if let Some(d) = without_b.strip_suffix('T') {
        (d, 1024 * 1024 * 1024 * 1024)
    } else {
        (without_b, 1)
    };

    let value: f64 = digits.trim().parse().map_err(|_| format!("'{raw}' is not a valid size (examples: 4096, 512M, 4G)"))?;
    // f64::from_str accepts "nan" and "inf"; NaN would cast to a silent
    // budget of 0 bytes, flipping the auto strategy to disk for no reason.
    if !value.is_finite() {
        return Err(format!("'{raw}' is not a valid size (examples: 4096, 512M, 4G)"));
    }
    if value < 0.0 {
        return Err(format!("'{raw}' must not be negative"));
    }
    Ok((value * multiplier as f64) as u64)
}

/// Parses `--min-quality`: a finite Phred score within the Phred+33
/// printable range (0..=93). Anything above the ceiling would trim every
/// base of every read and produce a plausible-looking but empty output.
fn parse_min_quality(raw: &str) -> Result<f64, String> {
    let value: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("'{raw}' is not a valid Phred quality score"))?;
    if !value.is_finite() || !(0.0..=93.0).contains(&value) {
        return Err(format!(
            "'{raw}' is outside the Phred+33 range 0-93 (typical Illumina data uses 0-40)"
        ));
    }
    Ok(value)
}

#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin.
    /// Several may be given (R1/R2, extra lanes); their counts and QC are
    /// aggregated. The format of each is detected from its content, not its
    /// extension.
    ///
    /// Mutually exclusive with `--paired-dir`: a run either counts the files
    /// named here, or discovers and counts every sample in a directory (see
    /// `--paired-dir`'s doc comment). Not `required` at the derive level
    /// (unlike before `--paired-dir` existed) because that alternative can
    /// satisfy the "give me something to count" requirement instead; `Cli::
    /// validate` enforces that exactly one of the two is actually given.
    #[arg(short, long, value_name = "FILE", num_args = 1.., required = false)]
    pub input: Vec<PathBuf>,

    /// Output path for k-mer frequencies (.csv or .parquet)
    #[arg(short, long, value_name = "FILE", default_value = "kmer_counts.parquet")]
    pub output: PathBuf,

    /// Length of k-mers (1 <= k <= 32)
    #[arg(short, long, default_value_t = 31)]
    pub kmer_size: usize,

    /// Minimum Phred quality score cutoff (0-93; typical data uses 0-40)
    #[arg(short = 'q', long, default_value_t = 20.0, value_parser = parse_min_quality)]
    pub min_quality: f64,

    /// Filter out k-mers with frequency below this cutoff (>= 1)
    #[arg(short = 'm', long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub min_count: u32,

    /// Filter out k-mers with frequency above this cutoff (repetitive regions)
    #[arg(short = 'M', long, value_name = "COUNT")]
    pub max_count: Option<u32>,

    /// Number of worker threads
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Path to export quality control summary JSON
    #[arg(long, value_name = "FILE", default_value = "qc_report.json")]
    pub qc: PathBuf,

    /// Optional path to export the k-mer frequency spectrum
    #[arg(long, value_name = "FILE")]
    pub histogram: Option<PathBuf>,

    /// Spectrum file format. "csv" (the default) writes the named-column
    /// CSV that has always shipped; "genomescope" writes the headerless
    /// space-separated `depth count` that `jellyfish histo` emits and
    /// GenomeScope 2.0 consumes.
    #[arg(long, value_enum, default_value = "csv")]
    pub histogram_format: CliHistogramFormat,

    /// Cap the spectrum at this depth: everything deeper is summed into
    /// this row rather than dropped (KMC's -cx convention), keeping the
    /// total number of distinct k-mers intact.
    #[arg(long, value_name = "DEPTH", value_parser = clap::value_parser!(u32).range(1..))]
    pub histogram_max: Option<u32>,

    /// Maximum RAM the automatic strategy chooser will target before
    /// switching from the in-memory to the disk-partitioned counting
    /// strategy. Accepts a plain byte count or a size with a K/M/G/T
    /// suffix (e.g. "4G"). Defaults to half of currently available system
    /// memory, or a fixed 4GB if that cannot be determined on this
    /// platform (see `mem_estimate::default_max_ram_bytes`).
    #[arg(long, value_name = "SIZE", value_parser = parse_byte_size)]
    pub max_ram: Option<u64>,

    /// Force a specific counting strategy instead of letting the memory
    /// estimate choose. "auto" (the default) picks based on the estimated
    /// peak memory versus --max-ram; "memory", "disk" and "binned" force
    /// one strategy outright, for benchmarking and debugging. "binned" is
    /// the experimental minimizer-partitioned strategy and "auto" never
    /// selects it.
    #[arg(long, value_enum, default_value = "auto")]
    pub strategy: CliStrategy,

    /// Collapse homopolymer runs (e.g. "AAAAAA" -> "A") before extracting
    /// k-mers. Off by default: with the flag absent, output is byte-for-byte
    /// identical to today's. Turn it on for long-read input (Oxford Nanopore,
    /// PacBio), where the dominant sequencing error is an insertion or
    /// deletion inside a homopolymer run rather than a substitution -- left
    /// uncompressed, that single indel shifts and corrupts every k-mer
    /// downstream of it. Short-read Illumina data has no need for this.
    ///
    /// This trades exact base-level positional correspondence with the
    /// original read for robustness to those indels: downstream tools that
    /// map a k-mer back to a reference coordinate (e.g. `fastdna.annotate`)
    /// are working with compressed-sequence offsets, not the original
    /// read's.
    #[arg(long)]
    pub hpc: bool,

    /// Directory of FASTQ files to discover and count as paired-end (or
    /// single-end) samples, one counting run per sample, instead of naming
    /// files by hand with `--input`. Uses `cohort::discover_samples`'s R1/R2
    /// pairing (`_R1`/`_R2`, `_1`/`_2`, and Illumina's `_R1_001` form -- see
    /// that module's doc comment for the exact rules); a discovered sample's
    /// mate files are fed into one counting run together, exactly as
    /// multiple `--input` files already are.
    ///
    /// Unlike ad hoc cohort listing, a file that carries a recognized pair
    /// suffix with no matching mate is a hard error here, not a recorded
    /// warning: this mode exists specifically for unattended counting across
    /// many samples, and a half-paired directory silently falling back to a
    /// single-end sample would produce a normal-looking run over corrupted
    /// grouping. An empty (or all-non-FASTQ) directory is also an error, not
    /// an empty cohort.
    ///
    /// Requires `--paired-output`; mutually exclusive with `--input`.
    /// `--output` is ignored in this mode (see `--paired-output`).
    #[arg(long, value_name = "DIR")]
    pub paired_dir: Option<PathBuf>,

    /// Output directory for `--paired-dir` mode: each discovered sample
    /// writes its own `<sample_id>.<parquet|csv>` file here (see
    /// `--paired-format`), named from `discover_samples`'s `sample_id`.
    /// Created if it does not already exist. Required together with
    /// `--paired-dir`; rejected otherwise.
    #[arg(long, value_name = "DIR")]
    pub paired_output: Option<PathBuf>,

    /// File format for `--paired-dir`'s per-sample outputs. Ignored without
    /// `--paired-dir` (not rejected: unlike `--paired-output`, a clap
    /// `default_value` makes "the user explicitly asked for the default"
    /// and "the user never touched this flag" indistinguishable here, so
    /// `Cli::validate` cannot draw a reliable line to reject on).
    #[arg(long, value_enum, default_value = "parquet")]
    pub paired_format: CliOutputFormat,
}

impl Cli {
    /// Cross-flag validation clap cannot express declaratively. Called by
    /// `main` right after parsing; a distinct method so the argument-parsing
    /// tests can exercise it without spawning the binary.
    pub fn validate(&self) -> Result<(), String> {
        // stdin is a single unnamed stream: there is no meaningful order in
        // which to read it alongside named files, and accepting the
        // combination would silently read the pipe once and the files
        // normally -- not what the command line says. Rejected outright
        // rather than guessed at.
        let stdin_count = self
            .input
            .iter()
            .filter(|p| p.as_os_str() == crate::fastq::STDIN_ARG)
            .count();
        if stdin_count > 0 && self.input.len() > 1 {
            return Err(format!(
                "--input '-' means standard input and must be the only input; got {} inputs",
                self.input.len()
            ));
        }

        if let Some(max) = self.max_count {
            if max < self.min_count {
                return Err(format!(
                    "--max-count {max} is below --min-count {}: this band keeps no k-mer at all",
                    self.min_count
                ));
            }
        }

        // `--paired-dir` and `--input` name two different ways to say "here
        // is what to count" -- accepting both would leave no defined order
        // to run them in, and neither has a self-evident priority over the
        // other. Checked here, before the counting run, rather than letting
        // one of the two silently win.
        if self.paired_dir.is_some() && !self.input.is_empty() {
            return Err(
                "--input and --paired-dir are mutually exclusive: a run either counts the files \
                 named by --input, or discovers samples from a directory with --paired-dir, not both"
                    .to_string(),
            );
        }

        match &self.paired_dir {
            Some(_) => {
                // Without an output directory there is nowhere for N
                // per-sample files to go; unlike `--output`'s single-file
                // default, guessing one here (the cohort directory itself?
                // the current directory?) would silently place a cohort's
                // worth of output somewhere the user did not ask for.
                if self.paired_output.is_none() {
                    return Err(
                        "--paired-dir requires --paired-output, naming the directory each \
                         discovered sample's counts are written into"
                            .to_string(),
                    );
                }
            }
            None => {
                if self.paired_output.is_some() {
                    return Err(
                        "--paired-output is only meaningful together with --paired-dir".to_string(),
                    );
                }
                // Mirrors the check `fastq::MultiSourceReader::validate` already
                // makes deeper in the run, but stated here means it fires
                // before the startup banner prints or any file is touched,
                // not after -- "bad input fails fast, loudly, and before the
                // counting run".
                if self.input.is_empty() {
                    return Err(
                        "no input given: pass --input <FILE>... or --paired-dir <DIR>".to_string(),
                    );
                }
            }
        }

        Ok(())
    }
}
