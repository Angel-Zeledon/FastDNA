// src/cli.rs

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// User-facing choice of counting strategy, forwarded into
/// `pipeline::MemoryPolicy::strategy`. `Auto` (the default) leaves the
/// choice to `pipeline::resolve_strategy`'s memory estimate; `Memory` and
/// `Disk` force one strategy outright, for benchmarking and debugging (see
/// `pipeline::resolve_strategy`'s doc comment).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliStrategy {
    Auto,
    Memory,
    Disk,
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
    if value < 0.0 {
        return Err(format!("'{raw}' must not be negative"));
    }
    Ok((value * multiplier as f64) as u64)
}

#[derive(Parser, Debug)]
#[command(name = "fastdna", version = "0.1.0", author = "FastDNA Team")]
pub struct Cli {
    /// Input FASTQ file (.fastq or .fastq.gz)
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Output path for k-mer frequencies (.csv or .parquet)
    #[arg(short, long, value_name = "FILE", default_value = "kmer_counts.parquet")]
    pub output: PathBuf,

    /// Length of k-mers (1 <= k <= 32)
    #[arg(short, long, default_value_t = 31)]
    pub kmer_size: usize,

    /// Minimum Phred quality score cutoff (0-40)
    #[arg(short = 'q', long, default_value_t = 20.0)]
    pub min_quality: f64,

    /// Filter out k-mers with frequency below this cutoff
    #[arg(short = 'm', long, default_value_t = 1)]
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

    /// Optional path to export frequency spectrum (Histogram CSV)
    #[arg(long, value_name = "FILE")]
    pub histogram: Option<PathBuf>,

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
    /// peak memory versus --max-ram; "memory" and "disk" force one
    /// strategy outright, for benchmarking and debugging.
    #[arg(long, value_enum, default_value = "auto")]
    pub strategy: CliStrategy,
}
