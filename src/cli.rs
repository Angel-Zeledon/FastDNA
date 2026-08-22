// src/cli.rs

use clap::Parser;
use std::path::PathBuf;

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

    /// Number of worker threads
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Path to export quality control summary JSON
    #[arg(long, value_name = "FILE", default_value = "qc_report.json")]
    pub qc: PathBuf,

    /// Optional path to export frequency spectrum (Histogram CSV)
    #[arg(long, value_name = "FILE")]
    pub histogram: Option<PathBuf>,
}