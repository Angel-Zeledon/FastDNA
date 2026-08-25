//! Per-bin occupancy report for the binned strategy on a real FASTQ file.
//!
//! This is the R3 mitigation `docs/design-minimizer-counting.md` §6 and §7
//! call for: "report per-bin occupancy behind a debug flag from step 4, and
//! benchmark on at least one real SRA sample before step 6." It drives the
//! exact production types (`fastq::FastqReader`, `binned::BinStore`,
//! `binned::BinWriter::push_sequence`) that `pipeline.rs`'s worker loop
//! drives, single-threaded, so the occupancy numbers are what a real run
//! would actually produce -- not a reimplementation of the bin function.
//!
//! Usage: `cargo run --release --example binned_occupancy_report -- <path> [k]`

use std::env;
use std::process::ExitCode;

use fastdna_core::binned::{BinStore, BinnedConfig};
use fastdna_core::fastq::FastqReader;

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: binned_occupancy_report <fastq[.gz]> [k]");
        return ExitCode::FAILURE;
    };
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(31);

    let mut reader = match FastqReader::from_path(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let config = BinnedConfig::new(k);
    let store = BinStore::new(config);
    let mut writer = store.writer();

    let mut records_seen: u64 = 0;
    // Same quality trim the default CLI run applies (-q 20, default window),
    // so the occupancy numbers reflect what the counting workers actually
    // see, not raw untrimmed reads.
    let min_qual = 20.0;
    let quality_window = 1usize;

    loop {
        match reader.next_record() {
            Ok(Some(mut record)) => {
                record.quality_trim_end(min_qual, quality_window);
                writer.push_sequence(&store, &record.seq);
                records_seen += 1;
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("read error after {records_seen} records: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let total_occurrences = writer.occurrences();
    writer.finish(&store);

    let occupancy = store.occupancy();
    let total_bytes: usize = occupancy.iter().sum();
    let nonempty = occupancy.iter().filter(|&&b| b > 0).count();
    let mean = total_bytes as f64 / occupancy.len() as f64;
    let max = occupancy.iter().copied().max().unwrap_or(0);
    let min_nonzero = occupancy.iter().copied().filter(|&b| b > 0).min().unwrap_or(0);

    let mut sorted = occupancy.clone();
    sorted.sort_unstable();
    let p50 = sorted[sorted.len() / 2];
    let p90 = sorted[(sorted.len() * 9) / 10];
    let p99 = sorted[(sorted.len() * 99) / 100];

    println!("file: {path}");
    println!("k = {k}, m = {}, num_bins = {}", config.m, config.num_bins);
    println!("records: {records_seen}, k-mer occurrences: {total_occurrences}");
    println!("super-k-mer store: {total_bytes} bytes ({:.3} MiB)", total_bytes as f64 / (1024.0 * 1024.0));
    println!("bins occupied: {nonempty}/{}", occupancy.len());
    println!("bin bytes -- mean: {mean:.0}, min(nonzero): {min_nonzero}, p50: {p50}, p90: {p90}, p99: {p99}, max: {max}");
    println!("skew (max / mean): {:.2}x", max as f64 / mean);
    println!("skew (max / p50): {:.2}x", max as f64 / p50.max(1) as f64);

    ExitCode::SUCCESS
}
