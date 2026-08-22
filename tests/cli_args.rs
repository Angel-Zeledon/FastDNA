//! Argument-parsing contract. These run clap directly rather than spawning the
//! binary, so they stay fast and need no extra dev-dependencies.

use clap::Parser;
use fastdna::cli::Cli;

#[test]
fn max_count_defaults_to_no_upper_bound() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq"]);
    assert_eq!(cli.max_count, None, "absence of the flag must mean no cap");
}

#[test]
fn max_count_is_parsed_when_supplied() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq", "--max-count", "10000"]);
    assert_eq!(cli.max_count, Some(10_000));
}

#[test]
fn min_count_still_defaults_to_one() {
    let cli = Cli::parse_from(["fastdna", "--input", "sample.fastq"]);
    assert_eq!(cli.min_count, 1);
}

#[test]
fn existing_short_flags_are_unchanged() {
    let cli = Cli::parse_from([
        "fastdna", "--input", "s.fastq", "-k", "21", "-q", "30", "-m", "5",
    ]);
    assert_eq!(cli.kmer_size, 21);
    assert_eq!(cli.min_quality, 30.0);
    assert_eq!(cli.min_count, 5);
}
