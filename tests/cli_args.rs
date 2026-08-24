//! Argument-parsing contract. These run clap directly rather than spawning the
//! binary, so they stay fast and need no extra dev-dependencies.

#![allow(clippy::expect_used)]

use clap::Parser;
use fastdna_core::cli::Cli;

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

#[test]
fn non_finite_max_ram_is_rejected() {
    // f64::from_str happily parses "nan" and "inf"; "nan" used to become a
    // silent budget of 0 bytes, flipping the auto strategy to disk.
    for bad in ["nan", "inf", "NaN", "-inf"] {
        assert!(
            Cli::try_parse_from(["fastdna", "--input", "s.fastq", "--max-ram", bad]).is_err(),
            "--max-ram {bad} must be rejected"
        );
    }
}

#[test]
fn min_quality_beyond_the_phred_ceiling_is_rejected() {
    // Phred+33 tops out at Q93 ('~'). Anything above trims every base and
    // used to produce a silent, plausible-looking empty output.
    for bad in ["94", "200", "9000", "nan", "inf", "-1"] {
        assert!(
            Cli::try_parse_from(["fastdna", "--input", "s.fastq", "--min-quality", bad]).is_err(),
            "-q {bad} must be rejected"
        );
    }
}

#[test]
fn min_quality_within_the_phred_range_still_parses() {
    for good in ["0", "20", "41", "93"] {
        assert!(
            Cli::try_parse_from(["fastdna", "--input", "s.fastq", "--min-quality", good]).is_ok(),
            "-q {good} is a legitimate Phred cutoff"
        );
    }
}

#[test]
fn min_count_zero_is_rejected() {
    // Frequencies are always >= 1, so a cutoff of 0 is a meaningless no-op
    // that silently behaves like 1.
    assert!(
        Cli::try_parse_from(["fastdna", "--input", "s.fastq", "--min-count", "0"]).is_err()
    );
}

#[test]
fn contradictory_count_band_is_rejected_by_validate() {
    let cli = Cli::parse_from(["fastdna", "--input", "s.fastq", "-m", "5", "-M", "2"]);
    let err = cli.validate().expect_err("-M below -m keeps nothing and must be an error");
    assert!(err.contains("--max-count"), "message must name the flags: {err}");
    assert!(err.contains("--min-count"), "message must name the flags: {err}");
}

#[test]
fn equal_and_ordered_count_bands_pass_validate() {
    let equal = Cli::parse_from(["fastdna", "--input", "s.fastq", "-m", "5", "-M", "5"]);
    assert!(equal.validate().is_ok(), "-m 5 -M 5 keeps exactly frequency 5");
    let ordered = Cli::parse_from(["fastdna", "--input", "s.fastq", "-m", "2", "-M", "10"]);
    assert!(ordered.validate().is_ok());
    let unbounded = Cli::parse_from(["fastdna", "--input", "s.fastq", "-m", "5"]);
    assert!(unbounded.validate().is_ok(), "no -M means no upper bound");
}

#[test]
fn qc_and_histogram_paths_are_available_to_the_binary() {
    let cli = Cli::parse_from([
        "fastdna", "--input", "s.fastq",
        "--qc", "my_qc.json",
        "--histogram", "my_hist.csv",
    ]);
    assert_eq!(cli.qc.to_string_lossy(), "my_qc.json");
    assert_eq!(
        cli.histogram.as_ref().map(|p| p.to_string_lossy().to_string()),
        Some("my_hist.csv".to_string())
    );
}
