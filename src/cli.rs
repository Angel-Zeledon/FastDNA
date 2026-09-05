// src/cli.rs

use clap::{Args, Parser, Subcommand, ValueEnum};
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

/// User-facing choice of pairwise metric for `fastdna dist`, mirroring
/// `fastdna.compare_all`'s Python `metric=` parameter plus `containment`,
/// which `compare_all` does not expose: that function's output shape is one
/// row per *unordered* pair, which only makes sense for a symmetric metric,
/// and containment is not symmetric (`A.containment(B) != B.containment(A)`
/// in general -- see `GenomeSketch::containment`'s doc comment). `DistArgs`
/// below reports both directions of every pair when this is `Containment`,
/// and one row per pair otherwise.
///
/// The clap spelling `"mash"` (not `"mash-distance"`) matches this task's
/// own specification; the Rust variant is still named after what it
/// actually computes, `GenomeSketch::mash_distance`, the same
/// spelling-vs-meaning split `CliHistogramFormat::GenomeScope` already
/// draws between its clap value name and its variant name.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliDistMetric {
    #[default]
    Jaccard,
    Containment,
    #[value(name = "mash")]
    MashDistance,
}

/// How several tables' individual frequencies for the same k-mer are folded
/// into the single `frequency` column a set operation's output has room
/// for -- the CLI's spelling of `setops::CombineOp` (see that enum's own
/// doc comment for what each value means and which operation defaults to
/// which). A separate enum from the library type for the same reason
/// `CliHistogramFormat`/`CliDistMetric` are already separate from their own
/// library counterparts: the CLI's value spelling is a CLI decision.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliCombineOp {
    Sum,
    Min,
    Max,
}

impl From<CliCombineOp> for crate::setops::CombineOp {
    fn from(value: CliCombineOp) -> Self {
        match value {
            CliCombineOp::Sum => crate::setops::CombineOp::Sum,
            CliCombineOp::Min => crate::setops::CombineOp::Min,
            CliCombineOp::Max => crate::setops::CombineOp::Max,
        }
    }
}

/// Which reads `fastdna filter` writes to its output -- the CLI's spelling
/// of `read_filter::FilterMode` (see that enum's own doc comment for what
/// each value means). Separate from the library type for the same reason
/// every other `Cli*` enum in this file is: the CLI's value spelling is a
/// CLI decision.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliFilterMode {
    Keep,
    Discard,
}

impl From<CliFilterMode> for crate::read_filter::FilterMode {
    fn from(value: CliFilterMode) -> Self {
        match value {
            CliFilterMode::Keep => crate::read_filter::FilterMode::Keep,
            CliFilterMode::Discard => crate::read_filter::FilterMode::Discard,
        }
    }
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

/// Parses `fastdna filter`'s `--min-fraction`: a finite fraction in
/// `0.0..=1.0`. Mirrors `read_filter::validate_min_fraction`'s own check
/// (that one runs again inside `read_filter::run_filter`, since a Python
/// caller reaches that function without going through this parser at all)
/// so a bad value on the command line is rejected immediately, before any
/// file is opened, with the same reasoning `parse_min_quality` above
/// already gives for its own range: a `NaN` or out-of-range fraction can
/// never be crossed (or is always crossed) by a genuine computed fraction,
/// silently keeping nothing or discarding nothing.
fn parse_min_fraction(raw: &str) -> Result<f64, String> {
    let value: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("'{raw}' is not a valid fraction (expected a number between 0.0 and 1.0)"))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("'{raw}' must be between 0.0 and 1.0 inclusive"));
    }
    Ok(value)
}

// `version` comes from Cargo.toml, not a literal. `pyproject.toml` already
// documents that the version lives in one place -- "the version stays
// sourced from Cargo.toml's [package] version alone ... rather than
// duplicating it" -- and this line was exactly the duplication that
// paragraph considered already avoided: `ffi.rs` uses
// env!("CARGO_PKG_VERSION") in both of its own sites while the CLI carried
// "0.1.0" written by hand, so the first version bump would have made
// `fastdna --version` and `fastdna.__version__` disagree.
//
// `author` is removed rather than fixed: Cargo.toml declares no `authors`,
// so there is nothing to read it from, and "FastDNA Team" is not an entity
// that exists. clap omits the field when it is not given.
//
// `args_conflicts_with_subcommands`: this is what lets `command` (below)
// and `count` (the flattened counting flags) coexist on one struct without
// ambiguity. Its effect (see `clap::Command::args_conflicts_with_
// subcommands`'s own doc comment) is that flags may only follow the
// *final* subcommand on the line -- `fastdna [count-flags]` or
// `fastdna <mode> [mode-flags]`, never both mixed -- which is exactly the
// two shapes this CLI needs and no more.
#[derive(Parser, Debug)]
#[command(name = "fastdna", version = env!("CARGO_PKG_VERSION"), args_conflicts_with_subcommands = true)]
pub struct Cli {
    /// Which mode to run. Absent -- no subcommand word at all -- means
    /// k-mer counting: every invocation written before this CLI grew
    /// subcommands stays valid unchanged, which is the compatibility
    /// contract `CHANGELOG.md` states for "the flags and subcommands
    /// documented in `fastdna --help`". Explicit `fastdna count ...` does
    /// exactly the same thing and exists for symmetry with the other four
    /// modes, and for scripts/pipelines that prefer to name the mode
    /// outright rather than rely on an implicit default.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// The counting flags, used when `command` is `None` -- i.e. every
    /// invocation that does not start with a subcommand word. Kept as its
    /// own `CountArgs` type (not inlined here) so the exact same field
    /// definitions back both this default path and the explicit
    /// `Command::Count` variant below with one source of truth, instead of
    /// two struct definitions that could silently drift apart.
    #[command(flatten)]
    pub count: CountArgs,
}

/// The five things `fastdna` can do. `Count` is listed first and is the
/// only one with a direct non-subcommand equivalent (see `Cli::command`'s
/// doc comment); the other four wrap Rust-core functionality
/// (`sketch.rs`, `hll.rs`, `preview.rs`) that previously was reachable only
/// from the Python binding (`ffi.rs`) -- pipeline users who never leave the
/// shell had no way to sketch a genome, estimate its cardinality, or peek
/// at a file's read geometry without writing Python.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Count k-mers in one or more FASTQ/FASTA files (identical to giving
    /// no subcommand at all).
    Count(CountArgs),
    /// Build a MinHash sketch of one file and save it for later comparison.
    Sketch(SketchArgs),
    /// Compare several sketches and/or files pairwise (Jaccard,
    /// containment, or Mash distance).
    Dist(DistArgs),
    /// Estimate the number of distinct k-mers in a file via HyperLogLog.
    Card(CardArgs),
    /// Preview a file's read geometry and composition without reading all
    /// of it.
    Peek(PeekArgs),
    /// Point-lookup a single k-mer's count in a sorted k-mer table (the
    /// Parquet file `count` already writes -- see `docs/feature-gap-
    /// analysis.md`'s S1 and `src/ktab.rs`'s module doc comment).
    Query(QueryArgs),
    /// Union of two or more k-mer tables: every k-mer present in any of
    /// them, with counts summed by default (`docs/feature-gap-analysis.md`'s
    /// S2, `src/setops.rs`).
    Union(UnionArgs),
    /// Intersection of two or more k-mer tables: only k-mers present in
    /// *every* one of them (`docs/feature-gap-analysis.md`'s S2,
    /// `src/setops.rs`).
    Intersect(IntersectArgs),
    /// Asymmetric difference: k-mers in one table absent (or below a
    /// tolerance) from one or more others -- reference subtraction / host
    /// removal (`docs/feature-gap-analysis.md`'s S2, `src/setops.rs`).
    Diff(DiffArgs),
    /// Streams a FASTQ/FASTA input against a reference k-mer table and
    /// writes reads that should be kept (host/contaminant removal, or
    /// targeted enrichment -- see `FilterArgs`'s doc comment for the full
    /// design) (`docs/feature-gap-analysis.md`'s S4, `src/read_filter.rs`).
    Filter(FilterArgs),
    /// Builds a cohort-wide k-mer presence/count matrix from several
    /// samples' FASTQ files and writes it as a Parquet file, independent of
    /// the Python GWAS module (`docs/feature-gap-analysis.md`'s S6,
    /// `src/cohort/matrix.rs`, `src/export.rs::export_cohort_matrix_
    /// parquet`).
    Matrix(MatrixArgs),
    /// Per-read k-mer profiles against a reference k-mer table: for each
    /// read, the RLE-compressed vector of counts its own canonical k-mers
    /// have in the reference, plus a per-read summary table
    /// (`docs/feature-gap-analysis.md`'s S3, `src/read_profile.rs`).
    Profile(ProfileArgs),
    /// Estimates the k-mer frequency spectrum (how many distinct k-mers
    /// occur exactly once, twice, ...) in one streaming pass and bounded
    /// memory, via an ntCard-style sketch (`docs/feature-gap-analysis.md`'s
    /// S7(a), `src/ntcard.rs`) -- the same question `count --histogram`
    /// answers exactly, approximated cheaply enough to run on input too
    /// large to count in full.
    Spectrum(SpectrumArgs),
}

/// The k-mer counting flags -- today's only behavior, and still the
/// default one (see `Cli::command`'s doc comment for how `command: None`
/// and `Command::Count` both resolve here).
#[derive(Args, Debug)]
pub struct CountArgs {
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

    /// Include the decoded `kmer_sequence` column in the output.
    ///
    /// Off by default: that column is entirely derivable from `kmer_u64`
    /// plus `--kmer-size` (it is exactly what `kmer::decode_kmer_into`
    /// computes), and on the benchmark file it is 1,667 MB against 430 MB
    /// for `kmer_u64` and 215 MB for `frequency` -- 2.6x the other two
    /// columns combined, and the majority of the ~17% of a run's wall
    /// time the export step costs. Every consumer in this crate and its
    /// Python API works in `u64` space and never reads it. Pass this flag
    /// to get it back, or reconstruct it
    /// from `kmer_u64` in Python via `KmerCounts.with_sequence()` without
    /// rereading the FASTQ.
    #[arg(long)]
    pub with_sequence: bool,

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

impl CountArgs {
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

/// `fastdna sketch`: builds a single-file MinHash sketch (`sketch::
/// GenomeSketch`) and saves it to disk, so later comparisons (`fastdna
/// dist`) reread a small JSON file instead of the original FASTQ/FASTA --
/// the same "compute the sketch once, compare many times" shape
/// `fastdna.sketch()` / `Sketch.save` already give Python callers, now
/// reachable without leaving the shell.
#[derive(Args, Debug)]
pub struct SketchArgs {
    /// Input FASTQ/FASTA file, optionally gzipped. A single file, unlike
    /// `count`'s `--input`: a sketch fingerprints one sample, and multiple
    /// lanes of the same sample should be concatenated (e.g. via `cat`)
    /// before sketching -- the same expectation `GenomeSketch::from_path`
    /// already imposes on the Python binding.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Length of k-mers (1 <= k <= 32). Defaults to 21, not `count`'s 31:
    /// shorter k is the literature's usual choice for sketching/comparison
    /// work (Mash's own default), matching `fastdna.sketch()`'s Python
    /// default so a sketch built from either surface is directly
    /// comparable.
    #[arg(short = 'k', long, default_value_t = 21)]
    pub kmer_size: usize,

    /// Number of smallest hashes to keep -- the MinHash fingerprint's size.
    /// Larger values estimate Jaccard/containment more precisely at the
    /// cost of a bigger saved sketch; 1000 matches `fastdna.sketch()`'s
    /// Python default.
    #[arg(long, default_value_t = 1000)]
    pub sketch_size: usize,

    /// Output path for the saved sketch (JSON; see `GenomeSketch::save`).
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,
}

/// `fastdna dist`: pairwise comparison across several sketches and/or
/// FASTQ/FASTA files, analogous to Python's `fastdna.compare_all`.
#[derive(Args, Debug)]
pub struct DistArgs {
    /// Two or more inputs to compare, in any mix: a previously saved
    /// sketch (a `.json` file, as written by `fastdna sketch` / `Sketch.
    /// save`) is loaded as-is; anything else is treated as a FASTQ/FASTA
    /// file and sketched on the fly using `--kmer-size`/`--sketch-size`
    /// below. `num_args = 2..` because a pairwise comparison of fewer than
    /// two things is not a comparison.
    #[arg(short, long, value_name = "FILE", num_args = 2.., required = true)]
    pub input: Vec<PathBuf>,

    /// Pairwise metric. "jaccard" (the default) and "mash" are symmetric,
    /// so each unordered pair is reported once; "containment" is
    /// asymmetric (`A.containment(B) != B.containment(A)` in general -- see
    /// `GenomeSketch::containment`'s doc comment), so both directions of
    /// every pair are reported.
    #[arg(long, value_enum, default_value = "jaccard")]
    pub metric: CliDistMetric,

    /// k-mer size used only for inputs sketched on the fly -- a saved
    /// `.json` sketch already has its own `k` baked in and ignores this.
    /// Comparing a saved sketch built at one `k` against an on-the-fly
    /// sketch at a different `k` fails the same way comparing two saved
    /// sketches of different `k` already does (a `MismatchedK` error).
    #[arg(short = 'k', long, default_value_t = 21)]
    pub kmer_size: usize,

    /// Sketch size used only for inputs sketched on the fly; see
    /// `--kmer-size`.
    #[arg(long, default_value_t = 1000)]
    pub sketch_size: usize,

    /// Optional output path for the result table (CSV: `sample_a,
    /// sample_b,<metric>`). Without this flag the same rows print to
    /// stdout instead, so `fastdna dist` is usable directly in a pipe.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// `fastdna card`: HyperLogLog cardinality estimate (`hll::
/// estimate_cardinality`) -- how many distinct canonical k-mers a whole
/// file holds, in bounded memory, without ever materializing the k-mer set
/// itself. See `hll.rs`'s module doc comment for how this differs from
/// `peek`'s exact-but-sampled count.
#[derive(Args, Debug)]
pub struct CardArgs {
    /// Input FASTQ/FASTA file, optionally gzipped.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Length of k-mers (1 <= k <= 32).
    #[arg(short = 'k', long, default_value_t = 31)]
    pub kmer_size: usize,

    /// HyperLogLog register precision: memory is `2^precision` bytes,
    /// standard error is `~1.04 / sqrt(2^precision)`. See `hll::
    /// HyperLogLog::new`'s doc comment for the full tradeoff; the default
    /// matches `fastdna.estimate_cardinality`'s Python default.
    #[arg(long, default_value_t = crate::hll::DEFAULT_PRECISION)]
    pub precision: u32,
}

/// `fastdna peek`: a quick preview of a FASTQ/FASTA file's read geometry
/// and composition (`preview::peek`) -- read length min/median/max, GC
/// content, and a suggested k, all computed from at most the first
/// `--n-reads` records, never the whole file. Analogous to Python's
/// `fastdna.peek()`.
#[derive(Args, Debug)]
pub struct PeekArgs {
    /// Input FASTQ/FASTA file, optionally gzipped.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Maximum number of records to sample. `preview::peek` itself rejects
    /// anything above `preview::MAX_N_READS`, so this stays a genuine
    /// preview rather than becoming an accidental full read.
    #[arg(long, default_value_t = 10_000)]
    pub n_reads: usize,
}

/// `fastdna query`: a single point lookup against a sorted k-mer table
/// (`ktab::KmerTable`) -- the random-access counterpart to `count`'s
/// write-only Parquet output. Deliberately one k-mer per invocation rather
/// than a batch/file-of-k-mers mode: `ktab::KmerTable::range`/`iter` (not
/// yet exposed as their own subcommand) are the right tool for anything
/// wider than a handful of ad hoc lookups, and this subcommand exists for
/// the ad hoc case a shell script or a quick sanity check actually has.
#[derive(Args, Debug)]
pub struct QueryArgs {
    /// Path to a sorted k-mer table: a `.parquet` file written by `fastdna
    /// count` (any counting run's default output already qualifies -- see
    /// `ktab.rs`'s module doc comment for why no separate conversion step
    /// exists).
    #[arg(long, value_name = "FILE")]
    pub table: PathBuf,

    /// The k-mer to look up: either a DNA sequence of exactly the table's
    /// own `k` (case-insensitive A/C/G/T/U; canonicalized the same way
    /// counting does), or a plain decimal integer, taken as the table's raw
    /// `kmer_u64` encoding directly (for a value already read from a
    /// `kmer_u64` column elsewhere).
    #[arg(long, value_name = "KMER")]
    pub kmer: String,
}

/// `fastdna union`: every k-mer present in any of `--input`'s tables,
/// written out as a new sorted k-mer table (see `setops::union`'s doc
/// comment for the streaming merge and the `--combine` semantics).
#[derive(Args, Debug)]
pub struct UnionArgs {
    /// Two or more sorted k-mer tables to combine (`.parquet` files written
    /// by `fastdna count`, or by a previous `union`/`intersect`/`diff` --
    /// set operations compose). `num_args = 2..` because a union of fewer
    /// than two tables is not a combination of anything.
    #[arg(short, long, value_name = "FILE", num_args = 2.., required = true)]
    pub input: Vec<PathBuf>,

    /// Output path for the resulting k-mer table (`.parquet`; immediately
    /// reopenable with `fastdna query`/`KmerTable.open`).
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// How to fold several tables' counts for the same k-mer into this
    /// output's single `frequency` column. "sum" (the default) is the
    /// natural "combine these samples" reading -- see `setops::CombineOp`'s
    /// doc comment for what "min"/"max" mean instead.
    #[arg(long, value_enum, default_value = "sum")]
    pub combine: CliCombineOp,
}

/// `fastdna intersect`: only k-mers present in *every* one of `--input`'s
/// tables, written out as a new sorted k-mer table (see `setops::
/// intersect`'s doc comment).
#[derive(Args, Debug)]
pub struct IntersectArgs {
    /// Two or more sorted k-mer tables. See `UnionArgs::input`'s doc
    /// comment for the same `num_args = 2..` reasoning.
    #[arg(short, long, value_name = "FILE", num_args = 2.., required = true)]
    pub input: Vec<PathBuf>,

    /// Output path for the resulting k-mer table (`.parquet`).
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// How to fold every table's count for a shared k-mer into this
    /// output's single `frequency` column. "min" (the default) is the
    /// conservative summary -- see `setops::CombineOp`'s doc comment.
    #[arg(long, value_enum, default_value = "min")]
    pub combine: CliCombineOp,
}

/// `fastdna diff`: every k-mer in `--input` absent (or below
/// `--max-subtract-count`) from every table named by `--subtract` -- the
/// reference-subtraction/host-removal use case (see `setops::diff`'s doc
/// comment).
#[derive(Args, Debug)]
pub struct DiffArgs {
    /// The table to filter (e.g. a sample's own k-mer counts).
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// One or more reference tables to subtract (a host genome, an adapter
    /// set, a known contaminant, or several of them at once -- a k-mer is
    /// dropped if it is flagged by *any* of them, see `setops::diff`'s doc
    /// comment).
    #[arg(long, value_name = "FILE", num_args = 1.., required = true)]
    pub subtract: Vec<PathBuf>,

    /// Output path for the filtered k-mer table (`.parquet`); its rows keep
    /// `--input`'s own original counts, never blended with `--subtract`'s.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// A k-mer's count in a `--subtract` table may reach this value without
    /// being treated as contamination; above it, the k-mer is dropped from
    /// the output. 0 (the default) is the strict "present at all, in any
    /// reference" reading -- raise it to tolerate a small amount of
    /// reference noise, e.g. a handful of spurious low-depth hits in a host
    /// genome, before a k-mer counts as real contamination.
    #[arg(long, value_name = "COUNT", default_value_t = 0)]
    pub max_subtract_count: u32,
}

/// `fastdna filter`: streams a FASTQ/FASTA input against a reference k-mer
/// table (`ktab::KmerTable`) and writes each read to `--output` (always as
/// FASTQ, even for FASTA input -- see `read_filter.rs`'s module doc
/// comment) according to `--mode` -- the classic `kmc_tools filter`/BBDuk
/// `ref=`/`k=` host-removal or targeted-enrichment workflow
/// (`docs/feature-gap-analysis.md`'s S4).
///
/// **Single-end by default.** Each `--input` file is filtered
/// independently, read by read, and several files are filtered as one
/// concatenated stream into `--output` -- the same convention `count`'s own
/// multi-file `--input` already uses.
///
/// **Paired-end, via `--input2`/`--output2`.** Given together (see
/// `FilterArgs::validate`), `--input` is read as the R1 stream and
/// `--input2` as R2 -- BBDuk's `in1=`/`in2=`/`out1=`/`out2=` convention,
/// spelled with this crate's own flag names. The two streams are read in
/// lock step and a pair is kept or discarded as a unit if *either* mate
/// matches the reference, never independently per mate, so the two output
/// streams (`--output` for R1, `--output2` for R2) can never drift out of
/// sync with each other. See `read_filter.rs`'s module doc comment
/// ("Paired-end (R1/R2) synchronized filtering") for the full design and
/// the reasoning behind "either mate matches".
#[derive(Args, Debug)]
pub struct FilterArgs {
    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin. In
    /// paired-end mode (`--input2` also given) this is the R1 side. See
    /// this subcommand's own doc comment for the single-end/paired-end
    /// scope note.
    #[arg(short, long, value_name = "FILE", num_args = 1.., required = true)]
    pub input: Vec<PathBuf>,

    /// R2 (reverse mate) input file(s) for paired-end filtering, required
    /// together with `--output2` (`FilterArgs::validate`), and omitted
    /// entirely for single-end filtering. `--input`/`--input2` do *not*
    /// need to name the same number of individual files -- each side is
    /// concatenated into one stream first (the same multi-file convention
    /// `--input` already uses on its own), and it is those two streams
    /// that are paired, record by record; only their *total* record counts
    /// need to agree. A genuine mismatch there is reported as an error
    /// naming whichever side ran out first, not silently truncated -- see
    /// `fastq::PairedSourceReader`'s doc comment.
    #[arg(long, value_name = "FILE", num_args = 1..)]
    pub input2: Vec<PathBuf>,

    /// The reference k-mer table to filter against: a `.parquet` file
    /// written by `fastdna count`, or by `union`/`intersect`/`diff` -- any
    /// valid `KmerTable` (`ktab::KmerTable::open`).
    #[arg(long, value_name = "FILE")]
    pub table: PathBuf,

    /// "keep" writes only reads (or, in paired-end mode, pairs) that match
    /// the reference (targeted enrichment: keep only reads that look like
    /// this organism/panel). "discard" writes only reads/pairs that do
    /// *not* match it (host/contaminant removal: remove this reference's
    /// reads and keep the rest of the sample).
    #[arg(long, value_enum)]
    pub mode: CliFilterMode,

    /// A read "matches" the reference when at least this fraction of its
    /// own canonical k-mers are found in `--table` (`>=`, inclusive, so a
    /// read sitting exactly on the threshold matches). 0.1 (the default) is
    /// a "keep if there is any real presence" reading, not a strict
    /// full-containment requirement; raise it towards 1.0 for a
    /// higher-confidence match, or lower it towards 0.0 for "any single hit
    /// at all is enough". A read that yields *no* canonical k-mers of its
    /// own (shorter than the table's `k`, or entirely ambiguous bases)
    /// never matches, regardless of this value -- see
    /// `read_filter::matching_fraction`'s doc comment. In paired-end mode
    /// this threshold is applied to each mate independently; the pair-level
    /// keep/discard decision then combines both mates' individual match
    /// results with a logical OR (see `FilterArgs`'s own doc comment).
    #[arg(long, default_value_t = 0.1, value_parser = parse_min_fraction)]
    pub min_fraction: f64,

    /// Output path for the reads this run keeps (FASTQ, optionally `.gz`;
    /// gzip is chosen by this path's own extension, matching every other
    /// gzip-aware path in this crate). In paired-end mode this is the R1
    /// output; see `--output2` for R2.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// R2 (reverse mate) output path, required together with `--input2`
    /// and omitted entirely for single-end filtering. See `--input2`'s doc
    /// comment and `FilterArgs`'s own doc comment.
    #[arg(long, value_name = "FILE")]
    pub output2: Option<PathBuf>,
}

impl FilterArgs {
    /// Whether `--input2`/`--output2` were given, switching this run into
    /// paired-end mode. Only meaningful after `validate` has confirmed the
    /// two are given together (or not at all) -- see that method.
    pub fn is_paired(&self) -> bool {
        !self.input2.is_empty()
    }

    /// Cross-flag validation clap cannot express declaratively:
    /// `--input2` and `--output2` must be given together, naming both a
    /// paired-end run's R2 input and its R2 output, or neither at all for
    /// a single-end run. Deliberately does *not* require `--input` and
    /// `--input2` to name the same number of *files* -- see `--input2`'s
    /// own doc comment for why that would reject legitimate input (e.g. an
    /// R1 side split across lanes with an already-concatenated R2 side);
    /// a genuine record-count mismatch between the two streams is instead
    /// caught while reading them (`fastq::PairedSourceReader`), the same
    /// "the check that matters lives where the real invariant is" shape
    /// `CountArgs::validate` already follows for `--paired-dir`.
    pub fn validate(&self) -> Result<(), String> {
        match (self.input2.is_empty(), self.output2.is_some()) {
            (true, false) | (false, true) => Ok(()),
            (true, true) => Err(
                "--output2 was given without --input2: paired-end filtering needs both an R2 \
                 input and an R2 output, or neither"
                    .to_string(),
            ),
            (false, false) => Err(
                "--input2 was given without --output2: paired-end filtering needs both an R2 \
                 input and an R2 output, or neither"
                    .to_string(),
            ),
        }
    }
}

/// `fastdna matrix`: builds a cohort-wide k-mer presence/count matrix
/// (`cohort::matrix::CohortMatrix`) from several samples' FASTQ files and
/// writes it as a long/COO-format Parquet table (`export::
/// export_cohort_matrix_parquet`) -- the CLI verb and generic,
/// FastDNA-tool-independent file artifact `docs/feature-gap-analysis.md`'s
/// S6 named as the one piece still missing once the matrix-building engine
/// had already landed. (It also had a Python binding, removed with the ML
/// layer on 2026-09-05; the CLI verb was always the artifact-producing
/// half and is unaffected.)
///
/// Two ways to name the cohort, mutually exclusive (`MatrixArgs::validate`):
/// `--input DIR` reuses `cohort::discover_samples`'s R1/R2 pairing (the same
/// convention `count`'s own `--paired-dir` already uses -- see
/// `CountArgs::paired_dir`'s doc comment), so a directory of paired or
/// single-end FASTQ files needs no manual per-sample file-list bookkeeping;
/// `--sample FILE...` names each sample's single file explicitly, with no
/// automatic pairing, for a cohort whose files are not laid out in one
/// directory or do not follow a naming convention `discover_samples`
/// recognizes.
#[derive(Args, Debug)]
pub struct MatrixArgs {
    /// Directory of FASTQ files to discover as cohort samples, one row of
    /// the output matrix per discovered sample (paired or single-end -- see
    /// `cohort::discover_samples`'s module doc comment for the exact
    /// pairing rules). An orphaned mate is a warning printed to the
    /// console, not a hard failure -- unlike `count --paired-dir`'s strict
    /// mode, this is the ad hoc cohort-listing use `discover_samples`
    /// itself is documented for. Mutually exclusive with `--sample`.
    #[arg(short, long, value_name = "DIR")]
    pub input: Option<PathBuf>,

    /// One file per sample, named explicitly -- no automatic R1/R2 pairing.
    /// The sample id is derived from each file's name the same way
    /// `python/fastdna/cohort_counts.py::_sample_id_from_path` does (a trailing
    /// `.gz` stripped first, then the remaining extension), so a sample id
    /// computed here agrees with the one that Python entry point would
    /// compute for the same file. Mutually exclusive with `--input`.
    #[arg(long, value_name = "FILE", num_args = 1..)]
    pub sample: Vec<PathBuf>,

    /// Output path for the cohort matrix (`.parquet`; see `export::
    /// export_cohort_matrix_parquet`'s doc comment for the schema).
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// Length of k-mers (1 <= k <= 32).
    #[arg(short = 'k', long, default_value_t = 31)]
    pub kmer_size: usize,

    /// Minimum Phred quality score cutoff, forwarded to each sample's
    /// counting run (same meaning as `count`'s `--min-quality`).
    #[arg(short = 'q', long, default_value_t = 20.0, value_parser = parse_min_quality)]
    pub min_quality: f64,

    /// Per-sample minimum depth for a k-mer to count as observed at all.
    /// Defaults to 2, not `count`'s 1, for the reason `cohort::matrix`
    /// states: at realistic coverage a depth-1 k-mer is overwhelmingly
    /// likely a sequencing error, and each one would otherwise become a
    /// private, cohort-meaningless column.
    #[arg(short = 'm', long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..))]
    pub min_count: u32,

    /// A k-mer must be observed in at least this many samples to become a
    /// column of the matrix. Defaults to 2: a k-mer private to a single
    /// sample carries no cohort-level signal and is usually a residual
    /// sequencing error `--min-count` did not catch.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..))]
    pub min_samples: u32,

    /// Hard cap on the number of columns (k-mers) kept, ranked by
    /// minor-sample-count -- see `cohort::matrix::build_cohort_matrix`'s
    /// doc comment for the exact ranking and tie-break rules. Unset (the
    /// default) keeps every k-mer that passes `--min-samples`.
    #[arg(long, value_name = "COUNT")]
    pub max_kmers: Option<usize>,

    /// Include the decoded `kmer_sequence` column in the output, alongside
    /// the compact `kmer_u64` encoding that is always written. Off by
    /// default for the same reason `count --with-sequence` is: the column
    /// is entirely derivable from `kmer_u64` plus `--kmer-size`.
    #[arg(long)]
    pub with_sequence: bool,

    /// Number of worker threads used for each sample's counting run.
    #[arg(short, long)]
    pub threads: Option<usize>,
}

/// `fastdna profile`: streams a FASTQ/FASTA input against a reference k-mer
/// table (`ktab::KmerTable`) and writes two Parquet files -- an
/// RLE-compressed per-read profile and a small per-read summary table --
/// via `read_profile::run_profile` (`docs/feature-gap-analysis.md`'s S3).
/// See `read_profile.rs`'s module doc comment for the output-format
/// decision (RLE, not one row per base) and the single-end-only scope this
/// shares with `fastdna filter`.
#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin.
    /// Single-end only -- see this subcommand's own doc comment.
    #[arg(short, long, value_name = "FILE", num_args = 1.., required = true)]
    pub input: Vec<PathBuf>,

    /// The reference k-mer table to profile against: a `.parquet` file
    /// written by `fastdna count` (or `union`/`intersect`/`diff`) -- any
    /// valid `KmerTable` (`ktab::KmerTable::open`).
    #[arg(long, value_name = "FILE")]
    pub table: PathBuf,

    /// Output path for the RLE-compressed per-read profile (`.parquet`;
    /// see `read_profile::profile_schema`).
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// Output path for the per-read summary table (`read_id`, `n_kmers`,
    /// `n_present_kmers`, `min_count`, `median_count`, `max_count`; see
    /// `read_profile::summary_schema`).
    ///
    /// Defaults to `read_profile_summary.parquet` **in `--output`'s own
    /// directory**, which is what this doc comment always claimed and what
    /// the code did not do: a bare relative `default_value` resolves against
    /// the process's working directory, so `-o /data/run1/prof.parquet`
    /// silently wrote the summary to `./read_profile_summary.parquet`
    /// instead. That is worse than untidy -- a loop profiling many samples
    /// into separate output directories had every iteration overwrite one
    /// shared summary file, keeping only the last, with no error.
    ///
    /// `None` here means "not given"; `resolve_summary_path` derives it.
    /// An explicit value is used exactly as passed.
    #[arg(long, value_name = "FILE")]
    pub summary: Option<PathBuf>,
}

impl ProfileArgs {
    /// The summary path to actually write: the caller's if they gave one,
    /// otherwise `read_profile_summary.parquet` beside `--output`.
    pub fn resolve_summary_path(&self) -> PathBuf {
        if let Some(explicit) = &self.summary {
            return explicit.clone();
        }
        match self.output.parent() {
            // `parent()` of a bare filename is `Some("")`, which would build
            // "/read_profile_summary.parquet" if joined blindly.
            Some(dir) if !dir.as_os_str().is_empty() => dir.join(DEFAULT_PROFILE_SUMMARY),
            _ => PathBuf::from(DEFAULT_PROFILE_SUMMARY),
        }
    }
}

/// File name used for `fastdna profile`'s summary table when `--summary` is
/// not given. Only the name: the directory comes from `--output`.
pub const DEFAULT_PROFILE_SUMMARY: &str = "read_profile_summary.parquet";

impl MatrixArgs {
    /// Cross-flag validation clap cannot express declaratively -- the same
    /// "name the cohort exactly one way" check `CountArgs::validate` makes
    /// between `--input` and `--paired-dir`.
    pub fn validate(&self) -> Result<(), String> {
        match (self.input.is_some(), self.sample.is_empty()) {
            (true, false) => Err(
                "--input and --sample are mutually exclusive: a cohort is either discovered from a \
                 directory with --input, or named file by file with --sample, not both"
                    .to_string(),
            ),
            (false, true) => {
                Err("no samples given: pass --input <DIR> or --sample <FILE>...".to_string())
            }
            _ => Ok(()),
        }
    }
}

/// `fastdna spectrum`: ntCard-style streaming estimate of the k-mer
/// frequency spectrum (`src/ntcard.rs`) -- how many distinct k-mers occur
/// exactly once, twice, ... in `2^precision * 16` bytes of memory,
/// independent of input size, without ever materializing the k-mer table
/// `count` builds. Output is the same `{depth: distinct k-mers}` shape as
/// `count --histogram`'s exact spectrum (and `KmerCounts.spectrum()` in
/// Python), so `python/fastdna/genomescope.py::profile_genome` can consume
/// either one interchangeably. See `src/ntcard.rs`'s module doc comment for
/// the algorithm and its measured accuracy.
#[derive(Args, Debug)]
pub struct SpectrumArgs {
    /// Input FASTQ/FASTA file(s), optionally gzipped, or "-" for stdin.
    /// Same multi-file/stdin convention as `count --input`: several lanes
    /// of the same sample are aggregated into one spectrum.
    #[arg(short, long, value_name = "FILE", num_args = 1.., required = true)]
    pub input: Vec<PathBuf>,

    /// Length of k-mers (1 <= k <= 32).
    #[arg(short = 'k', long, default_value_t = 31)]
    pub kmer_size: usize,

    /// ntCard bucket-table precision: memory is `2^precision * 16` bytes
    /// (twice `fastdna card`'s HyperLogLog at the same precision, since
    /// each bucket here keeps 16 bytes instead of 1 -- see `src/ntcard.rs`'s
    /// module doc comment for why). Same valid range and default as
    /// `fastdna card`'s own `--precision`.
    #[arg(long, default_value_t = crate::hll::DEFAULT_PRECISION)]
    pub precision: u32,

    /// Cap the estimated spectrum at this depth: everything deeper is
    /// folded into this row rather than dropped (the same KMC `-cx`
    /// convention `count --histogram-max` uses), keeping the estimated
    /// total distinct-k-mer count intact.
    #[arg(long, value_name = "DEPTH", value_parser = clap::value_parser!(u32).range(1..))]
    pub max_frequency: Option<u32>,

    /// Output path for the estimated spectrum. A ".json" extension
    /// (case-insensitive) writes `{"depth": distinct_kmers, ...}`; any
    /// other extension writes the same textual spectrum `count
    /// --histogram` writes, in `--format`.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// Text output format, used whenever `--output` does not end in
    /// ".json". Same values and default as `count --histogram-format`.
    #[arg(long, value_enum, default_value = "csv")]
    pub format: CliHistogramFormat,
}
