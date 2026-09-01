// examples/chimera_calibration.rs

// Like main.rs, this binary legitimately owns all of its own console
// output (it is a CLI report, not library code) -- the crate-wide clippy
// denials in Cargo.toml exist to keep the library core silent, not this
// entry point. Also allows unwrap/expect: `main.rs` reserves that stance
// for the shipped binary and its own `run()` returns `Result` instead, but
// this file is a throwaway local calibration script (see its own doc
// comment: gitignored `scratch/` inputs, not part of the crate's tested
// surface), where a hard `panic!` on a missing/malformed genome file is
// the right failure mode -- there is no caller to hand a `Result` back to.
#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unwrap_used, clippy::expect_used)]

//! Ad hoc calibration driver for `chimera_scan` (`src/chimera_scan.rs`),
//! run locally against real reference genomes staged in
//! `scratch/chimera_scan/genomes/` -- gitignored, not part of the
//! committed repository (see `CLAUDE.md`'s "Known non-obvious facts" for
//! this project's `scratch/` convention). Not part of the test suite or
//! this crate's public API surface.
//!
//! Run with `cargo run --release --example chimera_calibration` from the
//! crate root, after populating `scratch/chimera_scan/genomes/` with:
//!
//! - `ecoli_562.22432.fna`, `ecoli_562.22496.fna`,
//!   `other_bact_1328432.3.fna` -- real *E. coli* WGS draft assemblies
//!   (reused from `scratch/amr_repro/genomes/` at the top-level repo, per
//!   this task's brief).
//! - `bacillus_subtilis.fna` -- *Bacillus subtilis* subsp. *subtilis* str.
//!   168, GCF_000009045.1 (Firmicutes -- a different phylum from the
//!   *E. coli* above's Proteobacteria, the "hard" same-domain case).
//! - `methanocaldococcus_jannaschii.fna` -- GCF_000091665.1 (Archaea, the
//!   "easy" cross-domain case).
//! - `saccharomyces_cerevisiae.fna` -- GCF_000146045.2 (Eukaryote, the
//!   other "easy" cross-domain case).
//!
//! # What this measures, and why
//!
//! Two questions, both real, reportable results rather than a hand-picked
//! threshold:
//!
//! 1. **Sensitivity**: for synthetic chimeras built by concatenating real
//!    arms from two genomes at a controlled taxonomic distance (deterministic
//!    seed 9001, this project's own convention -- `scripts/bench/
//!    generate_reads_large.py`), what fraction of replicates have a
//!    divergence signal at the true junction at or above a candidate
//!    threshold? Swept across window size and taxonomic-distance category
//!    (same-domain-different-phylum "hard" vs. cross-domain "easy").
//! 2. **Negative-control false-positive rate**: run on real, complete,
//!    non-chimeric sequence (the same genomes, *not* concatenated), how
//!    many breakpoints get flagged per candidate threshold? This is the
//!    mandatory gate this task's brief describes -- genomic islands,
//!    prophages and rRNA operons are real biology that also shifts local
//!    composition, so some flagging here is expected; a *high* rate is the
//!    signal that composition alone is not sufficient without read-coverage
//!    corroboration.
//!
//! Output is plain CSV-ish text to stdout, meant to be captured and
//! reported, not machine-parsed by anything else in this repository.

use fastdna_core::chimera_scan::{divergence_profile, ChimeraScanParams, DivergenceSample};
use std::fs;
use std::path::{Path, PathBuf};

// -- deterministic PRNG (xorshift64, seed 9001) ------------------------------

struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 is undefined at seed 0; this project's own seed 9001
        // is nonzero, but guard anyway so a future caller changing the seed
        // does not get a silently-stuck generator.
        Xorshift64(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// -- FASTA loading ------------------------------------------------------------

fn genome_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scratch/chimera_scan/genomes")
}

/// Reads a FASTA file and returns its largest contig/chromosome/replicon's
/// sequence (uppercase-or-not bytes, exactly as written -- `kmer.rs`'s
/// extraction is already case-insensitive). "Largest contig", not "every
/// contig concatenated": a multi-contig assembly's own contig boundaries
/// are not biological signal, and concatenating them would plant synthetic
/// junctions of this loader's own making into what is supposed to be a
/// clean single-organism reference.
fn largest_contig(path: &Path) -> Vec<u8> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("chimera_calibration: failed to read {}: {e}", path.display()));

    let mut contigs: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for line in text.lines() {
        if let Some(_header) = line.strip_prefix('>') {
            if !current.is_empty() {
                contigs.push(std::mem::take(&mut current));
            }
        } else {
            current.extend(line.trim().as_bytes());
        }
    }
    if !current.is_empty() {
        contigs.push(current);
    }
    contigs
        .into_iter()
        .max_by_key(|c| c.len())
        .unwrap_or_else(|| panic!("chimera_calibration: {} contains no sequence", path.display()))
}

// -- synthetic chimera construction -------------------------------------------

/// One synthetic chimera: `sequence` is `arm_len` real bases from `a`
/// followed by `arm_len` real bases from `b`, at random (seeded) offsets;
/// `true_junction` is the exact, known boundary (== `arm_len`).
struct SyntheticChimera {
    sequence: Vec<u8>,
    true_junction: u32,
}

fn build_chimera(rng: &mut Xorshift64, a: &[u8], b: &[u8], arm_len: usize) -> Option<SyntheticChimera> {
    if a.len() < arm_len || b.len() < arm_len {
        return None;
    }
    let a_start = rng.next_range(a.len() - arm_len + 1);
    let b_start = rng.next_range(b.len() - arm_len + 1);

    let mut sequence = Vec::with_capacity(2 * arm_len);
    sequence.extend_from_slice(&a[a_start..a_start + arm_len]);
    sequence.extend_from_slice(&b[b_start..b_start + arm_len]);

    Some(SyntheticChimera { sequence, true_junction: arm_len as u32 })
}

// -- profile summaries ---------------------------------------------------------

/// The strongest divergence signal within `radius` bases of `center` --
/// "was there a strong signal at (or very near) the true junction".
fn max_near(profile: &[DivergenceSample], center: u32, radius: u32) -> f64 {
    profile
        .iter()
        .filter(|s| (s.position as i64 - center as i64).unsigned_abs() as u32 <= radius)
        .map(|s| s.divergence)
        .fold(0.0, f64::max)
}

/// The strongest divergence signal *outside* `radius` bases of `center` --
/// the background/noise level within an otherwise-homogeneous arm, for
/// comparison against `max_near`'s at-the-junction signal.
fn max_outside(profile: &[DivergenceSample], center: u32, radius: u32) -> f64 {
    profile
        .iter()
        .filter(|s| (s.position as i64 - center as i64).unsigned_abs() as u32 > radius)
        .map(|s| s.divergence)
        .fold(0.0, f64::max)
}

/// Fraction of `seq` that is G or C (case-insensitive) -- a coarse,
/// order-0 composition summary printed as a sanity check that the loaded
/// genomes are genuinely compositionally different from each other before
/// trusting any tetranucleotide-level divergence number computed from them.
fn gc_fraction(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return f64::NAN;
    }
    let gc = seq.iter().filter(|&&b| matches!(b, b'G' | b'g' | b'C' | b'c')).count();
    gc as f64 / seq.len() as f64
}

/// Counts flagged, clustered breakpoints in an already-computed profile at
/// a given threshold -- the same consecutive-candidate clustering rule
/// `chimera_scan::scan_sequence` uses, reimplemented here to operate on a
/// profile computed once per (genome, window) rather than recomputing the
/// k-mer extraction once per threshold in the sweep below.
fn count_clusters(profile: &[DivergenceSample], threshold: f64) -> usize {
    let mut clusters = 0usize;
    let mut in_cluster = false;
    for sample in profile {
        if sample.divergence >= threshold {
            if !in_cluster {
                clusters += 1;
                in_cluster = true;
            }
        } else {
            in_cluster = false;
        }
    }
    clusters
}

// -- driver ---------------------------------------------------------------------

const SEED: u64 = 9001;
const WINDOW_SIZES: &[usize] = &[300, 500, 1000, 2000, 4000];
const REPLICATES_PER_CELL: usize = 15;
// Revised downward from an initial [0.30 ..= 0.95] sweep: that sweep came
// back all-zero sensitivity at every window/category and near-zero
// negative-control flagging above 0.40 -- the raw per-replicate signal
// (see `## raw_replicate` output) showed real junction divergence at real
// genome pairs tops out around 0.05-0.4 depending on window size, an order
// of magnitude below the "AT-only vs. GC-only" synthetic construction
// `chimera_scan.rs`'s own unit tests use (which reaches ~1.0). This finer,
// lower range is where the actual sensitivity/specificity trade-off lives.
const THRESHOLDS: &[f64] = &[0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40];
const K: usize = 4;

fn step_for(window: usize) -> usize {
    (window / 5).max(20)
}

fn main() {
    let dir = genome_dir();
    println!("# chimera_scan calibration -- seed {SEED}, k={K}");
    println!("# genome directory: {}", dir.display());

    let ecoli_a = largest_contig(&dir.join("ecoli_562.22432.fna"));
    let ecoli_b = largest_contig(&dir.join("ecoli_562.22496.fna"));
    let ecoli_c = largest_contig(&dir.join("other_bact_1328432.3.fna"));
    let bacillus = largest_contig(&dir.join("bacillus_subtilis.fna"));
    let archaeon = largest_contig(&dir.join("methanocaldococcus_jannaschii.fna"));
    let eukaryote = largest_contig(&dir.join("saccharomyces_cerevisiae.fna"));

    println!("# largest contig sizes (bases) and GC fraction (sanity check -- these genomes must actually differ):");
    println!("#   ecoli_562.22432          {} bases, GC {:.4}", ecoli_a.len(), gc_fraction(&ecoli_a));
    println!("#   ecoli_562.22496          {} bases, GC {:.4}", ecoli_b.len(), gc_fraction(&ecoli_b));
    println!("#   other_bact_1328432.3     {} bases, GC {:.4}", ecoli_c.len(), gc_fraction(&ecoli_c));
    println!("#   bacillus_subtilis        {} bases, GC {:.4}", bacillus.len(), gc_fraction(&bacillus));
    println!("#   methanocaldococcus       {} bases, GC {:.4}", archaeon.len(), gc_fraction(&archaeon));
    println!("#   saccharomyces_cerevisiae {} bases, GC {:.4}", eukaryote.len(), gc_fraction(&eukaryote));

    // -- category -> (genome A, genome B) --------------------------------
    let categories: Vec<(&str, &[u8], &[u8])> = vec![
        ("hard_same_domain_diff_phylum", &ecoli_a, &bacillus),
        ("easy_cross_domain_bact_archaea", &ecoli_a, &archaeon),
        ("easy_cross_domain_bact_eukaryote", &ecoli_a, &eukaryote),
        ("easy_cross_domain_archaea_eukaryote", &archaeon, &eukaryote),
    ];

    // -- sensitivity sweep -------------------------------------------------
    // Raw per-replicate signal first (diagnostic: what does the divergence
    // actually look like at and away from a real junction, before any
    // threshold is applied), then the threshold-sweep summary.
    println!();
    println!("## raw_replicate: category,window,step,replicate,arm_len,junction_signal,background_signal");
    let mut rng = Xorshift64::new(SEED);
    // category -> window -> Vec<junction_signal>, reused below for the
    // threshold sweep so the sequence generation and divergence_profile
    // computation happens exactly once per replicate.
    let mut signals_by_cell: Vec<(&str, usize, Vec<f64>)> = Vec::new();

    for &window in WINDOW_SIZES {
        let step = step_for(window);
        let arm_len = window * 6;
        let params = ChimeraScanParams { window, step, k: K, threshold: None };

        for &(name, a, b) in &categories {
            let mut junction_signals: Vec<f64> = Vec::new();
            for replicate in 0..REPLICATES_PER_CELL {
                let Some(chimera) = build_chimera(&mut rng, a, b, arm_len) else {
                    continue;
                };
                let profile = divergence_profile(&chimera.sequence, &params);
                let junction_signal = max_near(&profile, chimera.true_junction, window as u32);
                let background_signal = max_outside(&profile, chimera.true_junction, window as u32);
                println!("{name},{window},{step},{replicate},{arm_len},{junction_signal:.6},{background_signal:.6}");
                junction_signals.push(junction_signal);
            }
            signals_by_cell.push((name, window, junction_signals));
        }
    }

    println!();
    println!("## sensitivity: category,window,step,threshold,detected,replicates,sensitivity");
    for &(name, window, ref junction_signals) in &signals_by_cell {
        let step = step_for(window);
        let n = junction_signals.len();
        if n == 0 {
            println!("{name},{window},{step},<all>,skipped (arm_len exceeds a genome's largest contig),0,n/a");
            continue;
        }
        for &t in THRESHOLDS {
            let detected = junction_signals.iter().filter(|&&d| d >= t).count();
            let sensitivity = detected as f64 / n as f64;
            println!("{name},{window},{step},{t:.2},{detected},{n},{sensitivity:.3}");
        }
    }

    // -- negative control ----------------------------------------------------
    println!();
    println!("## negative_control: genome,window,step,contig_bases,threshold,flagged_clusters,flags_per_mb");
    let negatives: Vec<(&str, &[u8])> = vec![
        ("ecoli_562.22432", &ecoli_a),
        ("ecoli_562.22496", &ecoli_b),
        ("other_bact_1328432.3", &ecoli_c),
        ("bacillus_subtilis", &bacillus),
        ("methanocaldococcus_jannaschii", &archaeon),
        ("saccharomyces_cerevisiae_largest_chr", &eukaryote),
    ];

    for &window in WINDOW_SIZES {
        let step = step_for(window);
        let params = ChimeraScanParams { window, step, k: K, threshold: None };

        for &(name, seq) in &negatives {
            let profile = divergence_profile(seq, &params);
            let mb = seq.len() as f64 / 1_000_000.0;
            for &t in THRESHOLDS {
                let clusters = count_clusters(&profile, t);
                let per_mb = if mb > 0.0 { clusters as f64 / mb } else { 0.0 };
                println!("{name},{window},{step},{},{t:.2},{clusters},{per_mb:.3}", seq.len());
            }
        }
    }
}
