// src/chimera_scan.rs
//! Composition-based chimera detection in (meta)genome assemblies: finds
//! places within a single contig where the local tetranucleotide (or, more
//! generally, k-mer) composition shifts abruptly, the signature of a
//! long-read assembler having fused sequence from two unrelated organisms
//! into one contig.
//!
//! # Calibration result: composition alone did not clear its own negative-
//! # control gate -- read this before trusting a flagged breakpoint
//!
//! `examples/chimera_calibration.rs` built synthetic multi-domain chimeras
//! from real reference genomes (deterministic seed 9001: *E. coli* fused
//! with *Bacillus subtilis* -- the "hard" same-domain-different-phylum
//! case -- and with *Methanocaldococcus jannaschii* and *Saccharomyces
//! cerevisiae* -- the "easy" cross-domain cases), swept window size
//! (300-4000 bases) and threshold (0.05-0.40 bits), and ran the *same*
//! calibrated thresholds against six real, complete (or largest-contig),
//! genuinely non-chimeric reference sequences as a negative control.
//!
//! **Finding: no (window, threshold) combination in the swept range gives
//! both usable sensitivity and an acceptable false-positive rate.** Every
//! operating point that detects a majority of real chimeras also flags
//! the overwhelming majority of real, non-chimeric genome content (at
//! small windows, hundreds to thousands of spurious breakpoints per
//! megabase -- ordinary multinomial sampling noise in a k-mer count, not
//! biology); every operating point with a clean negative control collapses
//! sensitivity, and does so hardest for the same-domain-different-phylum
//! case, whose true-junction divergence overlaps its own single-genome
//! background noise almost entirely (`hard_sens` reaches exactly `0.0` at
//! every window size once the threshold is high enough to keep negative-
//! control flagging under roughly one flag per genome). The best balanced
//! points found (e.g. window 2000, threshold 0.15-0.18) still flag 17-50%
//! of real, single-organism reference genomes at least once while
//! detecting well under half of even the "easy" cross-domain synthetic
//! chimeras. This is a real result, not a bug: unlike the maximally
//! disjoint A/T-only-vs-G/C-only construction this module's own unit tests
//! use, real cross-taxon tetranucleotide differences are compositionally
//! modest, and real single genomes have their own natural local
//! compositional heterogeneity (genomic islands, prophages, rRNA operons,
//! and apparently plain local variation even without any of those) of
//! comparable magnitude.
//!
//! **Conclusion:** tetranucleotide composition (Jensen-Shannon divergence,
//! this module's own measure) is not, on its own, a sufficient signal for
//! reliable multi-domain chimera calling against real assemblies. This
//! module and its `scan_chimeras()`/`fastdna.chimeras.scan_chimeras()`
//! surface remain correct and tested (see this module's own unit tests and
//! `python/tests/test_chimeras.py`) as a composition-divergence primitive,
//! and are useful as *one input among several* or for a caller who has
//! already accepted a much higher false-positive tolerance than a
//! production flagging pipeline would -- but a flagged breakpoint from
//! this module alone should not be read as "this is a chimera". Combining
//! it with read-coverage evidence (the `anvi-script-find-misassemblies`
//! approach this module's own doc comment below describes as
//! complementary) is the natural next step, not merely a nice-to-have; a
//! CLI subcommand and a from-this-module-alone "chimera report" were
//! deliberately not built on top of this until that combination -- or some
//! other way to close this gap -- exists. See this project's calibration
//! report (delivered alongside this change, not committed as source) for
//! the full sensitivity/false-positive tables this summary is drawn from.
//!
//! # Why composition, and why this is a complement to read-mapping tools
//!
//! Tetranucleotide (4-mer) composition is approximately constant within one
//! genome (codon usage, GC content and short-range dinucleotide/oligomer
//! biases are genome-wide properties, not local ones) and differs between
//! distantly related taxa. A chimeric contig -- one assembled from reads
//! belonging to two different organisms, stitched together at a spurious
//! junction -- therefore produces a step-like discontinuity in composition
//! exactly at the fusion point, detectable from the assembly alone.
//!
//! The one directly comparable published tool, `anvi-script-find-
//! misassemblies` (Nature Biotechnology, Jan 2026, Meren lab/anvi'o,
//! "Troubleshooting common errors in assemblies of long-read metagenomes"),
//! is read-mapping-based: it needs a BAM of long reads mapped back onto the
//! assembly and reports zero-coverage regions and clipped-read hotspots.
//! This module needs no reads at all -- only the assembly FASTA -- which
//! matters because most public MAGs (metagenome-assembled genomes) have no
//! deposited reads. The two approaches are complementary evidence, not
//! competitors; see this project's negative-control results (calibration
//! notes, not reproduced in source) for the honest read on how far
//! composition alone gets before read-coverage evidence becomes necessary.
//!
//! # Scope of this first pass: multi-domain chimeras only
//!
//! A chimera fusing a bacterial contig with an archaeal or eukaryotic one is
//! the largest, least ambiguous compositional distance this technique can
//! resolve, and is the only case this module's calibration was run against.
//! Cross-genus-within-domain chimeras (much smaller compositional distance,
//! much easier to false-positive on) are out of scope for this pass.
//!
//! # Algorithm
//!
//! For a contig of length `N`, candidate breakpoints are every position on
//! the grid `window, window + step, window + 2*step, ...` up to `N -
//! window` (so both the window immediately before and the window
//! immediately after a candidate always fit inside the contig). At each
//! candidate, [`divergence_profile`] builds a canonical k-mer frequency
//! distribution for the `window` bases immediately before it and for the
//! `window` bases immediately after it -- reusing [`crate::kmer::
//! extract_canonical_kmers_into`] rather than reimplementing 2-bit packing
//! or canonicalization -- and compares the two with the Jensen-Shannon
//! divergence ([`jensen_shannon_divergence`]), this project's own existing
//! convention for comparing k-mer frequency distributions (see
//! `python/fastdna/validate_generated.py`'s module doc comment, which uses
//! the same measure -- there via `scipy.spatial.distance.jensenshannon`,
//! computed as the *distance* i.e. `sqrt(divergence)` -- to compare
//! generated vs. reference k-mer spectra). This module reports the
//! divergence itself (bits, base-2 log, bounded `[0, 1]`), not its square
//! root, because the calibration sweep this module's threshold is derived
//! from was run against the divergence.
//!
//! [`scan_sequence`] turns a raw divergence profile into flagged
//! [`Breakpoint`]s: every candidate at or above `threshold` is kept, and a
//! run of *consecutive* (adjacent on the step grid) flagged candidates --
//! which is what a single real junction produces, since the window sliding
//! through the transition zone stays elevated for roughly one window's
//! width on either side of it -- is collapsed to the one candidate with the
//! highest divergence in that run, rather than reported as several
//! near-duplicate breakpoints for the same event.
//!
//! # Memory and parallelism
//!
//! Each contig's own sequence is read into memory once and scanned in a
//! single forward pass with two small reused buffers (a `Vec<u64>` and an
//! `FxHashMap<u64, u32>` per side) -- memory does not grow with contig
//! length or with how many candidate breakpoints exist. This is *not* an
//! O(1)-per-base rolling window the way `kmer.rs`'s counting hot path is:
//! each candidate's two frequency tables are rebuilt from scratch, an
//! O(window) cost paid once per `step` bases. That is a deliberate
//! simplicity/robustness trade for a QC/audit tool that runs at
//! contig-count scale (thousands of MAGs, each a handful of contigs), not
//! at the per-base scale `pipeline.rs`'s counting hot loop runs at.
//!
//! Parallelism is *across* contigs, not within one -- [`scan_sequences`]
//! hands each contig to its own rayon task with no shared mutable state and
//! no locks, the same per-worker-private-accumulation shape
//! `pipeline.rs`'s per-record workers already use, just at contig
//! granularity instead of read-batch granularity. [`scan_paths`] reads one
//! input file's contigs into memory before handing that file's contigs to
//! rayon, so peak memory is bounded by the largest single input file, not
//! by the sum of an entire directory of MAGs.

use crate::error::{FastDnaError, Result};
use crate::fastq::FastqReader;
use crate::kmer::extract_canonical_kmers_into;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};

/// Parameters controlling one chimera scan. `window` and `step` are in
/// bases; `k` is the k-mer size used for the composition vector (4, i.e.
/// tetranucleotide composition, is this technique's namesake and this
/// project's calibrated default, but the field is a parameter -- not
/// hardcoded -- because window size and k are exactly the two axes the
/// calibration sweep needs to vary).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChimeraScanParams {
    /// Number of bases compared on each side of a candidate breakpoint.
    pub window: usize,
    /// Distance in bases between one candidate breakpoint and the next.
    pub step: usize,
    /// K-mer size for the composition vector (1..=32, the same limit
    /// [`crate::kmer::extract_canonical_kmers_into`] imposes; 4 for the
    /// tetranucleotide composition this module is calibrated for).
    pub k: usize,
    /// Minimum Jensen-Shannon divergence (bits, `[0, 1]`) for a candidate
    /// breakpoint to be flagged. `None` disables flagging: every candidate
    /// is returned by [`scan_sequence`] as its own breakpoint with
    /// `confidence == divergence`, which is what the calibration sweep
    /// needs (the unfiltered curve, not a pre-thresholded one).
    pub threshold: Option<f64>,
}

impl ChimeraScanParams {
    /// Rejects a configuration that could never produce a meaningful scan,
    /// before any file is opened or any sequence is touched -- this
    /// crate's "bad input fails fast, loudly, and before the run" CLI
    /// philosophy (`CLAUDE.md`), applied to a library entry point that a
    /// Python caller can also reach directly with no CLI parser in front of
    /// it.
    pub fn validate(&self) -> Result<()> {
        if self.k == 0 || self.k > 32 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "k",
                reason: format!(
                    "k must be in 1..=32 (the 2-bit-packing limit every canonical k-mer \
                     extraction in this crate shares), got {}",
                    self.k
                ),
            });
        }
        if self.window == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "window",
                reason: "window must be at least 1 base".to_string(),
            });
        }
        if self.window < self.k {
            return Err(FastDnaError::InvalidConfig {
                parameter: "window",
                reason: format!(
                    "window ({}) must be at least k ({}) -- a window shorter than k can never \
                     contain a single k-mer, so its composition vector would always be empty",
                    self.window, self.k
                ),
            });
        }
        if self.step == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "step",
                reason: "step must be at least 1 base -- a step of 0 would never advance to the \
                          next candidate breakpoint"
                    .to_string(),
            });
        }
        if let Some(t) = self.threshold {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err(FastDnaError::InvalidConfig {
                    parameter: "threshold",
                    reason: format!(
                        "threshold must be a finite value in 0.0..=1.0 (Jensen-Shannon \
                         divergence, base-2 log, is bounded there), got {t}"
                    ),
                });
            }
        }
        Ok(())
    }
}

/// One candidate breakpoint's raw, unfiltered divergence -- [`divergence_
/// profile`]'s output element, before any thresholding or clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivergenceSample {
    /// 0-based position in the contig: the boundary between the "before"
    /// and "after" windows this sample compared (the after-window starts
    /// exactly here).
    pub position: u32,
    /// Jensen-Shannon divergence (bits, base-2 log) between the two
    /// windows' canonical k-mer frequency distributions, in `[0, 1]`.
    pub divergence: f64,
}

/// A flagged chimera candidate -- one row of [`scan_sequence`]/[`scan_
/// sequences`]/[`scan_paths`]'s output. A plain long/tidy shape (one row per
/// candidate breakpoint, `contig_id` a plain string) so it can be joined
/// later against taxonomic annotation by contig id.
#[derive(Debug, Clone, PartialEq)]
pub struct Breakpoint {
    /// Which contig this breakpoint was found in.
    pub contig_id: String,
    /// 0-based position in the contig (see [`DivergenceSample::position`]).
    pub position: u32,
    /// The raw Jensen-Shannon divergence (bits, `[0, 1]`) at this
    /// breakpoint -- the magnitude of the compositional shift.
    pub divergence: f64,
    /// A `[0, 1]` score derived from `divergence` and the threshold that
    /// produced this breakpoint: `0.0` right at the decision boundary,
    /// rising linearly to `1.0` at the maximum possible divergence. When
    /// `params.threshold` was `None` (unfiltered/calibration mode), this
    /// equals `divergence` directly -- there is no boundary to score
    /// against, so the raw magnitude is the only honest number to report.
    pub confidence: f64,
}

/// Jensen-Shannon divergence (bits, base-2 log, symmetric, bounded `[0,
/// 1]`) between two k-mer frequency distributions, each given as a raw
/// occurrence-count map over the same kind of key (a canonical k-mer u64).
/// This is the *divergence*, not the *distance* `python/fastdna/
/// validate_generated.py` reports (that module takes the square root of
/// the same quantity) -- see this module's doc comment for why this module
/// reports the un-rooted divergence.
///
/// `left`/`right` need not share the same key set: a key present in only
/// one side is treated as having probability `0` on the other, the
/// standard convention (a `0 * log(0/x)` term contributes `0`, which is
/// exactly what dropping it below does). An empty map on either side (a
/// window that yielded no canonical k-mers at all -- shorter than `k`, or
/// entirely ambiguous bases) makes the divergence undefined; this returns
/// `0.0` in that case (treated as "no signal observed", not as "maximally
/// divergent") since [`divergence_profile`]'s own window-length invariant
/// already guarantees this only happens for a degenerate, k-mer-free
/// window, not for a genuine comparison.
pub fn jensen_shannon_divergence(left: &FxHashMap<u64, u32>, right: &FxHashMap<u64, u32>) -> f64 {
    let left_total: u64 = left.values().map(|&c| c as u64).sum();
    let right_total: u64 = right.values().map(|&c| c as u64).sum();
    if left_total == 0 || right_total == 0 {
        return 0.0;
    }
    let left_total = left_total as f64;
    let right_total = right_total as f64;

    let mut divergence = 0.0f64;
    for (kmer, &lc) in left {
        let p = lc as f64 / left_total;
        let q = right.get(kmer).copied().unwrap_or(0) as f64 / right_total;
        divergence += kl_term(p, q);
    }
    for (kmer, &rc) in right {
        if left.contains_key(kmer) {
            continue;
        }
        let q = rc as f64 / right_total;
        divergence += kl_term(0.0, q);
    }
    // Floating-point summation of many small terms can overshoot the
    // theoretical [0, 1] bound by a handful of ULPs; clamping keeps the
    // contract exact for callers (in particular `Breakpoint::confidence`'s
    // own `[0, 1]` arithmetic below, which would otherwise silently accept
    // a divergence of 1.0000000000000002).
    divergence.clamp(0.0, 1.0)
}

/// One key's contribution to `0.5 * KL(P||M) + 0.5 * KL(Q||M)` where `M =
/// (P + Q) / 2`, for a single outcome with probabilities `p` under `P` and
/// `q` under `Q`. `p * log2(p / m)` is defined to be `0` when `p == 0` (the
/// standard convention that makes Kullback-Leibler divergence well-defined
/// whenever `Q`'s support covers `P`'s, which a shared `m = (p+q)/2` always
/// does here since `m > 0` whenever `p > 0` or `q > 0`).
#[inline]
fn kl_term(p: f64, q: f64) -> f64 {
    let m = 0.5 * (p + q);
    let mut term = 0.0;
    if p > 0.0 {
        term += 0.5 * p * (p / m).log2();
    }
    if q > 0.0 {
        term += 0.5 * q * (q / m).log2();
    }
    term
}

/// Computes the raw, unclustered divergence profile of `seq`: for every
/// candidate breakpoint on `params.step`'s grid, the Jensen-Shannon
/// divergence between the `params.window` bases immediately before it and
/// the `params.window` bases immediately after it. See this module's own
/// doc comment for the full algorithm and its memory/parallelism
/// properties.
///
/// A contig shorter than `2 * params.window` (there is no position with a
/// full window on both sides) or empty yields an empty result, not an
/// error -- the same "nothing to report, not a failure" convention
/// `kmer::extract_canonical_kmers_into` already uses for a read shorter
/// than `k`.
///
/// `params` is assumed already validated (`ChimeraScanParams::validate`);
/// this function does not re-validate it, matching every other `_into`-
/// style hot-path function in this crate.
pub fn divergence_profile(seq: &[u8], params: &ChimeraScanParams) -> Vec<DivergenceSample> {
    let window = params.window;
    if window == 0 || seq.len() < 2 * window {
        return Vec::new();
    }

    let mut samples = Vec::new();
    let mut left_kmers: Vec<u64> = Vec::new();
    let mut right_kmers: Vec<u64> = Vec::new();
    let mut left_counts: FxHashMap<u64, u32> = FxHashMap::default();
    let mut right_counts: FxHashMap<u64, u32> = FxHashMap::default();

    let mut pos = window;
    while pos + window <= seq.len() {
        let left = &seq[pos - window..pos];
        let right = &seq[pos..pos + window];

        extract_canonical_kmers_into(left, params.k, &mut left_kmers);
        extract_canonical_kmers_into(right, params.k, &mut right_kmers);

        left_counts.clear();
        for &km in &left_kmers {
            *left_counts.entry(km).or_insert(0) += 1;
        }
        right_counts.clear();
        for &km in &right_kmers {
            *right_counts.entry(km).or_insert(0) += 1;
        }

        let divergence = jensen_shannon_divergence(&left_counts, &right_counts);
        // `pos` is bounded by `seq.len()`, and a contig longer than
        // `u32::MAX` bases is not a real assembly on any platform this
        // crate targets -- the same cap `kmer::extract_canonical_kmers_
        // with_positions_into` already imposes on read positions, applied
        // here to contig positions for the same reason (a fixed-width
        // output row).
        samples.push(DivergenceSample { position: pos as u32, divergence });

        pos += params.step;
    }

    samples
}

/// Turns `contig_id`'s divergence profile into flagged [`Breakpoint`]s: see
/// this module's doc comment for the exact flagging/clustering rule.
///
/// When `params.threshold` is `None`, every candidate breakpoint is
/// returned unclustered, one row per candidate, `confidence == divergence`
/// -- this is the mode the calibration sweep runs to see the unfiltered
/// curve, not the mode a caller doing real chimera flagging wants.
pub fn scan_sequence(contig_id: &str, seq: &[u8], params: &ChimeraScanParams) -> Vec<Breakpoint> {
    let profile = divergence_profile(seq, params);

    let Some(threshold) = params.threshold else {
        return profile
            .into_iter()
            .map(|s| Breakpoint {
                contig_id: contig_id.to_string(),
                position: s.position,
                divergence: s.divergence,
                confidence: s.divergence,
            })
            .collect();
    };

    let mut breakpoints = Vec::new();
    let mut cluster_best: Option<usize> = None;

    for (i, sample) in profile.iter().enumerate() {
        if sample.divergence >= threshold {
            cluster_best = Some(match cluster_best {
                Some(best) if profile[best].divergence >= sample.divergence => best,
                _ => i,
            });
        } else if let Some(best) = cluster_best.take() {
            breakpoints.push(make_breakpoint(contig_id, &profile[best], threshold));
        }
    }
    if let Some(best) = cluster_best {
        breakpoints.push(make_breakpoint(contig_id, &profile[best], threshold));
    }

    breakpoints
}

/// Builds one flagged [`Breakpoint`] from a divergence sample known to be
/// `>= threshold`. See [`Breakpoint::confidence`]'s own doc comment for the
/// scoring rule.
fn make_breakpoint(contig_id: &str, sample: &DivergenceSample, threshold: f64) -> Breakpoint {
    let confidence = if threshold >= 1.0 {
        // threshold == 1.0 is the only value for which the linear rescale
        // below divides by zero; the only sample that can reach it is a
        // divergence of exactly 1.0 too (the flagging condition is `>=
        // threshold`), which is unambiguously maximal confidence.
        1.0
    } else {
        ((sample.divergence - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    };
    Breakpoint {
        contig_id: contig_id.to_string(),
        position: sample.position,
        divergence: sample.divergence,
        confidence,
    }
}

/// Scans several contigs in parallel, one rayon task per contig, no shared
/// mutable state -- see this module's doc comment's "Memory and
/// parallelism" section. Order of `contigs` is preserved in the output:
/// every breakpoint of the first contig, then every breakpoint of the
/// second, and so on, matching this crate's convention (`kmer.rs`'s own
/// buffer-reuse tests, `translate.rs`'s row ordering) that output order is
/// a property of the input order, not of thread scheduling.
pub fn scan_sequences(contigs: Vec<(String, Vec<u8>)>, params: &ChimeraScanParams) -> Vec<Breakpoint> {
    contigs
        .into_par_iter()
        .map(|(id, seq)| scan_sequence(&id, &seq, params))
        .collect::<Vec<Vec<Breakpoint>>>()
        .into_iter()
        .flatten()
        .collect()
}

/// The identifier part of a FASTA header: the marker byte (`>`) dropped,
/// then everything up to the first whitespace -- the same convention
/// `samtools faidx` and `ffi.rs::header_to_sequence_id` use. Duplicated
/// here (rather than reused from `ffi.rs`) because `ffi.rs` only compiles
/// under the `python` feature and this is a core, always-compiled module;
/// the logic is ten lines and self-contained enough that sharing it is not
/// worth a new always-public helper elsewhere for one caller on each side.
fn contig_id_from_header(header: &[u8]) -> String {
    let without_marker = match header.first() {
        Some(b'>') | Some(b'@') => &header[1..],
        _ => header,
    };
    let end = without_marker.iter().position(|b| b.is_ascii_whitespace()).unwrap_or(without_marker.len());
    String::from_utf8_lossy(&without_marker[..end]).into_owned()
}

/// Reads every contig of one FASTA/FASTQ(.gz) file into memory as `(contig_
/// id, sequence)` pairs. `contig_id` is `"{file stem}::{header accession}"`
/// -- the file stem prefix keeps ids unique across a directory of MAGs that
/// may reuse generic per-assembly header conventions (`contig_1`,
/// `NODE_1`, ...) across different files.
fn read_contigs(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let mut reader = FastqReader::from_path(path)
        .map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("contig").to_string();

    let mut contigs = Vec::new();
    let mut record_number: u64 = 0;
    loop {
        let record = reader.next_record().map_err(|e| FastDnaError::MalformedFastq {
            path: path.to_path_buf(),
            record: record_number + 1,
            reason: e.to_string(),
        })?;
        let Some(record) = record else { break };
        record_number += 1;

        let contig_id = format!("{stem}::{}", contig_id_from_header(&record.id));
        contigs.push((contig_id, record.seq));
    }
    Ok(contigs)
}

/// Scans every contig of every file in `paths` for chimera candidates.
/// `params` is validated once, up front (`ChimeraScanParams::validate`),
/// before any file is opened.
///
/// Files are read sequentially (I/O), one at a time; each file's contigs
/// are then scanned in parallel (`scan_sequences`) before moving to the
/// next file, so peak memory is bounded by the largest single input file's
/// total sequence bytes, not by the sum over every file in `paths` -- see
/// this module's doc comment's "Memory and parallelism" section.
pub fn scan_paths(paths: &[PathBuf], params: &ChimeraScanParams) -> Result<Vec<Breakpoint>> {
    params.validate()?;

    let mut all_breakpoints = Vec::new();
    for path in paths {
        let contigs = read_contigs(path)?;
        let mut found = scan_sequences(contigs, params);
        all_breakpoints.append(&mut found);
    }
    Ok(all_breakpoints)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    fn default_params() -> ChimeraScanParams {
        ChimeraScanParams { window: 200, step: 50, k: 4, threshold: None }
    }

    // -- ChimeraScanParams::validate ---------------------------------------

    #[test]
    fn validate_rejects_k_out_of_range() {
        let mut p = default_params();
        p.k = 0;
        assert!(p.validate().is_err());
        p.k = 33;
        assert!(p.validate().is_err());
        p.k = 4;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validate_rejects_window_shorter_than_k() {
        let mut p = default_params();
        p.window = 2;
        p.k = 4;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_window_or_step() {
        let mut p = default_params();
        p.window = 0;
        assert!(p.validate().is_err());

        let mut p = default_params();
        p.step = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_rejects_threshold_out_of_range() {
        let mut p = default_params();
        p.threshold = Some(-0.1);
        assert!(p.validate().is_err());
        p.threshold = Some(1.1);
        assert!(p.validate().is_err());
        p.threshold = Some(f64::NAN);
        assert!(p.validate().is_err());
        p.threshold = Some(0.5);
        assert!(p.validate().is_ok());
    }

    // -- jensen_shannon_divergence ------------------------------------------

    #[test]
    fn js_divergence_of_identical_distributions_is_zero() {
        let mut m: FxHashMap<u64, u32> = FxHashMap::default();
        m.insert(1, 5);
        m.insert(2, 5);
        assert_eq!(jensen_shannon_divergence(&m, &m), 0.0);
    }

    #[test]
    fn js_divergence_of_completely_disjoint_distributions_is_one() {
        let mut left: FxHashMap<u64, u32> = FxHashMap::default();
        left.insert(1, 10);
        let mut right: FxHashMap<u64, u32> = FxHashMap::default();
        right.insert(2, 10);
        // Two single-point distributions with disjoint support: the
        // textbook maximal case, JSD == 1 bit exactly.
        assert!((jensen_shannon_divergence(&left, &right) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn js_divergence_is_symmetric() {
        let mut left: FxHashMap<u64, u32> = FxHashMap::default();
        left.insert(1, 7);
        left.insert(2, 3);
        let mut right: FxHashMap<u64, u32> = FxHashMap::default();
        right.insert(2, 6);
        right.insert(3, 4);
        assert_eq!(jensen_shannon_divergence(&left, &right), jensen_shannon_divergence(&right, &left));
    }

    #[test]
    fn js_divergence_of_an_empty_map_is_zero_not_a_panic() {
        let empty: FxHashMap<u64, u32> = FxHashMap::default();
        let mut nonempty: FxHashMap<u64, u32> = FxHashMap::default();
        nonempty.insert(1, 5);
        assert_eq!(jensen_shannon_divergence(&empty, &nonempty), 0.0);
        assert_eq!(jensen_shannon_divergence(&nonempty, &empty), 0.0);
        assert_eq!(jensen_shannon_divergence(&empty, &empty), 0.0);
    }

    // -- divergence_profile / scan_sequence: synthetic ground truth ---------

    /// A deterministic pseudo-random ACGT sequence (xorshift, fixed seed),
    /// used as a stand-in for "genuine, non-repetitive, compositionally
    /// uniform sequence" -- a literal repeated motif would itself create
    /// artifactual periodicity in the k-mer counts that a real genome does
    /// not have.
    fn pseudo_random_dna(len: usize, mut seed: u64) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            // xorshift64
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            out.push(bases[(seed % 4) as usize]);
        }
        out
    }

    #[test]
    fn a_uniform_composition_sequence_produces_no_flagged_breakpoint() {
        // A 200-base window holds only ~197 k-mers spread across ~136
        // canonical tetranucleotide bins -- multinomial sampling noise
        // alone (not any real compositional shift) produces a JS
        // divergence of several tenths of a bit between two *independent*
        // random windows at that size, as this test itself discovered
        // (`default_params`'s window = 200 flagged several candidates at
        // threshold 0.25 on pure noise). A real calibration run sweeps
        // window size for exactly this reason; this test picks a window
        // large enough (1000 bases, ~5x the k-mer sample size) that the
        // noise floor sits well under a still-conservative 0.4 threshold,
        // which is asserted below rather than assumed.
        let seq = pseudo_random_dna(8000, 0xC0FFEE);
        let params = ChimeraScanParams { window: 1000, step: 100, k: 4, threshold: None };

        let profile = divergence_profile(&seq, &params);
        let max_divergence = profile.iter().map(|s| s.divergence).fold(0.0, f64::max);
        assert!(max_divergence < 0.4, "max divergence {max_divergence} was not small on a uniform sequence");

        let mut flagging = params;
        flagging.threshold = Some(0.4);
        let breakpoints = scan_sequence("uniform", &seq, &flagging);
        assert!(
            breakpoints.is_empty(),
            "a compositionally uniform sequence must not produce a flagged breakpoint, got {breakpoints:?}"
        );
    }

    #[test]
    fn a_hand_constructed_compositional_shift_is_detected_at_the_known_position() {
        // Left half: strongly AT-biased ("ATAT" repeated). Right half:
        // strongly GC-biased ("GCGC" repeated). The true junction is at
        // position 1000 by construction.
        let half = 1000;
        let mut seq = Vec::with_capacity(2 * half);
        seq.extend(b"AT".iter().cycle().take(half));
        seq.extend(b"GC".iter().cycle().take(half));
        let true_junction = half as u32;

        let mut params = default_params();
        params.threshold = Some(0.5);

        let breakpoints = scan_sequence("chimera", &seq, &params);
        assert_eq!(
            breakpoints.len(),
            1,
            "a single hand-constructed junction must produce exactly one clustered breakpoint, got {breakpoints:?}"
        );
        let bp = &breakpoints[0];
        // The reported position must land within one window of the true
        // junction -- the cluster's best candidate is wherever the two
        // windows are most cleanly split by the true junction, which is
        // the true junction itself for this construction.
        let distance = (bp.position as i64 - true_junction as i64).abs();
        assert!(distance <= params.window as i64, "reported position {} too far from true junction {true_junction}", bp.position);
        assert!(bp.divergence > 0.9, "divergence at a maximally disjoint composition shift should be near 1.0, got {}", bp.divergence);
        assert!((0.0..=1.0).contains(&bp.confidence));
    }

    #[test]
    fn a_shift_below_threshold_is_not_flagged() {
        // A moderate shift, not a maximal one: both sides share the full
        // A/C/G/T alphabet (unlike the fully-disjoint A/T-vs-G/C
        // construction used elsewhere in this file, which reaches the
        // exact 1.0 divergence bound), so this is guaranteed by
        // construction -- not merely by floating-point luck -- to land
        // strictly below 1.0. Left: "AAAT" repeated (75% A, 25% T). Right:
        // "ATTT" repeated (25% A, 75% T). A real but modest AT-skew shift.
        let half = 1000;
        let mut seq = Vec::with_capacity(2 * half);
        seq.extend(b"AAAT".iter().cycle().take(half));
        seq.extend(b"ATTT".iter().cycle().take(half));

        let params = default_params();
        let profile = divergence_profile(&seq, &params);
        let max_divergence = profile.iter().map(|s| s.divergence).fold(0.0, f64::max);
        assert!(
            max_divergence < 0.9,
            "test assumption violated: this moderate shift's divergence ({max_divergence}) was not \
             comfortably below the threshold used below"
        );

        let mut flagging = params;
        flagging.threshold = Some(0.95);
        let breakpoints = scan_sequence("moderate_shift", &seq, &flagging);
        assert!(
            breakpoints.is_empty(),
            "a shift whose divergence never reaches the threshold must not be flagged, got {breakpoints:?}"
        );
    }

    // -- edge cases -----------------------------------------------------------

    #[test]
    fn a_contig_shorter_than_one_window_yields_no_candidates() {
        let params = default_params(); // window = 200
        let seq = pseudo_random_dna(50, 1);
        assert!(divergence_profile(&seq, &params).is_empty());
        assert!(scan_sequence("short", &seq, &params).is_empty());

        let mut flagging = params;
        flagging.threshold = Some(0.1);
        assert!(scan_sequence("short", &seq, &flagging).is_empty());
    }

    #[test]
    fn a_contig_of_exactly_two_windows_yields_exactly_one_candidate() {
        let params = default_params(); // window = 200, step = 50
        let seq = pseudo_random_dna(400, 2);
        let profile = divergence_profile(&seq, &params);
        assert_eq!(profile.len(), 1);
        assert_eq!(profile[0].position, 200);
    }

    #[test]
    fn an_empty_contig_yields_nothing_and_does_not_panic() {
        let params = default_params();
        assert!(divergence_profile(b"", &params).is_empty());
        assert!(scan_sequence("empty", b"", &params).is_empty());
    }

    #[test]
    fn scan_sequences_preserves_contig_order() {
        let params = default_params();
        let half = 1000;
        let mut chimera = Vec::with_capacity(2 * half);
        chimera.extend(b"AT".iter().cycle().take(half));
        chimera.extend(b"GC".iter().cycle().take(half));
        let uniform = pseudo_random_dna(2000, 42);

        let mut flagging = params;
        flagging.threshold = Some(0.5);

        let contigs = vec![
            ("first".to_string(), chimera.clone()),
            ("second".to_string(), uniform.clone()),
            ("third".to_string(), chimera.clone()),
        ];
        let breakpoints = scan_sequences(contigs, &flagging);
        let ids: Vec<&str> = breakpoints.iter().map(|b| b.contig_id.as_str()).collect();
        assert_eq!(ids, vec!["first", "third"], "order must follow input order, and the uniform contig must contribute nothing");
    }

    // -- scan_paths: end-to-end over a real file -----------------------------

    #[test]
    fn scan_paths_reads_a_fasta_file_and_labels_contigs_by_file_stem_and_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample_mag.fasta");
        let half = 1000;
        let mut file = std::fs::File::create(&path).expect("create fixture");
        writeln!(file, ">contig_1 some description").unwrap();
        let mut seq = Vec::new();
        seq.extend(b"AT".iter().cycle().take(half));
        seq.extend(b"GC".iter().cycle().take(half));
        writeln!(file, "{}", String::from_utf8(seq).unwrap()).unwrap();
        drop(file);

        let params = ChimeraScanParams { window: 200, step: 50, k: 4, threshold: Some(0.5) };
        let breakpoints = scan_paths(&[path], &params).expect("scan_paths must succeed");
        assert_eq!(breakpoints.len(), 1);
        assert_eq!(breakpoints[0].contig_id, "sample_mag::contig_1");
    }

    #[test]
    fn scan_paths_rejects_an_invalid_configuration_before_touching_any_file() {
        let params = ChimeraScanParams { window: 200, step: 50, k: 0, threshold: None };
        // A nonexistent path: if validation ran after the file open, this
        // would fail with an I/O error instead of InvalidConfig.
        let missing = PathBuf::from("this/path/does/not/exist.fasta");
        let err = scan_paths(&[missing], &params).expect_err("must fail validation");
        assert!(matches!(err, FastDnaError::InvalidConfig { parameter: "k", .. }));
    }
}
