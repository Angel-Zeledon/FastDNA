// src/adaptive_bins.rs

//! A data-adaptive replacement for the binned strategy's static
//! hash-to-bin map, and the fix for the R3 finding in
//! `docs/design-minimizer-counting.md`.
//!
//! # The finding this module exists to fix
//!
//! Commit `41675fd` ran the binned strategy's real-data gate on two real ENA
//! samples. A bacterial WGS shotgun sample was well-behaved (5.64x max/mean
//! bin skew), but a 16S amplicon sample -- real amplicon reads are ~250 bp of
//! a single, near-identical conserved-region sequence repeated across nearly
//! every read -- put 11.6% of the entire super-k-mer store into one of 512
//! bins: **1516x max/median skew**. `minimizer::bin_of` masks the high bits
//! of a hashed signature, which is uniform *over the signature space*, but
//! says nothing about how often each signature actually occurs in a given
//! input; a handful of signatures dominating a low-diversity sample sail
//! straight through the mask into whichever few bins their hash happens to
//! land in; there's no mechanism to notice they're heavy. That is exactly
//! the case the KMC2 paper (SS2.4) describes and defends against with a
//! sampling pass, and this module builds the same kind of fix.
//!
//! # Why the signature space is small enough to histogram directly
//!
//! A signature is `mix64(canonical_mmer)` (`minimizer::signature_of_kmer`),
//! and `mix64` is a bijection (`minimizer::mix64`'s doc comment). So the set
//! of signature values a real run can ever produce has exactly the
//! cardinality of the set of eligible canonical m-mers: at most `4^m`, and
//! roughly half that after canonicalisation. At the crate's default `m = 7`
//! that is on the order of 8,000 distinct values -- small enough that a
//! `HashMap` keyed by signature holds an *exact* histogram of everything
//! observed, not a lossy sketch. [`MAX_TRACKED_SIGNATURES`] exists only as a
//! safety net for a caller who configures an unusually large `m`; it is not
//! load-bearing at the defaults.
//!
//! # Why this cannot make a run's counts wrong, even if the sampling is
//! # naive
//!
//! [`DynamicBinMap::bin_of`] is a total, deterministic function of the
//! signature alone -- same signature, same bin, every time it is consulted
//! during one run. That is the only property the counting path depends on
//! (`binned.rs`'s module doc comment: `bin(x) == bin(rc(x))`, and every
//! occurrence of one canonical k-mer reaching exactly one bin). Which bin a
//! signature lands in changes only how work is *distributed*, never what the
//! final merged table contains: `binned.rs::BinStore::finish` runs a genuine
//! k-way merge over the bins (`counter::k_way_merge_sorted_counts`), not a
//! concatenation that depends on bin order lining up with k-mer order --
//! that ordering trick belongs to `disk_spill.rs`'s high-bit bucketing, not
//! to this module (see `disk_spill.rs`'s doc comment for why bucket order
//! matters *there*, and does not here). So a bug in the histogram or the
//! greedy packer below could only ever degrade load balance, never
//! correctness -- a property `binned_counts_match_kmer_counter_exactly_*`
//! keeps checking regardless of which bin map produced the answer.
//!
//! # The algorithm
//!
//! KMC2-style: sample a bounded prefix of the input, build an exact
//! frequency histogram of the signatures observed (weighted by super-k-mer
//! bytes, matching the metric `binned.rs::BinStore::occupancy` reports and
//! the one the real-data gate measured skew against), then run a greedy
//! longest-processing-time-first (LPT) bin-packing: heaviest signature
//! first, always placed into the currently lightest bin. LPT is a
//! textbook 4/3-approximation to the optimal makespan for this exact
//! problem (minimize the maximum load over `n` bins given a fixed list of
//! weighted, non-splittable items) -- an exact optimum is NP-hard, and this
//! module does not need to reach for it: cutting a real-data skew of
//! 1516x down by even a very loose approximation is already the whole point.
//! Signatures the sample never saw fall back to the original static
//! `minimizer::bin_of`, so an input dominated by material the sample missed
//! degrades to today's behaviour rather than to something undefined.

use std::collections::HashMap;

use crate::minimizer::{bin_of as static_bin_of, INELIGIBLE_SIGNATURE};
use crate::superkmer::for_each_superkmer;

/// A safety cap on the number of distinct signatures one histogram tracks.
///
/// Not load-bearing at the crate's default `m = 7` (see the module doc
/// comment: the whole signature space is on the order of 8,000 values
/// there), but without a cap, a caller who configures a large `m` could
/// make the sampling pass itself grow without bound on adversarial or
/// highly diverse input. Once the cap is reached, a newly seen signature's
/// weight is simply not tracked -- it falls back to
/// [`DynamicBinMap::bin_of`]'s static path, which is exactly today's
/// behaviour for it, rather than to an approximation that could be wrong in
/// a way that is hard to reason about.
pub const MAX_TRACKED_SIGNATURES: usize = 1 << 20;

/// An exact (up to [`MAX_TRACKED_SIGNATURES`]) histogram of super-k-mer byte
/// weight per signature, accumulated over a sample of sequences.
#[derive(Debug, Default, Clone)]
pub struct SignatureHistogram {
    weights: HashMap<u64, u64>,
}

impl SignatureHistogram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scans one already quality-trimmed (and, if enabled,
    /// homopolymer-compressed) sequence -- the same preprocessing the real
    /// counting worker applies before it ever reaches a bin -- and adds each
    /// super-k-mer's byte length to its signature's running weight.
    ///
    /// This drives `superkmer::for_each_superkmer` with the same `k` and `m`
    /// the counting path uses, so the sampled distribution is a measurement
    /// of the exact windows phase 1 will produce, not an approximation of
    /// them from a different scan.
    pub fn observe_sequence(&mut self, seq: &[u8], k: usize, m: usize) {
        for_each_superkmer(seq, k, m, |_start, bases, signature| {
            if signature == INELIGIBLE_SIGNATURE {
                // Bin 0 is a dedicated, fixed overflow bin for these
                // (`minimizer.rs`'s doc comment); they are never part of the
                // packing problem below, so tracking their weight here would
                // only cost memory for no benefit.
                return;
            }
            if self.weights.len() >= MAX_TRACKED_SIGNATURES && !self.weights.contains_key(&signature) {
                return;
            }
            *self.weights.entry(signature).or_insert(0) += bases.len() as u64;
        });
    }

    /// Whether anything was observed at all -- an empty sample (a tiny or
    /// entirely-fallback input) has no basis for an adaptive map, and
    /// [`Self::build_bin_map`] returns the identity map in that case.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The number of distinct signatures currently tracked.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Builds a [`DynamicBinMap`] that balances this histogram's weight
    /// across `num_bins` bins.
    pub fn build_bin_map(&self, num_bins: usize) -> DynamicBinMap {
        DynamicBinMap::from_histogram(self, num_bins)
    }
}

/// A signature-to-bin map that overrides the static
/// `minimizer::bin_of` for the signatures a sample identified as heavy,
/// falling back to it for everything else.
///
/// Total and deterministic: [`Self::bin_of`] is a pure function of the
/// signature and never changes once built, which is what keeps the
/// disjoint-bin invariant `binned.rs` depends on intact regardless of which
/// map produced the assignment (see the module doc comment).
#[derive(Debug, Clone)]
pub struct DynamicBinMap {
    overrides: HashMap<u64, u32>,
    num_bins: usize,
}

impl DynamicBinMap {
    /// The static map, unchanged. Every signature falls through to
    /// `minimizer::bin_of`; this is what a `BinStore` built without an
    /// adaptive sample uses, so today's behaviour is this type's zero case
    /// rather than a separate code path.
    pub fn identity(num_bins: usize) -> Self {
        Self { overrides: HashMap::new(), num_bins }
    }

    /// Builds the greedy assignment described in the module doc comment.
    ///
    /// Falls back to [`Self::identity`] when there is nothing to balance
    /// against (an empty histogram) or nowhere to balance across
    /// (`num_bins <= 1`, where every signature has exactly one place to go
    /// regardless).
    pub fn from_histogram(hist: &SignatureHistogram, num_bins: usize) -> Self {
        if hist.weights.is_empty() || num_bins <= 1 {
            return Self::identity(num_bins);
        }

        // Heaviest first: LPT's approximation guarantee depends on placing
        // the largest items before the small ones fill in the gaps.
        // Signature value is a secondary key purely for determinism: two
        // signatures with equal sampled weight must still sort the same way
        // regardless of the `HashMap`'s iteration order, or the assignment
        // -- and therefore the whole run's bin contents -- would depend on
        // hash-seed-driven iteration order rather than only on the input.
        let mut entries: Vec<(u64, u64)> = hist.weights.iter().map(|(&sig, &w)| (sig, w)).collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Bin 0 is reserved as the dedicated overflow bin for
        // `INELIGIBLE_SIGNATURE` (`minimizer.rs`'s doc comment) and is never
        // a candidate here, so the packing only ever considers bins
        // `1..num_bins`.
        let mut loads = vec![0u64; num_bins];
        let mut overrides = HashMap::with_capacity(entries.len());

        for (signature, weight) in entries {
            let lightest = loads
                .iter()
                .enumerate()
                .skip(1)
                .min_by_key(|&(_, &load)| load)
                .map(|(bin, _)| bin)
                .unwrap_or(0);
            overrides.insert(signature, lightest as u32);
            loads[lightest] = loads[lightest].saturating_add(weight);
        }

        Self { overrides, num_bins }
    }

    /// The bin one signature routes to: the dedicated overflow bin for
    /// [`INELIGIBLE_SIGNATURE`], this map's override when the sample saw the
    /// signature, or the static `minimizer::bin_of` otherwise.
    #[inline]
    pub fn bin_of(&self, signature: u64) -> usize {
        if signature == INELIGIBLE_SIGNATURE {
            return 0;
        }
        match self.overrides.get(&signature) {
            Some(&bin) => bin as usize,
            None => static_bin_of(signature, self.num_bins),
        }
    }

    /// The bin count this map was built for.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// How many signatures carry an explicit override rather than falling
    /// back to the static map. Diagnostic only.
    pub fn overrides_len(&self) -> usize {
        self.overrides.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::minimizer::DEFAULT_NUM_BINS;

    // ---------------------------------------------------------------
    // SignatureHistogram
    // ---------------------------------------------------------------

    #[test]
    fn an_empty_sample_produces_an_empty_histogram() {
        let hist = SignatureHistogram::new();
        assert!(hist.is_empty());
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn a_pure_homopolymer_sequence_is_not_tracked() {
        let mut hist = SignatureHistogram::new();
        hist.observe_sequence(&[b'A'; 100], 31, 7);
        assert!(hist.is_empty(), "every window is the ineligible fallback; nothing should be tracked");
    }

    #[test]
    fn a_repeated_sequence_concentrates_weight_on_few_signatures() {
        let mut hist = SignatureHistogram::new();
        // The amplicon-style case: the same short sequence, over and over.
        let seq = b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCG";
        for _ in 0..500 {
            hist.observe_sequence(seq, 21, 7);
        }
        assert!(!hist.is_empty());
        // A single 43-base sequence has only a handful of 21-mer windows;
        // the distinct signature count must stay tiny even after 500 repeats.
        assert!(hist.len() < 20, "got {} distinct signatures from one repeated short sequence", hist.len());
    }

    // ---------------------------------------------------------------
    // DynamicBinMap
    // ---------------------------------------------------------------

    #[test]
    fn identity_map_matches_the_static_function_everywhere() {
        let map = DynamicBinMap::identity(DEFAULT_NUM_BINS);
        for sig in [0u64, 1, 42, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
            assert_eq!(map.bin_of(sig), static_bin_of(sig, DEFAULT_NUM_BINS));
        }
    }

    #[test]
    fn an_empty_histogram_builds_the_identity_map() {
        let hist = SignatureHistogram::new();
        let map = hist.build_bin_map(DEFAULT_NUM_BINS);
        assert_eq!(map.overrides_len(), 0);
        for sig in [1u64, 999_999, 0xABCD] {
            assert_eq!(map.bin_of(sig), static_bin_of(sig, DEFAULT_NUM_BINS));
        }
    }

    #[test]
    fn a_single_bin_always_returns_the_identity_map() {
        let mut hist = SignatureHistogram::new();
        hist.observe_sequence(b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCG", 21, 7);
        let map = hist.build_bin_map(1);
        assert_eq!(map.overrides_len(), 0);
    }

    #[test]
    fn the_ineligible_signature_always_routes_to_bin_zero_even_with_overrides() {
        let mut hist = SignatureHistogram::new();
        hist.observe_sequence(b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCG", 21, 7);
        let map = hist.build_bin_map(8);
        assert_eq!(map.bin_of(INELIGIBLE_SIGNATURE), 0);
    }

    #[test]
    fn every_bin_of_result_is_in_range_for_many_configurations() {
        let mut hist = SignatureHistogram::new();
        let seq = b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCGGGACATCAGT";
        for _ in 0..50 {
            hist.observe_sequence(seq, 25, 9);
        }
        for num_bins in [2usize, 4, 8, 64, DEFAULT_NUM_BINS] {
            let map = hist.build_bin_map(num_bins);
            for sig in [0u64, 1, 42, u64::MAX, 0x1234_5678_9ABC_DEF0] {
                assert!(map.bin_of(sig) < num_bins, "out of range for {num_bins} bins");
            }
        }
    }

    /// The property that actually matters: a synthetic input reproducing
    /// the real-data failure -- one dominant sequence repeated across
    /// nearly every read, so a handful of signatures carry almost all the
    /// weight -- must never come out *more* skewed than the static map,
    /// however few distinct signatures it reduces to (a sequence short
    /// enough to be a single super-k-mer has exactly one signature, and no
    /// packing strategy can spread a single item across more than one bin --
    /// that degenerate case is the reason this only asserts "not worse",
    /// not "better"; the mechanism that actually separates colliding heavy
    /// signatures is pinned directly by
    /// `greedy_assignment_separates_two_equally_heavy_signatures_that_collide_under_the_static_map`
    /// below).
    #[test]
    fn greedy_assignment_is_never_worse_than_the_static_map_on_a_dominant_repeated_sequence() {
        let dominant = b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCGGGACATCAGTTTGACCAACGGTTCAGGATCGACTGACTGATCGATTAGCTAGCATCGTAGCTAGCATCGATCGATCGTAGCTACGATCGATCGGGCATGCATCGATGCTAGCTGATCGATCGTAGCTAGCTACGATCG";
        let k = 31usize;
        let m = 7usize;
        let num_bins = DEFAULT_NUM_BINS;

        let mut hist = SignatureHistogram::new();
        // The amplicon case: the same conserved sequence, repeated many
        // times, is essentially all of the input.
        for _ in 0..2_000 {
            hist.observe_sequence(dominant, k, m);
        }
        assert!(!hist.is_empty());
        // A single super-k-mer would make this test unable to say anything
        // about redistribution at all; require the input to actually carry
        // more than one distinct signature; the collision test below covers
        // the exact mechanism at any count, including one.
        assert!(hist.len() > 1, "test input must produce more than one distinct signature, got {}", hist.len());

        let adaptive = hist.build_bin_map(num_bins);

        // Replay the exact same weighted signatures against both maps and
        // compare the resulting per-bin load.
        let load_under = |map: &dyn Fn(u64) -> usize| -> Vec<u64> {
            let mut loads = vec![0u64; num_bins];
            let mut scratch = SignatureHistogram::new();
            scratch.observe_sequence(dominant, k, m);
            for (&sig, &w) in &scratch.weights {
                loads[map(sig)] += w * 2_000;
            }
            loads
        };

        let static_loads = load_under(&|sig| static_bin_of(sig, num_bins));
        let adaptive_loads = load_under(&|sig| adaptive.bin_of(sig));

        let skew = |loads: &[u64]| -> f64 {
            let total: u64 = loads.iter().sum();
            let mean = total as f64 / loads.len() as f64;
            let max = loads.iter().copied().max().unwrap_or(0) as f64;
            if mean == 0.0 {
                0.0
            } else {
                max / mean
            }
        };

        let static_skew = skew(&static_loads);
        let adaptive_skew = skew(&adaptive_loads);
        println!("static max/mean skew: {static_skew:.2}x, adaptive max/mean skew: {adaptive_skew:.2}x");

        assert!(
            adaptive_skew <= static_skew,
            "adaptive assignment must not be worse than the static map: static={static_skew:.2}x adaptive={adaptive_skew:.2}x"
        );
    }

    /// The mechanism itself, isolated from any real sequence's idiosyncrasies:
    /// two equally heavy signatures engineered to land in the same bin under
    /// the static mask (`minimizer::bin_of` keys only on `signature >> 32`)
    /// must be placed in *different* bins by the greedy assignment. This is
    /// the exact failure the real 16S amplicon sample hit -- several heavy
    /// signatures happening to share a bucket by hash chance -- reproduced
    /// deterministically rather than hoped for from a synthetic sequence.
    #[test]
    fn greedy_assignment_separates_two_equally_heavy_signatures_that_collide_under_the_static_map() {
        let num_bins = 8usize;
        // High halves 8 and 16 both reduce to 0 under `& (num_bins - 1) = & 7`.
        let sig_a: u64 = (8u64 << 32) | 1;
        let sig_b: u64 = (16u64 << 32) | 2;
        assert_eq!(
            static_bin_of(sig_a, num_bins),
            static_bin_of(sig_b, num_bins),
            "test setup: these two signatures must collide under the static map"
        );

        let mut hist = SignatureHistogram::default();
        hist.weights.insert(sig_a, 1_000);
        hist.weights.insert(sig_b, 1_000);

        let map = hist.build_bin_map(num_bins);
        assert_ne!(
            map.bin_of(sig_a),
            map.bin_of(sig_b),
            "two equally heavy signatures that collide under the static map must be separated"
        );
    }

    /// Weight ties must not make the assignment depend on `HashMap`
    /// iteration order: the same histogram must always build the same map.
    #[test]
    fn building_the_same_histogram_twice_produces_the_same_map() {
        let mut hist = SignatureHistogram::new();
        let seq = b"ACGTGGTCAGATCCAGTCGATGCATCGATCGATCGATCGATCGGGACATCAGT";
        for _ in 0..30 {
            hist.observe_sequence(seq, 21, 7);
        }

        let a = hist.build_bin_map(64);
        let b = hist.build_bin_map(64);
        for sig in 0u64..5000 {
            assert_eq!(a.bin_of(sig), b.bin_of(sig));
        }
    }
}
