// src/sketch.rs
//! MinHash sketching for fast, approximate genome-to-genome comparison
//! (design doc §8). Instead of counting every k-mer, a `GenomeSketch` keeps
//! only the `sketch_size` smallest hashes as a fingerprint; comparing two
//! fingerprints estimates how similar the full k-mer sets are without ever
//! materializing them side by side. Two SARS-CoV-2 samples that would take
//! minutes to compare exactly can be compared in milliseconds this way.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// MinHash sketch representation for rapid genomic distance estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeSketch {
    pub sketch_size: usize,
    pub k: usize,
    pub hashes: Vec<u64>,
}

/// splitmix64's output mixer (Steele, Lea & Flood, 2014; public domain) --
/// multiply, xorshift, multiply, xorshift, final xorshift. Bottom-k
/// selection picks k-mers purely by the numeric value of their hash, so the
/// quality of this mixing step directly determines how well the sketch's
/// overlap estimates Jaccard similarity: the previous finalizer was a bare
/// `wrapping_mul`, i.e. a linear congruential step, which clusters values
/// (e.g. every input sharing a low bit pattern keeps sharing one) and biases
/// which k-mers end up in the "smallest" bucket. This is cheap enough to run
/// per k-mer on the hot path -- two multiplies and three xorshifts, no
/// branches, no allocation.
#[inline(always)]
fn finalize_hash(kmer: u64) -> u64 {
    let mut z = kmer;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Folds one already-finalized hash into a bottom-`sketch_size` working set,
/// evicting the current maximum once the set is full. Extracted out of
/// `from_kmers` so a later streaming construction path can share exactly
/// this selection logic.
#[inline]
fn insert_bottom_k(min_set: &mut BTreeSet<u64>, sketch_size: usize, hash: u64) {
    if sketch_size == 0 {
        return;
    }
    if min_set.len() < sketch_size {
        min_set.insert(hash);
    } else if let Some(&max_val) = min_set.iter().next_back() {
        if hash < max_val && min_set.insert(hash) {
            min_set.pop_last();
        }
    }
}

impl GenomeSketch {
    pub fn new(sketch_size: usize, k: usize) -> Self {
        Self { sketch_size, k, hashes: Vec::with_capacity(sketch_size) }
    }

    pub fn from_kmers(kmers: &[u64], sketch_size: usize, k: usize) -> Self {
        let mut min_set: BTreeSet<u64> = BTreeSet::new();

        for &kmer in kmers {
            insert_bottom_k(&mut min_set, sketch_size, finalize_hash(kmer));
        }

        Self { sketch_size, k, hashes: min_set.into_iter().collect() }
    }

    pub fn jaccard_similarity(&self, other: &GenomeSketch) -> f64 {
        assert_eq!(self.k, other.k, "k-mer sizes must match");
        let mut i = 0;
        let mut j = 0;
        let mut intersection = 0usize;
        let mut union_count = 0usize;

        while i < self.hashes.len() && j < other.hashes.len() && union_count < self.sketch_size {
            if self.hashes[i] == other.hashes[j] {
                intersection += 1;
                i += 1;
                j += 1;
            } else if self.hashes[i] < other.hashes[j] {
                i += 1;
            } else {
                j += 1;
            }
            union_count += 1;
        }

        if union_count == 0 {
            0.0
        } else {
            intersection as f64 / union_count as f64
        }
    }
}

#[cfg(test)]
// Matches the established pattern in `preview.rs`/`progress.rs`:
// `unwrap`/`expect` are denied under `src/` because production code must
// never panic on caller input, but that rule is not about test assertions,
// where spelling every check as a `match` would obscure what is tested.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn identical_inputs_give_jaccard_one() {
        let kmers: Vec<u64> = (0..2_000).collect();
        let a = GenomeSketch::from_kmers(&kmers, 256, 21);
        let b = GenomeSketch::from_kmers(&kmers, 256, 21);

        let j = a.jaccard_similarity(&b);
        assert_eq!(j, 1.0, "a sketch compared with itself must be exactly 1.0");
    }

    #[test]
    fn disjoint_inputs_give_jaccard_zero() {
        let a_kmers: Vec<u64> = (0..2_000).collect();
        let b_kmers: Vec<u64> = (2_000..4_000).collect();
        let a = GenomeSketch::from_kmers(&a_kmers, 256, 21);
        let b = GenomeSketch::from_kmers(&b_kmers, 256, 21);

        let j = a.jaccard_similarity(&b);
        assert_eq!(j, 0.0, "no shared k-mer means no shared hash, so intersection must be exactly zero");
    }

    /// The estimator test: two 100,000-element universes overlapping by
    /// exactly 50,000, giving a true Jaccard of 50,000 / 150,000 = 1/3. With
    /// `sketch_size = 512`, the bottom-k Jaccard estimator's standard error
    /// is approximately `sqrt(J * (1-J) / sketch_size)`
    /// = `sqrt((1/3) * (2/3) / 512)` ~= 2.08%. The tolerance below is set to
    /// roughly 4 standard errors (~8.3%, rounded up to 9 percentage points)
    /// -- tight enough that a broken or heavily biased estimator (the old
    /// bare `wrapping_mul` finalizer, or a bug that always returns zero or
    /// one) would fail this test, but loose enough that it is not testing
    /// for bit-for-bit reproduction of a single hash draw, which is the
    /// wrong thing to demand of a probabilistic estimator.
    #[test]
    fn known_overlap_estimates_jaccard_within_a_stated_tolerance() {
        let a_kmers: Vec<u64> = (0..100_000).collect();
        let b_kmers: Vec<u64> = (50_000..150_000).collect();
        let sketch_size = 512;

        let a = GenomeSketch::from_kmers(&a_kmers, sketch_size, 21);
        let b = GenomeSketch::from_kmers(&b_kmers, sketch_size, 21);

        let estimate = a.jaccard_similarity(&b);
        let true_jaccard = 50_000.0 / 150_000.0;
        let tolerance = 0.09;

        assert!(
            (estimate - true_jaccard).abs() < tolerance,
            "estimate {estimate} too far from true Jaccard {true_jaccard} (tolerance {tolerance})"
        );
    }

    #[test]
    fn finalize_hash_is_a_bijection_on_a_sample_of_inputs() {
        // A weak/linear finalizer (the original `wrapping_mul`) can map
        // distinct inputs to the same output when they share low bits with
        // the multiplier; splitmix64's mixer must not, over a reasonably
        // sized sample. This directly targets the failure mode the task
        // calls out: "a weak hash clusters values and biases the estimate".
        let mut seen = std::collections::HashSet::new();
        for kmer in 0u64..10_000 {
            assert!(seen.insert(finalize_hash(kmer)), "collision at input {kmer}");
        }
    }
}
