// src/cms.rs

use serde::{Deserialize, Serialize};

use crate::error::{FastDnaError, Result};

/// Bounded-memory probabilistic frequency estimator using Count-Min Sketch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMinSketch {
    depth: usize,
    width: usize,
    table: Vec<Vec<u32>>,
    seeds: Vec<u64>,
}

impl CountMinSketch {
    /// Creates a Count-Min Sketch with fixed memory allocation.
    ///
    /// Rejects degenerate dimensions instead of deferring the failure:
    /// `width == 0` used to panic with a divide-by-zero on the first
    /// `insert`, and `depth == 0` made `estimate` return `u32::MAX` for
    /// every key -- silent, maximally wrong frequency estimates.
    pub fn new(depth: usize, width: usize) -> Result<Self> {
        if depth == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "depth",
                reason: "a Count-Min Sketch needs at least one row".to_string(),
            });
        }
        if width == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "width",
                reason: "a Count-Min Sketch needs at least one counter per row".to_string(),
            });
        }

        let mut seeds = Vec::with_capacity(depth);
        let mut seed = 0x9E3779B97F4A7C15u64;
        for _ in 0..depth {
            // Force every multiplier odd: `wrapping_mul` by an even seed
            // zeroes the low bits of the product and discards the same
            // number of the key's top bits outright, making that row a
            // measurably weaker hash. The additive walk below alternates
            // parity, so without `| 1` half the rows were degraded.
            seeds.push(seed | 1);
            seed = seed.wrapping_add(0x517CC1B727220A95);
        }

        Ok(Self {
            depth,
            width,
            table: vec![vec![0; width]; depth],
            seeds,
        })
    }

    /// Default 16 MB sketch (depth=4, width=1,048,576).
    pub fn default_16mb() -> Self {
        // The fixed dimensions are trivially valid, so this cannot actually
        // fail; the fallback only exists to avoid unwrap in production code.
        Self::new(4, 1_048_576).unwrap_or(Self {
            depth: 0,
            width: 0,
            table: Vec::new(),
            seeds: Vec::new(),
        })
    }

    #[inline(always)]
    fn hash(&self, row: usize, kmer: u64) -> usize {
        let h = kmer.wrapping_mul(self.seeds[row]);
        ((h ^ (h >> 32)) as usize) % self.width
    }

    /// Increments the frequency count for a canonical k-mer.
    #[inline(always)]
    pub fn insert(&mut self, kmer: u64) {
        for row in 0..self.depth {
            let col = self.hash(row, kmer);
            self.table[row][col] = self.table[row][col].saturating_add(1);
        }
    }

    /// Estimates the frequency count for a k-mer (guaranteed to be >= true count).
    #[inline(always)]
    pub fn estimate(&self, kmer: u64) -> u32 {
        let mut min_count = u32::MAX;
        for row in 0..self.depth {
            let col = self.hash(row, kmer);
            min_count = min_count.min(self.table[row][col]);
        }
        min_count
    }

    /// Merges another sketch of identical dimensions and seeds.
    ///
    /// A dimension or seed mismatch is a caller mistake, reported as an
    /// error rather than a panic: this is a `pub` type on the library
    /// surface, and a panic here would cross a future FFI boundary as an
    /// unrecoverable exception (the same contract `sketch::jaccard` and
    /// `hll::merge` already follow). Mismatched *seeds* with matching
    /// dimensions would not panic at all -- they would silently add counts
    /// hashed under different functions, corrupting every later estimate.
    pub fn merge(&mut self, other: &CountMinSketch) -> Result<()> {
        if self.depth != other.depth || self.width != other.width {
            return Err(FastDnaError::InvalidConfig {
                parameter: "cms dimensions",
                reason: format!(
                    "cannot merge a {}x{} sketch into a {}x{} one",
                    other.depth, other.width, self.depth, self.width
                ),
            });
        }
        if self.seeds != other.seeds {
            return Err(FastDnaError::InvalidConfig {
                parameter: "cms seeds",
                reason: "sketches were built with different hash seeds".to_string(),
            });
        }

        for row in 0..self.depth {
            for col in 0..self.width {
                self.table[row][col] = self.table[row][col].saturating_add(other.table[row][col]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
// Same rationale as `sketch.rs`'s tests: the crate-wide unwrap/expect
// denial is about production code paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn zero_width_is_rejected_instead_of_a_deferred_division_panic() {
        match CountMinSketch::new(4, 0) {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "width"),
            other => panic!("width=0 must be InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn zero_depth_is_rejected_instead_of_estimating_u32_max_for_everything() {
        match CountMinSketch::new(0, 16) {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "depth"),
            other => panic!("depth=0 must be InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn estimate_is_at_least_the_true_count() {
        let mut cms = CountMinSketch::new(4, 1024).unwrap();
        for _ in 0..7 {
            cms.insert(42);
        }
        assert!(cms.estimate(42) >= 7, "CMS must never underestimate");
    }

    #[test]
    fn every_seed_is_odd_so_no_row_hash_is_degraded() {
        let cms = CountMinSketch::new(8, 16).unwrap();
        for (row, seed) in cms.seeds.iter().enumerate() {
            assert!(seed % 2 == 1, "seed for row {row} is even: {seed:#x}");
        }
    }

    #[test]
    fn merging_mismatched_dimensions_is_an_error_not_a_panic() {
        let mut a = CountMinSketch::new(4, 1024).unwrap();
        let b = CountMinSketch::new(4, 2048).unwrap();
        match a.merge(&b) {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => {
                assert_eq!(parameter, "cms dimensions");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn merging_identical_dimensions_adds_counts() {
        let mut a = CountMinSketch::new(4, 1024).unwrap();
        let mut b = CountMinSketch::new(4, 1024).unwrap();
        a.insert(7);
        b.insert(7);
        b.insert(7);
        a.merge(&b).unwrap();
        assert!(a.estimate(7) >= 3);
    }

    #[test]
    fn default_16mb_is_usable() {
        let mut cms = CountMinSketch::default_16mb();
        cms.insert(99);
        assert!(cms.estimate(99) >= 1);
    }
}
