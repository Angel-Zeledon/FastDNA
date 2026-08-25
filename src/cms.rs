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

    /// Increments the frequency count for a canonical k-mer.
    ///
    /// Iterates `seeds` and `table` as zipped slices rather than indexing
    /// both by `0..self.depth`. Per row that removes: the `self.seeds[row]`
    /// bounds check, the `self.table[row]` bounds check, and the address
    /// arithmetic for both -- 2 compare-and-branch pairs per row, so 8 per
    /// insert at the default depth of 4, on a path that runs once per
    /// k-mer. `width` is hoisted out of the loop for the same reason it
    /// cannot be hoisted automatically: the writes go through `table`'s heap
    /// buffer, which the optimizer cannot prove does not alias the `width`
    /// field, so it must otherwise reload it every row.
    ///
    /// The column mapping itself is untouched, so the table contents and
    /// every later `estimate` are bit-identical to before.
    #[inline(always)]
    pub fn insert(&mut self, kmer: u64) {
        debug_assert_eq!(self.seeds.len(), self.depth);
        debug_assert_eq!(self.table.len(), self.depth);
        let width = self.width;
        for (&seed, row) in self.seeds.iter().zip(self.table.iter_mut()) {
            let col = column(seed, width, kmer);
            row[col] = row[col].saturating_add(1);
        }
    }

    /// Estimates the frequency count for a k-mer (guaranteed to be >= true count).
    ///
    /// Same slice-zip treatment as `insert`, for the same per-row saving.
    #[inline(always)]
    pub fn estimate(&self, kmer: u64) -> u32 {
        debug_assert_eq!(self.seeds.len(), self.depth);
        debug_assert_eq!(self.table.len(), self.depth);
        let width = self.width;
        let mut min_count = u32::MAX;
        for (&seed, row) in self.seeds.iter().zip(self.table.iter()) {
            min_count = min_count.min(row[column(seed, width, kmer)]);
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

        // Zipped slice walk instead of `self.table[row][col]` /
        // `other.table[row][col]` indexing. The dimension guard above
        // already proved the shapes match, so every one of those bounds
        // checks was provably redundant -- and there are a lot of them: at
        // the `default_16mb` shape (4 x 1,048,576) the old loop performed
        // 4,194,304 compare-and-branch pairs for the two inner indexings
        // alone. Removing them also leaves a straight-line
        // `u32::saturating_add` over two contiguous slices, the shape the
        // vectorizer can actually widen; the indexed form's panic edges
        // blocked that.
        debug_assert_eq!(self.table.len(), other.table.len());
        for (self_row, other_row) in self.table.iter_mut().zip(other.table.iter()) {
            debug_assert_eq!(self_row.len(), other_row.len());
            for (a, &b) in self_row.iter_mut().zip(other_row.iter()) {
                *a = a.saturating_add(b);
            }
        }
        Ok(())
    }
}

/// The column a k-mer maps to in one row, given that row's seed.
///
/// Free-standing rather than a `&self` method so `insert` and `estimate` can
/// walk `seeds`/`table` as zipped slices while still sharing a single
/// definition of the mapping. The arithmetic is character-for-character the
/// one the previous `CountMinSketch::hash` method computed.
///
/// The `%` is the expensive part: a 64-bit hardware division, roughly 20-40
/// cycles of latency, executed `depth` times per `insert` and per
/// `estimate`. It cannot be removed without changing behaviour, because a
/// cheaper mapping (a mask, or Lemire's multiply-shift range reduction)
/// sends k-mers to different columns and so changes every stored count and
/// every estimate. See this module's note in the performance report: if
/// `width` were *required* to be a power of two, `% width` would become
/// `& (width - 1)`, but that is an API restriction on `new`, not a local
/// rewrite, so it is left to the caller-facing design to decide.
#[inline(always)]
fn column(seed: u64, width: usize, kmer: u64) -> usize {
    let h = kmer.wrapping_mul(seed);
    ((h ^ (h >> 32)) as usize) % width
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
