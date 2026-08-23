// src/cms.rs

use serde::{Deserialize, Serialize};

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
    pub fn new(depth: usize, width: usize) -> Self {
        let mut seeds = Vec::with_capacity(depth);
        let mut seed = 0x9E3779B97F4A7C15u64;
        for _ in 0..depth {
            seeds.push(seed);
            seed = seed.wrapping_add(0x517CC1B727220A95);
        }

        Self {
            depth,
            width,
            table: vec![vec![0; width]; depth],
            seeds,
        }
    }

    /// Default 16 MB sketch (depth=4, width=1,048,576).
    pub fn default_16mb() -> Self {
        Self::new(4, 1_048_576)
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

    /// Merges another sketch of identical dimensions.
    pub fn merge(&mut self, other: &CountMinSketch) {
        assert_eq!(self.depth, other.depth);
        assert_eq!(self.width, other.width);

        for row in 0..self.depth {
            for col in 0..self.width {
                self.table[row][col] = self.table[row][col].saturating_add(other.table[row][col]);
            }
        }
    }
}
