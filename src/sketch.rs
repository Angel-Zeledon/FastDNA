// src/sketch.rs

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// MinHash sketch representation for rapid genomic distance estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeSketch {
    pub sketch_size: usize,
    pub k: usize,
    pub hashes: Vec<u64>,
}

impl GenomeSketch {
    pub fn new(sketch_size: usize, k: usize) -> Self {
        Self {
            sketch_size,
            k,
            hashes: Vec::with_capacity(sketch_size),
        }
    }

    pub fn from_kmers(kmers: &[u64], sketch_size: usize, k: usize) -> Self {
        let mut min_set: BTreeSet<u64> = BTreeSet::new();

        for &kmer in kmers {
            let kmer_hash = kmer.wrapping_mul(0x517cc1b727220a95);

            if min_set.len() < sketch_size {
                min_set.insert(kmer_hash);
            } else if let Some(&max_val) = min_set.iter().next_back() {
                if kmer_hash < max_val && min_set.insert(kmer_hash) {
                    min_set.pop_last();
                }
            }
        }

        Self {
            sketch_size,
            k,
            hashes: min_set.into_iter().collect(),
        }
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