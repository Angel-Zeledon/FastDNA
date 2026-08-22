// src/counter.rs

use rustc_hash::FxHashMap;

/// In-memory frequency table for canonical 64-bit k-mers.
#[derive(Default)]
pub struct KmerCounter {
    table: FxHashMap<u64, u32>,
    total_kmers: u64,
}

impl KmerCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            total_kmers: 0,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, kmer: u64) {
        *self.table.entry(kmer).or_insert(0) += 1;
        self.total_kmers += 1;
    }

    pub fn insert_batch(&mut self, kmers: &[u64]) {
        for &kmer in kmers {
            self.insert(kmer);
        }
    }

    pub fn merge(&mut self, other: KmerCounter) {
        self.total_kmers += other.total_kmers;
        for (kmer, count) in other.table {
            *self.table.entry(kmer).or_insert(0) += count;
        }
    }

    #[inline(always)]
    pub fn distinct_kmers(&self) -> usize {
        self.table.len()
    }

    #[inline(always)]
    pub fn total_kmers(&self) -> u64 {
        self.total_kmers
    }

    #[inline(always)]
    pub fn get_count(&self, kmer: u64) -> u32 {
        self.table.get(&kmer).copied().unwrap_or(0)
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, u64, u32> {
        self.table.iter()
    }

    pub fn generate_histogram(&self) -> FxHashMap<u32, u64> {
        let mut histogram: FxHashMap<u32, u64> = FxHashMap::default();
        for &count in self.table.values() {
            *histogram.entry(count).or_insert(0) += 1;
        }
        histogram
    }

    pub fn top_kmers(&self, n: usize) -> Vec<(u64, u32)> {
        let mut entries: Vec<(u64, u32)> = self.table.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }
}