// src/counter.rs

use rustc_hash::FxHashMap;

/// Outcome of a `prune` call, for reporting what a filter removed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub dropped_min: u64,
    pub dropped_max: u64,
    pub kept: u64,
}

/// In-memory frequency table for canonical 64-bit k-mers.
#[derive(Debug, Default)]
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

    /// Removes k-mers outside the inclusive `[min, max]` frequency band.
    ///
    /// Both bounds are inclusive: a k-mer whose count equals `min` or `max` is
    /// kept. `total_kmers` is deliberately left unchanged — it is the
    /// normalization basis and must reflect the sample's true depth.
    pub fn prune(&mut self, min: u32, max: Option<u32>) -> PruneStats {
        let mut stats = PruneStats::default();

        self.table.retain(|_, count| {
            if *count < min {
                stats.dropped_min += 1;
                false
            } else if max.is_some_and(|cap| *count > cap) {
                stats.dropped_max += 1;
                false
            } else {
                stats.kept += 1;
                true
            }
        });

        stats
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a counter where k-mer `i` appears `i` times, for i in 1..=5.
    fn counter_with_graded_counts() -> KmerCounter {
        let mut c = KmerCounter::new();
        for kmer in 1u64..=5 {
            for _ in 0..kmer {
                c.insert(kmer);
            }
        }
        c
    }

    #[test]
    fn prune_drops_counts_below_min_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(3, None);

        assert_eq!(stats.dropped_min, 2, "k-mers seen 1 and 2 times must go");
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 3, "k-mers seen 3, 4 and 5 times must stay");
        assert_eq!(c.distinct_kmers(), 3);
        assert_eq!(c.get_count(3), 3, "count == min is kept");
        assert_eq!(c.get_count(2), 0, "count < min is gone");
    }

    #[test]
    fn prune_drops_counts_above_max_and_keeps_the_boundary() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, Some(4));

        assert_eq!(stats.dropped_max, 1, "only the k-mer seen 5 times is over the cap");
        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.kept, 4);
        assert_eq!(c.get_count(4), 4, "count == max is kept");
        assert_eq!(c.get_count(5), 0);
    }

    #[test]
    fn prune_applies_both_bounds_at_once() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(2, Some(4));

        assert_eq!(stats.dropped_min, 1);
        assert_eq!(stats.dropped_max, 1);
        assert_eq!(stats.kept, 3);
        assert_eq!(c.distinct_kmers(), 3);
    }

    #[test]
    fn prune_leaves_total_kmers_untouched() {
        let mut c = counter_with_graded_counts();
        let before = c.total_kmers();
        assert_eq!(before, 15, "1+2+3+4+5 occurrences");

        c.prune(4, None);

        assert_eq!(
            c.total_kmers(),
            before,
            "total_kmers is the normalization basis and must survive pruning"
        );
    }

    #[test]
    fn prune_with_permissive_bounds_drops_nothing() {
        let mut c = counter_with_graded_counts();

        let stats = c.prune(1, None);

        assert_eq!(stats.dropped_min, 0);
        assert_eq!(stats.dropped_max, 0);
        assert_eq!(stats.kept, 5);
    }
}
