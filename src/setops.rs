// src/setops.rs
//! Set operations between k-mer tables (`docs/feature-gap-analysis.md`'s
//! S2): union, intersection and asymmetric difference over one or more
//! `ktab::KmerTable`s -- the case-vs-control, reference-subtraction and
//! trio-binning workflows `kmc_tools simple`/FastK's `Logex` cover for
//! their own binary databases.
//!
//! # Why a new module, not folded into `ktab.rs`
//!
//! `ktab.rs` is a read layer: `KmerTable::open`/`get`/`range`/`iter` answer
//! questions about *one* table. This module answers questions about *several*
//! tables at once, and its own state (a merge cursor per input table, a
//! binary heap ordering them) has nothing to do with `KmerTable`'s own
//! internals -- it is built entirely on `KmerTable::iter`'s public streaming
//! contract, the same arm's-length relationship `disk_spill.rs` has with
//! `counter.rs` (a separate module built on `KmerCounter`'s public surface,
//! not merged into it). Keeping sorted-merge-join code here, rather than
//! growing `ktab.rs` into "read one table, or combine several", is the same
//! separation of concerns that split `counter.rs` from `disk_spill.rs` in
//! the first place.
//!
//! # Why this is a linear merge-join, not a hash-join
//!
//! `docs/feature-gap-analysis.md`'s own framing for S2: "every op is a linear
//! merge-join -- machinery `disk_spill.rs` already has." Every table's rows
//! are already globally sorted, deduplicated `(key, frequency)` pairs (`ktab.rs`'s `open` verifies the sort at row-group boundaries;
//! `counter.rs`'s `CountTable` invariant guarantees the dedup before a table
//! is ever written). Given that, combining two or more tables is exactly
//! what `disk_spill.rs::merge_sources_into` already does for its own
//! per-bucket run files: advance whichever source currently holds the
//! smallest pending key, and fold together every other source sitting at
//! that same key. `MultiTableMerge` below is that same binary-heap algorithm,
//! adapted to pull from `KmerTable::iter`'s `RangeIter`s (Parquet row groups,
//! decoded one batch at a time) instead of `disk_spill.rs`'s flat run files.
//!
//! Loading one side into a `HashSet`/`HashMap` -- the naive way to compute a
//! set operation -- was deliberately not done: that would require the whole
//! smaller table resident in memory before a single output row could be
//! produced, throwing away the exact reason a sorted, row-group-pruned
//! representation was worth building in the first place (`ktab.rs`'s own
//! module doc comment). The merge below never holds more than one Arrow
//! batch's worth of each input table in memory at a time (the same bound
//! `RangeIter` itself already provides), regardless of how large any input
//! table is.
//!
//! # The one merge primitive every operation is built from
//!
//! `MultiTableMerge` streams `MergedRow`s -- one per distinct k-mer across
//! every input table, carrying *each* table's own frequency for that k-mer
//! (or `None` if that table lacks it) -- in ascending key order.
//! `union`, `intersect` and `diff` are each a thin, different decision over
//! that same slice: whether to keep the row, and how to fold whichever
//! per-table frequencies are present down into the single `frequency` column
//! the output shape (below) provides. This is also what answers this task's
//! "intersect must expose each input's count, not just a boolean" -- the
//! merge itself always computes every table's individual count for a shared
//! k-mer; `intersect`'s `combine` argument is only how that Rust-level detail
//! gets folded into the one `frequency` column a k-mer table has room for.
//!
//! # Both table widths
//!
//! Everything here is generic over [`MergeSource`], implemented by
//! `ktab::KmerTable` (`u64` keys, `k <= 32`) and
//! `wide_ktab::WideKmerTable` (`u128` keys, `33 <= k <= 64`). That is
//! deliberately a *different* decision from the one `wide_kmer.rs` and
//! `wide_ktab.rs` make when they refuse to generify: the counting engine's
//! `u64` is a measured choice -- exactly equal to KMC3, every benchmark
//! taken on it -- so making *that* generic would put those results back in
//! question. A merge-join's key is a sort key and nothing else, and nothing
//! in this module was ever measured on the concrete type.
//!
//! `tests/wide_setops.rs` is what holds the two together: the same reads
//! counted at the same `k` by each engine give two tables of different
//! width holding the same k-mers, and every operation here must return the
//! same result over both, k-mer for k-mer.
//!
//! # Output shape
//!
//! Every operation here returns a plain `impl Iterator<Item = Result<(S::Key,
//! u32)>>` in ascending, deduplicated key order -- exactly the shape
//! `export::export_pairs_parquet` (or `export_wide_pairs_parquet`) writes to
//! Parquet with `ktab.rs`'s `fastdna.sorted_by`/`fastdna.k` footer metadata
//! attached, so a set operation's result is immediately reopenable as its own
//! table, with no conversion step: set operations compose (`union(a, b)`'s
//! output can be `diff`'d against `c` directly).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

use crate::error::{FastDnaError, Result};
use crate::ktab::{KmerTable, RangeIter};
use crate::wide_ktab::{WideKmerTable, WideRangeIter};

/// Rejects a set operation whose `--output` would overwrite one of its own
/// input tables. Shared by the CLI (`main.rs::guard_against_setops_output_
/// overwrite`, which calls straight into this) and the Python FFI
/// (`ffi.rs::ktab_union`/`ktab_intersect`/`ktab_diff`, which had no such
/// guard at all before this existed here -- `a.intersect(b, output=a_path)`
/// silently replaced `a` with the, frequently empty, intersection; see
/// `tests/review_findings.rs`/`python/tests/test_review_findings.py`).
///
/// Living here, in `fastdna_core`, rather than duplicated once per FFI
/// function means every current and future caller of a set operation --
/// CLI, Python, anything else built on this crate -- inherits the same
/// check from the same place: a set op's inputs are read lazily, row group
/// by row group, over the whole run (`MultiTableMerge`'s own doc comment),
/// so a truncated output file landing on top of a live input would corrupt
/// the very read still in progress.
pub fn guard_against_output_overwrite(input_paths: &[&Path], output: &Path) -> Result<()> {
    for input in input_paths {
        if crate::atomic::same_file(input, output) {
            return Err(FastDnaError::InvalidConfig {
                parameter: "output",
                reason: format!("points at the input table {} and would overwrite it", input.display()),
            });
        }
    }
    Ok(())
}

/// How several tables' individual frequencies for the same k-mer are folded
/// into the single `frequency` column the output shape provides. See
/// `union`/`intersect`'s own doc comments for which default this module
/// picks and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineOp {
    /// Total occurrences across every table that has this k-mer -- the
    /// natural "combine these samples into one" reading, and `union`'s
    /// default.
    Sum,
    /// The smallest count among the tables that have this k-mer -- a
    /// conservative summary ("this k-mer is at least this well
    /// supported everywhere it appears"), and `intersect`'s default,
    /// matching `kmc_tools simple`'s own default reducer for its
    /// `intersect` operation.
    Min,
    /// The largest count among the tables that have this k-mer.
    Max,
}

impl CombineOp {
    /// Folds every present count for one k-mer down to one `u32`. `counts`
    /// is never empty when this is called (every call site only invokes it
    /// on the `Some` entries of a `MergedRow`, and a `MergedRow` always has
    /// at least one), but `Min`/`Max` still fall back to `0` rather than
    /// panicking if that ever stopped being true -- an empty reduction is a
    /// caller bug, not a reason to crash a long-running merge over it.
    fn reduce(self, counts: impl Iterator<Item = u32>) -> u32 {
        match self {
            CombineOp::Sum => counts.fold(0u32, |acc, c| acc.saturating_add(c)),
            CombineOp::Min => counts.min().unwrap_or(0),
            CombineOp::Max => counts.max().unwrap_or(0),
        }
    }
}

/// One distinct k-mer's row out of `MultiTableMerge`: its value, and each
/// input table's own frequency for it (in the same order the tables were
/// given to `MultiTableMerge::new`), `None` where that table does not have
/// this k-mer at all.
///
/// `pub(crate)` (not `pub`): `cohort_vocab.rs`'s `rank_vocabulary` reuses
/// this struct's fields directly (`per_table.iter().filter(Option::is_some).
/// count()` is exactly "how many distinct tables contributed to this
/// k-mer", i.e. prevalence) rather than re-deriving the same information a
/// second, subtly different way. Still crate-private: nothing outside this
/// crate has a `KmerTable` to build a `MultiTableMerge` from in the first
/// place.
pub(crate) struct MergedRow<K> {
    pub(crate) kmer: K,
    pub(crate) per_table: Vec<Option<u32>>,
}

/// A k-mer table this module can merge: something that knows its own `k`
/// and can hand out its rows, ascending and deduplicated.
///
/// This is the one place `setops` is generic, and it is deliberately the
/// *narrowest* possible abstraction: a sort key and a stream of
/// `(key, count)`. Everything below -- the heap, the fold, the three
/// operations -- is already key-agnostic in its logic, so a trait with two
/// implementations replaces what would otherwise be a second copy of this
/// file differing only in an integer width.
///
/// This is not the "generify the counting engine" that `wide_kmer.rs` and
/// `wide_ktab.rs` argue against, and the difference is worth stating: the
/// counting engine's `u64` is a *measured* choice (exactly equal to KMC3,
/// every benchmark taken on it), so making it generic would put those
/// results in question. A merge-join's key is a sort key and nothing else.
/// Nothing here was ever measured on the concrete type.
///
/// `pub`, and therefore part of the compatibility contract, because it has
/// to be: `union`, `intersect` and `diff` are public and generic over it,
/// so their return type (`impl Iterator<Item = Result<(S::Key, u32)>>`)
/// names it. A caller cannot use those functions without being able to
/// name the trait that decides what comes out. Implemented for exactly the
/// two table types this crate has and not meant to be implemented outside
/// it, but sealing it would only trade a documented intent for a harder-to-
/// read one.
pub trait MergeSource {
    /// The packed k-mer type: `u64` for a narrow table, `u128` for a wide
    /// one. `Ord` is what the merge needs; `Display` is for the error
    /// messages that name an offending k-mer.
    type Key: Ord + Copy + std::fmt::Display;
    type Rows: Iterator<Item = Result<(Self::Key, u32)>>;

    /// The `k` this table's rows were packed with, from its own footer
    /// metadata -- never re-derived from row contents.
    fn k(&self) -> usize;

    /// A fresh, owned iterator over every row, ascending. Owned rather
    /// than borrowed so a merge carries no lifetime back to the tables it
    /// was built from -- see `MultiTableMerge`'s own doc comment.
    fn rows(&self) -> Result<Self::Rows>;
}

impl MergeSource for KmerTable {
    type Key = u64;
    type Rows = RangeIter;

    fn k(&self) -> usize {
        KmerTable::k(self)
    }

    fn rows(&self) -> Result<RangeIter> {
        self.iter()
    }
}

impl MergeSource for WideKmerTable {
    type Key = u128;
    type Rows = WideRangeIter;

    fn k(&self) -> usize {
        WideKmerTable::k(self)
    }

    fn rows(&self) -> Result<WideRangeIter> {
        self.iter()
    }
}

/// Streams `MergedRow`s across several `KmerTable`s in ascending `kmer_u64`
/// order via a binary-heap k-way merge over each table's own `KmerTable::
/// iter` -- see the module doc comment for why this is the right primitive
/// and how it relates to `disk_spill.rs::merge_sources_into`.
///
/// No duplicate keys can appear *within* one source: a `KmerTable`'s rows
/// are already deduplicated by construction (`counter.rs`'s `CountTable`
/// invariant, upheld before `export.rs` ever writes a row), unlike
/// `disk_spill.rs`'s raw run files, which really can hold the same k-mer
/// across several of one worker's own flushes. That is why this merge only
/// ever needs to record at most one pending count per source -- there is no
/// "fold duplicates within a source" step to run.
///
/// Owns every `RangeIter` it pulls from outright (`KmerTable::iter` returns
/// an owned iterator with its own file handle, not one borrowed from the
/// table), so this -- and everything built on it -- carries no lifetime
/// tied back to the `&[KmerTable]` it was built from.
pub(crate) struct MultiTableMerge<S: MergeSource> {
    sources: Vec<S::Rows>,
    heap: BinaryHeap<Reverse<(S::Key, usize)>>,
    /// The count belonging to whichever entry each source currently has *in
    /// the heap* -- mirrors `disk_spill.rs::merge_sources_into`'s own
    /// `pending` buffer and the same reasoning: a source's absence from the
    /// heap already encodes "exhausted", so this needs no `Option`.
    pending: Vec<u32>,
}

impl<S: MergeSource> MultiTableMerge<S> {
    pub(crate) fn new<'t>(tables: impl IntoIterator<Item = &'t S>) -> Result<Self>
    where
        S: 't,
    {
        let mut sources: Vec<S::Rows> = Vec::new();
        for table in tables {
            sources.push(table.rows()?);
        }

        let mut pending = vec![0u32; sources.len()];
        let mut heap = BinaryHeap::with_capacity(sources.len());
        for (idx, source) in sources.iter_mut().enumerate() {
            if let Some(item) = source.next() {
                let (kmer, count) = item?;
                pending[idx] = count;
                heap.push(Reverse((kmer, idx)));
            }
        }

        Ok(Self { sources, heap, pending })
    }

    /// Pulls one more entry from source `idx` and re-inserts it into the
    /// heap, if there is one. Shared by the "first entry at this key" and
    /// "another source folded into this key" paths in `next` below, so
    /// neither can forget to re-arm its source.
    fn advance(&mut self, idx: usize) -> Result<()> {
        match self.sources[idx].next() {
            Some(Ok((kmer, count))) => {
                self.pending[idx] = count;
                self.heap.push(Reverse((kmer, idx)));
                Ok(())
            }
            Some(Err(e)) => Err(e),
            None => Ok(()),
        }
    }
}

impl<S: MergeSource> Iterator for MultiTableMerge<S> {
    type Item = Result<MergedRow<S::Key>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Reverse((kmer, idx)) = self.heap.pop()?;
        let mut per_table: Vec<Option<u32>> = vec![None; self.sources.len()];
        per_table[idx] = Some(self.pending[idx]);

        if let Err(e) = self.advance(idx) {
            return Some(Err(e));
        }

        // Fold in every other source currently sitting at the same key --
        // mirrors `disk_spill.rs::merge_sources_into`'s own duplicate-key
        // loop, except every folded source contributes its own slot in
        // `per_table` instead of being summed away.
        while let Some(&Reverse((peek_kmer, _))) = self.heap.peek() {
            if peek_kmer != kmer {
                break;
            }
            let Some(Reverse((_, other_idx))) = self.heap.pop() else {
                break;
            };
            per_table[other_idx] = Some(self.pending[other_idx]);
            if let Err(e) = self.advance(other_idx) {
                return Some(Err(e));
            }
        }

        Some(Ok(MergedRow { kmer, per_table }))
    }
}

/// Rejects a set operation across tables built with different `k`: their
/// `kmer_u64` values would not even mean the same thing (a `k=21` encoding
/// and a `k=31` encoding of the same integer are unrelated sequences), so
/// merging them by raw integer comparison would silently produce a
/// nonsensical result rather than fail loudly.
///
/// `pub(crate)` rather than private: `similarity.rs` builds on the same
/// merge primitive and needs the identical guard -- one merge primitive,
/// one k-mismatch check, not two.
pub(crate) fn check_same_k<'t, S: MergeSource + 't>(
    tables: impl IntoIterator<Item = &'t S>,
) -> Result<()> {
    let mut tables = tables.into_iter();
    let Some(first) = tables.next() else { return Ok(()) };
    let k = first.k();
    for (idx, table) in tables.enumerate() {
        // `idx` counts from the second table (index 1 overall) since
        // `first` was already consumed above.
        if table.k() != k {
            return Err(FastDnaError::InvalidConfig {
                parameter: "input tables",
                reason: format!(
                    "table {} has k={}, but table 0 has k={k} -- a set operation requires every \
                     input table to share the same k (their packed encodings are only \
                     comparable when they do)",
                    idx + 1,
                    table.k()
                ),
            });
        }
    }
    Ok(())
}

fn require_min_tables<S>(tables: &[S], op: &'static str) -> Result<()> {
    if tables.len() < 2 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "input tables",
            reason: format!(
                "{op} needs at least two tables to combine; got {} (a single table is already \
                 its own {op})",
                tables.len()
            ),
        });
    }
    Ok(())
}

/// Union: every k-mer present in at least one of `tables`, with its
/// frequency folded across every table that has it via `combine`.
///
/// `CombineOp::Sum` is the recommended default: this is the "merge these
/// samples into one bigger sample" operation (combining replicate lanes,
/// pooling a cohort), and occurrences genuinely add up across samples the
/// same way they would if the reads had simply been counted together in
/// one run. `Min`/`Max` are available for a caller who wants "how often did
/// this k-mer show up, at worst/best, across the inputs" instead.
///
/// Streaming: nothing beyond one Arrow batch per input table (`RangeIter`'s
/// own bound) is held in memory at a time, however many distinct k-mers the
/// union produces.
pub fn union<S: MergeSource>(
    tables: &[S],
    combine: CombineOp,
) -> Result<impl Iterator<Item = Result<(S::Key, u32)>>> {
    check_same_k(tables)?;
    require_min_tables(tables, "union")?;
    let merge = MultiTableMerge::new(tables)?;
    Ok(merge.map(move |row| {
        row.map(|MergedRow { kmer, per_table }| (kmer, combine.reduce(per_table.into_iter().flatten())))
    }))
}

/// Intersection: only k-mers present in *every* one of `tables`, with the
/// frequency folded across all of them via `combine`. A k-mer missing from
/// even one input table is dropped entirely -- not emitted with a zero or
/// partial count.
///
/// `CombineOp::Min` is the recommended default (matching `kmc_tools
/// simple`'s own default reducer for `intersect`): the k-mer's support in
/// the intersection is only as strong as its rarest observation, so this is
/// the conservative single-number summary of "every table actually has
/// this, and here is how confidently". `Sum`/`Max` remain available for a
/// caller who wants a different summary of the same per-table counts this
/// merge already computes.
pub fn intersect<S: MergeSource>(
    tables: &[S],
    combine: CombineOp,
) -> Result<impl Iterator<Item = Result<(S::Key, u32)>>> {
    check_same_k(tables)?;
    require_min_tables(tables, "intersect")?;
    let merge = MultiTableMerge::new(tables)?;
    Ok(merge.filter_map(move |row| match row {
        Ok(MergedRow { kmer, per_table }) => {
            if per_table.iter().all(Option::is_some) {
                Some(Ok((kmer, combine.reduce(per_table.into_iter().flatten()))))
            } else {
                None
            }
        }
        Err(e) => Some(Err(e)),
    }))
}

/// Asymmetric difference: every k-mer in `a` that is absent from every
/// table in `subtract`, or that never exceeds `max_subtract_count` in any
/// of them -- the reference-subtraction/host-removal use case
/// (`docs/feature-gap-analysis.md`'s S2): filtering a sample's own k-mer
/// table against one or more host genomes, adapter sets, or known
/// contaminant references.
///
/// `max_subtract_count = 0` is the strict "present at all, in any
/// reference" reading -- the ordinary set-difference. Raising it tolerates
/// a small amount of reference noise (e.g. a handful of spurious low-depth
/// hits in the host reference) before a k-mer is treated as contamination
/// and dropped, which is what makes this "genuinely useful for [host
/// removal]" rather than a toy set-diff: a real reference genome and a
/// real sample share some k-mers by chance even with no true contamination
/// present, and a strict `> 0` threshold would treat every one of those as
/// disqualifying.
///
/// The kept row's frequency is always `a`'s own original count, never
/// blended with anything from `subtract` -- this is a filter over `a`'s
/// table (answering "what does `a` look like with these references
/// removed"), not a further combination of the two sides.
///
/// `subtract` may name more than one table (a trio's other two members, a
/// host plus an adapter reference, ...): a k-mer is dropped if it exceeds
/// the threshold in *any* of them, not only if it does in all of them --
/// the natural reading for "remove everything that looks like any of these
/// references."
pub fn diff<'a, S: MergeSource>(
    a: &'a S,
    subtract: &'a [S],
    max_subtract_count: u32,
) -> Result<impl Iterator<Item = Result<(S::Key, u32)>>> {
    if subtract.is_empty() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "subtract",
            reason: "diff needs at least one table to subtract from the input".to_string(),
        });
    }
    let all: Vec<&S> = std::iter::once(a).chain(subtract.iter()).collect();
    check_same_k(all.iter().copied())?;

    let merge = MultiTableMerge::new(all)?;
    Ok(merge.filter_map(move |row| match row {
        Ok(MergedRow { kmer, mut per_table }) => {
            // Index 0 is always `a` (see the `all` vector built above);
            // absent from `a` means this k-mer is not this operation's
            // concern at all, regardless of what the subtract tables hold.
            let a_count = per_table.first_mut().and_then(Option::take)?;
            // Index 0 (`a`'s own slot) was just taken above and is `None`
            // now, so `flatten()` already skips it -- what remains is
            // exactly the subtract tables' own counts.
            let worst_in_subtract = per_table.into_iter().flatten().max().unwrap_or(0);
            if worst_in_subtract > max_subtract_count {
                None
            } else {
                Some(Ok((kmer, a_count)))
            }
        }
        Err(e) => Some(Err(e)),
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::counter::KmerCounter;
    use crate::export;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("fastdna_setops_test");
        std::fs::create_dir_all(&dir).unwrap();
        let unique = std::process::id() as u64 * 1_000_000
            + std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as u64;
        dir.join(format!("{name}_{unique}.parquet"))
    }

    /// Builds a table through the real counting + export path (not a
    /// hand-built Parquet file), matching `ktab.rs`'s own test convention --
    /// these tests exercise exactly the files `fastdna count` produces.
    fn build_table(name: &str, k: usize, entries: &[u64]) -> (KmerTable, PathBuf) {
        let path = temp_path(name);
        let mut counter = KmerCounter::new();
        counter.insert_batch(entries);
        export::export_counts_parquet(&counter, &path, k, 1, false).unwrap();
        (KmerTable::open(&path).unwrap(), path)
    }

    fn collect(rows: impl Iterator<Item = Result<(u64, u32)>>) -> Vec<(u64, u32)> {
        rows.collect::<Result<Vec<_>>>().unwrap()
    }

    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn union_of_disjoint_tables_keeps_every_kmer_with_its_own_count() {
        let (a, pa) = build_table("union_disjoint_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("union_disjoint_b", 4, &[3, 3, 3]);

        let rows = collect(union(&[a, b], CombineOp::Sum).unwrap());
        assert_eq!(rows, vec![(1, 2), (2, 1), (3, 3)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn union_sums_overlapping_kmers_by_default_semantics() {
        let (a, pa) = build_table("union_overlap_a", 4, &[5, 5, 6]);
        let (b, pb) = build_table("union_overlap_b", 4, &[5, 7, 7, 7]);

        let rows = collect(union(&[a, b], CombineOp::Sum).unwrap());
        assert_eq!(rows, vec![(5, 3), (6, 1), (7, 3)], "kmer 5 must sum 2 + 1 = 3");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn union_combine_max_keeps_the_larger_per_table_count() {
        let (a, pa) = build_table("union_max_a", 4, &[5, 5]);
        let (b, pb) = build_table("union_max_b", 4, &[5, 5, 5]);

        let rows = collect(union(&[a, b], CombineOp::Max).unwrap());
        assert_eq!(rows, vec![(5, 3)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn union_of_three_tables_folds_every_source() {
        let (a, pa) = build_table("union3_a", 4, &[1]);
        let (b, pb) = build_table("union3_b", 4, &[1, 1]);
        let (c, pc) = build_table("union3_c", 4, &[1, 1, 1]);

        let rows = collect(union(&[a, b, c], CombineOp::Sum).unwrap());
        assert_eq!(rows, vec![(1, 6)]);

        cleanup(&[pa, pb, pc]);
    }

    #[test]
    fn union_with_an_empty_table_is_the_non_empty_side_unchanged() {
        let (a, pa) = build_table("union_empty_a", 4, &[]);
        let (b, pb) = build_table("union_empty_b", 4, &[9, 9]);

        let rows = collect(union(&[a, b], CombineOp::Sum).unwrap());
        assert_eq!(rows, vec![(9, 2)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_of_disjoint_tables_is_empty() {
        let (a, pa) = build_table("intersect_disjoint_a", 4, &[1, 2]);
        let (b, pb) = build_table("intersect_disjoint_b", 4, &[3, 4]);

        let rows = collect(intersect(&[a, b], CombineOp::Min).unwrap());
        assert!(rows.is_empty());

        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_keeps_only_shared_kmers_with_min_by_default() {
        let (a, pa) = build_table("intersect_shared_a", 4, &[1, 1, 1, 2]);
        let (b, pb) = build_table("intersect_shared_b", 4, &[1, 1, 3]);

        let rows = collect(intersect(&[a, b], CombineOp::Min).unwrap());
        assert_eq!(rows, vec![(1, 2)], "kmer 1: min(3, 2) = 2; kmer 2/3 are not shared");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_combine_sum_adds_the_shared_kmers_counts() {
        let (a, pa) = build_table("intersect_sum_a", 4, &[1, 1, 1]);
        let (b, pb) = build_table("intersect_sum_b", 4, &[1, 1]);

        let rows = collect(intersect(&[a, b], CombineOp::Sum).unwrap());
        assert_eq!(rows, vec![(1, 5)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_of_identical_tables_is_a_full_overlap() {
        let (a, pa) = build_table("intersect_full_a", 4, &[1, 1, 2, 3, 3, 3]);
        let (b, pb) = build_table("intersect_full_b", 4, &[1, 1, 2, 3, 3, 3]);

        let rows = collect(intersect(&[a, b], CombineOp::Min).unwrap());
        assert_eq!(rows, vec![(1, 2), (2, 1), (3, 3)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_requires_every_table_not_just_two_to_have_the_kmer() {
        let (a, pa) = build_table("intersect3_a", 4, &[1, 2]);
        let (b, pb) = build_table("intersect3_b", 4, &[1, 2]);
        let (c, pc) = build_table("intersect3_c", 4, &[1]);

        let rows = collect(intersect(&[a, b, c], CombineOp::Min).unwrap());
        assert_eq!(rows, vec![(1, 1)], "kmer 2 is missing from table c, so it must be dropped");

        cleanup(&[pa, pb, pc]);
    }

    #[test]
    fn intersect_with_an_empty_table_is_always_empty() {
        let (a, pa) = build_table("intersect_empty_a", 4, &[1, 1]);
        let (b, pb) = build_table("intersect_empty_b", 4, &[]);

        let rows = collect(intersect(&[a, b], CombineOp::Min).unwrap());
        assert!(rows.is_empty());

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_removes_every_kmer_present_at_all_in_subtract_by_default() {
        let (a, pa) = build_table("diff_basic_a", 4, &[1, 1, 2, 3]);
        let (b, pb) = build_table("diff_basic_b", 4, &[2]);

        let rows = collect(diff(&a, &[b], 0).unwrap());
        assert_eq!(rows, vec![(1, 2), (3, 1)], "kmer 2 must be removed, kept ones keep a's own count");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_with_no_overlap_returns_a_unchanged() {
        let (a, pa) = build_table("diff_no_overlap_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("diff_no_overlap_b", 4, &[9, 9]);

        let rows = collect(diff(&a, &[b], 0).unwrap());
        assert_eq!(rows, vec![(1, 2), (2, 1)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_of_a_against_itself_is_empty() {
        let (a, pa) = build_table("diff_self_a", 4, &[1, 1, 2, 3]);
        let (b, pb) = build_table("diff_self_b", 4, &[1, 1, 2, 3]);

        let rows = collect(diff(&a, &[b], 0).unwrap());
        assert!(rows.is_empty());

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_threshold_tolerates_low_level_noise_in_the_subtract_table() {
        let (a, pa) = build_table("diff_threshold_a", 4, &[1, 1, 1, 2]);
        // kmer 1 appears twice in the reference -- below a threshold of 2 it
        // should be tolerated (kept), not treated as contamination.
        let (b, pb) = build_table("diff_threshold_b", 4, &[1, 1]);

        let tolerant = collect(diff(&a, std::slice::from_ref(&b), 2).unwrap());
        assert_eq!(tolerant, vec![(1, 3), (2, 1)], "count <= threshold in b must be tolerated");

        let strict = collect(diff(&a, &[b], 0).unwrap());
        assert_eq!(strict, vec![(2, 1)], "threshold 0 drops any kmer present at all in b");

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_against_several_subtract_tables_drops_a_kmer_flagged_by_any_of_them() {
        let (a, pa) = build_table("diff_multi_a", 4, &[1, 2, 3]);
        let (host, ph) = build_table("diff_multi_host", 4, &[1]);
        let (adapter, pd) = build_table("diff_multi_adapter", 4, &[2]);

        let rows = collect(diff(&a, &[host, adapter], 0).unwrap());
        assert_eq!(rows, vec![(3, 1)], "1 flagged by host, 2 flagged by adapter, only 3 survives");

        cleanup(&[pa, ph, pd]);
    }

    #[test]
    fn diff_with_an_empty_subtract_table_keeps_a_unchanged() {
        let (a, pa) = build_table("diff_empty_subtract_a", 4, &[1, 2]);
        let (b, pb) = build_table("diff_empty_subtract_b", 4, &[]);

        let rows = collect(diff(&a, &[b], 0).unwrap());
        assert_eq!(rows, vec![(1, 1), (2, 1)]);

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_with_an_empty_input_table_a_is_empty() {
        let (a, pa) = build_table("diff_empty_a", 4, &[]);
        let (b, pb) = build_table("diff_empty_a_b", 4, &[1, 2]);

        let rows = collect(diff(&a, &[b], 0).unwrap());
        assert!(rows.is_empty());

        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_rejects_an_empty_subtract_list() {
        let (a, pa) = build_table("diff_no_subtract_a", 4, &[1]);
        match diff(&a, &[], 0) {
            Err(FastDnaError::InvalidConfig { .. }) => {}
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[pa]);
    }

    #[test]
    fn union_rejects_mismatched_k_across_tables() {
        let (a, pa) = build_table("mismatched_k_union_a", 4, &[1]);
        let (b, pb) = build_table("mismatched_k_union_b", 6, &[1]);

        match union(&[a, b], CombineOp::Sum) {
            Err(FastDnaError::InvalidConfig { reason, .. }) => {
                assert!(reason.contains('k'), "{reason}");
            }
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[pa, pb]);
    }

    #[test]
    fn intersect_rejects_mismatched_k_across_tables() {
        let (a, pa) = build_table("mismatched_k_intersect_a", 4, &[1]);
        let (b, pb) = build_table("mismatched_k_intersect_b", 8, &[1]);

        assert!(matches!(intersect(&[a, b], CombineOp::Min), Err(FastDnaError::InvalidConfig { .. })));
        cleanup(&[pa, pb]);
    }

    #[test]
    fn diff_rejects_mismatched_k_between_a_and_subtract() {
        let (a, pa) = build_table("mismatched_k_diff_a", 4, &[1]);
        let (b, pb) = build_table("mismatched_k_diff_b", 5, &[1]);

        assert!(matches!(diff(&a, &[b], 0), Err(FastDnaError::InvalidConfig { .. })));
        cleanup(&[pa, pb]);
    }

    #[test]
    fn union_requires_at_least_two_tables() {
        let (a, pa) = build_table("union_min_a", 4, &[1]);
        match union(std::slice::from_ref(&a), CombineOp::Sum) {
            Err(FastDnaError::InvalidConfig { .. }) => {}
            other => panic!("expected InvalidConfig, got is_ok={}", other.is_ok()),
        }
        cleanup(&[pa]);
    }

    #[test]
    fn intersect_requires_at_least_two_tables() {
        let (a, pa) = build_table("intersect_min_a", 4, &[1]);
        assert!(matches!(
            intersect(std::slice::from_ref(&a), CombineOp::Min),
            Err(FastDnaError::InvalidConfig { .. })
        ));
        cleanup(&[pa]);
    }

    /// A composability smoke test: `union`'s own output, written back out
    /// through `export::export_pairs_parquet` and reopened, is itself a
    /// valid `KmerTable` that `diff` can consume -- the exact "set
    /// operations should compose" property the module doc comment claims.
    #[test]
    fn union_output_written_and_reopened_composes_with_diff() {
        let (a, pa) = build_table("compose_a", 4, &[1, 1, 2]);
        let (b, pb) = build_table("compose_b", 4, &[2, 3]);

        let union_rows = union(&[a, b], CombineOp::Sum).unwrap();
        let union_path = temp_path("compose_union_out");
        export::export_pairs_parquet(union_rows, &union_path, 4).unwrap();
        let union_table = KmerTable::open(&union_path).unwrap();
        assert_eq!(collect(union_table.iter().unwrap()), vec![(1, 2), (2, 2), (3, 1)]);

        let (c, pc) = build_table("compose_c", 4, &[1]);
        let final_rows = collect(diff(&union_table, &[c], 0).unwrap());
        assert_eq!(final_rows, vec![(2, 2), (3, 1)], "kmer 1 must be removed by the second diff");

        cleanup(&[pa, pb, pc, union_path]);
    }
}
