// src/sketch.rs
//! MinHash sketching for fast, approximate genome-to-genome comparison
//! (design doc §8). Instead of counting every k-mer, a `GenomeSketch` keeps
//! only the `sketch_size` smallest hashes as a fingerprint; comparing two
//! fingerprints estimates how similar the full k-mer sets are without ever
//! materializing them side by side. Two SARS-CoV-2 samples that would take
//! minutes to compare exactly can be compared in milliseconds this way.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, FastqReader};
use crate::kmer;

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
pub(crate) fn finalize_hash(kmer: u64) -> u64 {
    let mut z = kmer;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The bottom-`capacity` working set: the `capacity` smallest *distinct*
/// hashes seen so far, evicting the current maximum once full. Shared by
/// `from_kmers` (in-memory) and `from_reader` (streaming) so both
/// construction paths run through identical selection logic -- see
/// `streaming_construction_matches_in_memory_path` below, which exists
/// specifically to prove that sharing pays off.
///
/// The ordered set is not an implementation detail that can be swapped for a
/// max-heap or an unsorted buffer: bottom-k here is over *distinct* hashes,
/// and a k-mer stream repeats the same k-mer many times, so the set's
/// deduplication is load-bearing. A heap holding duplicates would retain
/// fewer than `capacity` distinct hashes and change the result.
///
/// What *is* removable is the per-hash tree work on the rejection path.
/// `insert` runs once per k-mer -- the hottest line in this file -- and once
/// the set is full the overwhelming majority of calls are rejections (over a
/// stream of N hashes only about `capacity * ln(N / capacity)` are ever
/// accepted; for N = 10^8 and capacity = 1000 that is ~11,500 accepts against
/// ~10^8 rejects, i.e. 99.99% rejections). The previous version paid, on
/// *every* rejected hash: a `sketch_size == 0` test, a `len()` test, a
/// `BTreeSet::iter()` construction (which descends from the root to *both*
/// the leftmost and the rightmost leaf -- 2 x ~3 node visits at capacity
/// 1000, each a pointer chase into a cache line that the hot loop otherwise
/// never touches), and then the comparison. Caching the maximum in a plain
/// `u64` field reduces the rejected case to one `bool` test and one integer
/// comparison, both on data already in registers: ~6 pointer-chasing node
/// visits plus an iterator construction removed per rejected k-mer. The
/// `capacity == 0` test is hoisted into `new` (it becomes `full = true` with
/// `max = 0`, which rejects everything, since no `u64` is `< 0`), removing a
/// third comparison from every call.
///
/// The accepted path is unchanged in cost (the same `BTreeSet::insert` plus
/// `pop_last`, and one tree descent to re-read the new maximum), and the
/// selected hashes are bit-identical to the old logic's -- the cached value
/// is always exactly `set.iter().next_back()`, so every branch is taken on
/// the same condition as before.
struct BottomK {
    set: BTreeSet<u64>,
    capacity: usize,
    /// Cached copy of `set`'s current maximum. Meaningful only while
    /// `full`; before that nothing is ever evicted, so it is never read.
    max: u64,
    full: bool,
}

impl BottomK {
    fn new(capacity: usize) -> Self {
        // `capacity == 0` starts out "full" with a maximum of 0, so the
        // `hash < max` test below rejects every hash forever -- the same
        // no-op the old explicit `sketch_size == 0` guard produced, without
        // costing a comparison per k-mer.
        Self { set: BTreeSet::new(), capacity, max: 0, full: capacity == 0 }
    }

    #[inline]
    fn insert(&mut self, hash: u64) {
        if !self.full {
            self.set.insert(hash);
            if self.set.len() >= self.capacity {
                if let Some(&m) = self.set.iter().next_back() {
                    self.max = m;
                }
                self.full = true;
            }
            return;
        }
        // The hot rejection: one comparison, no tree touched.
        if hash >= self.max {
            return;
        }
        if self.set.insert(hash) {
            self.set.pop_last();
            if let Some(&m) = self.set.iter().next_back() {
                self.max = m;
            }
        }
    }

    /// Consumes the working set into the ascending, deduplicated hash list
    /// `GenomeSketch` stores -- `BTreeSet`'s iteration order, so no sort is
    /// needed here.
    fn into_hashes(self) -> Vec<u64> {
        self.set.into_iter().collect()
    }
}

impl GenomeSketch {
    pub fn new(sketch_size: usize, k: usize) -> Self {
        Self { sketch_size, k, hashes: Vec::with_capacity(sketch_size) }
    }

    /// Builds a sketch from every k-mer already held in memory as a slice.
    /// Peak memory is `O(kmers.len())` for the input plus `O(sketch_size)`
    /// for the working set -- fine for small inputs or when the k-mers are
    /// already materialized for another reason, but see `from_reader`/
    /// `from_path` for large FASTQ files, where materializing every k-mer
    /// up front would defeat the point of sketching.
    pub fn from_kmers(kmers: &[u64], sketch_size: usize, k: usize) -> Self {
        let mut bottom_k = BottomK::new(sketch_size);

        for &kmer in kmers {
            bottom_k.insert(finalize_hash(kmer));
        }

        Self { sketch_size, k, hashes: bottom_k.into_hashes() }
    }

    /// Builds a sketch by streaming a FASTQ file record by record,
    /// transparently decompressing `.gz` inputs. Memory stays bounded by
    /// `sketch_size` regardless of file size: unlike `from_kmers`, no
    /// intermediate list of every k-mer in the file is ever built.
    pub fn from_path<P: AsRef<Path>>(path: P, sketch_size: usize, k: usize) -> Result<Self> {
        let path_ref = path.as_ref();
        let reader = FastqReader::from_path(path_ref)
            .map_err(|e| FastDnaError::Io { path: path_ref.to_path_buf(), source: e })?;
        Self::from_reader(reader, sketch_size, k, path_ref)
    }

    /// The streaming construction logic itself, over any `BufRead` -- kept
    /// separate from `from_path` so it is testable against an in-memory
    /// buffer without touching the filesystem (same split as
    /// `preview::peek`/`peek_from_reader`).
    ///
    /// `source` is used only to attribute I/O and parse errors to a path;
    /// `FastqReader` itself has no path of its own and tracks no record
    /// count across calls (see the doc comment on `FastqReadError`), so the
    /// caller supplies both.
    fn from_reader<R: BufRead>(
        mut reader: FastqReader<R>,
        sketch_size: usize,
        k: usize,
        source: &Path,
    ) -> Result<Self> {
        // An out-of-range k makes `extract_canonical_kmers` return an empty
        // Vec for every read, so without this check the whole file "sketches"
        // to zero hashes and every later comparison quietly reports 0.0 --
        // silent garbage instead of the error `count()` and
        // `estimate_cardinality` already raise for the same mistake.
        if k == 0 || k > 32 {
            return Err(FastDnaError::InvalidK { k });
        }
        if sketch_size == 0 {
            return Err(FastDnaError::InvalidConfig {
                parameter: "sketch_size",
                reason: "must be at least 1".to_string(),
            });
        }

        let mut bottom_k = BottomK::new(sketch_size);
        let mut record_count: u64 = 0;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    record_count += 1;
                    for kmer in kmer::extract_canonical_kmers(&record.seq, k) {
                        bottom_k.insert(finalize_hash(kmer));
                    }
                }
                Ok(None) => break,
                // A genuine I/O failure, not a data problem -- see the
                // matching branch in `preview::peek_from_reader` and
                // `pipeline::process_stream_parallel`, which this mirrors.
                Err(FastqReadError::Io(source_err)) => {
                    return Err(FastDnaError::Io { path: source.to_path_buf(), source: source_err });
                }
                Err(FastqReadError::Malformed(reason)) => {
                    return Err(FastDnaError::MalformedFastq {
                        path: source.to_path_buf(),
                        record: record_count + 1,
                        reason,
                    });
                }
            }
        }

        Ok(Self { sketch_size, k, hashes: bottom_k.into_hashes() })
    }

    /// Estimates the Jaccard similarity `|A ∩ B| / |A ∪ B|` between the two
    /// sketches' underlying k-mer sets: symmetric, and penalized by size
    /// differences between the genomes being compared -- a small viral
    /// genome compared against a large metagenomic sample scores near zero
    /// under this metric even when the virus is entirely present in the
    /// sample. See `containment` for the question that actually answers.
    ///
    /// Returns `Err(FastDnaError::MismatchedK)` rather than panicking when
    /// `self.k != other.k`: comparing sketches built with different k has
    /// no biological meaning, so it must fail, but a `Result` failure
    /// crosses the eventual PyO3 boundary as a `ValueError`, whereas a
    /// panic crosses it as an unrecoverable `pyo3_runtime.PanicException`.
    pub fn jaccard(&self, other: &GenomeSketch) -> Result<f64> {
        if self.k != other.k {
            return Err(FastDnaError::MismatchedK { left: self.k, right: other.k });
        }

        // Bounded by the smaller of the two sketch sizes: a bottom-k
        // estimate over the union is only valid up to the point where
        // *both* sketches would still contain every hash that small, so
        // capping by `self.sketch_size` alone (the original behaviour)
        // would overcount the union when the two sketches were built with
        // different sizes.
        let cap = self.sketch_size.min(other.sketch_size);

        let mut i = 0;
        let mut j = 0;
        let mut intersection = 0usize;
        let mut union_count = 0usize;

        while i < self.hashes.len() && j < other.hashes.len() && union_count < cap {
            match self.hashes[i].cmp(&other.hashes[j]) {
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
            union_count += 1;
        }

        // One list is exhausted (or the cap was hit). If the exhausted
        // sketch is *not full*, it holds its set's complete hash list --
        // there is no truncation ceiling past its last element -- so every
        // remaining hash on the other side is a genuine union-only member
        // and must widen the denominator. Stopping here regardless (the
        // old behaviour) shrank the union only, a systematic upward bias
        // that reached 33x when a tiny sketch met a large one. When the
        // exhausted sketch *is* full, stopping is correct: hashes above
        // its ceiling cannot be judged present or absent in it.
        let a_full = self.hashes.len() >= self.sketch_size;
        let b_full = other.hashes.len() >= other.sketch_size;
        if union_count < cap {
            if i >= self.hashes.len() && !a_full {
                let remaining = other.hashes.len() - j;
                union_count += remaining.min(cap - union_count);
            } else if j >= other.hashes.len() && !b_full {
                let remaining = self.hashes.len() - i;
                union_count += remaining.min(cap - union_count);
            }
        }

        if union_count == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f64 / union_count as f64)
        }
    }

    /// Estimates containment: what fraction of `self`'s k-mers also appear
    /// in `other`. Asymmetric by design -- `A.containment(B)` and
    /// `B.containment(A)` answer different questions, and clinically the
    /// one that matters is usually "is this pathogen (small, `self`)
    /// present in this metagenomic sample (large, `other`)", which is a
    /// containment question, not a similarity one: Jaccard would penalize
    /// the size mismatch between the two and report near zero even when the
    /// pathogen is entirely present.
    ///
    /// Only the portion of `self`'s sketch that `other`'s sketch can
    /// actually attest to is used. If `other`'s sketch is full (holds
    /// exactly `other.sketch_size` hashes), it has already discarded every
    /// hash above its current maximum -- a `self` hash above that ceiling
    /// cannot be judged present or absent in `other`, only "not observed
    /// within the sketch", so it is excluded from both the numerator and
    /// the denominator rather than being counted as a miss. This is the
    /// standard bounded-MinHash containment estimator (as used by e.g. Mash
    /// Screen), needed precisely because only `other`'s own bottom-k sketch
    /// is available here, not its full k-mer set.
    ///
    /// Same `MismatchedK` behaviour as `jaccard`: comparing sketches built
    /// with different `k` has no biological meaning, so this fails as a
    /// `Result` instead of panicking.
    pub fn containment(&self, other: &GenomeSketch) -> Result<f64> {
        if self.k != other.k {
            return Err(FastDnaError::MismatchedK { left: self.k, right: other.k });
        }

        if self.hashes.is_empty() {
            return Ok(0.0);
        }

        let ceiling = if other.hashes.len() >= other.sketch_size {
            other.hashes.last().copied().unwrap_or(u64::MAX)
        } else {
            u64::MAX
        };

        // `self.hashes` is ascending (by construction from a `BTreeSet`, and
        // enforced on the `load` path by `validate_sketch_invariants`), so
        // the resolvable hashes are a *prefix*, not a scattered subset: one
        // `partition_point` finds its length in ceil(log2(n)) comparisons --
        // 10 at n = 1024 -- replacing n `h > ceiling` comparisons and n
        // `resolvable += 1` increments in the loop body. It also drops the
        // whole tail on the common "small query against a large reference"
        // shape instead of iterating it to `continue`.
        let resolvable = self.hashes.partition_point(|&h| h <= ceiling);
        if resolvable == 0 {
            return Ok(0.0);
        }

        // Both lists are sorted ascending, and the queries are issued in
        // ascending order, so each search can start where the previous one
        // stopped: everything before `lo` is already known to be smaller
        // than every remaining query. Searching `other.hashes[lo..]` costs
        // ceil(log2(m - lo)) comparisons, which is <= the ceil(log2(m)) the
        // full-slice search cost, on *every* call and for every input shape
        // -- so this is unconditionally fewer comparisons, never a trade.
        //
        // A full linear merge walk (O(n + m)) was considered and rejected:
        // it wins only when n is a large fraction of m (at m = 1024 it needs
        // n > ~114 to beat n*log2(m)), and loses badly on exactly the shape
        // `containment` exists to serve -- a small pathogen sketch queried
        // against a large sample sketch, where n = 1 costs 10 comparisons by
        // binary search and up to 1024 by merge. Narrowing the search range
        // captures the merge's win on the balanced shape without its loss on
        // the skewed one.
        let mut shared = 0usize;
        let mut lo = 0usize;
        for &h in &self.hashes[..resolvable] {
            match other.hashes[lo..].binary_search(&h) {
                Ok(offset) => {
                    shared += 1;
                    lo += offset + 1;
                }
                Err(offset) => lo += offset,
            }
        }

        Ok(shared as f64 / resolvable as f64)
    }

    /// Estimates the per-base mutation rate implied by `jaccard`, under
    /// the same Poisson mutation model Mash itself uses (Ondov et al.,
    /// 2016): D = -(1/k) * ln(2J / (1+J)), where J is the Jaccard
    /// estimate. Two sketches sharing every k-mer (J=1) give D=0; sharing
    /// none (J=0) give D=1 (this implementation's chosen convention --
    /// the bare formula diverges to infinity there, which is not a useful
    /// number to hand a caller).
    ///
    /// This exists because raw Jaccard alone is a weaker claim than what
    /// "Mash-style" comparison implies: Jaccard says "these sketches
    /// overlap this much", not "these genomes differ by roughly this
    /// fraction of their bases", and the two are related but not the same
    /// question. `k` matters here in a way it does not for `jaccard`
    /// itself, since the model assumes each mutation destroys up to `k`
    /// overlapping k-mers -- the same `MismatchedK` restriction applies,
    /// for the same reason.
    ///
    /// What this does *not* give: Mash's own tool also reports a p-value
    /// against a null (random-sequence) hypothesis, which needs a genome
    /// length estimate this type does not have. This is the distance
    /// alone, not the full statistical test.
    pub fn mash_distance(&self, other: &GenomeSketch) -> Result<f64> {
        let j = self.jaccard(other)?;
        if j <= 0.0 {
            return Ok(1.0);
        }
        if j >= 1.0 {
            return Ok(0.0);
        }
        let d = -(1.0 / self.k as f64) * (2.0 * j / (1.0 + j)).ln();
        Ok(d.clamp(0.0, 1.0))
    }

    /// Persists the sketch as JSON. This is what makes N-sample comparison
    /// stop being O(N^2) FASTQ reads: compute each sample's sketch once,
    /// save it, and every later comparison loads two small files instead of
    /// re-reading two large ones.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let to_err = |e: std::io::Error| FastDnaError::Io { path: path.to_path_buf(), source: e };

        // Write-to-temp + rename like every other writer (see `atomic.rs`):
        // a failed save must not leave a corrupt half-written sketch where a
        // good one used to be.
        let (file, pending) = crate::atomic::AtomicFile::create(path)?;
        let mut writer = BufWriter::new(file);
        // Mirrors `QcSummary::export_json`: `serde_json::Error` conflates a
        // propagated I/O failure (`is_io()` true) with a genuine
        // serialization failure, and only the former belongs in `Io`.
        serde_json::to_writer_pretty(&mut writer, self).map_err(|e| {
            if e.is_io() {
                FastDnaError::Io { path: path.to_path_buf(), source: e.into() }
            } else {
                FastDnaError::Export { path: path.to_path_buf(), reason: e.to_string() }
            }
        })?;
        writer.flush().map_err(to_err)?;
        drop(writer);
        pending.commit()?;
        Ok(())
    }

    /// Loads a sketch previously written by `save`. Uses `FastDnaError::Load`
    /// for a corrupt/foreign file: it used to reuse `Export`, which
    /// `error.rs` renders as "export failed for {path}", telling a caller
    /// that writing failed when what actually failed was reading. `Load`
    /// names the operation honestly instead.
    ///
    /// Also validates the two invariants `jaccard` and `containment`
    /// silently assume on `hashes` -- see `validate_sketch_invariants` --
    /// since serde will happily deserialize a `Vec<u64>` that violates
    /// either one, and neither estimator checks for it on every call (that
    /// would mean re-validating on every comparison instead of once, at
    /// the file boundary, which is where a hand-edited or corrupted file
    /// can actually introduce the problem).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let to_err = |e: std::io::Error| FastDnaError::Io { path: path.to_path_buf(), source: e };
        let load_err = |reason: String| FastDnaError::Load { path: path.to_path_buf(), reason };

        let file = File::open(path).map_err(to_err)?;
        let reader = BufReader::new(file);
        let sketch: GenomeSketch = serde_json::from_reader(reader).map_err(|e| {
            if e.is_io() {
                FastDnaError::Io { path: path.to_path_buf(), source: e.into() }
            } else {
                load_err(e.to_string())
            }
        })?;

        validate_sketch_invariants(&sketch).map_err(load_err)?;

        Ok(sketch)
    }
}

/// Verifies the two invariants `jaccard`'s merge walk and `containment`'s
/// binary search both silently assume about `hashes`: it is sorted in
/// strictly ascending order, and it holds no more than `sketch_size`
/// entries. `save` always produces data meeting both -- `from_kmers` and
/// `from_reader` build `hashes` from a `BTreeSet` capped at `sketch_size`,
/// which is deduplicated, ordered, and bounded by construction -- but
/// `load` reads arbitrary JSON, and serde will happily deserialize a
/// hand-edited or corrupted file that violates either one. Without this
/// check, an unsorted vector silently breaks `jaccard`'s merge-walk
/// early-termination logic, and an oversized one means `containment`'s
/// `hashes.last()` is not actually the sketch's true ceiling -- both
/// return quiet nonsense instead of an error.
fn validate_sketch_invariants(sketch: &GenomeSketch) -> std::result::Result<(), String> {
    if sketch.k == 0 || sketch.k > 32 {
        return Err(format!(
            "k is {}, outside the 1..=32 range 2-bit packing supports",
            sketch.k
        ));
    }
    if sketch.sketch_size == 0 {
        return Err("sketch_size is 0; a saved sketch must hold at least one slot".to_string());
    }
    if sketch.hashes.len() > sketch.sketch_size {
        return Err(format!(
            "hashes has {} entries, more than sketch_size ({})",
            sketch.hashes.len(),
            sketch.sketch_size
        ));
    }
    if !sketch.hashes.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("hashes is not sorted in strictly ascending order".to_string());
    }
    Ok(())
}

#[cfg(test)]
// Matches the established pattern in `preview.rs`/`progress.rs`:
// `unwrap`/`expect` are denied under `src/` because production code must
// never panic on caller input, but that rule is not about test assertions,
// where spelling every check as a `match` would obscure what is tested.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fastq_bytes(reads: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for (i, seq) in reads.iter().enumerate() {
            buf.extend_from_slice(format!("@r{i}\n{seq}\n+\n{}\n", "I".repeat(seq.len())).as_bytes());
        }
        buf
    }

    fn reader_over(reads: &[&str]) -> FastqReader<Cursor<Vec<u8>>> {
        FastqReader::new(Cursor::new(fastq_bytes(reads)))
    }

    #[test]
    fn identical_inputs_give_jaccard_one() {
        let kmers: Vec<u64> = (0..2_000).collect();
        let a = GenomeSketch::from_kmers(&kmers, 256, 21);
        let b = GenomeSketch::from_kmers(&kmers, 256, 21);

        let j = a.jaccard(&b).expect("same k must not error");
        assert_eq!(j, 1.0, "a sketch compared with itself must be exactly 1.0");
    }

    #[test]
    fn mash_distance_of_identical_sketches_is_zero() {
        let kmers: Vec<u64> = (0..2_000).collect();
        let a = GenomeSketch::from_kmers(&kmers, 256, 21);
        let b = GenomeSketch::from_kmers(&kmers, 256, 21);

        let d = a.mash_distance(&b).expect("same k must not error");
        assert_eq!(d, 0.0);
    }

    #[test]
    fn mash_distance_of_disjoint_sketches_is_one() {
        let a = GenomeSketch::from_kmers(&(0u64..2_000).collect::<Vec<_>>(), 256, 21);
        let b = GenomeSketch::from_kmers(&(1_000_000u64..1_002_000).collect::<Vec<_>>(), 256, 21);

        let d = a.mash_distance(&b).expect("same k must not error");
        assert_eq!(d, 1.0);
    }

    #[test]
    fn mash_distance_decreases_as_jaccard_increases() {
        // Two pairs of sketches at different overlap levels: the pair with
        // higher Jaccard must report a lower (or equal) Mash distance --
        // the formula is monotonically decreasing in J.
        let base: Vec<u64> = (0..1_000).collect();
        let mostly_shared: Vec<u64> = (0..900).chain(2_000_000..2_000_100).collect();
        let barely_shared: Vec<u64> = (0..50).chain(3_000_000..3_000_950).collect();

        let a = GenomeSketch::from_kmers(&base, 500, 21);
        let b = GenomeSketch::from_kmers(&mostly_shared, 500, 21);
        let c = GenomeSketch::from_kmers(&barely_shared, 500, 21);

        let d_close = a.mash_distance(&b).expect("same k must not error");
        let d_far = a.mash_distance(&c).expect("same k must not error");

        assert!(
            d_close < d_far,
            "higher overlap must give a smaller distance: d_close={d_close}, d_far={d_far}"
        );
    }

    #[test]
    fn mash_distance_rejects_mismatched_k_like_jaccard() {
        let a = GenomeSketch::from_kmers(&[1, 2, 3], 10, 21);
        let b = GenomeSketch::from_kmers(&[1, 2, 3], 10, 25);

        match a.mash_distance(&b) {
            Err(FastDnaError::MismatchedK { left, right }) => {
                assert_eq!((left, right), (21, 25));
            }
            other => panic!("expected MismatchedK, got {other:?}"),
        }
    }

    #[test]
    fn disjoint_inputs_give_jaccard_zero() {
        let a_kmers: Vec<u64> = (0..2_000).collect();
        let b_kmers: Vec<u64> = (2_000..4_000).collect();
        let a = GenomeSketch::from_kmers(&a_kmers, 256, 21);
        let b = GenomeSketch::from_kmers(&b_kmers, 256, 21);

        let j = a.jaccard(&b).expect("same k must not error");
        assert_eq!(j, 0.0, "no shared k-mer means no shared hash, so intersection must be exactly zero");
    }

    /// The estimator test: two 100,000-element universes overlapping by
    /// exactly 50,000, giving a true Jaccard of 50,000 / 150,000 = 1/3. With
    /// `sketch_size = 512`, the bottom-k Jaccard estimator's standard error
    /// is approximately `sqrt(J * (1-J) / sketch_size)`
    /// = `sqrt((1/3) * (2/3) / 512)` ~= 2.08%.
    ///
    /// The tolerance is 0.05 (~2.4 standard errors), chosen to actually
    /// discriminate between the current splitmix64-style finalizer and the
    /// bare `wrapping_mul` (linear congruential) finalizer it replaced, not
    /// merely to bound a probabilistic estimator loosely. Measured directly
    /// against this exact test: the current finalizer's estimate is
    /// 0.298828125, |diff| = 0.0345 from the true 1/3, comfortably under
    /// 0.05; the old bare-`wrapping_mul` finalizer's estimate is
    /// 0.275390625, |diff| = 0.0579, which *fails* at 0.05. The tolerance
    /// used to be 0.09, under which both finalizers pass and this test
    /// cannot distinguish the improvement from a revert of it.
    #[test]
    fn known_overlap_estimates_jaccard_within_a_stated_tolerance() {
        let a_kmers: Vec<u64> = (0..100_000).collect();
        let b_kmers: Vec<u64> = (50_000..150_000).collect();
        let sketch_size = 512;

        let a = GenomeSketch::from_kmers(&a_kmers, sketch_size, 21);
        let b = GenomeSketch::from_kmers(&b_kmers, sketch_size, 21);

        let estimate = a.jaccard(&b).expect("same k must not error");
        let true_jaccard = 50_000.0 / 150_000.0;
        let tolerance = 0.05;

        assert!(
            (estimate - true_jaccard).abs() < tolerance,
            "estimate {estimate} too far from true Jaccard {true_jaccard} (tolerance {tolerance})"
        );
    }

    /// Demonstrates why both `jaccard` and `containment` exist. `a` is a
    /// small genome (50 distinct k-mers) wholly contained within `b`, a
    /// larger sample (500 distinct k-mers, 0..500, which by construction
    /// includes all of `a`'s 0..50). `sketch_size` is chosen larger than
    /// either set so both sketches capture their full k-mer set with no
    /// truncation.
    ///
    /// `containment` is genuinely exact set arithmetic here: with no
    /// truncation, `a_in_b`/`b_in_a` are computed by binary search over the
    /// complete hash sets, so they equal the true fraction of shared
    /// k-mers with no estimation involved.
    ///
    /// `jaccard` is exact here too, since the union-side fix: with both
    /// sketches non-full they hold their complete hash sets, and the merge
    /// walk now counts the longer list's tail as union-only members instead
    /// of stopping at the shorter list's end, so the ratio is genuinely
    /// 50/500 regardless of where `a`'s maximum hash falls in `b`'s order.
    /// The check is still kept loose deliberately: this test is about the
    /// asymmetry story, and the exact-value contract lives in
    /// `jaccard_counts_the_full_union_when_a_non_full_sketch_exhausts`.
    #[test]
    fn containment_is_asymmetric_where_jaccard_is_low() {
        let a_kmers: Vec<u64> = (0..50).collect();
        let b_kmers: Vec<u64> = (0..500).collect();
        let sketch_size = 1_000;

        let a = GenomeSketch::from_kmers(&a_kmers, sketch_size, 21);
        let b = GenomeSketch::from_kmers(&b_kmers, sketch_size, 21);

        let a_in_b = a.containment(&b).expect("same k must not error");
        let b_in_a = b.containment(&a).expect("same k must not error");
        let jaccard = a.jaccard(&b).expect("same k must not error");

        assert_eq!(a_in_b, 1.0, "a is wholly inside b, so containment(a, b) must be 1.0: {a_in_b}");
        assert_eq!(b_in_a, 0.1, "only 50 of b's 500 k-mers are shared, so containment(b, a) must be 50/500: {b_in_a}");
        assert!(
            (jaccard - 0.1).abs() < 0.05,
            "jaccard must land in the same ballpark as b_in_a (0.1), pulled down by the size \
             mismatch between a and b -- but exact equality to b_in_a is a coincidence of this \
             mixer's tie-breaking, not a property that must hold: jaccard={jaccard}"
        );
        assert!(a_in_b > b_in_a, "containment must be asymmetric: a_in_b={a_in_b} must exceed b_in_a={b_in_a}");
    }

    /// The union side of the Jaccard estimate must not stop at the end of
    /// the shorter hash list when that sketch is *not full*: a non-full
    /// sketch holds its complete k-mer set, so every remaining hash in the
    /// other list is a genuine union-only member. The old walk stopped
    /// anyway, shrinking the denominator only -- a systematic upward bias
    /// that reached 33x on small-versus-large comparisons.
    #[test]
    fn jaccard_counts_the_full_union_when_a_non_full_sketch_exhausts() {
        let a = GenomeSketch::from_kmers(&(0u64..3).collect::<Vec<_>>(), 1_000, 21);
        let b = GenomeSketch::from_kmers(&(0u64..100).collect::<Vec<_>>(), 1_000, 21);

        let j = a.jaccard(&b).expect("same k must not error");
        assert!(
            (j - 0.03).abs() < 1e-9,
            "both sketches are exact (non-full), so Jaccard must be exactly 3/100, got {j}"
        );
    }

    #[test]
    fn jaccard_of_disjoint_non_full_sketches_counts_both_sides_of_the_union() {
        let a = GenomeSketch::from_kmers(&(0u64..10).collect::<Vec<_>>(), 1_000, 21);
        let b = GenomeSketch::from_kmers(&(1_000_000u64..1_000_030).collect::<Vec<_>>(), 1_000, 21);

        let j = a.jaccard(&b).expect("same k must not error");
        assert_eq!(j, 0.0, "disjoint sets must give exactly 0.0, got {j}");
    }

    #[test]
    fn from_reader_rejects_an_out_of_range_k_instead_of_an_empty_sketch() {
        for bad_k in [0usize, 33] {
            match GenomeSketch::from_reader(reader_over(&["ACGTACGT"]), 128, bad_k, Path::new("s.fastq")) {
                Err(FastDnaError::InvalidK { k }) => assert_eq!(k, bad_k),
                other => panic!("k={bad_k} must be InvalidK, got {other:?}"),
            }
        }
    }

    #[test]
    fn from_reader_rejects_a_zero_sketch_size() {
        match GenomeSketch::from_reader(reader_over(&["ACGTACGT"]), 0, 21, Path::new("s.fastq")) {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => {
                assert_eq!(parameter, "sketch_size");
            }
            other => panic!("sketch_size=0 must be InvalidConfig, got {other:?}"),
        }
    }

    /// `from_kmers` has no `sketch_size` validation of its own (unlike
    /// `from_reader`), so a zero size must simply select nothing. The old
    /// `insert_bottom_k` spelled that out as an explicit `sketch_size == 0`
    /// early return, paid on every k-mer; `BottomK` instead starts a
    /// zero-capacity set as already-full with a maximum of 0, so the
    /// `hash < max` test rejects everything (no `u64` is below 0) at no
    /// per-k-mer cost. That is a subtle enough encoding of the same
    /// behaviour to be worth pinning, including the `u64::MAX` and `0`
    /// boundary hashes where an off-by-one in the comparison would show.
    #[test]
    fn a_zero_sketch_size_selects_no_hashes_at_all() {
        for kmers in [
            vec![],
            vec![0u64],
            vec![u64::MAX],
            vec![0u64, u64::MAX, 7, 7, 1],
            (0u64..1_000).collect::<Vec<_>>(),
        ] {
            let sketch = GenomeSketch::from_kmers(&kmers, 0, 21);
            assert!(
                sketch.hashes.is_empty(),
                "sketch_size 0 must keep nothing, kept {:?} from {} k-mers",
                sketch.hashes,
                kmers.len()
            );
        }
    }

    /// The bottom-k discipline is over *distinct* hashes: a k-mer stream
    /// repeats the same k-mer many times, and the working set deduplicates.
    /// This pins that a full sketch fed nothing but repeats of hashes it
    /// already holds neither grows nor evicts -- the property that rules out
    /// swapping the ordered set for a plain max-heap, which would fill with
    /// duplicates and retain fewer than `sketch_size` distinct hashes.
    #[test]
    fn repeated_kmers_do_not_displace_distinct_ones_from_a_full_sketch() {
        let distinct: Vec<u64> = (0..64).collect();
        let baseline = GenomeSketch::from_kmers(&distinct, 8, 21);
        assert_eq!(baseline.hashes.len(), 8, "the sketch must be full for this test to mean anything");

        // The same k-mers, each repeated 50 times, in a different order.
        let mut repeated: Vec<u64> = Vec::new();
        for _ in 0..50 {
            for &kmer in distinct.iter().rev() {
                repeated.push(kmer);
            }
        }
        let with_repeats = GenomeSketch::from_kmers(&repeated, 8, 21);

        assert_eq!(
            with_repeats.hashes, baseline.hashes,
            "duplicates and input order must not change which hashes bottom-k selects"
        );
    }

    #[test]
    fn load_rejects_a_sketch_file_with_an_out_of_range_k() {
        let dir = std::env::temp_dir().join("fastdna_sketch_bad_k_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_k.json");
        std::fs::write(&path, r#"{"sketch_size":128,"k":0,"hashes":[1,2,3]}"#).unwrap();

        match GenomeSketch::load(&path) {
            Err(FastDnaError::Load { .. }) => {}
            other => panic!("k=0 in a saved sketch must be a Load error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mismatched_k_is_an_error_not_a_panic() {
        let a = GenomeSketch::from_kmers(&[1, 2, 3], 256, 21);
        let b = GenomeSketch::from_kmers(&[1, 2, 3], 256, 31);

        match a.jaccard(&b) {
            Err(FastDnaError::MismatchedK { left, right }) => {
                assert_eq!(left, 21);
                assert_eq!(right, 31);
            }
            other => panic!("expected Err(MismatchedK), got {other:?}"),
        }

        match a.containment(&b) {
            Err(FastDnaError::MismatchedK { left, right }) => {
                assert_eq!(left, 21);
                assert_eq!(right, 31);
            }
            other => panic!("expected Err(MismatchedK), got {other:?}"),
        }
    }

    /// Exercises the bounded-MinHash ceiling branch, which the only other
    /// containment test (`containment_is_asymmetric_where_jaccard_is_low`)
    /// never reaches: there, `other`'s sketch is built with
    /// `sketch_size = 1_000` against only 500 k-mers, so it never fills,
    /// `other.hashes.len() >= other.sketch_size` is always false, and
    /// `ceiling` is always `u64::MAX` -- the bounded estimator silently
    /// degenerates to plain membership testing, and this is precisely the
    /// Mash-Screen-style behaviour (as opposed to a naive `|A∩B|/|A|`) that
    /// the code exists to implement.
    ///
    /// Sketches are built directly from hand-picked hashes rather than
    /// through `from_kmers`, so which hashes land above/below the ceiling
    /// is controlled exactly instead of depending on `finalize_hash`'s
    /// output order.
    #[test]
    fn containment_ceiling_excludes_hashes_the_other_sketch_could_not_have_kept() {
        // `other` is full (hashes.len() == sketch_size), so it has
        // provably discarded every hash above its current maximum (30):
        // ceiling = 30.
        let other = GenomeSketch { sketch_size: 3, k: 21, hashes: vec![10, 20, 30] };
        // `self` is not full (hashes.len() < sketch_size), so nothing of
        // its own is truncated; two of its five hashes (35, 45) exceed
        // `other`'s ceiling and must be excluded from both the numerator
        // and the denominator, not counted as misses.
        let this = GenomeSketch { sketch_size: 10, k: 21, hashes: vec![10, 15, 25, 35, 45] };

        let containment = this.containment(&other).expect("same k must not error");

        // Only {10, 15, 25} are resolvable (<= ceiling 30); of those, only
        // 10 is present in `other`. resolvable = 3, shared = 1.
        assert_eq!(
            containment,
            1.0 / 3.0,
            "expected 1 shared of 3 resolvable hashes, got {containment}"
        );
        // The naive (unbounded) alternative would divide by all 5 of
        // self's hashes instead of just the 3 `other` can attest to,
        // giving 1/5 = 0.2 -- a different, wrong answer. Asserting
        // inequality here pins down that the ceiling logic is actually
        // changing the result, not merely present and inert.
        assert_ne!(containment, 1.0 / 5.0, "the ceiling exclusion must change the result versus the naive, unbounded estimator");
    }

    #[test]
    fn save_then_load_round_trips_to_an_identical_sketch() {
        let kmers: Vec<u64> = (0..500).collect();
        let original = GenomeSketch::from_kmers(&kmers, 128, 21);

        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("fastdna_sketch_save_load_roundtrip_test.sig");
        original.save(&path).expect("save must succeed");
        let loaded = GenomeSketch::load(&path).expect("load must succeed");

        assert_eq!(loaded.sketch_size, original.sketch_size);
        assert_eq!(loaded.k, original.k);
        assert_eq!(loaded.hashes, original.hashes);
    }

    #[test]
    fn load_of_a_missing_file_is_an_io_error() {
        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("fastdna_sketch_does_not_exist_test.sig");

        match GenomeSketch::load(&path) {
            Err(FastDnaError::Io { .. }) => {}
            other => panic!("expected Err(Io), got {other:?}"),
        }
    }

    /// A corrupt/foreign file must surface as `FastDnaError::Load`, not
    /// `Export` -- `error.rs` renders `Export` as "export failed for
    /// {path}", which would tell a caller that *writing* failed while
    /// *loading* is what actually failed.
    #[test]
    fn load_of_a_corrupt_file_is_a_load_error_not_an_export_error() {
        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("fastdna_sketch_corrupt_test.sig");
        std::fs::write(&path, b"not valid json { at all").unwrap();

        match GenomeSketch::load(&path) {
            Err(FastDnaError::Load { .. }) => {}
            other => panic!("expected Err(Load), got {other:?}"),
        }
    }

    /// `jaccard`'s merge walk and `containment`'s binary search both
    /// assume `hashes` holds at most `sketch_size` entries. A hand-edited
    /// or otherwise corrupted file can violate that even though it is
    /// well-formed JSON, so this must be caught explicitly rather than
    /// silently producing a sketch whose `containment` ceiling logic is
    /// wrong.
    #[test]
    fn load_rejects_a_sketch_with_more_hashes_than_sketch_size() {
        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("fastdna_sketch_oversized_test.sig");
        let bad = GenomeSketch { sketch_size: 2, k: 21, hashes: vec![1, 2, 3] };
        std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();

        match GenomeSketch::load(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("sketch_size"), "reason should explain the mismatch: {reason}");
            }
            other => panic!("expected Err(Load), got {other:?}"),
        }
    }

    /// `jaccard`'s merge walk depends on `hashes` being sorted ascending;
    /// an unsorted vector breaks its early-termination logic silently
    /// (wrong, not a crash), so `load` must reject it instead of handing
    /// back a sketch that estimators will misuse.
    #[test]
    fn load_rejects_a_sketch_with_unsorted_hashes() {
        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("fastdna_sketch_unsorted_test.sig");
        let bad = GenomeSketch { sketch_size: 10, k: 21, hashes: vec![5, 1, 3] };
        std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();

        match GenomeSketch::load(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("sorted"), "reason should explain the ordering violation: {reason}");
            }
            other => panic!("expected Err(Load), got {other:?}"),
        }
    }

    /// The rewrite-safety test: proves the streaming construction path
    /// (`from_reader`, over a `FastqReader`) selects exactly the same
    /// bottom-k hashes as the in-memory path (`from_kmers`, over a
    /// pre-extracted slice) when fed the same underlying k-mers. This is
    /// what guarantees the streaming rewrite -- done to bound memory by
    /// `sketch_size` instead of input size -- did not change behaviour.
    #[test]
    fn streaming_construction_matches_in_memory_path() {
        let k = 5;
        let sketch_size = 1_000; // Larger than the distinct k-mer count below, so nothing is truncated and this is an exact equivalence check, not an approximate one.
        let reads = ["ACGTACGGTTACAGTCAGTCAGCATCGATCGACTAGCATGGGTTAACCGGTT", "TTGGCCAATTGGCCTAGCTAGCTAGGGCATCGATCGATCG"];

        // The same k-mers, taken through both extraction paths, so the
        // comparison isolates the bottom-k selection logic itself.
        let mut kmers: Vec<u64> = Vec::new();
        for seq in &reads {
            kmers.extend(kmer::extract_canonical_kmers(seq.as_bytes(), k));
        }
        let in_memory = GenomeSketch::from_kmers(&kmers, sketch_size, k);

        let reader = reader_over(&reads);
        let streamed =
            GenomeSketch::from_reader(reader, sketch_size, k, Path::new("<memory>")).expect("streaming build must succeed");

        assert_eq!(streamed.sketch_size, in_memory.sketch_size);
        assert_eq!(streamed.k, in_memory.k);
        assert_eq!(streamed.hashes, in_memory.hashes, "streaming and in-memory construction must select identical hashes");
        assert!(!streamed.hashes.is_empty(), "the test data must actually produce k-mers");
    }

    /// The rewrite-safety test above never exercises eviction:
    /// `sketch_size = 1_000` against ~84 k-mers means the bottom-k working
    /// set never fills, so `insert_bottom_k`'s eviction branch
    /// (`min_set.insert(hash)` then `pop_last()` once already full) never
    /// runs on either path. Bounded memory under truncation is the entire
    /// reason `from_reader`'s streaming construction exists instead of
    /// collecting every k-mer into a `Vec` first, so that branch needs its
    /// own equivalence check, with a `sketch_size` well below the distinct
    /// k-mer count so the sketch is forced to fill and then keep evicting.
    #[test]
    fn streaming_construction_matches_in_memory_path_when_truncated_by_eviction() {
        let k = 5;
        let sketch_size = 8;
        let reads = ["ACGTACGGTTACAGTCAGTCAGCATCGATCGACTAGCATGGGTTAACCGGTT", "TTGGCCAATTGGCCTAGCTAGCTAGGGCATCGATCGATCG"];

        let mut kmers: Vec<u64> = Vec::new();
        for seq in &reads {
            kmers.extend(kmer::extract_canonical_kmers(seq.as_bytes(), k));
        }
        assert!(
            kmers.len() > sketch_size,
            "test data must produce more k-mers ({}) than sketch_size ({sketch_size}), or eviction never triggers",
            kmers.len()
        );

        let in_memory = GenomeSketch::from_kmers(&kmers, sketch_size, k);
        assert_eq!(
            in_memory.hashes.len(),
            sketch_size,
            "the sketch must be full (truncated) for this test to actually exercise eviction"
        );

        let reader = reader_over(&reads);
        let streamed = GenomeSketch::from_reader(reader, sketch_size, k, Path::new("<memory>"))
            .expect("streaming build must succeed");

        assert_eq!(streamed.hashes.len(), sketch_size);
        assert_eq!(
            streamed.hashes, in_memory.hashes,
            "streaming and in-memory construction must select identical hashes even under repeated eviction"
        );
    }

    #[test]
    fn from_path_reports_malformed_fastq_with_a_record_number() {
        let dir = tempfile::tempdir().expect("failed to create temp dir for test");
        let path = dir.path().join("bad.fastq");

        let mut bytes = fastq_bytes(&["ACGTACGT"]);
        bytes.extend_from_slice(b"not-a-header-line\n");
        std::fs::write(&path, &bytes).unwrap();

        let err = GenomeSketch::from_path(&path, 256, 5).unwrap_err();

        match err {
            FastDnaError::MalformedFastq { record, .. } => assert_eq!(record, 2),
            other => panic!("expected MalformedFastq, got {other:?}"),
        }
    }

    /// Replaces a previous test (`finalize_hash_is_a_bijection_on_a_sample_of_inputs`)
    /// whose rationale was factually wrong and whose assertion was vacuous.
    /// It claimed the old bare-`wrapping_mul` finalizer "can map distinct
    /// inputs to the same output when they share low bits with the
    /// multiplier"; measured directly, it cannot -- `0x517cc1b727220a95` is
    /// odd, so multiplying by it modulo 2^64 is a bijection on `u64` with
    /// exactly zero collisions, confirmed here over `0..10_000`. A
    /// bijection check can never distinguish a linear step from a real
    /// mixer, because both are bijections.
    ///
    /// What actually separates them is avalanche: flipping one input bit
    /// should flip roughly half of the 64 output bits, for every input bit
    /// position. Multiplication by an odd constant does not have this
    /// property -- a bit flip at input position `i` only affects output
    /// bits `>= i` (there is no carry propagation downward), so flipping
    /// the top input bit (63) can only ever change output bit 63. Measured
    /// directly against these exact finalizers over 64 sample inputs: the
    /// current splitmix64-style finalizer flips ~31.9 output bits on
    /// average per input-bit flip (minimum per-bit average ~31.0, close to
    /// the ideal 32); the old bare-`wrapping_mul` finalizer flips ~17.6 on
    /// average, and its worst input bit (63) flips only 1.0 output bits on
    /// average -- it does not mix at all for that bit. The thresholds below
    /// sit well clear of both measurements in both directions.
    #[test]
    fn finalize_hash_has_strong_avalanche_bit_diffusion() {
        let samples: Vec<u64> =
            (0u64..64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1)).collect();

        let mut total_flipped: u64 = 0;
        let mut comparisons: u64 = 0;
        let mut min_per_bit_avg = f64::INFINITY;

        for bit in 0u32..64 {
            let mut flipped_for_bit: u64 = 0;
            for &x in &samples {
                let base = finalize_hash(x);
                let perturbed = finalize_hash(x ^ (1u64 << bit));
                flipped_for_bit += (base ^ perturbed).count_ones() as u64;
            }
            total_flipped += flipped_for_bit;
            comparisons += samples.len() as u64;
            let bit_avg = flipped_for_bit as f64 / samples.len() as f64;
            if bit_avg < min_per_bit_avg {
                min_per_bit_avg = bit_avg;
            }
        }

        let overall_avg = total_flipped as f64 / comparisons as f64;

        assert!(
            overall_avg > 28.0,
            "overall avalanche too weak: {overall_avg} flipped bits/flip on average (want > 28, ideal 32)"
        );
        assert!(
            min_per_bit_avg > 20.0,
            "at least one input bit position diffuses poorly: {min_per_bit_avg} flipped bits/flip \
             on average (want > 20) -- a linear finalizer leaves some input bits (e.g. the top bit) \
             barely affecting the output"
        );
    }
}
