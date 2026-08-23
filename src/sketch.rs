// src/sketch.rs
//! MinHash sketching for fast, approximate genome-to-genome comparison
//! (design doc §8). Instead of counting every k-mer, a `GenomeSketch` keeps
//! only the `sketch_size` smallest hashes as a fingerprint; comparing two
//! fingerprints estimates how similar the full k-mer sets are without ever
//! materializing them side by side. Two SARS-CoV-2 samples that would take
//! minutes to compare exactly can be compared in milliseconds this way.

use std::collections::BTreeSet;
use std::io::BufRead;
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
fn finalize_hash(kmer: u64) -> u64 {
    let mut z = kmer;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Folds one already-finalized hash into a bottom-`sketch_size` working set,
/// evicting the current maximum once the set is full. Shared by `from_kmers`
/// (in-memory) and `from_reader` (streaming) so both construction paths run
/// through identical selection logic -- see
/// `streaming_construction_matches_in_memory_path` below, which exists
/// specifically to prove that sharing pays off.
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

    /// Builds a sketch from every k-mer already held in memory as a slice.
    /// Peak memory is `O(kmers.len())` for the input plus `O(sketch_size)`
    /// for the working set -- fine for small inputs or when the k-mers are
    /// already materialized for another reason, but see `from_reader`/
    /// `from_path` for large FASTQ files, where materializing every k-mer
    /// up front would defeat the point of sketching.
    pub fn from_kmers(kmers: &[u64], sketch_size: usize, k: usize) -> Self {
        let mut min_set: BTreeSet<u64> = BTreeSet::new();

        for &kmer in kmers {
            insert_bottom_k(&mut min_set, sketch_size, finalize_hash(kmer));
        }

        Self { sketch_size, k, hashes: min_set.into_iter().collect() }
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
        let mut min_set: BTreeSet<u64> = BTreeSet::new();
        let mut record_count: u64 = 0;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    record_count += 1;
                    for kmer in kmer::extract_canonical_kmers(&record.seq, k) {
                        insert_bottom_k(&mut min_set, sketch_size, finalize_hash(kmer));
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

        Ok(Self { sketch_size, k, hashes: min_set.into_iter().collect() })
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

    #[test]
    fn from_path_reports_malformed_fastq_with_a_record_number() {
        let dir = std::env::temp_dir().join("fastdna_sketch_malformed_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.fastq");

        let mut bytes = fastq_bytes(&["ACGTACGT"]);
        bytes.extend_from_slice(b"not-a-header-line\n");
        std::fs::write(&path, &bytes).unwrap();

        let err = GenomeSketch::from_path(&path, 256, 5).unwrap_err();
        let _ = std::fs::remove_file(&path);

        match err {
            FastDnaError::MalformedFastq { record, .. } => assert_eq!(record, 2),
            other => panic!("expected MalformedFastq, got {other:?}"),
        }
    }
}
