use std::cmp;

/// FEATURE 1: Fast reverse complement.
/// DNA has two strands. If we read "ATGC", its mirror is "GCAT".
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev() // reverse the order
        .map(|&c| match c {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            _ => c, // leave noise (N) untouched
        })
        .collect()
}

/// Returns the canonical form of the k-mer (the lexicographic minimum).
/// This halves RAM usage and improves downstream machine learning.
pub fn canonical_kmer(seq: &[u8]) -> Vec<u8> {
    let rev = reverse_complement(seq);
    // Compare the bytes of both sequences and keep the smaller one
    if seq < &rev[..] {
        seq.to_vec()
    } else {
        rev
    }
}

/// FEATURE 4: MinHash (probabilistic fingerprint).
/// Takes a sequence and returns a hash representing it.
pub fn hash_kmer(seq: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    seq.hash(&mut hasher);
    hasher.finish()
}
