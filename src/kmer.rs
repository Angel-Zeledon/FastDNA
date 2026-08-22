// src/kmer.rs

/// Converts an ASCII nucleotide to its 2-bit representation.
/// A/a = 00, C/c = 01, G/g = 10, T/t/U/u = 11.
/// Returns `None` for invalid or ambiguous characters (such as 'N').
#[inline(always)]
pub fn base_to_bits(base: u8) -> Option<u64> {
    match base {
        b'A' | b'a' => Some(0b00),
        b'C' | b'c' => Some(0b01),
        b'G' | b'g' => Some(0b10),
        b'T' | b't' | b'U' | b'u' => Some(0b11),
        _ => None,
    }
}

/// Computes the reverse complement of a u64-encoded k-mer in O(1)
/// using CPU register-level bit operations.
#[inline(always)]
pub fn reverse_complement_u64(kmer: u64, k: usize) -> u64 {
    let mut v = !kmer;
    v = ((v >> 2) & 0x3333_3333_3333_3333) | ((v & 0x3333_3333_3333_3333) << 2);
    v = ((v >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4);
    v = v.swap_bytes();
    v >> (64 - (2 * k))
}

/// Returns the canonical k-mer (the lexicographic minimum of the k-mer and its
/// reverse complement).
#[inline(always)]
pub fn canonical_kmer_u64(kmer: u64, k: usize) -> u64 {
    let rc = reverse_complement_u64(kmer, k);
    kmer.min(rc)
}

/// Extracts every canonical k-mer from a DNA sequence using an O(1)-per-base
/// rolling window. Ambiguous bases ('N') reset the window automatically instead
/// of producing corrupt k-mers.
pub fn extract_canonical_kmers(seq: &[u8], k: usize) -> Vec<u64> {
    if seq.len() < k || k == 0 || k > 32 {
        return Vec::new();
    }

    let mut kmers = Vec::with_capacity(seq.len() - k + 1);
    let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };

    let mut current_kmer: u64 = 0;
    let mut valid_len = 0;

    for &base in seq {
        if let Some(bits) = base_to_bits(base) {
            current_kmer = ((current_kmer << 2) | bits) & mask;
            valid_len += 1;

            if valid_len >= k {
                kmers.push(canonical_kmer_u64(current_kmer, k));
            }
        } else {
            current_kmer = 0;
            valid_len = 0;
        }
    }

    kmers
}

/// Decodes a u64 into its ASCII string representation of length k.
pub fn decode_kmer(mut kmer: u64, k: usize) -> String {
    let mut chars = vec![b'A'; k];
    for i in (0..k).rev() {
        chars[i] = match kmer & 0b11 {
            0b00 => b'A',
            0b01 => b'C',
            0b10 => b'G',
            0b11 => b'T',
            _ => unreachable!(),
        };
        kmer >>= 2;
    }
    String::from_utf8(chars).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_encoding_and_decoding() {
        let seq = "ACGT";
        let k = 4;
        let kmers = extract_canonical_kmers(seq.as_bytes(), k);
        assert_eq!(kmers.len(), 1);

        let decoded = decode_kmer(kmers[0], k);
        assert!(decoded == "ACGT" || decoded == "ACGT");
    }

    #[test]
    fn test_reverse_complement_symmetry() {
        // "AACG" (00 00 01 10) -> its reverse complement is "CGTT" (01 10 11 11)
        let k = 4;
        let original: u64 = 0b00_00_01_10;
        let rc = reverse_complement_u64(original, k);
        let expected_rc: u64 = 0b01_10_11_11;
        assert_eq!(rc, expected_rc);
        assert_eq!(reverse_complement_u64(rc, k), original);
    }

    #[test]
    fn test_ambiguous_base_reset() {
        let seq = b"ACGTNACGT";
        let kmers = extract_canonical_kmers(seq, 4);
        assert_eq!(kmers.len(), 2);
    }
}
