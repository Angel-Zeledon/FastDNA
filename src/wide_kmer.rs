// src/wide_kmer.rs
//! K-mer encoding for `33 <= k <= 64`, in `u128` instead of `u64`.
//!
//! # Why a second engine rather than making the first one generic
//!
//! `kmer.rs` packs two bits per base into a `u64`, which is exactly 32
//! bases and not one more. Supporting longer k-mers needs a wider integer,
//! and there were two ways to get one: make `kmer.rs` (and `counter.rs`,
//! and every caller) generic over the word type, or add a parallel
//! implementation for the wide case and route to it by `k`.
//!
//! This is the second. The reason is not taste:
//!
//! * The `u64` path is the one measured **exactly equal to KMC3** on real
//!   reads (`scripts/validation/kmc3_equivalence.py`), and the one every
//!   benchmark in `docs/BENCHMARKS.md` describes. Making it generic would
//!   put every one of those results back in question -- monomorphisation
//!   should preserve the codegen, but "should" is not what a validated
//!   number rests on, and re-validating the whole engine is a far larger
//!   job than writing this file.
//! * `u128` arithmetic is not free on a 64-bit machine: shifts and
//!   comparisons become multi-instruction sequences. A generic engine
//!   would make the common case (`k = 31`, the field's default) pay
//!   nothing at runtime but would make every future change to the hot loop
//!   reason about both widths at once. Two files that each do one thing
//!   are easier to keep correct than one file that does both.
//! * `k > 32` is a real but uncommon request. The right trade for a
//!   capability most runs never touch is isolation, not entanglement.
//!
//! The cost of that decision is duplication, and it is paid honestly:
//! every function here mirrors one in `kmer.rs`, and
//! `wide_matches_the_narrow_engine_exactly_where_they_overlap` asserts the
//! two agree k-mer for k-mer over the range where both are defined. That
//! test is the reason this file may diverge in *width* but cannot diverge
//! in *meaning*.
//!
//! # What is shared, deliberately
//!
//! The alphabet. [`crate::kmer::base_to_bits`] is the single definition of
//! which bytes are bases and what two bits each maps to, and this module
//! calls it rather than restating the table. Two encodings would be two
//! truths: a canonical k-mer computed here would not be comparable with
//! one computed there, and nothing in the type system would say so.
//!
//! # Range
//!
//! `1..=64`. The engine is mathematically fine below 33 and the
//! differential test relies on that, but `pipeline.rs` routes only
//! `33..=64` here -- below that the narrow engine is faster and is the one
//! with the external validation behind it.
//!
//! 64, not KMC3's 256: two bits per base in 128 bits is 64 bases, and
//! going beyond needs a byte-array key, which changes the sort, the
//! Parquet schema and every comparison in the counter. That is a different
//! project, and `docs/feature-gap-analysis.md` records it as still open
//! rather than pretending this closes it.

use crate::kmer::base_to_bits;

/// The largest `k` two bits per base fits in a `u128`.
pub const MAX_WIDE_K: usize = 64;

/// The smallest `k` that needs this engine at all; at or below it,
/// `kmer.rs` is both faster and externally validated.
pub const MIN_WIDE_K: usize = 33;

/// `0x3333...` and `0x0F0F...`, widened to 128 bits.
///
/// Written as shifted constructions rather than as 32 literal hex digits
/// because a typo in a 32-digit literal is invisible on review, and these
/// two masks are what make the reverse complement correct.
const MASK_2BIT_PAIRS: u128 = 0x3333_3333_3333_3333_3333_3333_3333_3333;
const MASK_4BIT_PAIRS: u128 = 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;

/// Complements a single 2-bit base code.
///
/// The identity is [`crate::kmer::complement_bits`]'s, at a different
/// width: `A<->T` and `C<->G` are `0b00<->0b11` and `0b01<->0b10` under
/// `base_to_bits`'s mapping, which is `0b11 ^ b`. It is restated here
/// rather than called across the width boundary only to avoid a cast in
/// the hot loop; if `base_to_bits` ever remaps the alphabet, both this and
/// its narrow twin must be re-derived together.
#[inline(always)]
pub fn complement_bits(bits: u128) -> u128 {
    0b11 ^ bits
}

/// The reverse complement of a `k`-mer packed two bits per base.
///
/// The same three-step shuffle `kmer::reverse_complement_u64` documents,
/// widened: complement every base with `!` (which flips both bits of every
/// 2-bit field, i.e. `complement_bits` applied in parallel), then reverse
/// the order of the bases by swapping adjacent 2-bit fields, then adjacent
/// 4-bit fields, then reversing the bytes -- and finally shift the result
/// down so the `k` bases sit in the low `2k` bits.
pub fn reverse_complement_u128(kmer: u128, k: usize) -> u128 {
    debug_assert!(
        (1..=MAX_WIDE_K).contains(&k),
        "reverse_complement_u128: k must be in 1..={MAX_WIDE_K}, got {k}"
    );
    let mut v = !kmer;
    v = ((v >> 2) & MASK_2BIT_PAIRS) | ((v & MASK_2BIT_PAIRS) << 2);
    v = ((v >> 4) & MASK_4BIT_PAIRS) | ((v & MASK_4BIT_PAIRS) << 4);
    v = v.swap_bytes();
    // `k == 64` fills the word, and `u128 >> 128` is undefined; the guard
    // is the same one `reverse_complement_u64` needs at `k == 32` and gets
    // for free from its mask. Here it is explicit.
    if k == MAX_WIDE_K {
        v
    } else {
        v >> (128 - (2 * k))
    }
}

/// The canonical k-mer: the smaller of the k-mer and its reverse
/// complement, exactly as [`crate::kmer::canonical_kmer_u64`] defines it
/// for the narrow engine.
#[inline(always)]
pub fn canonical_kmer_u128(kmer: u128, k: usize) -> u128 {
    kmer.min(reverse_complement_u128(kmer, k))
}

/// Extracts every canonical k-mer from `seq` into `out`, `O(1)` per base.
///
/// Behaviour matches [`crate::kmer::extract_canonical_kmers_into`] in every
/// respect that is not the width: `out` is cleared first, an ambiguous base
/// resets the window rather than corrupting the k-mers spanning it, and the
/// forward and reverse-complement registers are rolled together so the
/// canonical form costs a `min` rather than a recomputed reverse
/// complement.
pub fn extract_canonical_kmers_into(seq: &[u8], k: usize, out: &mut Vec<u128>) {
    out.clear();
    if seq.len() < k || k == 0 || k > MAX_WIDE_K {
        return;
    }
    out.reserve(seq.len() - k + 1);

    let mask = if k == MAX_WIDE_K { u128::MAX } else { (1u128 << (2 * k)) - 1 };
    let top_shift = 2 * (k - 1);

    let mut fwd: u128 = 0;
    let mut rev: u128 = 0;
    let mut valid_len = 0;

    for &base in seq {
        if let Some(bits) = base_to_bits(base) {
            let bits = u128::from(bits);
            fwd = ((fwd << 2) | bits) & mask;
            rev = (rev >> 2) | (complement_bits(bits) << top_shift);
            valid_len += 1;

            if valid_len >= k {
                out.push(fwd.min(rev));
            }
        } else {
            fwd = 0;
            rev = 0;
            valid_len = 0;
        }
    }
}

/// Allocating convenience over [`extract_canonical_kmers_into`], for tests
/// and one-off callers. The hot path uses the buffer-reusing form.
pub fn extract_canonical_kmers(seq: &[u8], k: usize) -> Vec<u128> {
    let mut out = Vec::new();
    extract_canonical_kmers_into(seq, k, &mut out);
    out
}

/// Decodes a packed k-mer back into its ASCII bases, appending to `out`.
///
/// The inverse of the packing in [`extract_canonical_kmers_into`]: the
/// most significant 2-bit field is the first base, so the bases come out
/// high-to-low.
pub fn decode_kmer_into(kmer: u128, k: usize, out: &mut Vec<u8>) {
    debug_assert!(
        (1..=MAX_WIDE_K).contains(&k),
        "decode_kmer_into: k must be in 1..={MAX_WIDE_K}, got {k}"
    );
    const BASES: [u8; 4] = *b"ACGT";
    out.reserve(k);
    for position in (0..k).rev() {
        let bits = ((kmer >> (2 * position)) & 0b11) as usize;
        out.push(BASES[bits]);
    }
}

/// Allocating form of [`decode_kmer_into`].
pub fn decode_kmer(kmer: u128, k: usize) -> String {
    let mut bytes = Vec::with_capacity(k);
    decode_kmer_into(kmer, k, &mut bytes);
    // Every byte pushed above is one of `ACGT`, so this cannot fail; the
    // lossy form keeps the crate's `unwrap_used` denial satisfied without
    // an `expect` that claims more than it can prove.
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The 16-byte big-endian form of a packed k-mer, which is how a wide
/// k-mer is stored in Parquet (`export.rs`).
///
/// Big-endian on purpose: byte-wise lexicographic order over these keys is
/// the same order as numeric order over the `u128` they encode, so a table
/// sorted by the counter's own `u128` comparison is also sorted by the
/// bytes a reader compares without decoding them. That is what lets the
/// wide table keep the `fastdna.sorted_by` contract `ktab.rs` established
/// for the narrow one.
#[inline]
pub fn to_key_bytes(kmer: u128) -> [u8; 16] {
    kmer.to_be_bytes()
}

/// Inverse of [`to_key_bytes`].
#[inline]
pub fn from_key_bytes(bytes: [u8; 16]) -> u128 {
    u128::from_be_bytes(bytes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn random_sequence(len: usize, seed: u64) -> Vec<u8> {
        // xorshift64*, so the test data is deterministic without a
        // dependency. Same approach the other modules' tests take.
        let mut state = seed | 1;
        let bases = b"ACGT";
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bases[(state % 4) as usize]
            })
            .collect()
    }

    /// **The load-bearing test of this file.** For every `k` both engines
    /// accept, the wide engine must produce exactly the narrow engine's
    /// k-mers -- same values, same order, same count -- on the same input.
    ///
    /// The narrow engine is the one validated against KMC3 on real reads.
    /// If this holds, the wide engine inherits that validation for the
    /// overlapping range, and any disagreement above `k = 32` is then a
    /// question about width alone rather than about whether the two agree
    /// on what a canonical k-mer is at all.
    #[test]
    fn wide_matches_the_narrow_engine_exactly_where_they_overlap() {
        for seed in [1u64, 7, 99, 4242] {
            let seq = random_sequence(500, seed);
            for k in 1..=32usize {
                let narrow = crate::kmer::extract_canonical_kmers(&seq, k);
                let wide = extract_canonical_kmers(&seq, k);
                assert_eq!(
                    narrow.len(),
                    wide.len(),
                    "k={k}, seed={seed}: engines disagree on how many k-mers a \
                     {}-base sequence yields",
                    seq.len()
                );
                for (index, (n, w)) in narrow.iter().zip(wide.iter()).enumerate() {
                    assert_eq!(
                        u128::from(*n),
                        *w,
                        "k={k}, seed={seed}, k-mer {index}: narrow {n:#x} vs wide {w:#x}"
                    );
                }
            }
        }
    }

    /// The same agreement on sequences containing ambiguous bases, which is
    /// where a window-reset bug would hide: both engines must drop exactly
    /// the k-mers that span an `N`, not merely a similar number of them.
    #[test]
    fn wide_matches_the_narrow_engine_across_ambiguous_bases() {
        let seq = b"ACGTACGTNNACGTACGTACGTNACGTACGTACGTACGT";
        for k in 1..=32usize {
            let narrow = crate::kmer::extract_canonical_kmers(seq, k);
            let wide = extract_canonical_kmers(seq, k);
            let narrow_wide: Vec<u128> = narrow.iter().map(|&n| u128::from(n)).collect();
            assert_eq!(narrow_wide, wide, "k={k}: engines disagree around ambiguous bases");
        }
    }

    /// A k-mer's reverse complement's reverse complement is itself, at
    /// every width this engine supports including the boundary where the
    /// shift would be undefined.
    #[test]
    fn reverse_complement_is_an_involution_at_every_k() {
        for k in [1usize, 31, 32, 33, 45, 63, 64] {
            let seq = random_sequence(k, k as u64 * 31 + 7);
            let kmers = extract_canonical_kmers(&seq, k);
            for kmer in kmers {
                let back = reverse_complement_u128(reverse_complement_u128(kmer, k), k);
                assert_eq!(kmer, back, "k={k}: rc(rc(x)) != x for {kmer:#x}");
            }
        }
    }

    /// Canonicality is what makes a count strand-independent: a sequence
    /// and its reverse complement must produce the same multiset of
    /// canonical k-mers.
    #[test]
    fn a_sequence_and_its_reverse_complement_yield_the_same_kmers() {
        for k in [33usize, 40, 63, 64] {
            let seq = random_sequence(200, k as u64 * 17 + 3);
            let rc: Vec<u8> = seq
                .iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'C' => b'G',
                    b'G' => b'C',
                    _ => b'A',
                })
                .collect();

            let mut forward = extract_canonical_kmers(&seq, k);
            let mut reverse = extract_canonical_kmers(&rc, k);
            forward.sort_unstable();
            reverse.sort_unstable();
            assert_eq!(forward, reverse, "k={k}: strand changes the canonical k-mer multiset");
        }
    }

    /// Encode/decode round-trips at every supported width, including 64
    /// where the packed value fills the word.
    #[test]
    fn decode_round_trips_the_encoding() {
        for k in [33usize, 34, 50, 63, 64] {
            let seq = random_sequence(k, k as u64 * 101 + 5);
            let kmers = extract_canonical_kmers(&seq, k);
            assert_eq!(kmers.len(), 1, "k={k}: a k-base sequence yields exactly one k-mer");
            let decoded = decode_kmer(kmers[0], k);
            assert_eq!(decoded.len(), k, "k={k}: decoded length");
            // The decoded string re-encodes to the value it came from.
            let re_encoded = extract_canonical_kmers(decoded.as_bytes(), k);
            assert_eq!(re_encoded, kmers, "k={k}: decode -> encode is not the identity");
        }
    }

    /// The property `to_key_bytes` exists for: byte order over the stored
    /// keys is numeric order over the values, which is what keeps a wide
    /// table sorted for a reader that never decodes it.
    #[test]
    fn big_endian_key_bytes_sort_like_the_numbers_they_encode() {
        let seq = random_sequence(4000, 20_260_905);
        let k = 41;
        let mut kmers = extract_canonical_kmers(&seq, k);
        kmers.sort_unstable();
        kmers.dedup();
        assert!(kmers.len() > 100, "need a real spread of values to compare orders");

        let keys: Vec<[u8; 16]> = kmers.iter().map(|&x| to_key_bytes(x)).collect();
        for window in keys.windows(2) {
            assert!(window[0] < window[1], "byte order disagrees with numeric order");
        }
        for (kmer, key) in kmers.iter().zip(keys.iter()) {
            assert_eq!(from_key_bytes(*key), *kmer, "key bytes do not round-trip");
        }
    }

    /// A sequence shorter than `k`, an empty one, and an all-`N` one all
    /// yield nothing rather than panicking or emitting a partial k-mer --
    /// the same contract `kmer.rs` holds.
    #[test]
    fn degenerate_inputs_yield_no_kmers() {
        assert!(extract_canonical_kmers(b"", 33).is_empty());
        assert!(extract_canonical_kmers(b"ACGT", 33).is_empty());
        assert!(extract_canonical_kmers(&[b'N'; 100], 33).is_empty());
        // Above the width this engine can hold, it declines rather than
        // truncating: 65 bases do not fit in 128 bits.
        assert!(extract_canonical_kmers(&random_sequence(200, 1), MAX_WIDE_K + 1).is_empty());
    }
}
