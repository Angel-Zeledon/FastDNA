// src/kmer.rs

/// Table sentinel for a byte that is not a nucleotide. `0xFF` cannot collide
/// with a real code, which is always one of `0b00..=0b11`.
const INVALID_BASE: u8 = 0xFF;

/// ASCII byte -> 2-bit nucleotide code, or [`INVALID_BASE`].
///
/// This is a table rather than a `match` for a reason that is visible in the
/// generated code, not guessed at. Written as
/// `match base { b'A' | b'a' => .., b'C' | b'c' => .., .. }` the compiler
/// lowers the nine accepted bytes to a **jump table with an indirect
/// branch**, run once per base:
///
/// ```text
/// addl   $-65, %r8d          ; base - 'A'
/// cmpl   $52, %r8d
/// ja     <invalid>
/// movslq (%r9,%r8,4), %r8    ; load from a 53-entry jump table
/// addq   %r9, %r8
/// jmpq   *%r8                ; indirect branch, 4 live targets
/// <target>: movl $N, %eax ; jmp <join>
/// ```
///
/// That is ~8 instructions, one table load and two branches -- and the
/// indirect one has four roughly equiprobable targets on real DNA, so it
/// mispredicts most of the time, at a full pipeline flush each. The table
/// form is one load from a 256-byte array plus one compare against the
/// sentinel and one branch that is taken only on an ambiguous base:
///
/// ```text
/// movzbl BASE_BITS(%rip,%rax), %eax
/// cmpb   $-1, %al
/// je     <invalid>
/// ```
///
/// Strictly fewer instructions, the same number of loads (the jump table was
/// already a load, from a 212-byte table), and an unpredictable indirect
/// branch traded for a predictable direct one. It is not a cache gamble: the
/// 256-byte table is four cache lines read on every base of every read for
/// the whole run, which is the most resident data in the program.
const BASE_BITS: [u8; 256] = {
    let mut table = [INVALID_BASE; 256];
    table[b'A' as usize] = 0b00;
    table[b'a' as usize] = 0b00;
    table[b'C' as usize] = 0b01;
    table[b'c' as usize] = 0b01;
    table[b'G' as usize] = 0b10;
    table[b'g' as usize] = 0b10;
    table[b'T' as usize] = 0b11;
    table[b't' as usize] = 0b11;
    // RNA: U pairs with A exactly as T does, so it shares T's code.
    table[b'U' as usize] = 0b11;
    table[b'u' as usize] = 0b11;
    table
};

/// Converts an ASCII nucleotide to its 2-bit representation.
/// A/a = 00, C/c = 01, G/g = 10, T/t/U/u = 11.
/// Returns `None` for invalid or ambiguous characters (such as 'N').
#[inline(always)]
pub fn base_to_bits(base: u8) -> Option<u64> {
    // `base` is a `u8` and the table has 256 entries, so this index can never
    // be out of range and the bounds check is removed.
    let bits = BASE_BITS[base as usize];
    if bits == INVALID_BASE {
        None
    } else {
        Some(bits as u64)
    }
}

/// Computes the reverse complement of a u64-encoded k-mer in O(1)
/// using CPU register-level bit operations.
///
/// `k` must be in `1..=32`: `k == 0` shifts by 64 (panics in debug, silently
/// wrong in release) and `k > 32` overflows the `64 - (2 * k)` subtraction
/// the same way. `extract_canonical_kmers` already guards its own calls, but
/// this function is `pub` and reachable directly, so callers outside this
/// module -- including a future Python binding -- get the same guarantee.
/// A `debug_assert!` is used rather than a runtime branch because this
/// function is `#[inline(always)]` on the k-mer extraction hot path, where a
/// branch that always evaluates true still costs measurable throughput; the
/// assert compiles to nothing in release builds.
#[inline(always)]
pub fn reverse_complement_u64(kmer: u64, k: usize) -> u64 {
    debug_assert!((1..=32).contains(&k), "reverse_complement_u64: k must be in 1..=32, got {k}");
    let mut v = !kmer;
    v = ((v >> 2) & 0x3333_3333_3333_3333) | ((v & 0x3333_3333_3333_3333) << 2);
    v = ((v >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4);
    v = v.swap_bytes();
    v >> (64 - (2 * k))
}

/// Returns the canonical k-mer (the lexicographic minimum of the k-mer and its
/// reverse complement).
///
/// Same `k` constraint as `reverse_complement_u64`, which this delegates to.
#[inline(always)]
pub fn canonical_kmer_u64(kmer: u64, k: usize) -> u64 {
    debug_assert!((1..=32).contains(&k), "canonical_kmer_u64: k must be in 1..=32, got {k}");
    let rc = reverse_complement_u64(kmer, k);
    kmer.min(rc)
}

/// Complements a single 2-bit base code.
///
/// Derived from `base_to_bits`, not assumed: that function maps
/// `A -> 0b00`, `C -> 0b01`, `G -> 0b10`, `T -> 0b11`. Watson-Crick pairing
/// is `A<->T` and `C<->G`, i.e. `0b00<->0b11` and `0b01<->0b10`, which is
/// exactly `b -> 0b11 - b`. For a 2-bit value `0b11 - b == 0b11 ^ b`
/// (no borrow is possible), so the complement is a single XOR:
///
/// ```text
/// 0b11 ^ 0b00 = 0b11   A -> T
/// 0b11 ^ 0b01 = 0b10   C -> G
/// 0b11 ^ 0b10 = 0b01   G -> C
/// 0b11 ^ 0b11 = 0b00   T -> A
/// ```
///
/// This identity holds only for this encoding. If `base_to_bits` ever
/// changes its mapping, this function and the rolled reverse complement in
/// `extract_canonical_kmers_into` must be re-derived together.
#[inline(always)]
pub fn complement_bits(bits: u64) -> u64 {
    0b11 ^ bits
}

/// Extracts every canonical k-mer from a DNA sequence into a caller-owned
/// buffer, using an O(1)-per-base rolling window. Ambiguous bases ('N') reset
/// the window automatically instead of producing corrupt k-mers.
///
/// `out` is cleared first, so the caller can hand the same `Vec` back on every
/// call and pay for its allocation once instead of once per read. The pipeline
/// workers do exactly that: at ~7 million reads for a 2.14 GB FASTQ, the
/// allocating [`extract_canonical_kmers`] wrapper costs 7 million
/// malloc/free pairs that this variant costs zero of.
///
/// # The reverse complement is rolled, not recomputed
///
/// The obvious implementation calls [`canonical_kmer_u64`] per k-mer, which
/// calls [`reverse_complement_u64`], which is a full 64-bit reversal:
/// `!kmer` (1 op), two mask-and-shift swap steps (5 ops each: two shifts,
/// two ANDs, one OR), `swap_bytes` (1 op) and a final shift (1 op) --
/// 13 ALU operations, once per k-mer. At 840 million k-mer occurrences that
/// is ~1.1e10 operations spent recomputing a value that only changes by one
/// base per step.
///
/// Instead both strands are rolled side by side. Writing the forward k-mer
/// with the most recently read base in the *low* two bits (which is what
/// `fwd = (fwd << 2) | bits` does), its reverse complement holds the same
/// base, complemented, in the *high* two bits -- because reversing the
/// sequence sends position `k-1` to position `0`. So on each new base:
///
/// ```text
/// fwd = ((fwd << 2) | bits) & mask                  // oldest base falls off the top
/// rev =  (rev >> 2) | (complement(bits) << 2*(k-1)) // oldest base falls off the bottom
/// ```
///
/// The base leaving the forward window is the one leaving the low end of
/// `rev`, so `rev >> 2` drops exactly the right base and no mask is needed:
/// `rev` never exceeds `2*k` bits. That is 4 ALU operations per base (XOR,
/// two shifts by loop-invariant amounts, OR) replacing 13 per k-mer -- a net
/// 9 operations saved per k-mer, ~7.6e9 on the 840-million-k-mer benchmark.
///
/// A run of `k` rolls after a window reset fully overwrites `rev`, so
/// resetting it to 0 alongside `fwd` on an ambiguous base keeps the two
/// strands in step; the first k-mer emitted after a reset has had exactly
/// `k` bases rolled into both.
pub fn extract_canonical_kmers_into(seq: &[u8], k: usize, out: &mut Vec<u64>) {
    out.clear();
    if seq.len() < k || k == 0 || k > 32 {
        return;
    }
    out.reserve(seq.len() - k + 1);

    let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    // Position of the highest base slot; loop-invariant, so the shift amount
    // is hoisted out of the loop below.
    let top_shift = 2 * (k - 1);

    let mut fwd: u64 = 0;
    let mut rev: u64 = 0;
    let mut valid_len = 0;

    for &base in seq {
        if let Some(bits) = base_to_bits(base) {
            fwd = ((fwd << 2) | bits) & mask;
            rev = (rev >> 2) | (complement_bits(bits) << top_shift);
            valid_len += 1;

            if valid_len >= k {
                // `rev` is already `reverse_complement_u64(fwd, k)` here, so
                // this is `canonical_kmer_u64(fwd, k)` without the recompute.
                out.push(fwd.min(rev));
            }
        } else {
            fwd = 0;
            rev = 0;
            valid_len = 0;
        }
    }
}

/// Collapses runs of the same base to a single occurrence, so that an
/// insertion or deletion inside a homopolymer run -- the dominant error mode
/// on long-read platforms such as Oxford Nanopore and PacBio, as opposed to
/// the substitution errors that dominate short-read Illumina data -- does
/// not shift every k-mer downstream of it in the compressed sequence. Two
/// consecutive bytes are the same base when [`base_to_bits`] maps them to
/// the same 2-bit code, so runs are matched case-insensitively and treat
/// `U` as `T`, exactly as k-mer extraction already does.
///
/// An ambiguous byte (`base_to_bits` returns `None`, e.g. `N`) is never
/// merged with its neighbours, in either direction: it is always emitted on
/// its own, so a run of `N`s is **not** collapsed. This keeps homopolymer
/// compression a conservative transform of the sequence alphabet itself,
/// rather than a transform reasoned about in terms of the k-mer window's
/// own reset behaviour (which would happen to make either choice
/// observationally equivalent downstream, but that is not a reason to let
/// this function assume it).
///
/// `out` is cleared first, matching [`extract_canonical_kmers_into`]'s
/// buffer-reuse convention. The output is always the same length as or
/// shorter than `seq`; it is a sequence, not a set of k-mers, so this is a
/// preprocessing step that runs *before* k-mer extraction, not a variant of
/// it.
///
/// This trades exact base-level positional correspondence with the
/// original read for robustness to indel errors: a compressed sequence's
/// byte offsets no longer line up one-to-one with the uncompressed read's,
/// which matters for anything doing reference-coordinate mapping downstream
/// (for example `python/fastdna/annotate.py`'s `locate_kmer`, which assumes
/// uncompressed coordinates). Re-deriving compressed-to-original coordinate
/// mapping is a separate concern this function does not attempt.
pub fn homopolymer_compress_into(seq: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(seq.len());

    let mut prev_bits: Option<u64> = None;
    for &base in seq {
        let bits = base_to_bits(base);
        if bits.is_some() && bits == prev_bits {
            continue;
        }
        out.push(base);
        prev_bits = bits;
    }
}

/// Extracts every canonical k-mer from a DNA sequence into a fresh `Vec`.
///
/// A thin wrapper over [`extract_canonical_kmers_into`]; identical results.
/// Callers on a hot path (once per read or more) should use the `_into`
/// variant with a buffer they own, so the allocation happens once rather
/// than once per call.
pub fn extract_canonical_kmers(seq: &[u8], k: usize) -> Vec<u64> {
    let mut kmers = Vec::new();
    extract_canonical_kmers_into(seq, k, &mut kmers);
    kmers
}

/// Decodes a u64 k-mer and *appends* its `k` ASCII bases to `out`.
///
/// The allocation-free half of [`decode_kmer`]. Every byte written comes from
/// the `b"ACGT"` literal, so the result is ASCII by construction and needs no
/// UTF-8 validation -- which is the whole point: exporters call this once per
/// distinct k-mer (53.8 million on the benchmark file), and the `String`
/// version pays one heap allocation, one matching free and one redundant
/// `String::from_utf8` scan of `k` bytes on every one of them.
pub fn decode_kmer_into(mut kmer: u64, k: usize, out: &mut Vec<u8>) {
    let start = out.len();
    // Same fill-with-'A'-then-overwrite shape as `vec![b'A'; k]` had, minus
    // the allocation: for k > 32 the top slots have no bits left and stay 'A',
    // which is the behaviour `decode_kmer` has always had.
    out.resize(start + k, b'A');
    let dst = &mut out[start..];
    for i in (0..k).rev() {
        dst[i] = b"ACGT"[(kmer & 0b11) as usize];
        kmer >>= 2;
    }
}

/// Decodes a u64 into its ASCII string representation of length k.
pub fn decode_kmer(kmer: u64, k: usize) -> String {
    let mut chars = Vec::with_capacity(k);
    decode_kmer_into(kmer, k, &mut chars);
    // All four possible byte values above come from the b"ACGT" literal,
    // which is valid ASCII/UTF-8 by construction, so this cannot fail.
    String::from_utf8(chars).unwrap_or_default()
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
        assert_eq!(decoded, "ACGT");
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

    /// `base_to_bits` became a 256-byte table lookup. Exhaustively pin it
    /// against the `match` it replaced, so no byte silently changed meaning
    /// -- an accepted byte that should be ambiguous would produce corrupt
    /// k-mers rather than resetting the window.
    #[test]
    fn base_to_bits_table_agrees_with_the_match_it_replaced_on_every_byte() {
        for byte in 0..=u8::MAX {
            let expected = match byte {
                b'A' | b'a' => Some(0b00u64),
                b'C' | b'c' => Some(0b01),
                b'G' | b'g' => Some(0b10),
                b'T' | b't' | b'U' | b'u' => Some(0b11),
                _ => None,
            };
            assert_eq!(base_to_bits(byte), expected, "byte {byte} (0x{byte:02X}) changed meaning");
        }
    }

    #[test]
    fn complement_bits_matches_the_base_to_bits_encoding() {
        // A<->T, C<->G, read straight out of `base_to_bits`.
        for (a, b) in [(b'A', b'T'), (b'C', b'G'), (b'G', b'C'), (b'T', b'A')] {
            let (Some(ab), Some(bb)) = (base_to_bits(a), base_to_bits(b)) else {
                panic!("ACGT must encode");
            };
            assert_eq!(complement_bits(ab), bb, "complement of {} must be {}", a as char, b as char);
        }
    }

    /// Reference window encoder: packs one k-length slice from scratch, with
    /// no rolling at all. Used to pin the rolled reverse complement.
    fn encode_window(window: &[u8]) -> Option<u64> {
        let mut v = 0u64;
        for &b in window {
            v = (v << 2) | base_to_bits(b)?;
        }
        Some(v)
    }

    /// The rolled reverse complement must equal the from-scratch
    /// `reverse_complement_u64` at every window position, for every k the
    /// 2-bit packing allows -- including both edges, k = 1 and k = 32 -- and
    /// across an ambiguous-base window reset.
    #[test]
    fn rolled_reverse_complement_matches_recomputed_at_every_position() {
        let seq: Vec<u8> = b"ACGTTGCAAACCGGTTNGATTACAGATTACAGATTACAGATTACACCGGTTAANCGCGCGATCG".to_vec();

        for k in 1..=32usize {
            let rolled = extract_canonical_kmers(&seq, k);

            let mut expected = Vec::new();
            for window in seq.windows(k) {
                if let Some(fwd) = encode_window(window) {
                    expected.push(fwd.min(reverse_complement_u64(fwd, k)));
                }
            }

            assert_eq!(rolled, expected, "rolled canonical k-mers diverge at k = {k}");
        }
    }

    #[test]
    fn extract_into_clears_and_reuses_the_buffer() {
        let mut buf = vec![0xDEAD_BEEFu64; 9];

        extract_canonical_kmers_into(b"ACGTACGT", 4, &mut buf);
        assert_eq!(buf, extract_canonical_kmers(b"ACGTACGT", 4));

        // A second call must not append to the first call's output, and a
        // call that yields nothing must leave the buffer empty rather than
        // stale.
        extract_canonical_kmers_into(b"ACG", 4, &mut buf);
        assert!(buf.is_empty(), "a too-short read must clear the reused buffer");

        extract_canonical_kmers_into(b"TTTTGGGG", 4, &mut buf);
        assert_eq!(buf, extract_canonical_kmers(b"TTTTGGGG", 4));
    }

    #[test]
    fn homopolymer_compress_collapses_runs_of_the_same_base() {
        let mut out = Vec::new();
        homopolymer_compress_into(b"AAACCGGGGT", &mut out);
        assert_eq!(out, b"ACGT");
    }

    #[test]
    fn homopolymer_compress_matches_runs_case_insensitively() {
        let mut out = Vec::new();
        homopolymer_compress_into(b"AaAaCcGgGgTt", &mut out);
        // Each byte in a matched run is dropped, keeping the first byte's
        // case, exactly as `base_to_bits` treats them as one code.
        assert_eq!(out, b"ACGT");
    }

    #[test]
    fn homopolymer_compress_does_not_collapse_a_run_of_ambiguous_bases() {
        let mut out = Vec::new();
        homopolymer_compress_into(b"AANNNAAA", &mut out);
        // The leading "AA" and trailing "AAA" are each their own homopolymer
        // run and collapse to one 'A' apiece; "NNN" in between must not
        // collapse at all.
        assert_eq!(out, b"ANNNA", "a run of N must be passed through untouched, not collapsed");
    }

    #[test]
    fn homopolymer_compress_treats_a_single_ambiguous_byte_between_runs_as_a_break() {
        let mut out = Vec::new();
        homopolymer_compress_into(b"AAANAAA", &mut out);
        assert_eq!(out, b"ANA");
    }

    #[test]
    fn homopolymer_compress_of_an_empty_sequence_is_empty() {
        let mut out = vec![0xAA];
        homopolymer_compress_into(b"", &mut out);
        assert!(out.is_empty(), "must clear the buffer, not just fail to grow it");
    }

    #[test]
    fn homopolymer_compress_of_a_sequence_with_no_run_is_unchanged() {
        let mut out = Vec::new();
        homopolymer_compress_into(b"ACGTACGT", &mut out);
        assert_eq!(out, b"ACGTACGT");
    }

    /// The falsifiable claim behind `--hpc`: a deletion strictly inside a
    /// homopolymer run changes the raw canonical k-mer set (the error
    /// propagates downstream, corrupting every k-mer that overlaps it), but
    /// after homopolymer compression the two reads become byte-identical,
    /// so their k-mer sets agree exactly. Constructed by hand, not
    /// probabilistically: "GATCAAAAAATCG" (a run of six A's) versus the same
    /// read with one A deleted from inside that run.
    #[test]
    fn a_deletion_inside_a_homopolymer_run_is_absorbed_by_compression() {
        let clean = b"GATCAAAAAATCG"; // run of 6 A's
        let with_deletion = b"GATCAAAAATCG"; // run of 5 A's: one deleted

        let k = 4;
        let raw_clean = extract_canonical_kmers(clean, k);
        let raw_deleted = extract_canonical_kmers(with_deletion, k);
        assert_ne!(
            raw_clean, raw_deleted,
            "without --hpc the indel must still change the k-mer set -- otherwise this test \
             proves nothing"
        );

        let mut compressed_clean = Vec::new();
        let mut compressed_deleted = Vec::new();
        homopolymer_compress_into(clean, &mut compressed_clean);
        homopolymer_compress_into(with_deletion, &mut compressed_deleted);
        assert_eq!(
            compressed_clean, compressed_deleted,
            "a run-internal indel must be fully absorbed by compression"
        );

        let hpc_clean = extract_canonical_kmers(&compressed_clean, k);
        let hpc_deleted = extract_canonical_kmers(&compressed_deleted, k);
        assert_eq!(
            hpc_clean, hpc_deleted,
            "with --hpc the two reads must agree on their k-mer set"
        );
    }

    #[test]
    fn decode_kmer_into_appends_the_same_bases_decode_kmer_returns() {
        let mut buf = b"prefix:".to_vec();
        let kmers = extract_canonical_kmers(b"ACGTTGCA", 4);

        let mut expected = String::from("prefix:");
        for &km in &kmers {
            decode_kmer_into(km, 4, &mut buf);
            expected.push_str(&decode_kmer(km, 4));
        }

        assert_eq!(String::from_utf8(buf).unwrap_or_default(), expected);
    }
}
