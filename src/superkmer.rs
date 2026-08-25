// src/superkmer.rs

//! Super-k-mers: maximal runs of consecutive k-mers that share a signature,
//! and the byte layout they are stored in
//! (`docs/design-minimizer-counting.md` §3.4).
//!
//! Nothing here has a caller in the counting pipeline yet. Like
//! [`crate::minimizer`], this module is a leaf, so the one property the whole
//! design rests on can be proved before anything depends on it.
//!
//! # Why super-k-mers exist
//!
//! The current counting path materialises **one 8-byte word per k-mer
//! occurrence** -- 120 words per 150 bp read, each overlapping its neighbour
//! by `k-1 = 30` bases. A partition-then-sort design has to buffer every
//! occurrence of a bin before it can count that bin, and a bin only closes at
//! end of input, so it must buffer *all* of them: 6.26 GiB as `u64` on the
//! benchmark input. Stored as super-k-mers that is 829 MiB, which fits in
//! memory comfortably. Super-k-mers are what make the partitioned design
//! viable in RAM; that, and not the raw compression ratio, is the point.
//!
//! Runs of consecutive k-mers sharing a bin exist only because the bin
//! function is *stable along a read*, which is exactly the locality property
//! a minimizer has by construction and high-bit bucketing of canonical
//! k-mers does not (consecutive k-mers have essentially unrelated high bits,
//! so every run would have length 1).
//!
//! # The record layout, verbatim from §3.4
//!
//! ```text
//!   offset 0   : u8   n_bases      (k ..= 255; 0 terminates a chunk)
//!   offset 1.. : u8[] packed bases, ceil(n_bases / 4) bytes,
//!                2 bits per base, A=00 C=01 G=10 T=11 -- the same encoding
//!                as kmer::base_to_bits -- first base in the high bits of
//!                the first byte
//!
//!   Record size = 1 + ceil(n_bases / 4) bytes.
//! ```
//!
//! A `u8` length caps a super-k-mer at 255 bases. That is not a limit on
//! read length: a run longer than the cap is *split*, which is always safe --
//! it only cuts a run in two, and every k-mer still reaches the same bin,
//! because both halves carry the same signature -- at a cost of one extra
//! `k-1` overlap per 255 bases (~12% expansion on ONT-length reads). A `u16`
//! length would instead cost one extra byte on each of 77 million records
//! (74 MiB) to save that 12% on a read type FastDNA does not properly
//! support yet.
//!
//! # The invariant this module has to guarantee
//!
//! > Expanding every super-k-mer of a read, canonicalising each k-mer, and
//! > collecting yields **exactly the same multiset** as
//! > `kmer::extract_canonical_kmers(seq, k)`.
//!
//! Break it -- an off-by-one at a super-k-mer boundary, an `N` cut that drops
//! or duplicates a window, a mishandled cap split -- and the output is a
//! well-formed sorted table with the correct schema, a plausible k-mer count,
//! and wrong numbers. Nothing crashes. That is categorically worse than any
//! failure the current code can produce, which is why
//! `expanding_super_kmers_reproduces_the_reference_multiset_exactly` is the
//! single most important test in the project and covers `N`, short reads,
//! all-`N` reads, homopolymers and the 255-base cap in one sweep.

use crate::kmer::{base_to_bits, complement_bits};
use crate::minimizer::{bin_of, SignatureScanner};

/// The largest number of bases one record can hold, set by the `u8` length
/// prefix. FastK caps its own super-mers the same way.
pub const MAX_SUPER_KMER_BASES: usize = u8::MAX as usize;

/// The size in bytes of a record holding `n_bases` bases: the length prefix
/// plus the 2-bit payload, with no padding between records.
#[inline(always)]
pub fn record_len(n_bases: usize) -> usize {
    1 + n_bases.div_ceil(4)
}

/// One super-k-mer's position within the sequence it was cut from, and the
/// signature every k-mer inside it shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperKmer {
    /// Index of the first base, within the sequence passed to
    /// [`split_into_superkmers`].
    pub start: usize,
    /// Number of bases; always in `k..=255`.
    pub len: usize,
    /// The shared signature (see [`crate::minimizer`]). Every k-mer this
    /// super-k-mer expands to has this same signature, and therefore the
    /// same bin.
    pub signature: u64,
}

impl SuperKmer {
    /// One past the last base.
    #[inline(always)]
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    /// How many k-mers this super-k-mer expands to.
    #[inline(always)]
    pub fn kmer_count(&self, k: usize) -> usize {
        self.len.saturating_sub(k) + usize::from(self.len >= k)
    }

    /// The bin every k-mer inside this super-k-mer routes to.
    #[inline(always)]
    pub fn bin(&self, num_bins: usize) -> usize {
        bin_of(self.signature, num_bins)
    }
}

/// Cuts `seq` into super-k-mers, calling `emit(start, bases, signature)` for
/// each one in left-to-right order.
///
/// `bases` is a subslice of `seq` beginning at index `start`, never a copy,
/// and is guaranteed to contain only unambiguous bases, to be at least `k`
/// and at most [`MAX_SUPER_KMER_BASES`] long, and to have every one of its
/// k-mers carry `signature`.
///
/// Three cuts happen, and each one is a place the multiset invariant could
/// break:
///
/// 1. **At every ambiguous base.** `kmer::extract_canonical_kmers` resets its
///    window at anything that is not `ACGT`/`U`, so no reference k-mer ever
///    spans one; this must not produce one either. The scanner is reset, not
///    merely interrupted, so no window straddles the gap.
/// 2. **At every signature change.** This is what makes a run a super-k-mer.
///    Cutting on the signature *value* (rather than on the position that
///    produced it) is deliberate: equal values always mean an equal bin,
///    which is the only thing correctness needs, and it merges the
///    occasional pair of adjacent runs that select the same m-mer at two
///    different positions.
/// 3. **At the 255-base cap.** A split run is still correct -- both halves
///    carry the same signature and reach the same bin -- and costs one
///    repeated `k-1` overlap.
///
/// A degenerate configuration (`k == 0`, `k > 32`, `m == 0`, `m > k`) emits
/// nothing, which matches `extract_canonical_kmers`'s own behaviour of
/// returning an empty vector rather than panicking.
pub fn for_each_superkmer<F>(seq: &[u8], k: usize, m: usize, mut emit: F)
where
    F: FnMut(usize, &[u8], u64),
{
    if k == 0 || k > 32 || m == 0 || m > k {
        return;
    }

    let mut scanner = SignatureScanner::new(k, m);

    // The open run, if any: where it starts in `seq`, the signature it
    // carries, and where its last k-mer starts (so the run ends
    // `last_kmer_start + k`).
    let mut run_start: usize = 0;
    let mut run_signature: u64 = 0;
    let mut last_kmer_start: usize = 0;
    let mut open = false;

    for (i, &base) in seq.iter().enumerate() {
        let Some(bits) = base_to_bits(base) else {
            // Ambiguous base: close whatever run is open *before* the gap,
            // then start over. Nothing carries across.
            if open {
                emit(run_start, &seq[run_start..last_kmer_start + k], run_signature);
                open = false;
            }
            scanner.reset();
            continue;
        };

        let Some(signature) = scanner.push(bits) else {
            continue;
        };

        // The k-mer whose window just closed starts here.
        let kmer_start = i + 1 - k;

        // `i - run_start + 1` is what the open run's length would become if
        // this k-mer joined it, so it fits while `i - run_start` is still
        // below the cap.
        let fits = i - run_start < MAX_SUPER_KMER_BASES;
        if open && run_signature == signature.value && fits {
            last_kmer_start = kmer_start;
            continue;
        }

        if open {
            emit(run_start, &seq[run_start..last_kmer_start + k], run_signature);
        }
        run_start = kmer_start;
        run_signature = signature.value;
        last_kmer_start = kmer_start;
        open = true;
    }

    if open {
        emit(run_start, &seq[run_start..last_kmer_start + k], run_signature);
    }
}

/// Cuts `seq` into super-k-mers and returns them as positions into `seq`.
///
/// The collecting form of [`for_each_superkmer`], for callers that want the
/// cut points rather than the bases -- the tests below, and any future
/// diagnostic that reports per-bin occupancy. The counting path uses the
/// callback form, which allocates nothing.
pub fn split_into_superkmers(seq: &[u8], k: usize, m: usize) -> Vec<SuperKmer> {
    let mut out = Vec::new();
    for_each_superkmer(seq, k, m, |start, bases, signature| {
        out.push(SuperKmer { start, len: bases.len(), signature });
    });
    out
}

/// Writes one super-k-mer record into the front of `dst`, returning how many
/// bytes it used.
///
/// Returns `None` -- rather than writing a partial or wrong record -- when
/// `dst` is too small, when `bases` is empty or longer than
/// [`MAX_SUPER_KMER_BASES`], or when `bases` contains anything that is not an
/// unambiguous nucleotide. The last case cannot arise from
/// [`for_each_superkmer`], whose slices are `ACGT`-only by construction; it
/// is rejected rather than silently encoded as `A` because an ambiguous base
/// reaching here would mean a cut was missed, which is exactly the silent
/// wrong-count failure this module exists to make impossible.
pub fn encode_record_into(bases: &[u8], dst: &mut [u8]) -> Option<usize> {
    let n = bases.len();
    if n == 0 || n > MAX_SUPER_KMER_BASES {
        return None;
    }
    let need = record_len(n);
    let out = dst.get_mut(..need)?;

    out[0] = n as u8;
    // Written, not assumed: the trailing bases of the last byte are padding
    // and must be zero for the layout to be canonical, and `dst` may be a
    // recycled buffer.
    out[1..].fill(0);

    for (i, &base) in bases.iter().enumerate() {
        let bits = base_to_bits(base)? as u8;
        // First base in the high bits of the first byte.
        out[1 + (i >> 2)] |= bits << (6 - 2 * (i & 0b11));
    }
    Some(need)
}

/// A borrowed view of one encoded super-k-mer record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperKmerRecord<'a> {
    n_bases: usize,
    packed: &'a [u8],
}

impl<'a> SuperKmerRecord<'a> {
    /// How many bases this record holds.
    #[inline(always)]
    pub fn n_bases(&self) -> usize {
        self.n_bases
    }

    /// How many k-mers this record expands to.
    #[inline(always)]
    pub fn kmer_count(&self, k: usize) -> usize {
        if self.n_bases < k || k == 0 {
            0
        } else {
            self.n_bases - k + 1
        }
    }

    /// The 2-bit code of base `i`, or `None` past the end.
    #[inline(always)]
    pub fn base_code(&self, i: usize) -> Option<u64> {
        if i >= self.n_bases {
            return None;
        }
        let byte = *self.packed.get(i >> 2)?;
        Some(((byte >> (6 - 2 * (i & 0b11))) & 0b11) as u64)
    }

    /// Appends every canonical k-mer of this record to `out`.
    ///
    /// Appends rather than clearing, because a bin's expansion buffer is
    /// filled from many records in sequence.
    ///
    /// Both strands are rolled side by side, exactly as
    /// `kmer::extract_canonical_kmers_into` does -- see that function for the
    /// derivation. The record contains no ambiguous base by construction, so
    /// unlike that function this loop has no window reset in it at all.
    pub fn expand_canonical_into(&self, k: usize, out: &mut Vec<u64>) {
        if k == 0 || k > 32 || self.n_bases < k {
            return;
        }
        out.reserve(self.n_bases - k + 1);

        let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
        let top_shift = 2 * (k - 1);
        let mut fwd: u64 = 0;
        let mut rev: u64 = 0;

        let mut remaining = self.n_bases;
        let mut index = 0usize;
        for &byte in self.packed {
            // The final byte of a record can hold fewer than four bases; its
            // padding must not be rolled in as `A`.
            let take = remaining.min(4);
            for slot in 0..take {
                let bits = ((byte >> (6 - 2 * slot)) & 0b11) as u64;
                fwd = ((fwd << 2) | bits) & mask;
                rev = (rev >> 2) | (complement_bits(bits) << top_shift);
                index += 1;
                if index >= k {
                    out.push(fwd.min(rev));
                }
            }
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
    }

    /// Appends this record's bases as ASCII `ACGT` to `out`. Used by tests
    /// and by any future diagnostic that needs to look at a bin's contents.
    pub fn decode_bases_into(&self, out: &mut Vec<u8>) {
        for i in 0..self.n_bases {
            let code = self.base_code(i).unwrap_or(0);
            out.push(b"ACGT"[code as usize]);
        }
    }
}

/// Iterates the records packed back-to-back in a chunk.
///
/// Stops at the end of `bytes` **or** at a zero length prefix, whichever
/// comes first: a chunk is zero-initialised and never written with a
/// zero-length record, so a zero byte marks the end of the written records
/// even for a caller that does not track the fill level separately. A
/// truncated final record (a length prefix whose payload runs past the end of
/// `bytes`) also ends the iteration rather than yielding a short read of it.
#[derive(Debug, Clone)]
pub struct RecordIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// Iterates the super-k-mer records encoded in `bytes`.
pub fn records(bytes: &[u8]) -> RecordIter<'_> {
    RecordIter { bytes, pos: 0 }
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = SuperKmerRecord<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_bases = *self.bytes.get(self.pos)? as usize;
        if n_bases == 0 {
            return None;
        }
        let payload_start = self.pos + 1;
        let payload_end = payload_start + n_bases.div_ceil(4);
        let packed = self.bytes.get(payload_start..payload_end)?;
        self.pos = payload_end;
        Some(SuperKmerRecord { n_bases, packed })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::kmer::extract_canonical_kmers;
    use crate::minimizer::{signature_of_kmer, DEFAULT_M, DEFAULT_NUM_BINS, INELIGIBLE_SIGNATURE};

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn random_sequence(rng: &mut Xorshift, len: usize) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..len).map(|_| bases[(rng.next() % 4) as usize]).collect()
    }

    /// Encodes every super-k-mer of `seq` through the real record layout,
    /// expands each one back, and returns the canonical k-mers in the order
    /// the bins would see them.
    fn round_trip_kmers(seq: &[u8], k: usize, m: usize) -> Vec<u64> {
        let mut out = Vec::new();
        let mut buffer = [0u8; 1 + MAX_SUPER_KMER_BASES / 4 + 1];
        for_each_superkmer(seq, k, m, |_start, bases, _signature| {
            let used = encode_record_into(bases, &mut buffer).expect("a super-k-mer must encode");
            let mut iter = records(&buffer[..used]);
            let record = iter.next().expect("one record");
            assert!(iter.next().is_none(), "a single super-k-mer must encode to a single record");
            record.expand_canonical_into(k, &mut out);
        });
        out
    }

    fn sorted(mut v: Vec<u64>) -> Vec<u64> {
        v.sort_unstable();
        v
    }

    // ---------------------------------------------------------------
    // The single most important test in the project.
    // ---------------------------------------------------------------

    /// §5 step 2, verbatim: expanding every super-k-mer of a read,
    /// canonicalising each k-mer, and collecting must yield **exactly the
    /// same multiset** as `kmer::extract_canonical_kmers(seq, k)` --
    /// including for reads containing `N`, reads shorter than `k`, all-`N`
    /// reads, pure-homopolymer reads, and reads at the 255-base super-k-mer
    /// cap.
    ///
    /// A multiset, not a set: a k-mer occurring twice in one read must be
    /// expanded twice, or every count in the output is quietly low. Sorting
    /// both sides and comparing is exactly a multiset comparison.
    #[test]
    fn expanding_super_kmers_reproduces_the_reference_multiset_exactly() {
        let mut rng = Xorshift(0x5EED_0000_1234_5678);
        let mut cases: Vec<(String, Vec<u8>)> = Vec::new();

        // Ordinary reads.
        for len in [31usize, 32, 40, 75, 100, 150, 151, 300] {
            cases.push((format!("random {len}bp"), random_sequence(&mut rng, len)));
        }

        // Reads shorter than k, and exactly k.
        for len in [0usize, 1, 5, 30, 31] {
            cases.push((format!("short {len}bp"), random_sequence(&mut rng, len)));
        }

        // Ambiguous bases: leading, trailing, interior, adjacent, and a run
        // long enough that no window can bridge it.
        let mut with_n = random_sequence(&mut rng, 150);
        with_n[0] = b'N';
        with_n[40] = b'N';
        with_n[41] = b'N';
        with_n[100] = b'N';
        with_n[149] = b'N';
        cases.push(("scattered N".into(), with_n));

        let mut n_run = random_sequence(&mut rng, 200);
        for slot in n_run.iter_mut().skip(60).take(50) {
            *slot = b'N';
        }
        cases.push(("50-base N run".into(), n_run));

        // Every base ambiguous.
        cases.push(("all N".into(), vec![b'N'; 150]));
        cases.push(("all N, short".into(), vec![b'N'; 10]));

        // Other bytes `base_to_bits` rejects, which must be cut at exactly
        // like `N` is.
        let mut mixed = random_sequence(&mut rng, 120);
        mixed[50] = b'n';
        mixed[70] = b'-';
        mixed[90] = b'R';
        cases.push(("lowercase n, gap, IUPAC R".into(), mixed));

        // Lowercase ACGT are *not* ambiguous and must be counted, so a
        // soft-masked read must not be cut at all.
        let mut soft = random_sequence(&mut rng, 120);
        for slot in soft.iter_mut().skip(30).take(40) {
            *slot = slot.to_ascii_lowercase();
        }
        cases.push(("soft-masked".into(), soft));

        // Pure homopolymers: every window falls back, so the whole read is
        // one run, and the cap has to split it.
        for base in [b'A', b'C', b'G', b'T'] {
            for len in [31usize, 100, 255, 256, 600] {
                cases.push((format!("poly-{} {len}bp", base as char), vec![base; len]));
            }
        }

        // A long constant-signature stretch built from real bases: a short
        // motif repeated, which keeps selecting the same m-mer.
        let motif = b"ACGTACGTAC";
        let repeated: Vec<u8> = motif.iter().cycle().copied().take(900).collect();
        cases.push(("repeated 10-mer, 900bp".into(), repeated));

        // Reads sitting exactly on the cap boundary.
        for len in [MAX_SUPER_KMER_BASES - 1, MAX_SUPER_KMER_BASES, MAX_SUPER_KMER_BASES + 1] {
            cases.push((format!("random at cap {len}bp"), random_sequence(&mut rng, len)));
        }

        // Plus a large sweep of random reads, so this is not just a handful
        // of hand-picked shapes.
        for i in 0..400 {
            let len = 20 + (rng.next() as usize % 340);
            let mut seq = random_sequence(&mut rng, len);
            // Roughly one read in three carries an ambiguous base.
            if rng.next().is_multiple_of(3) && len > 0 {
                let pos = rng.next() as usize % len;
                seq[pos] = b'N';
            }
            cases.push((format!("sweep {i} ({len}bp)"), seq));
        }

        let mut checked = 0usize;
        for (k, m) in [(31usize, DEFAULT_M), (31, 9), (21, 7), (15, 5), (8, 3), (32, 7), (5, 5), (1, 1)] {
            for (label, seq) in &cases {
                let expected = sorted(extract_canonical_kmers(seq, k));
                let actual = sorted(round_trip_kmers(seq, k, m));
                assert_eq!(
                    actual, expected,
                    "super-k-mer expansion diverges from the reference at k={k} m={m} on '{label}'"
                );
                checked += 1;
            }
        }
        println!("multiset invariant checked on {checked} (read, k, m) combinations");
    }

    // ---------------------------------------------------------------
    // The second property: one super-k-mer, one bin.
    // ---------------------------------------------------------------

    /// Every k-mer expanded from one super-k-mer must have the same
    /// `bin_of` -- checked against each k-mer's *own* signature, computed
    /// independently by `signature_of_kmer`, not against the run's recorded
    /// one. If this and the multiset property both hold, the design cannot
    /// produce a wrong count.
    #[test]
    fn every_kmer_of_a_super_kmer_shares_its_bin() {
        let mut rng = Xorshift(0xC0FF_EE00_D15E_A5E5);
        let mut sequences: Vec<Vec<u8>> = (0..120)
            .map(|_| {
                let len = 40 + (rng.next() as usize % 300);
                random_sequence(&mut rng, len)
            })
            .collect();
        sequences.push(vec![b'A'; 400]);
        sequences.push(b"ACGTACGTAC".iter().cycle().copied().take(500).collect());
        let mut with_n = random_sequence(&mut rng, 200);
        with_n[77] = b'N';
        sequences.push(with_n);

        for (k, m) in [(31usize, DEFAULT_M), (21, 7), (15, 5)] {
            for seq in &sequences {
                for sk in split_into_superkmers(seq, k, m) {
                    assert!(sk.len >= k, "a super-k-mer must hold at least one k-mer");
                    assert!(sk.len <= MAX_SUPER_KMER_BASES, "the cap must be respected");

                    let bases = &seq[sk.start..sk.end()];
                    let expected_bin = sk.bin(DEFAULT_NUM_BINS);
                    for window in bases.windows(k) {
                        let mut packed = 0u64;
                        for &b in window {
                            packed = (packed << 2) | base_to_bits(b).expect("no N inside a super-k-mer");
                        }
                        let own = signature_of_kmer(packed, k, m);
                        assert_eq!(
                            own, sk.signature,
                            "a k-mer's own signature differs from its super-k-mer's at k={k} m={m}"
                        );
                        assert_eq!(bin_of(own, DEFAULT_NUM_BINS), expected_bin);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // The cap, and the fallback bin.
    // ---------------------------------------------------------------

    /// The cap must split a long run rather than drop or truncate it, and it
    /// must cost exactly one repeated `k-1` overlap per split -- no more.
    #[test]
    fn the_255_base_cap_splits_a_long_run_without_losing_a_kmer() {
        let k = 31usize;
        let m = DEFAULT_M;
        let seq = vec![b'A'; 1_000];

        let parts = split_into_superkmers(&seq, k, m);
        assert!(parts.len() > 1, "a 1000-base homopolymer must be split by the cap");

        let total_kmers: usize = parts.iter().map(|p| p.kmer_count(k)).sum();
        assert_eq!(
            total_kmers,
            seq.len() - k + 1,
            "the split must preserve the k-mer count exactly, not merely approximately"
        );

        let mut expected_start = 0usize;
        for (i, part) in parts.iter().enumerate() {
            assert!(part.len <= MAX_SUPER_KMER_BASES, "part {i} exceeds the cap");
            assert_eq!(part.start, expected_start, "part {i} does not resume where the previous one's last k-mer began");
            // Every part except the last must be filled to the cap: cutting
            // early would mean more overlap bytes than the design budgets.
            if i + 1 < parts.len() {
                assert_eq!(part.len, MAX_SUPER_KMER_BASES, "part {i} was cut before the cap");
            }
            // The next part starts at this one's last k-mer, so exactly
            // `k - 1` bases are repeated.
            expected_start = part.end() - (k - 1);
        }

        // Stated as bytes, which is what the design budgets: the overlap
        // costs `k-1` extra stored bases per split.
        let stored: usize = parts.iter().map(|p| p.len).sum();
        assert_eq!(stored, seq.len() + (parts.len() - 1) * (k - 1));
    }

    #[test]
    fn an_all_ineligible_run_carries_the_fallback_signature_and_bin_zero() {
        let k = 31usize;
        let seq = vec![b'A'; 300];
        let parts = split_into_superkmers(&seq, k, DEFAULT_M);
        assert!(!parts.is_empty());
        for part in &parts {
            assert_eq!(part.signature, INELIGIBLE_SIGNATURE, "an all-A run has no eligible m-mer");
            assert_eq!(part.bin(DEFAULT_NUM_BINS), 0, "the fallback must route to bin 0");
        }
    }

    /// A read with no k-mer at all must produce no super-k-mer, so a bin
    /// never receives an empty or sub-k record.
    #[test]
    fn reads_with_no_kmers_produce_no_super_kmers() {
        let k = 31usize;
        for seq in [
            Vec::new(),
            b"ACGT".to_vec(),
            vec![b'N'; 100],
            b"ACGTNNACGT".to_vec(),
        ] {
            assert!(split_into_superkmers(&seq, k, DEFAULT_M).is_empty(), "unexpected super-k-mer");
        }
        // Degenerate configurations must be inert, not panicking.
        assert!(split_into_superkmers(b"ACGTACGTACGT", 0, 3).is_empty());
        assert!(split_into_superkmers(b"ACGTACGTACGT", 33, 7).is_empty());
        assert!(split_into_superkmers(b"ACGTACGTACGT", 5, 0).is_empty());
        assert!(split_into_superkmers(b"ACGTACGTACGT", 5, 9).is_empty());
    }

    /// Super-k-mers must tile the read's k-mer positions without a gap or an
    /// overlap in *k-mers* (they do overlap in bases, by `k-1`, which is the
    /// whole point). Checked positionally, independently of the expansion.
    #[test]
    fn super_kmers_tile_the_kmer_positions_of_a_read() {
        let mut rng = Xorshift(0x0BAD_F00D_0BAD_F00D);
        let k = 31usize;
        for _ in 0..200 {
            let len = 31 + (rng.next() as usize % 400);
            let mut seq = random_sequence(&mut rng, len);
            if rng.next().is_multiple_of(2) {
                let pos = rng.next() as usize % len;
                seq[pos] = b'N';
            }

            // The reference: every k-mer start whose window is N-free.
            let mut expected: Vec<usize> = Vec::new();
            for start in 0..=len - k {
                if seq[start..start + k].iter().all(|&b| base_to_bits(b).is_some()) {
                    expected.push(start);
                }
            }

            let mut actual: Vec<usize> = Vec::new();
            for part in split_into_superkmers(&seq, k, DEFAULT_M) {
                for offset in 0..part.kmer_count(k) {
                    actual.push(part.start + offset);
                }
            }

            assert_eq!(actual, expected, "super-k-mers do not tile the read's k-mer positions");
        }
    }

    // ---------------------------------------------------------------
    // The byte layout itself, pinned literally.
    // ---------------------------------------------------------------

    /// The §3.4 layout, byte for byte, on a value chosen so every field is
    /// visible: a length prefix, `A=00 C=01 G=10 T=11`, first base in the
    /// high bits, and zero padding in the final partial byte. This is the
    /// contract a bin's chunk carries; if it ever changes silently, every
    /// stored super-k-mer decodes to different bases.
    #[test]
    fn the_record_layout_is_exactly_the_bytes_the_design_specifies() {
        let mut buf = [0xFFu8; 8];
        // ACGT -> 00 01 10 11 -> 0b00011011 = 0x1B
        let used = encode_record_into(b"ACGT", &mut buf).unwrap();
        assert_eq!(used, 2, "1 length byte + ceil(4/4) payload bytes");
        assert_eq!(&buf[..2], &[4u8, 0b0001_1011]);

        // Five bases: the fifth lands in the high bits of a second byte and
        // the remaining three slots are zero padding.
        let mut buf = [0xFFu8; 8];
        let used = encode_record_into(b"ACGTG", &mut buf).unwrap();
        assert_eq!(used, 3);
        assert_eq!(&buf[..3], &[5u8, 0b0001_1011, 0b1000_0000]);

        // Lowercase and `U` share their uppercase/`T` codes, as
        // `base_to_bits` defines.
        let mut lower = [0u8; 8];
        let mut upper = [0u8; 8];
        encode_record_into(b"acgu", &mut lower).unwrap();
        encode_record_into(b"ACGT", &mut upper).unwrap();
        assert_eq!(lower, upper);

        // The formula, at every residue class.
        assert_eq!(record_len(1), 2);
        assert_eq!(record_len(4), 2);
        assert_eq!(record_len(5), 3);
        assert_eq!(record_len(31), 9);
        assert_eq!(record_len(255), 65);
    }

    #[test]
    fn encoding_refuses_what_it_cannot_represent() {
        let mut buf = [0u8; 128];
        assert_eq!(encode_record_into(b"", &mut buf), None, "an empty record is not representable");
        assert_eq!(
            encode_record_into(&vec![b'A'; MAX_SUPER_KMER_BASES + 1], &mut buf),
            None,
            "the u8 length prefix cannot hold 256"
        );
        assert_eq!(encode_record_into(b"ACGTN", &mut buf), None, "an ambiguous base means a cut was missed");
        let mut tiny = [0u8; 2];
        assert_eq!(encode_record_into(b"ACGTG", &mut tiny), None, "a record that does not fit must not be truncated");
        // A record that exactly fills its buffer is fine.
        let mut exact = [0u8; 3];
        assert_eq!(encode_record_into(b"ACGTG", &mut exact), Some(3));
    }

    #[test]
    fn records_round_trip_through_a_shared_chunk() {
        let mut rng = Xorshift(0x1357_9BDF_2468_ACE0);
        let inputs: Vec<Vec<u8>> = (0..64)
            .map(|_| {
                let len = 1 + (rng.next() as usize % MAX_SUPER_KMER_BASES);
                random_sequence(&mut rng, len)
            })
            .collect();

        // A zero-initialised chunk, exactly as `binned.rs` will use.
        let mut chunk = vec![0u8; 64 * (1 + MAX_SUPER_KMER_BASES.div_ceil(4))];
        let mut written = 0usize;
        for seq in &inputs {
            let used = encode_record_into(seq, &mut chunk[written..]).unwrap();
            assert_eq!(used, record_len(seq.len()));
            written += used;
        }

        // Read back with the fill level, and again without it: the trailing
        // zero must terminate the iteration on its own.
        for slice in [&chunk[..written], &chunk[..]] {
            let decoded: Vec<Vec<u8>> = records(slice)
                .map(|r| {
                    let mut bases = Vec::new();
                    r.decode_bases_into(&mut bases);
                    assert_eq!(r.n_bases(), bases.len());
                    bases
                })
                .collect();
            assert_eq!(decoded, *inputs);
        }
    }

    /// A truncated chunk must end the iteration, not yield a record whose
    /// payload runs off the end (which would decode neighbouring bytes as
    /// bases).
    #[test]
    fn a_truncated_record_ends_the_iteration_instead_of_being_read_short() {
        let mut chunk = [0u8; 16];
        let used = encode_record_into(b"ACGTACGTACGT", &mut chunk).unwrap();
        assert_eq!(records(&chunk[..used]).count(), 1);
        for cut in 1..used {
            assert_eq!(records(&chunk[..cut]).count(), 0, "a record cut at {cut} must not be yielded");
        }
    }

    #[test]
    fn expansion_matches_the_reference_extractor_on_a_single_record() {
        let mut rng = Xorshift(0x2222_4444_6666_8888);
        let mut buf = vec![0u8; record_len(MAX_SUPER_KMER_BASES)];
        for _ in 0..300 {
            let len = 1 + (rng.next() as usize % MAX_SUPER_KMER_BASES);
            let seq = random_sequence(&mut rng, len);
            let used = encode_record_into(&seq, &mut buf).unwrap();
            let record = records(&buf[..used]).next().unwrap();
            for k in [1usize, 4, 15, 31, 32] {
                let mut got = Vec::new();
                record.expand_canonical_into(k, &mut got);
                assert_eq!(got, extract_canonical_kmers(&seq, k), "k={k} len={len}");
                assert_eq!(got.len(), record.kmer_count(k));
            }
        }
    }

    /// The expansion buffer is shared across a whole bin, so
    /// `expand_canonical_into` must append rather than clear.
    #[test]
    fn expansion_appends_to_a_shared_buffer() {
        let mut buf = [0u8; 32];
        let used = encode_record_into(b"ACGTACGTACGT", &mut buf).unwrap();
        let record = records(&buf[..used]).next().unwrap();

        let mut out = vec![0xDEAD_BEEFu64];
        record.expand_canonical_into(4, &mut out);
        record.expand_canonical_into(4, &mut out);

        let one = extract_canonical_kmers(b"ACGTACGTACGT", 4);
        let mut expected = vec![0xDEAD_BEEFu64];
        expected.extend_from_slice(&one);
        expected.extend_from_slice(&one);
        assert_eq!(out, expected);
    }

    /// The compression the whole design is built on, measured rather than
    /// asserted from theory: bytes stored per k-mer occurrence, against the
    /// 8 bytes the current path spends, on reads shaped like the benchmark
    /// input (150 bp, k=31, m=7).
    #[test]
    fn stored_bytes_per_occurrence_are_near_the_designs_prediction() {
        let mut rng = Xorshift(0x9001_9001_9001_9001);
        let (k, m) = (31usize, DEFAULT_M);

        let mut records_written = 0usize;
        let mut bytes = 0usize;
        let mut occurrences = 0usize;
        for _ in 0..20_000 {
            let seq = random_sequence(&mut rng, 150);
            occurrences += extract_canonical_kmers(&seq, k).len();
            for_each_superkmer(&seq, k, m, |_start, bases, _| {
                records_written += 1;
                bytes += record_len(bases.len());
            });
        }

        let per_read = records_written as f64 / 20_000.0;
        let per_occurrence = bytes as f64 / occurrences as f64;
        println!(
            "150bp reads, k=31 m=7: {per_read:.2} super-k-mers/read, \
             {per_occurrence:.3} bytes/occurrence ({:.2}x smaller than 8 B)",
            8.0 / per_occurrence
        );

        // §3.4 predicts 11.08 super-k-mers per read and 1.035 bytes per
        // occurrence. A band, not an equality: the realised density carries
        // Roberts' "a few percent above" caveat and the canonical-m-mer
        // perturbation, both measured in `minimizer.rs`.
        assert!((9.5..=12.5).contains(&per_read), "super-k-mers per read: {per_read:.2}");
        assert!(per_occurrence < 1.15, "bytes per occurrence: {per_occurrence:.3}");
    }
}
