// src/minimizer.rs

//! Canonical minimizers: the bin function of the super-k-mer counting
//! strategy described in `docs/design-minimizer-counting.md` §3.3.
//!
//! Nothing in this module has a caller in the counting pipeline yet; it is
//! deliberately a leaf of pure functions plus two small rolling state
//! machines, so its invariants can be proven by test before anything is
//! wired to them.
//!
//! # The one invariant everything rests on
//!
//! `KmerCounter` counts *canonical* k-mers -- `min(x, rc(x))`
//! (`kmer::canonical_kmer_u64`). A bin function used to partition counting
//! work must therefore satisfy
//!
//! ```text
//! bin(x) == bin(rc(x))   for every k-mer x
//! ```
//!
//! or the two strand orientations of one physical k-mer land in different
//! bins, each bin counts its own half of the occurrences, and the exported
//! table contains the same canonical k-mer with a split count. That failure
//! is silent: the output is still a well-formed, sorted table with a
//! plausible k-mer count and wrong numbers.
//!
//! The signature is defined as
//!
//! ```text
//! sig(x) = min over the k-m+1 positions of  h(canon_m(mmer_i))
//! ```
//!
//! where `canon_m(y) = min(y, rc_m(y))` on m-mers and `h` is a fixed 64-bit
//! mixer. **Claim: `sig(x) == sig(rc(x))` for every k-mer `x`.** The multiset
//! of m-mers of `rc(x)` is exactly the multiset of reverse complements of the
//! m-mers of `x` (reversing the k-mer reverses the order of its m-mer
//! positions and complements each one), and `canon_m` maps a value and its
//! reverse complement to the same representative. The two multisets of
//! canonical m-mers are therefore identical, and so are their minima under
//! any fixed `h` -- and so is the eligibility filter below, which is also a
//! function of the canonical m-mer alone. This is KMC's per-m-mer
//! canonicalisation, not Roberts' window canonicalisation: the latter is
//! equally strand-invariant but requires evaluating a second,
//! reverse-complemented window for the same guarantee.
//!
//! `signature_is_invariant_under_reverse_complement` below is the executable
//! form of that proof.
//!
//! # Why a hash order and not a lexicographic one
//!
//! Roberts et al. 2004 (§2.4) and Marcais et al. 2017 (§2.4) both report
//! that lexicographic order on DNA is pathological on poly-A: runs of `A`
//! make many consecutive windows select a minimizer, pushing realised
//! density *above* the `2/(w+1)` a random order achieves. Zheng et al. 2020
//! shows a random (hash) order hits `2/(w+1)` essentially exactly at any `k`
//! a counter would use. Kraken2 reached the same conclusion independently
//! with its XOR shuffle. The mixer is applied to the *canonical* m-mer, or
//! the strand invariance above is lost.
//!
//! # The eligibility rule, and what it deliberately is not
//!
//! KMC2's published signature rule excludes canonical minimizers that start
//! with `AAA`, start with `ACA`, or contain `AA` anywhere but at the
//! beginning. It buys two distinct things: bin balance, and ~10-15% fewer
//! super-k-mers. The second is a *lexicographic-order artefact* -- KMC's own
//! worked example (`AAAAAAC` -> `AAAAACX` -> `AAAACXY`) is a cascade that
//! happens only because an A-rich prefix makes the successor m-mer also
//! likely minimal under lexicographic order. Under a random hash order that
//! cascade does not exist, so most of that benefit is not available to us
//! and adopting the full rule would not reproduce it.
//!
//! Bin balance survives any ordering, though: poly-A tracts are genuinely
//! over-represented in real reads, so whichever bin holds `A^m` still
//! receives far more than `1/4^m` of the data. This module therefore adopts
//! a narrow rule aimed only at that:
//!
//! > A canonical m-mer is **ineligible** as a signature if it is a
//! > homopolymer (all A, all C, all G or all T), or if it begins with `AAA`.
//!
//! Applied to the *canonical* m-mer the rule is automatically symmetric
//! across strands, so it cannot reintroduce the asymmetry the canonical
//! m-mer just eliminated. KMC needs explicit reverse-complement duals in its
//! rule for exactly this reason; testing the canonical form gets them free.
//!
//! When every m-mer of a window is ineligible -- a pure poly-A read, which
//! real data contains -- the signature is [`INELIGIBLE_SIGNATURE`] and the
//! window routes to a dedicated overflow bin 0. That bin can be large in
//! *occurrences* but is tiny in *distinct* k-mers, so it costs sort time,
//! not memory.


use crate::kmer::{canonical_kmer_u64, complement_bits};

/// Default m-mer length. Odd on purpose: an odd-length m-mer cannot equal
/// its own reverse complement, which removes the palindrome case from the
/// canonicalisation and keeps the induced hash distribution as close to
/// uniform as canonicalisation allows (Marcais et al., "k-nonical space",
/// 2024).
///
/// KMC3 raised its own default from 7 to 9 for finer-grained bin
/// assignment (4^9 signatures to distribute over 512 bins rather than 4^7).
/// We derive the bin from a hash rather than from a frequency-balanced
/// signature map, so that finer granularity buys us less than it buys KMC,
/// while the smaller `m` gives a wider window and therefore lower density:
/// `m = 7` gives `2/26 = 0.0769` against `m = 9`'s `2/24 = 0.0833`, an 8%
/// difference in super-k-mer count.
pub const DEFAULT_M: usize = 7;

/// Default number of bins. A power of two, the same default as KMC.
///
/// At 512 bins a benchmark-scale bin holds ~1.64M occurrences, so its whole
/// expansion buffer is 12.5 MiB -- L3-resident on the machines FastDNA runs
/// on, which makes the per-bin sort cache-local -- while keeping
/// `threads * bins * chunk_bytes` at a defensible 64 MiB for 8 threads.
/// This is a starting point sized the same way `DEFAULT_BUCKET_BITS` and
/// `RAW_FINALIZE_THRESHOLD` were, not a value proven by a sweep, which is
/// why every function here takes it as a parameter.
pub const DEFAULT_NUM_BINS: usize = 512;

/// The signature of a window in which no m-mer is eligible.
///
/// Zero is not an arbitrary sentinel that might collide with a real
/// signature: [`mix64`] is a bijection on `u64`, so `mix64(y) == 0` only for
/// `y == 0`, and `y == 0` is the all-`A` m-mer, which the homopolymer clause
/// of [`is_eligible`] always rejects. A signature of zero therefore means
/// "no eligible m-mer in this window" and nothing else --
/// `zero_signature_means_exactly_the_ineligible_fallback` pins it.
pub const INELIGIBLE_SIGNATURE: u64 = 0;

/// The fixed 64-bit mixer that defines the minimizer order.
///
/// This is the `fmix64` finalizer of MurmurHash3 (public domain), the same
/// avalanche function `splitmix64` uses. Two properties matter here and
/// neither is incidental:
///
/// * **It is a bijection.** Each step -- an xor-shift, a multiply by an odd
///   constant -- is invertible on `u64`, so distinct canonical m-mers always
///   receive distinct order keys. Ties in a window are then genuine repeats
///   of one m-mer, never hash collisions, which is what makes a run of equal
///   signatures a well-defined super-k-mer rather than an accident.
/// * **It avalanches from a small input.** An m-mer is only `2m` bits (14
///   for `m = 7`), so the first `x ^= x >> 33` is the identity on it; the
///   multiply is what spreads those low bits across the word, and the two
///   later shift-xors fold the high half back down. That is why
///   [`bin_of`] can take its bits from the *high* half and still see a
///   well-mixed value.
#[inline(always)]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// The 2-bit mask for an `m`-base m-mer.
#[inline(always)]
fn mmer_mask(m: usize) -> u64 {
    if m >= 32 {
        u64::MAX
    } else {
        (1u64 << (2 * m)) - 1
    }
}

/// The canonical form of a 2-bit packed m-mer: `min(y, rc(y))`.
///
/// Delegates to [`canonical_kmer_u64`] rather than reimplementing the bit
/// tricks, so an m-mer and a k-mer can never disagree about what "canonical"
/// means.
#[inline(always)]
pub fn canonical_mmer(mmer: u64, m: usize) -> u64 {
    canonical_kmer_u64(mmer, m)
}

/// Whether a **canonical** m-mer may serve as a signature.
///
/// Ineligible when the m-mer is a homopolymer (all four are rejected, not
/// just poly-A: poly-C/G/T tracts are over-represented in real reads for the
/// same reasons), or when it begins with `AAA`. See the module documentation
/// for why this is narrower than KMC's published rule and what that costs.
///
/// The caller is responsible for passing a canonical value; passing a raw
/// forward m-mer would break the strand invariance this rule is written to
/// preserve. That is a precondition, not a check, because this sits on the
/// per-base hot path.
#[inline(always)]
pub fn is_eligible(canonical: u64, m: usize) -> bool {
    debug_assert!((1..=31).contains(&m), "is_eligible: m must be in 1..=31, got {m}");

    // A homopolymer has the same 2-bit code in every one of its `m` slots.
    // `mask / 3` is `0b01_01_..._01` (m ones in the low bit of each slot),
    // so multiplying it by the m-mer's own last code reconstructs the
    // homopolymer that shares that code, and equality decides it in two
    // arithmetic operations rather than an m-step loop.
    let repunit = mmer_mask(m) / 0b11;
    if canonical == (canonical & 0b11) * repunit {
        return false;
    }

    // "Begins with AAA": the first base of a packed m-mer sits in the *high*
    // bits (the encoder is `v = (v << 2) | bits`), so the first three bases
    // are the top six bits. Vacuous for m < 3, where the homopolymer clause
    // above has already rejected the only all-A value.
    if m >= 3 && (canonical >> (2 * m - 6)) == 0 {
        return false;
    }

    true
}

/// The order key of one **canonical** m-mer: `Some(mix64(y))` when `y` is
/// eligible, `None` otherwise. `None` means "cannot be a signature", not
/// "compares greater than everything" -- an ineligible m-mer is removed from
/// the candidate set entirely.
#[inline(always)]
pub fn mmer_order_key(canonical: u64, m: usize) -> Option<u64> {
    if is_eligible(canonical, m) {
        Some(mix64(canonical))
    } else {
        None
    }
}

/// The bin a signature routes to.
///
/// Takes its bits from the high half of the mixer output, which is
/// independent of the low bits a caller might use for anything else, and
/// masks rather than divides -- `num_bins` is a power of two. The result is
/// always in `0..num_bins` for any `num_bins >= 1` (`x & (n-1) <= n-1`), so
/// the map is total even if a caller ignores the power-of-two convention;
/// it merely stops being uniform.
///
/// [`INELIGIBLE_SIGNATURE`] maps to bin 0, the dedicated overflow bin.
#[inline(always)]
pub fn bin_of(signature: u64, num_bins: usize) -> usize {
    debug_assert!(num_bins.is_power_of_two(), "bin_of: num_bins must be a power of two, got {num_bins}");
    ((signature >> 32) as usize) & (num_bins - 1)
}

/// The signature of one 2-bit packed k-mer, computed from scratch.
///
/// The reference definition: the minimum order key over every one of the
/// `k - m + 1` m-mer positions of `kmer`, or [`INELIGIBLE_SIGNATURE`] if no
/// position is eligible. Quadratic in the window compared with
/// [`SignatureScanner`], which is what the counting path uses; this exists so
/// the rolling scanner has something independent to be checked against, and
/// so the strand-invariance property can be stated over a single k-mer.
///
/// Returns [`INELIGIBLE_SIGNATURE`] for a degenerate configuration
/// (`m > k`, `k > 32`, `m == 0`) rather than panicking: there is no window
/// to take a minimum over, which is the same situation as no eligible m-mer.
pub fn signature_of_kmer(kmer: u64, k: usize, m: usize) -> u64 {
    if m == 0 || m > k || k > 32 {
        return INELIGIBLE_SIGNATURE;
    }

    let mask = mmer_mask(m);
    let mut best: Option<u64> = None;
    for i in 0..=(k - m) {
        // m-mer `i` counts bases from the left, and base 0 of a packed k-mer
        // occupies the highest 2-bit slot, so m-mer `i` ends `k - i - m`
        // bases above the bottom of the word.
        let mmer = (kmer >> (2 * (k - i - m))) & mask;
        if let Some(key) = mmer_order_key(canonical_mmer(mmer, m), m) {
            best = Some(match best {
                Some(b) if b <= key => b,
                _ => key,
            });
        }
    }
    best.unwrap_or(INELIGIBLE_SIGNATURE)
}

/// Rolls the canonical value of an m-base window one base at a time.
///
/// Both strands are rolled side by side, exactly as
/// `kmer::extract_canonical_kmers_into` does for k-mers, rather than
/// recomputing a full 13-operation reverse complement per position. See that
/// function's doc comment for the derivation; the only difference here is
/// the window width.
#[derive(Debug, Clone)]
pub struct MmerRoller {
    fwd: u64,
    rev: u64,
    mask: u64,
    top_shift: u32,
    m: usize,
    filled: usize,
}

impl MmerRoller {
    pub fn new(m: usize) -> Self {
        debug_assert!((1..=31).contains(&m), "MmerRoller: m must be in 1..=31, got {m}");
        Self {
            fwd: 0,
            rev: 0,
            mask: mmer_mask(m),
            top_shift: (2 * (m.max(1) - 1)) as u32,
            m,
            filled: 0,
        }
    }

    /// Drops every base rolled in so far. Used at an ambiguous base, and
    /// between reads.
    pub fn reset(&mut self) {
        self.fwd = 0;
        self.rev = 0;
        self.filled = 0;
    }

    /// Rolls one 2-bit base code in. Returns the canonical value of the
    /// m-mer ending at this base, or `None` while fewer than `m` bases have
    /// been rolled since the last [`reset`](Self::reset).
    #[inline(always)]
    pub fn push(&mut self, bits: u64) -> Option<u64> {
        self.fwd = ((self.fwd << 2) | bits) & self.mask;
        self.rev = (self.rev >> 2) | (complement_bits(bits) << self.top_shift);
        if self.filled < self.m {
            self.filled += 1;
        }
        if self.filled == self.m {
            Some(self.fwd.min(self.rev))
        } else {
            None
        }
    }
}

/// A monotonic deque holding the sliding-window minimum of a stream of
/// `(key, position)` pairs.
///
/// The deque holds keys in strictly increasing order: pushing a key pops
/// every strictly greater key off the back first, because a greater key that
/// arrived earlier can never be the minimum of any window that also contains
/// the new one. Amortised O(1) per push (each entry is pushed once and
/// popped once), which is the property Kraken2 documents for the same
/// structure -- "an average of O(1) time to calculate a new minimizer".
///
/// Ties keep the **earlier** position: the pop condition is a strict `>`, so
/// an equal key already in the deque survives and stays in front of the new
/// one. That is the leftmost-tie rule modern implementations use, a
/// deliberate deviation from Roberts et al. 2004's "each of the smallest
/// k-mers is a minimizer", and it is what makes a super-k-mer a single
/// well-defined run rather than an overlapping set.
/// # Storage: a fixed inline ring buffer, not a `VecDeque`
///
/// The deque never holds more than `k - m + 1` entries -- every entry is
/// inside the current window by the eviction invariant -- and `k <= 32` on
/// this path (the binned strategy is `u64`-keyed; `SignatureScanner::new`
/// asserts the range). So the maximum is 32 slots, which fits inline in the
/// scanner rather than behind a heap pointer, and a power-of-two capacity
/// turns the wrap into a mask.
///
/// This replaced a `VecDeque<(u64, usize)>` after `SignatureScanner::push`
/// measured as the largest single consumer of a binned run (31.8% of work,
/// `docs/BENCHMARKS.md`). Positions narrow to `u32` at the same time: a
/// position is an offset within one unambiguous stretch of one read, which
/// `kmer::extract_canonical_kmers_with_positions_into` already caps at
/// `u32::MAX` for the same reason.
#[derive(Debug, Clone)]
pub struct WindowMin {
    /// Ring buffer. Only `entries[(head + i) & MASK]` for `i < len` is live.
    entries: [(u64, u32); Self::CAPACITY],
    head: usize,
    len: usize,
}

impl Default for WindowMin {
    fn default() -> Self {
        Self { entries: [(0, 0); Self::CAPACITY], head: 0, len: 0 }
    }
}

impl WindowMin {
    /// `k - m + 1 <= k <= 32`, rounded to a power of two so the wrap is a
    /// mask rather than a compare-and-subtract.
    const CAPACITY: usize = 32;
    const MASK: usize = Self::CAPACITY - 1;

    /// `capacity` is accepted for source compatibility with the `VecDeque`
    /// form and checked rather than used: the buffer is always
    /// [`Self::CAPACITY`] slots, and a caller asking for more would be
    /// asking for a window this path cannot produce.
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(
            capacity <= Self::CAPACITY,
            "WindowMin: window of {capacity} exceeds the {} slots k <= 32 can need",
            Self::CAPACITY
        );
        Self::default()
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Admits `(key, position)`. Positions must be pushed in increasing
    /// order; skipping positions is allowed and is exactly how an ineligible
    /// m-mer is excluded from the candidate set.
    #[inline]
    pub fn push(&mut self, key: u64, position: usize) {
        // Ties keep the earlier position: `>` and not `>=`, so an equal key
        // already in the deque survives in front of the new one. See this
        // type's doc comment for why that rule is load-bearing.
        while self.len > 0 && self.entries[(self.head + self.len - 1) & Self::MASK].0 > key {
            self.len -= 1;
        }
        debug_assert!(self.len < Self::CAPACITY, "WindowMin overflow: window wider than k <= 32 allows");
        self.entries[(self.head + self.len) & Self::MASK] = (key, position as u32);
        self.len += 1;
    }

    /// Drops every entry whose position is below `first_position`.
    #[inline]
    pub fn evict_before(&mut self, first_position: usize) {
        let first = first_position as u32;
        while self.len > 0 && self.entries[self.head].1 < first {
            self.head = (self.head + 1) & Self::MASK;
            self.len -= 1;
        }
    }

    /// The minimum key currently in the window, with the position that
    /// produced it, or `None` if no candidate is in the window.
    #[inline]
    pub fn min(&self) -> Option<(u64, usize)> {
        if self.len == 0 {
            None
        } else {
            let (key, pos) = self.entries[self.head];
            Some((key, pos as usize))
        }
    }
}

/// One k-mer window's signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature {
    /// The minimum order key over the eligible canonical m-mers of this
    /// window, or [`INELIGIBLE_SIGNATURE`] when the window has none.
    pub value: u64,
    /// The m-mer position (counted from the start of the current stretch)
    /// that produced `value`, or `None` for the ineligible fallback.
    pub position: Option<usize>,
}

impl Signature {
    /// The bin this signature routes to.
    #[inline(always)]
    pub fn bin(self, num_bins: usize) -> usize {
        bin_of(self.value, num_bins)
    }
}

/// Rolling signature computation over one unambiguous stretch of sequence.
///
/// Fed one 2-bit base code at a time; yields a [`Signature`] for every
/// complete k-mer window, in order. Amortised O(1) per base: one m-mer roll
/// plus one amortised deque push and one amortised eviction.
///
/// The scanner knows nothing about ambiguous bases -- the caller feeds it
/// only unambiguous codes and calls [`reset`](Self::reset) at every `N`, so
/// no k-mer window ever spans one. That split of responsibility is why this
/// type has no notion of a "sequence" at all.
#[derive(Debug, Clone)]
pub struct SignatureScanner {
    roller: MmerRoller,
    window: WindowMin,
    k: usize,
    m: usize,
    /// Number of bases pushed since the last reset.
    bases_seen: usize,
}

impl SignatureScanner {
    /// `m` must be in `1..=k` and `k` in `1..=32`; the constructor
    /// `debug_assert!`s that rather than branching per base.
    pub fn new(k: usize, m: usize) -> Self {
        debug_assert!((1..=32).contains(&k), "SignatureScanner: k must be in 1..=32, got {k}");
        debug_assert!((1..=k).contains(&m), "SignatureScanner: m must be in 1..={k}, got {m}");
        Self {
            roller: MmerRoller::new(m),
            window: WindowMin::with_capacity(k - m + 1),
            k,
            m,
            bases_seen: 0,
        }
    }

    /// The window width in m-mer positions, `k - m + 1`.
    pub fn window_width(&self) -> usize {
        self.k - self.m + 1
    }

    /// Starts a fresh stretch: positions restart at zero and no window
    /// spans the discontinuity.
    pub fn reset(&mut self) {
        self.roller.reset();
        self.window.clear();
        self.bases_seen = 0;
    }

    /// Rolls one 2-bit base code in. Returns the signature of the k-mer
    /// window *ending at this base*, or `None` while fewer than `k` bases
    /// have been pushed since the last reset.
    #[inline]
    pub fn push(&mut self, bits: u64) -> Option<Signature> {
        let base_index = self.bases_seen;
        self.bases_seen += 1;

        if let Some(canonical) = self.roller.push(bits) {
            // The m-mer just completed starts `m - 1` bases back.
            let position = base_index + 1 - self.m;
            if let Some(key) = mmer_order_key(canonical, self.m) {
                self.window.push(key, position);
            }
        }

        if base_index + 1 < self.k {
            return None;
        }

        // The k-mer ending here starts at `base_index + 1 - k`, and its
        // window of m-mers starts at that same position.
        let first_position = base_index + 1 - self.k;
        self.window.evict_before(first_position);

        Some(match self.window.min() {
            Some((value, position)) => Signature { value, position: Some(position) },
            None => Signature { value: INELIGIBLE_SIGNATURE, position: None },
        })
    }
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: the unwrap/expect and
// stdout denials are about production paths. `print_stdout` is allowed here
// because the density test's job is to *report* a measured number, not only
// to assert a band around it.
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::kmer::{base_to_bits, reverse_complement_u64};

    /// The `VecDeque`-backed `WindowMin` this module used before the inline
    /// ring buffer replaced it, kept as both a correctness oracle and the
    /// other arm of `window_min_ab`.
    ///
    /// Two implementations of a monotonic deque can agree on every
    /// `min()` and still differ on a tie rule or an eviction boundary, and
    /// the tie rule here is load-bearing (it is what makes a super-k-mer a
    /// single run). `ring_and_deque_windows_agree_on_real_sequence` drives
    /// both from the same bases and requires the same answer at every step.
    #[derive(Debug, Clone, Default)]
    struct DequeWindowMin {
        entries: std::collections::VecDeque<(u64, usize)>,
    }

    impl DequeWindowMin {
        fn clear(&mut self) {
            self.entries.clear();
        }

        fn push(&mut self, key: u64, position: usize) {
            while let Some(&(back_key, _)) = self.entries.back() {
                if back_key > key {
                    self.entries.pop_back();
                } else {
                    break;
                }
            }
            self.entries.push_back((key, position));
        }

        fn evict_before(&mut self, first_position: usize) {
            while let Some(&(_, pos)) = self.entries.front() {
                if pos < first_position {
                    self.entries.pop_front();
                } else {
                    break;
                }
            }
        }

        fn min(&self) -> Option<(u64, usize)> {
            self.entries.front().copied()
        }
    }

    /// Deterministic ACGT bases, the shape a real read gives the scanner.
    fn bench_bases(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) & 0b11
            })
            .collect()
    }

    /// Drives both window implementations through the identical sequence of
    /// `push`/`evict_before`/`min` calls a `k`-window scan makes, and
    /// requires the same answer every time.
    ///
    /// This is the check the A/B rests on: a faster window that answers
    /// differently is not a faster window, it is a different bin function,
    /// and `tests/dual_strategy.rs` would only catch that at whole-file
    /// granularity.
    #[test]
    fn ring_and_deque_windows_agree_on_real_sequence() {
        for (k, m) in [(31usize, 7usize), (31, 9), (21, 7), (32, 1), (8, 7)] {
            let width = k - m + 1;
            let mut ring = WindowMin::with_capacity(width);
            let mut deque = DequeWindowMin::default();
            let mut roller = MmerRoller::new(m);

            for (i, bits) in bench_bases(4_000, 0xBEEF ^ k as u64).into_iter().enumerate() {
                if let Some(canonical) = roller.push(bits) {
                    let position = i + 1 - m;
                    if let Some(key) = mmer_order_key(canonical, m) {
                        ring.push(key, position);
                        deque.push(key, position);
                    }
                }
                if i + 1 >= k {
                    let first = i + 1 - k;
                    ring.evict_before(first);
                    deque.evict_before(first);
                    assert_eq!(ring.min(), deque.min(), "k={k} m={m} at base {i}");
                    assert_eq!(ring.is_empty(), deque.min().is_none(), "k={k} m={m} at base {i}");
                }
            }
            ring.clear();
            deque.clear();
            assert_eq!(ring.min(), deque.min());
        }
    }

    /// A/B measurement, not an assertion. `#[ignore]`d; run by hand with
    /// `cargo test --release window_min_ab -- --ignored --nocapture`.
    ///
    /// `SignatureScanner::push` measured as the largest single consumer of
    /// a binned run (31.8% of work, `docs/BENCHMARKS.md`), and the window is
    /// the part of it that was a heap-allocated `VecDeque`. Whether an
    /// inline ring buffer is actually faster is a question two whole-binary
    /// runs could not answer -- they disagreed, 1.104x one way and 0.939x
    /// the other, because a ~9 s run on a loaded host has a wider spread
    /// than the effect. This isolates it.
    #[test]
    #[ignore]
    fn window_min_ab() {
        use std::time::Instant;

        const BASES: usize = 4_000_000;
        const REPEATS: usize = 15;
        let (k, m) = (31usize, DEFAULT_M);
        let bases = bench_bases(BASES, 0x5EED);

        fn median(mut v: Vec<f64>) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            v[v.len() / 2]
        }

        // The window driven exactly as `SignatureScanner::push` drives it,
        // minus the roller, so the measurement is the window and not the
        // m-mer arithmetic both arms share.
        let keys: Vec<(usize, u64)> = {
            let mut roller = MmerRoller::new(m);
            let mut out = Vec::with_capacity(bases.len());
            for (i, &bits) in bases.iter().enumerate() {
                if let Some(canonical) = roller.push(bits) {
                    if let Some(key) = mmer_order_key(canonical, m) {
                        out.push((i + 1 - m, key));
                    }
                }
            }
            out
        };

        let mut ring_times = Vec::new();
        let mut deque_times = Vec::new();

        for _ in 0..REPEATS {
            // Interleaved, so drift in host load hits both arms.
            let mut ring = WindowMin::with_capacity(k - m + 1);
            let t = Instant::now();
            let mut acc = 0u64;
            let mut next = 0usize;
            for i in 0..bases.len() {
                if let Some(&(pos, key)) = keys.get(next) {
                    if pos + m == i + 1 {
                        ring.push(key, pos);
                        next += 1;
                    }
                }
                if i + 1 >= k {
                    ring.evict_before(i + 1 - k);
                    if let Some((v, _)) = ring.min() {
                        acc ^= v;
                    }
                }
            }
            ring_times.push(t.elapsed().as_secs_f64());
            std::hint::black_box(acc);

            let mut deque = DequeWindowMin::default();
            let t = Instant::now();
            let mut acc = 0u64;
            let mut next = 0usize;
            for i in 0..bases.len() {
                if let Some(&(pos, key)) = keys.get(next) {
                    if pos + m == i + 1 {
                        deque.push(key, pos);
                        next += 1;
                    }
                }
                if i + 1 >= k {
                    deque.evict_before(i + 1 - k);
                    if let Some((v, _)) = deque.min() {
                        acc ^= v;
                    }
                }
            }
            deque_times.push(t.elapsed().as_secs_f64());
            std::hint::black_box(acc);
        }

        let (r, d) = (median(ring_times), median(deque_times));
        println!("\nwindow_min A/B -- {BASES} bases, k={k} m={m}, {REPEATS} interleaved repeats");
        println!("  VecDeque     {d:.4} s");
        println!("  inline ring  {r:.4} s   {:.2}x\n", d / r);
    }

    /// Fixed-seed xorshift64, the same generator `tests/dual_strategy.rs`
    /// uses, so every test here is exactly reproducible with no RNG
    /// dependency.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn random_kmer(rng: &mut Xorshift, k: usize) -> u64 {
        let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
        rng.next() & mask
    }

    // ---------------------------------------------------------------
    // The strand invariance proof, executable.
    // ---------------------------------------------------------------

    /// The §3.3 invariant, and the single reason this module can be used as
    /// a bin function at all: if this ever fails, the counting path silently
    /// splits the count of every k-mer whose two orientations disagree.
    #[test]
    fn signature_is_invariant_under_reverse_complement() {
        let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
        for k in [7usize, 15, 21, 31, 32] {
            for m in [3usize, 5, 7, 9] {
                if m > k {
                    continue;
                }
                for _ in 0..2_000 {
                    let x = random_kmer(&mut rng, k);
                    let rc = reverse_complement_u64(x, k);
                    assert_eq!(
                        signature_of_kmer(x, k, m),
                        signature_of_kmer(rc, k, m),
                        "sig(x) != sig(rc(x)) at k={k} m={m} for kmer {x:#x}"
                    );
                }
            }
        }
    }

    /// The same invariant one level down: eligibility is a predicate on the
    /// canonical m-mer, so it cannot reintroduce the strand asymmetry the
    /// canonicalisation removed. Exhaustive over every m-mer for m = 7.
    #[test]
    fn eligibility_of_a_canonical_mmer_is_strand_symmetric() {
        let m = 7usize;
        for y in 0..(1u64 << (2 * m)) {
            let rc = reverse_complement_u64(y, m);
            assert_eq!(
                canonical_mmer(y, m),
                canonical_mmer(rc, m),
                "canonical form differs between strands for {y:#x}"
            );
            assert_eq!(
                is_eligible(canonical_mmer(y, m), m),
                is_eligible(canonical_mmer(rc, m), m),
                "eligibility differs between strands for {y:#x}"
            );
        }
    }

    /// `bin_of` is total for every signature a k-mer can produce, including
    /// the ineligible fallback, at every bin count the design allows.
    #[test]
    fn every_kmer_maps_into_the_bin_range() {
        let mut rng = Xorshift(0xDEAD_BEEF_CAFE_F00D);
        for num_bins in [1usize, 2, 64, 256, DEFAULT_NUM_BINS, 1024] {
            for _ in 0..20_000 {
                let k = 31;
                let sig = signature_of_kmer(random_kmer(&mut rng, k), k, DEFAULT_M);
                assert!(bin_of(sig, num_bins) < num_bins, "bin out of range for {num_bins} bins");
            }
            assert!(bin_of(INELIGIBLE_SIGNATURE, num_bins) < num_bins);
        }
    }

    // ---------------------------------------------------------------
    // The eligibility rule itself.
    // ---------------------------------------------------------------

    fn pack(bases: &str) -> u64 {
        let mut v = 0u64;
        for b in bases.bytes() {
            v = (v << 2) | base_to_bits(b).expect("test bases must be ACGT");
        }
        v
    }

    #[test]
    fn homopolymers_and_aaa_prefixes_are_ineligible() {
        let m = 7usize;
        for base in ["A", "C", "G", "T"] {
            let homo = pack(&base.repeat(m));
            assert!(!is_eligible(homo, m), "the {base}-homopolymer must be ineligible");
        }
        assert!(!is_eligible(pack("AAACGTC"), m), "an AAA prefix must be ineligible");
        assert!(!is_eligible(pack("AAAAGTC"), m), "AAAA also begins with AAA");
        // The boundary of the prefix rule: two leading As is not three.
        assert!(is_eligible(pack("AACGTCG"), m), "AA (not AAA) must stay eligible");
        assert!(is_eligible(pack("ACAGTCG"), m), "the declined KMC ACA clause must not be applied");
        assert!(is_eligible(pack("ACGAATC"), m), "the declined KMC interior-AA clause must not be applied");
    }

    /// A count, not an adjective: the rule must remove a small, known
    /// fraction of the m-mer space, not a large one. KMC's full rule
    /// disqualifies most of it; this narrow one must not.
    #[test]
    fn the_eligibility_rule_excludes_only_a_small_fraction_of_mmers() {
        let m = 7usize;
        let total = 1u64 << (2 * m);
        let ineligible = (0..total).filter(|&y| !is_eligible(canonical_mmer(y, m), m)).count();
        let fraction = ineligible as f64 / total as f64;
        println!("m=7: {ineligible}/{total} m-mers ineligible ({:.2}%)", fraction * 100.0);
        assert!(
            fraction < 0.06,
            "the narrow rule should exclude a few percent of m-mer space, got {:.2}%",
            fraction * 100.0
        );
    }

    /// Zero is reserved for the fallback by construction, not by luck: the
    /// mixer is a bijection, so only the all-`A` m-mer hashes to zero, and
    /// the homopolymer clause always rejects it.
    #[test]
    fn zero_signature_means_exactly_the_ineligible_fallback() {
        let m = 7usize;
        assert_eq!(mix64(0), 0, "the mixer must fix zero for the argument below to hold");
        for y in 0..(1u64 << (2 * m)) {
            if let Some(key) = mmer_order_key(canonical_mmer(y, m), m) {
                assert_ne!(key, INELIGIBLE_SIGNATURE, "an eligible m-mer produced the fallback signature");
            }
        }
    }

    #[test]
    fn an_all_ineligible_window_falls_back_to_bin_zero() {
        let k = 31usize;
        let m = DEFAULT_M;

        // A pure poly-A k-mer: every m-mer of it is the A-homopolymer.
        let poly_a = 0u64;
        assert_eq!(signature_of_kmer(poly_a, k, m), INELIGIBLE_SIGNATURE);
        assert_eq!(bin_of(signature_of_kmer(poly_a, k, m), DEFAULT_NUM_BINS), 0);

        // And through the rolling scanner, which is what the counting path
        // will actually use.
        let mut scanner = SignatureScanner::new(k, m);
        let mut last = None;
        for _ in 0..k {
            last = scanner.push(0);
        }
        let sig = last.expect("a k-base stretch must yield one window");
        assert_eq!(sig.value, INELIGIBLE_SIGNATURE);
        assert_eq!(sig.position, None);
        assert_eq!(sig.bin(DEFAULT_NUM_BINS), 0);

        // A poly-T k-mer canonicalises to poly-A m-mers, so it must land in
        // the same fallback rather than in a bin of its own.
        let mask = (1u64 << (2 * k)) - 1;
        assert_eq!(signature_of_kmer(mask, k, m), INELIGIBLE_SIGNATURE);
    }

    // ---------------------------------------------------------------
    // The rolling machinery, against the from-scratch definition.
    // ---------------------------------------------------------------

    fn random_sequence(rng: &mut Xorshift, len: usize) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..len).map(|_| bases[(rng.next() % 4) as usize]).collect()
    }

    fn codes(seq: &[u8]) -> Vec<u64> {
        seq.iter().map(|&b| base_to_bits(b).expect("ACGT only")).collect()
    }

    fn pack_window(codes: &[u64]) -> u64 {
        codes.iter().fold(0u64, |acc, &c| (acc << 2) | c)
    }

    /// The scanner must agree with `signature_of_kmer` -- the quadratic,
    /// from-scratch definition -- at every window position, for every
    /// combination of `k` and `m` the design allows near its defaults.
    #[test]
    fn the_rolling_scanner_matches_the_from_scratch_signature_at_every_window() {
        let mut rng = Xorshift(0x0123_4567_89AB_CDEF);
        let seq = random_sequence(&mut rng, 500);
        let c = codes(&seq);

        for k in [8usize, 15, 21, 31, 32] {
            for m in [1usize, 2, 3, 5, 7, 9, 11] {
                if m > k {
                    continue;
                }
                let mut scanner = SignatureScanner::new(k, m);
                let mut window_index = 0usize;
                for (i, &bits) in c.iter().enumerate() {
                    if let Some(sig) = scanner.push(bits) {
                        let start = i + 1 - k;
                        let expected = signature_of_kmer(pack_window(&c[start..start + k]), k, m);
                        assert_eq!(
                            sig.value, expected,
                            "rolling signature diverges at k={k} m={m} window {window_index}"
                        );
                        window_index += 1;
                    }
                }
                assert_eq!(window_index, c.len() - k + 1, "wrong number of windows at k={k} m={m}");
            }
        }
    }

    /// A reset must leave no trace of the previous stretch: the signatures
    /// after it must be exactly those of a scanner that only ever saw the
    /// second stretch. This is the property the `N` handling in
    /// `superkmer.rs` will depend on.
    #[test]
    fn reset_makes_the_scanner_indistinguishable_from_a_fresh_one() {
        let mut rng = Xorshift(0xFEED_FACE_1234_5678);
        let first = codes(&random_sequence(&mut rng, 90));
        let second = codes(&random_sequence(&mut rng, 120));
        let (k, m) = (31usize, DEFAULT_M);

        let mut reused = SignatureScanner::new(k, m);
        for &bits in &first {
            reused.push(bits);
        }
        reused.reset();
        let after_reset: Vec<Signature> = second.iter().filter_map(|&b| reused.push(b)).collect();

        let mut fresh = SignatureScanner::new(k, m);
        let from_fresh: Vec<Signature> = second.iter().filter_map(|&b| fresh.push(b)).collect();

        assert_eq!(after_reset, from_fresh);
    }

    /// The deque, against the O(w) definition of a sliding-window minimum,
    /// including the leftmost-tie rule.
    #[test]
    fn window_min_matches_a_brute_force_sliding_minimum() {
        let mut rng = Xorshift(0xABCD_0123_4567_89AB);
        // Deliberately few distinct keys, so ties are common and the
        // tie-breaking rule is actually exercised.
        let keys: Vec<u64> = (0..400).map(|_| rng.next() % 7).collect();
        let width = 11usize;

        let mut window = WindowMin::with_capacity(width);
        for (i, &key) in keys.iter().enumerate() {
            window.push(key, i);
            if i + 1 < width {
                continue;
            }
            let first = i + 1 - width;
            window.evict_before(first);

            let slice = &keys[first..=i];
            let best = slice.iter().copied().min().expect("non-empty");
            let best_at = first + slice.iter().position(|&k| k == best).expect("present");
            assert_eq!(window.min(), Some((best, best_at)), "window ending at {i}");
        }
    }

    // ---------------------------------------------------------------
    // Realised density -- measured, not assumed.
    // ---------------------------------------------------------------

    /// Roberts et al. 2004 predicts a random-order minimizer density of
    /// `2/(w+1)`, with the caveat that on real sequence "the actual
    /// proportion of k-mers that are minimizers can be a few percent above"
    /// it. Two further effects push in the other direction here: canonical
    /// m-mers make the induced hash distribution slightly non-uniform
    /// (Marcais et al. 2024), and the eligibility rule removes candidates,
    /// which lowers the *selected* fraction because the surviving candidates
    /// are sampled from a sparser set.
    ///
    /// So this measures rather than assumes, reports both the classical
    /// density (distinct selected positions per position) and the quantity
    /// the memory budget actually depends on (super-k-mer runs per window,
    /// i.e. how often the signature *changes*), and asserts a band wide
    /// enough to be a genuine check on the implementation rather than a
    /// restatement of whatever it happens to produce.
    #[test]
    fn realised_density_sits_in_a_band_around_two_over_w_plus_one() {
        let mut rng = Xorshift(0x5EED_D0DE_2026_0825);
        let seq = random_sequence(&mut rng, 2_000_000);
        let c = codes(&seq);

        for (k, m) in [(31usize, 7usize), (31, 9), (21, 7), (32, 5)] {
            let w = k - m + 1;
            let theory = 2.0 / (w as f64 + 1.0);

            let mut scanner = SignatureScanner::new(k, m);
            let mut windows = 0u64;
            let mut selected_positions = 0u64;
            let mut runs = 0u64;
            let mut fallbacks = 0u64;
            let mut last_position: Option<usize> = None;
            let mut last_value: Option<u64> = None;

            for &bits in &c {
                let Some(sig) = scanner.push(bits) else { continue };
                windows += 1;
                if sig.position.is_none() {
                    fallbacks += 1;
                }
                if sig.position != last_position {
                    selected_positions += 1;
                    last_position = sig.position;
                }
                if Some(sig.value) != last_value {
                    runs += 1;
                    last_value = Some(sig.value);
                }
            }

            let density = selected_positions as f64 / windows as f64;
            let run_density = runs as f64 / windows as f64;
            println!(
                "k={k} m={m} w={w}: theory 2/(w+1)={theory:.5}  \
                 realised density={density:.5} ({:+.1}%)  \
                 super-k-mer runs/window={run_density:.5}  fallback windows={fallbacks}",
                (density / theory - 1.0) * 100.0
            );

            assert!(
                (0.85 * theory..=1.15 * theory).contains(&density),
                "realised density {density:.5} is outside +-15% of 2/(w+1) = {theory:.5} at k={k} m={m}"
            );
            // Runs can only merge adjacent selections (two different
            // positions carrying the same m-mer), never split one.
            assert!(run_density <= density, "signature runs cannot outnumber selected positions");
        }
    }

    /// Real reads are not uniform random sequence. A poly-A tract is the
    /// case the eligibility rule exists for, and the case where the
    /// fallback bin has to absorb whole windows -- this pins that it does,
    /// and that ordinary sequence around it is unaffected.
    #[test]
    fn a_poly_a_tract_routes_to_the_fallback_and_its_flanks_do_not() {
        let mut rng = Xorshift(0x1111_2222_3333_4444);
        let (k, m) = (31usize, DEFAULT_M);
        let mut seq = random_sequence(&mut rng, 200);
        seq.extend(std::iter::repeat_n(b'A', 200));
        seq.extend(random_sequence(&mut rng, 200));

        let mut scanner = SignatureScanner::new(k, m);
        let mut fallback_windows = 0u64;
        let mut total_windows = 0u64;
        for &bits in &codes(&seq) {
            if let Some(sig) = scanner.push(bits) {
                total_windows += 1;
                if sig.position.is_none() {
                    fallback_windows += 1;
                    assert_eq!(sig.bin(DEFAULT_NUM_BINS), 0, "a fallback window must route to bin 0");
                }
            }
        }

        // The 200-base poly-A run contains 200 - k + 1 = 170 windows made
        // entirely of A, and every one of them must fall back. Windows that
        // straddle either end of the run may fall back too -- a straddling
        // m-mer is still ineligible whenever it happens to begin with three
        // As after canonicalisation -- so the count is bounded, not exact:
        // a straddling window is one of the at most `k - 1` on each side.
        let all_a_windows = (200 - k + 1) as u64;
        println!("poly-A tract: {fallback_windows} fallback windows of {total_windows} (all-A windows: {all_a_windows})");
        assert!(
            fallback_windows >= all_a_windows,
            "every all-A window must fall back, got {fallback_windows} < {all_a_windows}"
        );
        assert!(
            fallback_windows <= all_a_windows + 2 * (k as u64 - 1),
            "only windows straddling the tract may join it, got {fallback_windows}"
        );
        assert!(fallback_windows < total_windows / 2, "the flanking sequence must not fall back");
    }
}
