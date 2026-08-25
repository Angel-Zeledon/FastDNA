// src/translate.rs

//! DNA -> protein translation in all six reading frames, over selectable
//! NCBI genetic-code tables.
//!
//! # Where the codon assignments come from
//!
//! Every table below is transcribed **verbatim** from the NCBI "Genetic
//! Codes" reference page -- <https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi>
//! (NCBI Taxonomy, "The Genetic Codes", version 4.6) -- which is also the
//! source GenBank's own `/transl_table=` qualifier refers to. That page
//! prints each code as four aligned 64-character rows:
//!
//! ```text
//!   AAs  = FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG
//! Starts = ---M------**--*----M---------------M----------------------------
//! Base1  = TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG
//! Base2  = TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG
//! Base3  = TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG
//! ```
//!
//! The `AAs` row is stored here exactly as NCBI prints it, so the constant
//! in this file can be diffed character-for-character against the source
//! rather than trusted. The `Starts` row is deliberately *not* stored: this
//! module translates a reading frame, it does not call ORFs, and a start
//! codon only means "Met" when it is the first codon of a real CDS. Baking
//! the `Starts` row into plain frame translation would rewrite an internal
//! `TTG` or `GTG` to `M` in the middle of a protein, which is wrong.
//!
//! **Consequence worth stating explicitly:** NCBI table 11
//! (Bacterial/Archaeal/Plant Plastid) has an `AAs` row *identical* to table
//! 1 (Standard) -- the two codes differ only in their `Starts` row, i.e. in
//! which codons may initiate translation. Since this module does not apply
//! start codons (see above), `table=11` and `table=1` produce byte-identical
//! proteins here. That is not an omission; it is what the NCBI definitions
//! say. Table 11 is still offered by id because this library's users count
//! bacterial reads and asking them to pass `table=1` for a bacterial genome
//! would be asking them to write something that reads as wrong.
//!
//! # Why translation is fast here
//!
//! A codon is three bases. `kmer.rs` already encodes a base in two bits
//! (`base_to_bits`: A=00, C=01, G=10, T=11), so a codon is exactly six bits
//! -- an integer in `0..64`. A genetic code is therefore a plain
//! `[u8; 64]` lookup indexed by that integer: no string comparison, no
//! hashing, no per-codon allocation, one array index per amino acid. This
//! is the same 2-bit packing the k-mer path uses, reused rather than
//! reinvented, which is also what makes the reverse-strand frames cheap:
//! complementing a 2-bit base is a single XOR, so a reverse frame packs the
//! reverse complement of a codon directly as it reads it, with no
//! reverse-complemented copy of the sequence anywhere (see `pack_codon`).

use crate::error::{FastDnaError, Result};
use crate::kmer;

/// The amino-acid letter emitted for a codon that cannot be translated --
/// one containing `N` or any other non-ACGT byte.
///
/// `X` is the IUPAC/IUB code for "any amino acid" and is what UniProt,
/// EMBOSS and Biopython all emit for an untranslatable codon. Emitting a
/// placeholder rather than skipping the codon is the whole point: a skipped
/// codon silently shortens the protein and shifts every residue after it,
/// so an alignment, a motif offset or a k-mer index built on the result
/// would be wrong with nothing to point at.
pub const AMBIGUOUS_AA: u8 = b'X';

/// The amino-acid letter emitted for a stop codon. `*` is the NCBI/UniProt
/// convention and is what the `AAs` rows below already contain.
pub const STOP_AA: u8 = b'*';

/// The number of codons in a genetic code: 4 bases ^ 3 positions.
const CODON_COUNT: usize = 64;

/// NCBI table 1, The Standard Code. `AAs` row, verbatim.
const NCBI_AAS_TABLE_1: &[u8; CODON_COUNT] =
    b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

/// NCBI table 2, The Vertebrate Mitochondrial Code. `AAs` row, verbatim.
///
/// Differs from table 1 at four codons: `TGA` is Trp rather than a stop,
/// `ATA` is Met rather than Ile, and `AGA`/`AGG` are stops rather than Arg.
const NCBI_AAS_TABLE_2: &[u8; CODON_COUNT] =
    b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG";

/// NCBI table 4, The Mold, Protozoan, and Coelenterate Mitochondrial Code
/// and the Mycoplasma/Spiroplasma Code. `AAs` row, verbatim.
///
/// Differs from table 1 at exactly one codon: `TGA` is Trp, not a stop.
const NCBI_AAS_TABLE_4: &[u8; CODON_COUNT] =
    b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

/// NCBI table 11, The Bacterial, Archaeal and Plant Plastid Code. `AAs`
/// row, verbatim -- identical to table 1's, as NCBI itself prints it. See
/// this module's header for why that is expected rather than a copy-paste
/// slip.
const NCBI_AAS_TABLE_11: &[u8; CODON_COUNT] =
    b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

/// The base order NCBI's table rows are written in (`Base1`/`Base2`/`Base3`
/// cycle T, C, A, G), mapped to this crate's 2-bit encoding from `kmer.rs`
/// (A=0, C=1, G=2, T=3). Used only by `pack_ncbi_row`.
const NCBI_BASE_ORDER_TO_BITS: [usize; 4] = [0b11, 0b01, 0b00, 0b10];

/// Permutes an NCBI `AAs` row into an array indexed by this crate's 2-bit
/// codon packing.
///
/// NCBI orders its 64 entries by T, C, A, G at each of the three positions;
/// `kmer::base_to_bits` orders by A, C, G, T. Rather than hand-reordering
/// the table (which would make the constants above no longer diffable
/// against the source page -- the one property that makes them checkable),
/// the row is stored verbatim and permuted here, at compile time.
const fn pack_ncbi_row(aas: &[u8; CODON_COUNT]) -> [u8; CODON_COUNT] {
    let mut packed = [0u8; CODON_COUNT];
    let mut ncbi_index = 0;
    while ncbi_index < CODON_COUNT {
        let first = NCBI_BASE_ORDER_TO_BITS[ncbi_index / 16];
        let second = NCBI_BASE_ORDER_TO_BITS[(ncbi_index / 4) % 4];
        let third = NCBI_BASE_ORDER_TO_BITS[ncbi_index % 4];
        packed[(first << 4) | (second << 2) | third] = aas[ncbi_index];
        ncbi_index += 1;
    }
    packed
}

/// One NCBI genetic code, as a 64-entry lookup indexed by a 2-bit-packed
/// codon (see this module's header for why that representation is the fast
/// one here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranslationTable {
    /// The NCBI `transl_table` id this code corresponds to.
    pub id: u8,
    code: [u8; CODON_COUNT],
}

static TABLE_1: TranslationTable = TranslationTable { id: 1, code: pack_ncbi_row(NCBI_AAS_TABLE_1) };
static TABLE_2: TranslationTable = TranslationTable { id: 2, code: pack_ncbi_row(NCBI_AAS_TABLE_2) };
static TABLE_4: TranslationTable = TranslationTable { id: 4, code: pack_ncbi_row(NCBI_AAS_TABLE_4) };
static TABLE_11: TranslationTable =
    TranslationTable { id: 11, code: pack_ncbi_row(NCBI_AAS_TABLE_11) };

/// The NCBI table ids this module implements, in the order an error message
/// should list them.
pub const SUPPORTED_TABLE_IDS: [u8; 4] = [1, 2, 4, 11];

impl TranslationTable {
    /// Looks up a genetic code by its NCBI `transl_table` id.
    ///
    /// Returns `InvalidConfig` -- naming every supported id -- rather than
    /// silently falling back to the standard code for an unknown one: a
    /// caller who asks for table 5 and quietly gets table 1 back has no way
    /// to notice, and every downstream protein would be wrong.
    pub fn from_id(id: u8) -> Result<&'static TranslationTable> {
        match id {
            1 => Ok(&TABLE_1),
            2 => Ok(&TABLE_2),
            4 => Ok(&TABLE_4),
            11 => Ok(&TABLE_11),
            _ => Err(FastDnaError::InvalidConfig {
                parameter: "table",
                reason: format!(
                    "unsupported NCBI translation table {id}; supported ids are {:?}",
                    SUPPORTED_TABLE_IDS
                ),
            }),
        }
    }

    /// The amino acid a 2-bit-packed codon (an integer in `0..64`) encodes.
    /// Values at or above 64 cannot come from `pack_codon` and are reported
    /// as untranslatable rather than indexing out of bounds.
    #[inline(always)]
    pub fn translate_codon(&self, packed: u64) -> u8 {
        match self.code.get(packed as usize) {
            Some(&aa) => aa,
            None => AMBIGUOUS_AA,
        }
    }
}

/// What to do at a stop codon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopHandling {
    /// Emit `*` and keep going to the end of the frame. **The default.**
    ///
    /// Reason it is the default: a frame translation is only trustworthy if
    /// residue `i` of the protein corresponds to codon `i` of the frame. In
    /// a six-frame scan of raw reads or a whole contig -- what this library
    /// is for -- there is no reason to believe the first stop encountered is
    /// the real end of anything, and truncating there would silently discard
    /// every downstream ORF in that frame. Callers who want a single CDS's
    /// product ask for it explicitly.
    #[default]
    Translate,
    /// Stop at (and exclude) the first stop codon.
    ///
    /// The right choice when the input *is* a coding sequence -- a called
    /// gene, a CDS pulled from a GenBank record -- and the wanted answer is
    /// its protein product, without a trailing `*`. Matches Biopython's
    /// `Seq.translate(to_stop=True)`.
    StopAtFirst,
}

/// One of the six reading frames.
///
/// Written `+1`/`+2`/`+3` and `-1`/`-2`/`-3` in the literature and in this
/// crate's public APIs; `0` is not a frame. A negative frame translates the
/// reverse complement of the sequence, starting `|frame| - 1` bases in from
/// its 3' end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    offset: usize,
    reverse: bool,
}

/// The six frames, in the conventional order a six-frame translation is
/// reported in.
pub const ALL_FRAMES: [i8; 6] = [1, 2, 3, -1, -2, -3];

impl Frame {
    /// Parses the conventional `+1..+3`/`-1..-3` spelling.
    ///
    /// `0` and `|frame| > 3` are `InvalidConfig`, not a clamp: a caller who
    /// passes `4` meant something, and quietly translating frame 1 instead
    /// would hide it.
    pub fn from_i8(frame: i8) -> Result<Frame> {
        match frame {
            1..=3 => Ok(Frame { offset: (frame - 1) as usize, reverse: false }),
            -3..=-1 => Ok(Frame { offset: (-frame - 1) as usize, reverse: true }),
            _ => Err(FastDnaError::InvalidConfig {
                parameter: "frame",
                reason: format!(
                    "frame must be one of {ALL_FRAMES:?} (0 is not a reading frame), got {frame}"
                ),
            }),
        }
    }

    /// The `+1..+3`/`-1..-3` spelling of this frame.
    pub fn as_i8(self) -> i8 {
        let magnitude = (self.offset + 1) as i8;
        if self.reverse {
            -magnitude
        } else {
            magnitude
        }
    }

    /// How many bases into the (possibly reverse-complemented) sequence the
    /// first codon starts: 0, 1 or 2.
    pub fn offset(self) -> usize {
        self.offset
    }

    /// Whether this frame reads the reverse-complement strand.
    pub fn is_reverse(self) -> bool {
        self.reverse
    }
}

/// The 2-bit complement of a base. A(`00`) <-> T(`11`) and C(`01`) <->
/// G(`10`), so complementing a packed base is exactly `^ 0b11` -- the same
/// relationship `kmer::reverse_complement_u64` opens with its `!kmer`,
/// applied to one base instead of thirty-two.
const BASE_COMPLEMENT: u64 = 0b11;

/// Packs one codon-sized window into a 6-bit codon, or `None` if any of its
/// bases is not A/C/G/T (`N`, an IUPAC ambiguity code, or junk). `REVERSE`
/// packs the window's reverse complement instead of the window itself.
///
/// **Why `REVERSE` is a const parameter and not a runtime flag.** It is
/// resolved when the compiler monomorphizes this function, so a reverse
/// frame's loop contains no strand test at all; the previous form branched
/// on `frame.is_reverse()` twice per codon (once to compute the window's
/// start, once to decide whether to flip it).
///
/// **Why the reverse case is assembled rather than flipped.** The reverse
/// complement of the window `b0 b1 b2` is `comp(b2) comp(b1) comp(b0)`, and
/// complementing a 2-bit base is one XOR, so packing it directly costs three
/// XORs more than the forward case: 7 operations against 4. Producing the
/// same six bits by packing forwards and calling
/// `kmer::reverse_complement_u64(packed, 3)` costs the forward packing plus
/// one `!`, two shift/mask/shift/mask/or groups, a `swap_bytes` and a final
/// shift -- roughly 16 operations. The saving is per codon of every reverse
/// frame: a six-frame scan of a 4.6 Mb bacterial genome packs ~4.6 million
/// reverse codons.
#[inline(always)]
fn pack_codon<const REVERSE: bool>(window: &[u8]) -> Option<u64> {
    let &[first, second, third] = window else { return None };
    let first = kmer::base_to_bits(first)?;
    let second = kmer::base_to_bits(second)?;
    let third = kmer::base_to_bits(third)?;
    if REVERSE {
        Some(
            ((third ^ BASE_COMPLEMENT) << 4)
                | ((second ^ BASE_COMPLEMENT) << 2)
                | (first ^ BASE_COMPLEMENT),
        )
    } else {
        Some((first << 4) | (second << 2) | third)
    }
}

/// Translates a stream of codon-sized windows onto the end of `protein`.
///
/// The windows arrive from `chunks_exact(3)`/`rchunks_exact(3)`, which is
/// what removes the per-codon `offset + 3 * codon_index` multiply-add and
/// the three bounds checks the old index-based `pack_codon` paid on every
/// codon: the iterator does the striding, and the `&[u8; 3]` pattern is one
/// length test the optimizer can hoist out of the chunked loop.
#[inline]
fn translate_codons<'a, const REVERSE: bool, I>(
    codons: I,
    table: &TranslationTable,
    stop_handling: StopHandling,
    protein: &mut Vec<u8>,
) where
    I: Iterator<Item = &'a [u8]>,
{
    for window in codons {
        let amino_acid = match pack_codon::<REVERSE>(window) {
            Some(codon) => table.translate_codon(codon),
            None => AMBIGUOUS_AA,
        };

        if amino_acid == STOP_AA && stop_handling == StopHandling::StopAtFirst {
            break;
        }
        protein.push(amino_acid);
    }
}

/// Translates `seq` in one reading frame.
///
/// Behaviour that callers depend on, all of it deliberate:
///
/// - **Trailing partial codons are dropped.** A frame yields
///   `(len - offset) / 3` residues; the 1 or 2 leftover bases at the end
///   cannot be read as a codon and contribute nothing. For a reverse frame
///   the leftover bases are at the *5'* end of the original sequence, since
///   a reverse frame reads from the 3' end backwards.
/// - **A codon containing any non-ACGT base becomes `X`** and does not shift
///   anything after it (see `AMBIGUOUS_AA`).
/// - **Lowercase input is accepted**, because `kmer::base_to_bits` accepts
///   it -- soft-masked reference sequence translates the same as uppercase
///   rather than becoming a run of `X`.
/// - **`U` is read as `T`**, again inherited from `kmer::base_to_bits`, so
///   an RNA sequence translates without being transcribed back first.
///
/// Reverse frames never materialize a reverse-complemented copy of the
/// sequence. A codon is exactly three bases, and the reverse complement of a
/// three-base window is just its three complemented bases assembled in the
/// other order -- three XORs (see `pack_codon`). So a reverse frame walks the
/// original bytes from the 3' end with `rchunks_exact(3)` and complements
/// each window as it packs it: no second reverse-complement implementation,
/// and no allocation proportional to the sequence length.
///
/// Allocates once, for the returned `String`. Callers translating many
/// sequences in a row should use [`translate_into`], which fills a buffer
/// they own and therefore allocates once in total rather than once per
/// sequence.
pub fn translate(
    seq: &[u8],
    frame: Frame,
    table: &TranslationTable,
    stop_handling: StopHandling,
) -> String {
    // Sized exactly, so `translate_into`'s `reserve` is a no-op and the
    // vector never regrows: one allocation for the whole call, as before.
    let mut protein = Vec::with_capacity(seq.len().saturating_sub(frame.offset()) / 3);
    translate_into(seq, frame, table, stop_handling, &mut protein);

    // Every byte pushed came either from a `TranslationTable`'s code array
    // (an NCBI `AAs` row, ASCII by construction) or from `AMBIGUOUS_AA`, so
    // this cannot fail -- the same argument `kmer::decode_kmer` makes for
    // its own `from_utf8`. It moves the buffer rather than copying it.
    String::from_utf8(protein).unwrap_or_default()
}

/// [`translate`], writing into a caller-owned buffer instead of returning a
/// fresh `String`.
///
/// `protein` is cleared and then filled with the frame's residues, as ASCII
/// bytes. Its existing capacity is reused, so a caller that translates N
/// sequences through one buffer performs **one** heap allocation in total
/// where N calls to `translate` perform N allocations and N frees. That is
/// the whole reason this variant exists: `src/ffi.rs` translates every
/// record of a file in up to six frames, which is `6 * records` allocations
/// saved on a path whose per-row work is otherwise a memcpy into an Arrow
/// buffer.
///
/// The bytes are ASCII by construction (see `translate`), so
/// `std::str::from_utf8` over the result is the same validation `translate`
/// already runs -- moved to the caller, not added.
pub fn translate_into(
    seq: &[u8],
    frame: Frame,
    table: &TranslationTable,
    stop_handling: StopHandling,
    protein: &mut Vec<u8>,
) {
    protein.clear();
    let offset = frame.offset();
    if seq.len() < offset + 3 {
        return;
    }
    protein.reserve((seq.len() - offset) / 3);

    if frame.is_reverse() {
        // A reverse frame reads from the 3' end, so its codon boundaries are
        // anchored there: `offset` bases are dropped off the 3' end and the
        // windows walk backwards from what is left. `rchunks_exact` produces
        // exactly those windows, in exactly that order, and leaves the 1-2
        // unusable bases at the 5' end -- which is where a reverse frame's
        // partial codon belongs.
        translate_codons::<true, _>(
            seq[..seq.len() - offset].rchunks_exact(3),
            table,
            stop_handling,
            protein,
        );
    } else {
        translate_codons::<false, _>(
            seq[offset..].chunks_exact(3),
            table,
            stop_handling,
            protein,
        );
    }
}

/// Translates `seq` in all six frames, in `ALL_FRAMES` order, returning
/// `(frame, protein)` pairs.
pub fn translate_six_frames(
    seq: &[u8],
    table: &TranslationTable,
    stop_handling: StopHandling,
) -> Vec<(i8, String)> {
    // `Vec::with_capacity(6)` and a plain loop rather than `collect()`: a
    // `filter_map` reports a lower size hint of 0, so the collected vector
    // started at capacity 0 and grew 4 -> 8, which is two allocations, one
    // free and a 128-byte move for a result whose length is known to be
    // exactly six.
    let mut proteins = Vec::with_capacity(ALL_FRAMES.len());
    for frame_id in ALL_FRAMES {
        // Built directly instead of through `Frame::from_i8`: every id in
        // `ALL_FRAMES` is a valid frame by construction, so parsing them
        // would be six range matches and six `Result`s discarded on the spot.
        let frame =
            Frame { offset: (frame_id.unsigned_abs() - 1) as usize, reverse: frame_id < 0 };
        proteins.push((frame_id, translate(seq, frame, table, stop_handling)));
    }
    proteins
}

/// Extracts every k-mer of *amino acids* from a translated protein, as
/// overlapping windows of length `k`.
///
/// **These are not canonical k-mers, and this is not `kmer.rs`'s code path.**
/// Two independent reasons, both structural rather than a matter of effort:
///
/// 1. A protein has no reverse complement. Canonicalization in `kmer.rs`
///    exists because a DNA k-mer and its reverse complement are the same
///    physical double-stranded object, so collapsing them halves the table
///    and makes counts strand-independent. A peptide read backwards is a
///    different peptide, and `min(kmer, revcomp(kmer))` is not defined for
///    one. Every amino-acid k-mer here is therefore stored as it reads,
///    N-terminus to C-terminus.
/// 2. The alphabet does not fit. `kmer.rs` packs a base into 2 bits because
///    there are 4 of them; there are 20 amino acids plus `*` and `X`, which
///    needs 5 bits and would cap `k` at 12 in a `u64` while giving up the
///    O(1) rolling-window and reverse-complement tricks that are the entire
///    reason for the packed representation. So this path stays on plain
///    bytes: a window, a `String`, a hash map.
///
/// Windows containing `*` (a stop) or `X` (an untranslatable codon) are
/// **kept**, deliberately: dropping them here would be a policy this
/// function has no business choosing on the caller's behalf, and it is
/// trivial to filter the result. `k` larger than the protein yields an
/// empty result; `k == 0` is rejected.
pub fn amino_acid_kmers(protein: &str, k: usize) -> Result<Vec<&str>> {
    validate_aa_k(protein, k)?;
    let bytes = protein.as_bytes();
    if bytes.len() < k {
        return Ok(Vec::new());
    }
    Ok(bytes
        .windows(k)
        .filter_map(|window| std::str::from_utf8(window).ok())
        .collect())
}

/// Counts amino-acid k-mers (see `amino_acid_kmers` for what they are and
/// are not), returned sorted by k-mer so the output is deterministic --
/// a hash map's iteration order is not, and this feeds an Arrow table a
/// caller may diff between runs.
pub fn count_amino_acid_kmers(protein: &str, k: usize) -> Result<Vec<(String, u32)>> {
    let mut counts: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for kmer_str in amino_acid_kmers(protein, k)? {
        *counts.entry(kmer_str).or_insert(0) += 1;
    }
    Ok(counts.into_iter().map(|(kmer_str, count)| (kmer_str.to_string(), count)).collect())
}

/// Shared validation for the amino-acid k-mer path: a usable `k`, and a
/// protein this module can safely slice by bytes.
fn validate_aa_k(protein: &str, k: usize) -> Result<()> {
    if k == 0 {
        return Err(FastDnaError::InvalidConfig {
            parameter: "k",
            reason: "amino-acid k-mer length must be at least 1, got 0".to_string(),
        });
    }
    // Proteins produced by `translate` are ASCII by construction. A caller
    // can hand this function any `&str`, though, and a byte window over
    // multi-byte UTF-8 would split a character mid-sequence -- so it is
    // refused up front rather than silently dropped by the `from_utf8`
    // filter in `amino_acid_kmers`.
    if !protein.is_ascii() {
        return Err(FastDnaError::InvalidConfig {
            parameter: "protein",
            reason: "protein sequences must be ASCII amino-acid letters; \
                     got non-ASCII characters, which cannot be split into k-mers by byte"
                .to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: unwrap/expect denial is
// about production paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn standard() -> &'static TranslationTable {
        TranslationTable::from_id(1).unwrap()
    }

    fn tr(seq: &str, frame: i8, table_id: u8) -> String {
        translate(
            seq.as_bytes(),
            Frame::from_i8(frame).unwrap(),
            TranslationTable::from_id(table_id).unwrap(),
            StopHandling::Translate,
        )
    }

    // -- the NCBI rows, checked position by position -------------------

    /// The four table constants must be exactly 64 characters and contain
    /// only amino-acid letters or `*`. A transcription slip that dropped or
    /// added one character would otherwise shift every codon after it and
    /// still "work".
    #[test]
    fn every_ncbi_row_is_64_valid_amino_acid_letters() {
        for row in [NCBI_AAS_TABLE_1, NCBI_AAS_TABLE_2, NCBI_AAS_TABLE_4, NCBI_AAS_TABLE_11] {
            assert_eq!(row.len(), 64);
            for &c in row.iter() {
                assert!(
                    c == STOP_AA || c.is_ascii_uppercase(),
                    "unexpected character {:?} in an NCBI AAs row",
                    c as char
                );
            }
        }
    }

    /// Spot-checks the NCBI-order -> 2-bit-order permutation against codons
    /// whose assignment is common knowledge, at both ends and in the middle
    /// of the row. If `pack_ncbi_row` permuted wrongly, these would land on
    /// the wrong letters.
    #[test]
    fn the_packed_table_agrees_with_known_codon_assignments() {
        let known = [
            ("TTT", 'F'),
            ("ATG", 'M'),
            ("TGG", 'W'),
            ("TAA", '*'),
            ("TAG", '*'),
            ("TGA", '*'),
            ("GGG", 'G'),
            ("AAA", 'K'),
            ("CCC", 'P'),
            ("GTC", 'V'),
        ];
        for (codon, expected) in known {
            assert_eq!(tr(codon, 1, 1), expected.to_string(), "codon {codon}");
        }
    }

    /// `pack_codon::<true>` assembles a codon's reverse complement from three
    /// XORs instead of packing it forwards and calling
    /// `kmer::reverse_complement_u64(packed, 3)`. The two must agree on all
    /// 64 codons -- exhaustively, since the domain is 64 values and a
    /// spot-check would leave the other 60 untested.
    #[test]
    fn the_reverse_codon_packing_agrees_with_the_kmer_bit_trick_on_all_64_codons() {
        for first in b"ACGT" {
            for second in b"ACGT" {
                for third in b"ACGT" {
                    let window = [*first, *second, *third];
                    let forward = pack_codon::<false>(&window).unwrap();
                    assert_eq!(
                        pack_codon::<true>(&window).unwrap(),
                        kmer::reverse_complement_u64(forward, 3),
                        "codon {}",
                        std::str::from_utf8(&window).unwrap()
                    );
                }
            }
        }
    }

    // -- known-answer translation --------------------------------------

    /// Known answer from the Biopython tutorial's own translation example
    /// (Biopython Tutorial and Cookbook, "Translation"), which uses this
    /// exact sequence and reports `MAIVMGR*KGAR*` under the standard code.
    /// Verifiable by hand against the NCBI table: ATG GCC ATT GTA ATG GGC
    /// CGC TGA AAG GGT GCC CGA TAG.
    #[test]
    fn the_standard_code_translates_the_biopython_example() {
        assert_eq!(tr("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", 1, 1), "MAIVMGR*KGAR*");
    }

    /// NCBI table 4 reassigns `TGA` from stop to Trp and changes nothing
    /// else; on this sequence that is the single visible difference.
    #[test]
    fn table_4_reads_tga_as_tryptophan() {
        assert_eq!(tr("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", 1, 4), "MAIVMGRWKGAR*");
    }

    /// NCBI table 2 differs from table 1 at four codons. All four are
    /// exercised here rather than only the one the example sequence
    /// happens to contain.
    #[test]
    fn table_2_differs_from_table_1_at_its_four_reassigned_codons() {
        // TGA: stop -> Trp.
        assert_eq!(tr("TGA", 1, 1), "*");
        assert_eq!(tr("TGA", 1, 2), "W");
        // ATA: Ile -> Met.
        assert_eq!(tr("ATA", 1, 1), "I");
        assert_eq!(tr("ATA", 1, 2), "M");
        // AGA/AGG: Arg -> stop.
        assert_eq!(tr("AGAAGG", 1, 1), "RR");
        assert_eq!(tr("AGAAGG", 1, 2), "**");
    }

    /// NCBI's table 11 `AAs` row is character-for-character table 1's; the
    /// two codes differ only in start codons, which this module does not
    /// apply. Pinned as a test so the claim in the module header cannot
    /// quietly stop being true.
    #[test]
    fn table_11_matches_table_1_because_only_its_start_codons_differ() {
        assert_eq!(NCBI_AAS_TABLE_11, NCBI_AAS_TABLE_1);
        let seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG";
        assert_eq!(tr(seq, 1, 11), tr(seq, 1, 1));
    }

    #[test]
    fn an_unsupported_table_id_is_rejected_and_lists_the_supported_ones() {
        match TranslationTable::from_id(5) {
            Err(FastDnaError::InvalidConfig { parameter, reason }) => {
                assert_eq!(parameter, "table");
                assert!(reason.contains('5'), "reason must name the bad id: {reason}");
                assert!(reason.contains("11"), "reason must list what is supported: {reason}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    // -- frames ---------------------------------------------------------

    #[test]
    fn frames_round_trip_through_their_conventional_spelling() {
        for frame_id in ALL_FRAMES {
            assert_eq!(Frame::from_i8(frame_id).unwrap().as_i8(), frame_id);
        }
    }

    #[test]
    fn zero_and_out_of_range_frames_are_rejected() {
        for bad in [0i8, 4, -4, 127, -128] {
            assert!(Frame::from_i8(bad).is_err(), "frame {bad} must be rejected");
        }
    }

    /// The offset frames start one and two bases in, so each drops the
    /// leading base(s) and one more trailing partial codon.
    #[test]
    fn forward_frames_2_and_3_start_one_and_two_bases_in() {
        // Frame +2 skips one base: AATGGCC reads ATG GCC from index 1.
        assert_eq!(tr("AATGGCC", 2, 1), "MA");
        // Frame +3 skips two: AAATGGCC reads ATG GCC from index 2.
        assert_eq!(tr("AAATGGCC", 3, 1), "MA");
        // ...and the same sequences in frame +1 do *not* read ATG GCC,
        // which is what makes the two assertions above about the offset
        // rather than about the sequence happening to translate that way.
        assert_eq!(tr("AATGGCC", 1, 1), "NG");
    }

    /// A reverse frame translates the reverse complement. Checked against
    /// an explicitly reverse-complemented sequence translated forwards, so
    /// the test does not merely restate the implementation.
    #[test]
    fn a_reverse_frame_equals_the_forward_frame_of_the_reverse_complement() {
        let seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG";
        let rc = reverse_complement_str(seq);
        for offset in 1..=3i8 {
            assert_eq!(
                tr(seq, -offset, 1),
                tr(&rc, offset, 1),
                "reverse frame -{offset} must equal forward frame +{offset} of the revcomp"
            );
        }
    }

    /// The headline six-frame property: translating a sequence and
    /// translating its reverse complement must yield the same six proteins,
    /// as a set. (Not in the same order -- the reverse complement swaps
    /// which strand each frame reads.)
    #[test]
    fn six_frame_translation_is_invariant_under_reverse_complement() {
        let seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTTACGGT";
        let rc = reverse_complement_str(seq);

        let mut forward: Vec<String> = translate_six_frames(seq.as_bytes(), standard(), StopHandling::Translate)
            .into_iter()
            .map(|(_, protein)| protein)
            .collect();
        let mut reverse: Vec<String> = translate_six_frames(rc.as_bytes(), standard(), StopHandling::Translate)
            .into_iter()
            .map(|(_, protein)| protein)
            .collect();
        forward.sort();
        reverse.sort();

        assert_eq!(forward, reverse);
    }

    #[test]
    fn six_frame_translation_reports_all_six_frames_in_order() {
        let frames: Vec<i8> =
            translate_six_frames(b"ATGGCCATTGTAATGGGCCGC", standard(), StopHandling::Translate)
                .into_iter()
                .map(|(frame_id, _)| frame_id)
                .collect();
        assert_eq!(frames, ALL_FRAMES.to_vec());
    }

    // -- ambiguity, stops, edges ----------------------------------------

    /// A codon containing `N` must become exactly one `X` and leave every
    /// residue after it where it was. Skipping the codon instead would give
    /// "MM" here -- shorter, and silently misaligned with the DNA.
    #[test]
    fn a_codon_containing_n_becomes_x_without_shifting_the_frame() {
        assert_eq!(tr("ATGNNNATG", 1, 1), "MXM");
        // The ambiguity need not fill the codon: one bad base spoils it.
        assert_eq!(tr("ATGANGATG", 1, 1), "MXM");
        assert_eq!(tr("ATGNCCATG", 1, 1), "MXM");
        // Any IUPAC ambiguity code, not just N.
        assert_eq!(tr("ATGRYKATG", 1, 1), "MXM");
    }

    #[test]
    fn ambiguity_on_the_reverse_strand_also_becomes_x_in_place() {
        // Reverse complement of ATGNNNATG is CATNNNCAT -> H X H.
        assert_eq!(tr("ATGNNNATG", -1, 1), "HXH");
    }

    #[test]
    fn to_stop_truncates_at_the_first_stop_and_excludes_it() {
        let seq = b"ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG";
        assert_eq!(
            translate(seq, Frame::from_i8(1).unwrap(), standard(), StopHandling::StopAtFirst),
            "MAIVMGR"
        );
        assert_eq!(
            translate(seq, Frame::from_i8(1).unwrap(), standard(), StopHandling::Translate),
            "MAIVMGR*KGAR*"
        );
    }

    #[test]
    fn a_sequence_starting_with_a_stop_translates_to_nothing_under_to_stop() {
        assert_eq!(
            translate(b"TAAATGATG", Frame::from_i8(1).unwrap(), standard(), StopHandling::StopAtFirst),
            ""
        );
    }

    #[test]
    fn an_empty_sequence_translates_to_an_empty_protein_in_every_frame() {
        for frame_id in ALL_FRAMES {
            assert_eq!(tr("", frame_id, 1), "");
        }
    }

    #[test]
    fn a_sequence_shorter_than_a_codon_translates_to_nothing() {
        for seq in ["A", "AT", "ATG"] {
            let expected = if seq.len() == 3 { "M" } else { "" };
            assert_eq!(tr(seq, 1, 1), expected, "sequence {seq:?}");
        }
        // Two bases cannot fill a codon in any frame.
        for frame_id in ALL_FRAMES {
            assert_eq!(tr("AT", frame_id, 1), "");
        }
    }

    /// Trailing bases that do not complete a codon are dropped, in both
    /// directions -- on the reverse strand the dropped bases are the ones at
    /// the 5' end of the original sequence.
    #[test]
    fn trailing_partial_codons_are_dropped() {
        assert_eq!(tr("ATGGCCA", 1, 1), "MA"); // 7 bases -> 2 codons, 1 dropped
        assert_eq!(tr("ATGGCCAT", 1, 1), "MA"); // 8 bases -> 2 codons, 2 dropped
        // Reverse: revcomp("ATGGCCA") is "TGGCCAT" -> TGG CCA = "WP", T dropped.
        assert_eq!(tr("ATGGCCA", -1, 1), "WP");
    }

    /// `kmer::base_to_bits` accepts lowercase, so soft-masked reference
    /// sequence must translate identically rather than becoming `XXX`.
    #[test]
    fn lowercase_input_translates_the_same_as_uppercase() {
        let seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG";
        assert_eq!(tr(&seq.to_lowercase(), 1, 1), tr(seq, 1, 1));
        assert_eq!(tr("aTgGcC", 1, 1), "MA");
    }

    /// `kmer::base_to_bits` maps `U` to the same bits as `T`, so an RNA
    /// sequence translates without being back-transcribed first.
    #[test]
    fn uracil_is_read_as_thymine() {
        assert_eq!(tr("AUGGCC", 1, 1), "MA");
    }

    // -- amino-acid k-mers ----------------------------------------------

    #[test]
    fn amino_acid_kmers_are_overlapping_windows_in_reading_order() {
        assert_eq!(amino_acid_kmers("MAIVM", 3).unwrap(), vec!["MAI", "AIV", "IVM"]);
    }

    /// Hand-checkable: "MAMAM" has windows MA, AM, MA, AM -> MA:2, AM:2.
    #[test]
    fn amino_acid_kmer_counts_are_hand_checkable_and_sorted() {
        assert_eq!(
            count_amino_acid_kmers("MAMAM", 2).unwrap(),
            vec![("AM".to_string(), 2), ("MA".to_string(), 2)]
        );
    }

    /// Unlike DNA k-mers, amino-acid k-mers are not canonicalized: a
    /// peptide read backwards is a different peptide, so "MA" and "AM" must
    /// stay distinct rather than collapsing onto one another.
    #[test]
    fn amino_acid_kmers_are_not_canonicalized() {
        let counts = count_amino_acid_kmers("MAAM", 2).unwrap();
        let kmers: Vec<&str> = counts.iter().map(|(kmer_str, _)| kmer_str.as_str()).collect();
        assert_eq!(kmers, vec!["AA", "AM", "MA"]);
    }

    #[test]
    fn a_k_larger_than_the_protein_yields_no_kmers() {
        assert!(amino_acid_kmers("MAI", 4).unwrap().is_empty());
        assert!(count_amino_acid_kmers("MAI", 4).unwrap().is_empty());
        assert!(amino_acid_kmers("", 1).unwrap().is_empty());
    }

    #[test]
    fn a_zero_k_is_rejected_rather_than_yielding_empty_strings() {
        match amino_acid_kmers("MAIVM", 0) {
            Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "k"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        assert!(count_amino_acid_kmers("MAIVM", 0).is_err());
    }

    #[test]
    fn stop_and_ambiguous_residues_are_kept_in_amino_acid_kmers() {
        assert_eq!(amino_acid_kmers("M*X", 2).unwrap(), vec!["M*", "*X"]);
    }

    #[test]
    fn a_non_ascii_protein_is_rejected_with_an_actionable_message() {
        match amino_acid_kmers("MAÏVM", 2) {
            Err(FastDnaError::InvalidConfig { parameter, reason }) => {
                assert_eq!(parameter, "protein");
                assert!(reason.contains("ASCII"), "reason: {reason}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    /// Test-local reverse complement, written the naive way on purpose: the
    /// point of the tests that use it is to check `translate`'s reverse
    /// frames against an *independent* implementation, so reusing the crate's
    /// own bit-trick version here would make those tests circular.
    fn reverse_complement_str(seq: &str) -> String {
        seq.chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'C' => 'G',
                'G' => 'C',
                'T' => 'A',
                other => other,
            })
            .collect()
    }
}
