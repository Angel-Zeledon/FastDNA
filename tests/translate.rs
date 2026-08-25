//! DNA -> protein translation, exercised through the crate's public API.
//!
//! The per-function rules live in `src/translate.rs`'s own test module.
//! What this file pins down is the behaviour a *user* of the library sees:
//! that the public surface is reachable and coherent from outside the
//! crate, that translation composes with the FASTA/FASTQ reader the same
//! way the counting pipeline does, and that the properties which would
//! silently produce wrong biology (frame alignment, strand symmetry) hold
//! on real multi-record input rather than only on hand-picked codons.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;

use fastdna_core::fastq::FastqReader;
use fastdna_core::translate::{
    amino_acid_kmers, count_amino_acid_kmers, translate, translate_six_frames, Frame, StopHandling,
    TranslationTable, ALL_FRAMES,
};

fn table(id: u8) -> &'static TranslationTable {
    TranslationTable::from_id(id).expect("a supported NCBI table id")
}

fn tr(seq: &str, frame: i8, table_id: u8) -> String {
    translate(
        seq.as_bytes(),
        Frame::from_i8(frame).expect("a valid frame"),
        table(table_id),
        StopHandling::Translate,
    )
}

/// The Biopython tutorial's own translation example ("Translation", Biopython
/// Tutorial and Cookbook), whose published answer under the standard code is
/// `MAIVMGR*KGAR*`. Hand-checkable against NCBI table 1:
/// ATG GCC ATT GTA ATG GGC CGC TGA AAG GGT GCC CGA TAG.
const BIOPYTHON_EXAMPLE: &str = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG";
const BIOPYTHON_EXAMPLE_PROTEIN: &str = "MAIVMGR*KGAR*";

/// Naive reverse complement, written independently of the crate's own
/// bit-trick implementation on purpose: tests that compare a reverse frame
/// against it would be circular if they reused the code under test.
fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' => 'T',
            'C' => 'G',
            'G' => 'C',
            'T' => 'A',
            'a' => 't',
            'c' => 'g',
            'g' => 'c',
            't' => 'a',
            other => other,
        })
        .collect()
}

#[test]
fn the_published_standard_code_answer_is_reproduced_through_the_public_api() {
    assert_eq!(tr(BIOPYTHON_EXAMPLE, 1, 1), BIOPYTHON_EXAMPLE_PROTEIN);
}

/// The one difference between NCBI tables 1 and 4 is `TGA`: a stop in the
/// standard code, tryptophan in the mold/protozoan mitochondrial and
/// Mycoplasma code. On this sequence that changes exactly one residue.
#[test]
fn table_4_and_table_1_differ_only_where_ncbi_says_they_do() {
    let standard = tr(BIOPYTHON_EXAMPLE, 1, 1);
    let mold = tr(BIOPYTHON_EXAMPLE, 1, 4);

    assert_eq!(mold, "MAIVMGRWKGAR*");
    let differing: Vec<usize> = standard
        .bytes()
        .zip(mold.bytes())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(differing, vec![7], "only the TGA codon at residue 7 may differ");
}

/// Table 2 (Vertebrate Mitochondrial) reassigns `AGA`/`AGG` from arginine
/// to stop -- the difference that matters most in practice, because it
/// changes where a mitochondrial ORF ends.
#[test]
fn table_2_reads_aga_and_agg_as_stops() {
    assert_eq!(tr("ATGAGAAGGTTT", 1, 1), "MRRF");
    assert_eq!(tr("ATGAGAAGGTTT", 1, 2), "M**F");

    // ...and with `to_stop`, that reassignment truncates the product.
    let seq = b"ATGAGAAGGTTT";
    assert_eq!(
        translate(seq, Frame::from_i8(1).unwrap(), table(2), StopHandling::StopAtFirst),
        "M"
    );
    assert_eq!(
        translate(seq, Frame::from_i8(1).unwrap(), table(1), StopHandling::StopAtFirst),
        "MRRF"
    );
}

/// Six-frame translation of a sequence and of its reverse complement must
/// yield the same six proteins as a *set*: the reverse complement swaps
/// which strand each frame reads, so the order changes but the content
/// cannot. This is the property that makes a six-frame scan independent of
/// which strand a contig happened to be assembled on.
#[test]
fn six_frames_of_a_sequence_and_its_reverse_complement_are_the_same_set() {
    let seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTTACGGTCATNNNGGATCC";
    let rc = reverse_complement(seq);

    let mut forward: Vec<String> =
        translate_six_frames(seq.as_bytes(), table(1), StopHandling::Translate)
            .into_iter()
            .map(|(_, protein)| protein)
            .collect();
    let mut reverse: Vec<String> =
        translate_six_frames(rc.as_bytes(), table(1), StopHandling::Translate)
            .into_iter()
            .map(|(_, protein)| protein)
            .collect();

    assert_eq!(forward.len(), 6);
    forward.sort();
    reverse.sort();
    assert_eq!(forward, reverse);
    // Guards against the degenerate way this test could pass: six empty
    // strings compare equal to six empty strings.
    assert!(forward.iter().any(|p| !p.is_empty()), "the fixture must actually translate");
}

/// Every reverse frame must equal the corresponding forward frame of an
/// independently reverse-complemented sequence. This is the direct check on
/// the "complement each codon as it is packed instead of materializing a
/// reverse-complemented copy" implementation choice.
#[test]
fn reverse_frames_agree_with_translating_an_explicit_reverse_complement() {
    for seq in [BIOPYTHON_EXAMPLE, "ACGTACGTAC", "TTTTTTTTTTTT", "ATGNNNATGNNN", "AC"] {
        let rc = reverse_complement(seq);
        for offset in 1..=3i8 {
            assert_eq!(
                tr(seq, -offset, 1),
                tr(&rc, offset, 1),
                "frame -{offset} of {seq:?} must equal frame +{offset} of its reverse complement"
            );
        }
    }
}

/// An ambiguous codon becomes exactly one `X` and every residue after it
/// keeps its position. Checked as an alignment property -- same length,
/// same residues either side -- rather than only as a literal string, since
/// the failure mode being guarded against is a silent *shift*.
#[test]
fn an_ambiguous_codon_costs_one_residue_and_shifts_nothing() {
    let clean = "ATGGCCATTGTAATGGGCCGC";
    let spoiled = "ATGGCCATTGTANTGGGCCGC"; // the 5th codon's first base -> N

    let clean_protein = tr(clean, 1, 1);
    let spoiled_protein = tr(spoiled, 1, 1);

    assert_eq!(clean_protein, "MAIVMGR");
    assert_eq!(spoiled_protein.len(), clean_protein.len(), "the protein must not shorten");
    assert_eq!(&spoiled_protein[4..5], "X");
    assert_eq!(&spoiled_protein[..4], &clean_protein[..4], "residues before must be unchanged");
    assert_eq!(&spoiled_protein[5..], &clean_protein[5..], "residues after must not shift");
}

#[test]
fn empty_short_and_lowercase_inputs_behave_as_documented() {
    for frame_id in ALL_FRAMES {
        assert_eq!(tr("", frame_id, 1), "", "empty input, frame {frame_id}");
        assert_eq!(tr("AT", frame_id, 1), "", "two bases cannot fill a codon");
    }
    assert_eq!(tr("ATG", 1, 1), "M");
    assert_eq!(
        tr(&BIOPYTHON_EXAMPLE.to_lowercase(), 1, 1),
        BIOPYTHON_EXAMPLE_PROTEIN,
        "soft-masked (lowercase) sequence must translate identically"
    );
}

/// Translation composes with the reader the counting pipeline uses: a FASTA
/// file's records translate to exactly what translating their sequences
/// directly gives. This is the Rust-side counterpart of the Python
/// `translate_file` round-trip test.
#[test]
fn records_read_from_fasta_translate_to_the_same_proteins_as_their_raw_sequences() {
    let fasta = format!(
        ">gene1 first\n{}\n>gene2\nATGAGAAGGTTT\n>wrapped\nATGGCC\nATTGTA\n",
        BIOPYTHON_EXAMPLE
    );
    let mut reader = FastqReader::new(Cursor::new(fasta.into_bytes()));

    let mut seen: Vec<(String, String)> = Vec::new();
    while let Some(record) = reader.next_record().expect("valid FASTA") {
        let id = String::from_utf8(record.id.clone()).expect("ASCII header");
        let protein = translate(
            &record.seq,
            Frame::from_i8(1).unwrap(),
            table(1),
            StopHandling::Translate,
        );
        // The record's sequence, translated on its own, must agree.
        let direct = tr(std::str::from_utf8(&record.seq).unwrap(), 1, 1);
        assert_eq!(protein, direct);
        seen.push((id, protein));
    }

    assert_eq!(
        seen,
        vec![
            (">gene1 first".to_string(), BIOPYTHON_EXAMPLE_PROTEIN.to_string()),
            (">gene2".to_string(), "MRRF".to_string()),
            // Wrapped FASTA lines are joined by the reader before
            // translation, so this is one 12-base sequence, not two 6-base
            // ones -- which would translate to "MA" + "IV" only by luck of
            // both fragments being codon-aligned.
            (">wrapped".to_string(), "MAIV".to_string()),
        ]
    );
}

/// A FASTQ record and a FASTA record holding the same bases must translate
/// identically -- the reader's synthetic Q40 quality for FASTA is not
/// allowed to leak into the protein.
#[test]
fn fasta_and_fastq_records_with_the_same_bases_translate_identically() {
    let fasta = format!(">g\n{BIOPYTHON_EXAMPLE}\n");
    let fastq = format!("@g\n{}\n+\n{}\n", BIOPYTHON_EXAMPLE, "I".repeat(BIOPYTHON_EXAMPLE.len()));

    let proteins: Vec<String> = [fasta, fastq]
        .into_iter()
        .map(|text| {
            let mut reader = FastqReader::new(Cursor::new(text.into_bytes()));
            let record = reader.next_record().expect("valid input").expect("one record");
            translate(&record.seq, Frame::from_i8(1).unwrap(), table(1), StopHandling::Translate)
        })
        .collect();

    assert_eq!(proteins[0], proteins[1]);
    assert_eq!(proteins[0], BIOPYTHON_EXAMPLE_PROTEIN);
}

// -- amino-acid k-mers --------------------------------------------------

#[test]
fn amino_acid_kmers_of_a_translated_protein_are_hand_checkable() {
    let protein = tr(BIOPYTHON_EXAMPLE, 1, 1);
    assert_eq!(protein, "MAIVMGR*KGAR*");

    let kmers = amino_acid_kmers(&protein, 3).expect("k=3 over a 13-residue protein");
    assert_eq!(kmers.len(), protein.len() - 3 + 1);
    assert_eq!(kmers[0], "MAI");
    assert_eq!(kmers[kmers.len() - 1], "AR*");

    // "MAIVMGR*KGAR*" contains no repeated 3-mer, so every count is 1.
    let counts = count_amino_acid_kmers(&protein, 3).expect("k=3");
    assert_eq!(counts.len(), kmers.len());
    assert!(counts.iter().all(|(_, count)| *count == 1));
}

/// Amino-acid k-mers are not canonical (a peptide has no reverse
/// complement), so a protein and its reversal must give *different* k-mer
/// sets -- the exact opposite of the DNA k-mer invariant. Pinned here
/// because a future "optimization" that reused `kmer.rs`'s canonical path
/// would silently collapse these together.
#[test]
fn amino_acid_kmers_are_direction_sensitive_unlike_dna_kmers() {
    let forward = count_amino_acid_kmers("MAIVMGR", 3).expect("k=3");
    let reversed: String = "MAIVMGR".chars().rev().collect();
    let backward = count_amino_acid_kmers(&reversed, 3).expect("k=3");

    assert_ne!(forward, backward);
}

#[test]
fn amino_acid_kmer_edge_cases_are_rejected_or_empty_as_documented() {
    assert!(amino_acid_kmers("MAI", 4).expect("k > len is not an error").is_empty());
    assert!(amino_acid_kmers("MAIVM", 0).is_err(), "k = 0 must be rejected");
    assert!(count_amino_acid_kmers("MAIVM", 0).is_err(), "k = 0 must be rejected");
}

#[test]
fn an_unsupported_translation_table_is_rejected_rather_than_silently_substituted() {
    for bad in [0u8, 3, 5, 33, 255] {
        assert!(TranslationTable::from_id(bad).is_err(), "table {bad} must be rejected");
    }
    for good in [1u8, 2, 4, 11] {
        assert_eq!(TranslationTable::from_id(good).expect("supported").id, good);
    }
}
