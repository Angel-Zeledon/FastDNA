"""Tests for fastdna.translate -- DNA to protein translation.

The codon-level rules are pinned on the Rust side (`src/translate.rs`'s own
test module and `tests/translate.rs`). What this file covers is the Python
surface: the argument handling, the shapes of the returned tables, and the
two things a Python caller can get wrong in ways the Rust core never sees
-- passing a bare string where a list is expected, and assuming
`translate_file` agrees with `translate`.

Every expected protein in this file is either hand-checkable against the
NCBI codon table (cited at each use) or taken from a published source, not
from running the implementation and recording what it printed.
"""
from __future__ import annotations

import pytest

import fastdna
from fastdna.translate import (
    protein_kmers,
    translate,
    translate_file,
    translate_six_frames,
)

# The Biopython tutorial's own translation example ("Translation", Biopython
# Tutorial and Cookbook). Its published answer under the standard genetic
# code is `MAIVMGR*KGAR*`; hand-checkable against NCBI table 1 as
# ATG GCC ATT GTA ATG GGC CGC TGA AAG GGT GCC CGA TAG.
EXAMPLE = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
EXAMPLE_PROTEIN = "MAIVMGR*KGAR*"


def reverse_complement(seq):
    """Naive reverse complement, independent of the crate's own bit-trick
    implementation -- reusing the code under test would make the strand
    symmetry checks circular.
    """
    return "".join({"A": "T", "C": "G", "G": "C", "T": "A"}.get(c, c) for c in reversed(seq))


# -- translate ----------------------------------------------------------


def test_a_single_string_translates_to_a_single_string():
    """A bare `str` is one sequence, not an iterable of one-character
    sequences. See `translate`'s docstring for why this is handled
    explicitly rather than raising.
    """
    assert translate(EXAMPLE) == EXAMPLE_PROTEIN
    assert isinstance(translate(EXAMPLE), str)


def test_a_list_of_strings_translates_to_a_list_of_strings():
    result = translate([EXAMPLE, "ATGGCC"])
    assert result == [EXAMPLE_PROTEIN, "MA"]
    assert isinstance(result, list)


def test_a_one_element_list_still_returns_a_list_not_a_bare_string():
    """The return shape follows the *input* shape, not the element count --
    otherwise code that builds a list programmatically would get a string
    back whenever the list happened to have one entry.
    """
    assert translate([EXAMPLE]) == [EXAMPLE_PROTEIN]


def test_a_bare_string_is_never_iterated_per_character():
    """The regression this pins: `for s in "ACGT"` yields 'A','C','G','T',
    each of which is shorter than a codon, so a per-character iteration
    would silently return four empty proteins instead of one real one.
    This is the same footgun `fastdna.sklearn` hit with a bare path.
    """
    assert translate("ATGGCC") == "MA"
    assert translate("ATGGCC") != ["", "", "", "", "", ""]


def test_bytes_are_rejected_with_an_actionable_type_error():
    """`bytes` is iterable too, but iterating it yields ints, which would
    fail deep inside with something unreadable. Refused up front instead.
    """
    with pytest.raises(TypeError, match="bytes"):
        translate(b"ATGGCC")


def test_a_generator_of_sequences_is_accepted():
    """Anything iterable of strings works, including a one-shot iterator --
    the implementation must not consume it twice.
    """
    assert translate(iter([EXAMPLE, "ATGGCC"])) == [EXAMPLE_PROTEIN, "MA"]


def test_frames_select_the_reading_frame():
    # Frame +2 skips one base: AATGGCC reads ATG GCC.
    assert translate("AATGGCC", frame=2) == "MA"
    # Frame +3 skips two.
    assert translate("AAATGGCC", frame=3) == "MA"


def test_a_negative_frame_translates_the_reverse_complement():
    for offset in (1, 2, 3):
        assert translate(EXAMPLE, frame=-offset) == translate(
            reverse_complement(EXAMPLE), frame=offset
        )


def test_an_invalid_frame_raises_value_error():
    for bad in (0, 4, -4):
        with pytest.raises(ValueError, match="frame"):
            translate(EXAMPLE, frame=bad)


# -- translation tables -------------------------------------------------


def test_table_4_reads_tga_as_tryptophan_where_table_1_stops():
    """NCBI table 4 (Mold/Protozoan/Coelenterate Mitochondrial and
    Mycoplasma/Spiroplasma) reassigns TGA from stop to Trp and changes
    nothing else. Source: NCBI Taxonomy, "The Genetic Codes", table 4.
    """
    assert translate(EXAMPLE, table=1) == "MAIVMGR*KGAR*"
    assert translate(EXAMPLE, table=4) == "MAIVMGRWKGAR*"


def test_table_2_reads_aga_and_agg_as_stops_where_table_1_reads_arginine():
    """NCBI table 2 (Vertebrate Mitochondrial) reassigns AGA/AGG from Arg
    to stop, and ATA from Ile to Met. Source: NCBI Taxonomy, "The Genetic
    Codes", table 2. ATG AGA AGG TTT is hand-checkable: M R R F under table
    1, M * * F under table 2.
    """
    assert translate("ATGAGAAGGTTT", table=1) == "MRRF"
    assert translate("ATGAGAAGGTTT", table=2) == "M**F"
    # ATA: Ile in table 1, Met in table 2.
    assert translate("ATA", table=1) == "I"
    assert translate("ATA", table=2) == "M"


def test_table_11_matches_table_1_because_only_start_codons_differ():
    """NCBI's table 11 (Bacterial/Archaeal/Plant Plastid) has an `AAs` row
    identical to table 1's -- the codes differ only in which codons may
    *initiate* translation, which plain frame translation does not apply.
    Documented in `fastdna.translate`'s docstring; pinned here so the claim
    cannot quietly stop being true.
    """
    assert translate(EXAMPLE, table=11) == translate(EXAMPLE, table=1)


def test_an_unsupported_table_raises_value_error_listing_what_is_supported():
    with pytest.raises(ValueError) as excinfo:
        translate(EXAMPLE, table=5)
    message = str(excinfo.value)
    assert "5" in message
    assert "11" in message, f"the error must list the supported ids: {message}"


# -- stops, ambiguity, edges -------------------------------------------


def test_to_stop_truncates_at_the_first_stop_and_translating_through_does_not():
    assert translate(EXAMPLE, to_stop=True) == "MAIVMGR"
    assert translate(EXAMPLE, to_stop=False) == "MAIVMGR*KGAR*"
    # The default is to translate through -- see the module docstring.
    assert translate(EXAMPLE) == translate(EXAMPLE, to_stop=False)


def test_a_codon_containing_n_becomes_x_and_does_not_shift_the_frame():
    """One `X` per bad codon, in place. A skipped codon would give "MM"
    here -- shorter, and misaligned with the DNA from that point on.
    """
    assert translate("ATGNNNATG") == "MXM"
    # A single bad base spoils only its own codon.
    assert translate("ATGANGATG") == "MXM"
    # Same on the reverse strand: revcomp("ATGNNNATG") is CATNNNCAT -> HXH.
    assert translate("ATGNNNATG", frame=-1) == "HXH"


def test_an_empty_sequence_translates_to_an_empty_protein():
    assert translate("") == ""
    assert translate(["", ""]) == ["", ""]


def test_a_sequence_shorter_than_three_bases_translates_to_nothing():
    assert translate("A") == ""
    assert translate("AT") == ""
    assert translate("ATG") == "M"


def test_lowercase_input_translates_the_same_as_uppercase():
    assert translate(EXAMPLE.lower()) == EXAMPLE_PROTEIN
    assert translate("atgncc") == "MX"


def test_trailing_bases_that_do_not_complete_a_codon_are_dropped():
    assert translate("ATGGCCA") == "MA"
    assert translate("ATGGCCAT") == "MA"


# -- six frames ---------------------------------------------------------


def test_six_frame_translation_of_a_single_string_returns_one_dict():
    result = translate_six_frames(EXAMPLE)
    assert isinstance(result, dict)
    assert sorted(result) == [-3, -2, -1, 1, 2, 3]
    assert result[1] == EXAMPLE_PROTEIN


def test_six_frame_translation_of_a_list_returns_one_dict_per_sequence():
    result = translate_six_frames([EXAMPLE, "ATGGCC"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0][1] == EXAMPLE_PROTEIN
    assert result[1][1] == "MA"


def test_six_frames_of_a_sequence_and_its_reverse_complement_are_the_same_set():
    """The property that makes a six-frame scan independent of which strand
    a contig was assembled on: the reverse complement swaps which strand
    each frame reads, so the frame labels change but the six proteins do
    not.
    """
    forward = translate_six_frames(EXAMPLE)
    reverse = translate_six_frames(reverse_complement(EXAMPLE))

    assert sorted(forward.values()) == sorted(reverse.values())
    # Guards the degenerate pass: six empty strings also compare equal.
    assert any(forward.values())


def test_six_frame_translation_honours_the_table_and_to_stop_options():
    assert translate_six_frames(EXAMPLE, table=4)[1] == "MAIVMGRWKGAR*"
    assert translate_six_frames(EXAMPLE, to_stop=True)[1] == "MAIVMGR"


def test_six_frame_translation_rejects_bytes_like_translate_does():
    with pytest.raises(TypeError, match="bytes"):
        translate_six_frames(b"ATGGCC")


# -- translate_file -----------------------------------------------------


@pytest.fixture
def fasta_path(tmp_path):
    """A small FASTA file with a described header, a bare header, and a
    wrapped (multi-line) sequence -- the three shapes a real FASTA has.
    """
    path = tmp_path / "genes.fasta"
    # newline="\n" pins LF endings, as the rest of this package does when it
    # writes FASTA/FASTQ (see `fastdna.interop`): the reader strips CRLF
    # fine, but the fixture should be the same bytes on every platform.
    with open(path, "w", newline="\n") as handle:
        handle.write(
            f">gene1 a described gene\n{EXAMPLE}\n"
            ">gene2\nATGAGAAGGTTT\n"
            ">wrapped\nATGGCC\nATTGTA\n"
        )
    return path


def test_translate_file_returns_the_documented_schema(fasta_path):
    table = translate_file(fasta_path, frames=(1,))
    assert table.column_names == ["sequence_id", "frame", "protein"]
    assert table.num_rows == 3
    assert table.column("sequence_id").to_pylist() == ["gene1", "gene2", "wrapped"]
    assert table.column("frame").to_pylist() == [1, 1, 1]


def test_translate_file_uses_the_accession_not_the_whole_header(fasta_path):
    """`>gene1 a described gene` must key as `gene1`: the accession, the
    way BLAST and SAM define it. Keeping the description would make
    `sequence_id` unjoinable against anything else the user has.
    """
    ids = translate_file(fasta_path, frames=(1,)).column("sequence_id").to_pylist()
    assert "gene1" in ids
    assert not any(" " in seq_id for seq_id in ids)


def test_translate_file_matches_translating_the_same_sequences_directly(fasta_path):
    """The round-trip that makes `translate_file` trustworthy: streaming a
    file through the Rust reader must give exactly what translating its
    sequences in memory gives. A wrapped FASTA record is the interesting
    case -- its lines must be joined *before* translation, not translated
    per line.
    """
    sequences = [EXAMPLE, "ATGAGAAGGTTT", "ATGGCCATTGTA"]
    frames = (1, 2, 3, -1, -2, -3)

    from_file = translate_file(fasta_path, frames=frames)
    direct = [
        translate_six_frames(seq)[frame] for seq in sequences for frame in frames
    ]

    assert from_file.column("protein").to_pylist() == direct


def test_translate_file_emits_every_requested_frame_in_order(fasta_path):
    table = translate_file(fasta_path, frames=(1, -1))
    assert table.column("frame").to_pylist() == [1, -1, 1, -1, 1, -1]
    assert table.num_rows == 6


def test_translate_file_defaults_to_all_six_frames(fasta_path):
    table = translate_file(fasta_path)
    assert table.num_rows == 3 * 6
    assert set(table.column("frame").to_pylist()) == {1, 2, 3, -1, -2, -3}


def test_translate_file_reads_fastq_too(tmp_path):
    """The reader detects FASTA vs FASTQ by content, so the same function
    handles both without a format argument.
    """
    path = tmp_path / "reads.fastq"
    with open(path, "w", newline="\n") as handle:
        handle.write(f"@read1 desc\n{EXAMPLE}\n+\n{'I' * len(EXAMPLE)}\n")

    table = translate_file(path, frames=(1,))
    assert table.column("sequence_id").to_pylist() == ["read1"]
    assert table.column("protein").to_pylist() == [EXAMPLE_PROTEIN]


def test_translate_file_on_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        translate_file(tmp_path / "nope.fasta")


def test_translate_file_rejects_an_empty_frame_list(fasta_path):
    with pytest.raises(ValueError, match="frame"):
        translate_file(fasta_path, frames=())


# -- protein_kmers ------------------------------------------------------


def test_protein_kmers_counts_are_hand_checkable():
    """"MAMAM" has windows MA, AM, MA, AM -> MA:2, AM:2."""
    table = protein_kmers("MAMAM", k=2)
    assert table.column_names == ["sequence_id", "aa_kmer", "count"]
    assert dict(zip(table.column("aa_kmer").to_pylist(), table.column("count").to_pylist())) == {
        "MA": 2,
        "AM": 2,
    }


def test_protein_kmers_accepts_a_bare_string_without_iterating_it():
    """As with `translate`: a bare string is one protein. Iterating it per
    character would produce one row per residue with k-mers of length 1.
    """
    table = protein_kmers("MAIVM", k=3)
    assert table.column("aa_kmer").to_pylist() == ["AIV", "IVM", "MAI"]
    assert set(table.column("sequence_id").to_pylist()) == {"protein0"}


def test_protein_kmers_rejects_bytes():
    with pytest.raises(TypeError, match="bytes"):
        protein_kmers(b"MAIVM", k=3)


def test_protein_kmers_keeps_proteins_separate():
    table = protein_kmers(["MAM", "MAM"], k=2)
    assert table.column("sequence_id").to_pylist() == [
        "protein0",
        "protein0",
        "protein1",
        "protein1",
    ]
    assert table.column("count").to_pylist() == [1, 1, 1, 1]


def test_protein_kmers_are_not_canonical():
    """Unlike DNA k-mers, `MA` and `AM` are different peptides and must not
    collapse onto one another -- a protein has no reverse complement.
    """
    kmers = protein_kmers("MAAM", k=2).column("aa_kmer").to_pylist()
    assert sorted(kmers) == ["AA", "AM", "MA"]


def test_a_k_larger_than_the_protein_yields_no_rows():
    assert protein_kmers("MAI", k=4).num_rows == 0


def test_k_zero_is_rejected():
    with pytest.raises(ValueError, match="k"):
        protein_kmers("MAIVM", k=0)


def test_protein_kmers_composes_with_translate():
    """The intended pipeline: translate, then k-merize the protein."""
    protein = translate(EXAMPLE, to_stop=True)
    assert protein == "MAIVMGR"
    table = protein_kmers(protein, k=3)
    assert table.num_rows == len(protein) - 3 + 1


# -- package surface ----------------------------------------------------


def test_the_module_is_reached_as_a_submodule_not_a_package_root_export():
    """`fastdna.translate` is the *module*, deliberately.

    Re-exporting the `translate()` function from `fastdna/__init__.py`
    would bind the name `fastdna.translate` to the function and shadow the
    module of the same name -- so `import fastdna.translate` and
    `fastdna.translate(seq)` would mean different things depending on
    import order. The package already keeps its optional surfaces
    (`fastdna.interop`, `fastdna.taxonomy`, `fastdna.sklearn`, ...) as
    submodules that `__init__.py` does not import, and this follows that
    convention rather than making an exception that collides.
    """
    import fastdna.translate as translate_module

    assert fastdna.translate is translate_module
    assert translate_module.translate(EXAMPLE) == EXAMPLE_PROTEIN
    assert not hasattr(fastdna, "translate_six_frames"), (
        "the package root must stay free of these names -- see this test's docstring"
    )
