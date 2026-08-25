"""Tests for `fastdna.annotate` -- mapping a rule's causal k-mer back to the
annotated gene/feature it falls inside, in a user-supplied reference genome
plus its GFF3 or GenBank annotation.

Fixtures below build one small synthetic reference (`FULL_SEQUENCE`, 155 bp,
`seqid="chr1"`) with two annotated genes (`geneA` on the `+` strand,
`geneB` on the `-` strand) and several deliberately placed k-mers of
`_K = 12` bases:

- `SINGLE_GENE_KMER` occurs exactly once, inside `geneA`, on the `+` strand.
- `REPEAT_MOTIF` occurs twice: once inside `geneA`, once in an intergenic
  region -- the "same k-mer, two different answers" case.
- `RC_ONLY_KMER`'s reverse complement (not its literal sequence) occurs once,
  in an intergenic region -- the "only present as reverse complement of the
  reference strand" case.
- `ZERO_HIT_KMER` occurs nowhere, in either orientation.
- `NEAR_MISS_KMER` is `REPEAT_MOTIF` with one base substituted: absent under
  exact search, present under `max_mismatches=1` (at both of `REPEAT_MOTIF`'s
  real locations, since the same substituted position mismatches both
  copies identically).
- `INDEL_ONLY_KMER` is `REPEAT_MOTIF` with one base inserted and the last
  base dropped (same length, frame-shifted by one) -- the minimum Hamming
  distance from this sequence to *any* 12-mer window of the reference is 4,
  so it is not found even at `max_mismatches=2`; an indel-tolerant search
  would find it near-instantly, which is exactly the capability this module
  deliberately does not have (see `annotate.py`'s module docstring).

All of the above were computed once with a small script (not by hand) and
pinned here as literal strings so the test file has no runtime dependency on
how they were derived.
"""
from __future__ import annotations

import pathlib

import pytest

import pyarrow as pa

from fastdna.annotate import Annotation, Hit, annotate_rule, export_bed, load_annotation, locate_kmer

_K = 12

FULL_SEQUENCE = (
    "AAGCCCAATAAACCACTCTGGGGATATAGACTGGCCGAATAGCAACGACATGTGCGGCGACCCTTGCGAC"
    "AGTGACGCTTTCGCCGTTGCCTAAACCTACTGGCCGAATAATTTGAACCGCAGTACTGCTAGACTCCAGG"
    "CACAATACCTCGTCC"
)
assert len(FULL_SEQUENCE) == 155

GENE_A_START, GENE_A_END = 21, 50
GENE_B_START, GENE_B_END = 66, 95

SINGLE_GENE_KMER = "GGGATATAGACT"  # occurs once, at 21..32, inside geneA, '+'
REPEAT_MOTIF = "ACTGGCCGAATA"  # occurs at 30..41 (in geneA) and 99..110 (intergenic)
RC_ONLY_KMER = "GGAGTCTAGCAG"  # its reverse complement occurs at 126..137 (intergenic)
ZERO_HIT_KMER = "GTGTTACCAGAC"  # occurs nowhere, either orientation
NEAR_MISS_KMER = "ACTGGGCGAATA"  # REPEAT_MOTIF with a single substitution
INDEL_ONLY_KMER = "ACTGGGCCGAAT"  # REPEAT_MOTIF, one base inserted + one dropped

GFF3_TEXT = """##gff-version 3
chr1\ttest\tgene\t21\t50\t.\t+\t.\tID=geneA;Name=geneA
chr1\ttest\tgene\t66\t95\t.\t-\t.\tID=geneB;Name=geneB
"""


def write_fasta(tmp_path: pathlib.Path, seqid: str, sequence: str) -> pathlib.Path:
    p = tmp_path / "reference.fasta"
    p.write_text(f">{seqid} synthetic test reference\n{sequence}\n")
    return p


def write_gff3(tmp_path: pathlib.Path, text: str = GFF3_TEXT) -> pathlib.Path:
    p = tmp_path / "annotation.gff3"
    p.write_text(text)
    return p


@pytest.fixture
def reference_and_gff3(tmp_path):
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    gff3_path = write_gff3(tmp_path)
    return fasta_path, gff3_path


# ---------------------------------------------------------------------------
# load_annotation()
# ---------------------------------------------------------------------------


def test_load_annotation_from_gff3_returns_an_annotation(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3

    annotation = load_annotation(fasta_path, gff3_path)

    assert isinstance(annotation, Annotation)
    assert annotation.sequences == {"chr1": FULL_SEQUENCE}
    feature_names = sorted(f.name for f in annotation.features)
    assert feature_names == ["geneA", "geneB"]


# ---------------------------------------------------------------------------
# locate_kmer() -- single hit inside one annotated gene
# ---------------------------------------------------------------------------


def test_locate_kmer_finds_single_hit_inside_a_gene(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    hits = locate_kmer(annotation, SINGLE_GENE_KMER)

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, Hit)
    assert hit.seqid == "chr1"
    assert hit.start == 21
    assert hit.end == 32
    assert hit.strand == "+"
    assert hit.mismatches == 0
    assert hit.feature_type == "gene"
    assert hit.gene_name == "geneA"
    assert hit.feature_start == GENE_A_START
    assert hit.feature_end == GENE_A_END
    assert hit.feature_strand == "+"


# ---------------------------------------------------------------------------
# locate_kmer() -- zero hits
# ---------------------------------------------------------------------------


def test_locate_kmer_returns_empty_list_for_a_kmer_absent_from_the_reference(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    hits = locate_kmer(annotation, ZERO_HIT_KMER)

    assert hits == []


# ---------------------------------------------------------------------------
# locate_kmer() -- multiple hits (repeat region: once inside a gene, once
# intergenic)
# ---------------------------------------------------------------------------


def test_locate_kmer_returns_every_occurrence_of_a_repeated_kmer(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    hits = locate_kmer(annotation, REPEAT_MOTIF)

    assert len(hits) == 2
    by_start = {h.start: h for h in hits}
    assert set(by_start) == {30, 99}

    in_gene = by_start[30]
    assert in_gene.end == 41
    assert in_gene.strand == "+"
    assert in_gene.mismatches == 0
    assert in_gene.feature_type == "gene"
    assert in_gene.gene_name == "geneA"

    intergenic = by_start[99]
    assert intergenic.end == 110
    assert intergenic.strand == "+"
    assert intergenic.mismatches == 0
    assert intergenic.feature_type == "intergenic"
    assert intergenic.gene_name is None
    assert intergenic.feature_start is None
    assert intergenic.feature_end is None
    assert intergenic.feature_strand is None


# ---------------------------------------------------------------------------
# locate_kmer() -- intergenic hit, reported not dropped
# ---------------------------------------------------------------------------


def test_locate_kmer_reports_an_intergenic_hit_explicitly(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    # The second REPEAT_MOTIF occurrence (position 99) falls between geneB
    # (ends at 95) and the RC_ONLY_KMER region (starts at 126) -- nowhere
    # near an annotated feature.
    hits = locate_kmer(annotation, REPEAT_MOTIF)
    intergenic_hits = [h for h in hits if h.start == 99]

    assert len(intergenic_hits) == 1
    hit = intergenic_hits[0]
    assert hit.feature_type == "intergenic"
    assert hit.gene_name is None


# ---------------------------------------------------------------------------
# locate_kmer() -- reverse-complement-only hit
# ---------------------------------------------------------------------------


def test_locate_kmer_finds_a_kmer_only_present_as_the_reverse_complement(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    hits = locate_kmer(annotation, RC_ONLY_KMER)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.start == 126
    assert hit.end == 137
    assert hit.strand == "-"
    assert hit.mismatches == 0
    # Nothing is annotated out there -- still a real, reported answer.
    assert hit.feature_type == "intergenic"


def test_locate_kmer_reports_both_strands_distinctly_for_a_palindromic_kmer(reference_and_gff3):
    # A 12-base palindrome: its own reverse complement equals itself, so
    # every occurrence must be reported once per orientation, not collapsed
    # into a single hit.
    palindrome = "ACGTACGTACGT"  # rev-comp: complement each base then reverse
    # Sanity-check the fixture claim about this literal string before
    # trusting the test built on it.
    assert palindrome == palindrome.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    fasta_path = write_fasta(reference_and_gff3[0].parent, "chrP", "GG" + palindrome + "GG")
    gff3_path = reference_and_gff3[1]
    # chrP is not covered by the geneA/geneB GFF3 fixture, and the
    # seqid-mismatch guard in load_annotation() only fires when *no*
    # annotation seqid matches *any* FASTA seqid -- chr1 is absent from
    # this FASTA, so build a minimal empty-feature annotation directly.
    annotation = Annotation({"chrP": "GG" + palindrome + "GG"}, [])

    hits = locate_kmer(annotation, palindrome)

    assert len(hits) == 2
    strands = sorted(h.strand for h in hits)
    assert strands == ["+", "-"]
    assert all(h.start == 3 and h.end == 14 for h in hits)


# ---------------------------------------------------------------------------
# locate_kmer() -- max_mismatches
# ---------------------------------------------------------------------------


def test_exact_search_misses_a_single_base_substitution(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    assert locate_kmer(annotation, NEAR_MISS_KMER, max_mismatches=0) == []


def test_max_mismatches_finds_a_near_match_exact_search_would_miss(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    hits = locate_kmer(annotation, NEAR_MISS_KMER, max_mismatches=1)

    # Both real REPEAT_MOTIF locations are one substitution away from
    # NEAR_MISS_KMER, since the mutated position is the same relative
    # offset in both copies.
    assert len(hits) == 2
    assert {h.start for h in hits} == {30, 99}
    assert all(h.mismatches == 1 for h in hits)
    assert all(h.strand == "+" for h in hits)


def test_max_mismatches_does_not_find_a_match_that_requires_an_indel(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    # INDEL_ONLY_KMER is REPEAT_MOTIF with one base inserted and the last
    # base dropped -- a single-base indel away from a real, exact match.
    # Substitution-only search must not find it even at a mismatch budget
    # (2) well above what any real substitution-based near-match in this
    # fixture needs (1, per the test above): the minimum Hamming distance
    # from INDEL_ONLY_KMER to any 12-mer window of the reference is 4.
    assert locate_kmer(annotation, INDEL_ONLY_KMER, max_mismatches=2) == []


def test_max_mismatches_rejects_a_negative_value(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    with pytest.raises(ValueError):
        locate_kmer(annotation, SINGLE_GENE_KMER, max_mismatches=-1)


# ---------------------------------------------------------------------------
# load_annotation() -- GenBank input
# ---------------------------------------------------------------------------


def write_genbank(tmp_path: pathlib.Path) -> pathlib.Path:
    Bio = pytest.importorskip("Bio")  # noqa: F841
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    record = SeqRecord(Seq(FULL_SEQUENCE), id="chr1", name="chr1", description="synthetic test reference")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(GENE_A_START - 1, GENE_A_END, strand=1),
            type="gene",
            qualifiers={"gene": ["geneA"]},
        ),
        SeqFeature(
            FeatureLocation(GENE_B_START - 1, GENE_B_END, strand=-1),
            type="gene",
            qualifiers={"gene": ["geneB"]},
        ),
    ]
    path = tmp_path / "annotation.gbk"
    SeqIO.write(record, str(path), "genbank")
    return path


def test_load_annotation_from_genbank_returns_the_same_features_as_gff3(tmp_path):
    pytest.importorskip("Bio")
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    genbank_path = write_genbank(tmp_path)

    annotation = load_annotation(fasta_path, genbank_path)

    feature_names = sorted(f.name for f in annotation.features)
    assert feature_names == ["geneA", "geneB"]
    by_name = {f.name: f for f in annotation.features}
    assert by_name["geneA"].start == GENE_A_START
    assert by_name["geneA"].end == GENE_A_END
    assert by_name["geneA"].strand == "+"
    assert by_name["geneB"].start == GENE_B_START
    assert by_name["geneB"].end == GENE_B_END
    assert by_name["geneB"].strand == "-"


def test_locate_kmer_works_identically_against_a_genbank_loaded_annotation(tmp_path):
    pytest.importorskip("Bio")
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    genbank_path = write_genbank(tmp_path)
    annotation = load_annotation(fasta_path, genbank_path)

    hits = locate_kmer(annotation, SINGLE_GENE_KMER)

    assert len(hits) == 1
    assert hits[0].gene_name == "geneA"
    assert hits[0].feature_type == "gene"


# ---------------------------------------------------------------------------
# load_annotation() -- malformed / missing files
# ---------------------------------------------------------------------------


def test_load_annotation_raises_actionably_on_missing_fasta(tmp_path):
    gff3_path = write_gff3(tmp_path)
    with pytest.raises(FileNotFoundError, match="reference FASTA"):
        load_annotation(tmp_path / "does_not_exist.fasta", gff3_path)


def test_load_annotation_raises_actionably_on_missing_annotation(tmp_path):
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    with pytest.raises(FileNotFoundError, match="GFF3"):
        load_annotation(fasta_path, tmp_path / "does_not_exist.gff3")


def test_load_annotation_raises_actionably_on_unrecognized_annotation_extension(tmp_path):
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    bogus_path = tmp_path / "annotation.txt"
    bogus_path.write_text("not an annotation file")

    with pytest.raises(ValueError, match="does not recognize"):
        load_annotation(fasta_path, bogus_path)


def test_load_annotation_raises_actionably_on_malformed_gff3(tmp_path):
    fasta_path = write_fasta(tmp_path, "chr1", FULL_SEQUENCE)
    bad_gff3 = tmp_path / "bad.gff3"
    bad_gff3.write_text("##gff-version 3\nchr1\ttest\tgene\t21\t50\t.\t+\n")  # only 7 columns

    with pytest.raises(ValueError, match="malformed GFF3"):
        load_annotation(fasta_path, bad_gff3)


def test_load_annotation_raises_actionably_on_seqid_mismatch(tmp_path):
    fasta_path = write_fasta(tmp_path, "totally_different_seqid", FULL_SEQUENCE)
    gff3_path = write_gff3(tmp_path)

    with pytest.raises(ValueError, match="sequence ids"):
        load_annotation(fasta_path, gff3_path)


# ---------------------------------------------------------------------------
# annotate_rule() -- table shape, and a full SetCoveringClassifier round trip
# ---------------------------------------------------------------------------


def test_annotate_rule_accepts_a_plain_kmer_string(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    table = annotate_rule(SINGLE_GENE_KMER, annotation)

    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["gene_name"] == "geneA"
    assert row["feature_type"] == "gene"
    assert row["start"] == 21
    assert row["strand"] == "+"
    assert row["mismatches"] == 0
    assert row["feature_index"] is None
    assert row["presence"] is None


def test_annotate_rule_returns_an_empty_table_with_a_stable_schema_for_zero_hits(reference_and_gff3):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    table = annotate_rule(ZERO_HIT_KMER, annotation)

    assert table.num_rows == 0
    assert set(table.column_names) >= {
        "kmer_sequence", "seqid", "start", "end", "strand", "mismatches",
        "feature_type", "gene_name", "feature_start", "feature_end", "feature_strand",
    }


def test_annotate_rule_rejects_a_rule_with_a_placeholder_feature_name(reference_and_gff3):
    from fastdna.rules import Rule

    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    placeholder_rule = Rule(feature_index=2, feature_name="feature_2", presence=True)

    with pytest.raises(ValueError, match="real DNA sequence"):
        annotate_rule(placeholder_rule, annotation)


def test_full_set_covering_classifier_round_trip(reference_and_gff3):
    """A tiny synthetic classification problem where the single learned
    rule's k-mer is `SINGLE_GENE_KMER` -- confirms `annotate_rule()`
    composes end to end with a real, fitted `SetCoveringClassifier`, not
    just with a hand-built `Rule`.
    """
    np = pytest.importorskip("numpy")
    from fastdna.rules import SetCoveringClassifier

    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)

    feature_names = [SINGLE_GENE_KMER, ZERO_HIT_KMER]
    # Column 0 (SINGLE_GENE_KMER) present iff resistant; column 1 carries no
    # signal, so the greedy search should not need it.
    X = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 1],
        ]
    )
    y = np.array(["resistant", "resistant", "resistant", "sensitive", "sensitive", "sensitive"])

    clf = SetCoveringClassifier(max_rules=3)
    clf.fit(X, y, feature_names=feature_names)

    assert len(clf.rules_) >= 1
    causal_rule = clf.rules_[0]
    assert causal_rule.feature_name == SINGLE_GENE_KMER

    table = annotate_rule(causal_rule, annotation)

    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["gene_name"] == "geneA"
    assert row["feature_index"] == 0
    # classes_ sorts alphabetically, so classes_[1] (the "positive" class the
    # rules describe, per scikit-learn convention) is "sensitive", not
    # "resistant" -- and column 0 is exactly the *complement* of "sensitive"
    # in this fixture, so the greedy search learns "sensitive IF
    # absent(SINGLE_GENE_KMER)", i.e. presence=False. The actionable fact for
    # a caller reading `clf.explain()` is still "presence of this k-mer,
    # which sits inside geneA, tracks resistance" -- just expressed as the
    # negation of the positive class's rule.
    assert row["presence"] is False


# ---------------------------------------------------------------------------
# export_bed() -- BED6 conversion
# ---------------------------------------------------------------------------


def test_export_bed_converts_1_based_inclusive_to_0_based_half_open(reference_and_gff3, tmp_path):
    """`SINGLE_GENE_KMER` is a hand-checkable case: `annotate_rule()` reports
    seqid=chr1, start=21, end=32 (1-based inclusive, an 12 bp span) -- BED's
    0-based half-open form of the same span is chromStart=20, chromEnd=32.
    """
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    table = annotate_rule(SINGLE_GENE_KMER, annotation)

    out = tmp_path / "hits.bed"
    export_bed(table, out)

    lines = out.read_text().splitlines()
    assert len(lines) == 1
    chrom, start, end, name, score, strand = lines[0].split("\t")
    assert chrom == "chr1"
    assert start == "20"
    assert end == "32"
    assert name == SINGLE_GENE_KMER
    assert score == "0"
    assert strand == "+"


def test_export_bed_uses_lf_line_endings(reference_and_gff3, tmp_path):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    table = annotate_rule(REPEAT_MOTIF, annotation)  # two hits

    out = tmp_path / "hits.bed"
    export_bed(table, out)

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2


def test_export_bed_writes_zero_rows_as_an_empty_but_valid_file(reference_and_gff3, tmp_path):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    table = annotate_rule(ZERO_HIT_KMER, annotation)
    assert table.num_rows == 0

    out = tmp_path / "hits.bed"
    export_bed(table, out)

    assert out.exists()
    assert out.read_text() == ""


def test_export_bed_falls_back_to_intergenic_for_a_null_gene_name(reference_and_gff3, tmp_path):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    table = annotate_rule(RC_ONLY_KMER, annotation)  # its one hit is intergenic
    assert table.to_pylist()[0]["gene_name"] is None

    out = tmp_path / "hits.bed"
    export_bed(table, out, name_column="gene_name")

    name = out.read_text().splitlines()[0].split("\t")[3]
    assert name == "intergenic"


def test_export_bed_rejects_a_table_missing_required_columns(tmp_path):
    table = pa.table({"seqid": ["chr1"], "start": [1], "end": [10]})  # no strand column

    with pytest.raises(ValueError, match="strand"):
        export_bed(table, tmp_path / "hits.bed")


def test_export_bed_rejects_an_unknown_name_column(reference_and_gff3, tmp_path):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    table = annotate_rule(SINGLE_GENE_KMER, annotation)

    with pytest.raises(ValueError, match="not_a_real_column"):
        export_bed(table, tmp_path / "hits.bed", name_column="not_a_real_column")


def test_export_bed_handles_multiple_concatenated_rule_tables(reference_and_gff3, tmp_path):
    fasta_path, gff3_path = reference_and_gff3
    annotation = load_annotation(fasta_path, gff3_path)
    combined = pa.concat_tables(
        [annotate_rule(SINGLE_GENE_KMER, annotation), annotate_rule(REPEAT_MOTIF, annotation)]
    )

    out = tmp_path / "hits.bed"
    export_bed(combined, out)

    lines = out.read_text().splitlines()
    assert len(lines) == 3  # 1 hit + 2 hits
