"""Tests for fastdna.metagenomics -- Kraken2-style per-read classification
against a k-mer -> lowest-common-ancestor database.

The algorithm's arithmetic is pinned in Rust, against hand-counted k-mers
(`src/metagenomics.rs`). What these tests cover is the Python surface: that
the documented signatures work, that the Arrow tables have the promised
columns, that every documented validation failure surfaces as a `ValueError`
naming the offending row, and that the honesty-critical numbers -- the
confidence denominator, the abundance denominator, the memory cost -- are
what the docstrings say they are.
"""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from fastdna.metagenomics import KmerDatabase, build_database

# The toy tree used throughout:
#
#   1 root
#   └── 10 genus  Toyella
#       ├── 100 species  Toyella alpha
#       └── 200 species  Toyella beta
TAXONOMY = "\n".join(
    [
        "tax_id\tparent_tax_id\trank\tname\tsequence_ids",
        "1\t1\tno rank\troot\t",
        "10\t1\tgenus\tToyella\t",
        "100\t10\tspecies\tToyella alpha\tspecies_a",
        "200\t10\tspecies\tToyella beta\tspecies_b",
        "",
    ]
)

K = 11
SHARED = "GATTACAGATTACAGGCC"
ONLY_A = "TTGCACCGTAAGCTATCG"
ONLY_B = "ACGCGTTAACCGGATCAT"
# Present in neither reference, so every k-mer touching it matches nothing.
NOVEL = "CTCTAGGACTGA"


def write(tmp_path: pathlib.Path, name: str, text: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def write_reference(tmp_path: pathlib.Path) -> pathlib.Path:
    return write(
        tmp_path,
        "reference.fasta",
        f">species_a Toyella alpha\n{ONLY_A}{SHARED}\n>species_b\n{ONLY_B}{SHARED}\n",
    )


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[tuple[str, str]]) -> pathlib.Path:
    return write(
        tmp_path,
        name,
        "".join(f"@{read_id}\n{seq}\n+\n{'I' * len(seq)}\n" for read_id, seq in reads),
    )


@pytest.fixture()
def db(tmp_path):
    return build_database(write_reference(tmp_path), write(tmp_path, "taxonomy.tsv", TAXONOMY), k=K)


# ---------------------------------------------------------------------------
# build_database()
# ---------------------------------------------------------------------------


def test_build_database_returns_a_usable_database(db):
    assert db.k == K
    assert db.n_kmers > 0
    assert "KmerDatabase" in repr(db)


def test_memory_is_twelve_bytes_per_kmer_plus_the_taxonomy(db):
    # The number every scale claim in the docstrings rests on. If the
    # representation changes, this fails and the docstrings must change too.
    assert db.memory_bytes > 12 * db.n_kmers
    assert db.memory_bytes < 12 * db.n_kmers + 10_000, "the taxonomy term must stay small"


def test_build_database_writes_the_output_file_and_still_returns_the_database(tmp_path):
    out = tmp_path / "toy.fdb"
    built = build_database(
        write_reference(tmp_path), write(tmp_path, "taxonomy.tsv", TAXONOMY), k=K, output=out
    )
    assert out.exists()
    assert built.n_kmers > 0

    reloaded = KmerDatabase.load(out)
    assert reloaded.n_kmers == built.n_kmers
    assert reloaded.k == built.k


def test_a_saved_database_classifies_exactly_as_the_original(tmp_path, db):
    out = tmp_path / "toy.fdb"
    db.save(out)
    reads = write_fastq(tmp_path, "reads.fastq", [("a", ONLY_A), ("s", SHARED)])

    assert KmerDatabase.load(out).classify(reads) == db.classify(reads)


def test_loading_a_corrupt_database_raises_rather_than_answering_wrongly(tmp_path, db):
    out = tmp_path / "toy.fdb"
    db.save(out)
    out.write_bytes(out.read_bytes()[:-5])  # truncate mid-entry

    with pytest.raises(ValueError, match="truncated|corrupt"):
        KmerDatabase.load(out)


def test_loading_a_foreign_file_raises(tmp_path):
    path = write(tmp_path, "not_a_db.fdb", "just some text")
    with pytest.raises(ValueError, match="FastDNA"):
        KmerDatabase.load(path)


def test_loading_a_missing_database_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        KmerDatabase.load(tmp_path / "nope.fdb")


# ---------------------------------------------------------------------------
# taxonomy validation -- every failure names the offending row
# ---------------------------------------------------------------------------


def build_with_taxonomy(tmp_path, taxonomy_text):
    return build_database(
        write_reference(tmp_path), write(tmp_path, "taxonomy.tsv", taxonomy_text), k=K
    )


@pytest.mark.parametrize(
    ("name", "taxonomy", "expected"),
    [
        (
            "missing parent",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "1\t1\tno rank\troot\t\n"
            "100\t99\tspecies\tA\tspecies_a\n"
            "200\t1\tspecies\tB\tspecies_b\n",
            ["row 3", "99"],
        ),
        (
            "cycle",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "1\t1\tno rank\troot\t\n"
            "100\t200\tspecies\tA\tspecies_a\n"
            "200\t100\tspecies\tB\tspecies_b\n",
            ["100"],
        ),
        (
            "duplicate tax_id",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "1\t1\tno rank\troot\t\n"
            "100\t1\tspecies\tA\tspecies_a\n"
            "100\t1\tspecies\tB\tspecies_b\n",
            ["row 4", "100"],
        ),
        (
            "duplicate sequence id",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "1\t1\tno rank\troot\t\n"
            "100\t1\tspecies\tA\tspecies_a\n"
            "200\t1\tspecies\tB\tspecies_a\n",
            ["row 4", "species_a"],
        ),
        (
            "reserved tax_id 0",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "1\t1\tno rank\troot\t\n"
            "0\t1\tspecies\tA\tspecies_a\n",
            ["row 3", "reserved"],
        ),
        (
            "missing column",
            "tax_id\tparent_tax_id\trank\n1\t1\tno rank\n",
            ["name"],
        ),
        (
            "no root",
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n"
            "100\t200\tspecies\tA\tspecies_a\n"
            "200\t100\tspecies\tB\tspecies_b\n",
            ["root"],
        ),
    ],
)
def test_a_broken_taxonomy_raises_naming_the_offending_row(tmp_path, name, taxonomy, expected):
    with pytest.raises(ValueError) as caught:
        build_with_taxonomy(tmp_path, taxonomy)
    message = str(caught.value)
    for fragment in expected:
        assert fragment in message, f"{name}: {fragment!r} missing from {message!r}"


def test_a_reference_sequence_with_no_taxon_is_named(tmp_path):
    reference = write(tmp_path, "reference.fasta", f">species_a\n{ONLY_A}\n>surprise\n{ONLY_B}\n")
    with pytest.raises(ValueError, match="surprise"):
        build_database(reference, write(tmp_path, "taxonomy.tsv", TAXONOMY), k=K)


def test_a_taxonomy_sequence_missing_from_the_reference_is_named(tmp_path):
    reference = write(tmp_path, "reference.fasta", f">species_a\n{ONLY_A}\n")
    with pytest.raises(ValueError) as caught:
        build_database(reference, write(tmp_path, "taxonomy.tsv", TAXONOMY), k=K)
    assert "species_b" in str(caught.value)
    assert "row 5" in str(caught.value)


@pytest.mark.parametrize("k", [0, 33, 64])
def test_an_out_of_range_k_raises(tmp_path, k):
    with pytest.raises(ValueError, match="1 and 32"):
        build_database(
            write_reference(tmp_path), write(tmp_path, "taxonomy.tsv", TAXONOMY), k=k
        )


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


def calls(db, tmp_path, reads, **kwargs):
    path = write_fastq(tmp_path, "reads.fastq", reads)
    return db.classify(path, **kwargs)


def test_classify_returns_the_documented_schema(db, tmp_path):
    table = calls(db, tmp_path, [("r1", ONLY_A)])
    assert isinstance(table, pa.Table)
    assert table.column_names == [
        "read_id",
        "tax_id",
        "confidence",
        "n_kmers",
        "n_classified_kmers",
    ]
    assert table.schema.field("tax_id").type == pa.uint32()
    assert table.schema.field("confidence").type == pa.float64()
    assert table.schema.field("read_id").type == pa.string()


def test_species_specific_reads_classify_to_their_species(db, tmp_path):
    table = calls(db, tmp_path, [("a", ONLY_A), ("b", ONLY_B)])
    assert table.column("tax_id").to_pylist() == [100, 200]
    assert table.column("read_id").to_pylist() == ["a", "b"]


def test_a_read_of_shared_kmers_classifies_to_the_genus(db, tmp_path):
    # The k-mers in SHARED occur in both species, so they map to the LCA.
    table = calls(db, tmp_path, [("s", SHARED)])
    assert table.column("tax_id").to_pylist() == [10]


def test_a_read_split_between_two_species_classifies_to_their_ancestor(db, tmp_path):
    table = calls(db, tmp_path, [("chimera", ONLY_A + ONLY_B)])
    assert table.column("tax_id").to_pylist() == [10]


def test_confidence_is_over_all_kmers_not_over_matching_kmers(db, tmp_path):
    # ONLY_A (18 bases) + NOVEL (12) = 30 bases = 20 k-mers at k=11. The 8
    # k-mers wholly inside ONLY_A match species 100; the other 12 span the
    # junction or lie in NOVEL and match nothing. Kraken 2's confidence is
    # 8/20 = 0.4 -- not 8/8 = 1.0, which is what dividing by the *matching*
    # k-mers would give for a read that 60% of the time matched nothing.
    table = calls(db, tmp_path, [("partial", ONLY_A + NOVEL)])
    row = table.to_pylist()[0]
    assert row["n_kmers"] == 20
    assert row["n_classified_kmers"] == 8
    assert row["tax_id"] == 100
    assert row["confidence"] == pytest.approx(0.4)


def test_a_threshold_pushes_a_borderline_read_up_the_tree(db, tmp_path):
    # species_a's own sequence: 26 k-mers, 18 unique to species 100 and 8
    # inside SHARED and so at the genus. At threshold 0 the call is the
    # species with confidence 18/26; at 0.7 it needs ceil(0.7*26) = 19 and
    # has 18, so it moves up to the genus, whose clade holds all 26.
    reads = write_fastq(tmp_path, "reads.fastq", [("g", ONLY_A + SHARED)])

    unfiltered = db.classify(reads).to_pylist()[0]
    assert unfiltered["tax_id"] == 100
    assert unfiltered["confidence"] == pytest.approx(18 / 26)

    promoted = db.classify(reads, confidence_threshold=0.7).to_pylist()[0]
    assert promoted["tax_id"] == 10
    assert promoted["confidence"] == pytest.approx(1.0)


def test_a_threshold_of_one_leaves_a_partly_matching_read_unclassified(db, tmp_path):
    reads = write_fastq(tmp_path, "reads.fastq", [("partial", ONLY_A + NOVEL)])
    row = db.classify(reads, confidence_threshold=1.0).to_pylist()[0]
    assert row["tax_id"] == 0, "0 is unclassified, Kraken's convention"
    assert row["confidence"] == 0.0
    assert row["n_classified_kmers"] == 8, "the evidence is still reported"


def test_raising_the_threshold_never_moves_a_call_sideways(db, tmp_path):
    reads = write_fastq(
        tmp_path, "reads.fastq", [("g", ONLY_A + SHARED), ("p", ONLY_A + NOVEL), ("s", SHARED)]
    )
    ancestors = {100: {100, 10, 1}, 200: {200, 10, 1}, 10: {10, 1}, 1: {1}}

    previous = db.classify(reads).column("tax_id").to_pylist()
    for step in range(1, 11):
        current = db.classify(reads, confidence_threshold=step / 10).column("tax_id").to_pylist()
        for before, after in zip(previous, current):
            assert after == 0 or after in ancestors[before], (
                f"at threshold {step / 10}, a call moved from {before} to {after}"
            )
        previous = current


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_an_out_of_range_threshold_raises(db, tmp_path, threshold):
    reads = write_fastq(tmp_path, "reads.fastq", [("r", ONLY_A)])
    with pytest.raises(ValueError, match="confidence_threshold"):
        db.classify(reads, confidence_threshold=threshold)


# ---------------------------------------------------------------------------
# the degenerate reads: none may be silently dropped
# ---------------------------------------------------------------------------


def test_degenerate_reads_are_reported_as_unclassified_not_dropped(db, tmp_path):
    table = calls(
        db,
        tmp_path,
        [
            ("too_short", "ACGTA"),
            ("all_ambiguous", "N" * 20),
            ("matches_nothing", NOVEL + NOVEL),
            ("good", ONLY_A),
        ],
    )

    assert table.num_rows == 4, "every read gets a row, however useless"
    rows = {row["read_id"]: row for row in table.to_pylist()}

    assert rows["too_short"]["tax_id"] == 0
    assert rows["too_short"]["n_kmers"] == 0
    assert rows["too_short"]["confidence"] == 0.0, "0/0 is reported as 0.0, never NaN"

    assert rows["all_ambiguous"]["tax_id"] == 0
    assert rows["all_ambiguous"]["n_kmers"] == 0

    # Distinguishable from the two above: it had k-mers, none of them known.
    assert rows["matches_nothing"]["tax_id"] == 0
    assert rows["matches_nothing"]["n_kmers"] > 0
    assert rows["matches_nothing"]["n_classified_kmers"] == 0

    assert rows["good"]["tax_id"] == 100


def test_an_empty_read_file_gives_a_well_formed_empty_table(db, tmp_path):
    table = db.classify(write(tmp_path, "empty.fastq", ""))
    assert table.num_rows == 0
    assert table.column_names == [
        "read_id",
        "tax_id",
        "confidence",
        "n_kmers",
        "n_classified_kmers",
    ]


def test_a_read_with_some_ambiguity_still_classifies_on_its_clean_windows(db, tmp_path):
    table = calls(db, tmp_path, [("mixed", f"{ONLY_A}NNNN{ONLY_B}")])
    row = table.to_pylist()[0]
    assert row["n_kmers"] == 16, "8 clean windows per half; none span the Ns"
    assert row["tax_id"] == 10


def test_gzipped_and_fasta_reads_are_accepted(db, tmp_path):
    import gzip

    fasta = write(tmp_path, "reads.fasta", f">r1\n{ONLY_A}\n>r2\n{SHARED}\n")
    gz = tmp_path / "reads.fastq.gz"
    with gzip.open(gz, "wt") as handle:
        handle.write(f"@r1\n{ONLY_A}\n+\n{'I' * len(ONLY_A)}\n@r2\n{SHARED}\n+\n{'I' * len(SHARED)}\n")

    assert db.classify(fasta).column("tax_id").to_pylist() == [100, 10]
    assert db.classify(gz).column("tax_id").to_pylist() == [100, 10]


# ---------------------------------------------------------------------------
# abundance()
# ---------------------------------------------------------------------------


def test_abundance_reports_names_ranks_and_a_normalized_share(db, tmp_path):
    table = calls(
        db,
        tmp_path,
        [("a1", ONLY_A), ("a2", ONLY_A), ("b1", ONLY_B), ("none", NOVEL + NOVEL)],
    )
    report = db.abundance(table)

    assert report.column_names == ["tax_id", "name", "rank", "reads", "relative_abundance"]
    rows = report.to_pylist()

    assert rows[0]["tax_id"] == 100, "most reads first"
    assert rows[0]["name"] == "Toyella alpha"
    assert rows[0]["rank"] == "species"
    assert rows[0]["reads"] == 2
    assert rows[0]["relative_abundance"] == pytest.approx(0.5)

    by_id = {row["tax_id"]: row for row in rows}
    assert by_id[0]["name"] == "unclassified"
    assert by_id[0]["reads"] == 1


def test_relative_abundance_sums_to_one_over_every_read_including_unclassified(db, tmp_path):
    table = calls(
        db, tmp_path, [("a", ONLY_A), ("b", ONLY_B), ("n1", NOVEL + NOVEL), ("n2", NOVEL + NOVEL)]
    )
    report = db.abundance(table)
    assert sum(report.column("relative_abundance").to_pylist()) == pytest.approx(1.0)
    # Half the reads matched nothing, and the report must say so rather
    # than renormalizing the unclassified fraction away.
    unclassified = [r for r in report.to_pylist() if r["tax_id"] == 0][0]
    assert unclassified["relative_abundance"] == pytest.approx(0.5)


def test_abundance_accepts_a_plain_sequence_of_tax_ids(db):
    report = db.abundance([100, 100, 200])
    assert report.column("tax_id").to_pylist() == [100, 200]
    assert report.column("reads").to_pylist() == [2, 1]


def test_abundance_over_no_reads_is_empty_not_a_division_by_zero(db):
    report = db.abundance([])
    assert report.num_rows == 0
    assert report.column_names == ["tax_id", "name", "rank", "reads", "relative_abundance"]


def test_abundance_is_deterministic_under_ties(db):
    first = db.abundance([100, 200, 10]).column("tax_id").to_pylist()
    for _ in range(8):
        assert db.abundance([10, 200, 100]).column("tax_id").to_pylist() == first
    assert first == [10, 100, 200], "ties break by tax_id ascending"


# ---------------------------------------------------------------------------
# the honesty requirements, asserted rather than assumed
# ---------------------------------------------------------------------------


def test_the_module_documents_its_scale_ceiling_and_the_bracken_caveat():
    import fastdna.metagenomics as module

    # These are the two claims a user must not be able to miss: that the
    # database does not hold RefSeq, and that the abundance report is not
    # corrected for genome length. If either docstring is ever trimmed,
    # this fails rather than the caveat quietly disappearing.
    assert "RefSeq" in module.__doc__
    assert "12 bytes per distinct canonical k-mer" in module.__doc__
    assert "not been benchmarked against it" in module.__doc__
    assert "minimizers" in module.__doc__

    abundance_doc = module.KmerDatabase.abundance.__doc__
    assert "read abundance, not organism abundance" in abundance_doc
    assert "Bracken" in abundance_doc
    assert "FastDNA does not implement it" in abundance_doc

    classify_doc = module.KmerDatabase.classify.__doc__
    assert "Kraken 2's own definition" in classify_doc
