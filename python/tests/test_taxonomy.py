"""Tests for fastdna.taxonomy -- classification against a reference
database, and same-sample identity checks -- built entirely on top of the
stable `fastdna.sketch()` MinHash API tested in `test_sketch.py`.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna
from fastdna.taxonomy import (
    build_reference_database,
    check_sample_identity,
    classify,
    gather,
)


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def repeat_motif(motif: str, times: int) -> str:
    return motif * times


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


def test_classify_ranks_the_matching_reference_first(tmp_path):
    # Three references built from clearly different underlying sequence
    # content (distinct repeated motifs, not just shuffled reads of the
    # same alphabet), so a high score for the matching one is a real
    # signal, not a coincidence of k-mer alphabet overlap.
    motif_a = "ACGTGGCATCAGT"
    motif_b = "TTAGGCCTAAGGC"
    motif_c = "GATCGATCGGATT"

    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif_a, 5)] * 20)
    ref_a_path = write_fastq(tmp_path, "ref_a.fastq", [repeat_motif(motif_a, 5)] * 20)
    ref_b_path = write_fastq(tmp_path, "ref_b.fastq", [repeat_motif(motif_b, 5)] * 20)
    ref_c_path = write_fastq(tmp_path, "ref_c.fastq", [repeat_motif(motif_c, 5)] * 20)

    reference_db = build_reference_database(
        {"match": ref_a_path, "unrelated_b": ref_b_path, "unrelated_c": ref_c_path},
        k=11,
    )

    result = classify(str(query_path), reference_db, k=11, top_n=5)

    names = result.column("name").to_pylist()
    scores = result.column("score").to_pylist()

    assert names[0] == "match"
    assert scores[0] == pytest.approx(1.0)
    # The unrelated references must score meaningfully lower than the
    # true match -- not just "not exactly 1.0".
    for other_score in scores[1:]:
        assert other_score < scores[0] - 0.5


def test_classify_accepts_an_already_built_sketch(tmp_path):
    motif = "ACGTGGCATCAGT"
    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif, 5)] * 10)
    ref_path = write_fastq(tmp_path, "ref.fastq", [repeat_motif(motif, 5)] * 10)

    reference_db = build_reference_database([ref_path], k=11)
    query_sketch = fastdna.sketch(str(query_path), k=11)

    # Passing an already-built Sketch must work exactly like passing a
    # path, and must not require re-reading the FASTQ file.
    result = classify(query_sketch, reference_db, k=11)

    assert result.column("name").to_pylist()[0] == "ref"
    assert result.column("score").to_pylist()[0] == pytest.approx(1.0)


def test_classify_containment_and_jaccard_can_disagree(tmp_path):
    # A small query built entirely from one motif...
    motif = "ACGTGGCATCAGTA"
    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif, 4)] * 5)

    # ...fully contained inside a much larger reference (the same motif,
    # repeated far more, interspersed with a lot of additional sequence
    # content the query never touches)...
    large_reads = [repeat_motif(motif, 4)] * 5 + [repeat_motif("TTGACCGTAGGCCA", 4)] * 200
    large_ref_path = write_fastq(tmp_path, "large_ref.fastq", large_reads)

    # ...versus a small reference that is a comparably-sized but partial
    # match (shares only some of the query's content).
    partial_reads = [repeat_motif(motif, 4)] * 2 + [repeat_motif("CCTTGGAACCTTGG", 4)] * 2
    small_partial_ref_path = write_fastq(tmp_path, "small_partial_ref.fastq", partial_reads)

    reference_db = build_reference_database(
        {"large_full_match": large_ref_path, "small_partial_match": small_partial_ref_path},
        k=11,
        sketch_size=2000,
    )

    containment_result = classify(str(query_path), reference_db, k=11, sketch_size=2000, metric="containment")
    jaccard_result = classify(str(query_path), reference_db, k=11, sketch_size=2000, metric="jaccard")

    containment_top = containment_result.column("name").to_pylist()[0]
    jaccard_top = jaccard_result.column("name").to_pylist()[0]

    # Containment should rank the large reference -- which fully contains
    # the query's content -- on top, since containment doesn't penalize
    # the size mismatch.
    assert containment_top == "large_full_match"

    containment_scores = dict(
        zip(containment_result.column("name").to_pylist(), containment_result.column("score").to_pylist())
    )
    jaccard_scores = dict(zip(jaccard_result.column("name").to_pylist(), jaccard_result.column("score").to_pylist()))

    # The large reference's containment score should be (close to) 1.0 --
    # essentially everything in the query is found in it.
    assert containment_scores["large_full_match"] == pytest.approx(1.0, abs=0.05)

    # But its Jaccard score must be markedly lower than its containment
    # score, precisely because of the size mismatch Jaccard penalizes and
    # containment does not -- this is the disagreement the test exists to
    # demonstrate.
    assert jaccard_scores["large_full_match"] < containment_scores["large_full_match"] - 0.3


def test_classify_top_n_limits_result_rows(tmp_path):
    motif = "ACGTGGCATCAGT"
    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif, 5)] * 10)

    refs = {
        f"ref_{i}": write_fastq(tmp_path, f"ref_{i}.fastq", [repeat_motif(motif, 5)] * 10)
        for i in range(6)
    }
    reference_db = build_reference_database(refs, k=11)

    result = classify(str(query_path), reference_db, k=11, top_n=3)

    assert result.num_rows == 3


def test_classify_rejects_an_unknown_metric(tmp_path):
    motif = "ACGTGGCATCAGT"
    path = write_fastq(tmp_path, "a.fastq", [repeat_motif(motif, 5)] * 5)
    reference_db = build_reference_database([path], k=11)

    with pytest.raises(ValueError):
        classify(str(path), reference_db, k=11, metric="not_a_real_metric")


def test_classify_with_empty_reference_db_returns_empty_table(tmp_path):
    motif = "ACGTGGCATCAGT"
    path = write_fastq(tmp_path, "a.fastq", [repeat_motif(motif, 5)] * 5)

    result = classify(str(path), {}, k=11)

    assert result.num_rows == 0
    assert set(result.column_names) == {"name", "score"}


# ---------------------------------------------------------------------------
# build_reference_database()
# ---------------------------------------------------------------------------


def test_build_reference_database_from_dict_names(tmp_path):
    motif = "ACGTGGCATCAGT"
    ref_path = write_fastq(tmp_path, "ref.fastq", [repeat_motif(motif, 5)] * 5)

    db = build_reference_database({"my_custom_name": ref_path}, k=11)

    assert set(db.keys()) == {"my_custom_name"}
    assert db["my_custom_name"].k == 11


def test_build_reference_database_derives_names_from_filenames(tmp_path):
    ref_path = write_fastq(tmp_path, "listeria.fastq", ["ACGTGGCATCAGT" * 5] * 5)

    db = build_reference_database([ref_path], k=11)

    assert set(db.keys()) == {"listeria"}


def test_build_reference_database_feeds_classify_end_to_end(tmp_path):
    motif_match = "ACGTGGCATCAGT"
    motif_other = "TTAGGCCTAAGGC"

    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif_match, 5)] * 10)
    match_path = write_fastq(tmp_path, "salmonella.fastq", [repeat_motif(motif_match, 5)] * 10)
    other_path = write_fastq(tmp_path, "ecoli.fastq", [repeat_motif(motif_other, 5)] * 10)

    db = build_reference_database([match_path, other_path], k=11)
    result = classify(str(query_path), db, k=11)

    assert result.column("name").to_pylist()[0] == "salmonella"


# ---------------------------------------------------------------------------
# check_sample_identity()
# ---------------------------------------------------------------------------


def test_check_sample_identity_true_for_same_underlying_sample(tmp_path):
    # Two "sequencing runs" of the same sample: same underlying reads.
    reads = ["ACGTGGCATCAGTACGTGGCATCAGT"] * 30
    run_1 = write_fastq(tmp_path, "run1.fastq", reads)
    run_2 = write_fastq(tmp_path, "run2.fastq", reads)

    result = check_sample_identity(str(run_1), str(run_2), k=15)

    assert result.same_sample is True
    assert result.metric == "mash_distance"
    assert result.score == pytest.approx(0.0, abs=1e-6)


def test_check_sample_identity_false_for_clearly_different_samples(tmp_path):
    sample_a = write_fastq(tmp_path, "a.fastq", ["ACGTGGCATCAGTACGTGGCATCAGT"] * 30)
    sample_b = write_fastq(tmp_path, "b.fastq", ["TTAGGCCTAAGGCTTAGGCCTAAGGC"] * 30)

    result = check_sample_identity(str(sample_a), str(sample_b), k=15)

    assert result.same_sample is False
    # Disjoint k-mer content -> mash_distance close to 1 -> similarity
    # (1 - distance) close to 0, nowhere near the default threshold.
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_check_sample_identity_threshold_boundary(tmp_path):
    reads = ["ACGTGGCATCAGTACGTGGCATCAGT"] * 30
    run_1 = write_fastq(tmp_path, "run1.fastq", reads)
    run_2 = write_fastq(tmp_path, "run2.fastq", reads)

    # Identical underlying reads -> mash_distance == 0.0 -> similarity ==
    # 1.0, which must pass even an extremely strict threshold...
    strict = check_sample_identity(str(run_1), str(run_2), k=15, threshold=0.999999)
    assert strict.same_sample is True

    # ...but a threshold above the achievable similarity (> 1.0) must
    # always fail, proving the comparison is real and not hard-coded to
    # True.
    impossible = check_sample_identity(str(run_1), str(run_2), k=15, threshold=1.5)
    assert impossible.same_sample is False


def test_check_sample_identity_supports_jaccard_metric(tmp_path):
    reads = ["ACGTGGCATCAGTACGTGGCATCAGT"] * 30
    run_1 = write_fastq(tmp_path, "run1.fastq", reads)
    run_2 = write_fastq(tmp_path, "run2.fastq", reads)

    result = check_sample_identity(str(run_1), str(run_2), k=15, metric="jaccard")

    assert result.metric == "jaccard"
    assert result.score == pytest.approx(1.0)
    assert result.same_sample is True


def test_check_sample_identity_rejects_an_unknown_metric(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTGGCATCAGT"] * 10)

    with pytest.raises(ValueError):
        check_sample_identity(str(path), str(path), metric="not_a_real_metric")


# ---------------------------------------------------------------------------
# gather() -- stretch goal
# ---------------------------------------------------------------------------


def test_gather_picks_out_two_distinct_organisms_in_a_mixed_sample(tmp_path):
    motif_a = "ACGTGGCATCAGT"
    motif_b = "TTAGGCCTAAGGC"
    motif_c = "GATCGATCGGATT"

    # A "metagenomic" query made of two distinct organisms' content.
    query_reads = [repeat_motif(motif_a, 5)] * 20 + [repeat_motif(motif_b, 5)] * 20
    query_path = write_fastq(tmp_path, "mixed_query.fastq", query_reads)

    ref_a_path = write_fastq(tmp_path, "organism_a.fastq", [repeat_motif(motif_a, 5)] * 20)
    ref_b_path = write_fastq(tmp_path, "organism_b.fastq", [repeat_motif(motif_b, 5)] * 20)
    ref_c_path = write_fastq(tmp_path, "organism_c.fastq", [repeat_motif(motif_c, 5)] * 20)

    reference_db = build_reference_database(
        {"organism_a": ref_a_path, "organism_b": ref_b_path, "organism_c": ref_c_path},
        k=11,
        sketch_size=2000,
    )

    result = gather(str(query_path), reference_db, k=11, sketch_size=2000, min_containment=0.05)

    picked_names = set(result.column("name").to_pylist())
    assert {"organism_a", "organism_b"}.issubset(picked_names)
    assert "organism_c" not in picked_names


def test_gather_discounts_a_redundant_duplicate_reference(tmp_path):
    motif = "ACGTGGCATCAGT"
    query_path = write_fastq(tmp_path, "query.fastq", [repeat_motif(motif, 5)] * 20)

    # Two references built from identical content -- picking the second
    # after the first should contribute essentially nothing new.
    ref_1_path = write_fastq(tmp_path, "ref1.fastq", [repeat_motif(motif, 5)] * 20)
    ref_2_path = write_fastq(tmp_path, "ref2_duplicate.fastq", [repeat_motif(motif, 5)] * 20)

    reference_db = build_reference_database({"ref1": ref_1_path, "ref2_duplicate": ref_2_path}, k=11)

    result = gather(str(query_path), reference_db, k=11, min_containment=0.05)

    # Only the first (whichever is picked first) should clear the
    # min_containment bar on its *adjusted* score -- the duplicate's
    # adjusted score should be discounted to (near) zero and therefore
    # excluded.
    assert result.num_rows == 1


def test_gather_with_empty_reference_db_returns_empty_table(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTGGCATCAGT"] * 10)

    result = gather(str(path), {}, k=11)

    assert result.num_rows == 0
    assert set(result.column_names) == {"name", "containment", "adjusted_score"}


def test_gather_rejects_invalid_min_containment(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTGGCATCAGT"] * 10)
    reference_db = build_reference_database([path], k=11)

    with pytest.raises(ValueError):
        gather(str(path), reference_db, k=11, min_containment=1.5)
