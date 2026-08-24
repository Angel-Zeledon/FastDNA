import math

import pytest

from fastdna.active_learning import (
    prioritize_for_review,
    suggest_reference_additions,
    uncertainty_score,
)


# ---------------------------------------------------------------------------
# margin method
# ---------------------------------------------------------------------------


def test_margin_confident_result_has_low_uncertainty():
    confident = [("E. coli", 0.95), ("Shigella", 0.02), ("Salmonella", 0.01)]
    score = uncertainty_score(confident, method="margin")
    # margin = 0.95 - 0.02 = 0.93 -> uncertainty = 1 - 0.93 = 0.07
    assert score == pytest.approx(0.07)


def test_margin_torn_result_has_high_uncertainty():
    torn = [("E. coli", 0.51), ("Shigella", 0.49)]
    score = uncertainty_score(torn, method="margin")
    # margin = 0.51 - 0.49 = 0.02 -> uncertainty = 1 - 0.02 = 0.98
    assert score == pytest.approx(0.98)


def test_margin_orders_confident_below_torn():
    confident = [("E. coli", 0.95), ("Shigella", 0.02), ("Salmonella", 0.01)]
    torn = [("E. coli", 0.51), ("Shigella", 0.49)]

    confident_score = uncertainty_score(confident, method="margin")
    torn_score = uncertainty_score(torn, method="margin")

    assert confident_score < torn_score


# ---------------------------------------------------------------------------
# entropy method
# ---------------------------------------------------------------------------


def test_entropy_peaked_distribution_has_low_uncertainty():
    peaked = [("A", 0.98), ("B", 0.01), ("C", 0.01)]
    score = uncertainty_score(peaked, method="entropy")
    # Hand-computed Shannon entropy, base 2: -sum(p * log2(p))
    expected = -(0.98 * math.log2(0.98) + 0.01 * math.log2(0.01) * 2)
    assert score == pytest.approx(expected)
    assert score < 0.2  # sanity check: close to zero, sharply peaked


def test_entropy_uniform_distribution_has_high_uncertainty():
    uniform = [("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)]
    score = uncertainty_score(uniform, method="entropy")
    # Perfectly uniform over 4 classes -> entropy == log2(4) == 2.0 exactly.
    assert score == pytest.approx(2.0)


def test_entropy_orders_peaked_below_uniform():
    peaked = [("A", 0.98), ("B", 0.01), ("C", 0.01)]
    near_uniform = [("A", 0.34), ("B", 0.33), ("C", 0.33)]

    peaked_score = uncertainty_score(peaked, method="entropy")
    uniform_score = uncertainty_score(near_uniform, method="entropy")

    assert peaked_score < uniform_score


def test_entropy_normalizes_raw_nonnegative_scores_by_sum():
    # Raw containment-style scores that don't sum to 1: [0.8, 0.1] should
    # normalize to [0.8/0.9, 0.1/0.9] via simple sum-normalization.
    raw = [("RefA", 0.8), ("RefB", 0.1)]
    score = uncertainty_score(raw, method="entropy")
    p1, p2 = 0.8 / 0.9, 0.1 / 0.9
    expected = -(p1 * math.log2(p1) + p2 * math.log2(p2))
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# prioritize_for_review
# ---------------------------------------------------------------------------


def test_prioritize_for_review_sorts_most_uncertain_first():
    batch = {
        "sample_confident": [("E. coli", 0.95), ("Shigella", 0.02), ("Salmonella", 0.01)],
        "sample_torn": [("E. coli", 0.51), ("Shigella", 0.49)],
        "sample_medium": [("E. coli", 0.70), ("Shigella", 0.25), ("Salmonella", 0.05)],
    }

    queue = prioritize_for_review(batch, method="margin")

    ids_in_order = [entry[0] for entry in queue]
    assert ids_in_order == ["sample_torn", "sample_medium", "sample_confident"]

    # Scores themselves are genuinely descending.
    scores_in_order = [entry[1] for entry in queue]
    assert scores_in_order == sorted(scores_in_order, reverse=True)

    # Each entry carries (query_id, uncertainty_score, original_result).
    for query_id, score, original in queue:
        assert query_id in batch
        assert original == batch[query_id]
        assert isinstance(score, float)


def test_prioritize_for_review_accepts_list_of_pairs_too():
    batch = [
        ("s1", [("A", 0.9), ("B", 0.1)]),
        ("s2", [("A", 0.55), ("B", 0.45)]),
    ]
    queue = prioritize_for_review(batch, method="margin")
    assert [entry[0] for entry in queue] == ["s2", "s1"]


def test_prioritize_for_review_top_n_truncates():
    batch = {
        "sample_confident": [("E. coli", 0.95), ("Shigella", 0.02), ("Salmonella", 0.01)],
        "sample_torn": [("E. coli", 0.51), ("Shigella", 0.49)],
        "sample_medium": [("E. coli", 0.70), ("Shigella", 0.25), ("Salmonella", 0.05)],
    }

    queue = prioritize_for_review(batch, method="margin", top_n=2)

    assert len(queue) == 2
    assert [entry[0] for entry in queue] == ["sample_torn", "sample_medium"]


def test_prioritize_for_review_rejects_empty_batch():
    with pytest.raises(ValueError):
        prioritize_for_review({})
    with pytest.raises(ValueError):
        prioritize_for_review([])


# ---------------------------------------------------------------------------
# malformed input
# ---------------------------------------------------------------------------


def test_uncertainty_score_rejects_empty_ranked_results():
    with pytest.raises(ValueError):
        uncertainty_score([], method="margin")
    with pytest.raises(ValueError):
        uncertainty_score([], method="entropy")


def test_uncertainty_score_rejects_mismatched_probs_and_labels():
    probs = [0.5, 0.3, 0.2]
    labels = ["A", "B"]  # deliberately mismatched length
    with pytest.raises(ValueError):
        uncertainty_score((probs, labels), method="margin")


def test_uncertainty_score_rejects_empty_probability_vector():
    with pytest.raises(ValueError):
        uncertainty_score(([], []), method="margin")


def test_uncertainty_score_rejects_unknown_method():
    with pytest.raises(ValueError):
        uncertainty_score([("A", 0.9), ("B", 0.1)], method="bogus")


def test_uncertainty_score_rejects_malformed_entries():
    with pytest.raises(ValueError):
        uncertainty_score([("A", 0.9), "not-a-pair"], method="margin")


# ---------------------------------------------------------------------------
# both input shapes agree
# ---------------------------------------------------------------------------


def test_ranked_tuples_and_probability_vector_shapes_agree():
    # Same underlying confidence pattern (confident) expressed both ways.
    ranked_confident = [("E. coli", 0.95), ("Shigella", 0.03), ("Salmonella", 0.02)]
    probs_confident = ([0.95, 0.03, 0.02], ["E. coli", "Shigella", "Salmonella"])

    # Same underlying confidence pattern (torn) expressed both ways.
    ranked_torn = [("E. coli", 0.51), ("Shigella", 0.49)]
    probs_torn = ([0.51, 0.49], ["E. coli", "Not_E_coli"])

    for method in ("margin", "entropy"):
        confident_from_ranked = uncertainty_score(ranked_confident, method=method)
        confident_from_probs = uncertainty_score(probs_confident, method=method)
        torn_from_ranked = uncertainty_score(ranked_torn, method=method)
        torn_from_probs = uncertainty_score(probs_torn, method=method)

        # Values from equivalent inputs should match exactly (both shapes
        # reduce to the same underlying score list).
        assert confident_from_ranked == pytest.approx(confident_from_probs)

        # And the relative ranking (confident < torn) should hold within
        # each shape.
        assert confident_from_ranked < torn_from_ranked
        assert confident_from_probs < torn_from_probs


def test_probability_vector_shape_detected_correctly_with_three_classes():
    # A probability vector shape with != 2 classes is unambiguous even
    # with numeric-looking data throughout.
    probs = [0.6, 0.3, 0.1]
    labels = ["A", "B", "C"]
    score = uncertainty_score((probs, labels), method="margin")
    assert score == pytest.approx(1.0 - (0.6 - 0.3))


# ---------------------------------------------------------------------------
# suggest_reference_additions
# ---------------------------------------------------------------------------


def test_suggest_reference_additions_filters_by_threshold():
    batch = {
        "sample_confident": [("E. coli", 0.95), ("Shigella", 0.02), ("Salmonella", 0.01)],
        "sample_torn": [("E. coli", 0.51), ("Shigella", 0.49)],
        "sample_medium": [("E. coli", 0.70), ("Shigella", 0.25), ("Salmonella", 0.05)],
    }
    queue = prioritize_for_review(batch, method="margin")

    strong_candidates = suggest_reference_additions(queue, uncertainty_threshold=0.9)

    assert [entry[0] for entry in strong_candidates] == ["sample_torn"]

    # A threshold that admits everyone.
    all_candidates = suggest_reference_additions(queue, uncertainty_threshold=0.0)
    assert len(all_candidates) == len(queue)

    # A threshold nothing clears.
    none_candidates = suggest_reference_additions(queue, uncertainty_threshold=999)
    assert none_candidates == []
