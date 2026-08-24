"""Audit tests for `fastdna.active_learning` (commits 824a145, merge 90238b0).

Complements `test_active_learning.py`. That file pins both uncertainty
measures against hand-computed values and checks the sort direction, all of
which is sound. What it never touches:

  * the probability-vector shape with *numeric* class labels -- the shape
    scikit-learn hands you for a plain binary problem (`clf.classes_` is
    `[0, 1]`);
  * `_normalize`'s all-zero branch, which is exactly what
    `fastdna.taxonomy.classify` returns for a sample matching nothing --
    the module docstring's own motivating case;
  * `_normalize`'s softmax branch (negative scores), never executed;
  * `_scores_from_ranked_tuples`'s malformed-entry guard, which is
    unreachable by construction (see below), and which the test named
    `test_uncertainty_score_rejects_malformed_entries` does not in fact
    exercise.

Tests whose docstring begins with "EXPECTED TO FAIL" pin a reported defect
and are expected to be red until the module is patched or reverted; see the
audit report. They are deliberately not xfail-marked.
"""
from __future__ import annotations

import math

import pytest

from fastdna.active_learning import (
    _normalize,
    prioritize_for_review,
    suggest_reference_additions,
    uncertainty_score,
)


# ---------------------------------------------------------------------------
# Direction checks (these are the module's core claim, and they hold)
# ---------------------------------------------------------------------------


def test_margin_ranks_the_smallest_margin_first_across_a_wide_batch():
    """The headline claim: smallest top1-top2 gap = most uncertain = first
    in the queue. Checked over a batch whose true margin ordering is known
    by construction, so a reversed sort or a reversed `1 - margin` would
    invert the whole list rather than just perturb it.
    """
    batch = {
        "gap_0.00": [("A", 0.50), ("B", 0.50), ("C", 0.00)],
        "gap_0.05": [("A", 0.55), ("B", 0.50), ("C", 0.00)],
        "gap_0.20": [("A", 0.70), ("B", 0.50), ("C", 0.00)],
        "gap_0.60": [("A", 0.90), ("B", 0.30), ("C", 0.00)],
        "gap_0.95": [("A", 0.99), ("B", 0.04), ("C", 0.00)],
    }
    queue = prioritize_for_review(batch, method="margin")

    assert [qid for qid, _, _ in queue] == [
        "gap_0.00", "gap_0.05", "gap_0.20", "gap_0.60", "gap_0.95",
    ]
    assert [round(score, 10) for _, score, _ in queue] == [1.0, 0.95, 0.8, 0.4, 0.05]


def test_entropy_ranks_the_flattest_distribution_first_across_a_wide_batch():
    """Same for entropy: largest = most uncertain. Values are checked
    against `-sum(p*log2(p))` recomputed in the test.
    """
    distributions = {
        "uniform_4": [0.25, 0.25, 0.25, 0.25],
        "mild_4": [0.40, 0.30, 0.20, 0.10],
        "peaked_4": [0.90, 0.04, 0.03, 0.03],
        "certain_4": [1.00, 0.00, 0.00, 0.00],
    }
    batch = {
        name: [(f"c{i}", p) for i, p in enumerate(probs)]
        for name, probs in distributions.items()
    }
    queue = prioritize_for_review(batch, method="entropy")

    assert [qid for qid, _, _ in queue] == ["uniform_4", "mild_4", "peaked_4", "certain_4"]
    for qid, score, _ in queue:
        expected = -sum(p * math.log2(p) for p in distributions[qid] if p > 0)
        assert score == pytest.approx(expected)
    assert queue[0][1] == pytest.approx(2.0), "log2(4)"
    assert queue[-1][1] == pytest.approx(0.0), "all mass on one class"


# ---------------------------------------------------------------------------
# The scikit-learn binary case
# ---------------------------------------------------------------------------


def test_binary_probability_vector_with_integer_class_labels_is_not_misread():
    """EXPECTED TO FAIL -- pins a reported defect.

    `_is_ranked_tuples` decides between the module's two input shapes by
    asking whether *every* top-level element is a length-2 sequence with a
    numeric second element. For a two-class probability vector paired with
    integer class labels -- `([0.5, 0.5], [0, 1])`, which is exactly
    `(clf.predict_proba(x)[0].tolist(), list(clf.classes_))` for an
    ordinary binary scikit-learn classifier -- both elements pass that
    test, so the pair is read as a two-entry ranked list.

    The extracted "scores" are then the second element of each half:
    `[probs[1], labels[1]]` == `[p_class1, 1]`. Margin uncertainty
    collapses to `1 - (1 - p_class1)` == `p_class1`, i.e. the module
    returns the probability of the positive class as an uncertainty.

    That is not merely a wrong magnitude, it is an inverted ranking.
    Measured on this batch:

        coin flip   ([0.50, 0.50], [0, 1])  ->  0.500  (true 1.00)
        p1 = 0.99   ([0.01, 0.99], [0, 1])  ->  0.990  (true 0.02)
        p0 = 0.99   ([0.99, 0.01], [0, 1])  ->  0.010  (true 0.02)

    so `prioritize_for_review` puts the model's single most confident
    prediction at the *head* of the human review queue and the genuine
    50/50 coin flip below it. The module docstring calls this a "known
    limitation"; it is not signalled at runtime, not validated, and not
    covered by any existing test.

    Note the shape is only ambiguous because the ranked-tuples contract
    ("ordered best-first") is not checked: read as a ranked list,
    `([0.5, 0.5], [0, 1])` has *ascending* scores (0.5 then 1) and could be
    rejected on that basis alone.
    """
    batch = {
        "coin_flip": ([0.50, 0.50], [0, 1]),
        "confident_class_1": ([0.01, 0.99], [0, 1]),
        "confident_class_0": ([0.99, 0.01], [0, 1]),
    }
    queue = prioritize_for_review(batch, method="margin")

    assert queue[0][0] == "coin_flip", (
        "the 50/50 sample must head the review queue; got "
        f"{[(qid, round(score, 3)) for qid, score, _ in queue]}"
    )


def test_binary_string_labels_and_the_equivalent_ranked_list_agree_exactly():
    """The same two-class input with *string* labels is detected correctly.
    `test_active_learning.py::test_ranked_tuples_and_probability_vector_shapes_agree`
    builds this pair but only asserts equality for its three-class case, so
    the two-class agreement was never actually pinned.
    """
    for method in ("margin", "entropy"):
        from_probs = uncertainty_score(([0.51, 0.49], ["E. coli", "other"]), method=method)
        from_ranked = uncertainty_score([("E. coli", 0.51), ("other", 0.49)], method=method)
        assert from_probs == pytest.approx(from_ranked)


# ---------------------------------------------------------------------------
# _normalize's two unexecuted branches
# ---------------------------------------------------------------------------


def test_all_zero_scores_are_maximally_uncertain_under_both_methods():
    """`fastdna.taxonomy.classify` returns all-zero containment scores for
    a sample that matches nothing in the reference set -- the module
    docstring's own motivating scenario ("when a k-mer-based classifier
    doesn't confidently match a query against any known class"). Neither
    `_normalize`'s all-zero branch nor the resulting uncertainty was
    covered by any test.

    Behaviour is correct and pinned here: margin saturates at 1.0 and
    entropy at log2(n), both maxima, so such a sample sorts to the head of
    the review queue.
    """
    no_match = [("ebola", 0.0), ("sars_cov_2", 0.0), ("influenza_a", 0.0)]

    assert uncertainty_score(no_match, method="margin") == pytest.approx(1.0)
    assert uncertainty_score(no_match, method="entropy") == pytest.approx(math.log2(3))
    assert _normalize([0.0, 0.0, 0.0]) == [pytest.approx(1 / 3)] * 3

    strong_match = [("ebola", 0.98), ("sars_cov_2", 0.01), ("influenza_a", 0.0)]
    queue = prioritize_for_review(
        {"no_match": no_match, "strong_match": strong_match}, method="margin"
    )
    assert [qid for qid, _, _ in queue] == ["no_match", "strong_match"]


def test_negative_scores_take_the_softmax_branch_for_entropy_only():
    """Characterisation test (currently GREEN): `_normalize`'s softmax
    branch was never executed by any test. It is arithmetically correct
    (checked against a hand-written stable softmax below), but only
    `entropy` ever calls `_normalize`.

    `margin` reads the raw scores directly, so for signed inputs -- raw
    decision-function margins, log-odds -- the two methods do not merely
    disagree in scale: `[-1.0, 0.0, 1.0]` is a 1.0-wide raw gap, giving
    margin uncertainty exactly 0.0 ("perfectly confident"), while the
    softmax the same input takes for entropy is `[0.090, 0.245, 0.665]`,
    entropy 1.20 out of a maximum log2(3) = 1.58 ("quite uncertain").
    Recorded so the inconsistency is visible rather than latent.
    """
    scores = [-1.0, 0.0, 1.0]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    expected = [e / sum(exps) for e in exps]

    assert _normalize(scores) == pytest.approx(expected)

    signed = [("A", -1.0), ("B", 0.0), ("C", 1.0)]
    entropy = uncertainty_score(signed, method="entropy")
    margin = uncertainty_score(signed, method="margin")

    assert entropy == pytest.approx(-sum(p * math.log2(p) for p in expected))
    assert entropy > 0.75 * math.log2(3), "entropy calls this input uncertain"
    assert margin == pytest.approx(0.0), "margin calls the same input fully confident"


def test_unnormalised_scores_make_margin_and_entropy_disagree_on_the_ordering():
    """Characterisation test (currently GREEN): neither method validates
    that a probability vector sums to 1, and only `entropy` normalises.

    Two entries, one a confident distribution that does sum to 1
    (`[0.9, 0.1]`), the other a near-tied pair on a different scale
    (`[3.0, 4.0]`, i.e. 0.43/0.57 once normalised). Switching `method`
    reverses which one a reviewer is told to look at first. `margin` puts
    the near-tied entry *last* because its raw gap of 1.0 saturates
    `1 - margin` at 0.0.
    """
    batch = {"sums_to_one_confident": [("A", 0.9), ("B", 0.1)],
             "sums_to_seven_torn": [("A", 3.0), ("B", 4.0)]}

    by_margin = [qid for qid, _, _ in prioritize_for_review(batch, method="margin")]
    by_entropy = [qid for qid, _, _ in prioritize_for_review(batch, method="entropy")]

    assert by_margin == ["sums_to_one_confident", "sums_to_seven_torn"]
    assert by_entropy == ["sums_to_seven_torn", "sums_to_one_confident"]
    assert by_margin == list(reversed(by_entropy))


def test_entropy_is_not_comparable_across_results_with_different_lengths():
    """Characterisation test (currently GREEN): `prioritize_for_review`
    sorts raw entropies across a batch without normalising by `log2(n)`,
    so an entry with more candidates is systematically ranked as more
    uncertain.

    A five-way result with a clear 60% winner scores 1.771; a two-way
    coin flip -- the most uncertain a binary result can possibly be, at
    its log2(2) = 1.0 ceiling -- scores 1.000 and is ranked *below* it.
    Fine when every result has the same candidate count (which
    `classify(top_n=...)` gives against one reference DB); a real ordering
    hazard for the generic ranked-tuples shape the module also advertises.
    """
    batch = {
        "five_way_clear_winner": [("A", 0.6), ("B", 0.1), ("C", 0.1), ("D", 0.1), ("E", 0.1)],
        "two_way_coin_flip": [("A", 0.5), ("B", 0.5)],
    }
    queue = prioritize_for_review(batch, method="entropy")

    assert [qid for qid, _, _ in queue] == ["five_way_clear_winner", "two_way_coin_flip"]
    assert queue[0][1] == pytest.approx(1.7709505944546687)
    assert queue[1][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation: where the errors actually come from
# ---------------------------------------------------------------------------


def test_malformed_ranked_entries_report_a_misleading_probability_vector_error():
    """Characterisation test (currently GREEN): documents that
    `_scores_from_ranked_tuples`'s malformed-entry guard is unreachable.

    `_scores_from_ranked_tuples` is only ever called when
    `_is_ranked_tuples` already returned True, which guarantees every entry
    is a length-2 sequence -- so its own `"entry ... is not a (label,
    score) pair"` check can never fire. `active_learning.py`'s lines
    102-106 are dead code.

    Every malformed ranked list therefore falls through to the
    probability-vector branch and is diagnosed as a probs/labels length
    mismatch. For `[("A", 0.9), "not-a-pair"]` the message is "probs has 2
    entries but labels has 10", counting the characters of the string
    `"not-a-pair"` as class labels.

    `test_active_learning.py::test_uncertainty_score_rejects_malformed_entries`
    asserts only `pytest.raises(ValueError)`, so it passes via this path
    while reading as though it exercised the ranked-tuples guard.
    """
    with pytest.raises(ValueError) as excinfo:
        uncertainty_score([("A", 0.9), "not-a-pair"], method="margin")

    message = str(excinfo.value)
    assert "probs has 2 entries but labels has 10" in message
    assert "is not a (label, score) pair" not in message

    # A three-element entry: same story, diagnosed as a length mismatch.
    with pytest.raises(ValueError, match="probs has 3 entries but labels has 2"):
        uncertainty_score([("A", 0.9, "extra"), ("B", 0.1)], method="margin")

    # A one-entry list whose score is not numeric falls through to the
    # `len(result) != 2` guard (line 135), also previously unexecuted, and
    # is likewise reported as a probability-vector problem.
    with pytest.raises(ValueError, match=r"must be a \(probs, labels\) pair"):
        uncertainty_score([("A", "not-a-number")], method="margin")

    # A non-sequence top level takes `_is_ranked_tuples`'s `return False`
    # (line 90), also previously unexecuted.
    with pytest.raises(TypeError):
        uncertainty_score(0.5, method="margin")


def test_single_candidate_result_treats_the_absent_runner_up_as_zero():
    """`_margin_uncertainty`'s `len(ordered) > 1` guard was never
    exercised. With one candidate the runner-up is taken as 0.0, so the
    uncertainty is `1 - top1` -- meaning a single candidate scored 0.3
    (a weak lone hit) is reported as more uncertain than one scored 0.9.
    Reasonable, but undocumented and unpinned until now.
    """
    assert uncertainty_score([("only", 0.9)], method="margin") == pytest.approx(0.1)
    assert uncertainty_score([("only", 0.3)], method="margin") == pytest.approx(0.7)
    assert uncertainty_score([("only", 0.9)], method="entropy") == pytest.approx(0.0)


def test_suggest_reference_additions_is_inclusive_at_the_threshold():
    """The docstring says "entries with `uncertainty_score >=` this value";
    the existing test only checks thresholds strictly inside or outside the
    range, never one landing exactly on a score.
    """
    batch = {"exactly_0.8": [("A", 0.6), ("B", 0.4)], "below": [("A", 0.9), ("B", 0.1)]}
    queue = prioritize_for_review(batch, method="margin")
    assert queue[0][1] == pytest.approx(0.8)

    on_the_line = suggest_reference_additions(queue, uncertainty_threshold=0.8)
    assert [qid for qid, _, _ in on_the_line] == ["exactly_0.8"]

    just_above = suggest_reference_additions(queue, uncertainty_threshold=0.8000001)
    assert just_above == []
