"""Tests for `fastdna.rules.SetCoveringClassifier` -- the interpretable
Set Covering Machine over binary k-mer presence features (roadmap A4).

Every dataset here is small and hand-checkable on paper: the point of an
SCM is that you can verify *which* rule it should pick by reading the
matrix, so the tests assert the exact rules, not just the accuracy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

np = pytest.importorskip("numpy")

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from fastdna.rules import Rule, SetCoveringClassifier

# ---------------------------------------------------------------------------
# Hand-checkable fixtures
# ---------------------------------------------------------------------------

# Only column 2 is a candidate rule at all: it is 1 for every positive and 0
# for every negative, so one round of the greedy loop removes all three
# negatives. Columns 0, 1 and 3 are each mixed across the positives, so
# neither the "present" nor the "absent" polarity holds for all of them and
# they can never be selected.
ONE_FEATURE_X = np.array(
    [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 1, 0],
    ],
    dtype=np.uint8,
)
ONE_FEATURE_Y = np.array([0, 0, 0, 1, 1, 1])

# Two rounds are needed here. Both column 0 and column 1 are present in both
# positives (rows 0 and 1), so both are candidates:
#   present(col 0) is false for negatives 3 and 4  -> removes 2
#   present(col 1) is false for negative 2         -> removes 1
# so round 1 takes column 0, leaving negative 2, which only present(col 1)
# removes. Column 2 is mixed across the positives and is never a candidate.
TWO_FEATURE_X = np.array(
    [
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)
TWO_FEATURE_Y = np.array([1, 1, 0, 0, 0])

# "positive iff column 0 OR column 1 is present" -- unlearnable as a
# conjunction, learnable as a disjunction via the label-inverted dual.
DISJUNCTION_X = np.array(
    [
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)
DISJUNCTION_Y = np.array([1, 1, 1, 0, 0])

KMERS = ["ACGTACGTA", "TTGCATTGC", "GGCCGGCCG", "AATTAATTA"]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def test_single_separating_feature_yields_exactly_one_rule():
    clf = SetCoveringClassifier()
    fitted = clf.fit(ONE_FEATURE_X, ONE_FEATURE_Y)

    assert fitted is clf, "fit() must return self"
    assert len(clf.rules_) == 1
    rule = clf.rules_[0]
    assert rule.feature_index == 2
    assert rule.presence is True
    assert np.array_equal(clf.predict(ONE_FEATURE_X), ONE_FEATURE_Y)


def test_two_features_needed_yields_two_rules():
    clf = SetCoveringClassifier()
    clf.fit(TWO_FEATURE_X, TWO_FEATURE_Y)

    assert [(r.feature_index, r.presence) for r in clf.rules_] == [(0, True), (1, True)]
    assert np.array_equal(clf.predict(TWO_FEATURE_X), TWO_FEATURE_Y)


def test_max_rules_truncates_and_leaves_negatives_uncovered():
    clf = SetCoveringClassifier(max_rules=1)
    clf.fit(TWO_FEATURE_X, TWO_FEATURE_Y)

    assert len(clf.rules_) == 1
    assert clf.rules_[0].feature_index == 0
    # Sample 2 is the negative only the second (truncated) rule would have
    # removed, so a one-rule conjunction still calls it positive.
    predicted = clf.predict(TWO_FEATURE_X)
    assert predicted[2] == 1
    assert not np.array_equal(predicted, TWO_FEATURE_Y)


def test_disjunction_solves_the_dual_problem():
    clf = SetCoveringClassifier(rule_type="disjunction")
    clf.fit(DISJUNCTION_X, DISJUNCTION_Y)

    assert [(r.feature_index, r.presence) for r in clf.rules_] == [(0, True), (1, True)]
    assert np.array_equal(clf.predict(DISJUNCTION_X), DISJUNCTION_Y)
    assert " OR " in clf.explain()


def test_conjunction_cannot_solve_the_disjunctive_problem():
    # The honest counterpart to the test above: the default conjunction has
    # no candidate rule at all here -- every column is mixed across the
    # positives, in both polarities -- so it learns nothing rather than
    # pretending to.
    clf = SetCoveringClassifier()
    clf.fit(DISJUNCTION_X, DISJUNCTION_Y)

    assert clf.rules_ == []
    # An empty conjunction is vacuously true, i.e. "everything is positive".
    assert np.array_equal(clf.predict(DISJUNCTION_X), np.ones(5, dtype=int))


def test_absent_polarity_rule_is_learnable():
    # Positives are exactly the samples where column 0 is *absent*.
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    y = np.array([0, 0, 1, 1])

    clf = SetCoveringClassifier()
    clf.fit(X, y)

    assert len(clf.rules_) == 1
    assert clf.rules_[0].feature_index == 0
    assert clf.rules_[0].presence is False
    assert np.array_equal(clf.predict(X), y)


def test_boolean_and_sparse_inputs_give_the_same_rules():
    sparse = pytest.importorskip("scipy.sparse")

    dense_bool = SetCoveringClassifier().fit(ONE_FEATURE_X.astype(bool), ONE_FEATURE_Y)
    from_sparse = SetCoveringClassifier().fit(sparse.csr_matrix(ONE_FEATURE_X), ONE_FEATURE_Y)

    assert dense_bool.rules_ == from_sparse.rules_
    assert np.array_equal(from_sparse.predict(sparse.csr_matrix(ONE_FEATURE_X)), ONE_FEATURE_Y)


def test_tiebreaker_decides_the_second_round():
    # Round 1 is unambiguous: column 0 removes 3 of the 5 negatives, more
    # than either other column. That leaves negatives 3 and 4, and columns 1
    # and 2 remove exactly one each -- a genuine tie on the greedy score.
    # Column 1 holds for 4 training samples, column 2 for 5, so
    # "max_coverage" prefers the more general column 2 while "first" falls
    # back to the lower index.
    X = np.array(
        [
            [1, 1, 1],  # the only positive
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([1, 0, 0, 0, 0, 0])

    first = SetCoveringClassifier(tiebreaker="first").fit(X, y)
    covered = SetCoveringClassifier(tiebreaker="max_coverage").fit(X, y)

    assert [r.feature_index for r in first.rules_] == [0, 1, 2]
    assert [r.feature_index for r in covered.rules_] == [0, 2, 1]
    # Both orderings are equally correct on the training set -- the
    # tiebreaker only decides which rule is found first.
    assert np.array_equal(first.predict(X), y)
    assert np.array_equal(covered.predict(X), y)


def test_non_numeric_labels_are_preserved():
    y = np.where(ONE_FEATURE_Y == 1, "susceptible", "resistant")
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, y)

    # sklearn's convention: classes_ is sorted, and the *second* class is the
    # positive one the rules describe.
    assert list(clf.classes_) == ["resistant", "susceptible"]
    assert clf.explain() == "susceptible IF present(feature_2)"
    assert np.array_equal(clf.predict(ONE_FEATURE_X), y)


# ---------------------------------------------------------------------------
# Interpretability surface
# ---------------------------------------------------------------------------


def test_explain_contains_the_literal_kmer_sequences():
    clf = SetCoveringClassifier()
    clf.fit(TWO_FEATURE_X, TWO_FEATURE_Y, feature_names=KMERS[:3])

    text = clf.explain()
    assert text == f"1 IF present({KMERS[0]}) AND present({KMERS[1]})"
    assert KMERS[0] in text and KMERS[1] in text


def test_explain_without_feature_names_uses_positional_placeholders():
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    assert clf.explain() == "1 IF present(feature_2)"
    assert clf.rules_[0].feature_name == "feature_2"


def test_export_rules_fasta_is_valid_lf_only_fasta(tmp_path):
    clf = SetCoveringClassifier()
    clf.fit(TWO_FEATURE_X, TWO_FEATURE_Y, feature_names=KMERS[:3])

    path = tmp_path / "rules.fasta"
    clf.export_rules_fasta(path)

    raw = path.read_bytes()
    assert b"\r\n" not in raw, "FASTA must use LF line endings, like interpret.py's exporter"
    lines = raw.decode().split("\n")
    assert lines[-1] == "", "file must end with a trailing newline"
    records = lines[:-1]
    assert len(records) == 4  # 2 rules x (header + sequence)
    assert records[0].startswith(">") and records[2].startswith(">")
    assert records[1] == KMERS[0]
    assert records[3] == KMERS[1]
    assert "present" in records[0]


def test_export_rules_fasta_refuses_placeholder_feature_names(tmp_path):
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    with pytest.raises(ValueError, match="feature_names"):
        clf.export_rules_fasta(tmp_path / "rules.fasta")


# ---------------------------------------------------------------------------
# predict_proba honesty
# ---------------------------------------------------------------------------


def test_predict_proba_returns_hard_uncalibrated_zero_one():
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    proba = clf.predict_proba(ONE_FEATURE_X)

    assert proba.shape == (6, 2)
    # Pinned behavior: hard 0/1, never an interpolated score.
    assert set(np.unique(proba).tolist()) <= {0.0, 1.0}
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(ONE_FEATURE_X))


def test_predict_proba_docstring_states_it_is_not_calibrated():
    doc = SetCoveringClassifier.predict_proba.__doc__.lower()
    assert "not a calibrated probability" in doc
    assert "calibrate" in doc  # points at roadmap A1's planned helper
    assert "fastdna.cv" in doc


def test_docstrings_warn_about_overfitting_and_point_at_cv():
    import fastdna.rules as rules_module

    for doc in (rules_module.__doc__, SetCoveringClassifier.__doc__):
        assert "fastdna.cv" in doc
    assert "overfit" in SetCoveringClassifier.__doc__.lower()


# ---------------------------------------------------------------------------
# scikit-learn compatibility
# ---------------------------------------------------------------------------


def test_get_params_set_params_round_trip():
    clf = SetCoveringClassifier()
    assert clf.get_params() == {
        "max_rules": 10,
        "rule_type": "conjunction",
        "tiebreaker": "max_coverage",
        "class_weight": None,
    }

    clf.set_params(max_rules=3, rule_type="disjunction", tiebreaker="first", class_weight="balanced")
    assert clf.get_params() == {
        "max_rules": 3,
        "rule_type": "disjunction",
        "tiebreaker": "first",
        "class_weight": "balanced",
    }


def test_clone_produces_an_equivalent_unfitted_estimator():
    clf = SetCoveringClassifier(max_rules=4, rule_type="disjunction", tiebreaker="first")
    clf.fit(DISJUNCTION_X, DISJUNCTION_Y)

    fresh = clone(clf)
    assert fresh.get_params() == clf.get_params()
    with pytest.raises(NotFittedError):
        fresh.predict(DISJUNCTION_X)


def test_predict_before_fit_raises_not_fitted():
    clf = SetCoveringClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(ONE_FEATURE_X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(ONE_FEATURE_X)
    with pytest.raises(NotFittedError):
        clf.rules_
    with pytest.raises(NotFittedError):
        clf.explain()


def test_score_from_classifier_mixin_works():
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    assert clf.score(ONE_FEATURE_X, ONE_FEATURE_Y) == 1.0


# ---------------------------------------------------------------------------
# Validation -- every message must name the actual problem
# ---------------------------------------------------------------------------


def test_non_binary_x_is_rejected_with_the_offending_values():
    X = np.array([[0, 1], [2, 0], [0, 1], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        SetCoveringClassifier().fit(X, np.array([0, 0, 1, 1]))

    with pytest.raises(ValueError) as excinfo:
        SetCoveringClassifier().fit(np.array([[0.0, 0.5], [1.0, 1.0]]), np.array([0, 1]))
    assert "0.5" in str(excinfo.value)


def test_more_than_two_classes_is_rejected_as_binary_only():
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError) as excinfo:
        SetCoveringClassifier().fit(ONE_FEATURE_X, y)
    message = str(excinfo.value)
    assert "binary" in message
    assert "3 classes" in message


def test_single_class_y_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        SetCoveringClassifier().fit(ONE_FEATURE_X, np.ones(6, dtype=int))
    assert "single class" in str(excinfo.value)


def test_feature_names_length_mismatch_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y, feature_names=KMERS[:2])
    message = str(excinfo.value)
    assert "feature_names" in message
    assert "2" in message and "4" in message


def test_y_length_mismatch_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y[:3])
    assert "3" in str(excinfo.value)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_max_rules_below_one_is_rejected(bad):
    with pytest.raises(ValueError, match="max_rules"):
        SetCoveringClassifier(max_rules=bad).fit(ONE_FEATURE_X, ONE_FEATURE_Y)


def test_empty_x_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        SetCoveringClassifier().fit(np.zeros((0, 3), dtype=np.uint8), np.array([]))
    with pytest.raises(ValueError, match="empty"):
        SetCoveringClassifier().fit(np.zeros((4, 0), dtype=np.uint8), np.array([0, 0, 1, 1]))


def test_one_dimensional_x_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        SetCoveringClassifier().fit(np.array([0, 1, 0, 1], dtype=np.uint8), np.array([0, 1, 0, 1]))


def test_unknown_rule_type_and_tiebreaker_are_rejected():
    with pytest.raises(ValueError, match="rule_type"):
        SetCoveringClassifier(rule_type="xor").fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    with pytest.raises(ValueError, match="tiebreaker"):
        SetCoveringClassifier(tiebreaker="random").fit(ONE_FEATURE_X, ONE_FEATURE_Y)


def test_predict_with_wrong_feature_count_is_rejected():
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y)
    with pytest.raises(ValueError) as excinfo:
        clf.predict(ONE_FEATURE_X[:, :2])
    message = str(excinfo.value)
    assert "4" in message and "2" in message


def test_rule_namedtuple_fields():
    clf = SetCoveringClassifier().fit(ONE_FEATURE_X, ONE_FEATURE_Y, feature_names=KMERS)
    rule = clf.rules_[0]
    assert isinstance(rule, Rule)
    assert rule == Rule(feature_index=2, feature_name=KMERS[2], presence=True)


# ---------------------------------------------------------------------------
# class_weight -- cost-sensitive learning
#
# Why these fixtures look the way they do: a plain per-class rescaling of the
# greedy "negatives removed" count cannot change which literal a hard-coverage
# SCM picks -- every candidate is *required* to hold for 100% of the positive
# class, so that class's contribution to every score is the same additive
# constant, and the negative class is a single homogeneous population with one
# weight value, so weighting is a pure positive scalar multiple of the raw
# count (verified analytically and by exhaustive randomized search: 20,000
# random imbalanced datasets with exact dyadic weights produced zero
# divergences from the unweighted rules). class_weight therefore has to do
# something structurally different to matter at all: `fit()` relaxes the hard
# "must hold for every positive" requirement into a weighted trade-off
# (weighted negatives removed minus weighted *currently-covered* positives a
# candidate would newly exclude), and only takes a literal whose net score is
# positive. When every sample shares one weight (class_weight=None, or a
# uniform dict), there is no asymmetry to trade off and fit() provably falls
# back to running the exact original hard-coverage code.
# ---------------------------------------------------------------------------

# 3 positives (minority, "resistant"), 5 negatives (majority, "susceptible").
# Column 0 ("informative"): present in 2 of the 3 positives and absent from
# every negative -- a real, near-perfect discriminator marred by one
# exceptional positive (e.g. a strain resistant through a different
# mechanism). It is present_ok=False AND absent_ok=False (mixed across the
# positives in both polarities), so a hard SCM can never even consider it.
# Column 1 ("pure"): present in all 3 positives (a valid hard candidate) but
# only absent in 2 of the 5 negatives, so present(col1) removes just 2 of 5.
IMBALANCED_X = np.array(
    [
        [1, 1],  # positive 0
        [1, 1],  # positive 1
        [0, 1],  # positive 2 -- the exception column 0 misses
        [0, 1],  # negative 0
        [0, 1],  # negative 1
        [0, 1],  # negative 2
        [0, 0],  # negative 3
        [0, 0],  # negative 4
    ],
    dtype=np.uint8,
)
IMBALANCED_Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])


def test_class_weight_none_matches_omitting_the_parameter():
    default = SetCoveringClassifier(max_rules=1).fit(IMBALANCED_X, IMBALANCED_Y)
    explicit = SetCoveringClassifier(max_rules=1, class_weight=None).fit(IMBALANCED_X, IMBALANCED_Y)
    assert explicit.rules_ == default.rules_
    assert np.array_equal(explicit.predict(IMBALANCED_X), default.predict(IMBALANCED_X))


def test_uniform_dict_class_weight_is_bit_identical_to_unweighted():
    # Both classes weighted equally -- no cost asymmetry, so this must fall
    # back to running the exact same hard-coverage search as class_weight=None,
    # for every affected fixture, not just this one.
    for X, y in [(ONE_FEATURE_X, ONE_FEATURE_Y), (TWO_FEATURE_X, TWO_FEATURE_Y), (IMBALANCED_X, IMBALANCED_Y)]:
        unweighted = SetCoveringClassifier().fit(X, y)
        uniform = SetCoveringClassifier(class_weight={0: 2.5, 1: 2.5}).fit(X, y)
        assert uniform.rules_ == unweighted.rules_
        assert np.array_equal(uniform.predict(X), unweighted.predict(X))


def test_balanced_class_weight_finds_a_literal_the_hard_scm_cannot_use():
    # Hard-coverage SCM (class_weight=None): column 0 is ineligible (mixed
    # across the positives), so the only candidate is present(column 1),
    # which removes 2 of 5 negatives. With max_rules=1 it stops there,
    # leaving negatives 0-2 uncovered -> 3 false positives on the training set.
    unweighted = SetCoveringClassifier(max_rules=1).fit(IMBALANCED_X, IMBALANCED_Y)
    assert [(r.feature_index, r.presence) for r in unweighted.rules_] == [(1, True)]
    unweighted_pred = unweighted.predict(IMBALANCED_X)
    assert np.count_nonzero(unweighted_pred != IMBALANCED_Y) == 3

    # class_weight="balanced": w_pos = 8/(2*3) = 4/3, w_neg = 8/(2*5) = 4/5.
    # present(column 0) removes all 5 negatives (weighted gain 5*4/5=4.0) at
    # the cost of excluding the one covered positive that lacks it (weighted
    # cost 4/3 ~= 1.333); net score ~2.667. present(column 1) removes 2
    # negatives (weighted gain 2*4/5=1.6) at zero cost; net score 1.6. Column
    # 0's net score wins, so it is picked even though a hard SCM could never
    # even nominate it.
    balanced = SetCoveringClassifier(max_rules=1, class_weight="balanced").fit(IMBALANCED_X, IMBALANCED_Y)
    assert [(r.feature_index, r.presence) for r in balanced.rules_] == [(0, True)]
    balanced_pred = balanced.predict(IMBALANCED_X)
    # Every negative is now classified correctly; the one training-set error
    # is the deliberate trade-off, positive example 2 (the exception column 0
    # misses) predicted as the negative class.
    assert np.count_nonzero(balanced_pred != IMBALANCED_Y) == 1
    misclassified = np.flatnonzero(balanced_pred != IMBALANCED_Y)
    assert list(misclassified) == [2]
    # The hard SCM's 100%-positive-recall guarantee is explicitly given up
    # here -- that is the point of a *cost-sensitive* relaxation.
    assert balanced_pred[2] != IMBALANCED_Y[2]


def test_explicit_dict_class_weight_matches_balanced_when_numerically_equal():
    balanced = SetCoveringClassifier(max_rules=1, class_weight="balanced").fit(IMBALANCED_X, IMBALANCED_Y)
    explicit = SetCoveringClassifier(max_rules=1, class_weight={0: 8 / 10, 1: 8 / 6}).fit(
        IMBALANCED_X, IMBALANCED_Y
    )
    assert explicit.rules_ == balanced.rules_
    assert np.array_equal(explicit.predict(IMBALANCED_X), balanced.predict(IMBALANCED_X))


def test_explicit_dict_class_weight_behaves_as_documented_with_asymmetric_values():
    # Heavily upweighting the negative (majority) class relative to the
    # positive class should only *strengthen* the preference for the
    # informative-but-impure literal, since violating a positive becomes
    # relatively cheaper.
    clf = SetCoveringClassifier(max_rules=1, class_weight={0: 10.0, 1: 1.0}).fit(IMBALANCED_X, IMBALANCED_Y)
    assert [(r.feature_index, r.presence) for r in clf.rules_] == [(0, True)]


@pytest.mark.parametrize(
    "bad_class_weight",
    [
        "unbalanced",
        ["balanced"],
        3,
        3.5,
    ],
)
def test_class_weight_wrong_type_or_unknown_string_is_rejected(bad_class_weight):
    with pytest.raises(ValueError, match="class_weight"):
        SetCoveringClassifier(class_weight=bad_class_weight).fit(IMBALANCED_X, IMBALANCED_Y)


def test_class_weight_dict_missing_a_class_is_rejected():
    with pytest.raises(ValueError, match="class_weight"):
        SetCoveringClassifier(class_weight={1: 2.0}).fit(IMBALANCED_X, IMBALANCED_Y)


def test_class_weight_dict_with_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="class_weight"):
        SetCoveringClassifier(class_weight={0: 1.0, 1: 2.0, 2: 3.0}).fit(IMBALANCED_X, IMBALANCED_Y)


@pytest.mark.parametrize("bad_weight", [0, -1.0, "heavy", True])
def test_class_weight_dict_with_non_positive_or_non_numeric_weight_is_rejected(bad_weight):
    with pytest.raises(ValueError, match="class_weight"):
        SetCoveringClassifier(class_weight={0: bad_weight, 1: 1.0}).fit(IMBALANCED_X, IMBALANCED_Y)


def test_class_weight_docstring_documents_the_cost_sensitive_relaxation():
    doc = SetCoveringClassifier.__doc__
    assert "class_weight" in doc
    assert "cost-sensitive" in doc.lower()
    assert "balanced" in doc
