"""Tests for `fastdna.calibration` -- turning a fitted classifier's scores
into genuine, checkable probabilities.

The centerpiece test (`test_calibrating_scm_closes_the_gap_evaluation_warns_about`)
demonstrates the actual gap this module closes end-to-end, with three
disjoint splits (train / calibrate / evaluate) so the final check is not
circular: fit `SetCoveringClassifier` on one split, show its raw
`predict_proba()` trips `UncalibratedScoresWarning` under
`fastdna.evaluation.calibration_report`, wrap it with `calibrate()` fit on a
second split, and show the wrapper's output on a *third*, still-unseen
split produces a real (non-degenerate) reliability curve with a lower Brier
score than the raw hard output.
"""
from __future__ import annotations

import warnings

import pytest

pytest.importorskip("sklearn")

np = pytest.importorskip("numpy")

from fastdna.calibration import CalibratedEstimator, calibrate
from fastdna.evaluation import UncalibratedScoresWarning, calibration_report
from fastdna.rules import SetCoveringClassifier


def _make_noisy_kmer_dataset(rng, n_samples=240, n_features=30, n_informative=4, flip_rate=0.12):
    """A binary presence matrix where `n_informative` features are each
    correlated with the label but with `flip_rate` label noise -- unlike
    `test_rules.py`'s hand-built perfectly-separable fixtures, this is
    designed so that *no* fixed rule perfectly predicts the label, which is
    what makes "fraction of positives where the rule fired" a genuinely
    informative, non-trivial number for calibration to recover instead of
    trivially 0.0 or 1.0.
    """
    y = rng.integers(0, 2, size=n_samples)
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint8)
    informative_true_rule = y.copy()
    for col in range(n_informative):
        flips = rng.random(n_samples) < flip_rate
        X[:, col] = np.where(flips, 1 - informative_true_rule, informative_true_rule)
    labels = np.array(["resistant" if v else "susceptible" for v in y])
    feature_names = [f"kmer_{i}" for i in range(n_features)]
    return X, labels, feature_names


@pytest.fixture
def three_way_split():
    rng = np.random.default_rng(20260825)
    X, y, feature_names = _make_noisy_kmer_dataset(rng)
    n = len(y)
    idx = rng.permutation(n)
    third = n // 3
    train_idx, calib_idx, eval_idx = idx[:third], idx[third : 2 * third], idx[2 * third :]
    return {
        "feature_names": feature_names,
        "train": (X[train_idx], y[train_idx]),
        "calib": (X[calib_idx], y[calib_idx]),
        "eval": (X[eval_idx], y[eval_idx]),
    }


# ---------------------------------------------------------------------------
# The headline gap-closing test
# ---------------------------------------------------------------------------


def test_calibrating_scm_closes_the_gap_evaluation_warns_about(three_way_split):
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, y_eval = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5, class_weight="balanced")
    clf.fit(X_train, y_train, feature_names=feature_names)
    # SetCoveringClassifier's positive class is classes_[1] (sorted-label
    # convention, see its docstring); calibration_report's pos_label must
    # agree with whichever class predict_proba()[:, 1] actually scores, not
    # a hardcoded string -- getting this backwards would silently compare
    # each report against the wrong class's frequency.
    pos_label = clf.classes_[1]

    # Step 1: the raw model's output is exactly the hard 0/1 the whole
    # module exists to fix, and evaluation.py's own diagnostic catches it.
    raw_proba_eval = clf.predict_proba(X_eval)[:, 1]
    assert set(np.unique(raw_proba_eval).tolist()) <= {0.0, 1.0}
    with pytest.warns(UncalibratedScoresWarning):
        raw_report = calibration_report(y_eval, raw_proba_eval, pos_label=pos_label)
    # Degenerate: at most two distinct predicted-probability bins (0 and 1).
    assert raw_report.curve.num_rows <= 2

    # Step 2: calibrate on a split the model was NOT fitted on.
    calibrated = calibrate(clf, X_calib, y_calib, method="venn_abers")
    assert isinstance(calibrated, CalibratedEstimator)
    assert calibrated.classes_[1] == pos_label

    # Step 3: evaluate the wrapper on a THIRD split, unseen by both fit()
    # and calibrate() -- this is what makes the check non-circular.
    calibrated_proba_eval = calibrated.predict_proba(X_eval)[:, 1]

    # No warning this time: the calibrated output is not two-valued (unless
    # every held-out sample where the rule fired/didn't fire happened to
    # agree unanimously, vanishingly unlikely for a noisy 240-sample cohort
    # split three ways).
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UncalibratedScoresWarning)
        calibrated_report = calibration_report(y_eval, calibrated_proba_eval, pos_label=pos_label)

    assert calibrated_report.curve.num_rows >= 2

    # The actual numeric claim: a reliability curve close to the diagonal.
    # Check it directly rather than trusting the Brier score alone -- a
    # single scalar can hide a curve that is close on average but wrong in
    # both directions.
    mean_predicted = np.asarray(calibrated_report.curve["mean_predicted_probability"])
    fraction_positive = np.asarray(calibrated_report.curve["fraction_of_positives"])
    assert np.all(np.abs(mean_predicted - fraction_positive) < 0.35)

    # And the summary Brier score genuinely improves over the raw hard
    # output on the same held-out split -- calibration is not just "less
    # degenerate", it is measurably closer to the true frequencies.
    assert calibrated_report.brier_score < raw_report.brier_score


def test_platt_also_closes_the_gap(three_way_split):
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, y_eval = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5, class_weight="balanced")
    clf.fit(X_train, y_train, feature_names=feature_names)

    calibrated = calibrate(clf, X_calib, y_calib, method="platt")
    calibrated_proba_eval = calibrated.predict_proba(X_eval)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UncalibratedScoresWarning)
        report = calibration_report(y_eval, calibrated_proba_eval, pos_label=clf.classes_[1])

    assert report.curve.num_rows >= 1


# ---------------------------------------------------------------------------
# CalibratedEstimator mechanics
# ---------------------------------------------------------------------------


def test_predict_proba_shape_and_column_order(three_way_split):
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, _ = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5).fit(X_train, y_train, feature_names=feature_names)
    calibrated = calibrate(clf, X_calib, y_calib)

    proba = calibrated.predict_proba(X_eval)
    assert proba.shape == (len(X_eval), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(calibrated.classes_) == ["resistant", "susceptible"]


def test_predict_matches_argmax_of_predict_proba(three_way_split):
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, _ = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5).fit(X_train, y_train, feature_names=feature_names)
    calibrated = calibrate(clf, X_calib, y_calib)

    proba = calibrated.predict_proba(X_eval)
    predicted = calibrated.predict(X_eval)
    assert np.array_equal(calibrated.classes_[proba.argmax(axis=1)], predicted)


def test_predict_interval_bounds_the_point_estimate(three_way_split):
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, _ = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5).fit(X_train, y_train, feature_names=feature_names)
    calibrated = calibrate(clf, X_calib, y_calib, method="venn_abers")

    p0, p1 = calibrated.predict_interval(X_eval)
    point = calibrated.predict_proba(X_eval)[:, 1]

    assert np.all(p0 <= p1 + 1e-12)
    assert np.all(p0 - 1e-9 <= point)
    assert np.all(point <= p1 + 1e-9)


def test_predict_interval_rejects_platt():
    rng = np.random.default_rng(1)
    X, y, feature_names = _make_noisy_kmer_dataset(rng, n_samples=60)
    clf = SetCoveringClassifier(max_rules=3).fit(X, y, feature_names=feature_names)
    calibrated = calibrate(clf, X, y, method="platt")

    with pytest.raises(ValueError, match="only defined for method='venn_abers'"):
        calibrated.predict_interval(X)


def test_two_hard_scm_scores_collapse_to_two_calibrated_values(three_way_split):
    """SetCoveringClassifier's raw scores take only two distinct values
    (0.0, 1.0) across an entire dataset. Venn-ABERS calibration on it should
    therefore also collapse to (at most) two distinct calibrated
    probabilities -- one for "rule fired", one for "rule didn't" -- which
    are real empirical rates, not the hard 0/1 they replace.
    """
    X_train, y_train = three_way_split["train"]
    X_calib, y_calib = three_way_split["calib"]
    X_eval, _ = three_way_split["eval"]
    feature_names = three_way_split["feature_names"]

    clf = SetCoveringClassifier(max_rules=5, class_weight="balanced").fit(
        X_train, y_train, feature_names=feature_names
    )
    calibrated = calibrate(clf, X_calib, y_calib, method="venn_abers")

    proba_eval = calibrated.predict_proba(X_eval)[:, 1]
    distinct = np.unique(np.round(proba_eval, 10))
    assert len(distinct) <= 2
    # And they are not themselves 0.0/1.0 -- the whole point is that they
    # are real, non-degenerate rates (unless the calibration split happened
    # to be unanimous for a firing state, vanishingly unlikely here).
    assert not np.all(np.isin(distinct, [0.0, 1.0]))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_method_is_rejected():
    rng = np.random.default_rng(2)
    X, y, feature_names = _make_noisy_kmer_dataset(rng, n_samples=40)
    clf = SetCoveringClassifier(max_rules=3).fit(X, y, feature_names=feature_names)

    with pytest.raises(ValueError, match="method must be one of"):
        calibrate(clf, X, y, method="not_a_method")


def test_non_binary_y_calib_is_rejected():
    rng = np.random.default_rng(3)
    X, y, feature_names = _make_noisy_kmer_dataset(rng, n_samples=40)
    clf = SetCoveringClassifier(max_rules=3).fit(X, y, feature_names=feature_names)

    y_three_class = np.array(["a", "b", "c"] * (len(y) // 3) + ["a"] * (len(y) % 3))
    with pytest.raises(ValueError, match="binary y_calib"):
        calibrate(clf, X, y_three_class, method="venn_abers")


def test_mismatched_lengths_are_rejected():
    rng = np.random.default_rng(4)
    X, y, feature_names = _make_noisy_kmer_dataset(rng, n_samples=40)
    clf = SetCoveringClassifier(max_rules=3).fit(X, y, feature_names=feature_names)

    with pytest.raises(ValueError, match="one label per calibration sample"):
        calibrate(clf, X, y[:-1], method="venn_abers")


def test_estimator_without_scoring_method_is_rejected():
    class NoScores:
        pass

    rng = np.random.default_rng(5)
    X, y, _ = _make_noisy_kmer_dataset(rng, n_samples=10)
    with pytest.raises(TypeError, match="decision_function\\(\\) or"):
        calibrate(NoScores(), X, y)
