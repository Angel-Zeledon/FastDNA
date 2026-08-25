"""Tests for `fastdna.evaluation` -- honest reporting under class imbalance:
precision-recall (not headline ROC-AUC) and calibration.

Two families of fixtures: numbers cross-checked directly against
`sklearn.metrics`/`sklearn.calibration` (this module's whole value-add is
packaging their output, not re-deriving the math -- see `cv.py`'s
`permutation_importance_pvalues` for the same convention), and hand-built
"obviously correct" edge cases (perfect separation, a manually-calibrated
predictor) that do not depend on trusting sklearn at all.
"""
from __future__ import annotations

import warnings

import pytest

pytest.importorskip("sklearn")

np = pytest.importorskip("numpy")

import pyarrow as pa

from fastdna.evaluation import (
    CalibrationReport,
    PrecisionRecallReport,
    UncalibratedScoresWarning,
    calibration_report,
    precision_recall_report,
)

# ---------------------------------------------------------------------------
# precision_recall_report
# ---------------------------------------------------------------------------


def test_perfect_separation_has_average_precision_one():
    # Every negative scores below every positive: the ranking is perfect, so
    # average precision -- the area under the precision-recall curve -- is
    # exactly 1.0 regardless of the actual score values. Hand-verifiable
    # without trusting sklearn's internals: at every recall level the
    # top-ranked-so-far set is entirely positives, so precision is always 1.
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.8, 0.9]

    report = precision_recall_report(y_true, y_score)

    assert isinstance(report, PrecisionRecallReport)
    assert report.average_precision == pytest.approx(1.0)


def test_worst_case_ranking_has_low_average_precision():
    # Every positive scores below every negative -- the worst possible
    # ranking. Average precision must be far below 1 (and, for this
    # balanced 2-vs-2 case, at its minimum: the two positives are found
    # only after both negatives).
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]

    report = precision_recall_report(y_true, y_score)
    assert report.average_precision < 0.6


def test_average_precision_matches_sklearn_reference():
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    report = precision_recall_report(y_true, y_score)

    assert report.average_precision == pytest.approx(average_precision_score(y_true, y_score))
    expected_precision, expected_recall, expected_thresholds = precision_recall_curve(y_true, y_score)
    assert np.allclose(report.curve.column("precision").to_numpy(), expected_precision)
    assert np.allclose(report.curve.column("recall").to_numpy(), expected_recall)
    # sklearn's thresholds array is one shorter than precision/recall (the
    # final (precision=1, recall=0) point has no associated threshold); the
    # table pads that last row so every curve point still has one row.
    got_thresholds = report.curve.column("threshold").to_numpy()
    assert len(got_thresholds) == len(expected_thresholds) + 1
    assert np.allclose(got_thresholds[: len(expected_thresholds)], expected_thresholds)
    assert np.isnan(got_thresholds[-1])


def test_curve_is_an_arrow_table_with_the_documented_columns():
    report = precision_recall_report([0, 1, 0, 1], [0.2, 0.9, 0.1, 0.6])
    assert isinstance(report.curve, pa.Table)
    assert set(report.curve.column_names) == {"precision", "recall", "threshold"}


def test_non_numeric_string_labels_are_supported_via_pos_label():
    y_true = ["susceptible", "susceptible", "resistant", "resistant"]
    y_score = [0.1, 0.4, 0.35, 0.8]

    report = precision_recall_report(y_true, y_score, pos_label="resistant")
    numeric_equivalent = precision_recall_report([0, 0, 1, 1], y_score)
    assert report.average_precision == pytest.approx(numeric_equivalent.average_precision)


def test_docstring_cites_saito_rehmsmeier_and_does_not_pick_a_threshold():
    doc = precision_recall_report.__doc__
    assert "Saito" in doc and "Rehmsmeier" in doc
    assert "PLOS ONE" in doc or "PLoS ONE" in doc
    assert "0118432" in doc
    assert "threshold" in doc.lower()


# ---------------------------------------------------------------------------
# precision_recall_report -- validation
# ---------------------------------------------------------------------------


def test_precision_recall_report_rejects_length_mismatch():
    with pytest.raises(ValueError, match="3 entries"):
        precision_recall_report([0, 1, 0], [0.1, 0.2])


def test_precision_recall_report_rejects_non_binary_y_true():
    with pytest.raises(ValueError, match="binary|two classes|2 classes"):
        precision_recall_report([0, 1, 2], [0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        precision_recall_report([1, 1, 1], [0.1, 0.5, 0.9])


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


def test_well_calibrated_predictor_has_low_brier_and_near_diagonal_curve():
    # 50 samples predicted at p=0.2, of which exactly 10 (20%) are actually
    # positive; 50 predicted at p=0.8, of which exactly 40 (80%) are
    # actually positive. Both bins match their predicted probability
    # exactly -- a textbook well-calibrated predictor, entirely
    # hand-constructed (no sklearn trust required for *this* property).
    y_prob = np.array([0.2] * 50 + [0.8] * 50)
    y_true = np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10)

    report = calibration_report(y_true, y_prob, n_bins=2)

    assert isinstance(report, CalibrationReport)
    fractions = report.curve.column("fraction_of_positives").to_numpy()
    predicted = report.curve.column("mean_predicted_probability").to_numpy()
    assert np.allclose(fractions, predicted, atol=1e-9)

    # Brier score = mean squared error between predicted probability and
    # the 0/1 outcome; computed by hand from the same construction:
    # 50 samples at (p=0.2): 10 positives contribute (1-0.2)^2, 40
    # negatives contribute (0-0.2)^2; symmetric for the p=0.8 bin.
    expected_brier = (10 * 0.8**2 + 40 * 0.2**2 + 40 * 0.2**2 + 10 * 0.8**2) / 100
    assert report.brier_score == pytest.approx(expected_brier)


def test_badly_calibrated_predictor_has_higher_brier_and_off_diagonal_curve():
    # Same actual outcomes as the well-calibrated fixture, but the model is
    # overconfident in the wrong direction on the low-probability bin: it
    # says p=0.2 for a group that is actually 80% positive, and p=0.8 for a
    # group that is actually 20% positive -- inverted confidence.
    y_prob = np.array([0.2] * 50 + [0.8] * 50)
    y_true = np.array([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40)

    good = calibration_report(
        np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10), y_prob, n_bins=2
    )
    bad = calibration_report(y_true, y_prob, n_bins=2)

    assert bad.brier_score > good.brier_score
    bad_fractions = bad.curve.column("fraction_of_positives").to_numpy()
    bad_predicted = bad.curve.column("mean_predicted_probability").to_numpy()
    assert not np.allclose(bad_fractions, bad_predicted, atol=0.1)


def test_calibration_curve_matches_sklearn_reference():
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < y_prob).astype(int)

    report = calibration_report(y_true, y_prob, n_bins=5)
    expected_true, expected_pred = calibration_curve(y_true, y_prob, n_bins=5)

    assert np.allclose(report.curve.column("fraction_of_positives").to_numpy(), expected_true)
    assert np.allclose(report.curve.column("mean_predicted_probability").to_numpy(), expected_pred)
    assert report.brier_score == pytest.approx(brier_score_loss(y_true, y_prob))


def test_docstring_scopes_predict_proba_of_hard_classifiers_and_avoids_threshold_picking():
    doc = calibration_report.__doc__
    assert "SetCoveringClassifier" in doc
    assert "threshold" in doc.lower()


# ---------------------------------------------------------------------------
# calibration_report -- the SetCoveringClassifier degenerate case
# ---------------------------------------------------------------------------


def test_hard_zero_one_scores_warn_and_produce_a_degenerate_two_point_curve():
    sklearn_rules = pytest.importorskip("fastdna.rules")
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
    y = np.array([1, 1, 0, 0])
    clf = sklearn_rules.SetCoveringClassifier().fit(X, y)
    proba = clf.predict_proba(X)[:, 1]  # hard 0.0/1.0 -- see predict_proba's own docstring

    with pytest.warns(UncalibratedScoresWarning):
        report = calibration_report(y, proba, n_bins=10)

    # A hard 0/1 predictor can only ever populate (at most) two probability
    # bins -- at 0.0 and at 1.0 -- regardless of how many bins were asked
    # for; the result is real, just degenerate, and honestly labeled by the
    # warning rather than silently presented as a normal reliability curve.
    assert len(report.curve) <= 2


def test_calibration_report_does_not_warn_for_genuine_probabilities():
    y_true = np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10)
    y_prob = np.array([0.2] * 50 + [0.8] * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UncalibratedScoresWarning)
        calibration_report(y_true, y_prob, n_bins=2)  # must not raise


# ---------------------------------------------------------------------------
# calibration_report -- validation
# ---------------------------------------------------------------------------


def test_calibration_report_rejects_length_mismatch():
    with pytest.raises(ValueError, match="3 entries"):
        calibration_report([0, 1, 0], [0.1, 0.2], n_bins=2)


def test_calibration_report_rejects_probabilities_outside_zero_one():
    with pytest.raises(ValueError, match=r"\[0, ?1\]|between 0 and 1"):
        calibration_report([0, 1, 0, 1], [0.1, 0.5, -0.2, 1.3])


def test_calibration_report_rejects_non_finite_probabilities():
    with pytest.raises(ValueError):
        calibration_report([0, 1, 0, 1], [0.1, 0.5, float("nan"), 0.9])


def test_calibration_report_rejects_non_binary_y_true():
    with pytest.raises(ValueError, match="binary|two classes|2 classes"):
        calibration_report([0, 1, 2, 0], [0.1, 0.5, 0.9, 0.2])
