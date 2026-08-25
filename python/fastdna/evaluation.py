"""fastdna.evaluation -- honest classifier reporting under class imbalance.

## Why this module exists

Real clinical AMR datasets are often severely imbalanced: resistance to a
reserve antibiotic might be present in only 2-5% of samples. Two problems
follow, and this module exists to make both visible rather than hidden
behind a single headline number.

First, a classifier optimized for plain accuracy can trivially "learn" to
always predict the majority class and still score well -- 97% accuracy on a
3%-prevalence phenotype is what predicting "susceptible" for everyone gets
you, for free, with zero biological signal.

Second, ROC-AUC is famously misleading under severe imbalance: because the
false-positive rate it plots is normalized by the (huge) negative class, a
classifier can rack up an impressive-looking ROC curve while its precision
on the minority class -- the fraction of "resistant" calls that are actually
resistant -- is unusable in practice. Saito T, Rehmsmeier M, "The
precision-recall plot is more informative than the ROC plot when evaluating
binary classifiers on imbalanced datasets," PLOS ONE 10(3):e0118432, 2015,
demonstrate this directly: two classifiers with visibly different
precision-recall curves on an imbalanced dataset can have near-identical ROC
curves. The precision-recall plot, and the area under it (average
precision), is the honest metric for this regime; `precision_recall_report`
packages it.

A classifier's *scores* can also be uninformative in a different way: a
model that ranks cases correctly (good AUC, good average precision) but
whose predicted "probabilities" do not track real-world frequency is not
safe to use for anything that consumes the probability itself -- risk
stratification, a clinical decision threshold, combining it with other
evidence in a Bayesian update. `calibration_report` packages a reliability
diagram and the Brier score (Brier GW, "Verification of forecasts expressed
in terms of probability," Monthly Weather Review 78(1):1-3, 1950) for that
question.

## What this module does not do

Neither function here picks a "best" operating point or threshold and
presents it as *the* answer. `precision_recall_report` returns the whole
curve plus its area; `calibration_report` returns the whole reliability
diagram plus its summary statistic. Choosing a threshold is a decision that
depends on the relative cost of a false positive versus a false negative --
a clinical or operational judgment this module cannot make on the caller's
behalf, exactly as `fastdna.cv`'s lineage-aware splitters and permutation
p-values report an honest estimate rather than a verdict, and
`fastdna.gwas.prefilter_association` is screening-only by design. Read the
curve; do not just read the summary number.

`SetCoveringClassifier.predict_proba()` (`fastdna.rules`) is a documented,
deliberate exception to "probability": it returns a hard, uncalibrated 0/1
decision reshaped to look like one. `calibration_report` does not raise on
that input -- see its docstring for why a hard error would be the wrong
call -- but it warns loudly (`UncalibratedScoresWarning`) and the resulting
reliability diagram is honestly degenerate (at most two points) rather than
silently presented as a normal calibration curve. `fastdna.calibration.
calibrate` is the fix for that gap -- wrap the fitted model, fit on
held-out data, and the reliability diagram this function draws from the
wrapper's output on a further held-out set is how the fix is checked, not
assumed (`python/tests/test_calibration.py`).

## Package convention

Like `fastdna.rules` and `fastdna.cv`, this module is imported explicitly
(`from fastdna.evaluation import precision_recall_report`) and pulls in
scikit-learn, numpy and pyarrow at module scope; plain `import fastdna` does
not pay for them. Both functions return `pyarrow.Table`-backed curves,
matching `fastdna.cv.permutation_importance_pvalues`'s convention of
structured, Arrow-native results rather than bare tuples of arrays.

## Why this is not in `fastdna.cv`

`fastdna.cv` has a specific, narrow identity: population-structure-aware,
leakage-safe *splitting and significance testing* (`LineageKFold`,
`permutation_importance_pvalues`) -- its own docstring says explicitly that
it "deliberately does not implement any of the statistics downstream of
that." Precision-recall and calibration reporting are general classifier
diagnostics that apply identically whether or not the evaluation used a
lineage-blocked split; they belong beside `cv.py` in the same
"evaluation-rigor" family, not inside a module whose whole reason for
existing is genomic population structure.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pyarrow as pa

__all__ = [
    "UncalibratedScoresWarning",
    "PrecisionRecallReport",
    "CalibrationReport",
    "precision_recall_report",
    "calibration_report",
]


class UncalibratedScoresWarning(UserWarning):
    """Raised by `calibration_report` when every score it was given is
    exactly 0.0 or 1.0.

    A distinct category (rather than a bare `UserWarning`) so a caller who
    has genuinely understood the caveat can silence *this* warning
    specifically -- `warnings.filterwarnings("ignore",
    category=UncalibratedScoresWarning)` -- without also silencing every
    other warning FastDNA might legitimately need to raise.
    """


class PrecisionRecallReport(NamedTuple):
    """What `precision_recall_report` computed.

    Attributes
    ----------
    curve : pyarrow.Table
        Columns `precision`, `recall`, `threshold`, one row per point on
        the curve, in the order `sklearn.metrics.precision_recall_curve`
        returns them. The final row's `precision` is 1.0 and `recall` is
        0.0 by construction (the empty-selection endpoint) and has no
        associated threshold, so its `threshold` is `NaN` -- check for that
        before using the column, e.g. `pyarrow.compute.drop_null` after
        casting, or a plain `numpy.isnan` mask.
    average_precision : float
        The area under the curve above (`sklearn.metrics.
        average_precision_score`), a.k.a. average precision. This is the
        number to report instead of ROC-AUC under class imbalance -- see
        the module docstring -- but it is still a single summary of the
        whole curve; look at the curve itself before trusting it, the same
        way an R^2 does not excuse skipping the residual plot.
    """

    curve: pa.Table
    average_precision: float


class CalibrationReport(NamedTuple):
    """What `calibration_report` computed.

    Attributes
    ----------
    curve : pyarrow.Table
        The reliability diagram: columns `mean_predicted_probability` and
        `fraction_of_positives`, one row per bin actually populated (a bin
        with no samples in it is dropped, matching
        `sklearn.calibration.calibration_curve`). A perfectly calibrated
        predictor has every row on the `y = x` diagonal.
    brier_score : float
        `sklearn.metrics.brier_score_loss`: the mean squared error between
        each predicted probability and the actual 0/1 outcome. Lower is
        better calibrated (0.0 is a perfect predictor); a constant
        predictor at the true base rate already beats a badly overconfident
        one, so a Brier score is only informative relative to a comparison
        -- another model, or the base-rate predictor -- not in isolation.
    """

    curve: pa.Table
    brier_score: float


def _validate_binary_y_true(y_true, caller):
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"{caller} needs a 1-D y_true, got shape {y_true.shape}")
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            f"{caller} needs binary labels (exactly two classes) in y_true, got "
            f"{len(classes)} class{'es' if len(classes) != 1 else ''}: "
            f"{classes.tolist()[:5]}{', ...' if len(classes) > 5 else ''}."
        )
    return y_true, classes


def _validate_matching_length(y_true, scores, score_name, caller):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"{caller} needs a 1-D {score_name}, got shape {scores.shape}")
    if len(scores) != len(y_true):
        raise ValueError(
            f"{caller} needs one {score_name} per label: y_true has {len(y_true)} entries "
            f"but {score_name} has {len(scores)}."
        )
    return scores


def precision_recall_report(y_true, y_score, *, pos_label=None):
    """Precision-recall curve and average precision -- the metric to trust
    over ROC-AUC when the classes are imbalanced.

    See the module docstring for why: Saito & Rehmsmeier (PLOS ONE
    10(3):e0118432, 2015) show a classifier's ROC curve can look
    indistinguishable from a much worse one's while its precision-recall
    curve reveals the difference plainly, precisely because ROC's
    false-positive rate is normalized by a majority class that swamps the
    signal a precision-recall curve reports directly.

    This function does no new numerical work of its own --
    `sklearn.metrics.precision_recall_curve` and `average_precision_score`
    already do it correctly -- its value-add is packaging the result as a
    structured, documented `PrecisionRecallReport` rather than a bare tuple
    of same-length arrays whose meaning has to be remembered separately,
    matching `fastdna.cv.permutation_importance_pvalues`'s convention.

    This function does not pick a decision threshold for you. `y_score`
    ranks examples; deciding where on the resulting curve to operate -- the
    precision/recall trade-off appropriate for a specific clinical or
    operational cost of a false positive versus a false negative -- is a
    judgment call for the caller, informed by the returned `threshold`
    column, not something this function can make on your behalf.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Exactly two distinct labels.
    y_score : array-like of shape (n_samples,)
        A ranking score: higher means more likely positive. Need not be a
        probability or lie in [0, 1] -- a raw `decision_function()` output
        works exactly as well, since only the ranking it induces matters.
    pos_label : optional
        Which of the two labels in `y_true` is "positive". Defaults to the
        larger of the two (sklearn's own convention, and the one
        `fastdna.rules.SetCoveringClassifier` follows via `classes_[1]`)
        when not given.

    Returns
    -------
    PrecisionRecallReport
    """
    y_true, classes = _validate_binary_y_true(y_true, "precision_recall_report()")
    y_score = _validate_matching_length(y_true, y_score, "y_score", "precision_recall_report()")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("precision_recall_report() requires finite y_score values (no NaN/inf).")

    label = classes[1] if pos_label is None else pos_label

    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=label)
    average_precision = average_precision_score(y_true, y_score, pos_label=label)

    # precision/recall have one more row than thresholds (the final
    # (precision=1, recall=0) point has no associated threshold); pad with
    # NaN so every curve row still has a threshold cell, documented above.
    padded_thresholds = np.concatenate([thresholds, [np.nan]])

    curve = pa.table(
        {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "threshold": padded_thresholds.tolist(),
        }
    )
    return PrecisionRecallReport(curve=curve, average_precision=float(average_precision))


def calibration_report(y_true, y_prob, *, n_bins=10, strategy="uniform", pos_label=None):
    """Reliability diagram and Brier score for probability estimates.

    A model can rank cases well (good average precision) while its
    predicted probabilities are still untrustworthy numbers -- consistently
    too confident, too timid, or shifted -- and that failure is invisible
    to `precision_recall_report`, which only cares about ranking. This
    function checks the numbers themselves: within each bin of predicted
    probability, does the actual fraction of positives match what was
    predicted? `sklearn.calibration.calibration_curve` computes the binned
    comparison and `sklearn.metrics.brier_score_loss` the single-number
    summary (mean squared error against the 0/1 outcome); this function
    packages both together, the same convention `precision_recall_report`
    and `fastdna.cv.permutation_importance_pvalues` use.

    This function does not pick or recommend a decision threshold -- a
    calibration curve is about whether the *probabilities* are trustworthy,
    not about where to threshold them into a classification, which remains
    a cost-driven judgment call for the caller.

    **Scope: this is for genuinely probabilistic estimators, not
    `SetCoveringClassifier`.** `SetCoveringClassifier.predict_proba()`
    (`fastdna.rules`) is DELIBERATELY an uncalibrated hard 0/1 decision --
    read its own docstring -- because a fitted SCM is a boolean formula
    with no notion of confidence. Calling `calibration_report` on that
    output does not raise: a hard classifier's predictions are still a
    valid degenerate case of "probability" (at most two possible values,
    0.0 and 1.0), and refusing outright would only break every pipeline
    that already knows to interpret the result carefully. What it does
    instead is warn (`UncalibratedScoresWarning`) and let the honestly
    degenerate result -- a reliability diagram with at most two points, and
    a Brier score that reduces exactly to the model's own training-set
    error rate -- speak for itself, rather than silently presenting it as
    an ordinary calibration curve. The intended callers for this function
    are genuinely probabilistic estimators used elsewhere in this ecosystem
    -- logistic regression or a boosted tree over a `KmerVectorizer` matrix,
    or `fastdna.calibration.calibrate` (Venn-ABERS by default, referenced in
    `SetCoveringClassifier.predict_proba`'s own docstring) fitted on
    held-out, lineage-blocked folds from `fastdna.cv`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Exactly two distinct labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of the positive class. Must lie in
        `[0, 1]` -- unlike `precision_recall_report`'s `y_score`, a raw
        ranking score is not accepted here, because a calibration
        statement ("40% of samples scored 0.4 were actually positive") is
        only meaningful for an actual probability.
    n_bins, strategy
        Forwarded to `sklearn.calibration.calibration_curve`
        (`strategy="uniform"`: bins of equal width in `[0, 1]`;
        `"quantile"`: bins with an equal number of samples).
    pos_label : optional
        Which of the two labels in `y_true` is "positive". Defaults to the
        larger of the two, matching `precision_recall_report`.

    Returns
    -------
    CalibrationReport
    """
    y_true, classes = _validate_binary_y_true(y_true, "calibration_report()")
    y_prob = _validate_matching_length(y_true, y_prob, "y_prob", "calibration_report()")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("calibration_report() requires finite y_prob values (no NaN/inf).")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        bad = y_prob[(y_prob < 0.0) | (y_prob > 1.0)]
        raise ValueError(
            f"calibration_report() requires y_prob values in [0, 1] (a probability), got "
            f"values outside that range, e.g. {bad[:5].tolist()}. Raw scores that are not "
            "probabilities (e.g. a decision_function() output) belong in "
            "precision_recall_report()'s y_score instead, which has no such requirement."
        )

    if np.all((y_prob == 0.0) | (y_prob == 1.0)):
        warnings.warn(
            "calibration_report() was given y_prob values that are all exactly 0.0 or "
            "1.0 -- this looks like a hard decision rule (e.g. "
            "fastdna.rules.SetCoveringClassifier.predict_proba(), which is DELIBERATELY "
            "an uncalibrated hard 0/1, see its own docstring), not a genuine probability "
            "estimate. The reliability diagram returned is real but degenerate (at most "
            "two points, at 0.0 and 1.0), and the Brier score below reduces exactly to "
            "the model's plain training-set error rate; neither says anything about "
            "calibration quality. This function is meant for estimators with real "
            "probability estimates (logistic regression, a calibrated tree/boosting "
            "model, a Venn-ABERS wrapper, ...), not a hard rule's 0/1 output.",
            UncalibratedScoresWarning,
            stacklevel=2,
        )

    label = classes[1] if pos_label is None else pos_label

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, pos_label=label
    )
    brier = brier_score_loss(y_true, y_prob, pos_label=label)

    curve = pa.table(
        {
            "mean_predicted_probability": mean_predicted.tolist(),
            "fraction_of_positives": fraction_of_positives.tolist(),
        }
    )
    return CalibrationReport(curve=curve, brier_score=float(brier))
