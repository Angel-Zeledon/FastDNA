"""fastdna.calibration -- turning a fitted classifier's scores into
genuine, checkable probabilities.

Roadmap item A1 (`docs/ml-differentiation-roadmap.md`, bucket A, rank #1):
`calibrate` was named there alongside `fastdna.cv.lineage_groups`,
`LineageKFold` and `permutation_importance_pvalues`, and is referenced from
`fastdna.rules.SetCoveringClassifier.predict_proba`'s own docstring as the
fix for that model's deliberately hard, uncalibrated 0/1 output. It lives
in its own module rather than inside either of those, for the reason
`fastdna.evaluation` already gives for not absorbing it: `fastdna.cv`'s own
docstring says it "deliberately does not implement any of the statistics
downstream of" leakage-safe splitting, and `fastdna.evaluation` only
*diagnoses* whether scores are calibrated -- it never fixes them. This
module is the fix: wrap any fitted, score-producing estimator so that
`.predict_proba()` returns numbers a clinical or operational decision can
actually consume, then use `fastdna.evaluation.calibration_report` on
genuinely held-out data to check the fix worked (see
`python/tests/test_calibration.py` for that end-to-end proof; this module
does not self-check, for the same reason `sklearn.calibration.
CalibratedClassifierCV` does not -- checking calibration quality against the
very data the calibrator was fit on is close to circular, since an
isotonic/logistic fit already interpolates that data closely by
construction. A real check needs a third, disjoint split).

## Why this matters for a hard rule model like SetCoveringClassifier

A fitted Set Covering Machine is a boolean formula: `predict_proba()`
returns exactly `[1., 0.]` or `[0., 1.]`, because the model itself has no
notion of degree. That is not a defect to calibrate away in the usual sense
-- there is no latent continuous score inside an SCM to sharpen. What
calibration *can* honestly add is this: among the held-out samples where the
rule fired, what fraction were actually positive? That empirical rate -- not
the model's own confidence, which does not exist -- is a real number a
clinician can use ("84% of held-out isolates where this rule fired were
resistant"), and it is exactly what both methods below produce when handed
a hard classifier's two-valued output: two calibrated numbers, one per
firing state, estimated from data the rule never saw.

## The two methods, and why both

- **Venn-ABERS (default, `method="venn_abers"`)**: Vovk V, Petej I, "Venn-
  Abers predictors," Uncertainty in Artificial Intelligence (UAI) 2014,
  proceedings pp. 803-812. An Inductive Venn-ABERS Predictor (IVAP) fits two
  isotonic regressions per query point -- one where the query is
  hypothetically labelled negative, one positive -- and reports both
  resulting probabilities `p0 <= p1` as a distribution-free multiprobability
  interval; `p1 / (1 - p0 + p1)` is the standard single-number point
  estimate (same reference). "Distribution-free" is the reason this is the
  default: it carries no assumption that scores are linearly separable in
  log-odds (Platt scaling's assumption) or that the calibration set is
  large (isotonic regression alone can overfit on small calibration sets;
  IVAP's two-model interval is specifically what makes the small-sample
  case honest instead of silently overconfident) -- both properties matter
  for the clinical-microbiology cohorts (tens to low hundreds of isolates)
  this module targets. `docs/ml-differentiation-roadmap.md` cites Venn-ABERS
  use in clinical microbiology in this size regime as the motivating
  precedent.
- **Platt scaling (`method="platt"`)**: Platt J, "Probabilistic Outputs for
  Support Vector Machines and Comparisons to Regularized Likelihood
  Methods," Advances in Large Margin Classifiers, MIT Press, 1999 -- a 1-D
  logistic regression of the true label on the raw score. Cheaper (one
  fitted model, not one isotonic fit per distinct query score) and the
  right choice once a calibration set is large enough that its extra
  assumption (a sigmoid is the right shape) stops being the risk it is on a
  few dozen samples. Documented as the fallback, not the default, for that
  reason.

## What this module does not do

It does not pick a calibration set for you, and it does not run any
splitting itself. `X_calib`/`y_calib` must already be held out from
whatever the estimator was fit on -- calibrating on the training set would
launder overfitting into a confident-looking probability instead of fixing
anything. Use `fastdna.cv.LineageKFold` (or a plain held-out split, for a
non-genomic-cohort estimator) to produce that held-out data; this module
takes it as given, the same separation of concerns `fastdna.evaluation`
uses.

## Package convention

Imported explicitly (`from fastdna.calibration import calibrate`);
`scikit-learn` is imported lazily inside `calibrate()`/`predict_proba()`,
not at module scope, matching `fastdna.cv` and `fastdna.rules`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import _core

__all__ = ["CalibratedEstimator", "calibrate"]

_METHODS = ("venn_abers", "platt")


def _decision_scores(estimator, X, caller):
    """A 1-D float64 ranking score per row of `X`: `decision_function()` if
    the estimator has one (it is not itself a probability, but only the
    ranking it induces matters to either calibration method here), else
    `predict_proba()[:, 1]` -- which is exactly how `SetCoveringClassifier`
    reaches this path with its hard 0/1 output, the case this module exists
    to fix.
    """
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
    elif hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(X))[:, 1]
    else:
        # No leaf in the fastdna._core exception hierarchy inherits from
        # TypeError (every leaf is a ValueError/OSError/FileNotFoundError/
        # MemoryError/RuntimeError subclass), so this stays a bare TypeError
        # rather than losing isinstance(e, TypeError) compatibility for
        # existing callers (see python/tests/test_calibration.py's
        # pytest.raises(TypeError, ...) on this exact call).
        raise TypeError(
            f"{caller} needs an estimator exposing decision_function() or "
            f"predict_proba(), but {type(estimator).__name__} has neither."
        )
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(X):
        raise _core.InvalidConfigError(
            f"{caller}: the estimator's scores have {scores.shape[0]} entries but X has "
            f"{len(X)} rows -- decision_function()/predict_proba() returned the wrong shape."
        )
    if not np.all(np.isfinite(scores)):
        raise _core.InvalidConfigError(f"{caller}: the estimator produced non-finite scores (NaN/inf).")
    return scores


def _ivap_predict(calib_scores, calib_labels01, query_scores):
    """Inductive Venn-ABERS prediction (Vovk & Petej, UAI 2014) for each of
    `query_scores`, against the fixed calibration set `(calib_scores,
    calib_labels01)`.

    For a query score `s`: fit an isotonic regression on the calibration
    set augmented with `(s, 0)`, predict at `s` to get `p0`; fit a second
    one augmented with `(s, 1)` instead, predict at `s` to get `p1`. The
    interval `[p0, p1]` is the distribution-free multiprobability estimate;
    `p1 / (1 - p0 + p1)` is the point estimate this module reports as
    `predict_proba()`.

    Grouped by distinct query score rather than looped per-row: a hard
    classifier's scores (the motivating case for this module) take only two
    distinct values across an entire dataset, so this turns what would be
    `2 * n_query` isotonic fits into at most `2 * n_unique` -- for
    `SetCoveringClassifier` specifically, exactly 4 total regardless of how
    many samples are being scored.

    Returns `(point, p0, p1)`, each a float64 array shaped like
    `query_scores`.
    """
    from sklearn.isotonic import IsotonicRegression

    calib_scores = np.asarray(calib_scores, dtype=np.float64)
    calib_labels01 = np.asarray(calib_labels01, dtype=np.float64)
    query_scores = np.asarray(query_scores, dtype=np.float64)

    unique_scores, inverse = np.unique(query_scores, return_inverse=True)
    p0_by_unique = np.empty(unique_scores.shape[0], dtype=np.float64)
    p1_by_unique = np.empty(unique_scores.shape[0], dtype=np.float64)

    for i, s in enumerate(unique_scores):
        x_aug = np.concatenate([calib_scores, [s]])

        y_aug0 = np.concatenate([calib_labels01, [0.0]])
        iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso0.fit(x_aug, y_aug0)
        p0_by_unique[i] = iso0.predict([s])[0]

        y_aug1 = np.concatenate([calib_labels01, [1.0]])
        iso1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso1.fit(x_aug, y_aug1)
        p1_by_unique[i] = iso1.predict([s])[0]

    # Isotonicity guarantees p1 >= p0, so the denominator below is > 0
    # whenever p1 > 0 or p0 < 1 -- i.e. always, except the degenerate case
    # where the calibration set and query agree on a certain outcome at
    # both hypothetical labels, handled by falling back to that certainty.
    denom = 1.0 - p0_by_unique + p1_by_unique
    point_by_unique = np.divide(
        p1_by_unique, denom, out=p1_by_unique.copy(), where=denom > 0
    )

    p0 = p0_by_unique[inverse]
    p1 = p1_by_unique[inverse]
    point = point_by_unique[inverse]
    return point, p0, p1


class CalibratedEstimator:
    """A fitted estimator wrapped so `.predict_proba()` returns genuine,
    checkable probabilities. Returned by `calibrate()`; not constructed
    directly.

    Attributes
    ----------
    estimator_ : the wrapped, original estimator, unmodified.
    classes_ : numpy.ndarray of shape (2,)
        The two class labels, sorted -- `classes_[1]` is the positive
        class, matching `SetCoveringClassifier`/scikit-learn convention.
    method : {"venn_abers", "platt"}
        Which calibration method `calibrate()` fit.
    """

    def __init__(
        self,
        estimator: Any,  # any fitted scikit-learn-style estimator; too varied to type precisely
        classes: np.ndarray,
        method: str,
        *,
        calib_scores: Optional[np.ndarray] = None,
        calib_labels01: Optional[np.ndarray] = None,
        platt: Optional[Any] = None,  # a fitted sklearn.linear_model.LogisticRegression, or None
    ) -> None:
        self.estimator_ = estimator
        self.classes_ = classes
        self.method = method
        self._calib_scores = calib_scores
        self._calib_labels01 = calib_labels01
        self._platt = platt

    def predict_proba(self, X: Any) -> np.ndarray:  # array-like or scipy sparse matrix
        """`(n_samples, 2)` calibrated probabilities, columns ordered as
        `classes_` -- unlike the wrapped estimator's own `predict_proba()`
        (which, for `SetCoveringClassifier`, is a hard 0/1), these numbers
        are fit to track the true positive rate on held-out data, and
        `fastdna.evaluation.calibration_report` on a further, disjoint
        held-out set is how that claim is checked rather than assumed.
        """
        scores = _decision_scores(self.estimator_, X, "CalibratedEstimator.predict_proba")
        if self.method == "venn_abers":
            point, _p0, _p1 = _ivap_predict(self._calib_scores, self._calib_labels01, scores)
            positive = point
        else:
            positive = self._platt.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.column_stack((1.0 - positive, positive))

    def predict_interval(
        self, X: Any  # array-like or scipy sparse matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """The Venn-ABERS multiprobability interval `(p0, p1)`, each a
        float64 array of shape `(n_samples,)`: the two isotonic-regression
        probabilities described in the module docstring, `p0 <= p1`. The
        gap `p1 - p0` is itself informative -- it shrinks as the
        calibration set grows and widens exactly where the calibration data
        gives the method the least to go on, which is the "distribution-free
        validity guarantee" Venn-ABERS is chosen for here rather than a
        single interpolated number that hides how little evidence backs it.

        Only defined for `method="venn_abers"` -- Platt scaling produces a
        single sigmoid-fit probability with no analogous interval.
        """
        if self.method != "venn_abers":
            raise _core.InvalidConfigError(
                f"predict_interval() is only defined for method='venn_abers', but this "
                f"CalibratedEstimator was fit with method={self.method!r}. Platt scaling "
                "produces a single probability with no interval to report."
            )
        scores = _decision_scores(self.estimator_, X, "CalibratedEstimator.predict_interval")
        _point, p0, p1 = _ivap_predict(self._calib_scores, self._calib_labels01, scores)
        return p0, p1

    def predict(self, X: Any) -> np.ndarray:  # array-like or scipy sparse matrix
        """The predicted label per sample (the class with the higher
        calibrated probability), taken from `classes_`.
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


def calibrate(
    estimator: Any,  # any fitted scikit-learn-style estimator; too varied to type precisely
    X_calib: Any,  # array-like or scipy sparse matrix
    y_calib: Any,  # array-like, accepted by np.asarray()
    *,
    method: str = "venn_abers",
) -> CalibratedEstimator:
    """Wraps a fitted, score-producing estimator so its probabilities are
    genuinely calibrated, using a held-out calibration set.

    See the module docstring for the two methods and why Venn-ABERS is the
    default. `X_calib`/`y_calib` must be data the estimator was **not**
    fitted on -- `fastdna.cv.LineageKFold` is how to produce a leakage-safe
    held-out split for a genomic cohort; calibrating on the training set
    would fit the calibrator to the same overfitting the estimator already
    has, not correct for it.

    Parameters
    ----------
    estimator : a fitted estimator exposing `decision_function()` or
        `predict_proba()` -- e.g. a fitted `fastdna.rules.
        SetCoveringClassifier`, or any scikit-learn-compatible classifier.
    X_calib : array-like or scipy sparse matrix of shape (n_samples, n_features)
        Held-out samples, in whatever form `estimator`'s own scoring method
        accepts.
    y_calib : array-like of shape (n_samples,)
        Exactly two distinct labels, matching `estimator.classes_`'s two
        classes.
    method : {"venn_abers", "platt"}, default "venn_abers"

    Returns
    -------
    CalibratedEstimator
    """
    if method not in _METHODS:
        raise _core.InvalidConfigError(f"method must be one of {_METHODS}, got {method!r}")

    y_calib = np.asarray(y_calib)
    if y_calib.ndim != 1:
        raise _core.InvalidConfigError(f"calibrate() needs a 1-D y_calib, got shape {y_calib.shape}")
    classes = np.unique(y_calib)
    if len(classes) != 2:
        raise _core.InvalidConfigError(
            f"calibrate() needs binary y_calib (exactly two classes), got {len(classes)} "
            f"class{'es' if len(classes) != 1 else ''}: {classes.tolist()[:5]}"
        )

    scores = _decision_scores(estimator, X_calib, "calibrate()")
    if scores.shape[0] != len(y_calib):
        raise _core.InvalidConfigError(
            f"calibrate() needs one label per calibration sample: X_calib scored "
            f"{scores.shape[0]} rows but y_calib has {len(y_calib)} labels."
        )
    if scores.shape[0] < 2:
        raise _core.InvalidConfigError(
            f"calibrate() needs at least 2 calibration samples to fit anything, got "
            f"{scores.shape[0]}."
        )

    labels01 = (y_calib == classes[1]).astype(np.float64)

    if method == "venn_abers":
        return CalibratedEstimator(
            estimator, classes, "venn_abers", calib_scores=scores, calib_labels01=labels01
        )

    from sklearn.linear_model import LogisticRegression

    platt = LogisticRegression()
    platt.fit(scores.reshape(-1, 1), labels01)
    return CalibratedEstimator(estimator, classes, "platt", platt=platt)
