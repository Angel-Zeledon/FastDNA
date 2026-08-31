"""fastdna.mic -- log2(MIC) regression for antimicrobial resistance.

## Status: frozen

Per `docs/audit/PLAN.md` §2 ("Qué se poda"), this module is frozen: stable,
not accepting new features, and a candidate for extraction into a separate
`fastdna-contrib` package in a future release. Freezing is not deleting --
see that section for the full reasoning behind the boundary.

## Why this module exists

Minimum inhibitory concentration (MIC) is the standard antimicrobial
susceptibility phenotype: the lowest antibiotic concentration that
inhibits visible growth, read off a two-fold dilution series (e.g. 0.25,
0.5, 1, 2, 4, 8, 16 mg/L). Two consequences follow directly from that assay
design, and both are why this module exists rather than "just call
`sklearn.linear_model.Ridge` on the raw values":

1. **Raw MIC is not a linear scale.** The dilution series is geometric, so
   the gap between 1 and 2 mg/L is the same *biological* step (one
   dilution) as the gap between 64 and 128 mg/L, even though the second
   gap is 64x larger in raw units. Fitting a regressor directly on raw MIC
   values lets the largest concentrations dominate the loss purely because
   of their scale, not because they are more informative. `log2` turns the
   dilution series into evenly spaced integers, which is what a standard
   regression loss (squared error, absolute error) actually assumes when it
   treats every unit of error the same. This is standard practice in AMR
   genomics ML -- e.g. the PATRIC/BV-BRC and CRyPTIC AMR modelling
   literature reports log2(MIC) as the regression target, not raw MIC.
2. **The field's own accuracy standard is dilution-based, not
   error-based.** Clinical microbiology does not ask "how close in mg/L",
   it asks "how many two-fold dilution steps off". CLSI and FDA guidance
   for evaluating a susceptibility test method (an antimicrobial
   susceptibility testing device, or here, a genomic MIC predictor) against
   a reference method uses **essential agreement (EA)**: the fraction of
   predictions within +/-1 doubling dilution of the reference value. A
   model can have an unremarkable R^2 and still be clinically useful if its
   errors are almost all within one dilution step -- and, symmetrically, a
   model can have a deceptively high R^2 driven by a few widely-spread
   high-MIC outliers while missing the dilution step on most samples. This
   module reports EA alongside R^2/MAE precisely so neither reads the
   result alone.

## What this module provides

`log2_mic`
    Validates raw MIC values (must be strictly positive and finite -- see
    below) and returns their log2 transform. The one piece of arithmetic
    every other function in this module is built from.
`MicRegressor`
    A thin scikit-learn-compatible wrapper: fits any scikit-learn
    regressor on `log2_mic(y)` instead of raw `y`, and un-transforms
    (`2**prediction`) back to MIC units on `.predict()`. Does not vectorize
    FASTQ files itself -- see "Where the feature matrix comes from" below.
`mic_regression_report`
    Packages R^2, MAE (both computed in log2 space, per point 1 above) and
    essential agreement into one report, following the same
    "report-a-structured-result" convention as
    `fastdna.evaluation.precision_recall_report` /
    `fastdna.evaluation.calibration_report`. Like those two functions, it
    does not fit anything -- it takes already-predicted MIC values and
    reports on them, so it works equally well on `MicRegressor`'s output or
    on predictions from an entirely different pipeline.

## Where the feature matrix comes from

This module does not read FASTQ files or build a k-mer feature matrix
itself -- `fastdna.sklearn.KmerVectorizer` already does that, correctly
(leakage-safe vocabulary selection, see its own module docstring), and
duplicating it here would be a second, divergent implementation of the
same thing. `MicRegressor` is built to sit immediately downstream of it,
in a plain scikit-learn pipeline::

    from fastdna.sklearn import KmerVectorizer
    from fastdna.mic import MicRegressor, mic_regression_report
    from sklearn.linear_model import Ridge

    vectorizer = KmerVectorizer(k=31, top_features=10_000)
    X_train = vectorizer.fit_transform(train_paths)
    X_test = vectorizer.transform(test_paths)

    model = MicRegressor(Ridge(alpha=1.0)).fit(X_train, train_mic)
    predicted_mic = model.predict(X_test)

    report = mic_regression_report(test_mic, predicted_mic)
    print(report.r2, report.mae_log2, report.essential_agreement)

`X` can equally well be a `scipy.sparse` matrix from
`fastdna.gwas.cohort_presence_matrix`, or any other `(n_samples,
n_features)` matrix -- `MicRegressor` never looks at where `X` came from,
only at `y`.

## Population structure -- read this before trusting a headline number

AMR MIC is exactly the phenotype `docs/ml-differentiation-roadmap.md`
warns about: it is strongly correlated with lineage (a resistant clone
carries its whole accessory genome along with the resistance
determinant), so a random train/test split routinely scatters
near-identical genomes across the boundary and inflates every metric this
module reports. Evaluate with `fastdna.cv.LineageKFold` (built from
`fastdna.cv.lineage_groups`, itself built on `fastdna.compare_all`'s Mash
distances) rather than a plain random split whenever the cohort has any
clonal structure -- which, for bacterial AMR, is essentially always.
`mic_regression_report` computes an honest report of whatever predictions
it is given; it has no way to know whether those predictions came from a
leakage-safe split, exactly as `fastdna.gwas.prefilter_association` cannot
tell a real association from an uncorrected confound.

## Optional dependencies

`numpy` is imported at module scope (this module cannot do anything
without it); `scikit-learn` is imported lazily inside `MicRegressor.fit`
and `mic_regression_report`, matching `fastdna.calibration` and
`fastdna.cv`'s convention -- so `from fastdna.mic import log2_mic` alone
never requires scikit-learn to be installed.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np
import pyarrow as pa

from . import _core

__all__ = [
    "log2_mic",
    "MicRegressor",
    "MicRegressionReport",
    "mic_regression_report",
]


def log2_mic(mic_values: np.ndarray) -> np.ndarray:  # mic_values is array-like, coerced via np.asarray
    """Validates raw MIC values and returns their log2 transform.

    MIC values come from a two-fold dilution series, so log2 turns them
    into evenly spaced regression targets -- see the module docstring for
    why that matters. This function's entire job is failing loudly on the
    one thing that makes log2 undefined or silently wrong: a non-positive
    value. A real dilution assay never produces a MIC of exactly 0 or a
    negative concentration, so a 0/negative entry here is a data error
    (an unencoded "no growth at any tested dilution" sentinel, a unit
    conversion bug, ...), not a value log2 should be asked to swallow --
    `log2(0)` is `-inf` and `log2` of a negative number is complex/NaN,
    either of which would propagate silently into every downstream fit and
    metric.

    Parameters
    ----------
    mic_values : array-like of shape (n_samples,)
        Raw, positive MIC values, any concentration unit (mg/L, ug/mL, ...)
        as long as it is consistent across the whole cohort being compared.

    Returns
    -------
    numpy.ndarray of float64
        `log2(mic_values)`, same shape as the input.

    Raises
    ------
    ValueError
        If `mic_values` is not 1-D, is empty, contains a non-finite value
        (`NaN`/`inf`), or contains a value `<= 0`.
    """
    values = np.asarray(mic_values, dtype=np.float64)
    if values.ndim != 1:
        raise _core.InvalidConfigError(f"log2_mic() needs a 1-D array of MIC values, got shape {values.shape}")
    if values.size == 0:
        raise _core.InvalidConfigError("log2_mic() received an empty array of MIC values.")
    if not np.all(np.isfinite(values)):
        bad_count = int((~np.isfinite(values)).sum())
        raise _core.InvalidConfigError(
            f"log2_mic() requires finite MIC values (no NaN/inf), got {bad_count} non-finite "
            "value(s)."
        )
    if np.any(values <= 0):
        bad = values[values <= 0]
        raise _core.InvalidConfigError(
            f"log2_mic() requires strictly positive MIC values (log2 of zero or a negative value "
            f"is undefined), got {bad[:5].tolist()}{', ...' if bad.size > 5 else ''}. A MIC of 0 "
            "usually means 'no growth at the lowest tested dilution' -- encode it as that dilution's "
            "concentration (or half of it, a common lab convention for off-scale-low readings), not "
            "literally 0. A MIC reported as '>=X' (off-scale-high) should similarly be encoded as X, "
            "not left as a string or a sentinel."
        )
    return np.log2(values)


class MicRegressor:
    """Fits a scikit-learn regressor on `log2(MIC)` instead of raw MIC, and
    un-transforms predictions back to MIC units.

    A thin wrapper, not a new model: all of the actual regression is done
    by whatever `estimator` is passed in (or the default). Its only job is
    making the log2 transform structural rather than a step a caller has
    to remember to apply consistently on both the fitting side and the
    prediction side -- forgetting it on just one side is a silent,
    hard-to-notice bug (predictions off by a power of two look plausible at
    a glance).

    Parameters
    ----------
    estimator : a scikit-learn-compatible regressor, or None
        Must implement `fit(X, y)` and `predict(X)`. Defaults to
        `sklearn.linear_model.Ridge()` when `None` -- a reasonable,
        cheap-to-fit baseline for the high-dimensional, sparse k-mer
        feature matrices this module is meant to be used with (see the
        module docstring's `KmerVectorizer` example); pass anything else
        (a random forest, gradient boosting, a `Pipeline` ending in a
        regressor, ...) for a stronger model once a baseline is working.

    Attributes
    ----------
    estimator_ : the fitted (cloned) estimator, set by `fit()`.
    n_features_in_ : int
        `X.shape[1]` at fit time, scikit-learn's own convention.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,  # scikit-learn-compatible regressor; sklearn is imported lazily
    ) -> None:
        # scikit-learn convention: __init__ only assigns parameters, no
        # validation and no side effects, matching fastdna.sklearn.KmerVectorizer
        # and fastdna.calibration.calibrate's own estimator-parameter handling.
        self.estimator = estimator

    def fit(
        self,
        X: Any,  # array-like or scipy sparse matrix, shape (n_samples, n_features)
        y: np.ndarray,  # array-like of shape (n_samples,), raw MIC values; coerced via log2_mic()
    ) -> MicRegressor:
        """Fits `estimator` (or the default `Ridge()`) on `(X, log2_mic(y))`.

        Parameters
        ----------
        X : array-like or scipy sparse matrix of shape (n_samples, n_features)
            A feature matrix -- typically `fastdna.sklearn.KmerVectorizer.
            fit_transform()`/`.transform()`'s output, but any matrix the
            chosen `estimator` accepts works.
        y : array-like of shape (n_samples,)
            Raw (linear-scale) MIC values, one per row of `X`. Validated by
            `log2_mic()`: must be strictly positive and finite.

        Returns
        -------
        self, per the scikit-learn convention that `fit()` returns the
        fitted estimator.

        Raises
        ------
        ValueError
            Everything `log2_mic()` raises, plus a row-count mismatch
            between `X` and `y`.
        """
        from sklearn.base import clone
        from sklearn.linear_model import Ridge

        n_samples = X.shape[0]
        y_log2 = log2_mic(y)
        if y_log2.shape[0] != n_samples:
            raise _core.InvalidConfigError(
                f"MicRegressor.fit(): X has {n_samples} rows but y has {y_log2.shape[0]} entries -- "
                "they must line up one MIC value per sample (row) of X."
            )

        base = self.estimator if self.estimator is not None else Ridge()
        self.estimator_ = clone(base)
        self.estimator_.fit(X, y_log2)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else None
        return self

    def _check_fitted(self):
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "MicRegressor instance is not fitted yet. Call fit(X, y) before predict()/"
                "predict_log2()."
            )

    def predict_log2(self, X: Any) -> np.ndarray:  # X is array-like or scipy sparse matrix
        """Predicted `log2(MIC)`, i.e. the wrapped estimator's raw output --
        the space the model was actually fit in, and the one
        `mean_absolute_error` and R^2 are conventionally reported in for
        MIC regression (see the module docstring).

        Returns
        -------
        numpy.ndarray of float64, shape (n_samples,)
        """
        self._check_fitted()
        return np.asarray(self.estimator_.predict(X), dtype=np.float64).reshape(-1)

    def predict(self, X: Any) -> np.ndarray:  # X is array-like or scipy sparse matrix
        """Predicted MIC in original (linear) units: `2 ** predict_log2(X)`.

        Returns
        -------
        numpy.ndarray of float64, shape (n_samples,)
        """
        return np.exp2(self.predict_log2(X))


class MicRegressionReport(NamedTuple):
    """What `mic_regression_report` computed.

    Attributes
    ----------
    table : pyarrow.Table
        One row per sample, columns `mic_true`, `mic_pred` (both raw,
        linear-scale MIC), `log2_mic_true`, `log2_mic_pred`, and
        `log2_error` (`log2_mic_pred - log2_mic_true`, signed: positive
        means the prediction overestimated the MIC).
    r2 : float
        `sklearn.metrics.r2_score` computed on `log2(MIC)`, not raw MIC --
        see the module docstring for why raw-MIC R^2 would be dominated by
        whichever samples happen to have the largest concentrations.
    mae_log2 : float
        Mean absolute error in log2 space: the average number of two-fold
        dilution steps a prediction is off by. Directly comparable across
        cohorts and antibiotics in a way a raw-MIC MAE (in mg/L) is not.
    essential_agreement : float
        Fraction of predictions with `abs(log2_error) <= tolerance_log2`
        (default 1.0, i.e. within one two-fold dilution step) -- the
        standard "essential agreement" (EA) metric used to compare an
        antimicrobial susceptibility test method against a reference
        method (CLSI/FDA guidance for AST device validation; commonly
        reported alongside "categorical agreement" in AMR genomics papers
        that predict MIC rather than a resistant/susceptible category). A
        value of 1.0 means every prediction landed within one dilution
        step of the true MIC; this is the number to lead with when
        communicating results to a clinical-microbiology audience, ahead
        of R^2 or MAE.
    """

    table: pa.Table
    r2: float
    mae_log2: float
    essential_agreement: float


def mic_regression_report(
    mic_true: np.ndarray,  # array-like of shape (n_samples,), coerced via log2_mic()
    mic_pred: np.ndarray,  # array-like of shape (n_samples,), coerced via log2_mic()
    *,
    tolerance_log2: float = 1.0,
) -> MicRegressionReport:
    """R^2, MAE and essential agreement for predicted vs. true MIC values.

    Does no fitting -- like `fastdna.evaluation.precision_recall_report`
    and `fastdna.evaluation.calibration_report`, this only reports on
    already-produced predictions, so it works identically whether they
    came from `MicRegressor`, a hand-rolled model, or a different
    pipeline entirely.

    Parameters
    ----------
    mic_true, mic_pred : array-like of shape (n_samples,)
        Raw (linear-scale) MIC values -- ground truth and predictions,
        respectively. Both are validated by `log2_mic()`: strictly
        positive and finite. `mic_pred` from `MicRegressor.predict()` is
        already in this form; if working with a model whose native output
        is `log2(MIC)`, exponentiate it first (`2 ** log2_predictions`)
        before calling this function, so both `mic_true` and `mic_pred`
        are consistently on the raw MIC scale this function's signature
        documents.
    tolerance_log2 : float, default 1.0
        Essential-agreement tolerance in log2 units (dilution steps). 1.0
        (one two-fold dilution) is the standard clinical-microbiology
        threshold; a stricter or looser tolerance can be passed for a
        different reporting convention, but changing it changes what
        "essential agreement" means for this report, so state the value
        used alongside the number.

    Returns
    -------
    MicRegressionReport

    Raises
    ------
    ValueError
        Everything `log2_mic()` raises for either argument, plus a length
        mismatch between `mic_true` and `mic_pred`, fewer than 2 samples
        (R^2 is undefined for a single point), or a non-positive
        `tolerance_log2`.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    true_log2 = log2_mic(mic_true)
    pred_log2 = log2_mic(mic_pred)

    if true_log2.shape[0] != pred_log2.shape[0]:
        raise _core.InvalidConfigError(
            f"mic_regression_report() needs one prediction per true value: mic_true has "
            f"{true_log2.shape[0]} entries but mic_pred has {pred_log2.shape[0]}."
        )
    if true_log2.shape[0] < 2:
        raise _core.InvalidConfigError(
            f"mic_regression_report() needs at least 2 samples to compute R^2, got "
            f"{true_log2.shape[0]}."
        )
    if not isinstance(tolerance_log2, (int, float)) or isinstance(tolerance_log2, bool) or tolerance_log2 <= 0:
        raise _core.InvalidConfigError(f"tolerance_log2 must be a positive number, got {tolerance_log2!r}")

    r2 = float(r2_score(true_log2, pred_log2))
    mae = float(mean_absolute_error(true_log2, pred_log2))
    log2_error = pred_log2 - true_log2
    essential_agreement = float(np.mean(np.abs(log2_error) <= float(tolerance_log2)))

    mic_true_arr = np.asarray(mic_true, dtype=np.float64)
    mic_pred_arr = np.asarray(mic_pred, dtype=np.float64)
    table = pa.table(
        {
            "mic_true": pa.array(mic_true_arr, type=pa.float64()),
            "mic_pred": pa.array(mic_pred_arr, type=pa.float64()),
            "log2_mic_true": pa.array(true_log2, type=pa.float64()),
            "log2_mic_pred": pa.array(pred_log2, type=pa.float64()),
            "log2_error": pa.array(log2_error, type=pa.float64()),
        }
    )

    return MicRegressionReport(
        table=table,
        r2=r2,
        mae_log2=mae,
        essential_agreement=essential_agreement,
    )
