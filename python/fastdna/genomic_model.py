"""fastdna.genomic_model -- a thin deployment wrapper around a fitted
genomic classifier, with an honest domain-applicability check on every new
sample it is asked to score.

## The gap this closes

Everything upstream of this module (`fastdna.sklearn.KmerVectorizer`,
`fastdna.cv`, `fastdna.evaluation`) is about *fitting and evaluating* a
model. Nothing packages the result into something a lab can actually point
at a brand-new FASTQ file six months later and trust -- and "trust" is the
operative word: a classifier fitted on, say, an *E. coli* AMR cohort will
still cheerfully emit a confident-looking prediction for a *Klebsiella*
isolate, a heavily contaminated sample, or a lineage the training cohort
never included, because nothing about `estimator.predict()` itself knows
what it was trained on. That is exactly the setting a domain-applicability
check exists for in the wider ML-deployment literature (this is the same
concern "out-of-distribution detection" addresses generally; see e.g.
Yang et al., "Generalized Out-of-Distribution Detection: A Survey", IJCV
2024), applied here to genomic samples specifically.

`GenomicModel` does not reimplement that check. It reuses
`fastdna.anomaly.CohortOutlierFlagger` -- already-built, already-tested
machinery for exactly the question "does this new sample's genomic
profile plausibly belong to a fitted cohort's distribution" -- and wires
its verdict onto every prediction. See `fastdna.anomaly`'s own module
docstring for why a robust z-score over cohort Mash distance is the right
tool here (an absolute statistical criterion, not a fixed-quota outlier
detector, and unbounded so severity keeps ranking past the cohort's own
spread) and for its one honest ceiling (`mash_distance` saturates at 1.0).

## What "flagged, not suppressed" means here

This project's convention, followed identically by
`fastdna.evaluation.calibration_report`'s `UncalibratedScoresWarning` and
`fastdna.explain`'s per-feature `verdict` strings, is that uncertainty is
reported loudly, not hidden. `GenomicModel.predict()` never refuses to
return a prediction for an out-of-distribution sample -- suppressing the
number would just make a caller reach for `estimator.predict()` directly
and lose the check entirely. Instead every row of the returned table
carries `in_distribution` (bool) and `verdict` (a short human-readable
label) alongside the prediction, and `predict()` also raises
`OutOfDistributionWarning` (a distinct category, filterable independently
of other warnings) naming every flagged sample, so a caller who ignores
the returned columns still cannot silently miss it in a log.

## Duck-typing, not a hard import, against `fastdna.sklearn`

This module takes an already-fitted `estimator` (anything exposing
`.predict()`, and optionally `.predict_proba()` + `.classes_`) and an
already-fitted `vectorizer` (anything exposing `.transform(paths) ->
array-like`) rather than importing `fastdna.sklearn.KmerVectorizer`
directly. `fastdna.sklearn.KmerVectorizer` is the intended fit -- its
`.transform()` contract is exactly what this module calls -- but this
module was written against that *documented* contract rather than a hard
import of the class itself, the same choice `fastdna.interpret` made
for the same reason (see that module's own docstring): both were built in
parallel with other modules touching `fastdna/sklearn.py`, and a
duck-typed contract is what stays correct regardless of which one lands,
or changes shape, first. Any fitted scikit-learn-style transformer with a
matching `.transform(paths)` works here, not only `KmerVectorizer`.

## No bespoke persistence format

`docs/audit/ml-gaps.md`'s own sketch of this feature (`G-10`) shows
`model.save("amr_ecoli_v1.fdna-model")` / `GenomicModel.load(...)`. That
is not implemented here, deliberately: a `GenomicModel` is a plain Python
object holding a `vectorizer`, an `estimator` and a list of training
paths plus scalars -- nothing about it needs a new file format, and this
package has already made the opposite call once before for the same
reason (`fastdna.taxonomy.build_reference_database`'s own docstring: "a
second, bespoke multi-sketch file format... would add a format to
maintain without adding any capability [a plain dict] doesn't already
have"). As long as `estimator` and `vectorizer` are themselves picklable
(true of ordinary scikit-learn estimators and of `KmerVectorizer`, which
subclasses `sklearn.base.BaseEstimator`), a `GenomicModel` instance is
picklable as-is: `pickle.dump(model, f)` / `joblib.dump(model, path)`
round-trips it exactly, with no bespoke format to keep in sync with this
module's own evolution.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pyarrow as pa

from .anomaly import CohortOutlierFlagger

__all__ = ["GenomicModel", "OutOfDistributionWarning"]


class OutOfDistributionWarning(UserWarning):
    """Raised by `GenomicModel.predict()` when one or more queried samples
    fall outside the training cohort's established genomic-profile
    distribution (per `fastdna.anomaly.CohortOutlierFlagger`).

    A distinct category, not a bare `UserWarning`, so a caller who has
    genuinely understood the caveat can silence *this* warning specifically
    (`warnings.filterwarnings("ignore", category=OutOfDistributionWarning)`)
    without also silencing every other warning FastDNA might legitimately
    raise -- the same reasoning `evaluation.UncalibratedScoresWarning`
    documents for itself.
    """


def _validate_paths(paths, label):
    """Turns `paths` into a list of path strings, accepting a single bare
    path (str/PathLike) as a convenience for the common "score one new
    sample" case rather than forcing every caller to wrap it in a list.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError(f"{label} requires at least one path, got an empty sequence")
    return paths


class GenomicModel:
    """Wraps an already-fitted `(vectorizer, estimator)` pair together with
    the training cohort's own FASTQ paths, so scoring a new sample also
    answers "should this prediction be trusted" -- see the module
    docstring for why that check reuses `fastdna.anomaly.
    CohortOutlierFlagger` rather than reimplementing it, and for why this
    class deliberately has no bespoke `.save()`/`.load()`.

        from fastdna.sklearn import KmerVectorizer
        from fastdna.genomic_model import GenomicModel
        from sklearn.linear_model import LogisticRegression

        vec = KmerVectorizer(k=31, top_features=10_000).fit(train_paths)
        clf = LogisticRegression(max_iter=1000).fit(vec.transform(train_paths), y)

        model = GenomicModel(clf, vec, train_paths)
        result = model.predict("new_isolate.fastq.gz")
        # result.to_pylist()[0] -> {"sample": ..., "prediction": ...,
        #   "probability": ..., "in_distribution": ..., "outlier_score": ...,
        #   "cohort_distance": ..., "verdict": ...}

    Or, to fit `vectorizer`/`estimator` and this wrapper together in one
    call, see the `fit()` classmethod.

    Parameters
    ----------
    estimator : a fitted scikit-learn-style classifier
        Must expose `.predict(X)`. `.predict_proba(X)` and `.classes_`
        are used when present (to populate `probability`), but are not
        required -- an estimator lacking them (e.g.
        `fastdna.rules.SetCoveringClassifier`, whose `predict_proba()` is
        DELIBERATELY absent of real calibration -- see its own and
        `fastdna.evaluation.calibration_report`'s docstrings) simply gets
        `probability = NaN` in every row rather than an error.
    vectorizer : a fitted transformer exposing `.transform(paths)`
        Projects FASTQ(.gz) paths onto the same feature space `estimator`
        was fitted on. Typically a fitted `fastdna.sklearn.KmerVectorizer`
        -- see the module docstring for why this is duck-typed rather than
        a hard `isinstance` check against that specific class.
    training_paths : iterable of str or pathlib.Path
        The real FASTQ(.gz) paths of the cohort `estimator`/`vectorizer`
        were trained on -- **actual files**, not `sample_id` strings from
        a `fastdna.CohortCounts`/`KmerVectorizer(counts=...)` artifact,
        even if that artifact was used to fit them: the domain-
        applicability check sketches these files directly
        (`fastdna.anomaly.CohortOutlierFlagger`), which needs real paths
        to read. This is the identical constraint `fastdna.explain`'s own
        `paths` parameter documents, for the identical reason. At least 4
        paths are required (`fastdna.anomaly`'s own minimum for a robust
        reference distribution); see that module's `ValueError` message
        for why.
    k, sketch_size : int, default 21, 1000
        Forwarded to `CohortOutlierFlagger` for sketching every training
        and query sample. Independent of whatever `k` `vectorizer` itself
        uses for exact k-mer vectorization -- the domain check is a
        MinHash-sketch comparison, not a feature-space comparison, so the
        two `k`s may legitimately differ.
    outlier_method : {"robust_zscore"}, default "robust_zscore"
        Forwarded to `CohortOutlierFlagger`. Only `"robust_zscore"` is
        accepted (the same restriction `CohortOutlierFlagger` itself
        applies to new-sample queries; see its docstring for the measured
        reason `"isolation_forest"` cannot score an out-of-cohort query
        honestly).
    outlier_threshold : float, default 3.5
        Forwarded to `CohortOutlierFlagger`: modified z-scores above this
        (Iglewicz & Hoaglin, 1993) are flagged out-of-distribution.

    Attributes
    ----------
    training_paths : list of str
        As given, stringified.
    """

    def __init__(
        self,
        estimator,
        vectorizer,
        training_paths,
        *,
        k=21,
        sketch_size=1000,
        outlier_method="robust_zscore",
        outlier_threshold=3.5,
    ):
        if not hasattr(estimator, "predict"):
            raise TypeError(
                f"estimator must already be fitted and expose .predict(X); got a "
                f"{type(estimator).__name__} with no .predict method. Fit it first, e.g. "
                "estimator.fit(vectorizer.transform(training_paths), y)."
            )
        if not hasattr(vectorizer, "transform"):
            raise TypeError(
                f"vectorizer must already be fitted and expose .transform(paths); got a "
                f"{type(vectorizer).__name__} with no .transform method. "
                "fastdna.sklearn.KmerVectorizer is the intended fit here (see the module "
                "docstring), but any fitted transformer with a matching .transform(paths) "
                "works."
            )

        self.estimator = estimator
        self.vectorizer = vectorizer
        self.training_paths = _validate_paths(training_paths, "training_paths")
        self.k = k
        self.sketch_size = sketch_size
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        # Fitted once, here, and reused for every later predict() call --
        # the same "sketch once, query many times" principle
        # CohortOutlierFlagger's own docstring already applies to a cohort.
        self._flagger = CohortOutlierFlagger(
            k=k, sketch_size=sketch_size, method=outlier_method, threshold=outlier_threshold,
        ).fit(self.training_paths)

    @classmethod
    def fit(
        cls,
        paths,
        y,
        *,
        vectorizer,
        estimator,
        k=21,
        sketch_size=1000,
        outlier_method="robust_zscore",
        outlier_threshold=3.5,
    ):
        """Fits `vectorizer` and `estimator` on `paths`/`y`, fits the
        domain-applicability check on the same `paths`, and returns a
        ready-to-use `GenomicModel` -- the one-call convenience for the
        common case where nothing is fitted yet.

        Parameters
        ----------
        paths : iterable of str or pathlib.Path
            The training cohort's real FASTQ(.gz) paths (at least 4; see
            the class docstring).
        y : array-like, length `len(paths)`
            Training labels, forwarded to `vectorizer.fit_transform(paths,
            y)` (or `.fit(paths, y)` then `.transform(paths)`, if the
            vectorizer has no `fit_transform`) and to `estimator.fit(X,
            y)`.
        vectorizer : an UNFITTED transformer exposing `.fit`/`.transform`
            (or `.fit_transform`). Fitted here, in place -- unlike
            `estimator`, it is not cloned first, since a caller who wants
            to keep an unfitted copy can pass `sklearn.base.clone
            (vectorizer)` themselves, and `fastdna.sklearn.KmerVectorizer`
            has no meaningful "unfitted state" worth preserving once its
            vocabulary is decided.
        estimator : an UNFITTED scikit-learn-style estimator
            Cloned first via `sklearn.base.clone` when scikit-learn is
            importable (never mutating the object the caller passed in,
            matching `fastdna.cv.permutation_importance_pvalues`'s own
            convention), then fitted on `(X, y)` in place. If scikit-learn
            cannot be imported, `estimator` is fitted in place directly --
            it must already be an estimator instance, so scikit-learn
            being unavailable here would be a caller error regardless.
        k, sketch_size, outlier_method, outlier_threshold
            Forwarded to `GenomicModel.__init__`; see its docstring.

        Returns
        -------
        GenomicModel
        """
        paths = _validate_paths(paths, "paths")

        if hasattr(vectorizer, "fit_transform"):
            X = vectorizer.fit_transform(paths, y)
        else:
            vectorizer.fit(paths, y)
            X = vectorizer.transform(paths)

        try:
            from sklearn.base import clone

            estimator = clone(estimator)
        except ImportError:
            pass
        estimator.fit(X, y)

        return cls(
            estimator,
            vectorizer,
            paths,
            k=k,
            sketch_size=sketch_size,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
        )

    def predict(self, paths):
        """Scores `paths` and checks each against the training cohort's
        distribution in one call.

        Parameters
        ----------
        paths : str, pathlib.Path, or an iterable of either
            One new sample or several. A bare path is accepted directly
            (not just a one-element list) so the common "score one new
            isolate" case reads naturally:
            `model.predict("isolate.fastq.gz")`.

        Returns
        -------
        pyarrow.Table, one row per input path in input order, columns:
          * `sample` -- the path, stringified.
          * `prediction` -- `estimator.predict()`'s output for this
            sample, whatever type that estimator returns (a class label).
          * `probability` -- the predicted class's probability, from
            `estimator.predict_proba()` looked up against
            `estimator.classes_` at the predicted label's own index (so
            this is always the probability of the label actually
            predicted, not e.g. a fixed "positive class" column). `NaN`
            when the estimator exposes no `predict_proba`/`classes_`.
          * `in_distribution` -- `True` unless
            `CohortOutlierFlagger.predict()` flagged this sample (its
            modified z-score against the training cohort's own cohort-
            distance distribution exceeded `outlier_threshold`). **Check
            this before trusting `prediction`** -- see the module
            docstring.
          * `outlier_score` -- the same modified z-score
            (`CohortOutlierFlagger.outlier_scores()`; higher = more
            unusual), so severity is visible, not just a threshold crossed
            or not.
          * `cohort_distance` -- the raw median Mash distance to the
            training cohort the score above was computed from
            (`CohortOutlierFlagger.cohort_distance()`), in the same
            roughly-per-base-divergence units `Sketch.mash_distance`
            reports.
          * `verdict` -- `"in-distribution"` or `"OUT-OF-DISTRIBUTION"`,
            the same short human-readable label convention
            `fastdna.explain`'s `FeatureExplanation.verdict` uses.

        Also raises `OutOfDistributionWarning` (not an exception -- the
        prediction is still returned) naming every flagged sample, so a
        caller who does not inspect the returned columns still cannot
        silently miss the finding. See the module docstring for why this
        is a warning plus an annotated column, not a suppressed
        prediction.
        """
        paths = _validate_paths(paths, "paths")

        X = self.vectorizer.transform(paths)
        predictions = np.asarray(self.estimator.predict(X))
        if predictions.shape[0] != len(paths):
            raise ValueError(
                f"estimator.predict() returned {predictions.shape[0]} predictions for "
                f"{len(paths)} input paths -- vectorizer.transform() and estimator.predict() "
                "disagree about how many samples were given."
            )

        probabilities = np.full(len(paths), np.nan, dtype=np.float64)
        if hasattr(self.estimator, "predict_proba") and hasattr(self.estimator, "classes_"):
            proba = np.asarray(self.estimator.predict_proba(X))
            classes = list(self.estimator.classes_)
            for i, label in enumerate(predictions.tolist()):
                try:
                    class_index = classes.index(label)
                except ValueError:
                    # The predicted label is not among estimator.classes_ --
                    # can happen for an estimator whose predict()/
                    # predict_proba() disagree by construction (unusual, but
                    # not this module's place to paper over); left as NaN
                    # rather than guessing which column it might mean.
                    continue
                probabilities[i] = proba[i, class_index]

        flags = self._flagger.predict(paths)  # 1 = inlier, -1 = outlier
        outlier_scores = np.asarray(self._flagger.outlier_scores(paths), dtype=np.float64)
        cohort_distances = np.asarray(self._flagger.cohort_distance(paths), dtype=np.float64)
        in_distribution = flags == 1

        flagged = [
            (p, s, d)
            for p, ok, s, d in zip(paths, in_distribution, outlier_scores, cohort_distances)
            if not ok
        ]
        if flagged:
            details = "; ".join(
                f"{p} (outlier score {s:.2f} > threshold {self.outlier_threshold:g}, "
                f"cohort distance {d:.4f})"
                for p, s, d in flagged
            )
            warnings.warn(
                f"{len(flagged)} of {len(paths)} sample(s) fall outside the training "
                "cohort's established genomic-profile distribution. Treat the "
                "prediction(s) above as untrustworthy until the discrepancy is explained "
                "(wrong species, contamination, a lineage the training cohort did not "
                f"include, ...): {details}",
                OutOfDistributionWarning,
                stacklevel=2,
            )

        verdicts = ["in-distribution" if ok else "OUT-OF-DISTRIBUTION" for ok in in_distribution]

        return pa.table(
            {
                "sample": paths,
                "prediction": predictions.tolist(),
                "probability": probabilities.tolist(),
                "in_distribution": np.asarray(in_distribution, dtype=bool).tolist(),
                "outlier_score": outlier_scores.tolist(),
                "cohort_distance": cohort_distances.tolist(),
                "verdict": verdicts,
            }
        )

    def __repr__(self):
        return (
            f"GenomicModel(estimator={type(self.estimator).__name__}, "
            f"vectorizer={type(self.vectorizer).__name__}, "
            f"training_cohort={len(self.training_paths)} samples, "
            f"outlier_method={self.outlier_method!r}, outlier_threshold={self.outlier_threshold})"
        )
