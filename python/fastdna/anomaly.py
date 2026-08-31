"""fastdna.anomaly -- within-cohort QC outlier flagging.

Answers one narrow, operational question: *given a cohort of samples that
are supposed to be comparable -- the same species, the same study, the
same sequencing run -- which member does not look like the rest?* The
actionable causes are lab-side, not epidemiological: a swapped tube or
mis-assigned barcode, a contaminated library, a failed prep, a sample
that silently came from a different organism than the manifest says.

Status: frozen
--------------
Per `docs/audit/PLAN.md` §2 ("Qué se poda"), this module is frozen: stable,
not accepting new features, and a candidate for extraction into a separate
`fastdna-contrib` package in a future release. Freezing is not deleting --
see that section for the full reasoning behind the boundary.

What this module is *not*
-------------------------
It is not pathogen surveillance, novelty detection, or an
"emerging-variant" alarm, and the framing matters enough to spell out
because an earlier version of this file claimed exactly that and was
deleted for it (commit c8870e3; see `python/tests/test_anomaly.py`'s
module docstring for the audit findings).

An unsupervised outlier score is a *relative* statement: "this sample is
unlike the others you gave me". It carries no information about what the
sample actually is. A cohort's most unusual member is, in ordinary lab
data, overwhelmingly likely to be a contaminated or swapped sample rather
than a novel organism -- and the score cannot tell those apart, because
nothing in it ever consults a reference. Treating "most unusual in this
batch" as "possible new pathogen" is an unsupported leap with no support
in the 2024-2026 literature, and it produces false alarms at exactly the
rate ordinary lab noise occurs.

The question "what organism is this, and is anything unexpected present"
is a *reference-based screening* question, and there are validated
methods for it: containment screening against a reference database
(Mash Screen; sourmash gather; NCBI STAT, Genome Biology 2021). In this
package that is `fastdna.taxonomy.classify` / `fastdna.taxonomy.gather`
against a reference database built by
`fastdna.taxonomy.build_reference_database`. Use those for identity and
contamination *screening*; use this module for "which member of my
supposedly-homogeneous cohort is the odd one out", which is a real,
constant lab QC need that reference screening does not answer (all the
samples can match the expected reference and one can still be wrong).

Feature representation
----------------------
Each sample is summarized to a single scalar, its **cohort distance**:
the median `Sketch.mash_distance` from that sample to every *other*
member of the fitted cohort. The median (not the mean) is deliberate --
if the cohort itself contains two or three bad samples, a mean would drag
every other sample's statistic toward them and mask the very thing being
looked for. A median tolerates a minority of contaminated cohort members.

The leave-one-out rule is applied identically at fit time and at query
time: a sample's distance to *itself* is exactly `0.0` by construction
and says nothing about whether it fits the cohort, so whenever a queried
path is also a cohort member, that column is dropped before the median is
taken. Without this, cohort members and genuinely new samples would be
summarized by two different statistics and their scores would not be
comparable.

Scoring: modified z-score, not a one-class model
------------------------------------------------
The default `method="robust_zscore"` is the Iglewicz & Hoaglin (1993)
modified z-score of a sample's cohort distance against the cohort's own
distribution of cohort distances:

    z = 0.6745 * (d - median(d_cohort)) / MAD(d_cohort)

flagging `z > threshold`, with `threshold=3.5` -- the value those authors
recommend and the one the NIST/SEMATECH handbook repeats. Three
properties matter here, and each of them is a defect the deleted module
had:

1. **A homogeneous cohort produces zero flags.** The threshold is an
   absolute statistical criterion, so a cohort where nothing is wrong
   comes back with nothing flagged. Contrast a one-class SVM's `nu` or an
   isolation forest's `contamination`, both of which are *quotas*: they
   flag their configured fraction of any cohort, correct or not. The
   deleted module's default (`OneClassSVM`, sklearn's `nu=0.5`) flagged 4
   of 8 known-good samples.
2. **The score keeps ranking past the cohort's own spread.** `z` is
   unbounded and strictly monotone in cohort distance, so severity stays
   rankable however far out a sample sits. The deleted module's
   `OneClassSVM(gamma="scale")` underflowed to exactly `0.0` past roughly
   distance 0.13, making a 15%-divergent same-organism sample and a
   completely unrelated organism indistinguishable.
3. **Median/MAD are robust to a contaminated cohort.** The reference
   distribution is estimated from the same cohort being screened, so it
   has to survive that cohort containing bad samples -- which a mean and a
   standard deviation do not.

The one real ceiling, stated rather than hidden: `mash_distance`
saturates at exactly `1.0` for two sketches sharing no k-mers at all
(see `Sketch.mash_distance`). Past that point no scorer built on this
feature can rank severity, because the feature itself has stopped
changing. In practice everything at distance 1.0 is "a different
organism", which is already the strongest QC finding available.

Where `IsolationForest` is and is not offered
---------------------------------------------
`method="isolation_forest"` is available through `flag_cohort()` -- the
transductive path, where the sample being judged is part of the data the
forest was fitted on -- and is refused by `CohortOutlierFlagger`, which
scores *new* samples against an already-fitted cohort.

That asymmetry is the audit finding, kept rather than papered over. An
isolation forest's splits are axis-aligned thresholds drawn from the
range the training data spans. A query beyond that range is not isolated
any faster than the most extreme training point: it lands in the same
extreme leaf and receives that point's score exactly. So for an
out-of-cohort query -- precisely `CohortOutlierFlagger`'s job -- a wholly
different organism scores no more anomalous than the least typical cohort
member, which is a silent false negative. `python/tests/test_anomaly.py`
pins this as a characterisation test. Leaving the option reachable and
documented anyway is what got the previous module deleted; it is refused
here with a message pointing at `flag_cohort()` instead.

Inside `flag_cohort()` the forest is legitimate -- the candidate outlier
is in the fitted data, which is the contaminated-training-set setting
isolation forests were designed for -- and useful when what is wanted is
a fixed budget ("show me the ~10% most unusual samples") rather than an
absolute threshold. `contamination` defaults to `0.1` here rather than
sklearn's `"auto"`, which on a tight QC cohort flags roughly one in eight
known-good samples.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Union

import numpy as np
import pyarrow as pa

from . import sketch as _sketch

__all__ = ["CohortOutlierFlagger", "flag_cohort"]

_METHODS = ("robust_zscore", "isolation_forest")

# Below this many samples a median absolute deviation is not an estimate of
# anything: with 3 cohort distances the MAD is a single order statistic, and
# every query would be judged against noise dressed up as a threshold.
_MIN_COHORT = 4

# Iglewicz & Hoaglin's constant: 0.6745 is the 0.75 quantile of the standard
# normal, so 0.6745 * (x - median) / MAD is on the same scale as an ordinary
# z-score when the underlying data are normal.
_MAD_SCALE = 0.6745
# Their own fallback when the MAD is exactly zero (possible when more than
# half the cohort shares one distance value): the mean absolute deviation,
# scaled by 1.253314 to be comparably calibrated.
_MEAN_AD_SCALE = 1.253314

_DEFAULT_CONTAMINATION = 0.1


def _isolation_forest(detector_kwargs):
    """Builds the optional `sklearn.ensemble.IsolationForest`, imported
    lazily here rather than at module scope so that importing this module
    never requires scikit-learn -- only asking for
    `method="isolation_forest"` does. A missing scikit-learn raises a clear,
    actionable `ImportError` naming the package and the install command
    instead of a raw `ModuleNotFoundError` from inside sklearn's own
    import machinery.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError(
            "method='isolation_forest' requires the 'scikit-learn' package, which is "
            "not installed. Install it with `pip install scikit-learn` and try again, "
            "or use the default method='robust_zscore', which needs no extra package."
        ) from None

    kwargs = {"random_state": 0, "contamination": _DEFAULT_CONTAMINATION}
    kwargs.update(detector_kwargs)
    return IsolationForest(**kwargs)


def _validate_cohort_paths(paths):
    paths = [str(p) for p in paths]
    if len(paths) < _MIN_COHORT:
        raise ValueError(
            f"a cohort of at least {_MIN_COHORT} samples is required to estimate a "
            f"robust reference distribution, got {len(paths)}. With fewer samples the "
            "median absolute deviation of the cohort's own distances is not an "
            "estimate of anything, so every score would be noise. Add more samples "
            "from the same cohort, or -- for a one-off 'are these two the same "
            "sample' question -- use fastdna.taxonomy.check_sample_identity instead."
        )

    seen = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"paths contains a duplicate entry: {path!r}. Each cohort member may "
                "appear only once; a duplicated sample would contribute an exact 0.0 "
                "distance to its own twin and pull that sample's statistic toward "
                "'perfectly typical' for a reason that has nothing to do with QC."
            )
        seen.add(path)
    return paths


def _robust_center_scale(values):
    """The `(center, scale)` pair a modified z-score is computed against:
    the median, and the median absolute deviation rescaled by
    `_MAD_SCALE`.

    Returns `scale = 0.0` when the cohort's distances are literally all
    identical -- the only case in which no spread estimate exists at all.
    Callers must handle that explicitly rather than dividing by it (see
    `_robust_z`); Iglewicz & Hoaglin's mean-absolute-deviation fallback is
    tried first, since a zero MAD merely means more than half the cohort
    shares one value, which is not the same as no spread.
    """
    center = float(np.median(values))
    deviations = np.abs(values - center)

    mad = float(np.median(deviations))
    if mad > 0.0:
        return center, mad / _MAD_SCALE

    mean_ad = float(np.mean(deviations))
    if mean_ad > 0.0:
        return center, mean_ad * _MEAN_AD_SCALE / _MAD_SCALE

    return center, 0.0


def _robust_z(values, center, scale):
    """The modified z-score of `values` against a cohort's `(center, scale)`.

    With `scale == 0.0` (every cohort distance identical) any deviation at
    all is infinitely many "robust standard deviations" from the centre, so
    that is what is returned: `0.0` for a sample sitting exactly at the
    centre, `+/- inf` otherwise. This is the honest reading -- the cohort
    supplied no scale against which to call a deviation small -- and it is
    reported as `inf` rather than silently substituting an arbitrary epsilon
    that would make the threshold mean something the caller never chose.
    """
    values = np.asarray(values, dtype=np.float64)
    deviation = values - center
    if scale == 0.0:
        # Built by assignment rather than `np.where(..., np.sign(d) * inf)`:
        # the latter evaluates `0 * inf` for the on-centre entries and emits
        # an "invalid value" RuntimeWarning before discarding the NaN.
        z = np.zeros_like(deviation)
        z[deviation > 0.0] = np.inf
        z[deviation < 0.0] = -np.inf
        return z
    return deviation / scale


class CohortOutlierFlagger:
    """Flags samples that do not look like an established cohort, by
    comparing each sample's median Mash distance to that cohort against the
    cohort's own distribution of the same statistic (see the module
    docstring for the representation and for why the scorer is a robust
    z-score rather than a one-class model).

        from fastdna.anomaly import CohortOutlierFlagger

        flagger = CohortOutlierFlagger(k=21, sketch_size=1000).fit(reference_batch)
        flagger.predict(todays_batch)        # 1 = looks normal, -1 = review
        flagger.outlier_scores(todays_batch) # higher = more unusual

    Follows scikit-learn's `fit`/`predict` convention: `predict()` returns
    `1` for an inlier and `-1` for an outlier, and `score_samples()` returns
    a score where *higher* means more normal. `outlier_scores()` is the
    sign-flipped view, because "higher = more unusual" is what a QC reader
    expects and silently inverting sklearn's convention would be worse.

    For the common case where the cohort *is* the thing being screened --
    all the samples arrived together and any of them could be the bad one --
    use the module-level `flag_cohort()` instead; it fits and scores in one
    call.

    Parameters
    ----------
    k : int, default 21
        Forwarded to `fastdna.sketch()` for every sample this flagger ever
        sketches, cohort and query alike. Sketches built with different `k`
        cannot be compared, so this is fixed per instance, not per call.
    sketch_size : int, default 1000
        Forwarded to `fastdna.sketch()`, same reasoning.
    method : {"robust_zscore"}, default "robust_zscore"
        Only `"robust_zscore"` is accepted here. `"isolation_forest"` is
        recognized but refused with a message pointing at `flag_cohort()`;
        see the module docstring for the measured reason it cannot score
        out-of-cohort queries.
    threshold : float, default 3.5
        Modified z-scores above this are flagged. 3.5 is Iglewicz &
        Hoaglin's own recommendation. Raise it for a cohort known to be
        genuinely heterogeneous; lower it only if false negatives are more
        costly than the review burden.

    Attributes
    ----------
    cohort_paths_ : list of str
        The fitted cohort, in fit order.
    cohort_distances_ : numpy.ndarray
        Each cohort member's leave-one-out median distance to the rest.
    center_, scale_ : float
        The median and rescaled MAD of `cohort_distances_`, i.e. the
        reference distribution every later score is measured against.
    """

    def __init__(
        self,
        k: int = 21,
        sketch_size: int = 1000,
        method: str = "robust_zscore",
        threshold: float = 3.5,
        **detector_kwargs: Any,  # forwarded to the underlying detector; only robust_zscore's threshold is used here
    ) -> None:
        """Validation happens here, at construction, rather than being
        deferred to `fit()`: an unrecognized `method`, an unusable
        `threshold`, or keyword arguments with nothing to forward them to
        must fail loudly before any FASTQ file is read, not silently do
        nothing.
        """
        if method == "isolation_forest":
            raise ValueError(
                "method='isolation_forest' is not available on CohortOutlierFlagger, "
                "which scores NEW samples against an already-fitted cohort. An "
                "isolation forest's splits come from the range its training data "
                "spans, so any query beyond that range receives the same score as the "
                "most extreme cohort member -- a wholly different organism would be "
                "rated no more unusual than the least typical known-good sample. Use "
                "flag_cohort(paths, method='isolation_forest') instead, where every "
                "candidate is inside the fitted data, or keep the default "
                "method='robust_zscore', whose score is unbounded and stays "
                "meaningful past the cohort's own spread."
            )
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS!r}, got {method!r}")
        if not threshold > 0:
            raise ValueError(f"threshold must be a positive number of robust z units, got {threshold!r}")
        if detector_kwargs:
            raise TypeError(
                f"CohortOutlierFlagger got unexpected keyword arguments "
                f"{sorted(detector_kwargs)!r}. method='robust_zscore' has no underlying "
                "scikit-learn detector to forward them to -- its only knob is "
                "`threshold`. Detector keyword arguments (contamination=, "
                "n_estimators=, ...) apply to flag_cohort(..., method='isolation_forest')."
            )

        self.k = k
        self.sketch_size = sketch_size
        self.method = method
        self.threshold = threshold

        # Set by fit(): the cohort's paths (the leave-one-out rule needs
        # them, not just the sketches), its sketches in fit order, and the
        # reference distribution derived from them.
        self.cohort_paths_ = None
        self.cohort_distances_ = None
        self.center_ = None
        self.scale_ = None
        self._cohort_sketches = None

    def _sketch_path(self, path):
        return _sketch(str(path), k=self.k, sketch_size=self.sketch_size)

    def _require_fitted(self):
        if self._cohort_sketches is None:
            raise RuntimeError(
                "CohortOutlierFlagger is not fitted yet -- call .fit(cohort_paths) first."
            )

    def _features(self, sketches):
        """The `(len(sketches), len(cohort))` raw distance matrix: row `i`,
        column `j` is `sketches[i].mash_distance(cohort_sketches[j])`.

        Always measured against `self._cohort_sketches` -- the fixed basis
        `fit()` stored -- never against a fresh all-pairs comparison over
        whatever paths happen to share a `predict()` call. Recomputing the
        basis per call would change both the width and the meaning of the
        feature space depending on batching, making scores incomparable
        across calls.
        """
        # The Rust-side sketches, unwrapped once for the whole matrix.
        # `Sketch.mash_distance` is a one-line forwarder to exactly this
        # call, so going through it per cell added a Python frame and two
        # attribute lookups to each of the `len(sketches) * n_cohort`
        # comparisons -- 2,500 of each when a 50-sample cohort is fitted,
        # against 50 + 50 lookups here -- plus one NumPy scalar assignment
        # per cell where a whole row can be assigned at once.
        basis = [b._raw for b in self._cohort_sketches]
        X = np.empty((len(sketches), len(basis)), dtype=np.float64)
        for i, s in enumerate(sketches):
            distance = s._raw.mash_distance
            X[i] = [distance(b) for b in basis]
        return X

    def _summarize(self, X, query_paths):
        """Collapses each row of a distance matrix to its cohort distance:
        the median over the cohort columns, excluding the column belonging
        to the query itself when the query *is* a cohort member.

        The exclusion is what makes fit-time and query-time statistics the
        same quantity -- see the module docstring's leave-one-out note.
        """
        # The cohort's paths are the same for every query row, so they are
        # turned into an array once instead of being walked in a Python
        # list comprehension per row: a 50-sample cohort scored against 50
        # queries did 2,500 Python string comparisons and built 50 lists,
        # against 50 vectorized comparisons here. `!=` on a NumPy string
        # array is the same exact-equality test on the same `str` values.
        members = np.asarray(self.cohort_paths_)
        stats = np.empty(len(query_paths), dtype=np.float64)
        for i, path in enumerate(query_paths):
            stats[i] = float(np.median(X[i][members != path]))
        return stats

    def fit(self, paths: Iterable[Union[str, os.PathLike]]) -> "CohortOutlierFlagger":
        """`paths`: the cohort defining "normal" (FASTQ(.gz) file paths).

        Sketches each once, computes every member's leave-one-out cohort
        distance, and derives the robust reference distribution
        (`center_`/`scale_`) every later score is measured against.

        Returns `self`, matching scikit-learn's own `fit()` convention.
        """
        paths = _validate_cohort_paths(paths)

        self.cohort_paths_ = paths
        self._cohort_sketches = [self._sketch_path(p) for p in paths]

        X = self._features(self._cohort_sketches)
        self.cohort_distances_ = self._summarize(X, paths)
        self.center_, self.scale_ = _robust_center_scale(self.cohort_distances_)
        return self

    def _transform(self, paths):
        """The raw `(len(paths), len(cohort))` distance matrix for `paths`.

        Exposed (privately) because it is the honest intermediate a caller
        debugging a surprising flag wants: which cohort member is this
        sample close to, and which is it far from.
        """
        self._require_fitted()
        paths = [str(p) for p in paths]
        return self._features([self._sketch_path(p) for p in paths])

    def cohort_distance(self, paths: Iterable[Union[str, os.PathLike]]) -> np.ndarray:
        """Each path's cohort distance -- the median Mash distance to the
        fitted cohort, leave-one-out for cohort members themselves. This is
        the single scalar every score and label below is derived from, in
        the units `mash_distance` reports (roughly, per-base divergence).
        """
        self._require_fitted()
        paths = [str(p) for p in paths]
        return self._summarize(self._features([self._sketch_path(p) for p in paths]), paths)

    def score_samples(self, paths: Iterable[Union[str, os.PathLike]]) -> np.ndarray:
        """The scikit-learn-convention anomaly score: **higher means more
        normal**, one entry per path, same order as `paths`.

        Concretely `-z`, the negated modified z-score of each path's cohort
        distance. Unbounded and strictly monotone in that distance, so it
        ranks severity rather than only crossing a threshold -- up to the
        `mash_distance` ceiling of 1.0, past which the underlying feature
        itself stops changing (module docstring).
        """
        return -self.outlier_scores(paths)

    def outlier_scores(self, paths: Iterable[Union[str, os.PathLike]]) -> np.ndarray:
        """The same score with the sign a QC reader expects: **higher means
        more unusual**. Exactly `-score_samples(paths)`.

        The number is in robust-z units, so it is directly comparable to
        `threshold`: a value of 3.5 is the flagging boundary, 10 is far
        outside the cohort, and anything at the `mash_distance` ceiling
        (a sample sharing no k-mers with the cohort) will be enormous.
        """
        self._require_fitted()
        distances = self.cohort_distance(paths)
        return _robust_z(distances, self.center_, self.scale_)

    def predict(self, paths: Iterable[Union[str, os.PathLike]]) -> np.ndarray:
        """`1` (looks normal) or `-1` (flag for review) per path, in order --
        the same convention `IsolationForest.predict`/`OneClassSVM.predict`
        use, so this class does not invent a competing one.

        A sample is flagged when its modified z-score exceeds `threshold`.
        Note the test is one-sided: a sample that is *unusually close* to
        the cohort is not a QC problem (an over-similar pair is what
        `fastdna.taxonomy.check_sample_identity` is for, and it is a
        different question -- duplicate submission, not contamination).
        """
        scores = self.outlier_scores(paths)
        return np.where(scores > self.threshold, -1, 1)

    def __repr__(self) -> str:
        fitted = "unfitted" if self._cohort_sketches is None else f"fitted on {len(self.cohort_paths_)} samples"
        return (
            f"CohortOutlierFlagger(k={self.k}, sketch_size={self.sketch_size}, "
            f"method={self.method!r}, threshold={self.threshold}, {fitted})"
        )


def flag_cohort(
    paths: Iterable[Union[str, os.PathLike]],
    *,
    k: int = 21,
    sketch_size: int = 1000,
    method: str = "robust_zscore",
    threshold: float = 3.5,
    **detector_kwargs: Any,  # forwarded verbatim to IsolationForest when method="isolation_forest"
) -> pa.Table:
    """Screens a cohort against itself: "these samples all arrived together
    and should be comparable -- which one is not?"

    This is the transductive form of the question, and the one most lab QC
    actually asks. There is no held-out batch: every sample is both part of
    the reference and a candidate outlier, which is why the leave-one-out
    rule in `CohortOutlierFlagger._summarize` matters -- each sample is
    judged against the *others*, never against itself.

        from fastdna.anomaly import flag_cohort

        flag_cohort(todays_batch).to_pandas()

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        The cohort, at least `4` samples (see `_MIN_COHORT`). Duplicates are
        refused.
    k, sketch_size : forwarded to `fastdna.sketch()` for every sample.
    method : {"robust_zscore", "isolation_forest"}, default "robust_zscore"
        `"robust_zscore"` applies an absolute statistical criterion, so a
        cohort where nothing is wrong yields *zero* flags -- the right
        default for unattended QC. `"isolation_forest"` fits
        `sklearn.ensemble.IsolationForest` over the same cohort-distance
        statistic and flags a fixed `contamination` fraction; use it when
        what is wanted is "the N most unusual samples" regardless of whether
        any of them is actually bad, and note that it will always flag
        roughly that fraction of a perfectly good cohort. It is available
        here, and refused by `CohortOutlierFlagger`, for the reason set out
        in the module docstring.
    threshold : float, default 3.5
        Used by `"robust_zscore"` only; ignored by `"isolation_forest"`,
        whose flagging rate is set by `contamination` instead.
    **detector_kwargs
        Forwarded verbatim to `IsolationForest` (e.g. `contamination=0.05`,
        `n_estimators=200`). `contamination` defaults to 0.1 rather than
        sklearn's `"auto"`, which on a tight QC cohort flags roughly one in
        eight known-good samples. Passing any of these with the default
        method is a `TypeError`, not a silent no-op.

    Returns
    -------
    pyarrow.Table with columns `sample`, `cohort_distance`, `outlier_score`,
    `is_outlier`, one row per input path in input order -- a plain Arrow
    table, matching `compare_all()`/`taxonomy.classify()`'s convention, so
    it composes with `.to_pandas()`, DuckDB or Polars without conversion.
    `outlier_score` is "higher = more unusual" (robust z units under
    `"robust_zscore"`; the negated `IsolationForest.score_samples` under
    `"isolation_forest"`, which is bounded and therefore comparable only
    within one call).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS!r}, got {method!r}")

    paths = _validate_cohort_paths(paths)

    if method == "robust_zscore":
        flagger = CohortOutlierFlagger(k=k, sketch_size=sketch_size, threshold=threshold, **detector_kwargs)
        flagger.fit(paths)
        distances = flagger.cohort_distances_
        scores = _robust_z(distances, flagger.center_, flagger.scale_)
        is_outlier = scores > threshold
    else:
        # A CohortOutlierFlagger is still what computes the feature, so the
        # leave-one-out rule and the sketching parameters stay in one place;
        # only the scorer differs.
        flagger = CohortOutlierFlagger(k=k, sketch_size=sketch_size, threshold=threshold)
        flagger.fit(paths)
        distances = flagger.cohort_distances_

        forest = _isolation_forest(detector_kwargs)
        X = distances.reshape(-1, 1)
        forest.fit(X)
        scores = -forest.score_samples(X)
        is_outlier = forest.predict(X) == -1

    return pa.table(
        {
            "sample": paths,
            "cohort_distance": distances.astype(np.float64),
            "outlier_score": np.asarray(scores, dtype=np.float64),
            "is_outlier": np.asarray(is_outlier, dtype=bool),
        }
    )
