"""Unsupervised anomaly / outlier detection over a cohort of samples --
"does this new sample look like the ones we've already established as a
normal baseline" -- for emerging-pathogen surveillance and lab-contamination
detection (roadmap item 7, `docs/ml-genomics-roadmap.md`).

Deliberately built only on `fastdna.sketch()` / `Sketch.mash_distance()`
(existing, stable), not on the in-flight `KmerVectorizer` or `taxonomy`
modules being built by parallel agents -- see the roadmap doc's entry for
this feature.

Feature representation
-----------------------
Each sample is turned into a fixed-size numeric feature vector: its
`mash_distance` to every sample in the *fitted baseline cohort*, in the
baseline's fit order. This is roadmap option (a)(i) -- "each sample's row
of distances-to-everyone-else as its feature vector" -- not (a)(ii)
(`OneClassSVM(kernel="precomputed")`) and not the spectrum-based option
(b). Reasons:

* It reuses `compare_all`'s own technique (pairwise `mash_distance` over
  MinHash sketches) without needing a hand-picked distance-to-similarity
  transform or a guarantee that the resulting matrix is a valid (PSD)
  kernel, which `kernel="precomputed"` requires and a raw Mash-distance
  matrix is not guaranteed to be.
* A plain real-valued feature matrix works unchanged with *both*
  `IsolationForest` and `OneClassSVM(kernel="rbf")` (the default kernel),
  so `method=` genuinely swaps the underlying detector without also having
  to swap the feature pipeline -- see `GenomicAnomalyDetector.__init__`.
* Sequence-content similarity (what `mash_distance` measures) is exactly
  the "is this a different organism / a contaminated sample" question this
  feature targets; the spectrum-based option is a real, complementary
  signal for "something is wrong with this sequencing run" but is a
  distinct question, and scoping to one well-implemented representation
  beats two half-implemented ones (see the roadmap doc + the task brief
  this module was built from).

Why the default `method` is `"one_class_svm"`, not `"isolation_forest"`
-------------------------------------------------------------------------
`mash_distance` saturates at exactly `1.0` for any pair of sketches that
share no k-mers at all (see `Sketch.mash_distance`'s docstring) -- so a
sample from a genuinely different organism gets a feature vector that is
*uniformly* `1.0` across every baseline column, regardless of how
different it actually is, and that value sits outside the range any
baseline-only training column ever takes (baseline-to-baseline distances
stay well under `1.0`).

`IsolationForest`'s splits are axis-aligned thresholds chosen from the
*range the training data itself spans* in each column. A query value
that lies beyond that range on every axis is not, on that basis alone,
isolated in few splits -- at every split it simply joins whichever
fraction of the baseline also exceeds that split's threshold, which is
no different from how any ordinary baseline point gets bisected. Verified
empirically while building this module: with `IsolationForest`, a sample
from a wholly different reference genome was scored *less* anomalous
than the baseline points themselves (baseline cohort sizes from 8 to 30),
and a "same organism but much noisier" sample scored identically to it,
because both simply route to the same "always exceeds the threshold"
side of every split -- the forest cannot see *how far* past the range a
point lies, only which side of each threshold it falls on. This is a
structural property of axis-aligned isolation splits combined with this
feature representation's hard ceiling at `1.0`, not sample-size noise.

`OneClassSVM`'s RBF kernel, by contrast, is a smooth function of the raw
distance in feature space, so it keeps discriminating "clamped at 1.0 in
every column" from "close to baseline" and from "moderately elevated"
continuously rather than through a hard split boundary. It is the
default (`method="one_class_svm"`, `__init__`'s default) for that
reason; `"isolation_forest"` stays available via `method=` for other
feature representations (e.g. spectrum-derived ones) that do not share
this saturating-distance shape, but it is not this module's default.
See `python/tests/test_anomaly.py`'s
`test_out_of_distribution_sample_predicted_outlier` and
`test_score_samples_ranks_mild_and_severe_anomalies_correctly` for the
regression tests this reasoning is pinned by.

Fit/predict feature-space consistency
--------------------------------------
This is the one subtlety that actually matters for correctness here: a
sample's feature vector is *always* its `mash_distance` to the sketches
`fit()` stored, in that fixed order -- never a fresh `compare_all()` over
whatever set of paths happens to be passed to `predict()`/`score_samples()`
in a given call. A naive implementation that recomputed pairwise distances
over "the baseline plus whatever was just asked about" would silently
change the feature space's dimensionality *and* meaning depending on how
many other samples happened to be in the same batch -- making scores
incomparable across calls and the anomaly score meaningless. Concretely:
`detector.score_samples([a, b])` and calling `detector.score_samples([a])`
/ `detector.score_samples([b])` separately must (and do, here) produce the
same score for `a` and for `b` -- see `python/tests/test_anomaly.py`'s
`test_predict_is_independent_of_batching` for the regression test.

One consequence worth naming: a baseline sample's own feature vector
carries an exact `0.0` at the column for itself (`mash_distance` to
itself). This is intentional, not an artifact to work around -- it means
`fit()` sees each baseline point as "close to its own position, and at
some distance from every other baseline point", and a new in-distribution
sample tends to land a *near*-zero value in the column of whichever
baseline sample it resembles most, which is exactly the signal that makes
it read as an inlier.
"""

import numpy as np

from . import sketch as _sketch


class GenomicAnomalyDetector:
    """Flags samples that deviate from a cohort's established baseline,
    using an unsupervised scikit-learn outlier detector
    (`IsolationForest` or `OneClassSVM`) over Mash-distance-to-baseline
    feature vectors (see module docstring for why).

    Follows scikit-learn's own `fit`/`predict` convention: `predict()`
    returns `1` for an inlier ("looks normal") and `-1` for an outlier
    ("flag for review"), exactly like `IsolationForest.predict` /
    `OneClassSVM.predict` -- this class does not invent a different
    convention.
    """

    _METHODS = ("isolation_forest", "one_class_svm")

    def __init__(self, k=21, sketch_size=1000, method="one_class_svm", **detector_kwargs):
        """`k`/`sketch_size` are forwarded to `fastdna.sketch()` for every
        sample this detector ever sketches (baseline and query alike --
        sketches built with different `k` cannot be compared, so this is
        fixed once per detector instance, not per call).

        `method` selects the underlying scikit-learn detector:
        `"one_class_svm"` (the default -- `sklearn.svm.OneClassSVM`,
        default RBF kernel over the distance-vector features) or
        `"isolation_forest"` (`sklearn.ensemble.IsolationForest` over the
        same features). See the module docstring's "Why the default
        `method` is `\"one_class_svm\"`" section for why the default is
        the SVM and not the forest for this particular feature
        representation. `**detector_kwargs`
        is forwarded verbatim to whichever constructor `method` selects
        (e.g. `contamination=0.1` for the forest, `nu=0.05` for the SVM),
        so it is validated -- and the detector actually constructed -- here
        in `__init__`, not lazily deferred to `fit()`: an unrecognized
        `method` or a bad keyword must fail loudly at construction time,
        not silently do nothing.
        """
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got {method!r}")

        self.k = k
        self.sketch_size = sketch_size
        self.method = method
        self.detector_kwargs = dict(detector_kwargs)

        self._detector = self._build_detector()

        # Set by fit(): the baseline's paths (for introspection/repr) and
        # the baseline's own Sketch objects, in fit order -- the fixed
        # basis every feature vector (baseline or query) is measured
        # against. See the module docstring's "Fit/predict feature-space
        # consistency" section.
        self.baseline_paths_ = None
        self._baseline_sketches = None

    def _build_detector(self):
        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            kwargs = {"random_state": 0}
            kwargs.update(self.detector_kwargs)
            return IsolationForest(**kwargs)

        # method == "one_class_svm"
        from sklearn.svm import OneClassSVM

        return OneClassSVM(**self.detector_kwargs)

    def _sketch_path(self, path):
        return _sketch(str(path), k=self.k, sketch_size=self.sketch_size)

    def _require_fitted(self):
        if self._baseline_sketches is None:
            raise RuntimeError(
                "GenomicAnomalyDetector is not fitted yet -- call .fit(baseline_paths) first."
            )

    def _features(self, sketches):
        """Builds the `(len(sketches), len(baseline))` feature matrix:
        row i, column j is `sketches[i].mash_distance(baseline_sketches[j])`
        -- always relative to `self._baseline_sketches` (the fixed basis
        set by `fit()`), regardless of how many samples are in `sketches`
        or what else has been passed to `predict`/`score_samples` before
        or since. This is what keeps the feature space consistent between
        `fit()` and every later `predict()`/`score_samples()` call.
        """
        n_baseline = len(self._baseline_sketches)
        X = np.empty((len(sketches), n_baseline), dtype=np.float64)
        for i, s in enumerate(sketches):
            for j, b in enumerate(self._baseline_sketches):
                X[i, j] = s.mash_distance(b)
        return X

    def fit(self, paths):
        """`paths`: the established "normal" baseline cohort (FASTQ(.gz)
        file paths). Sketches each one once, builds each baseline sample's
        feature vector as its `mash_distance` to every sketch in this same
        list (including itself, at `0.0` -- see module docstring), and
        fits the underlying scikit-learn detector on that matrix.

        Returns `self`, matching scikit-learn's own `fit()` convention.
        """
        paths = [str(p) for p in paths]
        if len(paths) < 2:
            raise ValueError("fit() needs at least 2 baseline samples to build distance features")

        self.baseline_paths_ = paths
        self._baseline_sketches = [self._sketch_path(p) for p in paths]

        X = self._features(self._baseline_sketches)
        self._detector.fit(X)
        return self

    def _transform(self, paths):
        self._require_fitted()
        sketches = [self._sketch_path(p) for p in paths]
        return self._features(sketches)

    def predict(self, paths):
        """Returns a `numpy.ndarray` of `1` (inlier, "looks normal") or
        `-1` (outlier, "flag for review"), one entry per path in `paths`,
        in the same order -- the same convention as
        `IsolationForest.predict`/`OneClassSVM.predict`.

        Each path's feature vector is computed against the baseline
        `fit()` stored, independent of what other paths are in this same
        call (see module docstring).
        """
        X = self._transform(paths)
        return self._detector.predict(X)

    def score_samples(self, paths):
        """Returns the underlying detector's raw anomaly score for each
        path (higher means more "normal"/inlier-like, lower means more
        anomalous -- scikit-learn's own convention for both
        `IsolationForest.score_samples` and `OneClassSVM.score_samples`),
        one entry per path, same order as `paths`.

        Useful for ranking several flagged samples by how unusual they
        are, rather than only getting a binary `predict()` label.
        """
        X = self._transform(paths)
        return self._detector.score_samples(X)

    def __repr__(self):
        fitted = "unfitted" if self._baseline_sketches is None else f"fitted on {len(self.baseline_paths_)} samples"
        return f"GenomicAnomalyDetector(k={self.k}, sketch_size={self.sketch_size}, method={self.method!r}, {fitted})"
