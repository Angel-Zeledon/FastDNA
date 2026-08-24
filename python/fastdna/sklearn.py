"""scikit-learn integration for FastDNA (Tier 2, design doc §9.6).

Pure Python, built entirely on top of the stable `fastdna.count()` API --
no new Rust surface, no FFI additions. Lives in its own module, imported
explicitly (`from fastdna.sklearn import KmerVectorizer`), rather than in
`fastdna/__init__.py`, so importing `fastdna` itself never requires
scikit-learn or scipy to be installed. This matches the existing package's
own convention for heavier optional dependencies -- e.g. `KmerCounts.to_polars()`
does a local `import polars` instead of a module-level one, so plain
`import fastdna` never pays for (or requires) polars either.

## Why this module exists

Cross-validating a classifier over genomic samples usually means picking
*which k-mers* to use as features before fitting anything. Do that by
looking at the whole dataset -- including whatever ends up in a held-out
test fold -- and the model has effectively been shown the answer key: the
validation score comes out inflated because the vocabulary itself already
encodes which sequences correlate with the labels the model is later
"predicting". This is a well known failure mode in genomic ML (see the
design doc, §3, "Why feature selection is unsupervised"), and it is easy
to reintroduce by accident: a single call to build a shared vocabulary
before `train_test_split`, done once at the top of a notebook, is enough.

`KmerVectorizer` closes that door structurally rather than by
documentation or discipline. `scikit-learn`'s own `Pipeline`,
`cross_val_score`, and `GridSearchCV` all call `.fit()` on each pipeline
step *only* on the current training fold, at each split, before ever
calling `.transform()` on the held-out fold. Because `KmerVectorizer.fit()`
is the *only* place `self.vocabulary_` is ever assigned, and `transform()`
only ever reads that already-decided vocabulary, the held-out fold's
content cannot influence which k-mers become features -- not even
indirectly -- no matter how the caller wires this into a larger pipeline.
`transform()` also never receives or looks at `y`, so there is no path by
which the *label* could leak into feature selection either (design doc,
§3: features are ranked by prevalence, never by correlation with the
label).
"""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import fastdna


def _validate_paths(X, method_name):
    """Turns `X` into a list of path strings, rejecting the classic footgun
    of passing a single path instead of a list of them: a bare string (or
    `os.PathLike`) *is* iterable, so it would otherwise be iterated
    character by character and surface as a baffling per-character
    `FileNotFoundError` deep inside the counting loop.
    """
    if isinstance(X, (str, os.PathLike)):
        raise TypeError(
            f"KmerVectorizer.{method_name}() expects a list of paths, got a single "
            f"path {str(X)!r} -- wrap it in a list: [{str(X)!r}]"
        )
    return [str(p) for p in X]


class KmerVectorizer(BaseEstimator, TransformerMixin):
    """Projects FASTQ(.gz) files onto a fixed, learned k-mer vocabulary,
    producing a sparse count matrix suitable for `sklearn.pipeline.Pipeline`
    steps (`LogisticRegression`, `XGBClassifier`, etc.) and for
    `sklearn.model_selection.cross_val_score` / `GridSearchCV`.

        from fastdna.sklearn import KmerVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([
            ("kmers", KmerVectorizer(k=31, min_count=5, top_features=10_000)),
            ("clf",   LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(train_paths, train_labels)
        pipe.predict(test_paths)

    See the module docstring for why the vocabulary being decided
    exclusively inside `fit()` is the entire point of this class, not an
    incidental implementation detail.

    Parameters
    ----------
    k : int, default 31
        K-mer length, forwarded to `fastdna.count()`.
    min_count : int, default 1
        Per-sample minimum count a k-mer must reach to be kept at all,
        forwarded to `fastdna.count()`. Filters sequencing-error noise out
        before it can ever compete for a vocabulary slot; see
        `KmerCounts.suggest_min_count()` for picking this per-dataset
        rather than guessing.
    top_features : int or None, default 10_000
        Maximum vocabulary size selected during `fit()`. `None` means no
        cap: every k-mer observed across the training samples (after
        `min_count` pruning) becomes a feature. Leaving this uncapped is
        appropriate for a small, curated training cohort where the full
        union of observed k-mers is already modest; the numeric default
        exists for the common case where it is not, as a hard limit on
        matrix width (matching the design doc's own default of 10,000).
    threads : int or None, default None
        Forwarded to `fastdna.count()` for each sample; `None` uses the
        core's own default thread count per call.

    Attributes
    ----------
    vocabulary_ : numpy.ndarray of uint64
        The selected k-mers' `kmer_u64` encodings, in ranked order (see
        `fit()` for the ranking rule). Column `j` of any matrix returned
        by `transform()` corresponds to `vocabulary_[j]`.
    n_features_in_ : int
        `len(vocabulary_)`. scikit-learn's own convention for this
        attribute assumes `X` is a numeric `(n_samples, n_features)`
        design matrix; here `X` is a 1D list of file paths (the same
        situation `CountVectorizer` and other text vectorizers are in), so
        that convention does not literally apply. It is still set, to the
        one "how many features" number that actually becomes fixed once
        `fit()` returns: the width of whatever `transform()` will produce.
    """

    def __init__(self, k=31, min_count=1, top_features=10_000, threads=None):
        # scikit-learn convention: __init__ only assigns parameters, with
        # no validation and no other side effects, so that
        # get_params()/set_params()/clone() -- which cross-validation and
        # GridSearchCV rely on to build fresh, unfitted copies of this
        # estimator per fold/candidate -- can always reconstruct an
        # equivalent estimator from exactly these values. Validation
        # happens in fit(), where it belongs.
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads

    def fit(self, X, y=None):
        """Learns `self.vocabulary_` from `X` alone.

        Parameters
        ----------
        X : iterable of str or pathlib.Path
            FASTQ(.gz) file paths for the *training* samples only. Counted
            once each with `fastdna.count(path, k=self.k,
            min_count=self.min_count, threads=self.threads)`.
        y : ignored
            Accepted (and defaulting to `None`) purely to satisfy
            scikit-learn's estimator API -- `Pipeline.fit(X, y)` always
            passes it through to every step. Never read: feature selection
            here is prevalence-based, not label-based (see the module
            docstring and design doc §3, "Why feature selection is
            unsupervised") -- ranking by correlation with `y` computed over
            a fold that also gets evaluated on is exactly the leakage this
            class exists to make impossible, so `y` has no legitimate use
            inside this method at all.

        Returns
        -------
        self, per the scikit-learn convention that `fit()` always returns
        the (now-fitted) estimator, so calls chain: `vectorizer.fit(X).transform(X)`.
        """
        paths = _validate_paths(X, "fit")
        if not paths:
            raise ValueError("KmerVectorizer.fit() requires at least one sample path, got an empty X")
        seen: set[str] = set()
        for path in paths:
            if path in seen:
                raise ValueError(
                    f"X contains a duplicate path: {path!r}. Each training sample may "
                    "appear only once -- a duplicated path would double-count its "
                    "k-mers' prevalence, the vocabulary ranking's primary criterion."
                )
            seen.add(path)
        if self.top_features is not None and (
            not isinstance(self.top_features, (int, np.integer)) or isinstance(self.top_features, bool) or self.top_features <= 0
        ):
            raise ValueError(f"top_features must be a positive int or None, got {self.top_features!r}")

        # Two running tallies, keyed by each k-mer's `kmer_u64` encoding:
        #
        #   prevalence[kmer]  -- in how many of *these training* samples
        #                        the k-mer appears at all (0 or 1 per
        #                        sample, summed across samples)
        #   total_freq[kmer]  -- its summed raw count across those samples
        #
        # Prevalence (a.k.a. document frequency), not summed raw frequency,
        # is the primary selection criterion -- matching the design doc's
        # own choice (§7.6: "Ranked by descending prevalence"). A k-mer
        # present at moderate depth in *every* training sample is a more
        # trustworthy, sample-general signal than one present at enormous
        # depth in a single sample and absent from the rest (a PCR
        # duplicate, a contaminant, a library-prep artifact unique to one
        # file) -- the latter would dominate a total-frequency ranking
        # without being a feature that generalizes across samples at all,
        # which is the entire purpose of selecting features in the first
        # place. Ties in prevalence are broken by total_freq (still a
        # meaningful tiebreaker: among equally prevalent k-mers, more total
        # signal is preferable), and remaining ties by the raw `kmer_u64`
        # value purely for determinism, so `fit()` on identical input
        # always yields an identical `vocabulary_`, independent of Python
        # dict/set iteration order.
        prevalence: dict[int, int] = {}
        total_freq: dict[int, int] = {}
        sequence_of: dict[int, str] = {}

        for path in paths:
            counted = fastdna.count(path, k=self.k, min_count=self.min_count, threads=self.threads)
            table = counted.table
            kmers = table.column("kmer_u64").to_pylist()
            seqs = table.column("kmer_sequence").to_pylist()
            freqs = table.column("frequency").to_pylist()

            for kmer, seq, freq in zip(kmers, seqs, freqs):
                prevalence[kmer] = prevalence.get(kmer, 0) + 1
                total_freq[kmer] = total_freq.get(kmer, 0) + freq
                if kmer not in sequence_of:
                    sequence_of[kmer] = seq

        ranked = sorted(prevalence, key=lambda kmer: (-prevalence[kmer], -total_freq[kmer], kmer))
        if self.top_features is not None:
            ranked = ranked[: self.top_features]

        self.vocabulary_ = np.array(ranked, dtype=np.uint64)
        # Built once here, not recomputed per transform() call: an O(1)
        # dict lookup from kmer_u64 to its column index in the output
        # matrix.
        self._vocab_index_ = {kmer: i for i, kmer in enumerate(ranked)}
        # get_feature_names_out() needs the *decoded* sequence for each
        # selected k-mer, not its u64 encoding. The Rust core already
        # decoded every k-mer it counted into `kmer_sequence` (see
        # `KmerCounts.table`); no pure-Python 2-bit decoder exists anywhere
        # in this package (checked: only `src/kmer.rs::decode_kmer` does,
        # Rust-side), so this reuses that per-fit-sample table lookup
        # (`sequence_of`, populated above) instead of reimplementing
        # decoding here.
        self._feature_sequences_ = [sequence_of[kmer] for kmer in ranked]
        self.n_features_in_ = len(self.vocabulary_)
        return self

    def transform(self, X):
        """Projects `X` onto the vocabulary `fit()` already decided.

        Parameters
        ----------
        X : iterable of str or pathlib.Path
            FASTQ(.gz) file paths -- training samples, held-out samples, or
            entirely new ones; `transform()` makes no distinction, and
            that is the point (see module docstring). Each is counted
            fresh with the same `k`/`min_count`/`threads` used in `fit()`.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (len(X), len(self.vocabulary_))
        K-mers a sample contains that are not in `self.vocabulary_` are
        silently ignored (not an error -- an unseen sample routinely
        contains k-mers the training vocabulary never saw). Vocabulary
        k-mers a sample does not contain become 0. A CSR sparse matrix,
        not a dense array, because a real sample typically touches only a
        small fraction of a 10,000+ k-mer vocabulary, and this is the
        format scikit-learn estimators and XGBoost accept directly without
        an explicit densify step.
        """
        check_is_fitted(self, "vocabulary_")
        paths = _validate_paths(X, "transform")
        n_features = len(self.vocabulary_)

        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for i, path in enumerate(paths):
            counted = fastdna.count(path, k=self.k, min_count=self.min_count, threads=self.threads)
            table = counted.table
            kmers = table.column("kmer_u64").to_pylist()
            freqs = table.column("frequency").to_pylist()
            for kmer, freq in zip(kmers, freqs):
                col = self._vocab_index_.get(kmer)
                if col is not None:
                    rows.append(i)
                    cols.append(col)
                    data.append(freq)

        # float64: most downstream consumers (LogisticRegression, other
        # linear models, and anything that normalizes counts before
        # fitting) expect a floating dtype and would otherwise silently
        # upcast anyway; producing it directly avoids a hidden copy inside
        # whatever comes next in the pipeline. Matches what
        # `TfidfVectorizer` (scikit-learn's own closest analogue) returns.
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(paths), n_features), dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        """The selected vocabulary as decoded k-mer sequence strings, in
        the same order as the columns of any matrix `transform()` returns
        -- i.e. `get_feature_names_out()[j]` names column `j`.

        `input_features` is accepted (and ignored) only to match
        scikit-learn's `get_feature_names_out(input_features=None)`
        signature convention; it is meaningful for transformers whose
        output features are derived *from* named input features, which
        does not describe this one (its "input" is FASTQ file paths, not
        named columns).

        Pairing this with a fitted classifier's own feature-importance
        attribute (e.g. `model.feature_importances_` on an XGBoost/sklearn
        tree model, or `model.coef_` on a linear one) turns "feature 4,217
        mattered most" into an actual DNA sequence that can be looked up
        (e.g. via BLAST) -- the payoff the design doc calls out (§9.6) for
        keeping this as sequences rather than raw integer encodings.
        """
        check_is_fitted(self, "vocabulary_")
        return np.asarray(self._feature_sequences_, dtype=object)

    # `TransformerMixin.fit_transform` already does the right thing here
    # (`self.fit(X, y, **fit_params).transform(X)`) without an override:
    # `transform()` only ever reads `self.vocabulary_`/`self._vocab_index_`,
    # both of which `fit()` will have just finished setting, so there is no
    # correctness reason to special-case this. It does mean `fit_transform`
    # counts each training FASTQ file twice (once to build the vocabulary
    # in `fit()`, once to project in `transform()`) rather than once --
    # a real but deliberate cost, left as the simple, obviously-correct
    # choice for this first version rather than adding a same-call cache
    # (see the task's own guidance to prefer a working simple version over
    # a half-finished optimization).
