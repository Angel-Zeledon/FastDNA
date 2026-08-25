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
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import fastdna
from fastdna import _column_as_array


class _CohortCounts(NamedTuple):
    """Every sample's k-mer table, counted once and stacked end to end.

    The three arrays are the cohort's rows concatenated in `paths` order,
    so row `r` belongs to the sample whose block contains `r`;
    `row_counts[i]` is how many rows sample `i` contributed, which is all
    `np.repeat` needs to recover the row-to-sample mapping without a
    Python loop over the rows themselves.

    Keeping `sequences` as an Arrow array rather than a Python list is the
    point of holding this at all: only the k-mers that survive vocabulary
    selection ever need decoding into Python strings, and there are at
    most `top_features` of those against a cohort-wide row count that is
    routinely several orders of magnitude larger.
    """

    kmers: object  # pyarrow.UInt64Array, one row per (sample, k-mer)
    sequences: object  # pyarrow.StringArray, aligned with `kmers`
    frequencies: object  # pyarrow.UInt32Array, aligned with `kmers`
    row_counts: list  # rows contributed by each sample, in `paths` order


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

    def _validated_fit_paths(self, X):
        """The training-path checks `fit()` performs, shared verbatim with
        `fit_transform()` so both reject the same mistakes with the same
        messages (including naming `fit()`, which is the step that would
        have raised either way).
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
        return paths

    def _count_cohort(self, paths):
        """Counts each path **exactly once** and stacks the resulting k-mer
        tables into one `_CohortCounts`.

        Every caller here needs the same three columns of the same tables,
        so counting is done in one place and the result is passed around
        rather than recomputed: `fit_transform()` learns the vocabulary and
        projects from a single set of counts, where the previous
        `TransformerMixin.fit_transform` (`fit(X).transform(X)`) re-read and
        re-counted every training FASTQ a second time -- a 20-sample cohort
        did 40 counting passes for 20 samples' worth of data, and counting
        is by far the most expensive thing this class does.
        """
        kmer_arrays, sequence_arrays, frequency_arrays, row_counts = [], [], [], []
        for path in paths:
            table = fastdna.count(path, k=self.k, min_count=self.min_count, threads=self.threads).table
            kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
            sequence_arrays.append(_column_as_array(table.column("kmer_sequence")))
            frequency_arrays.append(_column_as_array(table.column("frequency")))
            row_counts.append(table.num_rows)
        return _CohortCounts(
            kmers=pa.concat_arrays(kmer_arrays),
            sequences=pa.concat_arrays(sequence_arrays),
            frequencies=pa.concat_arrays(frequency_arrays),
            row_counts=row_counts,
        )

    def _learn_vocabulary(self, counts):
        """Ranks the cohort's k-mers and assigns `vocabulary_`,
        `_feature_sequences_` and `n_features_in_`.

        Two running tallies, keyed by each k-mer's `kmer_u64` encoding:

          prevalence[kmer]  -- in how many of *these training* samples
                               the k-mer appears at all (0 or 1 per
                               sample, summed across samples)
          total_freq[kmer]  -- its summed raw count across those samples

        Prevalence (a.k.a. document frequency), not summed raw frequency,
        is the primary selection criterion -- matching the design doc's
        own choice (§7.6: "Ranked by descending prevalence"). A k-mer
        present at moderate depth in *every* training sample is a more
        trustworthy, sample-general signal than one present at enormous
        depth in a single sample and absent from the rest (a PCR
        duplicate, a contaminant, a library-prep artifact unique to one
        file) -- the latter would dominate a total-frequency ranking
        without being a feature that generalizes across samples at all,
        which is the entire purpose of selecting features in the first
        place. Ties in prevalence are broken by total_freq (still a
        meaningful tiebreaker: among equally prevalent k-mers, more total
        signal is preferable), and remaining ties by the raw `kmer_u64`
        value purely for determinism, so `fit()` on identical input
        always yields an identical `vocabulary_`.

        Both tallies are computed with one hash pass over the stacked
        cohort rather than a Python loop: `dictionary_encode` assigns each
        distinct k-mer a code (one C++ hash probe per row, replacing three
        Python dict operations per row), after which prevalence is just
        "how many rows carry this code" -- a k-mer appears at most once in
        a sample's count table, so its row count *is* its sample count --
        and total_freq is the same tally weighted by `frequency`. For a
        20-sample cohort of a million k-mers each, that is 20 million
        Python-level dict updates replaced by two `bincount` passes.
        """
        codes = pc.dictionary_encode(counts.kmers)
        row_code = np.asarray(codes.indices).astype(np.intp, copy=False)
        distinct = np.asarray(codes.dictionary)
        n_distinct = distinct.size

        prevalence = np.bincount(row_code, minlength=n_distinct).astype(np.int64)
        # float64 weights hold these sums exactly: `frequency` is uint32
        # and no partial sum comes anywhere near 2**53 (that bound is
        # ~9e15 occurrences of one k-mer), so the ordering below is the one
        # a Python `+=` loop over the same integers would have produced.
        total_freq = np.bincount(row_code, weights=np.asarray(counts.frequencies), minlength=n_distinct)

        # `lexsort` takes its primary key last, so this is exactly the
        # `(-prevalence, -total_freq, kmer)` ordering described above.
        ranked = np.lexsort((distinct, -total_freq, -prevalence))
        if self.top_features is not None:
            ranked = ranked[: self.top_features]

        self.vocabulary_ = distinct[ranked]
        # get_feature_names_out() needs the *decoded* sequence for each
        # selected k-mer, not its u64 encoding. The Rust core already
        # decoded every k-mer it counted into `kmer_sequence` (see
        # `KmerCounts.table`); no pure-Python 2-bit decoder exists anywhere
        # in this package (checked: only `src/kmer.rs::decode_kmer` does,
        # Rust-side), so this reuses that table column -- but takes only
        # the rows the selected k-mers first appeared in, so the number of
        # Python strings built is `len(vocabulary_)` rather than one per
        # k-mer of every training sample.
        first_row = np.empty(n_distinct, dtype=np.int64)
        descending = np.arange(row_code.size - 1, -1, -1)
        first_row[row_code[descending]] = descending
        self._feature_sequences_ = counts.sequences.take(pa.array(first_row[ranked])).to_pylist()
        self.n_features_in_ = len(self.vocabulary_)

    def _project(self, counts, n_samples):
        """The sparse `(n_samples, len(vocabulary_))` count matrix for an
        already-counted cohort.

        `index_in` resolves every row's `kmer_u64` against the vocabulary
        in one C++ hash pass (k-mers outside it come back null, i.e. the
        "silently ignored" case documented on `transform()`), replacing one
        Python dict lookup and up to three `list.append` calls per row of
        every sample's table.
        """
        column = np.asarray(pc.fill_null(pc.index_in(counts.kmers, value_set=pa.array(self.vocabulary_)), -1))
        kept = column >= 0
        rows = np.repeat(np.arange(n_samples, dtype=np.int64), counts.row_counts)[kept]
        # float64: most downstream consumers (LogisticRegression, other
        # linear models, and anything that normalizes counts before
        # fitting) expect a floating dtype and would otherwise silently
        # upcast anyway; producing it directly avoids a hidden copy inside
        # whatever comes next in the pipeline. Matches what
        # `TfidfVectorizer` (scikit-learn's own closest analogue) returns.
        data = np.asarray(counts.frequencies)[kept].astype(np.float64)
        return sparse.csr_matrix(
            (data, (rows, column[kept].astype(np.int64))),
            shape=(n_samples, len(self.vocabulary_)),
            dtype=np.float64,
        )

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
        paths = self._validated_fit_paths(X)
        self._learn_vocabulary(self._count_cohort(paths))
        return self

    def fit_transform(self, X, y=None):
        """`fit(X)` and `transform(X)` over **one** pass of counting.

        `TransformerMixin.fit_transform`'s default is
        `self.fit(X, y).transform(X)`, which counts every training FASTQ
        file twice -- once to rank the vocabulary, once to project onto it.
        This override counts each file once and reuses those counts for
        both steps, so a 20-sample cohort does 20 counting passes instead
        of 40. Nothing else changes: the vocabulary is still decided by
        `_learn_vocabulary()` from `X` alone before `_project()` is
        reached, and `_project()` still reads nothing but the vocabulary
        that decision produced -- the leakage guarantee in the module
        docstring is a property of that ordering, not of how many times
        the files were read. `fastdna.count()` is deterministic, so the
        matrix is identical to the one two passes produced.

        `y` is ignored, exactly as in `fit()`.
        """
        paths = self._validated_fit_paths(X)
        counts = self._count_cohort(paths)
        self._learn_vocabulary(counts)
        return self._project(counts, len(paths))

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
        return self._project(self._count_cohort(paths), len(paths))

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
