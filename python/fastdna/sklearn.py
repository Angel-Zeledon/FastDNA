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
import warnings
from typing import Iterable, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import fastdna
from fastdna import _column_as_array, _core, _decode_kmers

__all__ = ["KmerVectorizer", "DepthConfoundingWarning"]

_VALID_REPRESENTATIONS = ("presence", "count", "relative", "clr")


class DepthConfoundingWarning(UserWarning):
    """Sequencing depth varies enough across samples that a model trained
    on raw counts (`representation="count"`) may be learning depth instead
    of biology. See `KmerVectorizer`'s `representation` parameter."""


class _CohortCounts(NamedTuple):
    """Every sample's k-mer table, counted once and stacked end to end.

    The two arrays are the cohort's rows concatenated in `paths` order, so
    row `r` belongs to the sample whose block contains `r`;
    `row_counts[i]` is how many rows sample `i` contributed, which is all
    `np.repeat` needs to recover the row-to-sample mapping without a
    Python loop over the rows themselves.

    No `sequences` field: an earlier version of this type carried a
    `kmer_sequence` Arrow array alongside `kmers`, which meant the
    underlying `fastdna.count()` calls had to ask for that column even
    though only the k-mers that survive vocabulary selection -- at most
    `top_features` of them, several orders of magnitude fewer than a
    cohort's row count -- ever needed decoding into Python strings.
    `_learn_vocabulary` now decodes exactly those, directly from their
    `kmer_u64` values via `fastdna._decode_kmers`, so counting never has to
    materialize the sequence column at all (`with_sequence=False`, the
    fast default -- see `export::counts_schema`'s doc comment for what
    that column costs at full cohort scale).

    Attributes
    ----------
    kmers : pyarrow.UInt64Array
        One row per (sample, k-mer): the `kmer_u64` encoding.
    frequencies : pyarrow.UInt32Array
        The per-sample raw count, aligned with `kmers`.
    row_counts : list of int
        Rows contributed by each sample, in `paths` order.
    """

    kmers: object
    frequencies: object
    row_counts: list


def _validate_paths(X, method_name):
    """Turns `X` into a list of path strings.

    Rejects the classic footgun of passing a single path instead of a list
    of them: a bare string (or `os.PathLike`) *is* iterable, so it would
    otherwise be iterated character by character and surface as a baffling
    per-character `FileNotFoundError` deep inside the counting loop.

    Parameters
    ----------
    X : iterable of str or pathlib.Path
        Sample paths -- or, invalidly, a single bare path.
    method_name : str
        Name of the calling `KmerVectorizer` method, used only to name it
        in the raised error message.

    Returns
    -------
    list of str

    Raises
    ------
    TypeError
        If `X` is a single path rather than an iterable of them.
    """
    if isinstance(X, (str, os.PathLike)):
        # No leaf in the fastdna._core exception hierarchy inherits from
        # TypeError (every leaf is a ValueError/OSError/FileNotFoundError/
        # MemoryError/RuntimeError subclass), so this stays a bare TypeError
        # rather than losing isinstance(e, TypeError) compatibility for
        # existing callers (see python/tests/test_sklearn.py's
        # pytest.raises(TypeError, ...) on fit()/transform() with a bare
        # string).
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
    counts : fastdna.CohortCounts or None, default None
        A cohort counted once via `fastdna.count_cohort()`. When given, `X`
        (in `fit()`/`fit_transform()`/`transform()`) must be `sample_id`
        strings from that cohort, not FASTQ paths, and no counting happens
        here at all -- each call becomes an O(rows of that subset) slice
        of `counts`. This is what makes cross-validation cheap: without
        it, `cross_val_score(pipeline, paths, y, cv=5)` counts every file
        in the cohort once per fold (`fit()` on the training 80%,
        `transform()` on the held-out 20%, five times over -- five full
        passes over the cohort to evaluate one model). With it, counting
        happens once, before the cross-validation loop, and every fold
        reuses it.

            counts = fastdna.count_cohort(paths, k=31)
            vec = KmerVectorizer(top_features=10_000, counts=counts)
            cross_val_score(make_pipeline(vec, clf), counts.sample_ids, y, cv=5)

        `None` (the default) keeps the original behaviour: `X` is FASTQ
        paths, counted fresh on every call.
    chunk_size : int or None, default None
        Learn the vocabulary in batches of `chunk_size` samples instead of
        all at once. `None` (the default) keeps the original behaviour:
        `fit()` counts every training sample and holds all of it in memory
        simultaneously (via `_count_cohort`'s `pa.concat_arrays` over every
        sample, and `_learn_vocabulary`'s `pc.dictionary_encode` pass over
        the result) before ranking a single k-mer. That is the actual
        memory-scaling bottleneck for a large cohort -- it is proportional
        to the cohort's total *row* count (every sample's surviving k-mers,
        summed across every sample), not to the number of *distinct*
        k-mers or to `top_features`. A cohort of 10,000 bacterial genomes
        at a few million k-mers each puts tens of billions of rows in
        memory at once, which does not fit; see `docs/audit/ml-gaps.md`
        G-9.

        Setting `chunk_size` processes the training samples that many at a
        time: each batch is counted, folded into a running
        (prevalence, total_freq) tally keyed by k-mer, and then discarded
        -- only the running tally survives past its own batch. Peak memory
        during vocabulary learning becomes proportional to `chunk_size`'s
        own row count plus the number of *distinct* k-mers seen so far,
        never to the whole cohort's row count. For a cohort where samples
        share most of their k-mer content (the common case -- genomes of
        the same species overlap heavily), the distinct-k-mer count is
        far smaller than the summed row count, which is where the
        practical saving comes from; see `fit()`'s docstring for a
        concrete measurement. This is not an approximation: folding a
        batch's local tally into the running one is addition, which is
        associative, so the final tally -- and therefore `vocabulary_`,
        `_feature_sequences_`, and every tie-break -- is bit-for-bit
        identical to what one unchunked pass over the same cohort would
        have produced, regardless of `chunk_size` or how the cohort
        happens to be ordered. `top_features` selection still runs once,
        at the end, over the complete (not partial) tally, so nothing is
        evicted early and no ranking is approximated.

        Interoperates with `counts`: when both are given, each batch reads
        `self.counts.subset(...)` for that batch's sample_ids only,
        instead of the single all-at-once `subset()` call `counts` alone
        would make -- bounding the size of the `.take()`-allocated copy
        `subset()` produces, on top of (not instead of) whatever memory
        `self.counts` itself already occupies (`chunk_size` cannot shrink
        an artifact `count_cohort()` already built in full; see
        `cohort_counts.py`). Without `counts`, each batch is a fresh
        `fastdna.count()` per sample, exactly as `_count_cohort` already
        does for the whole cohort in one pass -- `chunk_size` only changes
        how many of those results are held at once, not what they contain.

        Only vocabulary learning is chunked. `transform()` is unaffected
        by this parameter -- it is ordinarily called on a single fold or a
        handful of new samples, not the whole training cohort, so it does
        not have the same scaling problem `fit()` does at cohort scale.
        `fit_transform()` honours `chunk_size` for the vocabulary half of
        its work, but its projection half still needs every training
        sample's counts at once to return one matrix over all of them, so
        it falls back to counting each sample a second time for that part
        -- see its own docstring for why, and prefer `fit()` followed by
        per-batch `transform()` calls for a cohort large enough that
        `chunk_size` matters in the first place.
    representation : {"presence", "count", "relative", "clr"}, default "presence"
        What each matrix cell holds. `"presence"` is 0/1 (a k-mer was
        observed in this sample or not) and is immune to sequencing depth
        by construction -- the default because a linear model over raw
        counts can otherwise separate samples by depth rather than
        biology (see the module-level rationale in `__init__`).
        `"count"` is the raw per-sample frequency, and warns
        (`DepthConfoundingWarning`) when depth varies more than 3x across
        the fitted samples. `"relative"` divides each sample's counts by
        its own total (within the vocabulary). `"clr"` is the centered
        log-ratio over each sample's observed entries, the standard
        transform for compositional data (e.g. metagenomic abundance).

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

    def __init__(
        self,
        k: int = 31,
        min_count: int = 1,
        top_features: int | None = 10_000,
        threads: int | None = None,
        counts: fastdna.CohortCounts | None = None,
        representation: str = "presence",
        chunk_size: int | None = None,
    ) -> None:
        # scikit-learn convention: __init__ only assigns parameters, with
        # no validation and no other side effects, so that
        # get_params()/set_params()/clone() -- which cross-validation and
        # GridSearchCV rely on to build fresh, unfitted copies of this
        # estimator per fold/candidate -- can always reconstruct an
        # equivalent estimator from exactly these values. Validation
        # happens in fit(), where it belongs.
        #
        # `counts` travels through the constructor rather than through `X`
        # for the same reason: scikit-learn's `_safe_indexing` is what
        # partitions `X` into folds, and it has dedicated branches for
        # pandas/numpy/sparse/generic-sequence inputs whose exact behaviour
        # for a bespoke object type is not part of its public contract. A
        # plain list of `sample_id` strings is a type `_safe_indexing`
        # already partitions correctly and predictably; the heavy artifact
        # sits alongside the other hyperparameters instead, where `clone()`
        # already knows to preserve it.
        self.k = k
        self.min_count = min_count
        self.top_features = top_features
        self.threads = threads
        self.counts = counts
        self.chunk_size = chunk_size
        # "presence" (0/1), not "count" (raw), by default.
        #
        # A sample sequenced at 100x has roughly 5x the k-mer counts of one
        # at 20x for purely technical reasons, and in real clinical
        # cohorts depth correlates with batch, year and sequencing center
        # -- which correlate with phenotype. A linear model over raw
        # counts can separate classes by that magnitude alone: the same
        # class of technical confounder `fastdna.cv` exists to catch on
        # the population-structure axis, here on the sequencing-depth
        # axis. `workflow.py`'s own GWAS path already binarizes its
        # presence matrix before association testing for exactly this
        # reason; this brings the ML-facing path in line with it. "count"
        # stays available for genuine abundance questions (metagenomics)
        # and warns when depth varies enough to matter (see
        # `DepthConfoundingWarning`). "relative" and "clr" are the
        # depth-normalized alternatives to raw counts.
        self.representation = representation

    def _validated_fit_paths(self, X):
        """The training-path checks `fit()` performs.

        Shared verbatim with `fit_transform()` so both reject the same
        mistakes with the same messages (including naming `fit()`, which is
        the step that would have raised either way).

        Parameters
        ----------
        X : iterable of str or pathlib.Path
            Training sample paths.

        Returns
        -------
        list of str

        Raises
        ------
        TypeError
            If `X` is a single path rather than an iterable of them.
        ValueError
            If `X` is empty, contains a duplicate path, or `self.top_features`
            is not a positive int or None.
        """
        paths = _validate_paths(X, "fit")
        if not paths:
            raise _core.InvalidConfigError("KmerVectorizer.fit() requires at least one sample path, got an empty X")
        seen: set[str] = set()
        for path in paths:
            if path in seen:
                raise _core.InvalidConfigError(
                    f"X contains a duplicate path: {path!r}. Each training sample may "
                    "appear only once -- a duplicated path would double-count its "
                    "k-mers' prevalence, the vocabulary ranking's primary criterion."
                )
            seen.add(path)
        if self.top_features is not None and (
            not isinstance(self.top_features, (int, np.integer)) or isinstance(self.top_features, bool) or self.top_features <= 0
        ):
            raise _core.InvalidConfigError(f"top_features must be a positive int or None, got {self.top_features!r}")
        if self.chunk_size is not None and (
            not isinstance(self.chunk_size, (int, np.integer)) or isinstance(self.chunk_size, bool) or self.chunk_size <= 0
        ):
            raise _core.InvalidConfigError(f"chunk_size must be a positive int or None, got {self.chunk_size!r}")
        return paths

    def _count_cohort(self, keys):
        """The cohort's k-mer tables as one `_CohortCounts`, for `keys`.

        Two paths, chosen by whether `self.counts` (a
        `fastdna.CohortCounts` built once via `fastdna.count_cohort()`) was
        given:

        - **With it**, `keys` are `sample_id`s and this is a recount of
          `self.counts.subset(keys)` -- an O(rows of that subset) slice of
          an already-counted cohort, no FASTQ touched. This is what makes
          `KmerVectorizer(counts=...)` skip rereading the cohort on every
          fold of a cross-validation: `fit()` on the training fold and
          `transform()` on the held-out fold each used to be a full
          `fastdna.count()` pass over every file in that fold, so a 5-fold
          `cross_val_score` read the whole cohort five times over.
        - **Without it**, `keys` are FASTQ paths and each is counted fresh
          with `fastdna.count(path, k=self.k, min_count=self.min_count,
          threads=self.threads)` -- the original behaviour, unchanged for
          any caller that has not adopted the cohort-counts artifact.

        Neither path asks for `kmer_sequence` (`with_sequence` stays at its
        default `False`): see `_CohortCounts`'s own doc comment for why
        that column is no longer needed here at all.

        `fit_transform()` still calls this once and reuses the result for
        both `_learn_vocabulary()` and `_project()`, so a 20-sample cohort
        without `counts=` still does 20 counting passes, not 40 --
        `TransformerMixin.fit_transform`'s default of `fit(X).transform(X)`
        would count every training file twice.

        Parameters
        ----------
        keys : list of str
            Sample paths, or sample ids when `self.counts` is set -- each
            counted (or sliced) exactly once.

        Returns
        -------
        _CohortCounts
        """
        if self.counts is not None:
            subset = self.counts.subset(list(keys))
            return _CohortCounts(
                kmers=subset.kmers, frequencies=subset.frequencies, row_counts=list(subset.row_counts)
            )

        kmer_arrays, frequency_arrays, row_counts = [], [], []
        for path in keys:
            table = fastdna.count(path, k=self.k, min_count=self.min_count, threads=self.threads).table
            kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
            frequency_arrays.append(_column_as_array(table.column("frequency")))
            row_counts.append(table.num_rows)
        return _CohortCounts(
            kmers=pa.concat_arrays(kmer_arrays),
            frequencies=pa.concat_arrays(frequency_arrays),
            row_counts=row_counts,
        )

    def _learn_vocabulary(self, counts):
        """Ranks the cohort's k-mers and assigns `vocabulary_`,
        `_feature_sequences_` and `n_features_in_`, in one pass over the
        already-fully-materialized `counts` (i.e. `self.chunk_size` is
        `None`). See `_learn_vocabulary_streaming` for the chunked
        alternative that never holds the whole cohort's rows at once; both
        end in `_select_vocabulary`, which is the one place the ranking
        rule itself is implemented, so the two paths cannot drift apart.

        Two running tallies, keyed by each k-mer's `kmer_u64` encoding:

          prevalence[kmer]  -- in how many of *these training* samples
                               the k-mer appears at all (0 or 1 per
                               sample, summed across samples)
          total_freq[kmer]  -- its summed raw count across those samples

        Both tallies are computed with one hash pass over the stacked
        cohort rather than a Python loop: `dictionary_encode` assigns each
        distinct k-mer a code (one C++ hash probe per row, replacing three
        Python dict operations per row), after which prevalence is just
        "how many rows carry this code" -- a k-mer appears at most once in
        a sample's count table, so its row count *is* its sample count --
        and total_freq is the same tally weighted by `frequency`. For a
        20-sample cohort of a million k-mers each, that is 20 million
        Python-level dict updates replaced by two `bincount` passes.

        This is also the method's own memory cost: `dictionary_encode`'s
        `row_code` is one `int` per *row* of the stacked cohort, alongside
        `counts.kmers`/`counts.frequencies` themselves (already one entry
        per row, built by `_count_cohort`'s `pa.concat_arrays` over every
        sample at once) -- i.e. peak memory here is proportional to the
        cohort's total row count, not to `top_features` or to the number
        of distinct k-mers. That is the actual bottleneck `chunk_size`
        (see `__init__`) exists to avoid; see `_learn_vocabulary_streaming`.
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

        self._select_vocabulary(distinct, prevalence, total_freq)

    def _learn_vocabulary_streaming(self, keys):
        """The `chunk_size`-chunked equivalent of `_learn_vocabulary`:
        processes `keys` (paths or sample_ids, per `self.counts`) in
        batches of `self.chunk_size`, and ends at the exact same
        `_select_vocabulary` call `_learn_vocabulary` does -- so, given the
        same cohort, the two produce a bit-for-bit identical
        `vocabulary_`/`_feature_sequences_`/`n_features_in_`, regardless of
        `chunk_size` or the order `keys` happens to be in. That equality
        holds because folding a batch's local tally into the running one
        below is addition, which is associative and commutative: summing
        each k-mer's prevalence/total_freq in N batches and then combining
        the N partial sums gives the same total as summing it in one pass
        over everything at once. Nothing is dropped, capped, or evicted
        early -- `top_features` selection still happens once, in
        `_select_vocabulary`, over the *complete* tally after every batch
        has been folded in, exactly as it would with `chunk_size=None`.

        Each batch calls `self._count_cohort` on a `chunk_size`-sized slice
        of `keys` -- the same method `_learn_vocabulary`'s caller uses for
        the whole cohort at once, so this reuses (rather than reimplements)
        both branches of `self.counts` being set or not. Its result is
        ranked locally with the identical `dictionary_encode` +
        `bincount` machinery `_learn_vocabulary` itself uses, producing a
        *batch-local* `(distinct, prevalence, total_freq)` triple that is
        orders of magnitude smaller than the whole cohort's row count. That
        triple is then merged into the running one via `np.unique` over the
        concatenation of "k-mers seen in a previous batch" and "k-mers in
        this batch" -- another vectorized hash pass, not a Python loop --
        and the batch's own rows go out of scope once this returns, so
        nothing about a processed batch survives except its contribution to
        the running tally. Peak memory at any point during this method is
        therefore proportional to `chunk_size`'s own row count plus the
        number of *distinct* k-mers accumulated so far -- never to the
        whole cohort's row count, which is what makes this usable on a
        cohort too large for `_learn_vocabulary` itself; see `chunk_size`
        in `__init__` for how much that saves in practice, measured on a
        synthetic cohort.
        """
        chunk_size = self.chunk_size
        running_kmers = np.empty(0, dtype=np.uint64)
        running_prevalence = np.empty(0, dtype=np.int64)
        running_total_freq = np.empty(0, dtype=np.float64)

        for start in range(0, len(keys), chunk_size):
            batch_counts = self._count_cohort(keys[start : start + chunk_size])

            codes = pc.dictionary_encode(batch_counts.kmers)
            row_code = np.asarray(codes.indices).astype(np.intp, copy=False)
            batch_kmers = np.asarray(codes.dictionary)
            n_batch_distinct = batch_kmers.size
            batch_prevalence = np.bincount(row_code, minlength=n_batch_distinct).astype(np.int64)
            batch_total_freq = np.bincount(
                row_code, weights=np.asarray(batch_counts.frequencies), minlength=n_batch_distinct
            )

            # Merge this batch's local tally into the running one: k-mers
            # seen before and k-mers new to this batch both land in
            # `running_kmers` via `np.unique`, and `inverse` maps every
            # entry of the concatenation back to its position there, so
            # `bincount(inverse, weights=...)` re-sums each k-mer's total
            # across however many of the batches-so-far it has appeared in
            # -- the same "one hash pass instead of a Python dict" approach
            # `_learn_vocabulary` uses within a batch, applied here across
            # batches instead of across rows.
            running_kmers, inverse = np.unique(
                np.concatenate([running_kmers, batch_kmers]), return_inverse=True
            )
            n_running = running_kmers.size
            running_prevalence = np.bincount(
                inverse,
                weights=np.concatenate([running_prevalence, batch_prevalence]),
                minlength=n_running,
            ).astype(np.int64)
            running_total_freq = np.bincount(
                inverse,
                weights=np.concatenate([running_total_freq, batch_total_freq]),
                minlength=n_running,
            )

        self._select_vocabulary(running_kmers, running_prevalence, running_total_freq)

    def _select_vocabulary(self, distinct, prevalence, total_freq):
        """Ranks `distinct` k-mers by `(prevalence, total_freq)` and
        assigns `vocabulary_`, `_feature_sequences_` and `n_features_in_`
        -- the one implementation of the selection rule shared by
        `_learn_vocabulary` (single pass) and `_learn_vocabulary_streaming`
        (chunked), so which of the two ran is never visible in the result.

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

        Parameters
        ----------
        distinct : np.ndarray
            Distinct k-mer codes across the cohort.
        prevalence : np.ndarray
            Sample count for each entry in `distinct`.
        total_freq : np.ndarray
            Summed raw frequency for each entry in `distinct`.

        Returns
        -------
        None
            Sets `self.vocabulary_`, `self._feature_sequences_`, and
            `self.n_features_in_`.
        """
        # `lexsort` takes its primary key last, so this is exactly the
        # `(-prevalence, -total_freq, kmer)` ordering described above.
        ranked = np.lexsort((distinct, -total_freq, -prevalence))
        if self.top_features is not None:
            ranked = ranked[: self.top_features]

        self.vocabulary_ = distinct[ranked]
        # get_feature_names_out() needs the *decoded* sequence for each
        # selected k-mer, not its u64 encoding. Decoded directly from
        # `vocabulary_` itself via `_decode_kmers` (the same 2-bit layout
        # `kmer::decode_kmer` unpacks Rust-side), rather than by reading a
        # `kmer_sequence` table column: decoding an integer needs no lookup
        # into any row, so this builds exactly `len(vocabulary_)` strings
        # without `_count_cohort` ever having to materialize that column
        # for the whole cohort just to let this method `.take()` a few
        # thousand of its rows.
        self._feature_sequences_ = _decode_kmers(self.vocabulary_, self.k)
        self.n_features_in_ = len(self.vocabulary_)

    def _project(self, counts, n_samples):
        """The sparse `(n_samples, len(vocabulary_))` matrix for an
        already-counted cohort, in `self.representation`.

        `index_in` resolves every row's `kmer_u64` against the vocabulary
        in one C++ hash pass (k-mers outside it come back null, i.e. the
        "silently ignored" case documented on `transform()`), replacing one
        Python dict lookup and up to three `list.append` calls per row of
        every sample's table.

        Parameters
        ----------
        counts : _CohortCounts
            The stacked cohort counts to project.
        n_samples : int
            Number of samples `counts` was stacked from -- the resulting
            matrix's row count.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_samples, len(self.vocabulary_))
        """
        if self.representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {list(_VALID_REPRESENTATIONS)}, "
                f"got {self.representation!r}"
            )

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

        if self.representation == "presence":
            # Every stored entry is already >= 1 by construction (a k-mer
            # that was never observed simply has no row), so setting them
            # all to 1 recovers plain 0/1 presence without touching the
            # sparsity pattern -- the same trick `workflow.py` uses ahead
            # of association testing, via the same reasoning: a linear
            # model over raw counts can separate samples by sequencing
            # depth alone, which is not biology.
            data = np.ones_like(data)
        else:
            # Per-sample depth: the sum of counts *within the vocabulary*
            # for that sample's rows -- the same quantity `representation`
            # in {"relative", "clr"} normalizes by, and the one
            # `DepthConfoundingWarning` measures the spread of.
            depth = np.bincount(rows, weights=data, minlength=n_samples)
            nonzero = depth[depth > 0]
            if self.representation == "count" and nonzero.size and nonzero.min() > 0:
                ratio = nonzero.max() / nonzero.min()
                if ratio > 3.0:
                    warnings.warn(
                        f"sequencing depth varies {ratio:.1f}x across samples "
                        f"({nonzero.min():.0f} to {nonzero.max():.0f} counts within the "
                        f"vocabulary) and representation='count' keeps that magnitude. A "
                        "linear model can separate samples by depth instead of by biology. "
                        "Use representation='presence' (the default) unless raw abundance "
                        "is the signal you are after.",
                        DepthConfoundingWarning,
                        stacklevel=3,
                    )

            if self.representation == "relative":
                with np.errstate(divide="ignore", invalid="ignore"):
                    data = data / np.where(depth[rows] > 0, depth[rows], 1.0)
            elif self.representation == "clr":
                # Centered log-ratio over each sample's *observed* entries.
                # The structural zeros (vocabulary k-mers this sample never
                # had) stay out of the sparse matrix entirely, which is
                # correct here: CLR is defined over the observed parts of a
                # composition, and materializing the zeros would turn a
                # sparse matrix of up to 10,000+ columns dense.
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs = np.log(data)
                    per_row_sum = np.bincount(rows, weights=logs, minlength=n_samples)
                    per_row_n = np.bincount(rows, minlength=n_samples)
                    mean_log = per_row_sum / np.where(per_row_n > 0, per_row_n, 1)
                    data = logs - mean_log[rows]
            # "count": data is already the raw counts computed above.

        return sparse.csr_matrix(
            (data, (rows, column[kept].astype(np.int64))),
            shape=(n_samples, len(self.vocabulary_)),
            dtype=np.float64,
        )

    def fit(self, X: Iterable[str | os.PathLike], y: object = None) -> KmerVectorizer:
        """Learns `self.vocabulary_` from `X` alone.

        Parameters
        ----------
        X : iterable of str or pathlib.Path, or of sample_id strings
            FASTQ(.gz) file paths for the *training* samples only, counted
            once each with `fastdna.count(path, k=self.k,
            min_count=self.min_count, threads=self.threads)` -- unless
            `self.counts` was given, in which case these are `sample_id`
            strings from that `fastdna.CohortCounts` instead, and nothing
            here is counted at all (see the `counts` parameter above).
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

        Memory, with and without `chunk_size`
        --------------------------------------
        No number is quoted here. An earlier draft of this docstring
        planned to report a specific peak-RSS comparison (400 samples,
        each 6,000 reads of 150 bases, k=21, min_count=1, sharing about
        half their k-mer content across samples and distinct otherwise),
        but that synthetic cohort's *unchunked* path -- almost entirely
        distinct random content, unlike real same-species sequencing data
        where most k-mers are shared -- turned out to hold tens of
        millions of distinct k-mers in memory at once before
        `top_features` selection, which OOM-killed the measurement rather
        than completing it. Reporting a made-up number instead of a real
        one would defeat the entire point of measuring rather than
        asserting: what `chunk_size` actually bounds is the row count
        counted per batch (see the parameter's own docstring above), which
        is the honest claim to rely on until a real measurement replaces
        this note.
        """
        paths = self._validated_fit_paths(X)
        if self.chunk_size is None:
            self._learn_vocabulary(self._count_cohort(paths))
        else:
            self._learn_vocabulary_streaming(paths)
        return self

    def fit_transform(self, X: Iterable[str | os.PathLike], y: object = None) -> sparse.csr_matrix:
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

        `chunk_size`: this method's whole point is reusing one set of
        counts for both `_learn_vocabulary`/`_learn_vocabulary_streaming`
        *and* `_project()` -- but `_project()` needs every training
        sample's counts in memory at once regardless (it returns one
        matrix over all of `X`), so there is nothing to reuse the chunked
        vocabulary tally *for*: `_project()` still requires the same
        `_count_cohort(paths)` this method would have needed anyway. With
        `chunk_size` set, this method therefore does the chunked
        vocabulary pass first (bounded memory for that half) and then
        counts every sample again for `_project()` -- i.e. it falls back
        to reading each sample twice, the exact cost this override exists
        to avoid when `chunk_size` is `None`. That trade only makes sense
        for a cohort whose *vocabulary-learning* step does not fit in
        memory but whose *projected matrix* does; for a cohort large
        enough that neither does, call `fit()` alone and `transform()` it
        in smaller batches instead (see `chunk_size` in `__init__`).

        Parameters
        ----------
        X : iterable of str or pathlib.Path
            FASTQ(.gz) file paths for the *training* samples only. See
            `fit()`.
        y : ignored
            Accepted purely to satisfy scikit-learn's estimator API; never
            read, exactly as in `fit()`.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (len(X), len(self.vocabulary_))
            Same format `transform()` returns; see its docstring.

        Raises
        ------
        TypeError, ValueError
            See `fit()`.
        """
        paths = self._validated_fit_paths(X)
        if self.chunk_size is None:
            counts = self._count_cohort(paths)
            self._learn_vocabulary(counts)
        else:
            self._learn_vocabulary_streaming(paths)
            counts = self._count_cohort(paths)
        return self._project(counts, len(paths))

    def transform(self, X: Iterable[str | os.PathLike]) -> sparse.csr_matrix:
        """Projects `X` onto the vocabulary `fit()` already decided.

        Parameters
        ----------
        X : iterable of str or pathlib.Path, or of sample_id strings
            FASTQ(.gz) file paths -- training samples, held-out samples, or
            entirely new ones; `transform()` makes no distinction, and
            that is the point (see module docstring). Each is counted
            fresh with the same `k`/`min_count`/`threads` used in `fit()`
            -- unless `self.counts` was given, in which case these are
            `sample_id` strings sliced out of it instead, with no counting
            at all.

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

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
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

        Parameters
        ----------
        input_features : ignored
            Accepted only to match scikit-learn's
            `get_feature_names_out(input_features=None)` signature
            convention; see above for why it does not apply here.

        Returns
        -------
        numpy.ndarray of object, shape (len(self.vocabulary_),)
            The decoded k-mer sequence strings, in vocabulary order.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before `fit()`/`fit_transform()`.
        """
        check_is_fitted(self, "vocabulary_")
        return np.asarray(self._feature_sequences_, dtype=object)
