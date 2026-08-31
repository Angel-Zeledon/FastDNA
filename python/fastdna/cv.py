"""fastdna.cv -- population-structure-aware, leakage-safe model evaluation.

## Why this module exists

Bacterial and viral populations are clonal: a cohort of genomes is not a
sample of independent observations but a set of nested clusters of close
relatives. Sampling is clonal too -- outbreak isolates, hospital
collections, sequencing campaigns all over-represent whichever lineages
happened to be circulating.

That structure confounds genomic machine learning in a specific,
well-documented way. A random cross-validation split scatters members of
one clone across the train/test boundary, so the "held-out" fold contains
near-copies of training samples. The model can score highly by
recognizing the lineage rather than the phenotype, and the reported
accuracy is an estimate of a question nobody asked. A PLOS Biology 2025
analysis over 24,000+ genomes showed this inflating published
antimicrobial-resistance prediction numbers; the Briefings in
Bioinformatics 2024 AMR benchmark evaluated its 78 datasets under three
different split strategies precisely for this reason; and arXiv
2502.07749 names phylogeny-aware cross-validation as the field's missing
standard tool.

The missing tool needs a phylogeny, or something that stands in for one.
That is the part FastDNA can supply without an aligner, a tree builder,
or a reference: `fastdna.compare_all()` already computes all-pairs Mash
distances from MinHash sketches, which is a serviceable proxy for genomic
relatedness at exactly the resolution this problem needs -- "are these two
isolates near-clonal?" -- and it runs in seconds on a cohort where
building a real phylogeny would take hours. `lineage_groups()` turns
those distances into cluster labels; `LineageKFold` turns the labels into
a scikit-learn splitter; `permutation_importance_pvalues()` puts a
significance estimate on the features that survive.

This module deliberately does not implement any of the statistics
downstream of that (mixed-model association testing, heritability, etc.).
pyseer and kmersGWAS exist and are well validated; re-deriving a
linear-mixed-model implementation here would be a worse version of
software people already trust.

## What this module does not fix

Lineage-blocked CV gives an *honest* estimate; it does not give a better
one. Scores usually go **down** relative to random CV -- that is the
point, and a drop is the finding, not a regression. It also cannot
rescue a design where the phenotype is perfectly confounded with lineage
(every resistant isolate in one clone): there is no split of such a
cohort that separates the two signals, and
`permutation_importance_pvalues(..., groups=...)` will say so by
returning p = 1.0 rather than pretending otherwise.

`scipy` is imported lazily inside `lineage_groups()` and `scikit-learn`
inside `LineageKFold`/`permutation_importance_pvalues()`, in the same
optional-dependency pattern `fastdna.embed` uses: importing this module
never requires either package, only calling into it does.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

import fastdna
from . import _core

__all__ = ["lineage_groups", "LineageKFold", "permutation_importance_pvalues"]


def _missing_dependency(caller, package, hint=None):
    return ImportError(
        f"{caller} requires the '{package}' package, which is not installed. "
        f"Install it with `{hint or f'pip install {package}'}` and try again."
    )


def _validate_paths(paths, caller):
    """Refuses the two ways a path list silently corrupts a distance matrix.

    Checked up front rather than after the sketching work is done: too few
    paths for a pairwise structure to exist at all, and a duplicate entry,
    which would map two rows to one index and leave the earlier one all
    zeros.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
    caller : str
        Name of the calling function, used only to name it in raised error
        messages.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If fewer than 2 paths are given, or a path is duplicated.
    """
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise _core.InvalidConfigError(
            f"{caller} needs at least 2 paths to compute pairwise distances, got {len(paths)}"
        )

    seen = set()
    for path in paths:
        if path in seen:
            raise _core.InvalidConfigError(
                f"paths contains a duplicate entry: {path!r}. Each sample may appear "
                "only once; a duplicated path would collapse two rows of the distance "
                "matrix onto one index and silently leave the earlier one all zeros."
            )
        seen.add(path)
    return paths


def _mash_distance_matrix(paths, k, sketch_size):
    """Dense, symmetric `n x n` Mash-distance matrix over `paths`, in
    `paths` order, built from `fastdna.compare_all(..., metric=
    "mash_distance")`'s long-format `(sample_a, sample_b, mash_distance)`
    table.

    `mash_distance` and not `jaccard`: it is already a distance (0 =
    identical), so no similarity-to-distance conversion is involved, and it
    is on an interpretable per-base-divergence scale, which is what makes a
    `distance_threshold` something a caller can reason about rather than
    tune blindly.

    Parameters
    ----------
    paths : list of str
        Sample paths, already validated by `_validate_paths()`.
    k, sketch_size : int
        Forwarded to `fastdna.compare_all()`.

    Returns
    -------
    numpy.ndarray of float64, shape (len(paths), len(paths))
    """
    table = fastdna.compare_all(paths, k=k, sketch_size=sketch_size, metric="mash_distance")

    n = len(paths)
    matrix = np.zeros((n, n), dtype=np.float64)

    # Filled in one vectorized pass rather than row by row: the table has
    # `n*(n-1)/2` rows, each of which cost two Python dict lookups and two
    # element assignments (19,900 iterations for a 200-sample cohort). The
    # values are copied through untouched, so the matrix is identical.
    i, j = fastdna._pair_positions(table, paths)
    values = np.asarray(fastdna._column_as_array(table.column("mash_distance")))
    matrix[i, j] = values
    matrix[j, i] = values

    np.fill_diagonal(matrix, 0.0)
    return matrix


def _relabel_by_first_appearance(labels):
    """Renumbers arbitrary cluster labels to `0, 1, 2, ...` in order of
    first appearance, so `groups[0]` is always `0` and the labelling is
    reproducible regardless of what `fcluster` happened to number things.

    Parameters
    ----------
    labels : array-like of int
        Arbitrary cluster labels, e.g. from `scipy.cluster.hierarchy.fcluster`.

    Returns
    -------
    numpy.ndarray of int64, same shape as `labels`
    """
    mapping = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        out[i] = mapping[label]
    return out


def lineage_groups(
    paths: Iterable[Union[str, os.PathLike]],
    *,
    k: int = 21,
    sketch_size: int = 1000,
    distance_threshold: float = 0.01,
) -> np.ndarray:
    """Assigns each of `paths` an integer lineage label, by single-linkage
    agglomerative clustering of the cohort's all-pairs Mash distances at
    `distance_threshold`.

    These labels are the `groups` a leakage-safe splitter needs: two
    samples sharing a label are close enough to be clonal relatives and
    must therefore never end up on opposite sides of a train/test boundary
    (see the module docstring).

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        FASTQ(.gz) files, at least two, no duplicates. Each is sketched
        exactly once.
    k, sketch_size : int
        Forwarded to the sketching inside `fastdna.compare_all()`. The
        defaults match `fastdna.sketch()`'s own (`k=21`, Mash's default
        length for comparison work).
    distance_threshold : float, default 0.01
        The Mash distance below which two samples are considered the same
        lineage -- roughly, 1% estimated per-base divergence. This is the
        one parameter that matters and it is genuinely dataset-dependent:
        it should sit *above* the within-lineage divergence of the organism
        at hand and *below* the between-lineage divergence. For a clonal
        bacterial pathogen 0.01 is a reasonable starting point; for a fast-
        evolving virus, or a cohort spanning species, inspect
        `fastdna.compare_all(paths, metric="mash_distance")` directly and
        pick a value from the gap in that distribution rather than
        accepting this default.

    Why **single** linkage: it merges two clusters when their *closest*
    members are within the threshold, so a chain of near-identical isolates
    stays one lineage even when its two extremes are further apart than the
    threshold. That is the conservative direction for this purpose -- the
    failure that matters is splitting a clone across folds, and complete or
    average linkage would do exactly that whenever a lineage is
    internally diverse. The cost is chaining: with a threshold set too
    loosely, distinct lineages linked by one intermediate sample merge into
    one group. That direction is safe (fewer, larger groups: a more
    pessimistic, never a leaky, evaluation), and it is why
    `LineageKFold`'s error message points at raising, not lowering, the
    threshold.

    Returns
    -------
    numpy.ndarray of int, one label per path in `paths` order. Labels are
    0-based and contiguous, numbered by first appearance, so `groups[0]`
    is always 0 and repeated calls on the same input give the identical
    array.
    """
    paths = _validate_paths(paths, "lineage_groups()")
    if not distance_threshold > 0:
        raise _core.InvalidConfigError(
            f"distance_threshold must be a positive Mash distance, got {distance_threshold!r}. "
            "At or below 0 every sample becomes its own lineage, which defeats the point."
        )

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise _missing_dependency("lineage_groups()", "scipy") from None

    matrix = _mash_distance_matrix(paths, k, sketch_size)
    # checks=False: the matrix is symmetric with a zero diagonal by
    # construction above, and squareform's own validation is strict about
    # floating-point symmetry in a way that would reject it spuriously.
    condensed = squareform(matrix, checks=False)
    labels = fcluster(linkage(condensed, method="single"), t=distance_threshold, criterion="distance")
    return _relabel_by_first_appearance(labels)


def _n_samples(X):
    """`X`'s sample count, whether it is a list of paths, a NumPy array, or
    a sparse matrix (as `KmerVectorizer.transform()` returns).

    Parameters
    ----------
    X : array-like, sparse matrix, or list

    Returns
    -------
    int
    """
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)


class LineageKFold:
    """A scikit-learn-compatible cross-validation splitter that never places
    two members of the same lineage on opposite sides of a train/test
    boundary.

        from fastdna.cv import LineageKFold
        from sklearn.model_selection import cross_val_score

        cv = LineageKFold(n_splits=5, paths=fastq_paths)
        scores = cross_val_score(pipeline, fastq_paths, y, cv=cv)

    Mechanically it is `sklearn.model_selection.GroupKFold` over groups
    derived from the genomes themselves by `lineage_groups()` -- the point
    is not the folding algorithm (GroupKFold is fine and well tested) but
    that the caller does not have to supply the groups from somewhere else.
    In practice "somewhere else" means a phylogeny nobody built, which is
    why random CV keeps getting used by default.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds. Must be at least 2 and at most the number of
        distinct lineages -- a fold boundary can only fall *between*
        lineages, so there is no way to make more folds than there are
        lineages without breaking one apart.
    paths : iterable of str or pathlib.Path, optional
        FASTQ(.gz) paths from which to derive the lineages. Mutually
        exclusive with `groups`; exactly one of the two is required.
    groups : array-like of int, optional
        Precomputed lineage labels (e.g. from an earlier `lineage_groups()`
        call, an MLST scheme, or a real phylogeny's clades). Use this to
        avoid re-sketching a cohort across several evaluations, or to plug
        in group structure FastDNA did not derive.
    k, sketch_size, distance_threshold
        Forwarded to `lineage_groups()` when `paths` is used; ignored when
        `groups` is supplied.

    Attributes
    ----------
    groups_ : numpy.ndarray
        The lineage labels actually used. When constructed from `paths`
        this is computed on first access (sketching a cohort is real work
        and should not happen inside a constructor) and cached, so a
        splitter reused across several `cross_val_score` calls sketches
        once.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        paths: Optional[Iterable[Union[str, os.PathLike]]] = None,
        groups: Optional[np.ndarray] = None,  # array-like of int, one lineage label per sample
        k: int = 21,
        sketch_size: int = 1000,
        distance_threshold: float = 0.01,
    ) -> None:
        if (paths is None) == (groups is None):
            raise _core.InvalidConfigError(
                "LineageKFold requires exactly one of paths= or groups=: pass paths= to "
                "derive lineages from the genomes with lineage_groups(), or groups= to "
                "supply labels you already have."
            )
        if not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool) or n_splits < 2:
            raise _core.InvalidConfigError(f"n_splits must be an integer >= 2, got {n_splits!r}")

        self.n_splits = int(n_splits)
        self.paths = None if paths is None else [str(p) for p in paths]
        self.k = k
        self.sketch_size = sketch_size
        self.distance_threshold = distance_threshold

        if groups is None:
            self._groups = None
        else:
            # Nothing to compute, so the n_splits/n_groups check can happen
            # now. Failing at construction beats failing halfway through a
            # GridSearchCV that has already spent minutes fitting.
            self._groups = np.asarray(groups)
            self._check_n_splits(self._groups)

    def _check_n_splits(self, groups):
        n_groups = len(np.unique(groups))
        if self.n_splits > n_groups:
            raise _core.InvalidConfigError(
                f"n_splits={self.n_splits} exceeds the number of distinct lineages "
                f"({n_groups}) found in this cohort. A fold boundary can only fall "
                "between lineages, so more folds than lineages is impossible without "
                "splitting a lineage across train and test -- exactly the leakage this "
                f"splitter exists to prevent. Either lower n_splits to at most "
                f"{n_groups}, or raise distance_threshold (currently "
                f"{self.distance_threshold!r}) so that near-clonal samples merge into "
                "fewer, larger lineages. If neither is acceptable, this cohort does not "
                "contain enough independent lineages to support the evaluation you are "
                "asking for, and that is itself the finding."
            )

    @property
    def groups_(self) -> np.ndarray:
        if self._groups is None:
            self._groups = lineage_groups(
                self.paths,
                k=self.k,
                sketch_size=self.sketch_size,
                distance_threshold=self.distance_threshold,
            )
            self._check_n_splits(self._groups)
        return self._groups

    def get_n_splits(
        self,
        X: Any = None,  # array-like, sparse matrix, or list; accepted and ignored (scikit-learn splitter API)
        y: Any = None,  # ignored; accepted for scikit-learn API compatibility
        groups: Any = None,  # ignored; accepted for scikit-learn API compatibility
    ) -> int:
        """The number of folds `split()` will produce -- `self.n_splits`.

        `X`/`y`/`groups` are accepted and ignored, matching scikit-learn's
        splitter signature; `cross_val_score` and `GridSearchCV` call this
        with all three.

        Parameters
        ----------
        X, y, groups : ignored

        Returns
        -------
        int
            `self.n_splits`.
        """
        return self.n_splits

    def split(
        self,
        X: Any,  # array-like, sparse matrix, or list; only its length is used
        y: Any = None,  # ignored; accepted for scikit-learn API compatibility
        groups: Any = None,  # ignored; accepted for scikit-learn API compatibility
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields `(train_indices, test_indices)` pairs, `n_splits` of them.

        Parameters
        ----------
        X : array-like, sparse matrix, or list
            Only its length is used -- it must have one entry per sample, in
            the same order as the `paths`/`groups` this splitter was built
            from. That ordering is the caller's responsibility and cannot be
            checked from here, so it is worth stating explicitly: passing a
            differently-ordered `X` silently misassigns lineages.
        y : ignored
        groups : ignored
            Accepted for scikit-learn API compatibility. Deliberately not
            honoured: this splitter's whole purpose is to use the lineage
            structure it derived itself, and quietly letting a caller-passed
            `groups` override that would turn a leakage-safe evaluation back
            into whatever the caller happened to pass -- including `None`,
            which is how `cross_val_score` calls it.
        """
        lineages = self.groups_
        n = _n_samples(X)
        if n != len(lineages):
            raise _core.InvalidConfigError(
                f"X has {n} samples but this splitter was built from {len(lineages)} "
                "lineage labels. X must have one entry per sample, in the same order as "
                "the paths= or groups= given at construction."
            )

        try:
            from sklearn.model_selection import GroupKFold
        except ImportError:
            raise _missing_dependency("LineageKFold.split()", "scikit-learn") from None

        # GroupKFold only reads X's length; a zero column keeps it from
        # having to validate whatever the caller actually passed (a list of
        # file paths is not an array and would be rejected on dtype).
        placeholder = np.zeros((n, 1))
        yield from GroupKFold(n_splits=self.n_splits).split(placeholder, y, lineages)

    def __repr__(self) -> str:
        source = "paths" if self.paths is not None else "groups"
        return f"LineageKFold(n_splits={self.n_splits}, from={source!r})"


def _model_importances(fitted, n_features, model_repr):
    """One non-negative importance per feature, from whichever attribute the
    fitted estimator exposes.

    `coef_` is taken in absolute value, and a multi-class `coef_` (one row
    per class) is reduced by mean absolute value across the class rows --
    absolute value *first*, because signed per-class coefficients partly
    cancel and averaging them before taking magnitude would report noise as
    importance (the same defect `fastdna.interpret` had to fix in its SHAP
    reduction).

    Parameters
    ----------
    fitted : fitted scikit-learn estimator
        Must expose `coef_` or `feature_importances_`.
    n_features : int
        Expected width of the importances vector, checked against what
        `fitted` actually reports.
    model_repr : str
        `type(model).__name__`, used only to name the model in raised
        error messages.

    Returns
    -------
    numpy.ndarray of float64, shape (n_features,)

    Raises
    ------
    ValueError
        If `fitted` exposes neither `coef_` nor `feature_importances_`, or
        the importances it reports do not have `n_features` entries.
    """
    importances = getattr(fitted, "feature_importances_", None)
    if importances is None:
        coef = getattr(fitted, "coef_", None)
        if coef is None:
            raise _core.InvalidConfigError(
                f"permutation_importance_pvalues() needs a model exposing per-feature "
                f"importances, but {model_repr} has neither `coef_` nor "
                "`feature_importances_` after fitting. Linear models (LogisticRegression, "
                "Lasso, LinearSVC) and tree ensembles (RandomForest, GradientBoosting, "
                "XGBoost) both do. For a model that exposes neither, a model-agnostic "
                "alternative is sklearn.inspection.permutation_importance, which measures "
                "importance by shuffling features rather than reading them off the model."
            )
        coef = np.abs(np.asarray(coef, dtype=np.float64))
        importances = coef.mean(axis=0) if coef.ndim > 1 else coef
    importances = np.asarray(importances, dtype=np.float64).ravel()

    if importances.shape[0] != n_features:
        raise _core.InvalidConfigError(
            f"the fitted model reports {importances.shape[0]} importances but X has "
            f"{n_features} columns"
        )
    return importances


def _permute(y, groups, rng):
    """One draw from the null distribution of the labels.

    Without `groups`, a free permutation of `y`: the null is "the labels are
    unrelated to the features".

    With `groups`, labels are shuffled *within* each lineage and never
    across lineages -- a restricted permutation. This preserves the
    lineage/phenotype association in every null draw, so what the resulting
    p-value tests is whether a feature explains the phenotype *beyond* what
    the population structure already explains. A free permutation would
    destroy that association too, and a feature that merely tags a lineage
    would then look highly significant: the very confounding this module
    exists to control.

    Parameters
    ----------
    y : array-like
        Labels to permute.
    groups : array-like of int or None
        Lineage labels restricting the permutation, or `None` for a free
        permutation.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, same shape as `y`
    """
    permuted = np.array(y, copy=True)
    if groups is None:
        return rng.permutation(permuted)

    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        permuted[members] = permuted[rng.permutation(members)]
    return permuted


def permutation_importance_pvalues(
    model: Any,  # unfitted scikit-learn estimator exposing coef_ or feature_importances_ once fitted
    X: Any,  # array-like or sparse matrix, shape (n_samples, n_features)
    y: np.ndarray,  # array-like, shape (n_samples,)
    feature_names: Iterable[str],
    *,
    n_permutations: int = 100,
    random_state: Optional[int] = None,
    groups: Optional[np.ndarray] = None,  # array-like, lineage labels; see _permute
) -> pa.Table:
    """Attaches an empirical p-value to each feature's importance, by
    refitting `model` on repeatedly permuted labels.

    A feature importance on its own is not evidence. A tree ensemble
    assigns a nonzero importance to every feature it ever split on,
    including pure noise, and with a k-mer matrix there are usually far
    more features than samples -- so the top of an importance ranking is
    populated by chance alone unless something says otherwise. This is what
    says otherwise: how often does a *randomly relabelled* dataset produce
    an importance at least as large for this feature?

    Parameters
    ----------
    model : unfitted scikit-learn estimator
        Cloned before every fit, so the object passed in is never mutated
        and each permutation starts from identical hyperparameters. Must
        expose `coef_` or `feature_importances_` once fitted.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Anything the estimator accepts -- typically
        `KmerVectorizer.transform()`'s CSR output.
    y : array-like, shape (n_samples,)
    feature_names : sequence of str, length n_features
        Typically `KmerVectorizer.get_feature_names_out()`, i.e. literal
        k-mer sequences. Length is checked against `X`'s width: a silent
        mismatch would attach a p-value to the wrong k-mer, which is
        invisible in the output and worse than an error.
    n_permutations : int, default 100
        Number of null refits. The smallest reportable p-value is
        `1 / (n_permutations + 1)`, so 100 permutations cannot resolve
        anything below ~0.0099; raise it if you intend to apply a
        multiple-testing correction across many features.
    random_state : int or None
        Seeds the permutation draws (`numpy.random.default_rng`), making
        the whole result reproducible.
    groups : array-like, optional
        Lineage labels (e.g. from `lineage_groups()`). Restricts each
        permutation to shuffle labels *within* a lineage, so the p-value
        asks whether a feature explains the phenotype beyond population
        structure -- see `_permute`. Strongly recommended for any clonal
        cohort; without it a feature that merely marks a lineage will score
        as highly significant.

        A design in which each lineage carries a single label is perfectly
        confounded: no within-lineage permutation can change `y`, every
        null refit reproduces the observed importances, and every p-value
        comes back at 1.0. That is the correct answer for such a cohort,
        not a bug -- the data cannot distinguish phenotype from lineage.

    Cost
    ----
    Exactly `n_permutations + 1` model fits, run serially. At the default
    that is 101 fits: fine for a linear model over a few thousand k-mers,
    slow for a large gradient-boosting model. Start with
    `n_permutations=20` to see the shape of the result, then raise it for
    the numbers you intend to report.

    Returns
    -------
    pyarrow.Table with columns `feature`, `importance`, `p_value`, one row
    per feature **in `feature_names` order** (not sorted), so the table
    lines up positionally with the model's own coefficient vector. Sort at
    the call site when you want a ranking:
    `table.sort_by([("p_value", "ascending")])`.

    p-values use the add-one estimator `(1 + #{perm >= observed}) /
    (1 + n_permutations)` (Phipson & Smyth, 2010), which never reports an
    impossible p = 0 -- with a finite number of permutations, "no draw beat
    it" is not evidence that no draw ever could.
    """
    if not isinstance(n_permutations, (int, np.integer)) or isinstance(n_permutations, bool) or n_permutations < 1:
        raise _core.InvalidConfigError(f"n_permutations must be a positive integer, got {n_permutations!r}")

    try:
        from sklearn.base import clone
    except ImportError:
        raise _missing_dependency("permutation_importance_pvalues()", "scikit-learn") from None

    feature_names = list(feature_names)
    n_samples = _n_samples(X)
    n_features = int(getattr(X, "shape", (n_samples, len(feature_names)))[1])

    if len(feature_names) != n_features:
        raise _core.InvalidConfigError(
            f"feature_names has {len(feature_names)} entries but X has {n_features} "
            "columns. A silent mismatch would attach a p-value to the WRONG feature, so "
            "this is refused rather than truncated. feature_names should be exactly "
            "`vectorizer.get_feature_names_out()` for the vectorizer that produced X."
        )

    y = np.asarray(y)
    if len(y) != n_samples:
        raise _core.InvalidConfigError(f"y has {len(y)} entries but X has {n_samples} samples")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n_samples:
            raise _core.InvalidConfigError(f"groups has {len(groups)} entries but X has {n_samples} samples")

    model_repr = type(model).__name__
    observed = _model_importances(clone(model).fit(X, y), n_features, model_repr)

    rng = np.random.default_rng(random_state)
    at_least_as_extreme = np.zeros(n_features, dtype=np.int64)
    for _ in range(n_permutations):
        permuted_importances = _model_importances(
            clone(model).fit(X, _permute(y, groups, rng)), n_features, model_repr
        )
        at_least_as_extreme += permuted_importances >= observed

    p_values = (1 + at_least_as_extreme) / (1 + n_permutations)

    return pa.table(
        {
            "feature": feature_names,
            "importance": observed.tolist(),
            "p_value": p_values.tolist(),
        }
    )
