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

import io
import os
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

import fastdna
from . import _core
from .cohort_counts import CohortCounts

__all__ = [
    "default_threshold_curve",
    "lineage_groups",
    "lineage_groups_at_thresholds",
    "lineage_groups_from_distances",
    "lineage_groups_from_tree",
    "LineageKFold",
    "permutation_importance_pvalues",
]


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


def _validate_cohort_counts(counts, caller):
    """The `CohortCounts` analogue of `_validate_paths()`'s length check --
    refuses a cohort too small for a pairwise structure to exist.

    No duplicate-entry check is needed here, unlike `_validate_paths()`:
    `CohortCounts.sample_ids` cannot contain a duplicate id in the first
    place -- `count_cohort()` raises before ever building the object if two
    input paths would derive the same `sample_id` (see its own docstring),
    and `CohortCounts.subset()` can only select from `sample_ids` already
    known to be unique, so there is no way to construct a `CohortCounts`
    with a repeated id to guard against here.

    Parameters
    ----------
    counts : fastdna.CohortCounts
    caller : str
        Name of the calling function, used only to name it in raised error
        messages.

    Returns
    -------
    fastdna.CohortCounts
        `counts`, unchanged -- returned for symmetry with `_validate_paths()`,
        which does transform its input.

    Raises
    ------
    ValueError
        If `counts` has fewer than 2 samples.
    """
    if len(counts) < 2:
        raise _core.InvalidConfigError(
            f"{caller} needs at least 2 samples to compute pairwise distances, got "
            f"{len(counts)} in this CohortCounts"
        )
    return counts


def _validate_paths_or_counts(samples, caller):
    """Validates `samples` for pairwise Mash-distance computation, accepting
    either FASTQ(.gz) paths (`_validate_paths()`) or an already-counted
    `fastdna.CohortCounts` (`_validate_cohort_counts()`) -- the two inputs
    `_mash_distance_matrix()` knows how to sketch from. Dispatches purely on
    `isinstance(samples, CohortCounts)`, so every existing call passing a
    plain iterable of paths is routed to `_validate_paths()` exactly as
    before.

    Parameters
    ----------
    samples : iterable of str/pathlib.Path, or fastdna.CohortCounts
    caller : str
        Name of the calling function, used only to name it in raised error
        messages.

    Returns
    -------
    list of str, or fastdna.CohortCounts
        Matches whichever of `_validate_paths()`/`_validate_cohort_counts()`
        actually ran.
    """
    if isinstance(samples, CohortCounts):
        return _validate_cohort_counts(samples, caller)
    return _validate_paths(samples, caller)


def _cohort_counts_sketches(counts, sketch_size):
    """One `fastdna.Sketch` per sample in `counts`, built directly from its
    already-counted k-mers via `fastdna.sketch_from_kmers()` -- no FASTQ
    file is read. The in-memory counterpart to sketching every one of
    `paths` by streaming it, used by `_mash_distance_matrix()` when it is
    given a `CohortCounts` instead of paths.

    Every sketch is built at `counts.k`, never at a separately-passed `k`
    argument -- see `_mash_distance_matrix()`'s own docstring for why that
    is the deliberate, documented choice here, not an oversight.

    Slicing is zero-copy (`pyarrow.Array.slice`) and each sample's whole row
    range is handed to `fastdna.sketch_from_kmers()` in one call, so this is
    one pass over `counts.kmers` in total, not one `CohortCounts.subset()`
    call per sample (which would rebuild a `sample_id -> position` lookup
    and recompute `offsets` on every call -- O(n) per sample, O(n^2) overall
    for an n-sample cohort).

    Parameters
    ----------
    counts : fastdna.CohortCounts
        Already validated by `_validate_cohort_counts()` (at least 2
        samples).
    sketch_size : int

    Returns
    -------
    list of fastdna.Sketch, one per `counts.sample_ids`, in that order.
    """
    offsets = counts.offsets
    sketches = []
    for i in range(len(counts)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        kmer_slice = counts.kmers.slice(start, end - start)
        sketches.append(fastdna.sketch_from_kmers(kmer_slice, k=counts.k, sketch_size=sketch_size))
    return sketches


def _mash_distance_matrix(samples, k, sketch_size):
    """Dense, symmetric `n x n` Mash-distance matrix over `samples`, in
    `samples` order.

    `samples` is either a plain iterable of FASTQ(.gz) paths (already
    validated by `_validate_paths()`) or an already-counted
    `fastdna.CohortCounts` (already validated by `_validate_cohort_counts()`
    -- see `_validate_paths_or_counts()`, which every caller of this
    function runs first). The two inputs take genuinely different code
    paths, not just a different source for the same sketching call:

    - **Paths**: built from `fastdna.compare_all(..., metric=
      "mash_distance")`'s long-format `(sample_a, sample_b, mash_distance)`
      table -- every file is opened and streamed to build its sketch, the
      same cost `fastdna.sketch()` has on its own.
    - **CohortCounts**: built from `_cohort_counts_sketches()`, which
      sketches every sample directly from its already-counted `kmer_u64`
      column (`fastdna.sketch_from_kmers()`) and this function then compares
      pairwise itself (`Sketch.mash_distance()`, one call per pair) -- no
      FASTQ file is read at all. This is the path that makes
      `fastdna.audit()`'s automatic leakage curve free of any additional
      file I/O when given a `CohortCounts` cohort: the same one-time count
      `KmerVectorizer(counts=...)` already reuses for the model also
      supplies the lineage/distance side.

    `mash_distance` and not `jaccard` in both cases: it is already a
    distance (0 = identical), so no similarity-to-distance conversion is
    involved, and it is on an interpretable per-base-divergence scale, which
    is what makes a `distance_threshold` something a caller can reason about
    rather than tune blindly.

    **A real, small behavioural difference between the two inputs.** If
    `samples` is a `CohortCounts` built with `min_count > 1`, singleton
    k-mers (those appearing once in a sample) were already dropped by
    `count_cohort()` before this function -- or `sketch_from_kmers()` --
    ever sees them, because `min_count` filters at counting time, not at
    sketching time. The paths input has no such filter: `fastdna.
    compare_all()` sketches every raw k-mer occurrence a file streams,
    unfiltered. This is closer to Mash's own `-m`/error-filtering
    convention (dropping likely-sequencing-error k-mers before they can
    pollute a sketch) than the paths-based path is, and it is not a defect
    to reconcile -- the two inputs are allowed to differ here, deliberately
    documented rather than silently matched, because "match them" would
    mean quietly re-filtering the paths-based path (a behaviour change with
    its own cost) to chase an input that most `CohortCounts` callers use
    with the library's own `min_count=1` default anyway.

    **Which `k` is used for the CohortCounts path, and why `k` can be
    silently ignored there.** When `samples` is a `CohortCounts`, every
    sketch is built at `samples.k` -- the k-mer size the cohort was actually
    counted at -- and this function's own `k` argument is not consulted at
    all for that path. This is a deliberate choice, not an omission: `k` is
    passed down through several layers (`lineage_groups_at_thresholds()`,
    `default_threshold_curve()`, `fastdna.audit()`) each with their own
    `k=21` default, and a plain Python default argument cannot distinguish
    "the caller explicitly asked for k=21" from "the caller never mentioned
    k and this is just the default" -- so there is no reliable signal here
    to compare against `samples.k` and decide whether a mismatch is real or
    just an unrelated default surfacing. Silently deferring to `samples.k`
    avoids that ambiguity entirely and is also almost always what a caller
    wants: `count_cohort()`'s own default is `k=31`, so the ordinary case of
    `audit(pipeline, counts, phenotype)` with every default left in place
    would otherwise "mismatch" on nearly every call, which would make an
    error (or even a warning) noise rather than signal. The alternative --
    raising `InvalidConfigError` on a mismatch -- was considered and
    rejected for exactly that reason: it cannot tell a genuine caller
    mistake from the completely ordinary case of two unrelated defaults
    (`k=21` here, `k=31` in `count_cohort()`) simply differing. A caller who
    needs a different k for sketching than the cohort was counted at should
    build a second `CohortCounts` at that k (`count_cohort(..., k=...)`) --
    there is no way to sketch at a k the k-mers were never extracted at
    without recounting anyway.

    Parameters
    ----------
    samples : list of str, or fastdna.CohortCounts
        Already validated by `_validate_paths_or_counts()`.
    k, sketch_size : int
        `sketch_size` is always honoured. `k` is honoured only for the
        paths input (forwarded to `fastdna.compare_all()`); ignored for a
        `CohortCounts` input, per the docstring section above.

    Returns
    -------
    numpy.ndarray of float64, shape (len(samples), len(samples))
    """
    if isinstance(samples, CohortCounts):
        sketches = _cohort_counts_sketches(samples, sketch_size)
        n = len(sketches)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = sketches[i].mash_distance(sketches[j])
                matrix[i, j] = d
                matrix[j, i] = d
        return matrix

    paths = samples
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


def _validate_distance_threshold(distance_threshold, caller):
    """The one check `lineage_groups()` and `lineage_groups_at_thresholds()`
    both apply to every threshold they are given: it must be a positive
    Mash distance. At or below 0 every sample becomes its own lineage,
    which defeats the point of clustering at all.

    Parameters
    ----------
    distance_threshold : float
    caller : str
        Name of the calling function, used only to name it in raised error
        messages.

    Raises
    ------
    ValueError
        If `distance_threshold` is not strictly positive.
    """
    if not distance_threshold > 0:
        raise _core.InvalidConfigError(
            f"distance_threshold must be a positive Mash distance, got {distance_threshold!r}. "
            "At or below 0 every sample becomes its own lineage, which defeats the point."
        )


def lineage_groups_at_thresholds(
    paths: Union[Iterable[Union[str, os.PathLike]], "CohortCounts"],
    thresholds: Sequence[float],
    *,
    k: int = 21,
    sketch_size: int = 1000,
) -> List[np.ndarray]:
    """`lineage_groups()`, evaluated at every threshold in `thresholds`, from
    a single all-pairs Mash distance matrix and a single dendrogram.

    See `lineage_groups()`'s own docstring for what a lineage label means,
    why single linkage is the right merge rule for this purpose, and what
    `distance_threshold` (here, each entry of `thresholds`) trades off --
    none of that is repeated here, because it does not change: this
    function computes exactly the same clustering `lineage_groups()` does,
    for each threshold given, and nothing about the *meaning* of a cut
    differs between the two.

    What is different, and the entire reason this function exists as its
    own entry point rather than as a documented idiom ("just call
    `lineage_groups()` in a loop"), is the cost model. `lineage_groups()`
    does two things: (1) sketch every sample and compute the `n x n`
    all-pairs Mash distance matrix -- the expensive part, dominated by
    `fastdna.compare_all()`'s `O(n^2)` sketch comparisons -- and (2) build a
    single-linkage dendrogram from that matrix and cut it once at
    `distance_threshold` -- both cheap, since `scipy.cluster.hierarchy.
    fcluster` cutting an already-built dendrogram is linear in the number of
    samples. Calling `lineage_groups()` once per threshold redoes step (1)
    every time even though it does not depend on the threshold at all, so
    sweeping `m` thresholds over a cohort costs `m` times the sketching and
    distance work for zero additional information. This function does step
    (1) exactly once, then reuses the same matrix and the same dendrogram
    for every cut in step (2) -- so sweeping `m` thresholds costs one
    sketching/distance pass plus `m` cheap cuts, not `m` of each. That
    difference is what makes it practical to report a leakage-vs-threshold
    curve (`fastdna.audit`'s `lineage_threshold_curve=`) instead of a single
    number at one threshold nobody has strong a priori grounds to pick.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path, or fastdna.CohortCounts
        Ordinarily FASTQ(.gz) files, at least two, no duplicates -- each is
        sketched exactly once, regardless of how many thresholds are given.
        May also be an already-counted `fastdna.CohortCounts` (e.g. from
        `fastdna.count_cohort()`), in which case every sample is sketched
        directly from its already-counted k-mers (`fastdna.
        sketch_from_kmers()`) instead of its FASTQ file being reopened and
        restreamed -- see `_mash_distance_matrix()`'s own docstring for
        exactly what changes between the two inputs, including the note on
        which `k` is actually used for sketching in each case (the `k`
        argument below is ignored for a `CohortCounts` input; `counts.k` is
        used instead) and the small, deliberate behavioural difference
        `min_count > 1` introduces on that path.
    thresholds : sequence of float
        The Mash-distance cut points to evaluate, each validated exactly as
        `lineage_groups()`'s own `distance_threshold` is (must be strictly
        positive). At least one threshold is required. Thresholds may
        repeat; a repeated value is cut (cheaply) more than once rather than
        deduplicated, since deduplicating here would desynchronize the
        output from `thresholds`' own order and length.
    k, sketch_size : int
        Forwarded to the sketching inside `fastdna.compare_all()` when
        `paths` is a plain iterable of paths, exactly as in
        `lineage_groups()`. Ignored when `paths` is a `CohortCounts` (see
        above).

    Returns
    -------
    list of numpy.ndarray of int
        One `groups` array per entry of `thresholds`, in the same order,
        each exactly what `lineage_groups(paths, k=k, sketch_size=
        sketch_size, distance_threshold=thresholds[i])` would return on its
        own -- 0-based, contiguous, first-appearance-ordered labels (see
        `lineage_groups()`'s `Returns` section).

    Raises
    ------
    ValueError
        Via `_validate_paths()`/`_validate_cohort_counts()`: fewer than 2
        paths/samples, or (paths only) a duplicate path. Via the
        per-threshold check above: `thresholds` is empty, or any entry is
        not strictly positive.
    """
    paths = _validate_paths_or_counts(paths, "lineage_groups_at_thresholds()")
    thresholds = list(thresholds)
    if len(thresholds) == 0:
        raise _core.InvalidConfigError(
            "lineage_groups_at_thresholds() needs at least 1 threshold, got 0"
        )
    for threshold in thresholds:
        _validate_distance_threshold(threshold, "lineage_groups_at_thresholds()")

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise _missing_dependency("lineage_groups_at_thresholds()", "scipy") from None

    matrix = _mash_distance_matrix(paths, k, sketch_size)
    # checks=False: see lineage_groups()'s identical comment -- the matrix
    # is symmetric with a zero diagonal by construction, and squareform's
    # own validation is strict about floating-point symmetry in a way that
    # would reject it spuriously.
    condensed = squareform(matrix, checks=False)
    dendrogram = linkage(condensed, method="single")

    return [
        _relabel_by_first_appearance(
            fcluster(dendrogram, t=threshold, criterion="distance")
        )
        for threshold in thresholds
    ]


def default_threshold_curve(
    paths: Union[Iterable[Union[str, os.PathLike]], "CohortCounts"],
    *,
    n_points: int = 5,
    k: int = 21,
    sketch_size: int = 1000,
) -> List[float]:
    """A data-driven set of Mash-distance thresholds spanning this cohort's
    own range of clustering granularity, for `fastdna.audit()` to sweep by
    default when a caller does not supply `lineage_threshold_curve=`
    explicitly.

    Why data-driven and not a fixed list of thresholds (e.g. `[0.001, 0.01,
    0.1]`): Mash distance has no universal scale a threshold can be picked
    against in the abstract -- what counts as "fine" or "coarse" clustering
    depends entirely on how divergent this particular cohort's own samples
    are from each other (see `lineage_groups()`'s own docstring on why
    `distance_threshold` is a real, cohort-specific tuning parameter, not a
    constant). A fixed threshold list picked without seeing the cohort can
    land entirely on one side of its actual range -- either cutting nothing
    (finer than every real divergence in the cohort, everyone their own
    lineage) or cutting everything into one lineage -- which is exactly the
    single-threshold fragility this function exists to route around.

    Instead, this builds the same single-linkage dendrogram `lineage_groups
    _at_thresholds()` does, reads off its merge heights (the Mash distance
    at which each successive pair of clusters merged -- a direct summary of
    this cohort's own divergence structure), and returns the thresholds at
    `n_points` evenly spaced quantiles (10th to 90th percentile) of that
    distribution. That spread is deliberately inside the two degenerate
    extremes (merging nothing, merging everything) rather than pinned to
    them: `fastdna.audit()`'s own per-point handling already reports (via
    `DegenerateLineagesWarning` and a `nan`-scored `LeakageCurvePoint`, see
    its own docstring) if the swept range still reaches one of those
    extremes for a given cohort, which is itself useful information, not
    something to engineer around by construction.

    This builds its own Mash distance matrix and dendrogram independently
    of any other call in the same `audit()` invocation (e.g. the primary
    threshold's own `lineage_groups_at_thresholds()` call) -- an extra
    `O(n^2)` sketching pass. That cost is real but is consistently small
    next to what it is paired with: `audit()`'s dominant cost is the
    `cross_val_score` fit/predict work per curve point (minutes, on a
    realistic cohort and estimator), not the sketching (seconds). Sharing
    one dendrogram across both would need this function and the primary
    threshold's own cut to be fused into one call, which is a real
    optimization but not one the cost model above makes worth the added
    coupling between them today.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path, or fastdna.CohortCounts
        Ordinarily FASTQ(.gz) files, at least two, no duplicates. May also
        be an already-counted `fastdna.CohortCounts`, sketched directly from
        its counted k-mers with no FASTQ file reread -- see
        `_mash_distance_matrix()`'s own docstring for exactly what changes
        (including which `k` is actually used: `counts.k`, not the `k`
        argument below, for this input).
    n_points : int, default 5
        How many thresholds to return. Reduced automatically (with no
        error) if the cohort's dendrogram does not have `n_points` distinct
        merge heights to draw quantiles from -- a small cohort simply has
        less granularity to show a curve across.
    k, sketch_size : int
        Forwarded to the sketching inside `fastdna.compare_all()` when
        `paths` is a plain iterable of paths, exactly as in
        `lineage_groups()`. Ignored when `paths` is a `CohortCounts` (see
        above).

    Returns
    -------
    list of float
        `n_points` (or fewer, see above) strictly positive, strictly
        increasing Mash-distance thresholds, suitable to pass directly as
        `fastdna.audit()`'s `lineage_threshold_curve=`.

    Raises
    ------
    ValueError
        Via `_validate_paths()`/`_validate_cohort_counts()`: fewer than 2
        paths/samples, or (paths only) a duplicate path.
    ValueError
        If `n_points` is not a positive integer.
    """
    paths = _validate_paths_or_counts(paths, "default_threshold_curve()")
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool) or n_points < 1:
        raise _core.InvalidConfigError(f"n_points must be a positive integer, got {n_points!r}")

    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise _missing_dependency("default_threshold_curve()", "scipy") from None

    matrix = _mash_distance_matrix(paths, k, sketch_size)
    condensed = squareform(matrix, checks=False)
    dendrogram = linkage(condensed, method="single")

    # linkage()'s 3rd column is each merge's height (the Mash distance at
    # which it happened), monotonically non-decreasing by construction for
    # single linkage. Only strictly positive heights are valid thresholds
    # (see _validate_distance_threshold); a height of exactly 0 (duplicate
    # or near-duplicate samples) is dropped rather than clipped upward,
    # since silently moving a caller's effective threshold would be worse
    # than just having fewer points to draw from.
    merge_heights = np.sort(dendrogram[:, 2])
    merge_heights = merge_heights[merge_heights > 0]

    if merge_heights.size == 0:
        # Every merge happened at distance 0 (e.g. every sample identical
        # under this k/sketch_size) -- there is no positive threshold this
        # cohort's own dendrogram can offer at all.
        return []

    quantile_ranks = np.linspace(0.10, 0.90, num=min(n_points, merge_heights.size))
    thresholds = np.quantile(merge_heights, quantile_ranks)
    # Deduplicated rather than left as repeats: a small cohort's merge
    # heights can collide at the sampled quantile ranks even after the
    # min(n_points, ...) cap above (e.g. two ranks landing either side of
    # the same repeated height), and a repeated threshold contributes no
    # additional information to the curve.
    thresholds = np.unique(thresholds)
    return [float(t) for t in thresholds]


def lineage_groups(
    paths: Union[Iterable[Union[str, os.PathLike]], "CohortCounts"],
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
    paths : iterable of str or pathlib.Path, or fastdna.CohortCounts
        Ordinarily FASTQ(.gz) files, at least two, no duplicates -- each is
        sketched exactly once. May also be an already-counted
        `fastdna.CohortCounts`, sketched directly from its counted k-mers
        with no FASTQ file reread -- see `lineage_groups_at_thresholds()`
        (which this delegates to) and `_mash_distance_matrix()`'s own
        docstring for exactly what changes, including which `k` is actually
        used for that input (`counts.k`, not the `k` argument below).
    k, sketch_size : int
        Forwarded to the sketching inside `fastdna.compare_all()` when
        `paths` is a plain iterable of paths. The defaults match
        `fastdna.sketch()`'s own (`k=21`, Mash's default length for
        comparison work). Ignored when `paths` is a `CohortCounts`.
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

    Implemented as the single-threshold case of `lineage_groups_at_
    thresholds()` -- see that function's docstring if you need this same
    clustering at more than one threshold, since calling this function in a
    loop redoes the expensive all-pairs sketching/distance pass once per
    call for no reason.
    """
    return lineage_groups_at_thresholds(
        paths, [distance_threshold], k=k, sketch_size=sketch_size
    )[0]


def _validate_distance_threshold(distance_threshold, caller):
    """Shared positivity check for `distance_threshold`, used by both
    `lineage_groups_from_distances()` and `lineage_groups_from_tree()`.
    `lineage_groups()` keeps its own copy of this same check inline rather
    than calling this helper, so as not to touch that function's body.

    Parameters
    ----------
    distance_threshold : float
    caller : str
        Name of the calling function, used only to name it in the raised
        error message.

    Raises
    ------
    ValueError
        If `distance_threshold` is not a positive number.
    """
    if not distance_threshold > 0:
        raise _core.InvalidConfigError(
            f"{caller} needs a positive distance_threshold, got {distance_threshold!r}. "
            "At or below 0 every sample becomes its own lineage, which defeats the point."
        )


def _validate_distance_matrix(distances, caller):
    """Refuses the three ways an already-computed distance matrix fails to
    be a valid input to single-linkage clustering: not square, a diagonal
    that is not (near) zero, and asymmetry.

    Symmetry and the diagonal are checked with a floating-point tolerance
    rather than exact equality, for the same reason `lineage_groups()`
    passes `checks=False` to `scipy.spatial.distance.squareform()`:
    squareform's own built-in validation is strict enough to reject a
    matrix that is symmetric by construction but carries ordinary
    floating-point rounding noise from whatever produced it (a caller's
    own pairwise-distance computation, not FastDNA's). This function's
    tolerance (`1e-6`) is deliberately looser than that, and its own
    `squareform(..., checks=False)` call in `_single_linkage_groups()`
    relies on this check having already run.

    Parameters
    ----------
    distances : array-like
    caller : str
        Name of the calling function, used only to name it in raised error
        messages.

    Returns
    -------
    numpy.ndarray of float64, shape (n, n)

    Raises
    ------
    ValueError
        If `distances` is not square, has fewer than 2 samples, has a
        diagonal that is not (near) zero, or is not (near) symmetric.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise _core.InvalidConfigError(f"{caller} needs a square distance matrix, got shape {distances.shape}")

    n = distances.shape[0]
    if n < 2:
        raise _core.InvalidConfigError(
            f"{caller} needs at least 2 samples to compute pairwise distances, got a "
            f"{n}x{n} matrix"
        )

    max_diagonal = float(np.max(np.abs(np.diagonal(distances))))
    if max_diagonal > 1e-6:
        raise _core.InvalidConfigError(
            f"{caller} needs a zero-diagonal distance matrix (a sample's distance to "
            f"itself must be 0), but the diagonal has values up to {max_diagonal!r}"
        )

    max_asymmetry = float(np.max(np.abs(distances - distances.T)))
    if max_asymmetry > 1e-6:
        raise _core.InvalidConfigError(
            f"{caller} needs a symmetric distance matrix (distances[i, j] must equal "
            f"distances[j, i]), but the maximum asymmetry found is {max_asymmetry!r}"
        )

    return distances


def _single_linkage_groups(distances, distance_threshold, caller):
    """The clustering core `lineage_groups_from_distances()` runs, matching
    `lineage_groups()`'s own `scipy.cluster.hierarchy.linkage(squareform(...),
    method="single")` + `fcluster(..., criterion="distance")` +
    `_relabel_by_first_appearance()` sequence exactly, so that "the same
    distance_threshold" means the same thing regardless of whether the
    distance matrix came from FastDNA's own Mash sketching or from
    somewhere else entirely.

    Parameters
    ----------
    distances : numpy.ndarray of float64, shape (n, n)
        Already validated by `_validate_distance_matrix()`.
    distance_threshold : float
    caller : str
        Name of the calling function, used only to name it in a raised
        ImportError.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise _missing_dependency(caller, "scipy") from None

    condensed = squareform(distances, checks=False)
    labels = fcluster(linkage(condensed, method="single"), t=distance_threshold, criterion="distance")
    return _relabel_by_first_appearance(labels)


def lineage_groups_from_distances(
    distances: np.ndarray,
    *,
    distance_threshold: float,
) -> np.ndarray:
    """Assigns each row of an already-computed distance matrix an integer
    lineage label, by the same single-linkage clustering `lineage_groups()`
    runs over its own Mash-distance matrix.

    Use this when a phylogenetic proxy for relatedness already exists from
    somewhere other than FastDNA's own sketching: a Roary/Panaroo gene-
    presence-absence distance, a core-genome SNP distance, DBGWAS unitig
    distances, or any other square distance matrix over the same samples
    `groups=` will be paired with. `lineage_groups_from_tree()` is the
    equivalent entry point starting from a Newick tree instead of an
    already-computed matrix.

    See `lineage_groups()`'s own docstring for what `distance_threshold`
    means and why single linkage is the conservative choice for this
    purpose (chaining, not splitting, clonal groups) -- that reasoning
    applies identically here and is not repeated.

    Parameters
    ----------
    distances : numpy.ndarray, shape (n, n)
        A square, symmetric, zero-diagonal distance matrix over `n`
        samples, any distance metric. Symmetry and the diagonal are
        checked with a floating-point tolerance (not exact equality),
        because a matrix computed elsewhere routinely carries ordinary
        rounding noise rather than being symmetric by literal
        construction.
    distance_threshold : float
        The distance below which two samples are considered the same
        lineage, on the same scale as `distances`. Must be positive.
        There is no dataset-independent default the way
        `lineage_groups()`'s Mash-distance `0.01` is: `distances` can be on
        any scale (a Jaccard distance, a SNP count, a normalized unitig
        distance), so inspect its distribution and pick a value from the
        gap between within-lineage and between-lineage distances, the same
        way `lineage_groups()`'s own docstring recommends for Mash
        distances.

    Returns
    -------
    numpy.ndarray of int, shape (n,)
        Integer lineage labels, 0-based and contiguous, numbered by first
        appearance -- the identical convention `lineage_groups()` uses.

    Raises
    ------
    ValueError
        If `distances` is not square, is not (near) symmetric, does not
        have a (near) zero diagonal, or `distance_threshold` is not
        positive.
    ImportError
        If scipy is not installed.

    Example
    -------
    A caller with a Roary `gene_presence_absence.csv` already has a
    sample-by-gene presence/absence matrix. Turning it into a distance
    matrix (e.g. `scipy.spatial.distance.squareform(scipy.spatial.distance.
    pdist(genes, metric="jaccard"))`) and calling
    `lineage_groups_from_distances(matrix, distance_threshold=0.1)`
    produces labels in the same shape `lineage_groups()` would, ready for
    `LineageKFold(groups=labels)` or `audit(groups=labels, paths=X)`.
    """
    _validate_distance_threshold(distance_threshold, "lineage_groups_from_distances()")
    distances = _validate_distance_matrix(distances, "lineage_groups_from_distances()")
    return _single_linkage_groups(distances, distance_threshold, "lineage_groups_from_distances()")


def lineage_groups_from_tree(
    tree: Union[str, os.PathLike],
    leaf_names: Sequence[str],
    *,
    distance_threshold: float,
) -> np.ndarray:
    """Assigns each of `leaf_names` an integer lineage label, derived from
    the all-pairs patristic distances (the sum of branch lengths along the
    path connecting two leaves) of a Newick-format phylogenetic tree.

    This is the entry point for a caller who already has a real tree --
    typically from IQ-TREE, RAxML, or a Gubbins recombination-corrected
    tree -- rather than FastDNA's own Mash-sketch proxy for one. It
    computes the patristic distance matrix over `leaf_names` and hands it
    to `lineage_groups_from_distances()`, so the two functions agree
    exactly on what "the same distance_threshold" means; see that
    function's docstring, and `lineage_groups()`'s, for the shared single-
    linkage reasoning this builds on.

    Parameters
    ----------
    tree : str or os.PathLike
        Either a path to a Newick file, or a Newick string directly (e.g.
        `"(A:0.1,(B:0.2,C:0.3):0.1);"`). A `str` is treated as a path when
        it names an existing file (checked with `os.path.isfile`);
        otherwise it is parsed as Newick text directly. An `os.PathLike`
        (e.g. `pathlib.Path`) is always treated as a path.
    leaf_names : sequence of str
        The sample_ids/paths the output labels correspond to, in the
        desired output order. Every entry must be a leaf label that
        actually exists in `tree` -- see Raises below.
    distance_threshold : float
        The patristic distance below which two leaves are considered the
        same lineage, on the tree's own branch-length scale (typically
        substitutions per site for a maximum-likelihood tree). This is
        *not* comparable to `lineage_groups()`'s Mash-distance default of
        `0.01`: inspect the tree's own branch lengths before picking a
        value, the same way `lineage_groups()`'s docstring recommends
        inspecting the Mash-distance distribution.

    Returns
    -------
    numpy.ndarray of int, shape (len(leaf_names),)
        Integer lineage labels, 0-based and contiguous, numbered by first
        appearance, in `leaf_names` order.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    ValueError
        If `tree` names a path that does not exist, if any of `leaf_names`
        is not a leaf label found in `tree` (the message names which are
        missing and shows a few of the tree's own leaf labels for
        comparison -- tree-building tools commonly spell sample names
        differently from a caller's own sample_ids, e.g. underscores vs
        spaces or an appended `_1`/accession suffix), or if
        `distance_threshold` is not positive.

    Example
    -------
    Given an IQ-TREE run's `sample.treefile`::

        groups = lineage_groups_from_tree(
            "sample.treefile", sample_ids, distance_threshold=0.02,
        )
        cv = LineageKFold(n_splits=5, groups=groups)

    or directly from a Newick string, without writing a file::

        newick = "(A:0.1,(B:0.05,C:0.05):0.2);"
        groups = lineage_groups_from_tree(newick, ["A", "B", "C"], distance_threshold=0.15)
        # groups == array([0, 1, 1]) -- B and C share a recent common
        # ancestor (patristic distance 0.05 + 0.05 = 0.10, under the
        # threshold); A is 0.1 + 0.2 = 0.3 from both, over it.
    """
    _validate_distance_threshold(distance_threshold, "lineage_groups_from_tree()")
    leaf_names = list(leaf_names)
    if len(leaf_names) < 2:
        raise _core.InvalidConfigError(
            f"lineage_groups_from_tree() needs at least 2 leaf_names to compute pairwise "
            f"distances, got {len(leaf_names)}"
        )

    try:
        from Bio import Phylo
    except ImportError:
        raise _missing_dependency("lineage_groups_from_tree()", "biopython") from None

    if isinstance(tree, os.PathLike):
        source = os.fspath(tree)
    elif isinstance(tree, str) and os.path.isfile(tree):
        source = tree
    else:
        source = io.StringIO(str(tree))

    try:
        parsed = Phylo.read(source, "newick")
    except FileNotFoundError as e:
        raise _core.IoNotFoundError(
            f"lineage_groups_from_tree() could not find the Newick tree file {source!r}."
        ) from e

    leaf_map = {clade.name: clade for clade in parsed.get_terminals() if clade.name is not None}
    missing = [name for name in leaf_names if name not in leaf_map]
    if missing:
        example_tree_leaves = sorted(leaf_map)[:5]
        raise _core.InvalidConfigError(
            f"lineage_groups_from_tree() could not find {len(missing)} of the requested "
            f"leaf_names as leaf labels in the tree: {missing[:5]}"
            f"{', ...' if len(missing) > 5 else ''}. A few leaf labels that ARE in the "
            f"tree, for comparison: {example_tree_leaves}. Tree-building tools commonly "
            "spell sample names differently from a caller's own sample_ids (underscores "
            "vs spaces, an appended '_1' or accession suffix) -- check for a naming "
            "mismatch before assuming the sample is truly absent from the tree."
        )

    n = len(leaf_names)
    distances = np.zeros((n, n), dtype=np.float64)
    clades = [leaf_map[name] for name in leaf_names]
    for i in range(n):
        for j in range(i + 1, n):
            d = parsed.distance(clades[i], clades[j])
            distances[i, j] = d
            distances[j, i] = d

    return lineage_groups_from_distances(distances, distance_threshold=distance_threshold)


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
