"""`embed_cohort` -- cohort visualization of pairwise genomic distances.

Turns `fastdna.compare_all()`'s long-format pairwise table into low-
dimensional coordinates (2D/3D) suitable for plotting a cohort, colored by
whatever metadata a caller has to hand -- "plot my samples in 2D based on
genomic similarity" is a routine exploratory step in microbiome/population
genomics (what QIIME2 calls a PCoA plot), and since `compare_all()` already
produces an N-sample pairwise table, this module is mostly the glue that
turns that long-format table into a dense distance matrix and hands it to a
standard dimensionality-reduction method.

This module deliberately does not import `umap` or `sklearn` at module load
time (see `_reduce` below): those are real, somewhat heavy optional
dependencies, and `fastdna`'s core package must stay importable without
them (docs/ml-genomics-roadmap.md, item 5).
"""

import numpy as np
import pyarrow as pa

import fastdna


def _distance_matrix(table, paths, metric):
    """Turns `compare_all`'s long-format `(sample_a, sample_b, metric)`
    table into a dense, symmetric `n x n` NumPy distance matrix, ordered
    to match `paths` (row/column `i` corresponds to `paths[i]`).

    Similarity-vs-distance handling (the part that is easy to get
    backwards): `mash_distance` is already a distance (0 = identical, 1 =
    maximally dissimilar), so it is copied through unchanged, with a zero
    diagonal. `jaccard` is a *similarity* (1 = identical, 0 = disjoint) --
    fed directly into a method expecting a distance matrix, it would
    embed every pair *backwards* (near-identical samples would look
    maximally far apart). This function converts `jaccard` to a proper
    distance via `1 - jaccard` before it ever reaches a `metric`/
    `dissimilarity="precomputed"` call, so every caller downstream of
    this function always receives a true distance matrix (0 = identical
    on the diagonal, larger = more dissimilar), regardless of which of
    `compare_all`'s two metrics produced it.
    """
    str_paths = [str(p) for p in paths]
    n = len(str_paths)

    dist = np.zeros((n, n), dtype=np.float64)

    # One vectorized fill rather than a Python loop over the table's
    # `n*(n-1)/2` rows: each row cost two dict lookups, a scalar subtract
    # and two element assignments, which for a 200-sample cohort is 19,900
    # iterations doing work NumPy does in four array operations. `1.0 - v`
    # elementwise on float64 is the same IEEE subtraction the scalar
    # version performed, so the matrix is unchanged value for value.
    i, j = fastdna._pair_positions(table, str_paths)
    values = np.asarray(fastdna._column_as_array(table.column(metric)))
    d = (1.0 - values) if metric == "jaccard" else values

    dist[i, j] = d
    dist[j, i] = d

    # Diagonal is always 0 after the conversion above: mash_distance(x, x)
    # would be 0 anyway, and 1 - jaccard(x, x) is 1 - 1 = 0 -- both
    # collapse to "identical to itself", which is what every embedding
    # method below expects of a precomputed distance matrix's diagonal.
    np.fill_diagonal(dist, 0.0)
    return dist


def _missing_dependency(package, extra_hint=None):
    hint = extra_hint or f"pip install {package}"
    return ImportError(
        f"embed_cohort(method=...) requires the '{package}' package, which "
        f"is not installed. Install it with `{hint}` and try again."
    )


def _reduce(distance_matrix, method, n_components, method_kwargs):
    """Runs the requested dimensionality-reduction method over an already-
    computed `n x n` distance matrix (see `_distance_matrix`), returning an
    `(n, n_components)` NumPy array of coordinates.

    Both optional dependencies (`umap-learn` for `"umap"`, `scikit-learn`
    for `"tsne"`/`"pcoa"`) are imported lazily, here, inside this function
    -- not at module import time -- so that importing `fastdna` (or even
    `fastdna.embed` itself) never requires either package; only actually
    calling `embed_cohort(..., method=...)` does, and only the one
    dependency that specific method needs.
    """
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise _missing_dependency("umap-learn") from None
        reducer = umap.UMAP(n_components=n_components, metric="precomputed", **method_kwargs)
        return np.asarray(reducer.fit_transform(distance_matrix))

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise _missing_dependency("scikit-learn") from None
        # TSNE with metric="precomputed" refuses its own default
        # init="pca" (PCA needs the original feature space, which a
        # precomputed distance matrix does not provide) -- init="random"
        # is required here, not just a stylistic choice.
        reducer = TSNE(n_components=n_components, metric="precomputed", init="random", **method_kwargs)
        return np.asarray(reducer.fit_transform(distance_matrix))

    if method == "pcoa":
        try:
            from sklearn.manifold import MDS
        except ImportError:
            raise _missing_dependency("scikit-learn") from None
        # `dissimilarity="precomputed"` is the portable spelling across the
        # sklearn versions this package supports. sklearn 1.9 deprecates it
        # in favour of `metric="precomputed"` (removal in 1.10) and emits a
        # FutureWarning -- but on sklearn <1.9 `MDS`'s `metric` parameter
        # means something entirely different (metric vs. non-metric MDS,
        # a bool), so switching early would silently mis-configure older
        # installs. A caller who wants the new spelling can pass
        # `metric="precomputed"` through `**method_kwargs` on a new enough
        # sklearn; revisit this when the supported floor moves past 1.10.
        reducer = MDS(n_components=n_components, dissimilarity="precomputed", **method_kwargs)
        return np.asarray(reducer.fit_transform(distance_matrix))

    raise ValueError(f"method must be 'umap', 'tsne', or 'pcoa', got {method!r}")


def embed_cohort(
    paths,
    *,
    k=21,
    sketch_size=1000,
    metric="mash_distance",
    method="umap",
    n_components=2,
    **method_kwargs,
):
    """Embeds a cohort of FASTQ(.gz) files into `n_components`-dimensional
    coordinates based on pairwise genomic (dis)similarity, for plotting
    "where does each sample sit relative to the others" -- the operation
    usually meant by a "PCoA plot" in microbiome/population genomics
    (QIIME2 and friends), generalized here to also support UMAP and t-SNE
    over the same precomputed distance matrix.

    Pipeline:
      1. `fastdna.compare_all(paths, k=k, sketch_size=sketch_size,
         metric=metric)` builds each sample's sketch once and computes
         every pairwise `metric` value (long-format table).
      2. That table is turned into a dense, symmetric `n x n` distance
         matrix in the same order as `paths` (see `_distance_matrix`).
         **Similarity-vs-distance handling**: `mash_distance` is already
         a distance (0 = identical) and is used as-is. `jaccard` is a
         *similarity* (1 = identical) and is converted to a distance via
         `1 - jaccard` before reaching any embedding method -- so the
         diagonal is always 0 ("identical to itself") and larger values
         always mean "more dissimilar", regardless of which metric was
         requested. Feeding a raw similarity matrix into a method that
         expects a distance would embed every pair backwards (near-
         identical samples pushed apart); this conversion is what
         prevents that.
      3. `method` runs over that distance matrix as a *precomputed*
         distance/dissimilarity matrix (never recomputed from raw
         features, since none exist here beyond the pairwise distances
         themselves):
           - `"umap"` (default): `umap.UMAP(metric="precomputed")`
             (requires the optional `umap-learn` package).
           - `"tsne"`: `sklearn.manifold.TSNE(metric="precomputed",
             init="random")` -- `init="random"` is required by sklearn
             whenever `metric="precomputed"`; its default `init="pca"`
             raises, since PCA needs original features, not a distance
             matrix (requires the optional `scikit-learn` package).
           - `"pcoa"`: `sklearn.manifold.MDS(dissimilarity="precomputed")`
             -- classical PCoA/multidimensional scaling, the standard
             method for this exact kind of data in the microbiome/
             population-genomics literature; needs no dependency beyond
             `scikit-learn` (also optional -- not a hard dependency of
             `fastdna` itself).
         Any `method_kwargs` are forwarded verbatim to the underlying
         estimator's constructor (e.g. `n_neighbors=` for UMAP,
         `perplexity=` for TSNE, `random_state=` for any of them).

    `umap-learn` and `scikit-learn` are both imported lazily, only inside
    this call and only for the specific `method` requested -- neither is
    a hard dependency of the `fastdna` package. A requested method whose
    dependency is not installed raises a clear `ImportError` naming the
    missing package and the `pip install` command to fix it, instead of a
    raw `ModuleNotFoundError` from deep inside sklearn/umap internals.

    Returns a `pyarrow.Table` with one row per input path (in `paths`
    order) and columns `sample`, `x`, `y` (and `z` when `n_components ==
    3`) -- consistent with `compare_all()` and `KmerCounts.table` already
    returning `pyarrow.Table`s, so the result composes directly with
    `.to_pandas()`, DuckDB, Polars, or a plotting library without any
    extra conversion step.
    """
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3, got {n_components!r}")

    str_paths = [str(p) for p in paths]

    # A duplicated path would make `_distance_matrix`'s path->index dict map
    # both occurrences to the last index, silently leaving the earlier
    # duplicate's row all zeros -- corrupt output, not a degenerate-but-valid
    # embedding. Refuse it explicitly, before any sketching work is done.
    seen = set()
    for p in str_paths:
        if p in seen:
            raise ValueError(
                f"paths contains a duplicate entry: {p!r}. Each sample may "
                "appear only once; a duplicated path would silently corrupt "
                "its distance-matrix row."
            )
        seen.add(p)

    table = fastdna.compare_all(str_paths, k=k, sketch_size=sketch_size, metric=metric)
    distance_matrix = _distance_matrix(table, str_paths, metric)

    coords = _reduce(distance_matrix, method, n_components, method_kwargs)

    columns = {"sample": str_paths, "x": coords[:, 0], "y": coords[:, 1]}
    if n_components == 3:
        columns["z"] = coords[:, 2]

    return pa.table(columns)
