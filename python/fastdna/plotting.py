"""fastdna.plotting -- paper-ready figures for two of this project's outputs:
a significance scatter over `gwas.prefilter_association`'s screening
results, and a Mash-distance population-structure view of the cohort
`cv.lineage_groups`/`cv.LineageKFold` derive leakage-safe CV splits from.

Follows `genomescope.py.plot_spectrum_fit`'s conventions exactly, since it is
the only existing plotting function in this codebase: `matplotlib` is
imported lazily, inside each function that actually draws, with an
actionable `ImportError` if it is missing; every function takes an optional
`ax` and always returns the axes it drew onto (a new one if `ax=None`), so a
caller can compose either figure into a larger one.

## Honesty, matching this project's discipline elsewhere

`plot_significance` visualizes `gwas.prefilter_association`'s output, which
is an **unadjusted screen**, not inference (see that function's own
docstring and its `ScreeningOnlyWarning`). This module does not relax that
caveat just because the output is now a plot instead of a table: no
threshold is invented here, `threshold_lines` only ever draws what the input
table itself already reported (see `plot_significance`'s docstring for
exactly what that means), and the title repeats the "screening, not
inference" framing so a reader who only ever sees the figure -- not the
docstring, not the warning -- still gets the caveat.

`plot_population_structure` visualizes a Mash-distance matrix. Without a
`groups` argument it is a plain distance visualization and says so; only
when `groups` is actually supplied (e.g. from `cv.lineage_groups()` or a
`cv.LineageKFold` fold assignment) does the figure claim to show how a CV
process split the cohort, because only then has the caller told this module
what that process actually decided.
"""
from __future__ import annotations

from . import _column_as_array

__all__ = ["plot_significance", "plot_population_structure"]

#: Columns `plot_significance` needs. These four are common to both
#: `prefilter_association` phenotype modes (binary and continuous) -- see
#: gwas.py's docstring for the two modes' full column lists -- so requiring
#: exactly these accepts either mode's table, plus any other table shaped
#: like one (the docstring says as much).
_REQUIRED_SIGNIFICANCE_COLUMNS = ("kmer_sequence", "p_value", "p_bonferroni", "q_value_bh")


def _missing_dependency(package, extra_hint=None):
    hint = extra_hint or f"pip install {package}"
    return ImportError(
        f"fastdna.plotting requires the '{package}' package, which is not "
        f"installed. Install it with `{hint}` and try again."
    )


def _numpy():
    """Imports numpy lazily, the same reason `genomescope._numpy` does:
    neither numpy nor matplotlib nor scipy is a runtime dependency of
    `fastdna`, so they are imported here, inside the call that needs them.
    """
    try:
        import numpy
    except ImportError:
        raise _missing_dependency("numpy") from None
    return numpy


def _matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise _missing_dependency("matplotlib") from None
    return plt


def _scipy_hierarchy():
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise _missing_dependency("scipy") from None
    return dendrogram, linkage, squareform


def plot_significance(table, *, x="index", positions=None, threshold_lines=True, ax=None):
    """Plots a significance scatter, in the visual style of a Manhattan
    plot, over `table` -- the `pyarrow.Table` :func:`fastdna.gwas.
    prefilter_association` returns (or any table with the same
    `p_value`/`p_bonferroni`/`q_value_bh`/`kmer_sequence`-shaped columns).

    **This is not a literal Manhattan plot.** A Manhattan plot places
    variants at their genomic coordinate; FastDNA is reference-free, so a
    k-mer screen ordinarily has no genomic coordinate to plot against. What
    this draws -- points ordered along x, `-log10(p_value)` on y, optional
    threshold reference lines -- is the same *visual style*, not the same
    claim, and calling it a Manhattan plot without that distinction would
    overclaim what a reference-free screen can show.

    Parameters
    ----------
    table : pyarrow.Table
        Must carry `kmer_sequence`, `p_value`, `p_bonferroni` and
        `q_value_bh` columns (every column `prefilter_association` returns
        in either phenotype mode includes these four). Missing columns raise
        `ValueError` naming them, rather than failing deep inside a
        `KeyError`.
    x : {"index", "position"}, default "index"
        `"index"` (the default, and the only option that needs nothing
        beyond `table` itself) plots k-mers in the table's existing row
        order -- i.e. `prefilter_association`'s own p-value-ascending sort.
        `"position"` plots against real genomic coordinates, for a caller
        who has them (e.g. from a future gene-annotation lookup step); it
        requires a companion `positions` array of the same length as
        `table`, and raises `ValueError` -- rather than silently falling
        back to index order -- if `positions` is not supplied.
    positions : array-like, optional
        Genomic coordinates, one per row of `table`, in the same order.
        Required when `x="position"`; ignored (and rejected, to avoid the
        caller believing it was used) when `x="index"`.
    threshold_lines : bool, default True
        Draws two horizontal reference lines: `-log10` of the *smallest*
        `p_bonferroni` value in `table`, and `-log10` of the *smallest*
        `q_value_bh` value in `table`. Deliberately not a conventional
        alpha=0.05 (or any other) cutoff: `prefilter_association` never
        reports an applied significance threshold -- only per-k-mer
        corrected values, "for orientation only, no threshold is applied"
        in its own words -- so this function has no threshold of the
        module's own choosing to draw. Taking the minimum of each reported
        column is the only threshold-shaped quantity that is genuinely
        *in* the table already: it marks how far this particular screen's
        most significant k-mer got on each correction, for orientation,
        without implying that crossing it means anything was confirmed.
        Nothing here corrects for population structure; see
        `prefilter_association`'s own docstring before treating anything
        on this plot as an association.
    ax : matplotlib.axes.Axes, optional
        Axes to draw onto. A new figure/axes is created if omitted. Always
        returned, so a caller can keep composing or styling it.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        `table` is missing a required column; `x` is not `"index"` or
        `"position"`; `x="position"` without `positions`; `positions` given
        with `x="index"`; or `positions` does not have one entry per row of
        `table`.

    `matplotlib` is imported lazily here and is not a dependency of
    `fastdna`; a missing install raises an `ImportError` naming the package
    and the install command.
    """
    plt = _matplotlib_pyplot()
    np = _numpy()

    if x not in ("index", "position"):
        raise ValueError(f"x must be 'index' or 'position', got {x!r}")

    missing = [c for c in _REQUIRED_SIGNIFICANCE_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(
            f"table is missing required column(s) {missing}. plot_significance() expects the "
            "shape gwas.prefilter_association() returns -- kmer_sequence, p_value, p_bonferroni "
            f"and q_value_bh at minimum -- but this table only has {list(table.column_names)}."
        )

    p_value = np.asarray(_column_as_array(table.column("p_value")), dtype=np.float64)
    p_bonferroni = np.asarray(_column_as_array(table.column("p_bonferroni")), dtype=np.float64)
    q_value_bh = np.asarray(_column_as_array(table.column("q_value_bh")), dtype=np.float64)
    n = p_value.shape[0]

    if x == "position":
        if positions is None:
            raise ValueError(
                "x='position' requires a companion `positions` array of real genomic "
                "coordinates (e.g. from a gene-annotation lookup step) -- pass "
                "positions=[...], or omit x (or pass x='index') to plot k-mers in the "
                "table's own row order instead, since fastdna is reference-free and usually "
                "has no genomic coordinate to plot a k-mer against."
            )
        positions = np.asarray(positions)
        if positions.shape[0] != n:
            raise ValueError(
                f"positions has {positions.shape[0]} entries but table has {n} rows; they "
                "must line up one-to-one, positions[i] for table row i."
            )
        x_values = positions
        x_label = "genomic position"
    else:
        if positions is not None:
            raise ValueError(
                "positions was given but x='index' does not use it (it would be silently "
                "ignored). Pass x='position' to plot against these coordinates."
            )
        x_values = np.arange(n)
        x_label = "k-mer index (table row order)"

    with np.errstate(divide="ignore"):
        y_values = -np.log10(p_value)

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x_values, y_values, s=10, color="0.25", linewidths=0, label="k-mer")

    if threshold_lines:
        min_bonferroni = float(np.min(p_bonferroni))
        min_q = float(np.min(q_value_bh))
        with np.errstate(divide="ignore"):
            ax.axhline(
                -np.log10(min_bonferroni),
                color="C3",
                linestyle="--",
                linewidth=1.0,
                label=f"min p_bonferroni = {min_bonferroni:.3g}",
            )
            ax.axhline(
                -np.log10(min_q),
                color="C0",
                linestyle=":",
                linewidth=1.0,
                label=f"min q_value_bh = {min_q:.3g}",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$-\log_{10}(p_{value})$")
    ax.set_title("k-mer significance screen (screening only -- see gwas.prefilter_association)")
    ax.legend(fontsize="small")
    return ax


def _group_color_map(groups, cmap):
    """Maps each distinct value of `groups` to a color from `cmap`, in
    order of first appearance, so repeated calls over the same labels are
    reproducible and small integer labels (as `lineage_groups()` returns)
    get the same colors every time.
    """
    seen = []
    for label in (groups.tolist() if hasattr(groups, "tolist") else list(groups)):
        if label not in seen:
            seen.append(label)
    return {label: cmap(i % cmap.N) for i, label in enumerate(seen)}


def plot_population_structure(distance_matrix, sample_ids, *, groups=None, kind="dendrogram", ax=None):
    """Plots the cohort's Mash-distance population structure -- a
    dendrogram or a heatmap -- optionally colored by group labels so a
    caller can see which samples a leakage-safe CV split (or any other
    grouping) considered the same lineage.

    Parameters
    ----------
    distance_matrix : array-like, shape (n, n)
        A square, symmetric Mash-distance matrix, e.g. from
        `fastdna.gwas.kinship_matrix` (which returns `1 - mash_distance`
        similarity -- pass `1 - similarity` to get a distance back) or
        directly from `fastdna.compare_all(paths, metric="mash_distance")`
        turned into a dense matrix (see `fastdna.cv._mash_distance_matrix`
        for exactly that conversion, which `cv.lineage_groups()` itself
        uses). A non-square matrix raises `ValueError`. A square matrix
        that is not symmetric within floating-point tolerance also raises,
        rather than being silently symmetrized: a genuine Mash-distance
        matrix is symmetric by construction, so disagreement between
        `(i, j)` and `(j, i)` almost always means the wrong array was
        passed in (a similarity instead of a distance, or a
        non-square-safe transpose bug), and this function has no principled
        way to decide which of the two disagreeing values is the real one.
    sample_ids : sequence of str, length n
        Labels aligned with `distance_matrix`'s rows/columns. A length
        mismatch raises `ValueError`.
    groups : array-like of length n, optional
        Per-sample group/fold labels, e.g. from `fastdna.cv.lineage_groups()`
        or a `fastdna.cv.LineageKFold` fold assignment. When given, leaves
        (dendrogram) or tick labels (heatmap) are colored by group so it is
        visually obvious which samples were treated as one lineage. When
        omitted, this is a plain visualization of the raw pairwise
        distances -- the title says so explicitly, and no claim is made
        about how any particular CV run split anything, because none was
        told to this function. A length mismatch against `sample_ids`
        raises `ValueError`.
    kind : {"dendrogram", "heatmap"}, default "dendrogram"
        `"dendrogram"` runs `scipy.cluster.hierarchy` **single linkage** --
        matching `fastdna.cv.lineage_groups()`'s own clustering method, so
        the tree shown here is consistent with the one that method's
        threshold cut. `"heatmap"` is a plain `sample x sample` distance
        heatmap, rows/columns ordered by the dendrogram's own leaf order
        (single linkage is still computed for this, purely for a
        consistent ordering -- no threshold is cut and no clusters are
        formed) so a heatmap and a dendrogram of the same cohort read the
        same way. Anything else raises `ValueError`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw onto. A new figure/axes is created if omitted. Always
        returned.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Non-square `distance_matrix`; `sample_ids` length does not match
        it; the matrix is not symmetric; `groups` length does not match it;
        `kind` is not `"dendrogram"` or `"heatmap"`.

    `matplotlib` and `scipy` are imported lazily here and are not
    dependencies of `fastdna`; a missing install raises an `ImportError`
    naming the package and the install command.
    """
    plt = _matplotlib_pyplot()
    np = _numpy()
    dendrogram, linkage, squareform = _scipy_hierarchy()

    if kind not in ("dendrogram", "heatmap"):
        raise ValueError(f"kind must be 'dendrogram' or 'heatmap', got {kind!r}")

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    sample_ids = list(sample_ids)
    n = len(sample_ids)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {matrix.shape}")
    if matrix.shape[0] != n:
        raise ValueError(
            f"sample_ids has {n} entries but distance_matrix is {matrix.shape[0]}x{matrix.shape[0]}; "
            "they must have the same length, one id per row/column."
        )
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError(
            "distance_matrix is not symmetric: some (i, j) entry disagrees with (j, i) by more "
            "than floating-point tolerance. A genuine Mash-distance matrix (e.g. from "
            "gwas.kinship_matrix or compare_all) is symmetric by construction, so this usually "
            "means a similarity (not a distance) was passed in, or the matrix was built from the "
            "wrong axis. This function refuses to guess which of the two disagreeing values is "
            "correct rather than silently averaging or transposing it."
        )

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != n:
            raise ValueError(f"groups has {groups.shape[0]} entries but there are {n} samples")

    # checks=False: the matrix is symmetric with a (near-)zero diagonal by
    # construction/validation above, and squareform's own strict
    # floating-point symmetry check would reject it spuriously -- the same
    # reasoning fastdna.cv.lineage_groups() uses for its own squareform call.
    condensed = squareform(matrix, checks=False)
    # Single linkage: matches lineage_groups()'s own clustering method (see
    # its docstring), so this figure is consistent with what that function
    # actually grouped, not a different clustering convention.
    tree = linkage(condensed, method="single")

    if ax is None:
        _, ax = plt.subplots()

    if kind == "dendrogram":
        ddata = dendrogram(tree, labels=sample_ids, ax=ax)
        leaf_order = ddata["leaves"]
        if groups is not None:
            colors = _group_color_map(groups, plt.cm.tab10)
            for tick_label in ax.get_xticklabels():
                tick_label.set_color(colors[groups[sample_ids.index(tick_label.get_text())]])
        ax.set_xlabel("sample")
        ax.set_ylabel("Mash distance")
        title = "population structure (Mash-distance dendrogram, single linkage)"
    else:
        leaf_order = dendrogram(tree, no_plot=True)["leaves"]
        ordered_matrix = matrix[np.ix_(leaf_order, leaf_order)]
        ordered_labels = [sample_ids[i] for i in leaf_order]

        image = ax.imshow(ordered_matrix, cmap="viridis_r", aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels(ordered_labels, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ordered_labels)
        if groups is not None:
            colors = _group_color_map(groups, plt.cm.tab10)
            for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                tick_label.set_color(colors[groups[sample_ids.index(tick_label.get_text())]])
        colorbar = ax.figure.colorbar(image, ax=ax)
        colorbar.set_label("Mash distance")
        title = "population structure (Mash-distance heatmap, dendrogram-ordered)"

    if groups is None:
        title += " -- raw pairwise distance only; no CV grouping is shown or claimed"
    else:
        title += " -- color shows the given group labels"
    ax.set_title(title)
    return ax
