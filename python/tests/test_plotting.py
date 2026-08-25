"""Tests for fastdna.plotting -- paper-ready figures for k-mer GWAS
screening results and Mash-distance population structure.

Matches test_genomescope.py's pattern for testing a matplotlib-dependent
function without a display: `pytest.importorskip("matplotlib")` +
`matplotlib.use("Agg")` inside each test that actually draws, and assertions
on what was drawn (line/point/image data) rather than a rendered image.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from fastdna.plotting import plot_population_structure, plot_significance

# ---------------------------------------------------------------------------
# Synthetic prefilter_association-shaped table
# ---------------------------------------------------------------------------


def _significance_table(n=10, seed=0, extra_columns=True):
    """A table with the same columns (and the same p_bonferroni/q_value_bh
    derivation) as `gwas.prefilter_association` returns, built directly from
    a set of raw p-values so every downstream test can compute its expected
    values from the same formulas rather than depending on gwas.py itself.
    """
    rng = np.random.default_rng(seed)
    p_value = np.sort(rng.uniform(1e-6, 0.9, size=n))
    p_bonferroni = np.clip(p_value * n, 0.0, 1.0)
    raw = p_value * n / np.arange(1, n + 1)
    q_value_bh = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0.0, 1.0)

    columns = {
        "kmer_sequence": [f"ACGT{i:04d}" for i in range(n)],
        "p_value": p_value,
        "p_bonferroni": p_bonferroni,
        "q_value_bh": q_value_bh,
    }
    if extra_columns:
        columns["odds_ratio"] = rng.uniform(0.1, 10.0, size=n)
    return pa.table(columns), p_value, p_bonferroni, q_value_bh


# ---------------------------------------------------------------------------
# plot_significance
# ---------------------------------------------------------------------------


def test_plot_significance_draws_one_point_per_row():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, p_value, _, _ = _significance_table(n=12)
    ax = plot_significance(table)

    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 12
    np.testing.assert_allclose(sorted(offsets[:, 0]), range(12))


def test_plot_significance_threshold_lines_use_the_tables_own_reported_values():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, _, p_bonferroni, q_value_bh = _significance_table(n=15, seed=7)
    ax = plot_significance(table, threshold_lines=True)

    expected_bonferroni = -np.log10(p_bonferroni.min())
    expected_q = -np.log10(q_value_bh.min())

    y_values = [line.get_ydata()[0] for line in ax.get_lines()]
    assert any(np.isclose(y, expected_bonferroni) for y in y_values)
    assert any(np.isclose(y, expected_q) for y in y_values)


def test_plot_significance_threshold_lines_false_draws_no_lines():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, *_ = _significance_table(n=8)
    ax = plot_significance(table, threshold_lines=False)

    assert ax.get_lines() == []


def test_plot_significance_position_without_positions_raises():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, *_ = _significance_table(n=5)
    with pytest.raises(ValueError) as excinfo:
        plot_significance(table, x="position")

    message = str(excinfo.value)
    assert "positions" in message


def test_plot_significance_position_with_positions_uses_them():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, *_ = _significance_table(n=6)
    positions = np.array([100, 250, 400, 900, 1200, 5000])
    ax = plot_significance(table, x="position", positions=positions)

    offsets = ax.collections[0].get_offsets()
    np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(positions))


def test_plot_significance_mismatched_positions_length_raises():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, *_ = _significance_table(n=6)
    with pytest.raises(ValueError):
        plot_significance(table, x="position", positions=[1, 2, 3])


def test_plot_significance_missing_columns_raises():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table = pa.table({"kmer_sequence": ["A", "C"], "p_value": [0.1, 0.2]})
    with pytest.raises(ValueError) as excinfo:
        plot_significance(table)

    message = str(excinfo.value)
    assert "p_bonferroni" in message
    assert "q_value_bh" in message


def test_plot_significance_ax_none_creates_new_axes_and_returns_it():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, *_ = _significance_table(n=5)
    ax = plot_significance(table)

    assert ax is not None
    assert ax.figure is not None
    plt.close("all")


def test_plot_significance_accepts_and_returns_a_given_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, *_ = _significance_table(n=5)
    _, ax = plt.subplots()
    returned = plot_significance(table, ax=ax)

    assert returned is ax
    assert len(ax.collections) == 1
    plt.close("all")


def test_plot_significance_x_must_be_index_or_position():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    table, *_ = _significance_table(n=5)
    with pytest.raises(ValueError):
        plot_significance(table, x="chromosome")


# ---------------------------------------------------------------------------
# plot_population_structure
# ---------------------------------------------------------------------------


def _synthetic_distance_matrix(n=8, seed=0):
    """A small, symmetric, zero-diagonal distance matrix with two obvious
    clusters (samples 0..n//2-1 close together, samples n//2..n-1 close
    together, the two groups far apart) -- enough structure for a dendrogram
    or heatmap to have something to show.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    labels = np.array([0] * half + [1] * (n - half))
    base = np.where(labels[:, None] == labels[None, :], 0.01, 0.5)
    noise = rng.uniform(0.0, 0.005, size=(n, n))
    noise = (noise + noise.T) / 2.0
    matrix = base + noise
    np.fill_diagonal(matrix, 0.0)
    sample_ids = [f"S{i}" for i in range(n)]
    return matrix, sample_ids, labels


def test_plot_population_structure_dendrogram_has_one_leaf_per_sample():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=9)
    ax = plot_population_structure(matrix, sample_ids, kind="dendrogram")

    assert len(ax.get_xticklabels()) == 9
    labels_drawn = {t.get_text() for t in ax.get_xticklabels()}
    assert labels_drawn == set(sample_ids)


def test_plot_population_structure_heatmap_has_the_right_shape():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=7)
    ax = plot_population_structure(matrix, sample_ids, kind="heatmap")

    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (7, 7)


def test_plot_population_structure_groups_coloring_is_applied_to_dendrogram_leaves():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, groups = _synthetic_distance_matrix(n=10)
    ax = plot_population_structure(matrix, sample_ids, groups=groups, kind="dendrogram")

    colors_by_text = {t.get_text(): t.get_color() for t in ax.get_xticklabels()}
    group_of_sample = dict(zip(sample_ids, groups.tolist()))

    # Every leaf whose sample is in group 0 shares one color, every leaf in
    # group 1 shares a different color, and the two colors differ -- i.e.
    # the coloring actually reflects the group labels, not just "some color
    # was set".
    colors_for_group = {0: set(), 1: set()}
    for sample_id, color in colors_by_text.items():
        colors_for_group[group_of_sample[sample_id]].add(color)

    assert len(colors_for_group[0]) == 1
    assert len(colors_for_group[1]) == 1
    assert colors_for_group[0] != colors_for_group[1]


def test_plot_population_structure_without_groups_does_not_claim_cv_grouping():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=6)
    ax = plot_population_structure(matrix, sample_ids, kind="dendrogram")

    title = ax.get_title().lower()
    assert "cv" not in title or "no cv" in title or "raw" in title


def test_plot_population_structure_mismatched_sample_ids_length_raises():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, _, _ = _synthetic_distance_matrix(n=6)
    with pytest.raises(ValueError):
        plot_population_structure(matrix, ["only", "three", "ids"])


def test_plot_population_structure_non_square_matrix_raises():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix = np.zeros((4, 5))
    with pytest.raises(ValueError):
        plot_population_structure(matrix, ["a", "b", "c", "d"])


def test_plot_population_structure_asymmetric_matrix_raises():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.2, 0.0, 0.3],  # disagrees with matrix[0, 1]
            [0.9, 0.3, 0.0],
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        plot_population_structure(matrix, ["a", "b", "c"])

    assert "symmetric" in str(excinfo.value)


def test_plot_population_structure_mismatched_groups_length_raises():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=6)
    with pytest.raises(ValueError):
        plot_population_structure(matrix, sample_ids, groups=[0, 0, 1])


def test_plot_population_structure_invalid_kind_raises():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=5)
    with pytest.raises(ValueError):
        plot_population_structure(matrix, sample_ids, kind="pie")


def test_plot_population_structure_ax_none_creates_new_and_passed_ax_is_returned():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=6)

    ax1 = plot_population_structure(matrix, sample_ids, kind="dendrogram")
    assert ax1 is not None

    _, ax2 = plt.subplots()
    returned = plot_population_structure(matrix, sample_ids, kind="heatmap", ax=ax2)
    assert returned is ax2
    plt.close("all")


def test_plot_population_structure_heatmap_ordered_by_dendrogram_leaf_order():
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    matrix, sample_ids, _ = _synthetic_distance_matrix(n=8, seed=3)
    ax = plot_population_structure(matrix, sample_ids, kind="heatmap")

    expected_order = dendrogram(linkage(squareform(matrix, checks=False), method="single"), no_plot=True)["leaves"]
    expected = matrix[np.ix_(expected_order, expected_order)]

    np.testing.assert_allclose(ax.images[0].get_array(), expected)
