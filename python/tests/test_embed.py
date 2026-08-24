"""Tests for fastdna.embed.embed_cohort() -- cohort visualization built on
top of fastdna.compare_all()'s pairwise distance/similarity table.
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import pyarrow as pa
import pytest

import fastdna
from fastdna.embed import _distance_matrix, embed_cohort


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def _random_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def _mutate(rng: random.Random, seq: str, n_mutations: int) -> str:
    bases = list(seq)
    positions = rng.sample(range(len(bases)), n_mutations)
    for pos in positions:
        choices = [b for b in "ACGT" if b != bases[pos]]
        bases[pos] = rng.choice(choices)
    return "".join(bases)


@pytest.fixture
def cohort_paths(tmp_path):
    """A 4-sample cohort: two near-identical samples derived from the same
    base sequence with a handful of point mutations, and two samples built
    from unrelated random sequences (clearly different from the first two
    and from each other).
    """
    rng = random.Random(1234)
    base = _random_seq(rng, 400)

    near_a = base
    near_b = _mutate(rng, base, n_mutations=8)  # ~2% divergence

    other_rng = random.Random(999)
    diff_c = _random_seq(other_rng, 400)
    diff_rng2 = random.Random(555)
    diff_d = _random_seq(diff_rng2, 400)

    paths = {
        "near_a": write_fastq(tmp_path, "near_a.fastq", [near_a]),
        "near_b": write_fastq(tmp_path, "near_b.fastq", [near_b]),
        "diff_c": write_fastq(tmp_path, "diff_c.fastq", [diff_c]),
        "diff_d": write_fastq(tmp_path, "diff_d.fastq", [diff_d]),
    }
    return {k: str(v) for k, v in paths.items()}


@pytest.fixture
def clustered_cohort_paths(tmp_path):
    """A 12-sample cohort in two clearly separated groups: six mutated
    variants of one random base sequence, and six of an unrelated one.

    UMAP is manifold learning -- it models each point's *local
    neighborhood* and normalizes away raw distance magnitudes, so a
    4-sample cohort (as in `cohort_paths`) is too small for its output to
    reflect the input distances in any reliable way. A two-cluster cohort
    of this size is the smallest input on which "same-group samples end up
    together" is a meaningful claim about UMAP rather than about noise.
    """
    rng = random.Random(20260822)
    base_one = _random_seq(rng, 600)
    base_two = _random_seq(rng, 600)

    group_a, group_b = [], []
    for i in range(6):
        a = write_fastq(tmp_path, f"ga{i}.fastq", [_mutate(rng, base_one, n_mutations=6)])
        b = write_fastq(tmp_path, f"gb{i}.fastq", [_mutate(rng, base_two, n_mutations=6)])
        group_a.append(str(a))
        group_b.append(str(b))

    return {"group_a": group_a, "group_b": group_b}


def _euclidean(row_a, row_b):
    return float(np.linalg.norm(np.array(row_a, dtype=np.float64) - np.array(row_b, dtype=np.float64)))


def _row(table, sample):
    idx = table.column("sample").to_pylist().index(sample)
    cols = [c for c in table.column_names if c != "sample"]
    return [table.column(c)[idx].as_py() for c in cols]


class TestPcoa:
    def test_shape_has_n_components_columns_and_one_row_per_path(self, cohort_paths):
        paths = list(cohort_paths.values())
        result = embed_cohort(paths, method="pcoa", n_components=2)

        assert isinstance(result, pa.Table)
        assert result.num_rows == len(paths)
        assert set(result.column_names) == {"sample", "x", "y"}

    def test_three_components_adds_a_z_column(self, cohort_paths):
        paths = list(cohort_paths.values())
        result = embed_cohort(paths, method="pcoa", n_components=3)

        assert set(result.column_names) == {"sample", "x", "y", "z"}
        assert result.num_rows == len(paths)

    def test_near_identical_samples_embed_closer_than_to_different_ones(self, cohort_paths):
        paths = list(cohort_paths.values())
        result = embed_cohort(
            paths,
            k=21,
            sketch_size=1000,
            metric="mash_distance",
            method="pcoa",
            n_components=2,
            random_state=42,
        )

        a = _row(result, cohort_paths["near_a"])
        b = _row(result, cohort_paths["near_b"])
        c = _row(result, cohort_paths["diff_c"])
        d = _row(result, cohort_paths["diff_d"])

        dist_ab = _euclidean(a, b)
        dist_ac = _euclidean(a, c)
        dist_ad = _euclidean(a, d)
        dist_bc = _euclidean(b, c)
        dist_bd = _euclidean(b, d)

        # The two near-identical samples must land closer to each other in
        # the embedded space than either lands to a clearly-different
        # sample -- this is the assertion that actually proves the
        # embedding preserves genomic-similarity structure, not merely
        # that the function ran without raising.
        assert dist_ab < dist_ac
        assert dist_ab < dist_ad
        assert dist_ab < dist_bc
        assert dist_ab < dist_bd


class TestSimilarityVsDistanceHandling:
    def test_distance_matrix_converts_jaccard_similarity_but_not_mash_distance(self):
        # Synthetic compare_all()-shaped table: "same" pair reports a
        # jaccard similarity of 1.0 (identical) and a mash_distance of
        # 0.0 (identical); "different" pair reports jaccard 0.0 (disjoint)
        # and mash_distance 1.0 (maximally dissimilar). A correct
        # _distance_matrix must turn both representations into the *same*
        # distance semantics: 0 = identical, 1 = maximally dissimilar.
        paths = ["a", "b", "c"]
        jaccard_table = pa.table(
            {
                "sample_a": ["a", "a", "b"],
                "sample_b": ["b", "c", "c"],
                "jaccard": [1.0, 0.0, 0.0],
            }
        )
        mash_table = pa.table(
            {
                "sample_a": ["a", "a", "b"],
                "sample_b": ["b", "c", "c"],
                "mash_distance": [0.0, 1.0, 1.0],
            }
        )

        jaccard_dist = _distance_matrix(jaccard_table, paths, "jaccard")
        mash_dist = _distance_matrix(mash_table, paths, "mash_distance")

        # Diagonal is always 0 (identical to itself) regardless of metric.
        assert np.allclose(np.diag(jaccard_dist), 0.0)
        assert np.allclose(np.diag(mash_dist), 0.0)

        # a-b: identical under both metrics -> distance 0.
        assert jaccard_dist[0, 1] == pytest.approx(0.0)
        assert mash_dist[0, 1] == pytest.approx(0.0)

        # a-c and b-c: maximally dissimilar under both metrics -> distance 1.
        assert jaccard_dist[0, 2] == pytest.approx(1.0)
        assert mash_dist[0, 2] == pytest.approx(1.0)
        assert jaccard_dist[1, 2] == pytest.approx(1.0)
        assert mash_dist[1, 2] == pytest.approx(1.0)

        # The two metrics must agree once converted to proper distances --
        # if jaccard had been fed through unconverted (as a similarity),
        # this would come out inverted (0 where mash says 1, and vice
        # versa).
        assert np.allclose(jaccard_dist, mash_dist)

    def test_embed_cohort_orders_pairs_correctly_for_both_metrics(self, cohort_paths):
        paths = list(cohort_paths.values())

        result_jaccard = embed_cohort(
            paths,
            k=21,
            sketch_size=1000,
            metric="jaccard",
            method="pcoa",
            n_components=2,
            random_state=42,
        )
        result_mash = embed_cohort(
            paths,
            k=21,
            sketch_size=1000,
            metric="mash_distance",
            method="pcoa",
            n_components=2,
            random_state=42,
        )

        for result in (result_jaccard, result_mash):
            a = _row(result, cohort_paths["near_a"])
            b = _row(result, cohort_paths["near_b"])
            c = _row(result, cohort_paths["diff_c"])

            dist_ab = _euclidean(a, b)
            dist_ac = _euclidean(a, c)

            # If jaccard's similarity semantics were silently treated as a
            # distance (i.e. not converted via 1 - jaccard), this ordering
            # would flip for the jaccard run: the near-identical pair
            # would appear "far" (low jaccard-as-distance value would
            # actually mean high similarity) while the different pair
            # would appear "close". Both metrics must agree on which pair
            # is closer.
            assert dist_ab < dist_ac


class TestMissingDependency:
    def test_umap_missing_dependency_raises_clear_importerror(self, cohort_paths, monkeypatch):
        monkeypatch.setitem(sys.modules, "umap", None)
        paths = list(cohort_paths.values())

        with pytest.raises(ImportError) as exc_info:
            embed_cohort(paths, method="umap", n_components=2)

        message = str(exc_info.value)
        assert "umap-learn" in message
        assert "pip install" in message

    def test_sklearn_missing_dependency_raises_clear_importerror_for_pcoa(self, cohort_paths, monkeypatch):
        monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
        paths = list(cohort_paths.values())

        with pytest.raises(ImportError) as exc_info:
            embed_cohort(paths, method="pcoa", n_components=2)

        message = str(exc_info.value)
        assert "scikit-learn" in message
        assert "pip install" in message

    def test_sklearn_missing_dependency_raises_clear_importerror_for_tsne(self, cohort_paths, monkeypatch):
        monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
        paths = list(cohort_paths.values())

        with pytest.raises(ImportError) as exc_info:
            embed_cohort(paths, method="tsne", n_components=2)

        message = str(exc_info.value)
        assert "scikit-learn" in message
        assert "pip install" in message


class TestValidation:
    def test_unknown_method_raises_valueerror(self, cohort_paths):
        paths = list(cohort_paths.values())
        with pytest.raises(ValueError):
            embed_cohort(paths, method="not_a_real_method")

    def test_unsupported_n_components_raises_valueerror(self, cohort_paths):
        paths = list(cohort_paths.values())
        with pytest.raises(ValueError):
            embed_cohort(paths, method="pcoa", n_components=1)


class TestTsne:
    def test_tsne_runs_over_a_precomputed_distance_matrix(self, cohort_paths):
        paths = list(cohort_paths.values())
        # `perplexity` must be < n_samples for sklearn's TSNE; with a
        # 4-sample cohort the default (30) would raise, so it is set
        # explicitly here and forwarded through `**method_kwargs`.
        result = embed_cohort(
            paths,
            k=21,
            sketch_size=1000,
            metric="mash_distance",
            method="tsne",
            n_components=2,
            perplexity=2,
            random_state=42,
        )

        assert result.num_rows == len(paths)
        assert set(result.column_names) == {"sample", "x", "y"}
        # Every coordinate must be a finite number -- a silent NaN here
        # would be the signature of TSNE having been handed something it
        # could not treat as a distance matrix (the init="pca" gotcha this
        # module guards against raises outright rather than producing
        # NaNs, but a finiteness check is the cheap catch-all).
        for column in ("x", "y"):
            assert all(np.isfinite(v) for v in result.column(column).to_pylist())


class TestUmap:
    def test_umap_separates_two_genomic_groups(self, clustered_cohort_paths):
        pytest.importorskip("umap")

        group_a = clustered_cohort_paths["group_a"]
        group_b = clustered_cohort_paths["group_b"]
        paths = group_a + group_b

        # `random_state` is required for a repeatable result: UMAP's
        # optimization is stochastic, and without a fixed seed this
        # assertion would flake.
        result = embed_cohort(
            paths,
            k=21,
            sketch_size=1000,
            metric="mash_distance",
            method="umap",
            n_components=2,
            n_neighbors=4,
            random_state=42,
        )

        assert result.num_rows == len(paths)
        assert set(result.column_names) == {"sample", "x", "y"}

        coords_a = [_row(result, p) for p in group_a]
        coords_b = [_row(result, p) for p in group_b]

        def mean_pairwise(left, right):
            pairs = [
                _euclidean(u, v)
                for i, u in enumerate(left)
                for j, v in enumerate(right)
                if left is not right or i < j
            ]
            return sum(pairs) / len(pairs)

        within = (mean_pairwise(coords_a, coords_a) + mean_pairwise(coords_b, coords_b)) / 2
        between = mean_pairwise(coords_a, coords_b)

        # Structural assertion rather than exact coordinates (which are
        # seed- and version-dependent): samples sharing a base sequence
        # must sit closer to each other on average than to samples from
        # the unrelated group.
        assert within < between
