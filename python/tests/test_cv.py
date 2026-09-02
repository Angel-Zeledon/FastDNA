"""Tests for fastdna.cv -- population-structure-aware, leakage-safe model
evaluation.

The lineage tests deliberately use *real* fastdna sketching over tiny
synthetic FASTQ files (the same fixture style as test_embed.py /
test_taxonomy.py) rather than hand-fed distance matrices: the whole claim
of `lineage_groups` is that FastDNA can derive leakage-safe groups from
the genomes themselves, and a test that skipped the sketching step would
not be testing that claim at all.

`permutation_importance_pvalues` is tested over a plain numeric matrix
instead, because it is deliberately agnostic about where its `X` came
from (a `KmerVectorizer` matrix, a dense array, anything an sklearn
estimator accepts) and refitting a model 30+ times over real FASTQ files
would make the suite slow for no extra coverage.
"""
from __future__ import annotations

import pathlib
import random

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

import pyarrow as pa

import fastdna
from fastdna.cv import (
    LineageKFold,
    default_threshold_curve,
    lineage_groups,
    lineage_groups_at_thresholds,
    lineage_groups_from_distances,
    lineage_groups_from_tree,
    permutation_importance_pvalues,
)


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def _random_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def _mutate(rng: random.Random, seq: str, rate: float) -> str:
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice("ACGT")
    return "".join(out)


# `k`/`sketch_size` small enough that the whole suite stays fast, large
# enough that a 600 bp genome still yields a meaningful bottom-k sketch.
_K = 21
_SKETCH_SIZE = 200
# Measured on this fixture: within-lineage mash distances top out around
# 0.023, between-lineage distances are 1.0 (unrelated random sequences),
# so any threshold in roughly [0.03, 0.9] recovers the true lineages.
# 0.05 is picked from the middle of that range, not tuned to the seed.
_THRESHOLD = 0.05


@pytest.fixture
def lineage_cohort(tmp_path):
    """Three "lineages" of four samples each: three unrelated 600 bp base
    sequences, each sampled four times with a 1% per-base mutation rate --
    the synthetic stand-in for clonal population structure, where members
    of one lineage are near-identical and members of different lineages
    share no k-mers at all.

    Returns `(paths, true_groups)` with `true_groups[i]` the lineage index
    of `paths[i]`.
    """
    rng = random.Random(20260824)
    bases = [_random_seq(rng, 600) for _ in range(3)]

    paths, true_groups = [], []
    for lineage, base in enumerate(bases):
        for i in range(4):
            sequence = _mutate(rng, base, rate=0.01)
            paths.append(str(write_fastq(tmp_path, f"lin{lineage}_{i}.fastq", [sequence] * 3)))
            true_groups.append(lineage)
    return paths, true_groups


def _as_partition(labels):
    """The set-of-frozensets view of a labelling, so two labellings can be
    compared for "same grouping" without depending on which integer each
    group happened to receive.
    """
    partition = {}
    for index, label in enumerate(labels):
        partition.setdefault(label, set()).add(index)
    return {frozenset(members) for members in partition.values()}


class TestLineageGroups:
    def test_two_synthetic_clusters_yield_two_groups(self, lineage_cohort):
        paths, true_groups = lineage_cohort
        two_clusters = [p for p, g in zip(paths, true_groups) if g in (0, 1)]
        expected = [g for g in true_groups if g in (0, 1)]

        groups = lineage_groups(two_clusters, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert len(set(groups)) == 2
        assert _as_partition(groups) == _as_partition(expected)

    def test_three_synthetic_lineages_are_recovered_exactly(self, lineage_cohort):
        paths, true_groups = lineage_cohort

        groups = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert _as_partition(groups) == _as_partition(true_groups)

    def test_returns_zero_based_contiguous_integer_labels_in_first_appearance_order(self, lineage_cohort):
        paths, _ = lineage_cohort

        groups = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert isinstance(groups, np.ndarray)
        assert np.issubdtype(groups.dtype, np.integer)
        assert sorted(set(groups.tolist())) == list(range(len(set(groups.tolist()))))
        # First-appearance ordering makes the labelling deterministic and
        # readable: paths[0] is always group 0.
        assert groups[0] == 0

    def test_a_threshold_below_within_lineage_divergence_splits_the_lineages(self, lineage_cohort):
        """The mechanism behind `LineageKFold`'s "raise distance_threshold"
        advice, pinned directly: the threshold *is* the knob that decides how
        many groups there are, and too tight a one shatters real lineages
        into singletons.
        """
        paths, _ = lineage_cohort

        tight = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=1e-6)

        assert len(set(tight.tolist())) > 3

    def test_a_threshold_above_every_distance_merges_everything(self, lineage_cohort):
        paths, _ = lineage_cohort

        merged = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=1.5)

        assert len(set(merged.tolist())) == 1

    def test_fewer_than_two_paths_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError) as exc_info:
            lineage_groups(paths[:1], k=_K, sketch_size=_SKETCH_SIZE)

        assert "2" in str(exc_info.value)

    def test_duplicate_paths_raise_naming_the_path(self, lineage_cohort):
        paths, _ = lineage_cohort
        duplicated = paths + [paths[0]]

        with pytest.raises(ValueError) as exc_info:
            lineage_groups(duplicated, k=_K, sketch_size=_SKETCH_SIZE)

        assert repr(paths[0]) in str(exc_info.value)

    def test_non_positive_distance_threshold_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError):
            lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=0.0)


class TestLineageGroupsAtThresholds:
    def test_agrees_with_lineage_groups_called_per_threshold(self, lineage_cohort):
        """The cross-check that makes this function trustworthy: cutting one
        dendrogram at several thresholds must give exactly the same
        groupings as calling `lineage_groups()` once per threshold.
        """
        paths, _ = lineage_cohort
        thresholds = [1e-6, _THRESHOLD, 1.5]

        batched = lineage_groups_at_thresholds(paths, thresholds, k=_K, sketch_size=_SKETCH_SIZE)

        for threshold, groups in zip(thresholds, batched):
            expected = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=threshold)
            assert groups.tolist() == expected.tolist(), f"mismatch at threshold {threshold}"

    def test_returns_results_in_thresholds_order(self, lineage_cohort):
        paths, _ = lineage_cohort
        # Deliberately out-of-order and including a very tight and a very
        # loose threshold, so a naive implementation that sorted internally
        # would be caught by this ordering check.
        thresholds = [1.5, 1e-6, _THRESHOLD]

        batched = lineage_groups_at_thresholds(paths, thresholds, k=_K, sketch_size=_SKETCH_SIZE)

        assert len(set(batched[0].tolist())) == 1  # 1.5: everything merges
        assert len(set(batched[1].tolist())) > 3  # 1e-6: shattered
        assert len(set(batched[2].tolist())) == 3  # _THRESHOLD: the true lineages

    def test_handles_a_single_element_thresholds_list(self, lineage_cohort):
        paths, _ = lineage_cohort

        batched = lineage_groups_at_thresholds(paths, [_THRESHOLD], k=_K, sketch_size=_SKETCH_SIZE)

        assert len(batched) == 1
        expected = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)
        assert batched[0].tolist() == expected.tolist()

    def test_fewer_than_two_paths_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError) as exc_info:
            lineage_groups_at_thresholds(paths[:1], [_THRESHOLD], k=_K, sketch_size=_SKETCH_SIZE)

        assert "2" in str(exc_info.value)

    def test_duplicate_paths_raise_naming_the_path(self, lineage_cohort):
        paths, _ = lineage_cohort
        duplicated = paths + [paths[0]]

        with pytest.raises(ValueError) as exc_info:
            lineage_groups_at_thresholds(duplicated, [_THRESHOLD], k=_K, sketch_size=_SKETCH_SIZE)

        assert repr(paths[0]) in str(exc_info.value)

    def test_non_positive_threshold_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError):
            lineage_groups_at_thresholds(paths, [_THRESHOLD, 0.0], k=_K, sketch_size=_SKETCH_SIZE)

    def test_empty_thresholds_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError):
            lineage_groups_at_thresholds(paths, [], k=_K, sketch_size=_SKETCH_SIZE)


class TestDefaultThresholdCurve:
    def test_returns_n_points_strictly_positive_increasing_thresholds(self, lineage_cohort):
        paths, _ = lineage_cohort

        thresholds = default_threshold_curve(paths, n_points=5, k=_K, sketch_size=_SKETCH_SIZE)

        assert len(thresholds) == 5
        assert all(t > 0 for t in thresholds)
        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds), "quantiles collided but were not deduplicated"

    def test_thresholds_span_a_real_range_of_this_cohort_own_granularity(self, lineage_cohort):
        """The whole point: thresholds are drawn from this cohort's own
        dendrogram, not fixed constants, so the finest and coarsest points
        must land on opposite sides of the fixture's known true-lineage
        threshold (0.03-0.9, see _THRESHOLD's own comment above) -- proof
        this is reading real merge heights, not returning an arbitrary
        fixed list independent of the data.
        """
        paths, _ = lineage_cohort

        thresholds = default_threshold_curve(paths, n_points=5, k=_K, sketch_size=_SKETCH_SIZE)

        finest_groups = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=thresholds[0])
        coarsest_groups = lineage_groups(
            paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=thresholds[-1]
        )
        assert len(set(finest_groups.tolist())) > len(set(coarsest_groups.tolist())), (
            "the finest and coarsest default thresholds resolved the same grouping -- "
            "the curve is not spanning this cohort's real granularity range"
        )

    def test_n_points_is_capped_to_available_distinct_merge_heights(self, tmp_path):
        """Two samples have exactly one merge height in their dendrogram --
        asking for more points than that must not raise or fabricate
        duplicates, just return what the cohort's own dendrogram has.
        """
        rng = random.Random(20260901)
        base = _random_seq(rng, 600)
        paths = [
            write_fastq(tmp_path, "a.fastq", [_mutate(rng, base, 0.01)] * 3).as_posix(),
            write_fastq(tmp_path, "b.fastq", [_mutate(rng, base, 0.01)] * 3).as_posix(),
        ]

        thresholds = default_threshold_curve(paths, n_points=5, k=_K, sketch_size=_SKETCH_SIZE)

        assert 1 <= len(thresholds) <= 5

    def test_fewer_than_two_paths_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError):
            default_threshold_curve(paths[:1], k=_K, sketch_size=_SKETCH_SIZE)

    def test_non_positive_n_points_raises(self, lineage_cohort):
        paths, _ = lineage_cohort

        with pytest.raises(ValueError):
            default_threshold_curve(paths, n_points=0, k=_K, sketch_size=_SKETCH_SIZE)


class TestCohortCountsInput:
    """`lineage_groups()`/`lineage_groups_at_thresholds()`/
    `default_threshold_curve()` all accept a `fastdna.CohortCounts` in
    place of a list of paths (see `_mash_distance_matrix()`'s own
    docstring). The equivalence check that matters: sketching from the
    already-counted k-mers must reproduce exactly what sketching by
    streaming the same files would have produced, and the whole point is
    that it does so without opening a single FASTQ file again.
    """

    @pytest.fixture
    def lineage_counts(self, lineage_cohort):
        paths, true_groups = lineage_cohort
        counts = fastdna.count_cohort(paths, k=_K)
        # count_cohort()'s sample_ids come from the filenames, in `paths`
        # order -- so `true_groups` (aligned to `paths`) is also aligned to
        # `counts.sample_ids` here, letting every test below reuse it as-is.
        return counts, true_groups

    def test_lineage_groups_from_counts_matches_lineage_groups_from_paths(self, lineage_cohort, lineage_counts):
        paths, _ = lineage_cohort
        counts, _ = lineage_counts

        from_paths = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)
        # k is intentionally NOT passed here (or passed and ignored, see the
        # next test) -- sketching for a CohortCounts input always uses
        # counts.k.
        from_counts = lineage_groups(counts, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert from_counts.tolist() == from_paths.tolist()

    def test_an_explicit_k_alongside_a_cohortcounts_is_silently_ignored(self, lineage_cohort, lineage_counts):
        """The documented design decision (`_mash_distance_matrix()`'s own
        docstring): `k=` cannot be told apart from "the caller never
        mentioned k", so a CohortCounts input always sketches at
        `counts.k`, and passing a different `k=` alongside it changes
        nothing -- rather than raising or silently producing a different
        (wrong) result.
        """
        paths, _ = lineage_cohort
        counts, _ = lineage_counts
        assert counts.k == _K

        default_k = lineage_groups(counts, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)
        # A deliberately wrong k, nowhere near _K -- if it were honoured,
        # sketching would use a completely different k-mer size than the
        # cohort was counted at and very likely change the grouping.
        wrong_k = lineage_groups(counts, k=5, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert wrong_k.tolist() == default_k.tolist()

    def test_lineage_groups_at_thresholds_from_counts_matches_from_paths(self, lineage_cohort, lineage_counts):
        paths, _ = lineage_cohort
        counts, _ = lineage_counts
        thresholds = [1e-6, _THRESHOLD, 1.5]

        from_paths = lineage_groups_at_thresholds(paths, thresholds, k=_K, sketch_size=_SKETCH_SIZE)
        from_counts = lineage_groups_at_thresholds(counts, thresholds, sketch_size=_SKETCH_SIZE)

        for threshold, expected, actual in zip(thresholds, from_paths, from_counts):
            assert actual.tolist() == expected.tolist(), f"mismatch at threshold {threshold}"

    def test_default_threshold_curve_from_counts_matches_from_paths(self, lineage_cohort, lineage_counts):
        paths, _ = lineage_cohort
        counts, _ = lineage_counts

        from_paths = default_threshold_curve(paths, n_points=5, k=_K, sketch_size=_SKETCH_SIZE)
        from_counts = default_threshold_curve(counts, n_points=5, sketch_size=_SKETCH_SIZE)

        assert from_counts == pytest.approx(from_paths)

    def test_fewer_than_two_samples_in_a_cohortcounts_raises(self, lineage_counts):
        counts, _ = lineage_counts
        one_sample = counts.subset([counts.sample_ids[0]])

        with pytest.raises(ValueError) as exc_info:
            lineage_groups(one_sample, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert "2" in str(exc_info.value)

    def test_lineage_groups_from_a_cohortcounts_never_rereads_a_fastq_file(self, lineage_counts, monkeypatch):
        """The point of accepting a CohortCounts at all, measured directly:
        deriving lineage groups from it must not touch `fastdna.count()`
        (which `count_cohort()` used, once, before this test starts) nor
        `fastdna._core.sketch()` (the streaming, real-file sketch path
        `fastdna.sketch()`/`fastdna.compare_all()` use for a plain paths
        input) -- every sketch must come from `sketch_from_kmers()` over
        the already-counted k-mers instead. Mirrors
        `test_cohort_counts.py::test_cross_validation_with_the_artifact_never_recounts`'s
        own spy pattern.
        """
        counts, _ = lineage_counts

        count_calls = {"n": 0}
        sketch_calls = {"n": 0}
        real_count = fastdna.count
        real_sketch = fastdna._core.sketch

        def count_spy(*args, **kwargs):
            count_calls["n"] += 1
            return real_count(*args, **kwargs)

        def sketch_spy(*args, **kwargs):
            sketch_calls["n"] += 1
            return real_sketch(*args, **kwargs)

        monkeypatch.setattr(fastdna, "count", count_spy)
        monkeypatch.setattr(fastdna._core, "sketch", sketch_spy)

        lineage_groups(counts, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)
        default_threshold_curve(counts, n_points=3, sketch_size=_SKETCH_SIZE)

        assert count_calls["n"] == 0, "deriving lineage groups from a CohortCounts must not call fastdna.count()"
        assert sketch_calls["n"] == 0, (
            "deriving lineage groups from a CohortCounts must not stream any FASTQ file via "
            "fastdna._core.sketch()"
        )


class TestLineageGroupsFromDistances:
    def test_two_clear_clusters_via_block_diagonal_matrix(self):
        """Four samples, two clusters of two: small within-block distance,
        large between-block distance, built by hand so the expected
        grouping is obvious.
        """
        distances = np.array(
            [
                [0.00, 0.01, 1.00, 1.00],
                [0.01, 0.00, 1.00, 1.00],
                [1.00, 1.00, 0.00, 0.02],
                [1.00, 1.00, 0.02, 0.00],
            ]
        )

        groups = lineage_groups_from_distances(distances, distance_threshold=0.1)

        assert _as_partition(groups) == _as_partition([0, 0, 1, 1])

    def test_agrees_with_lineage_groups_over_the_equivalent_mash_matrix(self, lineage_cohort):
        """The shared-math claim, pinned directly: clustering the exact
        Mash-distance matrix `lineage_groups()` builds internally, by hand
        via `lineage_groups_from_distances()`, must reproduce the identical
        labels `lineage_groups()` itself returns -- not just a plausible-
        looking grouping.
        """
        from fastdna.cv import _mash_distance_matrix

        paths, _ = lineage_cohort
        matrix = _mash_distance_matrix(paths, _K, _SKETCH_SIZE)

        via_distances = lineage_groups_from_distances(matrix, distance_threshold=_THRESHOLD)
        via_paths = lineage_groups(paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert via_distances.tolist() == via_paths.tolist()

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError) as exc_info:
            lineage_groups_from_distances(np.zeros((3, 4)), distance_threshold=0.1)

        assert "square" in str(exc_info.value)

    def test_asymmetric_matrix_raises(self):
        distances = np.array([[0.0, 0.5, 0.9], [0.1, 0.0, 0.8], [0.9, 0.8, 0.0]])

        with pytest.raises(ValueError) as exc_info:
            lineage_groups_from_distances(distances, distance_threshold=0.1)

        assert "symmetric" in str(exc_info.value)

    def test_nonzero_diagonal_raises(self):
        distances = np.array([[0.3, 0.5, 0.9], [0.5, 0.0, 0.8], [0.9, 0.8, 0.0]])

        with pytest.raises(ValueError) as exc_info:
            lineage_groups_from_distances(distances, distance_threshold=0.1)

        assert "diagonal" in str(exc_info.value)

    def test_non_positive_distance_threshold_raises(self):
        distances = np.array([[0.0, 0.5], [0.5, 0.0]])

        with pytest.raises(ValueError):
            lineage_groups_from_distances(distances, distance_threshold=0.0)


class TestLineageGroupsFromTree:
    # Root has two children: leaf A (branch 0.1) and an internal node
    # (branch 0.2) whose own two children are leaves B and C (branch 0.05
    # each). Patristic distances, computed by hand:
    #   A-B = 0.1 + 0.2 + 0.05 = 0.35
    #   A-C = 0.1 + 0.2 + 0.05 = 0.35
    #   B-C = 0.05 + 0.05      = 0.10
    # so a threshold of 0.15 merges B and C but leaves A on its own.
    _NEWICK = "(A:0.1,(B:0.05,C:0.05):0.2);"

    def test_newick_string_recovers_the_hand_computed_groups(self):
        pytest.importorskip("Bio")

        groups = lineage_groups_from_tree(self._NEWICK, ["A", "B", "C"], distance_threshold=0.15)

        assert groups.tolist() == [0, 1, 1]

    def test_newick_file_path_gives_the_same_result_as_the_string(self, tmp_path):
        pytest.importorskip("Bio")

        tree_path = tmp_path / "tree.nwk"
        tree_path.write_text(self._NEWICK)

        from_path = lineage_groups_from_tree(tree_path, ["A", "B", "C"], distance_threshold=0.15)
        from_str_path = lineage_groups_from_tree(str(tree_path), ["A", "B", "C"], distance_threshold=0.15)
        from_string = lineage_groups_from_tree(self._NEWICK, ["A", "B", "C"], distance_threshold=0.15)

        assert from_path.tolist() == from_string.tolist()
        assert from_str_path.tolist() == from_string.tolist()

    def test_missing_leaf_name_raises_with_leaf_label_guidance(self):
        pytest.importorskip("Bio")

        with pytest.raises(ValueError) as exc_info:
            lineage_groups_from_tree(self._NEWICK, ["A", "B", "not_in_tree"], distance_threshold=0.15)

        message = str(exc_info.value)
        assert "not_in_tree" in message
        # A few real tree leaf labels should be surfaced too, to help spot a
        # spelling mismatch.
        assert "A" in message and "B" in message and "C" in message

    def test_a_higher_threshold_merges_all_three_leaves(self):
        pytest.importorskip("Bio")

        groups = lineage_groups_from_tree(self._NEWICK, ["A", "B", "C"], distance_threshold=0.5)

        assert len(set(groups.tolist())) == 1

    def test_a_lower_threshold_splits_all_three_leaves(self):
        pytest.importorskip("Bio")

        groups = lineage_groups_from_tree(self._NEWICK, ["A", "B", "C"], distance_threshold=0.01)

        assert len(set(groups.tolist())) == 3

    def test_non_positive_distance_threshold_raises(self):
        pytest.importorskip("Bio")

        with pytest.raises(ValueError):
            lineage_groups_from_tree(self._NEWICK, ["A", "B", "C"], distance_threshold=0.0)


class TestLineageKFold:
    def test_split_never_places_a_lineage_across_the_train_test_boundary(self, lineage_cohort):
        """The single assertion this whole module exists for: no lineage may
        have members in a training fold and its matching test fold at the
        same time. That is exactly the clonal leakage random CV commits.
        """
        paths, true_groups = lineage_cohort
        splitter = LineageKFold(n_splits=3, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        n_splits_seen = 0
        for train_idx, test_idx in splitter.split(paths):
            n_splits_seen += 1
            train_lineages = {true_groups[i] for i in train_idx}
            test_lineages = {true_groups[i] for i in test_idx}
            assert not (train_lineages & test_lineages), (
                f"lineages {sorted(train_lineages & test_lineages)} appear on both sides "
                "of the train/test boundary"
            )
            assert len(train_idx) + len(test_idx) == len(paths)

        assert n_splits_seen == 3

    def test_every_sample_is_tested_exactly_once(self, lineage_cohort):
        paths, _ = lineage_cohort
        splitter = LineageKFold(n_splits=3, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        tested = [i for _, test_idx in splitter.split(paths) for i in test_idx]

        assert sorted(tested) == list(range(len(paths)))

    def test_random_kfold_leaks_where_lineage_kfold_does_not(self, lineage_cohort):
        """The comparison that makes the point: on the same cohort, plain
        `KFold(shuffle=True)` puts clonal relatives on both sides of the
        boundary -- which is what inflates published genomic-ML numbers --
        while `LineageKFold` does not.
        """
        from sklearn.model_selection import KFold

        paths, true_groups = lineage_cohort

        leaked = False
        for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=0).split(paths):
            if {true_groups[i] for i in train_idx} & {true_groups[i] for i in test_idx}:
                leaked = True
        assert leaked, "fixture no longer exhibits the leakage random CV is supposed to cause"

    def test_get_n_splits_reports_the_configured_number(self, lineage_cohort):
        paths, _ = lineage_cohort
        splitter = LineageKFold(n_splits=3, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        assert splitter.get_n_splits() == 3

    def test_groups_are_computed_once_and_cached(self, lineage_cohort):
        paths, _ = lineage_cohort
        splitter = LineageKFold(n_splits=3, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        first = splitter.groups_
        second = splitter.groups_

        assert first is second

    def test_n_splits_greater_than_n_groups_raises_actionable_error(self, lineage_cohort):
        """12 samples but only 3 lineages: a 4-fold split is impossible
        without breaking a lineage apart, and the caller needs to be told the
        one knob that fixes it.
        """
        paths, _ = lineage_cohort
        splitter = LineageKFold(n_splits=4, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD)

        with pytest.raises(ValueError) as exc_info:
            list(splitter.split(paths))

        message = str(exc_info.value)
        assert "distance_threshold" in message
        assert "4" in message and "3" in message

    def test_precomputed_groups_are_validated_eagerly(self):
        """When groups are handed in there is nothing to sketch, so the
        n_splits check can and should fail at construction rather than
        waiting for the first split().
        """
        with pytest.raises(ValueError) as exc_info:
            LineageKFold(n_splits=4, groups=[0, 0, 1, 1, 2, 2])

        assert "4" in str(exc_info.value) and "3" in str(exc_info.value)

    def test_the_remedy_offered_depends_on_where_the_groups_came_from(self, lineage_cohort):
        """The advice has to name a knob the caller can actually turn.

        When `groups=` is supplied, `distance_threshold` was never consulted:
        the grouping came in finished. Telling that caller to "raise
        distance_threshold" sends them to tune a parameter with no effect on
        their problem -- a message that sounds actionable and is not. This
        pins both halves: derived groupings get the threshold advice, supplied
        ones are told explicitly that it does not apply.
        """
        paths, _ = lineage_cohort
        derived = LineageKFold(
            n_splits=4, paths=paths, k=_K, sketch_size=_SKETCH_SIZE, distance_threshold=_THRESHOLD
        )
        with pytest.raises(ValueError) as derived_exc:
            list(derived.split(paths))
        derived_message = str(derived_exc.value)
        assert "raise distance_threshold" in derived_message

        with pytest.raises(ValueError) as supplied_exc:
            LineageKFold(n_splits=4, groups=[0, 0, 1, 1, 2, 2])
        supplied_message = str(supplied_exc.value)
        assert "supply coarser groups=" in supplied_message
        assert "not involved here" in supplied_message
        assert "raise distance_threshold" not in supplied_message

    def test_accepts_precomputed_groups_without_sketching(self):
        groups = [0, 0, 1, 1, 2, 2]
        splitter = LineageKFold(n_splits=3, groups=groups)

        X = list(range(6))
        for train_idx, test_idx in splitter.split(X):
            train_groups = {groups[i] for i in train_idx}
            test_groups = {groups[i] for i in test_idx}
            assert not (train_groups & test_groups)

    def test_requires_either_paths_or_groups(self):
        with pytest.raises(ValueError) as exc_info:
            LineageKFold(n_splits=3)

        message = str(exc_info.value)
        assert "paths" in message and "groups" in message

    def test_rejects_both_paths_and_groups(self, lineage_cohort):
        paths, true_groups = lineage_cohort

        with pytest.raises(ValueError):
            LineageKFold(n_splits=3, paths=paths, groups=true_groups)

    def test_split_rejects_an_x_of_the_wrong_length(self):
        splitter = LineageKFold(n_splits=3, groups=[0, 0, 1, 1, 2, 2])

        with pytest.raises(ValueError) as exc_info:
            list(splitter.split(list(range(5))))

        assert "5" in str(exc_info.value) and "6" in str(exc_info.value)

    def test_n_splits_below_two_raises(self):
        with pytest.raises(ValueError):
            LineageKFold(n_splits=1, groups=[0, 0, 1, 1])

    def test_usable_as_the_cv_argument_of_cross_val_score(self):
        """The API-compatibility claim: `LineageKFold` has to be droppable
        into scikit-learn's own machinery, not just callable by hand.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(3), 8)
        # Both classes present inside every lineage: a phenotype perfectly
        # confounded with lineage would leave some training fold with a
        # single class, which is a property of the *data*, not of the
        # splitter, and would make this test about the wrong thing.
        y = np.tile([0, 1], 12)
        X = rng.normal(size=(24, 3))
        X[:, 0] += y * 4.0

        scores = cross_val_score(
            LogisticRegression(max_iter=1000),
            X,
            y,
            cv=LineageKFold(n_splits=3, groups=groups),
        )

        assert len(scores) == 3


class TestPermutationImportancePvalues:
    @staticmethod
    def _informative_dataset(n=80, seed=0):
        """Feature 0 carries the label; features 1-3 are pure noise."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int)
        return X, y, ["kmer_informative", "kmer_noise_1", "kmer_noise_2", "kmer_noise_3"]

    def test_informative_feature_gets_a_low_p_value_and_noise_features_do_not(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset()

        table = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=30, random_state=0
        )

        p_values = dict(zip(table.column("feature").to_pylist(), table.column("p_value").to_pylist()))
        assert p_values["kmer_informative"] < 0.05
        for noise in ("kmer_noise_1", "kmer_noise_2", "kmer_noise_3"):
            assert p_values[noise] > 0.1

    def test_returns_an_arrow_table_in_feature_name_order(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset()

        table = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=5, random_state=0
        )

        assert isinstance(table, pa.Table)
        assert table.column_names == ["feature", "importance", "p_value"]
        assert table.column("feature").to_pylist() == names

    def test_p_values_respect_the_add_one_floor(self):
        """With `n_permutations` permutations the smallest attainable p-value
        is `1 / (n_permutations + 1)` -- the Phipson & Smyth (2010) add-one
        estimator, which never reports an impossible p = 0.
        """
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset()

        table = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=10, random_state=0
        )

        p_values = table.column("p_value").to_pylist()
        assert min(p_values) >= 1.0 / 11.0
        assert max(p_values) <= 1.0

    def test_random_state_makes_the_result_reproducible(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset()

        first = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=10, random_state=7
        )
        second = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=10, random_state=7
        )

        assert first.column("p_value").to_pylist() == second.column("p_value").to_pylist()

    def test_works_with_a_tree_models_feature_importances(self):
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = self._informative_dataset(n=60)

        table = permutation_importance_pvalues(
            RandomForestClassifier(n_estimators=10, random_state=0),
            X,
            y,
            names,
            n_permutations=10,
            random_state=0,
        )

        p_values = dict(zip(table.column("feature").to_pylist(), table.column("p_value").to_pylist()))
        assert p_values["kmer_informative"] < 0.2

    def test_groups_restrict_permutation_to_within_lineage(self):
        """With `groups=`, labels are permuted *within* each lineage, so the
        null keeps the lineage/phenotype association intact. When the
        phenotype is perfectly confounded with lineage -- every member of a
        lineage sharing one label -- no within-lineage permutation can change
        `y` at all, and every p-value must come back at the maximum. That is
        the honest answer: such a design carries no evidence that any feature
        is associated with the phenotype rather than with the lineage.
        """
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(4), 10)
        y = (groups % 2).astype(int)
        X = rng.normal(size=(40, 3))
        X[:, 0] += y * 3.0
        names = ["kmer_confounded", "kmer_noise_1", "kmer_noise_2"]

        table = permutation_importance_pvalues(
            LogisticRegression(max_iter=1000), X, y, names, n_permutations=10, random_state=0, groups=groups
        )

        assert table.column("p_value").to_pylist() == [1.0, 1.0, 1.0]

    def test_feature_name_length_mismatch_raises(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset()

        with pytest.raises(ValueError) as exc_info:
            permutation_importance_pvalues(
                LogisticRegression(max_iter=1000), X, y, names[:2], n_permutations=2, random_state=0
            )

        assert "4" in str(exc_info.value) and "2" in str(exc_info.value)

    def test_a_model_with_no_importance_attribute_raises_actionably(self):
        from sklearn.neighbors import KNeighborsClassifier

        X, y, names = self._informative_dataset(n=40)

        with pytest.raises(ValueError) as exc_info:
            permutation_importance_pvalues(
                KNeighborsClassifier(n_neighbors=3), X, y, names, n_permutations=2, random_state=0
            )

        message = str(exc_info.value)
        assert "coef_" in message and "feature_importances_" in message

    def test_non_positive_n_permutations_raises(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset(n=40)

        with pytest.raises(ValueError):
            permutation_importance_pvalues(
                LogisticRegression(max_iter=1000), X, y, names, n_permutations=0, random_state=0
            )

    def test_groups_length_mismatch_raises(self):
        from sklearn.linear_model import LogisticRegression

        X, y, names = self._informative_dataset(n=40)

        with pytest.raises(ValueError):
            permutation_importance_pvalues(
                LogisticRegression(max_iter=1000), X, y, names, n_permutations=2, random_state=0, groups=[0, 1]
            )
