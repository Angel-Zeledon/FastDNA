"""Tests for `fastdna.workflow.AssociationWorkflow` -- the single entry
point chaining `gwas`, `equivalence`, `cv`, a classifier, `evaluation`,
`gwas.prefilter_association`, `annotate` and `plotting`.

The synthetic cohort mirrors `test_cv.py`'s `lineage_cohort` fixture (three
unrelated 600 bp base sequences, each mutated at a 1% per-base rate into
four samples -- known-good parameters for `k=21`/`sketch_size=200`/
`distance_threshold=0.05` lineage recovery), plus one addition: within each
lineage, exactly two of the four samples also carry a homopolymer "marker"
read (`"G" * 25`), whose canonical 21-mer is `"C" * 21` (`'C' < 'G'`
byte-wise, matching `test_gwas.py`'s own `GGGGG -> CCCCC` example).

This construction makes the marker k-mer the *only* feature that can ever
satisfy `SetCoveringClassifier`'s "holds for every positive sample" rule:
every other informative k-mer comes from one lineage's own random
background sequence and therefore cannot be present in positives drawn
from a *different* lineage. Phenotype is assigned two-positive/two-negative
within every lineage, so it is deliberately uncorrelated with lineage
identity -- the workflow's classifier and its leave-one-lineage-out
cross-validation are both expected to recover the marker rule, exactly and
deterministically, not just "well".
"""

from __future__ import annotations

import pathlib
import random

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

import pyarrow as pa
import pyarrow.compute as pc

from fastdna.equivalence import EquivalenceClasses
from fastdna.evaluation import CalibrationReport, PrecisionRecallReport
from fastdna.gwas import ScreeningOnlyWarning, cohort_presence_matrix
from fastdna.workflow import AssociationResult, AssociationWorkflow

K = 21
SKETCH_SIZE = 200
DISTANCE_THRESHOLD = 0.05  # matches test_cv.py's own proven-good value for this fixture shape
MARKER_READ = "G" * 25
MARKER_KMER = "C" * K  # canonical form of the all-G 21-mer window


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


@pytest.fixture
def cohort(tmp_path):
    """Three lineages of four samples each; within every lineage, samples 0
    and 1 are phenotype-positive (carry `MARKER_READ`), samples 2 and 3 are
    negative. Returns `(paths, phenotype, sample_ids, lineage_of)`:
    `paths`/`phenotype` are aligned lists in generation order,
    `sample_ids[i]` is the id `gwas._resolve_cohort` derives from
    `paths[i]`'s file name, `lineage_of[i]` the true lineage index.
    """
    rng = random.Random(20260825)
    bases = [_random_seq(rng, 600) for _ in range(3)]

    paths, phenotype, sample_ids, lineage_of = [], [], [], []
    for lineage, base in enumerate(bases):
        for i in range(4):
            background = _mutate(rng, base, rate=0.01)
            is_positive = i < 2
            reads = [background] * 3 + ([MARKER_READ] * 3 if is_positive else [])
            name = f"lin{lineage}_s{i}.fastq"
            paths.append(str(write_fastq(tmp_path, name, reads)))
            phenotype.append(1 if is_positive else 0)
            sample_ids.append(f"lin{lineage}_s{i}")
            lineage_of.append(lineage)
    return paths, np.array(phenotype), sample_ids, lineage_of


def _workflow(paths, phenotype, **kwargs):
    kwargs.setdefault("matrix_kwargs", {"k": K})
    kwargs.setdefault("lineage_kwargs", {"k": K, "sketch_size": SKETCH_SIZE, "distance_threshold": DISTANCE_THRESHOLD})
    kwargs.setdefault("n_splits", 3)
    return AssociationWorkflow(paths, phenotype, **kwargs)


# ---------------------------------------------------------------------------
# The default pipeline
# ---------------------------------------------------------------------------


class TestDefaultPipeline:
    def test_returns_an_association_result(self, cohort):
        paths, phenotype, sample_ids, _ = cohort
        result = _workflow(paths, phenotype).run()

        assert isinstance(result, AssociationResult)
        assert result.matrix.shape[0] == 12
        assert result.sample_ids == sample_ids
        assert len(result.kmer_sequences) == result.matrix.shape[1]

    def test_run_caches_result_on_the_workflow(self, cohort):
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype)
        assert workflow.result_ is None

        result = workflow.run()
        assert workflow.result_ is result

    def test_lineage_groups_recover_the_three_true_lineages(self, cohort):
        paths, phenotype, _, lineage_of = cohort
        result = _workflow(paths, phenotype).run()

        assert len(set(result.groups.tolist())) == 3
        # Same partition as the true lineages, independent of which integer
        # label each group happened to receive.
        by_group = {}
        for i, g in enumerate(result.groups.tolist()):
            by_group.setdefault(g, set()).add(lineage_of[i])
        assert all(len(true_lineages) == 1 for true_lineages in by_group.values())

    def test_classifier_learns_the_marker_rule(self, cohort):
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype).run()

        rule_names = [rule.feature_name for rule in result.classifier.rules_]
        assert MARKER_KMER in rule_names
        marker_rule = next(r for r in result.classifier.rules_ if r.feature_name == MARKER_KMER)
        assert marker_rule.presence is True

    def test_held_out_cross_validation_recovers_the_marker_perfectly(self, cohort):
        paths, phenotype, sample_ids, _ = cohort
        result = _workflow(paths, phenotype).run()

        assert isinstance(result.cv_predictions, pa.Table)
        assert result.cv_predictions.column("sample_id").to_pylist() == sample_ids
        assert result.cv_predictions.column("y_true").to_pylist() == phenotype.tolist()

        cv_score = np.asarray(result.cv_predictions.column("cv_score").to_pylist())
        # Leave-one-lineage-out: every held-out fold's training set still
        # spans the other two lineages, in which the marker is still the
        # only feature present in 100% of the training positives -- so the
        # held-out fold's predictions should be exactly as clean as the
        # in-sample fit.
        assert np.array_equal(cv_score, phenotype.astype(np.float64))

        assert isinstance(result.precision_recall, PrecisionRecallReport)
        assert result.precision_recall.average_precision == pytest.approx(1.0)
        assert isinstance(result.calibration, CalibrationReport)

    def test_screening_runs_by_default_and_warns(self, cohort):
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype)

        with pytest.warns(ScreeningOnlyWarning):
            result = workflow.run()

        assert isinstance(result.screening, pa.Table)
        for column in ("kmer_sequence", "p_value", "p_bonferroni", "q_value_bh"):
            assert column in result.screening.column_names

    def test_importance_and_annotations_are_none_by_default(self, cohort):
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype).run()

        assert result.importance is None
        assert result.annotations is None


# ---------------------------------------------------------------------------
# Skippable / overridable stages
# ---------------------------------------------------------------------------


class TestSkippableStages:
    def test_collapse_equivalence_false_keeps_raw_columns(self, cohort):
        paths, phenotype, _, _ = cohort
        raw_matrix, _, raw_kmers = cohort_presence_matrix(paths, k=K)

        result = _workflow(paths, phenotype, collapse_equivalence=False).run()

        assert result.equivalence is None
        assert result.kmer_sequences == raw_kmers
        assert result.matrix.shape[1] == raw_matrix.shape[1]

    def test_collapse_equivalence_true_never_widens_the_matrix(self, cohort):
        paths, phenotype, _, _ = cohort
        raw_matrix, _, _ = cohort_presence_matrix(paths, k=K)

        result = _workflow(paths, phenotype, collapse_equivalence=True).run()

        assert isinstance(result.equivalence, EquivalenceClasses)
        assert result.matrix.shape[1] <= raw_matrix.shape[1]
        assert MARKER_KMER in result.kmer_sequences

    def test_cv_false_skips_cross_validated_evaluation_but_still_fits(self, cohort):
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype, cv=False).run()

        assert result.cv_predictions is None
        assert result.precision_recall is None
        assert result.calibration is None
        assert MARKER_KMER in [r.feature_name for r in result.classifier.rules_]

    def test_screen_false_skips_screening(self, cohort):
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype, screen=False).run()
        assert result.screening is None

    def test_precomputed_groups_are_used_instead_of_lineage_groups(self, cohort):
        paths, phenotype, _, lineage_of = cohort
        result = _workflow(paths, phenotype, groups=lineage_of).run()
        assert result.groups.tolist() == lineage_of

    def test_precomputed_groups_length_mismatch_raises(self, cohort):
        paths, phenotype, _, _ = cohort
        with pytest.raises(ValueError, match="groups has"):
            _workflow(paths, phenotype, groups=[0, 1, 2]).run()


# ---------------------------------------------------------------------------
# Phenotype alignment
# ---------------------------------------------------------------------------


class TestPhenotypeAlignment:
    def test_phenotype_as_mapping_is_matched_by_id_not_position(self, cohort):
        paths, phenotype, sample_ids, _ = cohort
        # Deliberately scrambled relative to `paths`' order, to prove the
        # mapping is matched by id rather than positionally re-zipped.
        scrambled = dict(zip(reversed(sample_ids), reversed(phenotype.tolist())))

        result = _workflow(paths, scrambled).run()

        assert result.cv_predictions.column("y_true").to_pylist() == phenotype.tolist()

    def test_phenotype_mapping_missing_a_sample_raises(self, cohort):
        paths, phenotype, sample_ids, _ = cohort
        incomplete = dict(zip(sample_ids[:-1], phenotype.tolist()[:-1]))
        with pytest.raises(ValueError, match="missing a value"):
            _workflow(paths, incomplete).run()

    def test_phenotype_array_length_mismatch_raises(self, cohort):
        paths, _, _, _ = cohort
        with pytest.raises(ValueError, match="phenotype has shape"):
            _workflow(paths, [0, 1, 0]).run()


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def write_reference(tmp_path):
    rng = random.Random(7)
    flank_a = _random_seq(rng, 50)
    flank_b = _random_seq(rng, 50)
    sequence = flank_a + ("G" * K) + flank_b
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(f">chr1 test reference\n{sequence}\n")

    start = len(flank_a) + 1  # 1-based inclusive
    end = start + K - 1
    gff = tmp_path / "reference.gff3"
    gff.write_text(
        "##gff-version 3\n"
        f"chr1\ttest\tgene\t{start}\t{end}\t.\t+\t.\tID=gene1;Name=markerGene\n"
    )
    return fasta, gff


class TestAnnotation:
    def test_reference_and_annotation_locate_the_marker_rule(self, cohort):
        paths, phenotype, _, _ = cohort
        reference_fasta, annotation_path = write_reference(pathlib.Path(paths[0]).parent)

        result = _workflow(
            paths, phenotype, reference_fasta=reference_fasta, annotation_path=annotation_path
        ).run()

        assert isinstance(result.annotations, pa.Table)
        assert result.annotations.num_rows > 0
        gene_names = set(result.annotations.column("gene_name").to_pylist())
        assert "markerGene" in gene_names
        marker_rows = result.annotations.filter(pc.field("kmer_sequence") == MARKER_KMER)
        assert marker_rows.num_rows > 0

    def test_reference_without_annotation_raises(self, cohort):
        paths, phenotype, _, _ = cohort
        with pytest.raises(ValueError, match="must be given together"):
            _workflow(paths, phenotype, reference_fasta="reference.fasta").run()

    def test_annotation_with_a_classifier_lacking_rules_raises(self, cohort):
        sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
        paths, phenotype, _, _ = cohort
        reference_fasta, annotation_path = write_reference(pathlib.Path(paths[0]).parent)

        workflow = _workflow(
            paths,
            phenotype,
            classifier=sklearn_linear_model.LogisticRegression(max_iter=1000),
            reference_fasta=reference_fasta,
            annotation_path=annotation_path,
        )
        with pytest.raises(ValueError, match="rules_"):
            workflow.run()


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


class TestPermutationImportance:
    def test_n_permutations_with_a_coefficient_model_runs(self, cohort):
        sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
        paths, phenotype, _, _ = cohort

        result = _workflow(
            paths,
            phenotype,
            classifier=sklearn_linear_model.LogisticRegression(max_iter=1000),
            n_permutations=5,
            permutation_random_state=0,
        ).run()

        assert isinstance(result.importance, pa.Table)
        assert result.importance.num_rows == len(result.kmer_sequences)
        for column in ("feature", "importance", "p_value"):
            assert column in result.importance.column_names

    def test_n_permutations_with_the_default_classifier_raises(self, cohort):
        paths, phenotype, _, _ = cohort
        with pytest.raises(ValueError, match="coef_"):
            _workflow(paths, phenotype, n_permutations=5).run()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_plot_significance_needs_run_first(self, cohort):
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype)
        with pytest.raises(ValueError, match="needs run"):
            workflow.plot_significance()

    def test_plot_significance_needs_screening(self, cohort):
        pytest.importorskip("matplotlib")
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype, screen=False)
        workflow.run()
        with pytest.raises(ValueError, match="screen=False"):
            workflow.plot_significance()

    def test_plot_significance_draws_onto_axes(self, cohort):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype)
        workflow.run()

        ax = workflow.plot_significance()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_population_structure_draws_onto_axes_and_caches_the_distance_matrix(self, cohort):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        paths, phenotype, _, _ = cohort
        workflow = _workflow(paths, phenotype)
        workflow.run()

        ax = workflow.plot_population_structure()
        assert isinstance(ax, matplotlib.axes.Axes)
        assert workflow._distance_matrix is not None

        cached = workflow._distance_matrix
        workflow.plot_population_structure()
        assert workflow._distance_matrix is cached


# ---------------------------------------------------------------------------
# HTML report export
# ---------------------------------------------------------------------------


class TestToReport:
    def test_to_report_bundles_this_results_own_classifier_and_calibration(self, cohort, tmp_path):
        pytest.importorskip("matplotlib")
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype).run()

        out = tmp_path / "report.html"
        result.to_report(out)

        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert result.classifier.explain() in text
        assert "data:image/png;base64," in text  # result.calibration was not None (cv=True by default)

    def test_to_report_auto_fills_sample_and_feature_counts(self, cohort, tmp_path):
        pytest.importorskip("matplotlib")
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype).run()

        out = tmp_path / "report.html"
        result.to_report(out)

        text = out.read_text(encoding="utf-8")
        assert str(len(result.sample_ids)) in text
        assert str(len(result.kmer_sequences)) in text

    def test_to_report_with_cv_false_has_no_calibration_but_still_writes(self, cohort, tmp_path):
        pytest.importorskip("matplotlib")
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype, cv=False).run()
        assert result.calibration is None

        out = tmp_path / "report.html"
        result.to_report(out)

        text = out.read_text(encoding="utf-8")
        assert "No calibration report or interval was provided" in text
        assert "data:image/png;base64," not in text

    def test_to_report_forwards_calibration_interval_and_extra_kwargs(self, cohort, tmp_path):
        np_ = pytest.importorskip("numpy")
        pytest.importorskip("matplotlib")
        paths, phenotype, _, _ = cohort
        result = _workflow(paths, phenotype).run()

        out = tmp_path / "report.html"
        result.to_report(
            out,
            calibration_interval=(np_.array([0.1]), np_.array([0.3])),
            title="Custom Title",
            metadata={"n_samples": "overridden"},
        )

        text = out.read_text(encoding="utf-8")
        assert "Custom Title" in text
        assert "overridden" in text
