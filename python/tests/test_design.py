"""Tests for fastdna.design: feasibility checks computed before an experiment.

The headline test is `test_auc_standard_error_matches_a_monte_carlo_estimate`,
which checks the Hanley-McNeil formula against simulation rather than against
a number transcribed from the paper. Transcribing a published constant tests
that the transcription is right; simulating tests that the formula means what
it is being used to mean.
"""
from __future__ import annotations

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from fastdna.design import (  # noqa: E402
    DesignConcern,
    DesignReport,
    auc_standard_error,
    check_design,
)


def _codes(report):
    return sorted(c.code for c in report.concerns)


class TestAucStandardError:
    def test_auc_standard_error_matches_a_monte_carlo_estimate(self):
        """External check on the formula, by simulation.

        Draws scores for two classes from distributions whose true AUC is
        known analytically, computes the empirical AUC many times, and
        compares the spread of those estimates against what
        `auc_standard_error` predicts for that sample size.

        The construction: for two normal distributions separated by `d` with
        unit variance, the true AUC is `Phi(d / sqrt(2))`. Choosing `d` to
        hit a target AUC makes the ground truth exact rather than estimated.

        Hanley-McNeil's exponential approximation is deliberately
        conservative (it assumes negative-exponential score distributions,
        which have heavier tails than the normals simulated here), so the
        formula is expected to sit at or slightly above the empirical
        spread. The assertion allows that direction generously and still
        catches a formula that is wrong by a factor.
        """
        from scipy.stats import norm
        from sklearn.metrics import roc_auc_score

        target_auc = 0.80
        n_pos = n_neg = 60
        separation = norm.ppf(target_auc) * np.sqrt(2.0)

        rng = np.random.default_rng(20260901)
        empirical = []
        for _ in range(400):
            positives = rng.normal(separation, 1.0, n_pos)
            negatives = rng.normal(0.0, 1.0, n_neg)
            y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
            scores = np.concatenate([positives, negatives])
            empirical.append(roc_auc_score(y, scores))

        empirical_se = float(np.std(empirical, ddof=1))
        predicted_se = auc_standard_error(target_auc, n_pos, n_neg)

        assert predicted_se == pytest.approx(empirical_se, rel=0.45), (
            f"formula predicts SE={predicted_se:.4f}, simulation gives "
            f"{empirical_se:.4f}"
        )
        # And in the conservative direction, which is the one that matters for
        # a feasibility check: it must not tell you an under-powered design is
        # adequate.
        assert predicted_se >= empirical_se * 0.9

    def test_standard_error_shrinks_with_sample_size(self):
        """The property the whole module rests on: more samples, tighter
        estimate, falling as 1/sqrt(n)."""
        small = auc_standard_error(0.75, 50, 50)
        large = auc_standard_error(0.75, 200, 200)
        assert large < small
        assert small / large == pytest.approx(2.0, rel=0.15)  # 4x n -> 2x tighter

    def test_an_empty_class_has_no_defined_standard_error(self):
        assert np.isnan(auc_standard_error(0.75, 0, 50))
        assert np.isnan(auc_standard_error(0.75, 50, 0))


class TestCheckDesign:
    def test_it_diagnoses_the_design_that_actually_failed(self):
        """REGRESSION, against a real failure. `docs/validation-real-data.md`
        records three attempts at a leakage demonstration; the ampicillin one
        fit 5,000 features to 80 samples and learned nothing. That was
        knowable from the shape alone, twenty minutes of counting and fitting
        before it was discovered.
        """
        phenotype = np.array([1] * 58 + [0] * 22)
        groups = np.array([i % 38 for i in range(80)])

        report = check_design(phenotype, n_features=5000, groups=groups, n_splits=5)

        assert isinstance(report, DesignReport)
        assert report.p_over_n == pytest.approx(62.5)
        assert "high_p_over_n" in _codes(report)
        # And the interval it could resolve was wider than the effect it
        # eventually reported (+0.075), which is the number that would have
        # saved the run.
        assert report.auc_ci_halfwidth > 0.075

    def test_a_well_shaped_design_raises_nothing(self):
        phenotype = np.array([1] * 250 + [0] * 250)
        groups = np.array([i % 60 for i in range(500)])

        report = check_design(phenotype, n_features=500, groups=groups, n_splits=5)

        assert report.concerns == ()
        assert report.p_over_n == pytest.approx(1.0)
        assert report.minority_count == 250

    def test_more_folds_than_groups_is_flagged_before_the_splitter_raises(self):
        """`cv.LineageKFold` already refuses this at construction. Catching it
        here is cheaper still: no sketching has happened yet, so the caller
        learns it while editing the plan rather than mid-run."""
        phenotype = np.array([0, 1] * 20)
        groups = np.array([i % 3 for i in range(40)])

        report = check_design(phenotype, n_features=10, groups=groups, n_splits=5)

        assert "fewer_groups_than_folds" in _codes(report)

    def test_a_dominant_lineage_is_flagged(self):
        """One group holding most of the cohort means a grouped fold is
        essentially that group, and its training set has none of it."""
        phenotype = np.array([0, 1] * 50)
        groups = np.array([0] * 60 + list(range(1, 41)))

        report = check_design(phenotype, n_features=10, groups=groups, n_splits=5)

        assert "dominant_group" in _codes(report)
        assert report.largest_group_fraction == pytest.approx(0.6)

    def test_severe_imbalance_points_at_the_module_that_handles_it(self):
        phenotype = np.array([1] * 5 + [0] * 95)

        report = check_design(phenotype, n_features=10, n_splits=5)

        codes = _codes(report)
        assert "class_imbalance" in codes
        assert "tiny_minority_class" in codes
        remedy = next(c.remedy for c in report.concerns if c.code == "class_imbalance")
        assert "evaluation" in remedy

    def test_a_cohort_too_small_to_resolve_the_effect_says_so(self):
        """The check that matters most and is least often made: whether the
        confidence interval on the hoped-for effect reaches chance."""
        phenotype = np.array([1] * 6 + [0] * 6)

        report = check_design(phenotype, n_features=2, n_splits=2, assumed_auc=0.75)

        assert "auc_ci_wider_than_effect" in _codes(report)

    def test_a_continuous_phenotype_reports_nan_rather_than_guessing(self):
        """Class balance and an AUC interval are undefined for a regression
        target. Reported as nan, not silently binned into a fake binary --
        the same refusal `audit.Confounding` makes for its own statistic."""
        phenotype = np.linspace(0.25, 64.0, 40)

        # 1,000 features over 40 samples: p/n = 25, comfortably past the
        # threshold, so the learnability check still fires on a target whose
        # class balance is undefined.
        report = check_design(phenotype, n_features=1000, n_splits=5)

        assert np.isnan(report.minority_fraction)
        assert np.isnan(report.auc_ci_halfwidth)
        assert "high_p_over_n" in _codes(report)  # still checkable

    def test_features_are_optional_and_their_absence_is_visible(self):
        phenotype = np.array([0, 1] * 40)
        report = check_design(phenotype, n_splits=5)
        assert np.isnan(report.p_over_n)
        assert "high_p_over_n" not in _codes(report)

    def test_groups_of_the_wrong_length_are_rejected(self):
        with pytest.raises(ValueError, match="same cohort"):
            check_design(np.array([0, 1, 0, 1]), groups=np.array([0, 1]))

    def test_degenerate_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            check_design(np.array([1]))
        with pytest.raises(ValueError, match="n_splits"):
            check_design(np.array([0, 1] * 10), n_splits=1)

    def test_the_report_renders_without_a_verdict(self):
        """Same posture as AuditReport: numbers and named concerns, no
        PASS/FAIL anywhere in the output."""
        phenotype = np.array([1] * 58 + [0] * 22)
        text = check_design(phenotype, n_features=5000, n_splits=5).to_markdown()

        assert "p/n" in text and "62.5" in text
        for verdict in ("PASS", "FAIL", "GOOD", "BAD", "INVALID"):
            assert verdict not in text
