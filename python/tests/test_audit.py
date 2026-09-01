"""Tests for fastdna.audit(): the random-CV vs. lineage-blocked-CV gap.

Two real-FASTQ positive/negative controls anchor the headline claim (the
same fixture style `test_cv.py`/`test_explain.py` use, for the same
reason: the whole point of `audit()` is that it derives the comparison
from the genomes themselves, so a test that skipped sketching would not
test that claim). Everything else -- input validation, the `covariates=`
path, the report's own formatting -- uses a precomputed feature matrix and
explicit `groups=` instead, which `audit()` supports directly (see its own
docstring) and which keeps the rest of the suite fast.
"""
from __future__ import annotations

import pathlib
import random
import warnings

import pytest

import fastdna

# Guarded, and in this order, on purpose -- see
# test_optional_dependencies.py::test_the_test_suite_itself_collects_in_the_environment_ci_builds,
# which exists specifically to catch a module reintroducing an unguarded
# `import numpy` above these skips.
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from fastdna.audit import (  # noqa: E402
    AuditReport,
    Confounding,
    CovariateAudit,
    DegenerateLineagesWarning,
    audit,
)
from fastdna.explain import explain  # noqa: E402
from fastdna.sklearn import KmerVectorizer  # noqa: E402


def test_fastdna_audit_survives_repeated_top_level_access():
    """`fastdna.audit` (and `fastdna.explain`) are re-exported at the top
    level via a lazy `fastdna.__getattr__`, specifically so a bare `import
    fastdna` never needs numpy (see `fastdna/__init__.py`'s own comment).
    A first version of that `__getattr__` returned the right function on
    the FIRST access but silently shadowed itself with the `audit`
    *submodule* (not callable) on every access after that, because `from
    .audit import audit` implicitly binds the submodule onto the package
    as an import-machinery side effect, and regular attribute lookup finds
    that binding before `__getattr__` is ever consulted again. This
    exercises the fix: every access must keep returning the same callable,
    not the submodule, and `fastdna.audit is fastdna.audit` must hold
    (bypassing `__getattr__` entirely once cached).
    """
    import fastdna as _fastdna_top_level

    first = _fastdna_top_level.audit
    second = _fastdna_top_level.audit
    third = _fastdna_top_level.explain

    assert first is audit
    assert second is audit
    assert first is second
    assert callable(_fastdna_top_level.audit)
    assert third is explain
    assert callable(_fastdna_top_level.explain)


def _write(tmp_path, name, reads):
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return str(p)


def _lineage_cohort(tmp_path, n_lineages=4, per_lineage=6, read_len=120, seed=0):
    """A cohort with real lineage structure: each lineage is a distinct
    random root sequence, and its members are sparsely point-mutated
    copies of it, so within-lineage Mash distance stays small and
    between-lineage distance stays large. Adapted from
    `test_explain.py`'s `_lineage_cohort` fixture. Returns
    `(paths, lineage_of)`.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    paths, lineage_of = [], []

    for lineage in range(n_lineages):
        root = "".join(rng.choice(bases, size=read_len * 4))
        for member in range(per_lineage):
            seq = list(root)
            for pos in rng.choice(len(seq), size=max(1, len(seq) // 200), replace=False):
                seq[pos] = str(rng.choice(bases))
            seq = "".join(seq)
            reads = [seq[i : i + read_len] for i in range(0, len(seq) - read_len, read_len // 2)]
            path = _write(tmp_path, f"L{lineage}_S{member}.fastq", reads)
            paths.append(path)
            lineage_of.append(lineage)

    return paths, np.array(lineage_of)


def _pipeline(k=21, top_features=200):
    return Pipeline(
        [
            ("kmers", KmerVectorizer(k=k, top_features=top_features, representation="presence")),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


# ---------------------------------------------------------------------------
# Positive/negative controls over real FASTQ, real sketching (slow-ish).
# ---------------------------------------------------------------------------


def test_audit_reports_a_real_gap_when_lineage_confounds_phenotype(tmp_path):
    """POSITIVE CONTROL. Phenotype is assigned entirely by lineage (every
    member of lineages 0/1 is a "case", every member of lineages 2/3 a
    "control"): the cohort cannot distinguish phenotype from lineage.
    Random CV can leak near-identical training relatives into the test
    fold and score high; lineage-blocked CV holds out entire lineages
    whose root sequence -- and therefore vocabulary -- the model never saw
    during training, so it should fall to roughly chance. The gap between
    the two must be large and positive.
    """
    paths, lineage_of = _lineage_cohort(tmp_path, n_lineages=4, per_lineage=6, seed=10)
    phenotype = np.array([1 if lineage in (0, 1) else 0 for lineage in lineage_of])

    report = audit(
        _pipeline(),
        paths,
        phenotype,
        n_splits=3,
        sketch_size=200,
        lineage_threshold=0.02,
        random_state=0,
    )

    assert isinstance(report, AuditReport)
    assert report.n_samples == len(paths)
    assert report.n_lineages == 4
    assert report.scoring == "roc_auc"
    # Random CV should look substantially better than lineage-blocked CV.
    # Measured on this fixture/seed: score_random=0.81, score_lineage=0.50,
    # gap=+0.31 -- some lineage-blocked folds are themselves NaN (a held-out
    # fold that is an entire, phenotype-pure lineage has an undefined
    # ROC-AUC; see AuditReport.per_fold's own docstring) and excluded from
    # the nanmean, which is expected for this deliberately fully-confounded
    # fixture, not a bug.
    assert report.score_random > 0.7, f"expected a clearly-above-chance random-CV AUC, got {report.score_random}"
    assert report.gap > 0.2, f"expected a large positive gap, got {report.gap}"

    # per_fold has exactly n_splits rows for each of "random"/"lineage".
    kinds = report.per_fold.column("cv_kind").to_pylist()
    assert kinds.count("random") == 3
    assert kinds.count("lineage") == 3


def test_audit_reports_a_small_gap_without_lineage_confounding(tmp_path):
    """NEGATIVE CONTROL. Phenotype is independent of lineage (assigned by
    an unrelated coin flip per sample), so no split strategy has any real
    signal to leak or to lose -- both CV variants should land close to
    each other (near chance), and the gap should be small in absolute
    value, unlike the positive control above.
    """
    paths, lineage_of = _lineage_cohort(tmp_path, n_lineages=4, per_lineage=6, seed=11)
    pheno_rng = np.random.default_rng(12)
    phenotype = pheno_rng.integers(0, 2, size=len(paths))

    report = audit(
        _pipeline(top_features=50),
        paths,
        phenotype,
        n_splits=3,
        sketch_size=200,
        lineage_threshold=0.02,
        random_state=0,
    )

    assert report.n_lineages == 4
    assert abs(report.gap) < 0.35, f"expected a small gap without confounding, got {report.gap}"


# ---------------------------------------------------------------------------
# Fast path: precomputed matrix + explicit groups=, no real FASTQ/sketching.
# ---------------------------------------------------------------------------


def _synthetic_matrix_cohort(seed=0, n_per_group=8, n_groups=4, n_features=20):
    """A plain numeric `(X, y, groups)` cohort with the same
    lineage-confounding structure as the FASTQ fixtures above, but without
    any sketching: `groups` is supplied directly to `audit()`, so
    `cv.lineage_groups` never runs and `X` never needs to look like a
    FASTQ path. Mirrors `test_cv.py`'s own rationale for testing
    `permutation_importance_pvalues` this way.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * n_groups
    groups = np.repeat(np.arange(n_groups), n_per_group)
    # Each group has its own informative feature block plus shared noise,
    # so a linear classifier can separate groups almost perfectly -- the
    # synthetic stand-in for "each lineage has its own private k-mers".
    X = rng.normal(size=(n_samples, n_features))
    for g in range(n_groups):
        X[groups == g, g % n_features] += 5.0
    return X, groups


def test_audit_accepts_a_precomputed_matrix_with_explicit_groups():
    X, groups = _synthetic_matrix_cohort(seed=1)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])

    # n_splits == n_groups here, so every lineage-blocked fold is an
    # entire, phenotype-pure group -- explicit scoring="accuracy" (defined
    # regardless of how many classes a fold contains) avoids the
    # undefined-ROC-AUC NaN case AuditReport.per_fold's docstring
    # describes, which is not what this test is about.
    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    assert report.n_samples == X.shape[0]
    assert report.n_lineages == 4
    assert np.isnan(report.lineage_threshold), "groups= was supplied directly; no threshold was used"
    assert report.score_random > report.score_lineage


def test_audit_warns_when_almost_every_sample_is_its_own_lineage():
    """Real motivation, not a hypothetical: this project's own bacterial
    AMR reproduction (`scratch/amr_repro/audit_report.json`) hit exactly
    this at the library's default `lineage_threshold` -- 149 of 150
    genomes each their own lineage, `gap` near zero -- and a naive reading
    of that report would have concluded "no leakage" when the real issue
    was that the grouping gave LineageKFold almost nothing to block on.
    """
    n_samples = 20
    X, groups = _synthetic_matrix_cohort(seed=2, n_per_group=1, n_groups=n_samples)
    phenotype = np.array([i % 2 for i in range(n_samples)])

    with pytest.warns(DegenerateLineagesWarning, match=r"20 of 20 samples"):
        report = audit(
            LogisticRegression(max_iter=1000),
            X,
            phenotype,
            groups=groups,
            n_splits=4,
            scoring="accuracy",
        )
    assert report.n_lineages == n_samples


def test_audit_does_not_warn_for_a_genuinely_clustered_cohort():
    X, groups = _synthetic_matrix_cohort(seed=1)  # 4 groups of 8: 4/32, well under the threshold
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateLineagesWarning)
        audit(LogisticRegression(max_iter=1000), X, phenotype, groups=groups, n_splits=4, scoring="accuracy")


def test_audit_covariates_reports_an_independent_gap_per_covariate():
    X, groups = _synthetic_matrix_cohort(seed=2)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])
    # "batch" here is literally the lineage labels again -- a covariate
    # perfectly correlated with lineage -- so its own blocked-CV gap
    # should also be large, exercising the covariates= path end to end.
    covariates = {"batch": groups}

    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        covariates=covariates,
        n_splits=4,
        scoring="accuracy",
    )

    assert len(report.covariates) == 1
    covariate_report = report.covariates[0]
    assert isinstance(covariate_report, CovariateAudit)
    assert covariate_report.name == "batch"
    assert covariate_report.n_groups == 4
    assert covariate_report.gap_vs_random == pytest.approx(
        report.score_random - covariate_report.score
    )

    kinds = report.per_fold.column("cv_kind").to_pylist()
    assert kinds.count("covariate:batch") == 4


def test_audit_rejects_a_non_mapping_covariates():
    X, groups = _synthetic_matrix_cohort(seed=3)
    phenotype = np.array([g % 2 for g in groups])
    with pytest.raises(TypeError, match="covariates"):
        audit(LogisticRegression(), X, phenotype, groups=groups, covariates=[1, 2, 3])


def test_audit_rejects_a_mismatched_covariate_length():
    X, groups = _synthetic_matrix_cohort(seed=4)
    phenotype = np.array([g % 2 for g in groups])
    with pytest.raises(ValueError, match="covariates"):
        audit(LogisticRegression(), X, phenotype, groups=groups, covariates={"batch": [0, 1, 2]})


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_audit_rejects_mismatched_phenotype_length():
    X, groups = _synthetic_matrix_cohort(seed=5)
    with pytest.raises(ValueError, match="phenotype"):
        audit(LogisticRegression(), X, [0, 1, 2], groups=groups)


def test_audit_rejects_mismatched_groups_length():
    X, groups = _synthetic_matrix_cohort(seed=6)
    phenotype = np.array([g % 2 for g in groups])
    with pytest.raises(ValueError, match="groups"):
        audit(LogisticRegression(), X, phenotype, groups=groups[:3])


def test_audit_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2"):
        audit(LogisticRegression(), [[0.0]], [0], groups=[0])


@pytest.mark.parametrize("bad_n_splits", [0, 1, -1, 2.5])
def test_audit_rejects_invalid_n_splits(bad_n_splits):
    X, groups = _synthetic_matrix_cohort(seed=7)
    phenotype = np.array([g % 2 for g in groups])
    with pytest.raises(ValueError, match="n_splits"):
        audit(LogisticRegression(), X, phenotype, groups=groups, n_splits=bad_n_splits)


def test_audit_forwards_n_splits_greater_than_lineages_to_lineagekfold():
    """`audit()` does not duplicate `LineageKFold`'s own n_splits-vs-lineage
    check; its error is allowed to propagate unchanged.
    """
    X, groups = _synthetic_matrix_cohort(seed=8, n_groups=2, n_per_group=6)
    phenotype = np.array([g % 2 for g in groups])
    with pytest.raises(ValueError, match="n_splits"):
        audit(LogisticRegression(), X, phenotype, groups=groups, n_splits=5)


# ---------------------------------------------------------------------------
# Report formatting.
# ---------------------------------------------------------------------------


def test_audit_report_to_markdown_and_repr():
    X, groups = _synthetic_matrix_cohort(seed=9)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])

    report = audit(LogisticRegression(max_iter=1000), X, phenotype, groups=groups, n_splits=4)

    markdown = report.to_markdown()
    assert "FastDNA leakage audit" in markdown
    assert "Gap" in markdown
    assert str(report) == markdown

    text_repr = repr(report)
    assert "AuditReport(" in text_repr
    assert f"n_samples={report.n_samples}" in text_repr


def test_audit_report_html_falls_back_without_pandas(monkeypatch):
    X, groups = _synthetic_matrix_cohort(seed=10)
    phenotype = np.array([g % 2 for g in groups])
    report = audit(LogisticRegression(max_iter=1000), X, phenotype, groups=groups, n_splits=4)

    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    html = report._repr_html_()
    assert html.startswith("<pre>")
    assert "FastDNA leakage audit" in html


# ---------------------------------------------------------------------------
# Phenotype-vs-lineage confounding (AuditReport.confounding).
#
# The positive and negative controls below are what make this number
# meaningful. A statistic that measures "how much of the phenotype is just
# lineage" has to fire when the phenotype IS the lineage, and stay quiet
# when it is independent of it -- without both, a plausible-looking value
# says nothing. They are deliberately built on the same
# `_synthetic_matrix_cohort` groups the rest of this file uses, with
# `groups=` passed explicitly, because `confounding` is computed from the
# two label vectors alone: no model is fitted and no genome is read for it,
# so sketching real FASTQ here would test `cv.lineage_groups`, not this.
# ---------------------------------------------------------------------------


def test_confounding_is_high_when_the_phenotype_is_the_lineage():
    """POSITIVE CONTROL. The phenotype is a pure function of the lineage
    label, so every bit of it is predictable from lineage alone and there
    is no biological signal to find. If this does not read high, the
    statistic is not measuring what it claims to.
    """
    X, groups = _synthetic_matrix_cohort(seed=20)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])

    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    assert isinstance(report.confounding, Confounding)
    assert report.confounding.statistic == "cramers_v"
    assert report.confounding.undefined_reason is None
    # Not `== 1.0`: Bergsma's correction subtracts phi2's expectation under
    # independence, so even a *perfect* association lands below 1 in any
    # finite sample. Here r=4, c=2, n=32 and phi2=1 give
    # phi2~ = 1 - 3/31 = 0.9032, r~ = 4 - 9/31, c~ = 2 - 1/31, and so
    # V~ = sqrt(0.9032 / 0.9677) = 0.966 -- which is exactly what this
    # returns. The bound below pins the behaviour (near-maximal) rather
    # than re-deriving that arithmetic, while staying tight enough that a
    # genuinely broken statistic could not slip past it.
    assert report.confounding.value > 0.9, (
        "a phenotype that is a deterministic function of the lineage label "
        f"must be maximally confounded; got {report.confounding.value:.3f}"
    )
    assert report.confounding.n_lineages == 4
    assert report.confounding.n_phenotype_levels == 2


def test_confounding_is_low_when_the_phenotype_is_independent_of_lineage():
    """NEGATIVE CONTROL. The phenotype alternates within every lineage, so
    each lineage carries both classes in equal measure and the lineage
    label predicts nothing. If the statistic reports confounding here it is
    inventing structure, which is the one failure mode that would make the
    number actively misleading rather than merely imprecise.
    """
    X, groups = _synthetic_matrix_cohort(seed=21)
    # Alternate inside each group rather than shuffling: this makes the
    # contingency table exactly uniform, so the expected value is 0 by
    # construction and the assertion needs no tolerance for sampling luck.
    phenotype = np.array([i % 2 for i in range(len(groups))])

    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    assert report.confounding.statistic == "cramers_v"
    assert report.confounding.value == pytest.approx(0.0, abs=1e-9), (
        "a phenotype distributed evenly across every lineage cannot be "
        f"confounded with lineage; got {report.confounding.value:.3f}"
    )


def test_confounding_uses_omega_squared_for_a_continuous_phenotype():
    """A regressor's phenotype has no contingency table, so the categorical
    branch cannot run. It must report `omega_squared` rather than silently
    binning the values into one -- and must still separate a
    lineage-determined phenotype from an independent one.
    """
    from sklearn.linear_model import Ridge

    X, groups = _synthetic_matrix_cohort(seed=22)
    rng = np.random.default_rng(22)

    # Determined by lineage (plus a little noise, so the within-group
    # variance is not exactly zero -- omega-squared is a variance ratio).
    by_lineage = groups * 10.0 + rng.normal(scale=0.1, size=len(groups))
    confounded = audit(
        Ridge(), X, by_lineage, groups=groups, n_splits=4, scoring="r2"
    ).confounding
    assert confounded.statistic == "omega_squared"
    assert confounded.n_phenotype_levels == 0, "a continuous phenotype has no levels"
    assert confounded.value > 0.9, (
        f"a phenotype set by lineage must read as confounded; got {confounded.value:.3f}"
    )

    # Independent of lineage: same spread, drawn without reference to it.
    independent = rng.normal(size=len(groups))
    clean = audit(
        Ridge(), X, independent, groups=groups, n_splits=4, scoring="r2"
    ).confounding
    assert clean.statistic == "omega_squared"
    assert clean.value < 0.3, (
        f"a lineage-independent phenotype must not read as confounded; got {clean.value:.3f}"
    )


def test_confounding_is_reported_but_never_turned_into_a_verdict():
    """`audit()` reports the number and stops there. `docs/audit/audit-api.md`
    wanted a CLEAN/INFLATED/CONFOUNDED enum built on this statistic;
    `audit.py`'s module docstring explains at length why that was declined
    (the field has not settled which number is 'correct', so the judgment
    belongs to the caller who knows the cohort). This pins that decision so
    a later change cannot quietly reintroduce the verdict.
    """
    X, groups = _synthetic_matrix_cohort(seed=23)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])
    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    for forbidden in ("verdict", "is_confounded", "is_clean", "is_inflated"):
        assert not hasattr(report, forbidden), (
            f"AuditReport grew a {forbidden!r} attribute -- audit.py's module "
            "docstring explains why an automated verdict was deliberately "
            "declined; reintroducing one needs that argument reopened, not a "
            "silent attribute"
        )
    assert not hasattr(report.confounding, "verdict")


def test_confounding_appears_in_the_rendered_report():
    X, groups = _synthetic_matrix_cohort(seed=24)
    phenotype = np.array([1 if g in (0, 1) else 0 for g in groups])
    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    markdown = report.to_markdown()
    # The statistic's name must travel with its value: the same number
    # means different things depending on which of the two it is.
    assert "cramers_v" in markdown or "Cramer" in markdown
    assert str(report.confounding) in markdown or f"{report.confounding.value:.3f}" in markdown


# ---------------------------------------------------------------------------
# Feature attribution to lineage (task 16) -- deliberately NOT a field here.
#
# `audit.py`'s own module docstring ("Feature attribution to lineage")
# explains why: the per-feature question task 16 named -- does a top
# feature's presence track lineage rather than phenotype -- is already
# answered, per feature, by `fastdna.explain()` as a separate composable
# call (its own positive/negative controls live in `test_explain.py`:
# `test_a_lineage_specific_kmer_is_flagged_as_a_lineage_marker` and
# `test_a_marker_associated_with_phenotype_within_lineages_is_credible`).
# Duplicating that Cochran-Mantel-Haenszel/lineage-fraction logic under a
# second name here, or growing a `feature_attribution` field `audit()`
# structurally has no fitted-model-plus-vectorizer to populate honestly,
# would be exactly the kind of re-implementation this module's own "What
# this module does not do" section already refuses. This test pins the
# absence -- the same way `test_confounding_is_reported_but_never_turned_
# into_a_verdict` above pins the "no verdict" decision -- so a later change
# cannot quietly reintroduce a duplicate without that argument being
# reopened.
# ---------------------------------------------------------------------------


def test_audit_report_has_no_feature_attribution_field():
    X, groups = _synthetic_matrix_cohort(seed=25)
    phenotype = np.array([g % 2 for g in groups])
    report = audit(
        LogisticRegression(max_iter=1000),
        X,
        phenotype,
        groups=groups,
        n_splits=4,
        scoring="accuracy",
    )

    for forbidden in ("feature_attribution", "signal_attributable_to_lineage"):
        assert not hasattr(report, forbidden), (
            f"AuditReport grew a {forbidden!r} attribute -- audit.py's module docstring's "
            "'Feature attribution to lineage' section explains why that question is answered "
            "by fastdna.explain() instead, as a separate composable call; reintroducing a "
            "field here needs that argument reopened, not a silent attribute"
        )


def test_explain_composes_with_audit_for_the_feature_attribution_question(tmp_path):
    """The module docstring's pointer is not aspirational: fitting the same
    kind of pipeline `audit()` itself takes, once, and calling
    `fastdna.explain()` on it actually answers task 16's question for a
    real planted marker, on this file's own fixtures.

    POSITIVE CONTROL: a marker present in only one of four lineages (and
    unrelated to the phenotype) must be flagged lineage-restricted.
    NEGATIVE CONTROL, in the same cohort: a marker present across every
    lineage, whose presence instead tracks the phenotype, must NOT be
    flagged lineage-restricted. If the positive control does not fire,
    that pins a real bug in the documented workflow, not a threshold to
    loosen.
    """
    paths, lineage_of = _lineage_cohort(tmp_path, n_lineages=4, per_lineage=6, seed=40)
    k = 21

    # A lineage-exclusive marker, appended to a single sample of lineage 0
    # only -- exactly k bases, so it is exactly one k-mer, and exactly the
    # construction `test_explain.py`'s own lineage-marker positive control
    # already validated as safe (does not disturb Mash-based clustering).
    lineage_marker = "ACGTTGCAAACCGGTTGATTA"
    assert len(lineage_marker) == k
    lineage_0_path = next(p for p, lineage in zip(paths, lineage_of) if lineage == 0)
    with open(lineage_0_path, "a") as fh:
        fh.write(f"@marker\n{lineage_marker}\n+\n{'I' * k}\n")

    # A phenotype-tracking marker, appended whenever phenotype == 1,
    # regardless of lineage -- present in every lineage, and correlated
    # with the phenotype rather than with lineage membership.
    phenotype_marker = "TTGGCCAATTGGCCAATTGGC"
    assert len(phenotype_marker) == k
    rng = np.random.default_rng(41)
    phenotype = rng.integers(0, 2, size=len(paths))
    for path, y in zip(paths, phenotype):
        if y == 1:
            with open(path, "a") as fh:
                fh.write(f"@pheno\n{phenotype_marker}\n+\n{'I' * k}\n")

    def _canonical(seq):
        complement = str.maketrans("ACGT", "TGCA")
        rc = seq.translate(complement)[::-1]
        return min(seq, rc)

    lineage_marker_kmer = _canonical(lineage_marker)
    phenotype_marker_kmer = _canonical(phenotype_marker)

    vec = KmerVectorizer(k=k, top_features=None, representation="presence").fit(paths)
    assert lineage_marker_kmer in vec._feature_sequences_, "fixture must produce the lineage marker k-mer"
    assert phenotype_marker_kmer in vec._feature_sequences_, "fixture must produce the phenotype marker k-mer"

    clf = LogisticRegression(max_iter=2000).fit(vec.transform(paths), phenotype)

    # top_n covers the whole vocabulary so both planted markers are
    # guaranteed to appear in the report regardless of where the fitted
    # coefficients happen to rank them -- the same reasoning
    # `test_explain.py` itself uses when it needs certainty rather than a
    # race to the top of |coefficient|.
    report = explain(vec, clf.coef_[0], paths, phenotype, top_n=len(vec.vocabulary_), lineage_threshold=0.02)
    by_kmer = {f.kmer: f for f in report.features}

    lineage_feature = by_kmer[lineage_marker_kmer]
    assert lineage_feature.n_lineages_present == 1, (
        "positive control: the planted marker should appear in exactly 1 lineage, got "
        f"{lineage_feature.n_lineages_present}"
    )
    assert lineage_feature.lineage_restricted, (
        "positive control: a marker exclusive to one lineage must be flagged lineage_restricted"
    )

    phenotype_feature = by_kmer[phenotype_marker_kmer]
    assert not phenotype_feature.lineage_restricted, (
        "negative control: a marker present across every lineage must not be flagged "
        f"lineage_restricted; got {phenotype_feature.n_lineages_present}/{phenotype_feature.n_lineages_total}"
    )
