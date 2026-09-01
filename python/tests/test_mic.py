"""Tests for `fastdna.mic` -- log2(MIC) regression for antimicrobial
resistance.

Three groups of tests, mirroring the module's own three-part structure:

* `log2_mic` -- the validation + transform every other function is built
  from, checked against both hand-computed values and its documented
  rejections (non-positive, non-finite, empty, non-1D).
* `MicRegressor` -- fit/predict round trip (`predict() == 2 **
  predict_log2()` by construction) and the fail-fast row-count check.
* `mic_regression_report` -- hand-computable fixtures (perfect predictions,
  and a fixture with a known, by-hand R^2/MAE/essential-agreement), plus
  validation, following the same "cross-check sklearn, then hand-verify a
  simple case" convention `test_evaluation.py` uses.

A final end-to-end test wires `fastdna.sklearn.KmerVectorizer` ->
`MicRegressor` -> `mic_regression_report` together over a tiny synthetic
FASTQ cohort, exactly the pipeline the module docstring documents, so the
documented example is actually exercised rather than merely asserted in
prose.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("sklearn")

np = pytest.importorskip("numpy")

import pyarrow as pa

from fastdna.mic import (
    MicRegressionReport,
    MicRegressor,
    log2_mic,
    mic_regression_report,
)

# ---------------------------------------------------------------------------
# log2_mic
# ---------------------------------------------------------------------------


def test_log2_mic_matches_hand_computed_values():
    # A real two-fold dilution series: 0.25, 0.5, 1, 2, 4, 8, 16 mg/L ->
    # log2 gives exactly -2, -1, 0, 1, 2, 3, 4.
    mic = [0.25, 0.5, 1, 2, 4, 8, 16]
    result = log2_mic(mic)
    assert np.allclose(result, [-2, -1, 0, 1, 2, 3, 4])
    assert result.dtype == np.float64


def test_log2_mic_rejects_zero():
    with pytest.raises(ValueError, match="positive"):
        log2_mic([1.0, 0.0, 2.0])


def test_log2_mic_rejects_negative():
    with pytest.raises(ValueError, match="positive"):
        log2_mic([1.0, -4.0, 2.0])


def test_log2_mic_rejects_nan_and_inf():
    with pytest.raises(ValueError, match="finite"):
        log2_mic([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="finite"):
        log2_mic([1.0, float("inf"), 2.0])


def test_log2_mic_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        log2_mic([])


def test_log2_mic_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        log2_mic([[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# MicRegressor
# ---------------------------------------------------------------------------


def test_mic_regressor_predict_is_two_to_the_predict_log2():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    true_log2 = X @ np.array([1.0, -0.5, 0.0, 0.25, 0.0]) + 2.0
    y = np.exp2(true_log2)

    model = MicRegressor(Ridge(alpha=0.1)).fit(X, y)
    log2_pred = model.predict_log2(X)
    pred = model.predict(X)

    assert np.allclose(pred, np.exp2(log2_pred))
    # Ridge on a genuinely linear-in-log2 relationship should recover it
    # closely, not just be self-consistent.
    assert np.corrcoef(log2_pred, true_log2)[0, 1] > 0.95


def test_mic_regressor_defaults_to_ridge_when_no_estimator_given():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    y = np.exp2(rng.normal(size=10))

    model = MicRegressor().fit(X, y)
    assert isinstance(model.estimator_, Ridge)


def test_mic_regressor_fit_rejects_row_count_mismatch():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 3))
    y = [1.0, 2.0, 4.0]  # only 3 values for 5 rows

    with pytest.raises(ValueError, match="5 rows"):
        MicRegressor().fit(X, y)


def test_mic_regressor_fit_rejects_non_positive_mic_via_log2_mic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 2))
    y = [1.0, 0.0, 2.0]

    with pytest.raises(ValueError, match="positive"):
        MicRegressor().fit(X, y)


def test_mic_regressor_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        MicRegressor().predict([[1.0, 2.0]])


def test_mic_regressor_accepts_a_custom_estimator():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 4))
    y = np.exp2(rng.normal(size=20))

    model = MicRegressor(RandomForestRegressor(n_estimators=10, random_state=0)).fit(X, y)
    assert isinstance(model.estimator_, RandomForestRegressor)
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert np.all(preds > 0)


def test_mic_regressor_is_clonable():
    """Regression test for a real bug found while scaling up
    scratch/hiv_repro_scaled: `sklearn.base.clone()` (which every
    `cross_val_score`/`GridSearchCV`/`fastdna.audit()` call makes once per
    fold) raised `TypeError: ... does not seem to be a scikit-learn
    estimator` on any `MicRegressor`, because the class defined neither
    `get_params()` nor `set_params()`. The module's own docstring example
    never exercises `clone()` (it calls `.fit()`/`.predict()` directly), so
    the break was invisible until `MicRegressor` was placed inside a
    cross-validated pipeline -- exactly the "compose with sklearn's own
    model-selection tools" use case `fastdna.cv`/`fastdna.audit` exist for.
    """
    from sklearn.base import clone
    from sklearn.linear_model import Ridge

    model = MicRegressor(Ridge(alpha=2.0))
    cloned = clone(model)

    assert cloned is not model
    assert isinstance(cloned.estimator, Ridge)
    assert cloned.estimator.alpha == pytest.approx(2.0)
    assert cloned.estimator is not model.estimator  # clone() deep-copies nested estimators too


def test_mic_regressor_get_params_exposes_nested_estimator_params():
    from sklearn.linear_model import Ridge

    model = MicRegressor(Ridge(alpha=3.0))
    params = model.get_params(deep=True)

    assert params["estimator"] is model.estimator
    assert params["estimator__alpha"] == pytest.approx(3.0)
    assert model.get_params(deep=False) == {"estimator": model.estimator}


def test_mic_regressor_set_params_supports_nested_syntax():
    from sklearn.linear_model import Ridge

    model = MicRegressor(Ridge(alpha=1.0))
    model.set_params(estimator__alpha=9.0)
    assert model.estimator.alpha == pytest.approx(9.0)

    other = Ridge(alpha=0.5)
    model.set_params(estimator=other)
    assert model.estimator is other


def test_mic_regressor_set_params_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unexpected parameter"):
        MicRegressor().set_params(not_a_real_param=1)


def test_mic_regressor_composes_with_cross_val_score():
    """The concrete composability check: `MicRegressor` wrapped in
    `sklearn.model_selection.cross_val_score` with an explicit `scoring=`
    (it has no default `.score()` -- see this test's own scoring= choice)
    runs end to end across multiple folds, each of which clones the
    estimator. Before the get_params()/set_params() fix above this raised
    on the very first fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score

    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 5))
    true_log2 = X @ np.array([1.0, -0.5, 0.0, 0.25, 0.0]) + 2.0
    y = np.exp2(true_log2)

    scores = cross_val_score(
        MicRegressor(Ridge(alpha=0.5)),
        X,
        y,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        scoring="r2",
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_generic_r2_scoring_differs_from_mic_regression_report_r2():
    """Pins the pitfall `MicRegressor.predict()`'s own docstring warns
    about: a generic `scoring="r2"` (what `cross_val_score`/`GridSearchCV`/
    `fastdna.audit()` compute by default, via `predict()`) is `r2_score`
    over RAW-MIC predictions, not the log2-space R^2 the MIC-regression
    literature reports and `mic_regression_report()` returns. Over a
    dilution-series-realistic MIC range (multiple octaves) with real fit
    noise, the two must not agree -- exp2's convexity means the same
    log2-space residual is a small raw-MIC error for a low-MIC sample and a
    huge one for a high-MIC sample, which raw-space R^2 punishes
    unevenly and log2-space R^2 does not.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(7)
    n = 60
    X = rng.normal(size=(n, 5))
    true_log2 = X @ np.array([1.0, -0.5, 0.3, 0.25, -0.2]) + 3.0  # spans several octaves
    noisy_log2 = true_log2 + rng.normal(scale=0.4, size=n)  # realistic, imperfect fit
    mic_true = np.exp2(noisy_log2)

    model = MicRegressor(Ridge(alpha=0.5)).fit(X, mic_true)

    generic_r2 = r2_score(mic_true, model.predict(X))
    report = mic_regression_report(mic_true, model.predict(X))

    assert not generic_r2 == pytest.approx(report.r2, abs=1e-6), (
        f"raw-space R^2 ({generic_r2}) unexpectedly matched log2-space R^2 ({report.r2}) -- "
        "the fixture no longer demonstrates the pitfall this test pins"
    )
    # The log2-space number is the one predict_log2() itself reproduces
    # directly, confirming report.r2 is scoring the space the model was
    # actually fit in, not an independent third computation.
    assert report.r2 == pytest.approx(r2_score(noisy_log2, model.predict_log2(X)), abs=1e-9)


# ---------------------------------------------------------------------------
# mic_regression_report
# ---------------------------------------------------------------------------


def test_perfect_predictions_give_r2_one_mae_zero_full_agreement():
    mic_true = [0.5, 1.0, 2.0, 4.0, 8.0]
    report = mic_regression_report(mic_true, mic_true)

    assert isinstance(report, MicRegressionReport)
    assert report.r2 == pytest.approx(1.0)
    assert report.mae_log2 == pytest.approx(0.0, abs=1e-9)
    assert report.essential_agreement == pytest.approx(1.0)


def test_hand_computed_mae_and_essential_agreement():
    # log2(true) = [0, 1, 2, 3]; log2(pred) = [1, 1, 2, 5] -> errors [1, 0, 0, 2]
    mic_true = [1.0, 2.0, 4.0, 8.0]
    mic_pred = [2.0, 2.0, 4.0, 32.0]

    report = mic_regression_report(mic_true, mic_pred, tolerance_log2=1.0)

    assert report.mae_log2 == pytest.approx((1 + 0 + 0 + 2) / 4)
    # Within +/-1 dilution: errors [1, 0, 0, 2] -> 3 of 4 within tolerance.
    assert report.essential_agreement == pytest.approx(3 / 4)


def test_essential_agreement_respects_custom_tolerance():
    mic_true = [1.0, 2.0, 4.0, 8.0]
    mic_pred = [2.0, 2.0, 4.0, 32.0]  # log2 errors [1, 0, 0, 2]

    strict = mic_regression_report(mic_true, mic_pred, tolerance_log2=0.5)
    loose = mic_regression_report(mic_true, mic_pred, tolerance_log2=2.0)

    assert strict.essential_agreement == pytest.approx(2 / 4)  # only the two exact hits
    assert loose.essential_agreement == pytest.approx(1.0)  # every error <= 2


def test_r2_and_mae_match_sklearn_reference_in_log2_space():
    from sklearn.metrics import mean_absolute_error, r2_score

    rng = np.random.default_rng(5)
    mic_true = np.exp2(rng.uniform(-2, 4, size=30))
    mic_pred = np.exp2(np.log2(mic_true) + rng.normal(scale=0.5, size=30))

    report = mic_regression_report(mic_true, mic_pred)

    assert report.r2 == pytest.approx(r2_score(np.log2(mic_true), np.log2(mic_pred)))
    assert report.mae_log2 == pytest.approx(mean_absolute_error(np.log2(mic_true), np.log2(mic_pred)))


def test_report_table_has_documented_columns_and_row_count():
    mic_true = [1.0, 2.0, 4.0]
    mic_pred = [1.0, 4.0, 4.0]
    report = mic_regression_report(mic_true, mic_pred)

    assert isinstance(report.table, pa.Table)
    assert set(report.table.column_names) == {
        "mic_true",
        "mic_pred",
        "log2_mic_true",
        "log2_mic_pred",
        "log2_error",
    }
    assert report.table.num_rows == 3
    assert np.allclose(report.table.column("mic_true").to_numpy(), mic_true)
    assert np.allclose(report.table.column("mic_pred").to_numpy(), mic_pred)
    assert np.allclose(
        report.table.column("log2_error").to_numpy(),
        np.log2(mic_pred) - np.log2(mic_true),
    )


def test_report_rejects_length_mismatch():
    with pytest.raises(ValueError, match="one prediction per true value"):
        mic_regression_report([1.0, 2.0, 4.0], [1.0, 2.0])


def test_report_rejects_fewer_than_two_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        mic_regression_report([1.0], [1.0])


def test_report_rejects_non_positive_tolerance():
    with pytest.raises(ValueError, match="tolerance_log2"):
        mic_regression_report([1.0, 2.0], [1.0, 2.0], tolerance_log2=0.0)
    with pytest.raises(ValueError, match="tolerance_log2"):
        mic_regression_report([1.0, 2.0], [1.0, 2.0], tolerance_log2=-1.0)


def test_report_propagates_log2_mic_validation_for_either_argument():
    with pytest.raises(ValueError, match="positive"):
        mic_regression_report([1.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        mic_regression_report([1.0, 2.0], [1.0, -1.0])


# ---------------------------------------------------------------------------
# End-to-end: KmerVectorizer -> MicRegressor -> mic_regression_report
# ---------------------------------------------------------------------------

BACKGROUND_READ = "ACGTACGTAC"  # shared "core genome" k-mer background
ACCESSORY_READ = "GGGGGGGGGG"  # resistance-associated marker k-mer


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def write_susceptible(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    return write_fastq(tmp_path, name, [BACKGROUND_READ] * 3)


def write_resistant(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    return write_fastq(tmp_path, name, [BACKGROUND_READ] * 3 + [ACCESSORY_READ] * 3)


def test_end_to_end_kmer_vectorizer_pipeline_separates_high_and_low_mic(tmp_path):
    """The pipeline documented in the module docstring: KmerVectorizer builds
    the feature matrix, MicRegressor fits log2(MIC) on it, and
    mic_regression_report scores the result. Not a claim of generalization
    (this is a training-set check on 8 tiny hand-built samples) -- it is a
    wiring test that the marker k-mer actually drives the fitted model's
    predictions in the expected direction, and that the whole pipeline runs
    without error end to end.
    """
    fastdna_sklearn = pytest.importorskip("fastdna.sklearn")
    from sklearn.linear_model import Ridge

    susceptible_paths = [write_susceptible(tmp_path, f"s{i}.fastq") for i in range(4)]
    resistant_paths = [write_resistant(tmp_path, f"r{i}.fastq") for i in range(4)]
    paths = susceptible_paths + resistant_paths
    # Susceptible: MIC 1 mg/L (log2 = 0). Resistant: MIC 16 mg/L (log2 = 4).
    mic_values = np.array([1.0] * 4 + [16.0] * 4)

    vectorizer = fastdna_sklearn.KmerVectorizer(k=5, min_count=1, top_features=None)
    X = vectorizer.fit_transform(paths)

    model = MicRegressor(Ridge(alpha=1.0)).fit(X, mic_values)
    predicted = model.predict(X)

    report = mic_regression_report(mic_values, predicted)
    assert isinstance(report, MicRegressionReport)
    assert report.table.num_rows == 8

    # The model must at least rank the two groups correctly on their own
    # training data: predicted MIC for every resistant sample should exceed
    # predicted MIC for every susceptible sample.
    assert predicted[:4].max() < predicted[4:].min()
