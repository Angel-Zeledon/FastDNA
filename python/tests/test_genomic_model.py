"""Tests for fastdna.genomic_model.GenomicModel -- a deployment wrapper
around a fitted (vectorizer, estimator) pair that checks every new
prediction against the training cohort's own genomic-profile distribution
(via fastdna.anomaly.CohortOutlierFlagger).

Reuses the same synthetic-FASTQ-cohort pattern `test_anomaly.py` already
established (a shared random reference, per-read mutation, and a wholly
different reference for the out-of-distribution case) so this file is not
inventing a second convention for the same kind of test data. On top of
that, a deterministic marker motif is embedded in half the cohort's reads
to give the classifier a real, learnable phenotype signal -- without one,
"the model predicts correctly" would not be a meaningful assertion.
"""
from __future__ import annotations

import pathlib
import random

import pytest

pytest.importorskip("sklearn")
np = pytest.importorskip("numpy")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from fastdna.genomic_model import GenomicModel, OutOfDistributionWarning  # noqa: E402
from fastdna.sklearn import KmerVectorizer  # noqa: E402


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def _random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    bases = "ACGT"
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice(bases)
    return "".join(out)


def _reads_from(reference: str, n_reads: int, read_len: int, mutation_rate: float, rng: random.Random) -> list[str]:
    reads = []
    for _ in range(n_reads):
        start = rng.randrange(0, len(reference) - read_len + 1)
        reads.append(_mutate(reference[start : start + read_len], mutation_rate, rng))
    return reads


# Same "one shared organism, mutated per replicate" cohort convention
# test_anomaly.py uses, plus a wholly different organism for the
# out-of-distribution case.
_REFERENCE = _random_sequence(3000, random.Random(1))
_OTHER_REFERENCE = _random_sequence(3000, random.Random(99))

# A 40 bp marker, long enough that a 21-mer window can sit entirely inside
# it (40 - 21 + 1 = 20 such windows per read it is embedded in) -- so
# embedding it turns into a handful of k-mers that are IDENTICAL across
# every read/sample it is embedded in, giving KmerVectorizer(representation
# ="presence") a perfectly separable, deterministic feature to learn from.
_MARKER = _random_sequence(40, random.Random(777))
_MARKER_OFFSET = 60  # position within each 150 bp read the marker overwrites
VEC_K = 21
FLAGGER_K = 15  # matches test_anomaly.py's own proven-working parameters
FLAGGER_SKETCH_SIZE = 200


def _reads_for(positive: bool, rng: random.Random) -> list[str]:
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    if positive:
        reads = [r[:_MARKER_OFFSET] + _MARKER + r[_MARKER_OFFSET + len(_MARKER) :] for r in reads]
    return reads


def _labeled_sample(tmp_path, idx, positive, seed):
    rng = random.Random(seed)
    reads = _reads_for(positive, rng)
    return write_fastq(tmp_path, f"sample_{idx}.fastq", reads)


def _out_of_distribution_sample(tmp_path, name="outlier.fastq"):
    rng = random.Random(5252)
    reads = _reads_from(_OTHER_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, name, reads)


@pytest.fixture
def labeled_cohort(tmp_path):
    """6 positive + 6 negative training samples, and their 0/1 labels in
    the same order.
    """
    paths = []
    labels = []
    for i in range(6):
        paths.append(str(_labeled_sample(tmp_path, f"pos{i}", True, seed=1000 + i)))
        labels.append(1)
    for i in range(6):
        paths.append(str(_labeled_sample(tmp_path, f"neg{i}", False, seed=2000 + i)))
        labels.append(0)
    return paths, np.array(labels)


@pytest.fixture
def fitted_model(labeled_cohort):
    paths, y = labeled_cohort
    vectorizer = KmerVectorizer(k=VEC_K, min_count=1, top_features=None)
    X = vectorizer.fit_transform(paths, y)
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    return GenomicModel(
        estimator, vectorizer, paths, k=FLAGGER_K, sketch_size=FLAGGER_SKETCH_SIZE
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construction_rejects_an_estimator_with_no_predict(labeled_cohort):
    paths, y = labeled_cohort
    vectorizer = KmerVectorizer(k=VEC_K, top_features=None).fit(paths)

    with pytest.raises(TypeError, match="predict"):
        GenomicModel(object(), vectorizer, paths)


def test_construction_rejects_a_vectorizer_with_no_transform(labeled_cohort):
    paths, y = labeled_cohort
    estimator = LogisticRegression().fit(np.zeros((len(paths), 1)), y)

    with pytest.raises(TypeError, match="transform"):
        GenomicModel(estimator, object(), paths)


def test_construction_requires_at_least_four_training_paths(tmp_path, labeled_cohort):
    paths, y = labeled_cohort
    # One positive, one negative -- both classes present, so fitting the
    # estimator itself succeeds and the failure under test is genuinely
    # GenomicModel's own minimum-cohort-size check, not an unrelated
    # single-class LogisticRegression error.
    two_paths = [paths[0], paths[6]]
    two_labels = np.array([y[0], y[6]])
    vectorizer = KmerVectorizer(k=VEC_K, top_features=None).fit(two_paths)
    X = vectorizer.transform(two_paths)
    estimator = LogisticRegression().fit(X, two_labels)

    with pytest.raises(ValueError):
        GenomicModel(estimator, vectorizer, two_paths)


def test_repr_reports_estimator_vectorizer_and_cohort_size(fitted_model, labeled_cohort):
    paths, _ = labeled_cohort
    text = repr(fitted_model)
    assert "LogisticRegression" in text
    assert "KmerVectorizer" in text
    assert f"{len(paths)} samples" in text


# ---------------------------------------------------------------------------
# predict(): the phenotype prediction itself
# ---------------------------------------------------------------------------


def test_in_distribution_positive_sample_predicted_correctly_and_not_flagged(tmp_path, fitted_model, recwarn):
    held_out = str(_labeled_sample(tmp_path, "held_out_pos", True, seed=9001))

    result = fitted_model.predict(held_out)
    row = result.to_pylist()[0]

    assert row["prediction"] == 1
    assert row["in_distribution"] is True
    assert row["verdict"] == "in-distribution"
    assert not any(issubclass(w.category, OutOfDistributionWarning) for w in recwarn.list)


def test_in_distribution_negative_sample_predicted_correctly_and_not_flagged(tmp_path, fitted_model, recwarn):
    held_out = str(_labeled_sample(tmp_path, "held_out_neg", False, seed=9002))

    result = fitted_model.predict(held_out)
    row = result.to_pylist()[0]

    assert row["prediction"] == 0
    assert row["in_distribution"] is True
    assert not any(issubclass(w.category, OutOfDistributionWarning) for w in recwarn.list)


def test_out_of_distribution_sample_is_flagged_and_warns(tmp_path, fitted_model):
    outlier = str(_out_of_distribution_sample(tmp_path))

    with pytest.warns(OutOfDistributionWarning, match=outlier):
        result = fitted_model.predict(outlier)

    row = result.to_pylist()[0]
    assert row["in_distribution"] is False
    assert row["verdict"] == "OUT-OF-DISTRIBUTION"
    assert row["outlier_score"] > fitted_model.outlier_threshold


def test_probability_populated_and_in_unit_interval_for_in_distribution_samples(tmp_path, fitted_model):
    held_out = str(_labeled_sample(tmp_path, "held_out_proba", True, seed=9003))

    result = fitted_model.predict(held_out)
    proba = result.to_pylist()[0]["probability"]

    assert proba is not None
    assert 0.0 <= proba <= 1.0


def test_predict_accepts_a_bare_path_and_a_list_identically(tmp_path, fitted_model):
    held_out = str(_labeled_sample(tmp_path, "held_out_bare", True, seed=9004))

    single = fitted_model.predict(held_out)
    batched = fitted_model.predict([held_out])

    assert single.to_pylist() == batched.to_pylist()


def test_predict_batches_multiple_samples_in_input_order(tmp_path, fitted_model):
    pos = str(_labeled_sample(tmp_path, "batch_pos", True, seed=9005))
    neg = str(_labeled_sample(tmp_path, "batch_neg", False, seed=9006))
    outlier = str(_out_of_distribution_sample(tmp_path, "batch_outlier.fastq"))

    with pytest.warns(OutOfDistributionWarning):
        result = fitted_model.predict([pos, neg, outlier])

    rows = result.to_pylist()
    assert [r["sample"] for r in rows] == [pos, neg, outlier]
    assert rows[0]["prediction"] == 1
    assert rows[1]["prediction"] == 0
    assert rows[0]["in_distribution"] is True
    assert rows[1]["in_distribution"] is True
    assert rows[2]["in_distribution"] is False


def test_predict_rejects_empty_input():
    # The empty-input check happens before predict() touches
    # self.vectorizer/self.estimator/self._flagger at all, so an
    # uninitialized instance (no real fitted cohort, no sketching paid for)
    # is enough to exercise it.
    model = GenomicModel.__new__(GenomicModel)
    with pytest.raises(ValueError, match="at least one path"):
        model.predict([])


# ---------------------------------------------------------------------------
# fit(): the one-call convenience
# ---------------------------------------------------------------------------


def test_fit_classmethod_produces_a_working_model(tmp_path, labeled_cohort):
    paths, y = labeled_cohort
    model = GenomicModel.fit(
        paths,
        y,
        vectorizer=KmerVectorizer(k=VEC_K, top_features=None),
        estimator=LogisticRegression(max_iter=1000),
        k=FLAGGER_K,
        sketch_size=FLAGGER_SKETCH_SIZE,
    )

    held_out = str(_labeled_sample(tmp_path, "fit_held_out", True, seed=9101))
    row = model.predict(held_out).to_pylist()[0]

    assert row["prediction"] == 1
    assert row["in_distribution"] is True


def test_fit_classmethod_does_not_mutate_the_estimator_passed_in(labeled_cohort):
    paths, y = labeled_cohort
    original_estimator = LogisticRegression(max_iter=1000)

    GenomicModel.fit(
        paths,
        y,
        vectorizer=KmerVectorizer(k=VEC_K, top_features=None),
        estimator=original_estimator,
        k=FLAGGER_K,
        sketch_size=FLAGGER_SKETCH_SIZE,
    )

    # sklearn.base.clone() is used internally (see the module docstring),
    # so the object the caller passed in must still be unfitted.
    assert not hasattr(original_estimator, "classes_")
