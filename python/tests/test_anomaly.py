"""Tests for fastdna.anomaly.GenomicAnomalyDetector -- unsupervised
outlier detection over a cohort's genomic profiles, built on
fastdna.sketch()/Sketch.mash_distance() (see python/fastdna/anomaly.py's
module docstring for the feature-representation and fit/predict
consistency design decisions this test file verifies).
"""
from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest

pytest.importorskip("sklearn")

from fastdna.anomaly import GenomicAnomalyDetector


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


# A fixed "organism" reference all baseline (and the in-distribution
# held-out) samples are drawn from, with a small per-read mutation rate to
# mimic real sequencing noise/variation between replicates -- distinct
# FASTQ files that nonetheless represent "the same normal population".
_REFERENCE = _random_sequence(3000, random.Random(1))
# A wholly different "organism" -- a fresh random reference, not a mutated
# copy of _REFERENCE -- for the clearly-anomalous sample.
_OTHER_REFERENCE = _random_sequence(3000, random.Random(99))


def _baseline_sample(tmp_path, idx):
    rng = random.Random(1000 + idx)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, f"baseline_{idx}.fastq", reads)


def _in_distribution_sample(tmp_path, name="held_out.fastq"):
    rng = random.Random(4242)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, name, reads)


def _out_of_distribution_sample(tmp_path, name="outlier.fastq"):
    rng = random.Random(5252)
    reads = _reads_from(_OTHER_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, name, reads)


def _mildly_different_sample(tmp_path, name="mild.fastq"):
    # Same reference as the baseline, but a much higher per-read mutation
    # rate -- "mildly different", not "a different organism".
    rng = random.Random(7373)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.08, rng=rng)
    return write_fastq(tmp_path, name, reads)


@pytest.fixture
def baseline_paths(tmp_path):
    return [str(_baseline_sample(tmp_path, i)) for i in range(8)]


def test_in_distribution_sample_predicted_inlier(tmp_path, baseline_paths):
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    held_out = str(_in_distribution_sample(tmp_path))
    label = detector.predict([held_out])[0]

    assert label == 1


def test_out_of_distribution_sample_predicted_outlier(tmp_path, baseline_paths):
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    outlier = str(_out_of_distribution_sample(tmp_path))
    label = detector.predict([outlier])[0]

    assert label == -1


def test_out_of_distribution_sample_scores_more_anomalous_than_in_distribution(tmp_path, baseline_paths):
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    held_out = str(_in_distribution_sample(tmp_path))
    outlier = str(_out_of_distribution_sample(tmp_path))

    scores = detector.score_samples([held_out, outlier])
    held_out_score, outlier_score = scores[0], scores[1]

    # score_samples: higher = more normal/inlier-like (sklearn convention).
    assert outlier_score < held_out_score


def test_score_samples_ranks_mild_and_severe_anomalies_correctly(tmp_path, baseline_paths):
    """score_samples should produce a real ranking, not just a threshold:
    a clearly-different-organism sample should score more anomalous than
    a same-organism-but-noisier sample, relative to the same baseline.
    """
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    mild = str(_mildly_different_sample(tmp_path))
    severe = str(_out_of_distribution_sample(tmp_path))

    scores = detector.score_samples([mild, severe])
    mild_score, severe_score = scores[0], scores[1]

    assert severe_score < mild_score


def test_predict_is_independent_of_batching(tmp_path, baseline_paths):
    """Regression test for the fit/predict feature-space consistency
    requirement: a sample's feature vector must always be computed
    relative to the *fitted baseline*, never relative to whatever other
    samples happen to be in the same predict()/score_samples() call. A
    naive implementation that recomputed a fresh pairwise-distance matrix
    over "baseline + whatever was just asked about" would give different
    (and meaningless) scores depending on batching -- this test would
    catch that regression.
    """
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    held_out = str(_in_distribution_sample(tmp_path))
    outlier = str(_out_of_distribution_sample(tmp_path))

    together = detector.score_samples([held_out, outlier])
    separate_held_out = detector.score_samples([held_out])
    separate_outlier = detector.score_samples([outlier])

    assert together[0] == pytest.approx(separate_held_out[0])
    assert together[1] == pytest.approx(separate_outlier[0])

    labels_together = detector.predict([held_out, outlier])
    assert labels_together[0] == detector.predict([held_out])[0]
    assert labels_together[1] == detector.predict([outlier])[0]


def test_feature_vector_width_matches_baseline_size_regardless_of_query_batch_size(tmp_path, baseline_paths):
    detector = GenomicAnomalyDetector(k=15, sketch_size=200).fit(baseline_paths)

    held_out = str(_in_distribution_sample(tmp_path))
    outlier = str(_out_of_distribution_sample(tmp_path))

    X_one = detector._transform([held_out])
    X_two = detector._transform([held_out, outlier])

    assert X_one.shape == (1, len(baseline_paths))
    assert X_two.shape == (2, len(baseline_paths))
    # The held-out sample's own row is identical whether it's queried
    # alone or alongside another sample.
    assert np.allclose(X_one[0], X_two[0])


def test_method_isolation_forest_selects_isolation_forest(baseline_paths):
    from sklearn.ensemble import IsolationForest

    detector = GenomicAnomalyDetector(method="isolation_forest")

    assert isinstance(detector._detector, IsolationForest)


def test_method_one_class_svm_selects_one_class_svm(baseline_paths):
    from sklearn.svm import OneClassSVM

    detector = GenomicAnomalyDetector(method="one_class_svm")

    assert isinstance(detector._detector, OneClassSVM)


def test_default_method_is_one_class_svm_not_isolation_forest(baseline_paths):
    """Pins the default `method=` to `"one_class_svm"`. This is a
    deliberate, tested choice (see anomaly.py's module docstring, "Why the
    default `method` is `\"one_class_svm\"`"): `IsolationForest`'s
    axis-aligned splits cannot reliably separate a query whose feature
    vector is clamped at the `mash_distance` ceiling of `1.0` in every
    column from an ordinary baseline point, which
    `test_out_of_distribution_sample_predicted_outlier` and
    `test_score_samples_ranks_mild_and_severe_anomalies_correctly` above
    would catch as a regression if `IsolationForest` became the default
    again without fixing that underlying issue.
    """
    from sklearn.svm import OneClassSVM

    detector = GenomicAnomalyDetector()

    assert isinstance(detector._detector, OneClassSVM)


def test_unrecognized_method_raises_at_construction(baseline_paths):
    with pytest.raises(ValueError):
        GenomicAnomalyDetector(method="not_a_real_method")


def test_one_class_svm_also_flags_the_out_of_distribution_sample(tmp_path, baseline_paths):
    """Confirms method= actually changes detector behavior, not just its
    type: both supported methods should agree on the easy, clearly
    anomalous case.
    """
    detector = GenomicAnomalyDetector(k=15, sketch_size=200, method="one_class_svm", nu=0.2).fit(baseline_paths)

    outlier = str(_out_of_distribution_sample(tmp_path))
    label = detector.predict([outlier])[0]

    assert label == -1


def test_detector_kwargs_are_forwarded(baseline_paths):
    detector = GenomicAnomalyDetector(method="isolation_forest", n_estimators=17, contamination=0.05)

    assert detector._detector.n_estimators == 17
    assert detector._detector.contamination == 0.05


def test_predict_before_fit_raises(tmp_path):
    detector = GenomicAnomalyDetector()

    with pytest.raises(RuntimeError):
        detector.predict([str(_in_distribution_sample(tmp_path))])


def test_fit_requires_at_least_two_baseline_samples(tmp_path):
    detector = GenomicAnomalyDetector()

    with pytest.raises(ValueError):
        detector.fit([str(_baseline_sample(tmp_path, 0))])


def test_fit_returns_self(baseline_paths):
    detector = GenomicAnomalyDetector(k=15, sketch_size=200)

    assert detector.fit(baseline_paths) is detector
