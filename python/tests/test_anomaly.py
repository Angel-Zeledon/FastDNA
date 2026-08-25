"""Tests for fastdna.anomaly.CohortOutlierFlagger -- within-cohort QC
outlier flagging over MinHash-sketch profiles, built on fastdna.sketch()
/ Sketch.mash_distance().

**History (read before changing the framing of this file).** An earlier
`fastdna/anomaly.py` shipped a `GenomicAnomalyDetector` sold as
"emerging-pathogen surveillance". A post-merge audit found that framing
had no literature support *and* that the implementation had every defect
its own docstring claimed to have solved: a reachable, documented
`method="isolation_forest"` that scored a wholly different organism as
*more* normal than the baseline it was fitted on; a default
`OneClassSVM(nu=0.5)` that flagged 4 of 8 known-good baseline samples;
and a `score_samples` that saturated to exactly `0.0` past a short
radius, so it could not rank severity at all. The module was deleted in
commit c8870e3 rather than patched, and this file was left module-skipped
as the specification for a replacement.

This is that replacement, re-scoped to the question the mechanics can
actually answer: *within-cohort QC outlier flagging* -- given a cohort
that is supposed to be homogeneous, which member does not look like the
rest (a swapped sample, a contamination suspect, a failed prep)? The
tests below keep the original file's contract wherever it still holds
(fit/predict/score_samples semantics, batch-independence, feature-space
consistency, "don't flag your own known-good cohort") and replace the
parts that only made sense under the surveillance framing.
"""
from __future__ import annotations

import pathlib
import random

import pytest

pytest.importorskip("sklearn")

np = pytest.importorskip("numpy")

import pyarrow as pa

from fastdna.anomaly import CohortOutlierFlagger, flag_cohort


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


# A fixed "organism" reference all cohort (and the in-distribution
# held-out) samples are drawn from, with a small per-read mutation rate to
# mimic real sequencing noise/variation between replicates -- distinct
# FASTQ files that nonetheless represent "the same normal population".
_REFERENCE = _random_sequence(3000, random.Random(1))
# A wholly different "organism" -- a fresh random reference, not a mutated
# copy of _REFERENCE -- for the clearly-anomalous sample. In QC terms this
# is the swapped-tube / heavily-contaminated case, not "a novel pathogen".
_OTHER_REFERENCE = _random_sequence(3000, random.Random(99))


def _cohort_sample(tmp_path, idx):
    rng = random.Random(1000 + idx)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, f"cohort_{idx}.fastq", reads)


def _in_distribution_sample(tmp_path, name="held_out.fastq"):
    rng = random.Random(4242)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, name, reads)


def _out_of_distribution_sample(tmp_path, name="outlier.fastq"):
    rng = random.Random(5252)
    reads = _reads_from(_OTHER_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.01, rng=rng)
    return write_fastq(tmp_path, name, reads)


def _mildly_different_sample(tmp_path, name="mild.fastq"):
    # Same reference as the cohort, but a much higher per-read mutation
    # rate -- "mildly different", not "a different organism".
    rng = random.Random(7373)
    reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=0.08, rng=rng)
    return write_fastq(tmp_path, name, reads)


@pytest.fixture
def cohort_paths(tmp_path):
    return [str(_cohort_sample(tmp_path, i)) for i in range(8)]


class TestFlaggingAgainstAFittedCohort:
    def test_in_distribution_sample_predicted_inlier(self, tmp_path, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        held_out = str(_in_distribution_sample(tmp_path))
        label = flagger.predict([held_out])[0]

        assert label == 1

    def test_out_of_distribution_sample_predicted_outlier(self, tmp_path, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        outlier = str(_out_of_distribution_sample(tmp_path))
        label = flagger.predict([outlier])[0]

        assert label == -1

    def test_out_of_distribution_sample_scores_more_anomalous_than_in_distribution(self, tmp_path, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        held_out = str(_in_distribution_sample(tmp_path))
        outlier = str(_out_of_distribution_sample(tmp_path))

        scores = flagger.score_samples([held_out, outlier])
        held_out_score, outlier_score = scores[0], scores[1]

        # score_samples: higher = more normal/inlier-like (sklearn convention).
        assert outlier_score < held_out_score

    def test_score_samples_ranks_mild_and_severe_anomalies_correctly(self, tmp_path, cohort_paths):
        """score_samples should produce a real ranking, not just a threshold:
        a clearly-different-organism sample should score more anomalous than
        a same-organism-but-noisier sample, relative to the same cohort.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        mild = str(_mildly_different_sample(tmp_path))
        severe = str(_out_of_distribution_sample(tmp_path))

        scores = flagger.score_samples([mild, severe])
        mild_score, severe_score = scores[0], scores[1]

        assert severe_score < mild_score

    def test_outlier_scores_is_the_sign_flipped_view_of_score_samples(self, tmp_path, cohort_paths):
        """`outlier_scores()` exists because "higher = more outlying" is the
        reading a QC user wants, while `score_samples()` must keep
        scikit-learn's opposite convention. They must stay exact negations of
        each other, or the two would silently disagree about which sample is
        worst.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        held_out = str(_in_distribution_sample(tmp_path))
        outlier = str(_out_of_distribution_sample(tmp_path))
        queries = [held_out, outlier]

        assert np.allclose(flagger.outlier_scores(queries), -flagger.score_samples(queries))
        assert flagger.outlier_scores(queries)[1] > flagger.outlier_scores(queries)[0]

    def test_predict_is_independent_of_batching(self, tmp_path, cohort_paths):
        """Regression test for the fit/predict feature-space consistency
        requirement: a sample's feature vector must always be computed
        relative to the *fitted cohort*, never relative to whatever other
        samples happen to be in the same predict()/score_samples() call. A
        naive implementation that recomputed a fresh pairwise-distance matrix
        over "cohort + whatever was just asked about" would give different
        (and meaningless) scores depending on batching -- this test would
        catch that regression.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        held_out = str(_in_distribution_sample(tmp_path))
        outlier = str(_out_of_distribution_sample(tmp_path))

        together = flagger.score_samples([held_out, outlier])
        separate_held_out = flagger.score_samples([held_out])
        separate_outlier = flagger.score_samples([outlier])

        assert together[0] == pytest.approx(separate_held_out[0])
        assert together[1] == pytest.approx(separate_outlier[0])

        labels_together = flagger.predict([held_out, outlier])
        assert labels_together[0] == flagger.predict([held_out])[0]
        assert labels_together[1] == flagger.predict([outlier])[0]

    def test_feature_vector_width_matches_cohort_size_regardless_of_query_batch_size(self, tmp_path, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        held_out = str(_in_distribution_sample(tmp_path))
        outlier = str(_out_of_distribution_sample(tmp_path))

        X_one = flagger._transform([held_out])
        X_two = flagger._transform([held_out, outlier])

        assert X_one.shape == (1, len(cohort_paths))
        assert X_two.shape == (2, len(cohort_paths))
        # The held-out sample's own row is identical whether it's queried
        # alone or alongside another sample.
        assert np.allclose(X_one[0], X_two[0])

    def test_cohort_distance_excludes_a_samples_own_column(self, cohort_paths):
        """A cohort member queried back against its own fitted cohort must be
        summarized leave-one-out: its distance to *itself* is exactly 0.0 by
        construction and carries no information about whether it fits the
        cohort. Including that zero would drag every cohort member's summary
        statistic down relative to a genuinely new sample's, making the two
        incomparable -- fit-time and query-time statistics must be computed
        the same way.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        # Re-querying the fitted cohort reproduces the fit-time statistics
        # exactly, which is only true if the self-column is dropped in both.
        assert np.allclose(flagger.cohort_distance(cohort_paths), flagger.cohort_distances_)
        assert (flagger.cohort_distance(cohort_paths) > 0).all()


class TestHomogeneousCohortIsNotFlagged:
    def test_known_good_cohort_flags_nothing(self, cohort_paths):
        """The cohort passed to `fit()` is, by definition, the established
        normal baseline. A flagger that flags most of it is unusable for QC
        regardless of how well it catches true outliers: the alerts are all
        noise. (The deleted module's default `OneClassSVM(nu=0.5)` flagged 4
        of these 8.)

        This is also the "all-similar cohort flags nothing" property that
        makes the default method usable unattended: its threshold is an
        absolute statistical criterion, so a homogeneous cohort produces
        *zero* flags rather than a fixed quota of them.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        labels = flagger.predict(cohort_paths)
        flagged = int((labels == -1).sum())

        assert flagged == 0, (
            f"{flagged} of {len(cohort_paths)} known-good cohort samples were "
            f"predicted outliers (labels={labels.tolist()})"
        )

    def test_flag_cohort_on_a_homogeneous_cohort_flags_nothing(self, cohort_paths):
        table = flag_cohort(cohort_paths, k=15, sketch_size=200)

        assert not any(table.column("is_outlier").to_pylist())


class TestFlagCohort:
    def test_returns_one_arrow_row_per_sample_in_input_order(self, tmp_path, cohort_paths):
        outlier = str(_out_of_distribution_sample(tmp_path))
        paths = cohort_paths + [outlier]

        table = flag_cohort(paths, k=15, sketch_size=200)

        assert isinstance(table, pa.Table)
        assert table.column_names == ["sample", "cohort_distance", "outlier_score", "is_outlier"]
        assert table.column("sample").to_pylist() == paths

    def test_flags_the_deterministic_outlier_and_nothing_else(self, tmp_path, cohort_paths):
        """The headline QC claim: drop one wrong sample into an otherwise
        homogeneous cohort and exactly that sample comes back flagged.
        """
        outlier = str(_out_of_distribution_sample(tmp_path))
        paths = cohort_paths + [outlier]

        table = flag_cohort(paths, k=15, sketch_size=200)
        flagged = [
            sample
            for sample, is_outlier in zip(table.column("sample").to_pylist(), table.column("is_outlier").to_pylist())
            if is_outlier
        ]

        assert flagged == [outlier]

    def test_the_outlier_has_the_highest_outlier_score(self, tmp_path, cohort_paths):
        outlier = str(_out_of_distribution_sample(tmp_path))
        paths = cohort_paths + [outlier]

        table = flag_cohort(paths, k=15, sketch_size=200)
        scores = table.column("outlier_score").to_pylist()

        assert scores.index(max(scores)) == len(paths) - 1

    def test_isolation_forest_flags_the_outlier_when_it_is_inside_the_fitted_cohort(self, tmp_path, cohort_paths):
        """`method="isolation_forest"` is only offered through this
        transductive path, where the sample being judged is part of the data
        the forest was fitted on -- see
        `TestIsolationForestIsCohortInternalOnly` for the measured reason it
        is not offered for out-of-cohort queries.
        """
        outlier = str(_out_of_distribution_sample(tmp_path))
        paths = cohort_paths + [outlier]

        table = flag_cohort(paths, k=15, sketch_size=200, method="isolation_forest")
        flagged = [
            sample
            for sample, is_outlier in zip(table.column("sample").to_pylist(), table.column("is_outlier").to_pylist())
            if is_outlier
        ]

        assert flagged == [outlier]


class TestIsolationForestIsCohortInternalOnly:
    def test_isolation_forest_rates_a_different_organism_as_more_normal_than_its_own_cohort(
        self, tmp_path, cohort_paths
    ):
        """Characterisation test pinning a measured, documented limitation --
        the exact defect that got the previous module deleted, kept visible
        here instead of being papered over.

        `IsolationForest`'s splits are axis-aligned thresholds drawn from the
        range the *training* data spans, so it cannot see how far past that
        range a query lies -- only which side of each threshold it falls on.
        Measured here: a wholly different organism, sitting at the
        `mash_distance` ceiling far beyond every cohort member, is scored
        *more normal* than the least typical cohort sample. In QC terms that
        is a silent false negative on the easiest possible case.

        That is why `CohortOutlierFlagger` refuses
        `method="isolation_forest"` at construction, and why the forest is
        reachable only through `flag_cohort()`, where every candidate is
        inside the fitted data.
        """
        from sklearn.ensemble import IsolationForest

        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)
        outlier = str(_out_of_distribution_sample(tmp_path))

        train = flagger.cohort_distances_.reshape(-1, 1)
        forest = IsolationForest(random_state=0, contamination=0.1).fit(train)

        query = flagger.cohort_distance([outlier]).reshape(-1, 1)
        assert float(query[0, 0]) > float(train.max()), "fixture no longer puts the query beyond the fitted range"

        worst_cohort_score = float(forest.score_samples(train).min())
        assert float(forest.score_samples(query)[0]) > worst_cohort_score, (
            "IsolationForest now scores an out-of-range query as more anomalous than "
            "every training point -- if this ever becomes reliably true, revisit whether "
            "method='isolation_forest' can be offered for out-of-cohort scoring"
        )

    def test_the_default_method_does_rank_beyond_the_fitted_range(self, tmp_path, cohort_paths):
        """The flip side of the test above: the default robust scorer is
        unbounded and strictly monotone in cohort distance, so it keeps
        separating samples arbitrarily far past the cohort's own spread.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)
        outlier = str(_out_of_distribution_sample(tmp_path))
        mild = str(_mildly_different_sample(tmp_path))

        worst_cohort = float(flagger.outlier_scores(cohort_paths).max())
        assert float(flagger.outlier_scores([mild])[0]) > worst_cohort
        assert float(flagger.outlier_scores([outlier])[0]) > float(flagger.outlier_scores([mild])[0])


class TestSeverityRanking:
    def _severity_series(self, tmp_path, rates):
        """One FASTQ per requested per-read mutation rate, all drawn from the
        same `_REFERENCE` the cohort uses -- a monotonically increasing
        "how far from normal is this" ladder.
        """
        paths = []
        for rate in rates:
            rng = random.Random(90000 + int(rate * 10000))
            reads = _reads_from(_REFERENCE, n_reads=200, read_len=150, mutation_rate=rate, rng=rng)
            paths.append(str(write_fastq(tmp_path, f"sev_{int(rate * 10000)}.fastq", reads)))
        return paths

    def test_score_samples_ranks_severity_across_the_full_distance_range(self, tmp_path, cohort_paths):
        """`score_samples`'s docstring promises a ranking, not just a binary
        label. The deleted module's `OneClassSVM(gamma="scale")` could not
        deliver one: over a near-zero-variance distance feature, gamma was
        enormous and every RBF term underflowed to exactly 0.0 past roughly
        mean distance 0.13, so a 15%-mutated same-organism sample and a
        completely unrelated organism both scored exactly 0.0.

        The replacement's score is a modified z-score of the cohort-distance
        statistic -- unbounded and strictly monotone in that statistic -- so
        this ladder must come out strictly ordered all the way out to a
        different organism.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200).fit(cohort_paths)

        ladder = self._severity_series(tmp_path, [0.06, 0.10, 0.15, 0.25])
        different_organism = str(_out_of_distribution_sample(tmp_path))
        queries = ladder + [different_organism]

        scores = [float(s) for s in flagger.score_samples(queries)]

        # Sanity: the ladder really does get progressively further away, so a
        # failure below is the scorer's, not the fixture's.
        distances = flagger.cohort_distance(queries)
        assert list(distances) == sorted(distances), (
            "the severity ladder is not monotonically increasing in cohort distance -- "
            "the ranking assertion below would not mean anything"
        )

        for i in range(len(scores) - 1):
            assert scores[i + 1] < scores[i], (
                f"score_samples does not rank severity {i} -> {i + 1}: "
                f"{scores[i]} vs {scores[i + 1]} (cohort distances "
                f"{distances[i]:.4f} vs {distances[i + 1]:.4f}). "
                f"full ladder scores = {scores}"
            )


class TestConstructionAndValidation:
    def test_default_method_is_robust_zscore(self):
        flagger = CohortOutlierFlagger()

        assert flagger.method == "robust_zscore"

    def test_isolation_forest_is_refused_by_the_out_of_cohort_class(self):
        """The previous module's fatal mistake was leaving a detector it had
        itself measured as broken for this feature representation reachable
        and documented. `CohortOutlierFlagger` scores *new* samples against a
        fitted cohort, which is exactly the case the forest cannot handle, so
        it is refused here rather than offered with a caveat.
        """
        with pytest.raises(ValueError) as exc_info:
            CohortOutlierFlagger(method="isolation_forest")

        message = str(exc_info.value)
        assert "flag_cohort" in message

    def test_unrecognized_method_raises_at_construction(self):
        with pytest.raises(ValueError):
            CohortOutlierFlagger(method="not_a_real_method")

    def test_non_positive_threshold_raises_at_construction(self):
        with pytest.raises(ValueError):
            CohortOutlierFlagger(threshold=0)

    def test_flag_cohort_forwards_detector_kwargs_to_the_forest(self, tmp_path, cohort_paths):
        outlier = str(_out_of_distribution_sample(tmp_path))
        table = flag_cohort(
            cohort_paths + [outlier],
            k=15,
            sketch_size=200,
            method="isolation_forest",
            n_estimators=17,
            contamination=0.2,
        )

        # 9 samples at contamination=0.2 -> roughly two flags, i.e. the
        # kwargs really did reach the forest rather than being swallowed.
        assert sum(table.column("is_outlier").to_pylist()) >= 2

    def test_flag_cohort_rejects_detector_kwargs_for_the_robust_method(self, cohort_paths):
        """The robust scorer has no detector to forward keyword arguments to,
        so silently accepting (and ignoring) `contamination=` would let a
        caller believe they had configured something they had not.
        """
        with pytest.raises(TypeError):
            flag_cohort(cohort_paths, k=15, sketch_size=200, contamination=0.1)

    def test_predict_before_fit_raises(self, tmp_path):
        flagger = CohortOutlierFlagger()

        with pytest.raises(RuntimeError):
            flagger.predict([str(_in_distribution_sample(tmp_path))])

    def test_fit_requires_a_minimum_cohort_size(self, tmp_path):
        """A modified z-score needs a median *and* a median absolute
        deviation of the cohort's own statistics. Below a handful of samples
        the MAD is not an estimate of anything, and every query would be
        judged against noise -- so this is refused loudly rather than
        answered with a confident, meaningless number.
        """
        flagger = CohortOutlierFlagger(k=15, sketch_size=200)
        too_few = [str(_cohort_sample(tmp_path, i)) for i in range(3)]

        with pytest.raises(ValueError) as exc_info:
            flagger.fit(too_few)

        message = str(exc_info.value)
        assert "3" in message and "4" in message

    def test_fit_rejects_a_duplicated_cohort_path(self, tmp_path, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200)
        duplicated = cohort_paths + [cohort_paths[0]]

        with pytest.raises(ValueError) as exc_info:
            flagger.fit(duplicated)

        assert repr(cohort_paths[0]) in str(exc_info.value)

    def test_fit_returns_self(self, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200)

        assert flagger.fit(cohort_paths) is flagger

    def test_repr_reports_fitted_state(self, cohort_paths):
        flagger = CohortOutlierFlagger(k=15, sketch_size=200)
        assert "unfitted" in repr(flagger)

        flagger.fit(cohort_paths)
        assert "8" in repr(flagger)
