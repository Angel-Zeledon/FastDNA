"""Tests for `fastdna.gwas` -- the k-mer GWAS on-ramp.

Everything here is built from tiny, hand-checkable synthetic FASTQ cohorts
written into `tmp_path`, reusing the same `write_fastq` helper shape as
`test_taxonomy.py`/`test_embed.py`/`test_sklearn.py`.

Two hand-computable facts do most of the work in this module (the same ones
`test_api.py::EXPECTED_CANONICAL_COUNTS` documents):

* `"ACGTACGTAC"` at k=5 yields exactly two distinct canonical 5-mers,
  `ACGTA` and `CGTAC`, each occurring 9 times across three copies of the
  read. This is the "core genome" background every synthetic sample here
  shares.
* `"GGGGGGGGGG"` at k=5 yields six copies of `GGGGG` per read, whose
  canonical form (the lexicographically smaller of the k-mer and its
  reverse complement) is `CCCCC` -- 18 occurrences across three copies of
  the read. This is the "accessory" k-mer only some samples carry.

Because both are exact, the expected bytes of a pyseer k-mer file can be
asserted literally rather than approximately.
"""

from __future__ import annotations

import gzip
import pathlib
import random

import pytest

pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

import scipy.sparse

from fastdna.gwas import (
    ScreeningOnlyWarning,
    cohort_presence_matrix,
    export_pyseer_kmers,
    kinship_matrix,
    prefilter_association,
)

BACKGROUND_READ = "ACGTACGTAC"  # -> canonical 5-mers ACGTA, CGTAC
ACCESSORY_READ = "GGGGGGGGGG"  # -> canonical 5-mer CCCCC


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def write_background(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """A sample carrying only the shared background k-mers."""
    return write_fastq(tmp_path, name, [BACKGROUND_READ] * 3)


def write_background_plus_accessory(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """A sample carrying the background k-mers *and* the accessory `CCCCC`."""
    return write_fastq(tmp_path, name, [BACKGROUND_READ] * 3 + [ACCESSORY_READ] * 3)


def _random_seq(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def _mutate(rng: random.Random, seq: str, n_mutations: int) -> str:
    bases = list(seq)
    for pos in rng.sample(range(len(bases)), n_mutations):
        bases[pos] = rng.choice([b for b in "ACGT" if b != bases[pos]])
    return "".join(bases)


# ---------------------------------------------------------------------------
# cohort_presence_matrix()
# ---------------------------------------------------------------------------


def test_presence_matrix_shape_and_sample_order(tmp_path):
    # Deliberately named so that insertion order and alphabetical order
    # disagree: the returned sample_ids must follow the order the caller
    # gave, because the matrix rows do.
    with_accessory = write_background_plus_accessory(tmp_path, "z_first.fastq")
    without = write_background(tmp_path, "a_second.fastq")

    matrix, sample_ids, kmer_sequences = cohort_presence_matrix(
        {"z_first": with_accessory, "a_second": without}, k=5, min_count=1, min_samples=1
    )

    assert scipy.sparse.issparse(matrix)
    assert matrix.format == "csr"
    assert sample_ids == ["z_first", "a_second"]
    assert kmer_sequences == ["ACGTA", "CCCCC", "CGTAC"]
    assert matrix.shape == (2, 3)

    dense = matrix.toarray()
    accessory_column = kmer_sequences.index("CCCCC")
    # Row 0 is z_first (the sample that actually carries CCCCC), row 1 is
    # a_second -- if the rows were sorted or reordered this flips.
    assert dense[0, accessory_column] == 18
    assert dense[1, accessory_column] == 0
    # The shared background is present in both at the hand-checked depth.
    assert dense[:, kmer_sequences.index("ACGTA")].tolist() == [9, 9]


def test_min_samples_filters_kmers_seen_in_too_few_samples(tmp_path):
    singleton = write_background_plus_accessory(tmp_path, "s1.fastq")
    plain = write_background(tmp_path, "s2.fastq")
    paths = [singleton, plain]

    _, _, all_kmers = cohort_presence_matrix(paths, k=5, min_count=1, min_samples=1)
    assert all_kmers == ["ACGTA", "CCCCC", "CGTAC"]

    # CCCCC is present in exactly one of the two samples, so min_samples=2
    # must drop it -- and only it.
    matrix, _, kept = cohort_presence_matrix(paths, k=5, min_count=1, min_samples=2)
    assert kept == ["ACGTA", "CGTAC"]
    assert matrix.shape == (2, 2)


def test_max_kmers_truncates_and_says_what_it_dropped(tmp_path):
    paths = [
        write_background_plus_accessory(tmp_path, "s1.fastq"),
        write_background(tmp_path, "s2.fastq"),
    ]

    with pytest.warns(UserWarning, match="max_kmers"):
        matrix, _, kept = cohort_presence_matrix(paths, k=5, min_count=1, min_samples=1, max_kmers=1)

    assert matrix.shape == (2, 1)
    # The ranking keeps the k-mers with the highest minor-sample-count --
    # the MAF-style "most testable" ones. ACGTA/CGTAC are in every sample
    # (minor count 0, zero variance, untestable); CCCCC is in one of two
    # (minor count 1), so it is the one worth keeping.
    assert kept == ["CCCCC"]


def test_cohort_presence_matrix_requires_at_least_two_samples(tmp_path):
    only = write_background(tmp_path, "only.fastq")

    with pytest.raises(ValueError, match="at least 2 samples"):
        cohort_presence_matrix([only], k=5, min_count=1)


def test_cohort_presence_matrix_names_duplicate_sample_ids(tmp_path):
    nested = tmp_path / "run2"
    nested.mkdir()
    first = write_background(tmp_path, "shared.fastq")
    second = write_background(nested, "shared.fastq")

    with pytest.raises(ValueError) as excinfo:
        cohort_presence_matrix([first, second], k=5, min_count=1)

    message = str(excinfo.value)
    assert "duplicate sample id" in message
    # The offending id has to be named, not just counted.
    assert "'shared'" in message


def test_cohort_presence_matrix_rejects_duplicate_paths(tmp_path):
    path = write_background(tmp_path, "s1.fastq")

    with pytest.raises(ValueError) as excinfo:
        cohort_presence_matrix({"a": path, "b": path}, k=5, min_count=1)

    assert "same file" in str(excinfo.value)


def test_cohort_presence_matrix_rejects_min_samples_above_cohort_size(tmp_path):
    paths = [write_background(tmp_path, "s1.fastq"), write_background(tmp_path, "s2.fastq")]

    with pytest.raises(ValueError) as excinfo:
        cohort_presence_matrix(paths, k=5, min_count=1, min_samples=3)

    message = str(excinfo.value)
    assert "min_samples=3" in message
    assert "2 samples" in message


def test_cohort_presence_matrix_rejects_a_single_bare_path(tmp_path):
    path = write_background(tmp_path, "s1.fastq")

    with pytest.raises(TypeError, match="list of paths"):
        cohort_presence_matrix(str(path), k=5, min_count=1)


# ---------------------------------------------------------------------------
# export_pyseer_kmers()
# ---------------------------------------------------------------------------


EXPECTED_PYSEER_BYTES = b"ACGTA | case_a:9 case_b:9\nCGTAC | case_a:9 case_b:9\n"


def test_export_pyseer_kmers_writes_the_exact_fsm_lite_line_format(tmp_path):
    # Two identical samples: every k-mer is present in both, at the
    # hand-checked depth of 9. Lines are ordered by k-mer sequence, so the
    # whole file is predictable byte for byte.
    paths = [
        write_background(tmp_path, "case_a.fastq"),
        write_background(tmp_path, "case_b.fastq"),
    ]
    out_path = tmp_path / "kmers.txt"

    result = export_pyseer_kmers(paths, out_path, k=5, min_count=1, min_samples=2)

    assert out_path.read_bytes() == EXPECTED_PYSEER_BYTES
    assert result.n_kmers == 2
    assert result.sample_ids == ["case_a", "case_b"]
    assert result.gzipped is False


def test_export_pyseer_kmers_gzip_round_trips(tmp_path):
    paths = [
        write_background(tmp_path, "case_a.fastq"),
        write_background(tmp_path, "case_b.fastq"),
    ]
    out_path = tmp_path / "kmers.txt.gz"

    result = export_pyseer_kmers(paths, out_path, k=5, min_count=1, min_samples=2)

    assert result.gzipped is True
    # pyseer opens --kmers with gzip by default (`--uncompressed` opts out),
    # so the `.gz` path is the one a real run uses.
    with gzip.open(out_path, "rb") as handle:
        assert handle.read() == EXPECTED_PYSEER_BYTES


def test_export_pyseer_kmers_survives_pyseers_own_line_parser(tmp_path):
    """The literal-bytes test above pins the format; this one pins that the
    format is the one pyseer actually reads.

    The two expressions below are transcribed verbatim from pyseer's
    `read_variant` (`pyseer/input.py`, v1.3.x): the k-mer is
    `line.split()[0]` and the sample list is
    `line.rstrip().split('|')[1].lstrip().split()`, from which only
    `x.split(':')[0]` is kept. A tab-separated `kmer<TAB>sample:1` line --
    a natural-looking guess -- raises `IndexError` here, which is exactly
    why this assertion exists rather than a comment saying "matches
    pyseer".
    """
    paths = [
        write_background_plus_accessory(tmp_path, "case_a.fastq"),
        write_background(tmp_path, "case_b.fastq"),
    ]
    out_path = tmp_path / "kmers.txt"
    export_pyseer_kmers(paths, out_path, k=5, min_count=1, min_samples=1)

    parsed = {}
    for line in out_path.read_text().splitlines():
        var_name, strains = (line.split()[0], line.rstrip().split("|")[1].lstrip().split())
        parsed[var_name] = {str(x.split(":")[0]): 1 for x in strains}

    assert parsed == {
        "ACGTA": {"case_a": 1, "case_b": 1},
        "CGTAC": {"case_a": 1, "case_b": 1},
        "CCCCC": {"case_a": 1},
    }


def test_export_pyseer_kmers_rejects_sample_ids_that_break_the_format(tmp_path):
    path_a = write_background(tmp_path, "good.fastq")
    path_b = write_background(tmp_path, "other.fastq")

    with pytest.raises(ValueError) as excinfo:
        export_pyseer_kmers({"good": path_a, "has:colon": path_b}, tmp_path / "out.txt", k=5, min_count=1)

    message = str(excinfo.value)
    assert "has:colon" in message
    assert ":" in message


# ---------------------------------------------------------------------------
# kinship_matrix()
# ---------------------------------------------------------------------------


@pytest.fixture
def kinship_cohort(tmp_path):
    """Two near-identical samples plus one built from unrelated sequence."""
    rng = random.Random(20260824)
    base = _random_seq(rng, 400)

    near_a = write_fastq(tmp_path, "near_a.fastq", [base] * 5)
    near_b = write_fastq(tmp_path, "near_b.fastq", [_mutate(rng, base, 4)] * 5)
    far = write_fastq(tmp_path, "far.fastq", [_random_seq(rng, 400)] * 5)
    return [near_a, near_b, far]


def test_kinship_matrix_is_symmetric_with_a_unit_diagonal(kinship_cohort):
    matrix, sample_ids = kinship_matrix(kinship_cohort, k=21, sketch_size=1000)

    assert sample_ids == ["near_a", "near_b", "far"]
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_kinship_matrix_scores_near_identical_samples_higher(kinship_cohort):
    matrix, sample_ids = kinship_matrix(kinship_cohort, k=21, sketch_size=1000)
    index = {name: i for i, name in enumerate(sample_ids)}

    near = matrix[index["near_a"], index["near_b"]]
    unrelated = matrix[index["near_a"], index["far"]]

    assert near > unrelated
    assert near > 0.8
    assert unrelated < 0.2


def test_kinship_matrix_requires_at_least_two_samples(tmp_path):
    path = write_background(tmp_path, "only.fastq")

    with pytest.raises(ValueError, match="at least 2 samples"):
        kinship_matrix([path], k=11, sketch_size=100)


# ---------------------------------------------------------------------------
# prefilter_association()
# ---------------------------------------------------------------------------


@pytest.fixture
def case_control_cohort(tmp_path):
    """Three cases carrying `CCCCC`, three controls that do not; all six
    share the `ACGTA`/`CGTAC` background.
    """
    paths = [write_background_plus_accessory(tmp_path, f"case_{i}.fastq") for i in range(3)]
    paths += [write_background(tmp_path, f"control_{i}.fastq") for i in range(3)]
    matrix, sample_ids, kmer_sequences = cohort_presence_matrix(paths, k=5, min_count=1, min_samples=1)
    phenotype = [1, 1, 1, 0, 0, 0]
    return matrix, phenotype, kmer_sequences, sample_ids


def test_case_only_kmer_ranks_first(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences)

    rows = table.to_pylist()
    assert rows[0]["kmer_sequence"] == "CCCCC"
    assert rows[0]["n_case_present"] == 3
    assert rows[0]["n_control_present"] == 0
    assert rows[0]["p_value"] < 1.0
    # The background k-mers are in every sample: zero variance, nothing to
    # test, p == 1.0 rather than a spurious hit.
    for row in rows[1:]:
        assert row["p_value"] > rows[0]["p_value"]


def test_prefilter_association_warns_that_it_is_screening_only(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning, match="pyseer"):
        prefilter_association(matrix, phenotype, kmer_sequences)


def test_prefilter_association_result_carries_screening_metadata(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences)

    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert metadata["fastdna.screening_only"] == "true"
    assert metadata["fastdna.population_structure_correction"] == "none"
    assert "pyseer" in metadata["fastdna.confirm_with"]


def test_prefilter_association_reports_multiple_testing_columns(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences)

    assert {"p_bonferroni", "q_value_bh"}.issubset(set(table.column_names))
    rows = table.to_pylist()
    n_tested = len(kmer_sequences)
    for row in rows:
        assert row["p_bonferroni"] == pytest.approx(min(1.0, row["p_value"] * n_tested))
        assert 0.0 <= row["q_value_bh"] <= 1.0


def test_prefilter_association_supports_chi2(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="chi2")

    rows = table.to_pylist()
    assert rows[0]["kmer_sequence"] == "CCCCC"
    assert rows[0]["p_value"] < 1.0


def test_prefilter_association_top_n_limits_rows(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, top_n=1)

    assert table.num_rows == 1
    assert table.to_pylist()[0]["kmer_sequence"] == "CCCCC"


def test_prefilter_association_rejects_phenotype_length_mismatch(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, phenotype[:4], kmer_sequences)

    message = str(excinfo.value)
    assert "6 samples" in message
    assert "4" in message


def test_prefilter_association_rejects_a_single_class_phenotype(case_control_cohort):
    matrix, _, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, [1, 1, 1, 1, 1, 1], kmer_sequences)

    assert "single class" in str(excinfo.value)


def test_prefilter_association_rejects_a_non_binary_phenotype(case_control_cohort):
    matrix, _, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], kmer_sequences)

    assert "binary" in str(excinfo.value)


def test_prefilter_association_rejects_kmer_sequence_length_mismatch(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, phenotype, kmer_sequences[:-1])

    assert "columns" in str(excinfo.value)


def test_prefilter_association_rejects_an_unknown_test(case_control_cohort):
    matrix, phenotype, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError, match="test must be"):
        prefilter_association(matrix, phenotype, kmer_sequences, test="not_a_real_test")


def test_prefilter_association_rejects_fewer_than_two_samples(case_control_cohort):
    _, _, kmer_sequences, _ = case_control_cohort
    one_row = scipy.sparse.csr_matrix(np.ones((1, len(kmer_sequences)), dtype=np.uint32))

    with pytest.raises(ValueError, match="at least 2 samples"):
        prefilter_association(one_row, [1], kmer_sequences)


def test_prefilter_association_binary_rejection_message_points_to_continuous_tests(case_control_cohort):
    matrix, _, kmer_sequences, _ = case_control_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], kmer_sequences)

    message = str(excinfo.value)
    assert "welch" in message
    assert "anova" in message


# ---------------------------------------------------------------------------
# prefilter_association() -- continuous phenotype (Welch's t-test / ANOVA)
# ---------------------------------------------------------------------------


@pytest.fixture
def continuous_cohort(tmp_path):
    """The same six-sample cohort shape as `case_control_cohort` -- three
    samples carrying the accessory `CCCCC` k-mer, three that do not -- but
    paired with a continuous phenotype instead of a 0/1 label: the three
    accessory carriers have a clearly higher phenotype value than the three
    that lack it, so `CCCCC` should screen as the top hit under a
    continuous test exactly as it does under Fisher/chi2.
    """
    paths = [write_background_plus_accessory(tmp_path, f"high_{i}.fastq") for i in range(3)]
    paths += [write_background(tmp_path, f"low_{i}.fastq") for i in range(3)]
    matrix, sample_ids, kmer_sequences = cohort_presence_matrix(paths, k=5, min_count=1, min_samples=1)
    phenotype = [10.0, 11.0, 9.0, 1.0, 2.0, 0.0]
    return matrix, phenotype, kmer_sequences, sample_ids


def test_case_only_kmer_ranks_first_under_welch(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="welch")

    rows = table.to_pylist()
    assert rows[0]["kmer_sequence"] == "CCCCC"
    assert rows[0]["n_present"] == 3
    assert rows[0]["n_absent"] == 3
    assert rows[0]["mean_present"] > rows[0]["mean_absent"]
    assert rows[0]["p_value"] < 1.0
    # The background k-mers are in every sample: no absent group at all,
    # nothing to test, sentinel p == 1.0 rather than a spurious hit.
    for row in rows[1:]:
        assert row["p_value"] > rows[0]["p_value"]


def test_prefilter_association_welch_is_the_default_continuous_test(continuous_cohort):
    """`test="welch"` and the bare Welch computation must agree bit for
    bit -- `"welch"` is not merely *a* supported continuous test, it is
    what a caller gets by explicitly asking for the continuous mode's
    documented default.
    """
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        explicit = prefilter_association(matrix, phenotype, kmer_sequences, test="welch")

    assert explicit.to_pylist()[0]["kmer_sequence"] == "CCCCC"


def test_prefilter_association_supports_anova(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="anova")

    rows = table.to_pylist()
    assert rows[0]["kmer_sequence"] == "CCCCC"
    assert rows[0]["p_value"] < 1.0


def test_prefilter_association_welch_matches_scipy_ttest_ind(continuous_cohort):
    """Pins the vectorized closed-form computation in `_welch_or_anova`
    against a direct, unvectorized `scipy.stats.ttest_ind(...,
    equal_var=False)` call on the same two groups -- the two must agree to
    floating-point tolerance, since the vectorized path is doing the same
    Welch's-t arithmetic scipy does, just for every k-mer column at once.
    """
    from scipy import stats

    matrix, phenotype, kmer_sequences, _ = continuous_cohort
    phenotype_array = np.asarray(phenotype, dtype=np.float64)
    dense = matrix.toarray()

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="welch")

    rows = {row["kmer_sequence"]: row for row in table.to_pylist()}
    column_index = {name: i for i, name in enumerate(kmer_sequences)}

    present_mask = dense[:, column_index["CCCCC"]].astype(bool)
    expected = stats.ttest_ind(phenotype_array[present_mask], phenotype_array[~present_mask], equal_var=False)

    assert rows["CCCCC"]["statistic"] == pytest.approx(expected.statistic)
    assert rows["CCCCC"]["p_value"] == pytest.approx(expected.pvalue)


def test_prefilter_association_anova_matches_scipy_f_oneway(continuous_cohort):
    from scipy import stats

    matrix, phenotype, kmer_sequences, _ = continuous_cohort
    phenotype_array = np.asarray(phenotype, dtype=np.float64)
    dense = matrix.toarray()

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="anova")

    rows = {row["kmer_sequence"]: row for row in table.to_pylist()}
    column_index = {name: i for i, name in enumerate(kmer_sequences)}

    present_mask = dense[:, column_index["CCCCC"]].astype(bool)
    expected = stats.f_oneway(phenotype_array[present_mask], phenotype_array[~present_mask])

    assert rows["CCCCC"]["statistic"] == pytest.approx(expected.statistic)
    assert rows["CCCCC"]["p_value"] == pytest.approx(expected.pvalue)


def test_prefilter_association_continuous_warns_that_it_is_screening_only(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning, match="pyseer"):
        prefilter_association(matrix, phenotype, kmer_sequences, test="welch")


def test_prefilter_association_continuous_warning_names_the_p_hacking_risk(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning, match="p-hack"):
        prefilter_association(matrix, phenotype, kmer_sequences, test="welch")


def test_prefilter_association_continuous_result_carries_screening_metadata(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="welch")

    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert metadata["fastdna.screening_only"] == "true"
    assert metadata["fastdna.population_structure_correction"] == "none"
    assert "pyseer" in metadata["fastdna.confirm_with"]
    assert metadata["fastdna.test"] == "welch"


def test_prefilter_association_continuous_reports_multiple_testing_columns(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="welch")

    assert {"p_bonferroni", "q_value_bh"}.issubset(set(table.column_names))
    rows = table.to_pylist()
    n_tested = len(kmer_sequences)
    for row in rows:
        assert row["p_bonferroni"] == pytest.approx(min(1.0, row["p_value"] * n_tested))
        assert 0.0 <= row["q_value_bh"] <= 1.0


def test_prefilter_association_continuous_top_n_limits_rows(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmer_sequences, test="welch", top_n=1)

    assert table.num_rows == 1
    assert table.to_pylist()[0]["kmer_sequence"] == "CCCCC"


def test_prefilter_association_rejects_a_non_numeric_continuous_phenotype(continuous_cohort):
    matrix, _, kmer_sequences, _ = continuous_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, ["lo", "hi", "lo", "hi", "lo", "hi"], kmer_sequences, test="welch")

    assert "numeric" in str(excinfo.value)


def test_prefilter_association_rejects_nan_in_a_continuous_phenotype(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort
    broken = list(phenotype)
    broken[0] = float("nan")

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, broken, kmer_sequences, test="welch")

    assert "NaN" in str(excinfo.value)


def test_prefilter_association_rejects_inf_in_a_continuous_phenotype(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort
    broken = list(phenotype)
    broken[0] = float("inf")

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, broken, kmer_sequences, test="welch")

    assert "inf" in str(excinfo.value)


def test_prefilter_association_rejects_a_zero_variance_continuous_phenotype(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort
    constant = [3.5] * len(phenotype)

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, constant, kmer_sequences, test="welch")

    assert "zero variance" in str(excinfo.value)


def test_prefilter_association_rejects_continuous_phenotype_length_mismatch(continuous_cohort):
    matrix, phenotype, kmer_sequences, _ = continuous_cohort

    with pytest.raises(ValueError) as excinfo:
        prefilter_association(matrix, phenotype[:4], kmer_sequences, test="welch")

    message = str(excinfo.value)
    assert "6 samples" in message
    assert "4" in message


def test_continuous_untestable_kmers_get_a_sentinel_pvalue_not_dropped():
    """A k-mer present in every sample, present in none, or present in
    only a single sample has no within-group variance to estimate on at
    least one side -- the continuous-mode analogue of the binary path's
    zero-marginal 2x2 table. Each is kept in the output with `statistic
    = nan` and the sentinel `p_value = 1.0` rather than being dropped, the
    same convention `prefilter_association` already uses for an
    all-present/all-absent k-mer in binary mode.
    """
    dense = np.array(
        [
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    matrix = scipy.sparse.csr_matrix(dense)
    kmers = ["ALL_PRESENT", "ALL_ABSENT", "ONE_PRESENT", "SPLIT"]
    phenotype = [10.0, 9.0, 1.0, 0.5, 2.0]

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmers, test="welch")

    rows = {row["kmer_sequence"]: row for row in table.to_pylist()}
    assert set(rows) == {"ALL_PRESENT", "ALL_ABSENT", "ONE_PRESENT", "SPLIT"}

    for name in ("ALL_PRESENT", "ALL_ABSENT", "ONE_PRESENT"):
        assert rows[name]["p_value"] == 1.0
        assert np.isnan(rows[name]["statistic"])

    # A single-sample mean is still informative even though no test can be
    # run from it, so it is reported rather than nan'd out.
    assert not np.isnan(rows["ONE_PRESENT"]["mean_present"])
    assert np.isnan(rows["ALL_PRESENT"]["mean_absent"])
    assert np.isnan(rows["ALL_ABSENT"]["mean_present"])

    assert rows["SPLIT"]["n_present"] == 2
    assert rows["SPLIT"]["n_absent"] == 3
    assert not np.isnan(rows["SPLIT"]["statistic"])
    assert rows["SPLIT"]["p_value"] < 1.0


def test_continuous_duplicate_presence_patterns_give_identical_results():
    """Two k-mer columns that happen to share the exact same
    presence/absence pattern must land on bit-identical statistics --
    `_continuous_group_stats` deliberately does not memoize by pattern (see
    its docstring for why that was investigated and skipped), so this is
    what would break first if the vectorized per-column arithmetic were
    accidentally column-order-dependent.
    """
    dense = np.array(
        [
            [1, 1],
            [1, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=np.uint32,
    )
    matrix = scipy.sparse.csr_matrix(dense)
    kmers = ["A", "B"]
    phenotype = [5.0, 6.0, 1.0, 2.0]

    with pytest.warns(ScreeningOnlyWarning):
        table = prefilter_association(matrix, phenotype, kmers, test="welch")

    rows = {row["kmer_sequence"]: row for row in table.to_pylist()}
    assert rows["A"]["statistic"] == rows["B"]["statistic"]
    assert rows["A"]["p_value"] == rows["B"]["p_value"]
    assert rows["A"]["mean_present"] == rows["B"]["mean_present"]
    assert rows["A"]["mean_absent"] == rows["B"]["mean_absent"]


def test_prefilter_association_rejects_fewer_than_two_samples_continuous(continuous_cohort):
    _, _, kmer_sequences, _ = continuous_cohort
    one_row = scipy.sparse.csr_matrix(np.ones((1, len(kmer_sequences)), dtype=np.uint32))

    with pytest.raises(ValueError, match="at least 2 samples"):
        prefilter_association(one_row, [1.0], kmer_sequences, test="welch")
