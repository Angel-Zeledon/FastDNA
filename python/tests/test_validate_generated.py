"""Tests for fastdna.validate_generated -- plausibility checks for
generated/synthetic DNA against a reference's k-mer composition, repeat
structure, and containment.

The composition check (Jensen-Shannon distance over per-k-mer relative
frequencies) is exercised with a real, deterministic positive/negative
control: "realistic" sequences are mutated copies of a strongly
base-composition-biased reference (mimicking a generative model that
learned local statistics correctly), and "implausible" sequences are drawn
with uniform per-base probability -- exactly the "uniform random k-mer
frequencies, which real genomes never have" case named by the task this
file backs. Real DNA is never perfectly uniform at the base level (GC
content skew is universal), so a strong, deliberate base-composition bias
in the reference is what makes this a real, non-tautological test rather
than a smoke test.
"""
from __future__ import annotations

import gzip
import pathlib
import random

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from fastdna.validate_generated import GenerativeValidationReport, validate_generated  # noqa: E402


def _weighted_sequence(length: int, weights: list[float], rng: random.Random) -> str:
    return "".join(rng.choices("ACGT", weights=weights, k=length))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    bases = "ACGT"
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice(bases)
    return "".join(out)


# A strongly AT-biased "organism" (90% A/T, 10% G/C) -- real genomes are
# never base-composition-uniform, but this bias is deliberately extreme so
# the test signal does not depend on subtle real-genome statistics.
_BIASED_WEIGHTS = [0.45, 0.05, 0.05, 0.45]  # A, C, G, T
_UNIFORM_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

_REFERENCE = _weighted_sequence(4000, _BIASED_WEIGHTS, random.Random(1))


def _biased_cohort(n_sequences: int, seed: int) -> list[str]:
    """Mutated copies of `_REFERENCE` -- same organism, same composition
    bias, standing in for "reference genomes" and for a generative model
    that captured the reference's local statistics correctly.
    """
    rng = random.Random(seed)
    return [_mutate(_REFERENCE, 0.02, rng) for _ in range(n_sequences)]


def _uniform_cohort(n_sequences: int, length: int, seed: int) -> list[str]:
    """Sequences with uniform per-base probability and no relationship to
    `_REFERENCE` at all -- the "obviously implausible" generated batch:
    real DNA is never base-composition-uniform, and unlike `_biased_cohort`
    this shares no ancestry with the reference either.
    """
    rng = random.Random(seed)
    return [_weighted_sequence(length, _UNIFORM_WEIGHTS, rng) for _ in range(n_sequences)]


def write_fasta(tmp_path: pathlib.Path, name: str, sequences: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f">seq{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    return p


def write_fasta_gz(tmp_path: pathlib.Path, name: str, sequences: list[str]) -> pathlib.Path:
    p = tmp_path / name
    text = "".join(f">seq{i}\n{seq}\n" for i, seq in enumerate(sequences))
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    return p


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_empty_generated_sequences():
    with pytest.raises(ValueError, match="generated"):
        validate_generated([], _biased_cohort(4, seed=1))


def test_rejects_empty_reference_sequences():
    with pytest.raises(ValueError, match="reference"):
        validate_generated(_biased_cohort(4, seed=1), [])


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        validate_generated(_biased_cohort(2, seed=1), _biased_cohort(2, seed=2), k=0)


def test_rejects_non_positive_repeat_k():
    with pytest.raises(ValueError):
        validate_generated(
            _biased_cohort(2, seed=1), _biased_cohort(2, seed=2), k=(3,), repeat_k=0
        )


# ---------------------------------------------------------------------------
# Composition check: the positive/negative control
# ---------------------------------------------------------------------------


def test_realistic_generated_sequences_pass_composition_check():
    reference = _biased_cohort(10, seed=100)
    realistic_generated = _biased_cohort(10, seed=200)

    report = validate_generated(realistic_generated, reference, k=(3, 6), check_containment=False)

    rows = {row["k"]: row for row in report.composition.to_pylist()}
    assert rows[3]["verdict"] == "realistic"
    assert rows[3]["js_distance"] < 0.10
    assert rows[6]["verdict"] == "realistic"


def test_uniform_random_generated_sequences_fail_composition_check():
    reference = _biased_cohort(10, seed=100)
    implausible_generated = _uniform_cohort(10, length=4000, seed=300)

    report = validate_generated(implausible_generated, reference, k=(3, 6), check_containment=False)

    rows = {row["k"]: row for row in report.composition.to_pylist()}
    assert rows[3]["verdict"] != "realistic"
    assert rows[6]["verdict"] != "realistic"


def test_uniform_random_diverges_more_than_realistic_at_every_k():
    """A relative comparison, robust to the exact default threshold bands:
    whatever the verdict labels say, the implausible batch must score a
    strictly larger JS distance than the realistic batch at every k.
    """
    reference = _biased_cohort(10, seed=100)
    realistic_generated = _biased_cohort(10, seed=200)
    implausible_generated = _uniform_cohort(10, length=4000, seed=300)

    k_values = (3, 6, 11, 21)
    good = validate_generated(realistic_generated, reference, k=k_values, check_containment=False)
    bad = validate_generated(implausible_generated, reference, k=k_values, check_containment=False)

    good_by_k = dict(zip(good.composition.column("k").to_pylist(), good.composition.column("js_distance").to_pylist()))
    bad_by_k = dict(zip(bad.composition.column("k").to_pylist(), bad.composition.column("js_distance").to_pylist()))

    for kv in k_values:
        assert bad_by_k[kv] > good_by_k[kv], f"k={kv}: expected implausible batch to diverge more"


def test_composition_reports_one_row_per_requested_k_in_order():
    reference = _biased_cohort(4, seed=1)
    generated = _biased_cohort(4, seed=2)

    report = validate_generated(generated, reference, k=(6, 3, 11), check_containment=False)

    assert report.composition.column("k").to_pylist() == [6, 3, 11]


def test_a_single_int_k_is_accepted_like_a_one_element_sequence():
    reference = _biased_cohort(4, seed=1)
    generated = _biased_cohort(4, seed=2)

    report = validate_generated(generated, reference, k=6, check_containment=False)

    assert report.composition.column("k").to_pylist() == [6]


# ---------------------------------------------------------------------------
# Containment (novelty / memorization)
# ---------------------------------------------------------------------------


def test_a_verbatim_copy_of_reference_scores_high_containment():
    reference = _biased_cohort(8, seed=100)
    # One of the "generated" sequences is an exact copy of a reference
    # sequence -- the memorization case.
    generated = [reference[0]] + _biased_cohort(3, seed=400)

    report = validate_generated(generated, reference, k=(11,), containment_k=15)

    assert report.mean_containment is not None
    assert report.high_containment_count >= 1


def test_unrelated_sequences_score_low_containment():
    reference = _biased_cohort(8, seed=100)
    unrelated = _uniform_cohort(4, length=4000, seed=500)

    report = validate_generated(unrelated, reference, k=(11,), containment_k=15)

    assert report.low_containment_count == len(unrelated)


def test_check_containment_false_skips_the_containment_computation():
    reference = _biased_cohort(4, seed=1)
    generated = _biased_cohort(4, seed=2)

    report = validate_generated(generated, reference, k=(6,), check_containment=False)

    assert report.mean_containment is None
    assert report.high_containment_count is None
    assert report.low_containment_count is None


# ---------------------------------------------------------------------------
# Repeat / coverage structure (smoke-level: the fit's convergence is an
# optimizer outcome, not something to pin exact boolean values on for tiny
# synthetic batches -- see the module docstring's own honesty caveat about
# what a False here can mean).
# ---------------------------------------------------------------------------


def test_repeat_structure_fields_are_populated_and_repeat_k_defaults_to_max_k():
    reference = _biased_cohort(8, seed=100)
    generated = _biased_cohort(8, seed=200)

    report = validate_generated(generated, reference, k=(3, 6, 11), check_containment=False)

    assert isinstance(report.generated_repeat_structure_found, bool)
    assert isinstance(report.reference_repeat_structure_found, bool)
    assert report.repeat_k == 11


def test_explicit_repeat_k_overrides_the_max_k_default():
    reference = _biased_cohort(8, seed=100)
    generated = _biased_cohort(8, seed=200)

    report = validate_generated(
        generated, reference, k=(3, 6, 11), repeat_k=6, check_containment=False
    )

    assert report.repeat_k == 6


# ---------------------------------------------------------------------------
# Input shapes: raw sequences, FASTA paths, gzipped FASTA paths
# ---------------------------------------------------------------------------


def test_accepts_fasta_file_paths(tmp_path):
    reference_path = write_fasta(tmp_path, "reference.fasta", _biased_cohort(6, seed=100))
    generated_path = write_fasta(tmp_path, "generated.fasta", _biased_cohort(6, seed=200))

    report = validate_generated(generated_path, reference_path, k=(6,), check_containment=False)

    assert report.generated_n_sequences == 6
    assert report.reference_n_sequences == 6


def test_accepts_gzipped_fasta_file_paths(tmp_path):
    reference_path = write_fasta_gz(tmp_path, "reference.fasta.gz", _biased_cohort(6, seed=100))
    generated_path = write_fasta_gz(tmp_path, "generated.fasta.gz", _biased_cohort(6, seed=200))

    report = validate_generated(generated_path, reference_path, k=(6,), check_containment=False)

    assert report.generated_n_sequences == 6
    assert report.reference_n_sequences == 6


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_report_repr_and_markdown_render_without_error():
    reference = _biased_cohort(6, seed=100)
    generated = _biased_cohort(6, seed=200)

    report = validate_generated(generated, reference, k=(3, 6), containment_k=15)

    assert isinstance(report, GenerativeValidationReport)
    assert "generated" in repr(report)
    markdown = str(report)
    assert "K-mer composition" in markdown
    assert "Repeat / coverage structure" in markdown
    assert "Novelty / memorization" in markdown  # check_containment defaults True
