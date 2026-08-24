"""Audit tests for `fastdna.taxonomy` (commits 2b23f44, merge a0a744f).

Complements `test_taxonomy.py`, which exercises only the extreme ends of
every scale it touches: `check_sample_identity` is tested at
`mash_distance == 0.0` (byte-identical files) and at `mash_distance == 1.0`
(fully disjoint k-mer content), so the `threshold` comparison that decides
every real-world borderline case is never actually evaluated anywhere in
between. The tests here fill in the middle, plus the branches
`test_taxonomy.py` never reaches at all (`.gz` name derivation,
`gather(max_references=...)`, and a query that matches nothing).

Tests whose docstring begins with "EXPECTED TO FAIL" pin a reported
defect and are expected to be red until the module is patched or reverted;
see the audit report. They are deliberately *not* xfail-marked, so the
defect stays visible in the suite's own output rather than being absorbed
into an "expected failure" tally.

All fixtures are seeded, so every number quoted in a docstring is
reproducible.
"""
from __future__ import annotations

import math
import pathlib
import random

import pytest

import fastdna
from fastdna.taxonomy import (
    _default_name,
    build_reference_database,
    check_sample_identity,
    classify,
    gather,
)


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list) -> str:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return str(p)


def random_loci(seed: int, count: int, length: int) -> list:
    """`count` independent random sequences of `length` bases. Distinct
    loci share essentially no k-mers at k >= 15, so set overlap between two
    samples is controlled exactly by which loci each one is built from.
    """
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=length)) for _ in range(count)]


# ---------------------------------------------------------------------------
# check_sample_identity: what does `threshold` actually mean?
# ---------------------------------------------------------------------------


@pytest.fixture
def half_shared_pair(tmp_path):
    """Two "samples" built so that their k-mer sets overlap by roughly a
    third: sample A is loci 0-9, sample B is loci 0-4 plus loci 10-14. Five
    loci of fifteen are common, so Jaccard ~= 5/15 = 0.33.

    This is a *sample swap*, not two runs of one specimen: two thirds of
    each file's content is absent from the other.
    """
    loci = random_loci(seed=7, count=20, length=60)
    a = write_fastq(tmp_path, "run_a.fastq", [loci[i] for i in range(10)] * 10)
    b = write_fastq(
        tmp_path,
        "run_b.fastq",
        [loci[i] for i in list(range(5)) + list(range(10, 15))] * 10,
    )
    return a, b


def test_half_shared_pair_really_does_share_only_about_a_third(half_shared_pair):
    """Guard rail for the two tests below: if this fixture ever stops
    producing a ~0.33-Jaccard pair, those tests stop measuring what their
    docstrings say they measure.
    """
    a, b = half_shared_pair
    sketch_a = fastdna.sketch(a, k=21)
    sketch_b = fastdna.sketch(b, k=21)

    assert sketch_a.jaccard(sketch_b) == pytest.approx(0.33, abs=0.05)


def test_identity_threshold_means_the_same_thing_for_both_metrics(half_shared_pair):
    """EXPECTED TO FAIL -- pins a reported defect.

    `check_sample_identity`'s docstring claims:

        "Both metrics are compared against `threshold` on the same 'closer
        to 1 is closer to identical' scale, so one threshold parameter
        works for either"

    and describes `threshold=0.9` as "requiring the two files to agree on
    roughly 90% of the chosen metric's signal".

    That claim is false. `1 - mash_distance` is not a rescaling of Jaccard;
    it is `1 + (1/k) * ln(2J / (1+J))`, a logarithm that compresses the
    whole low-Jaccard range into the top of the [0, 1] interval. At k=21,
    `1 - mash_distance >= 0.9` is satisfied by any pair with `J >= 0.066`.

    So at one and the same `threshold=0.9`, the default metric accepts a
    pair sharing 6.6% of its k-mers as "the same biological sample", while
    `metric="jaccard"` demands 90% -- roughly a 14x difference in
    strictness between two settings the docstring presents as
    interchangeable.

    Measured on this fixture (J = 0.333): mash_distance = 0.033, so
    `1 - 0.033 = 0.967 >= 0.9` and the default reports `same_sample=True`
    for a pair that has two thirds of its content different. `jaccard`
    reports `same_sample=False` on the same files and the same threshold.
    """
    a, b = half_shared_pair

    by_mash = check_sample_identity(a, b, k=21, threshold=0.9)
    by_jaccard = check_sample_identity(a, b, k=21, threshold=0.9, metric="jaccard")

    assert by_mash.same_sample == by_jaccard.same_sample, (
        "the two metrics disagree at the same threshold on the same files: "
        f"mash_distance={by_mash.score:.4f} -> same_sample={by_mash.same_sample}; "
        f"jaccard={by_jaccard.score:.4f} -> same_sample={by_jaccard.same_sample}"
    )


def test_identity_threshold_is_independent_of_k():
    """EXPECTED TO FAIL -- pins a reported defect.

    A same-sample decision must be a property of the two samples, not of
    the k-mer size the check happened to run at. Because the default metric
    is a *distance* whose magnitude carries a `1/k` factor, the default
    `threshold=0.9` silently means a different amount of shared content at
    every `k`:

        k=15  ->  requires J >= 0.126
        k=21  ->  requires J >= 0.066
        k=31  ->  requires J >= 0.023

    This test builds one pair of files with J = 0.053 (fixed by
    construction: 2 shared loci out of 38) and asks the same question at
    k=15, k=21 and k=31 with the same threshold. The verdict flips from
    False to False to True.

    A lab that raised its k from 21 to 31 for unrelated reasons would
    silently stop detecting a class of sample swap it used to catch.
    """
    loci = random_loci(seed=11, count=60, length=60)
    tmp = pathlib.Path(__import__("tempfile").mkdtemp())
    a = write_fastq(tmp, "a.fastq", [loci[i] for i in range(20)] * 6)
    b = write_fastq(tmp, "b.fastq", [loci[i] for i in [0, 1] + list(range(20, 38))] * 6)

    verdicts = {
        k: check_sample_identity(a, b, k=k, sketch_size=2000).same_sample
        for k in (15, 21, 31)
    }

    assert len(set(verdicts.values())) == 1, (
        "the same pair of files, the same threshold, three k values, "
        f"three different answers: {verdicts}"
    )


def test_mash_distance_and_jaccard_scales_are_documented_correctly():
    """EXPECTED TO FAIL -- pins the arithmetic behind the defect above,
    independently of any FASTQ file.

    Recomputes `1 - mash_distance` directly from Mash's own formula
    (Ondov et al. 2016, as implemented in `src/sketch.rs::mash_distance`)
    for a Jaccard of 0.1, and checks it against the docstring's claim that
    `threshold` measures "roughly 90% of the chosen metric's signal" on a
    common scale.

    If the two scales were interchangeable, a pair with J = 0.1 -- 10% of
    the signal -- would have to sit near 0.1 on the mash scale too. It
    sits at 0.919.
    """
    k = 21
    j = 0.1
    mash_distance = -(1.0 / k) * math.log(2 * j / (1 + j))
    mash_similarity = 1.0 - mash_distance

    assert mash_similarity == pytest.approx(j, abs=0.1), (
        f"J={j} maps to 1-mash_distance={mash_similarity:.4f}; a single "
        "`threshold` cannot mean the same thing on both scales"
    )


# ---------------------------------------------------------------------------
# classify: no-match and small-vs-large behaviour
# ---------------------------------------------------------------------------


def test_classify_returns_a_named_reference_even_when_nothing_matches(tmp_path):
    """Characterisation test (currently GREEN): records what `classify`
    does for a sample that matches nothing in the reference set.

    There is no "unknown" outcome and no minimum score: the query below
    shares zero k-mers with all three references, and `classify` still
    returns a full, score-ordered table naming a reference first. A caller
    that reads `result.column("name")[0]` -- the obvious way to use a
    function whose docstring says it "ranks the references ... by how well
    each one explains the query" -- gets a confident-looking pathogen name
    for a sample with no evidence behind it.

    The saving grace, pinned here so a change is noticed: the score really
    is exactly 0.0, so a caller who checks it can tell. Contrast `gather`,
    which has a `min_containment` floor and correctly returns nothing --
    also pinned below.
    """
    unrelated = random_loci(seed=3, count=4, length=200)
    query = write_fastq(tmp_path, "patient_sample.fastq", [unrelated[0]] * 30)
    db = build_reference_database(
        {
            "ebola": write_fastq(tmp_path, "ebola.fastq", [unrelated[1]] * 30),
            "sars_cov_2": write_fastq(tmp_path, "sars.fastq", [unrelated[2]] * 30),
            "influenza_a": write_fastq(tmp_path, "flu.fastq", [unrelated[3]] * 30),
        },
        k=21,
    )

    result = classify(query, db, k=21)

    assert result.num_rows == 3, "no floor: every reference is reported"
    assert result.column("score").to_pylist() == [0.0, 0.0, 0.0]
    assert result.column("name").to_pylist()[0] in {"ebola", "sars_cov_2", "influenza_a"}

    # gather, given the identical inputs, declines to name anything.
    assert gather(query, db, k=21).num_rows == 0


def test_containment_estimate_is_usable_when_query_and_reference_differ_in_size():
    """EXPECTED TO FAIL -- pins a reported defect.

    `classify`'s score is `Sketch.containment`, a bottom-k estimator whose
    denominator is not the query's sketch size but the number of query
    hashes below the reference sketch's own ceiling hash
    (`src/sketch.rs::containment`, the `resolvable` counter). When the
    reference's underlying genome is much larger than the query's, a
    `sketch_size=1000` reference sketch has a very low ceiling, and almost
    all of the query's 1000 hashes fall above it and are discarded from
    both numerator and denominator.

    The reported score is then a ratio of a handful of observations,
    printed to full float precision with nothing marking it as noise.
    Neither `classify` nor `build_reference_database` scales `sketch_size`
    with reference size, and neither docstring warns about this.

    Measured here: a reference ~100x the query's size, eight independent
    queries each built to have a *true* containment of exactly 0.5 (ten
    loci drawn from the reference, ten novel loci). Observed estimates
    span 0.125 to 0.667 -- absolute errors up to 0.375 on a [0, 1] score,
    i.e. any ranking between two references whose true containments differ
    by less than ~0.4 is decided by sampling noise.

    The same construction at a 10x size ratio stays within 0.07, so this
    is specifically a size-ratio effect, not a broken estimator.
    """
    loci = random_loci(seed=1234, count=2000, length=300)
    tmp = pathlib.Path(__import__("tempfile").mkdtemp())

    reference = write_fastq(tmp, "large_reference.fastq", loci)
    reference_sketch = fastdna.sketch(reference, k=21, sketch_size=1000)

    observed = []
    for seed in range(8):
        rng = random.Random(500 + seed)
        inside = [loci[i] for i in rng.sample(range(2000), 10)]
        outside = ["".join(rng.choices("ACGT", k=300)) for _ in range(10)]
        query = write_fastq(tmp, f"small_query_{seed}.fastq", inside + outside)
        observed.append(
            fastdna.sketch(query, k=21, sketch_size=1000).containment(reference_sketch)
        )

    worst = max(abs(value - 0.5) for value in observed)
    assert worst <= 0.15, (
        "containment estimates for a true containment of 0.5 against a "
        f"~100x larger reference: {[round(v, 3) for v in observed]} "
        f"(worst absolute error {worst:.3f})"
    )


# ---------------------------------------------------------------------------
# Branches `test_taxonomy.py` never reaches
# ---------------------------------------------------------------------------


def test_default_name_strips_a_gz_suffix_as_documented():
    """`_default_name`'s docstring promises `"ecoli.fastq.gz" -> "ecoli"`,
    but no test in `test_taxonomy.py` ever passes a `.gz` filename, so the
    `.gz` branch was never executed. Pinned here.
    """
    assert _default_name("ecoli.fastq.gz") == "ecoli"
    assert _default_name("ecoli.fastq") == "ecoli"
    assert _default_name("/some/dir/GCF_000005845.2.fastq.gz") == "GCF_000005845.2"


def test_gather_max_references_truncates_the_pick_list(tmp_path):
    """`gather`'s `max_references` parameter is never passed by any test in
    `test_taxonomy.py`, so its loop guard was only ever evaluated with
    `None`. Pinned here.
    """
    motifs = random_loci(seed=21, count=3, length=200)
    query = write_fastq(tmp_path, "mixed.fastq", [m for m in motifs for _ in range(20)])
    db = build_reference_database(
        {
            f"organism_{i}": write_fastq(tmp_path, f"org_{i}.fastq", [m] * 20)
            for i, m in enumerate(motifs)
        },
        k=21,
        sketch_size=2000,
    )

    unbounded = gather(query, db, k=21, sketch_size=2000, min_containment=0.05)
    bounded = gather(
        query, db, k=21, sketch_size=2000, min_containment=0.05, max_references=1
    )

    assert unbounded.num_rows >= 2
    assert bounded.num_rows == 1
    assert bounded.column("name").to_pylist()[0] == unbounded.column("name").to_pylist()[0]


def test_classify_score_column_is_the_query_side_of_containment(tmp_path):
    """Direction check for the asymmetric metric: `classify` must compute
    `query.containment(reference)` ("what fraction of the *query* is
    explained by this reference"), not the reverse.

    A tiny query fully inside a much larger reference scores 1.0 the right
    way round and near 0 the wrong way round, so this test would fail
    loudly if the two operands were ever swapped.
    """
    loci = random_loci(seed=99, count=40, length=200)
    query = write_fastq(tmp_path, "small_query.fastq", [loci[0]] * 20)
    reference = write_fastq(tmp_path, "big_reference.fastq", loci * 5)

    db = build_reference_database({"big": reference}, k=21, sketch_size=5000)
    score = classify(query, db, k=21, sketch_size=5000).column("score").to_pylist()[0]

    query_sketch = fastdna.sketch(query, k=21, sketch_size=5000)
    reference_sketch = db["big"]

    assert score == pytest.approx(query_sketch.containment(reference_sketch))
    assert score > 0.9, "the query is wholly inside the reference"
    assert reference_sketch.containment(query_sketch) < 0.2, (
        "the reverse containment must be small -- if `classify` ever "
        "returned this number instead, the assertion above would catch it"
    )
