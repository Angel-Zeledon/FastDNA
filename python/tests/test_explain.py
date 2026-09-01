"""fastdna.explain(): the three checks that separate a credible feature
from a false-causal one.

Each check gets its own positive control: a fixture constructed so that
one specific verdict is the only correct answer, and the test fails if
`explain()` reports anything else. Without these, "explain() runs" would
be unfalsifiable -- it would also pass against a version that always
reported "credible candidate".
"""
from __future__ import annotations

import pytest

import fastdna

# Guarded, and in this order, on purpose -- see
# test_optional_dependencies.py::test_the_test_suite_itself_collects_in_the_environment_ci_builds,
# which exists specifically to catch a module reintroducing an unguarded
# `import numpy` above these skips.
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from fastdna.explain import _cochran_mantel_haenszel, explain  # noqa: E402
from fastdna.sklearn import KmerVectorizer  # noqa: E402


def _write(tmp_path, name, reads):
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return str(p)


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _canonical(seq: str) -> str:
    """The same canonical form `fastdna` itself uses: the lexicographically
    smaller of a sequence and its reverse complement. Markers below are
    built to be exactly `k` bases, so this is what actually shows up in
    `KmerVectorizer._feature_sequences_` -- a raw forward slice is not
    guaranteed to be its own canonical form.
    """
    rc = seq.translate(_COMPLEMENT)[::-1]
    return min(seq, rc)


def _lineage_cohort(tmp_path, n_lineages=4, per_lineage=6, read_len=120, seed=0):
    """A cohort with real lineage structure: each lineage is a distinct
    random root sequence, and its members are point-mutated copies of it
    -- the same construction used elsewhere in this project's own test
    suite for exercising `fastdna.cv.lineage_groups`. Returns
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


# ---------------------------------------------------------------------------
# Check 1: identifiability (equivalence classes)
# ---------------------------------------------------------------------------


def test_kmers_from_a_shared_repeated_block_are_flagged_non_identifiable(tmp_path):
    """Two adjacent k-mers that always co-occur (part of the same repeated
    block) must land in the same equivalence class and be reported
    non-identifiable -- neither is more "causal" than the other.
    """
    k = 8
    # A 30-base block, present or absent as a whole in every read of a
    # sample: every k-mer inside it shares exactly the same presence
    # pattern across the cohort.
    block = "ACGTTGCAAACCGGTTGATTACAGATTAC"
    filler_a = "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    filler_b = "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"

    paths = []
    for i in range(6):
        has_block = i % 2 == 0
        reads = [block + filler_a] * 3 if has_block else [filler_b] * 3
        paths.append(_write(tmp_path, f"S{i}.fastq", reads))

    vec = KmerVectorizer(k=k, top_features=None, representation="presence").fit(paths)
    importances = np.ones(len(vec.vocabulary_))
    # A phenotype exactly matching "has the block" for a well-defined run.
    phenotype = np.array([1, 0, 1, 0, 1, 0])

    report = explain(vec, importances, paths, phenotype, top_n=len(vec.vocabulary_))

    # At least one k-mer entirely inside `block` must be reported with an
    # equivalence class larger than 1 (it shares its pattern with its
    # neighbors inside the same always-co-occurring block).
    class_sizes = [f.equivalence_class_size for f in report.features]
    assert max(class_sizes) > 1, "expected at least one multi-member equivalence class"
    non_identifiable = [f for f in report.features if f.equivalence_class_size > 1]
    assert non_identifiable, "no feature was flagged non-identifiable"
    for f in non_identifiable:
        assert f.verdict.startswith("non-identifiable")


def test_a_feature_with_a_unique_pattern_is_identifiable(tmp_path):
    """A feature present in a pattern no other k-mer shares must report an
    equivalence class of exactly 1.

    The marker is exactly `k` bases long, deliberately: a *longer* marker
    would produce several overlapping k-mers that all share the same
    presence pattern by construction (they are substrings of the same
    fixed text) and would therefore correctly form a multi-member
    equivalence class -- that is `explain()` working as intended, not a
    unique-pattern fixture. Isolating "unique presence" from "unique
    sequence" needs a marker that is only ever one k-mer.
    """
    k = 8
    marker = "ACGTTGCA"
    canonical_marker = _canonical(marker)
    # A background sequence shared verbatim by every sample, so its k-mers
    # all carry the "present in every sample" pattern -- distinct from the
    # marker's "present in sample 0 alone" pattern, and not exclusive to
    # any one sample the way an otherwise-empty read would be.
    background = "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAA"

    paths = []
    for i in range(6):
        reads = [background]
        if i == 0:
            reads.append(marker)  # exactly k bases: exactly one k-mer
        paths.append(_write(tmp_path, f"S{i}.fastq", reads))

    vec = KmerVectorizer(k=k, top_features=None, representation="presence").fit(paths)
    assert canonical_marker in vec._feature_sequences_, "fixture must actually produce the marker k-mer"
    marker_index = vec._feature_sequences_.index(canonical_marker)

    importances = np.zeros(len(vec.vocabulary_))
    importances[marker_index] = 1.0
    phenotype = np.array([1, 0, 0, 0, 0, 0])

    report = explain(vec, importances, paths, phenotype, top_n=1)
    assert report.features[0].kmer == canonical_marker
    assert report.features[0].equivalence_class_size == 1
    assert report.features[0].identifiable


# ---------------------------------------------------------------------------
# Check 2: lineage attribution
# ---------------------------------------------------------------------------


def test_a_lineage_specific_kmer_is_flagged_as_a_lineage_marker(tmp_path):
    """POSITIVE CONTROL. A marker present in only one of several lineages,
    with a phenotype that has nothing to do with the marker, must be
    flagged a lineage marker -- the marker's real information content is
    "which lineage", not "which phenotype".

    Deliberately does NOT reuse `_lineage_cohort`'s randomly-mutated
    backgrounds: sparse per-sample mutations there create a moving target
    of conserved-background k-mers whose exact presence subset shifts with
    every choice of which samples carry a marker, which made earlier
    versions of this test coincidentally collide with a real background
    k-mer's pattern (a genuine multi-member equivalence class, just not
    the one this test is trying to isolate). Every lineage here is instead
    an *identical* sequence repeated verbatim across its own members --
    still five distinguishable clusters for `lineage_groups` to find (Mash
    distance 0 within a lineage, large between lineages), with no sparse
    mutation noise to produce an unplanned coincidental match.
    """
    k = 21
    rng = np.random.default_rng(1)
    bases = np.array(list("ACGT"))
    n_lineages, per_lineage, read_len = 5, 4, 120

    paths, lineage_of = [], []
    for lineage in range(n_lineages):
        root = "".join(rng.choice(bases, size=read_len * 2))
        reads = [root[i : i + read_len] for i in range(0, len(root) - read_len, read_len // 2)]
        for member in range(per_lineage):
            path = _write(tmp_path, f"L{lineage}_S{member}.fastq", reads)
            paths.append(path)
            lineage_of.append(lineage)
    lineage_of = np.array(lineage_of)

    # Exactly k bases: a longer marker would produce several overlapping
    # k-mers that necessarily share one presence pattern (see
    # test_a_feature_with_a_unique_pattern_is_identifiable), which would
    # make every one of them non-identifiable rather than isolating the
    # lineage-restriction check this test is actually about.
    marker = "ACGTTGCAAACCGGTTGATTA"
    assert len(marker) == k
    canonical_marker = _canonical(marker)
    # Appended to exactly one sample of lineage 0: since every lineage
    # here is one fixed sequence repeated verbatim across its members,
    # every real background k-mer is present either in an entire lineage
    # or nowhere -- never in a strict subset of one. A marker exclusive to
    # a single sample therefore cannot coincidentally match any
    # background k-mer's pattern.
    lineage_0_paths = [p for p, lineage in zip(paths, lineage_of) if lineage == 0]
    with open(lineage_0_paths[0], "a") as fh:
        fh.write(f"@marker\n{marker}\n+\n{'I' * len(marker)}\n")

    vec = KmerVectorizer(k=k, top_features=None, representation="presence").fit(paths)
    assert canonical_marker in vec._feature_sequences_, "fixture must produce the marker k-mer"
    marker_kmer = canonical_marker
    marker_index = vec._feature_sequences_.index(marker_kmer)

    importances = np.zeros(len(vec.vocabulary_))
    importances[marker_index] = 1.0
    # Phenotype independent of lineage (alternating), so any association
    # the marker shows is purely a lineage artifact, not real signal.
    pheno_rng = np.random.default_rng(2)
    phenotype = pheno_rng.integers(0, 2, size=len(paths))

    report = explain(vec, importances, paths, phenotype, top_n=1, lineage_threshold=0.02)
    feature = report.features[0]
    assert feature.kmer == marker_kmer
    assert feature.equivalence_class_size == 1, (
        f"the marker's presence pattern (1 sample) must be unique; got a class of "
        f"{feature.equivalence_class_size}"
    )
    assert feature.n_lineages_present == 1, (
        f"the marker should appear in exactly 1 lineage, got {feature.n_lineages_present}"
    )
    assert feature.lineage_restricted
    assert feature.verdict == "lineage marker"


# ---------------------------------------------------------------------------
# Check 3: within-lineage association (Cochran-Mantel-Haenszel)
# ---------------------------------------------------------------------------


def test_cmh_detects_association_present_in_every_stratum():
    """Hand-built 2-stratum example: x perfectly predicts y within each
    stratum, so the CMH test must report a small p-value even though the
    marginal (pooled) table would be confounded by stratum composition.
    """
    # Two strata, x == y within each (perfect within-stratum association),
    # but stratum 0 is mostly y=1 and stratum 1 is mostly y=0 -- a marginal
    # test ignoring strata would be distorted by that imbalance.
    strata = np.array([0] * 20 + [1] * 20)
    x = np.array(([1] * 8 + [0] * 12) + ([1] * 4 + [0] * 16))
    y = x.copy()

    p = _cochran_mantel_haenszel(x, y, strata)
    assert p is not None
    assert p < 0.01, f"a perfect within-stratum association should have a tiny p-value, got {p}"


def test_cmh_reports_no_association_for_independent_noise():
    rng = np.random.default_rng(3)
    strata = rng.integers(0, 4, size=200)
    x = rng.integers(0, 2, size=200)
    y = rng.integers(0, 2, size=200)

    p = _cochran_mantel_haenszel(x, y, strata)
    assert p is not None
    assert p > 0.05, f"independent noise should not show significant association, got {p}"


def test_cmh_returns_none_with_too_few_informative_strata():
    # A single stratum: CMH needs at least 2 informative strata.
    strata = np.zeros(10, dtype=int)
    x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    assert _cochran_mantel_haenszel(x, y, strata) is None


def test_a_marker_associated_with_phenotype_within_lineages_is_credible(tmp_path):
    """POSITIVE CONTROL. A marker present across most lineages, and whose
    presence tracks the phenotype *within* each lineage (not merely
    correlated with lineage), must survive as a credible candidate.
    """
    paths, lineage_of = _lineage_cohort(tmp_path, n_lineages=6, per_lineage=6, seed=4)
    k = 21
    # Exactly k bases -- see the identifiability test above for why a
    # longer marker would confound this test with a linked-block result
    # instead of isolating the within-lineage association check.
    marker = "TTGGCCAATTGGCCAATTGGC"
    assert len(marker) == k
    canonical_marker = _canonical(marker)

    rng = np.random.default_rng(5)
    # Phenotype independent of lineage; the marker is appended precisely
    # when phenotype == 1, in every lineage -- a real, lineage-independent
    # signal.
    phenotype = rng.integers(0, 2, size=len(paths))
    for path, y in zip(paths, phenotype):
        if y == 1:
            with open(path, "a") as fh:
                fh.write(f"@marker\n{marker}\n+\n{'I' * len(marker)}\n")

    vec = KmerVectorizer(k=k, top_features=None, representation="presence").fit(paths)
    assert canonical_marker in vec._feature_sequences_, "fixture must produce the marker k-mer"
    marker_kmer = canonical_marker
    marker_index = vec._feature_sequences_.index(marker_kmer)

    importances = np.zeros(len(vec.vocabulary_))
    importances[marker_index] = 1.0

    report = explain(vec, importances, paths, phenotype, top_n=1, lineage_threshold=0.02)
    feature = report.features[0]
    assert feature.kmer == marker_kmer
    assert not feature.lineage_restricted, (
        f"the marker appears in {feature.n_lineages_present}/{feature.n_lineages_total} lineages "
        "and should not be flagged restricted"
    )
    assert feature.association_p_value is not None
    assert feature.association_p_value < 0.05
    assert feature.verdict == "credible candidate"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_importances_length_is_rejected(tmp_path):
    paths, _ = _lineage_cohort(tmp_path, n_lineages=2, per_lineage=3, seed=6)
    vec = KmerVectorizer(k=11, top_features=50).fit(paths)
    with pytest.raises(ValueError, match="importances"):
        explain(vec, np.ones(3), paths, np.zeros(len(paths)))


def test_non_binary_phenotype_is_rejected(tmp_path):
    paths, _ = _lineage_cohort(tmp_path, n_lineages=2, per_lineage=3, seed=7)
    vec = KmerVectorizer(k=11, top_features=50).fit(paths)
    phenotype = np.array([0, 1, 2] * (len(paths) // 3 + 1))[: len(paths)]
    with pytest.raises(ValueError, match="binary"):
        explain(vec, np.ones(len(vec.vocabulary_)), paths, phenotype)


def test_explain_without_phenotype_skips_association_but_still_runs(tmp_path):
    paths, _ = _lineage_cohort(tmp_path, n_lineages=3, per_lineage=4, seed=8)
    vec = KmerVectorizer(k=11, top_features=50).fit(paths)
    report = explain(vec, np.ones(len(vec.vocabulary_)), paths, phenotype=None, top_n=5)
    assert len(report.features) == 5
    for f in report.features:
        assert f.association_p_value is None
        assert f.survives_stratification is None
