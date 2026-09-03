"""FAILING regression tests written during the 2026-09-02 validation run.

Each test here pins a defect that exists at commit de96f0d. They are expected
to FAIL until the defect is fixed; none of them is a new feature request.
Same convention as `test_review_findings.py`: a dated header, one finding per
test, and a docstring explaining what the test pins and why.

FINDING 9 -- `KmerVectorizer`'s two defaults are individually well-reasoned
and jointly degenerate, and the result is an all-ones feature matrix.

  * `representation="presence"` (the default) encodes each feature 0/1
    rather than as a count, deliberately: `__init__`'s own comment explains
    that raw counts let a linear model separate classes by sequencing depth,
    which correlates with batch, year and center, which correlate with
    phenotype.
  * `_select_vocabulary` ranks candidate k-mers by *descending prevalence*,
    also deliberately: its docstring argues a k-mer present in every
    training sample is a more trustworthy signal than one present at
    enormous depth in a single sample.

Together they guarantee the top of the ranking is exactly the set of k-mers
present in *every* sample -- whose presence value is 1 in every sample, in
every fold, for every phenotype. Variance zero, information zero.

This is not a corner case; it is the normal case for the cohorts this
library is built for. Measured on the study's first real cohort (80
*Streptococcus pneumoniae* genomes, k=31): **460,795 k-mers are present in
all 80 genomes**, and 2,303,765 more sit in the informative 10-90%
prevalence band. `KmerVectorizer`'s documented example uses
`top_features=10_000`; the survey used 500. Both draw entirely from those
460,795 constants, so the matrix handed to the classifier is literally all
ones and `LogisticRegression` can only predict a constant.

What that produced downstream, and why no test caught it: `fastdna.audit()`
returned `score_random = 0.5000`, `score_lineage = 0.5000`, `gap = 0.0000`
-- a perfectly well-formed report whose every field is in range, reading as
"this cohort has no leakage". The same signature as the other eight defects
this validation effort found: *the wrong output was well-formed*.

An earlier draft of this docstring went one step further and claimed the
defect also explained `audit.py`'s p/n table (200 features -> 0.5326, 5,000
-> 0.5666 on an *E. coli* + ampicillin cohort). It does not, and the table's
own numbers are what rule it out: an all-ones matrix scores **exactly**
0.5000 in every fold -- constant scores, `roc_auc_score` = 0.5 by
construction, verified directly -- which is precisely why the survey's
report was recognisable as degenerate. A number that is not 0.5000 did not
come from an all-ones matrix. What that cohort's prevalence-ranked features
were is *near*-universal rather than universal: 80 diverse *E. coli*
assemblies share far fewer exact 31-mers than the 80 *S. pneumoniae*
genomes measured above, so the top of its ranking still varied slightly.
The severity of this defect therefore scales with how much exact k-mer core
a cohort shares -- total for the clonal cohorts, partial for the diverse
ones, and never in a direction that helps.

The fix has to be in the ranking rule, not in a filter bolted after it:
merely dropping the constants would promote the k-mers present in 79 of 80
samples (191,152 of them here), whose variance is 0.0123. For a binary
feature, informativeness *is* variance, maximised at prevalence n/2, so that
is what a presence-encoded vocabulary must rank by.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

from fastdna.sklearn import KmerVectorizer

K = 9

#: A core "genome" every sample carries, long enough that its k-mers alone
#: outnumber the `top_features` budget below -- which is the whole point: a
#: real bacterial cohort's core genome is hundreds of thousands of k-mers
#: against a budget of thousands, so the budget never reaches anything else.
CORE = (
    "ACGTTGCATTACGGCATTAGCCATGGATCCATTAGGCATCAGTTACGGATCAGTTACCGGA"
    "TTACGATCAGGCATTAGCATCAGGTTACGGATCAGCATTAGGCATCAGTTACGGATCAGTT"
    "ACGGATTACGATCAGGCATTAGCATCAGGTTACGGATCAGCATTAGGCATCAGTTACGGAT"
)

#: Carried by half the samples and absent from the other half -- the
#: accessory content a resistance gene actually looks like, and the only
#: thing in this fixture a classifier could possibly learn from.
ACCESSORY = "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAA"


def _cohort(tmp_path: pathlib.Path, n: int = 8) -> list[str]:
    """`n` samples sharing `CORE`; the first half also carry `ACCESSORY`."""
    paths = []
    for i in range(n):
        sequence = CORE + (ACCESSORY if i < n // 2 else "")
        path = tmp_path / f"sample{i}.fastq"
        path.write_text(f"@r0\n{sequence}\n+\n{'I' * len(sequence)}\n")
        paths.append(str(path))
    return paths


def test_presence_vocabulary_is_not_entirely_constant_features(tmp_path):
    """The defect itself: with the default `representation="presence"` and a
    `top_features` budget smaller than the core genome, every selected
    feature is present in every sample, so every column is all ones.

    Asserted as "at least one column varies" rather than as an exact
    vocabulary, because the point is not which k-mers get picked -- it is
    that a matrix a model cannot learn anything from must not be what
    `fit_transform` returns on a cohort that plainly does contain signal.
    """
    paths = _cohort(tmp_path)
    n_core_kmers = len(CORE) - K + 1

    matrix = KmerVectorizer(k=K, top_features=n_core_kmers // 2).fit_transform(paths)
    dense = np.asarray(matrix.todense())

    assert dense.std(axis=0).max() > 0, (
        "every selected feature is constant across the cohort: "
        f"{dense.shape[1]} columns, all equal to {dense[0, 0]}. A classifier "
        "on this matrix can only predict a constant, and audit() reports the "
        "resulting AUC of exactly 0.5 as 'no leakage'."
    )


def test_presence_vocabulary_prefers_the_feature_that_separates_the_cohort(tmp_path):
    """The positive half of the same finding: the accessory k-mers -- present
    in exactly half the samples, and the only ones carrying information --
    must be selected *before* the core, not after it.

    A budget of 10 features is deliberately far smaller than the accessory
    content, so this cannot pass by accident: the ten best features by
    variance are all accessory, and any ranking that puts core first fills
    all ten with constants.
    """
    paths = _cohort(tmp_path, n=8)

    vectorizer = KmerVectorizer(k=K, top_features=10).fit(paths)
    dense = np.asarray(vectorizer.transform(paths).todense())

    prevalence = dense.sum(axis=0)
    assert (prevalence == 4).all(), (
        "expected all 10 selected features to be the half-present accessory "
        f"k-mers; got prevalences {sorted(set(prevalence.tolist()))} across "
        "8 samples"
    )


def test_count_representation_keeps_ranking_by_prevalence(tmp_path):
    """The other half of the fix: this must not become a blanket change.

    Under `representation="count"` a k-mer present in every sample is *not*
    constant -- it varies in depth -- so `_select_vocabulary`'s original
    argument for prevalence-first ranking still holds there and must be left
    alone. Only the binary encoding needs variance ranking.
    """
    paths = _cohort(tmp_path, n=8)

    vectorizer = KmerVectorizer(
        k=K, top_features=10, representation="count"
    ).fit(paths)
    dense = np.asarray(vectorizer.transform(paths).todense())

    prevalence = (dense > 0).sum(axis=0)
    assert (prevalence == 8).all(), (
        "representation='count' must still rank by descending prevalence; "
        f"got prevalences {sorted(set(prevalence.tolist()))}"
    )
