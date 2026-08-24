"""Tests for fastdna.interpret -- mapping model feature importances over
KmerVectorizer-produced features back to literal k-mer DNA sequences.

**Stub note:** `fastdna.sklearn.KmerVectorizer` (docs/ml-genomics-roadmap.md,
item 1) had not landed in this worktree at the time `fastdna/interpret.py`
was written -- it was built and is tested here against that class's
*documented* contract only:

    class KmerVectorizer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            "Learns self.vocabulary_ from X (a list of FASTQ paths)."
        def transform(self, X):
            "Returns a scipy.sparse.csr_matrix, shape
             (len(X), len(self.vocabulary_))."
        def get_feature_names_out(self, input_features=None):
            "Returns the vocabulary as decoded k-mer sequence strings, in
             the same column order transform() produces."

`FakeVectorizer` below is a tiny hand-written stand-in matching that exact
shape -- NOT the real class, and not sklearn-compatible beyond what these
tests need (no real `fit`/`transform` estimation, just fixed data). It
exists purely so `fastdna.interpret`'s functions can be exercised against
something shaped like `KmerVectorizer`'s output (feature names + an
aligned sparse matrix) without importing a module that does not exist in
this isolated worktree.

**This is not integration coverage.** Once `fastdna.sklearn.KmerVectorizer`
is merged, a real integration test should be added that fits an actual
`KmerVectorizer` on real FASTQ files, trains a small model on its
`.transform()` output, and confirms `top_features`/
`export_top_features_fasta` correctly round-trip against the real class's
`get_feature_names_out()` ordering -- nothing in this file proves that.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import scipy.sparse as sp

from fastdna.interpret import export_top_features_fasta, top_features


class FakeVectorizer:
    """Stub matching the documented KmerVectorizer contract (see module
    docstring above) -- NOT the real fastdna.sklearn.KmerVectorizer.
    """

    def __init__(self, vocabulary, matrix):
        # vocabulary: list[str] of k-mer sequence strings.
        # matrix: 2-D array-like, columns aligned with `vocabulary`.
        self._vocabulary = list(vocabulary)
        self._matrix = sp.csr_matrix(np.asarray(matrix, dtype=float))
        self.vocabulary_ = None

    def fit(self, X, y=None):
        self.vocabulary_ = list(self._vocabulary)
        return self

    def transform(self, X):
        return self._matrix

    def get_feature_names_out(self, input_features=None):
        return list(self._vocabulary)


def test_fake_vectorizer_matches_documented_shape():
    fv = FakeVectorizer(["ACGT", "TTTT", "GGGG"], [[1, 0, 2], [0, 3, 0]])
    fv.fit(["a.fastq", "b.fastq"])

    assert fv.vocabulary_ == ["ACGT", "TTTT", "GGGG"]
    mat = fv.transform(["a.fastq", "b.fastq"])
    assert sp.issparse(mat)
    assert mat.shape == (2, 3)
    assert fv.get_feature_names_out() == ["ACGT", "TTTT", "GGGG"]


# --- top_features -----------------------------------------------------

def test_top_features_ranks_by_magnitude_for_signed_importances():
    # Hand-worked example: abs values are 0.1, 0.9, 0.05, 0.5, 0.3, 0.0 --
    # descending magnitude order is k2, k4, k5, k1, k3, k6.
    feature_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
    importances = [0.1, -0.9, 0.05, 0.5, -0.3, 0.0]

    table = top_features(importances, feature_names, n=3)

    assert table.column("rank").to_pylist() == [1, 2, 3]
    assert table.column("kmer").to_pylist() == ["k2", "k4", "k5"]
    # Literal (signed) values are preserved -- ranking used magnitude, but
    # the reported importance is the real, signed value.
    assert table.column("importance").to_pylist() == pytest.approx([-0.9, 0.5, -0.3])


def test_top_features_ascending_ranks_by_literal_value_not_magnitude():
    # Same data as above. Ascending literal order (smallest/most negative
    # first): k2 (-0.9), k5 (-0.3), k6 (0.0), k3 (0.05), k1 (0.1), k4 (0.5).
    feature_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
    importances = [0.1, -0.9, 0.05, 0.5, -0.3, 0.0]

    table = top_features(importances, feature_names, n=3, ascending=True)

    assert table.column("kmer").to_pylist() == ["k2", "k5", "k6"]
    assert table.column("importance").to_pylist() == pytest.approx([-0.9, -0.3, 0.0])


def test_top_features_on_nonnegative_importances_is_plain_descending():
    # Tree-model style feature_importances_: all non-negative, so
    # magnitude ranking and literal descending ranking coincide.
    feature_names = ["a", "b", "c", "d"]
    importances = [0.05, 0.4, 0.3, 0.25]

    table = top_features(importances, feature_names, n=2)

    assert table.column("kmer").to_pylist() == ["b", "c"]
    assert table.column("importance").to_pylist() == pytest.approx([0.4, 0.3])


def test_top_features_n_larger_than_available_features_is_not_an_error():
    feature_names = ["a", "b", "c"]
    importances = [0.1, 0.2, 0.3]

    table = top_features(importances, feature_names, n=100)

    assert table.num_rows == 3


def test_top_features_returns_a_pyarrow_table_with_expected_columns():
    table = top_features([0.5, -0.2], ["a", "b"], n=2)

    assert isinstance(table, pa.Table)
    assert set(table.column_names) == {"rank", "kmer", "importance"}


def test_top_features_raises_clear_error_on_mismatched_lengths():
    importances = [0.1, 0.2, 0.3]
    feature_names = ["a", "b"]  # deliberately one short

    with pytest.raises(ValueError, match="same length"):
        top_features(importances, feature_names)


def test_top_features_raises_on_2d_importances():
    importances = [[0.1, 0.2], [0.3, 0.4]]
    feature_names = ["a", "b"]

    with pytest.raises(ValueError):
        top_features(importances, feature_names)


# --- export_top_features_fasta -----------------------------------------

def parse_fasta(path):
    """Minimal manual FASTA parser -- good enough for the short,
    single-line-sequence records this module writes; no Biopython needed.
    """
    records = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                records.append((header, line))
    return records


def test_export_top_features_fasta_matches_top_features_selection(tmp_path):
    feature_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
    importances = [0.1, -0.9, 0.05, 0.5, -0.3, 0.0]
    out = tmp_path / "top_kmers.fasta"

    expected = top_features(importances, feature_names, n=3)
    export_top_features_fasta(importances, feature_names, str(out), n=3)

    records = parse_fasta(str(out))
    assert len(records) == 3 == expected.num_rows

    expected_kmers = expected.column("kmer").to_pylist()
    expected_ranks = expected.column("rank").to_pylist()
    expected_importances = expected.column("importance").to_pylist()

    for (header, sequence), kmer, rank, importance in zip(
        records, expected_kmers, expected_ranks, expected_importances
    ):
        assert sequence == kmer
        assert header == f"rank{rank}_importance{importance:.3f}"


def test_export_top_features_fasta_header_format_is_exact(tmp_path):
    out = tmp_path / "one.fasta"

    export_top_features_fasta([0.842], ["ACGTACGTACGTACGTACGTA"], str(out), n=1)

    content = out.read_text()
    lines = content.splitlines()
    assert lines == [">rank1_importance0.842", "ACGTACGTACGTACGTACGTA"]


def test_export_top_features_fasta_handles_negative_importance_in_header(tmp_path):
    out = tmp_path / "neg.fasta"

    export_top_features_fasta([-0.417], ["TTTTGGGGCCCCAAAATTTTG"], str(out), n=1)

    lines = out.read_text().splitlines()
    assert lines[0] == ">rank1_importance-0.417"


def test_export_top_features_fasta_returns_the_same_table_it_wrote(tmp_path):
    out = tmp_path / "ret.fasta"
    importances = [0.9, -0.5, 0.1]
    feature_names = ["a", "b", "c"]

    returned = export_top_features_fasta(importances, feature_names, str(out), n=2)

    assert isinstance(returned, pa.Table)
    assert returned.num_rows == 2
    assert returned.column("kmer").to_pylist() == ["a", "b"]


# --- explain_with_shap (only run if `shap` is importable) --------------
#
# `pytest.importorskip` is called *inside* the test function, not at module
# level: at module level it would skip the entire file's collection (every
# test above too) if `shap` is missing, which is not what we want -- only
# this one optional test should be skipped when the optional dependency
# isn't installed.


def test_explain_with_shap_on_a_real_fitted_logistic_regression():
    pytest.importorskip(
        "shap", reason="shap is an optional dependency of fastdna.interpret's explain_with_shap"
    )
    from sklearn.linear_model import LogisticRegression

    from fastdna.interpret import explain_with_shap

    rng = np.random.default_rng(0)
    n_samples, n_features = 40, 6
    X = sp.random(n_samples, n_features, density=0.5, random_state=0, format="csr")
    X.data[:] = rng.integers(0, 5, size=X.data.shape[0])
    # Make the label depend on feature 0's presence, so the model has a
    # real signal to find (and feature 0 should show up with a non-trivial
    # coefficient/SHAP value).
    y = (np.asarray(X[:, 0].todense()).ravel() > 0).astype(int)
    if y.sum() == 0 or y.sum() == n_samples:
        y[0] = 1 - y[0]  # guard against a degenerate single-class draw

    model = LogisticRegression(max_iter=1000).fit(X, y)
    feature_names = [f"kmer_{i}" for i in range(n_features)]

    table = explain_with_shap(model, X, feature_names, n=3)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert set(table.column_names) == {"rank", "kmer", "importance"}
    # SHAP importances here are mean absolute values -- non-negative.
    assert all(v >= 0 for v in table.column("importance").to_pylist())
