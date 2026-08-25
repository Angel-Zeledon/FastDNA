"""Tests for `fastdna.equivalence` -- collapsing k-mer columns that share an
identical presence/absence profile across a cohort into equivalence
classes.

Every fixture here is a small, hand-constructed CSR matrix (rows = samples,
columns = k-mers), built directly rather than through `fastdna.gwas.
cohort_presence_matrix`, so these tests exercise `collapse_equivalence_classes`
in isolation and do not depend on the Rust extension being built (matching
`test_rules.py`'s own scipy/numpy-only style). The one exception is
`test_composes_with_set_covering_classifier`, which is the module's own
"do these two things actually fit together" round trip and deliberately
pulls in `fastdna.rules.SetCoveringClassifier`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

import scipy.sparse as sparse

from fastdna.equivalence import EquivalenceClasses, collapse_equivalence_classes


def csr(rows_by_column, n_samples):
    """Builds a `(n_samples, len(rows_by_column))` CSR presence/count matrix
    from `rows_by_column`, a list where entry `j` is `{row: value}` for
    column `j`'s nonzero cells (value defaults to 1 for a plain presence
    matrix; tests that care about counts pass an explicit value).
    """
    row, col, data = [], [], []
    for j, cells in enumerate(rows_by_column):
        for r, v in (cells.items() if isinstance(cells, dict) else ((r, 1) for r in cells)):
            row.append(r)
            col.append(j)
            data.append(v)
    return sparse.csr_matrix(
        (data, (row, col)), shape=(n_samples, len(rows_by_column)), dtype=np.uint32
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_dense_input_with_an_actionable_message():
    dense = np.array([[1, 0], [0, 1]], dtype=np.uint32)
    with pytest.raises(TypeError, match="scipy.sparse"):
        collapse_equivalence_classes(dense, ["AAAAA", "CCCCC"])


def test_rejects_a_plain_list_of_lists():
    with pytest.raises(TypeError, match="scipy.sparse"):
        collapse_equivalence_classes([[1, 0], [0, 1]], ["AAAAA", "CCCCC"])


def test_rejects_mismatched_kmer_sequences_length():
    matrix = csr([{0: 1}, {1: 1}], n_samples=2)
    with pytest.raises(ValueError, match="2.*3|3.*2"):
        collapse_equivalence_classes(matrix, ["AAAAA", "CCCCC", "GGGGG"])


def test_rejects_a_matrix_with_zero_columns():
    matrix = sparse.csr_matrix((3, 0), dtype=np.uint32)
    with pytest.raises(ValueError, match="empty"):
        collapse_equivalence_classes(matrix, [])


def test_rejects_a_matrix_with_zero_rows():
    matrix = sparse.csr_matrix((0, 3), dtype=np.uint32)
    with pytest.raises(ValueError, match="empty"):
        collapse_equivalence_classes(matrix, ["AAAAA", "CCCCC", "GGGGG"])


def test_does_not_mutate_the_caller_supplied_matrix():
    matrix = csr([{0: 1}, {0: 1}], n_samples=2)
    original_nnz = matrix.nnz
    original_format = matrix.format
    collapse_equivalence_classes(matrix, ["AAAAA", "CCCCC"])
    assert matrix.nnz == original_nnz
    assert matrix.format == original_format
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1


# ---------------------------------------------------------------------------
# Core collapse semantics
# ---------------------------------------------------------------------------


def test_hand_verified_multi_class_collapse():
    # 4 samples, 5 k-mers.
    #   col 0 "AAAAA": rows {0, 1}
    #   col 1 "CCCCC": rows {0, 1}   -- identical pattern to col 0
    #   col 2 "GGGGG": rows {2, 3}
    #   col 3 "TTTTT": rows {2, 3}   -- identical pattern to col 2
    #   col 4 "ACGTA": rows {0, 2}   -- distinct from every other column
    # Expected classes (by presence pattern): {AAAAA, CCCCC}, {GGGGG, TTTTT},
    # {ACGTA} -- 3 classes total, representatives are the lexicographically
    # smallest member of each: AAAAA, GGGGG, ACGTA -> sorted: AAAAA, ACGTA,
    # GGGGG.
    kmers = ["AAAAA", "CCCCC", "GGGGG", "TTTTT", "ACGTA"]
    matrix = csr([{0, 1}, {0, 1}, {2, 3}, {2, 3}, {0, 2}], n_samples=4)

    result = collapse_equivalence_classes(matrix, kmers)

    assert isinstance(result, EquivalenceClasses)
    assert result.representative == ["AAAAA", "ACGTA", "GGGGG"]
    assert result.matrix.shape == (4, 3)
    assert sparse.issparse(result.matrix)

    dense = result.matrix.toarray()
    col_of = {name: i for i, name in enumerate(result.representative)}
    assert dense[:, col_of["AAAAA"]].tolist() == [1, 1, 0, 0]
    assert dense[:, col_of["GGGGG"]].tolist() == [0, 0, 1, 1]
    assert dense[:, col_of["ACGTA"]].tolist() == [1, 0, 1, 0]

    members = result.members.to_pylist()
    by_class = {}
    for row in members:
        by_class.setdefault(row["class_id"], set()).add(row["kmer_sequence"])
    assert set(frozenset(v) for v in by_class.values()) == {
        frozenset({"AAAAA", "CCCCC"}),
        frozenset({"GGGGG", "TTTTT"}),
        frozenset({"ACGTA"}),
    }


def test_all_identical_columns_collapse_to_one_class():
    kmers = ["TTTTT", "AAAAA", "CCCCC", "GGGGG"]
    matrix = csr([{0, 2}, {0, 2}, {0, 2}, {0, 2}], n_samples=3)

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.matrix.shape == (3, 1)
    assert result.representative == ["AAAAA"]  # lexicographically smallest of the four
    assert result.matrix.toarray()[:, 0].tolist() == [1, 0, 1]
    assert {row["kmer_sequence"] for row in result.members.to_pylist()} == set(kmers)
    assert {row["class_id"] for row in result.members.to_pylist()} == {0}


def test_all_distinct_columns_is_a_noop_modulo_column_order():
    kmers = ["TTTTT", "AAAAA", "GGGGG"]
    matrix = csr([{0}, {1}, {2}], n_samples=3)

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.matrix.shape == matrix.shape
    # Every column is its own class; the representative list is just the
    # input k-mers sorted lexicographically.
    assert result.representative == sorted(kmers)
    dense = result.matrix.toarray()
    col_of = {name: i for i, name in enumerate(result.representative)}
    assert dense[:, col_of["TTTTT"]].tolist() == [1, 0, 0]
    assert dense[:, col_of["AAAAA"]].tolist() == [0, 1, 0]
    assert dense[:, col_of["GGGGG"]].tolist() == [0, 0, 1]


def test_presence_not_counts_decides_class_membership():
    # Same nonzero row (sample 0), wildly different stored counts -- must
    # still collapse into one class, since collapsing is about presence,
    # not depth.
    kmers = ["AAAAA", "CCCCC"]
    matrix = csr([{0: 40}, {0: 12}], n_samples=2)

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.matrix.shape == (2, 1)
    assert {row["kmer_sequence"] for row in result.members.to_pylist()} == {"AAAAA", "CCCCC"}
    # The reduced matrix stores presence (0/1), not the original counts --
    # counts differed between the two members, so there is no single
    # "correct" count to keep; see the module docstring.
    assert result.matrix.toarray()[:, 0].tolist() == [1, 0]


def test_explicit_stored_zero_does_not_count_as_present():
    # A stored-but-zero entry (e.g. left behind by arithmetic upstream) must
    # not be treated as "present" -- eliminate_zeros() semantics.
    matrix = sparse.csr_matrix(
        (np.array([0, 1], dtype=np.uint32), (np.array([0, 0]), np.array([0, 1]))),
        shape=(2, 2),
    )
    kmers = ["AAAAA", "CCCCC"]

    result = collapse_equivalence_classes(matrix, kmers)

    # Column 0 has an explicit zero at row 0 and nothing else -> genuinely
    # empty presence set, same as column... well here it's the only
    # all-empty column, so it is a singleton class of its own, distinct
    # from column 1 (present at row 0).
    assert result.matrix.shape == (2, 2)
    assert set(result.representative) == {"AAAAA", "CCCCC"}


# ---------------------------------------------------------------------------
# All-zero columns
# ---------------------------------------------------------------------------


def test_a_single_all_zero_column_is_its_own_class():
    kmers = ["AAAAA", "CCCCC", "GGGGG"]
    matrix = csr([{0}, [], {1}], n_samples=2)  # column 1 ("CCCCC") is all-zero

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.matrix.shape == (2, 3)
    assert "CCCCC" in result.representative
    col = result.representative.index("CCCCC")
    assert result.matrix.toarray()[:, col].tolist() == [0, 0]

    members = result.members.to_pylist()
    ccccc_class = next(row["class_id"] for row in members if row["kmer_sequence"] == "CCCCC")
    assert {row["kmer_sequence"] for row in members if row["class_id"] == ccccc_class} == {"CCCCC"}


def test_two_all_zero_columns_collapse_into_one_class():
    # Consistent with the module's own rule ("identical presence pattern ->
    # same class"): two empty presence sets ARE identical, so two all-zero
    # columns collapse together like any other pair of matching columns --
    # no special case is carved out for the empty set.
    kmers = ["TTTTT", "AAAAA", "GGGGG"]
    matrix = csr([[], [], {0}], n_samples=2)  # cols 0 and 1 are both all-zero

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.matrix.shape == (2, 2)
    assert sorted(result.representative) == ["AAAAA", "GGGGG"]
    members = result.members.to_pylist()
    by_class = {}
    for row in members:
        by_class.setdefault(row["class_id"], set()).add(row["kmer_sequence"])
    assert frozenset({"AAAAA", "TTTTT"}) in {frozenset(v) for v in by_class.values()}


# ---------------------------------------------------------------------------
# Hash-collision safety
# ---------------------------------------------------------------------------


def test_hash_collision_does_not_merge_distinct_classes():
    # collapse_equivalence_classes buckets candidate columns by
    # `sum(row_index + 1 for row_index in column) * (n_samples + 1) +
    # len(column)` (documented in the module docstring as a deliberately
    # cheap, non-cryptographic hash -- the correctness guarantee comes from
    # verifying true equality within a bucket, not from the hash being
    # collision-free). With n_samples=4, columns with presence sets {0, 3}
    # and {1, 2} both have length 2 and weighted sum (0+1)+(3+1)=5 ==
    # (1+1)+(2+1)=5, so they land in the same candidate bucket despite being
    # genuinely different sets -- this is a forced collision by
    # construction, not a probabilistic one.
    kmers = ["AAAAA", "CCCCC", "GGGGG"]
    matrix = csr([{0, 3}, {1, 2}, {0, 3}], n_samples=4)

    result = collapse_equivalence_classes(matrix, kmers)

    # AAAAA and GGGGG (both {0, 3}) must merge; CCCCC ({1, 2}) must NOT be
    # merged with them despite the hash collision.
    assert result.matrix.shape == (4, 2)
    members = result.members.to_pylist()
    by_class = {}
    for row in members:
        by_class.setdefault(row["class_id"], set()).add(row["kmer_sequence"])
    classes = {frozenset(v) for v in by_class.values()}
    assert frozenset({"AAAAA", "GGGGG"}) in classes
    assert frozenset({"CCCCC"}) in classes

    dense = result.matrix.toarray()
    col_of = {name: i for i, name in enumerate(result.representative)}
    assert dense[:, col_of["AAAAA"]].tolist() == [1, 0, 0, 1]
    assert dense[:, col_of["CCCCC"]].tolist() == [0, 1, 1, 0]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_representative_choice_is_deterministic_across_runs():
    kmers = ["GGGGG", "AAAAA", "TTTTT", "CCCCC"]
    matrix = csr([{0}, {0}, {1}, {1}], n_samples=2)

    first = collapse_equivalence_classes(matrix, kmers)
    second = collapse_equivalence_classes(matrix, kmers)

    assert first.representative == second.representative
    assert first.members.to_pylist() == second.members.to_pylist()
    assert first.matrix.toarray().tolist() == second.matrix.toarray().tolist()


def test_class_order_matches_ascending_representative_order():
    kmers = ["ZZZZZ", "MMMMM", "AAAAA"]  # deliberately not already sorted
    matrix = csr([{0}, {1}, {0, 1}], n_samples=2)

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.representative == sorted(result.representative)


# ---------------------------------------------------------------------------
# `members` completeness
# ---------------------------------------------------------------------------


def test_members_table_accounts_for_every_input_kmer_exactly_once():
    kmers = ["AAAAA", "CCCCC", "GGGGG", "TTTTT", "ACGTA", "TGCAT"]
    matrix = csr([{0, 1}, {0, 1}, {2}, {2}, {0}, {1}], n_samples=3)

    result = collapse_equivalence_classes(matrix, kmers)

    members = result.members.to_pylist()
    assert sorted(row["kmer_sequence"] for row in members) == sorted(kmers)

    seen = set()
    for row in members:
        assert row["kmer_sequence"] not in seen, "a k-mer appeared in more than one class"
        seen.add(row["kmer_sequence"])
    assert seen == set(kmers)

    # Every class_id used in `members` must correspond to a real column of
    # `result.matrix`.
    class_ids = {row["class_id"] for row in members}
    assert class_ids == set(range(len(result.representative)))


def test_members_table_columns_and_row_order():
    kmers = ["CCCCC", "AAAAA"]
    matrix = csr([{0}, {0}], n_samples=1)

    result = collapse_equivalence_classes(matrix, kmers)

    assert result.members.column_names == ["class_id", "kmer_sequence"]
    # One class, both members, sorted by kmer_sequence within the class.
    assert result.members.to_pylist() == [
        {"class_id": 0, "kmer_sequence": "AAAAA"},
        {"class_id": 0, "kmer_sequence": "CCCCC"},
    ]


# ---------------------------------------------------------------------------
# Composability with fastdna.rules.SetCoveringClassifier
# ---------------------------------------------------------------------------


def test_composes_with_set_covering_classifier():
    sklearn = pytest.importorskip("sklearn")
    from fastdna.rules import SetCoveringClassifier

    # 6 samples, 5 k-mers. Columns 0 and 1 share the exact presence pattern
    # {0, 1, 2} (present in every "resistant" sample) and should collapse
    # into a single feature that the classifier can use as a perfect rule;
    # columns 2..4 are distinct accessory k-mers with no signal.
    kmers = ["AAAAA", "CCCCC", "GGGGG", "TTTTT", "ACGTA"]
    matrix = csr(
        [
            {0, 1, 2},  # AAAAA -- present in every resistant sample
            {0, 1, 2},  # CCCCC -- identical pattern to AAAAA
            {3},  # GGGGG -- noise
            {4},  # TTTTT -- noise
            {5},  # ACGTA -- noise
        ],
        n_samples=6,
    )
    y = np.array([1, 1, 1, 0, 0, 0])

    collapsed = collapse_equivalence_classes(matrix, kmers)
    assert collapsed.matrix.shape[1] < matrix.shape[1]  # something actually collapsed

    clf = SetCoveringClassifier(max_rules=5)
    clf.fit(collapsed.matrix, y, feature_names=collapsed.representative)

    assert np.array_equal(clf.predict(collapsed.matrix), y)
    explanation = clf.explain()
    assert "AAAAA" in explanation  # the surviving representative of the merged pair
    assert "CCCCC" not in explanation  # collapsed away, never seen by the classifier
