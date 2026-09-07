"""Tests for fastdna.similarity() -- exact pairwise similarity between
counted k-mer tables.

Tables are written directly here rather than produced by counting a FASTQ,
for the same reason `test_ktab.py` writes its own: what is under test is
the similarity arithmetic over a known pair of k-mer sets, and deriving
those sets by counting reads would make every assertion depend on the
counting path as well. The Rust side has its own tests against the real
counting + export path; these pin the numbers a caller sees.
"""
from __future__ import annotations

import pathlib
from typing import List, Tuple

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import fastdna


def write_kmer_table(path: pathlib.Path, k: int, rows: List[Tuple[int, int]]) -> str:
    """One sorted k-mer table, in the shape `KmerTable.open` requires."""
    schema = pa.schema(
        [
            pa.field("kmer_u64", pa.uint64(), nullable=False),
            pa.field("frequency", pa.uint32(), nullable=False),
        ]
    ).with_metadata({"fastdna.sorted_by": "kmer_u64", "fastdna.k": str(k)})
    table = pa.table(
        {
            "kmer_u64": pa.array([kmer for kmer, _ in rows], type=pa.uint64()),
            "frequency": pa.array([count for _, count in rows], type=pa.uint32()),
        },
        schema=schema,
    )
    pq.write_table(table, str(path))
    return str(path)


def test_identical_tables_are_maximally_similar(tmp_path):
    rows = [(1, 3), (5, 1), (9, 4)]
    a = write_kmer_table(tmp_path / "a.parquet", 21, rows)
    b = write_kmer_table(tmp_path / "b.parquet", 21, rows)

    row = fastdna.similarity([a, b]).to_pylist()[0]

    assert row["shared"] == 3
    assert row["only_a"] == 0 and row["only_b"] == 0
    assert row["jaccard"] == 1.0
    assert row["containment_ab"] == 1.0 and row["containment_ba"] == 1.0
    assert row["bray_curtis"] == 0.0


def test_disjoint_tables_share_nothing(tmp_path):
    a = write_kmer_table(tmp_path / "a.parquet", 21, [(1, 1), (2, 1)])
    b = write_kmer_table(tmp_path / "b.parquet", 21, [(8, 1), (9, 1)])

    row = fastdna.similarity([a, b]).to_pylist()[0]

    assert row["shared"] == 0
    assert row["jaccard"] == 0.0
    assert row["containment_ab"] == 0.0 and row["containment_ba"] == 0.0
    assert row["bray_curtis"] == 1.0


def test_partial_overlap_matches_the_arithmetic_by_hand(tmp_path):
    # A = {1,2,3}, B = {2,3,4}: shared 2, union 4 -> jaccard 0.5.
    # Counts: A = {1:5, 2:1, 3:1}, B = {2:3, 3:1, 4:1}.
    #   sum(min) over shared = min(1,3) + min(1,1) = 2
    #   totals = 7 and 5 -> bray_curtis = 1 - 2*2/12 = 2/3.
    a = write_kmer_table(tmp_path / "a.parquet", 21, [(1, 5), (2, 1), (3, 1)])
    b = write_kmer_table(tmp_path / "b.parquet", 21, [(2, 3), (3, 1), (4, 1)])

    row = fastdna.similarity([a, b]).to_pylist()[0]

    assert row["shared"] == 2
    assert row["only_a"] == 1 and row["only_b"] == 1
    assert row["jaccard"] == pytest.approx(0.5)
    assert row["bray_curtis"] == pytest.approx(2 / 3)


def test_containment_is_reported_in_both_directions(tmp_path):
    # A is a strict subset of B, so containment is 1.0 one way and not the
    # other -- the asymmetry the two columns exist for.
    a = write_kmer_table(tmp_path / "a.parquet", 21, [(1, 1), (2, 1)])
    b = write_kmer_table(tmp_path / "b.parquet", 21, [(1, 1), (2, 1), (3, 1), (4, 1)])

    row = fastdna.similarity([a, b]).to_pylist()[0]

    assert row["containment_ab"] == pytest.approx(1.0)
    assert row["containment_ba"] == pytest.approx(0.5)
    assert row["containment_ab"] != row["containment_ba"]


def test_every_pair_is_reported_once_and_labelled_by_path(tmp_path):
    paths = [
        write_kmer_table(tmp_path / f"{name}.parquet", 21, [(index, 1)])
        for index, name in enumerate("abc")
    ]

    table = fastdna.similarity(paths)

    assert table.num_rows == 3, "three tables make three unordered pairs"
    pairs = {(row["sample_a"], row["sample_b"]) for row in table.to_pylist()}
    assert pairs == {(paths[0], paths[1]), (paths[0], paths[2]), (paths[1], paths[2])}


def test_fewer_than_two_tables_is_rejected(tmp_path):
    a = write_kmer_table(tmp_path / "a.parquet", 21, [(1, 1)])
    with pytest.raises(ValueError, match="at least two"):
        fastdna.similarity([a])


def test_tables_built_with_different_k_are_rejected(tmp_path):
    a = write_kmer_table(tmp_path / "a.parquet", 21, [(1, 1)])
    b = write_kmer_table(tmp_path / "b.parquet", 31, [(1, 1)])
    with pytest.raises(ValueError):
        fastdna.similarity([a, b])
