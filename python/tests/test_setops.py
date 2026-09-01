"""Tests for fastdna.KmerTable's set operations -- the S2 union/intersect/
difference layer over sorted k-mer tables (docs/feature-gap-analysis.md),
wrapping src/setops.rs via `fastdna._core.ktab_union/ktab_intersect/
ktab_diff`.

Fixture tables are built with plain pyarrow, the same convention
`test_ktab.py` uses: `KmerTable.open`'s contract is "any sorted Parquet file
carrying the fastdna.sorted_by/fastdna.k footer metadata", not "only a file
this crate's own exporter wrote", so building fixtures this way exercises
that contract directly rather than only fastdna's own round trip.
"""
from __future__ import annotations

import pathlib
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fastdna


def write_kmer_table(path: pathlib.Path, k: int, rows: List[Tuple[int, int]]) -> None:
    kmers = [kmer for kmer, _ in rows]
    counts = [count for _, count in rows]
    schema = pa.schema(
        [
            pa.field("kmer_u64", pa.uint64(), nullable=False),
            pa.field("frequency", pa.uint32(), nullable=False),
        ]
    ).with_metadata({"fastdna.sorted_by": "kmer_u64", "fastdna.k": str(k)})
    table = pa.table(
        {"kmer_u64": pa.array(kmers, type=pa.uint64()), "frequency": pa.array(counts, type=pa.uint32())},
        schema=schema,
    )
    pq.write_table(table, str(path))


def open_table(path: pathlib.Path, k: int, rows: List[Tuple[int, int]]) -> "fastdna.KmerTable":
    write_kmer_table(path, k, rows)
    return fastdna.KmerTable.open(str(path))


def rows_for(table: "fastdna.KmerTable", candidates: List[int]) -> List[Tuple[int, int]]:
    """The subset of `candidates` present in `table`, with their counts --
    the only way to inspect a small `KmerTable`'s contents from Python today
    (see `KmerTable`'s own docstring on the missing batch/range binding).
    """
    return sorted((c, table[c]) for c in candidates if c in table)


def test_union_sums_overlapping_kmers_by_default(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 2), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1), (3, 3)])

    out = a.union(b, output=str(tmp_path / "union.parquet"))
    assert len(out) == 3
    assert rows_for(out, [1, 2, 3]) == [(1, 3), (2, 1), (3, 3)]


def test_union_output_is_reopenable_as_a_kmer_table(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(2, 1)])

    out_path = tmp_path / "union.parquet"
    out = a.union(b, output=str(out_path))
    assert isinstance(out, fastdna.KmerTable)

    reopened = fastdna.KmerTable.open(str(out_path))
    assert reopened.k == 4
    assert len(reopened) == 2


def test_union_combine_max_keeps_the_larger_count(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 5)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 9)])

    out = a.union(b, output=str(tmp_path / "union.parquet"), combine="max")
    assert out[1] == 9


def test_union_of_three_tables(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1)])
    c = open_table(tmp_path / "c.parquet", 4, [(1, 1)])

    out = a.union(b, c, output=str(tmp_path / "union.parquet"))
    assert out[1] == 3


def test_union_rejects_an_unknown_combine_value(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1)])

    with pytest.raises(ValueError):
        a.union(b, output=str(tmp_path / "union.parquet"), combine="bogus")


def test_intersect_keeps_only_shared_kmers_with_min_by_default(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 5), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 2), (3, 1)])

    out = a.intersect(b, output=str(tmp_path / "intersect.parquet"))
    assert len(out) == 1
    assert out[1] == 2


def test_intersect_of_disjoint_tables_is_empty(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(2, 1)])

    out = a.intersect(b, output=str(tmp_path / "intersect.parquet"))
    assert len(out) == 0


def test_intersect_requires_every_table_to_have_the_kmer(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1), (2, 1)])
    c = open_table(tmp_path / "c.parquet", 4, [(1, 1)])

    out = a.intersect(b, c, output=str(tmp_path / "intersect.parquet"))
    assert len(out) == 1
    assert out[1] == 1


def test_intersect_combine_sum(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 2)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 3)])

    out = a.intersect(b, output=str(tmp_path / "intersect.parquet"), combine="sum")
    assert out[1] == 5


def test_difference_removes_kmers_present_in_subtract_by_default(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 2), (2, 1), (3, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(2, 1)])

    out = a.difference(b, output=str(tmp_path / "diff.parquet"))
    assert len(out) == 2
    assert rows_for(out, [1, 2, 3]) == [(1, 2), (3, 1)]


def test_difference_keeps_a_unchanged_with_no_overlap(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(9, 1)])

    out = a.difference(b, output=str(tmp_path / "diff.parquet"))
    assert len(out) == 2


def test_difference_of_a_against_itself_is_empty(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1), (2, 1)])

    out = a.difference(b, output=str(tmp_path / "diff.parquet"))
    assert len(out) == 0


def test_difference_tolerates_noise_below_max_subtract_count(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 3), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 2)])

    tolerant = a.difference(b, output=str(tmp_path / "diff_tolerant.parquet"), max_subtract_count=2)
    assert len(tolerant) == 2

    strict = a.difference(b, output=str(tmp_path / "diff_strict.parquet"), max_subtract_count=0)
    assert len(strict) == 1
    assert strict[2] == 1


def test_difference_against_several_subtract_tables(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1), (2, 1), (3, 1)])
    host = open_table(tmp_path / "host.parquet", 4, [(1, 1)])
    adapter = open_table(tmp_path / "adapter.parquet", 4, [(2, 1)])

    out = a.difference(host, adapter, output=str(tmp_path / "diff.parquet"))
    assert len(out) == 1
    assert out[3] == 1


def test_union_of_an_empty_table_with_a_non_empty_one(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [])
    b = open_table(tmp_path / "b.parquet", 4, [(1, 1)])

    out = a.union(b, output=str(tmp_path / "union.parquet"))
    assert len(out) == 1
    assert out[1] == 1


def test_set_operations_compose(tmp_path):
    a = open_table(tmp_path / "a.parquet", 4, [(1, 1), (2, 1)])
    b = open_table(tmp_path / "b.parquet", 4, [(2, 1), (3, 1)])
    c = open_table(tmp_path / "c.parquet", 4, [(1, 1)])

    unioned = a.union(b, output=str(tmp_path / "union.parquet"))
    final = unioned.difference(c, output=str(tmp_path / "final.parquet"))

    assert len(final) == 2
    assert rows_for(final, [1, 2, 3]) == [(2, 2), (3, 1)]
