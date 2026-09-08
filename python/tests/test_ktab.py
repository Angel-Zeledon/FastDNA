"""Tests for fastdna.KmerTable -- the S1 random-access query layer over a
sorted k-mer table (docs/feature-gap-analysis.md), wrapping src/ktab.rs's
`KmerTable` via `fastdna._core.KmerTable`.

Fixture tables are built with plain pyarrow rather than fastdna's own
writer, deliberately: `KmerTable.open`'s contract is "any sorted Parquet
file carrying the `fastdna.sorted_by`/`fastdna.k` footer metadata", not
"only a file this crate's own exporter wrote" (see `src/ktab.rs`'s module
doc comment on the "no new file format" design decision) -- this proves
that contract rather than only exercising fastdna's own round trip.
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


def write_plain_parquet_with_no_ktab_metadata(path: pathlib.Path) -> None:
    """The right two columns, but none of `KmerTable.open`'s required
    footer metadata -- a plain Parquet file that happens to share the
    schema, not a k-mer table.
    """
    schema = pa.schema(
        [pa.field("kmer_u64", pa.uint64(), nullable=False), pa.field("frequency", pa.uint32(), nullable=False)]
    )
    table = pa.table(
        {"kmer_u64": pa.array([1, 2], type=pa.uint64()), "frequency": pa.array([1, 2], type=pa.uint32())},
        schema=schema,
    )
    pq.write_table(table, str(path))


def test_open_reports_k_and_length(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(0, 1), (6, 2), (27, 3)])

    table = fastdna.KmerTable.open(str(path))
    assert table.k == 4
    assert len(table) == 3


def test_get_finds_a_present_kmer_by_raw_encoding_and_by_sequence(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(0, 1), (6, 2), (27, 3)])

    table = fastdna.KmerTable.open(str(path))
    # "ACGT" packs to 0b00_01_10_11 = 27 and is its own reverse complement
    # (a palindrome: A<->T, C<->G), so it needs no canonicalization guess.
    assert table.get(27) == 3
    assert table.get("ACGT") == 3


def test_get_of_an_absent_kmer_is_none(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    # "GGCC" is also a palindrome (G<->C), distinct from "ACGT", and never
    # inserted -- a genuine miss.
    assert table.get("GGCC") is None
    assert table.get(999_999) is None


def test_getitem_returns_the_count_for_a_present_kmer(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    assert table["ACGT"] == 3
    assert table[27] == 3


def test_getitem_raises_keyerror_for_an_absent_kmer(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    with pytest.raises(KeyError):
        table["GGCC"]


def test_contains_reflects_presence_without_raising(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    assert "ACGT" in table
    assert "GGCC" not in table


def test_open_of_a_table_with_no_ktab_metadata_is_rejected(tmp_path):
    path = tmp_path / "not_a_table.parquet"
    write_plain_parquet_with_no_ktab_metadata(path)

    with pytest.raises(ValueError):
        fastdna.KmerTable.open(str(path))


def test_open_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        fastdna.KmerTable.open(str(tmp_path / "nope.parquet"))


def test_get_rejects_a_query_sequence_of_the_wrong_length(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    with pytest.raises(ValueError):
        table.get("AC")


def test_repr_reports_k_and_length(tmp_path):
    path = tmp_path / "counts.parquet"
    write_kmer_table(path, k=4, rows=[(27, 3)])

    table = fastdna.KmerTable.open(str(path))
    # `engine` joined the repr when `KmerTable` learned to open wide
    # (`kmer_bits`) tables too: with two widths behind one type, "which one
    # is this" is the first thing a repr has to answer.
    assert repr(table) == "KmerTable(k=4, len=1, engine=narrow)"


def test_open_of_an_empty_table_reports_zero_length_and_every_lookup_misses(tmp_path):
    path = tmp_path / "empty.parquet"
    write_kmer_table(path, k=4, rows=[])

    table = fastdna.KmerTable.open(str(path))
    assert len(table) == 0
    assert table.get("ACGT") is None
