"""Tests for fastdna.KmerTable.filter_reads -- the S4 read-filtering layer
(docs/feature-gap-analysis.md), wrapping src/read_filter.rs via
`fastdna._core.filter_reads`.

Fixture tables are built with plain pyarrow, the same convention
`test_ktab.py`/`test_setops.py` already use. The reference table's only row
is k-mer `0` at `k=4`, the canonical encoding of the homopolymer "AAAA"
(`A` is `0b00` in every base slot -- see `src/kmer.rs`'s `base_to_bits` --
so "AAAA"'s forward encoding is already `0`, smaller than its own reverse
complement "TTTT"'s `255`, making "AAAA" its own canonical form; the same
fact `src/ktab.rs`'s own test fixtures rely on). A read made entirely of
"A"s therefore matches this reference at every window; a read built only
from "G"/"C" bases can never produce that encoding in either direction and
shares nothing with it.
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


def write_fastq(path: pathlib.Path, reads: List[Tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for read_id, seq in reads:
            f.write(f"@{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")


def reference_table(tmp_path: pathlib.Path) -> "fastdna.KmerTable":
    path = tmp_path / "reference.parquet"
    write_kmer_table(path, k=4, rows=[(0, 5)])
    return fastdna.KmerTable.open(str(path))


def test_discard_mode_removes_the_matching_read_and_keeps_the_rest(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("matching", "AAAAAAAA"), ("non_matching", "GCGCGCGC")])

    out = tmp_path / "filtered.fastq"
    stats = table.filter_reads(str(sample), mode="discard", output=str(out), min_fraction=0.5)

    assert stats.reads_total == 2
    assert stats.reads_written == 1
    text = out.read_text()
    assert "@non_matching" in text
    assert "@matching" not in text


def test_keep_mode_writes_only_the_matching_read(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("matching", "AAAAAAAA"), ("non_matching", "GCGCGCGC")])

    out = tmp_path / "filtered.fastq"
    stats = table.filter_reads(str(sample), mode="keep", output=str(out), min_fraction=0.5)

    assert stats.reads_written == 1
    text = out.read_text()
    assert "@matching" in text
    assert "@non_matching" not in text


def test_filter_reads_accepts_a_single_path_not_only_a_sequence(tmp_path):
    """`inputs` may be a single path-like value, not just a list of them --
    `KmerTable.filter_reads` wraps it in a one-element list itself (see its
    own docstring / `python/fastdna/__init__.py`).
    """
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("only", "AAAAAAAA")])

    out = tmp_path / "filtered.fastq"
    stats = table.filter_reads(sample, mode="keep", output=out, min_fraction=0.5)
    assert stats.reads_written == 1


def test_filter_reads_rejects_an_unknown_mode(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("r1", "AAAAAAAA")])

    with pytest.raises(ValueError):
        table.filter_reads(str(sample), mode="bogus", output=str(tmp_path / "out.fastq"))


def test_filter_reads_concatenates_multiple_input_files(tmp_path):
    table = reference_table(tmp_path)
    a = tmp_path / "a.fastq"
    b = tmp_path / "b.fastq"
    write_fastq(a, [("a1", "AAAAAAAA")])
    write_fastq(b, [("b1", "GCGCGCGC")])

    out = tmp_path / "filtered.fastq"
    stats = table.filter_reads([str(a), str(b)], mode="discard", output=str(out), min_fraction=0.5)

    assert stats.reads_total == 2
    assert stats.reads_written == 1
    assert "@b1" in out.read_text()


def test_filter_reads_defaults_to_a_min_fraction_of_point_one(tmp_path):
    # A single matching k-mer out of many is enough to clear the default
    # 0.1 threshold under --mode keep: "AAAAGGGGGG" (10 bases, k=4) has 7
    # four-mer windows, exactly one of which ("AAAA", the first) is the
    # reference k-mer -- 1/7 ~= 0.143, just above the default 0.1.
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("mostly_g", "AAAAGGGGGG")])

    out = tmp_path / "filtered.fastq"
    stats = table.filter_reads(str(sample), mode="keep", output=str(out))
    assert stats.reads_written == 1
