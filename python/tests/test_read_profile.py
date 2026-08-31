"""Tests for fastdna.KmerTable.profile_reads -- the S3 per-read k-mer
profiling layer (docs/feature-gap-analysis.md), wrapping
src/read_profile.rs via `fastdna._core.profile_reads`.

Same fixture convention as `test_read_filter.py`: the reference table's only
row is k-mer `0` at `k=4`, the canonical encoding of the homopolymer "AAAA"
("A" is `0b00` in every base slot, already smaller than its own reverse
complement "TTTT"'s `255`, so "AAAA" is its own canonical form). A read made
entirely of "A"s therefore has every one of its k-mers match this single
reference row; a read built only from "G"/"C" bases can never produce that
encoding in either direction.
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


def test_profile_reads_writes_one_run_for_a_fully_matching_read(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    # 8 "A"s at k=4 -> 5 canonical k-mers, every one of them "AAAA" (count 5
    # in the reference).
    write_fastq(sample, [("r1", "AAAAAAAA")])

    profile_out = tmp_path / "profile.parquet"
    summary_out = tmp_path / "summary.parquet"
    stats = table.profile_reads(str(sample), output=str(profile_out), summary=str(summary_out))

    assert stats.reads_total == 1
    assert stats.reads_profiled == 1

    profile = pq.read_table(str(profile_out)).to_pylist()
    assert len(profile) == 1, "a uniform-count read must collapse to exactly one RLE run"
    row = profile[0]
    assert row["read_id"] == "r1"
    assert row["start"] == 0
    assert row["run_length"] == 5
    assert row["count"] == 5

    summary = pq.read_table(str(summary_out)).to_pylist()
    assert len(summary) == 1
    srow = summary[0]
    assert srow["read_id"] == "r1"
    assert srow["n_kmers"] == 5
    assert srow["n_present_kmers"] == 5
    assert srow["min_count"] == 5
    assert srow["median_count"] == 5.0
    assert srow["max_count"] == 5


def test_profile_reads_reports_absent_kmers_as_count_zero(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("r1", "GCGCGCGC")])

    profile_out = tmp_path / "profile.parquet"
    summary_out = tmp_path / "summary.parquet"
    table.profile_reads(str(sample), output=str(profile_out), summary=str(summary_out))

    profile = pq.read_table(str(profile_out)).to_pylist()
    assert all(row["count"] == 0 for row in profile)

    summary = pq.read_table(str(summary_out)).to_pylist()
    assert summary[0]["n_present_kmers"] == 0
    assert summary[0]["n_kmers"] > 0
    assert summary[0]["min_count"] == 0
    assert summary[0]["max_count"] == 0


def test_profile_reads_of_a_read_shorter_than_k_is_a_null_summary_row_and_no_profile_rows(tmp_path):
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("short", "ACG")])

    profile_out = tmp_path / "profile.parquet"
    summary_out = tmp_path / "summary.parquet"
    stats = table.profile_reads(str(sample), output=str(profile_out), summary=str(summary_out))

    assert stats.reads_total == 1
    assert stats.reads_profiled == 0, "a read with no k-mers of its own is not counted as profiled"

    profile = pq.read_table(str(profile_out)).to_pylist()
    assert profile == []

    summary = pq.read_table(str(summary_out)).to_pylist()
    assert len(summary) == 1
    srow = summary[0]
    assert srow["n_kmers"] == 0
    assert srow["n_present_kmers"] == 0
    assert srow["min_count"] is None
    assert srow["median_count"] is None
    assert srow["max_count"] is None


def test_profile_reads_accepts_a_single_path_not_only_a_sequence(tmp_path):
    """`inputs` may be a single path-like value, not just a list of them --
    `KmerTable.profile_reads` wraps it in a one-element list itself, the
    same convention `filter_reads` already uses.
    """
    table = reference_table(tmp_path)
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [("only", "AAAAAAAA")])

    stats = table.profile_reads(sample, output=tmp_path / "profile.parquet", summary=tmp_path / "summary.parquet")
    assert stats.reads_total == 1


def test_profile_reads_concatenates_multiple_input_files(tmp_path):
    table = reference_table(tmp_path)
    a = tmp_path / "a.fastq"
    b = tmp_path / "b.fastq"
    write_fastq(a, [("a1", "AAAAAAAA")])
    write_fastq(b, [("b1", "GCGCGCGC")])

    profile_out = tmp_path / "profile.parquet"
    summary_out = tmp_path / "summary.parquet"
    stats = table.profile_reads([str(a), str(b)], output=str(profile_out), summary=str(summary_out))

    assert stats.reads_total == 2
    summary_ids = {row["read_id"] for row in pq.read_table(str(summary_out)).to_pylist()}
    assert summary_ids == {"a1", "b1"}
