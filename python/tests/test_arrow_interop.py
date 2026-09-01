"""Verifies FastDNA's `KmerCounts.table` is consumed without friction by the
Arrow-native Python data tools people actually reach for, not just by
`pyarrow` itself. `.table` is documented as a zero-copy `pyarrow.Table`
(§9.2 of the design doc); these tests are the evidence for "consumed
without friction" rather than an assertion of it.
"""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest

import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


@pytest.fixture
def counts(tmp_path):
    # A handful of distinct 5-mers with varied frequencies, enough to give
    # the partitioning test below more than one bucket to actually split.
    #
    # with_sequence=True: this whole module is about interop with the
    # table FastDNA hands back, including the decoded kmer_sequence column
    # (off by default -- see count()'s own docstring), which
    # test_polars_from_arrow_preserves_schema_and_values below asserts on.
    reads = ["ACGTACGTAC"] * 6 + ["TTTTTGGGGG"] * 2 + ["AAACCCGGGT"] * 1
    path = write_fastq(tmp_path, reads)
    return fastdna.count(str(path), k=5, with_sequence=True)


def test_polars_from_arrow_preserves_schema_and_values(counts):
    pl = pytest.importorskip("polars")

    df = pl.from_arrow(counts.table)

    assert df.shape[0] == counts.table.num_rows
    assert set(df.columns) == {"kmer_u64", "kmer_sequence", "frequency"}
    # A round-trip through Polars must not silently change values: sum of
    # `frequency` must equal FastDNA's own `total_kmers` (every count
    # accounted for, none dropped or altered by the conversion).
    assert df["frequency"].sum() == counts.total_kmers
    assert df["kmer_u64"].dtype == pl.UInt64
    assert df["frequency"].dtype == pl.UInt32


def test_polars_can_filter_and_sort_the_zero_copy_table(counts):
    pl = pytest.importorskip("polars")

    df = pl.from_arrow(counts.table)
    top = df.sort("frequency", descending=True).head(1)

    assert top["frequency"][0] == df["frequency"].max()


def test_duckdb_queries_the_table_directly(counts):
    duckdb = pytest.importorskip("duckdb")

    # `counts_table` is a local variable that duckdb's relation API
    # resolves by name straight out of the calling frame -- no explicit
    # `register()` call, no copy into DuckDB's own storage first. This is
    # exactly the "SELECT straight off the pyarrow.Table" workflow the
    # design doc's zero-copy claim is meant to make possible.
    counts_table = counts.table  # noqa: F841 (referenced by name in the SQL string below)

    row = duckdb.sql("SELECT count(*) AS n, sum(frequency) AS total FROM counts_table").fetchone()

    assert row[0] == counts.table.num_rows
    assert row[1] == counts.total_kmers


def test_duckdb_can_join_against_a_second_kmer_count(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    path_a = write_fastq(tmp_path, ["ACGTACGTAC"] * 6)
    path_b = write_fastq(tmp_path, ["ACGTACGTAC"] * 3 + ["TTTTTGGGGG"] * 2)
    # with_sequence=True: the join below is on kmer_sequence, off by
    # default (see count()'s own docstring).
    table_a = fastdna.count(str(path_a), k=5, with_sequence=True).table  # noqa: F841
    table_b = fastdna.count(str(path_b), k=5, with_sequence=True).table  # noqa: F841

    # A realistic two-sample comparison: k-mers present in both, by sequence.
    row = duckdb.sql(
        "SELECT count(*) FROM table_a JOIN table_b USING (kmer_sequence)"
    ).fetchone()

    assert row[0] >= 1, "ACGTA/CGTAC should be shared between these two samples"


def test_write_dataset_partitioned_parquet_round_trips(counts, tmp_path):
    # Buckets frequency into a small number of named bands so partitioning
    # produces a handful of real partitions instead of one file per
    # distinct frequency -- the same pattern a user would apply before
    # partitioning a real (much larger) k-mer table for efficient
    # downstream filtering (e.g. skip the whole "low" partition entirely).
    freq = counts.table.column("frequency")
    band = pc.if_else(
        pc.greater_equal(freq, 5),
        "high",
        pc.if_else(pc.greater_equal(freq, 2), "medium", "low"),
    )
    banded = counts.table.append_column("freq_band", band)

    out_dir = tmp_path / "kmer_dataset"
    # Hive-style partitioning (`freq_band=high/...`) explicitly: the
    # `partitioning=[...]` shorthand defaults to bare directory names
    # (`high/`, not `freq_band=high/`) -- Hive style is what DuckDB, Polars,
    # and Spark all auto-detect without being told the partition column
    # names, which is the point of using it here.
    partitioning = ds.partitioning(pa.schema([("freq_band", pa.string())]), flavor="hive")
    ds.write_dataset(
        banded, str(out_dir), format="parquet", partitioning=partitioning,
        existing_data_behavior="overwrite_or_ignore",
    )

    partition_dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert partition_dirs, "expected at least one freq_band=... partition directory"
    assert all(name.startswith("freq_band=") for name in partition_dirs)

    # Read the whole partitioned dataset back and confirm nothing was lost
    # or duplicated across partitions.
    read_back = ds.dataset(str(out_dir), format="parquet").to_table()
    assert read_back.num_rows == banded.num_rows
    assert pc.sum(read_back.column("frequency")).as_py() == counts.total_kmers
