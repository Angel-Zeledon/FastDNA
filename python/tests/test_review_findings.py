"""FAILING regression tests written during the 2026-08-27 correctness review.

Each test here pins a defect that exists at commit bbe9301. They are expected
to FAIL until the defect is fixed; none of them is a new feature request.

FINDING 1 -- the Python FFI entry points added today (`_core.filter_reads`,
`_core.ktab_union` / `ktab_intersect` / `ktab_diff`) have no output-overwrite
guard, while their CLI counterparts do:

  * `main.rs::run_filter` explicitly rejects an `--output` that resolves to
    the reference `--table` or to any `--input` file, citing "irreversible
    data loss";
  * `main.rs::guard_against_setops_output_overwrite` rejects a set
    operation whose `--output` is one of its own input tables, citing "a
    truncated output file landing on top of a live input would corrupt the
    very read still in progress";
  * `main.rs::guard_against_input_overwrite` does the same for `count`.

`ffi.rs::filter_reads` / `ktab_union` / `ktab_intersect` / `ktab_diff` call
`read_filter::run_filter` / `export::export_pairs_parquet` directly and run
none of those checks, so the identical operation issued from Python silently
destroys the user's file. `atomic::AtomicFile` does not save it: the temp
file is renamed over the destination *after* the input has been fully
consumed, so the destination ends up holding the (frequently empty) result.
"""
from __future__ import annotations

import pathlib
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fastdna


def write_kmer_table(path: pathlib.Path, k: int, rows: List[Tuple[int, int]]) -> None:
    schema = pa.schema(
        [
            pa.field("kmer_u64", pa.uint64(), nullable=False),
            pa.field("frequency", pa.uint32(), nullable=False),
        ]
    ).with_metadata({"fastdna.sorted_by": "kmer_u64", "fastdna.k": str(k)})
    pq.write_table(
        pa.table(
            {
                "kmer_u64": pa.array([r[0] for r in rows], type=pa.uint64()),
                "frequency": pa.array([r[1] for r in rows], type=pa.uint32()),
            },
            schema=schema,
        ),
        str(path),
    )


def write_fastq(path: pathlib.Path, reads: List[Tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for read_id, seq in reads:
            f.write(f"@{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")


def test_filter_reads_must_not_overwrite_its_own_input_fastq(tmp_path):
    """`KmerTable.filter_reads(output=<one of its own inputs>)` currently
    replaces the user's FASTQ with the filter result -- which, under
    `mode="keep"` against a reference the sample does not match, is an empty
    file. 5 reads in, 0 reads written, 0 bytes left on disk: the input is
    gone and unrecoverable.

    `fastdna filter --input sample.fastq -o sample.fastq` on the command
    line is rejected up front by `main.rs::run_filter` for exactly this
    reason. The Python call must behave the same way.
    """
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [(f"r{i}", "GCGCGCGC") for i in range(5)])
    original = sample.read_bytes()

    reference = tmp_path / "ref.parquet"
    write_kmer_table(reference, k=4, rows=[(0, 1)])  # "AAAA" only -- no overlap
    table = fastdna.KmerTable.open(str(reference))

    with pytest.raises(ValueError):
        table.filter_reads(str(sample), mode="keep", output=str(sample), min_fraction=0.5)

    assert sample.read_bytes() == original, (
        "filter_reads overwrote its own input FASTQ: "
        f"{len(original)} bytes before, {sample.stat().st_size} bytes after"
    )


def test_filter_reads_must_not_overwrite_the_reference_table(tmp_path):
    """Same defect, aimed at the reference `KmerTable`'s own file: the table
    is streamed into a resident `ReferenceIndex` first, so the run
    "succeeds" and then renames an empty FASTQ over the Parquet table. The
    table is destroyed and no longer openable.
    """
    sample = tmp_path / "sample.fastq"
    write_fastq(sample, [(f"r{i}", "GCGCGCGC") for i in range(5)])

    reference = tmp_path / "ref.parquet"
    write_kmer_table(reference, k=4, rows=[(0, 1)])
    table = fastdna.KmerTable.open(str(reference))
    original_size = reference.stat().st_size

    with pytest.raises(ValueError):
        table.filter_reads(str(sample), mode="keep", output=str(reference), min_fraction=0.5)

    assert reference.stat().st_size == original_size
    fastdna.KmerTable.open(str(reference))  # must still be a valid table


def test_set_operations_must_not_overwrite_one_of_their_own_input_tables(tmp_path):
    """`a.union(b, output=a_path)` currently replaces `a` with the union.
    `fastdna union --input a.parquet b.parquet --output a.parquet` is
    rejected by `main.rs::guard_against_setops_output_overwrite`; the Python
    method must be too.

    The `intersect` half of this is the destructive one in practice: an
    empty intersection leaves the caller's input table holding zero rows.
    """
    a_path = tmp_path / "a.parquet"
    b_path = tmp_path / "b.parquet"
    write_kmer_table(a_path, 4, [(1, 1), (2, 1)])
    write_kmer_table(b_path, 4, [(9, 1)])
    a = fastdna.KmerTable.open(str(a_path))
    b = fastdna.KmerTable.open(str(b_path))

    with pytest.raises(ValueError):
        a.intersect(b, output=str(a_path))

    assert len(fastdna.KmerTable.open(str(a_path))) == 2, (
        "intersect(output=<its own input>) replaced a.parquet with the "
        "(empty) intersection"
    )
