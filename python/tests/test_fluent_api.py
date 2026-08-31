"""Tests for KmerCounts' chainable filter()/sort_by()/top() API.

These operate purely in Python (pyarrow.compute over the already-computed
table), not by re-counting, so they are correctness tests for pyarrow
compute usage, not benchmarks -- deliberately small inputs.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


@pytest.fixture
def counts(tmp_path):
    # Canonical 5-mers ACGTA and CGTAC occur 9 times each (3 reads x 3
    # each, see test_api.py's own EXPECTED_CANONICAL_COUNTS for the same
    # hand-checked reads); TTTTT/AAAAA-heavy reads add lower-frequency
    # k-mers so filter()/sort_by()/top() have more than one distinct value
    # to actually discriminate between.
    reads = ["ACGTACGTAC"] * 6 + ["TTTTTGGGGG"] * 2 + ["AAACCCGGGT"] * 1
    path = write_fastq(tmp_path, reads)
    return fastdna.count(str(path), k=5)


def test_filter_by_min_count_narrows_the_table(counts):
    filtered = counts.filter(min_count=5)

    assert filtered.distinct_kmers <= counts.distinct_kmers
    assert all(f >= 5 for f in filtered.table.column("frequency").to_pylist())


def test_filter_composes_across_calls(counts):
    once = counts.filter(min_count=2, max_count=10)
    twice = counts.filter(min_count=2).filter(max_count=10)

    assert once.table.to_pylist() == twice.table.to_pylist()


def test_filter_does_not_mutate_the_original(counts):
    before = counts.distinct_kmers
    counts.filter(min_count=999)

    assert counts.distinct_kmers == before, "filter() must return a new view, not mutate self"


def test_sort_by_frequency_descending_orders_correctly(counts):
    sorted_counts = counts.sort_by("frequency", descending=True)

    freqs = sorted_counts.table.column("frequency").to_pylist()
    assert freqs == sorted(freqs, reverse=True)


def test_top_returns_at_most_n_rows_and_the_highest_frequencies(counts):
    top2 = counts.top(2)

    assert len(top2) <= 2
    all_freqs = sorted(counts.table.column("frequency").to_pylist(), reverse=True)
    assert top2.table.column("frequency").to_pylist() == all_freqs[: len(top2)]


def test_chained_filter_sort_top_composes(counts):
    result = counts.filter(min_count=1).sort_by("frequency").top(3)

    assert len(result) <= 3
    freqs = result.table.column("frequency").to_pylist()
    assert freqs == sorted(freqs, reverse=True)
    assert all(f >= 1 for f in freqs)


def test_total_kmers_is_unaffected_by_filtering_or_truncation(counts):
    original_total = counts.total_kmers

    assert counts.filter(min_count=100).total_kmers == original_total
    assert counts.top(1).total_kmers == original_total


def test_distinct_kmers_reflects_the_current_view_not_the_original(counts):
    top1 = counts.top(1)

    assert top1.distinct_kmers == 1
    assert len(top1) == 1


def test_to_pandas_returns_a_dataframe_with_the_current_view(counts):
    pd = pytest.importorskip("pandas")

    df = counts.top(2).to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2
    # kmer_sequence is off by default (see count()'s own docstring); the
    # `counts` fixture above does not ask for it.
    assert set(df.columns) == {"kmer_u64", "frequency"}


# --------------------------------------------------------------------------
# top() rejects arguments pyarrow.Table.slice() would have accepted
# --------------------------------------------------------------------------


def test_top_rejects_arguments_pyarrow_would_have_accepted(counts):
    """`pyarrow.Table.slice` responds to None, negatives and floats instead
    of raising, and each response is plausible but wrong. Fixed here so
    `top()` does not inherit that behaviour.
    """
    # The worst failure: slice(0, None) means "to the end".
    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(None)

    # A negative n arrived at through arithmetic looked like "no k-mer passed".
    with pytest.raises(ValueError, match="n >= 0"):
        counts.top(-5)

    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(2.5)

    # bool is a subclass of int in Python; top(True) means nothing.
    with pytest.raises(TypeError, match="non-negative int"):
        counts.top(True)

    # The legitimate cases stay intact.
    assert len(counts.top(0)) == 0
    assert len(counts.top(10**9)) == len(counts)


# --------------------------------------------------------------------------
# filter() rejects negative bounds, but allows an empty (inverted) band
# --------------------------------------------------------------------------


def test_filter_rejects_negative_bounds_but_allows_an_empty_band(counts):
    """A negative threshold can never exclude anything over uint32
    frequencies, so `filter(min_count=-3)` used to return the whole table.
    An inverted band, by contrast, is a legitimate result and stays allowed.
    """
    with pytest.raises(ValueError, match="min_count must be >= 0"):
        counts.filter(min_count=-3)
    with pytest.raises(ValueError, match="max_count must be >= 0"):
        counts.filter(max_count=-1)
    with pytest.raises(TypeError, match="min_count must be an int"):
        counts.filter(min_count="5")
    with pytest.raises(TypeError, match="min_count must be an int"):
        counts.filter(min_count=True)

    # Inverted band: empty view, not an exception -- documented decision.
    assert len(counts.filter(min_count=10**6, max_count=1)) == 0

    # And the ordinary cases stay intact.
    assert len(counts.filter(min_count=0)) == len(counts)


# --------------------------------------------------------------------------
# sort_by() with an unknown column names the valid ones
# --------------------------------------------------------------------------


def test_sort_by_unknown_column_lists_the_valid_ones(counts):
    with pytest.raises(ValueError) as excinfo:
        counts.sort_by("freq")  # the real name is "frequency"

    message = str(excinfo.value)
    assert "freq" in message
    assert "frequency" in message, "the message must name the valid columns"
    assert "FieldRef" not in message, "Arrow's internal vocabulary must not leak through"
