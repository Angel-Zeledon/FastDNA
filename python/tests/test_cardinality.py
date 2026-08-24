"""Tests for fastdna.estimate_cardinality() -- HyperLogLog distinct-k-mer
estimation over a whole file, in bounded memory.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str], name: str = "sample.fastq") -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def test_estimate_is_close_to_the_exact_count(tmp_path):
    # ACGTACGTAC repeated has a small, known set of canonical 5-mers;
    # cross-check against fastdna.count()'s own exact distinct_kmers.
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 50)
    exact = fastdna.count(str(path), k=5).distinct_kmers

    estimate = fastdna.estimate_cardinality(str(path), k=5)

    assert abs(estimate - exact) / max(exact, 1) < 0.5


def test_estimate_scales_with_genuinely_distinct_content(tmp_path):
    small = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 20, name="small.fastq")
    larger = write_fastq(
        tmp_path,
        [
            "ACGTACGTACGTACGTACGT",
            "TTTTTGGGGGCCCCCAAAAA",
            "GATCGATCGATCGATCGATC",
            "CATGCATGCATGCATGCATG",
        ]
        * 20,
        name="larger.fastq",
    )

    small_estimate = fastdna.estimate_cardinality(str(small), k=7)
    larger_estimate = fastdna.estimate_cardinality(str(larger), k=7)

    assert larger_estimate > small_estimate


def test_precision_is_configurable(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 10)

    # Both must run without error and return a plausible non-negative
    # estimate; this is a smoke test for the parameter plumbing, not a
    # precision/accuracy claim (that is HyperLogLog's own Rust-side test
    # suite in src/hll.rs).
    low = fastdna.estimate_cardinality(str(path), k=5, precision=7)
    high = fastdna.estimate_cardinality(str(path), k=5, precision=16)

    assert low >= 0.0
    assert high >= 0.0


def test_invalid_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGT"])

    with pytest.raises(ValueError):
        fastdna.estimate_cardinality(str(path), k=99)
