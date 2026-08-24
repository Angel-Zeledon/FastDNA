"""Tests for fastdna.sketch()/load_sketch()/compare() -- the MinHash
fingerprinting API wrapping src/sketch.rs's GenomeSketch.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def test_identical_files_have_jaccard_one(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 20
    a = write_fastq(tmp_path, "a.fastq", reads)
    b = write_fastq(tmp_path, "b.fastq", reads)

    s1 = fastdna.sketch(str(a), k=5)
    s2 = fastdna.sketch(str(b), k=5)

    assert s1.jaccard(s2) == pytest.approx(1.0)
    assert s1.containment(s2) == pytest.approx(1.0)


def test_disjoint_alphabets_have_jaccard_zero(tmp_path):
    a = write_fastq(tmp_path, "a.fastq", ["AAAAAAAAAAAAAAAAAAAA"] * 20)
    b = write_fastq(tmp_path, "b.fastq", ["CCCCCCCCCCCCCCCCCCCC"] * 20)

    s1 = fastdna.sketch(str(a), k=5)
    s2 = fastdna.sketch(str(b), k=5)

    assert s1.jaccard(s2) == pytest.approx(0.0)


def test_mismatched_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)

    s1 = fastdna.sketch(str(path), k=5)
    s2 = fastdna.sketch(str(path), k=7)

    with pytest.raises(ValueError):
        s1.jaccard(s2)


def test_sketch_exposes_k_and_sketch_size(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 5)

    s = fastdna.sketch(str(path), k=9, sketch_size=50)

    assert s.k == 9
    assert s.sketch_size == 50


def test_save_and_load_round_trips(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)
    original = fastdna.sketch(str(path), k=5)

    out = tmp_path / "sample.sketch.json"
    original.save(str(out))
    loaded = fastdna.load_sketch(str(out))

    assert loaded.k == original.k
    assert loaded.sketch_size == original.sketch_size
    assert loaded.jaccard(original) == pytest.approx(1.0)


def test_compare_is_sugar_for_sketch_then_jaccard(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 20
    a = write_fastq(tmp_path, "a.fastq", reads)
    b = write_fastq(tmp_path, "b.fastq", reads)

    assert fastdna.compare(str(a), str(b), k=5) == pytest.approx(1.0)


def test_load_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        fastdna.load_sketch(str(tmp_path / "nope.sketch.json"))
