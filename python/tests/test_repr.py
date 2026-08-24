"""Tests for `_repr_html_` on `KmerCounts` and `Sketch` -- notebook-friendly
rich display. Jupyter calls `_repr_html_()` automatically when a cell's
last expression is one of these objects; these tests just verify it
returns a non-empty string containing the facts a newcomer would want to
see, and that it never raises (a crashing `_repr_html_` is worse than the
default `repr()` it replaces).
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
    reads = ["ACGTACGTAC"] * 6 + ["TTTTTGGGGG"] * 2 + ["AAACCCGGGT"] * 1
    path = write_fastq(tmp_path, reads)
    return fastdna.count(str(path), k=5)


def test_kmer_counts_repr_html_contains_key_facts(counts):
    html = counts._repr_html_()

    assert isinstance(html, str)
    assert html.strip() != ""
    assert f"k={counts.k}" in html
    assert f"{counts.distinct_kmers:,}" in html
    assert f"{counts.total_kmers:,}" in html
    assert "<table" in html


def test_kmer_counts_repr_html_does_not_raise_on_empty_view(counts):
    empty = counts.filter(min_count=10 ** 9)

    html = empty._repr_html_()

    assert isinstance(html, str)
    assert html.strip() != ""


def test_kmer_counts_repr_html_fallback_without_pandas(counts, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("simulated: pandas not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    html = counts._repr_html_()

    assert isinstance(html, str)
    assert html.strip() != ""
    assert "<table" in html


def test_sketch_repr_html_contains_key_facts(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGTACGT"] * 20)
    s = fastdna.sketch(str(path), k=5, sketch_size=100)

    html = s._repr_html_()

    assert isinstance(html, str)
    assert html.strip() != ""
    assert f"k={s.k}" in html
    assert f"{s.sketch_size:,}" in html
    assert "MinHash" in html
