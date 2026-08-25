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


def test_mash_distance_of_identical_files_is_zero(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 20
    a = write_fastq(tmp_path, "a.fastq", reads)
    b = write_fastq(tmp_path, "b.fastq", reads)

    s1 = fastdna.sketch(str(a), k=5)
    s2 = fastdna.sketch(str(b), k=5)

    assert s1.mash_distance(s2) == pytest.approx(0.0)


def test_mash_distance_of_disjoint_files_is_one(tmp_path):
    a = write_fastq(tmp_path, "a.fastq", ["AAAAAAAAAAAAAAAAAAAA"] * 20)
    b = write_fastq(tmp_path, "b.fastq", ["CCCCCCCCCCCCCCCCCCCC"] * 20)

    s1 = fastdna.sketch(str(a), k=5)
    s2 = fastdna.sketch(str(b), k=5)

    assert s1.mash_distance(s2) == pytest.approx(1.0)


def test_mash_distance_rejects_mismatched_k(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)

    s1 = fastdna.sketch(str(path), k=5)
    s2 = fastdna.sketch(str(path), k=7)

    with pytest.raises(ValueError):
        s1.mash_distance(s2)


def test_compare_all_returns_one_row_per_unordered_pair(tmp_path):
    paths = [
        write_fastq(tmp_path, f"s{i}.fastq", ["ACGTACGTACGTACGTACGTACGT"] * 10)
        for i in range(4)
    ]

    result = fastdna.compare_all([str(p) for p in paths], k=5)

    # n=4 -> n*(n-1)/2 = 6 unordered pairs, no self-comparisons, no
    # duplicate orderings of the same pair.
    assert result.num_rows == 6
    assert set(result.column_names) == {"sample_a", "sample_b", "jaccard"}
    pairs = set(zip(result.column("sample_a").to_pylist(), result.column("sample_b").to_pylist()))
    assert len(pairs) == 6
    for a, b in pairs:
        assert a != b


def test_compare_all_identical_files_score_one_on_jaccard(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 10
    paths = [write_fastq(tmp_path, f"s{i}.fastq", reads) for i in range(3)]

    result = fastdna.compare_all([str(p) for p in paths], k=5)

    assert all(v == pytest.approx(1.0) for v in result.column("jaccard").to_pylist())


def test_compare_all_supports_mash_distance_metric(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 10
    paths = [write_fastq(tmp_path, f"s{i}.fastq", reads) for i in range(3)]

    result = fastdna.compare_all([str(p) for p in paths], k=5, metric="mash_distance")

    assert "mash_distance" in result.column_names
    assert all(v == pytest.approx(0.0) for v in result.column("mash_distance").to_pylist())


def test_compare_all_rejects_an_unknown_metric(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 5)

    with pytest.raises(ValueError):
        fastdna.compare_all([str(path), str(path)], metric="not_a_real_metric")


# ---------------------------------------------------------------------------
# fastdna.frac_sketch() / load_frac_sketch() -- FracMinHash (scaled MinHash)
# ---------------------------------------------------------------------------


def test_frac_sketch_identical_files_have_containment_and_jaccard_one(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 20
    a = write_fastq(tmp_path, "a.fastq", reads)
    b = write_fastq(tmp_path, "b.fastq", reads)

    s1 = fastdna.frac_sketch(str(a), k=5, scale=4)
    s2 = fastdna.frac_sketch(str(b), k=5, scale=4)

    assert s1.jaccard(s2) == pytest.approx(1.0)
    assert s1.containment(s2) == pytest.approx(1.0)


def test_frac_sketch_disjoint_alphabets_have_containment_and_jaccard_zero(tmp_path):
    a = write_fastq(tmp_path, "a.fastq", ["AAAAAAAAAAAAAAAAAAAA"] * 20)
    b = write_fastq(tmp_path, "b.fastq", ["CCCCCCCCCCCCCCCCCCCC"] * 20)

    s1 = fastdna.frac_sketch(str(a), k=5, scale=4)
    s2 = fastdna.frac_sketch(str(b), k=5, scale=4)

    assert s1.jaccard(s2) == pytest.approx(0.0)
    assert s1.containment(s2) == pytest.approx(0.0)


def test_frac_sketch_mismatched_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)

    s1 = fastdna.frac_sketch(str(path), k=5, scale=4)
    s2 = fastdna.frac_sketch(str(path), k=7, scale=4)

    with pytest.raises(ValueError):
        s1.jaccard(s2)


def test_frac_sketch_mismatched_scale_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)

    s1 = fastdna.frac_sketch(str(path), k=5, scale=4)
    s2 = fastdna.frac_sketch(str(path), k=5, scale=8)

    with pytest.raises(ValueError):
        s1.containment(s2)


def test_frac_sketch_exposes_k_and_scale(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 5)

    s = fastdna.frac_sketch(str(path), k=9, scale=50)

    assert s.k == 9
    assert s.scale == 50


def test_frac_sketch_save_and_load_round_trips(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTACGTACGTACGT"] * 10)
    original = fastdna.frac_sketch(str(path), k=5, scale=4)

    out = tmp_path / "sample.frac_sketch.json"
    original.save(str(out))
    loaded = fastdna.load_frac_sketch(str(out))

    assert loaded.k == original.k
    assert loaded.scale == original.scale
    assert loaded.jaccard(original) == pytest.approx(1.0)


def test_frac_sketch_load_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        fastdna.load_frac_sketch(str(tmp_path / "nope.frac_sketch.json"))


def test_frac_sketch_repr_reports_k_and_scale():
    class _FakeRaw:
        k = 21
        scale = 1000

    sketch = fastdna.FracSketch(_FakeRaw())
    assert repr(sketch) == "FracSketch(k=21, scale=1000)"
