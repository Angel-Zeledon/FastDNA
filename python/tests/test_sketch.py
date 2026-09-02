"""Tests for fastdna.sketch()/load_sketch()/compare() -- the MinHash
fingerprinting API wrapping src/sketch.rs's GenomeSketch.
"""
from __future__ import annotations

import json
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
# fastdna.sketch_from_kmers() -- the in-memory counterpart to sketch(),
# built from an already-counted k-mer set with no FASTQ file read at all.
# ---------------------------------------------------------------------------


def test_sketch_from_kmers_is_bit_identical_to_the_streaming_sketch(tmp_path):
    """The equivalence this whole function exists to guarantee: sketching
    from a sample's complete, unfiltered k-mer set (here, `fastdna.count()`'s
    own `kmer_u64` column at `min_count=1`, i.e. every distinct k-mer the
    file contains) must select the exact same bottom-k hashes as streaming
    the file directly with `fastdna.sketch()` -- not merely an equal
    Jaccard/mash_distance, but the identical saved `hashes` array, since
    both go through the same `sketch::GenomeSketch::from_kmers`/
    `from_reader` bottom-k selection over the same finalized hashes.
    """
    reads = [
        "ACGTACGTACGTACGTACGTACGT",
        "TTTTACGTGGGGCCCCAAAATTTT",
        "GGGGCCCCTTTTAAAAACGTACGT",
        "ACGTACGTACGTACGTACGTACGT",  # a repeat, on purpose
    ]
    path = write_fastq(tmp_path, "a.fastq", reads)
    k = 5
    sketch_size = 500

    streamed = fastdna.sketch(str(path), k=k, sketch_size=sketch_size)

    # min_count=1 (the default): every distinct k-mer the file contains,
    # unfiltered -- the same content fastdna.sketch()'s own streaming
    # construction sees.
    counted = fastdna.count(str(path), k=k, min_count=1)
    kmers = counted.table.column("kmer_u64")
    from_kmers = fastdna.sketch_from_kmers(kmers, k=k, sketch_size=sketch_size)

    assert from_kmers.k == streamed.k
    assert from_kmers.sketch_size == streamed.sketch_size
    assert from_kmers.jaccard(streamed) == pytest.approx(1.0)

    streamed_out = tmp_path / "streamed.sketch.json"
    from_kmers_out = tmp_path / "from_kmers.sketch.json"
    streamed.save(str(streamed_out))
    from_kmers.save(str(from_kmers_out))

    streamed_hashes = json.loads(streamed_out.read_text())["hashes"]
    from_kmers_hashes = json.loads(from_kmers_out.read_text())["hashes"]
    assert from_kmers_hashes == streamed_hashes


def test_sketch_from_kmers_ignores_input_order_and_repeats(tmp_path):
    """Every occurrence or only the distinct values present: the docstring's
    claim that this makes no difference to the result, pinned directly --
    mirrors the Rust-level
    `repeated_kmers_do_not_displace_distinct_ones_from_a_full_sketch` test,
    at the FFI boundary this wraps."""
    distinct = list(range(200))
    baseline = fastdna.sketch_from_kmers(distinct, k=21, sketch_size=32)

    repeated = []
    for _ in range(5):
        repeated.extend(reversed(distinct))
    with_repeats = fastdna.sketch_from_kmers(repeated, k=21, sketch_size=32)

    assert with_repeats.jaccard(baseline) == pytest.approx(1.0)


def test_sketch_from_kmers_accepts_a_pyarrow_array(tmp_path):
    """The realistic caller shape: a `CohortCounts`/`KmerCounts` column,
    not a plain Python list."""
    import pyarrow as pa

    kmers = pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.uint64())
    s = fastdna.sketch_from_kmers(kmers, k=5, sketch_size=10)
    assert s.k == 5
    assert s.sketch_size == 10


def test_sketch_from_kmers_rejects_an_out_of_range_k():
    for bad_k in (0, 33):
        with pytest.raises(ValueError):
            fastdna.sketch_from_kmers([1, 2, 3], k=bad_k, sketch_size=10)


def test_sketch_from_kmers_rejects_a_zero_sketch_size():
    with pytest.raises(ValueError):
        fastdna.sketch_from_kmers([1, 2, 3], k=21, sketch_size=0)


def test_sketch_from_kmers_of_an_empty_input_is_an_empty_sketch():
    s = fastdna.sketch_from_kmers([], k=21, sketch_size=10)
    assert s.k == 21
    assert s.sketch_size == 10


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
