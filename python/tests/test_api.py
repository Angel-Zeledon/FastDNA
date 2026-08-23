from __future__ import annotations  # list[str] annotations need 3.9+ without this (requires-python is 3.8+)

import gzip
import pathlib

import pytest
import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def write_fastq_gz(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq.gz"
    content = "".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads))
    with gzip.open(p, "wt") as f:
        f.write(content)
    return p


# ["ACGTACGTAC"] * 3 at k=5 has a hand-checkable answer: each read yields the
# raw 5-mers ACGTA, CGTAC, GTACG, TACGT, ACGTA, CGTAC (6 per read, 18 total
# across 3 reads). Canonicalizing (lexicographically smaller of a k-mer and
# its reverse complement, under A<C<G<T) collapses ACGTA/TACGT to "ACGTA"
# and CGTAC/GTACG to "CGTAC", each occurring 9 times.
EXPECTED_CANONICAL_COUNTS = {"ACGTA": 9, "CGTAC": 9}


def test_count_returns_arrow_table_with_correct_values(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)
    r = fastdna.count(str(path), k=5)
    tbl = r.table

    assert tbl.num_rows == 2
    assert set(tbl.column_names) == {"kmer_u64", "kmer_sequence", "frequency"}

    rows = {row["kmer_sequence"]: row["frequency"] for row in tbl.to_pylist()}
    assert rows == EXPECTED_CANONICAL_COUNTS
    assert r.total_kmers == 18
    assert r.distinct_kmers == 2
    assert r.k == 5
    assert len(r) == 2


def test_qc_reports_read_and_base_metrics(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)
    r = fastdna.count(str(path), k=5)
    qc = r.qc

    # "ACGTACGTAC" is 5 G/C out of 10 bases; 3 identical reads.
    assert qc["total_reads"] == 3
    assert qc["total_bases"] == 30
    assert qc["gc_bases"] == 15
    assert qc["gc_content_pct"] == pytest.approx(50.0)
    # Quality string is all 'I' (Phred 40), well above the default min_quality.
    assert qc["q20_pct"] == pytest.approx(100.0)
    assert qc["q30_pct"] == pytest.approx(100.0)


def test_min_count_and_max_count_filter_the_table(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)

    # Both canonical 5-mers occur exactly 9 times.
    above_actual = fastdna.count(str(path), k=5, min_count=10)
    assert above_actual.table.num_rows == 0
    assert above_actual.distinct_kmers == 0
    # total_kmers is the normalization basis and must survive pruning.
    assert above_actual.total_kmers == 18

    below_actual = fastdna.count(str(path), k=5, max_count=8)
    assert below_actual.table.num_rows == 0

    exact_band = fastdna.count(str(path), k=5, min_count=9, max_count=9)
    assert exact_band.table.num_rows == 2


def test_gz_input_is_transparently_decompressed(tmp_path):
    path = write_fastq_gz(tmp_path, ["ACGTACGTAC"] * 3)
    r = fastdna.count(str(path), k=5)

    assert r.total_kmers == 18
    assert r.distinct_kmers == 2


def test_version_is_exposed():
    assert isinstance(fastdna.__version__, str)
    assert fastdna.__version__


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        fastdna.count(str(tmp_path / "nope.fastq"), k=5)


def test_invalid_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGT"])
    with pytest.raises(ValueError):
        fastdna.count(str(path), k=99)


@pytest.mark.timeout(30)
def test_zero_threads_raises_valueerror_not_hang(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 100)
    with pytest.raises(ValueError):
        fastdna.count(str(path), k=5, threads=0)


def test_progress_receives_events(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 500)
    seen = []
    fastdna.count(str(path), k=5, progress=seen.append, progress_interval=10)
    assert len(seen) > 1, "expected several progress events"


def test_progress_counts_are_monotonic_for_the_consumer(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 2000)
    seen = []
    fastdna.count(str(path), k=5, progress=seen.append, progress_interval=10, threads=4)
    reads = [e for e in seen if isinstance(e, int)]
    assert reads == sorted(reads), "adapter must serialize out-of-order worker events"


def test_a_panicking_callback_becomes_runtimeerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 500)

    def boom(_):
        raise ValueError("callback exploded")

    with pytest.raises(RuntimeError):
        fastdna.count(str(path), k=5, progress=boom, progress_interval=10)


def test_peek_reports_read_geometry(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 50)
    p = fastdna.peek(str(path))
    assert p.n_reads_sampled == 50
    assert p.read_length == (20, 20, 20)
    assert 0.0 <= p.gc_content <= 1.0
    # median_read_length=20 -> 20 // 3 = 6, clamped to 1..=32 (no-op), then
    # rounded down to the nearest odd value since 6 itself is even: 5. This
    # is deterministic for this input, not merely "some odd k in range".
    assert p.suggest_k() == 5


def test_peek_rejects_an_unreasonable_n_reads(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 5)
    with pytest.raises(ValueError):
        fastdna.peek(str(path), n_reads=10_000_001)


def test_peek_stops_before_reading_the_whole_file(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 100)
    p = fastdna.peek(str(path), n_reads=10)
    assert p.n_reads_sampled == 10


def test_build_info_reports_avx2(tmp_path):
    info = fastdna.build_info()
    assert "version" in info and "avx2" in info
    assert isinstance(info["avx2"], bool)
    assert info["version"] == fastdna.__version__
    assert info["max_k"] == 32
