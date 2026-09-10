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
    # with_sequence=True: this test asserts on the decoded kmer_sequence
    # column, which is off by default (see count()'s own docstring).
    r = fastdna.count(str(path), k=5, with_sequence=True)
    tbl = r.table

    assert tbl.num_rows == 2
    assert set(tbl.column_names) == {"kmer_u64", "kmer_sequence", "frequency"}

    rows = {row["kmer_sequence"]: row["frequency"] for row in tbl.to_pylist()}
    assert rows == EXPECTED_CANONICAL_COUNTS
    assert r.total_kmers == 18
    assert r.distinct_kmers == 2
    assert r.k == 5
    assert len(r) == 2


def test_hpc_absorbs_a_run_internal_indel(tmp_path):
    # Same fixture as the Rust integration test
    # (pipeline_hpc_flag_absorbs_a_run_internal_indel_end_to_end): a clean
    # read with a run of 6 A's, and the same read with one A deleted from
    # inside that run -- the kind of error long-read platforms make, not
    # Illumina.
    clean = "GATCAAAAAATCG"
    with_deletion = "GATCAAAAATCG"
    path = write_fastq(tmp_path, [clean, with_deletion])

    without_hpc = fastdna.count(str(path), k=4)
    # (13-4+1) + (12-4+1) = 19: the raw, uncompressed read lengths.
    assert without_hpc.total_kmers == 19

    with_hpc = fastdna.count(str(path), k=4, hpc=True)
    # Both reads collapse to the identical 8-base "GATCATCG", so total
    # occurrences is exactly double a single compressed read's: (8-4+1)*2.
    assert with_hpc.total_kmers == 10
    assert with_hpc.total_kmers != without_hpc.total_kmers


def test_hpc_off_by_default_matches_hpc_false_explicitly(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)
    default = fastdna.count(str(path), k=5)
    explicit = fastdna.count(str(path), k=5, hpc=False)
    assert default.total_kmers == explicit.total_kmers
    assert default.distinct_kmers == explicit.distinct_kmers


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
    # 64, and one number for the whole package since sketching and the
    # estimators reached it too. See
    # `test_wide_engine.py::test_build_info_reports_one_ceiling`.
    assert info["max_k"] == 64
    # `max_k_sketch` was reported for two days, while sketching and the
    # estimators still stopped at 32. They reach 64 now, so a second
    # ceiling would describe a discrepancy that no longer exists.
    assert "max_k_sketch" not in info


def test_progress_true_drives_a_tqdm_bar(tmp_path):
    """`count(progress=True)` is a documented feature whose entire
    implementation -- `_progress._tqdm_sink` -- was never executed by any
    test. Coverage put `_progress.py` at 55% with this block the bulk of
    the gap.

    What is checked is that the bar advances to the real read count, not
    merely that the call returns: a sink that swallowed every event would
    pass a smoke test and show a bar stuck at zero.
    """
    tqdm = pytest.importorskip("tqdm")

    reads = ["ACGTACGTACGTACGTACGTACGT"] * 500
    path = write_fastq(tmp_path, reads)

    seen = []

    class RecordingBar(tqdm.tqdm):
        def update(self, n=1):
            seen.append(n)
            return super().update(n)

    # The sink builds its own bar with `tqdm.tqdm(...)`, so the class is
    # swapped rather than an instance injected.
    original = tqdm.tqdm
    tqdm.tqdm = RecordingBar
    try:
        result = fastdna.count(path, k=21, progress=True, progress_interval=50)
    finally:
        tqdm.tqdm = original

    assert result.total_kmers > 0
    assert seen, "progress=True produced no bar updates at all"
    assert sum(seen) == len(reads), (
        f"the bar must advance to every read: {sum(seen)} of {len(reads)}"
    )


def test_progress_true_without_tqdm_is_silent_not_an_error(tmp_path, monkeypatch):
    """tqdm is optional. `progress=True` without it must count normally and
    simply not draw anything -- never raise.
    """
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name == "tqdm" or name.startswith("tqdm."):
            raise ImportError("tqdm blocked for this test")
        return real_import(name, *args, **kwargs)

    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 20)
    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    try:
        result = fastdna.count(path, k=11, progress=True)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert result.distinct_kmers > 0


def test_progress_rejects_a_non_callable(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGT"] * 5)
    with pytest.raises(TypeError):
        fastdna.count(path, k=11, progress="yes please")


def test_to_polars_round_trips_the_table(tmp_path):
    """`to_polars()` is public and was never executed by a test."""
    pl = pytest.importorskip("polars")

    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 10)
    counts = fastdna.count(path, k=11)
    frame = counts.to_polars()

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == counts.table.column_names
    assert frame.height == counts.table.num_rows
