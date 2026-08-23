import pathlib
import pytest
import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def test_count_returns_arrow_table(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 3)
    r = fastdna.count(str(path), k=5)
    tbl = r.table
    assert tbl.num_rows > 0
    assert set(tbl.column_names) == {"kmer_u64", "kmer_sequence", "frequency"}


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        fastdna.count(str(tmp_path / "nope.fastq"), k=5)


def test_invalid_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGT"])
    with pytest.raises(ValueError):
        fastdna.count(str(path), k=99)


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
