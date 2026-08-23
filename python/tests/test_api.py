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
