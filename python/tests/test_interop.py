"""Tests for `fastdna.interop` -- the in-memory-sequence convenience path.

The correctness check that matters here is not "it runs without crashing"
but "it produces identical counts to the direct file-based path": these
functions are documented as writing a temp FASTQ and calling
`fastdna.count()`/`fastdna.sketch()` on it, so the only thing worth
verifying is that the wiring in between is exact.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna
from fastdna.interop import count_from_sequences, sketch_from_sequences


def write_fastq(tmp_path: pathlib.Path, reads: list[str]) -> pathlib.Path:
    p = tmp_path / "sample.fastq"
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


READS = ["ACGTACGTAC"] * 6 + ["TTTTTGGGGG"] * 2 + ["AAACCCGGGT"] * 1


def test_count_from_plain_strings_matches_direct_file_count(tmp_path):
    direct_path = write_fastq(tmp_path, READS)
    direct = fastdna.count(str(direct_path), k=5)

    via_interop = count_from_sequences(READS, k=5)

    assert via_interop.distinct_kmers == direct.distinct_kmers
    assert via_interop.total_kmers == direct.total_kmers
    assert via_interop.k == direct.k
    assert (
        via_interop.table.sort_by("kmer_sequence").to_pylist()
        == direct.table.sort_by("kmer_sequence").to_pylist()
    )


def test_count_from_id_sequence_pairs_matches_direct_file_count(tmp_path):
    direct_path = write_fastq(tmp_path, READS)
    direct = fastdna.count(str(direct_path), k=5)

    pairs = [(f"read{i}", seq) for i, seq in enumerate(READS)]
    via_interop = count_from_sequences(pairs, k=5)

    assert via_interop.distinct_kmers == direct.distinct_kmers
    assert via_interop.total_kmers == direct.total_kmers


def test_count_from_sequences_passes_through_count_kwargs(tmp_path):
    via_interop = count_from_sequences(READS, k=5, min_count=5)

    assert all(f >= 5 for f in via_interop.table.column("frequency").to_pylist())


def test_count_from_sequences_raises_on_empty_input():
    with pytest.raises(ValueError):
        count_from_sequences([], k=5)


def test_count_from_sequences_skips_empty_strings(tmp_path):
    direct_path = write_fastq(tmp_path, READS)
    direct = fastdna.count(str(direct_path), k=5)

    via_interop = count_from_sequences(READS + ["", ""], k=5)

    assert via_interop.distinct_kmers == direct.distinct_kmers
    assert via_interop.total_kmers == direct.total_kmers


def test_sketch_from_sequences_matches_direct_file_sketch(tmp_path):
    reads = ["ACGTACGTACGTACGTACGTACGT"] * 20
    direct_path = write_fastq(tmp_path, reads)
    direct = fastdna.sketch(str(direct_path), k=5, sketch_size=100)

    via_interop = sketch_from_sequences(reads, k=5, sketch_size=100)

    assert via_interop.jaccard(direct) == pytest.approx(1.0)
    assert via_interop.k == direct.k
    assert via_interop.sketch_size == direct.sketch_size


class _FakeSeqRecord:
    """A minimal Biopython `SeqRecord` stand-in -- just enough duck typing
    (`.seq`, `.id`) to exercise the interop module's SeqRecord branch
    without depending on Biopython being installed for this test.
    """

    def __init__(self, seq, id):
        self.seq = seq
        self.id = id
        self.letter_annotations = {}


def test_count_from_duck_typed_seqrecords_matches_direct_file_count(tmp_path):
    direct_path = write_fastq(tmp_path, READS)
    direct = fastdna.count(str(direct_path), k=5)

    records = [_FakeSeqRecord(seq, f"r{i}") for i, seq in enumerate(READS)]
    via_interop = count_from_sequences(records, k=5)

    assert via_interop.distinct_kmers == direct.distinct_kmers
    assert via_interop.total_kmers == direct.total_kmers


def test_count_from_duck_typed_seqrecord_with_real_quality_uses_it(tmp_path):
    # A SeqRecord carrying real per-base Phred scores low enough to trigger
    # quality trimming (min_quality default is 20.0) should behave
    # differently from the same sequence with synthesized Q40 quality --
    # this confirms real quality, when present, is actually used rather
    # than silently overwritten by the uniform default.
    seq = "ACGTACGTAC"
    low_quality_record = _FakeSeqRecord(seq, "low_q")
    low_quality_record.letter_annotations = {"phred_quality": [2] * len(seq)}

    result = count_from_sequences([low_quality_record], k=5, min_quality=20.0)

    # The whole read should have been trimmed away by low quality, leaving
    # no k-mers -- unlike the synthesized-quality path, which never trims.
    assert result.total_kmers == 0


def test_biopython_seqrecords_real_objects(tmp_path):
    Bio_SeqIO = pytest.importorskip("Bio.SeqIO")
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    direct_path = write_fastq(tmp_path, READS)
    direct = fastdna.count(str(direct_path), k=5)

    records = [SeqRecord(Seq(seq), id=f"r{i}") for i, seq in enumerate(READS)]
    via_interop = count_from_sequences(records, k=5)

    assert via_interop.distinct_kmers == direct.distinct_kmers
    assert via_interop.total_kmers == direct.total_kmers

    # Also exercise real Bio.SeqIO.parse() end to end against a real FASTQ
    # file on disk, since that's the actual workflow this module targets.
    parsed = list(Bio_SeqIO.parse(str(direct_path), "fastq"))
    via_parse = count_from_sequences(parsed, k=5)
    assert via_parse.distinct_kmers == direct.distinct_kmers
    assert via_parse.total_kmers == direct.total_kmers
