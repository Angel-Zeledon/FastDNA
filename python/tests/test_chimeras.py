"""Tests for fastdna.chimeras.scan_chimeras, wrapping src/chimera_scan.rs
via `fastdna._core.scan_chimeras`.

Mirrors `src/chimera_scan.rs`'s own unit tests at the Python-facing
boundary rather than re-deriving new ground truth: a hand-constructed
compositional shift (A/T-rich vs. G/C-rich halves) with a known junction
position, a compositionally uniform sequence that must not be flagged,
and the `InvalidConfigError` mapping for out-of-range parameters.
"""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from fastdna import _core
from fastdna.chimeras import scan_chimeras


def write_fasta(tmp_path: pathlib.Path, filename: str, records: dict) -> pathlib.Path:
    path = tmp_path / filename
    path.write_text("".join(f">{header}\n{seq}\n" for header, seq in records.items()))
    return path


def _at_gc_chimera(half: int = 1000) -> str:
    left = "AT" * (half // 2)
    right = "GC" * (half // 2)
    return left + right


def test_scan_chimeras_detects_a_hand_constructed_junction(tmp_path):
    seq = _at_gc_chimera(half=1000)
    path = write_fasta(tmp_path, "sample.fasta", {"contig_1 description": seq})

    table = scan_chimeras(str(path), window_size=200, step=50, k=4, threshold=0.5)
    assert isinstance(table, pa.Table)
    assert table.column_names == ["contig_id", "position", "divergence", "confidence"]

    rows = table.to_pylist()
    assert len(rows) == 1, f"expected exactly one clustered breakpoint, got {rows}"
    row = rows[0]
    assert row["contig_id"] == "sample::contig_1"
    assert abs(row["position"] - 1000) <= 200
    assert row["divergence"] > 0.9
    assert 0.0 <= row["confidence"] <= 1.0


def test_scan_chimeras_with_no_threshold_returns_the_unfiltered_profile(tmp_path):
    seq = _at_gc_chimera(half=1000)
    path = write_fasta(tmp_path, "sample.fasta", {"contig_1": seq})

    table = scan_chimeras(str(path), window_size=200, step=50, k=4, threshold=None)
    rows = table.to_pylist()
    # Candidate breakpoints on the (window=200, step=50) grid over a
    # 2000-base contig: positions 200, 250, ..., 1800 -> 33 candidates.
    assert len(rows) == 33
    # confidence == divergence when unfiltered (no threshold to score
    # against -- see chimera_scan.rs::Breakpoint::confidence).
    for row in rows:
        assert row["confidence"] == pytest.approx(row["divergence"])


def test_scan_chimeras_does_not_flag_a_compositionally_uniform_sequence(tmp_path):
    # Deterministic, non-repetitive pseudo-random sequence (Python's own
    # PRNG, fixed seed) -- a literal repeated motif would introduce its own
    # artificial periodicity, unlike real, non-chimeric genomic sequence.
    import random

    rng = random.Random(12345)
    seq = "".join(rng.choice("ACGT") for _ in range(8000))
    path = write_fasta(tmp_path, "uniform.fasta", {"contig_1": seq})

    # window=1000: see chimera_scan.rs's own test for why a too-small window
    # (e.g. 200) has enough multinomial sampling noise on its own to flag a
    # uniform sequence at a moderate threshold.
    table = scan_chimeras(str(path), window_size=1000, step=100, k=4, threshold=0.4)
    assert table.num_rows == 0


def test_scan_chimeras_labels_contigs_by_file_stem_and_header_across_several_files(tmp_path):
    seq = _at_gc_chimera(half=1000)
    path_a = write_fasta(tmp_path, "mag_a.fasta", {"k99_1 some description": seq})
    path_b = write_fasta(tmp_path, "mag_b.fasta", {"k99_1 a different assembly": seq})

    table = scan_chimeras([str(path_a), str(path_b)], window_size=200, step=50, threshold=0.5)
    contig_ids = sorted(set(table.column("contig_id").to_pylist()))
    assert contig_ids == ["mag_a::k99_1", "mag_b::k99_1"]


def test_scan_chimeras_accepts_a_single_path_not_wrapped_in_a_list(tmp_path):
    seq = _at_gc_chimera(half=1000)
    path = write_fasta(tmp_path, "sample.fasta", {"contig_1": seq})
    table = scan_chimeras(path, window_size=200, step=50, threshold=0.5)
    assert table.num_rows == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 2, "step": 50, "k": 4},  # window < k
        {"window_size": 200, "step": 0, "k": 4},  # step must be >= 1
        {"window_size": 200, "step": 50, "k": 0},  # k out of range
        {"window_size": 200, "step": 50, "k": 33},  # k out of range
        {"window_size": 200, "step": 50, "k": 4, "threshold": 1.5},  # threshold out of range
        {"window_size": 200, "step": 50, "k": 4, "threshold": -0.1},  # threshold out of range
    ],
)
def test_scan_chimeras_rejects_invalid_configuration(tmp_path, kwargs):
    seq = _at_gc_chimera(half=1000)
    path = write_fasta(tmp_path, "sample.fasta", {"contig_1": seq})
    with pytest.raises(_core.InvalidConfigError):
        scan_chimeras(str(path), **kwargs)


def test_scan_chimeras_rejects_invalid_configuration_before_touching_a_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.fasta"
    with pytest.raises(_core.InvalidConfigError):
        scan_chimeras(str(missing), window_size=200, step=50, k=0)
