"""Tests for fastdna.estimate_spectrum() -- ntCard-style streaming k-mer
frequency-spectrum estimation (`docs/feature-gap-analysis.md`'s S7(a); Rust
core in `src/ntcard.rs`).

The estimator's actual accuracy is characterized and asserted against exact
spectra in `src/ntcard.rs`'s own Rust test suite (in particular
`ntcard_matches_the_exact_spectrum_within_a_measured_tolerance`); this file
only proves the FFI wiring itself -- signature, parameter plumbing, return
shape, error mapping -- the same scope `test_cardinality.py` covers for
`estimate_cardinality`.
"""
from __future__ import annotations

import pathlib

import pytest

import fastdna


def write_fastq(tmp_path: pathlib.Path, reads: list[str], name: str = "sample.fastq") -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def test_returns_a_plain_int_keyed_dict_matching_kmercounts_spectrum_shape(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTAC"] * 50)

    spectrum = fastdna.estimate_spectrum(str(path), k=5)

    assert isinstance(spectrum, dict)
    assert spectrum, "a non-empty file must produce a non-empty spectrum"
    for depth, count in spectrum.items():
        assert isinstance(depth, int) and depth >= 1
        assert isinstance(count, int) and count >= 0


def test_single_window_reads_report_the_exact_repeated_depth(tmp_path):
    # "AAAA" is exactly k=4 bases long, so every read yields exactly one
    # canonical k-mer occurrence -- the bucket-winner count `NtCardSketch`
    # tracks is exact (see `src/ntcard.rs`'s module doc comment), so the
    # one reported depth must be exactly the read count, even though the
    # distinct-k-mer count *at* that depth is an estimate.
    path = write_fastq(tmp_path, ["AAAA"] * 200)

    spectrum = fastdna.estimate_spectrum(str(path), k=4)

    assert list(spectrum.keys()) == [200]


def test_accepts_a_single_path_or_a_sequence_of_paths(tmp_path):
    # Two lanes of the same sample: aggregated into one spectrum, matching
    # `count()`'s own multi-file convention.
    lane1 = write_fastq(tmp_path, ["AAAA"] * 100, name="lane1.fastq")
    lane2 = write_fastq(tmp_path, ["AAAA"] * 100, name="lane2.fastq")

    single = fastdna.estimate_spectrum(str(lane1), k=4)
    combined = fastdna.estimate_spectrum([str(lane1), str(lane2)], k=4)

    assert list(single.keys()) == [100]
    assert list(combined.keys()) == [200]


def test_max_frequency_caps_the_reported_depth(tmp_path):
    path = write_fastq(tmp_path, ["AAAA"] * 200)

    spectrum = fastdna.estimate_spectrum(str(path), k=4, max_frequency=10)

    assert all(depth <= 10 for depth in spectrum), f"depth exceeded the cap: {spectrum}"


def test_precision_is_configurable(tmp_path):
    path = write_fastq(tmp_path, ["ACGTACGTACGTACGTACGT"] * 10)

    # Smoke test for parameter plumbing, not an accuracy claim -- that is
    # `src/ntcard.rs`'s own Rust test suite.
    low = fastdna.estimate_spectrum(str(path), k=5, precision=7)
    high = fastdna.estimate_spectrum(str(path), k=5, precision=16)

    assert isinstance(low, dict)
    assert isinstance(high, dict)


def test_invalid_k_raises_valueerror(tmp_path):
    path = write_fastq(tmp_path, ["ACGT"])

    with pytest.raises(ValueError):
        fastdna.estimate_spectrum(str(path), k=99)


def test_is_directly_consumable_by_genomescope_profile_genome(tmp_path):
    # The contract this module exists to satisfy: the estimated spectrum
    # must be swappable in wherever an exact `KmerCounts.spectrum()` is
    # accepted. A single repeated template at a plausible depth is enough
    # to exercise the real `profile_genome` code path end to end (not to
    # assert a particular genome-size answer, which is `test_genomescope.py`'s
    # job against an *exact* spectrum).
    from fastdna.genomescope import profile_genome

    reads = ["ACGTTGCAACCGGTTAGATCGATCGATCGGATCC" * 5] * 30
    path = write_fastq(tmp_path, reads)

    spectrum = fastdna.estimate_spectrum(str(path), k=21)

    # A `ValueError` here ("no coverage peak", likely given how little
    # distinct signal this tiny repeated-template fixture has) is an
    # expected, documented outcome -- what this test actually guards
    # against is a `TypeError`/shape error, which would mean the estimated
    # spectrum's dict did *not* interoperate with the exact-spectrum
    # contract `profile_genome` expects.
    try:
        profile = profile_genome(spectrum, k=21)
    except ValueError:
        pass
    else:
        assert profile.k == 21
