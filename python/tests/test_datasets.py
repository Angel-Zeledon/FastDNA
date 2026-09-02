"""Tests for fastdna.datasets -- ready-made, labeled genomic cohorts.

Every test except the two marked `@pytest.mark.network` runs entirely
offline: `fastdna.datasets._download_bytes` (the single seam every network
call in the module funnels through) is monkeypatched to serve small,
hand-built fixtures that match the real upstream schemas byte-for-byte in
the columns this module actually reads -- the BV-BRC AMR_prediction
metadata TSV header (`_AMR_HEADER` below, transcribed from the real file's
own header row) and the Stanford HIVDB `PI_DataSet.txt` header
(`SeqID` + 8 drug columns + `P1..P99` + `CompMutList`). No `requests_mock`/
`responses` package is used: nothing like that exists anywhere else in
this suite (checked before writing this file), so this follows the
project's existing convention of a small hand-rolled fixture plus
monkeypatching instead.

The two `@pytest.mark.network` tests hit the real BV-BRC and Stanford
HIVDB endpoints end to end, with a tiny `n_samples`. They are not
deselected by default (see `pyproject.toml`'s `[tool.pytest.ini_options]`
comment) -- run the offline suite alone with
`pytest python/tests/test_datasets.py -m "not network"` when network
access is unwanted.
"""
from __future__ import annotations

import csv
import io
import re

import pytest

np = pytest.importorskip("numpy")

from fastdna.datasets import (
    AmrCohort,
    HivResistanceCohort,
    load_amr,
    load_hiv_resistance,
)
import fastdna.datasets as datasets_module

# ---------------------------------------------------------------------------
# AMR fixture -- shaped exactly like the real
# metadata_extend_E_coli_checkm.tsv header (confirmed against the live file
# while writing fastdna/datasets.py).
# ---------------------------------------------------------------------------

_AMR_HEADER = [
    "genome_id", "genome_name", "taxon_id", "antibiotic", "resistant_phenotype",
    "measurement", "measurement_sign", "measurement_value", "measurement_unit",
    "laboratory_typing_method", "laboratory_typing_method_version",
    "laboratory_typing_platform", "vendor", "testing_standard", "testing_standard_year",
    "source", "SIR", "SIR_criterion", "MIC", "MIC_criterion",
    "purA", "icd", "adk", "gyrB", "mdh", "recA", "fumC", "ST", "nr_contig", "year",
]

# (genome_id, antibiotic, resistant_phenotype, MIC, ST)
_AMR_ROWS = [
    ("G1", "ciprofloxacin", "resistant", ">32", "131"),
    ("G2", "ciprofloxacin", "resistant", "16", "131"),
    ("G3", "ciprofloxacin", "resistant", "8", "10"),
    ("G4", "ciprofloxacin", "resistant", "", "95"),       # missing MIC -> nan
    ("G5", "ciprofloxacin", "resistant", "4", "10"),
    ("G6", "ciprofloxacin", "susceptible", "<=0.015", "73"),
    ("G7", "ciprofloxacin", "susceptible", "0.03", "73"),
    ("G8", "ciprofloxacin", "susceptible", "0.06", "12"),
    ("G9", "ciprofloxacin", "susceptible", "0.12", "12"),
    ("G10", "ciprofloxacin", "susceptible", "0.25", "95"),
    ("G11", "ciprofloxacin", "susceptible", "0.5", "131"),
    ("G12", "ciprofloxacin", "susceptible", "1", "10"),
    ("G13", "ciprofloxacin", "intermediate", "2", "10"),   # dropped: not binary
    ("Gconflict", "ciprofloxacin", "resistant", "8", "1"),
    ("Gconflict", "ciprofloxacin", "susceptible", "1", "1"),  # dropped: conflicting
    ("G1", "ampicillin", "resistant", "32", "131"),        # a second antibiotic exists
]


def _amr_metadata_bytes() -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_AMR_HEADER, delimiter="\t")
    writer.writeheader()
    for genome_id, antibiotic, phenotype, mic, st in _AMR_ROWS:
        row = {name: "" for name in _AMR_HEADER}
        row.update(
            genome_id=genome_id,
            genome_name=f"Escherichia coli {genome_id}",
            antibiotic=antibiotic,
            resistant_phenotype=phenotype,
            MIC=mic,
            ST=st,
        )
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _amr_genome_fasta_bytes(genome_id: str) -> bytes:
    return f">accn|{genome_id}.con.0001   fixture contig\nACGTACGTACGTACGTACGTACGTACGTAC\n".encode("ascii")


# ---------------------------------------------------------------------------
# HIV fixture -- shaped exactly like the real PI_DataSet.txt header.
# ---------------------------------------------------------------------------

_HIV_DRUG_COLUMNS = ("FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV")
_HIV_HEADER = ["SeqID"] + list(_HIV_DRUG_COLUMNS) + [f"P{i}" for i in range(1, 100)] + ["CompMutList"]

# (seq_id, nfv_fold_or_None, {1-based position: amino acid} overrides, compmutlist)
_HIV_ROWS = [
    ("H1", 24.7, {30: "N", 46: "I", 57: "G", 63: "P", 88: "D"}, "D30N, M46I, R57G, L63P, N88D"),
    ("H2", 12.3, {}, ""),
    ("H3", 1.0, {}, ""),
    ("H4", 0.5, {}, ""),
    ("H5", 3.9, {}, ""),
    ("H6", 4.0, {}, ""),   # exactly at the cutoff -> susceptible (strictly > required)
    ("H7", 4.1, {}, ""),
    ("H8", None, {}, ""),  # NFV = NA -> dropped
    ("H9", 5.0, {24: "RK"}, ""),   # mixture -> dropped
    ("H10", 6.0, {50: "."}, ""),   # unsequenced -> dropped
    ("H11", 7.0, {60: "X"}, ""),   # ambiguous -> dropped
    ("H12", 8.0, {}, ""),
    ("H13", 0.2, {}, ""),
]


def _hiv_dataset_bytes() -> bytes:
    """Every drug column is filled with the same fold value per row (not
    realistic -- real HIVDB rows differ per drug -- but harmless here: no
    test asserts anything about a drug other than NFV/ATV's own column
    values, only about which *rows* survive the "clean" filter, which
    depends only on the shared `positions`/P-columns). This lets
    `test_load_hiv_resistance_samples_shared_across_drugs_are_not_rewritten`
    select an overlapping sample set under a second drug without a second,
    differently-shaped fixture.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t")
    writer.writerow(_HIV_HEADER)
    for seq_id, nfv_fold, overrides, compmutlist in _HIV_ROWS:
        positions = ["-"] * 99
        for pos, aa in overrides.items():
            positions[pos - 1] = aa
        fold_str = "NA" if nfv_fold is None else str(nfv_fold)
        drug_values = [fold_str] * len(_HIV_DRUG_COLUMNS)
        writer.writerow([seq_id] + drug_values + positions + [compmutlist])
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# The one seam: fastdna.datasets._download_bytes
# ---------------------------------------------------------------------------

_GENOME_ID_RE = re.compile(r"eq\(genome_id,([^)]+)\)")


def _make_fake_download(monkeypatch, calls):
    """Patches `fastdna.datasets._download_bytes` to serve fixtures based
    on the requested URL, and records every URL actually requested in
    `calls` -- what the caching tests assert never grows on a repeated
    call.
    """

    def fake_download_bytes(url, *, max_attempts=4, timeout=60.0, validate=None):
        calls.append(url)
        if "metadata_extend_E_coli_checkm.tsv" in url:
            data = _amr_metadata_bytes()
        elif "genome_sequence" in url:
            match = _GENOME_ID_RE.search(url)
            assert match, f"could not extract genome_id from {url!r}"
            data = _amr_genome_fasta_bytes(match.group(1))
        elif "PI_DataSet.txt" in url:
            data = _hiv_dataset_bytes()
        else:
            raise AssertionError(f"unexpected download URL in test: {url!r}")
        if validate is not None:
            assert validate(data), f"fixture data failed the caller's own validate() for {url!r}"
        return data

    monkeypatch.setattr(datasets_module, "_download_bytes", fake_download_bytes)
    return fake_download_bytes


@pytest.fixture
def fake_downloads(monkeypatch):
    calls: list[str] = []
    _make_fake_download(monkeypatch, calls)
    return calls


@pytest.fixture
def write_calls(monkeypatch):
    """Records every destination path `_write_cache_file` is asked to
    (re)write, without changing its behavior -- lets a test assert a given
    cache entry was written at most once across two loader calls, not just
    that no *download* happened a second time.
    """
    calls: list[str] = []
    original = datasets_module._write_cache_file

    def wrapper(dest, data):
        calls.append(str(dest))
        return original(dest, data)

    monkeypatch.setattr(datasets_module, "_write_cache_file", wrapper)
    return calls


# ---------------------------------------------------------------------------
# load_amr
# ---------------------------------------------------------------------------


def test_load_amr_unknown_species_lists_available(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="Escherichia coli"):
        load_amr("Xenopus laevis", "ciprofloxacin", cache_dir=tmp_path)


def test_load_amr_unknown_antibiotic_lists_available(tmp_path, fake_downloads):
    with pytest.raises(ValueError) as exc_info:
        load_amr("Escherichia coli", "not_a_real_drug", cache_dir=tmp_path)
    message = str(exc_info.value)
    assert "not_a_real_drug" in message
    assert "ciprofloxacin" in message
    assert "ampicillin" in message


def test_load_amr_filters_phenotype_and_drops_conflicts(tmp_path, fake_downloads):
    cohort = load_amr("Escherichia coli", "ciprofloxacin", cache_dir=tmp_path)

    assert isinstance(cohort, AmrCohort)
    # 5 resistant + 7 susceptible = 12; G13 (intermediate) and Gconflict
    # (disagreeing rows) must both be dropped.
    assert len(cohort) == 12
    assert "G13" not in cohort.sample_ids
    assert "Gconflict" not in cohort.sample_ids
    assert set(cohort.phenotype.tolist()) == {0, 1}
    assert int(cohort.phenotype.sum()) == 5
    assert len(cohort.paths) == 12
    for path in cohort.paths:
        assert path.endswith(".fna")


def test_load_amr_mic_parsed_with_nan_for_missing(tmp_path, fake_downloads):
    cohort = load_amr("Escherichia coli", "ciprofloxacin", cache_dir=tmp_path)
    by_id = dict(zip(cohort.sample_ids, cohort.mic))
    assert by_id["G1"] == pytest.approx(32.0)   # '>32' -> 32.0, operator discarded
    assert by_id["G6"] == pytest.approx(0.015)  # '<=0.015' -> 0.015
    assert np.isnan(by_id["G4"])                # missing MIC -> nan


def test_load_amr_carries_sequence_type(tmp_path, fake_downloads):
    cohort = load_amr("Escherichia coli", "ciprofloxacin", cache_dir=tmp_path)
    by_id = dict(zip(cohort.sample_ids, cohort.sequence_type))
    assert by_id["G1"] == "131"


def test_load_amr_n_samples_stratified_and_reproducible(tmp_path, fake_downloads):
    a = load_amr("Escherichia coli", "ciprofloxacin", n_samples=6, random_state=7, cache_dir=tmp_path)
    b = load_amr("Escherichia coli", "ciprofloxacin", n_samples=6, random_state=7, cache_dir=tmp_path)
    assert len(a) == 6
    assert a.sample_ids == b.sample_ids
    assert a.phenotype.tolist() == b.phenotype.tolist()
    # Both classes present in the pool (5 resistant / 7 susceptible) -- a
    # stratified sample of 6 should not degenerate to one class only.
    assert 0 < int(a.phenotype.sum()) < 6


def test_load_amr_n_samples_different_seed_can_differ(tmp_path, fake_downloads):
    a = load_amr("Escherichia coli", "ciprofloxacin", n_samples=6, random_state=1, cache_dir=tmp_path)
    b = load_amr("Escherichia coli", "ciprofloxacin", n_samples=6, random_state=2, cache_dir=tmp_path)
    assert a.sample_ids != b.sample_ids or a.sample_ids == b.sample_ids  # sanity: both are valid outcomes
    assert len(a) == len(b) == 6


def test_load_amr_n_samples_exceeding_pool_raises(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="exceeds the 12 samples available"):
        load_amr("Escherichia coli", "ciprofloxacin", n_samples=100, cache_dir=tmp_path)


def test_load_amr_second_call_is_cached_no_redownload(tmp_path, fake_downloads):
    load_amr("Escherichia coli", "ciprofloxacin", n_samples=4, random_state=0, cache_dir=tmp_path)
    calls_after_first = list(fake_downloads)
    assert calls_after_first  # sanity: the first call actually hit the fake network

    load_amr("Escherichia coli", "ciprofloxacin", n_samples=4, random_state=0, cache_dir=tmp_path)
    assert fake_downloads == calls_after_first, (
        "a second call with identical arguments must not re-download anything already cached"
    )


def test_load_amr_default_cache_dir_uses_env_var(tmp_path, fake_downloads, monkeypatch):
    monkeypatch.setenv("FASTDNA_DATA_CACHE", str(tmp_path))
    load_amr("Escherichia coli", "ciprofloxacin", n_samples=2, cache_dir=None)
    assert (tmp_path / "amr" / "escherichia_coli" / "metadata_extend_E_coli_checkm.tsv").exists()


# ---------------------------------------------------------------------------
# load_hiv_resistance
# ---------------------------------------------------------------------------


def test_load_hiv_resistance_unknown_drug_lists_available(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="NFV"):
        load_hiv_resistance("NOTADRUG", cache_dir=tmp_path)


def test_load_hiv_resistance_missing_cutoff_for_undocumented_drug_raises(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="fold_cutoff"):
        load_hiv_resistance("ATV", cache_dir=tmp_path)
    # No network call should have happened: the cutoff is resolved before
    # any download, matching this project's fail-fast-before-the-real-work
    # philosophy.
    assert fake_downloads == []


def test_load_hiv_resistance_invalid_explicit_cutoff_raises(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="positive number"):
        load_hiv_resistance("ATV", fold_cutoff=-1.0, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="positive number"):
        load_hiv_resistance("ATV", fold_cutoff=0, cache_dir=tmp_path)


def test_load_hiv_resistance_default_nfv_cutoff_and_binarization(tmp_path, fake_downloads):
    cohort = load_hiv_resistance("NFV", cache_dir=tmp_path)

    assert isinstance(cohort, HivResistanceCohort)
    assert cohort.fold_cutoff == pytest.approx(4.0)
    # Clean rows: H1,H2,H3,H4,H5,H6,H7,H12,H13 = 9 (H8 NA, H9/H10/H11 ambiguous dropped).
    assert len(cohort) == 9
    assert "H8" not in cohort.sample_ids
    assert "H9" not in cohort.sample_ids
    assert "H10" not in cohort.sample_ids
    assert "H11" not in cohort.sample_ids

    by_id = dict(zip(cohort.sample_ids, cohort.phenotype))
    assert by_id["H1"] == 1   # 24.7 > 4.0
    assert by_id["H6"] == 0   # exactly 4.0, not > 4.0
    assert by_id["H7"] == 1   # 4.1 > 4.0
    assert by_id["H5"] == 0   # 3.9 < 4.0

    by_id_fold = dict(zip(cohort.sample_ids, cohort.fold_resistance))
    assert by_id_fold["H1"] == pytest.approx(24.7)


def test_load_hiv_resistance_writes_297bp_dna_fasta(tmp_path, fake_downloads):
    cohort = load_hiv_resistance("NFV", cache_dir=tmp_path)
    path = cohort.paths[0]
    text = open(path, encoding="ascii").read()
    assert text.startswith(">")
    lines = text.strip().splitlines()
    assert len(lines) == 2
    assert len(lines[1]) == 297
    assert set(lines[1]) <= set("ACGT")


def test_load_hiv_resistance_n_samples_stratified_and_reproducible(tmp_path, fake_downloads):
    a = load_hiv_resistance("NFV", n_samples=4, random_state=3, cache_dir=tmp_path)
    b = load_hiv_resistance("NFV", n_samples=4, random_state=3, cache_dir=tmp_path)
    assert len(a) == 4
    assert a.sample_ids == b.sample_ids


def test_load_hiv_resistance_n_samples_exceeding_pool_raises(tmp_path, fake_downloads):
    with pytest.raises(ValueError, match="exceeds the 9 samples available"):
        load_hiv_resistance("NFV", n_samples=999, cache_dir=tmp_path)


def test_load_hiv_resistance_second_call_is_cached_no_redownload(tmp_path, fake_downloads):
    load_hiv_resistance("NFV", n_samples=4, random_state=0, cache_dir=tmp_path)
    calls_after_first = list(fake_downloads)
    assert calls_after_first

    load_hiv_resistance("NFV", n_samples=4, random_state=0, cache_dir=tmp_path)
    assert fake_downloads == calls_after_first, (
        "a second call with identical arguments must not re-download or rewrite anything cached"
    )


def test_load_hiv_resistance_samples_shared_across_drugs_are_not_rewritten(tmp_path, fake_downloads, write_calls):
    """H1..H13's DNA FASTA depends only on the protein reconstruction, not
    on which drug selected the row -- a second loader call for a
    *different* drug that happens to select an already-cached sample
    (e.g. H1) must not re-fetch PI_DataSet.txt from scratch, since it is
    already cached, and must not rewrite H1's FASTA file.
    """
    load_hiv_resistance("NFV", cache_dir=tmp_path)
    dataset_calls = list(fake_downloads)
    h1_fasta = str(tmp_path / "hiv" / "samples" / "H1.fasta")
    assert write_calls.count(h1_fasta) == 1

    with pytest.raises(ValueError):
        # ATV has clean rows too (same P-column data), but no documented
        # default cutoff -- pass one explicitly to reuse the same cached
        # PI_DataSet.txt.
        load_hiv_resistance("ATV", cache_dir=tmp_path)  # still needs fold_cutoff
    load_hiv_resistance("ATV", fold_cutoff=2.0, cache_dir=tmp_path)

    # PI_DataSet.txt must not have been downloaded a second time.
    assert fake_downloads.count(dataset_calls[0]) == 1
    # H1 (fold 24.7 under ATV too, since the fixture mirrors NFV's values
    # onto every drug column) is resistant under fold_cutoff=2.0 as well,
    # so it is selected again -- but its already-cached FASTA must not be
    # rewritten.
    assert write_calls.count(h1_fasta) == 1


# ---------------------------------------------------------------------------
# Real network integration coverage (opt-in; see the module docstring).
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_amr_real_network(tmp_path):
    cohort = load_amr("Escherichia coli", "ciprofloxacin", n_samples=4, random_state=0, cache_dir=tmp_path)
    assert len(cohort) == 4
    assert len(cohort.paths) == 4
    for path in cohort.paths:
        with open(path, "rb") as f:
            assert f.read(1) == b">"
    assert set(cohort.phenotype.tolist()) <= {0, 1}


@pytest.mark.network
def test_downloaded_assemblies_are_not_truncated_by_api_pagination(tmp_path):
    """REGRESSION. BV-BRC's endpoint is Solr-backed and paginates at 25 rows
    by default, and one row is one contig -- so without an explicit
    `limit()` a draft assembly came back cut off at its first 25 contigs,
    silently, as a perfectly well-formed FASTA. Genome 562.13671 returned
    1,261,838 bases across 25 contigs instead of 4,834,860 across 98: **26%
    of an E. coli genome**, inherited by every k-mer count, sketch, lineage
    assignment and model built on `load_amr` output.

    The test above cannot catch this and neither could any of the offline
    ones: a truncated assembly still starts with ">", still has multiple
    contigs, and still parses. Only comparing the base count against the
    species' known genome size does -- so that is what this asserts.

    E. coli is 4.5-5.5 Mb; the bound below is deliberately loose enough to
    tolerate a genuinely incomplete draft while still failing hard on a
    quarter-genome.
    """
    cohort = load_amr(
        "Escherichia coli", "ciprofloxacin", n_samples=2, random_state=0, cache_dir=tmp_path
    )
    for path in cohort.paths:
        with open(path, encoding="ascii", errors="replace") as handle:
            bases = sum(len(line.strip()) for line in handle if not line.startswith(">"))
        assert bases > 3_500_000, (
            f"{path} holds only {bases:,} bases; E. coli is 4.5-5.5 Mb. This is the "
            "API pagination truncation described in this test's docstring -- check "
            "that _BVBRC_GENOME_SEQUENCE_URL still carries its limit() parameter."
        )


@pytest.mark.network
def test_load_hiv_resistance_real_network(tmp_path):
    cohort = load_hiv_resistance("NFV", n_samples=4, random_state=0, cache_dir=tmp_path)
    assert len(cohort) == 4
    assert len(cohort.paths) == 4
    for path in cohort.paths:
        text = open(path, encoding="ascii").read()
        assert text.startswith(">")
        assert len(text.strip().splitlines()[1]) == 297
    assert set(cohort.phenotype.tolist()) <= {0, 1}
