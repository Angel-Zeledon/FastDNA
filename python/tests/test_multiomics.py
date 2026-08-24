"""Tests for fastdna.multiomics -- joining k-mer feature tables with other
omics/clinical data by sample ID (kmer_feature_table, join_omics_layers,
normalize_sample_ids).
"""
from __future__ import annotations

import gzip
import pathlib

import pandas as pd
import pyarrow as pa
import pytest

import fastdna
from fastdna.multiomics import join_omics_layers, kmer_feature_table, normalize_sample_ids


def _fastq_text(reads: list[str]) -> str:
    return "".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads))


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(_fastq_text(reads))
    return p


def write_fastq_gz(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    """A genuinely gzip-compressed FASTQ, following test_api.py's own
    convention. `write_fastq` writes plain text, so using it for a
    `.gz`-named file produces something the reader correctly rejects with
    "invalid gzip header" -- the name alone does not make a file
    compressed.
    """
    p = tmp_path / name
    with gzip.open(p, "wt") as f:
        f.write(_fastq_text(reads))
    return p


# --------------------------------------------------------------------------
# kmer_feature_table
# --------------------------------------------------------------------------


def test_kmer_feature_table_shape_and_columns(tmp_path):
    a = write_fastq(tmp_path, "sample_a.fastq", ["ACGTACGTACGT"] * 5)
    b = write_fastq(tmp_path, "sample_b.fastq", ["TTTTGGGGCCCC"] * 5)

    table = kmer_feature_table([str(a), str(b)], k=5)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert "sample_id" in table.column_names
    # More than just sample_id: real k-mer columns are present.
    assert table.num_columns > 1
    assert set(table.column("sample_id").to_pylist()) == {"sample_a", "sample_b"}


def test_kmer_feature_table_sample_id_from_filename_strips_fastq_suffix(tmp_path):
    a = write_fastq_gz(tmp_path, "patient_007.fastq.gz", ["ACGTACGTACGT"] * 3)
    table = kmer_feature_table([str(a)], k=5)
    assert table.column("sample_id").to_pylist() == ["patient_007"]


def test_kmer_feature_table_row_matches_direct_count(tmp_path):
    # A sample whose own k-mer content we can check by hand: k=4 over a
    # short repeated sequence, plus a second, differently-composed sample
    # so the union-of-vocabulary column set actually has more than one
    # sample's k-mers in it.
    #
    # The two samples must differ in their *canonical* k-mers, which is a
    # stronger requirement than differing in their literal sequence:
    # FastDNA counts each k-mer and its reverse complement as one canonical
    # k-mer, so e.g. "AAAACCCC" and "GGGGTTTT" -- exact reverse complements
    # -- produce byte-for-byte identical count tables and would make this
    # test vacuous. "AAAAGGGG" shares only AAAA and CCCC (the canonical
    # form of GGGG) with "AAAACCCC", leaving AAAG/AAGG/AGGG genuinely
    # unique to sample b.
    reads_a = ["AAAACCCC"] * 4
    reads_b = ["AAAAGGGG"] * 4
    a = write_fastq(tmp_path, "s1.fastq", reads_a)
    b = write_fastq(tmp_path, "s2.fastq", reads_b)

    table = kmer_feature_table([str(a), str(b)], k=4)

    # Cross-check against calling fastdna.count() directly on sample a.
    direct = fastdna.count(str(a), k=4)
    direct_counts = dict(
        zip(direct.table.column("kmer_sequence").to_pylist(), direct.table.column("frequency").to_pylist())
    )

    row = table.to_pylist()
    row_a = next(r for r in row if r["sample_id"] == "s1")
    row_b = next(r for r in row if r["sample_id"] == "s2")

    # Hand-computed: each of the 4 identical reads "AAAACCCC" yields the 5
    # 4-mers AAAA, AAAC, AACC, ACCC, CCCC exactly once, so every one of
    # them has frequency 4 across the sample. (All five are already their
    # own canonical form here, being lexicographically <= their reverse
    # complements.) Asserting the literal numbers -- not just agreement
    # with count() -- is what makes this a real check of the wide table's
    # contents rather than a tautology.
    assert row_a["AAAA"] == 4
    assert row_a["AAAC"] == 4
    assert row_a["AACC"] == 4
    assert row_a["ACCC"] == 4
    assert row_a["CCCC"] == 4

    # And the same row must agree with a direct per-file count() call.
    for kmer, freq in direct_counts.items():
        assert row_a[kmer] == freq

    # A k-mer present only in sample b's reads must be 0 in sample a's row
    # (missing k-mers become 0, not absent/NaN).
    direct_b = fastdna.count(str(b), k=4)
    b_counts = dict(
        zip(direct_b.table.column("kmer_sequence").to_pylist(), direct_b.table.column("frequency").to_pylist())
    )
    only_in_b = [kmer for kmer in b_counts if kmer not in direct_counts]
    assert only_in_b, "expected at least one k-mer unique to sample b for this to be a real check"
    for kmer in only_in_b:
        assert row_a[kmer] == 0
        assert row_b[kmer] == b_counts[kmer]


def test_kmer_feature_table_top_features_restricts_columns(tmp_path):
    a = write_fastq(tmp_path, "s1.fastq", ["ACGTACGTACGTACGT"] * 6)
    b = write_fastq(tmp_path, "s2.fastq", ["TTTTGGGGCCCCAAAA"] * 6)

    full = kmer_feature_table([str(a), str(b)], k=4)
    limited = kmer_feature_table([str(a), str(b)], k=4, top_features=2)

    full_kmer_columns = [c for c in full.column_names if c != "sample_id"]
    limited_kmer_columns = [c for c in limited.column_names if c != "sample_id"]

    assert len(limited_kmer_columns) <= 2
    assert len(limited_kmer_columns) < len(full_kmer_columns)
    assert set(limited_kmer_columns).issubset(set(full_kmer_columns))


def test_kmer_feature_table_dict_input_uses_explicit_ids(tmp_path):
    a = write_fastq(tmp_path, "raw_file_1.fastq", ["ACGTACGTACGT"] * 3)
    b = write_fastq(tmp_path, "raw_file_2.fastq", ["TTTTGGGGCCCC"] * 3)

    table = kmer_feature_table({"patientA": str(a), "patientB": str(b)}, k=5)

    assert set(table.column("sample_id").to_pylist()) == {"patientA", "patientB"}


def test_kmer_feature_table_rejects_unknown_id_from(tmp_path):
    a = write_fastq(tmp_path, "s1.fastq", ["ACGTACGTACGT"] * 3)
    with pytest.raises(ValueError):
        kmer_feature_table([str(a)], k=5, id_from="basename_upper")


# --------------------------------------------------------------------------
# join_omics_layers -- inner join drops mismatched samples, with a report
# --------------------------------------------------------------------------


def test_inner_join_drops_mismatched_samples_and_reports_them():
    layer_a = pd.DataFrame({"sample_id": ["1", "2", "3"], "feat_a": [10, 20, 30]})
    layer_b = pd.DataFrame({"sample_id": ["2", "3", "4"], "feat_b": [200, 300, 400]})

    combined, report = join_omics_layers({"a": layer_a, "b": layer_b}, how="inner")

    assert set(combined["sample_id"]) == {"2", "3"}
    assert combined.shape[0] == 2
    assert report.row_count == 2
    assert sorted(report.kept_sample_ids) == ["2", "3"]

    # Sample "1" is present only in layer a -> missing from layer b.
    assert report.dropped_sample_ids["1"] == ["b"]
    # Sample "4" is present only in layer b -> missing from layer a.
    assert report.dropped_sample_ids["4"] == ["a"]
    assert set(report.dropped_sample_ids) == {"1", "4"}

    # The report also records exactly what each layer contained going in.
    assert report.layer_sample_ids["a"] == {"1", "2", "3"}
    assert report.layer_sample_ids["b"] == {"2", "3", "4"}

    # And the join actually carried over both layers' feature columns.
    row2 = combined[combined["sample_id"] == "2"].iloc[0]
    assert row2["feat_a"] == 20
    assert row2["feat_b"] == 200


def test_inner_join_of_three_layers_keeps_only_universal_samples():
    a = pd.DataFrame({"sample_id": ["1", "2", "3"], "x": [1, 2, 3]})
    b = pd.DataFrame({"sample_id": ["2", "3", "4"], "y": [1, 2, 3]})
    c = pd.DataFrame({"sample_id": ["1", "2", "3", "4"], "z": [1, 2, 3, 4]})

    combined, report = join_omics_layers({"a": a, "b": b, "c": c}, how="inner")

    assert set(combined["sample_id"]) == {"2", "3"}
    assert report.dropped_sample_ids["1"] == ["b"]
    assert report.dropped_sample_ids["4"] == ["a"]


def test_join_disambiguates_colliding_feature_columns_by_layer_name():
    # Three layers all carrying a "value" column: every one of them must
    # end up named after its own layer, and no layer's values may be lost
    # or overwritten by another's. (Leaning on pandas' own `suffixes=`
    # here would leave the third layer's column as a bare, ambiguous
    # "value".)
    a = pd.DataFrame({"sample_id": ["1", "2"], "value": [1, 2]})
    b = pd.DataFrame({"sample_id": ["1", "2"], "value": [10, 20]})
    c = pd.DataFrame({"sample_id": ["1", "2"], "value": [100, 200]})

    combined, _ = join_omics_layers({"a": a, "b": b, "c": c}, how="inner")

    assert "value" not in combined.columns
    assert {"value_a", "value_b", "value_c"}.issubset(set(combined.columns))

    row1 = combined[combined["sample_id"] == "1"].iloc[0]
    assert row1["value_a"] == 1
    assert row1["value_b"] == 10
    assert row1["value_c"] == 100


def test_join_leaves_non_colliding_column_names_untouched():
    a = pd.DataFrame({"sample_id": ["1", "2"], "age": [40, 50]})
    b = pd.DataFrame({"sample_id": ["1", "2"], "expression": [1.5, 2.5]})

    combined, _ = join_omics_layers({"a": a, "b": b}, how="inner")

    # Unique column names must not be renamed just because a join happened.
    assert "age" in combined.columns
    assert "expression" in combined.columns


# --------------------------------------------------------------------------
# join_omics_layers -- outer join keeps everything and fills sanely
# --------------------------------------------------------------------------


def test_outer_join_keeps_all_samples_and_fills_missing_values():
    layer_a = pd.DataFrame({"sample_id": ["1", "2", "3"], "count_feat": [10, 20, 30]})
    layer_b = pd.DataFrame({"sample_id": ["2", "3", "4"], "label": ["x", "y", "z"]})

    combined, report = join_omics_layers({"a": layer_a, "b": layer_b}, how="outer")

    assert set(combined["sample_id"]) == {"1", "2", "3", "4"}
    assert combined.shape[0] == 4
    assert report.row_count == 4
    # Nothing dropped under a full outer join.
    assert report.dropped_sample_ids == {}

    row1 = combined[combined["sample_id"] == "1"].iloc[0]
    row4 = combined[combined["sample_id"] == "4"].iloc[0]

    # Sample "1" has no row in layer b -> its non-numeric "label" column
    # is filled with the sentinel string "missing".
    assert row1["label"] == "missing"
    # Sample "4" has no row in layer a -> its numeric "count_feat" column
    # is filled with 0, not left as NaN.
    assert row4["count_feat"] == 0
    assert not pd.isna(row4["count_feat"])


# --------------------------------------------------------------------------
# normalize_sample_ids
# --------------------------------------------------------------------------


def test_normalize_sample_ids_matches_cosmetically_different_ids():
    ids = ["Sample-1", "sample_1", " SAMPLE 1 ", "sample--1", "sample.1"]
    normalized = normalize_sample_ids(ids)
    assert len(set(normalized)) == 1
    assert normalized[0] == "sample_1"


def test_normalize_sample_ids_does_not_merge_numerically_different_ids():
    # Zero-padding differences are explicitly out of scope -- these two
    # must NOT normalize to the same string.
    normalized = normalize_sample_ids(["sample_1", "sample_001"])
    assert normalized[0] != normalized[1]


def test_normalize_sample_ids_blank_ids_collapse_to_empty_string():
    # Documented edge case: an ID made only of whitespace/punctuation
    # normalizes to "". Pinned here so the behavior and the docstring
    # cannot drift apart silently.
    assert normalize_sample_ids(["   ", "---", "..."]) == ["", "", ""]


def test_normalize_sample_ids_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        normalize_sample_ids(["a"], strategy="fuzzy_match")


# --------------------------------------------------------------------------
# End-to-end: kmer_feature_table + normalize_sample_ids + join_omics_layers
# --------------------------------------------------------------------------


def test_end_to_end_kmer_table_joined_with_clinical_data(tmp_path):
    a = write_fastq(tmp_path, "Sample-1.fastq", ["ACGTACGTACGT"] * 5)
    b = write_fastq(tmp_path, "Sample-2.fastq", ["TTTTGGGGCCCC"] * 5)
    c = write_fastq(tmp_path, "Sample-3.fastq", ["AAAACCCCGGGG"] * 5)

    kmers = kmer_feature_table([str(a), str(b), str(c)], k=5)
    # kmer_feature_table's filename-derived ids are "Sample-1" etc (the
    # ".fastq" suffix is stripped, nothing else is touched).
    assert set(kmers.column("sample_id").to_pylist()) == {"Sample-1", "Sample-2", "Sample-3"}

    # A "clinical metadata" table using a slightly different sample_id
    # convention -- lowercase, underscore instead of hyphen -- for the
    # same three samples, plus an extra sample not present in the k-mer
    # layer at all.
    clinical = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2", "sample_3", "sample_4"],
            "age": [45, 60, 33, 51],
            "condition": ["case", "control", "case", "control"],
        }
    )

    # Normalize both sides' IDs onto the same convention before joining.
    kmers_df = kmers.to_pandas()
    kmers_df["sample_id"] = normalize_sample_ids(kmers_df["sample_id"])
    clinical["sample_id"] = normalize_sample_ids(clinical["sample_id"])

    combined, report = join_omics_layers({"kmers": kmers_df, "clinical": clinical}, how="inner")

    # sample_4 exists only in the clinical layer -> dropped by the inner
    # join and reported as missing from "kmers".
    assert set(combined["sample_id"]) == {"sample_1", "sample_2", "sample_3"}
    assert report.dropped_sample_ids == {"sample_4": ["kmers"]}

    # Both layers' columns made it into the combined table.
    assert "age" in combined.columns
    assert "condition" in combined.columns
    kmer_columns = [c for c in kmers_df.columns if c != "sample_id"]
    assert any(c in combined.columns for c in kmer_columns)
