"""fastdna.count_cohort() / CohortCounts: count once, reuse everywhere.

The type exists so a cohort is counted once and then sliced, rather than
re-read by every consumer. What has to hold is that a slice matches what
counting that sample alone would have produced, that the artifact is
immutable and cheap to share, and that a saved cohort reloads to exactly
the same rows.
"""
from __future__ import annotations

import pytest

import fastdna
from fastdna import _core

# Guarded, not imported at module scope, on purpose: a bare `import numpy`
# makes pytest raise during *collection* rather than skipping in an
# environment without it -- see
# `test_optional_dependencies.py::test_the_test_suite_itself_collects_in_the_environment_ci_builds`,
# which exists specifically to catch a module reintroducing this.
np = pytest.importorskip("numpy")


def _cohort(tmp_path, n=12):
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    paths = []
    for i in range(n):
        seq = "".join(rng.choice(bases, size=600))
        reads = [seq[j : j + 100] for j in range(0, 500, 50)]
        p = tmp_path / f"S{i:02d}.fastq"
        p.write_text("".join(f"@r{j}\n{r}\n+\n{'I' * len(r)}\n" for j, r in enumerate(reads)))
        paths.append(str(p))
    return paths


def test_count_cohort_derives_sample_ids_from_filenames(tmp_path):
    paths = _cohort(tmp_path, n=3)
    counts = fastdna.count_cohort(paths, k=11)

    assert len(counts) == 3
    assert set(counts.sample_ids) == {"S00", "S01", "S02"}
    assert sum(counts.row_counts) == len(counts.kmers)


def _write_fasta(tmp_path, name, seq):
    p = tmp_path / name
    p.write_text(f">contig1\n{seq}\n")
    return str(p)


def test_count_cohort_derives_sample_ids_from_fna_filenames(tmp_path):
    """Regression test: `.fna` is the standard extension genome archives
    (BV-BRC, NCBI) use for assembled nucleotide FASTA, and
    `fastdna.count()`'s Rust core content-sniffs FASTA regardless of
    extension -- but `_sample_id_from_path` used to only strip
    `.fastq`/`.fq`/`.fasta`/`.fa`/`.gz`, leaving a stray `.fna` in every
    sample_id derived from a real genome-assembly cohort. See
    `cohort_counts.py::_sample_id_from_path`'s docstring for the concrete
    failure this caused in `fastdna.audit()`-style usage: a `paths`/`groups`
    positional mismatch against sample_ids computed elsewhere.
    """
    rng = np.random.default_rng(1)
    bases = np.array(list("ACGT"))
    seq = "".join(rng.choice(bases, size=300))
    path = _write_fasta(tmp_path, "562.48346.fna", seq)

    counts = fastdna.count_cohort([path], k=11)
    assert counts.sample_ids == ("562.48346",)


def test_count_cohort_finds_fna_files_in_a_directory(tmp_path):
    """`.fna` (and gzipped `.fna.gz`) must be discovered by the
    directory-scan path, not just accepted when passed explicitly -- the
    directory filter used to omit both."""
    rng = np.random.default_rng(2)
    bases = np.array(list("ACGT"))
    for i in range(3):
        seq = "".join(rng.choice(bases, size=300))
        _write_fasta(tmp_path, f"genome_{i}.fna", seq)

    counts = fastdna.count_cohort(tmp_path, k=11)
    assert set(counts.sample_ids) == {"genome_0", "genome_1", "genome_2"}


def test_count_cohort_accepts_an_explicit_id_mapping(tmp_path):
    paths = _cohort(tmp_path, n=2)
    mapping = {"patientA": paths[0], "patientB": paths[1]}
    counts = fastdna.count_cohort(mapping, k=11)
    assert counts.sample_ids == ("patientA", "patientB")


def test_count_cohort_rejects_a_filename_collision(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for d in (a_dir, b_dir):
        (d / "S1.fastq").write_text("@r\nACGTACGTACGT\n+\nIIIIIIIIIIII\n")
    with pytest.raises(ValueError, match="S1"):
        fastdna.count_cohort([str(a_dir / "S1.fastq"), str(b_dir / "S1.fastq")], k=6)


def test_count_cohort_rejects_an_empty_input():
    with pytest.raises(ValueError):
        fastdna.count_cohort([], k=11)


def test_subset_preserves_per_sample_rows(tmp_path):
    paths = _cohort(tmp_path, n=6)
    counts = fastdna.count_cohort(paths, k=11)

    picked = [counts.sample_ids[0], counts.sample_ids[3]]
    sub = counts.subset(picked)

    assert sub.sample_ids == tuple(picked)
    assert len(sub.kmers) == sum(sub.row_counts)
    assert sub.row_counts == (counts.row_counts[0], counts.row_counts[3])
    # And the actual k-mer/frequency values must be untouched by the slice.
    assert sub.kmers.to_pylist() == counts.subset([picked[0]]).kmers.to_pylist() + counts.subset(
        [picked[1]]
    ).kmers.to_pylist()


def test_subset_names_a_missing_sample_instead_of_returning_less(tmp_path):
    """A fold silently missing samples would produce a plausible-looking
    score computed over the wrong cohort. It has to fail, not shrink."""
    paths = _cohort(tmp_path, n=4)
    counts = fastdna.count_cohort(paths, k=11)
    with pytest.raises(KeyError, match="no_existe"):
        counts.subset([counts.sample_ids[0], "no_existe"])


def test_subset_of_empty_selection_is_empty(tmp_path):
    paths = _cohort(tmp_path, n=3)
    counts = fastdna.count_cohort(paths, k=11)
    empty = counts.subset([])
    assert len(empty) == 0
    assert len(empty.kmers) == 0


def test_counts_artifact_is_immutable(tmp_path):
    """A fold that could mutate the shared artifact would be a silent way
    for one fold's view to leak into another's."""
    import dataclasses

    paths = _cohort(tmp_path, n=2)
    counts = fastdna.count_cohort(paths, k=11)
    with pytest.raises(dataclasses.FrozenInstanceError):
        counts.k = 21


def test_deepcopy_of_the_artifact_returns_the_same_object(tmp_path):
    """Deep-copying the artifact must be a no-op identity: it is frozen and
    meant to be shared (see the module docstring), and a real cohort's
    multi-hundred-million-row table copied once per consumer can exhaust
    memory outright. Any caller that hands the artifact to a framework
    which deep-copies its arguments hits exactly this path.
    """
    import copy

    paths = _cohort(tmp_path, n=2)
    counts = fastdna.count_cohort(paths, k=11)
    assert copy.deepcopy(counts) is counts


def test_save_load_round_trip_is_exact(tmp_path):
    paths = _cohort(tmp_path, n=5)
    counts = fastdna.count_cohort(paths, k=11)

    target = tmp_path / "cohort.parquet"
    counts.save(target)
    restored = fastdna.CohortCounts.load(target)

    assert restored.sample_ids == counts.sample_ids
    assert restored.row_counts == counts.row_counts
    assert restored.k == counts.k
    assert restored.min_count == counts.min_count
    assert restored.kmers.equals(counts.kmers)
    assert restored.frequencies.equals(counts.frequencies)
    # Types, not just values: uint32 frequencies silently widened to int64
    # would round-trip equal here but change what the Rust side accepts.
    assert restored.kmers.type == counts.kmers.type
    assert restored.frequencies.type == counts.frequencies.type


def test_a_loaded_cohort_subsets_to_the_same_rows(tmp_path):
    """Round-tripping the arrays is not enough: the offsets have to survive
    too, because `subset()` is what every fold of a cross-validation calls."""
    paths = _cohort(tmp_path, n=6)
    counts = fastdna.count_cohort(paths, k=11)
    target = tmp_path / "cohort.parquet"
    counts.save(target)

    wanted = ["S01", "S04"]
    original = counts.subset(wanted)
    restored = fastdna.CohortCounts.load(target).subset(wanted)

    assert restored.sample_ids == original.sample_ids
    assert restored.row_counts == original.row_counts
    assert restored.kmers.equals(original.kmers)
    assert restored.frequencies.equals(original.frequencies)


def test_load_rejects_a_plain_kmer_table(tmp_path):
    """`fastdna count -o x.parquet` writes the same two columns but none of
    the cohort structure. Loading it must say so, not invent one sample."""
    import pyarrow.parquet as pq

    paths = _cohort(tmp_path, n=2)
    plain = tmp_path / "plain.parquet"
    pq.write_table(fastdna.count(paths[0], k=11).table, plain)

    with pytest.raises(_core.InvalidConfigError) as excinfo:
        fastdna.CohortCounts.load(plain)
    assert "footer metadata" in str(excinfo.value)


def test_load_rejects_row_counts_that_do_not_sum_to_the_rows(tmp_path):
    """The silent-corruption case: a truncated or hand-edited cache whose
    offsets no longer match the data would mis-assign k-mers to samples and
    return a confident, wrong answer."""
    import pyarrow.parquet as pq

    paths = _cohort(tmp_path, n=3)
    counts = fastdna.count_cohort(paths, k=11)
    target = tmp_path / "cohort.parquet"
    counts.save(target)

    table = pq.read_table(target)
    metadata = dict(table.schema.metadata)
    bad = list(counts.row_counts)
    bad[0] += 7
    import json

    metadata[b"fastdna.row_counts"] = json.dumps(bad).encode()
    pq.write_table(table.replace_schema_metadata(metadata), target)

    with pytest.raises(_core.InvalidConfigError) as excinfo:
        fastdna.CohortCounts.load(target)
    assert "row counts" in str(excinfo.value)


def test_load_rejects_more_sample_ids_than_row_counts(tmp_path):
    import json

    import pyarrow.parquet as pq

    paths = _cohort(tmp_path, n=3)
    counts = fastdna.count_cohort(paths, k=11)
    target = tmp_path / "cohort.parquet"
    counts.save(target)

    table = pq.read_table(target)
    metadata = dict(table.schema.metadata)
    metadata[b"fastdna.sample_ids"] = json.dumps(
        list(counts.sample_ids) + ["S99"]
    ).encode()
    pq.write_table(table.replace_schema_metadata(metadata), target)

    with pytest.raises(_core.InvalidConfigError) as excinfo:
        fastdna.CohortCounts.load(target)
    assert "sample ids" in str(excinfo.value)


def test_an_interrupted_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """A cache is written by long jobs that get interrupted. If a failed
    save could truncate the existing file, the next run would load a partial
    cohort -- which is exactly the failure the whole file guards against."""
    paths = _cohort(tmp_path, n=3)
    counts = fastdna.count_cohort(paths, k=11)
    target = tmp_path / "cohort.parquet"
    counts.save(target)
    good = target.read_bytes()

    import pyarrow.parquet as pq

    real_write = pq.write_table

    def failing_write(table, where, *args, **kwargs):
        real_write(table, where, *args, **kwargs)  # write it, then die
        raise KeyboardInterrupt("laptop lid closed")

    monkeypatch.setattr(pq, "write_table", failing_write)
    with pytest.raises(KeyboardInterrupt):
        counts.save(target)
    monkeypatch.undo()

    assert target.read_bytes() == good
    assert not (tmp_path / "cohort.parquet.partial").exists()
    assert fastdna.CohortCounts.load(target).sample_ids == counts.sample_ids
