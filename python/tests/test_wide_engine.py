"""The `k > 32` engine, as reached from Python.

`fastdna.count(engine=...)` routes to one of two engines: the u64 one that
has always been here (`k <= 32`) and the u128 one (`33 <= k <= 64`). What
these tests pin is the part that is checkable without trusting either
engine's own arithmetic: **the two must agree, k-mer for k-mer, wherever
both are defined**. `tests/wide_engine.rs` makes the same check inside the
core; this makes it through the binding, which is where the routing,
the Arrow column type and the decoding actually live.
"""

import pathlib
import random
import shutil

import pyarrow as pa
import pytest

import fastdna
from fastdna._core import InvalidKError


def _write_reads(path, *, n_reads=40, read_len=120, genome_len=600, seed=11):
    """A small FASTQ with real k-mer redundancy: reads are overlapping
    windows of one random `genome_len`-base sequence, so k-mers repeat at
    depths a spectrum can be checked against instead of every count being
    1.
    """
    rng = random.Random(seed)
    genome = "".join(rng.choice("ACGT") for _ in range(genome_len))
    with open(path, "w") as handle:
        for i in range(n_reads):
            start = rng.randrange(0, genome_len - read_len)
            read = genome[start : start + read_len]
            handle.write(f"@read{i}\n{read}\n+\n{'I' * len(read)}\n")
    return genome


@pytest.fixture
def reads(tmp_path):
    path = tmp_path / "sample.fastq"
    _write_reads(path)
    return path


def _sequence_to_count(counts):
    """`{decoded k-mer: frequency}` for a `KmerCounts`, whichever engine
    produced it -- the one representation the two engines share, and so
    the only basis on which they can be compared directly.
    """
    table = counts.with_sequence().table
    return dict(
        zip(
            table.column("kmer_sequence").to_pylist(),
            table.column("frequency").to_pylist(),
        )
    )


def test_the_two_engines_agree_kmer_for_kmer_at_k32(reads):
    """The load-bearing test. `engine="wide"` at `k <= 32` exists so the
    u128 engine can be checked against the u64 one that KMC3 already
    validated; if these ever disagree, one of them is wrong.
    """
    narrow = fastdna.count(reads, k=32, engine="narrow")
    wide = fastdna.count(reads, k=32, engine="wide")

    assert narrow.engine == "narrow"
    assert wide.engine == "wide"
    assert narrow.total_kmers == wide.total_kmers
    assert narrow.distinct_kmers == wide.distinct_kmers
    assert _sequence_to_count(narrow) == _sequence_to_count(wide)
    assert narrow.spectrum() == wide.spectrum()


def test_auto_picks_the_engine_from_k(reads):
    """The default. No caller who never names an engine sees a change:
    `k <= 32` still counts narrow, and `k > 32` -- which used to raise --
    now counts wide.
    """
    assert fastdna.count(reads, k=31).engine == "narrow"
    assert fastdna.count(reads, k=32).engine == "narrow"
    assert fastdna.count(reads, k=33).engine == "wide"
    assert fastdna.count(reads, k=64).engine == "wide"


def test_a_wide_table_keys_on_16_byte_kmer_bits(reads):
    """Arrow has no 128-bit integer, so a wide table's key column is 16
    big-endian bytes rather than a `uint64` -- big-endian precisely so the
    column sorts bytewise the way the packed k-mer sorts numerically.
    """
    table = fastdna.count(reads, k=41).table
    assert table.column_names == ["kmer_bits", "frequency"]
    assert table.schema.field("kmer_bits").type == pa.binary(16)

    keys = table.column("kmer_bits").to_pylist()
    assert keys == sorted(keys), "a count table must be sorted by its key column"


def test_wide_counts_decode_to_k_bases(reads):
    """`with_sequence()` on a wide table decodes `kmer_bits`, and the
    Python decoder has to agree with the Rust one it mirrors -- so the
    same count is built both ways and compared.
    """
    k = 45
    decoded_in_python = fastdna.count(reads, k=k).with_sequence()
    decoded_in_rust = fastdna.count(reads, k=k, with_sequence=True)

    sequences = decoded_in_python.table.column("kmer_sequence").to_pylist()
    assert sequences, "the fixture should produce k-mers at k=45"
    assert {len(s) for s in sequences} == {k}
    assert set("".join(sequences)) <= set("ACGT")
    assert sequences == decoded_in_rust.table.column("kmer_sequence").to_pylist()


def test_decoding_survives_a_sliced_view(reads):
    """`top()`/`filter()` hand back an Arrow slice, whose values buffer
    still holds every row -- a decoder that ignored the array's offset
    would silently return the *first* n k-mers' sequences for the top n
    rows. Guards exactly that.
    """
    counts = fastdna.count(reads, k=41).sort_by("frequency", descending=True)
    top = counts.top(3)

    from_slice = top.with_sequence().table.column("kmer_sequence").to_pylist()
    from_whole = counts.with_sequence().table.column("kmer_sequence").to_pylist()[:3]
    assert from_slice == from_whole
    assert len(from_slice) == 3


def test_min_count_filters_a_wide_count(reads):
    """`min_count` is applied in the core, before the table is built, for
    both engines. Checked against the spectrum rather than a fixed number
    so the fixture can change without rewriting the expectation.
    """
    unfiltered = fastdna.count(reads, k=41)
    spectrum = unfiltered.spectrum()
    singletons = spectrum.get(1, 0)
    assert singletons > 0, "the fixture should contain some depth-1 k-mers"

    filtered = fastdna.count(reads, k=41, min_count=2)
    assert filtered.distinct_kmers == unfiltered.distinct_kmers - singletons
    assert 1 not in filtered.spectrum()


def test_narrow_refuses_a_k_it_cannot_pack(reads):
    """`engine="narrow"` is a pin, not a hint: it fails rather than
    silently upgrading, which is what a caller who needs a `kmer_u64`
    column downstream is asking for.
    """
    with pytest.raises(ValueError) as excinfo:
        fastdna.count(reads, k=33, engine="narrow")
    assert "wide" in str(excinfo.value)


@pytest.mark.parametrize("k", [0, 65, 100])
def test_k_outside_both_engines_is_rejected(reads, k):
    with pytest.raises(InvalidKError):
        fastdna.count(reads, k=k)


def test_an_unknown_engine_name_is_rejected(reads):
    with pytest.raises(ValueError) as excinfo:
        fastdna.count(reads, k=31, engine="u128")
    assert "auto" in str(excinfo.value)


def test_build_info_reports_one_ceiling():
    """One number again: counting, the table operations, sketching and the
    estimators all reach 64.

    This asserted two ceilings for two days, while sketching stopped at 32.
    A `max_k_sketch` pinned at 64 would read as if the discrepancy were
    still there, so the key is gone rather than updated.
    """
    info = fastdna.build_info()
    assert info["max_k"] == 64
    assert "max_k_sketch" not in info


def test_sketching_reaches_64_too(reads):
    """Sketching used to stop at 32 while counting reached 64, and this
    test pinned that. It is inverted rather than deleted: a sketch stores
    hashes, not k-mers, so the width was only ever in the extraction --
    `jaccard` and `containment` compare `u64` hashes either way.
    """
    wide = fastdna.sketch(reads, k=41, sketch_size=256)
    assert wide.k == 41
    same = fastdna.sketch(reads, k=41, sketch_size=256)
    assert abs(wide.jaccard(same) - 1.0) < 1e-12, "a file is identical to itself"

    # The one entry point still capped at 32, and for a reason that is not
    # about sketching: its argument is a list of `u64`-packed k-mers, and a
    # u64 cannot hold one longer than 32 bases.
    with pytest.raises(InvalidKError):
        fastdna.sketch_from_kmers([1, 2, 3], k=41)


def _count_to_parquet(reads, tmp_path, k):
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    """Writes a table through the CLI, because `fastdna.count()` returns an
    in-memory `KmerCounts` and cannot write a file carrying the footer
    metadata `KmerTable.open` requires -- the same limitation
    `KmerTable`'s own docstring states.
    """
    import subprocess

    binary = shutil.which("fastdna")
    if binary is None:
        for candidate in ("target/release/fastdna", "target/debug/fastdna"):
            if pathlib.Path(candidate).exists():
                binary = candidate
                break
    if binary is None:
        pytest.skip("the fastdna binary is not built; run `cargo build --release`")

    out = tmp_path / f"table_k{k}.parquet"
    subprocess.run(
        [
            binary, "count",
            "--input", str(reads),
            "-k", str(k),
            "-o", str(out),
            # Redirected out of the current directory on purpose: `--qc`
            # defaults to `qc_report.json` relative to the CWD and is
            # written unconditionally, so a test that leaves the default
            # alone drops a file in whatever directory pytest was run from.
            "--qc", str(tmp_path / "qc.json"),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_a_wide_table_opens_and_answers_lookups(reads, tmp_path):
    """The other half of reaching k>32 from Python: a table counted above
    32 has to be readable back, not just writable.
    """
    counts = fastdna.count(reads, k=41).with_sequence()
    expected = dict(
        zip(
            counts.table.column("kmer_sequence").to_pylist(),
            counts.table.column("frequency").to_pylist(),
        )
    )

    table = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 41))
    assert table.engine == "wide"
    assert table.k == 41
    assert len(table) == len(expected)

    for kmer, frequency in list(expected.items())[:20]:
        assert table.get(kmer) == frequency
        assert table[kmer] == frequency
        assert kmer in table

    absent = "A" * 41
    assert absent not in expected
    assert table.get(absent) is None
    with pytest.raises(KeyError):
        table[absent]


def test_a_narrow_table_still_opens_as_narrow(reads, tmp_path):
    table = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 21))
    assert table.engine == "narrow"
    assert table.k == 21
    assert len(table) > 0


def test_set_operations_work_on_a_wide_table(reads, tmp_path):
    """`setops` is generic over the key now, so `union`/`intersect`/
    `difference` take a wide table and hand back a wide one.

    The assertions are set-algebra identities over a table combined with
    itself -- answers that hold by definition, not ones this package
    computed: a self-union and a self-intersection are the table, and a
    self-difference is empty.
    """
    wide = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 41))
    assert wide.engine == "wide"

    unioned = wide.union(wide, output=str(tmp_path / "u.parquet"))
    assert unioned.engine == "wide", "a wide input must produce a wide output"
    assert unioned.k == 41
    assert len(unioned) == len(wide)

    intersected = wide.intersect(wide, output=str(tmp_path / "i.parquet"))
    assert len(intersected) == len(wide)

    differenced = wide.difference(wide, output=str(tmp_path / "d.parquet"))
    assert len(differenced) == 0, "a table minus itself is empty"


def test_mixing_widths_in_a_set_operation_is_refused_by_name(reads, tmp_path):
    """Two widths never share a `k`, so a mixed set is always a mistake."""
    narrow = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 21))
    wide = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 41))

    with pytest.raises(ValueError) as excinfo:
        narrow.union(wide, output=str(tmp_path / "mixed.parquet"))
    message = str(excinfo.value)
    assert "narrow" in message and "wide" in message, message


def test_read_filtering_takes_a_wide_table(reads, tmp_path):
    """`filter_reads` indexes a wide reference now. The expectation is
    definitional rather than a number this package produced: every read
    comes from the very file the reference was counted from, so at
    `min_fraction=1.0` keep-mode keeps all of them and discard-mode none.
    """
    wide = fastdna.KmerTable.open(_count_to_parquet(reads, tmp_path, 41))
    assert wide.engine == "wide"

    kept = tmp_path / "kept.fastq"
    stats = fastdna._core.filter_reads(
        table=wide._raw,
        inputs=[str(reads)],
        mode="keep",
        output=str(kept),
        min_fraction=1.0,
    )
    assert stats.reads_written == stats.reads_total
    assert stats.reads_total > 0

    dropped = tmp_path / "dropped.fastq"
    stats = fastdna._core.filter_reads(
        table=wide._raw,
        inputs=[str(reads)],
        mode="discard",
        output=str(dropped),
        min_fraction=1.0,
    )
    assert stats.reads_written == 0, "no read can be absent from its own reference"


def test_similarity_takes_wide_tables(reads, tmp_path):
    """`similarity` reads both widths now. Checked against the tables' own
    row counts, which come from the Parquet footer rather than from the
    merge that produced these numbers: `shared + only_a` must be `|A|`.
    """
    a = _count_to_parquet(reads, tmp_path, 41)
    # A second sample from the *same* genome but with fewer reads: its
    # k-mers are a proper subset of A's, so the pair overlaps partially.
    # A different seed would give an unrelated genome and a jaccard of 0,
    # which would satisfy nothing.
    other = tmp_path / "other.fastq"
    _write_reads(other, n_reads=12, seed=11)
    b = _count_to_parquet(other, tmp_path / "b", 41)

    table = fastdna.similarity([a, b])
    assert table.num_rows == 1
    row = {name: table.column(name)[0].as_py() for name in table.column_names}

    assert row["shared"] + row["only_a"] == len(fastdna.KmerTable.open(a))
    assert row["shared"] + row["only_b"] == len(fastdna.KmerTable.open(b))
    assert 0.0 < row["jaccard"] < 1.0, row


def test_similarity_refuses_a_mix_of_widths(reads, tmp_path):
    narrow = _count_to_parquet(reads, tmp_path, 21)
    wide = _count_to_parquet(reads, tmp_path, 41)

    with pytest.raises(ValueError) as excinfo:
        fastdna.similarity([narrow, wide])
    assert "width" in str(excinfo.value), excinfo.value


def test_non_canonical_counting_splits_reverse_complement_pairs(tmp_path):
    """`canonical=False` counts each k-mer as it reads forward -- the CLI's
    `--no-canonical`, KMC3's `-b`.

    Checked by a property rather than a number: canonicalising *merges* a
    k-mer with its reverse complement, so turning it off can only split
    rows, never invent or lose an occurrence. The reads are written on both
    strands, because with one strand there is nothing to merge and the two
    counts come out equal.
    """
    genome_rng = random.Random(5)
    genome = "".join(genome_rng.choice("ACGT") for _ in range(800))
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}

    reads = tmp_path / "both_strands.fastq"
    with reads.open("w") as handle:
        for i, start in enumerate(range(0, 600, 23)):
            read = genome[start : start + 150]
            if i % 2:
                read = "".join(complement[b] for b in reversed(read))
            handle.write(f"@r{i}\n{read}\n+\n{'I' * len(read)}\n")

    canonical = fastdna.count(reads, k=21)
    forward = fastdna.count(reads, k=21, canonical=False)

    assert forward.total_kmers == canonical.total_kmers, "occurrences cannot change"
    assert forward.distinct_kmers > canonical.distinct_kmers, "pairs must split"
    assert forward.distinct_kmers <= canonical.distinct_kmers * 2, "at most one split per row"


def test_a_table_records_its_counting_convention(reads, tmp_path):
    """The convention lives in the Parquet footer, so a reader never has to
    assume it. `KmerTable.canonical` is what a caller checks before
    combining two tables.
    """
    import subprocess

    binary = shutil.which("fastdna")
    if binary is None:
        for candidate in ("target/release/fastdna", "target/debug/fastdna"):
            if pathlib.Path(candidate).exists():
                binary = candidate
                break
    if binary is None:
        pytest.skip("the fastdna binary is not built; run `cargo build --release`")

    tables = {}
    for label, extra in (("canonical", []), ("forward", ["--no-canonical"])):
        out = tmp_path / f"{label}.parquet"
        subprocess.run(
            [binary, "count", "--input", str(reads), "-k", "21", "-o", str(out),
             "--qc", str(tmp_path / f"{label}.qc.json"), *extra],
            check=True, capture_output=True,
        )
        tables[label] = fastdna.KmerTable.open(out)

    assert tables["canonical"].canonical is True
    assert tables["forward"].canonical is False

    # Combining the two is refused: they disagree about what a key means,
    # and nothing downstream would catch it -- same width, same k.
    with pytest.raises(ValueError) as excinfo:
        tables["canonical"].union(tables["forward"], output=str(tmp_path / "mixed.parquet"))
    assert "canonical" in str(excinfo.value)


def test_decoding_agrees_with_and_without_numpy(reads, tmp_path, monkeypatch):
    """numpy is a fast path in `_decode_kmers`/`_decode_wide_kmers`, not a
    requirement -- it is not in this package's runtime dependencies, so a
    plain `pip install fastdna` does not have it.

    Both branches must produce identical strings. This blocks the import
    the way a bare install does and compares the two, rather than trusting
    that two implementations of the same bit layout agree.
    """
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy blocked for this test")
        return real_import(name, *args, **kwargs)

    for k in (21, 41):
        with_numpy = fastdna.count(reads, k=k).with_sequence()
        expected = with_numpy.table.column("kmer_sequence").to_pylist()
        assert expected, "the fixture must produce k-mers"

        monkeypatch.setattr(builtins, "__import__", no_numpy)
        try:
            without = fastdna.count(reads, k=k).with_sequence()
            got = without.table.column("kmer_sequence").to_pylist()
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)

        assert got == expected, f"k={k}: the two decoders disagree"


def test_sketch_from_kmers_works_without_numpy(monkeypatch):
    """Same reason: this reached for numpy unconditionally and raised
    `ModuleNotFoundError` on the install everyone gets.
    """
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy blocked for this test")
        return real_import(name, *args, **kwargs)

    kmers = [17, 42, 99, 17, 1234567]
    expected = fastdna.sketch_from_kmers(kmers, k=21, sketch_size=8)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    try:
        got = fastdna.sketch_from_kmers(kmers, k=21, sketch_size=8)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)

    assert got.k == expected.k
    assert abs(got.jaccard(expected) - 1.0) < 1e-12, "the two paths must sketch identically"
