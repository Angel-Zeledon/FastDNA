"""The `k > 32` engine, as reached from Python.

`fastdna.count(engine=...)` routes to one of two engines: the u64 one that
has always been here (`k <= 32`) and the u128 one (`33 <= k <= 64`). What
these tests pin is the part that is checkable without trusting either
engine's own arithmetic: **the two must agree, k-mer for k-mer, wherever
both are defined**. `tests/wide_engine.rs` makes the same check inside the
core; this makes it through the binding, which is where the routing,
the Arrow column type and the decoding actually live.
"""

import random

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


def test_build_info_reports_both_ceilings():
    """`max_k` is what `count()` reaches; `max_k_sketch` is the u64
    ceiling the rest of the package still has. Two numbers because a
    caller who read one and handed 41 to `sketch()` would be misled.
    """
    info = fastdna.build_info()
    assert info["max_k"] == 64
    assert info["max_k_sketch"] == 32


def test_the_u64_keyed_surface_still_stops_at_32(reads):
    """The honest limit of this feature: counting reaches 64, but
    sketching -- and every other k-mer-table operation -- does not. Pinned
    so the day one of them grows a wide form, this test is what says so.
    """
    with pytest.raises(InvalidKError):
        fastdna.sketch(reads, k=41)
