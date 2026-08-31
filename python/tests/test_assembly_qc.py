"""Tests for fastdna.assembly_qc -- Merqury-style (Rhie et al. 2020),
reference-free k-mer assembly QC (QV, completeness, spectra comparison).

Follows the house style of test_fluent_api.py/test_sketch.py: a
`write_fastq` helper building a FASTQ file from a list of read sequences,
plus (new here) a `write_fasta` helper building a FASTA assembly from a
list of contig sequences.
"""
from __future__ import annotations

import math
import pathlib
import random

import pytest

import fastdna
from fastdna.assembly_qc import AssemblyQC, _qv_from_counts, evaluate_assembly, evaluate_kmers


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def write_fasta(tmp_path: pathlib.Path, name: str, contigs: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f">contig{i}\n{s}\n" for i, s in enumerate(contigs)))
    return p


#: A fixed-seed pseudo-random 300 bp "reference genome" -- long enough that,
#: at k=21, its ~280 distinct canonical k-mers are (essentially certainly)
#: all unique, so k-mer-set membership tests behave the way a real
#: non-repetitive genomic region would, without actually needing a real
#: genome on disk.
_REFERENCE = "".join(random.Random(20260822).choices("ACGTACGT", k=300))
_K = 21


def _tile_reads(reference: str, read_len: int = 80) -> list[str]:
    """Dense (step=1) sliding-window tiling of `reference`: every possible
    `read_len`-length substring, so every k-mer (for any k <= read_len) of
    `reference` is guaranteed to occur in at least one read -- i.e. these
    reads are complete, error-free ground truth for `reference`'s own
    k-mer content.
    """
    return [reference[i : i + read_len] for i in range(0, len(reference) - read_len + 1)]


def _substitute(seq: str, positions: list[int]) -> str:
    """Returns `seq` with the base at each index in `positions` changed to
    a different base (deterministically: the next base in the ACGT cycle),
    simulating assembly consensus/base-calling errors.
    """
    chars = list(seq)
    cycle = "ACGT"
    for pos in positions:
        chars[pos] = cycle[(cycle.index(chars[pos]) + 1) % 4]
    return "".join(chars)


@pytest.fixture(scope="module")
def reads_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("assembly_qc_reads")
    reads = _tile_reads(_REFERENCE, read_len=80)
    return write_fastq(tmp_path, "reads.fastq", reads)


class TestPerfectAssembly:
    """The assembly *is* the reference the reads were tiled from: every
    assembly k-mer should be found in the reads (QV very high/infinite),
    and every reliable read k-mer should be found in the assembly
    (completeness == 1.0).
    """

    @pytest.fixture
    def result(self, tmp_path_factory, reads_path):
        tmp_path = tmp_path_factory.mktemp("perfect_assembly")
        assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        return evaluate_assembly(str(assembly), str(reads_path), k=_K)

    def test_qv_is_infinite(self, result):
        # No corruption anywhere: every assembly k-mer is, by construction,
        # a k-mer of the reference the (error-free) reads were tiled from,
        # so k_shared == k_total exactly -> error_rate == 0 -> QV == inf.
        assert result.error_rate == 0.0
        assert math.isinf(result.qv)

    def test_completeness_is_perfect(self, result):
        assert result.completeness == pytest.approx(1.0)

    def test_assembly_kmer_counts_are_sane(self, result):
        # 300 bp reference, k=21 -> 280 possible windows; no self-repeats
        # expected in a 300 bp pseudo-random sequence at k=21.
        assert result.assembly_distinct_kmers == 300 - _K + 1
        assert result.assembly_kmers_found_in_reads == result.assembly_distinct_kmers

    def test_spectra_table_has_expected_columns(self, result):
        names = result.spectra.column_names
        for col in ("depth", "total", "missing_in_assembly", "single_in_assembly", "multi_in_assembly"):
            assert col in names
        # Perfect assembly: no read k-mer, at any depth, is missing from
        # the assembly.
        assert sum(result.spectra.column("missing_in_assembly").to_pylist()) == 0


class TestBadAssembly:
    """Same reads, but the assembly has several substituted bases -- QV
    must come out measurably lower than the perfect case, proving the
    metric responds to real assembly errors (not just always reporting the
    same/near-infinite number).
    """

    @pytest.fixture
    def result(self, tmp_path_factory, reads_path):
        tmp_path = tmp_path_factory.mktemp("bad_assembly")
        # Five substitutions, spaced >= k apart so their corrupted-k-mer
        # windows do not overlap each other or the sequence ends.
        corrupted = _substitute(_REFERENCE, [50, 100, 150, 200, 250])
        assembly = write_fasta(tmp_path, "assembly.fasta", [corrupted])
        return evaluate_assembly(str(assembly), str(reads_path), k=_K)

    def test_qv_is_measurably_lower_than_perfect(self, result, tmp_path_factory, reads_path):
        tmp_path = tmp_path_factory.mktemp("bad_assembly_baseline")
        perfect_assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        perfect = evaluate_assembly(str(perfect_assembly), str(reads_path), k=_K)

        assert math.isinf(perfect.qv)
        assert math.isfinite(result.qv)
        assert result.qv < perfect.qv
        assert result.error_rate > 0.0

    def test_error_rate_reflects_corruption_extent(self, result):
        # 5 substitutions each corrupt up to k=21 overlapping k-mers ->
        # up to 105 of the ~280 distinct assembly k-mers should fail to be
        # found in the (unmodified) reads, i.e. k_shared/k_total ~ 175/280
        # = 0.625 (observed exactly: this test run's own
        # assembly_kmers_found_in_reads/assembly_distinct_kmers). Feeding
        # that ratio through the formula by hand
        # (error = 1 - 0.625**(1/21)) gives ~0.0221 -- note the exponent
        # (1/k) compresses a large fractional-k-mer loss into a much
        # smaller *per-base* error rate, which is the whole point of the
        # formula (see module docstring). Assert a wide-but-meaningful
        # band around that, rather than an exact float, since precise
        # edge behavior of which windows are affected is an
        # implementation detail of this test's own corruption placement,
        # not of the QV formula.
        assert 0.01 < result.error_rate < 0.05

    def test_missing_shared_count_dropped_from_perfect(self, result):
        # k_total is essentially unchanged (same assembly length, no
        # indels) but k_shared must have dropped.
        assert result.assembly_kmers_found_in_reads < result.assembly_distinct_kmers


class TestIncompleteAssembly:
    """Same reads, but the assembly is missing a chunk of the true
    sequence entirely (two contigs, with a gap where a third piece would
    have been) -- completeness must come out measurably lower than the
    perfect case, while QV (which only judges the k-mers the assembly
    *does* contain) stays high, since nothing present in this assembly is
    actually wrong.
    """

    @pytest.fixture
    def result(self, tmp_path_factory, reads_path):
        tmp_path = tmp_path_factory.mktemp("incomplete_assembly")
        # Drop reference[100:160] (60 bp) entirely; keep the two flanking
        # pieces as *separate* contigs so no fabricated "junction" k-mer
        # (spanning the artificial join) is introduced -- isolating the
        # completeness effect from the QV metric.
        contig1 = _REFERENCE[:100]
        contig2 = _REFERENCE[160:]
        assembly = write_fasta(tmp_path, "assembly.fasta", [contig1, contig2])
        return evaluate_assembly(str(assembly), str(reads_path), k=_K)

    def test_completeness_is_measurably_lower_than_perfect(self, result, tmp_path_factory, reads_path):
        tmp_path = tmp_path_factory.mktemp("incomplete_assembly_baseline")
        perfect_assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        perfect = evaluate_assembly(str(perfect_assembly), str(reads_path), k=_K)

        assert perfect.completeness == pytest.approx(1.0)
        assert result.completeness < perfect.completeness
        assert result.completeness < 0.9

    def test_completeness_is_not_degenerately_zero(self, result):
        # Most of the reference is still present -- only ~60-100 bp worth
        # of k-mers (the deleted chunk plus boundary effects) are missing
        # out of ~280 total.
        assert 0.3 < result.completeness < 0.95

    def test_qv_is_not_degraded_by_a_clean_deletion(self, result):
        # Every k-mer the (two-contig) assembly *does* contain is a real,
        # unmodified piece of the reference, so it should still be found
        # in the reads -- a missing chunk is a completeness problem, not a
        # QV (correctness-of-what's-there) problem.
        assert result.error_rate == 0.0
        assert math.isinf(result.qv)

    def test_missing_bucket_in_spectra_is_nonzero(self, result):
        # The deleted chunk's read k-mers must show up somewhere as
        # "missing from assembly" in the per-depth breakdown.
        assert sum(result.spectra.column("missing_in_assembly").to_pylist()) > 0


class TestQvFormulaArithmetic:
    """Direct arithmetic checks of `_qv_from_counts` against hand-computable
    values -- catches a formula bug (e.g. a wrong constant factor) that the
    "does it respond to bad/incomplete assemblies" tests above would not
    necessarily catch, since those only check *direction*, not magnitude.
    """

    def test_clean_hand_computed_example(self):
        # k_shared=81, k_total=100, k=2: ratio=0.81, and sqrt(0.81) == 0.9
        # exactly, so p_correct=0.9, error_rate=0.1, and
        # QV = -10*log10(0.1) = -10*(-1) = 10.0 -- exact by hand, no
        # calculator needed beyond sqrt(0.81)=0.9 and log10(0.1)=-1.
        error_rate, qv = _qv_from_counts(k_shared=81, k_total=100, k=2)
        assert error_rate == pytest.approx(0.1)
        assert qv == pytest.approx(10.0)

    def test_matches_plain_python_one_liner(self):
        # A second, independently-phrased check: recompute the formula
        # from scratch in this test (not by calling the function under
        # test with different inputs) and assert the two agree.
        k_shared, k_total, k = 90, 100, 21
        expected_p_correct = (k_shared / k_total) ** (1.0 / k)
        expected_error = 1.0 - expected_p_correct
        expected_qv = -10.0 * math.log10(expected_error)

        error_rate, qv = _qv_from_counts(k_shared, k_total, k)

        assert error_rate == pytest.approx(expected_error)
        assert qv == pytest.approx(expected_qv)

    def test_perfect_agreement_gives_zero_error_and_infinite_qv(self):
        error_rate, qv = _qv_from_counts(k_shared=100, k_total=100, k=21)
        assert error_rate == 0.0
        assert math.isinf(qv)

    def test_zero_agreement_gives_error_one_and_qv_zero(self):
        # ratio=0 -> p_correct = 0**(1/k) = 0 -> error_rate=1.0 exactly ->
        # QV = -10*log10(1) = 0.0 exactly, the worst score on this scale.
        error_rate, qv = _qv_from_counts(k_shared=0, k_total=100, k=21)
        assert error_rate == pytest.approx(1.0)
        assert qv == pytest.approx(0.0)

    def test_raises_on_empty_assembly(self):
        with pytest.raises(ValueError):
            _qv_from_counts(k_shared=0, k_total=0, k=21)


class TestFastaVsFastqAssemblyPaths:
    """The assembly side dispatches on file extension (module docstring):
    FASTA goes through the pure-Python fallback, but a caller who already
    converted their assembly to FASTQ (a common workaround: dummy quality
    scores) should get routed through the real `fastdna.count()` path and
    reach an equivalent answer.
    """

    def test_fastq_assembly_path_matches_fasta_assembly_path(self, tmp_path, reads_path):
        fasta_assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        # Dummy quality string of 'I' per base, one record -- exactly the
        # workaround described in the module docstring.
        fastq_assembly = write_fastq(tmp_path, "assembly.fastq", [_REFERENCE])

        from_fasta = evaluate_assembly(str(fasta_assembly), str(reads_path), k=_K)
        from_fastq = evaluate_assembly(str(fastq_assembly), str(reads_path), k=_K)

        assert from_fasta.assembly_distinct_kmers == from_fastq.assembly_distinct_kmers
        assert from_fasta.assembly_kmers_found_in_reads == from_fastq.assembly_kmers_found_in_reads
        assert from_fasta.qv == from_fastq.qv
        assert from_fasta.completeness == pytest.approx(from_fastq.completeness)


class TestEvaluateKmersLowLevelEntryPoint:
    """`evaluate_kmers` accepts a pre-built assembly k-mer multiset
    directly (a `Mapping[str, int]` or any iterable of canonical k-mer
    strings), for callers who already have the assembly's k-mers by some
    other means and want to skip file handling entirely.
    """

    def test_accepts_a_plain_mapping(self, reads_path):
        reads_counts = fastdna.count(str(reads_path), k=_K)
        from fastdna.assembly_qc import _count_fasta_kmers

        # Build the mapping "by some other means" -- here, just reusing
        # the module's own FASTA counter directly on an in-memory FASTA
        # written to a temp file, to get a real, correct k-mer multiset
        # without re-deriving the canonicalization logic in the test.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = write_fasta(pathlib.Path(d), "assembly.fasta", [_REFERENCE])
            assembly_kmers = _count_fasta_kmers(str(path), _K)

        result = evaluate_kmers(assembly_kmers, reads_counts, k=_K)

        assert isinstance(result, AssemblyQC)
        assert result.error_rate == 0.0
        assert math.isinf(result.qv)

    def test_accepts_an_iterable_of_kmer_strings(self, reads_path):
        # with_sequence=True: this test pulls decoded strings straight out
        # of the table below, which needs the column present.
        reads_counts = fastdna.count(str(reads_path), k=_K, with_sequence=True)

        # A minimal, deliberately tiny assembly k-mer "set" as a plain
        # list of canonical k-mer strings pulled straight from the reads'
        # own table -- exercises the Counter(iterable) fallback path.
        some_read_kmers = reads_counts.table.column("kmer_sequence").to_pylist()[:10]

        result = evaluate_kmers(some_read_kmers, reads_counts, k=_K)

        assert result.assembly_distinct_kmers == len(set(some_read_kmers))
        assert result.assembly_kmers_found_in_reads == result.assembly_distinct_kmers


class TestMinCountDefaulting:
    def test_none_uses_suggest_min_count(self, tmp_path, reads_path):
        reads_counts = fastdna.count(str(reads_path), k=_K)
        expected = reads_counts.suggest_min_count()

        assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        result = evaluate_assembly(str(assembly), str(reads_path), k=_K, min_count=None)

        assert result.min_count_used == expected

    def test_explicit_min_count_is_honored(self, tmp_path, reads_path):
        assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])
        result = evaluate_assembly(str(assembly), str(reads_path), k=_K, min_count=7)

        assert result.min_count_used == 7


def test_fasta_with_ambiguous_base_does_not_crash_and_skips_it(tmp_path, reads_path):
    # An 'N' in the assembly (common at contig gaps/low-confidence calls)
    # must reset the k-mer window rather than being silently treated as a
    # real base -- matching src/kmer.rs::extract_canonical_kmers.
    with_n = _REFERENCE[:150] + "N" + _REFERENCE[151:]
    assembly = write_fasta(tmp_path, "assembly.fasta", [with_n])

    result = evaluate_assembly(str(assembly), str(reads_path), k=_K)

    # The N destroys up to k windows around position 150, so k_total must
    # be strictly less than the no-N case, and nothing should raise.
    assert result.assembly_distinct_kmers < 300 - _K + 1
