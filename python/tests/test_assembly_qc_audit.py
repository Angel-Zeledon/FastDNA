"""Audit tests for `fastdna.assembly_qc` (commits c316001, merge 8502c77).

Complements `test_assembly_qc.py`. That file pins the QV *arithmetic*
well (`TestQvFormulaArithmetic` is genuinely value-pinned, not shape-only),
but every FASTQ it writes uses `'I' * len(s)` -- Phred 40 on every base of
every read -- so the read-side quality handling inside
`evaluate_assembly` is never exercised with anything a real sequencer
produces. It also never reaches the `.gz` FASTA path, the
`multi_in_assembly` spectra bucket, the NaN-completeness branch, or
`AssemblyQC.__repr__`.

Tests whose docstring begins with "EXPECTED TO FAIL" pin a reported
defect and are expected to be red until the module is patched or reverted;
see the audit report. They are deliberately not xfail-marked.
"""
from __future__ import annotations

import gzip
import math
import pathlib
import random

import pytest

import fastdna
from fastdna.assembly_qc import (
    _canonical_kmers,
    _qv_from_counts,
    evaluate_assembly,
    evaluate_kmers,
)


#: Same construction as `test_assembly_qc.py`: a seeded 300 bp pseudo-random
#: "genome" whose ~280 canonical 21-mers are all distinct.
_REFERENCE = "".join(random.Random(20260822).choices("ACGTACGT", k=300))
_K = 21


def write_fasta(tmp_path: pathlib.Path, name: str, contigs: list) -> str:
    p = tmp_path / name
    p.write_text("".join(f">contig{i}\n{s}\n" for i, s in enumerate(contigs)))
    return str(p)


def write_fastq_with_quality(tmp_path: pathlib.Path, name: str, reads: list, quality) -> str:
    """`quality` is a callable mapping a read sequence to its quality
    string, so a test can hand `evaluate_assembly` reads that are not
    uniformly Phred 40.
    """
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{quality(s)}\n" for i, s in enumerate(reads)))
    return str(p)


def tile_reads(reference: str, read_len: int = 80) -> list:
    return [reference[i : i + read_len] for i in range(0, len(reference) - read_len + 1)]


# ---------------------------------------------------------------------------
# The QV formula, cross-checked against Merqury's own script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k_shared,k_total,k",
    [(81, 100, 2), (90, 100, 21), (279, 280, 21), (1, 1000, 31), (250, 280, 21)],
)
def test_qv_matches_merqury_qv_sh_awk_expression_verbatim(k_shared, k_total, k):
    """Independent cross-check of `_qv_from_counts` against the literal
    expression in Merqury's own `qv.sh`:

        -10 * log(1 - (1 - ASM_ONLY/TOTAL) ** (1/k)) / log(10)

    where `ASM_ONLY` is the count of assembly k-mers absent from the reads
    and `TOTAL` the assembly's distinct k-mer count. This is phrased in
    terms of the *missing* count and uses natural logs divided by log(10),
    i.e. a different algebraic route to the same number than the module's
    `1 - (k_shared/k_total) ** (1/k)` and `math.log10`. Agreement across
    both phrasings is what rules out a transcription error in either.

    `test_assembly_qc.py::test_matches_plain_python_one_liner` re-derives
    the module's own phrasing, which cannot catch a shared mistake in the
    phrasing itself.
    """
    assembly_only = k_total - k_shared
    expected_qv = -10.0 * math.log(1 - (1 - assembly_only / k_total) ** (1.0 / k)) / math.log(10)

    error_rate, qv = _qv_from_counts(k_shared=k_shared, k_total=k_total, k=k)

    assert qv == pytest.approx(expected_qv)
    assert error_rate == pytest.approx(1 - (1 - assembly_only / k_total) ** (1.0 / k))


def test_qv_boundaries_match_the_merqury_convention():
    """Both ends of the scale, checked against what `qv.sh`'s awk would
    print: no missing k-mers -> `log(0)` -> +infinity; every k-mer missing
    -> `-10 * log(1)/log(10)` -> exactly 0.0, the worst score.
    """
    error_rate, qv = _qv_from_counts(k_shared=100, k_total=100, k=_K)
    assert error_rate == 0.0
    assert math.isinf(qv) and qv > 0

    error_rate, qv = _qv_from_counts(k_shared=0, k_total=100, k=_K)
    assert error_rate == 1.0
    assert qv == 0.0

    with pytest.raises(ValueError, match="no valid k-mers"):
        _qv_from_counts(k_shared=0, k_total=0, k=_K)


def test_qv_uses_assembly_vs_reads_and_completeness_uses_reads_vs_assembly(tmp_path):
    """Direction check for the two headline numbers, which are computed
    over *different* sets and must not be swapped:

      * QV grades assembly k-mers against the reads -- an assembly that is
        correct but incomplete must keep a perfect QV.
      * Completeness grades reliable read k-mers against the assembly -- an
        assembly that is complete but contains errors must keep perfect
        completeness.

    Building one assembly of each kind and asserting the *opposite* metric
    stays perfect in each case would fail if the two set comparisons were
    ever transposed.
    """
    reads = tile_reads(_REFERENCE)
    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", reads, lambda s: "I" * len(s)
    )

    # Correct but incomplete: two clean contigs, a 60 bp chunk dropped.
    incomplete = write_fasta(
        tmp_path, "incomplete.fasta", [_REFERENCE[:100], _REFERENCE[160:]]
    )
    incomplete_qc = evaluate_assembly(incomplete, reads_path, k=_K)

    assert math.isinf(incomplete_qc.qv), "a clean deletion is not a QV problem"
    assert incomplete_qc.completeness < 0.9, "but it is a completeness problem"

    # Complete but wrong: the whole reference plus an extra contig of pure
    # invention, so every real read k-mer is still present.
    invented = "".join(random.Random(4242).choices("ACGT", k=200))
    erroneous = write_fasta(tmp_path, "erroneous.fasta", [_REFERENCE, invented])
    erroneous_qc = evaluate_assembly(erroneous, reads_path, k=_K)

    assert erroneous_qc.completeness == pytest.approx(1.0), (
        "every reliable read k-mer is still in the assembly"
    )
    assert math.isfinite(erroneous_qc.qv), "but the invented contig is a QV problem"
    assert erroneous_qc.assembly_kmers_found_in_reads < erroneous_qc.assembly_distinct_kmers


# ---------------------------------------------------------------------------
# The read side: quality handling
# ---------------------------------------------------------------------------


def test_qv_is_not_fabricated_by_ordinary_read_quality_decay(tmp_path):
    """EXPECTED TO FAIL -- pins a reported defect.

    `evaluate_assembly` builds its ground-truth read k-mer set with
    `fastdna.count(reads_path, k=k)` and no `min_quality` argument, so it
    silently inherits `count`'s default `min_quality=20.0`, which
    3'-end-trims every read (`src/pipeline.rs`'s `quality_trim_end`). Every
    read k-mer that existed only in a trimmed tail vanishes from the
    ground truth, and every assembly k-mer that depended on it is then
    scored as an *assembly error*.

    Merqury builds its read k-mer database from the reads as given
    (`meryl count` on the FASTQ); nothing in this module's docstrings,
    `evaluate_assembly`'s signature, or `AssemblyQC`'s field descriptions
    mentions quality trimming, and there is no parameter to turn it off.

    This test grades one perfect assembly -- the exact sequence the reads
    were tiled from -- twice, against read sets that differ *only* in
    their quality strings:

      * all Phred 40  -> error_rate 0.00000, QV inf   (correct)
      * last 20 bases at Phred 2, an entirely ordinary Illumina 3' decay
                      -> error_rate 0.00316, QV 25.00 (fabricated)

    QV 25 reads as "one consensus error per ~316 bases" for an assembly
    with no errors at all. That is the number a researcher puts in a paper.
    """
    reads = tile_reads(_REFERENCE)
    assembly = write_fasta(tmp_path, "perfect.fasta", [_REFERENCE])

    pristine = write_fastq_with_quality(
        tmp_path, "q40.fastq", reads, lambda s: "I" * len(s)
    )
    decayed = write_fastq_with_quality(
        tmp_path, "decayed.fastq", reads, lambda s: "I" * (len(s) - 20) + "#" * 20
    )

    from_pristine = evaluate_assembly(assembly, pristine, k=_K)
    from_decayed = evaluate_assembly(assembly, decayed, k=_K)

    assert math.isinf(from_pristine.qv), "sanity: the all-Q40 baseline"
    assert from_decayed.qv == from_pristine.qv, (
        "a perfect assembly's QV changed because the reads' quality strings "
        f"changed: Q40 reads -> QV {from_pristine.qv}, 3'-decayed reads -> QV "
        f"{from_decayed.qv:.2f} (error_rate {from_decayed.error_rate:.5f}, "
        f"{from_decayed.assembly_kmers_found_in_reads}/"
        f"{from_decayed.assembly_distinct_kmers} assembly k-mers found)"
    )


def test_evaluate_kmers_lets_a_caller_avoid_the_quality_trim(tmp_path):
    """Characterisation test (currently GREEN): documents the only
    available workaround for the defect above -- go through the lower-level
    `evaluate_kmers` and build the read counts yourself with
    `min_quality=0.0`. Recorded so the workaround is not lost, and so it
    breaks loudly if `evaluate_kmers` ever starts doing its own counting.
    """
    from fastdna.assembly_qc import _count_fasta_kmers

    reads = tile_reads(_REFERENCE)
    decayed = write_fastq_with_quality(
        tmp_path, "decayed.fastq", reads, lambda s: "I" * (len(s) - 20) + "#" * 20
    )
    assembly = write_fasta(tmp_path, "perfect.fasta", [_REFERENCE])

    untrimmed_counts = fastdna.count(decayed, k=_K, min_quality=0.0)
    result = evaluate_kmers(_count_fasta_kmers(assembly, _K), untrimmed_counts, k=_K)

    assert result.error_rate == 0.0
    assert math.isinf(result.qv)


# ---------------------------------------------------------------------------
# Branches `test_assembly_qc.py` never reaches
# ---------------------------------------------------------------------------


def test_multi_in_assembly_spectra_bucket_is_reachable_and_correct(tmp_path):
    """`AssemblyQC.spectra`'s `multi_in_assembly` column is documented as
    "a potential erroneous duplication, or a collapsed repeat" -- the whole
    point of a spectra-cn breakdown -- and no test in
    `test_assembly_qc.py` ever produces a nonzero value in it. Line 421 of
    `assembly_qc.py` (`buckets[freq]["multi"] += 1`) was dead under the
    existing suite.

    An assembly containing the reference twice puts every read k-mer at
    assembly count 2, so the entire spectrum must land in `multi`.
    """
    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", tile_reads(_REFERENCE), lambda s: "I" * len(s)
    )
    duplicated = write_fasta(tmp_path, "duplicated.fasta", [_REFERENCE, _REFERENCE])

    result = evaluate_assembly(duplicated, reads_path, k=_K)
    spectra = result.spectra

    multi = sum(spectra.column("multi_in_assembly").to_pylist())
    single = sum(spectra.column("single_in_assembly").to_pylist())
    missing = sum(spectra.column("missing_in_assembly").to_pylist())
    total = sum(spectra.column("total").to_pylist())

    assert multi > 0, "the duplicated-contig case must populate multi_in_assembly"
    assert single == 0 and missing == 0
    assert multi + single + missing == total

    # A duplicated contig is not an error, so QV stays perfect...
    assert math.isinf(result.qv)
    # ...and the per-depth `total` column must agree with the reads' own
    # spectrum, as AssemblyQC's field docstring claims.
    reads_spectrum = fastdna.count(reads_path, k=_K).spectrum()
    depths = spectra.column("depth").to_pylist()
    totals = spectra.column("total").to_pylist()
    assert dict(zip(depths, totals)) == {int(d): int(c) for d, c in reads_spectrum.items()}


def test_gzipped_fasta_assembly_is_read(tmp_path):
    """The module docstring advertises `.fasta`/`.fa`/`.fna`(`.gz`). No
    existing test passes a `.gz` path, so `_open_text`'s `gzip.open` branch
    was never executed.
    """
    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", tile_reads(_REFERENCE), lambda s: "I" * len(s)
    )
    gz_path = tmp_path / "assembly.fasta.gz"
    with gzip.open(gz_path, "wt") as fh:
        fh.write(f">contig0\n{_REFERENCE}\n")

    result = evaluate_assembly(str(gz_path), reads_path, k=_K)

    assert result.assembly_distinct_kmers == 300 - _K + 1
    assert math.isinf(result.qv)


def test_wrapped_and_blank_line_fasta_is_parsed(tmp_path):
    """`_iter_fasta_sequences` claims to join multi-line records and ignore
    blank lines; the existing suite only ever writes single-line contigs,
    so the `continue` on the blank-line branch was never executed. A real
    assembler wraps at 60 or 80 columns.
    """
    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", tile_reads(_REFERENCE), lambda s: "I" * len(s)
    )
    wrapped = tmp_path / "wrapped.fasta"
    wrapped.write_text(
        ">contig0\n\n" + "\n".join(_REFERENCE[i : i + 60] for i in range(0, 300, 60)) + "\n\n"
    )
    single_line = write_fasta(tmp_path, "single.fasta", [_REFERENCE])

    wrapped_qc = evaluate_assembly(str(wrapped), reads_path, k=_K)
    single_qc = evaluate_assembly(single_line, reads_path, k=_K)

    assert wrapped_qc.assembly_distinct_kmers == single_qc.assembly_distinct_kmers == 280
    assert wrapped_qc.qv == single_qc.qv


def test_contig_shorter_than_k_contributes_no_kmers(tmp_path):
    """`_canonical_kmers`'s `k > len(seq)` early return was never executed;
    short contigs are routine output from real assemblers.
    """
    assert list(_canonical_kmers("ACGT", _K)) == []
    assert list(_canonical_kmers(_REFERENCE, 0)) == []

    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", tile_reads(_REFERENCE), lambda s: "I" * len(s)
    )
    with_short = write_fasta(tmp_path, "with_short.fasta", ["ACGT", _REFERENCE])

    result = evaluate_assembly(with_short, reads_path, k=_K)

    assert result.assembly_distinct_kmers == 300 - _K + 1
    assert math.isinf(result.qv), "a too-short contig must not be scored as an error"


def test_completeness_is_nan_when_no_read_kmer_is_reliable(tmp_path):
    """`AssemblyQC.completeness` is documented to be NaN when the reads
    contribute zero reliable k-mers at the threshold used. That branch, and
    `AssemblyQC.__repr__`'s matching NaN/inf formatting, were both
    unexecuted.
    """
    reads_path = write_fastq_with_quality(
        tmp_path, "reads.fastq", tile_reads(_REFERENCE), lambda s: "I" * len(s)
    )
    assembly = write_fasta(tmp_path, "assembly.fasta", [_REFERENCE])

    result = evaluate_assembly(assembly, reads_path, k=_K, min_count=10**9)

    assert math.isnan(result.completeness)
    assert result.min_count_used == 10**9
    assert "completeness=nan" in repr(result)
    assert "qv=inf" in repr(result)

    finite = evaluate_assembly(assembly, reads_path, k=_K, min_count=1)
    assert "completeness=1.0000" in repr(finite)
