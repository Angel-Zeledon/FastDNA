"""Merqury-style, reference-free genome assembly quality assessment.

Implements the k-mer-based approach of Rhie, Walenz, Koren & Phillippy,
"Merqury: reference-free quality, completeness, and phasing assessment for
genome assemblies", Genome Biology 21:245 (2020),
https://doi.org/10.1186/s13059-020-02134-9.

The idea: instead of grading an assembly against a curated truth reference
(which usually does not exist -- the assembly *is* the attempt at one), grade
it against the raw sequencing reads it was built from. A read is direct
observation of the sequenced DNA; an assembly k-mer that never occurs in any
read is, with very high probability for a reasonably sized read set, a
consensus/base-calling error rather than real biology the reads simply
missed. This gives two reference-free numbers:

* **QV** -- a Phred-like per-base accuracy estimate, from the *fraction of
  assembly k-mers absent from the reads*.
* **Completeness** -- what fraction of the reads' own *reliable* k-mers (i.e.
  plausibly real, not sequencing-error k-mers) turn up anywhere in the
  assembly. Low completeness means the assembly is missing real content the
  reads support.

plus a simplified per-depth "spectra" breakdown (Merqury's "spectra-cn"
idea, without the plot) a caller can chart themselves.

## The FASTA/FASTQ mismatch, and how this module handles it

`fastdna.count()` reads **FASTQ** (4-line records, via a Rust
`FastqReader` that expects an `@id` / sequence / `+` / quality-string
quartet) -- verified empirically and by reading `src/fastq.rs`: it has no
FASTA support, and feeding a 2-line FASTA record to it raises
`FastDnaError::MalformedFastq` (via `PyErr`), not a silently-wrong
"parsed" result, because that reader treats a record's second line as the
quality string it must decode as a Phred byte per sequence character
(and expects a `+` separator line it never finds).

A genome assembly is normally distributed as FASTA (no per-base quality --
there is nothing to report; it is a piece of the reads' own consensus, not
a re-sequenced observation), so `evaluate_assembly()`'s assembly side
cannot go through `fastdna.count()`'s fast Rust path. Two paths exist,
chosen automatically by the assembly file's extension:

1. **`.fasta`/`.fa`/`.fna`(`.gz`)** (the common case): parsed by a small
   pure-Python FASTA reader in this module, and k-mers extracted with a
   plain-Python reimplementation of the Rust core's own canonical-k-mer
   scheme (`src/kmer.rs`: A=00, C=01, G=10, T=11, canonical = the
   lexicographically smaller of the k-mer and its reverse complement --
   see `_canonical_kmers` below for why comparing the *strings*
   lexicographically, without reimplementing the 2-bit packing, gives
   exactly the same answer as the Rust core's own u64 comparison).
   **This path is genuinely slow** -- pure Python, one hash-set insert per
   base, no threading, no SIMD -- and is the honest cost of the Rust core
   having no FASTA support to hand off to. Fine for a bacterial-genome-scale
   assembly (megabases); expect real, possibly multi-minute+ latency for a
   mammalian-scale one (gigabases). This is a real limitation, not a
   nitpick to file away -- do not reach for this path expecting
   `fastdna.count()`-grade throughput.
2. **`.fastq`/`.fq`(`.gz`)**: if the assembly was *already* converted to
   FASTQ by the caller (a common workaround: append a dummy quality string
   such as all-`I` to each contig, e.g. with `seqkit fq2fa`'s inverse or a
   one-line script), this module recognizes the extension and routes it
   through the real `fastdna.count()` path instead -- full speed, no
   pure-Python fallback needed. This is why `evaluate_kmers()` (the
   lower-level entry point below) exists too: a caller who already has
   *both* sides as k-mer sets/counts by whatever means can skip file
   handling entirely.

## Confidence note on the QV formula

The formula implemented here (`_qv_from_counts`) was cross-checked against
two independent sources beyond memory: (a) the paper's own Methods text
(fetched via web search/fetch against the Genome Biology / PMC copy of
Rhie et al. 2020), which states P = (K_shared / K_total) ^ (1/k) for the
per-base correctness probability, and (b) the actual `qv.sh` script in the
reference implementation (github.com/marbl/merqury), whose `awk` line is
literally `(-10*log(1-(1-$1/$2)^(1/k))/log(10))` with `$1` = "assembly-only"
k-mer count and `$2` = total assembly k-mer count -- algebraically the same
formula (`1 - $1/$2 == K_shared/K_total`). Both sources describe K_total
and K_shared as **counts of distinct k-mers** in the assembly's own k-mer
set (not weighted by how many times each k-mer occurs across the assembly),
which is what this module implements. That said, this was established via
web research rather than a from-scratch derivation checked against the
published PDF's own typeset equation, so if a domain expert is available,
the one detail worth them double-checking directly against the paper's
Methods (or supplementary) PDF is exactly this distinct-vs-instance-count
question -- everything downstream (the `(1/k)` exponent, the `-10*log10`
Phred transform) is corroborated by both independent sources and is not in
doubt.
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pyarrow as pa

import fastdna

#: File extensions recognized as "already FASTQ" (see module docstring,
#: path 2): routed through the real `fastdna.count()` path instead of the
#: pure-Python FASTA fallback.
_FASTQ_EXTENSIONS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")

#: A k-mer containing any base outside A/C/G/T breaks the current window,
#: exactly like `kmer.rs::extract_canonical_kmers`'s `None` branch (an
#: ambiguous base such as 'N' cannot be assigned 2 bits, so no k-mer window
#: may span it). Splitting on this pattern yields the maximal runs of valid
#: bases -- the windows that may exist -- without testing every base
#: individually in Python.
_INVALID_RUN = re.compile("[^ACGT]+")

#: `str.translate` table for reverse-complementing an already-uppercased,
#: already-U-to-T-normalized sequence made only of A/C/G/T.
_COMPLEMENT = str.maketrans("ACGT", "TGCA")

#: `dict.get` default marking "this k-mer is not in the assembly at all",
#: which `None` and `0` cannot: a caller-supplied mapping may legitimately
#: store a k-mer with a count of 0, and that k-mer is shared with the reads
#: even though it buckets as `missing`. See `_compare`.
_ABSENT = object()


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _canonical_kmers(seq: str, k: int):
    """Yields every canonical k-mer substring of `seq` (a single contig's
    bases), in order, using the same ambiguous-base-resets-the-window rule
    as `kmer.rs::extract_canonical_kmers`: a run of valid bases shorter
    than `k` produces nothing, and a base outside A/C/G/T (case-
    insensitively; U/u is treated as T, matching the Rust core) discards
    the current run instead of poisoning the next `k - 1` windows with it.

    A k-mer's canonical form is itself or its reverse complement, whichever
    is lexicographically smaller. That matches `kmer.rs::canonical_kmer_u64`
    (`kmer.min(reverse_complement)` as **u64 values**) without
    reimplementing its 2-bit packing, because the two orderings coincide
    exactly: `kmer.rs::base_to_bits` assigns A=0b00 < C=0b01 < G=0b10 <
    T=0b11, the same order as plain ASCII/lexicographic
    `'A' < 'C' < 'G' < 'T'`; and a k-mer's u64 encoding packs its *first*
    base into the *most significant* bits
    (`current_kmer = (current_kmer << 2) | bits`), so comparing two k-mers'
    u64 values numerically is the same operation, digit by digit from the
    first base onward, as comparing their ACGT strings lexicographically.
    Whichever orientation the Rust core would pick as canonical is
    therefore always the same one plain Python string comparison picks
    here -- so a k-mer string produced by this module and one produced by
    `KmerCounts.table`'s `kmer_sequence` column are directly comparable
    without ever touching the u64 encoding.

    Two things are done once per *run* rather than once per *base* or once
    per *window*, which is what makes this the module's pure-Python
    fallback rather than its bottleneck:

    * Splitting on `_INVALID_RUN` finds the valid runs in C, instead of
      testing every one of the contig's bases for membership in an ACGT
      set and tracking the run start in Python -- one pass over a 5 Mb
      contig used to be 5 million set lookups plus 5 million `enumerate`
      steps.
    * The reverse complement of the whole run is built once, and each
      window's reverse complement is then a *slice* of it. The reverse
      complement of `run[end - k:end]` is exactly
      `rc_run[length - end:length - end + k]`, because `rc_run[j]` is the
      complement of `run[length - 1 - j]`. That replaces a `str.translate`
      plus a reversal -- two length-k string builds and two Python calls --
      per window with a single slice: a 5 Mb contig at k=21 does one
      length-5,000,000 reverse complement instead of five million
      length-21 ones.
    """
    if k <= 0 or k > len(seq):
        return
    seq = seq.upper().replace("U", "T")
    for run in _INVALID_RUN.split(seq):
        length = len(run)
        if length < k:
            continue
        rc_run = _revcomp(run)
        for end in range(k, length + 1):
            forward = run[end - k : end]
            reverse = rc_run[length - end : length - end + k]
            yield forward if forward <= reverse else reverse


def _open_text(path):
    """Opens `path` in text mode, transparently decompressing if it ends
    in `.gz`."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _iter_fasta_sequences(path):
    """Yields each record's full (multi-line-joined) sequence from a FASTA
    file, uppercased headers stripped. Blank lines are ignored, matching
    every common FASTA writer's tolerance for them.
    """
    with _open_text(path) as fh:
        chunks: list[str] = []
        has_record = False
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if has_record:
                    yield "".join(chunks)
                chunks = []
                has_record = True
            else:
                chunks.append(line)
        if has_record:
            yield "".join(chunks)


def _count_fasta_kmers(path, k: int) -> Counter:
    """The pure-Python fallback (module docstring, path 1): builds a
    `Counter` mapping each canonical k-mer found anywhere in `path`'s
    contigs to how many times it occurs across the whole assembly
    (across and within contigs -- a k-mer inside a collapsed repeat or
    duplicated region legitimately occurs more than once). See the module
    docstring for why this is a real performance limitation, not a
    corner cut for convenience.
    """
    counts: Counter = Counter()
    for contig_seq in _iter_fasta_sequences(path):
        counts.update(_canonical_kmers(contig_seq, k))
    return counts


def _is_fastq_path(path) -> bool:
    lower = str(path).lower()
    return lower.endswith(_FASTQ_EXTENSIONS)


def _assembly_kmer_counts(assembly_path, k: int) -> Counter:
    """Dispatches on `assembly_path`'s extension between the two paths
    described in the module docstring."""
    if _is_fastq_path(assembly_path):
        # Already FASTQ (a caller-side workaround, e.g. a dummy quality
        # string appended to each contig) -- use the real, fast Rust path.
        # `min_quality=0.0`: the quality string here is fabricated (an
        # assembly has no per-base quality of its own), so quality-based
        # trimming would silently depend on whatever filler character the
        # caller happened to pick rather than on anything real -- disabled
        # here to keep this path's behavior equivalent to the pure-FASTA
        # path, which has no quality signal to trim on at all.
        counted = fastdna.count(assembly_path, k=k, min_count=1, min_quality=0.0)
        table = counted.table
        return Counter(
            dict(zip(table.column("kmer_sequence").to_pylist(), table.column("frequency").to_pylist()))
        )
    return _count_fasta_kmers(assembly_path, k)


def _qv_from_counts(k_shared: int, k_total: int, k: int) -> tuple[float, float]:
    """The Merqury QV formula (see module docstring's "Confidence note").

    `k_shared`/`k_total` are counts of **distinct** k-mers (not weighted
    by how often each occurs in the assembly): `k_total` is the number of
    distinct canonical k-mers in the assembly's own k-mer set, and
    `k_shared` is how many of those also occur (at any frequency >= 1) in
    the reads' k-mer set.

    Returns `(error_rate, qv)`:
      * `error_rate` -- the estimated per-base probability of error,
        `1 - (k_shared / k_total) ** (1 / k)`. Derivation: an assembly
        k-mer absent from the reads is assumed to contain at least one
        base error (that is the whole premise of a reference-free
        comparison against reads); if a single base is correct with
        probability `1 - error_rate`, a whole, error-free k-mer spanning
        `k` such bases occurs with probability `(1 - error_rate) ** k`
        (independence assumption) -- so `k_shared / k_total`, the
        observed fraction of assembly k-mers that *are* found in the
        reads, is the sample estimate of `(1 - error_rate) ** k`, which
        this formula inverts.
      * `qv` -- the standard Phred transform, `-10 * log10(error_rate)`.
        `float('inf')` when `error_rate == 0.0` (every assembly k-mer was
        found in the reads -- a perfect score under this metric, not a
        bug: `-10 * log10(0)` diverges, and Merqury's own tooling reports
        this case as infinite/undefined rather than an arbitrary large
        number).

    Raises `ValueError` if `k_total == 0` (an assembly with no valid
    k-mers at all -- e.g. shorter than `k`, or empty -- for which no QV
    estimate is meaningful).
    """
    if k_total == 0:
        raise ValueError(
            "assembly contains no valid k-mers (empty, or every contig is shorter than k); cannot estimate QV"
        )
    ratio = k_shared / k_total
    p_correct = ratio ** (1.0 / k)
    error_rate = 1.0 - p_correct
    if error_rate <= 0.0:
        return 0.0, float("inf")
    return error_rate, -10.0 * math.log10(error_rate)


@dataclass
class AssemblyQC:
    """Result of :func:`evaluate_assembly`/:func:`evaluate_kmers`. See each
    field's own description; all "assembly k-mer" and "read k-mer" counts
    below are counts of **distinct** canonical k-mers, not weighted by how
    often a k-mer occurs in the assembly or how deeply a read k-mer is
    covered (except where a field's name says "reliable", "shared", etc.
    explicitly refers to a subset chosen by frequency -- see below).
    """

    #: k-mer size these k-mer sets were built/compared with.
    k: int

    #: The Phred-like per-base assembly accuracy estimate (dimensionless,
    #: `>= 0`; `float('inf')` means "no assembly k-mer was found missing
    #: from the reads"). See `_qv_from_counts` for the exact derivation.
    #: Higher is better -- Merqury's own convention (like Illumina's own
    #: Phred quality scores this mirrors).
    qv: float

    #: The per-base error-rate estimate `qv` is the Phred transform of, as
    #: a plain probability in `[0, 1]` (not percent). `error_rate == 0.0`
    #: corresponds to `qv == inf`.
    error_rate: float

    #: Fraction, in `[0, 1]`, of the reads' own *reliable* k-mers (see
    #: `min_count_used`) that occur anywhere in the assembly (at any
    #: frequency `>= 1`). `float('nan')` if the reads contributed zero
    #: reliable k-mers at the threshold used (nothing to measure recovery
    #: of). Higher is better: 1.0 means every real, reliably-observed
    #: piece of genome the reads support was recovered somewhere in the
    #: assembly.
    completeness: float

    #: The `min_count` threshold actually used to decide which read
    #: k-mers count as "reliable" ground truth versus likely sequencing
    #: errors for `completeness` -- either the caller's own explicit
    #: `min_count`, or (when `None` was passed) whatever
    #: `KmerCounts.suggest_min_count()` found from the reads' own
    #: frequency spectrum. Recorded here so a caller who did not pass an
    #: explicit threshold can see which one was actually used.
    min_count_used: int

    #: Number of distinct canonical k-mers in the assembly's own k-mer set
    #: (`K_total` in the module docstring's QV derivation).
    assembly_distinct_kmers: int

    #: Of `assembly_distinct_kmers`, how many were also found (at any
    #: frequency `>= 1`) in the reads' k-mer set (`K_shared`).
    assembly_kmers_found_in_reads: int

    #: Per-depth breakdown of how the reads' own k-mers map onto the
    #: assembly -- a simplified version of Merqury's "spectra-cn" plot,
    #: as a `pyarrow.Table` with columns:
    #:   * `depth` -- a frequency bucket from the reads' spectrum
    #:     (`KmerCounts.spectrum()`'s keys).
    #:   * `total` -- how many distinct read k-mers occur at exactly that
    #:     read depth (matches `KmerCounts.spectrum()[depth]`).
    #:   * `missing_in_assembly` -- of those, how many occur zero times in
    #:     the assembly (0 = a real k-mer the reads support that the
    #:     assembly does not contain at all -- a potential gap/deletion).
    #:   * `single_in_assembly` -- how many occur exactly once in the
    #:     assembly (the expected count for a well-assembled haploid
    #:     region).
    #:   * `multi_in_assembly` -- how many occur more than once in the
    #:     assembly (a potential erroneous duplication, or a collapsed
    #:     repeat resolved by the reads but not the assembly).
    #: This is data for a caller to plot, not a plot itself.
    spectra: pa.Table = field(repr=False)

    def __repr__(self):
        qv_str = "inf" if math.isinf(self.qv) else f"{self.qv:.2f}"
        comp_str = "nan" if isinstance(self.completeness, float) and math.isnan(self.completeness) else f"{self.completeness:.4f}"
        return f"AssemblyQC(k={self.k}, qv={qv_str}, completeness={comp_str}, min_count_used={self.min_count_used})"


def evaluate_kmers(
    assembly_kmers,
    reads_counts,
    *,
    k: int,
    min_count: int | None = None,
) -> AssemblyQC:
    """Lower-level entry point: compares an already-computed assembly
    k-mer multiset against an already-computed `fastdna.KmerCounts` for
    the reads, without touching any file itself.

    `assembly_kmers` is either a `Mapping[str, int]` from canonical k-mer
    string to how many times it occurs in the assembly (as produced by
    `_count_fasta_kmers`/`evaluate_assembly`'s own internals), or any
    iterable of canonical k-mer strings (each occurrence listed
    separately -- e.g. `["ACGT", "ACGT", "CCGA"]` for a k-mer occurring
    twice), which is turned into a `collections.Counter` first. Every
    k-mer must already be in the same canonical form `fastdna` itself
    uses (see `_canonical`/`KmerCounts.table`'s `kmer_sequence` column) --
    this function does not re-canonicalize.

    `reads_counts` is a `fastdna.KmerCounts` for the reads, built with the
    *same* `k` (mismatched `k` values make the comparison meaningless --
    two k-mer sets built at different `k` share almost nothing by
    construction, not because of any real assembly/read discrepancy).

    `min_count`: see `evaluate_assembly`.

    See `AssemblyQC` for the returned fields, and the module docstring for
    the QV formula and its provenance.
    """
    if not isinstance(assembly_kmers, Mapping):
        assembly_kmers = Counter(assembly_kmers)

    reads_table = reads_counts.table
    all_seqs = reads_table.column("kmer_sequence").to_pylist()
    all_freqs = reads_table.column("frequency").to_pylist()

    k_total = len(assembly_kmers)
    min_count_used = min_count if min_count is not None else reads_counts.suggest_min_count()

    # One pass over the reads' k-mer table, one hash lookup per row. The
    # version this replaced walked the same rows twice -- once for
    # completeness, once for the spectrum buckets -- with a lookup in each,
    # and separately built a `set` of every read k-mer only to count how
    # many assembly k-mers it contained. Against R read k-mers and A
    # assembly k-mers that was 2R iterations and 3R + A hash operations;
    # this is R and R. For a 30-million-k-mer read set the difference is
    # 60 million Python-level operations.
    #
    # `k_shared` is counted from this side instead: both `assembly_kmers`
    # and the reads' table hold each canonical k-mer exactly once, so
    # "assembly k-mers also seen in the reads" and "read k-mers also seen
    # in the assembly" are the same intersection and the same number. The
    # `_ABSENT` sentinel keeps that distinct from an assembly k-mer stored
    # with an explicit count of 0, which is *shared* but still buckets as
    # `missing`, exactly as before.
    lookup = assembly_kmers.get
    k_shared = 0
    reliable_total = 0
    reliable_found = 0
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: {"missing": 0, "single": 0, "multi": 0})
    for seq, freq in zip(all_seqs, all_freqs):
        asm_count = lookup(seq, _ABSENT)
        shared = asm_count is not _ABSENT
        if shared:
            k_shared += 1
        if freq >= min_count_used:
            reliable_total += 1
            if shared:
                reliable_found += 1
        if not shared or asm_count == 0:
            buckets[freq]["missing"] += 1
        elif asm_count == 1:
            buckets[freq]["single"] += 1
        else:
            buckets[freq]["multi"] += 1

    error_rate, qv = _qv_from_counts(k_shared, k_total, k)
    completeness = (reliable_found / reliable_total) if reliable_total > 0 else float("nan")

    depths = sorted(buckets)
    spectra = pa.table(
        {
            "depth": depths,
            "total": [buckets[d]["missing"] + buckets[d]["single"] + buckets[d]["multi"] for d in depths],
            "missing_in_assembly": [buckets[d]["missing"] for d in depths],
            "single_in_assembly": [buckets[d]["single"] for d in depths],
            "multi_in_assembly": [buckets[d]["multi"] for d in depths],
        }
    )

    return AssemblyQC(
        k=k,
        qv=qv,
        error_rate=error_rate,
        completeness=completeness,
        min_count_used=min_count_used,
        assembly_distinct_kmers=k_total,
        assembly_kmers_found_in_reads=k_shared,
        spectra=spectra,
    )


def evaluate_assembly(
    assembly_path,
    reads_path,
    *,
    k: int = 21,
    min_count: int | None = None,
    min_quality: float = 0.0,
) -> AssemblyQC:
    """Merqury-style reference-free assembly quality assessment (Rhie et
    al. 2020 -- see this module's own docstring for the full citation and
    the FASTA-handling discussion).

    `assembly_path`: the assembly to grade. Normally a FASTA file
    (`.fasta`/`.fa`/`.fna`, optionally `.gz`), parsed and k-mer-counted in
    pure Python since the Rust core has no FASTA support (see the module
    docstring -- **this is measurably slower** than `fastdna.count()`'s
    Rust path; expect it to dominate this function's runtime). If
    `assembly_path` instead has a FASTQ-like extension
    (`.fastq`/`.fq`(`.gz`)) -- i.e. the caller already converted their
    assembly to FASTQ with a dummy quality string, a common workaround --
    it is routed through the real, fast `fastdna.count()` path instead.

    `reads_path`: the raw FASTQ(.gz) reads the assembly was built from --
    genuinely FASTQ, so this side always goes through `fastdna.count()`
    at full speed. These reads are the ground truth this function grades
    the assembly against; an assembly built from a *different* read set
    than the one passed here is not a meaningful comparison.

    `k`: k-mer size for both sides. Defaults to `21` (not `fastdna.count`'s
    own `k=31` default), matching Merqury's own recommendation and this
    package's `fastdna.sketch`'s default -- `k=21` keeps the chance of two
    *unrelated* k-mers colliding by chance low while staying short enough
    that ordinary sequencing-error rates do not make every read k-mer
    unique. Both sides are always counted at the same `k`; there is no way
    to compare k-mer sets built at different `k` meaningfully.

    `min_count`: which read k-mers count as "reliable" ground truth for
    `completeness`, versus likely sequencing errors, expressed as a
    minimum observed frequency. If `None` (the default), uses the reads'
    own `KmerCounts.suggest_min_count()` -- the valley between the error
    peak and the true-coverage peak in *this* sample's own frequency
    spectrum -- rather than a hardcoded constant, per this package's own
    design position (see `fastdna.spectrum.suggest_min_count`'s docstring)
    that there is no single correct universal `min_count`.

    `min_quality`: the minimum per-base Phred quality `reads_path` is
    counted at, passed straight through to `fastdna.count`. Defaults to
    `0.0` here -- **not** `fastdna.count`'s own default of `20.0` -- which
    is a deliberate divergence, not an oversight: `fastdna.count(...,
    min_quality=20.0)` 3'-end-trims every read before counting, so any
    read k-mer that existed only in a trimmed tail silently vanishes from
    the ground truth this function grades the assembly against, and every
    assembly k-mer that depended on it is then scored as a *consensus
    error* that was never really there. Merqury itself builds its read
    k-mer database from the reads exactly as given (`meryl count` on the
    raw FASTQ, no quality trimming) -- this default matches that, and
    keeps an assembly's QV a property of the assembly, not of how sharply
    the caller's sequencer's quality happened to decay toward the read's
    3' end (ordinary, harmless decay on an otherwise-correct read
    dropped this module's own worked example's QV from `inf` to `25.00`
    with zero real assembly errors). Pass `min_quality=20.0` (or any other
    value) explicitly to opt back into `fastdna.count`'s own trimming
    behavior if that is genuinely wanted.

    Returns an `AssemblyQC` (see its own docstring for every field's exact
    meaning and units): `qv` (float, Phred-like, higher is better),
    `completeness` (float in `[0, 1]`, higher is better), and `spectra` (a
    `pyarrow.Table` a caller can plot).
    """
    reads_counts = fastdna.count(reads_path, k=k, min_quality=min_quality)
    assembly_kmers = _assembly_kmer_counts(assembly_path, k)
    return evaluate_kmers(assembly_kmers, reads_counts, k=k, min_count=min_count)
