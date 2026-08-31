"""fastdna.validate_generated -- plausibility checks for generated/synthetic
DNA sequences, against real k-mer spectra and statistics.

## The finding this answers

Nascimento, ... "Fundamental limitations of genomic language models for
realistic sequence generation" (bioRxiv, 2026,
https://www.biorxiv.org/content/10.64898/2026.01.17.700093v2) evaluated a
genomic foundation model (Evo 2) across prokaryotic, eukaryotic and viral
genomes and found that its generated sequences "captured local sequence
statistics, but consistently failed to preserve long-range genomic
organization, repeat composition and k-mer composition, and the
architecture of transcription-factor binding sites." A second, related
fact from the same literature: a genomic language model's usable context
is typically far shorter (Evo 2 aside, most sit around a few kb) than the
genome it is meant to model (an average bacterial genome is ~5 Mb) -- so
even a model that gets local statistics right has no mechanism to enforce
genome-scale structure.

**Generative genomic models are producing real output now** -- phage
design, CRISPR-Cas system design, synthetic genome segments -- and
checking "does this look like real DNA" today mostly means an ad hoc
script per paper. `validate_generated()` is that check, built entirely
from primitives this package already ships: `fastdna.count()`'s exact
k-mer counting, `fastdna.genomescope.profile_genome()`'s spectrum mixture
model, and `fastdna.sketch()`'s MinHash containment -- zero new counting
or sketching algorithms, per `docs/philosophy-narrow-not-broad.md`'s own
rule that a feature must be a direct extension of what this package's
Rust core already does fast and correctly.

## The three checks, and what each one can and cannot tell you

1. **K-mer composition divergence, at several `k`.** For each requested
   `k`, both `generated` and `reference` are exact-k-mer-counted (via the
   FASTQ-round-trip trick described below), turned into per-k-mer relative-
   frequency distributions, and compared with the Jensen-Shannon distance
   (Lin, "Divergence measures based on the Shannon entropy", IEEE Trans.
   Info. Theory 37(1), 1991; computed here via
   `scipy.spatial.distance.jensenshannon`, not re-derived by hand) -- a
   symmetric, bounded ([0, 1] at base-2 log) measure of how different two
   distributions are. Low `k` (the default includes 3 and 6) is
   dominated by base-composition effects -- GC content and short local
   motifs -- while high `k` (11, 21 by default) is where genuine long-range
   structure would have to show up; the gap between the two is exactly the
   failure mode the bioRxiv finding above describes ("captures local
   statistics, fails at long-range composition"), so reporting several `k`
   rather than one is the point, not an extra knob.

   **What this cannot tell you**: a low divergence at every `k` means the
   generated sequences are *compositionally* unremarkable relative to the
   reference -- it is not evidence of correct gene content, correct
   regulatory architecture, or biological function. `js_deviated`/
   `js_very_deviated` are illustrative starting thresholds, not values
   calibrated against a validation study (the same honest caveat
   `fastdna.cv.lineage_groups`'s own `distance_threshold` docstring makes
   for its default) -- inspect the raw `js_distance` column, do not just
   read the verdict label.

2. **Repeat/coverage structure, via `genomescope.profile_genome()`.**
   `fastdna.genomescope` fits a GenomeScope-style negative-binomial
   mixture to a *sequencing-coverage* spectrum (error component plus
   1x/2x/3x/4x haploid-coverage components). Nothing here is sequencing
   coverage: `generated`/`reference` are pooled into one synthetic k-mer
   depth spectrum each, where "depth" is how many times a k-mer recurs
   across the whole batch (within one sequence's own repeats, and across
   near-duplicate sequences in the batch, if any). That is a genuine
   reuse of the *same statistical shape* GenomeScope's mixture model
   describes -- a population of similar-but-not-identical sequences
   sharing k-mers at varying multiplicity is structurally the same
   histogram shape as reads at varying coverage -- but it is an analogy,
   not sequencing data, stated plainly so it is not mistaken for one.
   `repeat_structure_found` is `True` only when `profile_genome` both
   finds a coverage-like peak at all *and* the fit converges (see that
   function's own docstring for both conditions); `False` covers two
   different situations it cannot distinguish -- "this batch genuinely has
   no repeat-like structure" and "this batch is too small/heterogeneous
   for the model to separate an error-like tail from a peak" -- which is
   exactly why this function reports it for **both** `generated` and
   `reference` side by side: the same batch-size/heterogeneity limitation
   applies to both, so a `reference` that also fails to converge is the
   signal that the comparison itself is underpowered, not that the
   generated side is specifically deficient.

3. **Containment against the reference (novelty vs. memorization).** Each
   generated sequence is sketched (`fastdna.sketch()`) and its
   `containment()` against one pooled reference sketch is measured -- what
   fraction of *that generated sequence's* k-mers also appear somewhere in
   `reference` (see `Sketch.containment`'s own docstring for the
   asymmetric-containment reasoning; the same metric
   `fastdna.taxonomy.classify` uses for query-vs-reference screening).
   Very high containment (`>= high_containment`, default 0.95) flags a
   candidate near-verbatim copy of reference content -- "possible
   memorization" in the generative-model sense. Very low containment
   (`< low_containment`, default 0.10) flags a sequence sharing almost no
   k-mer content with the reference at all.

   **What this cannot tell you**: low containment is ambiguous by
   construction -- it looks identical whether the generated sequence is
   *implausible* (junk composition unrelated to any real genome) or
   *genuinely novel and biologically valid* (a real but previously-unseen
   organism/construct would also share little k-mer content with an
   unrelated reference). This function cannot and does not try to tell
   those apart; treat a low-containment flag as "needs a closer look", not
   "wrong".

## Why FASTA input goes through `fastdna.count()`, not a separate counter

`fastdna.count()` reads FASTQ (4-line records with a quality string), and
generated/reference sequences are ordinarily FASTA (no per-base quality --
there is nothing to report for either an assembled reference genome or a
model's raw sequence output). Rather than writing a second, slower,
pure-Python k-mer counter for this module the way `fastdna.assembly_qc`
had to for genuinely large assemblies, every sequence here is written to a
small temporary FASTQ with a dummy quality string (`"I" * len(seq)`,
`min_quality=0.0` so that fabricated string is never used to trim
anything) and counted through the real, fast Rust path -- the identical
"append a dummy quality string" workaround `fastdna.assembly_qc`'s own
module docstring documents and recognizes by extension. This keeps
`fastdna.count()` the *only* k-mer engine this module ever calls, per this
package's own convention (`fastdna.assembly_qc`, `fastdna.taxonomy`: reuse
the stable FFI surface, never re-implement it).

`numpy`/`scipy` are imported lazily, inside the calls that need them (the
same pattern `fastdna.genomescope` uses for its own optional
dependencies), so importing this module never requires either.
"""
from __future__ import annotations

import gzip
import math
import os
import tempfile
from dataclasses import dataclass

import pyarrow as pa

import fastdna
from . import _core
from .genomescope import profile_genome

__all__ = ["GenerativeValidationReport", "validate_generated"]

#: Default k-mer sizes checked by the composition test: two "local" scales
#: (dominated by base composition -- GC content and short motifs) and two
#: "long-range" scales (where the bioRxiv-2026 finding says genomic
#: language models tend to fail). Matches the scales used in
#: `docs/audit/ml-gaps.md`'s own worked example for this feature.
DEFAULT_K = (3, 6, 11, 21)

#: Jensen-Shannon *distance* (not divergence -- see the module docstring)
#: bands, in [0, 1] at base-2 log. Illustrative starting points, not
#: calibrated cutoffs; see the module docstring's check #1.
DEFAULT_JS_DEVIATED = 0.10
DEFAULT_JS_VERY_DEVIATED = 0.30

#: Containment bands for the novelty/memorization check (see check #3).
DEFAULT_HIGH_CONTAINMENT = 0.95
DEFAULT_LOW_CONTAINMENT = 0.10

_FASTQ_EXTENSIONS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")


def _missing_dependency(package):
    return ImportError(
        f"fastdna.validate_generated requires the '{package}' package, which is not "
        f"installed. Install it with `pip install {package}` and try again."
    )


def _numpy():
    try:
        import numpy
    except ImportError:
        raise _missing_dependency("numpy") from None
    return numpy


def _scipy_jensenshannon():
    try:
        from scipy.spatial.distance import jensenshannon
    except ImportError:
        raise _missing_dependency("scipy") from None
    return jensenshannon


def _open_text(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def _iter_fasta_sequences(path):
    """Yields each record's full (multi-line-joined) sequence from a FASTA
    file. A small, self-contained reader (not imported from
    `fastdna.assembly_qc`, which has its own equivalent for a different
    purpose) so this module has no coupling to that one's internals.
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


def _iter_fastq_sequences(path):
    """Yields just the sequence line of every FASTQ record (the quality
    line is not read at all -- this module discards real quality
    information exactly as it fabricates fake quality for FASTA input,
    since every check here is about k-mer composition, not base-call
    confidence).
    """
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                yield line.strip()


def _sequences_from_source(source, label):
    """Resolves `generated`/`reference` into a list of raw ACGT sequence
    strings.

    Accepts a path to a FASTA(.gz) or FASTQ(.gz) file (each record's
    sequence becomes one entry), or any iterable of sequence strings --
    the latter so a caller holding a generative model's output already in
    memory (the ordinary case -- nothing about calling an Evo2-style model
    writes a FASTA file first) does not have to round-trip it through disk
    just to call this function.
    """
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if path.lower().endswith(_FASTQ_EXTENSIONS):
            sequences = list(_iter_fastq_sequences(path))
        else:
            sequences = list(_iter_fasta_sequences(path))
    else:
        sequences = [str(s) for s in source]

    sequences = [s for s in sequences if s]
    if not sequences:
        raise _core.InvalidConfigError(f"{label} contains no sequences to validate")
    return sequences


def _count_sequences(sequences, k):
    """Counts canonical k-mers across `sequences` via `fastdna.count()`'s
    real Rust path -- see the module docstring's "Why FASTA input goes
    through fastdna.count()" section for the dummy-quality-FASTQ trick
    this wraps.
    """
    fd, temp_path = tempfile.mkstemp(suffix=".fastq")
    try:
        with os.fdopen(fd, "w") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f"@seq{i}\n{seq}\n+\n{'I' * len(seq)}\n")
        return fastdna.count(temp_path, k=k, min_count=1, min_quality=0.0)
    finally:
        os.unlink(temp_path)


def _kmer_probability_vector(counts, support, np):
    """`counts` (a `fastdna.KmerCounts`) projected onto `support` (a sorted
    array of every `kmer_u64` value observed on either side of a
    comparison), as a probability vector -- zero for any `support` entry
    `counts` never observed.
    """
    table = counts.table
    kmers = np.asarray(table.column("kmer_u64"))
    freq = np.asarray(table.column("frequency"), dtype=np.float64)
    vec = np.zeros(support.shape, dtype=np.float64)
    vec[np.searchsorted(support, kmers)] = freq
    total = vec.sum()
    return vec / total if total > 0 else vec


def _composition_distance(generated_counts, reference_counts):
    """Jensen-Shannon distance between `generated_counts`' and
    `reference_counts`' per-k-mer relative-frequency distributions. `nan`
    if either side has zero total k-mer mass at this `k` (too few
    sequences, or `k` longer than every sequence) -- there is no
    distribution to compare in that case, and reporting a numeric distance
    computed from an empty one would be a silent wrong answer.
    """
    np = _numpy()
    jensenshannon = _scipy_jensenshannon()

    table_g, table_r = generated_counts.table, reference_counts.table
    support = np.union1d(
        np.asarray(table_g.column("kmer_u64")), np.asarray(table_r.column("kmer_u64"))
    )
    if support.size == 0:
        return float("nan")

    p = _kmer_probability_vector(generated_counts, support, np)
    q = _kmer_probability_vector(reference_counts, support, np)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    return float(jensenshannon(p, q, base=2))


def _composition_verdict(distance, deviated, very_deviated):
    if not math.isfinite(distance):
        return "insufficient data"
    if distance >= very_deviated:
        return "very deviated"
    if distance >= deviated:
        return "deviated"
    return "realistic"


def _repeat_structure(counts):
    """`(found, profile)`: whether `counts`' own pooled k-mer depth
    spectrum fits GenomeScope's coverage-mixture model well enough to be
    called a genuine coverage-like peak -- see the module docstring's
    check #2 for what this can and cannot mean here. `found` is `False`,
    `profile` is `None` whenever `profile_genome` refuses outright (no
    peak at all, or an empty spectrum); a poor-but-attempted fit
    (`converged=False`) also counts as `found=False`, but its `profile` is
    still returned for inspection, matching `profile_genome`'s own
    "never hide a failed fit" stance.
    """
    try:
        profile = profile_genome(counts)
    except ValueError:
        return False, None
    return bool(profile.converged), profile


def _sketch_sequences(sequences, k, sketch_size):
    fd, temp_path = tempfile.mkstemp(suffix=".fastq")
    try:
        with os.fdopen(fd, "w") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f"@seq{i}\n{seq}\n+\n{'I' * len(seq)}\n")
        return fastdna.sketch(temp_path, k=k, sketch_size=sketch_size)
    finally:
        os.unlink(temp_path)


@dataclass(frozen=True)
class GenerativeValidationReport:
    """Result of :func:`validate_generated`. See the module docstring for
    what each check can and cannot tell you -- this is a plausibility/
    sanity check over composition and containment, never a guarantee of
    biological validity.
    """

    #: Number of sequences resolved from `generated`/`reference`.
    generated_n_sequences: int
    reference_n_sequences: int

    #: Check #1. `pyarrow.Table` with columns `k`, `js_distance` (float,
    #: `nan` if uncomputable at that `k`), `verdict` (`"realistic"` /
    #: `"deviated"` / `"very deviated"` / `"insufficient data"`), one row
    #: per requested `k`, in the order given.
    composition: pa.Table

    #: Check #2, computed once at `repeat_k` (see `validate_generated`'s
    #: own docstring for how it is chosen). `True` only when a genuine
    #: coverage-like peak was found *and* the fit converged; see
    #: `_repeat_structure`'s docstring for the two situations `False`
    #: cannot distinguish between. The underlying `GenomeProfile` objects
    #: (`None` when `profile_genome` refused outright) are kept for
    #: inspection.
    generated_repeat_structure_found: bool
    reference_repeat_structure_found: bool
    repeat_k: int
    generated_profile: object
    reference_profile: object

    #: Check #3, `None` for all three fields when `check_containment=False`
    #: was passed. `mean_containment` is the mean, across every generated
    #: sequence, of that sequence's containment in the pooled reference
    #: sketch. `high_containment_count`/`low_containment_count` are how
    #: many generated sequences cleared `high_containment`/fell below
    #: `low_containment` respectively (see the module docstring for what
    #: each band means, and its ambiguity).
    mean_containment: "float | None"
    high_containment_count: "int | None"
    low_containment_count: "int | None"

    def to_markdown(self) -> str:
        lines = [
            f"# FastDNA generative validation -- {self.generated_n_sequences} generated vs "
            f"{self.reference_n_sequences} reference",
            "",
            "## K-mer composition",
            "",
            "| k | JS distance | verdict |",
            "|---:|---:|---|",
        ]
        for row in self.composition.to_pylist():
            distance = row["js_distance"]
            distance_str = "nan" if not math.isfinite(distance) else f"{distance:.3f}"
            lines.append(f"| {row['k']} | {distance_str} | {row['verdict']} |")

        lines += [
            "",
            "## Repeat / coverage structure "
            f"(k={self.repeat_k}, GenomeScope-style spectrum fit -- see module docstring)",
            "",
            f"- generated: {'found' if self.generated_repeat_structure_found else 'not found'}",
            f"- reference: {'found' if self.reference_repeat_structure_found else 'not found'}",
        ]

        if self.mean_containment is not None:
            lines += [
                "",
                "## Novelty / memorization (containment against reference)",
                "",
                f"- mean containment: {self.mean_containment:.3f}",
                f"- generated sequences with containment >= high threshold "
                f"(possible copy): {self.high_containment_count}",
                f"- generated sequences with containment < low threshold "
                f"(no anchoring): {self.low_containment_count}",
            ]

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"GenerativeValidationReport({self.generated_n_sequences} generated, "
            f"{self.reference_n_sequences} reference, "
            f"k={self.composition.column('k').to_pylist()})"
        )

    def __str__(self):
        return self.to_markdown()


def validate_generated(
    generated,
    reference,
    *,
    k=DEFAULT_K,
    repeat_k=None,
    js_deviated=DEFAULT_JS_DEVIATED,
    js_very_deviated=DEFAULT_JS_VERY_DEVIATED,
    check_containment=True,
    containment_k=21,
    containment_sketch_size=1000,
    high_containment=DEFAULT_HIGH_CONTAINMENT,
    low_containment=DEFAULT_LOW_CONTAINMENT,
) -> GenerativeValidationReport:
    """Checks whether `generated` sequences' k-mer composition, repeat
    structure and reference containment are plausible for real DNA -- see
    the module docstring for the three checks, their literature grounding,
    and -- just as important -- what each one cannot tell you. This is a
    plausibility/sanity check, not a guarantee of biological validity.

    Parameters
    ----------
    generated, reference : path or iterable of str
        Each is a path to a FASTA(.gz)/FASTQ(.gz) file (every record's
        sequence becomes one entry) or an iterable of raw ACGT sequence
        strings -- e.g. a generative model's output already held in
        memory. `reference` is "a reference cohort or reference genome"
        (per the original request this implements): one big reference
        sequence and a multi-sequence cohort of related real genomes are
        both valid inputs, with no distinction made between them here.
    k : int or sequence of int, default (3, 6, 11, 21)
        K-mer size(s) for the composition check (check #1). See the module
        docstring for why several scales, spanning local (composition-
        driven) to long-range, matter here specifically.
    repeat_k : int or None, default None
        K-mer size for the repeat/coverage-structure check (check #2).
        `None` uses `max(k)` -- the most specific scale already being
        computed for check #1, reused here rather than adding a second
        k-mer size to count from scratch when it coincides with one
        already requested.
    js_deviated, js_very_deviated : float, default 0.10, 0.30
        Jensen-Shannon distance thresholds for the `composition` table's
        `verdict` column. See the module docstring's check #1 for why
        these are illustrative, not calibrated.
    check_containment : bool, default True
        Whether to run check #3 at all. It sketches every individual
        generated sequence (one `fastdna.sketch()` call each), which for a
        very large batch of generated sequences is the most expensive part
        of this function; set `False` to skip it when only the
        composition/repeat-structure checks are wanted.
    containment_k, containment_sketch_size : int, default 21, 1000
        Forwarded to `fastdna.sketch()` for check #3. Independent of `k`
        above -- sketch-based containment and exact-count-based
        composition are different techniques answering different
        questions, so there is no requirement they share a k-mer size.
    high_containment, low_containment : float, default 0.95, 0.10
        Containment thresholds for check #3's `high_containment_count`/
        `low_containment_count`. See the module docstring for what each
        band means and, for the low band specifically, its ambiguity.

    Returns
    -------
    GenerativeValidationReport
    """
    if isinstance(k, int):
        k_values = (k,)
    else:
        k_values = tuple(int(x) for x in k)
    if not k_values:
        raise _core.InvalidConfigError("k must be a positive int or a non-empty sequence of them")
    for kv in k_values:
        if kv < 1:
            raise _core.InvalidKError(f"every k must be >= 1, got {kv}")

    generated_sequences = _sequences_from_source(generated, "generated")
    reference_sequences = _sequences_from_source(reference, "reference")

    resolved_repeat_k = repeat_k if repeat_k is not None else max(k_values)
    if resolved_repeat_k < 1:
        raise _core.InvalidKError(f"repeat_k must be >= 1, got {resolved_repeat_k}")

    # Counted once per (role, k) and cached, since repeat_k routinely
    # coincides with one of the composition k values (the default is
    # exactly that) -- avoids re-counting the same sequences at the same k
    # twice.
    counts_cache: dict = {}

    def counts_for(role, sequences, kv):
        key = (role, kv)
        if key not in counts_cache:
            counts_cache[key] = _count_sequences(sequences, kv)
        return counts_cache[key]

    rows = {"k": [], "js_distance": [], "verdict": []}
    for kv in k_values:
        generated_counts = counts_for("generated", generated_sequences, kv)
        reference_counts = counts_for("reference", reference_sequences, kv)
        distance = _composition_distance(generated_counts, reference_counts)
        rows["k"].append(kv)
        rows["js_distance"].append(distance)
        rows["verdict"].append(_composition_verdict(distance, js_deviated, js_very_deviated))
    composition = pa.table(rows)

    generated_repeat_counts = counts_for("generated", generated_sequences, resolved_repeat_k)
    reference_repeat_counts = counts_for("reference", reference_sequences, resolved_repeat_k)
    generated_found, generated_profile = _repeat_structure(generated_repeat_counts)
    reference_found, reference_profile = _repeat_structure(reference_repeat_counts)

    mean_containment = high_count = low_count = None
    if check_containment:
        np = _numpy()
        reference_sketch = _sketch_sequences(reference_sequences, containment_k, containment_sketch_size)
        containments = [
            _sketch_sequences([seq], containment_k, containment_sketch_size).containment(reference_sketch)
            for seq in generated_sequences
        ]
        containments = np.asarray(containments, dtype=np.float64)
        mean_containment = float(containments.mean())
        high_count = int((containments >= high_containment).sum())
        low_count = int((containments < low_containment).sum())

    return GenerativeValidationReport(
        generated_n_sequences=len(generated_sequences),
        reference_n_sequences=len(reference_sequences),
        composition=composition,
        generated_repeat_structure_found=generated_found,
        reference_repeat_structure_found=reference_found,
        repeat_k=resolved_repeat_k,
        generated_profile=generated_profile,
        reference_profile=reference_profile,
        mean_containment=mean_containment,
        high_containment_count=high_count,
        low_containment_count=low_count,
    )
