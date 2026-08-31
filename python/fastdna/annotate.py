"""fastdna.annotate -- mapping a k-mer sequence (e.g. the causal literal of a
learned `fastdna.rules.SetCoveringClassifier` rule) back to the annotated
gene/feature it falls inside, in a user-supplied reference genome.

## Why this exists

`SetCoveringClassifier.explain()` already prints something a biologist can
paste into BLAST -- a literal DNA sequence, not an opaque embedding
dimension (see `rules.py`'s module docstring). But "resistant IF
present(ACGTACGT...GAT)" is not yet the answer a clinical microbiologist
wants. "This rule's k-mer falls inside *gyrA* (fluoroquinolone resistance),
position 247" is. This module closes exactly that gap, and only that gap:
it is a **lookup against the one or few reference sequences the caller
already has**, not a search against a large external database.

## Scope, stated as boundaries, not just features

- **Substitutions only, no indels.** A k-mer is fixed-length by
  construction (`src/kmer.rs`'s `extract_canonical_kmers`: every emitted
  k-mer is exactly `k` bases). Comparing two fixed-length strings can only
  ever express substitutions -- there is no "the reference has one extra
  base here" for a Hamming distance to represent. Tolerating indels would
  mean aligning two sequences of *different* lengths, which is a genuinely
  different, harder problem (gapped alignment) that this module does not
  attempt. `locate_kmer(..., max_mismatches=2)` will therefore never find a
  match that requires an insertion or deletion, even a single-base one, no
  matter how large `max_mismatches` is set -- see
  `test_max_mismatches_does_not_find_a_match_that_requires_an_indel` in the
  test suite for a concrete example.
- **No suffix array / FM-index / minimap2-BWA-class ambition.** Considered
  and deliberately rejected: those structures earn their keep at whole-
  database scale (many references, or one reference searched by millions
  of reads). This module's stated scale is "a handful of reference
  sequences, typically one genome, occasionally a handful of contigs",
  searched by a handful of k-mers (one per rule literal). At that scale a
  plain hash index of every k-length window of the reference -- built once
  per `load_annotation()` call and reused by every later `locate_kmer()`
  call at the same `k` -- gives O(1) amortized exact lookups for O(n) build
  cost and O(n) memory (`n` = total reference length), which is the whole
  budget an FM-index would also need just to build, for a workload that
  never asks it to. Building suffix-array machinery here would be solving
  a database-search problem nobody asked this module to solve.
- **Approximate search (`max_mismatches > 0`) is a direct scan, not an
  indexed one.** A substitution changes the string, so it cannot be found
  by hashing the *exact* window text -- there is nothing keyed by "this
  string, off by up to `m` characters" to look up. The approximate path
  instead slides a length-`k` window across each reference sequence and
  computes a Hamming distance directly: O(reference length * k) per call.
  That is linear in the input, not indexed, but at genome scale (single-
  digit to low-double-digit megabases for the bacterial/viral genomes this
  module targets) it is a small, one-off cost paid only when a caller
  actually asks for fuzzy matching -- not the common case, and not worth
  building (and invalidating on every new `k`) a second index for.

## Multi-hit and no-hit are both real answers

A k-mer can occur zero times, once, or many times (repeats, multi-copy
genes, plasmid duplicates) in a reference -- `locate_kmer()` returns
**every** occurrence, on **both** strands searched independently, never
just the first. A k-mer whose literal sequence and reverse complement both
happen to occur (including the degenerate case of a palindromic k-mer
matching the same position in both orientations) gets one `Hit` per
orientation, not one collapsed record: `fastdna`'s own k-mers are
*canonical* (`src/kmer.rs::canonical_kmer_u64` -- the lexicographic minimum
of the k-mer and its reverse complement, an encoding-level, strand-agnostic
choice), so the sequence a caller hands this module could correspond to
either strand of the real reference, and reporting only one orientation
would silently misrepresent which physical strand the evidence sits on.
A hit that lands entirely outside every annotated feature is reported with
`feature_type="intergenic"` rather than being dropped -- "no gene here" is
exactly as informative as "inside *gyrA*".

## File formats and dependency

`load_annotation()` reads the reference sequence itself from a plain FASTA
file with a small hand-written parser (no third-party dependency needed for
a strict `>header` / sequence-lines format -- the same reasoning
`fastdna.assembly_qc` already applies to its own FASTA reader). The
annotation file is one of:

- **GFF3** (`.gff`/`.gff3`), parsed here directly against the GFF3
  Specification, The Sequence Ontology Project, version 1.26 (18 August
  2020): https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
  -- nine tab-separated columns, `key=value` attributes separated by `;`,
  percent-encoded per the spec. Biopython ships no GFF3 reader of its own
  (that support lives in the separate `bcbio-gff` package, which is not a
  dependency of this project and is not added here); the format itself is
  a simple, fully-specified flat text format, so a direct parser is a
  bounded, precisely-scoped piece of code, not the kind of "reimplementing
  a whole parsing library" this project avoids elsewhere.
- **GenBank** (`.gb`/`.gbk`/`.gbff`/`.genbank`), parsed via Biopython's
  `Bio.SeqIO.parse(..., "genbank")`, which already handles the format's
  real complexity (joined/complemented locations, multi-line qualifiers,
  etc.) correctly -- reimplementing that here would be exactly the
  "hand-rolled parser standing in for a library" this project's
  conventions (see `interop.py`) avoid. `Bio` is imported lazily, only
  inside the GenBank code path, so `import fastdna.annotate` and the GFF3
  path never require Biopython -- matching `fastdna.embed`'s lazy-import-
  with-actionable-`ImportError` convention (`embed.py::_missing_dependency`)
  for an optional heavy dependency.
"""
from __future__ import annotations

import os
import pathlib
from collections import defaultdict
from typing import Any, NamedTuple, Optional, Union

import pyarrow as pa

from . import _core

__all__ = [
    "Annotation",
    "Feature",
    "Hit",
    "load_annotation",
    "locate_kmer",
    "annotate_rule",
    "export_bed",
]

_VALID_BASES = frozenset("ACGTN")
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

_GFF_EXTENSIONS = (".gff", ".gff3")
_GENBANK_EXTENSIONS = (".gb", ".gbk", ".gbff", ".genbank")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


class Feature(NamedTuple):
    """One annotated feature (gene, CDS, ...), 1-based inclusive
    coordinates -- the convention GFF3 uses on the wire and GenBank uses in
    its human-readable `21..50` locations, so `Hit`'s own coordinates (see
    below) can be compared to a feature's directly without an off-by-one
    translation at every call site.
    """

    seqid: str
    feature_type: str
    start: int
    end: int
    strand: str  # '+', '-', or '.' (GFF3's "not stranded / not relevant")
    name: str  # best-effort display name; "" if the record carries none


class Hit(NamedTuple):
    """One occurrence of a queried k-mer at one genomic position, in one
    orientation, inside (or outside) one overlapping feature.

    A single genomic position/orientation match produces **one `Hit` per
    overlapping feature** -- so a position covered by both a `gene` and a
    nested `CDS` (routine in GFF3 exports) yields two `Hit`s sharing the
    same `seqid`/`start`/`end`/`strand`/`mismatches`. A match with **no**
    overlapping feature at all yields exactly one `Hit` with
    `feature_type="intergenic"` and every `feature_*`/`gene_name` field
    `None` -- never zero `Hit`s, so an intergenic match is never
    indistinguishable from "not searched".
    """

    seqid: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    strand: str  # '+' if the query sequence itself matched, '-' if its
    # reverse complement did (see module docstring)
    mismatches: int
    feature_type: str  # a GFF3/GenBank feature type, or "intergenic"
    gene_name: Optional[str]
    feature_start: Optional[int]
    feature_end: Optional[int]
    feature_strand: Optional[str]


def _read_fasta(path: str) -> dict:
    """A small, dependency-free FASTA reader: `{record_id: uppercased
    sequence}`. `record_id` is the header's first whitespace-delimited
    token, matching the samtools/Biopython convention (the rest of the
    header line, if any, is a free-text description this module has no use
    for).
    """
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError as e:
        raise _core.IoNotFoundError(
            f"load_annotation() could not find the reference FASTA file {path!r}."
        ) from e

    sequences: dict[str, str] = {}
    current_id = None
    chunks: list[str] = []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks).upper()
                header = line[1:].split(maxsplit=1)
                current_id = header[0] if header else ""
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(chunks).upper()

    if not sequences:
        raise _core.InvalidConfigError(f"reference FASTA file {path!r} contains no sequences.")
    return sequences


def _gff3_attribute_name(attributes_field: str) -> str:
    """The best-effort display name for one GFF3 attributes column: `Name`,
    falling back to `gene`, `locus_tag`, then `ID`, then `""` -- the same
    fallback order GenBank parsing below uses for its own qualifiers, so a
    caller sees a consistent notion of "the gene's name" regardless of
    which format the annotation came in.
    """
    from urllib.parse import unquote

    parsed: dict[str, str] = {}
    for pair in attributes_field.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        parsed[unquote(key.strip())] = unquote(value.strip())

    for key in ("Name", "gene", "locus_tag", "ID"):
        if parsed.get(key):
            return parsed[key]
    return ""


def _parse_gff3(path: str) -> list:
    """Parses a GFF3 file's feature lines directly, per the GFF3
    Specification v1.26 (see module docstring): nine tab-separated columns,
    1-based inclusive `start`/`end`, `key=value;key=value` attributes.
    Comment lines (`#...`, including the `##gff-version` pragma) and blank
    lines are skipped; a `##FASTA` section (inline sequence appended at the
    end of the file) is not read -- this module always takes the reference
    sequence from `load_annotation()`'s separate FASTA argument, never from
    the annotation file itself, so anything after a `>` line is ignored.
    """
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError as e:
        raise _core.IoNotFoundError(
            f"load_annotation() could not find the GFF3 annotation file {path!r}."
        ) from e

    features = []
    with fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                break  # an inline ##FASTA section; nothing past here is a feature line

            fields = line.split("\t")
            if len(fields) != 9:
                raise _core.InvalidConfigError(
                    f"malformed GFF3 file {path!r} at line {line_no}: expected 9 "
                    f"tab-separated columns (GFF3 spec v1.26), got {len(fields)}: {line!r}"
                )
            seqid, _source, feature_type, start, end, _score, strand, _phase, attributes = fields
            try:
                start_i, end_i = int(start), int(end)
            except ValueError:
                raise _core.InvalidConfigError(
                    f"malformed GFF3 file {path!r} at line {line_no}: start/end must be "
                    f"integers, got {start!r}/{end!r}"
                )
            name = _gff3_attribute_name(attributes)
            strand_normalized = strand if strand in ("+", "-") else "."
            features.append(
                Feature(seqid, feature_type, start_i, end_i, strand_normalized, name)
            )
    return features


def _genbank_feature_name(qualifiers: dict) -> str:
    for key in ("gene", "locus_tag", "product", "label"):
        values = qualifiers.get(key)
        if values:
            return values[0]
    return ""


def _parse_genbank(path: str) -> list:
    """Parses a GenBank flat file's features via Biopython's own parser --
    see the module docstring for why this format is delegated instead of
    hand-rolled. Imported lazily so a bare `import fastdna.annotate` never
    requires Biopython; only calling this (via `load_annotation()` on a
    `.gb`/`.gbk`/`.gbff`/`.genbank` file) does.
    """
    try:
        from Bio import SeqIO
    except ImportError as e:
        raise ImportError(
            "load_annotation() needs the optional 'biopython' package to read a GenBank "
            "annotation file (.gb/.gbk/.gbff/.genbank). Install it with: pip install biopython"
        ) from e

    try:
        records = list(SeqIO.parse(path, "genbank"))
    except FileNotFoundError as e:
        raise _core.IoNotFoundError(
            f"load_annotation() could not find the GenBank annotation file {path!r}."
        ) from e

    if not records:
        raise _core.InvalidConfigError(f"GenBank annotation file {path!r} contains no records.")

    features = []
    for record in records:
        for feat in record.features:
            if feat.type == "source":
                continue
            # Biopython locations are 0-based half-open (Python slicing
            # convention); +1 on start converts to this module's 1-based
            # inclusive convention, matching GFF3 and Feature's own contract.
            start_i = int(feat.location.start) + 1
            end_i = int(feat.location.end)
            strand_value = feat.location.strand
            strand = "+" if strand_value == 1 else "-" if strand_value == -1 else "."
            name = _genbank_feature_name(feat.qualifiers)
            features.append(Feature(record.id, feat.type, start_i, end_i, strand, name))
    return features


class Annotation:
    """One or a few reference sequences plus their annotated features, as
    returned by `load_annotation()`. Holds a lazily-built, per-`k`-cached
    exact-match index of every length-`k` window of every sequence -- see
    the module docstring's efficiency argument for why this specific
    structure, at this scale, is the right tradeoff.
    """

    def __init__(self, sequences: dict[str, str], features: list[Feature]) -> None:
        self.sequences = sequences
        self.features = list(features)
        self._exact_index_cache: dict[int, dict] = {}

    def _exact_index(self, k: int) -> dict:
        cached = self._exact_index_cache.get(k)
        if cached is not None:
            return cached
        index: dict = defaultdict(list)
        for seqid, sequence in self.sequences.items():
            n = len(sequence)
            if n < k:
                continue
            for i in range(n - k + 1):
                index[sequence[i : i + k]].append((seqid, i))
        self._exact_index_cache[k] = index
        return index

    def __repr__(self) -> str:
        total_length = sum(len(s) for s in self.sequences.values())
        return (
            f"Annotation(sequences={len(self.sequences)}, "
            f"total_length={total_length}, features={len(self.features)})"
        )


def load_annotation(
    reference_fasta_path: Union[str, os.PathLike], annotation_path: Union[str, os.PathLike]
) -> Annotation:
    """Loads a reference FASTA plus its GFF3 or GenBank annotation into an
    `Annotation`, ready for repeated `locate_kmer()`/`annotate_rule()` calls.

    Parameters
    ----------
    reference_fasta_path : path-like
        A plain FASTA file (one or more records) -- the sequence
        `locate_kmer()` actually searches. Always the source of truth for
        sequence content, even when `annotation_path` is a GenBank file
        that also happens to embed a sequence: using one source
        consistently means a hit's coordinates are always reported against
        the same sequence the caller supplied, never silently against
        whatever a GenBank file's own `ORIGIN` block happened to contain.
    annotation_path : path-like
        A `.gff`/`.gff3` (GFF3) or `.gb`/`.gbk`/`.gbff`/`.genbank`
        (GenBank) file. The format is detected from the extension; an
        unrecognized one raises `ValueError` naming the supported ones
        rather than guessing.

    Raises `ValueError` if none of the annotation's `seqid`s match any
    record id in the reference FASTA -- almost always a sign the two files
    describe different assemblies, or that one uses accession numbers where
    the other uses plain contig names, and finding that out from a
    silent all-intergenic result later would be a far worse experience.
    """
    reference_fasta_path = str(reference_fasta_path)
    annotation_path = str(annotation_path)

    sequences = _read_fasta(reference_fasta_path)

    suffix = pathlib.Path(annotation_path).suffix.lower()
    if suffix in _GFF_EXTENSIONS:
        features = _parse_gff3(annotation_path)
    elif suffix in _GENBANK_EXTENSIONS:
        features = _parse_genbank(annotation_path)
    else:
        raise _core.InvalidConfigError(
            f"load_annotation() does not recognize the annotation file extension "
            f"{suffix!r} of {annotation_path!r}. Supported formats: GFF3 "
            f"({', '.join(_GFF_EXTENSIONS)}) or GenBank ({', '.join(_GENBANK_EXTENSIONS)})."
        )

    feature_seqids = {f.seqid for f in features}
    if features and not (feature_seqids & set(sequences)):
        raise _core.InvalidConfigError(
            f"none of the annotation's sequence ids ({sorted(feature_seqids)}) match any "
            f"sequence id in the reference FASTA ({sorted(sequences)}). Check that the "
            "annotation file and the reference FASTA describe the same assembly, and that "
            "their sequence/contig ids use the same naming convention."
        )

    return Annotation(sequences, features)


def _validate_kmer_sequence(kmer_sequence: str) -> str:
    kmer_sequence = str(kmer_sequence).upper()
    if not kmer_sequence:
        raise _core.InvalidConfigError("kmer_sequence must be a non-empty DNA sequence.")
    invalid = sorted(set(kmer_sequence) - _VALID_BASES)
    if invalid:
        raise _core.InvalidConfigError(
            f"kmer_sequence {kmer_sequence!r} contains characters that are not valid DNA "
            f"bases (A/C/G/T/N): {invalid}."
        )
    return kmer_sequence


def _validate_max_mismatches(max_mismatches) -> int:
    if isinstance(max_mismatches, bool) or not isinstance(max_mismatches, int) or max_mismatches < 0:
        raise _core.InvalidConfigError(
            f"max_mismatches must be a non-negative integer, got {max_mismatches!r}."
        )
    return max_mismatches


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _exact_hits(annotation: Annotation, kmer_sequence: str, rc_sequence: str) -> list:
    """`[(seqid, 0-based start, strand, mismatches=0)]` via the precomputed
    hash index -- see the module docstring's efficiency argument.
    """
    index = annotation._exact_index(len(kmer_sequence))
    palindrome = kmer_sequence == rc_sequence

    raw = [(seqid, pos, "+", 0) for seqid, pos in index.get(kmer_sequence, ())]
    if palindrome:
        # The sequence reads the same both ways: every '+' occurrence is
        # simultaneously a '-' occurrence at the identical position, and
        # both must be reported (see module docstring, "Multi-hit and
        # no-hit are both real answers").
        raw += [(seqid, pos, "-", 0) for seqid, pos in index.get(kmer_sequence, ())]
    else:
        raw += [(seqid, pos, "-", 0) for seqid, pos in index.get(rc_sequence, ())]
    return raw


def _approximate_hits(
    annotation: Annotation, kmer_sequence: str, rc_sequence: str, max_mismatches: int
) -> list:
    """`[(seqid, 0-based start, strand, mismatches)]` via a direct
    Hamming-distance scan -- see the module docstring for why this path is
    not indexed.
    """
    k = len(kmer_sequence)
    palindrome = kmer_sequence == rc_sequence
    raw = []
    for seqid, sequence in annotation.sequences.items():
        n = len(sequence)
        if n < k:
            continue
        for i in range(n - k + 1):
            window = sequence[i : i + k]
            mismatches_fwd = _hamming(window, kmer_sequence)
            if mismatches_fwd <= max_mismatches:
                raw.append((seqid, i, "+", mismatches_fwd))
            if palindrome:
                if mismatches_fwd <= max_mismatches:
                    raw.append((seqid, i, "-", mismatches_fwd))
            else:
                mismatches_rc = _hamming(window, rc_sequence)
                if mismatches_rc <= max_mismatches:
                    raw.append((seqid, i, "-", mismatches_rc))
    return raw


def locate_kmer(annotation: Annotation, kmer_sequence: str, *, max_mismatches: int = 0) -> list[Hit]:
    """Every occurrence of `kmer_sequence` (or its reverse complement) in
    `annotation`'s reference sequence(s), exactly or within
    `max_mismatches` substitutions, with the annotated feature(s) each
    occurrence falls inside (or `"intergenic"` if none) -- see the module
    docstring for the full scope statement (substitutions only, no
    suffix-array-caliber index) and `Hit`'s docstring for exactly what one
    returned entry means.

    Returns **every** hit, in both orientations independently -- zero hits
    (an empty list) is itself a real, meaningful answer ("not present in
    this reference"), not an error; a repeated/multi-copy k-mer returns one
    entry per occurrence rather than picking one.

    `max_mismatches=0` (the default) uses an exact hash-index lookup;
    `max_mismatches>0` uses a direct Hamming-distance scan of the whole
    reference. Both are substitution-only: no insertion or deletion is ever
    matched, at any `max_mismatches` (see module docstring).

    Results are sorted by `(seqid, start, strand, feature_type, gene_name)`
    for deterministic output across calls.
    """
    if not isinstance(annotation, Annotation):
        raise TypeError(
            f"annotation must be a fastdna.annotate.Annotation built by load_annotation(), "
            f"got {type(annotation).__name__!r}."
        )
    kmer_sequence = _validate_kmer_sequence(kmer_sequence)
    max_mismatches = _validate_max_mismatches(max_mismatches)

    rc_sequence = _reverse_complement(kmer_sequence)
    k = len(kmer_sequence)

    if max_mismatches == 0:
        raw_hits = _exact_hits(annotation, kmer_sequence, rc_sequence)
    else:
        raw_hits = _approximate_hits(annotation, kmer_sequence, rc_sequence, max_mismatches)

    hits = []
    for seqid, pos0, strand, mismatches in raw_hits:
        start1 = pos0 + 1
        end1 = pos0 + k
        overlapping = [
            f
            for f in annotation.features
            if f.seqid == seqid and f.start <= end1 and f.end >= start1
        ]
        if not overlapping:
            hits.append(Hit(seqid, start1, end1, strand, mismatches, "intergenic", None, None, None, None))
        else:
            for feature in overlapping:
                hits.append(
                    Hit(
                        seqid,
                        start1,
                        end1,
                        strand,
                        mismatches,
                        feature.feature_type,
                        feature.name or None,
                        feature.start,
                        feature.end,
                        feature.strand,
                    )
                )

    hits.sort(key=lambda h: (h.seqid, h.start, h.strand, h.feature_type, h.gene_name or ""))
    return hits


def annotate_rule(
    rule: Any,  # fastdna.rules.Rule (duck-typed, not imported here) or a plain k-mer sequence str
    annotation: Annotation,
    *,
    max_mismatches: int = 0,
) -> pa.Table:
    """Convenience wrapper: `locate_kmer()` for a `fastdna.rules.Rule` (or a
    plain k-mer sequence string), returned as a `pyarrow.Table` -- one row
    per `Hit`, ready to attach to `SetCoveringClassifier.explain()` output
    or write out alongside it.

    `rule` may be:

    - a `fastdna.rules.Rule` (duck-typed via `.feature_name`/`.presence`/
      `.feature_index` -- this module does not import `fastdna.rules`, the
      same "no hard dependency on the producing module" stance
      `fastdna.interpret` takes towards `fastdna.sklearn`): its
      `feature_name` is used as the k-mer sequence, and `presence`/
      `feature_index` are carried into the output table as extra columns.
    - a plain string: used directly as the k-mer sequence; the output
      table's `presence`/`feature_index` columns are all-null.

    Raises `ValueError` if the resulting sequence is not valid DNA -- the
    most common cause is a `Rule` whose classifier was fitted without
    `feature_names`, so `feature_name` is a positional placeholder like
    `"feature_2"` rather than a real k-mer (see
    `SetCoveringClassifier.export_rules_fasta`, which refuses the same
    input for the same reason).

    Always returns a `pyarrow.Table` with a fixed schema, zero rows for a
    zero-hit k-mer -- so a caller can `pa.concat_tables()` results across
    several rules without special-casing "this one had no hits".
    """
    if isinstance(rule, str):
        kmer_sequence = rule
        feature_index = None
        presence = None
    elif hasattr(rule, "feature_name") and hasattr(rule, "presence") and hasattr(rule, "feature_index"):
        kmer_sequence = rule.feature_name
        feature_index = rule.feature_index
        presence = rule.presence
    else:
        raise TypeError(
            f"rule must be a fastdna.rules.Rule or a plain k-mer sequence string, got "
            f"{type(rule).__name__!r}."
        )

    invalid = sorted(set(str(kmer_sequence).upper()) - _VALID_BASES)
    if invalid:
        raise _core.InvalidConfigError(
            f"annotate_rule() needs a real DNA sequence but rule.feature_name is "
            f"{kmer_sequence!r}, which contains non-DNA characters {invalid}. If this Rule "
            "came from a SetCoveringClassifier fitted without feature_names, feature_name is "
            "a positional placeholder (e.g. 'feature_2'), not a k-mer -- refit with "
            "feature_names=vectorizer.get_feature_names_out()."
        )

    hits = locate_kmer(annotation, kmer_sequence, max_mismatches=max_mismatches)

    n = len(hits)
    return pa.table(
        {
            "kmer_sequence": pa.array([kmer_sequence.upper()] * n, type=pa.string()),
            "feature_index": pa.array([feature_index] * n, type=pa.int64()),
            "presence": pa.array([presence] * n, type=pa.bool_()),
            "seqid": pa.array([h.seqid for h in hits], type=pa.string()),
            "start": pa.array([h.start for h in hits], type=pa.int64()),
            "end": pa.array([h.end for h in hits], type=pa.int64()),
            "strand": pa.array([h.strand for h in hits], type=pa.string()),
            "mismatches": pa.array([h.mismatches for h in hits], type=pa.int64()),
            "feature_type": pa.array([h.feature_type for h in hits], type=pa.string()),
            "gene_name": pa.array([h.gene_name for h in hits], type=pa.string()),
            "feature_start": pa.array([h.feature_start for h in hits], type=pa.int64()),
            "feature_end": pa.array([h.feature_end for h in hits], type=pa.int64()),
            "feature_strand": pa.array([h.feature_strand for h in hits], type=pa.string()),
        }
    )


def export_bed(
    table: pa.Table, path: Union[str, os.PathLike], *, name_column: str = "kmer_sequence"
) -> None:
    """Writes `table` -- `annotate_rule()`'s output, or `pa.concat_tables()`
    of several such tables -- as a standard BED6 file, for loading a rule's
    genomic hit positions into IGV, the UCSC Genome Browser, or any other
    BED-reading tool.

    This is a plain genomic-interval export, not a genotype export: it
    carries positions, not a `PLINK` BED/BIM/FAM-style biallelic-SNP-per-
    shared-reference-position genotype matrix. That format assumes every
    sample was called against the same fixed set of reference coordinates,
    which contradicts this project's deliberately reference-free k-mer
    approach (see `docs/philosophy-narrow-not-broad.md`) -- a caller who
    needs that specific format is better served by pyseer's own PLINK/VCF
    export paths (`fastdna.gwas.PyseerExport`) than by pretending a handful
    of annotated k-mer hits is a population-wide genotype call.

    Coordinate conversion: `annotate_rule()`/`Hit` report 1-based inclusive
    `start`/`end` (the GFF3/GenBank convention this module uses throughout
    -- see `Feature`'s docstring). BED is 0-based, half-open
    (`[chromStart, chromEnd)`, the UCSC convention): `chromStart = start -
    1`, `chromEnd = end` (the 1-based inclusive end and the 0-based
    half-open end are numerically identical, so only `start` shifts).

    Parameters
    ----------
    table : pyarrow.Table
        Must carry `seqid`, `start`, `end`, `strand` columns -- every table
        `annotate_rule()` returns does. Zero rows writes an empty (but
        valid) BED file.
    path : path-like
        Output file path. Written as plain TSV with LF line endings
        (`newline="\\n"`), matching `SetCoveringClassifier.
        export_rules_fasta`'s convention, so a Windows run produces the
        same bytes as a Linux one.
    name_column : str, default "kmer_sequence"
        Which column of `table` becomes BED's `name` field. The default is
        always populated (every `annotate_rule()` row carries the queried
        k-mer). Pass `"gene_name"` to label features by gene instead --
        rows with a null `gene_name` (e.g. `feature_type="intergenic"`
        hits) fall back to the literal string `"intergenic"` rather than
        writing an empty BED field, which is not valid BED.

    BED's `score` field (column 5) is always written as `0`: nothing in
    `table` is a score in BED's `0-1000` sense, and inventing one here
    would be exactly the kind of fabricated threshold this project's other
    modules (`gwas.prefilter_association`, `plotting.plot_significance`)
    are written to avoid.

    Raises
    ------
    ValueError
        `table` is missing a required column, or `name_column` is not one
        of its columns.
    """
    required = ("seqid", "start", "end", "strand")
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise _core.InvalidConfigError(
            f"export_bed() needs column(s) {missing}, but table only has "
            f"{list(table.column_names)}. Pass the pyarrow.Table annotate_rule() returns "
            "(or pa.concat_tables() of several)."
        )
    if name_column not in table.column_names:
        raise _core.InvalidConfigError(
            f"export_bed() name_column={name_column!r} is not a column of table "
            f"({list(table.column_names)})."
        )

    seqids = table.column("seqid").to_pylist()
    starts = table.column("start").to_pylist()
    ends = table.column("end").to_pylist()
    strands = table.column("strand").to_pylist()
    names = table.column(name_column).to_pylist()

    with open(path, "w", newline="\n") as f:
        for seqid, start, end, strand, name in zip(seqids, starts, ends, strands, names):
            bed_name = name if name else "intergenic"
            f.write(f"{seqid}\t{start - 1}\t{end}\t{bed_name}\t0\t{strand}\n")
