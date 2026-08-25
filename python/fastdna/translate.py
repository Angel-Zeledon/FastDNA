"""fastdna.translate -- DNA to protein, in any of the six reading frames.

k-mer counting answers "what sequence is present". Translation answers
"what would it make", which is the question behind gene finding, ORF
scanning, protein-level search (BLASTX/DIAMOND-style), and amino-acid k-mer
features for a classifier -- all of which start by turning nucleotides into
residues and none of which this package could do before.

The codon lookup itself lives in the Rust core (`src/translate.rs`), for
one substantive reason rather than habit: `kmer.rs` already packs a base
into two bits, so a codon is exactly six bits and a genetic code is a plain
64-entry array indexed by that integer. Translating is then one array index
per amino acid, with no string comparison and no per-codon allocation --
and the reverse-strand frames reuse the same O(1)
`reverse_complement_u64` bit trick the k-mer hot path uses, on one codon at
a time, instead of building a reverse-complemented copy of the sequence.

Genetic codes
-------------
`table` selects an NCBI `transl_table` id. Supported: **1** (Standard),
**2** (Vertebrate Mitochondrial), **4** (Mold/Protozoan/Coelenterate
Mitochondrial and Mycoplasma/Spiroplasma) and **11** (Bacterial/Archaeal/
Plant Plastid). Every codon assignment is transcribed verbatim from NCBI
Taxonomy, "The Genetic Codes"
(https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi), version 4.6 --
the same definitions GenBank's `/transl_table=` qualifier refers to. An
unsupported id raises `ValueError` naming the supported ones rather than
falling back to the standard code, which a caller would have no way to
notice.

**Table 11 gives the same proteins as table 1 here.** NCBI's two codes have
identical amino-acid rows; they differ only in which codons may *initiate*
translation. These functions translate a reading frame, they do not call
ORFs, so no start-codon rule is applied and the two tables coincide. Table
11 is still accepted by id, because writing `table=1` for a bacterial
genome reads as a mistake even when it is not one.

What happens to awkward input
-----------------------------
Three behaviours are worth knowing before trusting a result, all of them
chosen so that residue *i* of a protein always corresponds to codon *i* of
its frame:

- **A codon containing `N`, an IUPAC ambiguity code, or any other non-ACGT
  byte becomes `X`.** It is not skipped. Skipping would shorten the protein
  and shift every residue after it, so an alignment or a motif offset
  computed on the result would be wrong with nothing to point at.
- **Trailing 1-2 bases that cannot complete a codon are dropped.** A frame
  yields `(len(seq) - offset) // 3` residues. For a reverse frame the
  dropped bases are at the 5' end of the original sequence, since a reverse
  frame reads from the 3' end backwards.
- **Lowercase is accepted** (soft-masked reference sequence translates the
  same as uppercase rather than becoming a run of `X`), and **`U` is read
  as `T`**, so RNA translates without being back-transcribed first.

`to_stop`
---------
`to_stop=False` (the default) emits `*` at a stop codon and keeps going.
`to_stop=True` truncates at the first stop and excludes it, matching
Biopython's `Seq.translate(to_stop=True)`.

Translating through is the default because that is what a *scan* needs: in
a six-frame translation of raw reads or a contig there is no reason to
believe the first stop is the end of anything, and truncating there would
silently discard every downstream ORF in that frame. Use `to_stop=True`
when the input genuinely is a coding sequence and the wanted answer is its
protein product.

Passing sequences
-----------------
Every function here takes either a single `str` or an iterable of `str`,
and the return shape follows the input shape: one string in, one result
out; a list in, a list out. A bare string is **never** iterated character by
character -- that would silently translate `"ATGGCC"` as six one-character
sequences and return six empty proteins. `bytes` is rejected with a
`TypeError`, because iterating it yields integers and the failure would
otherwise surface as something unreadable deep inside. (`fastdna.sklearn`
hit this exact footgun with a bare path; it is handled explicitly here
rather than left to chance.)
"""
from __future__ import annotations

import pyarrow as pa

from . import _core

__all__ = [
    "translate",
    "translate_six_frames",
    "translate_file",
    "protein_kmers",
    "SUPPORTED_TABLES",
    "ALL_FRAMES",
]

#: NCBI `transl_table` ids this module implements. See the module docstring
#: for what each one is and why 11 coincides with 1 here.
SUPPORTED_TABLES = (1, 2, 4, 11)

#: The six reading frames, in the conventional reporting order. Negative
#: frames read the reverse-complement strand.
ALL_FRAMES = (1, 2, 3, -1, -2, -3)


def _as_sequence_list(sequences, *, argument, noun):
    """Normalizes `sequences` into `(list_of_str, was_single)`.

    `was_single` records whether the caller passed one bare string, so the
    caller of this helper can return a single result instead of a
    one-element list -- the return shape follows the input shape (see the
    module docstring).

    Rejects `bytes`/`bytearray` explicitly: both are iterable, but iterating
    them yields integers, which would fail somewhere far from the mistake.
    """
    if isinstance(sequences, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"{argument} must be str or an iterable of str, not bytes -- "
            f"iterating bytes yields integers, not {noun}. Decode it first: "
            f"{argument}.decode('ascii')"
        )
    if isinstance(sequences, str):
        return [sequences], True

    try:
        items = list(sequences)
    except TypeError as e:
        raise TypeError(
            f"{argument} must be a str or an iterable of str, got "
            f"{type(sequences).__name__}"
        ) from e

    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(
                f"{argument}[{index}] must be str, got {type(item).__name__}. "
                f"Every element must be a {noun} string; convert it first "
                f"(e.g. str(record.seq) for a Biopython SeqRecord)."
            )
    return items, False


def _default_ids(count, prefix):
    """Auto-numbered ids for callers who supplied bare sequences, matching
    the `seq0`, `seq1`, ... convention `fastdna.interop` already uses.
    """
    return [f"{prefix}{index}" for index in range(count)]


def _translate_table(ids, sequences, frames, table, to_stop):
    """The single call into the Rust core, wrapped into a `pyarrow.Table`.

    Everything public in this module funnels through here (or through
    `translate_file`, which shares the core's own row-building code), so the
    in-memory and streaming paths cannot disagree about a protein.
    """
    batch = _core.translate_sequences(
        ids=list(ids),
        sequences=list(sequences),
        frames=[int(frame) for frame in frames],
        table=int(table),
        to_stop=bool(to_stop),
    )
    return pa.Table.from_batches([batch])


def translate(sequences, *, frame=1, table=1, to_stop=False):
    """Translate DNA in one reading frame.

    Parameters
    ----------
    sequences : str or iterable of str
        A single DNA sequence, or several. A bare string is treated as one
        sequence, never iterated per character; `bytes` raises `TypeError`.
        See the module docstring for why.
    frame : int, default 1
        One of `1`, `2`, `3`, `-1`, `-2`, `-3`. Positive frames start
        `frame - 1` bases in from the 5' end; negative frames translate the
        reverse complement, starting `abs(frame) - 1` bases in from the 3'
        end. `0` is not a reading frame and raises `ValueError`.
    table : int, default 1
        NCBI `transl_table` id; one of `SUPPORTED_TABLES`. An unsupported
        id raises `ValueError` listing what is supported.
    to_stop : bool, default False
        `False` translates through stops, emitting `*`. `True` truncates at
        the first stop and excludes it. See the module docstring for which
        to use when.

    Returns
    -------
    str or list of str
        A single protein string if `sequences` was a single string;
        otherwise one protein per input sequence, in order. A sequence too
        short to fill one codon in the requested frame gives `""` -- an
        empty protein, not a dropped entry, so the output stays aligned
        with the input positionally.

    Raises
    ------
    TypeError
        `sequences` is `bytes`, or contains a non-string element.
    ValueError
        `frame` is not one of the six, or `table` is not supported.

    Examples
    --------
    >>> translate("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
    'MAIVMGR*KGAR*'
    >>> translate("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", to_stop=True)
    'MAIVMGR'
    >>> translate("ATGNNNATG")
    'MXM'
    """
    items, was_single = _as_sequence_list(sequences, argument="sequences", noun="DNA")
    result = _translate_table(
        _default_ids(len(items), "seq"), items, (frame,), table, to_stop
    )
    proteins = result.column("protein").to_pylist()
    return proteins[0] if was_single else proteins


def translate_six_frames(sequences, *, table=1, to_stop=False):
    """Translate DNA in all six reading frames at once.

    The reason to want all six: for an unannotated read or contig, nothing
    says which strand or which offset the coding sequence sits in, and
    checking one frame answers a question nobody asked. Translating a
    sequence and translating its reverse complement give the same six
    proteins as a set -- only the frame labels swap -- so a six-frame scan
    is independent of which strand the sequence happened to be recorded on.

    Parameters
    ----------
    sequences : str or iterable of str
        As in :func:`translate`.
    table : int, default 1
        NCBI `transl_table` id; one of `SUPPORTED_TABLES`.
    to_stop : bool, default False
        As in :func:`translate`, applied independently to each frame.

    Returns
    -------
    dict or list of dict
        `{frame: protein}` keyed by `1, 2, 3, -1, -2, -3`, for a single
        input string; otherwise one such dict per input sequence, in order.

    Raises
    ------
    TypeError
        `sequences` is `bytes`, or contains a non-string element.
    ValueError
        `table` is not supported.

    Examples
    --------
    >>> frames = translate_six_frames("ATGGCCATTGTAATGGGCCGCTGA")
    >>> frames[1]
    'MAIVMGR*'
    >>> sorted(frames)
    [-3, -2, -1, 1, 2, 3]
    """
    items, was_single = _as_sequence_list(sequences, argument="sequences", noun="DNA")
    result = _translate_table(
        _default_ids(len(items), "seq"), items, ALL_FRAMES, table, to_stop
    )

    frames_column = result.column("frame").to_pylist()
    proteins = result.column("protein").to_pylist()

    # Rows come back sequence-major (every frame of sequence 0, then every
    # frame of sequence 1, ...), which is the contract `proteins_schema` in
    # src/ffi.rs documents -- so a fixed-width regroup is correct here and
    # does not depend on row order within a sequence.
    width = len(ALL_FRAMES)
    grouped = [
        dict(zip(frames_column[start : start + width], proteins[start : start + width]))
        for start in range(0, len(proteins), width)
    ]
    return grouped[0] if was_single else grouped


def translate_file(path, *, frames=(1, 2, 3, -1, -2, -3), table=1):
    """Stream a FASTA or FASTQ file (optionally gzipped) through the Rust
    reader and translate every record.

    Uses the same reader the counting pipeline does, so the format is
    detected from the file's own content rather than its extension -- a
    `.fasta` file holding FASTQ, or a file with no extension at all, is read
    for what it actually contains -- and `.gz` input is decompressed
    transparently. Anything `fastdna.count()` can read, this can read.

    Unlike `count()`, the whole result is materialized: a protein is a third
    the length of its DNA, but six frames of it is twice the input size. Use
    this for genes, contigs and modest read sets rather than for a whole
    sequencing run.

    Parameters
    ----------
    path : str or os.PathLike
        The FASTA/FASTQ(.gz) file to read.
    frames : iterable of int, default (1, 2, 3, -1, -2, -3)
        Which reading frames to translate, in the order they should be
        emitted per record. Must not be empty.
    table : int, default 1
        NCBI `transl_table` id; one of `SUPPORTED_TABLES`.

    Returns
    -------
    pyarrow.Table
        Columns `sequence_id` (utf8), `frame` (int8), `protein` (utf8), one
        row per (record, frame) pair, record-major. `sequence_id` is the
        header with its `>`/`@` marker stripped and truncated at the first
        whitespace -- the accession, as BLAST and SAM define it, not the
        whole description line, so the column stays joinable against other
        tables.

        A `pyarrow.Table` rather than a wrapper class, matching
        `fastdna.compare_all` and `fastdna.interpret.top_features`: it
        already composes with pandas, polars and DuckDB, and a bespoke type
        for one function would not add anything they do not give.

    Raises
    ------
    FileNotFoundError
        No such file.
    ValueError
        `frames` is empty or contains an invalid frame, `table` is not
        supported, or the file is not parseable as FASTA/FASTQ (the message
        names the record number).

    Notes
    -----
    There is deliberately no `to_stop` option here. This function's job is a
    whole-file frame scan, and truncating each record at its first stop
    would discard exactly the downstream ORFs such a scan exists to find.
    Translate the sequences yourself with :func:`translate` if you want
    that.

    Examples
    --------
    >>> table = translate_file("genes.fasta", frames=(1,))  # doctest: +SKIP
    >>> table.column_names                                  # doctest: +SKIP
    ['sequence_id', 'frame', 'protein']
    """
    frame_list = [int(frame) for frame in frames]
    if not frame_list:
        raise ValueError(
            "frames must contain at least one reading frame; "
            f"pass e.g. frames=(1,) or frames={ALL_FRAMES} to translate all six"
        )

    batch = _core.translate_file(
        path=str(path),
        frames=frame_list,
        table=int(table),
        to_stop=False,
    )
    return pa.Table.from_batches([batch])


def protein_kmers(proteins, k=3):
    """Count k-mers of *amino acids* within each protein.

    The protein-level counterpart of `fastdna.count()`'s DNA k-mers, and a
    useful feature space in its own right: amino-acid k-mers tolerate
    synonymous substitutions that change a DNA k-mer entirely, so they stay
    informative across a much wider evolutionary distance than nucleotide
    k-mers do.

    Two differences from DNA k-mers matter enough to state rather than let
    a caller assume:

    - **These k-mers are not canonical.** A DNA k-mer and its reverse
      complement are the same physical double-stranded object, so
      `fastdna.count()` collapses them. A peptide read backwards is a
      different peptide, and has no reverse complement at all, so `"MA"`
      and `"AM"` stay distinct rows here.
    - **They do not use the 2-bit packed path.** That representation exists
      because there are four bases; there are twenty amino acids plus `*`
      and `X`, which needs five bits and would cap `k` at 12 while giving
      up the O(1) rolling window and reverse-complement tricks that justify
      packing in the first place. This is plain byte-window counting -- a
      genuinely separate code path from `src/kmer.rs`, not a reuse of it.

    Windows containing `*` (a stop) or `X` (an untranslatable codon) are
    kept. Filtering them is one line on your side; recovering them after a
    silent drop is not.

    Parameters
    ----------
    proteins : str or iterable of str
        A single protein string, or several. As in :func:`translate`, a
        bare string is one protein and is never iterated per residue;
        `bytes` raises `TypeError`.
    k : int, default 3
        K-mer length in residues. Must be at least 1. `k=3` is a common
        default for protein k-mer features; `k` larger than a protein
        simply yields no rows for it.

    Returns
    -------
    pyarrow.Table
        Columns `sequence_id` (utf8), `aa_kmer` (utf8), `count` (uint32).
        Counts are **per protein**, not pooled -- pooling is a one-line
        group-by on your side, whereas un-pooling is impossible once done
        here. `sequence_id` is auto-numbered `protein0`, `protein1`, ...
        Rows are protein-major and sorted by `aa_kmer` within each protein,
        so the table is identical across runs on the same input.

    Raises
    ------
    TypeError
        `proteins` is `bytes`, or contains a non-string element.
    ValueError
        `k` is 0, or a protein contains non-ASCII characters (which cannot
        be split into k-mers by byte).

    Examples
    --------
    >>> table = protein_kmers("MAMAM", k=2)
    >>> dict(zip(table.column("aa_kmer").to_pylist(),
    ...          table.column("count").to_pylist()))
    {'AM': 2, 'MA': 2}
    """
    items, _was_single = _as_sequence_list(proteins, argument="proteins", noun="protein")
    batch = _core.protein_kmers(
        ids=_default_ids(len(items), "protein"),
        proteins=items,
        k=int(k),
    )
    return pa.Table.from_batches([batch])
