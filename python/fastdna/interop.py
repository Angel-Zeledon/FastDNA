"""Convenience adapters for counting/sketching sequences that already live
in memory -- a Biopython `Bio.SeqIO.parse(...)` iterator of `SeqRecord`
objects, a list of plain strings, or `(id, sequence)` pairs -- instead of a
FASTQ file already on disk.

`fastdna.count()`/`fastdna.sketch()` only accept a file path: the Rust core
streams a FASTQ(.gz) file, and that streaming design is the entire reason
`count()` is fast and `sketch()` stays memory-bounded (see the README's
"How the Rust core actually works"). Bolting a second, in-memory ingestion
path onto the Rust side would mean a second code path to keep correct and
fast, for a convenience that is fundamentally about *not* having a file yet
-- not about a faster way to read one. So this module takes the honest,
simple route instead: write the given sequences out to a small temporary
FASTQ file, call the existing, already-tested `count()`/`sketch()` on it,
and clean the file up afterward.

**This is a convenience path, not a zero-copy fast path.** In particular:

- Plain strings and `(id, sequence)` pairs carry no real quality scores, so
  a uniform high-quality Phred string (`I`, Phred+33 for Q40) is
  synthesized for every base. If your in-memory data does carry real
  per-base quality (e.g. a Biopython `SeqRecord` with a
  `letter_annotations["phred_quality"]` populated from a real FASTQ read),
  it is used instead -- see `_quality_string_for` below -- but a bare
  string or a `SeqRecord` built from a FASTA file has no such information
  to use, and no such information is invented beyond the uniform default.
  Quality-based trimming (`min_quality`) on a uniform-Q40 input is
  therefore a no-op: there is nothing in the string to trim.
- Writing and re-reading a temp file costs real I/O `count()`/`sketch()` on
  an existing file also pays, plus this function's own write -- for a
  handful of in-memory records that's negligible; for millions of records
  built in a large Python-side pipeline, writing them to disk first and
  calling `fastdna.count()` directly is the same amount of work with one
  fewer layer, and may be worth doing yourself instead of through this
  module.
"""

import os
import tempfile

from . import count as _count
from . import sketch as _sketch

#: Uniform per-base quality used when a sequence carries no real quality
#: scores (plain strings, `(id, seq)` pairs, or a `SeqRecord` with no
#: `letter_annotations["phred_quality"]`). 'I' is Phred+33 for Q40 (a 1 in
#: 10,000 error rate) -- "trust this base", the same assumption implicit in
#: treating a FASTA-like sequence as if it were already known-good.
_DEFAULT_QUALITY_CHAR = "I"


def _sequence_and_id_for(item, index):
    """Duck-types one entry of `sequences` into `(id, sequence_str,
    quality_str_or_None)`.

    Accepts, in order of what is checked:

    - A Biopython `SeqRecord`-like object: anything with a `.seq`
      attribute. `str(record.seq)` gets the sequence; `.id` (falling back
      to an auto-numbered id if absent or falsy) names it; a populated
      `letter_annotations["phred_quality"]` (a list of per-base Phred
      ints, what `Bio.SeqIO.parse(..., "fastq")` actually populates) is
      converted back into a Phred+33 quality string when present, so a
      `SeqRecord` read from a *real* FASTQ file round-trips its real
      quality instead of losing it.
    - A `(id, sequence)` 2-tuple/list: used as given, no real quality.
    - A plain string: the sequence itself, auto-numbered id, no real
      quality.
    """
    seq_attr = getattr(item, "seq", None)
    if seq_attr is not None:
        sequence = str(seq_attr)
        record_id = getattr(item, "id", None) or f"seq{index}"
        quality = None
        letter_annotations = getattr(item, "letter_annotations", None)
        if letter_annotations and "phred_quality" in letter_annotations:
            phred_scores = letter_annotations["phred_quality"]
            if len(phred_scores) == len(sequence):
                quality = "".join(chr(int(q) + 33) for q in phred_scores)
        return record_id, sequence, quality

    if isinstance(item, (tuple, list)) and len(item) == 2:
        record_id, sequence = item
        return str(record_id), str(sequence), None

    # Plain string (or anything else str()-able that isn't one of the above).
    return f"seq{index}", str(item), None


def _write_fastq(sequences, fh):
    """Writes `sequences` (see `_sequence_and_id_for`) to the open text
    file `fh` as FASTQ records. Records with an empty sequence are skipped
    -- an empty read contributes no k-mers and a zero-length quality line
    would desync the FASTQ format.
    """
    n_written = 0
    for index, item in enumerate(sequences):
        record_id, sequence, quality = _sequence_and_id_for(item, index)
        if not sequence:
            continue
        if quality is None:
            quality = _DEFAULT_QUALITY_CHAR * len(sequence)
        fh.write(f"@{record_id}\n{sequence}\n+\n{quality}\n")
        n_written += 1
    return n_written


def _with_temp_fastq(sequences, fn):
    """Writes `sequences` to a temporary FASTQ file, calls `fn(path)`, and
    removes the file afterward regardless of whether `fn` raised.
    """
    fd, path = tempfile.mkstemp(suffix=".fastq", prefix="fastdna_interop_")
    try:
        # newline="\n" pins LF line endings rather than letting Python's
        # text mode translate them to CRLF on Windows: FASTQ is
        # conventionally an LF format, and pinning it here means this
        # module writes a byte-identical file on every platform rather
        # than one whose exact bytes depend on where it ran.
        with os.fdopen(fd, "w", newline="\n") as fh:
            n_written = _write_fastq(sequences, fh)
        if n_written == 0:
            raise ValueError("sequences produced no non-empty records to count/sketch")
        return fn(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass  # already gone, or never fully created -- nothing left to clean up


def count_from_sequences(sequences, *, k=31, **count_kwargs):
    """`fastdna.count()` for an in-memory iterable of sequences instead of
    a file already on disk.

    `sequences` may be:

    - an iterable of plain strings (each auto-numbered `seq0`, `seq1`, ...);
    - an iterable of `(id, sequence_string)` pairs;
    - an iterable of Biopython `SeqRecord`-like objects (duck-typed via a
      `.seq` attribute -- Biopython itself is never imported by this
      module, so it stays an optional dependency), e.g. the output of
      `Bio.SeqIO.parse(path, "fastq")` or `Bio.SeqIO.parse(path, "fasta")`.

    Writes `sequences` to a temporary FASTQ file (synthesizing uniform
    Q40 quality scores for any record that doesn't already carry real
    ones -- see this module's docstring) and calls `fastdna.count()` on
    it, deleting the temp file afterward. `k` and any other keyword
    accepted by `fastdna.count()` (`min_count`, `max_count`,
    `min_quality`, `threads`, `progress`, `progress_interval`) pass
    through unchanged.

    Raises `ValueError` if `sequences` contains no non-empty records.
    """
    return _with_temp_fastq(sequences, lambda path: _count(path, k=k, **count_kwargs))


def sketch_from_sequences(sequences, *, k=21, sketch_size=1000):
    """`fastdna.sketch()` for an in-memory iterable of sequences instead
    of a file already on disk. See `count_from_sequences` for what
    `sequences` may contain and the same "synthesized quality, temp file"
    caveats -- they apply identically here.
    """
    return _with_temp_fastq(sequences, lambda path: _sketch(path, k=k, sketch_size=sketch_size))
