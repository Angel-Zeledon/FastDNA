"""fastdna.gwas -- the k-mer GWAS on-ramp: cohort evidence in, established
association tools out.

## The design principle, stated plainly

FastDNA produces the per-sample k-mer evidence a k-mer GWAS needs and hands
it to the tools that already do the statistics properly. It does **not**
reimplement mixed-model association statistics, and this module will not
grow them.

That is a deliberate boundary, not a gap waiting to be filled. In a
bacterial or plant cohort, samples are not independent draws: they are
related by descent, and lineage is correlated with almost every phenotype
anyone measures. Every k-mer carried by a successful clone will look
associated with whatever that clone happens to cause. Correcting for that
population structure -- a linear mixed model with a kinship/similarity
matrix as the random effect, or an equivalent -- is the whole difficulty of
the field, and getting it subtly wrong does not produce an obviously broken
result: it produces a confident, publishable, false biological claim. The
established tools represent a decade of work on exactly that problem, and
they are one call away:

* Lees, Galardini, Bentley, Weiser & Corander, "pyseer: a comprehensive
  tool for microbial pangenome-wide association studies", *Bioinformatics*
  34(24):4310-4312 (2018). Fixed effects, LMM (FaST-LMM-derived) and
  elastic-net models over k-mers/unitigs/variants.
* Voichek & Weigel, "Identifying genetic variants underlying phenotypic
  variation in plants without complete genomes", *Nature Genetics*
  52:534-540 (2020) -- kmersGWAS: reference-free k-mer association with
  kinship correction and a permutation-based significance threshold.
* Jaillard, Lima, Tournoud, Mahe, van Belkum, Lacroix & Jacob, "A fast and
  agnostic method for bacterial genome-wide association studies:
  bridging the gap between k-mers and genetic events", *PLOS Genetics*
  14(11):e1007758 (2018) -- DBGWAS: de Bruijn graph GWAS, which also
  solves the "what does this significant k-mer actually mean" problem.

## What this module provides

`cohort_presence_matrix`
    One sparse `samples x k-mers` matrix from a list of FASTQ files -- the
    substrate for anything downstream (sklearn, Kover-style rule models,
    a quick look).
`export_pyseer_kmers`
    The same cohort written in pyseer's `--kmers` input format, so the
    handoff is one call rather than a scripting afternoon.
`kinship_matrix`
    A sample-by-sample similarity matrix derived from FastDNA's Mash
    distances, for pyseer's `--similarity` (the LMM's random effect).
`prefilter_association`
    A fast, **unadjusted** per-k-mer screen. Read its docstring before
    using its output for anything: it is a triage tool, not inference.

A pyseer LMM run assembled entirely from this module looks like::

    from fastdna.gwas import export_pyseer_kmers, kinship_matrix

    export_pyseer_kmers(paths, "kmers.txt.gz", k=31)
    similarity, sample_ids = kinship_matrix(paths)
    # write `similarity` as a tab-separated square matrix with sample ids
    # as both header and index (pandas: DataFrame(similarity,
    # index=sample_ids, columns=sample_ids).to_csv("K.tsv", sep="\\t"))

    # pyseer --lmm --phenotypes pheno.tsv --kmers kmers.txt.gz \\
    #        --similarity K.tsv --output-patterns patterns.txt

## Optional dependencies

`numpy` and `scipy` are imported at module load (this module cannot do
anything without them), which is why this lives in its own module rather
than in `fastdna/__init__.py` -- plain `import fastdna` must not require
either. That is the same convention `fastdna.sklearn` follows.
"""

from __future__ import annotations

import gzip
import os
import pathlib
import warnings
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse

from . import _column_as_array, _pair_positions, compare_all as _compare_all, count as _count

__all__ = [
    "ScreeningOnlyWarning",
    "PyseerExport",
    "cohort_presence_matrix",
    "export_pyseer_kmers",
    "kinship_matrix",
    "prefilter_association",
]


class ScreeningOnlyWarning(UserWarning):
    """Raised by :func:`prefilter_association` on every call.

    A distinct category (rather than a bare `UserWarning`) so that a
    caller who has genuinely understood the caveat can silence *this*
    warning specifically -- `warnings.filterwarnings("ignore",
    category=ScreeningOnlyWarning)` -- without also silencing every other
    warning FastDNA might legitimately need to raise.
    """


# The characters pyseer's own k-mer line parser uses as delimiters. A
# sample id containing any of them would be silently mis-split rather than
# rejected, so `export_pyseer_kmers` refuses them up front.
_PYSEER_RESERVED = (":", "|")


def _sample_id_from_path(path) -> str:
    """Derives a sample id from a file name: strips a trailing `.gz` first,
    then the remaining suffix, so `"ERR1234.fastq.gz"` -> `"ERR1234"` and
    not `"ERR1234.fastq"`.

    `pathlib.Path.stem` only removes the *last* suffix, and FASTQ files
    almost always carry a double extension; a stray `.fastq` left in every
    sample id would silently fail to match the sample names in a phenotype
    file, which is exactly the kind of mismatch that turns into "pyseer
    says 0 samples overlap" an hour later.
    """
    name = pathlib.Path(str(path)).name
    if name.lower().endswith(".gz"):
        name = name[: -len(".gz")]
    return pathlib.Path(name).stem


def _resolve_cohort(paths, *, caller):
    """Normalizes `paths` into `(sample_ids, path_strings)` and rejects the
    cohort-level mistakes that are cheap to catch here and expensive to
    diagnose later.

    Accepts either an iterable of paths (ids derived from file names, see
    `_sample_id_from_path`) or an explicit `{sample_id: path}` mapping, for
    accession-numbered files or any case where file names are not usable
    ids. A bare `str`/`os.PathLike` is rejected rather than iterated: a
    string is iterable, so it would otherwise be walked character by
    character and surface as a bewildering per-character
    `FileNotFoundError`.

    Duplicate ids and duplicate paths are both hard errors. A duplicated
    sample silently doubles its lineage's weight in every prevalence count,
    every association test, and every kinship row -- i.e. it manufactures
    exactly the population-structure artifact this module warns about.
    """
    if isinstance(paths, (str, os.PathLike)):
        raise TypeError(
            f"{caller}() expects a list of paths (or a {{sample_id: path}} mapping), got a "
            f"single path {str(paths)!r} -- wrap it in a list: [{str(paths)!r}]"
        )

    if isinstance(paths, Mapping):
        items = [(str(name), str(path)) for name, path in paths.items()]
    else:
        items = [(_sample_id_from_path(path), str(path)) for path in paths]

    if len(items) < 2:
        raise ValueError(
            f"{caller}() needs at least 2 samples to compare, got {len(items)}. "
            "A cohort of one has no contrast to measure and no population structure "
            "to correct for."
        )

    by_id: dict[str, list[str]] = {}
    for sample_id, path in items:
        by_id.setdefault(sample_id, []).append(path)
    collisions = {sample_id: found for sample_id, found in by_id.items() if len(found) > 1}
    if collisions:
        detail = "; ".join(
            f"{sample_id!r} <- " + ", ".join(repr(p) for p in found) for sample_id, found in sorted(collisions.items())
        )
        raise ValueError(
            f"{caller}(): duplicate sample id(s) derived from the given file names: {detail}. "
            "Sample ids must be unique -- pass an explicit {sample_id: path} mapping to name "
            "them yourself (files from different runs or directories routinely share a name)."
        )

    by_path: dict[str, list[str]] = {}
    for sample_id, path in items:
        by_path.setdefault(path, []).append(sample_id)
    repeated = {path: ids for path, ids in by_path.items() if len(ids) > 1}
    if repeated:
        detail = "; ".join(f"{path!r} as " + ", ".join(repr(i) for i in ids) for path, ids in sorted(repeated.items()))
        raise ValueError(
            f"{caller}(): the same file appears more than once in the cohort: {detail}. "
            "Counting one sample twice inflates its lineage's weight in every prevalence "
            "count and every association test -- remove the duplicate entry."
        )

    return [sample_id for sample_id, _ in items], [path for _, path in items]


def _positive_int_or_none(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive int or None, got {value!r}")
    return int(value)


def cohort_presence_matrix(paths, *, k=31, min_count=2, min_samples=2, max_kmers=None):
    """Counts every sample once and returns the cohort as one sparse matrix.

    This is the substrate a k-mer GWAS (and any other cohort-level model)
    is built on: rows are samples, columns are canonical k-mers, and the
    stored values are that k-mer's count in that sample.

    Parameters
    ----------
    paths : iterable of path, or mapping of str to path
        The cohort's FASTQ(.gz) files. An iterable derives each sample id
        from its file name; a `{sample_id: path}` mapping names them
        explicitly, which is what you want whenever the ids have to match a
        phenotype file. Order is preserved: row `i` of the matrix is
        `sample_ids[i]`.
    k : int, default 31
        K-mer length, forwarded to `fastdna.count()`. 31 is the usual
        bacterial-GWAS choice; use `fastdna.peek()` on short reads before
        trusting it.
    min_count : int, default 2
        Per-sample minimum depth for a k-mer to count as observed at all,
        forwarded to `fastdna.count()`. The default of 2 (not 1) is a
        deliberate error filter: at realistic coverage the overwhelming
        majority of depth-1 k-mers are sequencing errors, and each one
        would otherwise become a private, perfectly lineage-correlated
        column -- the exact shape of a false positive. See
        `KmerCounts.suggest_min_count()` to pick this from a sample's own
        spectrum rather than guessing.
    min_samples : int, default 2
        A k-mer must be observed in at least this many samples to become a
        column. Must not exceed the cohort size.
    max_kmers : int or None, default None
        Hard cap on the number of columns. `None` means no cap.

    Returns
    -------
    (matrix, sample_ids, kmer_sequences)
        `matrix` : `scipy.sparse.csr_matrix` of `uint32`, shape
        `(len(sample_ids), len(kmer_sequences))`. Despite the function's
        name the stored values are *counts*, not booleans -- presence is
        `matrix > 0` (or `matrix.astype(bool)`), which is the variant model
        k-mer GWAS actually uses, while the counts are kept because they
        are free here and cost a re-count to recover. "Presence" is in the
        name because presence/absence is the question the matrix exists to
        answer.
        `sample_ids` : `list[str]`, in the caller's order, indexing rows.
        `kmer_sequences` : `list[str]` of decoded canonical k-mers,
        indexing columns, sorted lexicographically.

    Column selection and `max_kmers`
    --------------------------------
    Columns are chosen in two steps, and neither is silent:

    1. K-mers observed in fewer than `min_samples` samples are dropped.
    2. If `max_kmers` is set and more survive, they are ranked by
       *minor-sample-count* -- `min(n_present, n_absent)`, the direct
       analogue of minor allele count in a SNP GWAS -- and the top
       `max_kmers` are kept. That ranking keeps the k-mers with the most
       power to discriminate anything and drops the ones nearest to being
       invariant across the cohort (a k-mer present in every sample has a
       minor-sample-count of 0 and cannot be associated with any
       phenotype). Truncation always raises a `UserWarning` naming how many
       k-mers were dropped and at which cutoff: a silently truncated
       feature matrix is a silently truncated result.

    Memory
    ------
    `max_kmers` bounds the *returned matrix*, which is what usually breaks
    a machine: a deep 200-sample cohort can easily exceed 10^8 distinct
    k-mers, and a dense-enough matrix over them will not fit in RAM. It
    does not bound the intermediate tally built while counting, which
    scales with the cohort's distinct-k-mer union regardless; `min_count`
    (and `k`) are the levers for that, and this is stated rather than
    implied because the distinction matters when a run dies anyway.

    Raises
    ------
    TypeError
        If `paths` is a single path rather than a collection of them.
    ValueError
        For fewer than 2 samples, duplicate sample ids (named), the same
        file listed twice, `min_samples` above the cohort size, or a
        non-positive `min_samples`/`max_kmers`.
    """
    sample_ids, path_strings = _resolve_cohort(paths, caller="cohort_presence_matrix")
    n_samples = len(sample_ids)

    max_kmers = _positive_int_or_none(max_kmers, "max_kmers")
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError(f"min_samples must be a positive int, got {min_samples!r}")
    min_samples = int(min_samples)
    if min_samples > n_samples:
        raise ValueError(
            f"min_samples={min_samples} exceeds the cohort size ({n_samples} samples): no k-mer "
            f"can be present in more samples than exist, so the matrix would come back empty. "
            f"Use min_samples <= {n_samples}."
        )

    # Every sample counted once, then stacked end to end so the whole
    # cohort is tallied, filtered and turned into a matrix by vectorized
    # passes over one pair of arrays. The version this replaced walked the
    # concatenation twice in Python -- once building three dicts keyed by
    # `kmer_u64` (three dict operations per row) and once looking every row
    # up again to emit a (row, column, value) triple -- so a 200-sample
    # cohort of a million k-mers each did 4x10^8 Python-level dict
    # operations. The counting itself, which dominates either way, is
    # unchanged: one `count()` per sample, exactly as before.
    kmer_arrays, sequence_arrays, frequency_arrays, row_counts = [], [], [], []
    for path in path_strings:
        table = _count(path, k=k, min_count=min_count).table
        kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
        sequence_arrays.append(_column_as_array(table.column("kmer_sequence")))
        frequency_arrays.append(_column_as_array(table.column("frequency")))
        row_counts.append(table.num_rows)

    all_kmers = pa.concat_arrays(kmer_arrays)
    all_sequences = pa.concat_arrays(sequence_arrays)
    all_frequencies = np.asarray(pa.concat_arrays(frequency_arrays))

    # `dictionary_encode` gives each distinct k-mer a code in one C++ hash
    # pass; `dictionary` holds the distinct `kmer_u64` values in order of
    # first appearance, which is the order the dict this replaced iterated
    # in. A k-mer appears at most once in a sample's count table, so "how
    # many rows carry this code" *is* its prevalence across samples.
    codes = pc.dictionary_encode(all_kmers)
    row_code = np.asarray(codes.indices).astype(np.intp, copy=False)
    distinct = np.asarray(codes.dictionary)
    n_distinct = distinct.size

    prevalence = np.bincount(row_code, minlength=n_distinct).astype(np.int64)
    # float64 weights are exact here: `frequency` is uint32 and no partial
    # sum comes anywhere near 2**53, so this recovers the same integers the
    # Python `+=` loop produced.
    total_freq = np.bincount(row_code, weights=all_frequencies, minlength=n_distinct)

    # Positions into `distinct`, ascending -- i.e. first-appearance order,
    # matching the dict iteration this replaced.
    surviving = np.flatnonzero(prevalence >= min_samples)

    if max_kmers is not None and surviving.size > max_kmers:
        surviving_prevalence = prevalence[surviving]
        minor_sample_count = np.minimum(surviving_prevalence, n_samples - surviving_prevalence)

        # Ranked by minor-sample-count first (the MAF analogue), then by
        # prevalence and total depth, then by the raw u64 encoding purely
        # so the result is reproducible independent of iteration order.
        # `lexsort` takes its primary key last.
        order = np.lexsort(
            (distinct[surviving], -total_freq[surviving], -surviving_prevalence, -minor_sample_count)
        )
        ranked = surviving[order]
        kept, dropped = ranked[:max_kmers], ranked[max_kmers:]
        cutoff = int(min(prevalence[dropped[0]], n_samples - prevalence[dropped[0]]))
        warnings.warn(
            f"max_kmers={max_kmers} truncated the cohort matrix: {surviving.size} k-mers passed "
            f"min_samples={min_samples}, {dropped.size} of them were dropped. Kept the "
            f"{max_kmers} with the highest minor-sample-count (min(present, absent) across the "
            f"{n_samples} samples, the minor-allele-count analogue); every dropped k-mer had a "
            f"minor-sample-count of {cutoff} or lower. Raise max_kmers, or raise min_count/"
            f"min_samples to shrink the candidate set on biological grounds instead.",
            UserWarning,
            stacklevel=2,
        )
        surviving = kept

    # The decoded sequence of each surviving k-mer, taken from the row it
    # first appeared in. Only the surviving k-mers are ever materialized as
    # Python strings -- the version this replaced built one string per row
    # of every sample's table, i.e. the whole cohort, to keep a handful.
    first_row = np.empty(n_distinct, dtype=np.int64)
    descending = np.arange(row_code.size - 1, -1, -1)
    first_row[row_code[descending]] = descending
    surviving_sequences = all_sequences.take(pa.array(first_row[surviving]))

    # Lexicographic column order: stable, hand-checkable, and independent
    # of how the ranking above happened to break ties. Arrow sorts these
    # byte-wise, which for equal-length ACGT strings is the same order
    # Python's `sorted` gives, and no two distinct k-mer encodings decode
    # to the same sequence, so there are no ties to break.
    lexicographic = np.asarray(pc.sort_indices(surviving_sequences))
    selected = surviving[lexicographic]
    kmer_sequences = surviving_sequences.take(pa.array(lexicographic)).to_pylist()

    # One lookup table from k-mer code to output column (-1 = not selected)
    # turns the whole cohort's rows into (row, column, value) triples with
    # a single fancy-index, replacing the per-row dict lookup and three
    # `list.append` calls the previous version did.
    column_of_code = np.full(n_distinct, -1, dtype=np.int64)
    column_of_code[selected] = np.arange(selected.size, dtype=np.int64)
    column_of_row = column_of_code[row_code]
    kept_rows = column_of_row >= 0

    matrix = sparse.csr_matrix(
        (
            all_frequencies[kept_rows].astype(np.uint32),
            (
                np.repeat(np.arange(n_samples, dtype=np.int64), row_counts)[kept_rows],
                column_of_row[kept_rows],
            ),
        ),
        shape=(n_samples, selected.size),
        dtype=np.uint32,
    )
    return matrix, sample_ids, kmer_sequences


class PyseerExport(NamedTuple):
    """What :func:`export_pyseer_kmers` wrote, so a caller can log or assert
    on it without re-reading the file it just produced.
    """

    path: pathlib.Path
    n_kmers: int
    n_samples: int
    sample_ids: list
    gzipped: bool


def export_pyseer_kmers(paths, out_path, *, k=31, min_count=2, min_samples=2):
    """Writes the cohort as a pyseer `--kmers` input file.

    Format targeted
    ---------------
    The `fsm-lite` output format, which is what pyseer 1.3.x consumes via
    `--kmers`. Verified against two primary sources rather than assumed:

    * `fsm-lite`'s own emitter (`fsm-lite.cpp`), which writes
      `cout << s + " |";` for the pattern and then, per sample,
      `cout << ' ' << id << ':' << count;` -- i.e. one line per k-mer::

          <kmer> | <sample_id>:<count> <sample_id>:<count> ...

    * pyseer's reader (`pyseer/input.py`, `read_variant`), which parses
      each line as `line.split()[0]` for the k-mer and
      `line.rstrip().split('|')[1].lstrip().split()` for the sample list,
      then keeps only `x.split(':')[0]` from each entry.

    Two consequences worth stating, because both are easy to get wrong:
    the `|` is **required** (pyseer indexes `split('|')[1]`, so a
    tab-separated `kmer<TAB>sample:1 ...` line raises `IndexError` deep
    inside pyseer rather than failing cleanly), and the per-sample counts
    are **discarded** by pyseer, which maps every listed sample to presence
    `1`. The real FastDNA counts are written anyway, matching what
    `fsm-lite` emits, so the same file stays useful to anything that does
    read them.

    Compression
    -----------
    pyseer assumes this file is gzipped (`--uncompressed` opts out), so a
    `out_path` ending in `.gz` (case-insensitive) is gzip-compressed and
    anything else is written as plain text. Prefer the `.gz` form: it is
    both pyseer's default and, for a real cohort, several-fold smaller.

    Parameters
    ----------
    paths : iterable of path, or mapping of str to path
        As in :func:`cohort_presence_matrix`. The sample ids written into
        every line are the ones this resolves to, and they must match the
        sample names in the phenotype file passed to pyseer's
        `--phenotypes` -- pyseer intersects the two and will report an
        empty overlap rather than guessing.
    k, min_count, min_samples
        Forwarded to :func:`cohort_presence_matrix`; see there. `max_kmers`
        is deliberately not exposed: a truncated k-mer file handed to
        pyseer would silently change which hypotheses were tested, and the
        Bonferroni-style threshold pyseer derives from the number of
        patterns along with it.

    Returns
    -------
    PyseerExport
        `(path, n_kmers, n_samples, sample_ids, gzipped)`.

    Raises
    ------
    ValueError
        Everything :func:`cohort_presence_matrix` raises, plus any sample
        id containing whitespace, `':'` or `'|'` -- pyseer's line parser
        splits on exactly those, so such an id would be silently
        mis-parsed into a different (or truncated) sample name instead of
        failing.
    """
    matrix, sample_ids, kmer_sequences = cohort_presence_matrix(
        paths, k=k, min_count=min_count, min_samples=min_samples, max_kmers=None
    )

    for sample_id in sample_ids:
        offending = [ch for ch in _PYSEER_RESERVED if ch in sample_id]
        if any(ch.isspace() for ch in sample_id):
            offending.append("whitespace")
        if offending:
            raise ValueError(
                f"sample id {sample_id!r} contains {', '.join(repr(o) for o in offending)}, which the "
                "pyseer/fsm-lite k-mer line format reserves: pyseer splits each line on '|' and then "
                "each entry on ':', so this id would be silently mis-parsed as a different sample. "
                "Rename it by passing an explicit {sample_id: path} mapping."
            )

    out_path = pathlib.Path(str(out_path))
    gzipped = out_path.name.lower().endswith(".gz")

    # Binary mode throughout, not text mode: on Windows a text-mode write
    # would translate every '\n' into '\r\n', and pyseer's parser would
    # then carry a trailing '\r' into the last sample id of every line.
    csc = matrix.tocsc()
    opener = (lambda: gzip.open(out_path, "wb")) if gzipped else (lambda: open(out_path, "wb"))
    with opener() as handle:
        for column, sequence in enumerate(kmer_sequences):
            start, end = csc.indptr[column], csc.indptr[column + 1]
            entries = " ".join(
                f"{sample_ids[row]}:{count}"
                for row, count in zip(csc.indices[start:end].tolist(), csc.data[start:end].tolist())
            )
            handle.write(f"{sequence} | {entries}\n".encode("ascii"))

    return PyseerExport(
        path=out_path,
        n_kmers=len(kmer_sequences),
        n_samples=len(sample_ids),
        sample_ids=sample_ids,
        gzipped=gzipped,
    )


def kinship_matrix(paths, *, k=21, sketch_size=10_000):
    """A sample-by-sample similarity matrix derived from FastDNA's Mash
    distances -- the population-structure covariance a mixed-model GWAS
    needs as its random effect.

    Built from `fastdna.compare_all(..., metric="mash_distance")`, with
    `similarity = 1 - mash_distance`. Sketching each sample once and
    comparing sketches makes this O(N) FASTQ reads and O(N^2) cheap sketch
    comparisons, rather than the O(N^2) full-genome comparisons an exact
    all-pairs distance would cost.

    Parameters
    ----------
    paths : iterable of path, or mapping of str to path
        As in :func:`cohort_presence_matrix`.
    k : int, default 21
        Sketching k, matching `fastdna.sketch()`'s own default (and Mash's).
        Shorter than the k used for association: a sketch is answering "how
        related are these genomes", where 31-mers are needlessly brittle to
        single substitutions.
    sketch_size : int, default 10_000
        MinHash sketch size. Larger than `fastdna.sketch()`'s default of
        1,000 because a kinship matrix is used to *correct* every test in
        the study: sketch noise here does not average out, it propagates
        into every p-value. 10,000 is the usual "careful" setting; raise it
        for closely related isolates, where the distances being separated
        are small.

    Returns
    -------
    (matrix, sample_ids)
        `matrix` : `numpy.ndarray` of shape `(n, n)`, symmetric, with an
        exact 1.0 diagonal (a sample is identical to itself by definition;
        it is not estimated). `sample_ids` : `list[str]` indexing both
        axes, in the caller's order.

    Feeding it to pyseer
    --------------------
    pyseer's `--similarity` expects a tab-separated square matrix with
    sample names as both the header row and the index column::

        import pandas as pd
        similarity, sample_ids = kinship_matrix(paths)
        pd.DataFrame(similarity, index=sample_ids, columns=sample_ids).to_csv("K.tsv", sep="\\t")

    Honest limits
    -------------
    This is a *sequence-similarity* matrix, not a kinship matrix estimated
    from a phylogeny. pyseer's own documented route is
    `phylogeny_distance.py --lmm` over a core-genome tree, and where a
    reliable tree exists it should be preferred: a Mash-derived matrix
    inherits every distortion that whole-genome k-mer content carries but
    shared ancestry does not -- recombination, plasmids, mobile elements,
    contamination, and coverage differences all move Mash distance without
    moving relatedness. It is a defensible structure covariate when no tree
    is available (the situation kmersGWAS is designed for), and it is
    strictly better than running with no correction at all, which is the
    only alternative this module would otherwise be offering.

    Raises
    ------
    TypeError, ValueError
        As in :func:`cohort_presence_matrix` (single path, fewer than 2
        samples, duplicate ids, duplicate files).
    """
    sample_ids, path_strings = _resolve_cohort(paths, caller="kinship_matrix")
    n = len(sample_ids)

    table = _compare_all(path_strings, k=k, sketch_size=sketch_size, metric="mash_distance")

    similarity = np.eye(n, dtype=np.float64)
    # Filled in one vectorized pass rather than row by row: the table has
    # `n*(n-1)/2` rows, each of which cost two Python dict lookups, a
    # scalar subtract and two element assignments (19,900 iterations for a
    # 200-sample cohort). `1.0 - d` elementwise on float64 is the same IEEE
    # subtraction, so every entry is bit-identical; `np.eye`'s exact 1.0
    # diagonal is left alone, since the table carries no self-pairs.
    i, j = _pair_positions(table, path_strings)
    value = 1.0 - np.asarray(_column_as_array(table.column("mash_distance")))
    similarity[i, j] = value
    similarity[j, i] = value

    return similarity, sample_ids


def _benjamini_hochberg(sorted_pvalues):
    """BH q-values for an already-ascending array of p-values.

    Written out rather than pulled from `statsmodels` (a dependency this
    package does not otherwise need) -- it is four lines, and the
    monotonicity enforcement (`np.minimum.accumulate` from the largest
    p-value down) is the only part anyone gets wrong.
    """
    m = sorted_pvalues.size
    if m == 0:
        return sorted_pvalues
    raw = sorted_pvalues * m / np.arange(1, m + 1)
    return np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0.0, 1.0)


def _contingency_counts(matrix, case_mask):
    """Per-column `(a, b, c, d)` = (case present, case absent, control
    present, control absent), computed with two sparse row-slice sums
    rather than a Python loop over columns.
    """
    csr = (matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))).copy()
    # A caller-supplied matrix may carry explicitly stored zeros; dropping
    # them first is what makes `.astype(bool)` mean "present" rather than
    # "has a stored entry".
    csr.eliminate_zeros()
    present = csr.astype(bool)

    n_case = int(case_mask.sum())
    n_control = int(case_mask.size - n_case)
    a = np.asarray(present[case_mask].sum(axis=0)).ravel().astype(np.int64)
    c = np.asarray(present[~case_mask].sum(axis=0)).ravel().astype(np.int64)
    return a, n_case - a, c, n_control - c


_SCREENING_METADATA = {
    "fastdna.screening_only": "true",
    "fastdna.population_structure_correction": "none",
    "fastdna.multiple_testing": (
        "p_bonferroni and q_value_bh are reported for orientation only; no threshold is applied "
        "and neither corrects for population structure"
    ),
    "fastdna.confirm_with": (
        "pyseer (--lmm with --similarity), kmersGWAS, DBGWAS, or an equivalent mixed-model tool, "
        "before any result here is described as an association"
    ),
}


def prefilter_association(matrix, phenotype, kmer_sequences, *, test="fisher", top_n=None):
    """A fast, **unadjusted** per-k-mer screen of a binary phenotype.

    This is triage, not inference. Read this paragraph before using the
    output for anything: every k-mer is tested independently against the
    phenotype with no correction for population structure, no random
    effect, and no model of relatedness between samples. In any real
    bacterial or plant cohort the samples are related by descent and
    lineage is correlated with the phenotype, so a successful clone's
    entire accessory genome -- hundreds of thousands of k-mers that have
    nothing to do with the trait -- will come out of this function with
    small p-values. That is not a subtle bias; it is the dominant signal in
    an uncorrected k-mer screen. The Bonferroni and Benjamini-Hochberg
    columns do not fix it either: multiple-testing control assumes the
    tests are exchangeable, which structure-confounded tests are not.
    Nothing this function returns may be reported as an association until
    it has been confirmed with pyseer (`--lmm` with a `--similarity` matrix
    -- see :func:`kinship_matrix`), kmersGWAS, DBGWAS, or an equivalent
    mixed model.

    What it is genuinely good for: sanity-checking a cohort before
    committing to a long run ("does *anything* separate cases from
    controls?"), shortlisting k-mers for a downstream model, and debugging
    a phenotype file. Every call emits a :class:`ScreeningOnlyWarning`
    saying so, and the returned table carries the same statement in its
    Arrow schema metadata (`fastdna.screening_only`,
    `fastdna.population_structure_correction`, `fastdna.confirm_with`), so
    the caveat survives being saved to Parquet and read back by someone
    who never saw this docstring.

    Parameters
    ----------
    matrix : scipy.sparse matrix or 2-D array, shape (n_samples, n_kmers)
        As returned by :func:`cohort_presence_matrix`. Values are
        binarized: any non-zero entry is "present". Row order must match
        `phenotype`.
    phenotype : array-like of length n_samples
        A binary phenotype -- exactly two distinct values (`0`/`1`,
        `False`/`True`, or any pair of comparable labels). The greater of
        the two is treated as the case group, so the usual 0/1 and
        `False`/`True` encodings mean what they look like. Continuous
        phenotypes are rejected rather than thresholded behind your back;
        use pyseer's `--continuous` for those.
    kmer_sequences : sequence of str, length n_kmers
        Column labels, as returned by :func:`cohort_presence_matrix`. They
        are decoded DNA, so a hit is directly BLASTable -- which is most of
        the point of screening at the k-mer level at all.
    test : {"fisher", "chi2"}, default "fisher"
        `"fisher"` is Fisher's exact test, correct at any cell count and
        the right default for cohorts where a k-mer may appear in a handful
        of samples. `"chi2"` is Pearson's chi-squared without continuity
        correction -- faster, but its asymptotic approximation is
        unreliable when any expected cell count falls below ~5, which for
        rare k-mers is the normal case rather than the exception.
    top_n : int or None, default None
        Return only the `top_n` most significant rows. The multiple-testing
        columns are always computed over *all* k-mers tested, not over the
        truncated view, so `top_n` changes what you see and never what the
        numbers mean.

    Returns
    -------
    pyarrow.Table
        Sorted by `p_value` ascending (ties broken by `kmer_sequence`, so
        the order is reproducible), with columns:
        `kmer_sequence`, `n_case_present`, `n_case_absent`,
        `n_control_present`, `n_control_absent`, `odds_ratio`, `p_value`,
        `p_bonferroni`, `q_value_bh`.
        `odds_ratio` is the sample odds ratio `(a*d)/(b*c)`, `nan` when a
        zero cell makes it undefined. K-mers present in every sample or in
        none have no variance to test and are reported with `p_value` 1.0
        and a `nan` odds ratio rather than being silently dropped -- their
        presence in the table is itself information about the cohort.

    Warns
    -----
    ScreeningOnlyWarning
        On every call, unconditionally.

    Raises
    ------
    ValueError
        If `matrix` has fewer than 2 rows; if `len(phenotype)` does not
        equal the number of rows; if `len(kmer_sequences)` does not equal
        the number of columns; if the phenotype has a single class or more
        than two; or if `test` is not one of the two supported names.
    """
    from scipy import stats

    if test not in ("fisher", "chi2"):
        raise ValueError(f"test must be 'fisher' or 'chi2', got {test!r}")
    top_n = _positive_int_or_none(top_n, "top_n")

    n_samples, n_kmers = matrix.shape
    if n_samples < 2:
        raise ValueError(
            f"prefilter_association() needs at least 2 samples, got a matrix with {n_samples} row(s). "
            "A single sample has no case/control contrast to test."
        )

    phenotype_array = np.asarray(phenotype)
    if phenotype_array.ndim != 1:
        raise ValueError(f"phenotype must be 1-dimensional, got shape {phenotype_array.shape}")
    if phenotype_array.size != n_samples:
        raise ValueError(
            f"phenotype has {phenotype_array.size} entries but the matrix has {n_samples} samples (rows). "
            "They must line up element by element: phenotype[i] is the phenotype of the sample in "
            "row i, in the same order cohort_presence_matrix() returned its sample_ids."
        )
    if len(kmer_sequences) != n_kmers:
        raise ValueError(
            f"kmer_sequences has {len(kmer_sequences)} entries but the matrix has {n_kmers} columns. "
            "Pass the kmer_sequences that cohort_presence_matrix() returned alongside this matrix -- "
            "a mismatched list would label every result with the wrong k-mer."
        )

    classes = np.unique(phenotype_array)
    if classes.size < 2:
        raise ValueError(
            f"phenotype has a single class ({classes.tolist()!r}): there is nothing to associate against. "
            "Check that the phenotype column was read correctly and that both cases and controls "
            "are present in this cohort."
        )
    if classes.size > 2:
        raise ValueError(
            f"phenotype has {classes.size} distinct values, but prefilter_association() only supports a "
            "binary phenotype. Encode it as 0/1 (or False/True) yourself if it really is binary; for a "
            "genuinely continuous phenotype use pyseer's --continuous, which models it properly "
            "instead of thresholding it arbitrarily."
        )

    warnings.warn(
        "prefilter_association() is an UNADJUSTED screen, not inference: no population-structure "
        "correction, no random effect, no relatedness model. In a cohort with any lineage structure "
        "-- i.e. every real one -- a successful clone's whole accessory genome will score as "
        "significant here. The p_bonferroni/q_value_bh columns control multiplicity, not "
        "confounding. Confirm anything of interest with pyseer (--lmm plus a --similarity matrix, "
        "see fastdna.gwas.kinship_matrix), kmersGWAS or DBGWAS before calling it an association.",
        ScreeningOnlyWarning,
        stacklevel=2,
    )

    case_value = classes[1]
    case_mask = np.asarray(phenotype_array == case_value)
    a, b, c, d = _contingency_counts(matrix, case_mask)

    p_values = np.ones(n_kmers, dtype=np.float64)
    odds_ratios = np.full(n_kmers, np.nan, dtype=np.float64)

    # Chosen once rather than re-tested per k-mer.
    run_test = stats.fisher_exact if test == "fisher" else partial(stats.chi2_contingency, correction=False)

    # A k-mer's 2x2 table is fully determined by `(a, c)`: `b` is
    # `n_case - a` and `d` is `n_control - c`, both fixed for the cohort.
    # There are therefore at most `(n_case + 1) * (n_control + 1)` distinct
    # tables no matter how many k-mers were tested, and the same table
    # always yields the same p-value -- so each one is handed to scipy
    # once. A 200-sample cohort (100 cases, 100 controls) has at most
    # 10,201 distinct tables, so a 10-million-k-mer screen makes ~10^4
    # `fisher_exact` calls instead of 10^7. The cached value is the number
    # scipy returned for that exact table, so no p-value moves.
    p_value_of = {}
    for j in range(n_kmers):
        aj, bj, cj, dj = int(a[j]), int(b[j]), int(c[j]), int(d[j])
        # A column that is all-present or all-absent has a zero marginal:
        # no variance, no test to run (and `chi2_contingency` would raise).
        if (aj + cj) == 0 or (bj + dj) == 0:
            continue
        if bj * cj:
            odds_ratios[j] = (aj * dj) / (bj * cj)
        elif aj * dj:
            odds_ratios[j] = np.inf
        cached = p_value_of.get((aj, cj))
        if cached is None:
            result = run_test([[aj, bj], [cj, dj]])
            # scipy >= 1.11 returns a result object with `.pvalue`; older
            # releases return a plain tuple whose second element is the
            # p-value for both tests.
            cached = float(result.pvalue if hasattr(result, "pvalue") else result[1])
            p_value_of[(aj, cj)] = cached
        p_values[j] = cached

    sequences = np.asarray(list(kmer_sequences), dtype=object)
    # Primary key is the last argument to lexsort: p ascending, ties broken
    # by sequence so repeated runs agree row for row.
    order = np.lexsort((sequences.astype(str), p_values))

    sorted_p = p_values[order]
    q_values = _benjamini_hochberg(sorted_p)
    bonferroni = np.clip(sorted_p * n_kmers, 0.0, 1.0)

    keep = slice(None) if top_n is None else slice(0, top_n)
    table = pa.table(
        {
            "kmer_sequence": pa.array([str(s) for s in sequences[order][keep]], type=pa.string()),
            "n_case_present": pa.array(a[order][keep], type=pa.int32()),
            "n_case_absent": pa.array(b[order][keep], type=pa.int32()),
            "n_control_present": pa.array(c[order][keep], type=pa.int32()),
            "n_control_absent": pa.array(d[order][keep], type=pa.int32()),
            "odds_ratio": pa.array(odds_ratios[order][keep], type=pa.float64()),
            "p_value": pa.array(sorted_p[keep], type=pa.float64()),
            "p_bonferroni": pa.array(bonferroni[keep], type=pa.float64()),
            "q_value_bh": pa.array(q_values[keep], type=pa.float64()),
        }
    )
    return table.replace_schema_metadata(
        {**_SCREENING_METADATA, "fastdna.n_kmers_tested": str(n_kmers), "fastdna.test": test}
    )
