"""fastdna.cohort_counts -- a cohort's k-mer counts, counted once.

## Why this exists

`KmerVectorizer.fit()` counts every training FASTQ; `.transform()` counts
every file it is given. scikit-learn's own `Pipeline`, `cross_val_score`
and `GridSearchCV` call `fit()` on the training fold and `transform()` on
the held-out fold at *every* split -- so a 5-fold `cross_val_score` over a
cohort read that cohort's FASTQ files five times over, and
`fastdna.audit()` (which runs both a random and a lineage-blocked CV
scheme, plus the sketching `compare_all()` needs for lineage detection)
reads it closer to eleven times. Counting is by far the most expensive
thing this package does; none of those re-reads are necessary, because
`fastdna.count()` is a pure, deterministic function of a FASTQ file's
bytes.

`CohortCounts` is that count, taken once and reused: `count_cohort()`
counts every sample exactly once and stacks the results end to end;
`CohortCounts.subset()` slices out the rows belonging to a given set of
`sample_id`s in O(those rows) with no FASTQ touched. `KmerVectorizer`'s
`counts=` parameter (`fastdna.sklearn`) is what turns that into "no
recounting inside cross-validation" in practice.

## Why counting ahead of time does not reopen the leakage door

The leakage `KmerVectorizer` exists to prevent is the *vocabulary* --
which k-mers become features -- being decided from data that includes a
held-out fold. Counting a k-mer's occurrences in one sample looks at
nothing but that sample: it does not compare samples, does not look at
labels, and does not decide anything about which k-mers matter. Counting
the whole cohort up front is therefore not a decision that could leak;
the decision that could (`KmerVectorizer._learn_vocabulary`) still runs
only inside `fit()`, only on the training fold's own rows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional, Sequence, Union

import pyarrow as pa

if TYPE_CHECKING:
    import numpy as np

import fastdna
from fastdna import _column_as_array, _PathLike

__all__ = ["CohortCounts", "count_cohort"]


@dataclass(frozen=True)
class CohortCounts:
    """A cohort's k-mer tables, counted once and stacked end to end.

    `kmers`/`frequencies` are the concatenation of every sample's own
    `fastdna.count()` columns, in `sample_ids` order; `row_counts[i]` is
    how many rows sample `i` contributed, which is what `offsets`/`subset`
    use to slice the stack back apart without rereading anything.

    `frozen=True` on purpose: this is a data artifact meant to be shared
    across every fold of a cross-validation (that sharing is the entire
    point of building it once), and a fold that could mutate it would be
    a silent way for one fold's view to leak into another's.

    Attributes
    ----------
    sample_ids : tuple of str
        Every sample's id, in the order its rows were stacked.
    kmers : pyarrow.Array of uint64
        Every sample's `kmer_u64` column, concatenated in `sample_ids`
        order.
    frequencies : pyarrow.Array of uint32
        Every sample's `frequency` column, concatenated in `sample_ids`
        order, aligned row-for-row with `kmers`.
    row_counts : tuple of int
        `row_counts[i]` is how many rows `sample_ids[i]` contributed --
        what `offsets`/`subset` use to slice the stack back apart.
    k : int
        The k-mer size every sample was counted at.
    min_count : int
        The `min_count` every sample was counted at.
    """

    sample_ids: tuple[str, ...]
    kmers: pa.Array
    frequencies: pa.Array
    row_counts: tuple[int, ...]
    k: int
    min_count: int

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def offsets(self) -> np.ndarray:
        """The start index of each sample's rows within `kmers`, plus one
        trailing entry for the end of the last sample -- `offsets[i]` to
        `offsets[i+1]` is sample `i`'s row range. A `numpy.ndarray`;
        `numpy` is imported lazily here (not at module scope) so plain
        `import fastdna` never requires it -- matching this package's own
        convention, e.g. `KmerCounts.to_polars()`'s local `import polars`.
        """
        import numpy as np

        return np.concatenate([[0], np.cumsum(self.row_counts)]).astype(np.int64)

    def subset(self, sample_ids: Sequence[str]) -> "CohortCounts":
        """The counts for a subset of `sample_ids`, with none of them
        recounted -- an O(rows in that subset) slice of the already-built
        arrays.

        This is what `KmerVectorizer(counts=...)` calls once per
        `fit()`/`transform()` in place of rereading FASTQ files, and it is
        the whole reason `CohortCounts` exists: a cross-validation fold
        becomes a slice of one array instead of a fresh counting pass.

        Parameters
        ----------
        sample_ids : sequence of str
            The subset to slice out, in the order the result should have.

        Returns
        -------
        CohortCounts
            A new instance holding only `sample_ids`' rows.

        Raises
        ------
        KeyError
            Naming every `sample_id` that is not in this cohort, rather
            than silently returning a shorter result: a fold quietly
            missing samples would produce a plausible-looking score
            computed over the wrong cohort.
        """
        import numpy as np

        position = {sid: i for i, sid in enumerate(self.sample_ids)}
        missing = [s for s in sample_ids if s not in position]
        if missing:
            preview = missing[:5]
            suffix = " ..." if len(missing) > 5 else ""
            raise KeyError(
                f"these sample_id values are not in this CohortCounts: {preview}{suffix}. "
                f"The cohort has {len(self.sample_ids)} samples."
            )

        starts = self.offsets
        if sample_ids:
            take = np.concatenate(
                [np.arange(starts[position[s]], starts[position[s] + 1]) for s in sample_ids]
            )
        else:
            take = np.empty(0, dtype=np.int64)

        indices = pa.array(take, type=pa.int64())
        return CohortCounts(
            sample_ids=tuple(sample_ids),
            kmers=self.kmers.take(indices),
            frequencies=self.frequencies.take(indices),
            row_counts=tuple(self.row_counts[position[s]] for s in sample_ids),
            k=self.k,
            min_count=self.min_count,
        )


def _sample_id_from_path(path: _PathLike) -> str:
    """The file's stem with a trailing `.fastq`/`.fq`/`.fasta`/`.fa`/`.fna`/
    `.gz` suffix stripped. `pathlib.Path.stem` only strips the *last*
    suffix, so a `.fastq.gz` file's stem would otherwise keep a stray
    `.fastq` in every sample_id -- this strips every suffix in the tuple
    that matches, in order, so a double extension like `.fna.gz` loses both
    parts (`.gz` first, then `.fna`) the same way `.fastq.gz` already does.

    `.fna` matters beyond FASTQ read files: `fastdna.count()`'s Rust core
    sniffs FASTA vs. FASTQ from a stream's first byte, not from the
    extension (`src/fastq.rs::sniff_format`), and `.fna` -- not `.fasta` or
    `.fa` -- is the extension BV-BRC, NCBI and most other genome archives
    actually use for assembled nucleotide FASTA (e.g. BV-BRC's own
    `genome_sequence` API and its `ftp://ftp.bvbrc.org/genomes/<id>/<id>.fna`
    layout). Without it here, a cohort built from real downloaded genome
    assemblies got sample_ids with a stray `.fna` still attached (breaking
    any caller-side matching against `sample_id`s from elsewhere, e.g.
    `fastdna.audit(..., groups=...)`'s own `paths` positional alignment),
    while `count_cohort(directory)` silently found nothing at all to count
    if pointed at a directory of `.fna` files.
    """
    name = os.path.basename(str(path))
    for suffix in (".gz", ".fastq", ".fq", ".fasta", ".fa", ".fna"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def count_cohort(
    # directory path, {sample_id: path} mapping, or iterable of paths -- see docstring
    samples: Union[_PathLike, Mapping[str, _PathLike], Iterable[_PathLike]],
    *,
    k: int = 31,
    min_count: int = 1,
    threads: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> CohortCounts:
    """Counts every sample in `samples` exactly once and returns the
    result as one `CohortCounts`.

    `with_sequence` is never requested here: nothing that consumes a
    `CohortCounts` (`KmerVectorizer`, `fastdna.audit()`) needs the decoded
    `kmer_sequence` column, and it is the majority of a count's cost (see
    `export::counts_schema`'s doc comment). `KmerVectorizer` decodes the
    handful of k-mers that survive vocabulary selection directly from
    their `kmer_u64` values instead.

    Parameters
    ----------
    samples : str, os.PathLike, Mapping[str, path], or iterable of path
        A directory (every FASTQ/FASTA file in it, sample_id derived from
        the filename), a `Mapping[sample_id, path]` (explicit ids), or a
        list/iterable of paths (ids derived from the filenames).
    k : int, default 31
        Forwarded to `fastdna.count()` for every sample.
    min_count : int, default 1
        Forwarded to `fastdna.count()` for every sample.
    threads : int, optional
        Forwarded to `fastdna.count()` for every sample.
    progress : callable, optional
        Called as `progress(completed, total, sample_id)` after each
        sample finishes counting, for a caller that wants to report
        progress across a cohort's worth of files.

    Returns
    -------
    CohortCounts
        Every sample's counts, stacked in `samples` order.

    Raises
    ------
    ValueError
        If `samples` is empty, if `samples` is a directory containing no
        FASTQ/FASTA files, or if two different paths derive the same
        sample_id -- silently keeping only one would drop a sample from
        the cohort without saying so.
    """
    if isinstance(samples, Mapping):
        pairs = [(str(sid), str(p)) for sid, p in samples.items()]
    elif isinstance(samples, (str, os.PathLike)) and os.path.isdir(samples):
        entries = sorted(
            os.path.join(samples, f)
            for f in os.listdir(samples)
            if f.lower().endswith(
                (".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fasta", ".fa", ".fna", ".fna.gz")
            )
        )
        if not entries:
            raise ValueError(f"no FASTQ/FASTA files found in {samples}")
        pairs = [(_sample_id_from_path(p), p) for p in entries]
    else:
        paths = [str(p) for p in samples]
        if not paths:
            raise ValueError("count_cohort() requires at least one sample, got an empty input")
        pairs = [(_sample_id_from_path(p), p) for p in paths]

    seen: dict[str, str] = {}
    for sid, path in pairs:
        if sid in seen:
            raise ValueError(
                f"two paths both derive sample_id {sid!r}: {seen[sid]!r} and {path!r}. "
                "Pass an explicit {sample_id: path} dict to disambiguate."
            )
        seen[sid] = path

    kmer_arrays, freq_arrays, row_counts, ids = [], [], [], []
    for i, (sid, path) in enumerate(pairs):
        table = fastdna.count(path, k=k, min_count=min_count, threads=threads).table
        kmer_arrays.append(_column_as_array(table.column("kmer_u64")))
        freq_arrays.append(_column_as_array(table.column("frequency")))
        row_counts.append(table.num_rows)
        ids.append(sid)
        if progress is not None:
            progress(i + 1, len(pairs), sid)

    if not kmer_arrays:
        kmers = pa.array([], type=pa.uint64())
        freqs = pa.array([], type=pa.uint32())
    else:
        kmers = pa.concat_arrays(kmer_arrays)
        freqs = pa.concat_arrays(freq_arrays)

    return CohortCounts(
        sample_ids=tuple(ids), kmers=kmers, frequencies=freqs, row_counts=tuple(row_counts), k=k,
        min_count=min_count,
    )
