"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import pyarrow as pa
import pyarrow.compute as pc

from . import _core
from ._progress import make_progress_adapter
from .spectrum import suggest_min_count as _suggest_min_count

if TYPE_CHECKING:
    # `pandas`/`polars` are soft dependencies (see `KmerCounts.to_pandas`/
    # `.to_polars`) never imported at module load time; this import only
    # runs for static type checkers, so their return-type annotations can
    # name the real classes without requiring either package at runtime.
    import pandas
    import polars

__version__ = _core.__version__

__all__ = [
    "count",
    "peek",
    "build_info",
    "KmerCounts",
    "Sketch",
    "sketch",
    "load_sketch",
    "FracSketch",
    "frac_sketch",
    "load_frac_sketch",
    "KmerTable",
    "compare",
    "compare_all",
    "estimate_cardinality",
    "estimate_spectrum",
    # Re-exported at the bottom of this module (see the comment there).
    "CohortCounts",
    "count_cohort",
    "audit",
    "explain",
]

# A path accepted anywhere in this module: a `str`, or anything implementing
# `os.PathLike` (e.g. `pathlib.Path`) -- every such parameter is converted
# with `str(path)` before crossing into the Rust core, which only accepts
# `str`.
_PathLike = Union[str, os.PathLike]


def _column_as_array(column):
    """One contiguous `pyarrow.Array` from a `pyarrow.Table` column.

    `Table.column()` hands back a `ChunkedArray`, and whether
    `ChunkedArray.combine_chunks()` returns an `Array` or another
    `ChunkedArray` has changed across the pyarrow versions this package
    supports (`pyarrow>=14`). Doing the flattening explicitly keeps
    `pa.concat_arrays()` -- which every cohort-level module uses to stack
    per-sample count tables into one array before a single vectorized pass
    over them -- working on all of them.
    """
    if isinstance(column, pa.ChunkedArray):
        chunks = column.chunks
        if len(chunks) == 1:
            return chunks[0]
        if not chunks:
            return pa.array([], type=column.type)
        return pa.concat_arrays(chunks)
    return column


def _decode_kmers(bits, k):
    """Decodes packed 2-bit-per-base k-mers (`kmer_u64` values) into their
    ASCII sequence strings, vectorized over every value in `bits` at once.

    `bits` is anything `numpy.asarray` accepts -- a `pyarrow.Array`/
    `ChunkedArray` (via its own `to_numpy`), a plain Python sequence of
    ints, or an existing numpy array.

    Shared by `KmerCounts.with_sequence()` (decodes every row of a table)
    and `fastdna.sklearn.KmerVectorizer` (decodes only the k-mers that
    survive vocabulary selection -- typically `top_features`, several
    orders of magnitude fewer than a cohort's row count): both need the
    same bit layout, and `kmer::decode_kmer` is the only other place that
    layout is implemented, Rust-side, so this is the one Python copy of it
    rather than a second one drifting from the first.

    Position `k-1-i` (from the right) comes from bits `2*i`/`2*i+1` of the
    packed k-mer -- the same layout `kmer::decode_kmer_into` unpacks in
    Rust, most significant bits first.
    """
    import numpy as np

    # `pyarrow.Array`/`ChunkedArray` -> numpy via their own `to_numpy()`
    # (uniform across the pyarrow>=14 versions this package supports,
    # unlike relying on `np.asarray`'s buffer-protocol auto-detection to
    # do the same thing); anything else (a plain sequence, an existing
    # numpy array) goes through `np.asarray` directly.
    if isinstance(bits, (pa.Array, pa.ChunkedArray)):
        bits = bits.to_numpy(zero_copy_only=False)
    bits = np.asarray(bits, dtype=np.uint64)
    alphabet = np.frombuffer(b"ACGT", dtype=np.uint8)
    codes = np.empty((len(bits), k), dtype=np.uint8)
    for i in range(k):
        codes[:, k - 1 - i] = alphabet[(bits >> np.uint64(2 * i)) & np.uint64(0b11)]
    return [row.tobytes().decode("ascii") for row in codes]


def _pair_positions(table, paths):
    """Row and column indices, into `paths`, for every row of a
    :func:`compare_all` long-format table.

    Every consumer of that table (`embed`, `cv`, `gwas.kinship_matrix`)
    needs the same thing: turn each `(sample_a, sample_b)` pair of path
    strings into the pair of positions it occupies in a dense `n x n`
    matrix. Doing it here, once, with Arrow's own hash lookup resolves all
    `n*(n-1)/2` rows in two C++ passes instead of two Python dict lookups
    per row -- a 200-sample cohort has 19,900 of those rows.

    Raises `KeyError` naming the offending path if the table mentions a
    sample that is not in `paths`, which is what the dict lookup this
    replaced did (silently mapping it to nothing would leave a row of the
    matrix all zeros).
    """
    value_set = pa.array([str(p) for p in paths], type=pa.string())
    positions = []
    for column_name in ("sample_a", "sample_b"):
        samples = _column_as_array(table.column(column_name))
        found = pc.index_in(samples, value_set=value_set)
        if found.null_count:
            raise KeyError(pc.filter(samples, pc.is_null(found))[0].as_py())
        positions.append(found.to_numpy(zero_copy_only=False))
    return positions[0], positions[1]


def _fallback_table_html(table):
    """A minimal hand-built HTML `<table>` over a small `pyarrow.Table`,
    used by `_repr_html_` methods when `pandas` (their preferred renderer)
    is not installed. No styling beyond what a Jupyter cell already
    applies to a bare `<table>` -- this exists so rich display degrades
    gracefully rather than not at all, not to look identical to pandas'
    own `to_html()` output.
    """
    columns = table.column_names
    header = "".join(f"<th>{col}</th>" for col in columns)
    rows_html = []
    for row in table.to_pylist():
        cells = "".join(f"<td>{row[col]}</td>" for col in columns)
        rows_html.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


class KmerCounts:
    """The result of :func:`count`, and of chaining any of its own
    filtering/ordering methods.

    Wraps the Rust-side `_core.KmerCounts` object, whose `.table` getter
    already returns a zero-copy `pyarrow.RecordBatch` (via Arrow's C Data
    Interface). This class only wraps that single batch into a
    `pyarrow.Table` -- itself a cheap reference wrap, not a copy -- so the
    public API matches the design doc's `r.table # pyarrow.Table`.

    `.filter()`, `.sort_by()` and `.top()` are chainable: each returns a
    new `KmerCounts` holding a derived `pyarrow.Table` (built with
    `pyarrow.compute`, entirely in Python -- no new Rust-side work, per
    this package's own rule that anything expressible in pure Python
    should not cross the FFI boundary) rather than mutating this one, the
    same immutable-view convention pandas/polars use for their own chained
    calls:

        fastdna.count("sample.fastq.gz", k=31) \\
            .filter(min_count=5) \\
            .sort_by("frequency") \\
            .top(20) \\
            .to_pandas()

    `.total_kmers` always reports the *original* sample's total -- the
    normalization basis a filtered or truncated view must not silently
    change out from under a caller computing, say, relative abundance.
    `.distinct_kmers`/`len()` reflect the *current* view instead: after
    `.top(20)`, `len(result) == 20`, matching what `.table` itself
    contains.
    """

    def __init__(self, raw: "_core.KmerCounts", _table: Optional[pa.Table] = None):
        self._raw = raw
        # `None` means "no view derived yet -- use the Rust side's own
        # table verbatim"; set once `.filter()`/`.sort_by()`/`.top()` (or
        # anything chained off one of those) produces a derived table, so
        # a chain of several calls does not re-wrap `self._raw.table`
        # (and re-pay its zero-copy-but-not-free construction) at every
        # step.
        self._table = _table

    @property
    def table(self) -> pa.Table:
        if self._table is not None:
            return self._table
        return pa.Table.from_batches([self._raw.table])

    @property
    def qc(self) -> Dict[str, Any]:
        return self._raw.qc

    @property
    def total_kmers(self) -> int:
        return self._raw.total_kmers

    @property
    def distinct_kmers(self) -> int:
        return self.table.num_rows

    @property
    def k(self) -> int:
        return self._raw.k

    def filter(
        self, min_count: Optional[int] = None, max_count: Optional[int] = None
    ) -> "KmerCounts":
        """Returns a new `KmerCounts` restricted to k-mers whose frequency
        falls in `[min_count, max_count]` (either bound optional, both
        inclusive) -- applied on top of whatever view this one already
        holds, so `.filter(min_count=5).filter(max_count=100)` composes
        rather than the second call replacing the first.

        This is a *view* over already-counted data, not a re-count: it
        cannot recover k-mers `count()`'s own `min_count`/`max_count`
        already dropped during counting. Use it to explore a single
        `count()` result at several thresholds without re-reading the
        FASTQ file for each one.

        A negative bound is rejected rather than applied. Frequencies are
        `uint32`, so no negative threshold can exclude anything and
        `filter(min_count=-3)` silently returned the whole table -- a
        `min_count` that went negative through arithmetic looked like a
        filter that ran and kept everything. The CLI already rejects this
        (`--min-count` is `value_parser!(u32).range(1..)`); this brings the
        Python path in line.

        An *inverted* band (`min_count > max_count`) is deliberately NOT
        rejected here, unlike in the CLI. On a command line that
        combination can only be a typo; in Python both bounds are commonly
        computed from the data, and an empty view is the correct answer to
        "no k-mer falls in this band".
        """
        for name, value in (("min_count", min_count), ("max_count", max_count)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int or None, got {type(value).__name__}: {value!r}"
                )
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if min_count is None and max_count is None:
            return KmerCounts(self._raw, self.table)

        freq = self.table.column("frequency")
        mask = None
        if min_count is not None:
            mask = pc.greater_equal(freq, min_count)
        if max_count is not None:
            upper = pc.less_equal(freq, max_count)
            mask = upper if mask is None else pc.and_(mask, upper)
        return KmerCounts(self._raw, self.table.filter(mask))

    def sort_by(self, column: str = "frequency", *, descending: bool = True) -> "KmerCounts":
        """Returns a new `KmerCounts` with the current view sorted by
        `column` (any of `table.column_names`; `frequency` by default,
        matching what "the most/least common k-mers" means in practice).

        The name is checked here rather than left to pyarrow, whose own
        message for an unknown key talks about `FieldRef` -- an Arrow
        concept a FastDNA caller has no reason to know -- and buries the
        valid column names inside a dump of the table's first rows.
        """
        available = self.table.column_names
        if column not in available:
            raise ValueError(
                f"unknown column {column!r}; sort_by accepts one of {available}"
            )
        order = "descending" if descending else "ascending"
        return KmerCounts(self._raw, self.table.sort_by([(column, order)]))

    def top(self, n: int) -> "KmerCounts":
        """The `n` most frequent k-mers in the current view, as a new
        `KmerCounts` -- sugar for `.sort_by("frequency").table.slice(0, n)`
        that stays chainable, e.g. `counts.filter(min_count=5).top(20)`.

        `n` must be a non-negative `int`. The validation below is not
        defensive boilerplate: `pyarrow.Table.slice()` accepts all three of
        the values it rejects and answers each of them plausibly rather
        than raising. `slice(0, None)` means "to the end", so `top(None)`
        used to return the *entire* table -- the exact opposite of what the
        name promises, and a config key that resolved to `None` would
        silently hand back 53 million rows. `slice(0, -5)` returns an empty
        table, so an `n` that went negative through arithmetic looked like
        the biological result "no k-mer passed the filter" instead of a
        bug. A float was truncated silently. `bool` is rejected ahead of
        `int` on purpose: `isinstance(True, int)` is `True` in Python, so
        `top(True)` would otherwise slip through as `n = 1`.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(
                f"top(n) needs a non-negative int, got {type(n).__name__}: {n!r}"
            )
        if n < 0:
            raise ValueError(f"top(n) needs n >= 0, got {n}")
        return self.sort_by("frequency", descending=True)._head(n)

    def _head(self, n):
        return KmerCounts(self._raw, self.table.slice(0, n))

    def with_sequence(self) -> "KmerCounts":
        """The current view with a `kmer_sequence` column added, decoded
        from `kmer_u64`.

        `count(with_sequence=False)` (the default) skips building this
        column: it is entirely derivable from `kmer_u64` plus `k`, and
        skipping it saves ~17% of a run's wall time and roughly half its
        output size on the benchmark file (see `count()`'s own docstring).
        This method exists so turning it off by default does not take the
        capability away from anyone -- only the cost from callers who never
        read it. It does not reread the FASTQ: decoding a `u64` back into
        its `k` bases is a pure function of the integer, done here once per
        row already in the table.

        A no-op if `kmer_sequence` is already present (e.g. this view came
        from `count(with_sequence=True)`).
        """
        if "kmer_sequence" in self.table.column_names:
            return self

        bits = self.table.column("kmer_u64")
        decoded = _decode_kmers(bits, self.k)
        sequences = pa.array(decoded, type=pa.string())

        table = self.table.append_column("kmer_sequence", sequences)
        # `kmer_u64, kmer_sequence, frequency` -- the same column order
        # `count(with_sequence=True)` produces, so a caller cannot tell
        # which path built a given table from its shape alone.
        table = table.select(["kmer_u64", "kmer_sequence", "frequency"])
        return KmerCounts(self._raw, table)

    def to_pandas(self) -> "pandas.DataFrame":
        """The current view as a `pandas.DataFrame` (requires `pandas`)."""
        return self.table.to_pandas()

    def to_polars(self) -> "polars.DataFrame":
        """The current view as a `polars.DataFrame` (requires `polars`)."""
        import polars as pl

        return pl.from_arrow(self.table)

    def spectrum(self) -> Dict[int, int]:
        """`{depth: number of distinct k-mers observed at that depth}`.

        Exposes the Rust core's `KmerCounter::generate_histogram` so
        `suggest_min_count()` -- and any user code -- can work with the
        frequency spectrum directly. Always computed from the *original*
        counts, like `.total_kmers` -- not from a filtered/truncated view,
        since a spectrum missing the error peak `.filter(min_count=...)`
        would have discarded is not the spectrum `suggest_min_count()`
        needs to find the valley in.
        """
        return dict(self._raw.spectrum())

    def suggest_min_count(self) -> int:
        """The `min_count` detected from this sample's own frequency
        spectrum: the valley between the error peak (frequency 1-2) and the
        true coverage peak. See `fastdna.spectrum.suggest_min_count` for why
        a single universal default (e.g. 5) is wrong -- this differs per
        sample.
        """
        return _suggest_min_count(self.spectrum())

    def __len__(self) -> int:
        return self.distinct_kmers

    def __repr__(self) -> str:
        return f"KmerCounts(k={self.k}, distinct={self.distinct_kmers}, total={self.total_kmers})"

    def _repr_html_(self):
        """Rich display for Jupyter/IPython: a compact HTML summary (k,
        distinct/total k-mer counts) plus a preview of the first few rows
        of `.table`.

        `pandas` is a soft dependency used only for its `to_html()`
        convenience (the same soft-dependency pattern `to_pandas()` above
        already relies on); when it isn't installed, this falls back to a
        small hand-built HTML table over `.table` directly rather than
        raising -- a notebook should never see a traceback just because it
        displayed a result.
        """
        preview_rows = min(10, self.distinct_kmers)
        head = self.table.slice(0, preview_rows)

        try:
            import pandas  # noqa: F401

            table_html = head.to_pandas().to_html(index=False)
        except ImportError:
            table_html = _fallback_table_html(head)

        return (
            "<div>"
            f"<p><b>KmerCounts</b> &mdash; k={self.k}, "
            f"distinct_kmers={self.distinct_kmers:,}, "
            f"total_kmers={self.total_kmers:,}</p>"
            f"{table_html}"
            "</div>"
        )


def count(
    path: _PathLike,
    *,
    k: int = 31,
    min_count: int = 1,
    max_count: Optional[int] = None,
    min_quality: float = 20.0,
    threads: Optional[int] = None,
    progress: Optional[Union[bool, Callable[[Any], None]]] = None,
    progress_interval: int = 100_000,
    hpc: bool = False,
    with_sequence: bool = False,
) -> KmerCounts:
    """Count canonical k-mers in a single FASTQ(.gz) file.

    `hpc=True` collapses homopolymer runs (e.g. "AAAAAA" -> "A") before
    k-mer extraction. Off by default -- output is byte-for-byte identical to
    `hpc=False`. Turn it on for long-read input (Oxford Nanopore, PacBio),
    where an insertion or deletion inside a homopolymer run, not a
    substitution, is the dominant sequencing error; left uncompressed, that
    single indel shifts and corrupts every k-mer downstream of it. Short-read
    Illumina data has no need for this. Trades exact base-level positional
    correspondence with the original read for robustness to those indels:
    downstream tools that map a k-mer back to a reference coordinate (e.g.
    `fastdna.annotate`) are working with compressed-sequence offsets, not the
    original read's.

    `threads=None` uses the core's own default thread count. Passing
    `threads=0` explicitly raises `ValueError` rather than hanging -- the
    core guards against the zero-worker-tasks deadlock described in the
    design doc's error-handling section (§12, `InvalidConfig`).

    `progress` accepts `None` (silent, the default -- a library does not
    write to stdout unasked), `True` (drives a `tqdm` bar if `tqdm` is
    installed), or a callable receiving each event. The callback is invoked
    concurrently from several worker threads and its `ReadsProcessed`
    events can arrive out of order (design doc, "Callback contract"); this
    function always wraps whatever is passed in a serializing adapter
    (`fastdna._progress`) before handing it to the Rust core, so neither
    `tqdm` nor a user callback has to be thread-safe itself.

    `progress_interval` controls how often (in reads) `ReadsProcessed` is
    emitted. The default matches the core's own default; tests and small
    inputs should lower it, since a run shorter than the interval never
    emits a single event. Lowering it below the core's default batch size
    (8,192) also shrinks the internal batch size to match: a batch larger
    than `progress_interval` would coarsen progress no matter how small the
    interval asked for, so a smaller interval means a smaller batch --
    which means more, smaller batches crossing the internal channel. A
    smoother progress bar therefore trades a little throughput for it; the
    default interval leaves the default batch size untouched.

    Pressing Ctrl-C during a call raises `KeyboardInterrupt` promptly
    rather than waiting for the run to finish -- but only when `count()` is
    called from the interpreter's main thread; `PyErr_CheckSignals` is a
    no-op anywhere else, so calling this from a `threading.Thread` makes
    Ctrl-C do nothing until the run finishes on its own.

    `with_sequence=False` by default: `.table` does not include the decoded
    `kmer_sequence` column. That column is entirely derivable from
    `kmer_u64` plus `k` -- it is exactly what the Rust core's
    `decode_kmer_into` computes -- and on the benchmark file it is 1,667 MB
    against 430 MB for `kmer_u64` and 215 MB for `frequency`, 2.6x the
    other two columns combined; decoding it is also the majority of the
    ~17% of a run's wall time the export/table-build step costs. Every
    ML-facing module this package ships (`fastdna.sklearn`, `.cv`, `.gwas`)
    works in `u64` space and never reads it. Pass `with_sequence=True` to
    get it back from the count itself, or call `.with_sequence()` on an
    already-built `KmerCounts` to reconstruct it locally without rereading
    the FASTQ.
    """
    adapter = make_progress_adapter(progress)
    raw = _core.count(
        path=str(path),
        k=k,
        min_count=min_count,
        max_count=max_count,
        min_quality=min_quality,
        threads=threads,
        progress=adapter,
        progress_interval=progress_interval,
        hpc=hpc,
        with_sequence=with_sequence,
    )
    return KmerCounts(raw)


def peek(path: _PathLike, *, n_reads: int = 10_000) -> "_core.Preview":
    """Samples the first `n_reads` records of a FASTQ(.gz) file and reports
    read-length geometry, GC content, and a suggested `k` -- in
    milliseconds, without reading the rest of the file.

    `k=31` is everyone's default, and it is wrong for short reads: with
    50 bp reads it leaves 20 k-mers per read and amplifies every sequencing
    error. `peek` answers "what `k` suits *these* data?" before a long run
    commits to a wrong one.
    """
    return _core.peek(path=str(path), n_reads=n_reads)


def build_info() -> Dict[str, Any]:
    """Reports the installed version, the maximum supported k, and whether
    AVX2 is live on *this* CPU -- without which "it's slow on my Mac" is
    undiagnosable remotely.
    """
    return _core.build_info()


class Sketch:
    """A MinHash fingerprint of a FASTQ file's canonical k-mer set (design
    doc §9.4), built by :func:`sketch` or :func:`load_sketch`.

    Comparing two full k-mer sets exactly means materializing both --
    expensive, and pointless for the question a sketch answers, which is
    "how similar are these two samples", not "what exactly is in each".
    A `Sketch` keeps only the `sketch_size` smallest hash values seen; two
    sketches' overlap estimates the Jaccard similarity (or containment) of
    the full sets behind them without either one ever being fully in
    memory at once.

    `.save()`/:func:`load_sketch` exist so an N-sample comparison stops
    being O(N) FASTQ re-reads: compute each sample's sketch once, save it,
    and every later comparison loads two small files instead.
    """

    def __init__(self, raw: "_core.Sketch"):
        self._raw = raw

    @property
    def k(self) -> int:
        return self._raw.k

    @property
    def sketch_size(self) -> int:
        return self._raw.sketch_size

    def jaccard(self, other: "Sketch") -> float:
        """Symmetric similarity: the fraction of the union of both
        sketches' k-mer sets that is shared. Penalizes genome-size
        differences -- two sketches from genomes of very different sizes
        report a low Jaccard even if the smaller is entirely contained in
        the larger; see `.containment()` for the question that does not.
        Raises `ValueError` if the two sketches were built with different
        `k`.
        """
        return self._raw.jaccard(other._raw)

    def containment(self, other: "Sketch") -> float:
        """Asymmetric containment: what fraction of *this* sketch's
        k-mers also appear in `other`. `a.containment(b)` and
        `b.containment(a)` are different questions -- the one that matters
        clinically is usually "is this (small) pathogen present in this
        (large) metagenomic sample", which containment answers and
        Jaccard does not (Jaccard would report near zero from the size
        mismatch alone, even with the pathogen entirely present).
        """
        return self._raw.containment(other._raw)

    def mash_distance(self, other: "Sketch") -> float:
        """Estimates the per-base mutation rate implied by `.jaccard()`,
        under the Poisson mutation model Mash itself uses (Ondov et al.,
        2016) -- the same k-mer overlap turned into an evolutionary-
        distance estimate. Two identical sketches give `0.0`; two sketches
        sharing no k-mers give `1.0`.

        `.jaccard()` alone is a weaker claim than what "Mash-style"
        comparison implies: it says "these sketches overlap this much",
        not "these genomes differ by roughly this fraction of their
        bases". This method is the latter question; note it does not
        include Mash's own p-value against a null hypothesis, which needs
        a genome-length estimate this method does not have.
        """
        return self._raw.mash_distance(other._raw)

    def save(self, path: _PathLike) -> None:
        """Persists this sketch as JSON, for :func:`load_sketch` later."""
        self._raw.save(str(path))

    def __repr__(self) -> str:
        return f"Sketch(k={self.k}, sketch_size={self.sketch_size})"

    def _repr_html_(self):
        """Rich display for Jupyter/IPython: k, sketch_size, and a
        one-line explanation of what a sketch is -- a newcomer looking at
        a bare `Sketch` in a notebook cell will not have read the README
        first, so this spells out the "MinHash fingerprint, not a full
        k-mer set" distinction inline rather than assuming it.

        No optional dependency involved (unlike `KmerCounts._repr_html_`):
        a sketch has no table to preview, just a handful of scalars.
        """
        return (
            "<div>"
            f"<p><b>Sketch</b> &mdash; k={self.k}, sketch_size={self.sketch_size:,}</p>"
            "<p style='color: #666; font-size: 0.9em;'>"
            "A MinHash fingerprint of a FASTQ file's canonical k-mer set: "
            f"only the {self.sketch_size:,} smallest hash values seen are kept, "
            "so comparing two sketches (<code>.jaccard()</code>, "
            "<code>.containment()</code>, <code>.mash_distance()</code>) "
            "estimates similarity between the full k-mer sets behind them "
            "without materializing either one."
            "</p>"
            "</div>"
        )


def sketch(path: _PathLike, *, k: int = 21, sketch_size: int = 1000) -> Sketch:
    """Builds a MinHash sketch of a single FASTQ(.gz) file by streaming it
    -- memory stays bounded by `sketch_size` regardless of file size,
    unlike `count()`, which must hold every distinct k-mer at once.

    `k=21` (not `count()`'s `k=31`) matches the shorter k typical for
    sketching/comparison work in the literature (e.g. Mash's own default);
    pass an explicit `k` to compare against sketches built elsewhere.
    """
    return Sketch(_core.sketch(str(path), k, sketch_size))


def load_sketch(path: _PathLike) -> Sketch:
    """Loads a sketch previously written by `Sketch.save`."""
    return Sketch(_core.load_sketch(str(path)))


class FracSketch:
    """A FracMinHash ("scaled MinHash") fingerprint of a FASTQ file's
    canonical k-mer set, built by :func:`frac_sketch` or
    :func:`load_frac_sketch`.

    `Sketch`'s bottom-k keeps a *fixed count* of the smallest hash values
    no matter how large the underlying k-mer set is -- fine for `.jaccard()`
    between comparably-sized samples, but biased for `.containment()`
    between very differently-sized ones (e.g. a small pathogen sketch
    queried against a large metagenomic sample): the large sketch's ceiling
    collapses to a tiny fraction of hash space, so only a handful of the
    small sketch's hashes are ever "resolvable" against it, and the ratio
    stops being a fine-grained estimate at all (see the Rust-side
    `FracSketch` doc comment in `src/sketch.rs` for the measured example --
    a true containment of 0.5 collapsing to a reported 1.0).

    `FracSketch` keeps every hash below a fixed *threshold* instead of a
    fixed *count*, so its size scales automatically with the set's true
    cardinality (`~|set| / scale` entries) and never truncates based on
    what the *other* sketch being compared happens to look like. Prefer
    this over `Sketch` for `.containment()` queries where the two sides can
    differ a lot in size -- this is exactly `fastdna.taxonomy.classify()`'s
    and `fastdna.taxonomy.gather()`'s use case, which is why both accept a
    `scale` parameter to switch to `FracSketch` internally.
    """

    def __init__(self, raw: "_core.FracSketch"):
        self._raw = raw

    @property
    def k(self) -> int:
        return self._raw.k

    @property
    def scale(self) -> int:
        return self._raw.scale

    def containment(self, other: "FracSketch") -> float:
        """Asymmetric containment: what fraction of *this* sketch's
        k-mers also appear in `other`. Unlike `Sketch.containment`, the
        estimate does not lose resolution as `other`'s underlying set
        grows relative to this one -- see the class docstring.
        """
        return self._raw.containment(other._raw)

    def jaccard(self, other: "FracSketch") -> float:
        """Symmetric similarity: the fraction of the union of both
        sketches' k-mer sets that is shared.
        """
        return self._raw.jaccard(other._raw)

    def save(self, path: _PathLike) -> None:
        """Persists this sketch as JSON, for :func:`load_frac_sketch`
        later.
        """
        self._raw.save(str(path))

    def __repr__(self) -> str:
        return f"FracSketch(k={self.k}, scale={self.scale})"


def frac_sketch(path: _PathLike, *, k: int = 21, scale: int = 1000) -> FracSketch:
    """Builds a FracMinHash ("scaled MinHash") sketch of a single
    FASTQ(.gz) file by streaming it. Unlike :func:`sketch`, memory is
    bounded by `~|distinct k-mers| / scale`, not by a fixed constant -- the
    sketch's size tracks the file's true k-mer cardinality, which is what
    removes the size-mismatch bias `Sketch.containment` has (see
    `FracSketch`'s docstring).

    `scale=1000` by default (a k-mer's hash is kept with probability
    1/1000) -- the same order of magnitude commonly used for scaled
    sketching in the literature. A smaller `scale` keeps more hashes (finer
    resolution, more memory); a larger one keeps fewer.
    """
    return FracSketch(_core.frac_sketch(str(path), k, scale))


def load_frac_sketch(path: _PathLike) -> FracSketch:
    """Loads a FracSketch previously written by `FracSketch.save`."""
    return FracSketch(_core.load_frac_sketch(str(path)))


class KmerTable:
    """Random-access query handle over a sorted k-mer table --
    `docs/feature-gap-analysis.md`'s S1: the database-with-a-query-API gap
    against KMC3's `.kmc_pre`/`.kmc_suf`, FastK's `.ktab` and Jellyfish's
    `.jf`.

    No new file format and no separate build step: any `.parquet` file the
    Rust core's exporter writes is already a valid k-mer table (see
    `src/ktab.rs`'s module doc comment for why), so `fastdna count -o
    counts.parquet` (the CLI) followed by `KmerTable.open("counts.parquet")`
    is the whole workflow, no conversion step in between. `count()` (this
    module's Python function) currently only returns an in-memory
    `KmerCounts` rather than writing a file itself; a Python-only pipeline
    that needs a table to `open()` later should either shell out to the CLI
    or write `.table` with `pyarrow.parquet.write_table` directly (which
    will not carry this table's footer metadata, so `KmerTable.open` will
    reject it -- writing a table Python itself can later query is future
    work, not yet wired through `count()`).

    Behaves like a read-only mapping from a canonical k-mer to its
    frequency: `table["ACGT..."]`/`table.get(...)` accept either a DNA
    sequence of exactly `table.k` bases or the table's raw `kmer_u64`
    integer encoding directly, `len(table)` is the distinct-k-mer count,
    and `kmer in table` tests presence without raising.

    Point lookups (`get`/`__getitem__`) decode at most one Parquet row
    group per call, pruned via that row group's own min/max k-mer
    statistics -- not an O(1) in-memory index. Looping `get()` over many
    k-mers against the same table is fine (the OS page cache absorbs
    repeated file opens within one process), but there is currently no
    batched lookup or range-iteration entry point exposed to Python; the
    Rust-side `KmerTable::range`/`iter` exist (`src/ktab.rs`) for a caller
    who needs that and is comfortable adding a small binding, but nothing
    in Python reaches them yet.
    """

    def __init__(self, raw: "_core.KmerTable"):
        self._raw = raw

    @classmethod
    def open(cls, path: _PathLike) -> "KmerTable":
        """Opens `path` and validates it as a queryable k-mer table (footer
        metadata and per-row-group statistics only, not a full scan).
        Raises `ValueError` (specifically `fastdna._core.LoadError`, a
        `ValueError` subclass) for a file that is not one, matching every
        other `*.load`/`.open` entry point in this package.
        """
        return cls(_core.KmerTable.open(str(path)))

    @property
    def k(self) -> int:
        return self._raw.k

    def get(self, kmer: Union[str, int]) -> Optional[int]:
        """The frequency recorded for `kmer`, or `None` if it is absent."""
        return self._raw.get(kmer)

    def __getitem__(self, kmer: Union[str, int]) -> int:
        return self._raw[kmer]

    def __contains__(self, kmer: Union[str, int]) -> bool:
        return kmer in self._raw

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"KmerTable(k={self.k}, len={len(self)})"

    def union(self, *others: "KmerTable", output: _PathLike, combine: str = "sum") -> "KmerTable":
        """Every k-mer present in `self` or any of `others`, written to
        `output` and reopened as a new `KmerTable` -- the Python-visible
        counterpart of `fastdna union` / `setops::union`
        (`docs/feature-gap-analysis.md`'s S2). Streams a linear merge-join
        over every table's already-sorted rows rather than materializing
        any of them in memory (see `src/setops.rs`'s module doc comment).

        `combine` ("sum", the default, "min" or "max") decides how several
        tables' own frequencies for the same k-mer are folded into the
        result's single `frequency` column; "sum" is the natural "combine
        these samples into one" reading. The result is itself a valid
        `KmerTable` -- set operations compose, e.g. `a.union(b, output=...)
        .difference(c, output=...)`.
        """
        raw = _core.ktab_union([self._raw, *(o._raw for o in others)], str(output), combine)
        return KmerTable(raw)

    def intersect(self, *others: "KmerTable", output: _PathLike, combine: str = "min") -> "KmerTable":
        """Only k-mers present in `self` *and* every one of `others`,
        written to `output` and reopened as a new `KmerTable` -- the
        Python-visible counterpart of `fastdna intersect` / `setops::
        intersect`.

        `combine` ("min", the default, "sum" or "max") folds every table's
        own count for a shared k-mer into the result's single `frequency`
        column; "min" is the conservative reading ("this k-mer's support is
        only as strong as its rarest observation"), matching kmc_tools'
        own default reducer for `simple ... intersect`.
        """
        raw = _core.ktab_intersect([self._raw, *(o._raw for o in others)], str(output), combine)
        return KmerTable(raw)

    def difference(
        self, *subtract: "KmerTable", output: _PathLike, max_subtract_count: int = 0
    ) -> "KmerTable":
        """Every k-mer in `self` that is absent from every table in
        `subtract`, or that never exceeds `max_subtract_count` in any of
        them -- the reference-subtraction/host-removal use case (see
        `setops::diff`'s doc comment for the full rationale, including why
        a nonzero threshold matters for a real host genome). Written to
        `output` and reopened as a new `KmerTable` -- the Python-visible
        counterpart of `fastdna diff` / `setops::diff`.

        Kept rows carry `self`'s own original counts, never blended with
        `subtract`'s -- this filters `self`'s table, it does not combine
        the two sides.
        """
        raw = _core.ktab_diff(
            self._raw, [t._raw for t in subtract], str(output), max_subtract_count
        )
        return KmerTable(raw)

    def filter_reads(
        self,
        inputs: Union[_PathLike, Sequence[_PathLike]],
        *,
        mode: str,
        output: _PathLike,
        min_fraction: float = 0.1,
    ) -> "_core.FilterStats":
        """Streams one or more FASTQ/FASTA files against this table and
        writes reads that should be kept to `output` -- the Python-visible
        counterpart of `fastdna filter` (`docs/feature-gap-analysis.md`'s
        S4). Output is always FASTQ, even when every input was FASTA (the
        existing synthetic-quality convention `src/fastq.rs`'s FASTA reader
        already establishes), gzipped iff `output`'s own extension says so.

        `mode="keep"` writes only reads that match this table (targeted
        enrichment: keep only reads that look like this organism/panel);
        `mode="discard"` writes only reads that do *not* match it
        (host/contaminant removal: remove this reference's reads, keep the
        rest of the sample).

        A read "matches" when at least `min_fraction` of its own canonical
        k-mers are found in this table (`>=`, inclusive -- a read exactly at
        the threshold matches). A read with no k-mers of its own (shorter
        than `self.k`, or entirely ambiguous bases) never matches,
        regardless of `min_fraction` -- see `src/read_filter.rs`'s module
        doc comment for why.

        Single-end: each of `inputs` is filtered independently, read by
        read, and several files are filtered as one concatenated stream
        into `output` -- the same convention `fastdna count`'s own
        multi-file input already uses. Passing a sample's R1 and R2 files
        both in `inputs` here filters each mate independently and can
        desynchronize them (a mate written while its partner is silently
        dropped) -- for paired-end input, use `filter_reads_paired`
        instead, which filters both mates in lock step and decides once
        per pair. See `src/read_filter.rs`'s module doc comment for the
        full explanation of both modes.
        """
        if isinstance(inputs, (str, os.PathLike)):
            inputs = [inputs]
        return _core.filter_reads(
            self._raw, [str(p) for p in inputs], mode, str(output), min_fraction
        )

    def filter_reads_paired(
        self,
        inputs: Union[_PathLike, Sequence[_PathLike]],
        inputs2: Union[_PathLike, Sequence[_PathLike]],
        *,
        mode: str,
        output: _PathLike,
        output2: _PathLike,
        min_fraction: float = 0.1,
    ) -> "_core.PairedFilterStats":
        """Streams synchronized R1/R2 FASTQ/FASTA file pairs against this
        table and writes pairs that should be kept to `output`/`output2` --
        the paired-end counterpart of `filter_reads` (`fastdna filter
        --input2 ... --output2 ...`, `docs/feature-gap-analysis.md`'s S4,
        `src/read_filter.rs`).

        `inputs`/`inputs2` are each concatenated into one stream first (the
        same multi-file convention `filter_reads`'s own `inputs` already
        uses), and it is those two streams that are paired, record by
        record -- they do not need to hold the same number of *files*, only
        the same *total* record count. Either may be a single path-like
        value instead of a list, wrapped in a one-element list the same way
        `filter_reads` already does.

        A pair is kept or discarded as *one unit*: `mode="keep"` writes a
        pair if *either* mate matches this table; `mode="discard"` writes a
        pair only if *neither* mate matches it -- never independently per
        mate, which is what two separate `filter_reads` calls over
        `inputs`/`inputs2` would do instead, and which can desynchronize
        the two output files (a mate written while its partner is silently
        dropped). Both mates of a kept pair are always written together, to
        `output`/`output2` respectively, so the two output files can never
        drift out of sync with each other. See `src/read_filter.rs`'s
        module doc comment ("Paired-end (R1/R2) synchronized filtering")
        for the full design and why "either mate matches" (not "both") is
        the standard convention real pipelines (BBDuk, `kmc_tools filter`)
        rely on.

        `min_fraction` and the "a read with no k-mers of its own never
        matches" rule are exactly as `filter_reads` documents, applied to
        each mate independently before the pair-level OR.

        Raises `ValueError` if the two input streams desynchronize -- one
        side has more total records than the other, so they cannot be kept
        paired past that point -- or if `output` and `output2` resolve to
        the same file.
        """
        if isinstance(inputs, (str, os.PathLike)):
            inputs = [inputs]
        if isinstance(inputs2, (str, os.PathLike)):
            inputs2 = [inputs2]
        return _core.filter_reads_paired(
            self._raw,
            [str(p) for p in inputs],
            [str(p) for p in inputs2],
            mode,
            str(output),
            str(output2),
            min_fraction,
        )

    def profile_reads(
        self,
        inputs: Union[_PathLike, Sequence[_PathLike]],
        *,
        output: _PathLike,
        summary: _PathLike = "read_profile_summary.parquet",
    ) -> "_core.ProfileStats":
        """Streams one or more FASTQ/FASTA files against this table and
        writes each read's k-mer profile -- the Python-visible counterpart
        of `fastdna profile` (`docs/feature-gap-analysis.md`'s S3,
        `src/read_profile.rs`).

        Two Parquet files are written:

        - `output`: the RLE-compressed per-read profile (`read_id`,
          `start`, `run_length`, `count`) -- read position `i` maps to the
          count this table records for the k-mer starting at position `i`
          of that read. See `src/read_profile.rs`'s module doc comment for
          why this is run-length-encoded rather than one row per base (real
          sequencing data yields long runs of identical counts, the same
          regularity FastK's own `.prof` format exploits), and for the
          lossless round-trip this format guarantees.
        - `summary` (defaults alongside `output`): one row per read --
          `read_id`, `n_kmers`, `n_present_kmers` (how many of that read's
          own canonical k-mers were found in this table at all),
          `min_count`, `median_count`, `max_count` over the full per-position
          count sequence (including positions whose k-mer was entirely
          absent, i.e. count `0`) -- enough to drive error detection and QV
          estimation without decompressing a single RLE run.

        A read shorter than `self.k` (or entirely ambiguous bases) yields no
        profile rows and an all-null summary row, not an error -- ordinary
        input, not malformed input.

        Single-end only, the same scope `filter_reads` documents: each of
        `inputs` is profiled independently, read by read, and several files
        are profiled as one concatenated stream into the same pair of
        output files.
        """
        if isinstance(inputs, (str, os.PathLike)):
            inputs = [inputs]
        return _core.profile_reads(
            self._raw, [str(p) for p in inputs], str(output), str(summary)
        )


def compare(path_a: _PathLike, path_b: _PathLike, *, k: int = 21, sketch_size: int = 1000) -> float:
    """Sugar for building two sketches and comparing them in one call:
    `fastdna.compare(a, b)` is `fastdna.sketch(a).jaccard(fastdna.sketch(b))`.

    Convenient for a one-off comparison; for comparing the same sample
    against many others, build and save each sketch once instead (see
    `Sketch.save`/`load_sketch`) rather than re-reading any FASTQ file
    more than once.
    """
    return sketch(path_a, k=k, sketch_size=sketch_size).jaccard(sketch(path_b, k=k, sketch_size=sketch_size))


def compare_all(
    paths: Sequence[_PathLike], *, k: int = 21, sketch_size: int = 1000, metric: str = "jaccard"
) -> pa.Table:
    """Builds a sketch for each of `paths` once, then compares every pair,
    returning a `pyarrow.Table` in long format (`sample_a`, `sample_b`,
    the metric column) -- one row per unordered pair (`n*(n-1)/2` rows for
    `n` paths), since both metrics are symmetric (asking for both
    orderings of a pair would be redundant).

    Building N sketches once and comparing them pairwise is O(N) FASTQ
    reads plus O(N^2) sketch comparisons -- cheap, since each comparison
    is O(sketch_size), not O(genome size). Comparing full k-mer sets
    directly, pair by pair, would be O(N^2) FASTQ reads: the exact cost
    sketching exists to avoid.

    `metric` is `"jaccard"` (symmetric similarity, the default) or
    `"mash_distance"` (Poisson-model evolutionary distance -- see
    `Sketch.mash_distance`).

    The result is a plain `pyarrow.Table`, not a `KmerCounts`: it already
    composes with `.sort_by()`/DuckDB/Polars on its own, and inventing a
    second chainable wrapper type for one function would not add anything
    those tools do not already give it.
    """
    if metric not in ("jaccard", "mash_distance"):
        raise ValueError(f"metric must be 'jaccard' or 'mash_distance', got {metric!r}")

    str_paths = [str(p) for p in paths]
    # The Rust-side sketch objects, unwrapped once. `Sketch.jaccard` /
    # `Sketch.mash_distance` are one-line forwarders to exactly these, so
    # calling them per pair added a Python frame and two `._raw` attribute
    # lookups to each of the `n*(n-1)/2` comparisons -- 19,900 of each for
    # a 200-sample cohort, against `n` lookups here. Same Rust call, same
    # value, same `ValueError` on a mismatched `k` (it is raised by the
    # Rust method, not by the wrapper).
    raw_sketches = [sketch(p, k=k, sketch_size=sketch_size)._raw for p in str_paths]

    sample_a, sample_b, values = [], [], []
    for i, path in enumerate(str_paths):
        # Bound once per row instead of once per pair, and the row's
        # results are extended in one call rather than appended one at a
        # time: `n` bindings and `3n` list operations replace `n*(n-1)/2`
        # of each.
        compare = getattr(raw_sketches[i], metric)
        others = raw_sketches[i + 1:]
        values.extend([compare(other) for other in others])
        sample_a.extend([path] * len(others))
        sample_b.extend(str_paths[i + 1:])

    return pa.table({"sample_a": sample_a, "sample_b": sample_b, metric: values})


def estimate_cardinality(path: _PathLike, *, k: int = 31, precision: int = 14) -> float:
    """Estimates the number of *distinct* canonical k-mers across an
    entire FASTQ(.gz) file using HyperLogLog, in a fixed, small amount of
    memory (`2**precision` bytes, 16 KB at the default) regardless of
    file size.

    This is a different question from `peek().sample_distinct_kmers`,
    which is exact but only covers a sampled prefix: this covers the
    whole file (at the cost of a full streaming pass, the same I/O
    `count()` itself pays), with a small, known error instead of exact
    precision -- `precision=14`'s standard error is ~0.8%. Useful for a
    file too large to `count()` exactly in available memory, when
    "roughly how many distinct k-mers" is enough to plan around.
    """
    return _core.estimate_cardinality(str(path), k, precision)


def estimate_spectrum(
    paths: Union[_PathLike, Sequence[_PathLike]],
    *,
    k: int = 31,
    precision: int = 14,
    max_frequency: Optional[int] = None,
) -> Dict[int, int]:
    """Estimates the k-mer frequency spectrum -- how many distinct k-mers
    occur exactly once, twice, ... -- across one or more FASTQ(.gz)/
    FASTA(.gz) files, in one streaming pass and a fixed, small amount of
    memory (`2**precision * 16` bytes, 256 KB at the default) regardless of
    input size, via an ntCard-style sketch (`docs/feature-gap-analysis.md`'s
    S7(a); Rust core in `src/ntcard.rs`).

    This is the same question `KmerCounts.spectrum()` answers exactly, from
    a completed `count()`: `{depth: distinct k-mers observed at that
    depth}`. The two are directly interchangeable wherever a caller accepts
    either -- in particular, `fastdna.genomescope.profile_genome(spectrum,
    k=k)` fits its coverage model over either shape the same way. Useful
    when a file is too large to `count()` in available memory (or the exact
    table is simply not needed), the same trade `estimate_cardinality`
    already makes for the distinct-k-mer count alone.

    `paths` is a single path or a sequence of paths (several lanes of the
    same sample are aggregated into one spectrum, matching `count`'s own
    multi-file `--input`); `"-"` means standard input.

    Accuracy is a per-run, measured property, not a guaranteed bound on
    every input -- see `src/ntcard.rs`'s module doc comment for the actual
    numbers this implementation was measured at, and why f1 (the
    error/noise class) is the hardest one to estimate precisely.
    """
    if isinstance(paths, (str, os.PathLike)):
        path_list = [str(paths)]
    else:
        path_list = [str(p) for p in paths]
    return _core.estimate_spectrum(path_list, k, precision, max_frequency)


# Imported at the end of the module, deliberately: `cohort_counts.py` does
# `from fastdna import _column_as_array` at its own import time, which
# needs that name to already be bound on this module. Everything
# `cohort_counts.count_cohort()` calls at runtime (`fastdna.count`) is
# resolved when it is actually called, not at import time, so only the
# early binding matters here.
from .cohort_counts import CohortCounts, count_cohort  # noqa: E402

# `fastdna.audit(...)` and `fastdna.explain(...)` -- the two functions this
# package's own differentiator (docs/audit/ml-gaps.md's "unified thesis")
# is built around -- were previously reachable only via `from fastdna.audit
# import audit` / `from fastdna.explain import explain`, one import per
# function, from a submodule whose name collides with the function itself
# (`fastdna.audit.audit`, not `fastdna.audit(...)`). Re-exported here so
# the natural spelling works; neither module imports anything from this
# one at its own import time, so this carries none of `cohort_counts`'s
# circular-import constraint above -- it is placed at the end purely for
# consistency with that re-export, not because it needs to be.
from .audit import audit  # noqa: E402
from .explain import explain  # noqa: E402
