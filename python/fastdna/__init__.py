"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

import pyarrow as pa
import pyarrow.compute as pc

from . import _core
from ._progress import make_progress_adapter
from .spectrum import suggest_min_count as _suggest_min_count

__version__ = _core.__version__


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

    def __init__(self, raw, _table=None):
        self._raw = raw
        # `None` means "no view derived yet -- use the Rust side's own
        # table verbatim"; set once `.filter()`/`.sort_by()`/`.top()` (or
        # anything chained off one of those) produces a derived table, so
        # a chain of several calls does not re-wrap `self._raw.table`
        # (and re-pay its zero-copy-but-not-free construction) at every
        # step.
        self._table = _table

    @property
    def table(self):
        if self._table is not None:
            return self._table
        return pa.Table.from_batches([self._raw.table])

    @property
    def qc(self):
        return self._raw.qc

    @property
    def total_kmers(self):
        return self._raw.total_kmers

    @property
    def distinct_kmers(self):
        return self.table.num_rows

    @property
    def k(self):
        return self._raw.k

    def filter(self, min_count=None, max_count=None):
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
        """
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

    def sort_by(self, column="frequency", *, descending=True):
        """Returns a new `KmerCounts` with the current view sorted by
        `column` (any of `table.column_names`; `frequency` by default,
        matching what "the most/least common k-mers" means in practice).
        """
        order = "descending" if descending else "ascending"
        return KmerCounts(self._raw, self.table.sort_by([(column, order)]))

    def top(self, n):
        """The `n` most frequent k-mers in the current view, as a new
        `KmerCounts` -- sugar for `.sort_by("frequency").table.slice(0, n)`
        that stays chainable, e.g. `counts.filter(min_count=5).top(20)`.
        """
        return self.sort_by("frequency", descending=True)._head(n)

    def _head(self, n):
        return KmerCounts(self._raw, self.table.slice(0, n))

    def to_pandas(self):
        """The current view as a `pandas.DataFrame` (requires `pandas`)."""
        return self.table.to_pandas()

    def to_polars(self):
        """The current view as a `polars.DataFrame` (requires `polars`)."""
        import polars as pl

        return pl.from_arrow(self.table)

    def spectrum(self):
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

    def suggest_min_count(self):
        """The `min_count` detected from this sample's own frequency
        spectrum: the valley between the error peak (frequency 1-2) and the
        true coverage peak. See `fastdna.spectrum.suggest_min_count` for why
        a single universal default (e.g. 5) is wrong -- this differs per
        sample.
        """
        return _suggest_min_count(self.spectrum())

    def __len__(self):
        return self.distinct_kmers

    def __repr__(self):
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
    path,
    *,
    k=31,
    min_count=1,
    max_count=None,
    min_quality=20.0,
    threads=None,
    progress=None,
    progress_interval=100_000,
):
    """Count canonical k-mers in a single FASTQ(.gz) file.

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
    )
    return KmerCounts(raw)


def peek(path, *, n_reads=10_000):
    """Samples the first `n_reads` records of a FASTQ(.gz) file and reports
    read-length geometry, GC content, and a suggested `k` -- in
    milliseconds, without reading the rest of the file.

    `k=31` is everyone's default, and it is wrong for short reads: with
    50 bp reads it leaves 20 k-mers per read and amplifies every sequencing
    error. `peek` answers "what `k` suits *these* data?" before a long run
    commits to a wrong one.
    """
    return _core.peek(path=str(path), n_reads=n_reads)


def build_info():
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

    def __init__(self, raw):
        self._raw = raw

    @property
    def k(self):
        return self._raw.k

    @property
    def sketch_size(self):
        return self._raw.sketch_size

    def jaccard(self, other):
        """Symmetric similarity: the fraction of the union of both
        sketches' k-mer sets that is shared. Penalizes genome-size
        differences -- two sketches from genomes of very different sizes
        report a low Jaccard even if the smaller is entirely contained in
        the larger; see `.containment()` for the question that does not.
        Raises `ValueError` if the two sketches were built with different
        `k`.
        """
        return self._raw.jaccard(other._raw)

    def containment(self, other):
        """Asymmetric containment: what fraction of *this* sketch's
        k-mers also appear in `other`. `a.containment(b)` and
        `b.containment(a)` are different questions -- the one that matters
        clinically is usually "is this (small) pathogen present in this
        (large) metagenomic sample", which containment answers and
        Jaccard does not (Jaccard would report near zero from the size
        mismatch alone, even with the pathogen entirely present).
        """
        return self._raw.containment(other._raw)

    def mash_distance(self, other):
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

    def save(self, path):
        """Persists this sketch as JSON, for :func:`load_sketch` later."""
        self._raw.save(str(path))

    def __repr__(self):
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


def sketch(path, *, k=21, sketch_size=1000):
    """Builds a MinHash sketch of a single FASTQ(.gz) file by streaming it
    -- memory stays bounded by `sketch_size` regardless of file size,
    unlike `count()`, which must hold every distinct k-mer at once.

    `k=21` (not `count()`'s `k=31`) matches the shorter k typical for
    sketching/comparison work in the literature (e.g. Mash's own default);
    pass an explicit `k` to compare against sketches built elsewhere.
    """
    return Sketch(_core.sketch(str(path), k, sketch_size))


def load_sketch(path):
    """Loads a sketch previously written by `Sketch.save`."""
    return Sketch(_core.load_sketch(str(path)))


def compare(path_a, path_b, *, k=21, sketch_size=1000):
    """Sugar for building two sketches and comparing them in one call:
    `fastdna.compare(a, b)` is `fastdna.sketch(a).jaccard(fastdna.sketch(b))`.

    Convenient for a one-off comparison; for comparing the same sample
    against many others, build and save each sketch once instead (see
    `Sketch.save`/`load_sketch`) rather than re-reading any FASTQ file
    more than once.
    """
    return sketch(path_a, k=k, sketch_size=sketch_size).jaccard(sketch(path_b, k=k, sketch_size=sketch_size))


def compare_all(paths, *, k=21, sketch_size=1000, metric="jaccard"):
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
    sketches = [sketch(p, k=k, sketch_size=sketch_size) for p in str_paths]

    sample_a, sample_b, values = [], [], []
    for i in range(len(str_paths)):
        for j in range(i + 1, len(str_paths)):
            value = getattr(sketches[i], metric)(sketches[j])
            sample_a.append(str_paths[i])
            sample_b.append(str_paths[j])
            values.append(value)

    return pa.table({"sample_a": sample_a, "sample_b": sample_b, metric: values})


def estimate_cardinality(path, *, k=31, precision=14):
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
