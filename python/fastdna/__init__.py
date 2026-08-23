"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

from . import _core
from ._progress import make_progress_adapter
from .spectrum import suggest_min_count as _suggest_min_count

__version__ = _core.__version__


class KmerCounts:
    """The result of :func:`count`.

    Wraps the Rust-side `_core.KmerCounts` object, whose `.table` getter
    already returns a zero-copy `pyarrow.RecordBatch` (via Arrow's C Data
    Interface). This class only wraps that single batch into a
    `pyarrow.Table` -- itself a cheap reference wrap, not a copy -- so the
    public API matches the design doc's `r.table # pyarrow.Table`.
    """

    def __init__(self, raw):
        self._raw = raw

    @property
    def table(self):
        import pyarrow as pa

        return pa.Table.from_batches([self._raw.table])

    @property
    def qc(self):
        return self._raw.qc

    @property
    def total_kmers(self):
        return self._raw.total_kmers

    @property
    def distinct_kmers(self):
        return self._raw.distinct_kmers

    @property
    def k(self):
        return self._raw.k

    def spectrum(self):
        """`{depth: number of distinct k-mers observed at that depth}`.

        Exposes the Rust core's `KmerCounter::generate_histogram` so
        `suggest_min_count()` -- and any user code -- can work with the
        frequency spectrum directly.
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
