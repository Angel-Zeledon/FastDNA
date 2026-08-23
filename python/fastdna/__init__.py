"""FastDNA -- a fast genomic k-mer counter.

This module re-exports the small FFI surface defined in `src/ffi.rs`
(compiled as the `fastdna._core` extension module) and adds nothing heavy:
anything that can be expressed in pure Python lives here instead of crossing
the Rust/Python boundary, per the packaging design (docs/superpowers/specs/
2026-08-22-fastdna-python-design.md, §9).
"""

from . import _core

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

    def __len__(self):
        return self.distinct_kmers

    def __repr__(self):
        return f"KmerCounts(distinct={self.distinct_kmers}, total={self.total_kmers})"


def count(path, *, k=31, min_count=1, max_count=None, min_quality=20.0, threads=None):
    """Count canonical k-mers in a single FASTQ(.gz) file.

    `threads=None` uses the core's own default thread count. Passing
    `threads=0` explicitly raises `ValueError` rather than hanging -- the
    core guards against the zero-worker-tasks deadlock described in the
    design doc's error-handling section (§12, `InvalidConfig`).
    """
    raw = _core.count(
        str(path),
        k,
        min_count,
        max_count,
        min_quality,
        threads,
    )
    return KmerCounts(raw)
