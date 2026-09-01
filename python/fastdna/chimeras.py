"""fastdna.chimeras -- composition-based chimera detection in (meta)genome
assemblies: finds places within a single contig where the local
tetranucleotide composition shifts abruptly, the signature of a long-read
assembler having fused sequence from two unrelated organisms into one
contig.

    from fastdna.chimeras import scan_chimeras

    # threshold=None: the unfiltered divergence profile -- see the warning
    # below on why this module ships with no calibrated default threshold.
    candidates = scan_chimeras("mag_001.fasta", window_size=2000, step=200)
    print(candidates.to_pandas())

This is a thin surface over `src/chimera_scan.rs`, which holds the
algorithm and every design decision (Jensen-Shannon divergence between
canonical k-mer frequency distributions on either side of a candidate
breakpoint, clustering flagged candidates into one row per real event,
parallelized across contigs); see that module's own doc comment for the
full picture, including how this complements (does not replace)
read-mapping-based tools like `anvi-script-find-misassemblies`.

**Calibration result -- read this before trusting a flagged breakpoint.**
This module's own calibration run (real synthetic chimeras built from real
reference genomes at a controlled taxonomic distance, checked against a
negative control of real, non-chimeric genomes -- see `src/chimera_scan.rs`'s
module doc comment for the full numbers) found **no window size or
threshold that gives both usable sensitivity and an acceptable
false-positive rate**. Every setting that catches most real chimeras also
flags the majority of ordinary, non-chimeric genome content; every setting
with a clean negative control misses most chimeras, and misses essentially
all same-domain-different-phylum ones. Composition alone is not, on its
own, a sufficient signal for reliable multi-domain chimera calling against
real assemblies -- this is a tested, correct composition-divergence
primitive, not a validated chimera classifier. There is deliberately no
calibrated numeric default for `threshold` -- it defaults to `None`,
which returns the raw unfiltered profile rather than silently
substituting a value this calibration run found does not actually work,
so this function cannot be called as if a safe default existed.
Read-coverage evidence is likely a necessary complement, not an optional
enhancement; no CLI subcommand or standalone "chimera report" was built on
top of this pending that (or some other way to close the gap).

**`window_size` matters more than it looks like it should, independent of
the finding above.** A window that is too small has enough multinomial
sampling noise in its own k-mer counts to produce a large Jensen-Shannon
divergence against an *identical-composition* neighbor purely by chance --
`src/chimera_scan.rs`'s own unit tests measured ~0.35 bits of noise-floor
divergence at a 200-base window and k=4, and the calibration run measured
noise-floor divergence exceeding 0.3 bits even at a 1000-base window for
some real genome pairs -- comparable in magnitude to real compositional
shifts at the same window size.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import pyarrow as pa

from . import _core

__all__ = ["scan_chimeras"]

# A path accepted anywhere in this module: a `str`, or anything implementing
# `os.PathLike` (e.g. `pathlib.Path`) -- matches `fastdna/__init__.py`'s own
# `_PathLike` convention.
_PathLike = Union[str, os.PathLike]


def scan_chimeras(
    paths: Union[_PathLike, Sequence[_PathLike]],
    *,
    window_size: int,
    step: int,
    k: int = 4,
    threshold: Optional[float] = None,
) -> pa.Table:
    """Scans every contig of one or more assembly FASTA/FASTQ(.gz) files
    for composition-based chimera candidates.

    Parameters
    ----------
    paths : path-like, or a sequence of path-like
        One assembly file, or several (e.g. every MAG in a directory,
        listed explicitly -- this function does not itself expand a
        directory; a caller sweeping a whole directory should glob it
        first, e.g. ``glob.glob("mags/*.fasta")``). Each record in each
        file is treated as one contig; a multi-contig MAG FASTA is exactly
        the expected input shape.
    window_size : int
        Number of bases compared on each side of a candidate breakpoint.
        Must be at least `k`. See this module's docstring for why picking
        this too small produces divergence from sampling noise alone, not
        real compositional structure -- there is no library-wide default
        because the right value depends on `k` and on how short the
        candidate contigs are; use a value from the calibration sweep for
        this project's synthetic ground truth (see the calibration
        report), or run your own sweep against reference genomes of your
        own study system.
    step : int
        Distance in bases between one candidate breakpoint and the next.
        Must be at least 1. A smaller step gives finer breakpoint
        localization at proportionally higher runtime; it does not change
        the noise floor `window_size` controls.
    k : int, default 4
        K-mer size for the composition vector. Defaults to `4`
        (tetranucleotide composition), this technique's namesake and the
        value this project's calibration was run at. Must be in `1..=32`.
    threshold : float, optional
        Minimum Jensen-Shannon divergence (bits, base-2 log, bounded
        `[0, 1]`) for a candidate breakpoint to be flagged and returned.
        `None` (the default) disables flagging: every candidate breakpoint
        on the `window_size`/`step` grid is returned, one row each, with
        `confidence` equal to `divergence` -- this is the *unfiltered*
        mode a calibration sweep needs, not the mode a caller looking for
        actual chimera candidates in a MAG collection wants. Passing a
        threshold from a calibration run (rather than an arbitrary guess)
        is what makes the `confidence` column below meaningful.

    Returns
    -------
    pyarrow.Table
        One row per candidate breakpoint, columns:

        * ``contig_id`` -- ``"{file stem}::{header accession}"``, unique
          across every file in `paths`.
        * ``position`` -- 0-based position in the contig where the
          "before" window ends and the "after" window begins.
        * ``divergence`` -- the raw Jensen-Shannon divergence (bits,
          `[0, 1]`) at that position: the magnitude of the compositional
          shift.
        * ``confidence`` -- `[0, 1]`, `0.0` right at `threshold`'s
          decision boundary rising linearly to `1.0` at the maximum
          possible divergence; equals `divergence` directly when
          `threshold` is `None`. See `src/chimera_scan.rs::Breakpoint::
          confidence`'s doc comment for the exact rule.

        A plain long/tidy table -- built to be joined against taxonomic
        annotation by ``contig_id``, not to be read as one row per contig.

    Raises
    ------
    fastdna._core.InvalidConfigError
        If `window_size`, `step`, `k`, or `threshold` is out of range
        (`window_size < k`, `step < 1`, `k` outside `1..=32`, or
        `threshold` outside `[0.0, 1.0]`), before any file is opened.
    """
    if isinstance(paths, (str, os.PathLike)):
        path_list = [str(paths)]
    else:
        path_list = [str(p) for p in paths]

    batch = _core.scan_chimeras(path_list, window_size, step, k, threshold)
    return pa.Table.from_batches([batch])
