"""fastdna.provenance -- the block of evidence that lets a third party
re-run a reported number, and that states plainly which parts of it are
guarantees and which are only hints.

## Why this exists

`fastdna.audit()` exists because a published AUC is not, on its own,
believable: `python/fastdna/cv.py` and `python/fastdna/audit.py` both open
by explaining that a random cross-validation split on a clonal cohort
inflates it. A reviewer who accepts that argument and asks for the audit
table then hits the *next* unanswerable question: **which run produced this
table?** A number pasted into a methods section carries no record of which
`fastdna` was installed, which `scikit-learn` did the fitting, which
interpreter and CPU ran it, which files went in, or which parameters were
passed. Without that, "we ran `fastdna.audit()` and got a gap of 0.23" is
not a reproducible claim; it is an anecdote.

This module produces that record. `capture()` returns a frozen
:class:`Provenance` that can be printed into a paper's methods section
(`to_markdown()`), stored as JSON next to the results (`to_dict()` /
`to_json()`), and compared against a later run's block (`matches()`) to
answer "is this the same run configuration, or has something moved?".

## Relationship to the rest of the package (what is NOT duplicated here)

- **`fastdna.report.to_report()` already renders HTML.** This module
  deliberately does not grow a second HTML renderer. `to_metadata()`
  returns a flat `{label: str}` mapping that is exactly the shape
  `to_report(metadata=...)` already consumes, so a provenance block becomes
  the "Run metadata" table of the existing report rather than a competing
  artifact. `_repr_html_` here is a one-line `<pre>` wrapper for notebooks
  (the same fallback `fastdna.audit.AuditReport._repr_html_` uses when
  pandas is absent), not a report.
- **`fastdna.build_info()` already answers "which binary am I running".**
  Its `version` / `max_k` / `avx2` dict is recorded verbatim rather than
  re-derived: `max_k` is a hard behavioural bound on any result, and `avx2`
  selects a different code path in the counting engine.
- **`fastdna.audit.AuditReport` is not modified.** A `Provenance` composes
  with it from the outside -- `print(report.to_markdown() + "\\n\\n" +
  prov.to_markdown())`, or `to_report(path, metadata=prov.to_metadata())`
  -- so `audit()` can grow a `provenance` field later without this module
  changing.
- **Nothing here is timed.** `docs/BENCHMARKS.md`'s discipline against
  quoting wall-clock numbers measured on an uncontrolled machine (restated
  in `fastdna.report`'s own module docstring) applies here too: a
  provenance block records *what ran*, never *how fast it ran*.

## What is recorded, and why each item is on the list

**Versions.** `fastdna.__version__`, plus `numpy`, `pyarrow`, `scipy` and
`scikit-learn` -- the four packages whose behaviour can move a number this
library reports. `pandas`, `polars`, `matplotlib` and `duckdb` are
deliberately *not* tracked: they render, convert or plot results that are
already computed, so upgrading one of them cannot change a score. Versions
come from installed distribution metadata (`importlib.metadata`), which
does not import the package -- so capturing a block in an environment
without scipy costs nothing and reports `None` for it, rather than raising.

**Environment.** Python version and implementation, `platform.platform()`
(which on Linux includes the glibc version), and `platform.machine()` --
architecture matters because it decides which SIMD path and which BLAS
kernels run, and those change floating-point *reduction order*.

**Threads, and why they are not dismissed as timing noise.** It is tempting
to record thread counts only as a performance footnote. That is not quite
honest: a threaded BLAS splits a dot product across workers and sums the
partial results in an order that depends on the worker count, so
`OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=8` can produce results that differ
in the last few bits -- enough to flip a marginal comparison, never enough
to change a conclusion that was not already marginal. So the *requested*
thread limits are recorded (as environment variables), together with
`os.cpu_count()`, without which an unset limit is uninterpretable ("unset"
means "as many as this machine has"). What is *not* recorded is any
duration, rate or throughput -- that is the part that is genuinely noise
for reproducibility.

**Environment variables: an allowlist, never a dump.** Only a fixed,
documented set is read (see `_TRACKED_ENVIRONMENT`). Dumping `os.environ`
into a block that is meant to be pasted into a paper or committed next to
results is how API tokens get published. `FASTDNA_SPILL_DIR` is excluded on
purpose despite being a FastDNA variable: it selects *where* temporary
files go, not what is computed, and it is a filesystem path that would leak
a machine's layout into a public document.

**Inputs, by content and not by path.** A path proves nothing -- `cohort/
sample_01.fastq.gz` is a different file on every machine, and on the same
machine after someone re-runs the basecaller. Each input is therefore
recorded with a digest. See the next section for exactly how strong that
digest is, because the honest answer is "it depends, and the block says
which".

**Parameters.** Whatever mapping the caller passes, key-sorted and coerced
to JSON-representable values.

**A timestamp**, in UTC, ISO 8601, second resolution, `...Z`-suffixed.

## The input digest: two modes, and the exact guarantee of each

`capture(input_identity=...)` takes one of two values.

`"sha256"` -- the plain SHA-256 of the file's bytes, with no framing added,
so it is byte-identical to what `sha256sum` / `shasum -a 256` /
`Get-FileHash -Algorithm SHA256` print for the same file and can be checked
by a reviewer with no FastDNA installed. **Guarantee:** two files with the
same digest are the same bytes, to cryptographic confidence. **Cost:** one
full read of the file. On a 50 GB FASTQ that is minutes of I/O per sample,
and a 400-sample cohort makes it hours -- which is why it is not the
default.

`"sampled"` (the default) -- SHA-256 over a domain-separating header
(scheme name, file size, chunk size, chunk count) followed by
`sampled_chunks` evenly-spaced `chunk_bytes` windows, the first anchored at
offset 0 and the last at `size - chunk_bytes`. At the defaults that is 4 MiB
read per file regardless of how large the file is. Because the header is
part of the hashed input, a sampled digest can never collide with, or be
mistaken for, a plain SHA-256 of the same file, and changing `chunk_bytes`
or `sampled_chunks` changes every digest rather than silently producing
incomparable ones that look comparable. A file small enough that the
windows would cover it entirely is read whole instead and reported as
`"sha256"` -- so each :class:`InputFile` states its own
`digest_method`, and a block can honestly mix the two.

**What the sampled digest does and does not give you:**

- It **does** detect: a different file substituted at the same path, a
  truncation or extension, a corrupted download, a re-basecalled or
  re-trimmed FASTQ (all of which change the length, the header, or the
  first/last records), and any accidental edit that happens to land in a
  sampled window.
- It **does not** detect an edit confined to the bytes between the windows.
  On a 50 GB file at the defaults, the windows cover 4 MiB -- about 0.008%
  of it. A single flipped base in the other 99.99% is invisible to this
  digest. It is a *change detector*, not a commitment to the content.
- It is **not adversarial**. The scheme is published in this docstring;
  anyone who wants two different files with the same sampled digest can
  construct them in seconds. Use `input_identity="sha256"` for anything
  where someone has an incentive to lie.

Neither mode is claimed to be more than that, anywhere in the output: the
markdown block prints the method next to every digest and repeats the
caveat in prose, because a provenance block that overstates its guarantees
is worse than no provenance block at all -- it converts "we do not know" into
a false "we checked".

## What a provenance block does NOT prove

Stated explicitly, in the same spirit as `fastdna.cv`'s and
`fastdna.audit`'s own "what this does not do" sections:

1. **It does not make a result correct, or even reproducible.** It records
   what would have to be reconstructed. A run that depended on an unseeded
   RNG, on wall-clock, or on network state stays irreproducible with a
   perfect provenance block attached.
2. **It records the parameters it was handed, not the parameters the run
   used.** Nothing here inspects a call frame or an estimator. If the
   caller passes a stale dict, the block faithfully records the stale dict.
   The block is only as truthful as its caller; it is evidence *offered*,
   not evidence *collected*.
3. **A matching digest is not "the same data" in the biological sense, and
   a differing one is not necessarily different data.** Re-compressing the
   identical reads at a different gzip level, or re-ordering records,
   changes the bytes and therefore the digest, in both modes. Conversely,
   see the sampled-mode limits above.
4. **Versions come from installed distribution metadata, not from the
   modules that were actually imported.** An editable install pointing at a
   working tree, a shadowing copy earlier on `sys.path`, or a locally
   patched package all report the metadata version and are not detected.
5. **The numerical stack is only partly captured.** Which BLAS/LAPACK a
   wheel was linked against, the exact CPU microarchitecture beyond
   `machine`, the compiler and its flags, and any GPU involved are not
   recorded, and all of them can move floating-point results in the last
   bits.
6. **The timestamp is this machine's clock**, not a trusted one. It orders
   events on one machine; it is not proof that anything happened when it
   says it did.
7. **It is not a data-availability statement.** Recording an identity for
   `cohort/sample_01.fastq.gz` does not publish that file, and a reader who
   cannot obtain the data cannot reproduce anything no matter how complete
   this block is.

## Package convention

Import-safe with only `pyarrow` (and, in practice, the compiled `_core`
that `fastdna/__init__.py` needs) present: everything in this module comes
from the standard library, and the optional packages whose versions are
reported are queried through `importlib.metadata` *inside* the function
that needs them -- they are never imported, at module load time or at all.
"""

from __future__ import annotations

import collections.abc
import datetime
import hashlib
import html
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from . import _core

__all__ = ["SCHEMA_VERSION", "InputFile", "Provenance", "capture"]

#: Version of the dict/JSON layout `Provenance.to_dict()` emits, written
#: into every serialized block under the `"provenance_schema"` key. A block
#: stored next to results outlives the code that wrote it; a reader years
#: later needs to know which layout it is looking at without guessing from
#: which keys happen to be present.
SCHEMA_VERSION = 1

# A path accepted anywhere in this module -- the same alias, and the same
# `str()`-at-the-boundary convention, `fastdna/__init__.py` uses.
_PathLike = Union[str, os.PathLike]

# (import name, distribution name) for every package whose *behaviour* can
# change a number this library reports. Keyed in the output by the
# distribution name, because that is the string a reader has to type to
# reproduce the environment (`pip install scikit-learn`, not `sklearn`).
# See the module docstring for why pandas/polars/matplotlib/duckdb are
# absent: they present results that are already computed.
_TRACKED_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
)

# The fixed allowlist of environment variables read. Never `os.environ`
# wholesale -- see the module docstring. Each entry earns its place by
# changing either which code path runs or the order in which floating-point
# results are accumulated:
#   FASTDNA_MAX_RAM_BYTES / FASTDNA_STRATEGY -- select the counting engine's
#       in-memory vs. spill-to-disk strategy (README, "Two environment
#       variables reach the same strategy chooser").
#   MKL_/NUMEXPR_/OMP_/OPENBLAS_NUM_THREADS -- change BLAS reduction order.
#   PYTHONHASHSEED -- changes `hash()` of str/bytes, and therefore the
#       iteration order of any set or dict keyed by them.
_TRACKED_ENVIRONMENT: Tuple[str, ...] = (
    "FASTDNA_MAX_RAM_BYTES",
    "FASTDNA_STRATEGY",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)

# `<fastdna.sklearn.KmerVectorizer object at 0x7f3c8a1b2d90>` -- CPython's
# default `object.__repr__`, whose hex id is a memory address and therefore
# differs between two otherwise-identical runs. Detected so that a
# parameter holding such an object does not silently make the whole block
# non-deterministic; see `_jsonable`.
_DEFAULT_REPR = re.compile(r" object at 0x[0-9a-fA-F]+>$")


def _utc_iso(moment: datetime.datetime) -> str:
    """`moment` as `YYYY-MM-DDTHH:MM:SSZ`.

    Second resolution on purpose: sub-second precision in a block meant to
    be pasted into a methods section is noise, and the trailing `Z` is
    preferred over `+00:00` because it is the form ISO 8601 timestamps are
    conventionally written in and read back by every parser that accepts
    the offset form anyway.
    """
    return moment.astimezone(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cell(value: Any) -> str:
    """`value` rendered safely into one markdown table cell.

    A recorded parameter can hold an arbitrary `repr` (a scikit-learn
    `Pipeline`'s spans several lines) and an environment variable can hold
    anything at all. An unescaped `|` silently splits a row into extra
    columns and a raw newline ends the table early -- in both cases the
    rendered block is wrong in a way that is easy to miss and impossible to
    check, which for a document whose only job is to be checkable is the
    worst failure mode available.
    """
    text = value if isinstance(value, str) else json.dumps(value)
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _jsonable(value: Any) -> Any:
    """`value` reduced to something `json.dumps` accepts, deterministically.

    JSON scalars pass through untouched. Mappings and sequences are
    converted element-wise (mapping keys are stringified, since JSON has no
    other kind). A zero-dimensional numpy scalar -- which is what
    `numpy.float64(0.71)` is -- is unwrapped via its own `.item()`,
    duck-typed rather than by importing numpy, so this module keeps its
    "standard library only" property. That check comes *first*, ahead of
    the scalar passthrough, precisely because `numpy.float64` subclasses
    Python `float`: without it, a numpy scalar would slip through the
    `isinstance` test unconverted and leave a numpy object embedded in
    something documented as a plain dict.

    Anything else is recorded as its `repr()`, which is context for a
    reader rather than a value that can be reconstructed; a fitted
    scikit-learn estimator, for instance, reprs as its constructor call,
    which is exactly what belongs in a methods section. The one exception
    is an object relying on CPython's default `object.__repr__`: that
    embeds a memory address, which would differ between two identical runs
    and quietly break the determinism this block's whole value rests on, so
    it is replaced by the stable `<module.QualName>` form instead.
    """
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        try:
            return _jsonable(item())
        except Exception:  # pragma: no cover -- defensive: a foreign .item()
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, collections.abc.Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        # A set has no inherent order, and its iteration order depends on
        # PYTHONHASHSEED for str/bytes members -- sorted by the rendered
        # form so the same set always serializes the same way.
        if isinstance(value, (set, frozenset)):
            items.sort(key=repr)
        return items
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    text = repr(value)
    if _DEFAULT_REPR.search(text):
        cls = type(value)
        return "<{}.{}>".format(cls.__module__, cls.__qualname__)
    return text


def _package_versions() -> Dict[str, Optional[str]]:
    """`{distribution name: version or None}` for `_TRACKED_PACKAGES`.

    Read from installed distribution metadata, which -- unlike
    `import scipy; scipy.__version__` -- neither imports the package (scipy
    and scikit-learn cost hundreds of milliseconds and pull in numpy) nor
    fails when it is absent. `None` means "no installed distribution of
    that name was found", which is the honest answer for the several
    packages in the list that are optional here.

    The `sys.modules` fallback exists for the one case metadata gets wrong:
    a package that is importable but carries no distribution metadata (a
    vendored or hand-placed copy). It only consults modules that are
    *already* imported, so it never triggers an import of its own, and its
    result therefore does not depend on this function's call order.
    """
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:  # pragma: no cover -- Python < 3.8, below the floor
        return {distribution: None for _, distribution in _TRACKED_PACKAGES}

    versions: Dict[str, Optional[str]] = {}
    for module_name, distribution in _TRACKED_PACKAGES:
        try:
            versions[distribution] = str(importlib_metadata.version(distribution))
            continue
        except Exception:
            pass
        module = sys.modules.get(module_name)
        fallback = getattr(module, "__version__", None)
        versions[distribution] = str(fallback) if fallback is not None else None
    return versions


def _fastdna_version() -> str:
    """`fastdna.__version__`, or `"unknown"` if it cannot be read.

    Imported here rather than at module scope for two reasons: this module
    is a submodule of the package it is reading a version from (a module
    -scope `from . import __version__` would make the import order between
    the two load-bearing), and `fastdna/__init__.py` itself requires the
    compiled `_core` extension, which a pure-Python consumer of this module
    should not be made to depend on just to stamp a timestamp.
    """
    try:
        from . import __version__
    except Exception:
        return "unknown"
    return str(__version__)


def _fastdna_build() -> Dict[str, Any]:
    """`fastdna.build_info()` verbatim (`version`, `max_k`, `avx2`), or an
    empty dict when the compiled extension is unavailable.

    Recorded rather than re-derived: `max_k` is a hard bound on any result
    (a k above it never ran), and `avx2` reports whether this *specific*
    CPU took the vectorized path, which `platform.machine()` alone cannot
    say. Both are already this package's own answer to "which binary
    produced this", so re-implementing either here would be a second copy
    that could drift from the first.
    """
    try:
        from . import build_info

        return {str(key): _jsonable(value) for key, value in dict(build_info()).items()}
    except Exception:
        return {}


def _environment() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in _TRACKED_ENVIRONMENT}


def _sample_offsets(size: int, chunk_bytes: int, chunk_count: int) -> Tuple[int, ...]:
    """Byte offsets of the windows the sampled digest reads: `chunk_count`
    of them, evenly spaced, the first at 0 and the last ending exactly at
    the end of the file.

    Anchoring the first and last windows at the file's two ends is what
    makes the scheme detect truncation and appended content, the two ways a
    FASTQ most often changes without anyone meaning it to.
    """
    if chunk_count == 1:
        return (0,)
    span = size - chunk_bytes
    return tuple(round(index * span / (chunk_count - 1)) for index in range(chunk_count))


def _digest_file(path: str, size: int, identity: str, chunk_bytes: int, chunk_count: int) -> Tuple[str, str]:
    """`(hex digest, method)` for one file. See the module docstring's "The
    input digest" section for what each method guarantees.

    Returns `"sha256"` as the method whenever the whole file was hashed --
    including in `"sampled"` mode for a file small enough that the windows
    would have covered it anyway, since reading it in pieces would cost the
    same and give a weaker claim for no reason.
    """
    digest = hashlib.sha256()
    if identity == "sha256" or size <= chunk_bytes * chunk_count:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest(), "sha256"

    # The framing header is hashed first so a sampled digest is drawn from a
    # different space than a plain SHA-256 (it can never be mistaken for
    # one, or compared against one and appear to disagree "about the
    # content"), and so that two blocks taken with different window
    # settings never produce equal digests that would imply a match the
    # settings do not support.
    digest.update(
        "fastdna-provenance-sampled-v1\n{}\n{}\n{}\n".format(size, chunk_bytes, chunk_count).encode("ascii")
    )
    with open(path, "rb") as handle:
        for offset in _sample_offsets(size, chunk_bytes, chunk_count):
            handle.seek(offset)
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest(), "sha256-sampled"


@dataclass(frozen=True)
class InputFile:
    """One input file's recorded identity.

    Attributes
    ----------
    path : str
        The path exactly as the caller gave it, stringified. Context, not
        identity -- deliberately not resolved to an absolute path, which
        would bake one machine's directory layout (and often a username)
        into a document meant to be published, and which would differ
        between two machines holding the same file.
    size_bytes : int
        `os.stat().st_size`.
    digest : str
        Lowercase hex SHA-256, over either the whole file or the sampled
        windows -- `digest_method` says which.
    digest_method : str
        `"sha256"` (the whole file; byte-identical to what `sha256sum`
        prints) or `"sha256-sampled"` (framed sample; a change detector,
        not a commitment to the content -- see the module docstring).
    modified_utc : str
        The file's mtime, ISO 8601 UTC. **Advisory only, and deliberately
        not part of `digest`.** `touch` changes it without changing a byte,
        and copying a dataset to another machine usually changes it too --
        so including it would make the recorded identity of unchanged data
        differ between machines, which is the opposite of what a
        content-addressed identity is for. It is reported because "this
        file was rewritten yesterday" is a useful thing for a human to
        notice, not because anything here trusts it.
    """

    path: str
    size_bytes: int
    digest: str
    digest_method: str
    modified_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "digest_method": self.digest_method,
            "modified_utc": self.modified_utc,
        }

    def __repr__(self) -> str:
        return "InputFile(path={!r}, size_bytes={}, {}={}...)".format(
            self.path, self.size_bytes, self.digest_method, self.digest[:12]
        )


@dataclass(frozen=True)
class Provenance:
    """Everything :func:`capture` recorded about one run.

    Frozen for the same reason `fastdna.audit.AuditReport` is (that class's
    design note: "a report is evidence; mutating it after generating it is
    exactly what must not be possible"). The three mapping-valued fields
    are `types.MappingProxyType` rather than plain dicts, because
    `frozen=True` only stops *rebinding* a field -- a plain dict field
    would still be mutable in place, which for a block of evidence is the
    same hole with an extra step.

    Attributes
    ----------
    fastdna_version : str
        `fastdna.__version__`, or `"unknown"` if the package version could
        not be read.
    fastdna_build : Mapping[str, Any]
        `fastdna.build_info()` verbatim; empty when the compiled extension
        is unavailable.
    package_versions : Mapping[str, str or None]
        `{distribution name: version}` for the four packages that can move
        a number (see the module docstring); `None` for any not installed.
    python_version, python_implementation : str
        e.g. `"3.11.9"`, `"CPython"`.
    platform : str
        `platform.platform()` -- OS, release, and on Linux the glibc
        version.
    machine : str
        `platform.machine()` -- e.g. `"x86_64"`, `"arm64"`. Decides which
        SIMD and BLAS kernels ran, and therefore the floating-point
        reduction order.
    cpu_count : int or None
        `os.cpu_count()`. Recorded because it is what an *unset*
        `OMP_NUM_THREADS` resolves to; without it the recorded thread
        limits cannot be interpreted.
    environment : Mapping[str, str or None]
        The allowlisted environment variables (see `_TRACKED_ENVIRONMENT`);
        `None` for any that was unset. Never a dump of `os.environ`.
    inputs : tuple of InputFile
        In the order given to :func:`capture`, which is usually the sample
        order the results themselves are indexed by, and is preserved
        rather than sorted for exactly that reason.
    parameters : Mapping[str, Any]
        Whatever mapping the caller passed, key-sorted and JSON-reduced.
        Recorded as offered -- nothing here verifies it against the run
        (module docstring, "what this does not prove", item 2).
    generated_utc : str
        When :func:`capture` ran, ISO 8601 UTC to the second. The **only**
        field expected to differ between two otherwise identical runs; see
        :meth:`matches`.
    """

    fastdna_version: str
    fastdna_build: Mapping[str, Any]
    package_versions: Mapping[str, Optional[str]]
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    cpu_count: Optional[int]
    environment: Mapping[str, Optional[str]]
    inputs: Tuple[InputFile, ...]
    parameters: Mapping[str, Any]
    generated_utc: str

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A plain, JSON-representable `dict` of this block, for storing
        next to the results it describes.

        Every nested mapping is a real `dict` with sorted keys and every
        value is something `json.dumps` accepts, so
        `json.dumps(prov.to_dict())` never raises and two identical runs
        produce byte-identical JSON apart from `generated_utc`. Carries
        `"provenance_schema": SCHEMA_VERSION` so a reader can tell which
        layout it is parsing.
        """
        return {
            "provenance_schema": SCHEMA_VERSION,
            "generated_utc": self.generated_utc,
            "fastdna_version": self.fastdna_version,
            "fastdna_build": dict(sorted(self.fastdna_build.items())),
            "package_versions": dict(sorted(self.package_versions.items())),
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "environment": dict(sorted(self.environment.items())),
            "inputs": [entry.to_dict() for entry in self.inputs],
            "parameters": dict(sorted(self.parameters.items())),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """`to_dict()` rendered as JSON. `indent=None` gives the compact
        single-line form; the default is the readable one, since this file
        is normally committed next to results and read by people.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_metadata(self) -> Dict[str, str]:
        """A flat `{label: string}` mapping, ready to hand straight to
        `fastdna.report.to_report(metadata=...)`.

        That function renders its `metadata` as a two-column table of
        `str(key)`/`str(value)` pairs, so a nested structure would come out
        as a stringified dict. Flattening here -- rather than teaching
        `report.py` about this class, or growing a second HTML renderer in
        this module -- is what keeps one report artifact instead of two.
        Absent packages and unset variables are rendered as
        `"not installed"` / `"unset"` rather than `"None"`, which reads as
        a value.
        """
        flat: Dict[str, str] = {
            "fastdna version": self.fastdna_version,
            "python": "{} {}".format(self.python_implementation, self.python_version),
            "platform": self.platform,
            "machine": self.machine,
            "cpu count": "unknown" if self.cpu_count is None else str(self.cpu_count),
            "generated (UTC)": self.generated_utc,
        }
        for name, version in sorted(self.package_versions.items()):
            flat[name] = version if version is not None else "not installed"
        for key, value in sorted(self.fastdna_build.items()):
            if key != "version":  # already reported as "fastdna version"
                flat["build: {}".format(key)] = str(value)
        for name, value in sorted(self.environment.items()):
            flat["env: {}".format(name)] = value if value is not None else "unset"
        for key, value in sorted(self.parameters.items()):
            flat["param: {}".format(key)] = json.dumps(value) if not isinstance(value, str) else value
        for entry in self.inputs:
            flat["input: {}".format(entry.path)] = "{}:{} ({} bytes)".format(
                entry.digest_method, entry.digest, entry.size_bytes
            )
        return flat

    # -- rendering --------------------------------------------------------

    def to_markdown(self) -> str:
        """This block as markdown, sized to paste into a paper's methods
        section or a repository's results directory.

        Deterministic: every table's rows are key-sorted (inputs excepted,
        which keep the caller's order -- see `inputs`), so two identical
        runs differ only in the `generated_utc` line. The closing paragraph
        restating what the digests do not prove is not optional garnish; a
        digest column with no statement of its strength is exactly the
        overstatement this module exists to avoid.
        """
        lines = [
            "# Provenance",
            "",
            "Generated {} (UTC) by `fastdna.provenance` (schema {}).".format(
                self.generated_utc, SCHEMA_VERSION
            ),
            "",
            "| software | version |",
            "|---|---|",
            "| fastdna | {} |".format(_cell(self.fastdna_version)),
        ]
        for name, version in sorted(self.package_versions.items()):
            lines.append("| {} | {} |".format(name, _cell(version) if version is not None else "not installed"))
        for key, value in sorted(self.fastdna_build.items()):
            if key != "version":
                lines.append("| fastdna build: {} | {} |".format(key, _cell(value)))

        lines += [
            "",
            "| environment | value |",
            "|---|---|",
            "| python | {} {} |".format(self.python_implementation, self.python_version),
            "| platform | {} |".format(_cell(self.platform)),
            "| machine | {} |".format(_cell(self.machine)),
            "| cpu_count | {} |".format("unknown" if self.cpu_count is None else self.cpu_count),
        ]
        for name, value in sorted(self.environment.items()):
            lines.append("| {} | {} |".format(name, _cell(value) if value is not None else "unset"))

        lines += ["", "| parameter | value |", "|---|---|"]
        if self.parameters:
            for key, value in sorted(self.parameters.items()):
                lines.append("| {} | {} |".format(_cell(key), _cell(value)))
        else:
            lines.append("| (none recorded) | |")

        lines += ["", "| input | bytes | identity |", "|---|---:|---|"]
        if self.inputs:
            for entry in self.inputs:
                lines.append(
                    "| `{}` | {} | `{}:{}` |".format(
                        entry.path, entry.size_bytes, entry.digest_method, entry.digest
                    )
                )
        else:
            lines.append("| (none recorded) | | |")

        lines += ["", self._caveat()]
        return "\n".join(lines)

    def _caveat(self) -> str:
        """The prose paragraph printed under every rendering, stating what
        the digests above do and do not establish -- worded for the modes
        actually present in `inputs`, so a block that hashed every file in
        full does not carry a warning about sampling that does not apply to
        it.
        """
        methods = {entry.digest_method for entry in self.inputs}
        pieces = [
            "This block records what ran, not that the result is correct: it lists the "
            "software, environment, inputs and parameters that would have to be "
            "reconstructed to re-run it. The parameters are recorded as they were passed "
            "to `capture()`; nothing here inspected the run itself. Package versions come "
            "from installed distribution metadata, so an editable or shadowed install is "
            "not detected. The BLAS/LAPACK build, the exact CPU microarchitecture and the "
            "compiler flags are not captured, and each can move a floating-point result in "
            "its last bits."
        ]
        if "sha256-sampled" in methods:
            pieces.append(
                "Identities marked `sha256-sampled` are a SHA-256 over the file's size and "
                "a few evenly-spaced windows of it, not over all of its bytes. They detect "
                "substitution, truncation, re-generation and corruption that touches a "
                "sampled window; they do NOT detect an edit confined to the bytes between "
                "the windows, and they are trivially forgeable by anyone who knows the "
                "scheme (it is published in `fastdna.provenance`'s docstring). Re-capture "
                "with `input_identity=\"sha256\"` where a full-content guarantee is needed."
            )
        if "sha256" in methods:
            pieces.append(
                "Identities marked `sha256` are the plain SHA-256 of the whole file, "
                "byte-identical to what `sha256sum` prints, and establish byte equality to "
                "cryptographic confidence. They still say nothing about whether two "
                "differently-compressed copies of the same reads are the same data."
            )
        return "\n\n".join(pieces)

    # -- comparison -------------------------------------------------------

    def matches(self, other: "Provenance") -> bool:
        """`True` when every field except `generated_utc` is equal.

        This is the operational form of this module's central claim: two
        captures of the same run configuration, on the same machine, must
        produce the same block apart from the moment they were taken. It is
        also what a third party runs after re-executing a published
        analysis -- "did anything move?" -- with the answer's negative case
        readable by diffing `to_dict()`.
        """
        if not isinstance(other, Provenance):
            raise TypeError(
                "Provenance.matches() compares two Provenance blocks, got {}. To compare "
                "against a block loaded from JSON, compare the two dicts directly after "
                "removing their 'generated_utc' keys.".format(type(other).__name__)
            )
        mine = self.to_dict()
        theirs = other.to_dict()
        mine.pop("generated_utc")
        theirs.pop("generated_utc")
        return mine == theirs

    # -- display ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "Provenance(fastdna={}, python={}, machine={}, n_inputs={}, "
            "n_parameters={}, generated_utc={})".format(
                self.fastdna_version,
                self.python_version,
                self.machine,
                len(self.inputs),
                len(self.parameters),
                self.generated_utc,
            )
        )

    def __str__(self) -> str:
        return self.to_markdown()

    def _repr_html_(self) -> str:
        """Rich display for Jupyter/IPython: the markdown block in a `<pre>`.

        Deliberately not a styled HTML table. `fastdna.report.to_report()`
        is this package's HTML artifact, and `to_metadata()` above feeds
        this block into it; a second, competing renderer here would be the
        duplication `docs/philosophy-narrow-not-broad.md` argues against.
        This is the same `<pre>` fallback `fastdna.audit.AuditReport.
        _repr_html_` already uses when pandas is unavailable.
        """
        return "<pre>{}</pre>".format(html.escape(self.to_markdown()))


def capture(
    *,
    inputs: Optional[Iterable[_PathLike]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    input_identity: str = "sampled",
    chunk_bytes: int = 1 << 20,
    sampled_chunks: int = 4,
) -> Provenance:
    """Records the software, environment, inputs and parameters behind one
    run, and returns them as a frozen :class:`Provenance`.

    Read the module docstring before relying on any of it -- in particular
    "The input digest: two modes, and the exact guarantee of each" and
    "What a provenance block does NOT prove". This function is cheap and
    honest; it is not a verification step, and treating it as one is the
    one way to use it wrongly.

    Parameters
    ----------
    inputs : iterable of path-like, optional
        The files the run consumed -- FASTQ(.gz), Parquet, a phenotype
        table, anything whose content determines the result. Each is
        `stat`ed and digested (see `input_identity`). Order is preserved,
        because it is usually the sample order the results are indexed by.
        `None` (the default) records no inputs, which is the right call for
        a block describing only an environment.
    parameters : mapping, optional
        The parameters the run used, e.g. `{"k": 31, "n_splits": 5,
        "random_state": 0}`. Keys are stringified and sorted; values are
        reduced to JSON-representable forms (see `_jsonable`), so a numpy
        scalar or a scikit-learn estimator can be passed directly. Recorded
        as given -- nothing here checks it against anything.
    input_identity : {"sampled", "sha256"}, default "sampled"
        How each input is digested. `"sampled"` reads
        `sampled_chunks * chunk_bytes` bytes per file (4 MiB at the
        defaults) regardless of file size; `"sha256"` reads every byte and
        gives a full cryptographic content identity. The default is the
        cheap one because a 400-sample cohort of 50 GB FASTQ files makes
        the expensive one an hours-long operation, and a provenance block
        nobody runs protects nothing -- but every digest carries its method
        in the output and the rendered block states the difference in
        prose, so the cheap option is never presented as the strong one.
    chunk_bytes : int, default 1048576
        Bytes per sampled window. Ignored when `input_identity="sha256"`.
        Part of the hashed framing header, so changing it changes every
        sampled digest rather than producing incomparable digests that look
        comparable.
    sampled_chunks : int, default 4
        Number of evenly-spaced windows, the first at the start of the file
        and the last at its end. Ignored when `input_identity="sha256"`.
        Also part of the framing header.

    Returns
    -------
    Provenance

    Raises
    ------
    ValueError
        If `input_identity` is not one of the two accepted values, if
        `chunk_bytes` or `sampled_chunks` is not a positive int, or if an
        entry of `inputs` exists but is not a regular file (a directory has
        no single content identity -- digest a manifest of it instead).
    FileNotFoundError
        If an entry of `inputs` does not exist. A provenance block naming a
        file nobody can find is worse than none, so this is not softened
        into a placeholder record.

    Examples
    --------
    >>> from fastdna import provenance
    >>> prov = provenance.capture(parameters={"k": 31, "random_state": 0})
    >>> prov.matches(provenance.capture(parameters={"random_state": 0, "k": 31}))
    True

    Alongside an audit, without either module knowing about the other::

        report = fastdna.audit(estimator, paths, phenotype, n_splits=5)
        prov = provenance.capture(inputs=paths,
                                  parameters={"n_splits": 5, "estimator": estimator})
        print(report.to_markdown(), "\\n\\n", prov.to_markdown())

    Or folded into this package's existing HTML report::

        from fastdna.report import to_report
        to_report("run.html", metadata=prov.to_metadata())
    """
    if input_identity not in ("sampled", "sha256"):
        raise _core.InvalidConfigError(
            "input_identity must be 'sampled' (a framed digest over a few windows of each "
            "file -- a change detector) or 'sha256' (the whole file -- a content identity), "
            "got {!r}. See fastdna.provenance's module docstring for what each one "
            "guarantees.".format(input_identity)
        )
    for name, value in (("chunk_bytes", chunk_bytes), ("sampled_chunks", sampled_chunks)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _core.InvalidConfigError("{} must be an int >= 1, got {!r}".format(name, value))

    recorded: List[InputFile] = []
    for entry in inputs or ():
        path = os.fspath(entry) if isinstance(entry, os.PathLike) else str(entry)
        if not os.path.exists(path):
            raise _core.IoNotFoundError(
                "provenance input {!r} does not exist; a block naming a file that cannot be "
                "found records nothing checkable.".format(path)
            )
        if not os.path.isfile(path):
            raise _core.InvalidConfigError(
                "provenance input {!r} is not a regular file. A directory has no single "
                "content identity -- pass its files individually, or digest a manifest of "
                "them.".format(path)
            )
        stat = os.stat(path)
        digest, method = _digest_file(path, stat.st_size, input_identity, chunk_bytes, sampled_chunks)
        recorded.append(
            InputFile(
                path=path,
                size_bytes=int(stat.st_size),
                digest=digest,
                digest_method=method,
                modified_utc=_utc_iso(datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc)),
            )
        )

    normalized_parameters = {str(key): _jsonable(value) for key, value in (parameters or {}).items()}

    return Provenance(
        fastdna_version=_fastdna_version(),
        fastdna_build=MappingProxyType(_fastdna_build()),
        package_versions=MappingProxyType(_package_versions()),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        environment=MappingProxyType(_environment()),
        inputs=tuple(recorded),
        parameters=MappingProxyType(dict(sorted(normalized_parameters.items()))),
        generated_utc=_utc_iso(datetime.datetime.now(datetime.timezone.utc)),
    )
