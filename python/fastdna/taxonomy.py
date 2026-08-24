"""fastdna.taxonomy -- k-mer sketch-based classification and sample-identity
checks, built entirely on top of the stable MinHash sketching API in
`fastdna/__init__.py` (`Sketch`, `sketch()`, `load_sketch()`). Nothing in
this module touches the Rust core directly or adds a new file format --
per the packaging design's own rule (see `__init__.py`'s module docstring),
anything expressible in pure Python on top of the existing FFI surface
belongs here, not in `src/`.

Two questions genomics labs ask constantly, both answerable from MinHash
sketch comparisons without ever materializing a full k-mer set:

1. "What does this sample most resemble, out of a reference set?"
   (`build_reference_database` + `classify`) -- sourmash/Mash-style
   classification/screening of a query against known references (reference
   genomes, known pathogen signatures, ...).
2. "Are these two sequencing runs actually the same biological sample?"
   (`check_sample_identity`) -- a lab QC / sample-swap-detection check,
   where a mismatch is a real, actionable lab error.

A third, harder question -- "what *set* of organisms is present in this
sample" (relevant for metagenomic, multi-organism samples) -- is covered
approximately by `gather`; see its docstring for why that one is
necessarily heuristic given what a MinHash sketch can and cannot expose.
"""
from __future__ import annotations

import math
import pathlib
from typing import NamedTuple

import pyarrow as pa

from . import Sketch, sketch as _sketch

__all__ = [
    "build_reference_database",
    "classify",
    "check_sample_identity",
    "SampleIdentityResult",
    "gather",
]

_CLASSIFY_METRICS = ("containment", "jaccard")
_IDENTITY_METRICS = ("mash_distance", "jaccard")


def _default_name(path) -> str:
    """Derives a reference's display name from its file name: strips a
    trailing `.gz` (routine for FASTQ) and then any remaining suffix, so
    `"ecoli.fastq.gz"` -> `"ecoli"` rather than `"ecoli.fastq"` or
    `"ecoli.fastq.gz"` itself.
    """
    name = pathlib.Path(str(path)).name
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    return pathlib.Path(name).stem


def _resolve_sketch(path_or_sketch, *, k, sketch_size) -> Sketch:
    """Accepts either a file path (sketched here with `k`/`sketch_size`)
    or an already-built `Sketch` (returned unchanged, `k`/`sketch_size`
    ignored) -- the "don't rebuild what the caller already has" rule
    `classify()`'s docstring calls out, factored out since both `classify`
    and `gather` need it.
    """
    if isinstance(path_or_sketch, Sketch):
        return path_or_sketch
    return _sketch(str(path_or_sketch), k=k, sketch_size=sketch_size)


def build_reference_database(paths_or_dict, *, k=21, sketch_size=1000):
    """Builds a `{name: Sketch}` reference database once, up front, so
    `classify()` (or `gather()`) never re-reads a reference FASTQ file per
    query -- the same "sketch once, compare many times" principle
    `compare_all()` already applies to all-pairs comparison, applied here
    to one-query-vs-many-references.

    `paths_or_dict` accepts either an iterable of paths (names derived
    from each filename, see `_default_name`) or an explicit
    `{name: path}` dict, for when filenames aren't distinctive names on
    their own (e.g. accession numbers, or several files that would
    otherwise collide on the same derived name).

    Returns a plain `dict[str, Sketch]` -- deliberately not a bespoke
    index/database type. `Sketch.save()`/`fastdna.load_sketch()` already
    exist for persisting one sketch; a caller who wants a persisted
    *database* can save each of this dict's values under its key as a
    filename and rebuild the same dict later with
    `{name: load_sketch(path) for name, path in ...}`. Inventing a second,
    bespoke multi-sketch file format on top of that would add a format to
    maintain without adding any capability this dict-of-Sketch doesn't
    already have.
    """
    if isinstance(paths_or_dict, dict):
        items = list(paths_or_dict.items())
    else:
        items = [(_default_name(p), p) for p in paths_or_dict]

    return {name: _sketch(str(path), k=k, sketch_size=sketch_size) for name, path in items}


def _empty_score_table(name_column="name", score_column="score"):
    return pa.table({name_column: pa.array([], type=pa.string()), score_column: pa.array([], type=pa.float64())})


def classify(
    query_path_or_sketch,
    reference_db,
    *,
    k=21,
    sketch_size=1000,
    top_n=5,
    metric="containment",
    min_score=0.0,
):
    """Ranks the references in `reference_db` (a `{name: Sketch}` mapping,
    as returned by `build_reference_database`, or any raw dict of the same
    shape) by how well each one explains the query. Returns the `top_n`
    matches as a `pyarrow.Table` with columns `("name", "score")`, sorted
    by score descending -- the same "return a small, composable Arrow
    table rather than a bespoke result class" convention `compare_all()`
    already uses.

    Why `metric="containment"` is the default -- and the metric this
    function exists to make easy to use correctly: classification/
    screening queries (a clinical isolate, a metagenomic read set, a
    suspected-pathogen sample) are typically much smaller and
    compositionally different from the full reference genomes they're
    being checked against. `jaccard` penalizes exactly that size
    mismatch -- a query fully explained by a reference can still score
    low on Jaccard purely because the reference has many more k-mers the
    query never touches (see `Sketch.jaccard`'s own docstring). `a.
    containment(b)`, "what fraction of the query's k-mers also appear in
    this reference", is the question classification actually asks, and
    it does not have that penalty. `metric="jaccard"` remains available
    for callers who want a symmetric similarity instead (e.g. comparing
    two sketches of comparable size/composition).

    `query_path_or_sketch` accepts either a file path (sketched here with
    `k`/`sketch_size`) or an already-built `Sketch` (used unchanged) --
    so classifying one query sketch against several different
    `reference_db`s, or reusing a sketch already built for another
    purpose, costs one sketch build total rather than one per call.

    `min_score`: references scoring at or below this value are dropped
    from the result entirely, the same "there is a floor below which a
    ranked position is not a finding" idea `gather`'s `min_containment`
    already applies. Without it, a query sharing literally zero k-mers
    with *every* reference in `reference_db` still produced a full,
    confidently-sorted `top_n` table -- `result.column("name")[0]` handed
    back a named reference backed by no evidence at all, which is exactly
    the kind of silent-wrong-answer this package's docstrings elsewhere
    (`gather`, `check_sample_identity`) go out of their way to avoid. The
    default `min_score=0.0` is deliberately the weakest possible floor --
    both `containment` and `jaccard` are bounded below by 0, so this only
    ever drops a reference with *zero* real overlap with the query, never
    a low-but-real one; a caller who wants a stronger bar (analogous to
    `gather`'s `min_containment=0.1`, chosen there for a different,
    iterative algorithm and not reused here as a default) can pass a
    higher `min_score` explicitly.
    """
    if metric not in _CLASSIFY_METRICS:
        raise ValueError(f"metric must be one of {_CLASSIFY_METRICS!r}, got {metric!r}")

    if not reference_db:
        return _empty_score_table()

    query = _resolve_sketch(query_path_or_sketch, k=k, sketch_size=sketch_size)

    names, scores = [], []
    for name, reference in reference_db.items():
        score_fn = getattr(query, metric)
        score = score_fn(reference)
        if score <= min_score:
            continue
        names.append(name)
        scores.append(score)

    if not names:
        return _empty_score_table()

    table = pa.table({"name": names, "score": scores})
    table = table.sort_by([("score", "descending")])
    return table.slice(0, top_n)


class SampleIdentityResult(NamedTuple):
    """The result of `check_sample_identity`: a bool plus enough context
    (the raw score, which metric produced it, and the threshold it was
    compared against) that a caller can see *how* close or far the call
    was -- not just trust a bare `True`/`False` with no way to audit a
    borderline case, which is exactly the kind of case a lab QC workflow
    needs to be able to inspect.
    """

    same_sample: bool
    score: float
    metric: str
    threshold: float


def _implied_jaccard_from_mash_distance(mash_distance: float, k: int) -> float:
    """Recovers the Jaccard similarity a `mash_distance` score was itself
    computed from, by inverting `src/sketch.rs::mash_distance`'s own
    formula exactly.

    That Rust method computes `d = -(1/k) * ln(2J / (1+J))` from `J =
    self.jaccard(other)` -- literally the same sketch-vs-sketch Jaccard
    estimate `check_sample_identity(..., metric="jaccard")` would report
    for the identical pair of files (see `sketch.rs`'s `mash_distance`,
    which calls its own `jaccard` first and only then applies the log
    transform). Solving that formula for `J` given `d`:

        exp(-k*d) = 2J / (1+J)
        =>  J = e / (2 - e),   where e = exp(-k*d)

    is therefore not an approximation but an exact algebraic round trip:
    it recovers the same `J` Rust started from, up to floating-point
    precision, for any `d` produced by the general-case branch of that
    formula. The two clamped edge cases (`J <= 0 -> d = 1.0` and
    `J >= 1 -> d = 0.0`, both special-cased in `mash_distance` itself
    rather than going through the log) are inverted the same way here.

    Why this matters: `k` appears on *both* legs of this round trip --
    once when Rust folded it into `d`, once again here un-folding it --
    so it cancels exactly. That cancellation is what makes comparing this
    recovered `J` against `threshold`, instead of comparing `1 - d`
    directly, independent of `k` (see `check_sample_identity`'s
    docstring).
    """
    if mash_distance >= 1.0:
        return 0.0
    if mash_distance <= 0.0:
        return 1.0
    e = math.exp(-k * mash_distance)
    return e / (2.0 - e)


def check_sample_identity(path_a, path_b, *, k=21, sketch_size=1000, threshold=0.9, metric="mash_distance"):
    """Answers "are these two FASTQ files plausibly the same underlying
    biological sample" -- e.g. two sequencing runs of one specimen -- for
    a lab QC / sample-swap-detection workflow, where a "no" is a real,
    actionable finding (wrong tube, wrong barcode, contamination), not
    just a data point.

    Why `metric="mash_distance"` is the default: two runs of the *same*
    sample differ only by sequencing error and coverage depth, not by
    real biological divergence, so their true `mash_distance` (Ondov et
    al.'s Poisson-model estimate of the fraction of sites that differ)
    should sit very close to 0 regardless of how the two runs' read
    counts happen to compare -- an actual sample swap (a different
    specimen, let alone a different organism) shows up as a distance the
    sequencing-error floor alone cannot explain. Plain `jaccard` is a
    similar but weaker signal here: it is a raw set-overlap fraction, not
    an evolutionary-distance estimate, so it is more sensitive to the two
    runs simply having different depth/coverage even when they come from
    the same sample (see `Sketch.jaccard` vs `Sketch.mash_distance` in
    `fastdna/__init__.py`). `metric="jaccard"` remains available for
    callers who specifically want that raw overlap signal instead.

    On the threshold, and a defect this docstring used to describe
    incorrectly: an earlier version of this function compared `1 -
    mash_distance` directly against `threshold`, on the claim that this
    put both metrics on the same "closer to 1 is closer to identical"
    scale. That claim was false. `mash_distance` is `D = -(1/k) *
    ln(2J/(1+J))` -- a *logarithm* of the Jaccard `J` it was computed
    from, not a linear rescaling of it -- so `1 - D` compresses the whole
    low-`J` range into the top of `[0, 1]`: at `k=21`, `1 - D >= 0.9` was
    satisfied by any pair sharing as little as `J >= 0.066` of its
    k-mers, roughly 14x looser than what `metric="jaccard"` demands at
    the identical `threshold=0.9`. Worse, that `1/k` factor meant the
    same `threshold=0.9` silently demanded a different amount of real
    overlap at every `k` (`J >= 0.126` at `k=15` vs. `J >= 0.023` at
    `k=31`) -- a lab that changed `k` for unrelated reasons would
    silently stop catching a class of sample swap it used to catch.

    The fix: `metric="mash_distance"` still reports `.score` as the raw
    Mash distance (unchanged -- it is a genuinely useful number on its
    own, and `metric` must keep meaning what it says), but the pass/fail
    decision now compares `threshold` against the Jaccard similarity
    `mash_distance` was itself derived from, recovered by inverting the
    formula exactly (`_implied_jaccard_from_mash_distance` -- an exact
    round trip, not an approximation, because the same `k` folded into
    `mash_distance` cancels back out when un-folding it). This makes
    `threshold` mean the *same* thing -- "the two files' sketches must
    share at least this fraction of their k-mer content" -- for both
    metrics, at every `k`, rather than pretending two different scales
    were secretly one. `threshold=0.9` is a deliberately generous default
    on that now-consistent Jaccard-equivalent scale: it absorbs the
    MinHash sampling noise a finite `sketch_size` introduces and the
    coverage differences ordinary between two runs of one library, while
    still requiring the two files to agree on roughly 90% of their k-mer
    content -- comfortably above where an actual sample swap (a different
    specimen, let alone a different species) typically lands, per the
    module docstring's own numbers (same-organism-different-specimen
    pairs sit around J = 0.3-0.8). Tighten it (e.g. `threshold=0.98`) for
    a stricter same-run check when coverage is known to be comparable;
    loosen it if the two runs are known to differ a lot in depth or
    library prep and some slack is expected.

    Returns a `SampleIdentityResult` (not just a bool): `.score` is the
    raw metric value actually computed (so a caller can see, e.g., "this
    was mash_distance=0.11, just over the line" rather than a bare
    `False`), and `.metric`/`.threshold` record what produced the
    decision.
    """
    if metric not in _IDENTITY_METRICS:
        raise ValueError(f"metric must be one of {_IDENTITY_METRICS!r}, got {metric!r}")

    sketch_a = _sketch(str(path_a), k=k, sketch_size=sketch_size)
    sketch_b = _sketch(str(path_b), k=k, sketch_size=sketch_size)

    if metric == "mash_distance":
        score = sketch_a.mash_distance(sketch_b)
        similarity = _implied_jaccard_from_mash_distance(score, k)
    else:
        score = sketch_a.jaccard(sketch_b)
        similarity = score

    return SampleIdentityResult(
        same_sample=similarity >= threshold,
        score=score,
        metric=metric,
        threshold=threshold,
    )


def gather(query_path_or_sketch, reference_db, *, k=21, sketch_size=1000, min_containment=0.1, max_references=None):
    """Approximates sourmash's `gather`: iteratively picks the single
    reference that currently best explains the query, records it, and
    repeats against the remaining references -- answering "what *set* of
    organisms is present in this sample" (relevant for a metagenomic,
    multi-organism sample) rather than `classify()`'s "single best
    match".

    This is a genuinely approximate, best-effort heuristic, not a port of
    sourmash's algorithm -- said plainly, up front, because that
    algorithm's core trick does not carry over to `GenomeSketch` as it
    exists here. sourmash's real `gather` can *subtract* the exact hashes
    a picked reference explained from the query's FracMinHash sketch
    before scoring the next round, because it can see and remove
    individual hash values. `fastdna`'s `Sketch` exposes no such
    operation -- only `jaccard`/`containment`/`mash_distance` scalar
    comparisons -- so this function cannot know *which* of the query's
    k-mers a picked reference explained, only a similarity number.

    The approximation used instead: at each round, rank the remaining
    references by `query.containment(reference)` (the same metric
    `classify()` uses, and for the same reason -- see its docstring),
    then discount that raw score by how much the candidate reference
    itself already overlaps with references already picked
    (`candidate.containment(picked)`, the largest such overlap across all
    previously-picked references). A reference that is near-identical to
    one already picked -- a duplicate strain, say -- gets its score
    driven toward zero, so gather does not keep "re-explaining" the same
    already-covered signal under a different name; a reference covering
    genuinely different content keeps its full raw score. The best
    surviving (discounted) candidate is picked, and the process repeats
    until no remaining reference clears `min_containment` on this
    discounted score, `reference_db` is exhausted, or `max_references`
    picks have been made.

    Caveat this function cannot remove: taking the *maximum* pairwise
    overlap with any single already-picked reference (rather than
    attempting to combine overlaps across several picked references,
    which risks double-counting and overstating redundancy, i.e. being
    dishonest in the *other* direction) is a conservative choice. It can
    under-penalize a reference whose content is collectively, but not
    individually, redundant with several already-picked references --
    this function can therefore report more references, or higher scores
    for later ones, than an exact per-hash accounting would. Treat the
    output as an approximate, order-and-magnitude guide to composition,
    not exact abundance fractions.

    Returns a `pyarrow.Table` with columns `("name", "containment",
    "adjusted_score")`, in the order references were picked (i.e. best
    first): `containment` is the raw, undiscounted `query.containment
    (reference)` at the round it was picked; `adjusted_score` is that
    value after the redundancy discount described above, which is what
    `min_containment` is actually compared against.
    """
    if not 0.0 <= min_containment <= 1.0:
        raise ValueError(f"min_containment must be in [0.0, 1.0], got {min_containment!r}")

    if not reference_db:
        return pa.table(
            {
                "name": pa.array([], type=pa.string()),
                "containment": pa.array([], type=pa.float64()),
                "adjusted_score": pa.array([], type=pa.float64()),
            }
        )

    query = _resolve_sketch(query_path_or_sketch, k=k, sketch_size=sketch_size)

    remaining = dict(reference_db)
    picked = []  # [(name, Sketch), ...] in pick order

    names, raw_scores, adjusted_scores = [], [], []

    while remaining and (max_references is None or len(picked) < max_references):
        best_name = None
        best_raw = None
        best_adjusted = -1.0

        for name, reference in remaining.items():
            raw = query.containment(reference)

            redundancy = 0.0
            for _, picked_reference in picked:
                overlap = reference.containment(picked_reference)
                if overlap > redundancy:
                    redundancy = overlap

            adjusted = raw * (1.0 - redundancy)
            if adjusted > best_adjusted:
                best_name, best_raw, best_adjusted = name, raw, adjusted

        if best_name is None or best_adjusted < min_containment:
            break

        names.append(best_name)
        raw_scores.append(best_raw)
        adjusted_scores.append(best_adjusted)
        picked.append((best_name, remaining.pop(best_name)))

    return pa.table({"name": names, "containment": raw_scores, "adjusted_score": adjusted_scores})
