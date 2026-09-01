"""fastdna.genomescope -- reference-free genome profiling from a k-mer
frequency spectrum: genome size, heterozygosity, repeat content, coverage
and sequencing error rate, without an assembly and without a reference.

Scientific basis: Ranallo-Benavidez, Jaron & Schatz, "GenomeScope 2.0 and
Smudgeplot for reference-free profiling of polyploid genomes", *Nature
Communications* 11, 1432 (2020). The idea it formalizes is that a
sequenced genome's k-mer spectrum is a *mixture*: sequencing errors pile up
at low depth, k-mers spanning a heterozygous site are split across the two
haplotypes and appear at ~1x haploid coverage, homozygous k-mers appear at
~2x, and k-mers inside repeats appear at higher multiples. Fit that
mixture and the biology falls out of the fitted parameters -- which is why
GenomeScope is the standard first look at an unassembled genome.

**Every number this module returns is an estimate from a model fit, not a
measurement.** It is only as good as the model's assumptions: diploid,
uniform-ish coverage, sequencing errors that mostly produce unique k-mers,
and a spectrum deep enough for the error and coverage components to
actually separate. When those assumptions do not hold, the honest outcomes
are the ones this module produces on purpose:

  * a spectrum with no coverage peak above the error component raises
    (`profile_genome`) instead of returning a confident genome size derived
    from noise;
  * a fit the optimizer could not complete -- or completed onto a curve
    that misses the data -- comes back as `converged=False` with the
    optimizer's own status in `GenomeProfile.model_fit`, never as a silent
    "success".

Diploid only for now (`ploidy=2`). Polyploid profiling -- GenomeScope 2.0's
own headline addition -- is planned; see item A2 of
`docs/ml-differentiation-roadmap.md`. `ploidy != 2` raises
`NotImplementedError` rather than fitting a diploid model and relabeling
the output.

`numpy` and `scipy` are imported lazily, inside the call that needs them
(the pattern `embed.py` uses for its own optional dependencies), so
importing `fastdna` -- or this module -- never requires either.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from . import _PathLike, _core
from .spectrum import suggest_min_count

if TYPE_CHECKING:
    # Only for static type checkers -- see `profile_genome`'s `source`
    # parameter. `fastdna.KmerCounts` is not imported at runtime here (this
    # module is imported *by* the package; see `_spectrum_and_k`'s own
    # deferred `import fastdna`).
    from . import KmerCounts

__all__ = ["GenomeProfile", "profile_genome", "plot_spectrum_fit"]

#: Total copy numbers modeled, i.e. the multiples of *haploid* coverage the
#: mixture's components sit at. In a diploid: 1 = heterozygous (one
#: haplotype carries the k-mer), 2 = homozygous (both do), 3 and 4 = the
#: first two repeat classes (a heterozygous k-mer duplicated elsewhere, and
#: a homozygous k-mer present twice per haplotype). GenomeScope models the
#: same four by default; components beyond 4 carry too little of a typical
#: spectrum's mass to be identifiable, and are absorbed by the 4x component
#: (see `repeat_fraction`, which is a floor for exactly this reason).
COMPONENT_COPY_NUMBERS = (1, 2, 3, 4)

#: Default fitted-range cap, as a multiple of the observed coverage peak's
#: depth. The peak sits at ~2x haploid coverage, so 10x it reaches ~20x
#: haploid coverage -- comfortably past the 4x component's right tail,
#: while excluding the high-copy tail (rDNA arrays, adapter dimers,
#: contaminants) whose enormous depths would otherwise dominate a
#: least-squares fit and force a dense array with one bin per depth.
DEFAULT_MAX_COVERAGE_MULTIPLE = 10.0

#: How far the fitted curve may sit from the observed spectrum, as RMSE
#: divided by the mean number of distinct k-mers per fitted bin, before the
#: fit is reported as `converged=False`. A local optimizer routinely
#: "succeeds" -- its own convergence criteria met -- on a curve that misses
#: the data entirely; reporting that as converged would hand a caller a
#: confident genome size backed by a fit that never matched their spectrum,
#: which is the exact failure mode this module exists to avoid. 0.5 (the
#: fitted curve is off by half the average bin height) is far outside
#: anything a genuine fit produces and far inside anything an unfittable
#: spectrum produces, so it separates the two without policing ordinary
#: fit quality.
MAX_RELATIVE_RMSE = 0.5

#: Hard cap on optimizer function evaluations, so a pathological spectrum
#: ends as `converged=False` in bounded time rather than running forever.
MAX_FIT_EVALUATIONS = 20_000

_NO_PEAK_MESSAGE = (
    "no coverage peak found above the error component: the spectrum has no "
    "local maximum past the error/signal valley, so there is nothing to fit "
    "a coverage model to. The data may be too shallow -- sequenced coverage "
    "below ~10x cannot be profiled this way, because the genome's own k-mers "
    "never rise clear of the sequencing-error component. Plot "
    "KmerCounts.spectrum() and confirm it has two peaks -- an error peak at "
    "low depth and a coverage peak -- before profiling it."
)


def _missing_dependency(package, extra_hint=None):
    hint = extra_hint or f"pip install {package}"
    return ImportError(
        f"fastdna.genomescope requires the '{package}' package, which is not "
        f"installed. Install it with `{hint}` and try again."
    )


def _numpy():
    """Imports numpy lazily. See the module docstring: neither numpy nor
    scipy is a runtime dependency of `fastdna`, so they are imported here,
    inside the call that needs them, and a missing one raises an
    `ImportError` naming the package and the install command instead of a
    raw `ModuleNotFoundError` from somewhere deeper.
    """
    try:
        import numpy
    except ImportError:
        raise _missing_dependency("numpy") from None
    return numpy


def _scipy():
    """Imports the two scipy pieces the fit needs (`least_squares` and the
    negative-binomial pmf), lazily, for the same reason as `_numpy`.
    """
    try:
        from scipy.optimize import least_squares
        from scipy.stats import nbinom
    except ImportError:
        raise _missing_dependency("scipy") from None
    return least_squares, nbinom


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _clean_spectrum(spectrum):
    """Validates and normalizes a `{depth: distinct k-mers}` mapping into
    plain `int -> int`, rejecting the shapes that would otherwise fail much
    later with an unrelated error: a non-mapping, a depth below 1 (there is
    no such thing as a k-mer observed zero or minus-three times), or a
    negative k-mer count.
    """
    try:
        items = list(spectrum.items())
    except AttributeError:
        raise _core.InvalidConfigError(
            f"spectrum must be a mapping of {{depth: distinct k-mers}} (e.g. "
            f"KmerCounts.spectrum()), got {type(spectrum).__name__}"
        ) from None

    cleaned = {}
    for depth, n_kmers in items:
        try:
            depth = int(depth)
            n_kmers = int(n_kmers)
        except (TypeError, ValueError):
            raise _core.InvalidConfigError(
                f"spectrum keys and values must be integers, got "
                f"{depth!r}: {n_kmers!r}"
            ) from None
        if depth < 1:
            raise _core.InvalidConfigError(f"spectrum depths must be >= 1, got {depth}")
        if n_kmers < 0:
            raise _core.InvalidConfigError(f"spectrum counts must be >= 0, got {n_kmers} at depth {depth}")
        cleaned[depth] = n_kmers

    if not cleaned:
        raise _core.InvalidConfigError("spectrum is empty: there is nothing to profile")
    return cleaned


def _spectrum_and_k(source, k):
    """Resolves `profile_genome`'s `source` into `(spectrum, k)`.

    Three accepted shapes, checked in this order:

      * a mapping -- an already-computed spectrum. `k` must be given
        explicitly: it is not recoverable from the spectrum, and both
        `heterozygosity` and `error_rate` are k-dependent, so guessing it
        would silently produce wrong biology rather than an error.
      * anything exposing `.spectrum()` and `.k` -- a `fastdna.KmerCounts`
        (or the raw `_core` object underneath it). Its own `k` is used; a
        conflicting explicit `k` is refused rather than quietly ignored,
        since one of the two must be wrong.
      * a path -- counted here with `fastdna.count(path, k=...)`, whose
        default `min_count=1` is what makes this work at all: the error
        component lives at depth 1, and a pre-filtered count has already
        thrown it away.
    """
    if hasattr(source, "items"):
        if k is None:
            raise _core.InvalidConfigError(
                "k must be given when profiling a bare spectrum mapping: it "
                "cannot be recovered from the spectrum itself, and both the "
                "heterozygosity and error-rate estimates depend on it. Pass "
                "the k the spectrum was counted with, e.g. "
                "profile_genome(spectrum, k=21)."
            )
        return _clean_spectrum(source), int(k)

    if hasattr(source, "spectrum") and hasattr(source, "k"):
        source_k = int(source.k)
        if k is not None and int(k) != source_k:
            raise _core.InvalidConfigError(
                f"k={k} conflicts with the k-mer size these counts were "
                f"actually built with (k={source_k}); omit k to use the "
                f"counts' own value."
            )
        return _clean_spectrum(dict(source.spectrum())), source_k

    if isinstance(source, (str, os.PathLike)):
        import fastdna  # deferred: this module is imported *by* the package

        counts = fastdna.count(str(source)) if k is None else fastdna.count(str(source), k=int(k))
        return _clean_spectrum(counts.spectrum()), int(counts.k)

    raise TypeError(
        "source must be a KmerCounts, a {depth: distinct k-mers} mapping, or "
        f"a path to a FASTQ(.gz) file, got {type(source).__name__}"
    )


def _highest_peak(spectrum, above):
    """The tallest local maximum at a depth strictly greater than `above`
    (the error/signal valley), as `(depth, count)`, or `(None, None)` when
    there is none.

    Only the *observed* depths are walked (missing depths count as zero
    neighbours) -- the same rule `spectrum.suggest_min_count` follows, and
    for the same reason: a lone contaminant k-mer at depth 10^7 must not
    force a 10-million-entry scan. Height, not position, picks the winner,
    so that lone k-mer (a local maximum of height 1) never outranks the
    real coverage peak.
    """
    best_depth, best_count = None, 0
    for depth, count in spectrum.items():
        if depth <= above or count <= 0:
            continue
        if count >= spectrum.get(depth - 1, 0) and count >= spectrum.get(depth + 1, 0):
            if count > best_count or (count == best_count and best_depth is not None and depth < best_depth):
                best_depth, best_count = depth, count
    if best_depth is None:
        return None, None
    return best_depth, best_count


# ---------------------------------------------------------------------------
# The mixture model
# ---------------------------------------------------------------------------


def _model_counts(depths, params, nbinom, np):
    """The fitted mixture evaluated at `depths`.

    Component `i` (copy number `i`) is a negative binomial with mean
    `i * coverage` and size `i * dispersion`. That parameterization is not
    arbitrary: the depth of a k-mer present in `i` copies is the *sum* of
    `i` independent single-copy depths, and a sum of `i` iid negative
    binomials with size `s` and mean `m` is again negative binomial with
    size `i*s` and mean `i*m`. So one shared `dispersion` describes the
    single-copy coverage noise and every component inherits it correctly,
    instead of each peak carrying its own free width.

    Amplitudes are in units of *distinct k-mers*: each component's pmf sums
    to 1, so `amplitude_i` is the number of distinct k-mers the model puts
    at copy number `i`.
    """
    coverage, dispersion = params[0], params[1]
    # nbinom's `p` is size / (size + mean); with size = i*dispersion and
    # mean = i*coverage the factor of i cancels, so every component shares
    # one p.
    p = dispersion / (dispersion + coverage)
    total = np.zeros(depths.shape, dtype=float)
    for amplitude, copies in zip(params[2:], COMPONENT_COPY_NUMBERS):
        total = total + amplitude * nbinom.pmf(depths, copies * dispersion, p)
    # A pathological parameter vector (driven to a bound, say) can make the
    # pmf underflow to nan/inf; least_squares refuses non-finite residuals
    # outright, which would turn a bad fit into an exception instead of a
    # `converged=False` result.
    return np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


def _initial_parameters(coverage, peak_count):
    """A starting parameter vector for a given candidate haploid coverage.

    `dispersion = 2 * coverage` puts the single-copy variance at 1.5x its
    mean -- mildly overdispersed relative to Poisson, which is what real
    sequencing coverage looks like. The homozygous amplitude is seeded from
    the observed peak height times the width of a Poisson of mean
    `2*coverage` (a peak of height `h` and width `w` holds about `h*w`
    k-mers); the others start at plausible fractions of it, since their
    relative sizes are exactly what the fit is there to determine.
    """
    homozygous = peak_count * math.sqrt(2.0 * math.pi * 2.0 * coverage)
    return [coverage, 2.0 * coverage, 0.25 * homozygous, homozygous, 0.05 * homozygous, 0.05 * homozygous]


def _fit(depths, observed, peak_depth, peak_count, max_coverage, np, least_squares, nbinom):
    """Fits the mixture, returning `(params, info)`.

    Two starting points are tried: `peak_depth / 2` (the observed peak is
    the *homozygous* 2x peak -- the ordinary diploid reading) and
    `peak_depth` (the observed peak is the 1x heterozygous peak, which
    happens in very heterozygous samples where the het peak is the taller
    one). This ambiguity is real and inherent -- a perfectly homozygous
    sample at coverage `c` and a fully heterozygous sample at coverage
    `2c` produce the *same* spectrum -- so the tie is broken by fit cost,
    and, on a genuine tie, in favour of the homozygous reading (the first
    start wins, since improvement is strict). A sample with no visible
    heterozygous shoulder therefore gets the conventional GenomeScope
    answer, and `heterozygosity` near 0 is the flag that the ambiguity was
    resolved by convention rather than by evidence.

    An optimizer that raises (rather than merely failing to converge) is
    caught here: the initial parameters are returned instead, flagged as
    not converged, so a caller always gets an inspectable `GenomeProfile`.
    """
    lower = [1e-6, 1e-6] + [0.0] * len(COMPONENT_COPY_NUMBERS)
    upper = [float(max_coverage), 1e9] + [float("inf")] * len(COMPONENT_COPY_NUMBERS)

    best = None
    for candidate in (peak_depth / 2.0, float(peak_depth)):
        x0 = _initial_parameters(min(max(candidate, 1e-3), float(max_coverage) * 0.999), peak_count)
        x0 = [min(max(value, lo), hi) for value, lo, hi in zip(x0, lower, upper)]
        try:
            result = least_squares(
                lambda params: _model_counts(depths, params, nbinom, np) - observed,
                x0=x0,
                bounds=(lower, upper),
                x_scale="jac",
                max_nfev=MAX_FIT_EVALUATIONS,
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            if best is None:
                best = {
                    "params": np.asarray(x0, dtype=float),
                    "cost": float("inf"),
                    "success": False,
                    "status": -2,
                    "message": f"{type(exc).__name__}: {exc}",
                    "n_evaluations": 0,
                    "initial_coverage": candidate,
                }
            continue

        if best is None or result.cost < best["cost"]:
            best = {
                "params": np.asarray(result.x, dtype=float),
                "cost": float(result.cost),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "n_evaluations": int(result.nfev),
                "initial_coverage": candidate,
            }

    return best["params"], best


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GenomeProfile:
    """Result of :func:`profile_genome`.

    Every field is an **estimate from a model fit**, not a measurement --
    see the module docstring. Check `converged` before quoting any of them:
    the fields are still populated when it is `False` (so the failure can
    be inspected and plotted), but they then describe a curve that did not
    match the data.
    """

    #: Estimated *haploid* genome size, in base pairs. Derived as
    #: `non-error k-mer mass / (ploidy * haploid_coverage)`: every haploid
    #: position contributes `ploidy * coverage` k-mer observations,
    #: whether it is homozygous (one distinct k-mer seen twice as deep) or
    #: heterozygous (two distinct k-mers each at 1x). Counted in k-mer
    #: space, so it is short by about `k` per chromosome/contig -- tens of
    #: bases on a real genome, i.e. far below the model's own error.
    #: k-mers deeper than `model_fit["max_coverage"]` are excluded (see
    #: that key), which makes this a slight *under*-estimate for genomes
    #: with very high-copy repeats.
    genome_size_bp: int

    #: Estimated coverage of a single haploid copy, in x -- the position of
    #: the 1x (heterozygous) component. The homozygous peak, i.e. the
    #: tallest one in an ordinary spectrum, sits at twice this.
    haploid_coverage: float

    #: Estimated fraction of bases that are heterozygous (0.01 = 1%, the
    #: usual way this rate is quoted). Recovered from the ratio of the 1x
    #: and 2x amplitudes: a k-mer is homozygous only if its whole window
    #: spans no heterozygous site, with probability `(1-r)**k`, and each
    #: heterozygous position yields two distinct k-mers against a
    #: homozygous position's one. `float('nan')` when both amplitudes fit
    #: to zero, which leaves the ratio undefined.
    heterozygosity: float

    #: Fraction of the genome's k-mer mass sitting in the repeat
    #: components (copy number >= 3), in `[0, 1]`. A **floor**, not an
    #: exact figure: everything above copy number 4 is absorbed into the 4x
    #: component, and everything deeper than `model_fit["max_coverage"]` is
    #: excluded from the fit entirely.
    repeat_fraction: float

    #: Estimated per-base sequencing error rate, as a probability (0.001 =
    #: 0.1%). Derived from the k-mer mass below the error/signal cutoff
    #: that the fitted genomic components do not explain: a k-mer is
    #: erroneous if any of its `k` bases is, with probability
    #: `1 - (1-e)**k`, inverted here for `e`.
    error_rate: float

    #: Estimated size of the *non-repetitive* haploid genome, in base
    #: pairs: the 1x and 2x components only (`amplitude_1 / 2 +
    #: amplitude_2`, halving the heterozygous component because its two
    #: distinct k-mers describe one haploid position). Compare against
    #: `genome_size_bp` for the repeat burden.
    unique_size_bp: int

    #: The k the spectrum was counted with. Not decoration: heterozygosity
    #: and the error rate are both k-dependent, so a profile is only
    #: interpretable alongside it.
    k: int

    #: `True` only when the optimizer reported success **and** the fitted
    #: curve stays within `MAX_RELATIVE_RMSE` of the observed spectrum. A
    #: local optimizer will happily converge onto a curve that misses the
    #: data; calling that "converged" would be a silent wrong answer, so
    #: both conditions must hold. `model_fit["convergence_reason"]` says
    #: which one failed.
    converged: bool

    #: The fitted parameters and fit diagnostics, for inspection and for
    #: :func:`plot_spectrum_fit`: `haploid_coverage`, `dispersion`,
    #: `amplitudes` (one per `copy_numbers` entry, in distinct k-mers),
    #: `copy_numbers`, `fit_min_depth` (the error/signal cutoff),
    #: `max_coverage` (the fitted range's upper bound), `initial_coverage`,
    #: `rmse` / `relative_rmse`, `cost`, `optimizer_success`, `status`,
    #: `optimizer_message`, `n_function_evaluations`, `convergence_reason`,
    #: `error_kmers` / `genomic_kmers` (the k-mer mass split the error rate
    #: comes from) and `kmers_above_max_coverage` (the mass excluded by the
    #: cap -- large values mean a high-copy tail this profile ignored).
    model_fit: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_genome(
    source: Union["KmerCounts", Mapping[int, int], _PathLike],
    *,
    k: Optional[int] = None,
    ploidy: int = 2,
    max_coverage: Optional[int] = None,
) -> GenomeProfile:
    """Fits a GenomeScope-style negative-binomial mixture to a k-mer
    frequency spectrum and reports what the fit implies about the genome:
    size, heterozygosity, repeat content, haploid coverage and sequencing
    error rate -- with no reference and no assembly.

    `ploidy` must be 2. Polyploid profiling (GenomeScope 2.0's own headline
    feature) raises `NotImplementedError`; see roadmap item A2. Fitting a
    diploid model to a tetraploid and relabeling the result would be worse
    than refusing.

    How the fit works: `spectrum.suggest_min_count` locates the valley
    between the error component and the coverage peak; depths at or above
    it are fitted, depths below it are treated as (mostly) error and feed
    `error_rate` instead. The mixture has four components at 1x/2x/3x/4x
    haploid coverage sharing one dispersion (see `_model_counts`), fitted
    with `scipy.optimize.least_squares` from two starting points (see
    `_fit`).

    Parameters
    ----------
    source : KmerCounts, mapping of int to int, or path-like
        One of:

        * a `fastdna.KmerCounts` (its `.spectrum()` and `.k` are used);
        * a `{depth: distinct k-mers}` mapping, in which case `k` is
          required (it cannot be recovered from the spectrum, and two of
          the outputs depend on it);
        * a path to a FASTQ(.gz) file, counted here via `fastdna.count()`
          with an unfiltered `min_count=1` -- the error component at
          depth 1 is *data* for this model, not noise to be filtered away
          first.
    k : int, optional
        K-mer size. Required (and used as-is) when `source` is a bare
        mapping; must agree with `source.k` when `source` is a
        `KmerCounts`; ignored (and passed to `fastdna.count()`) when
        `source` is a path.
    ploidy : int, default 2
        Must be 2 -- see above.
    max_coverage : int, optional
        Caps the depth range the fit runs over; defaults to
        `DEFAULT_MAX_COVERAGE_MULTIPLE` times the observed coverage
        peak's depth (bounded by the deepest observed k-mer). The cap is
        load-bearing twice over: a single k-mer at depth 10^7 (adapter
        dimer, rDNA array, contaminant) would otherwise both dominate a
        least-squares fit and force a dense array with one bin per depth.
        k-mers above the cap are excluded from every estimate and their
        total mass is reported as `model_fit["kmers_above_max_coverage"]`.

    Returns
    -------
    GenomeProfile
        **Check `.converged` first**: a profile whose fit failed is still
        returned (populated, inspectable, plottable) rather than raised,
        with the optimizer's status in `.model_fit`, but its numbers
        describe a curve that did not match the spectrum. Nothing here is
        a measurement; it is what one model, fitted to one spectrum,
        implies.

    Raises
    ------
    NotImplementedError
        `ploidy` is not 2.
    ValueError
        `k` is missing or conflicts with `source`'s own k, or the
        spectrum has no coverage peak above the error component --
        typically data too shallow to profile, below roughly 10x, where
        the genome's own k-mers never rise clear of the error component.
        That is a refusal by design: a mixture fitted to a pure error
        decay produces a confident, meaningless genome size.
    TypeError
        `source` is not a `KmerCounts`, a mapping, or a path.
    """
    if ploidy != 2:
        raise NotImplementedError(
            f"only diploid profiling is implemented (ploidy=2), got "
            f"ploidy={ploidy!r}. Polyploid support -- the GenomeScope 2.0 "
            "model with per-ploidy components -- is planned; see item A2 of "
            "docs/ml-differentiation-roadmap.md. Fitting the diploid model "
            "and relabeling its output would give you wrong numbers rather "
            "than no numbers."
        )

    np = _numpy()
    least_squares, nbinom = _scipy()

    spectrum, k = _spectrum_and_k(source, k)
    if k < 1:
        raise _core.InvalidKError(f"k must be >= 1, got {k}")

    # The error/signal boundary. `suggest_min_count` returns `default` when
    # the spectrum's shape is not unambiguous -- passing `None` as that
    # default turns "no valley I am confident about" into the refusal below
    # rather than into a silently-assumed cutoff of 2.
    fit_min_depth = suggest_min_count(spectrum, default=None)
    if fit_min_depth is None:
        raise _core.InvalidConfigError(_NO_PEAK_MESSAGE)

    peak_depth, peak_count = _highest_peak(spectrum, above=fit_min_depth)
    if peak_depth is None:
        raise _core.InvalidConfigError(_NO_PEAK_MESSAGE)

    observed_max_depth = max(spectrum)
    if max_coverage is None:
        max_coverage = min(int(round(DEFAULT_MAX_COVERAGE_MULTIPLE * peak_depth)), observed_max_depth)
    else:
        max_coverage = int(max_coverage)
        if max_coverage <= peak_depth:
            raise _core.InvalidConfigError(
                f"max_coverage={max_coverage} is at or below the coverage "
                f"peak (depth {peak_depth}); the fitted range would exclude "
                "the peak the model is built around. Pass a larger value, or "
                "omit it to use "
                f"{DEFAULT_MAX_COVERAGE_MULTIPLE:g}x the peak depth."
            )
    max_coverage = max(max_coverage, peak_depth + 1)

    depths = np.arange(fit_min_depth, max_coverage + 1, dtype=float)
    observed = np.array([spectrum.get(int(d), 0) for d in depths], dtype=float)

    params, info = _fit(depths, observed, peak_depth, peak_count, max_coverage, np, least_squares, nbinom)

    fitted = _model_counts(depths, params, nbinom, np)
    residuals = fitted - observed
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mean_observed = float(np.mean(observed))
    relative_rmse = rmse / mean_observed if mean_observed > 0 else float("inf")

    converged = bool(info["success"]) and relative_rmse <= MAX_RELATIVE_RMSE
    if not info["success"]:
        reason = f"optimizer did not converge: {info['message']}"
    elif relative_rmse > MAX_RELATIVE_RMSE:
        reason = (
            f"optimizer converged, but the fitted curve misses the spectrum "
            f"(relative RMSE {relative_rmse:.2f} > {MAX_RELATIVE_RMSE}); the "
            "spectrum is not shaped like a negative-binomial coverage mixture"
        )
    else:
        reason = "converged"

    coverage = float(params[0])
    dispersion = float(params[1])
    amplitudes = [float(a) for a in params[2:]]

    # --- k-mer mass accounting -------------------------------------------
    # Mass = sum(depth * distinct k-mers), i.e. total k-mer *observations*,
    # which is what coverage divides into to give a genome size.
    mass_in_range = float(np.dot(depths, observed))
    mass_below = float(sum(d * n for d, n in spectrum.items() if d < fit_min_depth))
    mass_above = float(sum(d * n for d, n in spectrum.items() if d > max_coverage))

    # Part of the genome's own signal falls below the cutoff (the 1x
    # component's left tail), and it should count towards the genome, not
    # towards errors. The model says how much; it can never be more than
    # the mass actually observed down there, hence the clamp -- without it
    # a bad fit could invent genomic mass and drive the error rate to zero.
    if fit_min_depth > 1:
        low_depths = np.arange(1, fit_min_depth, dtype=float)
        genomic_mass_below = float(np.dot(low_depths, _model_counts(low_depths, params, nbinom, np)))
        genomic_mass_below = min(genomic_mass_below, mass_below)
    else:
        genomic_mass_below = 0.0

    genomic_mass = mass_in_range + genomic_mass_below
    error_mass = max(0.0, mass_below - genomic_mass_below)

    genome_size = genomic_mass / (ploidy * coverage) if coverage > 0 else 0.0

    # --- biology from the fitted amplitudes -------------------------------
    heterozygous, homozygous = amplitudes[0], amplitudes[1]
    denominator = heterozygous + 2.0 * homozygous
    if denominator > 0:
        homozygous_kmer_fraction = min(1.0, max(0.0, 2.0 * homozygous / denominator))
        heterozygosity = 1.0 - homozygous_kmer_fraction ** (1.0 / k)
    else:
        heterozygosity = float("nan")

    weighted = [amp * copies for amp, copies in zip(amplitudes, COMPONENT_COPY_NUMBERS)]
    total_weight = sum(weighted)
    repeat_fraction = sum(weighted[2:]) / total_weight if total_weight > 0 else float("nan")

    observed_mass = error_mass + genomic_mass
    if observed_mass > 0:
        error_kmer_fraction = min(1.0, error_mass / observed_mass)
        error_rate = 1.0 - (1.0 - error_kmer_fraction) ** (1.0 / k)
    else:
        error_rate = float("nan")

    unique_size = amplitudes[0] / 2.0 + amplitudes[1]

    return GenomeProfile(
        genome_size_bp=int(round(genome_size)),
        haploid_coverage=coverage,
        heterozygosity=heterozygosity,
        repeat_fraction=repeat_fraction,
        error_rate=error_rate,
        unique_size_bp=int(round(unique_size)),
        k=k,
        converged=converged,
        model_fit={
            "haploid_coverage": coverage,
            "dispersion": dispersion,
            "amplitudes": amplitudes,
            "copy_numbers": list(COMPONENT_COPY_NUMBERS),
            "fit_min_depth": int(fit_min_depth),
            "max_coverage": int(max_coverage),
            "peak_depth": int(peak_depth),
            "initial_coverage": float(info["initial_coverage"]),
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "cost": float(info["cost"]),
            "optimizer_success": bool(info["success"]),
            "status": int(info["status"]),
            "optimizer_message": str(info["message"]),
            "n_function_evaluations": int(info["n_evaluations"]),
            "convergence_reason": reason,
            "genomic_kmers": genomic_mass,
            "error_kmers": error_mass,
            "kmers_above_max_coverage": mass_above,
        },
    )


def plot_spectrum_fit(
    profile: GenomeProfile,
    spectrum: Any,  # {depth: distinct k-mers} mapping or KmerCounts
    ax: Any = None,  # matplotlib.axes.Axes; matplotlib is an optional, lazily imported dependency
) -> Any:  # matplotlib.axes.Axes; matplotlib is optional and lazily imported, not a module-level dependency
    """Plots an observed spectrum against the model :func:`profile_genome`
    fitted to it -- the standard GenomeScope figure, and the fastest way to
    see whether a profile is trustworthy.

    Reading it is the point: a good fit hugs the observed histogram through
    both peaks, and the per-component dashed curves show which peak the
    model called heterozygous and which homozygous. A fit whose components
    are shifted by a factor of two against the visible peaks means the
    coverage was resolved the wrong way round -- a mistake that halves or
    doubles `genome_size_bp` while every returned number still looks
    plausible in isolation. `converged=False` is annotated in the title,
    but the curve is drawn either way: an unconverged fit is exactly the
    one worth looking at.

    Parameters
    ----------
    profile : GenomeProfile
        The fit result to plot, as returned by :func:`profile_genome`.
    spectrum : mapping of int to int, or KmerCounts
        The same `{depth: distinct k-mers}` mapping (or `KmerCounts`) the
        profile was fitted from.
    ax : matplotlib.axes.Axes, optional
        Axes to draw onto. A new figure/axes is created if omitted.
        Typed `Any` here (rather than `matplotlib.axes.Axes`) because
        `matplotlib` is imported lazily inside this function and is not a
        module-level dependency of this file -- a precise type hint would
        force an eager `matplotlib` import at type-checking time.

    Returns
    -------
    Any
        The `matplotlib.axes.Axes` drawn onto (see the `ax` parameter for
        why this is typed `Any`), so a caller can keep styling it.

    Raises
    ------
    ImportError
        `matplotlib` is not installed. Named explicitly, with the install
        command, rather than a raw `ModuleNotFoundError`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise _missing_dependency("matplotlib") from None

    np = _numpy()
    _, nbinom = _scipy()

    if hasattr(spectrum, "spectrum"):
        spectrum = dict(spectrum.spectrum())
    spectrum = _clean_spectrum(spectrum)

    fit = profile.model_fit
    fit_min_depth = int(fit["fit_min_depth"])
    max_coverage = int(fit["max_coverage"])

    if ax is None:
        _, ax = plt.subplots()

    # Draw from depth 1 so the error component is visible (it is what the
    # fitted range starts *above*), but scale the y-axis to the fitted
    # range: the error peak is routinely an order of magnitude taller than
    # the coverage peak and would otherwise flatten the part of the plot
    # this figure exists to show.
    all_depths = np.arange(1, max_coverage + 1, dtype=float)
    observed = np.array([spectrum.get(int(d), 0) for d in all_depths], dtype=float)
    ax.plot(all_depths, observed, color="0.35", linewidth=1.0, label="observed spectrum")

    params = [fit["haploid_coverage"], fit["dispersion"], *fit["amplitudes"]]
    ax.plot(all_depths, _model_counts(all_depths, params, nbinom, np), color="C3", linewidth=2.0, label="fitted model")

    for index, copies in enumerate(fit["copy_numbers"]):
        component = [0.0] * len(fit["amplitudes"])
        component[index] = fit["amplitudes"][index]
        ax.plot(
            all_depths,
            _model_counts(all_depths, [params[0], params[1], *component], nbinom, np),
            linestyle="--",
            linewidth=1.0,
            label=f"{copies}x component",
        )

    ax.axvline(fit_min_depth, color="0.6", linestyle=":", linewidth=1.0, label="error/signal cutoff")

    in_range = observed[fit_min_depth - 1 :]
    ceiling = float(in_range.max()) if in_range.size and in_range.max() > 0 else float(observed.max() or 1.0)
    ax.set_ylim(0, ceiling * 1.25)
    ax.set_xlim(0, max_coverage)
    ax.set_xlabel("k-mer depth")
    ax.set_ylabel("distinct k-mers")

    status = "" if profile.converged else "  [NOT CONVERGED]"
    ax.set_title(
        f"k={profile.k}  coverage={profile.haploid_coverage:.1f}x  "
        f"genome={profile.genome_size_bp / 1e6:.2f} Mb  "
        f"het={profile.heterozygosity:.3%}{status}"
    )
    ax.legend(fontsize="small")
    return ax
