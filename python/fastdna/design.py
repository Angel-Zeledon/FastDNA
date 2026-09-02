"""fastdna.design -- can this experiment answer the question, before you run it.

## Why this module exists

`audit()` tells you whether a result is real. This tells you whether the
experiment is capable of producing one, which is a different question and a
cheaper one: it needs no genomes, no counting and no fitting, only the shape
of the design.

The motivating failure is recorded in `docs/validation-real-data.md`. Three
attempts were made to demonstrate population-structure leakage on real
cohorts. Two of them could not have worked, for reasons visible from the
design alone before a single k-mer was counted:

- 80 samples against 5,000 features. An unregularised model with p/n = 62
  cannot learn anything that generalises, so "no signal" was guaranteed and
  told us nothing about the biology.
- 5-fold cross-validation over a cohort whose 80 samples fall into 38
  lineages, several of them singletons: held-out folds too small and too
  unbalanced for a fold-level AUC to mean much.

Each cost roughly twenty minutes of counting and fitting to discover. Both
are arithmetic.

There is a second, subtler use. A design check run *before* the experiment
is a form of pre-registration: it fixes what the experiment was expected to
be able to detect, in a machine-readable artifact, before any result exists
to be disappointed by. The gap between "this design can detect an AUC of
0.75" and "this design found 0.62" is interpretable; the gap between "we ran
something" and "we found 0.62" is not.

## What it does not do

**No verdict.** Same posture as `audit()` and `evaluation`: this reports
numbers and names concerns. It does not return PASS/FAIL, and it does not
refuse to let you run anything. A design that looks marginal here may be the
only cohort that exists for a question worth asking, and that is the
caller's call to make, with the numbers in front of them.

**No power analysis for the thing you actually care about.** The AUC
confidence interval below is a sampling-variance estimate under
Hanley-McNeil's assumptions; it says how precisely *this many samples* can
pin down an AUC, not whether your biology is detectable. A design can clear
every check here and still fail because the phenotype has no genomic basis.

**No opinion about features.** `p/n` is reported and flagged, not corrected:
the right response might be fewer features, stronger regularisation, or more
samples, and which one depends on things this function cannot see.

## Reference

Hanley JA, McNeil BJ. "The meaning and use of the area under a receiver
operating characteristic (ROC) curve." Radiology 143(1):29-36, 1982. The
standard-error formula in `auc_standard_error()` is theirs, including the
`Q1`/`Q2` exponential approximations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DesignReport",
    "DesignConcern",
    "auc_standard_error",
    "check_design",
]

#: Above this parameter-to-sample ratio, an unregularised linear model is
#: fitting noise. Not a law of nature -- a strongly regularised model or one
#: with genuine sparse signal can do fine past it -- which is why crossing it
#: raises a concern rather than an error. The value is the conventional
#: rule-of-thumb boundary; this project's own measured case sat at 62, where
#: the model learned nothing and the leakage gap grew with the ratio on data
#: containing no signal at all (see `audit`'s module docstring).
_HIGH_PN_RATIO = 5.0

#: Fewer positives than this in the smallest class makes a fold-level metric
#: dominated by which individual samples landed where. Ten events per
#: variable is the classical epidemiological rule; ten events total is a far
#: weaker bar, and failing even that is worth saying out loud.
_MIN_MINORITY_COUNT = 10

#: A single group holding more than this fraction of the cohort means one
#: fold of a grouped split is mostly that group -- and the corresponding
#: training set is missing it entirely.
_DOMINANT_GROUP_FRACTION = 0.30


@dataclass(frozen=True)
class DesignConcern:
    """One named problem with a design, and what would address it.

    Attributes
    ----------
    code : str
        Stable identifier, e.g. `"high_p_over_n"`. Suitable for filtering;
        the human-readable content is in `message`.
    message : str
        What is wrong, with the numbers that make it wrong.
    remedy : str
        What would change it. Phrased as options rather than instructions
        where more than one is legitimate.
    """

    code: str
    message: str
    remedy: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message} -- {self.remedy}"


@dataclass(frozen=True)
class DesignReport:
    """What a design can and cannot resolve, computed without running it.

    Attributes
    ----------
    n_samples, n_features : int
        As given.
    p_over_n : float
        `n_features / n_samples`. Reported even when no features are given
        (then `nan`), because its absence is itself worth noticing in a
        report about learnability.
    minority_fraction : float
        Fraction of the cohort in the smaller class, for a binary phenotype;
        `nan` for a continuous one.
    minority_count : int
        Absolute count in the smaller class -- the number that actually
        governs whether a fold-level metric is stable, since a fraction of a
        small cohort is still a small number.
    n_groups : int
        Distinct lineages/groups, or `n_samples` when none were supplied
        (each sample its own group, i.e. no grouping).
    largest_group_fraction : float
        Share of the cohort in its biggest group. High values mean one fold
        of a grouped split is mostly that group.
    min_test_fold_size : int
        Smallest test fold this many splits can produce, ignoring grouping.
        A lower bound, not a prediction: grouped splitters routinely do worse
        because they cannot break a group.
    auc_ci_halfwidth : float
        Half-width of the 95% confidence interval on an AUC of
        `assumed_auc`, at this cohort's class sizes (Hanley-McNeil). The
        number to compare against the effect you hope to see: an interval
        wider than your expected effect means the experiment cannot resolve
        it whatever the biology does.
    assumed_auc : float
        The AUC the interval above was computed at.
    concerns : tuple of DesignConcern
        Empty when nothing crossed a threshold. Never a verdict.
    """

    n_samples: int
    n_features: int
    p_over_n: float
    minority_fraction: float
    minority_count: int
    n_groups: int
    largest_group_fraction: float
    min_test_fold_size: int
    auc_ci_halfwidth: float
    assumed_auc: float
    concerns: Tuple[DesignConcern, ...] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        lines = [
            f"# Design check -- {self.n_samples} samples, {self.n_features} features",
            "",
            "| property | value |",
            "|---|---:|",
            f"| p/n | {self.p_over_n:.4g} |",
            f"| minority class | {self.minority_count} ({self.minority_fraction:.1%}) |",
            f"| groups | {self.n_groups} |",
            f"| largest group | {self.largest_group_fraction:.1%} |",
            f"| smallest test fold | {self.min_test_fold_size} |",
            f"| 95% CI half-width on AUC={self.assumed_auc:g} | ±{self.auc_ci_halfwidth:.4g} |",
        ]
        if self.concerns:
            lines += ["", "## Concerns", ""]
            lines += [f"- **{c.code}** — {c.message} *{c.remedy}*" for c in self.concerns]
        else:
            lines += ["", "No concerns crossed their thresholds."]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return (
            f"DesignReport(n_samples={self.n_samples}, p_over_n={self.p_over_n:.4g}, "
            f"minority={self.minority_count}, n_groups={self.n_groups}, "
            f"auc_ci_halfwidth={self.auc_ci_halfwidth:.4g}, "
            f"n_concerns={len(self.concerns)})"
        )


def auc_standard_error(auc: float, n_positive: int, n_negative: int) -> float:
    """Standard error of an AUC estimate, per Hanley & McNeil (1982).

    Uses their exponential approximations `Q1 = A/(2-A)` and
    `Q2 = 2A^2/(1+A)`, which assume negative-exponential score
    distributions. That assumption is conservative for most real
    classifiers -- it tends to overestimate the variance slightly -- which is
    the right direction for a feasibility check: it will not tell you an
    under-powered design is fine.

    Returns `nan` when either class is empty, since an AUC is undefined
    there at all.
    """
    if n_positive <= 0 or n_negative <= 0:
        return float("nan")
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    numerator = (
        a * (1.0 - a)
        + (n_positive - 1) * (q1 - a * a)
        + (n_negative - 1) * (q2 - a * a)
    )
    return math.sqrt(max(numerator, 0.0) / (n_positive * n_negative))


def check_design(
    phenotype: Sequence[Any],
    *,
    n_features: Optional[int] = None,
    groups: Optional[Sequence[Any]] = None,
    n_splits: int = 5,
    assumed_auc: float = 0.75,
) -> DesignReport:
    """Reports what an experiment of this shape can resolve, before running it.

    Parameters
    ----------
    phenotype : sequence
        The labels. Binary (two distinct values) enables the class-balance
        and AUC-interval checks; anything else is treated as continuous and
        those are reported as `nan` rather than guessed at.
    n_features : int, optional
        How many features the model will see -- e.g. `KmerVectorizer`'s
        `top_features`. Omitted means the `p/n` check cannot run, and
        `p_over_n` is `nan`.
    groups : sequence, optional
        Lineage labels, as `cv.LineageKFold` would receive. Omitted means
        every sample is its own group, which is what an ungrouped split
        effectively assumes.
    n_splits : int, default 5
        The intended number of cross-validation folds.
    assumed_auc : float, default 0.75
        The AUC to size the confidence interval at. The default is a
        deliberately optimistic-but-plausible published effect: if the
        interval is too wide even here, it will be wider at the AUC you
        actually get.

    Returns
    -------
    DesignReport

    Examples
    --------
    The design that failed in this project's own leakage attempt, diagnosed
    without counting a single k-mer:

    >>> import numpy as np
    >>> from fastdna.design import check_design
    >>> phenotype = np.array([1] * 58 + [0] * 22)
    >>> report = check_design(phenotype, n_features=5000, n_splits=5)
    >>> report.p_over_n
    62.5
    >>> [c.code for c in report.concerns]
    ['high_p_over_n']
    """
    labels = np.asarray(phenotype)
    n_samples = int(labels.shape[0])
    if n_samples < 2:
        raise ValueError(f"check_design needs at least 2 samples, got {n_samples}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    concerns: List[DesignConcern] = []

    # --- learnability -----------------------------------------------------
    p_over_n = float(n_features) / n_samples if n_features is not None else float("nan")
    if n_features is not None and p_over_n > _HIGH_PN_RATIO:
        concerns.append(DesignConcern(
            "high_p_over_n",
            f"{n_features} features against {n_samples} samples (p/n = {p_over_n:.3g}). "
            "An unregularised linear model has enough capacity to fit noise exactly, "
            "so a poor score will not distinguish 'no biological signal' from "
            "'not enough samples to find it'",
            "Reduce features, regularise strongly, or add samples. Note that a "
            "leakage gap also grows with p/n on signal-free data, so a large gap "
            "measured at a high ratio is not evidence of population structure",
        ))

    # --- class balance ----------------------------------------------------
    distinct = np.unique(labels[~_isnan_mask(labels)]) if labels.dtype.kind == "f" else np.unique(labels)
    is_binary = distinct.size == 2
    if is_binary:
        counts = np.array([int((labels == value).sum()) for value in distinct])
        minority_count = int(counts.min())
        minority_fraction = minority_count / n_samples
        n_positive, n_negative = int(counts.max()), minority_count
        if minority_count < _MIN_MINORITY_COUNT:
            concerns.append(DesignConcern(
                "tiny_minority_class",
                f"the smaller class has {minority_count} samples",
                f"any per-fold metric is governed by which individual samples land "
                f"where; {_MIN_MINORITY_COUNT}+ is a weak floor, not a target",
            ))
        if minority_fraction < 0.2:
            concerns.append(DesignConcern(
                "class_imbalance",
                f"the smaller class is {minority_fraction:.1%} of the cohort",
                "prefer average precision over ROC-AUC (see fastdna.evaluation, "
                "which exists for exactly this) and stratify the splits",
            ))
    else:
        minority_count, minority_fraction = -1, float("nan")
        n_positive = n_negative = 0

    # --- grouping ---------------------------------------------------------
    if groups is None:
        group_labels = np.arange(n_samples)
    else:
        group_labels = np.asarray(groups)
        if group_labels.shape[0] != n_samples:
            raise ValueError(
                f"groups has {group_labels.shape[0]} entries but phenotype has "
                f"{n_samples}; they must describe the same cohort"
            )
    unique_groups, group_sizes = np.unique(group_labels, return_counts=True)
    n_groups = int(unique_groups.size)
    largest_group_fraction = float(group_sizes.max()) / n_samples

    if n_groups < n_splits:
        concerns.append(DesignConcern(
            "fewer_groups_than_folds",
            f"{n_groups} groups but {n_splits} folds requested",
            f"a grouped splitter cannot place a fold boundary inside a group; "
            f"lower n_splits to at most {n_groups}, or use coarser groups",
        ))
    if groups is not None and largest_group_fraction > _DOMINANT_GROUP_FRACTION:
        concerns.append(DesignConcern(
            "dominant_group",
            f"the largest group holds {largest_group_fraction:.1%} of the cohort",
            "whichever fold receives it is mostly one lineage, and the matching "
            "training set contains none of it -- expect high fold-to-fold variance",
        ))

    # --- fold sizes -------------------------------------------------------
    min_test_fold_size = n_samples // n_splits
    if min_test_fold_size < _MIN_MINORITY_COUNT:
        concerns.append(DesignConcern(
            "small_test_folds",
            f"{n_splits} folds over {n_samples} samples leaves ~{min_test_fold_size} "
            "samples per test fold",
            "fewer folds, or more samples; a fold-level AUC over a handful of "
            "samples is mostly noise, and grouped splits make folds smaller still",
        ))

    # --- resolvable effect ------------------------------------------------
    if is_binary:
        standard_error = auc_standard_error(assumed_auc, n_positive, n_negative)
        ci_halfwidth = 1.96 * standard_error
        if ci_halfwidth > (assumed_auc - 0.5):
            concerns.append(DesignConcern(
                "auc_ci_wider_than_effect",
                f"the 95% CI on an AUC of {assumed_auc:g} is ±{ci_halfwidth:.3g}, which "
                f"reaches chance (0.5)",
                "this cohort cannot distinguish that effect from no effect at all; "
                "it needs more samples before the question is answerable",
            ))
    else:
        ci_halfwidth = float("nan")

    return DesignReport(
        n_samples=n_samples,
        n_features=int(n_features) if n_features is not None else 0,
        p_over_n=p_over_n,
        minority_fraction=minority_fraction,
        minority_count=minority_count,
        n_groups=n_groups,
        largest_group_fraction=largest_group_fraction,
        min_test_fold_size=min_test_fold_size,
        auc_ci_halfwidth=ci_halfwidth,
        assumed_auc=float(assumed_auc),
        concerns=tuple(concerns),
    )


def _isnan_mask(values: np.ndarray) -> np.ndarray:
    """`np.isnan` where that is meaningful, all-False where it is not."""
    if values.dtype.kind == "f":
        return np.isnan(values)
    return np.zeros(values.shape, dtype=bool)
