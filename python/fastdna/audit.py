"""fastdna.audit -- how much of a model's score is real, not lineage?

## The problem this answers

`python/fastdna/cv.py`'s own docstring already states the finding this
module makes concrete: a random cross-validation split on a clonal
bacterial or viral cohort scatters near-copies of the same lineage across
the train/test boundary, so a held-out fold is not really held out. The
model can score highly by recognizing the lineage rather than the
phenotype. That is not a hypothetical -- PLOS Biology 2025 measured it
inflating published AMR-prediction numbers over 24,000+ genomes, and arXiv
2502.07749 names phylogeny-aware cross-validation the field's missing
standard tool (both already cited by `cv.py`).

`fastdna.cv.LineageKFold` is the fix. What was still missing is the
*demonstration*, run against the caller's own model and data rather than
cited as an abstract literature finding: fit the same classifier, with the
same number of folds, once under ordinary cross-validation and once under
`LineageKFold`, and report the gap between the two scores. That is what
`audit()` does, and nothing more -- it composes `cv.lineage_groups` and
`cv.LineageKFold` (both used unmodified) with a plain
`sklearn.model_selection.cross_val_score` call for the random-CV baseline.

## Relationship to `docs/audit/audit-api.md`

That document is an earlier, more ambitious design draft, written before
`cv.py`/`evaluation.py`/`workflow.py` existed in their current form. This
module intentionally narrows it: no majority-class or lineage-only
baselines, no automatic "elbow" threshold selection, and -- most
deliberately -- no `CLEAN`/`INFLATED`/`CONFOUNDED` verdict enum. The
draft's own §0 already settles why a verdict would be premature: a 2026
preprint on AMR-prediction pipelines found random CV sometimes predicting
clinical performance *better* than phylogeny-aware splits (cited in
`docs/audit/PLAN.md`'s "the risk that is faced head-on" section) -- the
field has not settled which number is "correct", so `audit()` reports both
and the gap, and leaves the judgment to the caller, the same posture
`cv.py` and `evaluation.py` already take about thresholds and split
choice. See those two docs' "what this module does not do" style sections;
the corresponding section below states the same thing for this module.

The draft's phenotype-vs-lineage confounding statistic (its §3.3) *is*
computed, as of the section immediately below -- which also explains why
that is not a reversal of the paragraph above.

## Phenotype-vs-lineage confounding (`AuditReport.confounding`)

The gap answers "how much of my score survives when the model cannot see
near-copies of its training samples". It does not answer the question a
reviewer asks immediately afterwards: *could this cohort have separated
the two signals at all?* `cv.py`'s own "What this module does not fix"
section names the design that cannot -- every resistant isolate in one
clone -- and a gap alone does not distinguish that cohort from one where
the model simply was not very good. A `score_lineage` at chance is
consistent with both.

`AuditReport.confounding` measures it directly and separately: how much
of the phenotype is predictable from the lineage assignment *alone*. No
model is fitted, no k-mer is read, and no `cross_val_score` is involved
-- it is a property of the cohort's design and its labels, and it would
be the same number for whatever estimator the caller tries next. That is
precisely why it is worth reporting beside two numbers that are not.

**Which statistic, and why that one.** `Confounding.statistic` names it,
because a phenotype `audit()` accepts does not always have a contingency
table:

- Categorical phenotype (binary or multi-class): **Cramer's V** over the
  `n_lineages x n_phenotype_levels` contingency table, with **Bergsma's
  (2013) bias correction** (*A bias-correction for Cramer's V and
  Tschuprow's T*, J. Korean Statistical Society 42(3):323-328).
  `docs/audit/audit-api.md` §3.3 named plain Cramer's V and gave the
  right reason to prefer it over raw mutual information: it is
  normalized to `[0, 1]` regardless of how many lineages there are, so a
  value is comparable between a 5-lineage and a 50-lineage cohort, which
  raw mutual information is not. The bias correction is not optional
  decoration on top of that. Uncorrected V has an expected value of
  roughly `sqrt((r-1)(c-1)/(n-1))` under *independence*, which grows with
  the lineage count `r` -- and `r` here is set by a Mash-distance
  threshold, not by the sample size. A 40-sample cohort that
  `lineage_groups()` happens to split into 20 lineages would read ~0.72
  on a coin-flip phenotype, comfortably past `audit-api.md`'s own "HIGH"
  band, purely from small-sample bias; in the limit where every sample is
  its own lineage, uncorrected V is exactly 1.0 for *every* phenotype.
  Bergsma's correction subtracts that chance term before normalizing, so
  the coin-flip cohort reads ~0 and the saturated cohort is reported as
  undefined rather than as a spurious 1.0.
- Continuous phenotype: **omega-squared**, the bias-corrected form of the
  one-way-ANOVA eta-squared of phenotype on lineage that `audit-api.md`
  §3.3 named for this case (Olejnik & Algina 2003, *Generalized eta and
  omega squared statistics*). Same bias problem in the same direction --
  eta-squared's expectation under independence is `(k-1)/(n-1)`, again
  growing with the lineage count -- and the same fix. It occupies the
  same `[0, 1]` scale and answers the same question: what fraction of the
  phenotype's variation lies between lineages rather than within them.

  **A continuous phenotype is deliberately not binned** into a
  contingency table in order to force Cramer's V onto it. Binning would
  make the reported number depend on a bin count nobody chose on
  evidence, and two callers binning the same cohort differently would
  report different "confounding" for identical data. Omega-squared
  requires no such choice and is the standard estimator of this exact
  quantity.

Both are `nan` when the cohort makes them genuinely undefined rather than
merely small, and `Confounding.undefined_reason` then says, in a
sentence, which degeneracy it was.

**What this number cannot tell you.** It is an association between two
label vectors, so:

- **It is not causal.** It does not say lineage *drives* the phenotype. A
  real resistance mechanism that arose once and then spread clonally
  produces a high value; so does a sampling artifact in which one
  hospital contributed one clone along with all its resistant isolates.
  Those are different findings with the same number, and telling them
  apart needs the cohort's metadata, not more statistics (`covariates=`
  is the nearest this module comes, and it is not a decomposition either
  -- see the next section).
- **It does not bound the gap.** A high value does not guarantee a large
  `gap` and a low one does not guarantee a small gap. The gap depends on
  what the *model* found; this number does not know the model exists.
- **It carries no p-value.** The asymptotic chi-squared p-value
  `audit-api.md` §3.3 asked for is invalid on exactly the tables this
  module produces: a lineage-by-phenotype table over a clonal cohort is
  sparse, with many small or singleton lineages, and the chi-squared
  approximation needs expected cell counts that such a table does not
  have. Nor would it be informative -- in a clonal cohort the null "the
  phenotype is independent of lineage" is essentially never true, so a
  significant p-value is a foregone conclusion while the magnitude is the
  actual question. `cv.permutation_importance_pvalues(..., groups=...)`
  already answers the significance-flavoured question at the only level
  this project can answer it honestly: per feature, by within-lineage
  permutation. (Not needing a p-value is also why nothing in this
  computation imports `scipy`: the chi-squared *statistic* is four numpy
  operations on a contingency table, and only its p-value would have
  needed a distribution function. `audit(groups=...)` therefore still
  runs without `scipy` installed, exactly as it did before this number
  existed.)
- **It inherits the lineage assignment.** Everything
  `lineage_groups()`'s `distance_threshold` decides, this number
  inherits. `n_lineages` and `lineage_threshold` are reported next to it
  for that reason; a value read without them is not interpretable.

### Why this is a number and not a verdict

`docs/audit/audit-api.md` names this statistic in §3.3 *and* converts it
into a `CONFOUNDED` verdict in §3.6 at `V >= 0.6`, with §5's CLI exiting
non-zero on that verdict; `docs/audit/tasks/14-audit-nucleo.md` deferred
both to "task 15". This is task 15, and it lands the statistic while
still declining the verdict, because the two were never one decision:

- **The measurement is not the contested quantity.** What the 2026
  preprint unsettled (above) is *which cross-validation score better
  predicts clinical performance*. It says nothing against the claim that
  a particular cohort's phenotype is statistically associated with its
  lineage structure. That association is a fact about two label vectors;
  it is not a prediction about anything, and no finding about split
  strategy can make it false. Withholding it would not be caution -- it
  would be suppressing the one number in this report that does not depend
  on a modelling choice.
- **The threshold is.** `audit-api.md`'s `< 0.3` / `0.3-0.6` / `> 0.6`
  LOW/MODERATE/HIGH bands, and the `CONFOUNDED` verdict built on the 0.6
  cut-point, are where a measurement becomes a judgment about a cohort
  this module has never seen -- using a cut-point no cited study
  established, on a statistic whose value moves with a Mash-distance
  threshold the caller picked. A CLI exiting 1 on that would fail
  someone's pipeline on a number with no published provenance, which is a
  worse failure than the inflated score it was meant to prevent, because
  it looks authoritative.

So `Confounding.value` is reported, `Confounding.statistic` says what it
is, and the bands are not implemented. A caller who wants
`report.confounding.value >= 0.6` to mean something in *their* pipeline
writes that comparison in one line, having chosen the cut-point
themselves for a cohort they know -- which is the whole of the
difference.

## The `covariates=` parameter (G-13)

A gap between random and lineage-blocked CV is not proof the *cause* is
lineage specifically -- sequencing batch, site, or year can correlate with
both phenotype and lineage in a real clinical cohort (`docs/audit/
ml-gaps.md`, G-13). `covariates=` accepts a mapping of covariate name to
one label per sample (e.g. `{"batch": batch_ids}`) and, for each one, runs
the *same* comparison a third way: `cv.LineageKFold` reused verbatim with
`groups=` set to that covariate's own labels instead of the lineage labels.
This gives an honest, independently interpretable number per covariate --
"if I had blocked CV by this covariate instead, what would my score have
been?" -- computed with no new statistics, only another application of the
same splitter.

**What `covariates=` deliberately does not attempt**: a variance
decomposition that partitions the gap into percentages attributable to
lineage vs. each covariate vs. "unexplained", the way `docs/audit/
ml-gaps.md`'s "second round" notes sketch as an aspiration. Lineage and a
covariate like sequencing batch are frequently correlated with each other
in a real cohort (a batch is often also a lineage-enriched collection
event), so their individual blocked-CV gaps are not independent
contributions and do not sum to anything meaningful. Reporting a fabricated
"62% lineage, 18% batch, ..." split when the underlying quantities are not
additive would be exactly the kind of unearned precision this project's
own `docs/philosophy-narrow-not-broad.md` and every other honesty-focused
module here (`cv.py`, `evaluation.py`) explicitly refuse to ship. Each
covariate's gap is reported on its own, next to the lineage gap, and the
markdown/repr output says so explicitly rather than implying a
decomposition that was not computed.

## Feature attribution to lineage (`fastdna.explain()`, not a field here)

`docs/audit/tasks/14-audit-nucleo.md` named two things this module would
leave for later work, `Confounding` and `FeatureAttribution`; the section
above is task 15's answer for the first. Task 16, the second, asked: for
a model's top-N features, does each one's presence track lineage rather
than phenotype (`docs/audit/audit-api.md` §3.4 -- "restringido a linaje"
and "invariante de linaje", checked per k-mer against the same lineage
labels `cv.lineage_groups()` gives `audit()` itself)?

That question is already answered, per feature, by `fastdna.explain()`,
which exists as a separate, composable call and covers §3.4's two checks
more concretely than the draft asked for: for each of a fitted model's
top features it reports how many of the cohort's lineages the k-mer
appears in (`FeatureExplanation.n_lineages_present`/`.n_lineages_total`,
`.lineage_restricted` -- §3.4's "en cuantos linajes distintos aparece"),
and whether its presence still associates with the phenotype *inside*
each lineage separately, by a named, degeneracy-guarded
Cochran-Mantel-Haenszel test rather than an unspecified "correlaciona
... dentro de cada linaje" (`.association_p_value`,
`.survives_stratification`). It also runs a check `audit-api.md` §3.4
never asked for, an equivalence-class/identifiability test, because a
lineage-marker verdict on a k-mer that is one of 847 indistinguishable
ones does not single out that particular k-mer. `explain.py`'s own
module docstring calls itself "the other half" of what `audit()`
answers, for exactly this reason.

`AuditReport` therefore carries no `feature_attribution` field, not even
the `None`-valued placeholder `14-audit-nucleo.md` anticipated. Adding
one here would either duplicate `explain()`'s stratified test and
lineage-fraction check under a second name -- precisely what "No
re-implementation of `cv.py` or `evaluation.py`" below already refuses
to do for machinery that exists elsewhere -- or it would need `audit()`
to hold a single fitted vectorizer and a single importances array to run
that logic against, which it structurally never has: every
`cross_val_score` call above clones and fits (then discards) one
estimator per fold, and none of those fitted estimators survive the
call. `explain()`'s contract is built the other way around on purpose --
an already-fitted vectorizer and an importances array supplied *by the
caller*, who has both because they fit their own pipeline once, the same
expectation `cv.permutation_importance_pvalues` already places on its
own caller for its per-feature p-values. Fitting an extra,
whole-cohort-only model inside `audit()` just to populate this field
would be new, silent computation nobody asked for, in the same spirit
the `covariates=` section above already refuses to fabricate a variance
decomposition nobody computed.

A caller who wants both reads them as two independent pieces of evidence
on the same cohort, exactly as `explain()`'s own docstring demonstrates:
fit the pipeline once, then

```python
explanation = fastdna.explain(vectorizer, model.coef_[0], paths, phenotype, top_n=20)
```

alongside `audit(pipeline, paths, phenotype)` -- composed, not merged,
because a linear model refit on the whole cohort is not any one of
`audit()`'s per-fold fits, and claiming otherwise would overstate how
tightly "the CV gap" and "which k-mers drove it" actually connect.

## A positive gap does not by itself mean population-structure leakage

Overfitting produces one too, and the two are not distinguishable from the
gap alone. Measured on a real *E. coli* + ampicillin cohort (80 complete
BV-BRC assemblies, MLST as groups) where the model has essentially no
signal to begin with -- random-CV AUC hovers at 0.53-0.58 regardless -- only
`top_features` was varied:

    features   p/n     random-CV   lineage-blocked   gap
         200   2.5        0.5326            0.5199   +0.013
       1,000  12.5        0.5799            0.5465   +0.034
       5,000  62.5        0.5666            0.4918   +0.075

The gap grows monotonically with the parameter-to-sample ratio on data
containing no learnable biology. The mechanism is straightforward once
stated: a model with enough capacity memorises individual samples, and a
memorised sample's near-clones sit in its own lineage, so grouping them out
of the training fold costs more score the more memorisation there was. That
is a real drop in a real held-out score -- `LineageKFold` is not
malfunctioning -- but attributing it to *population structure confounding
the phenotype* would be wrong.

What tells them apart is not this function: it is whether the model has
genuine signal in the first place. `score_random` far above chance with a
large gap is the leakage story; `score_random` near chance with a large gap
is capacity. Read the two numbers together, never the gap alone -- which is
also why `gap` is reported next to both scores rather than on its own.

## The gap detects dominant leakage, and is insensitive to partial leakage

Calibrated against known injected leakage
(`scripts/validation/leakage_calibration.py`, which builds cohorts whose
phenotype is `lambda`-determined by lineage and `(1-lambda)`-determined by a
transferable marker, 10 replicates per point):

    lambda    random   blocked        gap
      0.00    1.0000    1.0000    +0.0000
      0.20    0.9331    0.9056    +0.0275
      0.40    0.8674    0.8573    +0.0101
      0.60    0.8147    0.7908    +0.0239
      0.80    0.8530    0.7296    +0.1234
      1.00    0.8730    0.4463    +0.4267

The extremes behave exactly as intended -- no gap when the signal transfers,
a large one when it does not -- but the response is a threshold, not a
slope: flat at 0.01-0.03 until lineage supplies roughly 80% of the
phenotype, then sharply rising.

The mechanism is that a model learns the cheapest available signal. While a
transferable feature still explains a good share of the labels, the model
prefers it over memorising several clones' worth of k-mers, and a
transferable feature costs nothing when a lineage is held out.

**The practical consequence, which belongs in any interpretation of a small
gap**: a gap near zero means population structure is not the *dominant*
explanation of the phenotype. It does not mean the cohort is unconfounded.
`AuditReport.confounding` answers that second question directly -- it is
computed from the label vectors with no model involved, and so is unaffected
by whichever signal the model happened to find easiest.

That is not an assumption. On the same injected-leakage sweep, measured
alongside the gap:

    lambda    gap        confounding
      0.0    +0.0000        0.0769
      0.2    +0.0322        0.1859
      0.4    +0.0103        0.3272
      0.6    +0.0290        0.5576
      0.8    +0.1559        0.7698
      1.0    +0.4963        0.9864

    Spearman(lambda, gap)         = 0.829
    Spearman(lambda, confounding) = 1.000

`confounding` rises monotonically -- and close to linearly -- across exactly
the range where the gap is flat. The two are not competing estimates of one
quantity; they answer different questions, and the sweep shows each
answering its own:

- **`confounding`: how much leakage is AVAILABLE in this cohort?** A
  property of the design. No model, so no model-dependent blind spot.
- **`gap`: how much did THIS model take?** A property of the model on this
  cohort. It moves only once lineage is the easiest signal available, which
  is the same reason a model with a transferable alternative ignores
  lineage and shows no gap.

**So read them as a pair.** High confounding with a small gap is the
informative combination and the easiest to misread: the cohort *could* have
fooled a model, and this particular one resisted -- which says something
about the model, not about the cohort's safety for the next one.

## What this module does not do

Same posture as `cv.py` and `evaluation.py`, stated here for this module
specifically:

- **No verdict.** No enum, no red/yellow/green label, no fixed numeric
  threshold above which a gap counts as "bad" -- see above for why. Read
  `score_random`, `score_lineage` and `gap`. The same holds for
  `confounding`: the number is computed and reported, `audit-api.md`'s
  LOW/MODERATE/HIGH bands over it are not, and the reasoning for keeping
  those two decisions apart is in "Why this is a number and not a
  verdict" above.
- **No fix.** Exactly like `cv.LineageKFold`, `audit()` reports an honest
  estimate; it does not produce a better model. A large gap means the
  cohort/design cannot currently distinguish phenotype from lineage, and
  no amount of hyperparameter tuning changes that -- see `cv.py`'s own
  "What this module does not fix" section, which applies unchanged here.
- **No re-implementation of `cv.py` or `evaluation.py`.** `lineage_groups`
  and `LineageKFold` are imported and called, not re-derived; nothing here
  computes a Mash distance, a linkage, or a fold split independently.
- **No feature-attribution field.** `docs/audit/tasks/14-audit-nucleo.md`
  left `FeatureAttribution` as the other placeholder beside `Confounding`
  (task 15, above); `fastdna.explain()` already answers "does this top
  feature track lineage rather than phenotype" per feature, as its own
  composable call, more concretely than the original sketch asked for --
  see "Feature attribution to lineage" above for why that stays a
  separate function rather than a field grown here.
- **No repeated-counting optimization, beyond what accepting a
  `CohortCounts` directly makes possible.** Each `cross_val_score` call
  still re-transforms every training and test path through whatever
  vectorizer `estimator` wraps, exactly like `sklearn.model_selection.
  cross_val_score` always does -- `audit()` does not change that. What it
  does do now (this was the gap this note used to describe, before `paths`
  could be a `CohortCounts`): a caller can pass `audit(pipeline_using_
  KmerVectorizer(counts=cc), cc, phenotype)` -- the same `CohortCounts`
  object as both `paths` and the pipeline's own `counts=` -- and every
  `cross_val_score` fold's `.transform()` call resolves to `CohortCounts.
  subset()` (`fastdna.sklearn.KmerVectorizer`'s own `counts=` contract, an
  O(rows in that fold) slice, no FASTQ reread), *and* lineage-group
  derivation for both the primary threshold and the automatic/explicit
  leakage curve reads from that same `CohortCounts` too (`cv.
  lineage_groups_at_thresholds()`'s `CohortCounts` support), instead of
  needing real FASTQ paths that only `KmerVectorizer`'s *fit/transform*
  side could take advantage of. Before this, the fast, no-reread model path
  (`KmerVectorizer(counts=...)`) and automatic leakage-curve derivation
  (which needed real file paths to sketch from) were mutually exclusive
  within one `audit()` call -- a caller had to either give up the curve (by
  precomputing `groups=` and losing `auto_leakage_curve` entirely, since
  `groups=` supplied directly skips curve derivation) or give up the fast
  path (by passing real paths, forcing every fold to re-read and re-count
  its FASTQ files). Passing a `CohortCounts` as `paths` removes that
  exclusion: both sides read from the one upfront count. See `paths`'
  own parameter docs above for the exact mechanism.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

from . import _core
from .cohort_counts import CohortCounts
from .cv import LineageKFold, default_threshold_curve, lineage_groups, lineage_groups_at_thresholds

__all__ = [
    "audit",
    "AuditReport",
    "Confounding",
    "CovariateAudit",
    "LeakageCurvePoint",
    "DegenerateLineagesWarning",
]

# Below this fraction of samples-that-are-their-own-lineage, `LineageKFold`
# barely differs from an ordinary random splitter: with e.g. 149 lineages
# over 150 samples, almost every fold's "held-out lineage" is a single
# sample indistinguishable from a random one, so `gap` reads near zero not
# because there is no leakage to find, but because the derived grouping
# gave the splitter almost nothing to block on. Both real cohorts this
# project has run through `audit()` so far hit exactly this: the library's
# own default `lineage_threshold=0.01` produced 149/150 lineages on a real
# E. coli cohort (`scratch/amr_repro/audit_report.json`) and a small,
# easy-to-miss `gap` that a coarser, data-driven threshold on the *same*
# cohort turned into a large one (`docs/audit/ml-gaps.md`'s "the unified
# thesis" is only as trustworthy as this number is legible). `audit()`
# warns rather than silently reporting a small `gap` as if it settled the
# question.
_DEGENERATE_LINEAGE_FRACTION = 0.9

#: Used only when a cohort's own dendrogram cannot offer a threshold at all
#: (every merge height is exactly 0, i.e. every sample is identical to every
#: other at this sketch resolution). This was `audit()`'s fixed default
#: before the threshold became data-driven; it survives as a last resort,
#: not as a recommendation -- on a real E. coli cohort it produced 198
#: lineages out of 200 samples.
_FALLBACK_LINEAGE_THRESHOLD = 0.01


class DegenerateLineagesWarning(UserWarning):
    """`groups` (derived or supplied to `audit()`) puts almost every sample
    in its own lineage, so `LineageKFold` has little to block on and
    `AuditReport.gap` may read near zero even when real, coarser-scale
    leakage exists -- see `audit()`'s own module-level note above this
    class for why, and its `lineage_threshold` parameter for the knob that
    controls it when `groups` is derived rather than supplied directly."""


@dataclass(frozen=True)
class Confounding:
    """How much of the phenotype is predictable from the lineage assignment
    alone -- a property of the cohort's *design*, computed from the two
    label vectors with no model fitted and no k-mer read.

    See the module docstring's "Phenotype-vs-lineage confounding" section
    for what this measures, which statistic is used for which kind of
    phenotype and why, the four specific things the number cannot tell you
    (it is not causal, it does not bound `AuditReport.gap`, it carries no
    p-value, and it inherits whatever `cv.lineage_groups`'s
    `distance_threshold` decided), and -- in its own subsection -- why this
    is deliberately a reported number rather than the `CONFOUNDED` verdict
    `docs/audit/audit-api.md` §3.6 wanted to build on it.

    Attributes
    ----------
    statistic : str
        Which statistic `value` is. This is reported rather than assumed
        because `audit()` accepts phenotypes that do not all have a
        contingency table:

        - `"cramers_v"` -- categorical (binary or multi-class) phenotype:
          Cramer's V over the `n_lineages x n_phenotype_levels`
          contingency table, with Bergsma's (2013) bias correction.
        - `"omega_squared"` -- continuous phenotype: the bias-corrected
          eta-squared of a one-way ANOVA of phenotype on lineage. The
          same `[0, 1]` scale and the same question, without binning a
          continuous phenotype into a table (see the module docstring for
          why binning is refused).
        - `"undefined"` -- neither could be computed on this cohort;
          `value` is `nan` and `undefined_reason` says why.
    value : float
        The statistic, in `[0, 1]`. 0 means the phenotype is distributed
        across lineages no more unevenly than chance alone would make it;
        1 means the lineage label determines the phenotype exactly. `nan`
        exactly when `statistic == "undefined"`.

        Both statistics are reported *after* their bias correction, so a
        value near 0 genuinely means "no association beyond chance" even
        in a small cohort with many lineages -- the case where the
        uncorrected forms read high for no reason (module docstring).
        Both corrections can go slightly negative when the observed
        between-lineage variation falls below its chance expectation;
        that is reported as `0.0`, since "less structure than chance"
        and "no structure" are the same finding here.
    n_lineages : int
        Distinct lineage labels the statistic was computed over -- the
        contingency table's row count, and `AuditReport.n_lineages`. Read
        it together with `value`: the same value means different things
        over 3 lineages and over 300.
    n_phenotype_levels : int
        Distinct phenotype values, i.e. the contingency table's column
        count, for a categorical phenotype. `0` for a continuous one,
        which has no levels and no table.
    undefined_reason : str or None
        `None` when `value` is finite. Otherwise a sentence naming the
        degeneracy that made the statistic undefined -- a single lineage,
        a single phenotype value, a phenotype with one distinct value per
        sample, or the saturated case where every sample is its own
        lineage (in which the *uncorrected* Cramer's V would be a
        confident, meaningless 1.0 for any phenotype whatsoever, which is
        the outcome the bias correction exists to refuse).
    """

    statistic: str
    value: float
    n_lineages: int
    n_phenotype_levels: int
    undefined_reason: Optional[str]

    def __str__(self):
        if self.undefined_reason is not None:
            return f"not defined ({self.undefined_reason})"
        if self.statistic == "cramers_v":
            detail = (
                f"bias-corrected Cramer's V over a {self.n_lineages} x "
                f"{self.n_phenotype_levels} lineage-by-phenotype table"
            )
        else:
            detail = (
                "bias-corrected eta-squared (omega-squared) of a continuous phenotype "
                f"over {self.n_lineages} lineages"
            )
        return f"{self.value:.4g} ({detail})"


@dataclass(frozen=True)
class CovariateAudit:
    """One covariate's own blocked-CV comparison against the random-CV
    baseline -- see the module docstring's `covariates=` section for what
    this is and, importantly, what it is not (not a share of the gap, not
    additive with the lineage gap or with other covariates').

    Attributes
    ----------
    name : str
        The key this covariate was given under in `covariates=`.
    n_groups : int
        Number of distinct values this covariate takes across the cohort
        -- the number of groups `cv.LineageKFold(groups=...)` blocked on.
    score : float
        Mean `cross_val_score` under a `cv.LineageKFold` blocked on this
        covariate's own labels (not the lineage labels).
    score_std : float
        Standard deviation across that splitter's folds.
    gap_vs_random : float
        `AuditReport.score_random - score` -- directly comparable to
        `AuditReport.gap` (the lineage-blocked equivalent), but computed
        independently; the two do not sum to anything.
    """

    name: str
    n_groups: int
    score: float
    score_std: float
    gap_vs_random: float


@dataclass(frozen=True)
class LeakageCurvePoint:
    """One point on the leakage-vs-granularity curve `audit()` builds when
    `lineage_threshold_curve=` is given: the same lineage-blocked-CV
    comparison `AuditReport`'s top-level `score_lineage`/`gap`/`n_lineages`/
    `confounding` fields report at `lineage_threshold`, computed instead at
    one other Mash-distance threshold.

    This exists because a single `gap` at a single threshold is not a
    robust finding on its own -- see this module's motivating case (a real
    bacterial AMR cohort read `gap=-0.018` at the library's default
    threshold, where 149/150 genomes were each their own lineage, and
    `gap=+0.148` at a coarser, data-driven threshold on the *same* cohort).
    A curve across thresholds turns that single, contestable number into a
    result: where leakage appears as the blocking granularity coarsens, not
    a single figure that depends on a constant nobody had strong grounds to
    pick. `n_lineages` is reported alongside every point for the same
    reason `AuditReport.n_lineages` is reported alongside `gap` itself --
    the same `gap` value means something different at 3 lineages than at
    150.

    Attributes
    ----------
    threshold : float
        The Mash-distance threshold this point was cut at -- one entry of
        the `lineage_threshold_curve` sequence given to `audit()`.
    n_lineages : int
        Distinct lineage labels found at this threshold. Coarser (larger)
        thresholds merge more samples into fewer, larger lineages, so this
        is monotonically non-increasing as `threshold` increases, for a
        fixed cohort and dendrogram (`cv.lineage_groups_at_thresholds`'s
        "one dendrogram, many cuts" property: every cut comes from the same
        underlying clustering).
    score_lineage : float
        Mean `cross_val_score` under `cv.LineageKFold(groups=...)` at this
        threshold's grouping -- the same quantity `AuditReport.score_lineage`
        is, at this point's threshold instead of the primary one. `nan` if
        this threshold's grouping has fewer lineages than `n_splits` (see
        `n_lineages` on this same point for why) -- a wide sweep routinely
        includes thresholds too coarse to block on, and that must not
        discard the rest of the curve or of `audit()`'s report.
    score_lineage_std : float
        Standard deviation across that splitter's folds. `nan` under the
        same condition as `score_lineage`.
    gap : float
        `AuditReport.score_random` (the single, threshold-independent
        random-CV baseline) minus this point's `score_lineage`. Directly
        comparable across points, and to `AuditReport.gap` itself, because
        `score_random` is the same number everywhere on the curve -- only
        the lineage-blocked side changes with `threshold`. `nan` under the
        same condition as `score_lineage`.
    confounding : Confounding
        `AuditReport.confounding`, recomputed at this threshold's grouping
        -- see `Confounding` and the module docstring's "Phenotype-vs-
        lineage confounding" section for what this measures.

    The point whose `threshold` equals `AuditReport.lineage_threshold`
    itself is not recomputed: `audit()` reuses the primary point's already-
    computed `score_lineage`/`score_lineage_std`/`gap`/`confounding`
    (and `n_lineages`) rather than refitting the same splitter a second
    time, exactly the reuse `cv.lineage_groups_at_thresholds`'s single-
    dendrogram cost model exists to make possible.
    """

    threshold: float
    n_lineages: int
    score_lineage: float
    score_lineage_std: float
    gap: float
    confounding: Confounding


@dataclass(frozen=True)
class AuditReport:
    """What :func:`audit` computed. See the module docstring for what each
    number means, how it was computed, and -- for `covariates` -- what it
    deliberately does not claim.

    Attributes
    ----------
    n_samples : int
    n_splits : int
        Folds used for both the random and the lineage-blocked evaluation.
    scoring : str or None
        The resolved `sklearn` scorer name actually used (`None` means
        each `cross_val_score` call fell back to `estimator`'s own default
        scorer -- accuracy for a classifier, R^2 for a regressor).
    score_random : float
        Mean `cross_val_score` under ordinary (random) cross-validation.
    score_random_std : float
        Standard deviation across the random splitter's folds.
    score_lineage : float
        Mean `cross_val_score` under `cv.LineageKFold`.
    score_lineage_std : float
        Standard deviation across the lineage-blocked splitter's folds.
    gap : float
        `score_random - score_lineage` -- the headline number. Positive
        means random CV reported a higher score, i.e. population structure
        was likely leaking across the train/test boundary under random CV.
        Can be negative or near zero; neither is an error (see the module
        docstring's discussion of the 2026 preprint finding the reverse
        pattern on some cohorts).

        **`nan` when the grouping was degenerate** -- see
        `gap_undefined_reason`. A degenerate grouping is one where
        `LineageKFold` had almost nothing to block on, so the difference
        between the two splitters measures nothing; reporting it as a small
        float invites exactly the wrong conclusion ("no leakage here"). The
        two `score_*` fields stay populated, because those were really
        measured; only their difference is withheld.
    gap_undefined_reason : str, optional
        `None` on an ordinary report. A short sentence naming why `gap` is
        `nan`, in the same "return a reason, not a number" convention
        `Confounding.undefined_reason` already uses for its own degenerate
        cases.

        This exists because the failure it guards against is silent: a
        degenerate grouping produces a small, well-formed float, and "small
        gap" reads as "no leakage" to anyone who does not also check how
        many lineages were found. The two are indistinguishable from the
        number alone, which is why the number is withheld rather than
        annotated. This module's own notes record the same reversal on an
        earlier cohort (`-0.018` at the default, `+0.148` at a coarser,
        data-driven threshold).

        A worked example of how far the reading can be off is deliberately
        NOT cited here: the one this project measured turned out to be an
        artefact of a data-truncation bug (see the retraction at the top of
        `docs/validation-real-data.md`'s Track D). The guard stands on the
        argument above, which needs no cohort.
    n_lineages : int
        Distinct lineage labels found (or supplied via `groups=`).
    lineage_threshold : float
        The Mash-distance threshold used by `cv.lineage_groups` (irrelevant,
        and reported as `float("nan")`, when `groups=` was supplied
        directly instead of being derived).
    confounding : Confounding
        How much of the phenotype the lineage assignment alone predicts --
        see `Confounding` and the module docstring's
        "Phenotype-vs-lineage confounding" section. Unlike every other
        number in this report it involves no model and no fold: it is
        computed from `phenotype` and the lineage labels directly, and
        answers the prior question the gap cannot ("could this cohort ever
        have separated the two signals?"). It is a measurement, not a
        verdict, and it is deliberately reported without the
        LOW/MODERATE/HIGH bands `docs/audit/audit-api.md` proposed for it.
    per_fold : pyarrow.Table
        Columns `cv_kind` (`"random"`, `"lineage"`, or
        `"covariate:<name>"`), `fold` (0-based), `score` -- one row per
        fold of every splitter run, the raw numbers `score_random`/
        `score_lineage`/`CovariateAudit.score` are (`numpy.nanmean`-)
        averages of. A `score` of `NaN` here is itself informative, not a
        computation error: with `scoring="roc_auc"` (the default for a
        binary phenotype -- see `_default_scoring`), a group-blocked fold
        whose held-out set happens to be entirely one phenotype class has
        an undefined ROC-AUC (`sklearn.metrics.roc_auc_score` itself
        returns `nan` for that fold, with `UndefinedMetricWarning`). A
        strongly lineage-confounded cohort makes this common for
        lineage-/covariate-blocked folds specifically, because a whole
        lineage can be (close to) one phenotype class by construction --
        which is itself evidence for the confounding this function
        measures, not a bug to hide. `score_lineage`/`CovariateAudit.score`
        are `numpy.nanmean` over the valid folds (`nan` only if every fold
        of that splitter was undefined); pick `scoring="accuracy"` (or
        another metric defined for a single-class fold) explicitly if NaN
        folds make the aggregate too sparse to be useful for a given
        cohort.
    covariates : tuple of CovariateAudit
        Empty when `covariates=` was not given to :func:`audit`.
    leakage_curve : tuple of LeakageCurvePoint, or None
        By default (`groups=` omitted, `auto_leakage_curve=True`, the
        default), always populated with a data-driven curve -- see
        `auto_leakage_curve`'s own docstring for why a single-threshold
        `gap` is not, by itself, a robust finding. `None` only when
        `groups=` was supplied directly (there is no dendrogram to sweep
        when the caller hands in labels instead of letting `audit()`
        derive them), or when a caller explicitly passed
        `auto_leakage_curve=False` and no `lineage_threshold_curve=`. When
        present, one `LeakageCurvePoint` per swept threshold (explicit
        `lineage_threshold_curve`, or the automatic curve's own points), in
        the same order, each the same random-vs-lineage-blocked comparison
        this report's own top-level fields make at `lineage_threshold`,
        made instead at that point's own threshold, from the same single
        dendrogram (see `cv.lineage_groups_at_thresholds`).
    """

    n_samples: int
    n_splits: int
    scoring: Optional[str]
    score_random: float
    score_random_std: float
    score_lineage: float
    score_lineage_std: float
    gap: float
    n_lineages: int
    lineage_threshold: float
    confounding: Confounding
    per_fold: pa.Table
    covariates: Tuple[CovariateAudit, ...]
    leakage_curve: Optional[Tuple[LeakageCurvePoint, ...]] = None
    gap_undefined_reason: Optional[str] = None

    @property
    def _gap_label(self) -> str:
        """How `gap` renders in every human-facing view. A withheld gap must
        not print as `+nan`: that reads like a numerical accident rather than
        a deliberate refusal to answer, and the whole point of withholding it
        is that the reader changes what they do next."""
        if self.gap_undefined_reason:
            return "undefined (degenerate grouping)"
        return f"{self.gap:+.4g}"

    def to_markdown(self) -> str:
        scoring_label = self.scoring if self.scoring is not None else "estimator default"
        threshold_label = "n/a (groups= supplied directly)" if np.isnan(self.lineage_threshold) else f"{self.lineage_threshold:g}"
        lines = [
            f"# FastDNA leakage audit -- {self.n_samples} samples, n_splits={self.n_splits}, "
            f"scoring={scoring_label}",
            "",
            "| cv | score | std |",
            "|---|---:|---:|",
            f"| random | {self.score_random:.4g} | {self.score_random_std:.4g} |",
            f"| lineage-blocked | {self.score_lineage:.4g} | {self.score_lineage_std:.4g} |",
        ]
        for c in self.covariates:
            lines.append(f"| covariate:{c.name} | {c.score:.4g} | {c.score_std:.4g} |")
        lines += [
            "",
            f"Gap (random - lineage-blocked): {self._gap_label}",
            f"{self.n_lineages} lineages (Mash distance threshold {threshold_label}).",
            f"Phenotype-vs-lineage confounding: {self.confounding}",
        ]
        if self.gap_undefined_reason:
            lines += ["", f"**Gap withheld.** {self.gap_undefined_reason}"]
        if self.covariates:
            lines += [
                "",
                "| covariate | groups | gap vs random |",
                "|---|---:|---:|",
            ]
            for c in self.covariates:
                lines.append(f"| {c.name} | {c.n_groups} | {c.gap_vs_random:+.4g} |")
            lines.append(
                "Each covariate's gap is independent of the lineage gap above and of every "
                "other covariate's -- they are not shares of one total and do not sum to "
                "anything meaningful (see the module docstring)."
            )
        if self.leakage_curve is not None:
            lines += [
                "",
                "## Leakage curve",
                "",
                "| threshold | lineages | score_lineage | gap | confounding |",
                "|---:|---:|---:|---:|---:|",
            ]
            for point in self.leakage_curve:
                lines.append(
                    f"| {point.threshold:g} | {point.n_lineages} | {point.score_lineage:.4g} | "
                    f"{point.gap:+.4g} | {point.confounding.value:.4g} |"
                )
            lines.append(
                "The gap is not a single number but a curve across blocking granularity: "
                "each row cuts the same dendrogram at a different Mash-distance threshold, so "
                "a threshold is not a free parameter chosen to make the gap read one way or "
                "the other -- see where it changes, not just its value at one point."
            )
        lines += [
            "",
            "A positive gap means random cross-validation reported a higher score than "
            "lineage-blocked cross-validation did -- consistent with population structure "
            "leaking across the train/test boundary under the random split. This module "
            "does not decide whether that gap is acceptable for your use case; see "
            "`fastdna.cv`'s own docstring for why the lineage-blocked number is honest, not "
            "necessarily 'better'.",
            "",
            "The confounding figure is a separate, model-free measurement: how much of the "
            "phenotype the lineage assignment alone predicts, on a 0-1 scale, bias-corrected "
            "for the number of lineages. It is reported without a threshold on purpose -- a "
            "high value means this cohort's design cannot separate phenotype from lineage no "
            "matter which splitter or model is used, but where 'high' starts is a judgment "
            "about your cohort, not a constant this module can supply.",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"AuditReport(n_samples={self.n_samples}, score_random={self.score_random:.4g}, "
            f"score_lineage={self.score_lineage:.4g}, gap={self._gap_label}, "
            f"n_lineages={self.n_lineages}, "
            f"confounding={self.confounding.value:.4g} ({self.confounding.statistic}), "
            f"n_covariates={len(self.covariates)})"
        )

    def __str__(self):
        return self.to_markdown()

    def _repr_html_(self):
        try:
            import pandas as pd
        except ImportError:
            return f"<pre>{self.to_markdown()}</pre>"

        rows = {
            "cv": ["random", "lineage-blocked"] + [f"covariate:{c.name}" for c in self.covariates],
            "score": [self.score_random, self.score_lineage] + [c.score for c in self.covariates],
            "std": [self.score_random_std, self.score_lineage_std] + [c.score_std for c in self.covariates],
            "gap_vs_random": [self.gap, self.gap] + [c.gap_vs_random for c in self.covariates],
        }
        # The random row's own "gap vs random" is 0 by definition, not `self.gap`
        # (which is random-vs-lineage) -- fixed up explicitly rather than
        # left to read as a copy-paste artifact.
        rows["gap_vs_random"][0] = 0.0
        df = pd.DataFrame(rows)
        header = (
            f"<p><b>AuditReport</b> &mdash; {self.n_samples} samples, gap={self._gap_label}, "
            f"{self.n_lineages} lineages, phenotype-vs-lineage confounding "
            f"{self.confounding}</p>"
        )
        return header + df.to_html(index=False)


def _safe_is_classifier(estimator):
    """`sklearn.base.is_classifier`, but treating a duck-typed estimator
    that scikit-learn cannot answer the question for (anything that is
    not a `BaseEstimator` subclass and defines no `__sklearn_tags__` of
    its own -- `fastdna.mic.MicRegressor` is exactly this shape) as "no
    answer" rather than letting it propagate as an exception.

    Every caller in this module documents `is_classifier`/`is_regressor`
    returning `False` for such an object as the deliberate, no-crash
    fallback path (see `_phenotype_is_categorical`'s own docstring). That
    was true through scikit-learn's older tag machinery; the 1.6
    `__sklearn_tags__` rewrite changed `is_classifier`/`is_regressor` to
    raise `AttributeError` for a non-`BaseEstimator` duck-typed estimator
    instead of returning `False`, which silently broke every one of this
    module's own documented fallbacks -- including `audit()` itself, which
    crashed outright (before ever reaching a fold) on a plain
    `MicRegressor()`. This wrapper (and `_safe_is_regressor` below)
    restores the documented behaviour under both old and new scikit-learn
    tag machinery.
    """
    from sklearn.base import is_classifier

    try:
        return is_classifier(estimator)
    except AttributeError:
        return False


def _safe_is_regressor(estimator):
    """The `is_regressor` counterpart of `_safe_is_classifier` -- see its
    docstring for why this wrapper exists."""
    from sklearn.base import is_regressor

    try:
        return is_regressor(estimator)
    except AttributeError:
        return False


def _default_scoring(estimator, y):
    """`"roc_auc"` for a binary-classification `estimator`/`y` pair, `None`
    (each `cross_val_score` call then falls back to `estimator`'s own
    default scorer -- accuracy for a classifier, R^2 for a regressor)
    otherwise.

    Narrowed from `docs/audit/audit-api.md`'s original "roc_auc if binary,
    r2 if continuous, detected not guessed": accuracy is exactly the
    metric `fastdna.evaluation`'s own module docstring documents as
    misleadingly high under class imbalance (a 97%-accuracy "always
    predict the majority class" model), so a binary audit defaults away
    from it. Multi-class and regression targets fall back to sklearn's own
    default rather than guessing a specific metric this module cannot
    justify picking on the caller's behalf.
    """
    if _safe_is_classifier(estimator) and np.unique(np.asarray(y)).size == 2:
        return "roc_auc"
    return None


def _phenotype_is_categorical(estimator, y):
    """Whether the phenotype should be treated as a set of levels (Cramer's
    V over a contingency table) or as a continuous quantity (omega-squared
    over a one-way ANOVA) -- see `_confounding`.

    Decided from `estimator` first, on the same reasoning `_default_scoring`
    already uses: the caller chose a classifier or a regressor, and that
    choice *is* their statement about what their labels are. It is a more
    reliable signal than the array's dtype, which cannot tell integer-coded
    classes from an integer-valued measurement, and happily reports
    `float64` for the perfectly ordinary `y = np.array([0.0, 1.0, 1.0])`.

    Only when `estimator` is neither (a duck-typed object with no
    `is_classifier`/`is_regressor` answer) does this fall back to the dtype,
    and that fallback is a guess, stated as one: non-numeric, boolean and
    integer arrays are read as categorical, floating-point ones as
    continuous. An integer-coded continuous phenotype (say an MIC in
    doubling dilutions) reaching this branch would be measured with
    Cramer's V instead of omega-squared -- `Confounding.statistic` says
    which one ran, so the misreading is visible in the report rather than
    silent.
    """
    if _safe_is_classifier(estimator):
        return True
    if _safe_is_regressor(estimator):
        return False
    return np.asarray(y).dtype.kind in "biuOSU"


def _contingency_table(groups, phenotype):
    """The `n_lineages x n_phenotype_levels` count table, plus its two axis
    sizes, built with `numpy.bincount` over a flattened pair index.

    Deliberately not `scipy.stats.contingency.crosstab` and deliberately
    not followed by `scipy.stats.chi2_contingency`: the chi-squared
    *statistic* is four array operations (below), and only its *p-value*
    would need a distribution function from `scipy`. `audit()` does not
    report that p-value (module docstring: it is invalid on a sparse
    lineage-by-phenotype table and uninformative even where it is valid),
    so taking a `scipy` dependency for it would be paying a real cost --
    `audit(..., groups=...)` currently runs with `scipy` absent -- to
    obtain a number this module has decided not to publish.
    """
    _, lineage_index = np.unique(np.asarray(groups), return_inverse=True)
    _, phenotype_index = np.unique(np.asarray(phenotype), return_inverse=True)
    lineage_index = np.asarray(lineage_index).ravel()
    phenotype_index = np.asarray(phenotype_index).ravel()

    n_rows = int(lineage_index.max()) + 1
    n_cols = int(phenotype_index.max()) + 1
    flat = lineage_index * n_cols + phenotype_index
    observed = np.bincount(flat, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    return observed.astype(np.float64), n_rows, n_cols


def _cramers_v_bias_corrected(groups, phenotype):
    """Bergsma's (2013) bias-corrected Cramer's V of the lineage-by-phenotype
    contingency table, as `(value, n_lineages, n_levels, undefined_reason)`.

    With `phi2 = chi2 / n` over an `r x c` table:

        phi2~ = max(0, phi2 - (r-1)(c-1)/(n-1))
        r~    = r - (r-1)^2/(n-1)
        c~    = c - (c-1)^2/(n-1)
        V~    = sqrt(phi2~ / min(r~ - 1, c~ - 1))

    The subtracted term is exactly `phi2`'s expectation under independence,
    so `V~` answers "how much association is there *beyond* what a table
    this shape produces by chance" -- which is the only version of the
    question worth asking here, because the table's shape is not a free
    choice: `r` is however many lineages `cv.lineage_groups` found at the
    caller's Mash threshold, and on a clonal cohort it is routinely a large
    fraction of `n`. See the module docstring for the worked case (a
    40-sample, 20-lineage cohort reading 0.72 on a coin-flip phenotype
    uncorrected, ~0 corrected).

    Returns `nan` with a reason, rather than a number, in the four cases
    where `min(r~ - 1, c~ - 1) <= 0` -- which is to say wherever the table
    is degenerate: `r < 2`, `c < 2`, `r == n` or `c == n`. `r == n` is not
    a hypothetical here (a `distance_threshold` set below the cohort's
    within-lineage divergence makes every sample a singleton lineage), and
    it is the case that most needs refusing: uncorrected V is *exactly*
    1.0 there for every phenotype, including a coin flip, so a report that
    printed it would state maximum confounding on the strength of nothing
    at all.
    """
    observed, r, c = _contingency_table(groups, phenotype)
    n = float(observed.sum())

    if r < 2:
        return (
            float("nan"),
            r,
            c,
            "the cohort has only one lineage, so there is no lineage structure for the "
            "phenotype to be confounded with",
        )
    if c < 2:
        return (
            float("nan"),
            r,
            c,
            "the phenotype takes a single value across the whole cohort, so there is "
            "nothing for the lineage assignment to predict",
        )
    if r >= n:
        return (
            float("nan"),
            r,
            c,
            f"every one of the {int(n)} samples is its own lineage, so the contingency "
            "table is saturated: an uncorrected Cramer's V would read exactly 1.0 here "
            "for any phenotype at all, and the bias-corrected form is undefined (its "
            "denominator is exactly 0). Raise lineage_threshold so near-clonal samples "
            "merge into fewer, larger lineages, or supply groups= from a coarser "
            "assignment",
        )
    if c >= n:
        return (
            float("nan"),
            r,
            c,
            f"the phenotype takes a distinct value for each of the {int(n)} samples "
            "while being treated as categorical, which makes the contingency table "
            "saturated and the bias-corrected statistic undefined. A phenotype with one "
            "level per sample is a continuous measurement; pass it to a regressor (or as "
            "a float array) so it is measured with omega-squared instead",
        )

    # chi-squared without scipy: expected counts are the outer product of the
    # margins over n, and every one of them is strictly positive because both
    # margins come from np.unique levels that each have at least one member.
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    chi2 = float((((observed - expected) ** 2) / expected).sum())

    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    c_corrected = c - (c - 1) ** 2 / (n - 1)
    denominator = min(r_corrected - 1.0, c_corrected - 1.0)

    value = float(np.sqrt(phi2_corrected / denominator))
    # Clamped only against floating-point overshoot: the algebra bounds V~ by
    # 1 whenever the four guards above have passed (at perfect association
    # the ratio reduces to (n-r)/(n-c) or (c-1)/(r-1), both <= 1).
    return min(1.0, value), r, c, None


def _omega_squared(groups, phenotype):
    """Bias-corrected eta-squared (omega-squared) of a one-way ANOVA of a
    continuous `phenotype` on `groups`, as `(value, n_lineages,
    undefined_reason)`.

    The fraction of the phenotype's variance that lies *between* lineages
    rather than within them, on the same `[0, 1]` scale as Cramer's V and
    with the same chance term removed (Olejnik & Algina 2003):

        omega^2 = (SS_between - (k-1) * MS_within) / (SS_total + MS_within)

    Plain eta-squared -- `SS_between / SS_total`, which is what
    `docs/audit/audit-api.md` §3.3 named -- has expectation `(k-1)/(n-1)`
    under the null of no lineage effect, so on a 40-sample cohort split
    into 20 lineages it reads ~0.49 for a phenotype drawn independently of
    lineage. That is the same defect, in the same direction and for the
    same reason, as uncorrected Cramer's V, and it gets the same treatment
    so that the two statistics remain comparable: a `0.1` reported for a
    binary phenotype and a `0.1` reported for a continuous one both mean
    "a tenth more structure than chance", not "a tenth by one convention
    and something else by another".

    Negative values -- less between-lineage variation than chance produces
    -- are reported as `0.0`, the usual convention, and the honest one
    here: "no confounding" and "less confounding than chance" are the same
    finding, and a negative number on a scale documented as `[0, 1]` would
    invite being read as its own kind of signal.

    `nan` with a reason when the ANOVA has no residual degrees of freedom
    (`k == n`, every lineage a singleton -- the exact analogue of the
    saturated contingency table), when the phenotype is constant, or when
    it is not numeric at all.
    """
    groups = np.asarray(groups)
    try:
        y = np.asarray(phenotype, dtype=np.float64)
    except (TypeError, ValueError):
        return (
            float("nan"),
            int(np.unique(groups).size),
            "the phenotype was treated as continuous (the estimator is a regressor, or "
            "its dtype is floating-point) but could not be read as an array of numbers",
        )

    _, lineage_index = np.unique(groups, return_inverse=True)
    lineage_index = np.asarray(lineage_index).ravel()
    k = int(lineage_index.max()) + 1
    n = y.size

    if k < 2:
        return (
            float("nan"),
            k,
            "the cohort has only one lineage, so there is no lineage structure for the "
            "phenotype to be confounded with",
        )
    if k >= n:
        return (
            float("nan"),
            k,
            f"every one of the {n} samples is its own lineage, so the one-way ANOVA has "
            "no within-lineage degrees of freedom left: every sample is its own group "
            "mean, an uncorrected eta-squared would read exactly 1.0 here for any "
            "phenotype at all, and the bias-corrected form is undefined. Raise "
            "lineage_threshold so near-clonal samples merge into fewer, larger lineages",
        )

    grand_mean = float(y.mean())
    ss_total = float(((y - grand_mean) ** 2).sum())
    if ss_total <= 0.0:
        return (
            float("nan"),
            k,
            "the phenotype has zero variance (every sample carries the same value), so "
            "there is no variation for the lineage assignment to explain",
        )

    counts = np.bincount(lineage_index, minlength=k).astype(np.float64)
    group_means = np.bincount(lineage_index, weights=y, minlength=k) / counts
    ss_between = float((counts * (group_means - grand_mean) ** 2).sum())
    ss_within = max(0.0, ss_total - ss_between)
    ms_within = ss_within / (n - k)

    value = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return float(min(1.0, max(0.0, value))), k, None


def _confounding(estimator, groups, phenotype):
    """The `Confounding` reported by :func:`audit` -- see that dataclass and
    the module docstring's "Phenotype-vs-lineage confounding" section.

    Dispatches on `_phenotype_is_categorical`: Cramer's V (bias-corrected)
    over the contingency table for a categorical phenotype, omega-squared
    for a continuous one. Both are bias-corrected forms on a `[0, 1]`
    scale, so the two branches produce numbers meaning the same thing.

    `estimator` is read only to decide which of the two the phenotype is;
    it is never fitted, cloned or otherwise touched here. Nothing about
    this number depends on the model -- that independence is the point of
    reporting it beside `AuditReport.gap`, which depends on the model
    entirely.
    """
    if _phenotype_is_categorical(estimator, phenotype):
        value, n_lineages, n_levels, reason = _cramers_v_bias_corrected(groups, phenotype)
        statistic = "undefined" if reason is not None else "cramers_v"
        return Confounding(
            statistic=statistic,
            value=value,
            n_lineages=n_lineages,
            n_phenotype_levels=n_levels,
            undefined_reason=reason,
        )

    value, n_lineages, reason = _omega_squared(groups, phenotype)
    return Confounding(
        statistic="undefined" if reason is not None else "omega_squared",
        value=value,
        n_lineages=n_lineages,
        # A continuous phenotype has no levels and no contingency table; 0
        # says that, rather than reporting a count of distinct float values
        # that would look like a table dimension and is not one.
        n_phenotype_levels=0,
        undefined_reason=reason,
    )


def _random_cv_splitter(estimator, y, n_splits, random_state):
    """The ordinary, non-lineage-aware baseline splitter `audit()` compares
    against: `StratifiedKFold` when `estimator` is a classifier and every
    class has at least `n_splits` members (`StratifiedKFold`'s own
    requirement), `KFold` otherwise.

    This is the same automatic choice `sklearn.model_selection.
    cross_val_score(..., cv=<int>)` makes internally via `check_cv`, made
    explicit here so it is stated rather than hidden -- and, unlike
    `check_cv`'s own default, always shuffled: an unshuffled split over
    paths in whatever order the caller's list happens to be in is its own
    source of leakage if that order correlates with anything (e.g.
    collection batch), which is exactly the class of confound this module
    exists to surface, not reproduce by accident.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if _safe_is_classifier(estimator):
        counts = np.unique(np.asarray(y), return_counts=True)[1]
        if counts.min() >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _nan_aware_mean(values):
    """`numpy.nanmean`, except an all-NaN input returns `nan` directly
    instead of numpy's own `RuntimeWarning` + `nan` for that case -- the
    quiet, expected result here (see `_nan_aware_std` and the module/
    `AuditReport` docstrings for why NaN folds happen at all: a
    `scoring="roc_auc"`-scored fold whose held-out set is, by construction
    of a group-blocked split, entirely one phenotype class).
    """
    values = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _nan_aware_std(values):
    """`numpy.nanstd`, with the same all-NaN short-circuit as
    `_nan_aware_mean`."""
    values = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanstd(values))


def _validate_covariates(covariates, n_samples):
    if covariates is None:
        return {}
    if not isinstance(covariates, Mapping):
        raise TypeError(
            f"covariates must be a mapping of {{name: array-like}}, got "
            f"{type(covariates).__name__}. Pass e.g. covariates={{'batch': batch_ids}}."
        )
    validated = {}
    for name, values in covariates.items():
        values = np.asarray(values)
        if values.shape[0] != n_samples:
            raise ValueError(
                f"covariates[{name!r}] has {values.shape[0]} entries but there are "
                f"{n_samples} samples. Pass one covariate value per sample, in the same "
                "order as paths."
            )
        validated[name] = values
    return validated


def audit(
    estimator: Any,  # unfitted/fitted sklearn estimator or Pipeline; soft dependency, not imported at module level
    paths: Union[Sequence[Any], "CohortCounts"],  # ordinarily real FASTQ(.gz) paths, a precomputed feature matrix when groups= is given, or a fastdna.CohortCounts -- see below
    phenotype: Union[Sequence[Any], np.ndarray],
    *,
    groups: Optional[Union[Sequence[int], np.ndarray]] = None,
    covariates: Optional[Mapping[str, Union[Sequence[Any], np.ndarray]]] = None,
    n_splits: int = 5,
    scoring: Optional[Union[str, Callable[..., float]]] = None,
    k: int = 21,
    sketch_size: int = 1000,
    lineage_threshold: Optional[float] = None,
    lineage_threshold_curve: Optional[Sequence[float]] = None,
    auto_leakage_curve: bool = True,
    default_curve_points: int = 5,
    random_state: Optional[int] = 0,
) -> AuditReport:
    """Fits and scores `estimator` under ordinary (random) cross-validation
    and under `cv.LineageKFold`, and reports the gap -- the arXiv 2502.07749
    / PLOS Biology 2025 finding (see the module docstring) made concrete
    for this specific model and cohort, rather than cited as an abstract
    claim about the field.

    Parameters
    ----------
    estimator : unfitted (or already-fitted -- see below) scikit-learn
        estimator, or a `sklearn.pipeline.Pipeline` whose first step is a
        `fastdna.sklearn.KmerVectorizer` (or anything duck-typed like it)
        Cloned with `sklearn.base.clone` before every fit, the same
        convention `cv.permutation_importance_pvalues` uses -- so
        `estimator` may already be fitted (its fit state is discarded;
        only its constructor parameters survive `clone`) or not, and it is
        never mutated. When `paths` is a list of real FASTQ paths (the
        ordinary case; see `paths` below), `estimator` needs to accept
        those paths directly as `X`, which a `Pipeline([("kmers",
        KmerVectorizer(...)), ("clf", ...)])` does -- exactly the usage
        `cv.LineageKFold`'s own docstring already demonstrates with
        `cross_val_score`. When `paths` is a `fastdna.CohortCounts`
        instead (see `paths` below), `estimator` needs to accept
        `CohortCounts.sample_ids` as `X`, which a
        `Pipeline([("kmers", KmerVectorizer(counts=paths)), ("clf", ...)])`
        does -- the same `counts=` contract `fastdna.sklearn.
        KmerVectorizer` already documents for cross-validation without
        recounting.
    paths : sequence, or fastdna.CohortCounts
        The cohort this function evaluates `estimator` over. Three shapes,
        all pre-existing behaviour except the third:

        - **A list of real FASTQ(.gz) paths** (the ordinary case, and the
          default assumption everywhere else in this docstring): the `X`
          given to `cross_val_score` for every splitter this function runs,
          in order, and also what `cv.lineage_groups`/`cv.
          lineage_groups_at_thresholds`/`cv.default_threshold_curve` read
          the cohort's genomes from to derive `groups` below (each real
          file is opened and streamed once per distinct sketching pass
          this function makes).
        - **A precomputed feature matrix**, when `groups=` is supplied
          directly instead of derived: lineage detection never runs in that
          case, so `paths` is passed to `estimator` unexamined and does not
          need to be real files -- it can be a precomputed feature matrix
          and `estimator` a plain classifier, exactly like
          `cv.LineageKFold(groups=...)`'s own `paths`-free usage.
        - **A `fastdna.CohortCounts`** (e.g. from `fastdna.count_cohort()`):
          the same cohort, already counted once. `X` for `cross_val_score`
          becomes `list(paths.sample_ids)` -- what `KmerVectorizer(counts=
          paths)` inside `estimator`'s pipeline expects to receive, via its
          own `.subset()` contract (`python/fastdna/sklearn.py`'s
          `_count_cohort()`) -- and lineage-group derivation (both the
          primary threshold and the automatic/explicit curve, see
          `lineage_threshold_curve`/`auto_leakage_curve` below) also reads
          from this same `CohortCounts`, via `cv.lineage_groups_at_
          thresholds()`'s own `CohortCounts` support, instead of `cv.
          lineage_groups`/`cv.compare_all()` reopening every FASTQ file.
          Net effect: after the one `count_cohort()` call that built
          `paths`, `audit()` reads zero additional FASTQ bytes, no matter
          how many curve points `auto_leakage_curve`/
          `lineage_threshold_curve` sweep -- this is what closes the gap
          documented in the module docstring's "No repeated-counting
          optimization" section between the fast, no-reread model path
          (`KmerVectorizer(counts=...)`) and automatic leakage-curve
          derivation, which previously required real file paths and could
          not be used in the same call. `k`/`sketch_size` below still
          govern the *comparison* sketch size, but `k` specifically is
          always overridden to `paths.k` in this case -- see `cv.
          _mash_distance_matrix()`'s own docstring for exactly why a
          silently-deferred `k` is the deliberate choice here, not an
          oversight -- and `min_count > 1` on this `CohortCounts` changes
          what the derived lineage groups see (singleton k-mers already
          dropped before sketching), the same real, documented difference
          noted there.
    phenotype : array-like, length `len(paths)`
        One label per sample, in `paths` order (`paths.sample_ids` order,
        when `paths` is a `CohortCounts`).
    groups : array-like of int, optional
        Precomputed lineage labels (e.g. from an earlier
        `cv.lineage_groups()` call, or a real phylogeny's clades). `None`
        (the default) derives them with `cv.lineage_groups(paths, k=k,
        sketch_size=sketch_size, distance_threshold=lineage_threshold)`,
        which needs `paths` to be either real FASTQ files or a
        `fastdna.CohortCounts` -- see `paths` above for what changes
        between the two.
    covariates : mapping of str to array-like, optional
        One additional blocked-CV comparison per entry -- see the module
        docstring's `covariates=` section for exactly what is computed and
        what is deliberately not claimed. Each array must have one value
        per sample, in `paths` order.
    n_splits : int, default 5
        Folds for both the random and the lineage-blocked evaluation (and
        every covariate's, if any) -- the same splitter count is what
        makes the resulting gap attributable to the *splitter*, not to a
        difference in fold count. Forwarded to `cv.LineageKFold`, which
        raises if it exceeds the number of distinct lineages found; that
        error is not caught or relaxed here.
    scoring : str or callable, optional
        Forwarded to every `cross_val_score` call. `None` (the default)
        resolves to `"roc_auc"` for a binary-classification `estimator`
        (see `_default_scoring`), or to `None` itself (each estimator's
        own default scorer) otherwise.
    k, sketch_size
        Forwarded to `cv.lineage_groups` when `groups` is not given;
        ignored otherwise. `k=21` matches `cv.lineage_groups`'s own default;
        `sketch_size=1000` matches `fastdna.sketch()`'s. `k` specifically is
        further overridden -- silently, and always -- to `paths.k` when
        `paths` is a `fastdna.CohortCounts`; see `paths` above and `cv.
        _mash_distance_matrix()`'s own docstring for why.
    lineage_threshold : float, optional
        The Mash distance at which `cv.lineage_groups` cuts the dendrogram.
        Ignored when `groups=` is supplied.

        **`None` (the default) derives it from this cohort**, as the median
        of `cv.default_threshold_curve`'s points -- i.e. the middle of the
        range of granularities this cohort's own merge heights actually
        span. Pass a float to pin it.

        It is not a constant because Mash distance has no universal scale:
        what separates two lineages depends on how divergent the cohort is,
        which is the same argument `default_threshold_curve` already makes
        for the curve. The previous fixed default of 0.01 was measurably
        wrong on real data, and wrong in *both directions* depending on the
        cohort -- which is the clearest possible argument against any
        constant. On BV-BRC *E. coli* assemblies it put 198 of 200 genomes
        in their own lineage on one cohort, and merged 80 genomes into 15
        groups on another whose real structure is 38 sequence types. The
        derived value on the second is 0.0041, giving 40 groups and
        recovering the MLST partition at **ARI 0.931**, against 0.512 for
        the constant.

        Deriving is usually free: when a leakage curve is computed at all
        (the default, see `auto_leakage_curve`) its points are already in
        hand. With `auto_leakage_curve=False` this costs one extra
        dendrogram pass, which is still cheaper than being quietly wrong.
    lineage_threshold_curve : sequence of float, optional
        Additional Mash-distance thresholds at which to repeat the
        lineage-blocked comparison, producing `AuditReport.leakage_curve` --
        see `LeakageCurvePoint`. Given explicitly, exactly those thresholds
        are swept and `auto_leakage_curve` is not consulted. `None` (the
        default) does NOT mean "no curve" -- see `auto_leakage_curve`
        below, which is what actually controls that.

        When a curve is computed (explicit or automatic), `[lineage_
        threshold] + <the distinct entries of the curve>` is swept in a
        single call to `cv.lineage_groups_at_thresholds`, so the grouping
        this function already needs at `lineage_threshold` (for every
        top-level `score_lineage`/`gap`/`n_lineages`/`confounding` field)
        and every point on the curve all come from one sketching pass and
        one dendrogram, not one per threshold -- see that function's own
        docstring for why that matters. A curve entry equal to
        `lineage_threshold` reuses that grouping's already-computed
        `LeakageCurvePoint` rather than recomputing it a second time with a
        fresh `cross_val_score` call. `DegenerateLineagesWarning` is
        evaluated per distinct threshold on the curve, not only at
        `lineage_threshold` -- a curve is exactly the tool for showing
        *where* a grouping degenerates into (almost) one lineage per
        sample, so that warning firing partway along the curve is itself
        part of the result, not noise to suppress. Silently ignored
        (matching how `lineage_threshold` itself is already documented as
        ignored) whenever `groups=` is supplied directly -- there is no
        dendrogram to cut at other thresholds when the lineage labels did
        not come from one.
    auto_leakage_curve : bool, default True
        Whether `lineage_threshold_curve=None` means "sweep a data-driven
        default curve" (`True`, the default) or "no curve" (`False`, this
        function's original behaviour, before `leakage_curve` existed).

        This defaults to `True` on purpose, not as a convenience: a single
        `gap` at one arbitrary `lineage_threshold` is not a robust finding
        by itself -- see this module's own motivating case (a real
        bacterial AMR cohort read `gap=-0.018` at this function's default
        threshold, where 149/150 genomes were each their own lineage, and
        `gap=+0.148` at a coarser, data-driven threshold on the *same*
        cohort) -- so reporting one number as if it settled the question,
        merely because a caller did not think to ask for a curve, silently
        exports exactly the fragility this module exists to catch. When
        `True` and `lineage_threshold_curve` is `None` and `groups=` was
        not supplied directly, the default curve comes from `cv.
        default_threshold_curve(paths, n_points=default_curve_points, k=k,
        sketch_size=sketch_size)` -- thresholds drawn from THIS cohort's
        own dendrogram (see that function's own docstring for why a fixed,
        cohort-independent threshold list would not do), not arbitrary
        constants.

        This makes every default call to `audit()` cost roughly `1 +
        default_curve_points` lineage-blocked `cross_val_score` runs
        instead of 1 -- real, and dominant, wall-clock on a realistic
        cohort and estimator (minutes per run, not seconds; the extra
        dendrogram sketch this also costs is comparatively negligible).
        Pass `auto_leakage_curve=False` to keep the original single-
        threshold, single-`cross_val_score`-pair behaviour when that cost
        is not worth paying for a given call -- e.g. a quick sanity check
        during development, or a caller that already knows, from its own
        prior curve, that this cohort's gap is stable across granularity.
    default_curve_points : int, default 5
        `n_points` forwarded to `cv.default_threshold_curve()` when the
        curve is automatic (`auto_leakage_curve=True` and `lineage_
        threshold_curve=None`). Ignored otherwise -- including when
        `lineage_threshold_curve` is given explicitly, which controls its
        own point count directly via how many thresholds it contains.
    random_state : int or None, default 0
        Seeds the random-CV splitter's shuffle (and every covariate's
        `StratifiedKFold`/`KFold`, when applicable through
        `_random_cv_splitter`). `cv.LineageKFold` itself has no randomness
        to seed -- `GroupKFold` assigns folds deterministically from the
        group labels.

    Returns
    -------
    AuditReport
        Whose `confounding` field is the one number in it that is computed
        from `phenotype` and the lineage labels alone -- no fit, no fold,
        no dependence on `estimator` beyond asking it whether the phenotype
        is categorical or continuous (see `Confounding`), and whose
        `leakage_curve` field is `None` only when `groups=` was supplied
        directly, or when `auto_leakage_curve=False` and `lineage_
        threshold_curve` was not given -- by default (`groups=` omitted,
        `auto_leakage_curve=True`) it is always populated (see
        `LeakageCurvePoint` and the `lineage_threshold_curve`/`auto_
        leakage_curve` parameters above).
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    if not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")

    # `paths` doubles as two different things below: the `X` given to every
    # `cross_val_score` call, and the source `cv.lineage_groups`/`cv.
    # lineage_groups_at_thresholds`/`cv.default_threshold_curve` derive
    # lineage groups from. Those used to always be the same object (a list
    # of real FASTQ paths); they no longer are once `paths` may be a
    # `fastdna.CohortCounts` -- see this function's own docstring's `paths`
    # entry. `lineage_source` keeps the object those three `cv` calls need
    # (the `CohortCounts` itself, so they can sketch from its already-
    # counted k-mers with no FASTQ reread); `paths` is reassigned to
    # `list(counts.sample_ids)`, exactly what `KmerVectorizer(counts=...)`'s
    # own `subset()` contract expects as `X` (see `sklearn.py`'s
    # `_count_cohort()`), so every `cross_val_score` call below needs no
    # change at all -- it already just uses `paths`.
    if isinstance(paths, CohortCounts):
        counts = paths
        n_samples = len(counts)
        if n_samples < 2:
            raise ValueError(f"audit() needs at least 2 samples, got {n_samples}")
        paths = list(counts.sample_ids)
        lineage_source = counts
    else:
        paths = list(paths)
        n_samples = len(paths)
        if n_samples < 2:
            raise ValueError(f"audit() needs at least 2 samples, got {n_samples}")
        lineage_source = paths

    phenotype = np.asarray(phenotype)
    if phenotype.shape[0] != n_samples:
        raise ValueError(
            f"phenotype has {phenotype.shape[0]} entries but paths has {n_samples} samples. "
            "Pass one phenotype value per path, in the same order."
        )

    groups_supplied_directly = groups is not None
    # Populated only when a curve is actually being swept -- groups was not
    # supplied directly, and either lineage_threshold_curve was given
    # explicitly or auto_leakage_curve derived one (the default; see its
    # own docstring). Used below to build AuditReport.leakage_curve from
    # the SAME dendrogram the primary `groups` grouping already came from.
    curve_thresholds = None
    threshold_to_groups = None

    if groups_supplied_directly:
        groups = np.asarray(groups)
        if groups.shape[0] != n_samples:
            raise ValueError(
                f"groups has {groups.shape[0]} entries but paths has {n_samples} samples. Pass "
                "one lineage label per path, in the same order as paths, or omit groups= to "
                "derive them automatically with cv.lineage_groups()."
            )
        resolved_lineage_threshold = float("nan")
        # lineage_threshold_curve/auto_leakage_curve are both no-ops here:
        # there is no dendrogram to sweep when the caller handed in labels
        # directly (see this function's own docstring).
    else:
        if lineage_threshold_curve is not None:
            resolved_curve_thresholds = [float(t) for t in lineage_threshold_curve]
        elif auto_leakage_curve:
            # lineage_threshold_curve=None no longer means "no curve" --
            # see auto_leakage_curve's own docstring for why a data-driven
            # default curve is what this function reports unless a caller
            # opts out. This is a SEPARATE sketching/dendrogram pass from
            # the one below (see default_threshold_curve's own docstring
            # for why that tradeoff was made); an empty result (every
            # merge height was exactly 0) falls through to the plain,
            # curve-free path rather than erroring, since there is
            # genuinely no positive threshold this cohort's own dendrogram
            # can offer.
            resolved_curve_thresholds = default_threshold_curve(
                lineage_source, n_points=default_curve_points, k=k, sketch_size=sketch_size
            ) or None
        else:
            resolved_curve_thresholds = None

        # `lineage_threshold=None` (the default) means "read it off this
        # cohort's own dendrogram" rather than a fixed constant. A single
        # constant cannot work here: Mash distance has no universal scale,
        # so what separates lineages depends entirely on how divergent this
        # particular cohort is -- the same argument `default_threshold_curve`
        # already makes for the curve, applied to the primary cut too.
        #
        # A fixed constant cannot work, and the evidence for that is that
        # the SAME constant fails in OPPOSITE directions on data of
        # different completeness. Measured on real BV-BRC E. coli cohorts:
        # `0.01` put 198 of 200 genomes in their own lineage on one cohort
        # (far too fine) and merged 80 genomes into 15 groups on another
        # (far too coarse, against 38 real sequence types). The derived
        # value on the second lands at 0.0041 -> 40 groups, recovering the
        # MLST partition at ARI 0.931 where the constant scores 0.512.
        # See `scripts/validation/lineage_leakage_experiment.py`.
        derived_threshold = None
        if lineage_threshold is None:
            curve_for_median = resolved_curve_thresholds
            if not curve_for_median:
                # No curve was computed (auto_leakage_curve=False and none
                # supplied), so pay for one dendrogram pass to derive the
                # threshold. Still cheaper than being silently wrong.
                curve_for_median = default_threshold_curve(
                    lineage_source, n_points=default_curve_points, k=k, sketch_size=sketch_size
                )
            derived_threshold = (
                float(np.median(np.asarray(curve_for_median, dtype=float)))
                if curve_for_median
                else _FALLBACK_LINEAGE_THRESHOLD
            )
        effective_lineage_threshold = (
            derived_threshold if lineage_threshold is None else float(lineage_threshold)
        )

        if resolved_curve_thresholds is not None:
            curve_thresholds = resolved_curve_thresholds
            primary_threshold = float(effective_lineage_threshold)
            # One dendrogram, cut at the primary threshold plus every
            # DISTINCT threshold the curve asks for -- duplicates
            # (including a curve entry equal to lineage_threshold itself)
            # are cut once and shared, not recomputed, per cv.
            # lineage_groups_at_thresholds's cost model.
            batch_thresholds = [primary_threshold]
            seen_thresholds = {primary_threshold}
            for t in curve_thresholds:
                if t not in seen_thresholds:
                    batch_thresholds.append(t)
                    seen_thresholds.add(t)
            batch_groups = lineage_groups_at_thresholds(
                lineage_source, batch_thresholds, k=k, sketch_size=sketch_size
            )
            threshold_to_groups = dict(zip(batch_thresholds, batch_groups))
            groups = threshold_to_groups[primary_threshold]
            resolved_lineage_threshold = primary_threshold
        else:
            groups = lineage_groups(
                lineage_source, k=k, sketch_size=sketch_size,
                distance_threshold=effective_lineage_threshold,
            )
            resolved_lineage_threshold = float(effective_lineage_threshold)

    def _warn_if_degenerate(n_lineages_here, grouping_label):
        """Warns, and returns a reason string when the grouping is degenerate
        (`None` otherwise) so the caller can withhold `gap` rather than
        report a difference that measures nothing."""
        if n_samples > 0 and n_lineages_here / n_samples >= _DEGENERATE_LINEAGE_FRACTION:
            warnings.warn(
                f"{n_lineages_here} of {n_samples} samples are each their own lineage (or "
                f"nearly so) under {grouping_label} -- LineageKFold has little to block on, and "
                "the gap computed from it may read near zero even when real, coarser-scale "
                "leakage exists. If groups= was not supplied directly, try a larger threshold; "
                "see DegenerateLineagesWarning's own docstring.",
                DegenerateLineagesWarning,
                stacklevel=3,
            )
            return (
                f"{n_lineages_here} of {n_samples} samples are each their own lineage under "
                f"{grouping_label}, so LineageKFold had almost nothing to block on and the "
                "difference between the two splitters does not measure leakage. The score_* "
                "fields are still the scores actually measured."
            )
        return None

    n_lineages = int(np.unique(groups).size)
    gap_undefined_reason = _warn_if_degenerate(
        n_lineages,
        "the current grouping" if groups_supplied_directly else f"the grouping at threshold {resolved_lineage_threshold:g}",
    )
    validated_covariates = _validate_covariates(covariates, n_samples)

    # Computed before any fitting: it needs no model, and a caller reading a
    # traceback from a failed cross_val_score below should be able to see
    # from the code that this number was never going to depend on it.
    confounding = _confounding(estimator, groups, phenotype)

    resolved_scoring = scoring if scoring is not None else _default_scoring(estimator, phenotype)

    random_cv = _random_cv_splitter(estimator, phenotype, n_splits, random_state)
    lineage_cv = LineageKFold(n_splits=n_splits, groups=groups)

    scores_random = np.asarray(
        cross_val_score(clone(estimator), paths, phenotype, cv=random_cv, scoring=resolved_scoring),
        dtype=np.float64,
    )
    scores_lineage = np.asarray(
        cross_val_score(clone(estimator), paths, phenotype, cv=lineage_cv, scoring=resolved_scoring),
        dtype=np.float64,
    )

    per_fold_cv_kind, per_fold_index, per_fold_score = [], [], []

    def _record(kind, scores):
        for fold, score in enumerate(scores):
            per_fold_cv_kind.append(kind)
            per_fold_index.append(fold)
            per_fold_score.append(float(score))

    _record("random", scores_random)
    _record("lineage", scores_lineage)

    mean_random = _nan_aware_mean(scores_random)
    mean_lineage = _nan_aware_mean(scores_lineage)

    covariate_reports = []
    for name, values in validated_covariates.items():
        covariate_cv = LineageKFold(n_splits=n_splits, groups=values)
        scores_covariate = np.asarray(
            cross_val_score(clone(estimator), paths, phenotype, cv=covariate_cv, scoring=resolved_scoring),
            dtype=np.float64,
        )
        _record(f"covariate:{name}", scores_covariate)
        covariate_score = _nan_aware_mean(scores_covariate)
        covariate_reports.append(
            CovariateAudit(
                name=name,
                n_groups=int(np.unique(values).size),
                score=covariate_score,
                score_std=_nan_aware_std(scores_covariate),
                gap_vs_random=float(mean_random - covariate_score),
            )
        )

    per_fold = pa.table(
        {"cv_kind": per_fold_cv_kind, "fold": per_fold_index, "score": per_fold_score}
    )

    leakage_curve = None
    if threshold_to_groups is not None:
        # One LeakageCurvePoint per DISTINCT threshold in the sweep, keyed
        # by threshold value; the primary threshold's point reuses the
        # scores/confounding already computed above instead of refitting.
        points_by_threshold = {}
        for threshold in batch_thresholds:
            threshold_groups = threshold_to_groups[threshold]
            if threshold == resolved_lineage_threshold:
                points_by_threshold[threshold] = LeakageCurvePoint(
                    threshold=threshold,
                    n_lineages=n_lineages,
                    score_lineage=mean_lineage,
                    score_lineage_std=_nan_aware_std(scores_lineage),
                    gap=float(mean_random - mean_lineage),
                    confounding=confounding,
                )
                continue

            threshold_n_lineages = int(np.unique(threshold_groups).size)
            _warn_if_degenerate(threshold_n_lineages, f"the grouping at threshold {threshold:g}")

            # A curve deliberately sweeps a wide threshold range, and a
            # coarse-enough threshold can collapse the lineage count below
            # n_splits (LineageKFold's own requirement) even when the
            # primary threshold's grouping is fine. That must fail THIS
            # point only, not the whole curve -- and not the rest of
            # audit()'s report, which was already computed above -- so
            # this point is reported with NaN scores/gap rather than
            # raising, and n_lineages on the point itself is the reason
            # why. confounding is model-free and never depends on
            # n_splits, so it is still computed even when the CV
            # comparison could not be.
            try:
                threshold_cv = LineageKFold(n_splits=n_splits, groups=threshold_groups)
                threshold_scores = np.asarray(
                    cross_val_score(
                        clone(estimator), paths, phenotype, cv=threshold_cv, scoring=resolved_scoring
                    ),
                    dtype=np.float64,
                )
                threshold_score = _nan_aware_mean(threshold_scores)
                threshold_score_std = _nan_aware_std(threshold_scores)
                threshold_gap = float(mean_random - threshold_score)
            except _core.InvalidConfigError:
                threshold_score = float("nan")
                threshold_score_std = float("nan")
                threshold_gap = float("nan")

            points_by_threshold[threshold] = LeakageCurvePoint(
                threshold=threshold,
                n_lineages=threshold_n_lineages,
                score_lineage=threshold_score,
                score_lineage_std=threshold_score_std,
                gap=threshold_gap,
                confounding=_confounding(estimator, threshold_groups, phenotype),
            )

        # Re-expanded to curve_thresholds' own order/length -- a curve entry
        # repeating a threshold (including the primary one) gets the SAME
        # LeakageCurvePoint object, not a fresh recomputation.
        leakage_curve = tuple(points_by_threshold[t] for t in curve_thresholds)

    return AuditReport(
        n_samples=n_samples,
        n_splits=int(n_splits),
        scoring=resolved_scoring,
        score_random=mean_random,
        score_random_std=_nan_aware_std(scores_random),
        score_lineage=mean_lineage,
        score_lineage_std=_nan_aware_std(scores_lineage),
        # Withheld, not reported small, when the grouping was degenerate:
        # `LineageKFold` had nothing to block on, so this difference is not a
        # measurement of leakage and a caller who reads it as one draws the
        # opposite of the right conclusion. Same convention as
        # `Confounding.value`/`undefined_reason`.
        gap=float("nan") if gap_undefined_reason else float(mean_random - mean_lineage),
        n_lineages=n_lineages,
        lineage_threshold=resolved_lineage_threshold,
        confounding=confounding,
        per_fold=per_fold,
        covariates=tuple(covariate_reports),
        leakage_curve=leakage_curve,
        gap_undefined_reason=gap_undefined_reason,
    )
