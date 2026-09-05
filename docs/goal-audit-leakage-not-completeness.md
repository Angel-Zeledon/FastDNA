# Goal: the tool that audits leakage, and that is audited itself

Date: 2026-09-05. Status: **active goal, supersedes
`docs/goal-most-complete-genomics-ml-library.md`** (2026-08-27) and
reinstates the scope discipline of `docs/philosophy-narrow-not-broad.md`
(2026-08-25) — for a narrower and more specific target than that document
originally named.

## The decision

The target is not "the most complete library for genomics ML". It is:

> **the tool that measures whether a genomics ML result survives lineage
> blocking — and that is audited hard enough to be believed when it says
> so.**

Everything else in this repository is either an input to that claim, or
already-shipped surface that stays where it is and stops generating new
work.

## Why this reverses the 2026-08-27 decision

Not a change of taste. Five days of evidence, all of it in this repository.

### Nothing that went wrong was caused by a missing module

Nine defects have been found since 2026-08-31. **None was reachable by the
~1,750 tests.** Every one was found by validating an *already-shipped*
module against a truth computed outside it — an external implementation
(KMC3, Mash, Merqury, GenomeScope2) or a constructed case whose answer was
known in advance. The eight found before 2026-09-02 are listed in
`docs/CHECKPOINT-2026-09-02.md`; the ninth is in
`docs/CHECKPOINT-2026-09-03.md`.

They share one signature: **the wrong output was well-formed.** A truncated
FASTA still starts with `>`. A vocabulary of constant features still
produces a matrix of the requested shape. An audit run on that matrix still
returns every field in range — `score_random = 0.5000`,
`score_lineage = 0.5000`, `gap = 0.0000` — and reads as "this cohort has no
leakage".

Two of them retracted claims this repository had already written down:
Track D's leakage gap (an artefact of `load_amr` returning 26% of each
genome) and the leakage survey's first measured cohort (the constant-feature
vocabulary above).

Not one of the nine was a coverage gap. Every one was a **missing control on
something already shipped**. The 2026-08-27 posture — "nothing in
`feature-gap-analysis.md` or `ml-differentiation-roadmap.md` is 'someday'
anymore; every Q/S/A/B item is a real commitment, in priority order, not a
menu" — directs effort onto the axis that has produced zero failures, and
none onto the axis that has produced nine.

### The only asset here that does not depend on breadth

The leakage survey, in progress as this is written (2026-09-05, 7 of 30
cohorts measured): mean random-CV AUC 0.80, **mean gap +0.103 (SE 0.032)**,
6 of 7 cohorts losing more than 0.05 AUC under lineage blocking, and one
cohort *negative* (−0.052). That last number matters more than the mean: a
detector that always reported leakage would be worth nothing.

That is a finding about the field, not about this library. It is the only
thing here whose value does not depend on anyone agreeing that FastDNA is
complete.

### The 2026-08-25 evidence was never refuted, only overruled

`philosophy-narrow-not-broad.md` argued from outcomes: SAMtools 57,470
citations against Biopython's 5,675 for ~1% of the scope; AnnData's 1.68M
monthly downloads against scikit-bio's 101,280; `exon` and `biobear` (the
two broadest Rust-Arrow-for-genomics projects) dead while `oxbow` and
`polars-bio` (the two narrowest) live. The 2026-08-27 reversal disputed none
of it — it argued the expansion was *within* genomics ML and therefore on a
different axis.

That may be true and still not decide the question. The binding constraint
was never scope breadth in the abstract. It is that 34 Python modules plus
the Rust crate is more surface than this project can hold to the standard
its own mission demands, and the nine defects are what that gap looks like
from the inside.

## This is not a deletion program

Nothing shipped gets ripped out. The modules that exist keep their tests and
their documented behaviour; the compatibility contract in `CHANGELOG.md`
still holds. What changes is where new effort goes:

- A shipped module that cannot be checked against a truth outside itself
  gets **documented as unvalidated** (`docs/validation-real-data.md`'s
  existing coverage tables are the right place), not quietly improved.
- A new module has to earn its place by being needed for the claim above,
  not by closing a row in a gap table.

## In scope

1. **The audit triad and its trustworthiness.** `gap`, `confounding`,
   `explain` — and, for each, calibration against known injected truth,
   negative controls, and guards that refuse to return a number rather than
   return a meaningless one (`gap_undefined_reason`,
   `ConstantPredictionWarning`, `DegenerateLineagesWarning`,
   `EmptyVocabularyWarning` are the pattern to extend).
2. **The counting engine underneath it.** Already exactly equivalent to
   KMC3 on real data; it is not in question and needs no new features to
   serve this goal.
3. **The empirical study.** `scripts/validation/leakage_survey.py` across
   30 published species-antibiotic cohorts, *with its own controls* — a
   label-permutation run being the first, since the whole point is that a
   positive gap alone does not distinguish leakage from overfitting
   (`audit.py` says so at length, and its own p/n table is currently marked
   pending re-measurement).

## Out of scope

- Everything `philosophy-narrow-not-broad.md` already ruled out — sequence
  alignment, pangenome graphs, general interval algebra (`minimap2`/`vg`/
  `bedtools`'s territory, zero code reuse with the k-mer engine).
  Unchanged, and it was never the contested part.
- **New: adding modules.** `feature-gap-analysis.md`,
  `ml-genomics-roadmap.md` and `ml-differentiation-roadmap.md` revert from
  commitments to a menu. Their research and their status audits stay
  useful; their claim on the roadmap does not.
- **Demoted, not dropped: `pip`/`conda` publishability.** 2026-08-27 made it
  a completeness blocker. It is now downstream of the publishing decision,
  which is deliberately withheld until the claim above holds — in the
  user's words, *do not publish until we are the best at the thing we said
  we do: being the ones who audit.*

## What "done" means

A standing bar, re-checked periodically — not a ship date, and explicitly
not an empty "Still ahead" list:

- The survey's headline number exists **and has survived its own controls**
  (permuted labels give a mean gap indistinguishable from zero; the result
  does not move when the feature budget moves).
- Every number this repository publishes about itself is traceable to a
  measurement whose method is written down, and every retracted one is
  retracted *in the repository*, not only in a session log.
- Each leg of the triad has a calibration curve against injected truth and
  a documented blind spot. `gap` has both
  (`scripts/validation/leakage_calibration.py`, and its insensitivity to
  partial leakage is written down). `confounding` has the curve
  (Spearman rho = 1.000 against injected leakage) but its blind spot is
  recorded as "nothing yet found", which is not the same as none.
  `explain` has neither: it was checked against known provenance, which is
  a validation, not a calibration.

Related: [[philosophy-narrow-not-broad]] (reinstated conclusion, valid
technical corrections), [[goal-most-complete-genomics-ml-library]]
(superseded), `docs/validation-real-data.md`,
`docs/CHECKPOINT-2026-09-03.md`.
