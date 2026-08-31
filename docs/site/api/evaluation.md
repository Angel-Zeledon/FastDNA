# Evaluation and validation

The modules that answer *"is this score real?"* rather than *"how high is
it?"*.

Genomic cohorts are clonal, so a random cross-validation split puts
near-copies of training samples in the held-out fold and the model can score
well by recognizing a lineage instead of the phenotype. `fastdna.cv` supplies
the splits that do not do that; `fastdna.audit` measures how much of a score
survives them; `fastdna.explain` asks the same question of individual
features; `fastdna.evaluation` and `fastdna.calibration` deal with what a
headline number hides under class imbalance and what a raw score does or does
not mean as a probability.

Scores usually go **down** here. That is the point.

## Leakage-safe cross-validation

::: fastdna.cv

<!-- Carried over by hand because it is currently dropped in rendering:
     `lineage_groups`'s docstring continues in prose *inside* its
     `Parameters` block instead of closing the section first, so griffe's
     NumPy parser reads the paragraph as a run of nameless parameters and
     emits nothing for it. Delete this note (and the `warnings: false` in
     mkdocs.yml) once python/fastdna/cv.py moves that paragraph out of the
     section. -->
!!! note "Why `lineage_groups` uses **single** linkage"

    Single linkage merges two clusters when their *closest* members are
    within the threshold, so a chain of near-identical isolates stays one
    lineage even when its two extremes are further apart than the threshold.
    That is the conservative direction here: the failure that matters is
    splitting a clone across folds, and complete or average linkage would do
    exactly that whenever a lineage is internally diverse.

    The cost is chaining — with a threshold set too loosely, distinct
    lineages linked by one intermediate sample merge into a single group.
    That direction is safe (fewer, larger groups mean a more pessimistic,
    never a leakier, evaluation), and it is why `LineageKFold`'s error
    message points at *raising* the threshold rather than lowering it.

## How much of the score is real

::: fastdna.audit

## Which features can you believe

::: fastdna.explain

::: fastdna.interpret

## Reporting under class imbalance

::: fastdna.evaluation

## Turning scores into probabilities

::: fastdna.calibration

## Deciding what a human should look at

::: fastdna.active_learning

## Cohort QC outliers

::: fastdna.anomaly
