# Cohorts and features

Turning a directory of FASTQ files into a feature matrix a model can be
fitted on — and doing it without counting the same file eleven times.

The ordering here is the order the pieces are used: count the cohort once
(`CohortCounts`), project it onto a learned vocabulary
(`fastdna.sklearn.KmerVectorizer`), optionally join it to other omics or
clinical layers (`fastdna.multiomics`), collapse columns that are
indistinguishable in this cohort (`fastdna.equivalence`), and hand the
evidence to established association tooling (`fastdna.gwas`).

## Counting a cohort once

<!-- The module docstring on its own, then its two public names under the
     `fastdna.*` paths they are actually re-exported at (both appear in
     `fastdna.__all__`), rather than under `fastdna.cohort_counts.*`. -->
::: fastdna.cohort_counts
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

::: fastdna.count_cohort

::: fastdna.CohortCounts

## scikit-learn integration

::: fastdna.sklearn

## Joining other data by sample ID

::: fastdna.multiomics

## Collapsing indistinguishable features

::: fastdna.equivalence

## Association: evidence out, statistics elsewhere

::: fastdna.gwas
