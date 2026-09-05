# Cohorts

Counting many samples once, and slicing that result per sample afterwards,
instead of re-reading a file every time some consumer needs its k-mers.

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
