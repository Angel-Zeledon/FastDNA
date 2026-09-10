# Historical documents

**Nothing in this directory describes the code as it is today.** These are
the working documents from states the project has passed through — plans,
specs, checkpoints, an audit, a generated architecture page — kept because
they record why decisions were made, not because they are still true.

Most of them describe a machine-learning layer (a scikit-learn vectorizer,
lineage-aware cross-validation, a leakage audit, association screening,
metagenomic classification, taxonomy, and more) that was **removed on
2026-09-05**, before any version published it. If a document here mentions
`fastdna.sklearn`, `fastdna.audit`, `fastdna.cv`, `fastdna.gwas` or
anything similar, it is referring to code that no longer exists. The
reasoning behind the removal, with the numbers behind it, is in
[`../goal-fast-kmer-counter.md`](../goal-fast-kmer-counter.md).

For what the project *is*, read these instead:

| | |
|---|---|
| [`../../README.md`](../../README.md) | what it does and how to use it |
| [`../goal-fast-kmer-counter.md`](../goal-fast-kmer-counter.md) | the current mission, and what is deliberately out of scope |
| [`../BENCHMARKS.md`](../BENCHMARKS.md) | every performance number, with its method and its caveats |
| [`../design-minimizer-counting.md`](../design-minimizer-counting.md) | how the default counting strategy works |
| [`../feature-gap-analysis.md`](../feature-gap-analysis.md) | what exists, what does not, and why |

## What is here

| | |
|---|---|
| `CHECKPOINT-*.md` | session-end states, five of them, 2026-08-25 to 2026-09-03 |
| `superpowers/` | the plans and specs that drove the work, with their task lists |
| `audit-2026-08-24.md` | a correctness review of the tree as it stood that day |
| `PERFORMANCE_PLAN.md` | the optimisation plan that produced the numbers now in `BENCHMARKS.md` |
| `ARCHITECTURE.html` | a generated architecture page, superseded by the module map in `CLAUDE.md` |
