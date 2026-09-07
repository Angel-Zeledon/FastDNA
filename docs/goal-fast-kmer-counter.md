# Goal: a fast k-mer counter, and the things that are actually k-mer counting

Date: 2026-09-05. Status: **active goal.** Supersedes and replaces both
`goal-most-complete-genomics-ml-library.md` (2026-08-27) and
`goal-audit-leakage-not-completeness.md` (2026-09-05, earlier the same day);
both are deleted rather than kept with banners, because the code they
planned for is gone. `philosophy-narrow-not-broad.md` (2026-08-25) is the
operative document again, and its evidence is why.

## The decision

FastDNA is **a fast, exact k-mer counter with a small set of capabilities
that are counting**: sketching and distance, cardinality and spectrum
estimation, k-mer tables with set operations, read filtering against a
reference table, exact similarity between tables, cohort counting, QC,
and preview. That is the CLI surface (`count`, `sketch`, `dist`, `card`,
`peek`, `query`, `union`, `intersect`, `diff`, `similarity`, `filter`,
`matrix`, `profile`, `spectrum`) and the Python API that mirrors it.

Two things were added under this goal rather than inherited from before it,
and both had to earn it the same way: `similarity` (exact Jaccard,
containment and abundance-weighted Bray-Curtis between counted tables --
checkable against `kmc_tools`' own set operations, and it agrees exactly)
and the `k > 32` engine (`--engine`, up to k=64 -- checkable against the
narrow engine, which they agree with k-mer for k-mer wherever both are
defined).

The machine-learning layer is **removed**, not frozen: the scikit-learn
vectorizer, lineage-aware cross-validation, the leakage audit, feature
explanation, association screening, MIC regression, active learning,
calibration, anomaly detection, rule models, multi-omics joins, dataset
loaders, metagenomic classification, taxonomy, chimera scanning,
translation, assembly QC, genome profiling, plotting, reporting and workflow
helpers. 31 Python modules, 4 Rust modules, and everything that existed to
serve them.

## Why

Three reasons, in the order they carry weight.

**1. Every serious defect this project found was in that layer.** Nine of
them between 2026-08-31 and 2026-09-05, none reachable by the ~1,750 tests
that existed, each found only by checking a shipped module against a truth
computed outside it. They shared one signature — *the wrong output was well
formed*: a FASTA truncated to 26% of a genome still starts with `>`; a
feature matrix of literal ones still has the requested shape; an audit run
on it still returns `score_random = 0.5000, score_lineage = 0.5000,
gap = 0.0000`, every field in range, reading as "no leakage". Two of the
nine retracted claims the repository had already written down.

The counting engine, over the same period, was checked against KMC3 on real
reads and came back **exactly equal** — 19,062,700 distinct k-mers,
163,051,083 total, to the digit. HyperLogLog landed within −0.26% of the
exact count, ntCard within ±2%, MinHash distances at r = 0.997 against
Mash 2.3 with no bias and error shrinking as 1/sqrt(sketch size).

One of those two halves is validated. The other kept producing well-formed
wrong answers faster than they could be caught.

**2. Maintaining the ML layer cost more than it produced.** 525 of 997 test
functions, 26 of 34 Python modules touching scikit-learn or scipy, a test
extra pulling in shap (and therefore numba and llvmlite), and a validation
suite whose long pole was a four-hour cohort study. What it produced in
return was one unfinished empirical result.

**3. The 2026-08-25 evidence was never refuted.**
`philosophy-narrow-not-broad.md` argued from outcomes in this exact
ecosystem: SAMtools 57,470 citations against Biopython's 5,675 for ~1% of
the scope; AnnData at 1.68M monthly downloads against scikit-bio's 101,280;
`exon` and `biobear`, the two broadest Rust-Arrow-for-genomics projects,
dead, while `oxbow` and `polars-bio`, the two narrowest, live. The
2026-08-27 expansion disputed none of that — it argued the expansion was
*within* genomics ML and therefore on a different axis. Ten days later the
narrow document's prediction is the one that matched.

## What was kept, and the rule for what gets added

Kept: the counting engine (`kmer`, `counter`, `fastq`, `pipeline`, the
memory estimator and both strategies, minimizers and super-k-mers), the
things built directly on it (`sketch`, `hll`, `ntcard`, `ktab`, `setops`,
`read_filter`, `read_profile`, `qc`, `preview`, `export`, `cohort/`), the
CLI, the Python bindings, and WASM.

The rule for anything new: **it has to be k-mer counting, or an operation on
counted k-mers, and it has to be checkable against something outside this
repository.** If the only way to know the answer is right is to read this
project's own code, it does not belong here.

Unchanged from `philosophy-narrow-not-broad.md`, and never the contested
part: no sequence alignment, no pangenome graphs, no genomic interval
algebra. `minimap2`, `vg`/`pggb` and `bedtools` own those, each would reuse
approximately none of the k-mer engine, and a published API surface is close
to permanent for a project maintained by one person.

## What this cost, stated plainly

The removal is a breaking change and `CHANGELOG.md` records it as one. Any
user of `fastdna.sklearn`, `fastdna.audit`, `fastdna.cv` or any other
removed module is broken by it, with no deprecation period. Pre-1.0 permits
that; honesty requires saying it rather than describing the change as
"focus".

Everything removed is in the git history at commit `60b5f82` and can be
recovered from there. The leakage survey's partial result — 9 of 30 cohorts,
mean gap +0.079 — was discarded with it, unfinished and unpublished.
