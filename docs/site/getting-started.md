# Getting started

Everything below assumes FastDNA is importable — see
[Installation](installation.md) if `import fastdna` fails.

## Count one file

```python
import fastdna

result = fastdna.count("sample.fastq.gz", k=31)
print(result)
# KmerCounts(k=31, distinct=1423950, total=23999892)
```

[`count()`](api/counting.md#fastdna.count) streams the FASTQ (transparently
gzip-decoded, multi-member gzip included, as real SRA/ENA downloads require),
trims read ends below `min_quality` (Phred 20 by default), extracts every
k-mer as a **canonical** 2-bit-packed `u64` — so a read and its
reverse-complement strand contribute to the same count — and returns a
[`KmerCounts`](api/counting.md#fastdna.KmerCounts).

The counts are an Arrow table, handed over with no copy:

```python
table = result.table               # a pyarrow.Table
print(table.column_names)
# ['kmer_u64', 'frequency']
```

!!! note "`kmer_sequence` is opt-in"

    The decoded `kmer_sequence` column is **not** built by default. It is
    entirely derivable from `kmer_u64` plus `k`, and on the 2.14 GB benchmark
    file it weighs 1,667 MB against 430 MB for `kmer_u64` and 215 MB for
    `frequency` — 2.6x the other two columns combined — while decoding it
    accounts for most of the ~17% of wall time the table-building step costs.
    Every ML-facing module in this package works in `u64` space and never
    reads it.

    Ask for it with `count(..., with_sequence=True)`, or add it afterwards
    with `result.with_sequence()`, which decodes from the integers already in
    memory rather than rereading the FASTQ.

Quality-control figures for the run come back alongside the counts:

```python
print(result.qc)
# {'total_reads': ..., 'total_bases': ..., 'q20_bases': ..., 'q30_bases': ...,
#  'gc_bases': ..., 'gc_content_pct': ..., 'q20_pct': ..., 'q30_pct': ...}

result.k               # 31
result.distinct_kmers  # distinct canonical k-mers kept
result.total_kmers     # total occurrences counted
len(result)            # == result.distinct_kmers
```

## Pick `k` and `min_count` from the data, not from habit

```python
preview = fastdna.peek("sample.fastq.gz")      # samples the first 10,000 reads
print(preview)
# Preview(n_reads_sampled=10000, read_length=(150, 150, 150), gc_content=0.412, suggested_k=31)

preview.read_length            # (min, median, max) among the sampled reads
preview.gc_content
preview.sample_distinct_kmers  # exact, but only over the sampled prefix
preview.suggest_k()            # largest odd k <= median_read_length / 3, clamped to 1..=32
```

[`peek()`](api/counting.md#fastdna.peek) reads a prefix, not the file, so it
answers in milliseconds. `k=31` is everyone's default and it is wrong for
short reads: with 50 bp reads it leaves 20 k-mers per read and amplifies
every sequencing error.

The same logic applies to `min_count`. There is no universal 5 — the right
cutoff is the valley between the error peak (frequency 1–2) and this
sample's own coverage peak, and it moves with coverage:

```python
result.spectrum()            # {depth: number of distinct k-mers at that depth}
result.suggest_min_count()   # the valley, detected from this sample's spectrum
```

## Chain views instead of recounting

`filter`, `sort_by`, `top` and `with_sequence` all return a new
`KmerCounts` over the *same* counted data. None of them rereads the FASTQ:

```python
top20 = (
    result
    .filter(min_count=5)     # composes with any earlier filter
    .top(20)                 # 20 most frequent, still chainable
    .with_sequence()         # decode those 20 into ACGT strings
)

top20.to_pandas()            # requires pandas
top20.to_polars()            # requires polars
```

`filter()` is a view over already-counted data: it cannot recover k-mers that
`count()`'s own `min_count`/`max_count` dropped during counting. That is the
point — explore one `count()` result at several thresholds without paying for
the run again.

The table is a plain `pyarrow.Table`, so DuckDB, Polars and pandas consume it
directly with no export step.

## Compare samples without counting them in full

A MinHash sketch is a fixed-size fingerprint of a sample's canonical k-mer
set. It answers "how similar are these two samples", not "what exactly is in
each" — and it is what makes cohort-scale work affordable:

```python
a = fastdna.sketch("sample_a.fastq.gz", k=21, sketch_size=1000)
b = fastdna.sketch("sample_b.fastq.gz", k=21, sketch_size=1000)

a.jaccard(b)         # symmetric similarity
a.containment(b)     # asymmetric: how much of a is inside b
a.mash_distance(b)   # Poisson-model evolutionary distance, 0 = identical

a.save("a.sketch")
a2 = fastdna.load_sketch("a.sketch")
```

`fastdna.compare(a_path, b_path)` is sugar for the two-sketch case. For a
whole cohort, use
[`compare_all()`](api/sketching.md#fastdna.compare_all) instead of looping —
it builds each sketch once and compares every pair, which is `O(N)` FASTQ
reads plus `O(N²)` cheap sketch comparisons, rather than the `O(N²)` FASTQ
reads a naive pairwise loop costs:

```python
pairs = fastdna.compare_all(paths, k=21, metric="mash_distance")
# pyarrow.Table in long format: sample_a, sample_b, mash_distance
# one row per unordered pair, n*(n-1)/2 rows
```

There is also a **FracMinHash** ("scaled") variant —
`fastdna.frac_sketch` / `fastdna.FracSketch` / `fastdna.load_frac_sketch` —
whose sketch size grows with the genome instead of being fixed, which is what
makes containment meaningful between samples of very different size.

For a file too large to count exactly in the memory available,
`fastdna.estimate_cardinality()` gives the number of *distinct* canonical
k-mers via HyperLogLog in a fixed 16 KB (at the default `precision=14`, a
~0.8% standard error).

## Cohort work: count once, reuse everywhere

Counting is by far the most expensive thing this package does, and
scikit-learn's `Pipeline`/`cross_val_score`/`GridSearchCV` call `fit()` and
`transform()` at *every* split — so a naive 5-fold cross-validation reads the
cohort's FASTQ files five times over. `count_cohort()` counts each sample
exactly once; `KmerVectorizer(counts=...)` then slices that stack per fold
instead of recounting:

```python
import fastdna
from fastdna.sklearn import KmerVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

counts = fastdna.count_cohort("samples/", k=31, min_count=5)
# CohortCounts: every sample's table, stacked end to end, frozen

pipe = Pipeline([
    ("kmers", KmerVectorizer(k=31, min_count=5, top_features=10_000, counts=counts)),
    ("clf",   LogisticRegression(max_iter=1000)),
])
pipe.fit(train_paths, train_labels)
```

Counting ahead of time does not reopen the leakage door: a k-mer's count in
one sample looks at nothing but that sample. The decision that *could* leak —
which k-mers become features — still happens only inside `fit()`, on the
training fold's rows alone.

## Cross-validate against population structure, not around it

Bacterial and viral cohorts are clonal. A random CV split scatters members of
one clone across the train/test boundary, so the "held-out" fold contains
near-copies of the training data and the model can score highly by
recognizing the lineage rather than the phenotype. FastDNA already has the
all-pairs Mash distances needed to stand in for a phylogeny, in seconds
rather than the hours a real tree would take:

```python
from fastdna.cv import lineage_groups, LineageKFold
from sklearn.model_selection import cross_val_score

cv = LineageKFold(n_splits=5, paths=paths)
scores = cross_val_score(pipe, paths, labels, cv=cv)

# Or derive the labels yourself and reuse them across several evaluations,
# so the cohort is sketched once rather than once per splitter:
groups = lineage_groups(paths, distance_threshold=0.01)   # one integer label per sample
cv = LineageKFold(n_splits=5, groups=groups)
```

`LineageKFold` requires exactly one of `paths=` or `groups=` — it will not
silently fall back to random folds. It is `GroupKFold` underneath; the
contribution is that the groups come from the genomes themselves, so the
caller does not need a phylogeny nobody built.

Expect scores to go **down** relative to random CV. That drop is the finding,
not a regression: the honest estimate is the one you can defend.
[`fastdna.audit.audit()`](api/evaluation.md#fastdna.audit) measures exactly
that gap — it runs both a random and a lineage-blocked scheme and reports how
much of the score survives — and
[`fastdna.cv`](api/evaluation.md#fastdna.cv) documents what this technique
cannot fix (a cohort where the phenotype is perfectly confounded with lineage
has no honest split, and `permutation_importance_pvalues` will report p = 1.0
rather than pretend otherwise).

## The whole association study in one call

[`AssociationWorkflow`](api/models.md#fastdna.workflow) composes the six
modules a k-mer association study normally needs, in the right order, with
the sample-id bookkeeping done once and correctly:

```python
from fastdna.workflow import AssociationWorkflow

workflow = AssociationWorkflow(fastq_paths, phenotype, n_splits=5)
result = workflow.run()

print(result.classifier.explain())              # rules as literal DNA sequences
print(result.precision_recall.average_precision)
workflow.plot_significance()                    # needs matplotlib
workflow.plot_population_structure()            # needs matplotlib + scipy
```

It is a convenience layer, not a new capability: it calls the same public
functions a hand-written script would, keeps every intermediate artifact, and
suppresses none of the underlying warnings — the screening step still raises
`ScreeningOnlyWarning`, the rule model's `predict_proba` is still a hard 0/1
decision, and the calibration report still raises
`UncalibratedScoresWarning` when it should. Every stage is skippable or
replaceable.

## The command-line interface

The `fastdna` binary is a separate Cargo target that does not ship with the
Python package (see [Installation](installation.md)). It counts one
FASTQ(.gz) file and writes the frequency table plus a QC report:

```bash
fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

The output format follows the extension: `.parquet` writes Arrow/Snappy
Parquet, anything else writes CSV. Every exporter writes the same three
columns: `kmer_u64` (`uint64`), `kmer_sequence` (string), `frequency`
(`uint32`).

A whole directory of paired-end samples is one invocation — R1/R2 mates are
discovered and counted *together*, one run per sample:

```bash
fastdna --paired-dir samples/ --paired-output counts/ -k 31
```

Unlike ad hoc cohort listing, an unpaired file is a hard error in this mode
rather than a warning, because it exists to run unattended: silently falling
back to single-end would produce a normal-looking output for a directory that
was actually half-paired.

At startup the CLI prints the counting strategy it will use, the estimated
peak memory and the budget it compared against:

```
Strategy:       in-memory (estimated peak 2.28GB, budget 3.35GB)
```

The choice is made *before* the run by a calibrated estimator; if the
predicted peak exceeds the budget (`--max-ram`, or half of available RAM by
default) the disk-partitioned strategy runs instead of the run failing.
`--strategy memory|disk` overrides the chooser outright. Both strategies
produce identical, exact counts.

The complete flag table, the environment-variable escape hatches
(`FASTDNA_STRATEGY`, `FASTDNA_MAX_RAM_BYTES`) and their precedence rules are
documented in the
[repository README](https://github.com/Angel-Zeledon/FastDNA/blob/master/README.md#command-line-interface),
which is the single source of truth for the CLI surface.

## Where to go from here

The [API reference](api/counting.md) covers every public function, class and
submodule — including areas not touched above: metagenomic classification
against a k-mer/LCA database, Merqury-style reference-free assembly QC,
GenomeScope-style genome profiling, six-frame translation, MIC regression,
interpretable rule models, and HTML reporting.
