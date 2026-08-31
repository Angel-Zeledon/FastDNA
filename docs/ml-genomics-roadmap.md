# FastDNA x Machine Learning — feature roadmap

> **Audit pass, 2026-08-27**: every "Status" line below was re-checked
> against the actual code on this date (not just against this doc's own
> "dispatched" claims), per `docs/goal-most-complete-genomics-ml-library.md`'s
> instruction to verify before executing anything on these lists. All nine
> dispatched wave-1/wave-2 modules exist and do substantially what they were
> scoped to do; one bundled sub-item (Snakemake/Nextflow workflow templates,
> part of feature 4) was found genuinely incomplete despite the bundle being
> marked dispatched. The analysis/reasoning below is otherwise unchanged from
> the original write-up -- this is a status correction, not a rewrite.
>
> **Follow-up, same date (2026-08-27), later pass**: this audit's own claim
> that "no `workflow_templates/` directory ... exists anywhere in this repo"
> (below) was itself imprecise -- a `workflow_templates/` directory *did*
> exist (commit `9d692f8`, before this audit was written), with a single-file
> `Snakefile.example` and `fastdna.nf` per engine. What that commit shipped
> was real but narrow: each file only shelled out plain single-file `fastdna
> count`, with no config file, no paired-end handling, and no combined/
> cohort-level output -- not the runnable, cohort-scale pipeline this
> roadmap item and `docs/goal-most-complete-genomics-ml-library.md` call for.
> That gap is now closed: `workflow_templates/snakemake/` (`config.yaml` +
> `workflow/Snakefile` + `README.md`) and `workflow_templates/nextflow/`
> (`main.nf` + `nextflow.config` + `README.md`) both ship a real paired-end
> cohort pipeline -- per-sample counting via `fastdna count`'s multi-file
> `--input`, combined into one cohort table via `fastdna union` -- verified
> by a real `fastdna` release build plus real end-to-end `snakemake`/
> `nextflow run` executions against it (see `CHANGELOG.md`'s `[Unreleased]`
> entry for the exact verification steps). The old single-file stubs were
> removed as superseded. Feature 4 is fully shipped as of this date.

Living reference for the "make FastDNA the library every biotechnologist knows"
push. Ten features, in two waves. Each entry: what it is, why it matters, what
it's built on, and its status. Written so the vision survives even if the
person building any one piece changes.

**Guiding principle** (stated explicitly so it doesn't get lost across many
parallel contributors): FastDNA does not try to be an aligner, a variant
caller, or a full assembler — those are mature disciplines with a decade+ of
incumbent tools. Every feature here is either (a) a direct extension of exact
k-mer counting or MinHash sketching, the two things FastDNA's Rust core is
already fast and correct at, or (b) glue that makes FastDNA's Arrow-native
output trivial to use from the Python ML ecosystem people already have
installed. Nothing here reinvents alignment, variant calling, or assembly.

---

## Wave 1 — core ML-facing functionality

### 1. `fastdna.sklearn.KmerVectorizer`

**What**: a scikit-learn-compatible transformer (`BaseEstimator` +
`TransformerMixin`) that turns a list of FASTQ paths into a numeric feature
matrix, usable as a step in `sklearn.pipeline.Pipeline`.

**Why it matters most**: genomics ML has a well-known, easy-to-commit
mistake — picking which k-mers to use as features by looking at the *whole*
dataset (including the test set) before splitting. Because `Pipeline` /
`cross_val_score` call `.fit()` only on the training fold at each split, and
`KmerVectorizer.fit()` is the only place the vocabulary is ever decided,
leakage becomes structurally impossible rather than a discipline the caller
has to remember. This is the single highest-leverage feature for a data
scientist audience: it connects anyone who already knows `sklearn` to
FastDNA with zero genomics-specific plumbing.

**Built on**: `fastdna.count()` (existing, stable).

**Status**: **Shipped.** `python/fastdna/sklearn.py`. Grown well past the
original scope since dispatch (per `CHANGELOG.md`): `chunk_size=` for
batched vocabulary construction, `counts=` to reuse a precomputed
`CohortCounts` instead of recounting, `representation=` (presence/count/
relative/clr) with a `DepthConfoundingWarning`.

---

### 2. `fastdna.taxonomy` — k-mer classification & sample identity

**What**: `classify(query, reference_db)` ranks reference sketches by how
well they explain a query sample (containment-based, sourmash/Mash-Screen
style); `check_sample_identity(a, b)` answers "are these plausibly the same
biological sample" for lab QC / sample-swap detection.

**Why it matters**: "what's in this sample" and "did we mix up two samples"
are asked in every sequencing lab, every day. Containment (not Jaccard) is
the right metric because a query is usually much smaller/differently
composed than a reference genome — Jaccard would falsely report low
similarity purely from the size mismatch.

**Built on**: `fastdna.sketch()` / `Sketch.containment()` /
`Sketch.mash_distance()` (existing, stable).

**Status**: **Shipped, including the stretch goal.** `python/fastdna/
taxonomy.py`'s `classify()` and `check_sample_identity()` both exist, and
the "gather"-style multi-organism decomposition stretch goal was also
completed: `taxonomy.gather()` (line ~426) is a real function, not just
offered-and-declined. Both `classify`/`gather` accept a `scale` parameter to
use `FracSketch` (see `ml-differentiation-roadmap.md`'s B1) internally
instead of the fixed-count `Sketch`, for unbiased containment at large size
mismatches.

---

### 3. `fastdna.assembly_qc` — Merqury-style assembly evaluation

**What**: `evaluate_assembly(assembly_fasta, reads_fastq)` grades a genome
assembly's quality using the *reads* as ground truth (no reference genome
needed): a QV (Phred-scaled accuracy estimate), completeness (%), and a
spectra comparison (reads' frequency spectrum vs. how those k-mers land in
the assembly — 0 copies = possible gap, >1 in a haploid assembly = possible
collapsed repeat).

**Why it matters**: assembly QC without a truth reference is a real, common
need (most assemblies don't have one), and Merqury's k-mer-based approach
(Rhie et al., 2020) is the field's established method — reusing
`KmerCounts.suggest_min_count()` for the error/signal threshold reuses this
project's own valley-detection instead of a hardcoded constant.

**Built on**: `fastdna.count()`, `KmerCounts.spectrum()` /
`.suggest_min_count()` (existing, stable). The assembly side now also goes
through `fastdna.count()` directly (the Rust core's native, content-sniffed
FASTA support -- `feature-gap-analysis.md`'s Q1); a pure-Python FASTA
k-mer extractor is kept in the module, unused by `evaluate_assembly()`
itself, as a manual building block for `evaluate_kmers()`'s lower-level
entry point.

**Status**: **Shipped, including the B4 follow-up --
see `ml-differentiation-roadmap.md`'s B4.** `python/fastdna/assembly_qc.py::
evaluate_assembly()` works and is tested. Its module docstring previously
explained that the Rust core had "no FASTA support" and so the assembly
side had to go through a slow pure-Python k-mer extractor unless the
caller manually renamed the assembly to `.fastq`; that gap is now closed.
`src/fastq.rs` gained native FASTA support (`feature-gap-analysis.md`'s
Q1), and `evaluate_assembly()`'s assembly-side extraction
(`_assembly_kmer_counts`) now always calls `fastdna.count()` directly,
regardless of the assembly file's extension. Note: this was measured, not
assumed, to be a correctness/maintenance win rather than an unconditional
wall-clock one -- see `assembly_qc.py`'s own module docstring for the
measured caveat (decoding k-mers back to Python strings has a real fixed
cost that dominates at small-to-medium fixture sizes).

---

### 4. Ecosystem & adoption infrastructure

**What**: four independent pieces bundled into one task —
(a) `_repr_html_` on `KmerCounts`/`Sketch` for Jupyter,
(b) `fastdna.interop` accepting Biopython `SeqRecord`s / plain sequence
iterables (not just file paths) by writing a temp FASTQ under the hood,
(c) Snakemake and Nextflow workflow templates,
(d) a bioconda recipe skeleton.

**Why it matters**: a library's *reach* depends as much on how it fits
existing workflows as on what it computes. Bioinformaticians install from
bioconda, not just PyPI; pipelines run in Snakemake/Nextflow, not standalone
scripts; and a notebook that renders a wall of `repr()` text instead of a
readable summary loses people in the first five minutes.

**Built on**: existing stable API for (a)/(b); pure documentation/config for
(c)/(d).

**Status**: **Shipped, 4 of 4 sub-items landed** (as of the 2026-08-27
follow-up pass; (c) was the last to close -- see this doc's top-of-file note).
- (a) **Shipped.** `_repr_html_` exists on both `KmerCounts`
  (`python/fastdna/__init__.py:389`) and `Sketch` (`__init__.py:602`).
- (b) **Shipped.** `python/fastdna/interop.py`'s `count_from_sequences()`/
  `sketch_from_sequences()` accept plain strings, `(id, seq)` pairs, and
  duck-typed Biopython `SeqRecord`-like objects.
- (c) **Shipped 2026-08-27.** `workflow_templates/snakemake/` (`config.yaml`,
  `workflow/Snakefile`, `README.md`) and `workflow_templates/nextflow/`
  (`main.nf`, `nextflow.config`, `README.md`): a config-driven cohort
  pipeline that counts paired-end FASTQ samples per-sample via `fastdna
  count`'s multi-file `--input`, then folds every sample's Parquet table
  into one cohort-level table via `fastdna union`. Verified against a real
  release build of `fastdna` (exact flags confirmed via `--help` output) and
  by running both templates end-to-end for real (Docker `snakemake/
  snakemake` and a real `nextflow run`) over a small paired-end fixture --
  see `CHANGELOG.md`'s `[Unreleased]` entry for the full verification
  record. An earlier, narrower pair of single-file stubs (commit `9d692f8`:
  `Snakefile.example`/`fastdna.nf`, single-sample-only, no config, no
  combined output) was superseded and removed.
- (d) **Shipped, unverified/unsubmitted** — same evidence as
  `feature-gap-analysis.md`'s Q5: `recipe/meta.yaml` exists with a
  placeholder `source.sha256`.

---

## Wave 2 — ML-native extensions

### 5. `fastdna.embed_cohort` — cohort visualization via UMAP

**What**: `embed_cohort(paths) -> 2D/3D coordinates`, built directly on
`fastdna.compare_all(paths, metric="mash_distance")`'s N×N distance table —
UMAP (or t-SNE/PCoA as alternatives) over a precomputed distance matrix is a
well-supported, standard operation, and `compare_all` already produces
exactly the right input shape.

**Why it matters**: "plot my cohort in 2D, colored by whatever metadata I
have, based on genomic similarity" is a constant exploratory step in
microbiome/population genomics (this is what's usually done today with 16S
amplicon data via QIIME2/PCoA) — with `compare_all` already existing, this
is close to a thin wrapper, not new algorithmic work, and one of the fastest
wins in this whole roadmap.

**Built on**: `fastdna.compare_all()` (existing, stable, wave-1-independent
— this needed no mocking or contract assumptions to build against).

**Status**: **Shipped.** `python/fastdna/embed.py`.

---

### 6. `fastdna.interpret` — feature importance back to real DNA

**What**: utilities that take a trained model's feature importances (SHAP
values, linear coefficients, tree feature_importances_) over
`KmerVectorizer`-produced features and map them back to literal k-mer
sequences (via `get_feature_names_out()`), plus a convenience export of the
top-N important k-mers as a FASTA file ready to paste into BLAST.

**Why it matters**: this is a genuine differentiator against black-box deep
learning approaches to genomic ML (DNABERT-style transformers, raw sequence
embeddings) — a SHAP value on a k-mer feature points at one exact 21-base
motif a biologist can go look up, not an opaque embedding dimension.
"Interpretable by construction" is a real positioning argument, not just a
nice-to-have.

**Built on**: the `KmerVectorizer` contract from feature 1
(`vocabulary_`/`get_feature_names_out()`), which had not yet landed when
this was dispatched — built and tested against a documented stand-in
matching that contract; needs an integration pass once both are merged (see
the agent's own report for exactly what it assumed).

**Status**: **Shipped.** `python/fastdna/interpret.py`.

---

### 7. `fastdna.anomaly` — outlier detection for surveillance

**What (as originally scoped)**: wraps an unsupervised outlier detector
(isolation forest / one-class SVM, via scikit-learn) over per-sample
genomic-profile features to flag samples that deviate from a cohort's
established baseline, framed around emerging-pathogen surveillance.

**Status**: **Shipped, but not as originally framed --
superseded by `ml-differentiation-roadmap.md`'s A5, deliberately.** The
original "surveillance"-framed module was deleted (commit `c8870e3`: "audit:
surveillance framing had no literature support") and rebuilt as *cohort QC
outlier flagging* (reference-based contamination screening, Mash Screen /
NCBI STAT style — a claim the literature does support). The shipped module,
`python/fastdna/anomaly.py`'s `CohortOutlierFlagger`/`flag_cohort()`, fills
this roadmap slot's practical need (unsupervised deviation-from-baseline
detection over sketch/spectrum profiles) under the more defensible framing.
See A5 in `ml-differentiation-roadmap.md` for the full rationale.

---

### 8. `fastdna.active_learning` — uncertainty-driven curation

**What**: a generic uncertainty-sampling utility over any ranked
`(label, score)`-shaped classification output — not hard-wired to
`taxonomy.classify()` specifically, so it works whether the scores come from
`taxonomy.classify()`, a `KmerVectorizer` + sklearn classifier's
`predict_proba()`, or anything else shaped the same way. Flags the
least-confident results as candidates for expert review/labeling.

**Why it matters**: when `taxonomy.classify()` doesn't match a query
confidently against any reference, that's exactly the signal an active
learning loop needs to prioritize "this sample needs a human/expert label,
and probably a new reference added" — turning classification uncertainty
into an actionable curation queue rather than a silently-ignored low score.

**Built on**: deliberately generic (any ranked-score input), so it did not
need to depend on the in-flight `taxonomy` module's not-yet-landed exact
API — the connection to `taxonomy.classify()` is by input shape, documented
in the agent's own module, not a hard import.

**Status**: **Shipped.** `python/fastdna/active_learning.py`.

---

### 9. `fastdna.multiomics` — joining k-mer features with other data

**What**: utilities to join a k-mer feature table (from `count()`'s
`.table`, or a `KmerVectorizer`-produced matrix) with arbitrary other
tabular data (other omics layers, clinical metadata) by sample ID, producing
one combined table ready for a multi-modal model.

**Why it matters**: real studies are rarely genomics-only — clinical
metadata, other omics layers (transcriptomics, proteomics), and sequencing
data all need to line up by sample before a model can use them together.
Since FastDNA already speaks Arrow/pandas/polars natively, this is mostly
about making the *joining* convention explicit and correct (sample ID
matching, handling samples present in one layer but not another) rather
than inventing new computation.

**Built on**: `fastdna.count()` / `.table` (existing, stable) plus generic
tabular-join logic — does not require `KmerVectorizer` to exist, though it
becomes more useful alongside it.

**Status**: **Shipped.** `python/fastdna/multiomics.py`. `kmer_feature_table()`
was since rewritten with vectorized Arrow operations instead of per-cell
Python dicts (`CHANGELOG.md`, performance pass).

---

## Cross-cutting notes for whoever integrates all of this later

- **Wave 1 and wave 2 were dispatched as ten total isolated, parallel
  worktree agents** (four in wave 1, five in wave 2 — item 10 from the
  original ten-item conversation, "foundation-model-style pretraining front
  end," was folded into this doc as future/aspirational rather than
  dispatched as its own task; see the note at the very end). Each agent
  worked from a snapshot of the repo at dispatch time and could not see any
  other agent's in-flight work — features 6 and 8 above were explicitly
  scoped to depend only on already-stable APIs or on documented contracts
  rather than another in-flight module, specifically to keep true
  parallelism safe. The integration pass this section anticipated has
  happened in substance (all nine dispatched modules are live, tested, and
  cross-referenced correctly with each other's real APIs as of the
  2026-08-27 audit). Workflow templates (feature 4c) were the one leftover
  gap this audit found, and were closed the same day -- see feature 4's
  status above and `CHANGELOG.md`'s `[Unreleased]` entry.
- **Every new module is pure Python**, importable explicitly
  (`from fastdna.sklearn import KmerVectorizer`, etc.), matching the
  project's existing convention of keeping heavier/optional dependencies
  (scikit-learn, scipy, UMAP, Biopython) out of the core `fastdna` package's
  hard dependencies.
- **A tenth, not-yet-scoped idea worth recording**: a lightweight k-mer
  co-occurrence embedding (a classical, much cheaper alternative to
  transformer-based DNA foundation models like DNABERT/Nucleotide
  Transformer) — build a co-occurrence matrix of k-mers across a large
  cohort and factorize it (SVD/GloVe-style) into dense vectors, enabling
  similarity search and transfer learning without the compute cost of a
  full sequence-transformer pretraining run. **Status: still not shipped,
  and no longer just "not started yet" -- `ml-differentiation-roadmap.md`'s
  C3 has since evaluated this idea specifically and parked it** ("no
  2022-2026 evidence they beat presence/absence on phenotype tasks"). Kept
  here as a record of the idea's origin; treat C3's framing as the current
  word on its priority, not this paragraph's "natural wave 3 anchor".
