# ML differentiation roadmap (researched 2026-08-24)

> **Audit pass, 2026-08-27**: every bucket item below was re-checked against
> the actual code on this date, per
> `docs/goal-most-complete-genomics-ml-library.md`'s instruction to verify
> before executing anything on this list. Of the Bucket A/B items, A5, A1,
> A2, A3, A4, B1, B3 are already shipped, and B4 shipped in this pass
> (`assembly_qc.py`'s FASTA path is now wired through `fastdna.count()`,
> see its entry below). **B2's finishing touch has since landed too** (a
> `fastdna matrix` CLI verb and generic Parquet export for `CohortMatrix`,
> see its entry below) -- every Bucket A/B item is now shipped. Bucket C is
> still exactly as parked as this doc already said. The citation-backed
> analysis below is otherwise unchanged -- this is a status correction, not
> a rewrite.

Literature-backed plan to make FastDNA's ML layer a genuine differentiator.
Complements `ml-genomics-roadmap.md` (waves 1-2, landed) and
`feature-gap-analysis.md` (engine-side gaps). Every claim carries a citation;
be suspicious of any future addition that does not.

## Core strategic finding

The 2024-2026 literature does NOT support "k-mers beat deep learning on
accuracy" as a blanket claim (a 2026 Frontiers E. coli study found gene-based
features beating k-mers for AMR). The defensible differentiators are:

1. **Rigor.** Population structure confounds AMR ML (PLOS Biology 2025,
   24k+ genomes): random CV splits leak clonal relatives across train/test
   and inflate every published number; phylogeny-aware CV is the field's
   named missing tool (arXiv 2502.07749; BiB 2024 benchmark used three split
   methods for this reason). This is not an unsolved problem in general --
   homology-based splitting already exists for protein ML (CD-HIT,
   MMseqs2), and population-structure correction already exists for GWAS
   specifically (pyseer, mixed models over a precomputed kinship matrix).
   What is missing is a general-purpose, sklearn-native `cv=` splitter that
   derives the grouping directly from raw genomic sequence -- no separate
   tool, no precomputed matrix -- and composes with *any* estimator, not
   just an association test. FastDNA can, because it already computes Mash
   distances (`compare_all`).
2. **Interpretability.** gLM embeddings underperform supervised baselines on
   regulatory tasks (Koo lab, Genome Biology 2025); Evo 2 has documented
   blind spots and H100-class inference costs. A SHAP value on an exact
   31-mer is a BLASTable sequence — `interpret.py` already exploits this.
3. **Be the on-ramp.** kmersGWAS/pyseer/GenomeScope/Merqury are entrenched;
   feed them from one Python call instead of competing.

One-line thesis: *the only k-mer library where the evaluation is
genomics-aware (lineage-blocked CV, calibrated, permutation-tested) and every
model output is a literal DNA sequence you can BLAST.*

## Bucket A — quick wins on the current core

- **A5 (do first, hygiene): `anomaly.py` replacement.** **Shipped.** The
  module was deliberately deleted in commit c8870e3 (audit: surveillance
  framing had no literature support); `test_anomaly.py` sat skipped
  awaiting a real replacement. Rebuilt as *cohort QC outlier flagging*
  (validated framing: reference-based contamination screening à la Mash
  Screen / NCBI STAT, Genome Biology 2021) in `python/fastdna/anomaly.py`'s
  `CohortOutlierFlagger`/`flag_cohort()` — the "emerging pathogen
  surveillance" framing this item explicitly rejected does not appear.
- **A1 `fastdna.cv` — lineage-aware leakage-safe evaluation (rank #1).**
  **Shipped**, split across two modules. `python/fastdna/cv.py`: `lineage_
  groups()`, `LineageKFold`, `permutation_importance_pvalues()` — the
  `compare_all()` → single-linkage-clusters-at-a-Mash-threshold →
  sklearn-CV-splitter pipeline this item specifies. Venn-ABERS calibration
  (`calibrate()`) shipped in the companion module `python/fastdna/
  calibration.py` instead of `cv.py` itself (a reasonable split — that
  module's own docstring cites the same Vovk & Petej UAI 2014 IVAP method
  this item names, and this doc's own citation list). `CalibratedEstimator`
  is the returned type.
- **A2 `fastdna.genomescope` — spectrum mixture-model profiling (rank #2).**
  **Shipped.** `python/fastdna/genomescope.py`'s `profile_genome()` /
  `GenomeProfile` implement the GenomeScope 2.0 model (genome size,
  heterozygosity, repeats, coverage, error rate) over `KmerCounts.
  spectrum()`, plus `plot_spectrum_fit()`. Not independently verified in
  this audit pass whether it is diploid-only-first or reports
  `converged=False` honestly per the item's stated scope-down — read
  `genomescope.py`'s own docstring before relying on that detail.
- **A3 `fastdna.gwas` — cohort matrix + pyseer/kmersGWAS bridge (rank #3).**
  **Shipped.** `python/fastdna/gwas.py`: `cohort_presence_matrix()` (backed
  by the real Rust artifact `src/cohort/matrix.rs`, see B2 below),
  `export_pyseer_kmers()`, `kinship_matrix()`. Also ships
  `prefilter_association()`, a univariate/Benjamini-Hochberg prefilter —
  consistent with this item's explicit "do NOT reimplement LMM association
  statistics" instruction, since a prefilter is not an LMM.
- **A4 Kover-style set-covering rule models for AMR (rank #4).**
  **Shipped, public-benchmark reproduction unverified.** `python/fastdna/
  rules.py`'s `SetCoveringClassifier` implements the set-covering rule-model
  approach this item names. This audit pass found no artifact (script,
  notebook, or test) reproducing a public BV-BRC dataset result, which this
  item's own text says is required "before claiming parity" with the BiB
  2024 benchmark's top-ranked method — treat that specific claim as still
  open even though the estimator itself is shipped.

## Bucket B — modest Rust additions

- **B1 FracMinHash (scaled) sketches (rank #5).** **Shipped.**
  `src/sketch.rs::FracSketch` (Rust, with `containment`/`jaccard`/`save`/
  `load`, extensive unit tests including a direct measurement of the
  bottom-k bias it fixes) and the Python surface `frac_sketch()`/
  `FracSketch`/`load_frac_sketch()` in `python/fastdna/__init__.py`. Already
  used internally by `taxonomy.classify()`/`taxonomy.gather()` via their
  `scale` parameter, exactly as this item anticipated.
- **B2 Rust cohort mode → one Arrow samples×kmers matrix (rank #6).**
  **Shipped**, in the shape the finishing touch below always meant: a
  `fastdna matrix` CLI verb and a generic Parquet export of `CohortMatrix`,
  independent of `gwas.py`. `src/cohort/matrix.rs::CohortMatrix` is real,
  tested Rust-core work building a cohort-wide k-mer matrix directly from
  each sample's own sorted `KmerCounter` table (see that file's module doc
  comment for the double-decode/double-copy cost it removes versus the old
  per-sample-`count()` approach); it is wired into Python as `gwas.py::
  cohort_presence_matrix()` (returning `scipy.sparse.csr_matrix`, left
  unchanged) and now also into `export::export_cohort_matrix_parquet`
  (`src/export.rs`) plus `cli::MatrixArgs`/`main.rs::run_matrix` (`fastdna
  matrix --input DIR|--sample FILE... -o cohort.parquet`). The Parquet file
  is a long/COO table (`sample_id, kmer_u64, count`, `kmer_sequence`
  opt-in) rather than a dense wide grid or a second Arrow-in-memory path --
  the same shape `CohortMatrix` already holds, usable directly from
  DuckDB/pandas/polars with no FastDNA-specific reader. Two things this
  entry named that remain genuinely out of scope: the optional
  feature-hashing output was not added (no consumer currently needs it, and
  it would be a second, divergent column contract on top of the one just
  shipped); and the exporter-schema-contract watch-item has still only been
  tested against the one consumer this pass added (`export_cohort_matrix_
  parquet` itself, via `tests/matrix_cli.rs`), not a second independent
  reader outside this repo.
- **B3 Per-read screening (rank #7).** **Shipped.** `src/metagenomics.rs`'s
  `ReadClassification`/`classify_source`/`classification_batch` produce
  exactly the `read_id, n_kmers, n_matched`-shaped Arrow stream this item
  specifies (fields: `read_id, tax_id, confidence, n_kmers,
  n_classified_kmers`), against a reference set — here a Kraken-style
  taxonomy-tagged database, a superset of the plain "reference set" this
  item names. Python surface: `python/fastdna/metagenomics.py::
  KmerDatabase.classify()`. This is real substrate for the Mykrobe-style-
  probes and honest-contamination-path uses this item cites, though nobody
  has yet built either of those consumers on top of it.
- **B4 Native FASTA counting (rank #8).** **Shipped (2026-08-27).**
  `feature-gap-analysis.md`'s Q1 shipped native FASTA support in the Rust
  core (`src/fastq.rs::Format::Fasta`), and `python/fastdna/
  assembly_qc.py::_assembly_kmer_counts` now uses it: `evaluate_assembly()`'s
  assembly side always goes through `fastdna.count()` (no more extension-
  based dispatch to a pure-Python fallback), matching the module's updated
  docstring. See `feature-gap-analysis.md`'s own B4 entry for the file-level
  citation and equivalence tests
  (`tests/fasta_input.rs`, `python/tests/test_assembly_qc.py::
  TestFastPathMatchesPurePythonFallback`). See `ml-genomics-roadmap.md`'s
  feature 3 for the same finding from the other doc's angle.

## Bucket C — ambitious bets

- **C1 k-mer → DNA-LM hybrid**: **Not shipped.** No evidence of any
  DNABERT-2/NT/Evo 2 integration anywhere in the repo. Still exactly the
  "k-mers shortlist, LM annotates the hits' read contexts" idea this item
  describes, unstarted.
- **C2 real-time surveillance mode**: **Not shipped**, still explicitly
  parked pending B1/B3 (both now shipped, see above) + a pilot user. The
  code prerequisites this item was waiting on now exist; the pilot-user
  gate is a business/adoption condition this audit cannot resolve from the
  code alone.
- **C3 co-occurrence k-mer embeddings**: **Not shipped, still parked** — no
  2022-2026 evidence found (in this pass either) that this beats
  presence/absence on phenotype tasks. Note for cross-reference:
  `ml-genomics-roadmap.md`'s "tenth idea" (a co-occurrence-embedding wave-3
  anchor) is the same idea this item already evaluated and parked — that
  doc has been updated to point back here rather than presenting it as a
  fresh, unevaluated opportunity.

## Claims to avoid

Blanket "k-mers outperform DL"; isolation-forest "surveillance"; UMAP as a
differentiator (commodity); homegrown LMM GWAS stats; building/serving a
foundation model.

## Execution order

Re-derived 2026-08-27 against what is actually still open. The original
order was A5 → A1 → B1 → A2 → A3 → B4 → B2 → A4 (+public benchmark) → B3 →
C1, all strictly by rank; since A5/A1/A2/A3/A4(estimator)/B1/B2/B3/B4 have
all now shipped, only A4's benchmark-reproduction claim remains genuinely
open in Buckets A/B:

1. **A4's public-benchmark reproduction** against a BV-BRC dataset, to
   actually claim the parity this item's text requires before claiming it.
2. **C1**, now that its stated prerequisites (B1, B3) are both shipped —
   still the correct next ambitious bet per the original ordering.
3. **C2**, once a pilot user is available (unchanged gate).
4. **C3** stays parked; re-open only if new 2026+ evidence appears.

Full citation list: see the 2026-08-24 research session notes (PLOS Biology
2025 population-structure confounding; BiB 2024 AMR benchmark; arXiv
2502.07749; GenomeScope 2.0 Nat Commun 2020; Irber 2022 FracMinHash; sourmash
publications; Voichek & Weigel kmersGWAS; DBGWAS PLOS Genetics 2018; pyseer
docs; Mykrobe 2019; DNABERT-2 ICLR 2024; BEND 2023; Koo Genome Biology 2025;
NCBI STAT Genome Biology 2021; Venn-ABERS medRxiv 2025).
