# ML differentiation roadmap (researched 2026-08-24)

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
   methods for this reason). No mainstream library derives leakage-safe
   splits from the genomes themselves. FastDNA can, because it already
   computes Mash distances (`compare_all`).
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

- **A5 (do first, hygiene): `anomaly.py` replacement.** The module was
  deliberately deleted in commit c8870e3 (audit: surveillance framing had no
  literature support); `test_anomaly.py` sits skipped awaiting a real
  replacement. Rebuild as *cohort QC outlier flagging* (validated framing:
  reference-based contamination screening à la Mash Screen / NCBI STAT,
  Genome Biology 2021), not "emerging pathogen surveillance".
- **A1 `fastdna.cv` — lineage-aware leakage-safe evaluation (rank #1).**
  `compare_all()` → single-linkage clusters at a Mash threshold → sklearn CV
  splitter; permutation-test biomarker significance; Venn-ABERS calibration
  (used in clinical microbiology, medRxiv 2025). API: `lineage_groups`,
  `LineageKFold`, `permutation_importance_pvalues`, `calibrate`. Effort S-M.
- **A2 `fastdna.genomescope` — spectrum mixture-model profiling (rank #2).**
  GenomeScope 2.0 model (Nat Commun 2020) over `KmerCounts.spectrum()`:
  genome size, heterozygosity, repeats, coverage, error rate. Ship
  diploid-only first; report `converged=False` honestly. Effort M.
- **A3 `fastdna.gwas` — cohort matrix + pyseer/kmersGWAS bridge (rank #3).**
  Export pyseer `--kmers` files and the kinship matrix from sketches. Do NOT
  reimplement LMM association statistics. Effort M.
- **A4 Kover-style set-covering rule models for AMR (rank #4).** Top-ranked
  in the BiB 2024 78-dataset benchmark; rules are literal k-mers. Must
  reproduce a public BV-BRC dataset result before claiming parity. Effort M.

## Bucket B — modest Rust additions

- **B1 FracMinHash (scaled) sketches (rank #5).** Bottom-k containment is
  biased for wildly different set sizes — exactly `taxonomy.gather`'s case.
  FracMinHash (Irber 2022) fixes it, enables sketch subtraction (iterative
  gather) and ANI with confidence intervals (Hera 2023). Validate against
  sourmash outputs. Effort M.
- **B2 Rust cohort mode → one Arrow samples×kmers matrix (rank #6).** Plus
  optional feature-hashing output. Watch the exporter schema contract.
- **B3 Per-read screening (rank #7).** `read_id, n_kmers, n_matched` Arrow
  stream against a reference set; substrate for Mykrobe-style probes and the
  honest contamination path.
- **B4 Native FASTA counting (rank #8).** `assembly_qc.py` counts assembly
  k-mers in pure Python today — unusable at 3 Gb; Merqury parity needs it.

## Bucket C — ambitious bets

- **C1 k-mer → DNA-LM hybrid**: k-mers shortlist, LM annotates the hits'
  read contexts (DNABERT-2 / NT / Evo 2 as optional extras).
- **C2 real-time surveillance mode**: parked until B1/B3 + a pilot user.
- **C3 co-occurrence k-mer embeddings**: parked — no 2022-2026 evidence they
  beat presence/absence on phenotype tasks.

## Claims to avoid

Blanket "k-mers outperform DL"; isolation-forest "surveillance"; UMAP as a
differentiator (commodity); homegrown LMM GWAS stats; building/serving a
foundation model.

## Execution order

A5 → A1 → B1 → A2 → A3 → B4 → B2 → A4 (+public benchmark) → B3 → C1.

Full citation list: see the 2026-08-24 research session notes (PLOS Biology
2025 population-structure confounding; BiB 2024 AMR benchmark; arXiv
2502.07749; GenomeScope 2.0 Nat Commun 2020; Irber 2022 FracMinHash; sourmash
publications; Voichek & Weigel kmersGWAS; DBGWAS PLOS Genetics 2018; pyseer
docs; Mykrobe 2019; DNABERT-2 ICLR 2024; BEND 2023; Koo Genome Biology 2025;
NCBI STAT Genome Biology 2021; Venn-ABERS medRxiv 2025).
