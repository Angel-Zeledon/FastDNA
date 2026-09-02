# Real-data validation: does this actually work outside our own tests?

Date: 2026-08-25. Everything built this session (`fastdna.rules`,
`fastdna.cv`, `fastdna.gwas`, `fastdna.metagenomics`) had only ever been
exercised against synthetic data and unit tests. This is the first run
against real, public genomes with independently established ground truth --
laboratory-confirmed resistance phenotypes for Track A, submitter-declared
species identity for Track B. Every number below came out of an actual run
of the actual pipeline; nothing here was tuned to make the result look
better.

## Track A -- AMR genotype-to-phenotype prediction

### Dataset

50 real *Staphylococcus aureus* genome assemblies (25 laboratory-confirmed
oxacillin-resistant, 25 susceptible), downloaded from **BV-BRC** (the
successor to PATRIC; public, unauthenticated Data API,
`https://www.bv-brc.org/api`) -- the same warehouse the Kover/Drouin
literature and the Briefings in Bioinformatics 2024 AMR benchmark draw from.
Genome ids and their `resistant_phenotype` calls came straight from
BV-BRC's `genome_amr` table (`eq(taxon_id,1280)&eq(antibiotic,oxacillin)`);
the first 25 of each class, sorted by genome id, were taken verbatim -- no
cherry-picking. Assemblies were reconstructed from BV-BRC's
`genome_sequence` table (per-contig FASTA records), not from raw reads,
which is why 50 genomes downloaded in under two minutes instead of hours.

The exact PATRIC/Kover-style raw-read dataset (Drouin et al. 2016/2019) was
not attempted: those studies pull tens of thousands of raw FASTQ read sets
(hundreds of MB to GB each), which is not a "practical" download for a
same-session validation run. BV-BRC's own AMR metadata for the same
species/antibiotic, with real laboratory phenotypes, is the closest
available substitute that is still genuinely independent ground truth
(EUCAST/CLSI broth dilution, disk diffusion, VITEK 2, or MIC calls -- not
something this session produced).

### Pipeline run, unmodified

```
cohort_presence_matrix(paths, k=31, min_count=1, min_samples=2, max_kmers=200_000)
  -> 50 samples x 200,000 k-mers, 5,039,911 nonzero entries
     (6,909,838 k-mers passed min_samples=2 before truncation)

lineage_groups(paths, k=21, sketch_size=1000, distance_threshold=0.01)
  -> 6 distinct lineages, sizes [11, 31, 2, 2, 3, 1]

LineageKFold(n_splits=5, groups=groups)
  -> SetCoveringClassifier(max_rules=10), fit/predict per fold
```

### Result

| fold | train | test | accuracy | rules learned |
|------|-------|------|----------|----------------|
| 0 | 19 | 31 | 1.000 | 1 |
| 1 | 39 | 11 | 1.000 | 1 |
| 2 | 47 | 3  | 1.000 | 1 |
| 3 | 47 | 3  | 1.000 | 1 |
| 4 | 48 | 2  | 1.000 | 1 |

**Pooled lineage-blocked CV accuracy: 100% (50/50).** Average precision
1.0, Brier score 0.0 (on the hard 0/1 output --
`calibration_report` correctly raised `UncalibratedScoresWarning`, as
documented). A random (non-lineage-blocked) 5-fold CV gives the identical
100% -- see "Honest caveat" below for what that does and does not mean.

Every fold converged to **exactly one rule**:

```
1 IF present(AAAAAAAGAAAATGGACTCGTTACAGTGTCA)
```

### Verdict: matches the published literature, for the expected biological reason

Oxacillin (methicillin) resistance in *S. aureus* is driven almost entirely
by carriage of the SCC*mec* cassette and its *mecA* gene -- Bradley et al.,
"Rapid antibiotic-resistance predictions from genome sequence data for
*Staphylococcus aureus* and *Mycobacterium tuberculosis*", *Nature
Communications* 6:10063 (2015), report ~99% concordance between a
resistance-gene panel (led by *mecA* presence/absence) and phenotypic
oxacillin susceptibility testing across hundreds of isolates. A single,
strong, mostly-binary genetic determinant is exactly the regime where a
single-k-mer rule should reach near-total accuracy, and it did: **100% on
this 50-genome sample is consistent with, and at this sample size
indistinguishable from, the ~99% figure in the literature.**

More importantly, the rule is not a black box coincidence. Its k-mer was
looked up directly in BV-BRC's own gene annotations for one of the
resistant genomes (1280.10000): the k-mer occurs at position 1655 of contig
`1280.10000.con.0022`, **148 bp from the annotated boundary of the gene
BV-BRC itself labels** `"Penicillin-binding protein PBP2a, methicillin
resistance determinant MecA, transpeptidase"` **(coordinates
`complement(1804..3810)` on the same contig)**. The rule's one k-mer sits
just outside the annotated CDS, not inside it -- so this is the *mecA*
locus/its immediate flank, not literally the coding sequence base-for-base
-- but it is emphatically not a spurious lineage marker unrelated to
resistance biology. `rules.py`'s central claim -- "the explanation *is* the
biology, not a proxy for it" -- held up against an independent, real
annotation database that had no part in fitting the model.

### Honest caveat

Lineage-blocked and random CV gave the *same* accuracy here (both 100%),
so this dataset does not demonstrate the accuracy *drop* `fastdna.cv`'s own
docstring says to expect from correcting population-structure leakage --
see `cv.py`: "Scores usually go **down** relative to random CV -- that is
the point." A signal this strong (near-monogenic resistance) swamps any
lineage-confounding effect enough that even a leakage-prone split can't
inflate the score further; it is already ~perfect. This dataset is not
proof that `LineageKFold` matters here -- it is proof the classifier's
predictions were not simply riding a lineage tag it would be *convenient*
but wrong to credit. A phenotype with a weaker or more polygenic genetic
basis (most antibiotics are not this clean) is where the random-vs-blocked
gap this module exists to catch would actually show up, and that dataset
was not built for this run.

No result here has been reproduced against the exact original Kover/PATRIC
benchmark split, so this is *consistent with* the literature, not a
reproduction of a specific published number on the identical dataset --
the roadmap's stated precondition for "claiming parity with Kover"
(`rules.py`'s own docstring) is not yet met.

## Track B -- Metagenomic classification

### Setup

A 5-species reference panel (*Escherichia coli*, *Staphylococcus aureus*,
*Klebsiella pneumoniae*, *Pseudomonas aeruginosa*, *Salmonella enterica*),
one complete/near-complete assembly per species, downloaded the same way
as Track A (BV-BRC `genome_sequence`), built into a `KmerDatabase` via
`fastdna.metagenomics.build_database(k=31)`: **24,856,216 distinct
k-mers, 298 MB resident** -- in line with the module's own documented
~12 bytes/k-mer model.

Real WGS Illumina reads for one of the five species -- run **DRR002015**
("*Escherichia coli* 2W14"), downloaded from ENA
(`ftp.sra.ebi.ac.uk/vol1/fastq/DRR002/DRR002015/DRR002015_1.fastq.gz`,
~97 MB, R1 only) -- were classified against the panel.

A true Zymo/HMP mock-community shotgun dataset (mixed known-abundance
reads from several organisms at once) would be a stronger test of
`abundance()`'s composition-recovery claim specifically, but no such
accession could be reliably located without a working web-search tool in
this environment (the session's web-search budget was exhausted by
concurrent sibling agents before this task reached it). A single real,
publicly submitted, species-labeled read set is a weaker substitute --
it validates per-read classification precision/specificity, not
abundance-recovery -- but its ground truth is just as real: reads
deposited at ENA as *E. coli* must, barring contamination, actually be
*E. coli*.

**Kraken 2 was not installed.** This is a Windows host with no `kraken2`
binary, no `conda`/`bioconda`, and no confirmed WSL2 access from this
session; Kraken 2 has no native Windows build. Attempting a build from
source was judged out of scope for a same-session validation run. This is
stated plainly rather than skipped silently, per instructions.

### Result

2,343,637 reads classified:

| tax_id | name | rank | reads | relative_abundance |
|---|---|---|---|---|
| 562 | *Escherichia coli* | species | 1,657,184 | 70.71% |
| 0 | unclassified | -- | 635,159 | 27.10% |
| 1 | root | no rank | 37,286 | 1.59% |
| 28901 | *Salmonella enterica* | species | 7,558 | 0.322% |
| 573 | *Klebsiella pneumoniae* | species | 6,186 | 0.264% |
| 287 | *Pseudomonas aeruginosa* | species | 262 | 0.0112% |
| 1280 | *Staphylococcus aureus* | species | 2 | 0.0001% |

### Verdict: correct, with an honest, expected limitation

Of the 1,671,192 reads called to a **specific species** (excluding
`unclassified` and the ambiguous `root` calls), **99.16% were correctly
*E. coli***. The two cross-reactions of any real size were *Salmonella
enterica* (0.32%) and *Klebsiella pneumoniae* (0.26%) -- both, like
*E. coli*, members of family Enterobacteriaceae, which really do share
substantial genomic sequence; *Pseudomonas aeruginosa* (a different
phylum-adjacent genus, order Pseudomonadales) scored two orders of
magnitude lower, and *Staphylococcus aureus* (Gram-positive, a completely
different phylum) essentially zero (2 reads out of 2.3 million). The
cross-reaction pattern is exactly what real taxonomic relatedness
predicts, not noise.

The 27.10% unclassified rate is real and is exactly what the module's own
docstring warns about: this uses **one single reference assembly per
species with exact 64-bit k-mer matching and no spaced seeds** -- unlike
Kraken 2's compact hash table with spaced seeds, an exact match has zero
tolerance for a substitution. A real Illumina read set inevitably carries
both sequencing errors and genuine strain-level divergence from whichever
single *E. coli* assembly was chosen as the reference, and every k-mer
touching such a difference fails to match anything. This is the "spaced
seeds also cost real sensitivity" trade-off the module's own docstring
already names, now measured on real data instead of asserted: **a
~27% sensitivity cost for a ~30x smaller, exact-match, no-tolerance
database.** Whether that trade-off is acceptable depends entirely on the
use case (targeted panel screening vs. an open-world survey), which is
exactly the scope the module's docstring says it is built for.

## Bottom line

Both tracks ran the actual, unmodified pipeline end-to-end on real,
independently sourced data and got results that agree with the relevant
published/expected numbers, for defensible biological reasons that were
checked, not assumed. Neither is a reproduction of one specific paper's
exact benchmark on the exact same split -- that remains open work, and
should not be described as done.

---

# Track C -- against the field's own tools (2026-09-01)

Tracks A and B above validate *outcomes* against biological ground truth.
This track validates *components* against the established implementation of
the same algorithm, which is a different and complementary question: Track B
can only say "the classification looks right", not "the k-mer counts
underneath it are the same numbers KMC3 would produce".

**Unlike everything above, this track is executable.** Each row is produced
by a committed script, and `.github/workflows/validation.yml` runs them on a
schedule. That is deliberate: the two tracks above exist only as this
document, so if a regression broke them tomorrow, nothing would notice --
the exact failure mode that let `audit.py` cite an artefact
(`scratch/amr_repro/audit_report.json`) that is not in the repository.

| what | against | result | script |
|---|---|---|---|
| k-mer counting | **KMC3 3.2.4** | **exact match**, k=31/21/15 | `kmc3_equivalence.py` |
| HyperLogLog cardinality | exact count | -0.26% (bound ~0.8% at p=14) | `estimator_accuracy.py` |
| ntCard spectrum (f1/f2/f3) | exact histogram | -0.63% / +2.04% / -1.45% | `estimator_accuracy.py` |
| genome size | **GenomeScope2 2.0.1** | -0.29% | `estimator_accuracy.py` |
| MinHash distances | **Mash 2.3** | r=0.997, bias ~0 | `sketch_vs_mash.py` |
| assembly QV | **Merqury 1.4.1** | 18.4192 vs 18.4205 | `assembly_qc_vs_merqury.py` |
| species classification | published species identity | 12/12 held-out genomes | `taxonomy_real_species.py` |

Data: ENA `DRR002015` (*E. coli*, 2,343,637 reads) for the read-based rows;
8 BV-BRC *E. coli* assemblies for the Mash comparison. Reference tools run
from pinned biocontainers, so a rerun a year from now compares against the
same versions.

Three details that matter more than the headline percentages:

- **The counting comparison allows no tolerance at all.** With `-q 0`
  (trimming off) and `-ci1` (singletons kept), both tools solve the identical
  problem, so any difference is a bug. The comparison also has teeth: KMC's
  own default `-ci2` returns 4,766,994 distinct instead of 19,062,700, a 75%
  difference, so a flag mismatch could not slip through unnoticed.
- **The Mash comparison asserts bias, not equality.** MinHash is sampling;
  two correct implementations disagree by an amount falling as
  `1/sqrt(sketch_size)`. Measured: error fell 3.06x for a 10x sketch, against
  the 3.16x that `sqrt(10)` predicts, with bias ~0 at both sizes. An
  implementation whose error did *not* shrink with sketch size would be
  sampling the wrong space, and that check would catch it even if both tools
  shared a bug.
- **The QV row separates a formula from a pipeline.** The suite already
  transcribed Merqury's `qv.sh` arithmetic verbatim and checked FastDNA
  computes the same expression -- good evidence for the formula, none for
  which k-mers reach its numerator and denominator, since the fixtures
  behind it are 300 pseudorandom bases at a uniform Phred 40. Run on the
  same real (assembly, reads) pair, the two agree to four significant
  figures on a log-scale metric.
- **The genome-size row is the one that closed a real gap.** Before it,
  `profile_genome`'s central claim was tested only against a spectrum drawn
  from the same model family it fits, with an end-to-end assertion that
  accepted any answer between 2,500 and 10,000 bp on a synthetic 5 kb
  "genome". It now lands within 0.3% of GenomeScope2 on the same histogram,
  and reports heterozygosity ~0.001 -- the correct answer for a haploid
  organism, not an artefact.

Stated rather than buried: *E. coli* is haploid while both genome-size tools
assume diploid (`ploidy=2` / `-p 2`). They are outside their design
assumption in the same way, which is what keeps them comparable to each
other. A diploid cohort with a published genome size would exercise more of
the model and is still open work.

### What Track C does not cover

- **No Kraken 2 comparison.** `fastdna.metagenomics` is the one module whose
  reference implementation is not checked head to head here, because a fair
  comparison needs a Kraken 2 database built over the full NCBI taxonomy
  (gigabytes of downloads and a build step), not the handful of genomes the
  other rows use. What exists instead: Track B ran real *E. coli* reads
  against a 5-species panel and got 99.16% of species-level calls right with
  a biologically coherent cross-reaction pattern, and the taxonomy row above
  gets 12/12 on held-out real genomes. Both measure the same claim against
  ground truth rather than against Kraken 2's answer, which is weaker
  evidence for compatibility and equally strong evidence for correctness.
- **No diploid genome-size case**, per the note above.
- **No degraded-quality read set.** DRR002015 is uniformly Q30+
  (`q20_pct = 100.0`), so the quality-trimming path is never exercised by
  any row in this table. The README attributes its small KMC3/FASTK count
  differences to trimming; that explanation remains unverified, and a run
  with real 3'-end quality decay is what would settle it.

## Track D -- the leakage demonstration (2026-09-01)

The honest caveat in Track A said the *S. aureus* + oxacillin cohort could
not demonstrate `LineageKFold`'s value, because near-monogenic resistance
(*mecA*) leaves nothing for population structure to inflate. That
demonstration now exists, on a phenotype chosen for the opposite property:
*E. coli* + ciprofloxacin is polygenic (stepwise `gyrA`/`parC`, efflux,
`qnr`) and clone-associated (ST131).

> # RETRACTED (2026-09-01, same day)
>
> **The gap reported below is an artefact of the truncation bug, not a
> demonstration of population-structure leakage. Do not cite it.**
>
> These numbers were measured before it was discovered that `load_amr`
> returned only each assembly's first 25 contigs. Re-measuring after the fix
> did not reproduce them. A three-way comparison isolates the cause -- B and
> C differ *only* in whether the genomes were truncated:
>
> | | genomes | n | top_features | random-CV AUC | gap |
> |---|---|---:|---:|---:|---:|
> | A (as published below) | truncated | 200 | 10,000 | 0.755 | **+0.165** |
> | B | **complete** | 80 | 5,000 | **0.480** | -0.041 |
> | C | **truncated** | 80 | 5,000 | **0.715** | +0.007 |
>
> On complete genomes the model does not beat chance (0.480), so there is no
> signal for population structure to inflate. The apparent predictive power
> came from the truncation.
>
> **Why truncation created signal.** The cut was not uniform: it kept the
> first 25 contigs, so a 2-contig assembly survived intact and a 553-contig
> one was reduced to 3%. Across this cohort the surviving fraction ranged
> from 3.2% to 100% (median 24%), did **not** correlate with the phenotype
> (r=+0.072, p=0.52), and had **41% of its variance explained by sequence
> type** (R²=0.407). The bug therefore introduced an artificial variable
> encoding lineage -- and lineage does predict this phenotype (ST131 is 68%
> resistant). A model could learn that variable; blocking by lineage removed
> it; that produced the gap. The large gap additionally needed the 200-genome
> cohort: at n=80 the same truncated data gives +0.007.
>
> Worth stating plainly: `audit()` was not wrong. It correctly detected
> leakage from population structure. The structure had simply been
> manufactured by a data bug rather than by biology -- which is exactly why
> a leakage audit is a measurement about a *cohort*, not a verdict about a
> *method*.
>
> **What survives**, because it was measured on the complete genomes and
> improved rather than collapsed: `cv.lineage_groups` recovers the MLST
> partition almost exactly, ARI **0.931** (40 inferred groups against 38 real
> sequence types), where the truncated data gave 0.003. The lineage inference
> is sound; the input had been broken.
>
> **A replacement attempt also failed, and is reported rather than dropped.**
> *E. coli* + ampicillin was tried next, chosen mechanistically before its
> result was known: resistance there is carried by an accessory gene
> (`blaTEM`), which presence/absence of k-mers can see, unlike
> ciprofloxacin's `gyrA`/`parC` point mutations. On complete genomes, n=80:
> random-CV AUC **0.566**, gap **+0.075** with ±0.12 on the blocked side. A
> gap from "barely above chance" to "chance" is noise, not a demonstration.
>
> The likely cause is the experiment's design rather than the library:
> **5,000 features against 80 samples** (p/n = 62) with an unregularised
> logistic regression cannot learn anything that generalises. Attempt A had
> the same ratio and was rescued only by the spurious signal.
>
> What a valid demonstration would need, so the next attempt does not
> rediscover this: **n ≥ ~500 complete genomes** (which does not fit in 16 GB
> with `counts=` resident -- 200 already caused swap thrashing, so it wants
> `KmerVectorizer(disk_backed=True)` without `counts=`), a phenotype driven
> by accessory-gene content, and regularisation matched to p >> n.
>
> Changing antibiotic after attempt B is not the cherry-picking the script's
> own pre-registration forbids, and the distinction is worth being explicit
> about: B produced **no signal at all** (AUC 0.480), which makes the leakage
> question unanswerable rather than answered. Picking a new cohort because
> the gap was small would be cherry-picking; picking one because the model
> had nothing to audit is not. The unfavourable result of that new attempt is
> recorded above.
>
> The original section is kept below, unedited, because a retraction that
> deletes its own evidence is not a retraction.

200 BV-BRC genomes, 66 resistant / 134 susceptible, 79 sequence types:

| grouping | lineages | random CV | lineage-blocked | gap |
|---|---:|---:|---:|---:|
| MLST (ground truth) | 79 | 0.7548 | 0.5899 | **+0.165** |
| sketching @ 0.03 | 95 | 0.7548 | 0.5797 | **+0.175** |
| sketching @ 0.01 (old default) | 198 | 0.7548 | 0.7661 | -0.011 |

Chance is 0.5, so the model's lift over chance falls from 0.255 to 0.090:
**about 65% of its apparent predictive power was lineage memorisation.**
The cohort's confounding is measurable without any model at all -- a
predictor using *only* the sequence type, leave-one-out, scores AUC 0.815
(ST131 is 68% resistant; ST73 is 0%).

Two findings for the library itself came out of this, both since fixed:
sketch-derived grouping **reproduces** the MLST answer once the threshold is
right (so `cv.lineage_groups` is sound and only its default was
miscalibrated), and the old fixed `lineage_threshold=0.01` fell outside the
range this cohort's own dendrogram spans (0.019-0.044). See
`scripts/validation/lineage_leakage_experiment.py`, which carries a written
pre-registration that a near-zero gap is a reportable outcome rather than a
reason to try another antibiotic.

This is a demonstration that the tool works, not a discovery: PLOS Biology
2025 already measured this inflation across 24,000+ genomes, and arXiv
2502.07749 already named phylogeny-aware cross-validation the field's
missing standard tool. What is new here is that it runs end to end from
public accessions in one command.

---

# Track E -- calibrating the detector against known leakage (2026-09-01)

Tracks A-D validate against reality. This one validates against a
*constructed* truth, and it answers a question the others structurally
cannot: **when `audit()` reports a gap, is that the right number?**

On real data nobody knows the true amount of leakage, so a real-cohort
result is always a single point with no reference. Track D is the sharp
illustration -- it produced a confident +0.165 that turned out to measure a
data bug. Injecting a known amount of leakage removes that ambiguity
entirely.

## Construction

Two deliberately separable sources of phenotype, mixed by `lambda`
(`scripts/validation/leakage_calibration.py`):

- **Biological**: a marker sequence inserted independently of lineage. A
  model that learns it generalises to unseen lineages, because the marker is
  there too.
- **Lineage**: membership of designated resistant clones, predictable from
  k-mers characteristic of each clone -- which work under random CV and are
  absent from a held-out lineage.

`lambda = 0` means the phenotype is entirely the marker; `lambda = 1` means
it is entirely lineage. Three predictions were written down before running.

## Result

n=150 over 6 lineages, 1,000 features, 10 replicates per point:

| lambda | random CV | lineage-blocked | gap | sd | se |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0000 | 1.0000 | **+0.0000** | 0.000 | — |
| 0.2 | 0.9331 | 0.9056 | +0.0275 | 0.013 | 0.004 |
| 0.4 | 0.8674 | 0.8573 | +0.0101 | 0.021 | 0.006 |
| 0.6 | 0.8147 | 0.7908 | +0.0239 | 0.027 | 0.009 |
| 0.8 | 0.8530 | 0.7296 | +0.1234 | 0.037 | 0.012 |
| 1.0 | 0.8730 | 0.4463 | **+0.4267** | 0.083 | 0.026 |

Spearman(lambda, gap) = 0.829.

**Predictions 1 and 2 hold. Prediction 3 -- monotone increase -- does not.**
The response is a threshold rather than a slope: flat at 0.01-0.03 until
lineage supplies roughly 80% of the phenotype, then rising sharply. Ten
replicates make the standard errors small enough to separate the middle
points, and the Spearman coefficient is identical to the three-replicate run
(0.8286 both times), so this is a property of the system, not sampling
noise.

## What it establishes

1. **The detector is correct at the extremes.** Transferable signal produces
   *exactly* zero gap; pure lineage signal produces +0.43. It is measuring
   what it claims to measure.
2. **It is a detector, not a meter -- and the report's other number is the
   meter.** The gap identifies leakage as the *dominant* explanation and is
   insensitive to partial leakage coexisting with real biological signal.
   Measuring `confounding` on the same sweep closes that range:

   | lambda | gap | confounding |
   |---:|---:|---:|
   | 0.0 | +0.0000 | 0.0769 |
   | 0.2 | +0.0322 | 0.1859 |
   | 0.4 | +0.0103 | 0.3272 |
   | 0.6 | +0.0290 | 0.5576 |
   | 0.8 | +0.1559 | 0.7698 |
   | 1.0 | +0.4963 | 0.9864 |

   Spearman(lambda, gap) = 0.829; **Spearman(lambda, confounding) = 1.000**.

   `confounding` is monotone and near-linear precisely where the gap is
   flat, because it is computed from the labels with no model and so has no
   model-dependent threshold. The two answer different questions -- how much
   leakage the cohort makes *available*, versus how much this model *took* --
   and must be read as a pair. High confounding with a small gap is the
   informative combination: the cohort could have fooled a model, and this
   one did not.
3. **The overfitting confound is ruled out.** Both extremes share p/n, model
   and capacity; only the phenotype's source differs, and the gap moves from
   0.000 to +0.427. The capacity-driven gap documented in `audit`'s module
   docstring cannot explain this.

Point 2 is a limitation found by measurement rather than reported by a user,
which is the outcome this kind of calibration exists to produce.

---

# Track F -- module claims against constructed truth (2026-09-01)

Tracks C and E check FastDNA against something outside it: an established
implementation, or an injected quantity. Eighteen modules have neither
available -- there is no reference tool for "flag the sample that does not
belong" -- but their claims are still checkable, because the right answer
can be **built**.

Each check constructs a case whose answer exists before the call is made:

| module | constructed truth | result |
|---|---|---|
| `calibration` | P(y=1\|x) known analytically | error to TRUE probability 0.212 -> 0.044 |
| `anomaly` | one injected contaminant | flagged, 0 false positives |
| `genomic_model` | fit on one organism, predict another | silent in-domain, warns out |
| `validate_generated` | **the null**: identical distributions | no deviation claimed |
| `spectrum` | valley at a chosen depth | exact in 5/5 spectra |
| `embed` | known lineage labels | silhouette +0.996 (PCoA), +0.966 (UMAP) |
| `equivalence` | 3 features duplicated 4x | 12 columns -> 3 classes |
| `active_learning` | separation from tie to certainty | monotone, 1.000 -> 0.020 |
| `multiomics` | layers in different orders | joins by id, not position |
| `interop` | **real Biopython** SeqRecords | identical to plain strings |
| `mic` | **CLSI** essential agreement | 1 dilution agrees, 2 do not |
| `gwas` | one column set equal to phenotype | recovered, p=1.2e-59 |
| `annotate` | GFF written here | in-gene named, intergenic not |
| `rules` | one feature determines phenotype | that rule returned, accuracy 1.000 |
| `interpret` | known importances | ranked correctly both directions |
| `read_profile` | read lifted from the reference | all present vs all absent |
| `provenance` | hash recomputed independently | matches, and moves with content |
| `workflow` | end-to-end run | both promised warnings emitted |

Two have genuinely external anchors rather than constructed ones: `mic` is
checked against CLSI's essential-agreement convention, and `interop`
imports real Biopython, because the claim is interoperability with that
specific library and a stand-in cannot establish it.

**One of the eighteen failed**, and it is the reason this track exists.
`validate_generated` called identical distributions "very deviated" at
k=6, 11 and 21 -- a false positive that fired on any input at high k. It is
kept here as a permanent null control rather than trusted to stay fixed;
see the CHANGELOG entry for the coverage guard.

## What this track cannot establish

Passing here means the module does what it says on a case built to have a
knowable answer. It does not mean the module is useful on real data, and
the constructed cases are deliberately clean -- a single contaminant, one
causal feature, well-separated lineages. Real cohorts are messier in ways
these cases do not reproduce, which is what Tracks A-D are for.
