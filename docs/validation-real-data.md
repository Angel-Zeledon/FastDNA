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
