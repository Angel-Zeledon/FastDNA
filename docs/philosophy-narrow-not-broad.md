# Why FastDNA stays narrow: a decision record

> **REINSTATED 2026-09-05, and this time acted on.** This document was
> superseded on 2026-08-27 by a goal of becoming "the most complete library
> for genomics ML". That expansion was reversed on 2026-09-05 and the ML
> layer was removed outright — 31 Python modules and 4 Rust modules. The
> current decision is `docs/goal-fast-kmer-counter.md`; the evidence below
> is why it reads the way it does, and it was never refuted during the
> expansion, only overruled.
>
> Two things to read this document for: the outcome comparisons in this
> exact ecosystem (SAMtools/Biopython, AnnData/scikit-bio, `exon`+`biobear`
> dead against `oxbow`+`polars-bio` alive), and the two technical
> corrections at the end (the reverse-complement bit-trick derivation, and
> why a counting engine needs a total map and syncmers do not provide one).
> The boundary it draws against a general-purpose bioinformatics framework
> — no aligner, no pangenome graphs, no interval algebra — was never in
> dispute and still holds.

Date: 2026-08-25. This records a strategic decision, not a technical design.
Two external proposals (received the same day, unsolicited) argued for
turning FastDNA into a broad next-generation bioinformatics framework —
alignment, pangenome graphs, genomic interval algebra, a DLPack/PyTorch
bridge, long-read assembly, all in Rust with a first-class PyO3 layer. This
document is why that was declined, and what "complete" means for this
project instead.

## The evidence, not a preference

Two controlled comparisons already exist in the field, and both point the
same way:

- **SAMtools vs. Biopython**: published the same year, in the same venue,
  for the same community. SAMtools does one format, one job. Biopython does
  everything. SAMtools: 57,470 citations. Biopython: 5,675 — roughly 10x for
  about 1% of the scope.
- **AnnData vs. scikit-bio**: scikit-bio carries DOE funding, a *Nature
  Methods* paper, and ~500 functions, and has 101,280 downloads/month —
  1.4% of Biopython's. AnnData does one thing — a data structure — and has
  1.68M downloads/month, more than pysam, more than all of scikit-bio
  combined.
- **In the specific Rust+Arrow-for-genomics niche this proposal targets**:
  the two broadest projects that existed, `exon` ("an OLAP engine for
  biology") and `biobear`, are both dead. The two narrowest, `oxbow` and
  `polars-bio`, are alive and actively publishing.

**The finding that changes the calculus most**: `biotite`, a mature Python
bio library with 3.65M downloads/month, migrated its core to Rust in June
2026 and has an open PR to implement k-mer tables in Rust. "A fast Rust
kernel under a Python API" is therefore not a differentiator FastDNA owns
alone — convergence should be assumed, and competing on that axis is not
enough.

## Against the four proposed modules, specifically

Each of the modules proposed across both documents (pairwise alignment /
WFA / bit-parallel Myers edit distance; pangenome graphs / succinct r-index
structures; genomic interval algebra as a `bedtools` replacement; a
DLPack/PyTorch bridge for tokenized k-mer/syncmer features) has its own
decade-plus-entrenched incumbent with institutional backing: `minimap2`
(Heng Li, Broad/Harvard) for alignment, `vg`/`pggb` (Erik Garrison, backed
by the Human Pangenome Reference Consortium's infrastructure) for graphs,
`bedtools` for interval algebra. **Each reuses approximately none of
FastDNA's existing k-mer engine** — building any of them is starting over
against someone with a decade's head start, not extending what already
exists.

The maintenance-debt argument compounds this: a published API surface is
close to permanent. Biopython still ships `Bio.pairwise2` — deprecated and
flagged as redundant — four years after flagging it, because too much
third-party code imports it. FastDNA is effectively maintained by one
person with AI assistance. Every module shipped is a multi-year commitment
that specific team size cannot easily absorb.

## Two specific technical corrections made to the second proposal

1. **Its example `reverse_complement_2bit` snippet** (`kmer.swap_bytes()`
   then `(!reversed) & mask`) is structurally the same "recompute the whole
   reverse complement per k-mer" approach this project's own optimization
   pass replaced *earlier the same day*, verified by reading actual emitted
   assembly rather than assuming: the old form cost ~13 ALU operations per
   k-mer; the replacement rolls the reverse complement incrementally
   alongside the forward k-mer, ~4 operations per base, derived from the
   real `base_to_bits` encoding (`A/C/G/T -> 00/01/10/11`, so
   Watson-Crick pairing is exactly `b -> 0b11 - b`, which for a 2-bit value
   has no borrow and therefore equals `b ^ 0b11`). The proposal's own
   example code is not competitive with what this repository already ships.
2. **The proposal's "syncmers instead of minimizers" for the counting
   engine directly contradicts `docs/design-minimizer-counting.md`**,
   which already rejected syncmers for exactly this use case with a
   specific argument: a counter needs a total map from every k-mer to a
   bin; a syncmer rule is a selection predicate that declines to select
   most k-mers, so you cannot bucket what the rule refuses to select. A
   survey of eight real k-mer counters (KMC3, FastK, Gerbil, BCALM2,
   Bifrost, SSHash, Discount, Brisk) found none partitions by syncmers.
   Syncmers have a legitimate place in *sampling* for ML feature selection,
   where discarding most k-mers is the point — not in the exhaustive
   counting engine, which is where the proposal placed them.

## What "complete" means here instead

Complete within FastDNA's own territory, not across every subfield of
bioinformatics. The concrete plan already exists and reuses the actual
engine: a query layer over the sorted Parquet output, set operations
between k-mer tables, read filtering by k-mer content, per-read k-mer
profiles (`docs/superpowers/plans/2026-08-24-completeness-phase.md`). None
of that has a Gene Myers-caliber incumbent waiting — it is mapped territory
this project already owns. The ML layer is the other real differentiator:
no other k-mer library derives cross-validation groups from the samples'
own genetic distances the way `python/fastdna/cv.py` does.
