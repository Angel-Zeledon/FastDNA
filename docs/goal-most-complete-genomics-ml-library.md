# Goal: FastDNA becomes the most complete library for genomics ML

Date: 2026-08-27. Status: **active goal, supersedes the "stay narrow" conclusion
in `docs/philosophy-narrow-not-broad.md`** (that document is kept for its still-valid
technical corrections — the reverse-complement bit-trick derivation, and the
argument for why a *counting* engine needs a total map and syncmers don't
provide one — but its strategic conclusion no longer holds; see the notice at
its top).

## The decision

The target is no longer "the narrowest, sharpest tool in one corner of
k-mer counting." The target is: **the most complete library for genomics
machine learning that exists** — the one place a biotechnologist or ML
practitioner can go from raw FASTQ/FASTA to a trained, interpretable,
leakage-safe model without stitching together five different tools.

This is a scope expansion *within genomics ML*, not a decision to become a
general-purpose bioinformatics framework. The two are different axes:

- **Still explicitly out of scope**, and for the same reasons
  `philosophy-narrow-not-broad.md` gave (decade-plus incumbents, zero code
  reuse with FastDNA's k-mer engine): sequence alignment (`minimap2`'s
  territory), pangenome graphs (`vg`/`pggb`'s), general genomic interval
  algebra (`bedtools`'s). Building these would not make FastDNA more
  complete *for genomics ML* — it would make it a worse, later clone of
  tools with a decade's head start, unrelated to the ML mission.
- **Now explicitly in scope, aggressively**: every item in
  `docs/feature-gap-analysis.md` (engine-side completeness vs. KMC3/FastK/
  Jellyfish/Mash — Q1-Q5 and S1-S7) and every item in
  `docs/ml-genomics-roadmap.md` / `docs/ml-differentiation-roadmap.md`
  (the ML layer, waves 1-2 plus buckets A/B/C). "Complete" now means: close
  these lists, don't just cherry-pick the cheapest ones.

## Why this changes the execution posture

Previously (per `philosophy-narrow-not-broad.md` and the ranked lists in the
gap docs), the posture was: pick the highest value/effort items, ship them
opportunistically, stay comfortable being narrow. That posture is retired.
The new posture:

1. **Nothing in `feature-gap-analysis.md` or `ml-differentiation-roadmap.md`
   is "someday" anymore.** Every Q/S/A/B item is a real commitment, in
   priority order, not a menu.
2. **Publishability is a completeness blocker, not a nice-to-have.** A
   library nobody can `pip install` or `conda install` cannot be "the most
   complete library" in any sense that matters to an actual user — Q5
   (Bioconda) and the PyPI/crates.io publish step move up accordingly.
3. **Before adding new scope, audit what's already shipped.** This repo is
   under active multi-agent development and has repeatedly outpaced its own
   planning docs (see `docs/CHECKPOINT-2026-08-26.md`'s explicit warning:
   "the repo is far more advanced than this session's own history alone
   would suggest"). FASTA input and multi-file/stdin input (`feature-gap-
   analysis.md`'s Q1/Q2) are a confirmed example — already implemented
   (`src/fastq.rs`'s `Format::Fasta`, `MultiSourceReader`) by the time this
   goal was written, despite the gap doc still listing them as missing.
   **Verify against the actual code before executing any item on these
   lists — don't re-implement what already landed.**

## What "done" means for this goal

Not a single ship date — a standing bar. Re-check against it periodically:

- Every Q/S item in `feature-gap-analysis.md` is either shipped or has a
  documented, reasoned exception (like the "deliberately not copying" list
  already does for a few items).
- Every A/B item in `ml-differentiation-roadmap.md` is shipped, in its
  documented execution order, or explicitly deferred with a reason.
- The package is installable via `pip install fastdna` and
  `conda install -c bioconda fastdna` — not just buildable from source.
- The README's "Still ahead" section in the Roadmap is empty, or honestly
  small.

Related: [[philosophy-narrow-not-broad]] (superseded conclusion, valid
technical corrections), `docs/feature-gap-analysis.md`,
`docs/ml-genomics-roadmap.md`, `docs/ml-differentiation-roadmap.md`.
