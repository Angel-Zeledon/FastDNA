"""Measures how much a k-mer AMR classifier's score is inflated by
population structure, on a real cohort, using this project's own `audit()`.

WHAT THIS IS, AND WHAT IT IS NOT.

It is **not** a discovery. That the field's published AMR-prediction numbers
are inflated by clonal population structure was already measured at scale --
PLOS Biology 2025, 24,000+ genomes -- and arXiv 2502.07749 already named
phylogeny-aware cross-validation as the standard tool the field is missing.
Both are cited by `python/fastdna/cv.py` and `audit.py`.

It **is** the demonstration that FastDNA is that missing tool, end to end:
one command that goes from a public accession list to a quantified answer
for a specific model and cohort, with no manual phylogeny step. The claim
being evidenced is a software claim -- "this exists, it is installable, and
it works on real data" -- not a biological one.

WHY THIS COHORT.

`docs/validation-real-data.md` ran the same audit on *Staphylococcus aureus*
+ oxacillin and found **no gap**: lineage-blocked and random CV both scored
100%. That was the correct result and a poor demonstration, for a reason
that document states plainly -- oxacillin resistance is near-monogenic
(carriage of *mecA*), and a signal that strong swamps any lineage
confounding. To show that `LineageKFold` changes an answer, the phenotype
has to be one where population structure can plausibly stand in for the
phenotype:

  * *Escherichia coli* + **ciprofloxacin** is polygenic -- fluoroquinolone
    resistance accumulates through stepwise chromosomal mutations in `gyrA`
    and `parC`, plus efflux-pump upregulation and plasmid-borne `qnr`. No
    single gene carries it the way `mecA` carries oxacillin.
  * It is also strongly clone-associated: ST131 is a pandemic lineage that
    is disproportionately fluoroquinolone-resistant. A model that learns
    "this genome is ST131" scores well without learning any resistance
    biology at all -- exactly the failure `audit()` exists to expose.

HONEST PRE-REGISTRATION. A gap near zero is a publishable outcome of this
script and must be reported as such, not retried with a different antibiotic
until the number looks good. If the gap is small here, the honest reading is
that this cohort's signal is also strong enough to survive lineage blocking
-- which is information, not failure. The one thing that would be dishonest
is shopping for a cohort.

Usage:

    python scripts/validation/lineage_leakage_experiment.py
    python scripts/validation/lineage_leakage_experiment.py --n-samples 300 --antibiotic gentamicin

First run downloads assemblies from BV-BRC into `~/.fastdna/datasets`
(roughly 5 MB per genome); later runs reuse that cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--species", default="Escherichia coli")
    parser.add_argument("--antibiotic", default="ciprofloxacin")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--k", type=int, default=31, help="k for the feature matrix")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-features", type=int, default=10_000)
    # Exposed because it is the parameter most likely to make this
    # experiment say nothing. At `audit()`'s default of 0.01 Mash distance,
    # a diverse E. coli sample can put every genome in its own lineage --
    # and `LineageKFold` over N singleton groups is just K-fold, so the gap
    # collapses to zero for a reason that has nothing to do with leakage.
    # The automatic leakage curve sweeps thresholds precisely so this is
    # visible rather than silent; this flag is for pinning one afterwards.
    parser.add_argument("--lineage-threshold", type=float, default=0.01)
    # `sketch` (default) derives lineages the way a user with nothing but
    # FASTQ files must: MinHash distance under a threshold. `mlst` instead
    # uses the cohort's own `sequence_type` -- the field's established
    # typing standard, published per genome by BV-BRC -- as ground-truth
    # groups. The second is the stronger experiment (no threshold to argue
    # about) but only possible because this particular loader carries ST;
    # a user's own cohort usually will not, which is exactly why the
    # sketch path is the default rather than a fallback.
    parser.add_argument("--groups", choices=("sketch", "mlst"), default="sketch")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    # Imported inside main so `--help` works without the scientific stack.
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    import fastdna
    from fastdna.audit import audit
    from fastdna.datasets import load_amr
    from fastdna.sklearn import KmerVectorizer

    print(f"cohort   : {args.species} / {args.antibiotic}, n={args.n_samples}")
    started = time.monotonic()
    cohort = load_amr(
        args.species,
        args.antibiotic,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )
    y = np.asarray(cohort.phenotype)
    print(f"           {len(cohort.paths)} genomes, "
          f"{int(y.sum())} resistant / {int((1 - y).sum())} susceptible "
          f"({time.monotonic() - started:.0f}s)")

    # Counted once here rather than once per fold. Without this, every fold
    # of every splitter re-reads every genome -- the exact cost
    # `CohortCounts` exists to remove.
    started = time.monotonic()
    counts = fastdna.count_cohort(cohort.paths, k=args.k)
    print(f"counting : k={args.k} over {len(cohort.paths)} genomes "
          f"({time.monotonic() - started:.0f}s)")

    pipeline = Pipeline([
        ("kmers", KmerVectorizer(counts=counts, top_features=args.top_features)),
        # Plain L2 logistic regression on k-mer presence: the field's
        # workhorse, deliberately not tuned. A tuned model would confound
        # "the gap" with "the tuning", and the gap is the measurement.
        ("clf", LogisticRegression(max_iter=2000, random_state=args.random_state)),
    ])

    groups = None
    if args.groups == "mlst":
        raw_st = [str(s).strip() for s in cohort.sequence_type]
        if any(s in ("", "nan", "None") for s in raw_st):
            print("WARNING: some genomes carry no sequence_type; they each become "
                  "their own group, which weakens the blocking.", file=sys.stderr)
        levels = {st: i for i, st in enumerate(sorted(set(raw_st)))}
        groups = np.array([levels[s] for s in raw_st])
        print(f"lineages : MLST ground truth -- {len(levels)} distinct sequence types")

        # Independent check of `cv.lineage_groups`: does sketch-derived
        # clustering recover the same partition MLST does? This is the
        # comparison against known lineages that the test suite never makes
        # (its cohorts are synthetic), and it is free here because both
        # labellings exist for the same 200 genomes.
        try:
            from sklearn.metrics import adjusted_rand_score
            from fastdna.cv import lineage_groups

            inferred = lineage_groups(
                counts, sketch_size=1000, distance_threshold=args.lineage_threshold
            )
            ari = adjusted_rand_score(groups, np.asarray(inferred))
            print(f"           sketch-vs-MLST agreement: ARI = {ari:.3f} "
                  f"({len(set(np.asarray(inferred).tolist()))} inferred groups)")
        except Exception as exc:  # never let the side-check kill the run
            print(f"           (sketch-vs-MLST check skipped: {exc})")

    print(f"auditing : {args.n_splits}-fold, scoring=roc_auc, groups={args.groups}")
    started = time.monotonic()
    report = audit(
        pipeline,
        counts,
        y,
        groups=groups,
        scoring="roc_auc",
        n_splits=args.n_splits,
        lineage_threshold=args.lineage_threshold,
        random_state=args.random_state,
    )
    print(f"           done in {time.monotonic() - started:.0f}s")

    print("\n" + "=" * 72)
    print(f"  random CV        AUC = {report.score_random:.4f} ± {report.score_random_std:.4f}")
    print(f"  lineage-blocked  AUC = {report.score_lineage:.4f} ± {report.score_lineage_std:.4f}")
    print(f"  GAP                  = {report.gap:+.4f}")
    print(f"  lineages found       = {report.n_lineages} (threshold {report.lineage_threshold})")
    print("=" * 72)

    if report.leakage_curve:
        print("\nleakage curve (how the score moves as lineage blocking tightens):")
        print("  threshold  lineages  blocked AUC     gap")
        for point in report.leakage_curve:
            # A point whose n_lineages approaches n_samples is degenerate:
            # every sample is its own group, so LineageKFold degrades to
            # plain K-fold and its gap is uninformative rather than small.
            flag = " <- degenerate" if point.n_lineages >= 0.9 * report.n_samples else ""
            print(f"  {point.threshold:<10.4f} {point.n_lineages:<9} "
                  f"{point.score_lineage:.4f}         {point.gap:+.4f}{flag}")

    if args.json is not None:
        args.json.write_text(json.dumps({
            "species": args.species,
            "antibiotic": args.antibiotic,
            "n_samples": int(report.n_samples),
            "n_resistant": int(y.sum()),
            "k": args.k,
            "scoring": report.scoring,
            "score_random": report.score_random,
            "score_random_std": report.score_random_std,
            "score_lineage": report.score_lineage,
            "score_lineage_std": report.score_lineage_std,
            "gap": report.gap,
            "n_lineages": int(report.n_lineages),
            "lineage_threshold": report.lineage_threshold,
            "leakage_curve": [
                {
                    "threshold": p.threshold,
                    "n_lineages": int(p.n_lineages),
                    "score_lineage": p.score_lineage,
                    "gap": p.gap,
                }
                for p in (report.leakage_curve or ())
            ],
        }, indent=2))
        print(f"\nwrote {args.json}")

    # Deliberately always 0 on a completed run: a small gap is a real
    # result, not a failure. Only an exception (a broken pipeline, an
    # unreachable cohort) should make this script non-zero.
    return 0


if __name__ == "__main__":
    sys.exit(main())
