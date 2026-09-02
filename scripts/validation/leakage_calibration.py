"""Calibrates `audit()` against known leakage: does what it detects track
what was injected?

## Why this, and not another real cohort

A detector is not demonstrated by one case where it fired. It is
demonstrated by showing that its output tracks the thing it claims to
measure, across the range -- which requires knowing the true answer, which
in turn requires constructing it.

`docs/validation-real-data.md` records three attempts to demonstrate leakage
on real cohorts. One was an artefact; two had no signal to audit. Even a
successful one would have been a single point: "on this cohort, the gap was
X", with no way to check whether X was the right answer, because on real
data nobody knows the right answer. That is precisely the gap this script
fills.

## The construction

Two sources of phenotype, deliberately separable, mixed by `lambda`:

  * **Biological.** A marker sequence inserted into a random half of the
    cohort, independent of lineage. A model that learns the marker
    generalises to lineages it has never seen, because the marker is there
    too.
  * **Lineage.** Membership of designated "resistant" lineages. A model can
    predict this from the k-mers that are simply characteristic of each
    clone -- which works under random CV and fails completely on a held-out
    lineage, because those k-mers are not there.

For each sample, the phenotype is drawn from the lineage source with
probability `lambda` and from the marker source otherwise. So:

    lambda = 0.0   phenotype is the marker; a model that learns it
                   generalises; the honest and the naive estimate should
                   agree, i.e. gap ~ 0
    lambda = 1.0   phenotype is lineage; nothing transferable predicts it;
                   random CV scores well by memorising clones and
                   lineage-blocked CV falls toward chance, i.e. gap large

Intermediate values interpolate. The prediction under test is therefore not
"the gap is positive" but **"the gap increases monotonically with lambda"**,
which is a far harder claim to satisfy by accident.

## Pre-registered before running

  1. Gap at `lambda=0` is near zero (the marker transfers).
  2. Gap at `lambda=1` is clearly positive.
  3. Gap increases monotonically in between; Spearman correlation between
     injected `lambda` and measured gap is high.

Failure of (3) with (1) and (2) holding would mean the gap is a detector of
the extremes but not a measure -- still useful, and a different claim from
the one this script tests. That outcome gets reported, not retried.

## Design sizing, done before generating anything

`fastdna.design.check_design` was used to pick the cohort shape rather than
discovering it afterwards, which is the lesson of the failed attempts:
n=200 over 20 lineages with 200 features gives p/n = 1.0, no design concerns,
and a 95% CI on an AUC of 0.75 of +/-0.068 -- narrow enough that the
differences this sweep looks for are resolvable. Replicates per lambda
average down what remains.

Usage:

    python scripts/validation/leakage_calibration.py
    python scripts/validation/leakage_calibration.py --replicates 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple


def _write_fastq(directory: Path, name: str, reads: List[str]) -> str:
    path = directory / name
    with path.open("w") as handle:
        for index, read in enumerate(reads):
            handle.write(f"@{name}_{index}\n{read}\n+\n{'I' * len(read)}\n")
    return str(path)


def build_cohort(
    directory: Path,
    n_lineages: int,
    per_lineage: int,
    lam: float,
    seed: int,
    read_len: int = 120,
):
    """A cohort whose phenotype is `lam`-determined by lineage and
    `(1-lam)`-determined by a transferable marker.

    Returns `(paths, phenotype, lineage_of)`.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))

    # One marker, fixed across the cohort, long enough to contribute many
    # distinct k-mers and so be learnable at k=21.
    marker = "".join(rng.choice(bases, size=200))

    # Which lineages are the "resistant" ones, for the lineage source.
    resistant_lineages = set(range(n_lineages // 2))

    paths, phenotype, lineage_of = [], [], []
    for lineage in range(n_lineages):
        root = "".join(rng.choice(bases, size=read_len * 6))
        for member in range(per_lineage):
            # Lineage members: sparse point mutations off a shared root, so
            # within-lineage distance stays small and between-lineage large.
            sequence = list(root)
            for position in rng.choice(len(sequence), size=max(1, len(sequence) // 200), replace=False):
                sequence[position] = str(rng.choice(bases))

            # The two candidate labels for this sample.
            label_from_lineage = int(lineage in resistant_lineages)
            label_from_marker = int(rng.random() < 0.5)

            # Which source decides this sample's phenotype.
            use_lineage = rng.random() < lam
            label = label_from_lineage if use_lineage else label_from_marker

            # The marker is present iff the MARKER source would have called
            # this sample positive -- independent of lineage, and of the
            # final label whenever the lineage source won the draw. That
            # independence is what makes the marker transferable at lam=0
            # and uninformative at lam=1.
            if label_from_marker:
                insert_at = int(rng.integers(0, len(sequence) - 1))
                sequence = sequence[:insert_at] + list(marker) + sequence[insert_at:]

            sequence = "".join(sequence)
            reads = [
                sequence[i : i + read_len]
                for i in range(0, len(sequence) - read_len, read_len // 2)
            ]
            paths.append(_write_fastq(directory, f"lam{lam:.2f}_s{seed}_L{lineage}_M{member}.fastq", reads))
            phenotype.append(label)
            lineage_of.append(lineage)

    return paths, np.asarray(phenotype), np.asarray(lineage_of)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n-lineages", type=int, default=20)
    parser.add_argument("--per-lineage", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=200)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import warnings

    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from fastdna.audit import audit
    from fastdna.design import check_design
    from fastdna.sklearn import KmerVectorizer

    warnings.filterwarnings("ignore")

    n_samples = args.n_lineages * args.per_lineage
    probe_y = np.array([i % 2 for i in range(n_samples)])
    probe_groups = np.array([i // args.per_lineage for i in range(n_samples)])
    design = check_design(
        probe_y, n_features=args.top_features, groups=probe_groups, n_splits=args.n_splits
    )
    print(f"design   : n={n_samples}, p/n={design.p_over_n:.3g}, "
          f"{design.n_groups} lineages, 95% CI on AUC=0.75 = "
          f"+/-{design.auc_ci_halfwidth:.4g}")
    for concern in design.concerns:
        print(f"  WARNING [{concern.code}] {concern.message}", file=sys.stderr)
    print(f"sweep    : lambda in {args.lambdas}, {args.replicates} replicates each\n")

    print(f"  {'lambda':>7}  {'random':>8}  {'blocked':>8}  {'gap':>8}  {'sd':>6}")
    rows = []
    for lam in args.lambdas:
        gaps, randoms, blockeds = [], [], []
        for replicate in range(args.replicates):
            with tempfile.TemporaryDirectory() as tmp:
                paths, y, groups = build_cohort(
                    Path(tmp), args.n_lineages, args.per_lineage, lam,
                    seed=1000 * replicate + int(lam * 100),
                )
                pipeline = Pipeline([
                    ("kmers", KmerVectorizer(k=args.k, top_features=args.top_features,
                                             representation="presence")),
                    ("clf", LogisticRegression(max_iter=2000, random_state=0)),
                ])
                report = audit(
                    pipeline, paths, y, groups=groups, scoring="roc_auc",
                    n_splits=args.n_splits, random_state=0,
                )
            gaps.append(report.gap)
            randoms.append(report.score_random)
            blockeds.append(report.score_lineage)
        rows.append({
            "lambda": lam,
            "gap_mean": float(np.nanmean(gaps)),
            "gap_sd": float(np.nanstd(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
            "score_random_mean": float(np.nanmean(randoms)),
            "score_lineage_mean": float(np.nanmean(blockeds)),
            "gaps": [float(g) for g in gaps],
        })
        print(f"  {lam:>7.2f}  {rows[-1]['score_random_mean']:>8.4f}  "
              f"{rows[-1]['score_lineage_mean']:>8.4f}  {rows[-1]['gap_mean']:>+8.4f}  "
              f"{rows[-1]['gap_sd']:>6.4f}")

    lams = np.array([r["lambda"] for r in rows])
    means = np.array([r["gap_mean"] for r in rows])
    rho = float(spearmanr(lams, means).statistic)
    print(f"\n  Spearman(lambda, gap) = {rho:.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps(
            {"rows": rows, "spearman": rho, "design": {
                "n_samples": n_samples, "p_over_n": design.p_over_n,
                "auc_ci_halfwidth": design.auc_ci_halfwidth}}, indent=2))
        print(f"  wrote {args.json}")

    # The three pre-registered predictions, checked explicitly.
    zero_gap, one_gap = means[0], means[-1]
    print("\n  pre-registered predictions:")
    print(f"    gap at lambda=0 near zero      : {zero_gap:+.4f}  "
          f"{'OK' if abs(zero_gap) < 0.10 else 'NOT MET'}")
    print(f"    gap at lambda=1 clearly positive: {one_gap:+.4f}  "
          f"{'OK' if one_gap > 0.10 else 'NOT MET'}")
    print(f"    monotone increase (rho >= 0.9) : {rho:.4f}    "
          f"{'OK' if rho >= 0.9 else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
