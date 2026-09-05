"""The negative control for `leakage_survey.py`: the same cohorts, labels shuffled.

## What the survey cannot decide on its own

The survey reports a mean gap -- random-CV AUC minus lineage-blocked AUC --
across published species-antibiotic cohorts. A positive mean is consistent
with the story the study is about (phenotype tracks lineage, so a random
split leaks relatives into the test fold). It is also consistent with a
story that has nothing to do with leakage: `python/fastdna/audit.py` says so
at length, under "A positive gap does not by itself mean population-structure
leakage". A model with enough capacity memorises individual samples; a
memorised sample's near-clones sit in its own lineage; blocking that lineage
out of the training fold therefore costs score *whether or not the phenotype
has anything to do with lineage*.

The survey's own running numbers show exactly the shape that worry predicts:
at 8 cohorts, the gap correlated +0.71 with `score_random`. That is either
"cohorts with real signal are also the confounded ones" (the leakage story)
or "cohorts where the model fits harder pay more for blocking" (the capacity
story). No amount of additional cohorts distinguishes them, because both
stories predict the same monotone relationship.

## What permuting the labels does distinguish

Shuffling `y` within a cohort destroys the phenotype-lineage association and
changes nothing else: the same 80 genomes, the same k-mers, the same
lineages, the same class balance, the same p/n, the same estimator with the
same capacity. Under permuted labels there is no leakage left to find --
but the model can still memorise, and `LineageKFold` still removes its
memorised near-clones from the training fold.

So the permuted gap **is** the capacity contribution, measured rather than
argued. Two possible outcomes, both informative:

* permuted mean gap ~ 0 -- capacity does not produce a gap on these cohorts,
  and the observed gap is what the study says it is.
* permuted mean gap comparable to the observed one -- the survey measured
  overfitting, and the headline claim does not survive. Better to find that
  here than in review.

The comparison is **paired**: every cohort contributes
`(observed_gap, permuted_gap)`, and the statistic is the mean of their
per-cohort differences with its standard error. Pairing matters because
cohorts differ enormously in how much signal and how many lineages they
have; comparing two unpaired means would drown the effect in that spread.

## Cost, and why it is affordable

Counting is not repeated: every cohort the survey measured left its counts in
`--cache-dir`, and this script reads them through the same `counts_for` the
survey uses (`CohortCounts.load`). What it pays is the audit itself -- two
5-fold cross-validations per replicate, measured at 110-600 s per cohort
depending on cohort size.

`--reps` replicates per cohort tighten the null estimate; the default of 1 is
already enough for the paired comparison above, since the null is averaged
across cohorts rather than within them.

## Sharing the cohort derivation with the survey

`cohort_labels` and `counts_for` are imported from `leakage_survey.py` rather
than reimplemented. The lineage grouping in particular has to be identical:
two scripts that disagreed about which genomes share a lineage would each
produce a perfectly plausible set of gaps that could not be compared to the
other's, and nothing in either output would say so.

## Usage

```bash
export PATH="$HOME/.cargo/bin:$PATH"
python scripts/validation/leakage_permutation_control.py \
    --json survey.json --out permutation_control.json
```

Resumable on the same terms as the survey: results are written after every
replicate, cohorts already recorded are skipped, and a replicate that
errored is retried on the next run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakage_survey import (  # noqa: E402
    COHORT_SIZE,
    DEFAULT_CACHE_DIR,
    TOP_FEATURES,
    cohort_labels,
    counts_for,
)


def permuted_gap(species: str, antibiotic: str, n_samples: int, top_features: int,
                 rep: int, cache_dir: Optional[Path]) -> Dict:
    """One replicate: the same cohort, `y` shuffled, audited identically."""
    import warnings

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from fastdna.audit import audit
    from fastdna.sklearn import KmerVectorizer

    warnings.filterwarnings("ignore")
    started = time.monotonic()

    cohort, y, groups, grouping = cohort_labels(species, antibiotic, n_samples)
    counts, was_cached = counts_for(cohort.paths, cache_dir, species, antibiotic,
                                    n_samples)

    # A plain shuffle: it preserves the class balance exactly (the same
    # labels, reordered), so the permuted cohort keeps the design the real
    # one had -- same n, same minority size, same p/n. Only the association
    # between phenotype and genome is broken, which is the one thing this
    # control exists to remove. Seeded per (cohort, replicate) so the run
    # reproduces and two replicates of one cohort are different draws.
    #
    # `crc32`, not `hash()`: Python salts `hash()` of a str per process
    # (PYTHONHASHSEED), so a seed derived from it would differ between runs
    # while every line of output claimed to be reproducible -- a wrong
    # answer in perfectly good form, which is the failure mode this whole
    # directory exists to catch.
    seed = zlib.crc32(f"{species}|{antibiotic}|{rep}".encode())
    y_permuted = np.random.default_rng(seed).permutation(y)

    pipeline = Pipeline([
        ("kmers", KmerVectorizer(counts=counts, top_features=top_features)),
        ("clf", LogisticRegression(max_iter=2000, random_state=0)),
    ])
    report = audit(pipeline, counts, y_permuted, groups=groups, scoring="roc_auc",
                   n_splits=5, random_state=0)

    return {
        "species": species,
        "antibiotic": antibiotic,
        "rep": rep,
        "grouping": grouping,
        "n_samples": int(report.n_samples),
        "n_lineages": int(report.n_lineages),
        "score_random": report.score_random,
        "score_lineage": report.score_lineage,
        "gap": report.gap,
        "gap_undefined_reason": report.gap_undefined_reason,
        # The sanity check on the permutation itself: shuffling the labels
        # must also destroy the phenotype-lineage association `confounding`
        # measures. A permuted cohort that still read as confounded would
        # mean the shuffle did not do what this script claims it does.
        "confounding": report.confounding.value,
        "counts_cached": was_cached,
        "seed": seed,
        "seconds": round(time.monotonic() - started, 1),
    }


def summarise(observed: Dict[tuple, float], rows: List[Dict]) -> None:
    import numpy as np

    usable = [r for r in rows if "error" not in r and r.get("gap") == r.get("gap")]
    if not usable:
        print("\nno usable replicates yet")
        return

    # Average the replicates of a cohort first, so a cohort with three
    # replicates does not outweigh one with a single replicate in the
    # paired comparison below.
    per_cohort: Dict[tuple, List[float]] = {}
    for r in usable:
        per_cohort.setdefault((r["species"], r["antibiotic"]), []).append(r["gap"])

    paired = [(observed[key], float(np.mean(gaps)))
              for key, gaps in per_cohort.items() if key in observed]
    if not paired:
        print("\nno cohort has both an observed and a permuted gap yet")
        return

    obs = np.array([p[0] for p in paired])
    perm = np.array([p[1] for p in paired])
    diff = obs - perm

    def se(values):
        """`" (SE x)"`, or nothing at all with a single cohort -- a standard
        error over one number is not a small standard error, it is an
        undefined one, and printing 0.0000 there would read as certainty."""
        return (f"  (SE {values.std(ddof=1) / np.sqrt(len(values)):.4f})"
                if len(values) > 1 else "")

    print(f"\n{'=' * 72}")
    print(f"  {len(paired)} cohorts with both gaps "
          f"({len(usable)} permuted replicates)")
    print(f"  mean observed gap          {obs.mean():+.4f}{se(obs)}")
    print(f"  mean permuted gap          {perm.mean():+.4f}{se(perm)}")
    print(f"  mean paired difference     {diff.mean():+.4f}{se(diff)}")
    print(f"  cohorts where observed > permuted   {(diff > 0).sum()} of {len(diff)}")
    print("=" * 72)
    print("  Read it as: the permuted column is what capacity alone buys.")
    print("  A paired difference indistinguishable from zero means the survey")
    print("  measured overfitting, not leakage.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=Path("survey.json"),
                        help="the survey's own output; supplies the cohorts and "
                             "the observed gaps to compare against")
    parser.add_argument("--out", type=Path, default=Path("permutation_control.json"))
    parser.add_argument("--reps", type=int, default=1,
                        help="permutations per cohort")
    parser.add_argument("--n-samples", type=int, default=COHORT_SIZE)
    parser.add_argument("--top-features", type=int, default=TOP_FEATURES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    if not args.json.is_file():
        print(f"{args.json} does not exist -- run leakage_survey.py first", file=sys.stderr)
        return 1

    survey = json.loads(args.json.read_text())
    observed = {(r["species"], r["antibiotic"]): r["gap"] for r in survey
                if "error" not in r and "skipped" not in r
                and r.get("gap") == r.get("gap")}
    print(f"{len(observed)} measured cohorts in {args.json}")

    rows: List[Dict] = []
    if args.out.is_file():
        recorded = json.loads(args.out.read_text())
        rows = [r for r in recorded if "error" not in r]
        retrying = len(recorded) - len(rows)
        print(f"resuming: {len(rows)} replicates already run"
              + (f"; retrying {retrying} that errored" if retrying else ""))
    done = {(r["species"], r["antibiotic"], r["rep"]) for r in rows}

    for (species, antibiotic), observed_gap in observed.items():
        for rep in range(args.reps):
            if (species, antibiotic, rep) in done:
                continue
            print(f"\n[{len(rows) + 1}] {species} / {antibiotic}  rep {rep}"
                  f"  (observed gap {observed_gap:+.4f})", flush=True)
            try:
                row = permuted_gap(species, antibiotic, args.n_samples,
                                   args.top_features, rep, args.cache_dir)
            except Exception as exc:  # one bad cohort must not end the run
                row = {"species": species, "antibiotic": antibiotic, "rep": rep,
                       "error": str(exc)[:300]}
                print(f"    ERROR: {row['error'][:110]}", flush=True)
            else:
                gap_label = ("withheld" if row["gap"] != row["gap"]
                             else f"{row['gap']:+.4f}")
                print(f"    permuted: random {row['score_random']:.4f} -> blocked "
                      f"{row['score_lineage']:.4f}   gap {gap_label}   "
                      f"confounding {row['confounding']:.3f}   ({row['seconds']:.0f}s)",
                      flush=True)

            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2))

    summarise(observed, rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
