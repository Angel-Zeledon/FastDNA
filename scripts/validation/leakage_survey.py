"""The study: how much does lineage leakage inflate AMR prediction, measured
across many published species-antibiotic cohorts.

## What this produces

One row per species-antibiotic combination: the random-CV AUC, the
lineage-blocked AUC, the gap between them, and the cohort's confounding.
The headline is the aggregate -- "across N cohorts the mean AUC falls from X
to Y" -- not any single row.

## Why many small cohorts rather than one large one

This is the design decision that makes the study run on a laptop, and it is
statistical rather than pragmatic. A single cohort of 80 genomes gives a
95% CI on its AUC of +/-0.107, which is too wide to say much. The *mean* of
30 such cohorts has a CI of +/-0.107/sqrt(30) = +/-0.020 -- tighter than one
cohort of 300 would give, and 300 does not fit in 18 GB with complete
bacterial genomes (measured: it drives this machine into swap).

Aggregation buys the precision that memory cannot. It also makes the claim
stronger: a result that holds across 30 species-antibiotic pairs is evidence
about the field, while one cohort is an anecdote.

## Cost, measured rather than estimated

Roughly 5-8 minutes per cohort at n=80 (download is cached after the first
cohort of each species, since combinations share genomes). Thirty cohorts is
about four hours -- one overnight run.

The dominant cost is that every fold re-counts its genomes. `CohortCounts`
has no `save()`/`load()`, so the counting work cannot be reused across the
two splitters and five folds. Implementing that (the G-6 gap in
`docs/audit/ml-gaps.md`) would cut this roughly tenfold and is the single
best speed-up available.

## Resumability

Results are appended to the output JSON after every cohort, and cohorts
already present are skipped on a re-run. A four-hour job must survive a
closed laptop lid, an OOM kill, or a Ctrl-C.

## Before it runs anything

Every candidate is passed through `fastdna.design.check_design` first, and
those whose design cannot resolve the effect are excluded with their reason
recorded. Deciding that afterwards is how this project previously lost an
hour to two unrunnable experiments.

Usage:

    python scripts/validation/leakage_survey.py --list          # candidates only
    python scripts/validation/leakage_survey.py --max-cohorts 30 --json survey.json
    python scripts/validation/leakage_survey.py --json survey.json   # resumes
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

#: Genomes per cohort. Chosen because it fits comfortably (measured: 150
#: pushes this machine into swap with complete genomes) and because the
#: aggregate precision comes from the number of cohorts, not their size.
COHORT_SIZE = 80

#: Kept modest for the same reason: p/n = 500/80 = 6.25 already trips the
#: design check's ratio threshold, and a larger vocabulary buys nothing when
#: the phenotype is carried by an accessory gene.
TOP_FEATURES = 500

#: Minimum samples of the smaller class in the source metadata before a
#: combination is worth downloading at all.
MIN_MINORITY = 25


def candidate_cohorts() -> List[Dict]:
    """Every species-antibiotic pair with enough labelled samples, read from
    metadata alone -- no genome is downloaded to build this list."""
    from fastdna.datasets import _AMR_METADATA_BASE_URL, _AMR_SPECIES_FILES

    candidates = []
    for species, filename in _AMR_SPECIES_FILES.items():
        try:
            raw = urllib.request.urlopen(
                _AMR_METADATA_BASE_URL + filename, timeout=180
            ).read().decode("utf-8", "replace")
        except Exception as exc:  # a species' metadata being down is not fatal
            print(f"  WARNING: {species} metadata unavailable ({exc})", file=sys.stderr)
            continue

        counts = defaultdict(lambda: [0, 0])
        for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
            antibiotic = (row.get("antibiotic") or "").strip()
            phenotype = (row.get("resistant_phenotype") or "").strip().lower()
            if antibiotic and phenotype in ("resistant", "susceptible"):
                counts[antibiotic][0 if phenotype == "resistant" else 1] += 1

        for antibiotic, (n_resistant, n_susceptible) in counts.items():
            if min(n_resistant, n_susceptible) < MIN_MINORITY:
                continue
            candidates.append({
                "species": species,
                "antibiotic": antibiotic,
                "n_resistant": n_resistant,
                "n_susceptible": n_susceptible,
                "minority_fraction": min(n_resistant, n_susceptible)
                / (n_resistant + n_susceptible),
            })

    # Balanced cohorts first: they carry the most information per genome
    # downloaded, and a 50/50 split keeps ROC-AUC interpretable.
    candidates.sort(key=lambda c: -c["minority_fraction"])
    return candidates


def run_one(species: str, antibiotic: str, n_samples: int, top_features: int) -> Dict:
    """Download, count, audit. Returns a row, or a row carrying `error`."""
    import warnings

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    import fastdna
    from fastdna.audit import audit
    from fastdna.datasets import load_amr
    from fastdna.design import check_design
    from fastdna.sklearn import KmerVectorizer

    warnings.filterwarnings("ignore")
    started = time.monotonic()

    cohort = load_amr(species, antibiotic, n_samples=n_samples, random_state=0)
    y = np.asarray(cohort.phenotype).astype(int)

    # MLST as ground-truth lineage where the loader supplies it: it removes
    # the threshold question entirely. Sketch-derived grouping is the
    # fallback, and is what a user without ST would rely on.
    sequence_types = [str(s) for s in cohort.sequence_type]
    usable_st = sum(1 for s in sequence_types if s not in ("", "nan", "None"))
    if usable_st == len(sequence_types):
        levels = {st: i for i, st in enumerate(sorted(set(sequence_types)))}
        groups = np.array([levels[s] for s in sequence_types])
        grouping = "mlst"
    else:
        groups, grouping = None, "sketch"

    design = check_design(
        y, n_features=top_features,
        groups=groups if groups is not None else np.arange(len(y)),
        n_splits=5,
    )
    blocking = [c.code for c in design.concerns
                if c.code in {"fewer_groups_than_folds", "auc_ci_wider_than_effect",
                              "tiny_minority_class"}]
    if blocking:
        return {"species": species, "antibiotic": antibiotic, "skipped": blocking,
                "n_samples": len(y)}

    counts = fastdna.count_cohort(cohort.paths, k=31)
    pipeline = Pipeline([
        ("kmers", KmerVectorizer(counts=counts, top_features=top_features)),
        ("clf", LogisticRegression(max_iter=2000, random_state=0)),
    ])
    report = audit(pipeline, counts, y, groups=groups, scoring="roc_auc",
                   n_splits=5, random_state=0)

    return {
        "species": species,
        "antibiotic": antibiotic,
        "n_samples": int(report.n_samples),
        "n_resistant": int(y.sum()),
        "grouping": grouping,
        "n_lineages": int(report.n_lineages),
        "score_random": report.score_random,
        "score_lineage": report.score_lineage,
        "gap": report.gap,
        "gap_undefined_reason": report.gap_undefined_reason,
        "confounding": report.confounding.value,
        "confounding_statistic": report.confounding.statistic,
        "design_ci_halfwidth": design.auc_ci_halfwidth,
        "design_concerns": [c.code for c in design.concerns],
        "seconds": round(time.monotonic() - started, 1),
    }


def summarise(rows: List[Dict]) -> None:
    import numpy as np

    usable = [r for r in rows if "error" not in r and "skipped" not in r
              and r.get("gap") is not None and r["gap"] == r["gap"]]
    if not usable:
        print("\nno usable rows yet")
        return

    random_scores = np.array([r["score_random"] for r in usable])
    blocked = np.array([r["score_lineage"] for r in usable])
    gaps = np.array([r["gap"] for r in usable])
    confounding = np.array([r["confounding"] for r in usable])

    print(f"\n{'=' * 72}")
    print(f"  {len(usable)} cohorts")
    print(f"  mean random-CV AUC        {random_scores.mean():.4f}")
    print(f"  mean lineage-blocked AUC  {blocked.mean():.4f}")
    print(f"  mean gap                  {gaps.mean():+.4f}  "
          f"(SE {gaps.std(ddof=1) / np.sqrt(len(gaps)):.4f})")
    print(f"  mean confounding          {confounding.mean():.4f}")
    print(f"  cohorts losing >0.05 AUC  {(gaps > 0.05).sum()} of {len(usable)}")
    print(f"  cohorts falling to chance {(blocked < 0.55).sum()} of {len(usable)}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=Path("leakage_survey.json"))
    parser.add_argument("--max-cohorts", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=COHORT_SIZE)
    parser.add_argument("--top-features", type=int, default=TOP_FEATURES)
    parser.add_argument("--list", action="store_true", help="show candidates and exit")
    args = parser.parse_args()

    candidates = candidate_cohorts()
    print(f"{len(candidates)} candidate cohorts with >= {MIN_MINORITY} in the smaller class")

    if args.list:
        for c in candidates[:60]:
            print(f"  {c['species']:<26} {c['antibiotic']:<30} "
                  f"R={c['n_resistant']:<6} S={c['n_susceptible']:<6} "
                  f"minority={c['minority_fraction']:.0%}")
        return 0

    # Resume: anything already recorded is not re-run.
    rows: List[Dict] = []
    if args.json.is_file():
        rows = json.loads(args.json.read_text())
        print(f"resuming: {len(rows)} cohorts already in {args.json}")
    done = {(r["species"], r["antibiotic"]) for r in rows}

    for candidate in candidates:
        if len([r for r in rows if "skipped" not in r]) >= args.max_cohorts:
            break
        key = (candidate["species"], candidate["antibiotic"])
        if key in done:
            continue

        print(f"\n[{len(rows) + 1}] {candidate['species']} / {candidate['antibiotic']}",
              flush=True)
        try:
            row = run_one(candidate["species"], candidate["antibiotic"],
                          args.n_samples, args.top_features)
        except Exception as exc:  # one bad cohort must not end the night's run
            row = {"species": key[0], "antibiotic": key[1], "error": str(exc)[:300]}

        if "skipped" in row:
            print(f"    skipped: {row['skipped']}", flush=True)
        elif "error" in row:
            print(f"    ERROR: {row['error'][:110]}", flush=True)
        else:
            print(f"    random {row['score_random']:.4f} -> blocked "
                  f"{row['score_lineage']:.4f}   gap {row['gap']:+.4f}   "
                  f"confounding {row['confounding']:.3f}   ({row['seconds']:.0f}s)",
                  flush=True)

        rows.append(row)
        # Written after every cohort, not at the end: this job is long enough
        # that it will be interrupted at least once.
        args.json.write_text(json.dumps(rows, indent=2))

    summarise(rows)
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
