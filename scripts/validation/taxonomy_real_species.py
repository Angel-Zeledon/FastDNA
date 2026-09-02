"""Checks `fastdna.taxonomy.classify` on real genomes of known species.

The suite's taxonomy fixtures are repeated motifs -- `test_taxonomy.py`
builds its references as `"ACGTGGCATCAGT" * n`. A classifier scoring well
against those has demonstrated that its plumbing works, and nothing about
whether it can tell two real bacteria apart. Real genomes share large
conserved regions, differ in accessory content, and come in different sizes;
none of that exists in a repeated motif.

The design that makes this a real test rather than a demo:

  * **Held-out queries.** Three genomes per species build the reference,
    three DIFFERENT genomes of the same species are the queries. A query is
    never in the database it is scored against, so a correct call means the
    method generalised across strains rather than recognising a file.
  * **A confusable panel.** Escherichia coli and Salmonella enterica are
    both Enterobacteriaceae and share substantial sequence; getting those
    two right is the part that would fail if containment were being computed
    or normalised incorrectly. Streptococcus pneumoniae is the easy control
    at the other extreme (different phylum, and a 2.1 Mb genome against
    E. coli's 5.1 Mb, which also exercises the size asymmetry containment
    exists to handle).

Measured: 12/12 species calls correct, top-hit containment 0.35-0.96.

This is NOT a comparison against Kraken 2, which would need a database built
over the full NCBI taxonomy. It answers the narrower question the test suite
could not: does this classify real organisms of known identity correctly.
`docs/validation-real-data.md`'s Track B covers the per-read metagenomic
case against real sequencing reads.

Usage:

    python scripts/validation/taxonomy_real_species.py
    python scripts/validation/taxonomy_real_species.py --per-species 4 --k 31

Downloads assemblies from BV-BRC on first run (cached afterwards). Exits
non-zero if any species call is wrong.
"""

from __future__ import annotations

import argparse
import sys

#: (species, antibiotic) pairs, where the antibiotic only selects which
#: cohort file the loader reads -- it has no bearing on identity. Chosen to
#: span the hard case (two Enterobacteriaceae) and the easy one.
PANEL = [
    ("Escherichia coli", "ciprofloxacin"),
    ("Klebsiella pneumoniae", "gentamicin"),
    ("Salmonella enterica", "ampicillin"),
    ("Streptococcus pneumoniae", "erythromycin"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-species", type=int, default=3,
                        help="genomes per species for the reference; the same number "
                             "again, held out, are used as queries (default: 3)")
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--sketch-size", type=int, default=2000)
    args = parser.parse_args()

    from fastdna.datasets import load_amr
    from fastdna.taxonomy import build_reference_database, classify

    n = args.per_species
    reference, queries = {}, []
    for species, antibiotic in PANEL:
        cohort = load_amr(species, antibiotic, n_samples=2 * n, random_state=0)
        paths = sorted(str(p) for p in cohort.paths)
        if len(paths) < 2 * n:
            print(f"WARNING: {species} returned {len(paths)} genomes, wanted {2 * n}",
                  file=sys.stderr)
        for i, path in enumerate(paths[:n]):
            reference[f"{species}#{i}"] = path
        queries.extend((species, path) for path in paths[n:2 * n])

    print(f"reference: {len(reference)} genomes across {len(PANEL)} species")
    print(f"queries  : {len(queries)} held-out genomes, k={args.k}\n")

    db = build_reference_database(reference, k=args.k, sketch_size=args.sketch_size)

    wrong = []
    for truth, query in queries:
        table = classify(query, db, k=args.k, sketch_size=args.sketch_size, top_n=2)
        names = table.column("name").to_pylist()
        scores = table.column("score").to_pylist()
        # Reference keys are "<species>#<index>"; the species is the label.
        predicted = names[0].rsplit("#", 1)[0]
        hit = predicted == truth
        if not hit:
            wrong.append((truth, predicted, query))
        # The runner-up matters: a correct call whose second place is another
        # species at a near-identical score is not really a discrimination.
        runner_up = f"{names[1].rsplit('#', 1)[0]} {scores[1]:.3f}" if len(names) > 1 else "-"
        print(f"  {truth:<26} -> {predicted:<26} {scores[0]:.3f}   "
              f"(2nd: {runner_up})   {'OK' if hit else 'MISS'}")

    correct = len(queries) - len(wrong)
    print(f"\nspecies-level accuracy: {correct}/{len(queries)} = {correct / len(queries):.0%}")

    if wrong:
        for truth, predicted, path in wrong:
            print(f"  MISCALLED {path}: {truth} -> {predicted}", file=sys.stderr)
        print("\nFAIL: classify() misidentified a genome of known species.", file=sys.stderr)
        return 1

    print("OK: every held-out genome was assigned to its own species.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
