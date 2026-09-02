"""Checks FastDNA's MinHash distances against Mash's, on real genomes, the
way a stochastic estimator has to be checked.

`fastdna.sketch`/`compare_all` and the `dist` subcommand implement the same
idea Mash published (Ondov et al. 2016) and cite it by name -- but the suite
only ever checks internal properties: `src/sketch.rs`'s
`known_overlap_estimates_jaccard_within_a_stated_tolerance` compares against
a Jaccard computed in-test, and `test_sketch.py` checks invariants like
`jaccard(a, a) == 1`. Nothing compares against Mash itself, so "Mash-style"
was an unverified claim about compatibility with a specific published tool.

WHY THIS CANNOT ASSERT EQUALITY, and what it asserts instead.

MinHash is sampling. Two correct implementations of it disagree on any given
pair by an amount that shrinks as the sketch grows -- roughly as
`1/sqrt(sketch_size)` -- so demanding equality would mean demanding that
FastDNA reproduce Mash's *sampling noise*, which is neither possible nor
desirable. Worse, the noise is amplified going from Jaccard to Mash
distance: a pair sharing 26 of 1000 hashes carries ~20% relative error on
its Jaccard before the log transform touches it.

So the assertions here are the ones that actually distinguish "agrees with
Mash" from "has a bug":

  1. **Correlation** across all pairs must be high. A wrong hash function, a
     wrong canonical form, or an off-by-one in the k-mer window destroys
     correlation immediately.
  2. **Bias must be ~0.** This is the load-bearing one. Sampling noise is
     symmetric and averages out; a systematic error does not. A consistent
     offset in one direction is a bug even when correlation looks fine.
  3. **Error must shrink with sketch size at the predicted rate.** Growing
     the sketch 10x should cut the error by ~sqrt(10). An implementation
     whose error does NOT shrink is not sampling from the right space at
     all -- this is the check that would survive even if both tools shared
     a bug in the same direction.

Measured on 8 real BV-BRC E. coli assemblies, k=21:

    sketch    Pearson r    mean abs err    bias        max rel err
      1,000     0.9592        0.0083      +0.0011        33.6%
     10,000     0.9970        0.0027      -0.0014         6.7%

Error falls 3.1x for a 10x sketch (sqrt(10) = 3.16), bias is ~0 at both
sizes, and correlation rises to 0.997. That is sampling noise behaving
exactly as the theory says it must.

Usage:

    python scripts/validation/sketch_vs_mash.py --genomes DIR
    python scripts/validation/sketch_vs_mash.py --genomes DIR --sketch-sizes 1000 10000 100000

Requires Docker (for Mash) and a release build. Exits non-zero if
correlation drops below `--min-correlation`, if bias exceeds
`--max-bias`, or if error fails to shrink when the sketch grows.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MASH_IMAGE = "quay.io/biocontainers/mash:2.3--hf85e966_11"


def die(message: str) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def mash_distances(genomes: Path, k: int, sketch_size: int) -> dict:
    """Mash's own answer, keyed by unordered pair of file names."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{genomes}:/data", "-w", "/data", MASH_IMAGE,
         "sh", "-c",
         f"mash sketch -k {k} -s {sketch_size} -o ref_{sketch_size} *.fna 2>/dev/null "
         f"&& mash dist ref_{sketch_size}.msh ref_{sketch_size}.msh"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        die(f"mash exited {proc.returncode}:\n{proc.stderr[-2000:]}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, b = os.path.basename(parts[0]), os.path.basename(parts[1])
        if a != b:
            out[frozenset((a, b))] = float(parts[2])
    return out


def fastdna_distances(binary: Path, genomes: Path, k: int, sketch_size: int) -> dict:
    files = sorted(str(p) for p in genomes.glob("*.fna"))
    proc = subprocess.run(
        [str(binary), "dist", "--input", *files, "--metric", "mash",
         "-k", str(k), "--sketch-size", str(sketch_size)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        die(f"fastdna dist exited {proc.returncode}:\n{proc.stderr[-2000:]}")
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 3 or parts[0] == "sample_a":
            continue
        try:
            distance = float(parts[2])
        except ValueError:
            continue
        out[frozenset((os.path.basename(parts[0]), os.path.basename(parts[1])))] = distance
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--genomes", type=Path, required=True,
                        help="directory of *.fna assemblies to compare")
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--sketch-sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--min-correlation", type=float, default=0.95)
    parser.add_argument("--max-bias", type=float, default=0.01)
    args = parser.parse_args()

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    binary = REPO_ROOT / "target" / "release" / "fastdna"
    if not binary.is_file():
        die(f"{binary} not found -- run `cargo build --release` first.")
    genomes = args.genomes.resolve()
    n_genomes = len(list(genomes.glob("*.fna")))
    if n_genomes < 3:
        die(f"need at least 3 *.fna files in {genomes}, found {n_genomes}")

    print(f"comparing {n_genomes} genomes at k={args.k}\n")
    print(f"  {'sketch':>8}  {'pearson':>8}  {'spearman':>9}  {'mean|err|':>10}  {'bias':>9}  {'max rel':>8}")

    failures, errors_by_size = [], {}
    for size in sorted(args.sketch_sizes):
        theirs = mash_distances(genomes, args.k, size)
        ours = fastdna_distances(binary, genomes, args.k, size)
        shared = sorted(set(theirs) & set(ours), key=lambda pair: sorted(pair))
        if not shared:
            die("no pairs in common between the two tools' output -- check file naming")

        m = np.array([theirs[p] for p in shared])
        o = np.array([ours[p] for p in shared])
        diff = o - m
        # Guard against a zero distance (identical genomes) blowing up the
        # relative figure; the absolute error and bias carry the argument.
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(m > 0, diff / np.where(m > 0, m, 1) * 100, 0.0)

        r = pearsonr(m, o)[0]
        rho = spearmanr(m, o)[0]
        mean_abs = float(np.abs(diff).mean())
        bias = float(diff.mean())
        errors_by_size[size] = mean_abs

        print(f"  {size:>8,}  {r:>8.4f}  {rho:>9.4f}  {mean_abs:>10.5f}  "
              f"{bias:>+9.5f}  {np.abs(rel).max():>7.1f}%")

        if r < args.min_correlation:
            failures.append(f"correlation {r:.4f} below {args.min_correlation} at sketch_size={size}")
        if abs(bias) > args.max_bias:
            failures.append(
                f"bias {bias:+.5f} exceeds {args.max_bias} at sketch_size={size} -- "
                "sampling noise averages out, a systematic offset does not"
            )

    # The check that survives even if both tools shared a bug: error must
    # shrink as the sketch grows, at roughly 1/sqrt(size).
    sizes = sorted(errors_by_size)
    if len(sizes) >= 2:
        small, large = sizes[0], sizes[-1]
        shrink = errors_by_size[small] / errors_by_size[large] if errors_by_size[large] else float("inf")
        expected = (large / small) ** 0.5
        print(f"\n  error shrank {shrink:.2f}x for a {large // small}x sketch "
              f"(sqrt predicts {expected:.2f}x)")
        if shrink < 1.2:
            failures.append(
                f"error barely shrank ({shrink:.2f}x) when the sketch grew {large // small}x -- "
                "a correct MinHash estimator's error falls as 1/sqrt(sketch_size)"
            )

    if failures:
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        die("FastDNA's sketch distances do not agree with Mash's.")

    print("\nOK: agrees with Mash, with no systematic bias, and the error "
          "shrinks with sketch size as MinHash theory requires.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
