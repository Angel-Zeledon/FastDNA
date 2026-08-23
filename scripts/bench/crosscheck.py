"""Cross-checks all four canonical k-mer counting implementations
(`fastdna` itself, plus the three Python baselines in this directory)
against the same FASTQ file and **asserts** they agree within a stated
tolerance.

This exists because the README's "four independently-written
implementations land within 0.001% of each other" claim is only worth
publishing if something *runs* it, on demand, and fails loudly the moment
it stops being true -- a benchmark number written down once and never
re-checked is exactly the kind of claim that quietly outlives the code
that produced it.

Usage:

    python scripts/bench/crosscheck.py               # full README-scale check
    python scripts/bench/crosscheck.py --quick        # small, fast sanity check
    python scripts/bench/crosscheck.py --file r.fastq --k 21   # your own file

By default this generates the *same* dataset the README's benchmark table
was measured against (1 Mbp genome, 30x coverage, 150 bp reads, seed 1337
-- see `generate_reads.py`) and runs all four implementations over it, so
it is the real comparison, not a toy. That takes on the order of minutes,
because the two slower Python baselines are O(reads x k) pure-Python loops.
Pass `--quick` for a small generated file (a few seconds, all four
implementations) when you just want a fast sanity check that nothing is
badly broken -- it is not a substitute for running the default mode before
trusting the README's numbers.

Exits non-zero (and prints why) if any implementation disagrees with the
others by more than `--tolerance-pct` percent, if a baseline's output can't
be parsed, or if a required dependency (Biopython, NumPy) is missing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Matches the README's own benchmark dataset exactly, so a full (non-quick)
# run of this script reproduces the numbers the README's Benchmarks section
# quotes, not just "some file of similar shape".
FULL_GENOME_SIZE = 1_000_000
FULL_COVERAGE = 30
FULL_READ_LEN = 150
FULL_SEED = 1337

# Small enough that even the O(reads x k) pure-Python baselines finish in a
# couple of seconds; still large enough to have both an error peak and a
# real coverage peak, so it exercises the same code paths as the full run.
QUICK_GENOME_SIZE = 20_000
QUICK_COVERAGE = 15
QUICK_READ_LEN = 100
QUICK_SEED = 42

_COUNT_RE = re.compile(r"distinct_kmers=(\d+)\s+total_kmers=(\d+)")


def generate_dataset(out_path: Path, *, quick: bool) -> None:
    genome, coverage, read_len, seed = (
        (QUICK_GENOME_SIZE, QUICK_COVERAGE, QUICK_READ_LEN, QUICK_SEED)
        if quick
        else (FULL_GENOME_SIZE, FULL_COVERAGE, FULL_READ_LEN, FULL_SEED)
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "generate_reads.py"),
            str(genome),
            str(coverage),
            str(read_len),
            str(out_path),
            str(seed),
        ],
        check=True,
    )


def run_python_baseline(script_name: str, path: Path, k: int) -> tuple[int, int]:
    """Runs one of the `*_baseline.py` / `naive_python.py` scripts as a
    subprocess and parses its `distinct_kmers=... total_kmers=...` stdout
    line. Subprocess isolation (rather than importing these modules
    in-process) keeps this script from depending on their internals, and
    means a missing dependency (e.g. Biopython) surfaces as this script's
    own clear failure rather than an import-time crash here.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(HERE / script_name), str(path), str(k)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{script_name} exited with status {e.returncode}.\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}"
        ) from e

    match = _COUNT_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"{script_name} produced unparseable output: {result.stdout!r}")
    return int(match.group(1)), int(match.group(2))


def run_fastdna(path: Path, k: int) -> tuple[int, int]:
    import fastdna

    r = fastdna.count(str(path), k=k)
    return r.distinct_kmers, r.total_kmers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="use this FASTQ file instead of generating one")
    parser.add_argument("--k", type=int, default=31, help="k-mer length (default: 31, matching the README)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="generate a small dataset instead of the full README-scale one (seconds instead of minutes)",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=0.05,
        help="maximum allowed relative spread between the min and max distinct/total counts, "
        "as a percentage (default: 0.05, comfortably above the ~0.001%% spread FastDNA's quality "
        "trimming is expected to produce against baselines that don't trim)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the generated FASTQ file (ignored when --file is given)",
    )
    args = parser.parse_args()

    tmp_dir: str | None = None
    try:
        if args.file is not None:
            path = args.file
            if not path.exists():
                print(f"error: {path} does not exist", file=sys.stderr)
                return 1
        else:
            tmp_dir = tempfile.mkdtemp(prefix="fastdna-crosscheck-")
            path = Path(tmp_dir) / "crosscheck.fastq"
            print(f"generating {'quick' if args.quick else 'full README-scale'} dataset at {path} ...")
            generate_dataset(path, quick=args.quick)

        k = args.k
        print(f"\nrunning all four implementations on {path} at k={k} ...\n")

        try:
            results = {
                "fastdna": run_fastdna(path, k),
                "naive_python": run_python_baseline("naive_python.py", path, k),
                "biopython": run_python_baseline("biopython_baseline.py", path, k),
                "numpy": run_python_baseline("numpy_baseline.py", path, k),
            }
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        name_width = max(len(name) for name in results)
        for name, (distinct, total) in results.items():
            print(f"  {name:<{name_width}}  distinct={distinct:>10d}  total={total:>12d}")

        distinct_values = [v[0] for v in results.values()]
        total_values = [v[1] for v in results.values()]

        def spread_pct(values: list[int]) -> float:
            lo, hi = min(values), max(values)
            if hi == 0:
                return 0.0
            return (hi - lo) / hi * 100

        distinct_spread = spread_pct(distinct_values)
        total_spread = spread_pct(total_values)
        print(
            f"\ndistinct spread: {distinct_spread:.4f}%  |  total spread: {total_spread:.4f}%  "
            f"(tolerance: {args.tolerance_pct}%)"
        )

        if distinct_spread > args.tolerance_pct or total_spread > args.tolerance_pct:
            print(
                f"\nFAIL: spread exceeds tolerance ({args.tolerance_pct}%). The four implementations "
                "disagree by more than quality trimming alone should explain -- this is exactly the "
                "'fast because it does less work' failure mode this script exists to catch.",
                file=sys.stderr,
            )
            return 1

        print("\nOK: all four implementations agree within tolerance.")
        return 0
    finally:
        if tmp_dir is not None and not args.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir is not None and args.keep:
            print(f"\nkept generated dataset at {tmp_dir}")


if __name__ == "__main__":
    sys.exit(main())
