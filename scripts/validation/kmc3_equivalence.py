"""Asserts that FastDNA's k-mer counts are **exactly** KMC3's, on real
sequencing data, and fails loudly when they stop being.

Why this is a separate script from `scripts/bench/crosscheck.py`, which
already cross-checks four implementations: that one compares FastDNA against
three *Python baselines this repository wrote itself*, on a *synthetic*
dataset it generates, and it allows a tolerance. All three of those are
deliberate there and all three are weaknesses here:

  * A baseline written by the same project that wrote the thing under test
    shares its blind spots. KMC3 is a decade-old, independently authored,
    widely cited counter -- if FastDNA agrees with it, the agreement means
    something a self-written baseline cannot establish.
  * Synthetic reads have no adapter contamination, no quality collapse at
    the 3' end, no runs of `N`, and no genuine biological repeats. Real data
    is where a counting bug actually hides.
  * A tolerance is right when comparing implementations that legitimately
    do different work (`crosscheck.py`'s baselines do not quality-trim).
    Here there is no such excuse: with trimming disabled on FastDNA's side,
    both tools are solving the *identical* mathematical problem -- count
    distinct canonical k-mers of length k -- and the only acceptable
    difference is zero. See "The fair-comparison flags" below.

Both engines are in scope. KMC3 counts to k=256, so it is an outside answer
for the `u128` engine (`33 <= k <= 64`) as well as the `u64` one -- and
`--engine wide` at `k <= 32` checks the wide engine *directly* against KMC3
rather than only against the narrow engine, which is the strongest form the
overlap check can take.

`docs/BENCHMARKS.md` already records an exact agreement with KMC3 at 2.14 GB,
and `scripts/bench/kmc3_fastk_comparison.sh` records the steps that produced
it -- but that script says of itself that it is "a record of what was
actually run, not a one-command installer", and nothing re-runs it. A
benchmark number nobody re-checks quietly outlives the code that produced it.
This script exists so that claim is executable by a stranger, on one command,
with no manual toolchain setup: KMC3 arrives as a pinned biocontainer.

Usage:

    python scripts/validation/kmc3_equivalence.py              # downloads a real ENA run
    python scripts/validation/kmc3_equivalence.py --file r.fastq.gz --k 21
    python scripts/validation/kmc3_equivalence.py --k 41            # the u128 engine
    python scripts/validation/kmc3_equivalence.py --k 31 --engine wide   # u128 in the overlap
    python scripts/validation/kmc3_equivalence.py --keep       # keep the downloaded data

Requirements: Docker (for KMC3) and a release build of FastDNA
(`cargo build --release`). Both are checked before any work starts.

Exits non-zero if the two tools disagree on distinct k-mers, on total
k-mers, or on the read count, or if either tool fails to run.

NOT a speed benchmark. On an arm64 host the KMC3 container runs under
amd64 emulation, so its wall time here is meaningless. Timings are printed
for orientation only and must never be quoted as a comparison; the real
performance numbers live in `docs/BENCHMARKS.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pinned by tag, not by `:latest`: the whole point is that a rerun a year
# from now compares against the same reference implementation this run did.
KMC_IMAGE = "quay.io/biocontainers/kmc:3.2.4--h5ca1c30_4"

# DRR002015 (Escherichia coli 2W14), R1 only, ~97 MB gzipped, 2,343,637
# reads of 100 bp. Chosen for continuity, not novelty: it is the exact run
# `docs/validation-real-data.md` used for its metagenomic-classification
# track, so a discrepancy here is comparable against a result this project
# has already published internally.
DEFAULT_URL = (
    "https://ftp.sra.ebi.ac.uk/vol1/fastq/DRR002/DRR002015/DRR002015_1.fastq.gz"
)
DEFAULT_ACCESSION = "DRR002015"


def die(message: str) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_prerequisites() -> Path:
    """Fails before doing any slow work, rather than after the download."""
    if shutil.which("docker") is None:
        die("docker not found on PATH; it is how this script gets KMC3.")
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, check=True, timeout=60
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        die("the docker daemon is not responding; start it and retry.")

    binary = REPO_ROOT / "target" / "release" / "fastdna"
    if not binary.is_file():
        die(f"{binary} not found -- run `cargo build --release` first.")
    return binary


def fetch_reads(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"reusing cached reads: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    print("  (this is a real public sequencing run; expect ~100 MB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    # Rename only once the transfer completed, so an interrupted run cannot
    # leave a truncated file that the `is_file()` check above would happily
    # reuse as if it were whole.
    tmp.rename(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def run_kmc3(reads: Path, k: int, threads: int) -> tuple[int, int, int]:
    """Returns (distinct, total, reads) as reported by KMC3 itself."""
    workdir = reads.parent / "kmc_work"
    workdir.mkdir(exist_ok=True)

    # `-ci1` is load-bearing. KMC3's own default (`-ci2`) EXCLUDES k-mers
    # seen only once, and FastDNA's default `min_count=1` does not; without
    # this flag the two tools answer different questions and disagree by
    # millions of singletons, which would look like a counting bug.
    command = [
        "docker", "run", "--rm",
        # The biocontainer is a single-architecture amd64 image, so an arm64
        # host needs this to select emulation explicitly rather than fail.
        "--platform", "linux/amd64",
        "-v", f"{reads.parent}:/data",
        "-w", "/data",
        KMC_IMAGE,
        "kmc", f"-k{k}", "-ci1", "-fq", f"-t{threads}", "-m8",
        reads.name, "kmc_out", "kmc_work",
    ]
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        die(f"KMC3 exited {proc.returncode}:\n{proc.stderr[-2000:]}")

    def grab(pattern: str) -> int:
        match = re.search(pattern + r"\s*:\s*(\d+)", proc.stdout)
        if match is None:
            die(f"could not parse KMC3 output for {pattern!r}:\n{proc.stdout[-2000:]}")
        return int(match.group(1))

    distinct = grab(r"No\. of unique counted k-mers")
    total = grab(r"Total no\. of k-mers")
    n_reads = grab(r"Total no\. of reads")
    print(f"KMC3     : {distinct:>12,} distinct | {total:>13,} total | {n_reads:>10,} reads  ({elapsed:.1f}s, emulated)")
    return distinct, total, n_reads


def run_fastdna(binary: Path, reads: Path, k: int, engine: str) -> tuple[int, int, int]:
    out = reads.parent / "fastdna_out.parquet"
    qc = reads.parent / "fastdna_qc.json"

    # `-q 0` disables quality trimming. This is the flag that makes the
    # comparison fair: KMC3 does not trim, and FastDNA's default (Q20)
    # would legitimately drop low-quality 3' bases, producing FEWER k-mers
    # for a correct reason. With `-q 0` the trimming loop's condition
    # (`window mean >= min_qual`) is true on its first iteration for every
    # read, so nothing is trimmed. `-m 1` keeps singletons, matching `-ci1`.
    command = [
        str(binary), "--input", str(reads), "--output", str(out),
        "--qc", str(qc), "-k", str(k), "-q", "0", "-m", "1",
        "--engine", engine,
    ]
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        die(f"fastdna exited {proc.returncode}:\n{proc.stderr[-2000:]}")

    def grab(pattern: str) -> int:
        match = re.search(pattern, proc.stdout)
        if match is None:
            die(f"could not parse fastdna output for {pattern!r}:\n{proc.stdout[-2000:]}")
        return int(match.group(1))

    distinct = grab(r"Distinct k-mers:\s*(\d+)")
    total = grab(r"Total k-mers Indexed:\s*(\d+)")
    n_reads = grab(r"Total Reads:\s*(\d+)")
    print(f"FastDNA  : {distinct:>12,} distinct | {total:>13,} total | {n_reads:>10,} reads  ({elapsed:.1f}s, native)")

    # Cross-check the banner against the artefact it claims to have written:
    # a bug that miscounts and a bug that misreports are different bugs, and
    # this script should not be fooled by the second one.
    try:
        import pyarrow.parquet as pq

        rows = pq.read_metadata(out).num_rows
        if rows != distinct:
            die(
                f"fastdna reported {distinct:,} distinct k-mers but wrote {rows:,} "
                "rows -- the banner and the Parquet file disagree with each other."
            )
    except ImportError:
        print("  (pyarrow absent: skipped the Parquet row-count cross-check)")
    return distinct, total, n_reads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, help="use this FASTQ(.gz) instead of downloading")
    parser.add_argument("--url", default=DEFAULT_URL, help="ENA URL to download when --file is absent")
    parser.add_argument("--k", type=int, default=31, help="k-mer length (default: 31)")
    parser.add_argument(
        "--engine",
        choices=("auto", "narrow", "wide"),
        default="auto",
        help=(
            "which FastDNA engine to check. 'auto' (the default) is what a user gets: the "
            "u64 engine at k<=32, the u128 one above. 'wide' forces the u128 engine into "
            "the OVERLAP RANGE, which is the point of exposing this here -- at k<=32 both "
            "engines answer the same question, so running the wide one against KMC3 checks "
            "it directly rather than only against the narrow engine that KMC3 already "
            "validated. 'narrow' pins the u64 engine and fails above k=32."
        ),
    )
    parser.add_argument("--threads", type=int, default=4, help="threads for KMC3 (default: 4)")
    parser.add_argument("--workdir", type=Path, default=Path.home() / ".fastdna" / "validation")
    parser.add_argument("--keep", action="store_true", help="keep intermediate KMC3/Parquet artefacts")
    parser.add_argument("--json", type=Path, help="also write the result as JSON to this path")
    args = parser.parse_args()

    binary = check_prerequisites()

    if args.file is not None:
        reads = args.file.resolve()
        if not reads.is_file():
            die(f"{reads} does not exist")
    else:
        reads = fetch_reads(args.url, args.workdir / f"{DEFAULT_ACCESSION}_1.fastq.gz")

    engine_note = "" if args.engine == "auto" else f", --engine {args.engine}"
    print(
        f"\ncomparing at k={args.k}{engine_note}, singletons included, "
        "quality trimming disabled"
    )
    print("-" * 78)
    kmc = run_kmc3(reads, args.k, args.threads)
    fastdna = run_fastdna(binary, reads, args.k, args.engine)
    print("-" * 78)

    labels = ("distinct k-mers", "total k-mers", "reads")
    mismatches = [
        (label, a, b) for label, a, b in zip(labels, kmc, fastdna) if a != b
    ]

    if args.json is not None:
        args.json.write_text(json.dumps({
            "k": args.k,
            "engine": args.engine,
            "input": str(reads),
            "kmc3": dict(zip(labels, kmc)),
            "fastdna": dict(zip(labels, fastdna)),
            "agree": not mismatches,
        }, indent=2))

    if not args.keep:
        for leftover in ("kmc_work", "kmc_out.kmc_pre", "kmc_out.kmc_suf",
                         "fastdna_out.parquet", "fastdna_qc.json"):
            target = reads.parent / leftover
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.is_file():
                target.unlink()

    if mismatches:
        for label, expected, got in mismatches:
            print(f"  {label}: KMC3 {expected:,} vs FastDNA {got:,} "
                  f"(difference {got - expected:+,})", file=sys.stderr)
        die(
            "FastDNA and KMC3 disagree on real data. With trimming disabled and "
            "singletons kept, both tools count distinct canonical k-mers of the "
            "same length over the same reads -- there is no legitimate reason for "
            "any difference. Suspect canonicalisation, the handling of ambiguous "
            "bases, or read parsing."
        )

    print("OK: FastDNA's counts are exactly KMC3's on real sequencing data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
