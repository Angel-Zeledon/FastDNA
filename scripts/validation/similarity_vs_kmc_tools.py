"""`fastdna similarity` against KMC3's own set operations.

## What this checks, and why it needs an outside tool at all

`fastdna similarity` reports exact Jaccard and containment between two
counted k-mer tables. KMC3 does not report similarity -- but `kmc_tools`
computes the set operations those numbers are made of, so it can be used as
an oracle for them:

    |A ∩ B|  = k-mers in `kmc_tools simple A B intersect`
    |A|, |B| = each database's own k-mer count
    jaccard  = |A ∩ B| / (|A| + |B| - |A ∩ B|)

If FastDNA's Jaccard and KMC3's arithmetic disagree, one of them is wrong,
and the answer does not depend on reading FastDNA's own source -- which is
the property `CLAUDE.md` requires of anything that ships here.

This is deliberately *not* a check that could be written inside the test
suite. A unit test asserting `jaccard == 0.5` on a hand-built pair proves
the formula was typed correctly; it cannot notice that the merge dropped a
k-mer, that canonicalisation disagrees with the field's convention, or that
`KmerTable` and KMC3 mean different things by "distinct". Only a second
implementation can.

## Why emulation is acceptable here and not in scripts/bench

The KMC3 biocontainer is amd64-only, so on an arm64 host it runs emulated.
`scripts/bench/head_to_head.py` refuses to run there, because emulation
distorts *timings*. It does not distort *results*: an emulated KMC3 counts
exactly the k-mers a native one counts, which is all this script reads.

## The input

Two halves of one read set, which is the case that actually exercises the
arithmetic: two files from the same underlying genome share most of their
true k-mers and almost none of their error k-mers, so the intersection is
large, the exclusive sets are large, and Jaccard lands well away from both
0 and 1. Two unrelated files would agree trivially at 0, and a file against
itself trivially at 1; neither would catch an off-by-one in the merge.

Usage:

    cargo build --release
    python scripts/validation/similarity_vs_kmc_tools.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KMC_IMAGE = "quay.io/biocontainers/kmc:3.2.4--h5ca1c30_4"

#: Jaccard is a ratio of integers both tools compute exactly, so the only
#: tolerance needed is for float formatting on the way through CSV.
JACCARD_TOLERANCE = 1e-9


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def docker(work: Path, args: list[str]) -> str:
    """One `kmc`/`kmc_tools` invocation inside the pinned biocontainer."""
    result = run([
        "docker", "run", "--rm",
        # amd64-only image; an arm64 host needs this to pick emulation
        # explicitly rather than fail. Results are unaffected -- see the
        # module docstring.
        "--platform", "linux/amd64",
        "-v", f"{work}:/data", "-w", "/data", KMC_IMAGE, *args,
    ])
    if result.returncode != 0:
        die(f"{' '.join(args[:2])} exited {result.returncode}:\n{result.stderr[-3000:]}\n{result.stdout[-3000:]}")
    return result.stdout


def split_reads(source: Path, work: Path, reads_per_half: int) -> tuple[Path, Path]:
    """Two disjoint halves of one FASTQ, four lines per record."""
    a, b = work / "half_a.fastq", work / "half_b.fastq"
    if a.is_file() and b.is_file():
        return a, b
    with source.open() as fh, a.open("w") as out_a, b.open("w") as out_b:
        for index in range(reads_per_half * 2):
            record = [fh.readline() for _ in range(4)]
            if not record[0]:
                die(f"{source} has fewer than {reads_per_half * 2} reads")
            (out_a if index < reads_per_half else out_b).writelines(record)
    return a, b


def kmc_count(work: Path, reads: Path, name: str, k: int) -> int:
    """Builds a KMC database and returns its distinct k-mer count."""
    scratch = work / f"{name}_work"
    scratch.mkdir(exist_ok=True)
    stdout = docker(work, [
        "kmc", f"-k{k}", "-ci1", "-fq", "-t4", "-m4",
        reads.name, name, scratch.name,
    ])
    match = re.search(r"No\. of unique counted k-mers\s*:\s*(\d+)", stdout)
    if match is None:
        die(f"could not parse KMC3's k-mer count for {name}:\n{stdout[-2000:]}")
    return int(match.group(1))


def kmc_intersection_size(work: Path, a: str, b: str, k: int) -> int:
    """|A ∩ B|, via `kmc_tools simple ... intersect` and a dump line count.

    The dump is counted rather than trusted to a summary line because
    `kmc_tools`'s own stdout for `simple` reports nothing about the result's
    size, and the versions that do print statistics differ in wording
    between releases -- counting the k-mers it actually wrote cannot drift.
    """
    docker(work, ["kmc_tools", "simple", a, "-ci1", b, "-ci1", "intersect", "isect", "-ocmin"])
    docker(work, ["kmc_tools", "transform", "isect", "dump", "isect.txt"])
    dump = work / "isect.txt"
    with dump.open("rb") as fh:
        return sum(1 for _ in fh)


def fastdna_count(binary: Path, reads: Path, out: Path, k: int) -> None:
    result = run([str(binary), "--input", str(reads), "--output", str(out),
                  "-k", str(k), "-m", "1", "-q", "0"])
    if result.returncode != 0:
        die(f"fastdna count exited {result.returncode}:\n{result.stderr[-2000:]}")


def fastdna_similarity(binary: Path, tables: list[Path], k: int) -> dict:
    result = run([str(binary), "similarity", "--input", *[str(t) for t in tables]])
    if result.returncode != 0:
        die(f"fastdna similarity exited {result.returncode}:\n{result.stderr[-2000:]}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        die(f"fastdna similarity produced no data rows:\n{result.stdout}")
    header, row = lines[0].split(","), lines[1].split(",")
    return dict(zip(header, row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path,
                        default=Path.home() / ".fastdna" / "validation" / "similarity")
    parser.add_argument("--reads", type=Path,
                        help="FASTQ to split in half; generated if omitted")
    parser.add_argument("--reads-per-half", type=int, default=200_000)
    parser.add_argument("--k", type=int, default=31)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    binary = REPO_ROOT / "target" / "release" / "fastdna"
    if not binary.is_file():
        die(f"{binary} not found -- run `cargo build --release` first.")

    args.workdir.mkdir(parents=True, exist_ok=True)
    source = args.reads
    if source is None:
        source = args.workdir / "source.fastq"
        if not source.is_file():
            generator = REPO_ROOT / "scripts" / "bench" / "generate_reads_large.py"
            proc = run([sys.executable, str(generator), "2000000", "30", "150",
                        str(source), "31337"])
            if proc.returncode != 0:
                die(f"read generation failed:\n{proc.stderr[-2000:]}")

    print("input")
    a_reads, b_reads = split_reads(source, args.workdir, args.reads_per_half)
    print(f"  {a_reads.name} and {b_reads.name}, {args.reads_per_half} reads each")

    print("\nKMC3 (the oracle)")
    size_a = kmc_count(args.workdir, a_reads, "kmc_a", args.k)
    size_b = kmc_count(args.workdir, b_reads, "kmc_b", args.k)
    shared = kmc_intersection_size(args.workdir, "kmc_a", "kmc_b", args.k)
    union = size_a + size_b - shared
    oracle = {
        "distinct_a": size_a,
        "distinct_b": size_b,
        "shared": shared,
        "jaccard": shared / union if union else 1.0,
        "containment_ab": shared / size_a if size_a else 1.0,
        "containment_ba": shared / size_b if size_b else 1.0,
    }
    print(f"  |A| = {size_a:,}   |B| = {size_b:,}   |A n B| = {shared:,}")
    print(f"  jaccard = {oracle['jaccard']:.9f}")

    print("\nFastDNA")
    table_a, table_b = args.workdir / "a.parquet", args.workdir / "b.parquet"
    fastdna_count(binary, a_reads, table_a, args.k)
    fastdna_count(binary, b_reads, table_b, args.k)
    ours = fastdna_similarity(binary, [table_a, table_b], args.k)
    print(f"  |A| = {int(ours['shared']) + int(ours['only_a']):,}   "
          f"|B| = {int(ours['shared']) + int(ours['only_b']):,}   "
          f"|A n B| = {int(ours['shared']):,}")
    print(f"  jaccard = {float(ours['jaccard']):.9f}")

    print("\ncomparison")
    failures = []
    for label, theirs, mine in [
        ("|A|", size_a, int(ours["shared"]) + int(ours["only_a"])),
        ("|B|", size_b, int(ours["shared"]) + int(ours["only_b"])),
        ("|A n B|", shared, int(ours["shared"])),
    ]:
        ok = theirs == mine
        print(f"  {label:<10} KMC3 {theirs:>12,}   FastDNA {mine:>12,}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{label}: KMC3 {theirs} vs FastDNA {mine}")

    for label in ("jaccard", "containment_ab", "containment_ba"):
        theirs, mine = oracle[label], float(ours[label])
        ok = abs(theirs - mine) <= JACCARD_TOLERANCE
        print(f"  {label:<15} KMC3 {theirs:.9f}   FastDNA {mine:.9f}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{label}: KMC3 {theirs} vs FastDNA {mine}")

    if args.json:
        args.json.write_text(json.dumps({"kmc3": oracle, "fastdna": ours}, indent=2))
        print(f"\nwrote {args.json}")

    if failures:
        die("FastDNA and KMC3 disagree:\n  " + "\n  ".join(failures))
    print("\nOK: exact set sizes and every derived ratio agree with KMC3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
