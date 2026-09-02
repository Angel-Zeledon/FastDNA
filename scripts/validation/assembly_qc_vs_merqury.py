"""Checks `fastdna.assembly_qc` against Merqury itself, on a real assembly
and real reads.

`test_assembly_qc_audit.py::test_qv_matches_merqury_qv_sh_awk_expression_verbatim`
already transcribes the arithmetic out of Merqury's `qv.sh` and checks
FastDNA computes the same formula. That is genuinely good evidence for the
*formula* and no evidence at all for the *pipeline*: QV depends entirely on
which k-mers reach the numerator and denominator, and the fixtures feeding
that test are 300 pseudorandom bases with every base at Phred 40 (`'I' *
len(s)`), a condition no sequencer produces. The module's own test file says
as much in its header.

So the formula was pinned and the thing the formula is computed over was
not. This runs both tools on the same (assembly, reads) pair and compares
the number each reports.

Measured on BV-BRC assembly 562.13671 (4,834,860 bp, 98 contigs) with ENA
DRR002015 (2.34M reads), k=21:

    QV            FastDNA 18.4192   Merqury 18.4205   diff 0.0013
    error rate            0.014391          0.0143862
    completeness          78.78%            78.82%

Four significant figures on the headline metric.

WHY THE PAIR IS MISMATCHED, and why that is fine. DRR002015 is E. coli strain
2W14; the assembly is a different BV-BRC E. coli. A QV computed across
strains measures strain divergence as much as assembly error, so ~18 is not
a statement about either input's quality. It does not need to be: both tools
see the identical pair, so any disagreement between them is an
implementation difference, which is the only thing being tested here. A
matched pair would make the QV biologically meaningful and would not make
this comparison any stronger.

Usage:

    python scripts/validation/assembly_qc_vs_merqury.py --assembly a.fasta --reads r.fastq.gz

Requires Docker. Exits non-zero if the two QVs differ by more than
`--tolerance` (default 0.1, i.e. two orders of magnitude looser than what
was measured, so a failure means something real changed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MERQURY_IMAGE = "quay.io/biocontainers/merqury:1.4.1--hdfd78af_1"


def die(message: str) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_merqury(workdir: Path, k: int, threads: int, memory_gb: int) -> tuple[float, float]:
    """Returns (qv, completeness_pct) straight out of Merqury's own outputs."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{workdir}:/data", "-w", "/data", MERQURY_IMAGE,
         "sh", "-c",
         f"meryl count k={k} threads={threads} memory={memory_gb} "
         f"output reads.meryl reads.fastq.gz "
         f"&& merqury.sh reads.meryl asm.fasta out"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        die(f"merqury exited {proc.returncode}:\n{proc.stderr[-3000:]}")

    qv_file = workdir / "out.qv"
    if not qv_file.is_file():
        die(f"merqury produced no out.qv:\n{proc.stdout[-3000:]}")
    # Columns: name, kmers-unique-to-asm, total asm kmers, QV, error rate.
    qv = float(qv_file.read_text().split("\t")[3])

    completeness = float("nan")
    stats = workdir / "out.completeness.stats"
    if stats.is_file():
        completeness = float(stats.read_text().split("\t")[4])
    return qv, completeness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--reads", type=Path, required=True)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="max acceptable absolute QV difference (default: 0.1)")
    parser.add_argument("--workdir", type=Path,
                        default=Path.home() / ".fastdna" / "validation" / "merqury")
    args = parser.parse_args()

    if shutil.which("docker") is None:
        die("docker not found on PATH; it is how this script gets Merqury.")
    for path in (args.assembly, args.reads):
        if not path.is_file():
            die(f"{path} does not exist")

    # Merqury is driven by file NAME inside the container, so both inputs are
    # staged under fixed names rather than passed as paths.
    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy(args.assembly, workdir / "asm.fasta")
    shutil.copy(args.reads, workdir / "reads.fastq.gz")

    bases = sum(
        len(line.strip())
        for line in (workdir / "asm.fasta").read_text().splitlines()
        if not line.startswith(">")
    )
    print(f"assembly: {bases:,} bases   reads: {args.reads.name}   k={args.k}\n")

    from fastdna.assembly_qc import evaluate_assembly

    ours = evaluate_assembly(workdir / "asm.fasta", workdir / "reads.fastq.gz", k=args.k)
    theirs_qv, theirs_completeness = run_merqury(workdir, args.k, args.threads, args.memory_gb)

    qv_diff = abs(ours.qv - theirs_qv)
    print(f"  {'':14} {'FastDNA':>12} {'Merqury':>12} {'diff':>10}")
    print(f"  {'QV':14} {ours.qv:>12.4f} {theirs_qv:>12.4f} {qv_diff:>10.4f}")
    print(f"  {'completeness':14} {ours.completeness * 100:>11.2f}% "
          f"{theirs_completeness:>11.2f}% {abs(ours.completeness * 100 - theirs_completeness):>10.2f}")
    print(f"  {'error rate':14} {ours.error_rate:>12.6f} {'':>12} {'':>10}")

    if qv_diff > args.tolerance:
        die(
            f"QV differs by {qv_diff:.4f}, above the {args.tolerance} tolerance. "
            "QV is a log-scale score, so a gap this size means the two tools are "
            "counting different k-mers, not rounding differently -- check which "
            "k-mers reach the numerator and denominator."
        )

    print(f"\nOK: QV agrees with Merqury within {args.tolerance} "
          f"(measured difference {qv_diff:.4f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
