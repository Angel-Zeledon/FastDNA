"""FastDNA against KMC3 and FASTK, on one machine, in one process tree.

## Why this exists as a script rather than as a record of commands

`kmc3_fastk_comparison.sh` next to this file is the transcript of the run
that produced the historical table in `docs/BENCHMARKS.md`: a human, in
WSL2, on one laptop, once. That table has been wrong twice -- once because
the tools were not counting the same k-mers, once because the machine was
busy -- and both times the correction arrived months later, by hand. A
speed claim with no mechanism to re-derive it is exactly the failure this
repository keeps finding in itself.

So this is the same comparison, automated, asserting rather than reporting
the one thing that must hold (all tools agree on the count), and runnable
by `.github/workflows/validation.yml` on a schedule.

## Why it cannot run on an Apple silicon developer machine

The k-mer counters here are distributed as x86-64 Linux binaries: KMC3 ships
one, and the biocontainer the correctness scripts use is amd64-only. On an
arm64 host they run under emulation, and emulation does not cost every
program the same -- a SIMD-heavy inner loop pays far more than a
memory-bound one. Timing an emulated KMC3 against a native FastDNA would
produce a flattering number that means nothing, which is worse than no
number.

This script used to refuse outright off x86-64. It no longer has to for
KMC3: **KMC 3.2.4's own Makefile handles `aarch64`** (`D_ARCH=ARM64`,
`-march=armv8.4-a`), so on an ARM host `fetch_kmc` clones and builds it
instead of downloading the wrong architecture. Both tools then run native
on the same machine, which is the condition that made the refusal
necessary in the first place. FASTK ships x86-64 binaries only and has no
such escape, so `--with-fastk` keeps the guard.

What that does *not* fix is a busy or small machine. Measured on an M3 Pro
with an 11-core, 7.7 GB Docker VM and unrelated host load, consecutive runs
of the **same** tool on the same file varied by up to 7x (KMC3: 19.95 s
then 150.92 s; FastDNA binned: 18.01 s then 105.91 s). No median over a
handful of runs survives that. Architecture was one blocker; a quiet host
with room for the working set is the other, and this script cannot supply
it.

## What is compared, and on what terms

All tools count canonical k-mers, keep singletons, and read the same file:

* FastDNA `--strategy auto` -- what a user actually gets. Reported
  alongside which strategy `auto` chose, because that is the number the
  README's claim is about.
* FastDNA with each strategy forced, so a regression in the chooser is
  distinguishable from a regression in a strategy.
* KMC3 `-ci1` (keep singletons; without it KMC3 drops them by default and
  the tools answer different questions).
* FASTK `-t1`, optional: it is built from source, which takes minutes, so
  it is opt-in via `--with-fastk`.

Quality trimming is **off** for FastDNA (`-q 0`). It is a real feature and
the other two do not have it; leaving it on would mean FastDNA counts fewer
k-mers and the comparison would have a discrepancy to explain away instead
of an equality to assert.

Usage:

    python scripts/bench/head_to_head.py --json results.json
    python scripts/bench/head_to_head.py --genome-size 35000000 --with-fastk
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: KMC3's own published Linux x86-64 release, pinned. Not a container: this
#: script's whole point is measuring native speed, and the biocontainer used
#: by `scripts/validation/` is amd64-only and would be emulated here. Off
#: x86-64 this URL is skipped and `build_kmc_from_source` compiles the same
#: pinned version instead.
KMC_VERSION = "3.2.4"
KMC_RELEASE_URL = (
    "https://github.com/refresh-bio/KMC/releases/download/v3.2.4/KMC3.2.4.linux.x64.tar.gz"
)
FASTK_REPO = "https://github.com/thegenemyers/FASTK.git"


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def timed(command: list[str], cwd: Path | None = None) -> tuple[float, int, str]:
    """`(wall_seconds, peak_rss_bytes, stdout)` for one child process.

    Peak RSS comes from the platform's own `time` utility rather than from
    `resource.getrusage(RUSAGE_CHILDREN)`, which reports a high-water mark
    across *every* child this process has ever reaped -- so the second tool
    measured would inherit the first one's peak and every number after the
    largest would be wrong in the same direction.
    """
    if not Path("/usr/bin/time").is_file():
        # Not on every minimal image, and its absence must not take the
        # benchmark down: a timing with no peak-RSS column is still a
        # timing, and reporting 0 for the column is honest about the gap.
        started = time.monotonic()
        proc = run(command, cwd=cwd)
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            die(f"{command[0]} exited {proc.returncode}:\n{proc.stderr[-3000:]}")
        return elapsed, 0, proc.stdout

    if sys.platform == "darwin":
        wrapper, pattern = ["/usr/bin/time", "-l"], r"(\d+)\s+maximum resident set size"
        scale = 1
    else:
        wrapper, pattern = ["/usr/bin/time", "-v"], r"Maximum resident set size \(kbytes\):\s*(\d+)"
        scale = 1024

    started = time.monotonic()
    proc = run(wrapper + command, cwd=cwd)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        die(f"{command[0]} exited {proc.returncode}:\n{proc.stderr[-3000:]}")

    match = re.search(pattern, proc.stderr)
    peak = int(match.group(1)) * scale if match else 0
    return elapsed, peak, proc.stdout


def build_kmc_from_source(tools: Path) -> Path:
    """Compiles KMC3 for *this* machine, for hosts its release archive does
    not cover.

    KMC ships x86-64 Linux binaries only, which is why this script used to
    refuse to report timings anywhere else: an emulated competitor against a
    native FastDNA measures the emulator. But KMC 3.2.4's own Makefile
    handles `aarch64` (it sets `D_ARCH=ARM64` and `-march=armv8.4-a`), so on
    an ARM host the honest move is to build it rather than emulate it --
    both tools native, same machine, same compiler family.

    Requires `git`, `make` and a C++ toolchain. Returns the built binary.
    """
    binary = tools / "bin" / "kmc"
    if binary.is_file():
        return binary
    tools.mkdir(parents=True, exist_ok=True)
    src = tools / "KMC"
    if not src.is_dir():
        print(f"  building KMC3 {KMC_VERSION} from source for {platform.machine()}")
        clone = run(["git", "clone", "--depth", "1", "--branch", f"v{KMC_VERSION}",
                     "https://github.com/refresh-bio/KMC.git", str(src)])
        if clone.returncode != 0:
            die(f"git clone of KMC failed:\n{clone.stderr[-2000:]}")
    made = run(["make", "-j", str(os.cpu_count() or 4), "kmc"], cwd=src)
    if made.returncode != 0:
        made = run(["make", "-j", str(os.cpu_count() or 4)], cwd=src)
    built = next((p for p in src.rglob("kmc") if p.is_file() and os.access(p, os.X_OK)), None)
    if built is None:
        die(f"KMC did not build here:\n{made.stderr[-3000:]}")
    (tools / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, binary)
    binary.chmod(0o755)
    return binary


def fetch_kmc(tools: Path) -> Path:
    binary = tools / "bin" / "kmc"
    if binary.is_file():
        return binary
    # Off x86-64 the release archive is the wrong architecture; build instead.
    if platform.machine() not in ("x86_64", "AMD64"):
        return build_kmc_from_source(tools)
    tools.mkdir(parents=True, exist_ok=True)
    archive = tools / "kmc.tar.gz"
    print(f"  downloading {KMC_RELEASE_URL}")
    urllib.request.urlretrieve(KMC_RELEASE_URL, archive)
    with tarfile.open(archive) as tar:
        # `filter="data"` is the extraction policy Python 3.12 warns about
        # not choosing and 3.14 makes the default; naming it keeps this
        # working identically across all three.
        if sys.version_info >= (3, 12):
            tar.extractall(tools, filter="data")
        else:
            tar.extractall(tools)
    if not binary.is_file():
        die(f"KMC3 archive did not contain bin/kmc under {tools}")
    binary.chmod(0o755)
    (tools / "bin" / "kmc_tools").chmod(0o755)
    return binary


def build_fastk(tools: Path) -> Path:
    binary = tools / "FASTK" / "FastK"
    if binary.is_file():
        return binary
    print("  building FASTK from source (minutes)")
    if run(["git", "clone", "-q", FASTK_REPO, str(tools / "FASTK")]).returncode != 0:
        die("git clone of FASTK failed")
    proc = run(["make", "-C", str(tools / "FASTK"), "-j4"])
    if proc.returncode != 0:
        die(f"FASTK build failed:\n{proc.stderr[-3000:]}")
    return binary


def generate_reads(out: Path, genome_size: int, coverage: int, read_len: int, seed: int) -> Path:
    if out.is_file():
        print(f"  reusing {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return out
    generator = REPO_ROOT / "scripts" / "bench" / "generate_reads_large.py"
    proc = run([sys.executable, str(generator), str(genome_size), str(coverage),
                str(read_len), str(out), str(seed)])
    if proc.returncode != 0:
        die(f"read generation failed:\n{proc.stderr[-2000:]}")
    print(f"  {proc.stdout.strip()}")
    return out


def count_fastdna(binary: Path, reads: Path, workdir: Path, k: int, threads: int,
                  strategy: str) -> dict:
    output = workdir / f"fastdna_{strategy}.parquet"
    elapsed, peak, stdout = timed([
        str(binary), "--input", str(reads), "--output", str(output),
        "-k", str(k), "-m", "1", "-q", "0", "-t", str(threads),
        "--strategy", strategy,
    ])
    distinct = re.search(r"Distinct k-mers:\s*(\d+)", stdout)
    used = re.search(r"Strategy Used:\s*(\S+)", stdout)
    return {
        "tool": f"fastdna ({strategy})",
        "seconds": round(elapsed, 2),
        "peak_bytes": peak,
        "distinct": int(distinct.group(1)) if distinct else None,
        "strategy_used": used.group(1) if used else None,
    }


def count_kmc(binary: Path, reads: Path, workdir: Path, k: int, threads: int) -> dict:
    scratch = workdir / "kmc_work"
    scratch.mkdir(exist_ok=True)
    elapsed, peak, stdout = timed([
        str(binary), f"-k{k}", "-ci1", "-fq", f"-t{threads}", "-m4",
        str(reads), str(workdir / "kmc_out"), str(scratch),
    ])
    match = re.search(r"No\. of unique counted k-mers\s*:\s*(\d+)", stdout)
    return {
        "tool": "kmc3",
        "seconds": round(elapsed, 2),
        "peak_bytes": peak,
        "distinct": int(match.group(1)) if match else None,
    }


def count_fastk(binary: Path, reads: Path, workdir: Path, k: int, threads: int) -> dict:
    elapsed, peak, _ = timed([
        str(binary), f"-k{k}", "-t1", f"-T{threads}", "-N" + str(workdir / "fastk_out"),
        str(reads),
    ])
    # FastK reports nothing countable on stdout; its table is read back with
    # Histex, which is not built here. The equality assertion below therefore
    # covers FastDNA and KMC3 only, and this row is timing-only -- stated
    # rather than silently implied by a missing column.
    return {"tool": "fastk", "seconds": round(elapsed, 2), "peak_bytes": peak, "distinct": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, default=Path.home() / ".fastdna" / "headtohead")
    parser.add_argument("--genome-size", type=int, default=6_000_000,
                        help="reference length; 6e6 gives a ~390 MB file that fits a CI runner")
    parser.add_argument("--coverage", type=int, default=30)
    parser.add_argument("--read-len", type=int, default=150)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--k", type=int, default=31)
    parser.add_argument("--threads", type=int, default=0, help="0 = every core")
    parser.add_argument("--with-fastk", action="store_true",
                        help="also build and run FASTK (adds minutes to a cold run)")
    parser.add_argument("--allow-emulation", action="store_true",
                        help="time FASTK on a non-x86-64 host anyway, under emulation; "
                             "see the module docstring. KMC3 no longer needs this -- it is "
                             "built from source there instead")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    machine = platform.machine()
    if machine not in ("x86_64", "AMD64"):
        # No longer a refusal: `fetch_kmc` builds KMC3 from source here, so
        # both tools are native and the emulation objection is gone. FASTK
        # is still x86-64-only, which is why `--with-fastk` keeps the guard.
        print(f"note: {machine} host -- KMC3 will be built from source so both tools run native")
        if args.with_fastk and not args.allow_emulation:
            die(
                f"this host is {machine} and FASTK ships x86-64 binaries only, so --with-fastk "
                "would time an emulated competitor against a native FastDNA. Drop --with-fastk, "
                "run on x86-64, or pass --allow-emulation for numbers labelled as meaningless."
            )

    binary = REPO_ROOT / "target" / "release" / "fastdna"
    if not binary.is_file():
        die(f"{binary} not found -- run `cargo build --release` first.")

    threads = args.threads or (len(__import__("os").sched_getaffinity(0))
                               if hasattr(__import__("os"), "sched_getaffinity")
                               else __import__("os").cpu_count() or 1)

    args.workdir.mkdir(parents=True, exist_ok=True)
    tools = args.workdir / "tools"

    print("inputs")
    reads = generate_reads(args.workdir / f"bench_{args.genome_size}_{args.seed}.fastq",
                           args.genome_size, args.coverage, args.read_len, args.seed)

    print("\ntools")
    kmc = fetch_kmc(tools)
    fastk = build_fastk(tools) if args.with_fastk else None

    print(f"\ncounting (k={args.k}, {threads} threads, singletons kept)")
    rows = [
        count_fastdna(binary, reads, args.workdir, args.k, threads, "auto"),
        count_fastdna(binary, reads, args.workdir, args.k, threads, "binned"),
        count_fastdna(binary, reads, args.workdir, args.k, threads, "memory"),
        count_kmc(kmc, reads, args.workdir, args.k, threads),
    ]
    if fastk is not None:
        rows.append(count_fastk(fastk, reads, args.workdir, args.k, threads))

    print(f"\n{'tool':<22}{'time':>9}{'peak RSS':>12}{'distinct k-mers':>18}")
    for row in sorted(rows, key=lambda r: r["seconds"]):
        peak = f"{row['peak_bytes'] / 1e9:.2f} GB" if row["peak_bytes"] else "n/a"
        distinct = f"{row['distinct']:,}" if row["distinct"] is not None else "-"
        print(f"{row['tool']:<22}{row['seconds']:>8.2f}s{peak:>12}{distinct:>18}")
    if emulated:
        print("\nWARNING: emulated host -- these timings compare nothing. See --allow-emulation.")

    # The one thing that must hold. Everything above is a measurement; this
    # is an assertion, because two counters that disagree on the answer are
    # not being compared on speed at all.
    counted = {row["tool"]: row["distinct"] for row in rows if row["distinct"] is not None}
    if len(set(counted.values())) > 1:
        die(f"tools disagree on the distinct k-mer count: {counted}")
    print(f"\nOK: every tool that reports a count agrees on {next(iter(counted.values())):,} "
          f"distinct k-mers.")

    if args.json:
        args.json.write_text(json.dumps({
            "machine": machine,
            "emulated": emulated,
            "k": args.k,
            "threads": threads,
            "input_bytes": reads.stat().st_size,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
