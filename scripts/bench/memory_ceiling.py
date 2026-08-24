"""Finds FastDNA's practical memory ceiling on this machine by running the
release CLI against progressively larger FASTQ files while sampling the
process's peak resident set size (RSS).

FastDNA's counting table lives fully in memory -- there is no out-of-core
/ disk-spilling mode (unlike KMC3's disk-resident-bin approach). This
script exists to turn that architectural fact into a measured number
instead of a guess: the largest input this machine can process, and what
actually happens past that point (an allocation failure, the OS killing
the process, or swapping so severely that the run is impractical long
before either of those).

Usage:
    python scripts/bench/memory_ceiling.py <fastdna.exe path> <file1> [file2 ...]

Each file is run once, at the CLI's default settings (k=31, default thread
count). Prints one line per file: size on disk, wall time, peak RSS, and
the outcome. Stops early once a run fails, since the point past the
ceiling does not need re-demonstrating at every larger size.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "{:.2f}{}".format(n, unit)
        n /= 1024
    return "{:.2f}TB".format(n)


def run_and_track_peak_rss(exe, input_path, output_path, qc_path, timeout_s):
    proc = subprocess.Popen(
        [str(exe), "--input", str(input_path), "--output", str(output_path),
         "-k", "31", "--qc", str(qc_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    ps_proc = psutil.Process(proc.pid)
    peak = [0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    rss += child.memory_info().rss
                peak[0] = max(peak[0], rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.2)

    t0 = time.perf_counter()
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    timed_out = False
    try:
        _, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        _, stderr = proc.communicate()
    finally:
        stop.set()
        poller.join(timeout=2)

    elapsed = time.perf_counter() - t0
    return proc.returncode, peak[0], elapsed, timed_out, (stderr or "")[-2000:]


def main():
    exe = Path(sys.argv[1])
    files = [Path(p) for p in sys.argv[2:]]
    if not files:
        print("usage: memory_ceiling.py <fastdna.exe> <file1> [file2 ...]", file=sys.stderr)
        return 1

    tmp_dir = files[0].parent
    for f in files:
        size = f.stat().st_size
        out = tmp_dir / (f.stem + ".memtest.parquet")
        qc = tmp_dir / (f.stem + ".memtest.qc.json")

        print("--- {} ({} on disk) ---".format(f.name, format_bytes(size)), flush=True)
        rc, peak, elapsed, timed_out, stderr_tail = run_and_track_peak_rss(
            exe, f, out, qc, timeout_s=1800
        )

        if timed_out:
            print("  TIMED OUT after {:.1f}s, peak RSS observed: {}".format(elapsed, format_bytes(peak)))
            print("  (killed -- treating this as the practical ceiling; stopping)")
            return 0
        if rc != 0:
            print("  FAILED, exit code {}, elapsed {:.1f}s, peak RSS observed: {}".format(rc, elapsed, format_bytes(peak)))
            print("  stderr tail:\n{}".format(stderr_tail))
            print("  (stopping -- this input is past the practical ceiling)")
            return 0

        print("  OK, {:.1f}s, peak RSS: {}".format(elapsed, format_bytes(peak)), flush=True)

        try:
            out.unlink(missing_ok=True)
            qc.unlink(missing_ok=True)
        except OSError:
            pass

    print("\nAll files completed without hitting a ceiling on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
