"""Naive pure-Python canonical k-mer counter -- collections.Counter, no
external libraries, no quality trimming. Represents the way most people
would first write this without reaching for a specialized tool.
"""
import sys
import time
from collections import Counter

COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def canonical(kmer):
    rc = revcomp(kmer)
    return kmer if kmer <= rc else rc


def count_file(path, k):
    counts = Counter()
    total_reads = 0
    with open(path, "r") as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # '+'
            f.readline()  # qual (ignored -- no quality trimming, see module docstring)
            total_reads += 1
            n = len(seq)
            if n < k:
                continue
            for i in range(n - k + 1):
                sub = seq[i:i + k]
                if "N" in sub:
                    continue
                counts[canonical(sub)] += 1
    return counts, total_reads


if __name__ == "__main__":
    path, k = sys.argv[1], int(sys.argv[2])
    t0 = time.perf_counter()
    counts, reads = count_file(path, k)
    elapsed = time.perf_counter() - t0
    total_kmers = sum(counts.values())
    print(f"reads={reads} distinct_kmers={len(counts)} total_kmers={total_kmers} elapsed={elapsed:.4f}s")
