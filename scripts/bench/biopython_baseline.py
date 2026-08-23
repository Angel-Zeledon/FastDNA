"""Canonical k-mer counter using Biopython's FASTQ parser (Bio.SeqIO) for
I/O, otherwise the same naive per-base Python loop as naive_python.py.
Represents a bioinformatician reaching for the standard Python parsing
library instead of hand-rolling FASTQ parsing, before reaching for a
specialized k-mer counting tool.
"""
import sys
import time
from collections import Counter
from Bio import SeqIO

COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def canonical(kmer):
    rc = revcomp(kmer)
    return kmer if kmer <= rc else rc


def count_file(path, k):
    counts = Counter()
    total_reads = 0
    for record in SeqIO.parse(path, "fastq"):
        seq = str(record.seq)
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
