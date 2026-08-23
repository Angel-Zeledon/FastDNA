"""Vectorized canonical k-mer counter using NumPy: 2-bit-packs each base
with vectorized array ops, builds every k-mer's packed uint64 via a
convolution-like cumulative shift (no per-k-mer Python-level loop for the
extraction itself), computes each k-mer's reverse complement with bit ops
mirroring src/kmer.rs, and reduces via np.unique. This is close to "the
fastest a fluent NumPy user would reasonably write it" -- a much stronger
baseline than naive_python.py or biopython_baseline.py.
"""
import sys
import time
import numpy as np

CODE = np.full(256, -1, dtype=np.int64)
for ch, val in zip(b"ACGT", [0, 1, 2, 3]):
    CODE[ch] = val


def read_fastq_sequences(path):
    seqs = []
    with open(path, "rb") as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().rstrip(b"\n")
            f.readline()
            f.readline()
            seqs.append(seq)
    return seqs


def encode_kmers_for_read(seq_bytes, k):
    arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    codes = CODE[arr]
    n = len(codes)
    if n < k:
        return None
    # A base outside ACGT (e.g. 'N') poisons every k-mer window it touches;
    # mark those windows invalid rather than special-casing them, mirroring
    # extract_canonical_kmers's window reset in spirit (not letter -- see
    # the accompanying README methodology note on this baseline's
    # simplification).
    invalid_base = codes < 0
    codes = np.where(invalid_base, 0, codes).astype(np.uint64)

    # Build every k-length window's packed 2-bit value via k shifted views
    # summed with the right power-of-4 weight -- vectorized over all
    # windows in the read at once, not a per-window Python loop.
    packed = np.zeros(n - k + 1, dtype=np.uint64)
    for j in range(k):
        packed = (packed << np.uint64(2)) | codes[j:n - k + 1 + j]

    if invalid_base.any():
        window_bad = np.zeros(n - k + 1, dtype=bool)
        for j in range(k):
            window_bad |= invalid_base[j:n - k + 1 + j]
        packed = packed[~window_bad]

    return packed


MASK2 = np.uint64(0x3333333333333333)
MASK4 = np.uint64(0x0F0F0F0F0F0F0F0F)


MASK8 = np.uint64(0x00FF00FF00FF00FF)
MASK16 = np.uint64(0x0000FFFF0000FFFF)
MASK32 = np.uint64(0x00000000FFFFFFFF)


def swap_bytes_u64(v):
    # The vectorized equivalent of Rust's u64::swap_bytes(): reverse the 8
    # bytes of each 64-bit lane via three mask-and-shift passes (8 -> 16 ->
    # 32-bit swaps), not a dtype/byteswap() trick (an earlier version of
    # this baseline used astype(">u8").byteswap(), which does not do what
    # it looks like it does and silently corrupted every reverse complement
    # -- caught by cross-checking this baseline's distinct-k-mer count
    # against naive_python.py on the same subset).
    v = ((v & MASK8) << np.uint64(8)) | ((v >> np.uint64(8)) & MASK8)
    v = ((v & MASK16) << np.uint64(16)) | ((v >> np.uint64(16)) & MASK16)
    v = ((v & MASK32) << np.uint64(32)) | (v >> np.uint64(32))
    return v


def revcomp_u64_array(kmers, k):
    v = ~kmers
    v = ((v >> np.uint64(2)) & MASK2) | ((v & MASK2) << np.uint64(2))
    v = ((v >> np.uint64(4)) & MASK4) | ((v & MASK4) << np.uint64(4))
    v = swap_bytes_u64(v)
    return v >> np.uint64(64 - 2 * k)


def count_file(path, k):
    seqs = read_fastq_sequences(path)
    all_kmers = []
    for seq in seqs:
        packed = encode_kmers_for_read(seq, k)
        if packed is not None and len(packed):
            all_kmers.append(packed)
    kmers = np.concatenate(all_kmers) if all_kmers else np.array([], dtype=np.uint64)
    rc = revcomp_u64_array(kmers, k)
    canon = np.minimum(kmers, rc)
    uniq, counts = np.unique(canon, return_counts=True)
    return len(seqs), len(uniq), int(counts.sum())


if __name__ == "__main__":
    path, k = sys.argv[1], int(sys.argv[2])
    t0 = time.perf_counter()
    reads, distinct, total = count_file(path, k)
    elapsed = time.perf_counter() - t0
    print(f"reads={reads} distinct_kmers={distinct} total_kmers={total} elapsed={elapsed:.4f}s")
