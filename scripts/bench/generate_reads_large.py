"""Vectorized (NumPy) synthetic FASTQ generator for multi-GB benchmark
datasets -- scripts/bench/generate_reads.py's pure-Python per-base loop
does not scale to this size in reasonable time (extrapolated: ~1 hour for
2GB). Same statistical model (Illumina-like quality decay, Phred-derived
substitution rate, both strands sampled), vectorized with NumPy and
processed in bounded-memory chunks so output size does not bound peak RAM.

Usage: gen_large_fastq.py <genome_size> <coverage> <read_len> <out_path> <seed> [chunk_reads]
"""
import sys
import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
COMP = np.zeros(256, dtype=np.uint8)
for a, b in zip(b"ACGT", b"TGCA"):
    COMP[a] = b


def main():
    genome_size = int(sys.argv[1])
    coverage = float(sys.argv[2])
    read_len = int(sys.argv[3])
    out_path = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1337
    chunk_reads = int(sys.argv[6]) if len(sys.argv) > 6 else 200_000

    rng = np.random.default_rng(seed)

    genome = BASES[rng.integers(0, 4, size=genome_size, dtype=np.int64)]

    n_reads = int(genome_size * coverage / read_len)

    pos = np.arange(read_len)
    base_q_curve = 38 - (13 * (pos / read_len) ** 2)

    written = 0
    with open(out_path, "wb") as f:
        while written < n_reads:
            n = min(chunk_reads, n_reads - written)

            starts = rng.integers(0, genome_size - read_len + 1, size=n)
            idx = starts[:, None] + pos[None, :]
            reads = genome[idx].copy()

            rc_mask = rng.integers(0, 2, size=n).astype(bool)
            if rc_mask.any():
                rc_reads = COMP[reads[rc_mask][:, ::-1]]
                reads[rc_mask] = rc_reads

            quals = np.broadcast_to(base_q_curve, (n, read_len)).copy()
            dip_mask = rng.random((n, read_len)) < 0.02
            dip_amount = rng.integers(5, 16, size=(n, read_len))
            quals = quals - np.where(dip_mask, dip_amount, 0)
            quals = np.clip(quals, 2, 40).astype(np.int64)

            p_err = 10.0 ** (-quals / 10.0)
            err_mask = rng.random((n, read_len)) < p_err
            if err_mask.any():
                base_idx = np.searchsorted(BASES, reads)
                shift = rng.integers(1, 4, size=(n, read_len))
                new_idx = (base_idx + shift) % 4
                reads = np.where(err_mask, BASES[new_idx], reads)

            qual_bytes = (quals + 33).astype(np.uint8)

            lines = []
            ids = np.arange(written, written + n)
            for i in range(n):
                lines.append(b"@read%d_pos%d\n" % (ids[i], starts[i]))
                lines.append(reads[i].tobytes())
                lines.append(b"\n+\n")
                lines.append(qual_bytes[i].tobytes())
                lines.append(b"\n")
            f.write(b"".join(lines))

            written += n

    print(f"wrote {n_reads} reads, {n_reads * read_len} bases, genome={genome_size}bp, cov={coverage}x -> {out_path}")


if __name__ == "__main__":
    main()
