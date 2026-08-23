"""Generates a synthetic FASTQ dataset for benchmarking, deterministically
(fixed seed), so numbers are reproducible.

Simulates: a random "reference genome" of `genome_size` bases, then reads of
`read_len` bases sampled uniformly across it to `coverage`x depth, with a
realistic Illumina-like quality profile (high quality, gently decaying
towards the 3' end, occasional low-quality dips) and a `error_rate` per-base
substitution rate applied according to that quality (lower quality => more
likely to be an error) -- so the file has a genuine two-peak k-mer frequency
spectrum (sequencing errors at low depth, true genome content at the
coverage peak), not uniformly-random noise.
"""
import random
import sys

BASES = "ACGT"


def make_genome(size, rng):
    return "".join(rng.choice(BASES) for _ in range(size))


def quality_profile(read_len, rng):
    # Phred+33, starts ~Q38, decays gently to ~Q25 by the read end, with
    # occasional single-base dips -- a coarse but standard approximation of
    # an Illumina quality curve.
    quals = []
    for pos in range(read_len):
        base_q = 38 - int(13 * (pos / read_len) ** 2)
        if rng.random() < 0.02:
            base_q -= rng.randint(5, 15)
        base_q = max(2, min(40, base_q))
        quals.append(base_q)
    return quals


def mutate(seq, quals, rng):
    out = list(seq)
    for i, q in enumerate(quals):
        # Error probability derived from the Phred score itself: p = 10^(-Q/10).
        p_err = 10 ** (-q / 10)
        if rng.random() < p_err:
            out[i] = rng.choice([b for b in BASES if b != out[i]])
    return "".join(out)


def revcomp(seq):
    comp = str.maketrans("ACGT", "TGCA")
    return seq.translate(comp)[::-1]


def main():
    genome_size = int(sys.argv[1])
    coverage = float(sys.argv[2])
    read_len = int(sys.argv[3])
    out_path = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1337

    rng = random.Random(seed)
    genome = make_genome(genome_size, rng)
    n_reads = int(genome_size * coverage / read_len)

    with open(out_path, "w") as f:
        for i in range(n_reads):
            start = rng.randint(0, genome_size - read_len)
            frag = genome[start:start + read_len]
            if rng.random() < 0.5:
                frag = revcomp(frag)
            quals = quality_profile(read_len, rng)
            frag = mutate(frag, quals, rng)
            qual_str = "".join(chr(q + 33) for q in quals)
            f.write(f"@read{i}_pos{start}\n{frag}\n+\n{qual_str}\n")

    print(f"wrote {n_reads} reads, {n_reads * read_len} bases, genome={genome_size}bp, cov={coverage}x -> {out_path}")


if __name__ == "__main__":
    main()
