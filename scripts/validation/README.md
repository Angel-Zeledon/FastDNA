# `scripts/validation/`

Checks FastDNA against things outside FastDNA: the established
implementation of each algorithm it borrows, and organisms whose properties
are published. Run on a schedule by
[`.github/workflows/validation.yml`](../../.github/workflows/validation.yml).

## Why this exists separately from `tests/` and `scripts/bench/`

The test suites answer "does the code do what its author intended". These
answer "is what the author intended actually right", which no amount of unit
testing can reach. Every finding of the session that created this directory
came from here rather than from the ~1,700 tests, and each was invisible to
assertions about *shape*:

- `load_amr` was returning the first 25 contigs of every assembly -- 26% of
  an *E. coli* genome. The truncated FASTA starts with `>`, is multi-contig
  and parses fine. Only comparing its base count against the species' known
  genome size distinguishes it.
- `audit()` reported `gap = -0.011` ("no leakage") on a cohort whose real
  gap is +0.165. A small float is a perfectly valid float.
- `genomescope`'s end-to-end test accepted any answer between 2,500 and
  10,000 bp on a synthetic 5 kb genome, so its central claim could not fail.

`scripts/bench/` is the neighbouring directory and answers a third question
-- how fast -- against baselines this project wrote itself, on generated
data. Useful, and not a correctness check.

## The scripts

| script | checks | against |
|---|---|---|
| `kmc3_equivalence.py` | k-mer counting | KMC3 3.2.4, **exact equality** |
| `estimator_accuracy.py` | HyperLogLog, ntCard, genome size | the exact count, and GenomeScope2 |
| `sketch_vs_mash.py` | MinHash distances | Mash 2.3 |
| `assembly_qc_vs_merqury.py` | assembly QV | Merqury 1.4.1 |
| `taxonomy_real_species.py` | species classification | published species identity |
| `lineage_leakage_experiment.py` | the leakage audit end to end | MLST as ground truth |

Reference tools arrive as **pinned biocontainers**, not manual builds, so a
rerun a year from now compares against the same versions rather than
whatever is current.

## Running them

```bash
cargo build --release                        # every script needs this
maturin develop --release --features python  # and this
python scripts/validation/kmc3_equivalence.py
```

Docker is required for every script that names an external tool. Data
downloads are cached under `~/.fastdna/` and shared between scripts, so the
first run is slow and later ones are not.

## What "passing" means here

Each script asserts the tightest claim its subject can actually support, and
the difference matters:

- **Exact equality** for counting. With trimming off and singletons kept,
  FastDNA and KMC3 solve the identical problem, so any difference is a bug.
- **A stated bound** for the estimators, taken from each one's own
  documentation rather than from what it happens to score.
- **Bias and scaling** for MinHash. Demanding equality there would demand
  FastDNA reproduce Mash's *sampling noise*; what a correct implementation
  owes instead is no systematic offset, and error that shrinks as
  `1/sqrt(sketch_size)`. Both are asserted.

Writing an assertion looser than the claim would make these decorative;
writing one tighter than the algorithm permits would make them flaky. Both
failures are worse than no script.

## What is not covered

Listed in `docs/validation-real-data.md` under "What Track C does not
cover": no Kraken 2 head-to-head, no diploid genome-size case, and no read
set with real 3'-end quality decay -- which means the quality-trimming path
is exercised by none of these.
