# `scripts/validation/`

Checks FastDNA against things outside FastDNA: the established
implementation of each algorithm it borrows, and organisms whose properties
are published. Run on a schedule by
[`.github/workflows/validation.yml`](../../.github/workflows/validation.yml).

## Why this exists separately from `tests/` and `scripts/bench/`

The test suites answer "does the code do what its author intended". These
answer "is what the author intended actually right", which no amount of unit
testing can reach. Every serious defect found in this project came from here
rather than from the test suite, and each was invisible to assertions about
*shape*: a truncated FASTA still starts with `>`, a well-formed number is
still a number. A plausible answer is not a correct one, and only a source
of truth outside the code can tell the two apart.

`scripts/bench/` is the neighbouring directory and answers a third question
-- how fast -- against baselines this project wrote itself, on generated
data. Useful, and not a correctness check.

## The scripts

| script | checks | against |
|---|---|---|
| `kmc3_equivalence.py` | k-mer counting | KMC3 3.2.4, **exact equality** |
| `estimator_accuracy.py` | HyperLogLog, ntCard spectrum | the exact count from the same file |
| `sketch_vs_mash.py` | MinHash distances | Mash 2.3 |
| `similarity_vs_kmc_tools.py` | exact Jaccard and containment between tables | KMC3's own set operations, **exact equality** |

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
  documentation rather than from what it happens to score. A stated bound
  that nothing measures is a promise, not a property.
- **Bias and scaling** for MinHash. Demanding equality there would demand
  FastDNA reproduce Mash's *sampling noise*; what a correct implementation
  owes instead is no systematic offset, and error that shrinks as
  `1/sqrt(sketch_size)`. Both are asserted.

Writing an assertion looser than the claim would make these decorative;
writing one tighter than the algorithm permits would make them flaky. Both
failures are worse than no script.

## What is not covered

No read set with real 3'-end quality decay, which means the quality-trimming
path is exercised by none of these -- the one gap in this directory that is
about the counting engine itself rather than about something the project no
longer ships.
