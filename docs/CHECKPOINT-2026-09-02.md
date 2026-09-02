# Checkpoint — 2026-09-02

Written to hand this work to the next session. The goal that drove every
commit below, in the user's words: **do not publish until we are the best at
the thing we said we do — being the ones who audit.**

Nothing has been pushed. 52 commits sit on local `master` (`5aadfb1..HEAD`),
tree clean.

---

## The one thing to do next

Run the study. It is written, tested, and waiting:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
python scripts/validation/leakage_survey.py --max-cohorts 30 --json survey.json
```

Roughly four hours. It is resumable — re-running the same command picks up
where it stopped, and counted cohorts are cached under `cache/cohort_counts/`
so a re-run that changes the *analysis* starts at the audit rather than at
counting.

What it produces is the paper's headline: across N published
species-antibiotic cohorts, the mean AUC falls from X (random CV) to Y
(lineage-blocked). Nothing else in this repository is blocked on anything
else; this number is.

---

## What this session established

### The tool was validated in two layers, and the layers found real bugs

Layer one compares against an **external implementation** of the same
algorithm. Layer two, where no reference tool exists, compares against
**constructed truth**.

| checked | against | result |
|---|---|---|
| k-mer counting | KMC3 3.2.4 | **exact equality** — 19,062,700 distinct / 163,051,083 total, identical |
| HyperLogLog | the exact count | −0.26% |
| ntCard | the exact count | −0.63 / +2.04 / −1.45% |
| genome size | GenomeScope2 | −0.29% |
| MinHash distances | Mash 2.3 | r = 0.997, bias ~0, error shrinks 3.06× for a 10× sketch (√10 = 3.16) |
| assembly QV | Merqury 1.4.1 | 18.4192 vs 18.4205 |
| species classification | published identity | 12/12 held-out genomes |
| 22 further modules | constructed truth | `scripts/validation/known_truth_checks.py` |

**Eight defects were found this way, and none of them was reachable by the
~1,750 tests.** They share one signature: *the wrong output was well-formed*.

- `load_amr()` returned the **first 25 contigs** of every assembly — 26% of
  an *E. coli* genome. The truncated FASTA starts with `>`, is multi-contig,
  and parses fine. Only comparing base counts against the species' known
  genome size distinguishes it. (`a868ae6`)
- `audit()`'s hardcoded `lineage_threshold = 0.01` put 198 of 200 genomes in
  their own lineage — `LineageKFold` had nothing to block on — and still
  returned a small, well-formed `gap` reading as "no leakage". The same
  constant merged a different cohort into 15 groups where 38 sequence types
  exist. (`ab85899`, `905298d`)
- `genomescope`'s end-to-end test accepted any answer between 2,500 and
  10,000 bp on a synthetic 5 kb genome, so its central claim could not fail.
- `validate_generated` called identical distributions "very deviated" —
  Jensen-Shannon divergence is coverage-dependent, ~0.42/√ratio under the
  null. (`eaaa75b`)
- `profile`'s default summary was written to the process's working directory
  rather than beside `--output`, contradicting its own docs. (`7ad7d81`)

### One claim was retracted

Track D originally reported a leakage gap on real AMR data. It was an
artefact of the truncation bug above. A three-way control isolated
truncation as the cause and the claim was withdrawn (`69e60d3`), along with
the number that had been used to justify the threshold change (`2be95ae`).
The retraction is in the repository, not just in this file.

### The audit is a triad, not a single number

The session's most useful finding is that **`gap` alone is not enough**, and
the docs now say so out loud:

| meter | detects | blind to |
|---|---|---|
| `gap` | dominant leakage | partial leakage; and it goes *positive under plain overfitting too* |
| `confounding` | how much the phenotype tracks lineage — Spearman ρ = **1.000** against injected leakage | nothing yet found |
| `explain` | *where* a class sits: 3 lineages vs 1, localised to a gene vs scattered | — |

`confounding` covers exactly the range `gap` cannot see (`6dbe7c8`,
`86ef39c`). `explain()` gained spatial reporting — "localised to geneA" vs
"scattered over N bp" (`642eb13`). `fastdna.design.check_design()` is new
(`9f6ac18`) and runs *before* an experiment, so an unanswerable design is
refused rather than run for twenty minutes.

### Speed: counting is now paid once, ever

`count_cohort()` already removed the recount between folds (measured: **zero**
`fastdna.count()` calls in a 3-fold CV). This session added
`CohortCounts.save()/load()` (`e3c3c35`), which extends that across
*processes* — closing the G-6 gap in `docs/audit/ml-gaps.md`. One Parquet
file, cohort structure in footer metadata under the `fastdna.` prefix
`src/ktab.rs` established, so the file is still a valid k-mer table to any
reader that has never heard of `CohortCounts`. `load()` validates rather than
trusts: `row_counts` that do not sum to the row count would slice samples
apart at the wrong offsets and return a confident, wrong answer.

---

## Design decisions that are settled — do not relitigate

**n = 80 genomes per cohort, 30 cohorts, 500 features.** This is statistical,
not a compromise. One cohort of 80 gives a 95% CI on AUC of ±0.107. The
*mean* of 30 such cohorts gives ±0.107/√30 = **±0.020** — tighter than one
cohort of 300 would give, and 300 does not fit in 18 GB with complete
bacterial genomes (measured: 200 drove this machine to 7.6 GB of swap at 15%
CPU). Aggregation buys the precision the RAM cannot, *and* makes the claim
stronger: a result across 30 species-antibiotic pairs is evidence about the
field; one cohort is an anecdote.

An earlier hypothesis — that p/n was the problem — **was wrong**: 200
features scored AUC 0.533, worse than 5000's 0.567. What the data actually
showed is that the `gap` correlates with p/n *on signal-free data*, which is
the overfitting confound now documented in `audit.py`'s docstring.

---

## Open items, in the order they matter

1. **Run `leakage_survey.py`.** Everything above is preparation for it.
2. **Kraken 2 head-to-head** for `taxonomy` — the one reference-tool
   comparison layer one is missing.
3. **bioLeak numeric comparison — blocked.** bioLeak 0.3.8 installs and
   loads in `bioconductor/bioconductor_docker:RELEASE_3_20`, but the
   comparison never ran: four successive failures (Bioconductor
   `SummarizedExperiment` deps, then `group` expecting a column name, then a
   `delta_lsi` signature mismatch, finally a missing `glmnet` learner). The
   exported API is known — `audit_leakage`, `delta_lsi`, `dlsi_ci`,
   `audit_perm_gap`, `make_split_plan` and 15 others. This is the closest
   prior art and the comparison is worth finishing.
4. **The publishing decision.** Still deliberately withheld.

---

## Environment

```bash
export PATH="$HOME/.cargo/bin:$PATH"    # cargo/maturin are not on the default PATH
.venv/bin/python -m pytest python/tests -q
cargo test --release
```

State at this checkpoint: **1,035 Python tests + 729 Rust passing**, clippy
at its 8-site baseline (the ratchet), 1 intentional xfail.
