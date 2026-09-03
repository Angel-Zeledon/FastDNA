# Checkpoint — 2026-09-03

Continues `CHECKPOINT-2026-09-02.md`, which named one thing to do next:
*run the leakage survey*. It was right that nothing else was blocking it.
It was wrong that the survey was ready to run.

---

## The survey was measuring nothing, and said so in a well-formed way

The 2026-09-02 run completed one cohort before stopping. Its row:

```json
{"species": "Streptococcus pneumoniae", "antibiotic": "penicillin",
 "score_random": 0.5, "score_lineage": 0.5, "gap": 0.0}
```

Read as "this cohort has no leakage". It is the signature of a defect, not
a finding — every field in range, and the reading exactly wrong. That row
is archived as `survey.INVALID-presence-ranking-defect.json`; do not cite
it.

**Finding 9** (`python/tests/test_review_findings_2026_09_02.py`, fixed in
`2dcd669`): `KmerVectorizer`'s two defaults were individually well-reasoned
and jointly degenerate. Ranking by descending prevalence puts the k-mers
present in *every* sample at the top; `representation="presence"` encodes
those as 1 in every row. The matrix handed to the classifier was literally
all ones — 460,795 k-mers are present in all 80 *S. pneumoniae* genomes,
against a budget of 500.

The rule now splits by representation: `presence` ranks by
`min(prevalence, n_samples - prevalence)` — for a binary feature,
informativeness *is* variance, maximised at prevalence n/2 — and drops the
constants outright, so a generous `top_features` cannot refill the
vocabulary with them. Count-valued representations keep the prevalence
ranking, where a universal k-mer really does still vary. The same pair is
implemented Rust-side for `disk_backed=True`
(`cohort_vocab::VocabularyRanking`), with equivalence tests that fail if
the two drift apart.

`fastdna.gwas` and `src/cohort/matrix.rs` already ranked by minor-sample
count. `KmerVectorizer` was the one place that did not.

## What the same cohort says now

Same genomes, same cached counts, same `audit()` call — only the
vocabulary rule changed:

| | 2026-09-02 | 2026-09-03 |
|---|---|---|
| `score_random` | 0.5000 | **0.7875** |
| `score_lineage` | 0.5000 | **0.6485** |
| `gap` | 0.0000 | **+0.139** |
| `confounding` | 0.451 | 0.451 (unchanged — it never used the vectorizer) |

Signal at 0.79 with a +0.14 drop under lineage blocking is the shape the
study went looking for: by `audit.py`'s own rule (`score_random` far above
chance *and* a large gap), the leakage reading rather than the capacity
one. One cohort is an anecdote; the survey's whole point is the mean over
30. But the instrument now reads.

Cost with counts cached: **113 s/cohort**, versus 329 s when the counting
was paid.

## Recognising a contaminated run, elsewhere

An all-ones matrix scores **exactly** 0.5000 in every fold (constant
scores; verified directly). So a `score_random` that is not 0.5000 did not
come from one. That rules the defect *out* as the explanation for
`audit.py`'s p/n table (0.5326 / 0.5666 on *E. coli* + ampicillin) — an
earlier draft of the regression test's docstring claimed otherwise, and
that claim was corrected rather than propagated. The table is still marked
**pending re-measurement**: it was measured under the old ranking, so its
three commands select different features today.

Why that cohort escaped: 80 diverse *E. coli* assemblies share far fewer
exact 31-mers than 80 *S. pneumoniae* genomes, so its prevalence-ranked top
features were *near*-universal rather than universal. Severity scales with
how much exact core a cohort shares — total for clonal cohorts, partial for
diverse ones, never in a direction that helps.

---

## Open items, in the order they matter

1. **Run `leakage_survey.py`.** Unchanged from yesterday, now for real:

   ```bash
   export PATH="$HOME/.cargo/bin:$PATH"
   python scripts/validation/leakage_survey.py --max-cohorts 30 --json survey.json
   ```

   Resumable. The two cohorts already counted are cached under
   `cache/cohort_counts/` (4.7 GB), so those two start at the audit.
   Delete nothing: the counting is unaffected by the fix, only the ranking
   downstream of it was wrong.

2. **Re-measure `audit.py`'s p/n table** under the new rule
   (`scripts/validation/lineage_leakage_experiment.py`, varying only
   `--top-features`). Cheap, and it is the only observation behind the
   "overfitting also produces a gap" claim.

3. **Kraken 2 head-to-head.**
   `scripts/validation/metagenomics_vs_kraken2.py` is written and its CLI
   runs; it drives Kraken 2 through Docker (`staphb/kraken2:2.1.3`, which
   is available on this machine) and has **never been executed end to
   end**. Two arms: `-k 31 -l 31 -s 0` (same algorithm, where disagreement
   is a bug in one of the two) and Kraken 2's own defaults (which puts a
   number on the sensitivity cost of exact k-mers).

4. **bioLeak numeric comparison — still blocked** (see yesterday's
   checkpoint for the four failure modes).

5. **The publishing decision.** Still deliberately withheld.

---

## Environment

```bash
export PATH="$HOME/.cargo/bin:$PATH"    # cargo/maturin are not on the default PATH
.venv/bin/python -m maturin develop --release --features python
.venv/bin/python -m pytest python/tests -q -m "not network"
cargo test --all-targets
```

Verified against this working tree: **1,041 Python passing (1 xfail)** and
**735 Rust passing (0 failed, 1 intentionally ignored)**, clippy at exactly
its 8-site baseline.
