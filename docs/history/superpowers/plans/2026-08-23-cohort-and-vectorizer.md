# Cohort Engine and KmerVectorizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn FastDNA from "counts k-mers" into "is a scikit-learn component" — the capability no competitor offers, and the one that makes v1 worth switching to.

**Architecture:** A cohort engine that processes a directory of samples into one matrix, a Rust primitive that projects new samples onto a fixed vocabulary, and a pure-Python scikit-learn transformer built on top of it. Feature selection lives inside `fit`, so a `Pipeline` under cross-validation cannot leak.

**Tech Stack:** Rust 2021 (rayon, rustc-hash, arrow), PyO3, scikit-learn (Python-side only, as a soft dependency).

**Spec:** `docs/history/superpowers/specs/2026-08-22-fastdna-python-design.md` — §7 (cohort engine), §7.8 (vocabulary projection), §9.3/§9.6 (Cohort and KmerVectorizer). Phases E and F of §15.

**Depends on:** the packaging phase (`2026-08-22-python-packaging-v1.md`) being complete, since everything here crosses the FFI boundary it establishes.

## Global Constraints

- **English only** — code, comments, doc comments, commit messages, docs. No exceptions.
- **No `println!`, `eprintln!`, `.unwrap()`, `.expect()` under `src/` except `src/main.rs`.** Enforced by clippy lints in `Cargo.toml`; a violation fails the build.
- **Every public core function returns `Result<T, FastDnaError>`.**
- **Raw counts only.** Rust never normalizes. It emits counts plus per-sample depth so normalization happens inside the CV loop, where it neither leaks nor freezes a statistical choice into a binary.
- **Feature selection is unsupervised** — ranked by prevalence, never by correlation with the label. Rust never receives labels, so it cannot leak even by accident.
- **`total_kmers` is the normalization basis** and must survive pruning unchanged.
- Existing behaviour must not regress: `cargo test`, `cargo clippy --all-targets`, and `pytest python/tests` stay green.

---

## Why This Plan Is The Product

The competitive position, from research done 2026-08-22:

- **oxli** (Rust + PyO3, from the sourmash lab) has `count`/`consume`/`get`. No cohort, no matrix, no ML, no Arrow. It is the architectural twin and the maturity leader.
- **iMOKA** is the functional rival: cohorts → k-mer matrix → random forest. But it is a Docker/Singularity workflow in C++/Java, not something you `import`. **And its feature reduction runs across the whole cohort before classification — which is data leakage.**
- **KMC3** wins on raw counting speed and always will; it has a decade of disk-partitioned counting behind it.
- **sourmash** owns sketching.

Every piece exists. Nobody has joined them. `KmerVectorizer` is the join, and its leakage guarantee is a **correctness** claim rather than a benchmark claim — far harder for a rival to refute or copy without restructuring.

---

## File Structure

**Created:**
- `src/cohort/discovery.rs` — find samples, group R1/R2 pairs. One job.
- `src/cohort/engine.rs` — the two-pass count/prevalence/spill engine.
- `src/cohort/select.rs` — prevalence ranking and deterministic feature selection.
- `src/cohort/matrix.rs` — wide and sparse Arrow output.
- `src/cohort/mod.rs` — re-exports; `SampleMeta`, `CohortMatrix`, `CohortOpts`.
- `src/project.rs` — `count_projected`, the fixed-vocabulary primitive.
- `python/fastdna/sklearn.py` — `KmerVectorizer`.
- `python/fastdna/normalize.py` — CPM, CLR, log1p.
- Tests per task.

**Modified:**
- `src/lib.rs`, `src/ffi.rs`, `python/fastdna/__init__.py`.

The cohort engine is split across five files rather than one because a single `cohort.rs` would exceed what can be held in context at once, and discovery/selection/matrix have genuinely separate responsibilities and separate tests.

---

### Task 1: Sample discovery and R1/R2 pairing

Getting this wrong corrupts every downstream result silently, so it lands first and alone.

**Files:**
- Create: `src/cohort/discovery.rs`, `src/cohort/mod.rs`
- Modify: `src/lib.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Produces: `pub struct SampleFiles { pub sample_id: String, pub files: Vec<PathBuf> }` and `pub fn discover_samples(dir: &Path) -> Result<Vec<SampleFiles>>`

**Behaviour:**

| Input | Result |
|---|---|
| `pat_001_R1.fastq.gz` + `pat_001_R2.fastq.gz` | one sample, two files |
| `pat_001.fastq.gz` | one single-end sample |
| `pat_001_R1.fastq.gz` with no `_R2` | one sample, **and a recorded orphan warning** |
| directory with no recognizable FASTQ | `Err(NoSamplesFound)` |

Recognized extensions: `.fastq`, `.fq`, `.fastq.gz`, `.fq.gz`, matched case-insensitively. Pair suffixes: `_R1`/`_R2`/`_1`/`_2`, also with `.` as separator. `sample_id`s are sorted lexicographically so **matrix row order is reproducible across runs and platforms** — without that, two runs on the same data produce matrices whose rows do not correspond, and a saved model silently mispredicts.

**The Illumina form must pair, and an earlier version of this plan got it wrong.**
`sample_S1_L001_R1_001.fastq.gz` is the most common real-world FASTQ filename
shape, and its `_R1` is not the trailing component — `_001` is. A rule that only
matches trailing suffixes assigns it no pair role, its `_R2_001` mate becomes a
*separate sample with a different id*, and because neither carries a role **no
orphan warning fires**. That turns 500 patients into 1000 half-samples silently,
which is the exact failure this task exists to prevent.

So: after stripping the extension, a stem matching `<prefix>_R[12]_<digits>`
(an `_R1`/`_R2` followed by an underscore and a numeric-only run to the end)
yields sample id `<prefix>` with the corresponding role. Trailing-suffix rules
apply to everything else. `pat_R1_extra.fastq`, whose trailing part is not
numeric, stays single-end.

This plan originally required `pat_R1_001.fastq` to be treated as single-end,
as a guard against a naive `contains("_R1")` implementation. That guard was
right in intent and wrong in effect: it excluded the one convention that matters
most. Keep testing that `contains` is not used — via `pat_R1_extra.fastq` —
without breaking Illumina.

An orphan is a warning, not an error, but it must be *recorded* and surfaced on the result — a silent single-end fallback is how a half-loaded cohort looks successful.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(names: &[&str]) -> tempfile::TempDir { /* create empty files */ }

    #[test]
    fn pairs_r1_and_r2_into_one_sample() {
        let d = fixture(&["pat_001_R1.fastq.gz", "pat_001_R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_001");
        assert_eq!(s[0].files.len(), 2);
    }

    #[test]
    fn unsuffixed_file_is_a_single_end_sample() {
        let d = fixture(&["pat_001.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].files.len(), 1);
    }

    #[test]
    fn orphan_r1_is_processed_but_recorded() {
        let d = fixture(&["pat_001_R1.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert!(!s[0].orphan_warning.is_empty(), "an orphan must not be silent");
    }

    #[test]
    fn sample_order_is_lexicographic_and_reproducible() {
        let d = fixture(&["pat_010.fastq", "pat_002.fastq", "pat_001.fastq"]);
        let ids: Vec<_> = discover_samples(d.path()).expect("valid")
            .into_iter().map(|s| s.sample_id).collect();
        assert_eq!(ids, vec!["pat_001", "pat_002", "pat_010"]);
    }

    #[test]
    fn empty_directory_is_an_error_not_an_empty_cohort() {
        let d = fixture(&[]);
        assert!(matches!(discover_samples(d.path()), Err(FastDnaError::NoSamplesFound { .. })));
    }

    #[test]
    fn dots_and_dashes_in_names_do_not_confuse_pairing() {
        let d = fixture(&["p-1.a_R1.fastq.gz", "p-1.a_R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1, "got {:?}", s);
    }
}
```

`tempfile` is a dev-dependency only — add it under `[dev-dependencies]`, not `[dependencies]`, so it never ships in a wheel.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib cohort::discovery`
Expected: `cannot find function discover_samples`.

- [ ] **Step 3: Implement, then verify**

Run: `cargo test && cargo clippy --all-targets`
Expected: green, clean.

- [ ] **Step 4: Commit**

```bash
git commit -m "Add cohort sample discovery with R1/R2 pairing

Counting R1 and R2 as separate patients is a silent error that corrupts
every downstream result, so pairing is its own tested unit. Sample ids sort
lexicographically because matrix row order must be reproducible across runs
and platforms -- otherwise a saved model silently mispredicts on a
re-generated matrix. An orphaned R1 is recorded rather than silently treated
as single-end."
```

---

### Task 2: The cohort engine

**Files:**
- Create: `src/cohort/engine.rs`
- Test: `tests/cohort_engine.rs`

**Interfaces:**
- Consumes: `discover_samples`, `process_stream_parallel`, `KmerCounter::prune`.
- Produces: `SampleMeta`, and `count_cohort(dir, opts, progress, cancel) -> Result<CohortRaw>` carrying per-sample counters (in RAM or spilled) plus the global prevalence table.

**Parallelism: across samples, not within.** 500 samples are 500 independent units; each is processed on a sequential path and rayon distributes samples across threads. Parallelizing inside each one *on top of* that oversubscribes the pool and adds synchronization for no gain. `process_stream_parallel` stays for single-sample `count`, where it is the right strategy.

```rust
pub struct SampleMeta {
    pub sample_id: String,
    pub files: Vec<PathBuf>,
    pub total_reads: u64,
    pub total_kmers: u64,      // after trim, before prune -- the normalization basis
    pub distinct_raw: u64,
    pub distinct_pruned: u64,
    pub dropped_min: u64,
    pub dropped_max: u64,
}
```

**Fast path vs spill.** Estimate `Σ distinct_pruned × 12 bytes`. Below `max_ram` (default 4 GB), keep everything in memory and never touch disk. Above it, spill each sample's counts to a temp file as raw little-endian `(u64, u32)` pairs — not Parquet, because temp files are read exactly once and never inspected, so schema and compression overhead buy nothing.

A `Drop` guard removes temp files on error and on a panic propagating into Python. Leaving hundreds of gigabytes of scratch behind after a failed run is its own bug.

**Guards.** If the prevalence table would exceed `max_vocab_ram` (default 2 GB), return `VocabTooLarge` with the estimate and a suggestion — never attempt it.

- [ ] **Step 1: Write the failing tests** — a synthetic 6-sample cohort with hand-computed counts; assert per-sample `total_kmers` is unchanged by pruning; assert the RAM path and the spill path produce **byte-identical** output for the same input (force the spill with a tiny `max_ram`). That last one is the invariant that stops the engine having two behaviours.
- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Verify** — `cargo test && cargo clippy --all-targets`
- [ ] **Step 5: Commit**

---

### Task 3: Feature selection and matrix output

**Files:**
- Create: `src/cohort/select.rs`, `src/cohort/matrix.rs`
- Test: inline + `tests/cohort_matrix.rs`

**Interfaces:**
- Produces: `select_features(prevalence, top_n) -> Vec<u64>` (sorted); `CohortMatrix` with `to_arrow_wide()` and `to_arrow_sparse()`.

**Ranking:** prevalence descending, ties broken by total count, then by `kmer_u64`. The final tie-break exists so selection is **deterministic** — without it, two runs on identical data can pick different features and produce incompatible matrices.

**Wide format:** dense `n_samples × top_n`. Guard: `n_samples × n_features × 4` bytes against `max_matrix_bytes` (default 4 GB); over it, return `MatrixTooLarge` naming the estimate and suggesting a lower `top_features` or `sparse`. **A matrix that does not fit is never attempted.**

**Semantics of zero, and this must be documented on the Python side too:** a cell is `0` both when the k-mer was absent and when it was present but pruned below `min_count`. Per-sample pruning turns sub-threshold counts into explicit zeros. That is correct for the goal — those counts are sequencing noise — but a user interpreting sparsity must know some zeros mean "filtered", not "absent".

**Sparse format:** two Arrow tables (triplets + vocabulary), reconstructed in Python with one `scipy.sparse.csr_matrix` call. XGBoost and scikit-learn consume CSR directly.

- [ ] Steps 1-5 as above, with a test asserting wide and sparse encode **the same data**, and a test that provokes ties to pin determinism.

---

### Task 4: `count_projected` — the enabler

**Files:**
- Create: `src/project.rs`
- Test: `tests/projection.rs`

**Interfaces:**
- Produces: `count_projected(samples, vocabulary: &[u64], opts) -> Result<CohortMatrix>`

Counts new samples and projects them onto an **externally supplied** vocabulary rather than deriving one. K-mers outside the vocabulary are discarded; vocabulary entries the sample lacks become 0. Column order follows the vocabulary exactly.

Simpler than `count_cohort`: no prevalence pass, no selection, therefore **no spill** — each sample is counted, projected, released. One pass, memory bounded by vocabulary size.

**Without this there is no `transform`, and without `transform` there is no scikit-learn integration and no structural leakage guarantee.** It is the enabler for the whole product claim.

- [ ] Steps 1-5, with tests that k-mers absent from the vocabulary are dropped, that column order matches the vocabulary exactly, and that a sample sharing nothing with the vocabulary yields an all-zero row rather than an error.

---

### Task 5: `Cohort` across the FFI boundary

**Files:**
- Modify: `src/ffi.rs`, `python/fastdna/__init__.py`
- Create: `python/fastdna/normalize.py`
- Test: `python/tests/test_cohort.py`

**Interfaces:**
- Produces: `fastdna.count_cohort(dir, *, k, min_count, max_count, format, top_features, threads, progress) -> Cohort` with `.matrix`, `.samples`, `.vocabulary`, `.dropped`, `.to_pandas()`, `.to_numpy()`, `.to_scipy()`, `.normalize(method)`, `.save()`, and `fastdna.load_cohort()`.

`.samples` travels attached to `.matrix` rather than as a loose file: it is harder to normalize incorrectly when depth sits in the same object as the counts. `.normalize()` therefore needs no extra arguments.

`save`/`load_cohort` exist because a 500-sample cohort takes tens of minutes and losing it to a kernel restart is unacceptable.

Normalization (`cpm`, `clr`, `log1p`, `relative`) lives in **pure Python** — it is array arithmetic numpy already does well, and freezing it into Rust is exactly the mistake avoided by emitting raw counts.

- [ ] Steps 1-5, including a test that `.samples` carries `total_kmers` for every row of `.matrix`, and that `format="sparse"` round-trips through `to_scipy()`.

---

### Task 6: `KmerVectorizer` — the differentiator

**Files:**
- Create: `python/fastdna/sklearn.py`
- Test: `python/tests/test_sklearn.py`

**Interfaces:**
- Produces: `KmerVectorizer(k=31, min_count=5, top_features=10_000, threads=None)` implementing `fit(paths, y=None)`, `transform(paths)`, `fit_transform`, `get_feature_names_out()`, `get_params`/`set_params`.

```python
pipe = Pipeline([
    ("kmers", KmerVectorizer(k=31, min_count=5, top_features=10_000)),
    ("clf",   XGBClassifier()),
])
scores = cross_val_score(pipe, fastq_paths, y, cv=5)
```

- `fit` counts the training samples, selects the vocabulary by prevalence, stores `self.vocabulary_`.
- `transform` projects onto the **already-learned** vocabulary via `count_projected`. Unseen k-mers are ignored; missing ones become 0.
- `get_feature_names_out()` returns the k-mer sequences.

**Why this is the product.** Inside a `Pipeline`, scikit-learn calls `fit` **only on the training fold** at each CV iteration. The vocabulary never sees the test fold. Leakage stops being a documented risk that depends on the user's discipline and becomes **structurally impossible**. iMOKA's feature reduction runs across the whole cohort; ours cannot.

**The biological payoff.** `get_feature_names_out()` paired with `model.feature_importances_` gives the concrete DNA sequences driving the prediction — which go straight into BLAST. Without it the model is a black box that happens to be right; with it, it is a publishable result.

**scikit-learn is a soft dependency.** Import it lazily inside `sklearn.py` so `import fastdna` works without it, and raise a clear `ImportError` naming `pip install scikit-learn` if the module is used without it. A wheel that hard-requires scikit-learn is a wheel many users cannot install.

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_transform_shapes(cohort_paths):
    v = KmerVectorizer(k=11, min_count=1, top_features=50)
    X = v.fit_transform(cohort_paths)
    assert X.shape == (len(cohort_paths), 50)
    assert len(v.get_feature_names_out()) == 50


def test_transform_ignores_kmers_unseen_during_fit(train_paths, novel_path):
    v = KmerVectorizer(k=11, min_count=1, top_features=20).fit(train_paths)
    X = v.transform([novel_path])
    assert X.shape == (1, 20), "columns must be the fitted vocabulary, not the new sample's"


def test_feature_names_are_valid_dna(train_paths):
    v = KmerVectorizer(k=11, min_count=1, top_features=10).fit(train_paths)
    for name in v.get_feature_names_out():
        assert len(name) == 11 and set(name) <= set("ACGT")


def test_fit_never_sees_the_test_fold(cohort_paths, labels):
    """The anti-leakage guarantee, as an executable test rather than a footnote."""
    seen = []

    class Spy(KmerVectorizer):
        def fit(self, X, y=None):
            seen.append(list(X))
            return super().fit(X, y)

    pipe = Pipeline([("kmers", Spy(k=11, min_count=1, top_features=20)),
                     ("clf", LogisticRegression())])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    list(cross_val_score(pipe, cohort_paths, labels, cv=cv))

    for fold_train, (_, test_idx) in zip(seen, cv.split(cohort_paths, labels)):
        test_paths = {cohort_paths[i] for i in test_idx}
        assert not (set(fold_train) & test_paths), "fit saw a test-fold sample"


def test_sklearn_estimator_contract():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(KmerVectorizer())
```

The fourth test is the one that matters: it turns the leakage guarantee into something CI enforces, rather than a claim in a README.

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Verify** — `pytest python/tests -v`
- [ ] **Step 5: Commit**

---

## Verification

- [ ] `cargo test` and `cargo clippy --all-targets` green
- [ ] `pytest python/tests` green, including the leakage test
- [ ] RAM path and spill path produce byte-identical matrices
- [ ] Wide and sparse encode the same data
- [ ] A cohort matrix that would exceed the byte cap returns `MatrixTooLarge` rather than being attempted
- [ ] `check_estimator(KmerVectorizer())` passes
- [ ] End-to-end: a synthetic cohort with a planted biomarker, run through `Pipeline` + `cross_val_score`, recovers the planted k-mer in `get_feature_names_out()` ranked by `feature_importances_`

That last one is the acceptance test for the whole product claim: plant a signal, and prove the library finds it through the same API a researcher would use.

## Out of Scope

- Incremental cohorts (add/remove a sample without recomputing). iMOKA has this and it is a real clinical need — patient 501 arrives next week. The §7.4 per-sample spill files make it a small extension rather than a rewrite. **Scheduled next, not forgotten.**
- Supervised feature selection. It belongs in Python inside the CV loop, never in Rust.
- `Sketch`/`compare`. Its own plan.
