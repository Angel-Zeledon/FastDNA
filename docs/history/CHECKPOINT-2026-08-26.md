# Checkpoint — 2026-08-26

> **Resuelto en una sesión posterior, mismo día.** Los 5 agentes en
> background listados abajo ya no estaban vivos (`ListAgents` no los vio;
> murieron con la sesión anterior, tal como este documento anticipaba que
> podía pasar). Se verificó que los cinco ya habían dejado trabajo real en
> el árbol de trabajo, se corrió la verificación Rust+Python completa por
> Docker (ver comando más abajo), se corrigió un docstring con
> placeholders sin resolver en `sklearn.py`, y se actualizó
> `CHANGELOG.md`. Resultado: **364 tests de lib Rust + todas las suites de
> integración, 0 fallos, 1 ignorado, exactamente 8 warnings de clippy
> (mismas ubicaciones que el baseline)**; **712 tests Python pasados, 40
> skipped, 1 xfailed, 0 fallos** (baseline de esta sesión era 357/652 --
> ambos subieron, ninguno bajó). Ningún `git commit` se hizo tampoco en
> esta sesión de resolución -- todo sigue en el working tree. Detalles
> completos en la respuesta de esa sesión al usuario. Lo único de esta
> checkpoint que quedó explícitamente sin resolver: el backend gzip
> zlib-ng (G-4) nunca se implementó (Cargo.toml sigue con `flate2`
> default/miniz_oxide); y `fastdna.audit`/`GenomicModel`/
> `validate_generated` **no** se reexportaron en `__init__.py` como este
> documento planeaba en "Where to pick up" -- resultaron no necesitarlo
> (sus propios tests, ya escritos por los agentes, los importan como
> `from fastdna.<modulo> import ...`, el mismo patrón que el resto de la
> capa ML), así que forzar la reexportación de `audit` habría
> sombreado el submódulo del mismo nombre sin que ningún test lo pidiera.


Written because the interactive session that did this work was about to end
mid-task. Everything below is what was true when it was written, not
assumed — resume from "Where to pick up," but verify the live-agent status
first since background work may have kept running or may have died with the
session; this document cannot know which.

## Repository state

- `\\guayaba.online\UPIT\dev\angel.z\Documents\a\FastDNA`, branch `master`.
- **No `git commit` was made this session.** Everything below is uncommitted
  in the working tree, on top of whatever `master` HEAD was at session start
  (the tree already contained a large amount of prior work from other
  sessions/agents — see "Context: this is a shared, actively-developed repo"
  below).
- No local Rust or Python toolchain on this machine. All verification goes
  through Docker — see "Verification workflow" below for the exact,
  working command pattern (including a real gotcha found this session).

## Context: this is a shared, actively-developed repo

Partway through this session it became clear the repo is far more advanced
than this session's own history alone would suggest — there is other,
independent work landing on `master` (real commits, not this session's),
including a whole separate roadmap:

- `docs/philosophy-narrow-not-broad.md` — an explicit decision to keep
  FastDNA narrow (SAMtools-vs-Biopython-style argument). Read this before
  adding any new module — it argues against exactly the kind of
  monolithic API this session's own `docs/audit/` planning originally
  sketched, which is why the in-flight `audit()` work below was redesigned
  to reuse `cv.py`/`evaluation.py`/`workflow.py` rather than reimplement
  them.
- `docs/ml-genomics-roadmap.md`, `docs/ml-differentiation-roadmap.md`,
  `docs/feature-gap-analysis.md` — ten ML modules already shipped (wave 1-2:
  `sklearn.py`, `taxonomy.py`, `assembly_qc.py`, `interop.py`, `embed.py`,
  `interpret.py`, `anomaly.py`, `active_learning.py`, `multiomics.py`, plus
  ecosystem glue), and a full competitive/performance analysis against
  KMC3/FastK/Jellyfish/Mash.
- `docs/history/CHECKPOINT-2026-08-25.md`, `docs/history/PERFORMANCE_PLAN.md` — a benchmark
  session against KMC3/FastK on real data (WSL2, a different machine
  referenced as `C:\Users\Jahir\Documents\DNA-Rust\bench\`), with its own
  "Outcome" section recording what shipped (MSD-partition sort in
  `counter.rs`, rolled reverse-complement in `kmer.rs`, allocation removal
  in `fastq.rs`/`sketch.rs`, an Arrow-native rewrite of
  `multiomics.kmer_feature_table`) and what was explicitly declined
  (promoting the `binned` strategy — real-data-gated and declined, see
  `docs/validation-real-data.md` and commit `41675fd`).
- `git log --oneline` has 100+ commits, many `Merge branch
  'worktree-agent-...'` — confirms this pattern of parallel agents landing
  work directly on `master` predates and continues alongside this session.

**Practical implication for whoever resumes**: before building anything new,
check whether it already exists under a different name. This session found
and avoided three near-duplications this way (an `audit()`-shaped feature
already partly covered by `cv.py`+`evaluation.py`; a GenomicModel-shaped
feature overlapping `anomaly.py`; `cms.rs`/binned-promotion already decided
by the other track).

## What's done and verified this session

All of the following passed a full Rust (`cargo build/test/clippy
--all-targets`) and Python (`maturin develop --features python` +
`pytest python/tests -q`) run at the time it was verified:

- **H-10, the exception hierarchy** (`src/ffi.rs`) — the flagship fix this
  session. Thirteen leaf exception classes with **real** Python multiple
  inheritance (`InvalidKError(FastDnaError, ValueError)`,
  `IoNotFoundError(FastDnaError, FileNotFoundError)`, etc.), built via a
  short embedded-Python `class` snippet run once at import time
  (`py.run_bound`, since `pyo3::create_exception!` only supports one base).
  Verified via real MRO inspection and `isinstance` checks, not just
  compilation. Two real pitfalls hit and fixed along the way (both now
  documented in the code comments, in case they recur):
  1. Naming the macro-generated Rust type `FastDnaError` shadowed
     `crate::error::FastDnaError` (the import), causing 54 unrelated
     compile errors. Fixed by naming it `FastDnaErrorBase` in Rust while
     keeping the Python-visible name `FastDnaError`.
  2. `Cargo.toml` denies `clippy::unwrap_used` and `clippy::expect_used`
     crate-wide (`[lints]`, line ~59-60) — any "should never happen"
     assertion must be `.unwrap_or_else(|| panic!(...))`, not
     `.unwrap()`/`.expect()`.
  - Also fixed: a stray leftover Spanish comment in `src/error.rs`
    (the user's "todo el código en inglés" rule, applied opportunistically
    while already in that file).
- Everything from earlier in this session (context predates this
  checkpoint's own start, carried forward unverified-again but was green
  at the time): Tasks 25/33/34/35 (`with_sequence` opt-in, `CohortCounts`,
  `representation=` on `KmerVectorizer`, `explain()`), H-01/06/08/11-15/21/
  23-25 hardening, `cms.rs` + `binned_occupancy_report.rs` deletion,
  `CHANGELOG.md`.

**Baseline to compare any future run against** (measured after H-10, this
is the number to not regress below):
- Rust: 357 lib tests + all integration suites, **0 failed**, 1 ignored.
  Clippy: exactly 8 preexisting warnings, same locations as always
  (`src/metagenomics.rs:1094`, `src/binned.rs:553`, `src/counter.rs:905`,
  `src/minimizer.rs:682`, `src/superkmer.rs:388`, `src/superkmer.rs:480`,
  `src/translate.rs:424` x2, `tests/dual_strategy.rs:75`) — line numbers in
  `counter.rs` may drift if the in-flight SoA work (below) lands.
- Python: **652 passed, 40 skipped, 1 xfailed, 0 failed** with minimal
  extras (pyarrow, pandas, scikit-learn, scipy installed; matplotlib,
  biopython, polars, duckdb, umap-learn, shap not installed in this
  container). A prior session measured 699 passed with full extras.

## In flight — 5 background agents launched, status unknown at write time

All five were told: don't touch `CHANGELOG.md`; don't touch
`python/fastdna/__init__.py` (the orchestrating session reserved that for
itself, to do one final integration pass adding imports/`__all__` entries
after everyone lands, avoiding concurrent edits to a shared file); don't
`git commit`/`push`/`cargo publish`/`pip upload`; use an isolated
robocopy-mirror + Docker build dir so parallel agents don't race each
other; confirm the full test suite (not just their own new file) before
finishing.

**Gotcha discovered and included in their instructions**: the
`fastdna-pytest` Docker image's venv is on `PATH` but `maturin develop`
still fails with "Couldn't find a virtualenv" unless `VIRTUAL_ENV=/venv` is
exported first. The exact working command:

```powershell
$src = "\\guayaba.online\UPIT\dev\angel.z\Documents\a\FastDNA"
$build = "C:\Users\angel.z\AppData\Local\Temp\claude\--guayaba-online-UPIT-dev-angel-z-Documents-a\78d8cc1d-c970-4027-b312-db431c032344\scratchpad\build-<name>"
robocopy $src $build /MIR /XD .git target /NFL /NDL /NJH /NJS /NP
docker run --rm -v "${build}:/w" -v fastdna-cargo-registry:/usr/local/cargo/registry -v fastdna-target-<name>:/w/target -w /w fastdna-pytest bash -c "export VIRTUAL_ENV=/venv && maturin develop --features python && python -m pytest python/tests -q"
```
(For Rust-only work, swap the image for `rust:1-slim-bookworm` and drop the
`VIRTUAL_ENV`/`maturin` parts — `cargo build/test/clippy --all-targets`
directly. Clippy needs `rustup component add clippy` first in that image,
every time, since `--rm` containers don't persist it.)

Each agent's first launch **failed immediately** on an account-level API
session limit ("resets 10:10am America/Guatemala") before writing anything
— safe, nothing was half-edited. All five were relaunched with the same
prompts; relaunch succeeded (no immediate failure) as of this writing.

| Task | Agent ID (current) | Build dir / volume | Scope (files it owns) |
|---|---|---|---|
| `fastdna.audit()` — leakage-gap measurement (Task 14, folds in G-13 `covariates=`) | `a21fd7a456a4c1c9f` | `build-audit` / `fastdna-target-audit` | new: `python/fastdna/audit.py`, `python/tests/test_audit.py` |
| `GenomicModel` (Task 38/G-10) + `validate_generated()` (Task 39/G-12) | `a2235536f9774dc00` | `build-deploy` / `fastdna-target-deploy` | new file(s) under `python/fastdna/`, `python/tests/` (name left to the agent's judgment) |
| Streaming vocabulary construction for `KmerVectorizer` (Task 36/G-9) + `__all__`/annotations for that one file | `a0e795e32fe2344c3` | `build-vocab` / `fastdna-target-vocab` | `python/fastdna/sklearn.py`, `python/tests/test_sklearn.py` only |
| Type annotations (H-04/H-05) + `__all__` (H-09) + `py.typed` marker, package-wide except `sklearn.py` | `a43d252e5c2cfc5a3` | `build-types` / `fastdna-target-types` | `python/fastdna/__init__.py` + every submodule except `sklearn.py`; `pyproject.toml`; new `python/fastdna/py.typed` |
| `counter.rs` struct-of-arrays layout (G-3) + zlib-ng gzip backend (G-4) | `af54f4998ca91a27a` | `build-speed` / `fastdna-target-speed` | `Cargo.toml`, `src/counter.rs`, `src/mem_estimate.rs` only |

**Known wrinkle on the TYPES agent (`a43d252e5c2cfc5a3`)**: it reported
`status: completed` once already, but its own result text was "All five
background subagents are running. I'll pause here..." — meaning it had
delegated its own work to (at least) 5 further sub-agents (one per
file-group) and stopped when it ran out of live children to wait on, not
because the task was actually finished. It was resumed via `SendMessage`
and told to check its children, re-verify everything itself, and give a
real final report (files touched, `py.typed`-in-wheel confirmation, test
count). Two of its children had already reported back by the time this checkpoint
was finalized: annotations landed cleanly in `annotate.py`,
`assembly_qc.py`, `translate.py`, `calibration.py`, `embed.py` (first
child), and `genomescope.py`, `cv.py`, `mic.py`, `active_learning.py`,
`cohort_counts.py` (second child) — both verified only via `python -m
py_compile`, not yet the full pytest suite; that consolidated
re-verification pass still needs to happen. **Whoever resumes should check
this agent's status first** — it may need another nudge if it stalled again
the same way. At this rate (5 files/child, 2 children in ~10-15 min) there
are likely 1-3 more children covering the rest of `python/fastdna/`'s ~26
modules (minus `sklearn.py`) still to report.

At the moment this checkpoint was written: 8 subagents showing `running` in
`ListAgents` (the 5 above, plus what's left of the TYPES agent's own
children after some had already finished and been absorbed). None had
reported a hard failure.

## Where to pick up

1. Run `ListAgents` first. For anything still `running`, just wait for its
   completion notification (background agents notify this session
   automatically). For anything `completed`, check whether its result text
   is a real final report (files changed + test counts) or another
   TYPES-style false-alarm pause — if the latter, `SendMessage` it to
   finish and re-verify.
2. Once all 5 (and TYPES's children) are in, do the integration pass this
   session deliberately deferred to avoid concurrent-edit conflicts:
   - Add `from .audit import audit, AuditReport` (or whatever `audit.py`
     actually exports) to `python/fastdna/__init__.py`.
   - Add the equivalent import(s) for whatever the GenomicModel/
     validate_generated agent actually named its file(s).
   - Fold both into `__init__.py`'s `__all__` (which the TYPES agent should
     have already added for everything that existed before these two
     landed).
   - Update `CHANGELOG.md` in one pass covering everything from this
     session (H-10 plus whatever the 5 agents actually shipped — verify
     against their real final reports, not this document's plan, the same
     lesson learned from the CHANGELOG.md work earlier this session).
   - Run the full combined Rust + Python verification one more time after
     integration, since touching `__init__.py` is exactly the kind of
     shared-file edit that can silently break an otherwise-fine module.
3. Re-run the full baseline comparison (357+ Rust / 652+ Python, 0 failed,
   8 clippy warnings) after integration, before considering this batch done.

## Explicitly out of scope for this batch (per direct user instruction)

- CI/infra work (H-18 coverage script, H-19 wasm.rs tests, H-20 mkdocs
  site, H-26 CI running cargo test/clippy/pytest).
- Publishing (Task 12: actual PyPI/crates.io publish — irreversible,
  requires the user's explicit go-ahead; Task 13 Bioconda, blocked on it;
  H-02 fixing the README's `pip install` instructions in the meantime is
  still fair game if anyone picks it up, it just wasn't assigned this
  round).

## Still genuinely not started, not assigned this round either

- Task 37 (G-11): public benchmark dataset + reproducible notebook — likely
  blocked on real network/data access this environment doesn't have, same
  constraint that blocked G-2's real-data validation earlier (though the
  other track's `docs/validation-real-data.md` shows real-data validation
  *is* possible from wherever "Jahir" was working — worth asking whether
  that access is available before writing this off as blocked here too).
- Fase 4 long-term bets (k>32 via u128, gene-annotation loop-closing,
  foundation-model bridge) — nothing started, not attempted this session.
