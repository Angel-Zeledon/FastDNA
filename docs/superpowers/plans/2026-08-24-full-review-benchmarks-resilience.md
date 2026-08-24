# Full Review, Exhaustive Testing, Resilience Hardening & KMC3/FastK Benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the entire fastdna (Rust + Python bindings) and fastdna-ml codebases, exercise every user-facing feature end-to-end, fix every defect found (TDD), and re-run the KMC3/FastK large-scale benchmark including the new disk-partitioned strategy.

**Architecture:** Review-driven: parallel read-only review agents produce a findings list; each confirmed finding becomes a TDD fix task. Functional testing is a scripted probe matrix over the real release binary. Benchmarks run in WSL2 Ubuntu per `scripts/bench/kmc3_fastk_comparison.sh` (KMC3 prebuilt, FastK from source), FastDNA measured as the native Windows binary and additionally as a Linux build for like-for-like I/O.

**Tech Stack:** Rust (cargo, clippy with unwrap/expect/print denied), pyo3/maturin Python package (host Python 3.12), Docker (fastdna-ml), WSL2 Ubuntu (KMC 3.2.4, FastK, /usr/bin/time -v).

**Spec:** The user request of 2026-08-24: full code review (Rust + Python), run benchmarks vs KMC3 and FastK, test literally every feature, make the app super error-resilient.

## Global Constraints

- English only in code, comments, tests, commit messages, docs (project-wide rule).
- Chat/reporting to the user in Spanish (user memory).
- Output schema `kmer_u64: u64, kmer_sequence: utf8, frequency: u32` must not change (fastdna-ml contract).
- Clippy lints `unwrap_used`, `expect_used`, `print_stdout`, `print_stderr` are `deny` — fixes must not introduce them (CLI main.rs uses whatever mechanism already exists there).
- `panic = "unwind"` in release must stay (FFI safety).
- Producer channel stays bounded (memory flatness).
- Commit frequently on `master`; each fix its own commit.

## Known state (verified 2026-08-24)

- `cargo check` clean; `cargo test` all green (114 unit + 46 integration).
- CLAUDE.md is stale: `bio.rs` no longer exists, pipeline bug fixed, `--qc`/`--histogram`/`--max-ram`/`--strategy` all wired; counter.rs, disk_spill.rs, mem_estimate.rs, error.rs, ffi.rs, hll.rs, cohort/, preview.rs exist now.
- 10 agent worktree branches all merged into master; worktree dirs still on disk under `.claude/worktrees/`.
- README already documents a 2.14 GB KMC3/FastK comparison (pre-disk-strategy numbers).
- WSL2 Ubuntu + Docker present; kmc/FastK/hyperfine not installed on Windows host.

---

### Task 1: Parallel deep code review (read-only)

**Files:** none modified. Output: findings list in scratchpad.

- [ ] Dispatch 4 parallel read-only review agents:
  - R1: Rust core correctness — `src/kmer.rs`, `src/counter.rs`, `src/fastq.rs`, `src/qc.rs`, `src/pipeline.rs` (races, overflow, edge cases k=1/k=32, ambiguous bases, trimming invariants).
  - R2: Rust I/O + strategies — `src/export.rs`, `src/disk_spill.rs`, `src/mem_estimate.rs`, `src/cli.rs`, `src/main.rs`, `src/error.rs`, `src/progress.rs`, `src/preview.rs` (error handling gaps, partial writes, tmp-file cleanup, invalid-input handling).
  - R3: Rust auxiliary + FFI — `src/cms.rs`, `src/sketch.rs`, `src/hll.rs`, `src/cohort/`, `src/ffi.rs`, `src/wasm.rs` (panic paths crossing FFI, unwrap-equivalents, API misuse resilience).
  - R4: Python — `python/fastdna/*.py` and `fastdna-ml/` (`train_classifier.py`, `generate_dataset.py`, `baseline_python.py`, `download_covid.py`, Dockerfile, docker-compose, .bat/.sh scripts) — error handling, schema contract, portability (`/tmp/cohort` on Windows!), correctness.
- [ ] Each finding recorded as: file:line, defect, concrete failure scenario, severity.
- [ ] Adversarially verify every High/Critical finding (read the code yourself) before accepting it.

### Task 2: Exhaustive CLI probe matrix

**Files:** Create probe script in scratchpad; findings feed Task 4.

- [ ] Build release binary: `cargo build --release`.
- [ ] Happy paths: `.fastq`→`.csv`, `.fastq`→`.parquet`, `.fastq.gz` input, `-k 1`, `-k 32`, `-q 0`, `-q 40`, `-m`/`-M` filters, `-t 1`/`-t 8`, `--qc out.json`, `--histogram out.csv`, `--strategy memory`, `--strategy disk`, `--strategy auto` with tiny `--max-ram` (forces disk), `--max-ram 512K` parse forms (`4G`, plain bytes).
- [ ] Error paths (must exit non-zero with a clear message, never panic/hang): missing input, input is a directory, `-k 0`, `-k 33`, `-q 41`, `-m 0`, `-M < -m`, invalid `--strategy`, invalid `--max-ram` (`abc`, `-1`, `9999999T`), output to nonexistent dir, output to unwritable path, `.gz` that is not gzip, truncated `.gz`, empty file, file with only `\n`, binary garbage input, FASTA-not-FASTQ input, quality line shorter than sequence, CRLF file, huge header line, read shorter than k, all-N reads, unicode in paths, output path same as input path.
- [ ] Verify CSV and Parquet outputs of the same input agree; verify `memory` and `disk` strategies produce identical counts.
- [ ] Verify QC JSON parses and percentages are sane; histogram CSV header/rows sane.
- [ ] Record every deviation as a finding.

### Task 3: Python + ML pipeline exercise

- [ ] Run host pytest: `python -m pytest python/tests -q` (244 tests) — record failures.
- [ ] Smoke the public Python API surface per `python/fastdna/__init__.py` exports (count, spectrum, sketch, taxonomy, embed, sklearn vectorizer, progress callback, cohort discovery) against `test.fastq`.
- [ ] fastdna-ml: confirm `bin/fastdna.exe` is stale vs current build (hash compare); rebuild+copy if stale.
- [ ] `docker compose build ml-env` then run `train_classifier.py` in the container; record failures.
- [ ] Record every deviation as a finding.

### Task 4: Fix all confirmed findings (TDD, one commit each)

For each confirmed finding, in severity order:

- [ ] Write a failing test that reproduces it (Rust: `tests/` or module unit tests; Python: `python/tests/`).
- [ ] Run it, confirm it fails for the expected reason.
- [ ] Minimal fix; run the test, confirm pass.
- [ ] Run the full affected suite (`cargo test` / pytest) — no regressions.
- [ ] `cargo clippy --all-targets` clean.
- [ ] Commit: `fix: <finding summary>`.

Known candidate already identified during recon (verify before fixing): `fastdna-ml/ml_pipeline/train_classifier.py:66` hardcodes `/tmp/cohort` — fine inside Docker, breaks host-run on Windows; use `tempfile.gettempdir()`.

### Task 5: KMC3 + FastK benchmark in WSL

- [ ] Copy repo's bench scripts into WSL-native filesystem (`~/fastdna-bench`).
- [ ] Install/build tools per `scripts/bench/kmc3_fastk_comparison.sh` (KMC 3.2.4 prebuilt, FastK from source).
- [ ] Generate the canonical 2.14 GB dataset (`generate_reads_large.py 35000000 30 150 bench2gb.fastq 9001 200000`) on ext4.
- [ ] Build FastDNA Windows release (`cargo build --release`) and Linux release inside WSL (`cargo build --release` from `/mnt/c` source into a WSL-side target dir).
- [ ] Measure with `/usr/bin/time -v` (peak RSS + wall clock), k=31, all threads, singletons included (`kmc -ci1`, `FastK -t1`, fastdna `-m 1 -q 0`... match README's exact flags): KMC3, FastK, FastDNA memory strategy, FastDNA disk strategy (`--strategy disk`), FastDNA auto.
- [ ] Cross-check distinct k-mer counts across all tools (must agree given matching trimming settings).
- [ ] Update README benchmark section with the new table (including disk strategy row) and date; note hardware.
- [ ] Commit: `docs: refresh KMC3/FastK benchmark with disk-strategy numbers`.

### Task 6: Documentation truth pass

- [ ] Rewrite stale CLAUDE.md sections (crate compiles; current module list; current CLI; strategies; benchmarks pointer).
- [ ] Prune leftover `.claude/worktrees/` dirs and stale root artifacts (`counts.parquet`, `qc_report.json`, `hist.csv`, `test.fastq` — keep `test.fastq` if tests/scripts reference it).
- [ ] Commit: `docs: bring CLAUDE.md in line with the current codebase`.

### Task 7: Final verification + report

- [ ] `cargo test`, `cargo clippy --all-targets`, host pytest, one full CLI happy-path run.
- [ ] Report to user (Spanish): findings fixed, test/probe results, benchmark table, remaining risks.
