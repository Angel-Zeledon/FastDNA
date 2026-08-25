# Completeness Phase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the remaining in-domain gaps between FastDNA and the field's dedicated k-mer tools, in the order that unblocks the most downstream work.

**Architecture:** The keystone is a queryable on-disk k-mer database. The merge phase already emits a globally sorted `(u64, u32)` table (`disk_spill::merge_buckets`, and `counter.rs`'s finalized order), so a fixed-record file plus a sampled index gives O(log n) lookup with no hashing. Set operations, read filtering and per-read profiles are then linear merge-joins or streaming lookups over that format — the same machinery `disk_spill.rs` already implements.

**Tech Stack:** Rust (existing crate conventions), pyo3/maturin Python surface, Arrow output.

**Spec:** `docs/feature-gap-analysis.md` (engine gaps, researched against KMC3/FastK/Jellyfish/sourmash/ntCard) and `docs/ml-differentiation-roadmap.md` (B1 FracMinHash, B3 per-read screening).

## Global Constraints

- English only in code, comments, tests, docs, commit messages.
- Clippy denies `unwrap_used`, `expect_used`, `print_stdout`, `print_stderr` outside test modules and `main.rs`.
- `panic = "unwind"` in release must stay (FFI safety).
- The counts schema `kmer_u64: u64, kmer_sequence: utf8, frequency: u32` must not change.
- Every output writer goes through `atomic.rs` — never `File::create` a destination directly.
- The producer channel stays bounded.
- TDD per behavior; commit per coherent feature.
- **Validation standard:** every feature that claims parity with an existing tool must be measured against it, not asserted. The bar set on 2026-08-24: KMC3, FastK and FastDNA agreeing exactly on 11,914,413 distinct k-mers.

---

### Task 1: Queryable k-mer database (`src/ktab.rs`)

Blocks tasks 2, 3 and 6. Do this first.

- [ ] Format: header (magic, format version, `k`, record count, index stride) + sorted fixed-width `(u64 kmer, u32 count)` records + a sampled index of every Nth key. Version the header from day one — a format that cannot say what it is becomes unreadable the first time it changes.
- [ ] `write(counter, path)` via `atomic.rs`; `open(path)` memory-maps or buffers, validates the header, and reports a corrupt/foreign/truncated file as `FastDnaError::Load` (never a panic, never a silent wrong answer — mirror `sketch.rs::load`'s contract).
- [ ] `get(kmer) -> Option<u32>` (binary search within the indexed range) and `get_many(&[u64])` for batch lookup.
- [ ] Iteration in sorted order, so set operations can stream without loading.
- [ ] CLI: `fastdna query <db> <kmer-or-file>`; Python: `KmerTable.open(path).get("ACGT...")`.
- [ ] Tests: round-trip; lookup of present/absent k-mers; a k-mer sequence whose length disagrees with the database's `k` is an actionable error; truncated file, wrong magic, and a future format version each produce a distinct clear error; lookup on an empty database; k=1 and k=32 edges.
- [ ] Document the memory cost per k-mer and the database size for a given distinct count.

### Task 2: Set operations (`src/setops.rs`)

- [ ] `union`, `intersect`, `difference`, `symmetric_difference` over two sorted databases, streaming (never both resident).
- [ ] Count-combining policy is explicit, not implicit: `sum`, `min`, `max`, `left`, `subtract` — kmc_tools' counter operations. Default must be documented and justified.
- [ ] Reject operands built with different `k` (`FastDnaError::MismatchedK`, as `sketch.rs` already does for the same class of mistake).
- [ ] CLI `fastdna ops <intersect|union|difference|symdiff> A B -o C --counts sum`; Python methods on the table type.
- [ ] Tests: hand-checkable small tables for each operation and each count policy; empty operand on each side; disjoint inputs; identical inputs; mismatched `k` errors; output stays globally sorted.

### Task 3: Read filtering by k-mer content (`src/filter.rs`)

- [ ] Stream reads through the existing reader, look each read's canonical k-mers up in a database, keep or discard on a threshold (absolute hit count and fraction-of-k-mers, both supported and distinctly named).
- [ ] Write surviving records as FASTQ/FASTA (gzip by extension) through `atomic.rs`; `--invert` to keep the complement (host removal vs. target enrichment are the same operation with opposite polarity).
- [ ] Report kept/discarded counts on stderr and in the QC JSON.
- [ ] Tests: a read sharing every k-mer is kept and one sharing none is discarded at any threshold; boundary threshold arithmetic on a hand-checkable read; `--invert` is exactly the complement; reads shorter than `k` are handled explicitly (decide, document, test); paired-file input keeps mates in sync or errors loudly if it cannot.

### Task 4: FracMinHash (scaled) sketches (`src/sketch.rs`)

Fixes a real bias, not a new feature: bottom-k containment is biased when set sizes differ wildly, which is exactly `taxonomy.classify`/`gather`'s case.

- [ ] `scaled` construction: keep every hash below `u64::MAX / scaled`. Existing `sketch_size` bottom-k path stays; the two are alternative modes and mixing them in one comparison must be an error, not a silent wrong number.
- [ ] `subtract` (required for iterative gather) and `ani` with the confidence interval from Hera et al. 2023.
- [ ] Switch `taxonomy.gather` to iterative minimum-set-cover over scaled sketches (Irber et al. 2022).
- [ ] Tests: containment of a small set inside a large one is unbiased where bottom-k is not — assert the specific improvement against a computable ground truth, and keep a test that pins the old estimator's bias so the fix cannot silently regress; comparing a scaled sketch with a bottom-k sketch errors; ANI on sequences of known divergence; validate against sourmash output on a shared input if sourmash can be installed.

### Task 5: CLI subcommands and distribution

- [ ] Restructure clap into subcommands — `count` (the default, so every existing command line keeps working: pin that with a test), `sketch`, `dist`, `card`, `peek`, `query`, `ops`, `filter`, `profile`.
- [ ] Backward compatibility is a hard requirement, not a nicety: `fastdna -i x.fastq -o y.parquet` must behave exactly as before.
- [ ] Bioconda: real URL and sha256 in `recipe/meta.yaml`, then submit. Needs a tagged release first.

### Task 6: Per-read k-mer profiles (`src/profile.rs`)

FastK's signature feature; nothing else in the field has it. Makes `assembly_qc.py` exact rather than approximate.

- [ ] Two-pass: count (exists), then stream reads looking each k-mer up in the Task 1 database, emitting run-length-encoded count vectors.
- [ ] Arrow output: `read_id: utf8`, `profile: list<uint32>` (or an RLE pair of lists — decide, document the tradeoff, and say which downstream consumer drove the choice).
- [ ] Python `profile_reads(path, table)`; rewrite `assembly_qc.py`'s QV estimation on top of it and state in its docstring that it is now exact.
- [ ] Tests: a read whose k-mers are all in the database profiles as all-present; a single substitution produces the characteristic ~k-long dip and the dip's position and width are asserted exactly; RLE round-trips; a read shorter than `k` yields an empty profile, not an error.

### Task 7: Minimizer partitioning (`src/disk_spill.rs`)

Performance only. Do last — benchmarks matter at scales the current strategy already handles correctly.

- [ ] Replace `bucket_of`'s high-bit split with a minimizer signature. The module's own header documents this as the anticipated evolution and notes the spill/merge machinery does not care how `bucket_of` computes its answer.
- [ ] High-bit bucketing keeps bucket order aligned with k-mer order, which is what makes bucket-order concatenation already sorted; a minimizer scheme loses that, so either sort per bucket and concatenate by content order, or keep a hybrid. Decide with a measurement, not a preference.
- [ ] Measure bucket balance on a real skewed-composition input before and after; report the change honestly, including if it does not help.

### Task 8: Documentation truth pass

- [ ] README: the CLI section must cover FASTA/multi-file/stdin input and `--histogram-format`/`--histogram-max` (landed 2026-08-24, still undocumented), plus every subcommand from Task 5 and the new Python modules (`cv`, `gwas`, `rules`, `genomescope`, `anomaly`, `metagenomics`, `translate`).
- [ ] `docs/BENCHMARKS.md`: refreshed KMC3/FastK/FastDNA table including the disk strategy, measured on a quiet machine.
- [ ] CLAUDE.md: module list and conventions kept current.

## Deferred, with reasons

- **k > 32 via u128**: real demand (k=63 in assembly), but it touches `kmer.rs`'s core representation and every module built on `u64`. Worth a plan of its own, not a step in this one.
- **Unitigs (compacted de Bruijn graph)**: the field's own evolution beyond raw k-mers and a natural partner for `gwas.py` (DBGWAS, unitig-caller). Larger than anything above; plan separately once the database backbone exists.
- **ntCard-style streaming spectrum estimate**: useful before committing to a full count; smaller value than the tasks above.
- **`cms.rs`**: still wired to nothing. Either give it a purpose (an `--approximate` counting mode) or delete it — dead code on the public surface is a liability.
