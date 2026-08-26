# Performance Plan — Closing the Gap with KMC3/FastK

Status: **planned, not yet implemented**. This document records the plan agreed
on 2026-08-25 before any of the work below was executed, so progress can be
checked against it honestly later. It builds directly on `docs/design-minimizer-counting.md`
and `docs/CHECKPOINT-2026-08-25.md` — read those first for the underlying
diagnosis and measured numbers this plan is based on.

## Goal

Be faster than KMC3 at k-mer counting. Beating KMC3 specifically is the
realistic target, not FastK: in the cross-tool benchmark already recorded in
`docs/design-minimizer-counting.md` (2.14 GB FASTQ, 840,000,000 k-mer
occurrences, k=31, 8 threads, WSL2), KMC3 itself is already 2.9x slower than
FastK (110.4 s vs. 38.3 s). Even the full minimizer/super-k-mer redesign
described below is honestly predicted, in that same document, to likely leave
FastDNA still ~2x behind FastK — but that redesign directly targets the
mechanism (super-k-mer compression) that should let FastDNA close on, and
potentially pass, KMC3.

## Diagnosis (already established, not new)

FastDNA's default strategies sort all raw k-mer occurrences as independent
`u64` values. KMC3 instead compresses its input into far fewer **super-k-mers**
before sorting: 70,635,757 of them from the same 840M occurrences — an 11.9x
reduction in the number of items that need to be sorted. This is the
documented root cause of the current speed gap, not a memory or parallelism
problem (all four tools — FastK, KMC3, FastDNA in-memory, FastDNA disk — agree
exactly on 53,776,394 distinct k-mers; correctness is not in question).

FastDNA already has a super-k-mer-based counting path — the `binned` strategy
(`src/binned.rs`, `src/minimizer.rs`, `src/superkmer.rs`) — implementing
exactly this compression (measured at 1.035 bytes/occurrence vs. 8 bytes/occurrence
for the default path, a 7.7x reduction). It is opt-in only
(`--strategy binned` / `FASTDNA_STRATEGY=binned`) and never selected by the
automatic strategy chooser. As of 2026-08-25, one of its two known blockers to
promotion — a real-data bin-skew finding (1516x max/median skew on a 16S
amplicon sample) — was fixed by `src/adaptive_bins.rs`, a KMC2-style
data-adaptive bin map, already merged to `master`. The second blocker — a
calibrated peak-memory model for `binned`, needed before `resolve_strategy`
can safely consider it — does not exist yet.

## Plan: six independent workstreams, in parallel

Five are low-risk, mechanical optimizations on already-correct code (same
output, faster path). One is the strategic piece that actually targets the
KMC3 gap. All six are scoped to disjoint files so they can run without
conflicting, each as an isolated worktree branch reviewed and merged by the
coordinator afterward — the same pattern already used successfully today for
the adaptive bin map, the paired-end CLI mode, and the MIC regression module.

1. **`src/kmer.rs`, `src/export.rs`** — roll the reverse complement
   incrementally instead of recomputing it per k-mer window; remove a
   per-read allocation in the k-mer extraction path; build the exported
   `kmer_sequence` Arrow column from one contiguous byte buffer instead of
   one `String` allocation per k-mer (53.8 million of them at benchmark
   scale).
2. **`src/counter.rs`** (+ `Cargo.toml` only if real measurement justifies
   it) — MSD-partition the raw occurrence buffer into cache-resident buckets
   before `sort_unstable`; empirically sweep the finalize/compaction
   threshold. The existing `lto = "fat"` / `codegen-units = 1` setting stays
   untouched — it is already deliberately unbenchmarked, and documented as
   such, because this machine's own timing is unreliable across Docker/WSL
   (see `docs/CHECKPOINT-2026-08-25.md` and this repo's benchmark notes).
3. **`src/fastq.rs`** — reduce roughly 21 million per-record allocations
   measured in the real profiling run; investigate block reads in place of
   four `read_until` calls per FASTQ record, without weakening malformed-record
   error reporting (CLAUDE.md: bad input must fail loudly, not slowly and
   loudly is not enough — losing the exact failing record number would be a
   regression, not an acceptable trade).
4. **`src/sketch.rs`, `src/hll.rs`, `src/cms.rs`** — replace a `BTreeSet`-based
   per-k-mer insertion in sketch construction with a structure better suited
   to its actual (bounded) access pattern; check whether `containment`'s
   comparison should be a linear merge over sorted sequences instead of
   repeated binary search. Must not change which items are sampled/hashed —
   a sketching bias change is a correctness regression here, not a acceptable
   side effect of "still statistically valid."
5. **`python/fastdna/`** — vectorize at least one row-by-row Python loop over
   an Arrow table that should be a columnar operation; stop rebuilding a
   sample's sketch inside an O(n²) all-pairs comparison (`compare_all` and
   whatever calls it) when it can be built once per sample and reused.
6. **The strategic piece — connect `binned`/`adaptive_bins` toward automatic
   promotion.** Build the calibrated (or honestly-labeled structural, if real
   calibration isn't possible in the working environment) peak-memory model
   for the `binned` strategy in `src/mem_estimate.rs`, matching the existing
   in-memory/disk model's style and its explicit honesty about what is and
   isn't measured. Decide — and justify with real evidence, not
   assumption — whether `resolve_strategy` should now be able to select
   `binned` automatically, or whether it should stay opt-in pending real
   ENA-sample re-validation (the adaptive bin map was validated against a
   synthetic reproduction of the real skew case, not yet re-run against the
   original `DRR021372` sample). This workstream also owns getting a real,
   honest before/after benchmark number for `binned` vs. the default
   strategy on the native Windows release build — the number that actually
   answers "are we closer to KMC3 now."

## Constraints that apply to every workstream

- Every change must be bit-for-bit output-identical to today's behavior
  unless the whole point of the change is a documented, deliberate new
  capability (workstream 6's strategy selection is the one exception, and
  even there the change must be evidence-based, not assumed).
- `cargo test`, `cargo clippy --all-targets` (Rust) and
  `python -m pytest python/tests -q` (Python) must stay clean for every
  workstream before it is considered mergeable.
- No workstream merges to `master` on its own — each is reviewed and merged
  by the coordinator, the same way today's `adaptive_bins`, `--paired-dir`,
  and `fastdna.mic` work was.
- Benchmark numbers must come from the native Windows release build via
  `Measure-Command`, not WSL/Docker — this repository's own benchmark
  culture already treats those environments as unreliable on this machine
  (documented swings of 25–90 s on identical input for the same binary).
- No invented numbers. If a workstream cannot get a real measurement, it
  says so plainly rather than estimating one — the standard this repository
  already holds itself to (see the `Cargo.toml` LTO comment, and
  `docs/design-minimizer-counting.md`'s explicit "predictions recorded in
  advance so they can be falsified").

## What this plan does not promise

Beating FastK is not promised and is not the target. Beating KMC3 is the
target, is plausible given the diagnosis above, but is not guaranteed until
workstream 6 produces a real number. This document exists so that number,
whenever it lands, can be checked against a plan that was written down before
the fact.
