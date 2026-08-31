# Performance Plan — Closing the Gap with KMC3/FastK

Status: **executed 2026-08-25; see "Outcome" at the end of this document for
what each workstream actually produced, including the two that turned out to
be already done before this plan was written and the one measurement the
environment could not take.** The body below is the plan exactly as agreed,
left unedited so it can still be read as the prediction it was.

This document records the plan agreed
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

---

## Outcome — recorded after execution, 2026-08-25

The plan's own standard applies to this section: no invented numbers, and
where a measurement could not be taken, that is stated rather than estimated
around.

### The environment, first, because it determined three of the six results

**There is no Rust toolchain on the machine this was executed on** — no
`cargo`/`rustc` on the Windows host, none in WSL. All Rust work below was
compiled and tested in a Linux container (`rust:1-slim-bookworm`), and the
Python suite was run against a `maturin`-built extension in a second
container. That satisfies "`cargo test` / `cargo clippy` / `pytest` must stay
clean", which are correctness gates and environment-independent.

It does **not** satisfy this plan's benchmarking constraint, which requires
numbers from the native Windows release build via `Measure-Command`. That
build cannot be produced here. Consequently **no whole-program timing exists
for any of this work**, including the one number workstream 6 was chartered
to produce.

### Workstreams 1, 3 and 4 — already implemented before this plan was written

Not re-done, because on inspection all three were already in `master`:

- **1 (`kmer.rs`, `export.rs`)** — commit `106ae78`, 2026-08-25 08:34, some
  eleven hours before this document was committed at 19:53. The rolled
  reverse complement, the removed per-read allocation
  (`extract_canonical_kmers_into`), and the contiguous `kmer_sequence` buffer
  (`ChunkBuffers`, `decode_kmer_into`) are all present.
- **3 (`fastq.rs`)** — commits `a149b57` and `2a2de0f`. The ~21 million
  per-record allocations are gone via `next_record_into` and
  `skip_marked_line`; the four `read_until` calls per record were kept
  deliberately, with the reasoning recorded in the code (std's `read_until`
  is a word-at-a-time `memchr`, and the exact failing record number the
  plan named as non-negotiable is preserved).
- **4 (`sketch.rs`)** — commit `a149b57`. `BottomK` already reduces the
  rejection path (99.99% of calls) to one `bool` test and one integer
  compare via a cached maximum, leaving the `BTreeSet` on the ~11,500
  accepted hashes where its deduplication is load-bearing. `containment`'s
  linear-merge alternative was already evaluated and rejected in the code,
  with the crossover arithmetic written out — the narrowed binary search
  wins on the small-query-against-large-reference shape `containment` exists
  to serve.

**The plan was stale on these three when it was written.** That is the useful
finding: it was drafted from the diagnosis documents rather than from the
tree as it stood that evening.

### Workstream 2 (`counter.rs`) — one change shipped, one deliberately not

**Shipped: MSD partition before `sort_unstable`.** `compact_raw` now splits
the raw buffer into 1024 ascending buckets on the top significant bits and
sorts each in place; because bucket order is key order, the concatenation is
already sorted and everything downstream is untouched. Measured in isolation
at the real call size on real canonical k-mers (not uniform `u64`, which
matters here — `min(forward, revcomp)` skews exactly the high bits this
buckets on), three coverage shapes, 20 runs each: **1.23–1.30x** against
`sort_unstable`. Re-measured with 1/4/8/14 concurrent workers, since the
change trades memory bandwidth and a 16 MB per-worker scratch buffer for
cache locality and every worker calls it at once: **1.26–1.37x, holding at
every thread count**. Full detail, including why this does not contradict the
LSD radix sort the same file records trying and reverting, is in
`msd_partition`'s doc comment.

These numbers are from a Linux container, not the native Windows release
build. They are isolated, CPU-bound, in-memory microbenchmarks with no I/O —
the same measurement class `compact_raw`'s existing radix note used to
justify a reversion — which is why they were considered sufficient to act on.
**A native Windows confirmation is still owed.**

The extra scratch buffer is accounted for in
`mem_estimate::PER_WORKER_RAW_TRANSIENT_BYTES` (now `threshold * 32`, was
`* 24`). `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD` was deliberately *not*
re-fit, since that would mean inventing what the five calibration runs would
have measured against code they never ran against; the effect is +1.5% at 8
threads and moves the prediction in the safe direction.

**Not shipped: a threshold change.** `RAW_FINALIZE_THRESHOLD` was swept over
500k–16M, 40,000,000 occurrences per point, two coverage shapes. The sweep
found no better value: the two shapes disagree about which direction to move
(high coverage is fastest at 8M, where low coverage is 37% slower than at its
own best), and the sweep only covers one side of the trade — the other side,
`consolidate`/merge cost driven by run count, needs the whole-file benchmark
that does not exist here. The table and the reasoning are recorded on the
constant. It stays at 2,000,000.

### Workstream 5 (`python/fastdna/`) — done

`multiomics.kmer_feature_table` was the row-by-row loop over Arrow data. It
built a `{kmer: frequency}` Python dict per sample, accumulated cross-sample
totals one `(sample, k-mer)` pair at a time, then materialised the output
with one `dict.get()` per cell — three Python passes over a table that is
`samples x vocabulary` cells by construction. It is now three Arrow passes: a
hash-aggregate for the totals, a two-key sort for the vocabulary, and one
`index_in`/`take`/`fill_null` per *sample* for the matrix.

Output is unchanged, checked rather than asserted: 400 randomised trials
(varying sample counts including zero, vocabularies, and `top_features`
values) comparing the new implementation against the pre-change body copied
verbatim, agreeing on column order, schema and every value.

Measured on native Windows CPython (no Rust build involved, so this plan's
build constraint does not apply), best of 5 runs:

```text
  10 samples x 20,000 k-mers |  0.721 s -> 0.426 s   1.69x
  50 samples x 20,000 k-mers |  0.945 s -> 0.637 s   1.48x
 200 samples x 20,000 k-mers |  3.101 s -> 1.652 s   1.88x
 200 samples x  5,000 k-mers |  0.723 s -> 0.427 s   1.69x
 500 samples x  4,000 k-mers |  1.333 s -> 0.720 s   1.85x
```

The plan's second item for this workstream — "stop rebuilding a sample's
sketch inside an O(n^2) all-pairs comparison" — was already done:
`compare_all` builds `n` sketches once and compares the unwrapped Rust
objects pairwise.

### Workstream 6 — model built, promotion declined, benchmark not obtainable

**Built:** `mem_estimate::estimate_binned_peak_bytes`, modelling the binned
strategy's three phases (accumulation, per-bin counting, cross-bin merge) and
returning their maximum. It reproduces `docs/design-minimizer-counting.md`
4.1's published table term for term and phase for phase — 777/829/64/113/821
MiB and 1.63/2.08/2.36 GiB — pinned by
`binned_estimate_reproduces_the_design_document_arithmetic` so the code and
that document cannot drift apart silently.

It is labelled **structural, not calibrated**, which this plan explicitly
permits, and the label is load-bearing: unlike the in-memory model's
constant, not one term has been compared against an observed RSS of a real
binned run. Of its two estimated inputs, `SUPERKMER_BYTES_PER_OCCURRENCE`
(1.035) is partly corroborated — `superkmer.rs`'s own test measures the
encoder storing 0.998 bytes per occurrence, so the model over-charges by
3.7%, which is the safe direction — while `DISTINCT_PER_OCCURRENCE` rests
entirely on one file's coverage shape and is the term most likely to be badly
wrong on unlike input. Both are documented with the direction they fail in.

**Promotion to `auto`: declined, on the evidence the plan itself named.** The
outstanding blocker is not the memory model any more — it is that
`adaptive_bins.rs` was validated against a *synthetic reproduction* of the
`DRR021372` bin-skew shape, never against the original ENA sample, for lack
of network access in the implementing environment. That is unchanged here.
`resolve_strategy`'s automatic branch is bit-identical to before, pinned by
`automatic_decisions_still_report_the_in_memory_model_unchanged` alongside
the pre-existing `the_automatic_chooser_never_selects_the_binned_strategy`.

**One real gap the model closed.** `StrategyDecision.estimated_peak_bytes` is
what the CLI prints. It was computed from the in-memory model regardless of
which strategy was chosen, so `--strategy binned` announced the in-memory
strategy's peak — on the benchmark input, 8.34 GB for a run the design says
costs 2.36 GiB. A user reaches for `binned` precisely to fit a machine it
otherwise would not fit, which makes that the worst line to leave wrong. It
now reports the chosen strategy's own model. (`Disk` still reports the
in-memory figure, which is also not right for it; building and justifying a
disk model was not in scope, and substituting a guess would be the invented
number this plan forbids, so it is left visibly as it was.)

**Not obtained: the before/after benchmark.** This workstream also owned "a
real, honest before/after benchmark number for `binned` vs. the default
strategy on the native Windows release build — the number that actually
answers *are we closer to KMC3 now*". It does not exist. There is no Rust
toolchain on the Windows host, so that build cannot be produced, and this
plan rules out substituting a WSL or Docker figure. **The question this plan
was written to answer therefore remains open**, and nothing above should be
read as having answered it.

### Verification

`cargo test --all-targets`: 358 lib tests plus all integration suites,
0 failures — including `tests/dual_strategy.rs`, which requires the counting
strategies to agree bit-for-bit and so covers the MSD partition end to end.
`cargo clippy --all-targets`: the same 8 warnings as the pre-change baseline,
none of them in a changed file, zero introduced. `pytest python/tests`: 629
passed, 40 skipped, 1 xfailed, 0 failures, against a real `maturin`-built
extension rather than the no-Rust stub.
