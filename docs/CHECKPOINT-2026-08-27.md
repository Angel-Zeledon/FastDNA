# Checkpoint — 2026-08-27

Written because the interactive session driving this work was ending mid-task
(background agent still running when the session was about to close). Verify
live-agent status first (`ListAgents`) before assuming anything below is
still in flight — it may have finished, or died with the session, same
uncertainty `docs/CHECKPOINT-2026-08-26.md` flagged for its own agents.

## What changed this session

1. **Strategic goal pivot, committed on `recovery/verified-2026-08-26`**:
   new `docs/goal-most-complete-genomics-ml-library.md` — FastDNA now
   aggressively pursues being the most complete library for genomics ML
   (closing every item in the gap docs, not cherry-picking).
   `docs/philosophy-narrow-not-broad.md` marked SUPERSEDED at the top (its
   "stay narrow" conclusion reversed; its technical corrections — the
   reverse-complement bit trick, the syncmer-vs-total-map argument — still
   hold, and the boundary against becoming a general bioinformatics
   framework — no aligner/pangenome/interval-algebra — still holds too).
2. **Q4 (CLI subcommands) shipped, verified, merged** (commits `3eff7fe`
   docs-audit, `c2addaf` implementation, both on `recovery/verified-2026-08-26`
   via fast-forward merge). `fastdna sketch|dist|card|peek` added
   alongside `count` (backward-compatible, verified byte-for-byte). Also
   corrected `docs/feature-gap-analysis.md`, `docs/ml-genomics-roadmap.md`,
   `docs/ml-differentiation-roadmap.md` status markers against actual code
   (many items marked "missing" were already shipped — see those docs'
   2026-08-27 audit notes for the full corrected table with file citations).
   Verified: Rust 364 lib + 522 integration tests, 0 failed, 1 ignored
   (intentional), exactly the same 8 preexisting clippy warnings as
   baseline, +18 new passing tests.
3. **S1 (binary k-mer DB + query layer) dispatched, failed on the account's
   API rate limit before writing any code** — see "In flight" below.
4. **Repo object corruption hit and recovered from, on the host network
   share.** Two commits made directly on the host path
   (`\\guayaba.online\UPIT\dev\angel.z\Documents\a\FastDNA`) — the goal-doc
   pivot and an earlier draft of this very checkpoint — silently wrote
   invalid/missing git objects (`git fsck` showed missing blobs/trees,
   `git status`/`git commit` started failing with "bad tree object HEAD" /
   "invalid object ... for docs/goal-most-complete-genomics-ml-library.md").
   This is the same failure class the repo already needed a "Repair
   corrupted commit trees" fix for once before (commit `fa70f7a`) — writing
   many small git objects directly onto this UNC/DFS network share appears
   unreliable. **Recovery pattern that worked**: confirm a recent commit's
   tree is still fully readable via `git cat-file -p <commit>` /
   `git cat-file -p <tree>` (here, `c2addaf` was intact), `git reset --soft`
   the branch back to that commit (does NOT touch the index or working
   directory — critically, plain `git reset <commit>`/`git reset` with no
   args tries to diff/rebuild the index and can itself fail trying to read
   the *broken* current tree; `--soft` sidesteps that by only moving HEAD),
   then redo the actual file-content commit **inside a local, non-network-
   share git clone** (there's already one sititng at
   `...\scratchpad\build-cli-subcommands`, intact, used for the Q4 merge)
   rather than retrying the write on the flaky share, and bring the
   resulting good commit back into the host repo with `git remote add` +
   `git fetch` + `git merge --ff-only` — the exact same pattern already
   proven for merging Q4 back. **Lesson for future sessions**: prefer
   committing doc/strategy changes in a local clone and merging back, same
   as code changes, rather than committing directly on the host UNC path,
   given this has now bitten the repo twice.

## Repository state

- `\\guayaba.online\UPIT\dev\angel.z\Documents\a\FastDNA`, branch
  `recovery/verified-2026-08-26`. HEAD should be at the commit created by
  this checkpoint's own recovery process (a single commit combining the
  goal-pivot docs + this checkpoint, made in the local clone and merged
  back) — check `git log --oneline -5` to confirm, and run `git fsck
  --full` once to confirm no new corruption before trusting it further.
- **Real, unrelated, still-uncommitted work sits in the working tree**:
  `src/counter.rs` and `src/minimizer.rs` (a SIMD/consolidation-threshold
  performance effort, not part of this session's work) — `git status`
  shows them modified. **Do not discard, commit blindly, or overwrite
  these** without understanding what they are first; they predate this
  session's changes and were deliberately left alone throughout (every
  agent dispatched this session was explicitly told not to touch them).
- No local Rust/Python toolchain — verification is Docker-only, and Docker
  **must** be invoked from a PowerShell tool, never Bash (path mangling).
  See the repo's own verification memory / `docs/CHECKPOINT-2026-08-26.md`
  for the exact working `docker run` command pattern.
- Baseline to not regress below: Rust 364 lib tests + 522 integration
  tests, 0 failed, 1 ignored; clippy exactly 8 warnings (locations listed
  in `docs/CHECKPOINT-2026-08-26.md`); Python 712 passed, 40 skipped, 1
  xfailed, 0 failed with the container's default extras.

## In flight — S1: binary k-mer database + random-access query API

First dispatch failed immediately with "You've hit your session limit ·
resets 4:20pm (America/Guatemala)" — an account-level API rate limit, the
same kind `docs/CHECKPOINT-2026-08-26.md` already documented hitting once
before (that time, a bare relaunch with the same prompt succeeded once the
limit cleared). It died mid-research, before writing any code — no commits
exist anywhere from this attempt; nothing is lost by simply redispatching
once the limit clears.

Task, for whoever redispatches it: implement S1 per
`docs/feature-gap-analysis.md`'s S1 entry — a sorted, exact, random-access
on-disk k-mer table format (either a custom `.ktab`-style binary file +
sampled index, or sorted Parquet with row-group pruning — pick and justify
one), a Rust reader/writer, a `fastdna query` CLI subcommand (follow the
`clap` `Subcommand` pattern `src/cli.rs`/`src/main.rs` established for Q4),
and a Python `KmerTable` type via `src/ffi.rs`. Fine to scope down to the
in-memory-strategy table shape only (not `disk_spill.rs`'s) if reconciling
both shapes is a bigger problem than expected, as long as that's stated
clearly. **Do the work in a local clone, not directly on the host UNC
path** (see the corruption note above), on a new branch off
`recovery/verified-2026-08-26`, and merge back the same way Q4 was merged.

## Still queued after S1 (per the corrected `feature-gap-analysis.md`)

- S2 (set operations between tables) and S4 (read filtering) — both
  explicitly depend on S1 landing first.
- Q5 (Bioconda) — recipe skeleton exists (`recipe/meta.yaml`) but has
  placeholder URL/sha256 and was never submitted; publishing (Bioconda,
  PyPI, crates.io) is **explicitly held for the user's direct go-ahead**,
  not something to dispatch autonomously — it's irreversible in a way code
  changes on an unpushed local branch are not.
- `assembly_qc.py` still uses a slow pure-Python FASTA k-mer extractor
  instead of the fast Rust path Q1 already made available (B4, found
  partially-shipped during this session's audit) — small, real gap, not
  yet queued as its own task.
- Snakemake/Nextflow workflow templates (part of the original wave-1
  "ecosystem" bundle, found not actually shipped during this session's
  audit despite being marked "dispatched" in `ml-genomics-roadmap.md`).

## Explicitly not done, by design

No `git push` was performed anywhere this session — everything above is
local-only on `recovery/verified-2026-08-26` and its unmerged feature
branches. No publishing (Bioconda/PyPI/crates.io) was attempted.
