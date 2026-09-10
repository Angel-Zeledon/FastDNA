# Checkpoint — 2026-08-25

State of the work when the machine went offline. Everything below was
verified, not assumed. Resume from "Where to pick up".

## Repository state

- Branch `master`, HEAD `bb26baf`. **The tree compiles and all tests pass**
  (252 Rust, 375 Python + 1 skipped + 1 xfailed), clippy clean.
- One uncommitted change: `Cargo.toml` gains `lto = "fat"` and
  `codegen-units = 1` in `[profile.release]`. **This was never benchmarked**
  — it was added and the verifying build was interrupted. Either measure it
  or revert it; do not assume it helps.
- Two unmerged feature branches, both complete-as-far-as-they-got:
  - `worktree-agent-ac648a57b9cf83988` — **DNA→protein translation, finished**
    (3 commits, 2,235 lines, 42 Rust + 41 Python tests). Merge decision is
    open: the landscape review says the feature has no demand (four dead
    Rust crates do this; OrfM has 24 stars), but the code exists and is good.
    Recommendation was: merge but do not headline it, since amino-acid
    k-mers feed the ML layer. **After merging, run `maturin develop` or
    `test_translate.py` fails on the new FFI functions.**
  - `worktree-agent-a2229f6338db9bddd` — **metagenomics, one commit only**
    (a validated taxonomy tree). Recommendation was to drop it: three Rust
    Kraken2 rewrites already exist and the published one has 3 citations.
    Cheap to discard.
- Older `worktree-agent-*` branches are all merged into master already.

## Work that was in flight when the machine went offline

Five optimization agents were editing the main working tree with disjoint
file scopes. **None had written anything yet** — `git status` showed only the
`Cargo.toml` change above, so nothing is half-edited. Their assignments,
worth re-launching:

| Scope | Target |
|---|---|
| `kmer.rs`, `export.rs` | Roll the reverse complement instead of recomputing it per k-mer; remove the per-read `Vec` allocation; build Arrow's `StringArray` from one contiguous buffer instead of 53.8M `String`s |
| `counter.rs`, `Cargo.toml` | LTO/codegen-units; MSD partition into cache-resident buckets before `sort_unstable`; sweep `RAW_FINALIZE_THRESHOLD` |
| `fastq.rs` | 21M per-record allocations; block reads instead of four `read_until` per record |
| `sketch.rs`, `hll.rs`, `cms.rs` | `BTreeSet` insertion per k-mer; `containment`'s binary search vs a linear merge |
| `python/fastdna/` | Python loops over Arrow rows; sketches rebuilt per pair in the O(n²) comparison |

## The benchmark, and why it matters

Measured on 2026-08-25, all in one WSL2 environment, 2.14 GB FASTQ,
840,000,000 k-mer occurrences, k=31, 8 threads, singletons kept
(`kmc -ci1`, `FastK -t1`, `fastdna -m 1 -q 0`):

| Tool | Time | Peak RSS |
|---|---:|---:|
| FastK | 38.3 s | 2.86 GB |
| KMC3 | 110.4 s | 9.06 GB |
| FastDNA in-memory | 388.2 s | 10.34 GB |
| FastDNA disk | 553.6 s | 1.21 GB |

Native Windows build, same machine and file: in-memory 133.1 s / 8.34 GB,
disk 281.8 s / 1.10 GB, auto 284.7 s / 1.09 GB. All FastDNA outputs are
byte-identical to each other.

**All four counts agree exactly at 53,776,394 distinct k-mers.** Correctness
is settled; speed is the problem. Counting with near-zero export is 110.1 s
of the 133.1 s Windows run, so export is ~23 s and is not the bottleneck.

The diagnosis: FastDNA sorts all 840M occurrences as independent `u64`s.
KMC3 turns the same input into 70,635,757 super-k-mers first — 11.9x fewer
items to sort. See `docs/design-minimizer-counting.md`.

## Where to pick up

1. **Step 0 is a mandatory gate before any minimizer work.** Measure how the
   110 s splits between the single-threaded producer, the workers, and the
   reduce. The cheapest version needs no code: run `mid.fastq` at `-t 1`,
   `-t 2`, `-t 4`, `-t 8` and see whether wall time scales or plateaus. If it
   plateaus, the single-threaded FASTQ parse is the ceiling and the
   super-k-mer redesign fixes the wrong thing.
2. **Re-launch the five optimization agents** (table above). They are
   independent of the minimizer decision.
3. **Then implement `docs/design-minimizer-counting.md`**, which records its
   predictions in advance so they can be falsified: counting 30-50 s, peak
   RSS 2.2-3.0 GB, and — the sharpest claim — peak memory becoming
   thread-independent within 250 MiB across 1/4/8 threads, where today it
   varies by gigabytes. The design is honest that we would likely still be
   ~2x behind FastK afterward.
4. **Decide the two open branches** (translation: merge; metagenomics: drop).

## Two things not to lose

- **Licensing.** KMC3 is **GPL-3.0** (verified against its LICENSE file).
  FastDNA ships as a Python extension, so copying any KMC code would
  propagate GPL-3 to every user of the package. Work from the papers only.
  FastK's license is **not** canonical BSD-3 — its clause reads "source code
  **or derivative codes** must retain the above copyright notice", and GitHub
  classifies it as unidentified. Reading either for understanding is fine;
  copying is not.
- **The disk strategy defaults into RAM on most Linux systems.** The first
  Linux disk run failed with "No space left on device" because `/tmp` is a
  6.4 GB tmpfs there — a RAM-backed filesystem. The disk strategy was
  spilling into memory, defeating its own purpose. `FASTDNA_SPILL_DIR` is
  the workaround and is what made the run succeed. **The real fix is still
  open**: detect that the spill directory is RAM-backed and warn or avoid it,
  rather than letting a user discover this through a failure ten minutes in.

## Benchmark assets (WSL, survive a reboot)

- `~/fastdna-bench/bench2gb.fastq` — the 2.14 GB dataset (md5 `857f3c57c2b825e6c5943e1eee3d23c9`, identical to the Windows copy at `C:\Users\Jahir\Documents\DNA-Rust\bench\`).
- `~/fastdna-bench/tools/bin/kmc`, `~/fastdna-bench/tools/FASTK/FastK` — built and working.
- `~/fastdna-src/` — FastDNA built for Linux, for same-environment comparison.
- `C:\Users\Jahir\Documents\DNA-Rust\bench\mid.fastq` — 376 MB slice, 144,000,000 occurrences, **37,597,718 distinct** at k=31. This is the fast iteration target; idle baseline 25.42 s / 3.32 GB in-memory.
- `scripts/bench/measure_windows.ps1` — wall clock and peak RSS as JSON.
- **`scripts/bench/kmc3_fastk_comparison.sh` has a bug**: it passes `-P fastk_work` with a space. FastK takes the value attached (`-Pdir`); with a space it treats the directory as an input file and dies. Fix it before trusting that script.
