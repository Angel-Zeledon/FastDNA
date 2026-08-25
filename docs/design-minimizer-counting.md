# Design: minimizer-partitioned super-k-mer counting

Status: **step 0 passed; implementation in progress.** This document is the
argument that decided whether we implement. See the update below for the
gate measurement; the rest of the document is unchanged from the proposal.

Date: 2026-08-25.

### Step 0 result (2026-08-25, later the same day)

Measured directly rather than inferred from wall-clock comparisons across
separate process launches (unreliable on the machine available — see
§4.2's own admission that time predictions are lower-confidence for exactly
this reason). Isolated the producer's own cost: parse `mid.fastq` (376 MB,
144,000,000 k-mer occurrences, the same generator scaled down) end to end,
discarding every record, with no channel, no workers, no k-mer extraction —
a direct upper bound, not an inference.

| Pass | Time | Note |
|---|---:|---:|
| Parse only (producer, isolated) | 781 ms / 785 ms (two runs) | Upper bound on producer cost |
| Parse + canonical k-mer extraction (still single-threaded, no counting) | 1353 ms / 1485 ms | Isolates extraction's own marginal cost: ~600 ms |
| Full single-threaded pipeline (producer + one worker: extraction + insertion + sort + compact) | 16,735 ms / 16,679 ms | The denominator |

**Producer upper bound: 4.7% of total, both runs, nearly identical absolute
times (785 ms vs 781 ms; 16,679 ms vs 16,735 ms).** Extraction adds another
~3.6%. The remaining **~92% is insertion, sorting and compaction** —
confirming §1.1's diagnosis directly rather than by elimination. Gate
threshold was producer >~36% of total (design's ~40s/110.1s); actual result
is roughly 8x under it. **Gate passed. Proceeding to steps 1-4.**

---

## 0. Provenance, and licensing settled first

### 0.1 How claims are tagged

| Tag | Meaning |
|---|---|
| **[code]** | Verified by reading FastDNA's own source on 2026-08-25. File and line given. |
| **[paper]** | Published literature. DOI or URL given. Quotations are verbatim. |
| **[fastk-src]** | Read from FastK's source tree (`~/fastdna-bench/tools/FASTK` in WSL, and the public repo) — headers, comments, README and constant declarations only, to resolve what an unpublished tool actually does. **No FastK code is reproduced, transliterated, or paraphrased line-by-line here or in the design proposed.** |
| **[judgment]** | My own inference or arithmetic. Falsifiable, and marked so it can be attacked separately from the facts under it. |

### 0.2 KMC — GPL-3.0. Read the paper, never the source.

I verified this myself by fetching
<https://raw.githubusercontent.com/refresh-bio/KMC/master/LICENSE> on
2026-08-25. It is the verbatim text of **GNU General Public License, Version 3,
29 June 2007**. Independently corroborated: every `kmc_core` source header
carries *"This file is a part of KMC software distributed under GNU GPL 3
licence."* There is no dual-licence or commercial exception in the repository.

FastDNA is dual **MIT OR Apache-2.0** and ships as a pyo3 extension module
(`python/fastdna/_core.pyd`, per `CLAUDE.md`). GPL-3 contamination of the Rust
crate would propagate to every user of the Python package and to `fastdna-ml`.

**Therefore: no KMC source may be opened during implementation of this design.**
Everything attributed to KMC below comes from the KMC2 paper and its
supplement, the KMC3 note, and the public option documentation. The
*signature rule itself* — "no `AAA` prefix, no `ACA` prefix, no interior
`AA`" — is a sentence in a published paper and is an idea, not copyrightable
expression; reimplementing it from that sentence is lawful. Transcribing KMC's
function that implements it is not, and this design does not require it.

### 0.3 FastK — permissive, but **not** canonical BSD-3-Clause. Read the caveat.

I read `~/fastdna-bench/tools/FASTK/LICENSE` directly in WSL **[fastk-src]**.
Header: `Copyright (c) 2020, Dr. Eugene W. Myers (EWM). All rights reserved.`
Clause 1, verbatim:

> · Redistributions of source code **or derivative codes** must retain the
> above copyright notice, this list of conditions and the following
> disclaimer.

Plus the standard binary-reproduction clause and a no-endorsement clause.

**Do not call this "BSD-3-Clause" without qualification.** GitHub's own licence
detector reports `{"key": "other", "spdx_id": "NOASSERTION"}` for the
repository, because clause 1 is modified from the canonical BSD text — "or
derivative codes" extends the attribution obligation past verbatim
redistribution in a way plain BSD-3 does not. It is permissive and
copyleft-free, and it would be compatible with our licensing *if we carried the
notice*. But "derivative codes" is exactly the entanglement we do not want in a
hot path we intend to maintain for years.

**This design therefore takes nothing from FastK but understanding.** Where
FastK's own comments taught me something structural it is tagged
**[fastk-src]** and stated as a fact about the algorithm.

### 0.4 A note on what "reading source" was for

FastK is **unpublished** — there is no paper or preprint. This was confirmed
independently: ntStat (*PLoS Comput Biol* 2026,
<https://doi.org/10.1371/journal.pcbi.1014158>) writes *"the widely used and
efficient, though unpublished, tool FASTK"* **[paper]**. For FastK there is no
"read the paper instead" option, so its headers and README are the primary
source, and are cited as such.

---

## 1. The problem, with the measured evidence

Benchmark of 2026-08-25: one 2.14 GB FASTQ, 7,000,000 reads × 150 bp, `k=31`,
8 threads, singletons kept, no quality trimming. All three tools agree on
**53,776,394 distinct k-mers out of 840,000,000 occurrences**. Correctness is
not in question.

| Tool | Wall clock | Peak RSS | Environment |
|---|---:|---:|---|
| FastK | 38.3 s | 2.86 GB | WSL2 Linux |
| KMC3 | 110.4 s | 9.06 GB | WSL2 Linux |
| FastDNA in-memory | 388.2 s | 10.34 GB | WSL2 Linux (12 GB VM) |
| FastDNA disk | 553.6 s | 1.21 GB | WSL2 Linux |
| FastDNA in-memory | 133.1 s | 8.34 GB | native Windows, same machine |
| FastDNA disk | 281.8 s | 1.10 GB | native Windows, same machine |

Of the 133.1 s Windows in-memory run, **110.1 s is counting**; Parquet export
is ~23 s and is not the bottleneck.

Note in passing that the same binary is **2.9× slower under WSL** while its
peak RSS (10.34 GB) exceeds what a 12 GB VM can hold alongside kernel and page
cache. That is not an OS quality difference — that run went to swap. It is the
first piece of evidence that FastDNA's problem is memory, and it returns in §5.

### 1.1 What the code does — verified

**[code]** `src/pipeline.rs:342-343`, the rayon worker hot path:

```rust
let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
local_counter.insert_batch(&canon_kmers);
```

`extract_canonical_kmers` (`src/kmer.rs:53`) returns a `Vec<u64>` with **one
8-byte word per k-mer occurrence** — 120 words per 150 bp read,
840,000,000 words over the file, **6.26 GiB of independent integers**, each
overlapping its neighbour by 30 of its 31 bases. `insert_batch`
(`src/counter.rs:458`) appends them to a `Vec<u64>`.

**The diagnosis in the brief is confirmed**: FastDNA materialises every
occurrence as an independent `u64` and exploits none of the k−1 overlap.

### 1.2 The compounding defect — also confirmed

**[code]** `src/pipeline.rs:317-321`: each of `config.num_threads` workers
builds its own `KmerCounter::with_capacity(131_072)`. `src/mem_estimate.rs:44-60`
explains why that is expensive, in its own words: batches reach workers
essentially at random and the same k-mers recur throughout a real FASTQ, so
**every worker's private table converges toward the entire distinct-k-mer
set**, not 1/N of it.

The calibrated model (`src/mem_estimate.rs:140`) is

```
peak ≈ 777 MiB + threads × 48 MiB + 1.168 × threads × occurrences
```

For 840,000,000 occurrences at 8 threads it predicts **8.43 GiB** against
**8.34 GB measured** — 1% error **[judgment: arithmetic mine, constants from
code]**. The model is sound; the `threads × occurrences` scaling is real.

### 1.3 Why the existing eager-compaction machinery cannot fix it

`counter.rs` already fights this: `RAW_FINALIZE_THRESHOLD = 2_000_000` forces a
sort-and-compact every 2M occurrences and `MAX_PENDING_RUNS = 8` bounds
unconsolidated runs. `docs/BENCHMARKS.md:261-279` records the measured cost
honestly — on *this exact file* the bounded buffer made every run **slower**:
"unbounded buffer 6.49 GB / 109.9 s and 8.81 GB / 87.6 s; current (bounded)
7.10 GB / 173.1 s and 7.66 GB / 125.3 s", because at ~105M occurrences per
worker each of ~52 eager compactions re-merges the entire running table, which
on a high-diversity sample never stops growing.

**[judgment]** That is the structural trap. With per-worker tables you either
buffer all occurrences (memory blows up) or compact repeatedly against a table
that grows to full size (time blows up). Tuning the threshold moves along that
curve; it does not leave it. The way off is to make each table *small* — which
means partitioning k-mer space so no worker ever holds more than a slice.

---

## 2. Literature

### 2.1 Minimizers — Roberts et al. 2004

Roberts M, Hayes W, Hunt BR, Mount SM, Yorke JA, "Reducing storage requirements
for biological sequence comparison", *Bioinformatics* 20(18):3363–3369, 2004.
<https://doi.org/10.1093/bioinformatics/bth408> **[paper]**

Definition, verbatim (§2.1, "Interior minimizers"):

> a set of w consecutive k-mers covers a string of exactly w + k − 1 letters …
> To find a minimizer, we examine w consecutive k-mers and select the smallest,
> in the sense of our chosen ordering. **In the case of a tie, each of the
> smallest k-mers is a minimizer.**

Note the order is a *parameter*, and note the tie rule — "all tied are
minimizers", not "leftmost wins". Modern implementations, including this
design, break ties by position; that is a deliberate deviation, and it is what
makes a super-k-mer a single well-defined run.

The sharing guarantee, verbatim (Property 1′):

> **If two strings have a substring of length w + k − 1 in common, then they
> have a (w, k)-minimizer in common.**

Density, verbatim (§3):

> the k-mers at position 1 and w + 1 each have equal probability of 1/(w + 1)
> of being minimizers. Thus, the probability that the minimizers of the two
> adjacent windows differ is 2/(w + 1), and hence on average, about a fraction
> **2/(w + 1)** of all k-mers are (w, k)-minimizers, **independent of k**.

And the caveat that matters for our budget, verbatim:

> Owing to correlations between adjacent k-mers … In our tests on random
> sequences and DNA sequences, **the actual proportion of k-mers that are
> minimizers can be a few percent above 2/(w + 1)**.

Why lexicographic order is bad, verbatim (§2.4, "Orderings"):

> If a string contains many consecutive zeros (or A s in the case of genomic
> data), then several consecutive k-mers may be minimizers … One can mitigate
> this effect by choosing an ordering in which the letters that occur least
> frequently are deemed minimal … **We assign the values 0, 1, 2, 3 to C, A, T,
> G, respectively, for the odd numbered bases of k-mers, and reverse the
> ordering for even numbered bases.**

Their reverse-complement handling — worth flagging because it is **not** what
KMC does and **not** what this design does, verbatim:

> we choose the minimizer of each window W to be **the smaller of the two
> minimizers from W and its reverse complement**.

That canonicalises the *window*. KMC (§2.2) canonicalises each *m-mer*
independently. The distinction matters for us and is settled in §3.3.

Two properties carry the whole design:

1. **Density** `2/(w+1)` under a random order — how much we compress.
2. **Locality** — consecutive windows overlap in `w−1` m-mers, so the minimizer
   usually does not change. **Super-k-mers exist only because of this**, and it
   is exactly what plain high-bit bucketing lacks (§3.1).

### 2.2 KMC2/KMC3 — signatures, super-k-mers, bins

Deorowicz S, Kokot M, Grabowski Sz, Debudaj-Grabysz A, "KMC 2: Fast and
resource-frugal k-mer counting", *Bioinformatics* 31(10):1569–1576, 2015,
<https://doi.org/10.1093/bioinformatics/btv022> (preprint
<https://arxiv.org/abs/1407.1507>). Kokot M, Długosz M, Deorowicz S, "KMC 3",
*Bioinformatics* 33(17):2759–2761, 2017,
<https://doi.org/10.1093/bioinformatics/btx304>. **[paper]**

**Signature — the exact rule, verified against the paper.** The rule stated in
the task brief is **correct**. KMC2 §2.2, verbatim:

> Since the origin of both problems are runs of As (especially as signature
> prefixes), **we propose to use as signatures canonical minimizers, but only
> such that do not start with AAA, neither start with ACA, neither contain AA
> anywhere except at their beginning.**

Signature length is `-p`, **default 7 in KMC2, raised to 9 in KMC3** ("larger
default signature length (nine instead of seven)", KMC3 note). `nBins` default
is **512**.

**Why each exclusion exists — the paper gives two distinct reasons, verbatim
(§2.1):**

> 1. The distribution of bin sizes is far from uniform. In particular, **the
>    bin associated with the minimizer AA…A is usually huge.** Other minimizers
>    with a few As in their prefix also tend to produce large bins.
> 2. **When a minimizer starts with a few As, then it often implies several new
>    super k-mers spanning a single k-mer only.** To given an example, with
>    m = 7 and AAAAAAC as the minimizer: when the minimizer falls off the
>    sliding window, so the current k-mer starts with AAAAAC, then AAAAACX (for
>    some X) will likely be the new minimizer; but unfortunately for yet
>    another window AAAACXY (for some Y) also has a fair chance to be a
>    minimizer, etc.

Reason 1 is bin balance. **Reason 2 is a density effect and it is specific to
lexicographic order** — the cascade only happens because an A-rich prefix makes
the *successor* m-mer also likely minimal. This is the key to §3.3.

**Super-k-mer**, verbatim: "contiguous areas containing k-mers having the same
canonical minimizer; they dub these areas as super k-mers."

**Measured reduction from the exclusions**, verbatim (§3):

> **using our signatures diminishes the average number of super k-mers in a
> read by about 10–15 percent.** Also the number of k-mers in the largest
> (disk) bin is significantly reduced, sometimes more than twice.

KMC2 Table 6 (*G. gallus*, 100 bp reads, k=28, avg super-k-mers per read):
plain minimizers 7.919 at p=7, signatures **6.728**. **[judgment]** Random-order
theory for that configuration is `2/(28−7+2) × (100−7+1) = 8.17`; plain
minimizers land at theory, signatures **18% below it**.

**Strand handling.** KMC canonicalises **each m-mer independently** and takes
the minimum of those canonical values — not Roberts' window canonicalisation.
The KMC2 paper says so in §2.1: signatures are "canonical minimizers, i.e., the
minima of all canonical m-mers from the k-mer", and Figure 1's worked example
annotates `Minimizer: rev comp(CGTT) = AACG`. Ineligible m-mers are mapped to a
sentinel value and routed to a dedicated last bin.

**Bins, and no global merge**, verbatim (§2.4):

> The number of possible signatures, 4^m, can be, however, quite large, e.g.,
> 16,384 for typical value m = 7. Thus, **to reduce the number of bins to at
> most 512, some signatures are merged** … in a preprocessing stage KMC 2 reads
> a small fraction of the input data, builds a histogram of found signatures,
> and finally merges the least frequent signatures.

and

> the bins, read from disk to a queue in the memory, are sorted and compacted
> by multiple sorter threads. **Finally, the completer stores the sorted bins
> in the output database on disk.**

**KMC's output is grouped by signature, not globally sorted by k-mer.** Its
`.kmc_pre` file is a signature→prefix-array map and a lookup is a binary search
within a signature's range. This matters in §3.4: FastDNA's sorted-output
contract is *stricter than KMC's*, so we cannot simply adopt KMC's ending.

KMC also uses **(k,x)-mers** as a further sort-input reduction: verbatim, "with
setting x = 3 the number of (k,x)-mers becomes about twice smaller than the
number of k-mers", yielding ">20% (and even 38% for H. sapiens 2 and k = 55)"
time savings. Noted as a later optimisation (§5, step 7).

**Our benchmark's KMC3 number, now explained.** KMC3 reported **70,635,757
super-k-mers** for 840,000,000 k-mers: **11.892 k-mers per super-k-mer**,
10.09 super-k-mers per 150 bp read. **[judgment]** Random-order theory at
KMC3's default `p=9` (`w = 23`) is `2/24 × 142 = 11.83` super-k-mers per read;
KMC achieved 10.09, **15% below theory** — landing exactly in the "10–15
percent" band the KMC2 paper reports for its exclusion rule. My earlier
suspicion that the exclusions *lower* density, not merely rebalance bins, is
confirmed by the paper's own Table 6. This is a real effect, and §3.3 decides
whether to buy it.

### 2.3 FastK

<https://github.com/thegenemyers/FASTK>. Unpublished (§0.4). All of the
following is **[fastk-src]**.

`FastK.h`'s header states the architecture: "a **novel minimizer-based
distribution scheme** that permits problems of arbitrary size, and a
**two-staged 'super-mer then weighted k-mer' sort** to acheive greater speed
when error rates are low (1% or less)."

`split.c`'s header: "First the **minimal core prefix trie** is found over the
**first 1 Gbp** of the data set and then the entire data set is scanned and
partitioned into **super-mers** that are sent to file buckets according to the
trie."

`count.c`'s header, phase 2: "Sort the super-mers. Sort the k-mers from each
super-mer **weighted by the # of times that super-mer occurs**."

Five facts that change how I read the benchmark:

1. **FastK partitions by minimizer too.** It is not an alternative to the
   super-k-mer architecture; it is a refinement of it. Both tools that beat us
   use the same core idea, so this design is not choosing between them.
2. **Its minimizer is a "padded" minimizer with a data-adaptive alphabet
   order**: a 5-base seed length extended by a padding refinement loop, over
   bases remapped by *observed frequency* rather than lexicographic order —
   Roberts' 2004 advice, taken to its logical end. Its minimizers are
   canonical.
3. **Its partition is learned, not fixed**: the trie is balanced from a 1 Gbp
   sample, and the number of buckets is *memory*-derived (from `-M`, default
   12 GB), not a constant. Compare KMC's static exclusion rule.
4. **Its second sort is over *distinct* super-mers, weighted.** With 30×
   coverage and low error, super-mer occurrences collapse to far fewer distinct
   super-mers, and the k-mer sort then runs over `distinct × ~12` items rather
   than 840M. **[judgment]** This is very likely a large part of the 38.3 s and
   is the biggest idea in FastK that is *not* in KMC. Its own README concedes
   the catch: "its relative speedup decreases with increasing error rate", and
   the header scopes it to "1% or less".
5. **No SIMD.** A grep for `immintrin|__m256|_mm_|AVX` across FastK's core
   files returns zero hits; its two main sorts are MSD radix. (KMC3, by
   contrast, ships an AVX2/SSE4/NEON radix sorter, RADULS.) FastK's speed is
   algorithmic, not vectorised — relevant to `src/simd.rs`'s dead AVX2 path.

FastK's `.ktab` **is** globally sorted: README, verbatim — "The k-mers in each
part are lexicographically ordered and the k-mers in part i are all less than
the k-mers in part i+1, i.e. the concatention of the N files in order of thread
index is sorted." So FastK does reconcile a minimizer partition with globally
sorted output, which is the §3.4 problem, solved.

FastK's canonical definition matches ours exactly: "the lexicographically
smaller of the two alternatives is termed canonical."

The only FastK-vs-KMC3 benchmark that exists is FastK's own README claim,
"about 2 times faster than KMC3 when counting 40-mers in a 50X HiFi data set".
**No independent published replication was found.**

### 2.4 Has the field moved past minimizers for this? No.

**Density theory improved; the practical answer did not change.**

- Marçais G, Pellow D, Bork D, Orenstein Y, Shamir R, Kingsford C, "Improving
  the performance of minimizers and winnowing schemes", *Bioinformatics*
  33(14):i110–i117, 2017, <https://doi.org/10.1093/bioinformatics/btx235>
  **[paper]**: "the winnowing scheme with a random ordering has expected
  density of **2/(w+1)**"; "the minimizers scheme with **lexicographic ordering
  has density greater than 2/(w+1)**"; and on why — "homo-polymer runs,
  particularly repeated As, can cause a lexicographic minimizer algorithm to
  select many consecutive k-mers as minimizers in genomics applications."
  Universal-hitting-set orders reach "densities below **1.8/(w+1)**".
- Zheng H, Kingsford C, Marçais G, "Improved design and analysis of practical
  minimizers", *Bioinformatics* 36(Suppl 1):i119–i127, 2020,
  <https://doi.org/10.1093/bioinformatics/btaa472> **[paper]**: a random
  minimizer's density is `2/(w+1) + o(1/w)` provided `k ≥ (3+ε)·log_σ(w)`. For
  DNA and `w ≈ 25` that threshold is around `k ≥ 7`. **At any k a counter would
  use, a good 64-bit hash order hits 2/(w+1) essentially exactly.** The
  absolute floor is `1/w`; the best practical published schemes reach
  ~`1.67/w`.
- Marçais G, DeBlasio D, Kingsford C, "Asymptotically optimal minimizers
  schemes", *Bioinformatics* 34(13):i13–i22, 2018,
  <https://doi.org/10.1093/bioinformatics/bty258>; Orenstein Y et al., DOCKS,
  *PLoS Comput Biol* 13(10):e1005777, 2017,
  <https://doi.org/10.1371/journal.pcbi.1005777> **[paper]**: optimality is
  asymptotic in k, and DOCKS-style hitting sets are computable only to k ≈ 13
  with a day of precomputation.

**[judgment]** Total remaining headroom from twenty years of density theory is
roughly 10–17% fewer sampled positions, available only via a precomputed,
`w`-specific universal hitting set. For a k-mer counter that is not worth it.
Use a random hash order.

**Syncmers do not apply here.** Edgar R, "Syncmers are more sensitive than
minimizers for selecting conserved k-mers in biological sequences", *PeerJ*
9:e10805, 2021, <https://doi.org/10.7717/peerj.10805> **[paper]**. An *open
syncmer* is a k-mer whose minimal s-mer sits at the first position; a *closed
syncmer* is one whose minimal s-mer is first or last. Densities: closed
`2/(k−s+1)`, open `1/(k−s+1)`. The headline advantage is context-freedom —
"Unlike a minimizer, a syncmer is identified by its sequence alone", so "minimizers
can be deleted by mutations in flanking sequence, which cannot happen with
syncmers."

That is exactly why they are wrong for this job. **A counter needs a total map
from every k-mer to a bin; a syncmer rule is a selection predicate that
declines to select most k-mers**, and open syncmers with an offset have no
window guarantee at all — a stretch of sequence can contain none. You cannot
bucket what the rule declines to select. A survey search across KMC3, FastK,
Gerbil, BCALM2, Bifrost, SSHash, Discount and Brisk found **no k-mer counter
that partitions by syncmers**; the 2024 review "When less is more: sketching
with minimizers in genomics" (*Genome Biology*,
<https://doi.org/10.1186/s13059-024-03414-4>) places every bucketing use on
minimizers and every syncmer use on seeding and sketching. **[judgment] Anyone
proposing syncmers for this problem has confused sampling with partitioning.**

Where syncmers *do* legitimately enter is as the *ordering inside* a minimizer
scheme, which stays total: the miniception samples the smallest closed syncmer
in the window, and "The open-closed mod-minimizer algorithm" (*Algorithms Mol
Biol* 2025, <https://doi.org/10.1186/s13015-025-00270-0>) extends it. A
possible future refinement, not a replacement.

**Other modern work, and the honest verdict.**

- **SSHash** (Pibiri GE, "Sparse and skew hashing of k-mers", *Bioinformatics*
  38(Suppl 1):i185–i194, 2022,
  <https://doi.org/10.1093/bioinformatics/btac245>) **[paper]** uses minimizers
  for exactly the bucketing role proposed here — "A super-k-mer of 𝒮 [is] a
  maximal sequence of consecutive k-mers having the same minimizer" — but to
  *index* a known k-mer set (8.28 bits/k-mer on human). A candidate for
  FastDNA's planned `S1` binary table (`docs/feature-gap-analysis.md`), not for
  the counting hot path.
- **Kraken2** (Wood DE, Lu J, Langmead B, *Genome Biology* 20:257, 2019,
  <https://doi.org/10.1186/s13059-019-1891-0>) **[paper]** uses "the
  lexicographically smallest canonical ℓ-mer", a deque-based sliding-window
  minimum ("an average of O(1) time to calculate a new minimizer"), and an
  **XOR shuffle** that "serves to permute the ordering of the ℓ-mers … and
  helps to **avoid a bias toward low-complexity ℓ-mers**" — Roberts' lesson,
  reached independently, and the same conclusion this design reaches in §3.3.
  Kraken2 also notes the ordering constraint we depend on: "By performing
  canonicalization of the minimizer candidates **prior to** applying the spaced
  seed mask, we ensure the result is the same whether applied to the ℓ-mer or
  its reverse complement."
- **Is anything beating KMC3/FastK on exact counting in 2026? No.** ntStat
  (*PLoS Comput Biol* 2026, <https://doi.org/10.1371/journal.pcbi.1014158>)
  **[paper]** reports "**KMC3 was faster overall** but at a high disk usage and
  memory cost" — "3.7× for k = 25 and 1.6× for k = 64" — nine years after
  KMC3 shipped. KCOSS (*Bioinformatics* 2022) wins only on assembled genomes;
  CHTKC (*Brief Bioinform* 2021) concludes defensively that hash tables
  "remain a feasible solution"; Gerbil is still the main GPU counter (2017) and
  no 2023–2026 GPU counter displacing KMC3/FastK was found. Recent activity has
  moved to dictionaries, indexes and graph construction (GGCAT, Brisk), not
  counting.
- **Bin skew is a known, named problem.** Discount (*Bioinformatics* 2021,
  <https://doi.org/10.1093/bioinformatics/btab156>) **[paper]** states it
  plainly: "**minimizers are known to generate bins of very different sizes**,
  which can pose challenges for distributed and parallel processing, as well as
  generally increase memory requirements." Its universal frequency ordering
  achieves counting "in as little as **1/8 of the memory** of comparable
  approaches". This is §6's R3.
- **The minimizer step itself is cheap and there is Rust prior art.**
  simd-minimizers (<https://doi.org/10.1101/2025.01.27.634998>) **[paper]**
  reports finding all canonical minimizers of a **3.2 Gbp human genome in 6.7
  seconds** using AVX2 and a deterministic two-stacks sliding-window minimum,
  "over 15× faster than the existing implementation in the minimizer-iter
  crate". **[judgment]** At ~478 Mbp/s that puts our 1.05 Gbase at ~2.2 s
  single-threaded, ~0.3 s across 8 workers — **the minimizer computation is not
  a new bottleneck**, which pre-empts the obvious objection to this design.

**Conclusion [judgment]:** minimizer → super-k-mer → bin → sort-and-compact is
still the state of the art for exact k-mer counting in 2026. FastK's weighted
super-mer sort is the most recent meaningful advance on it. There is no newer
paradigm we are missing.

---

## 3. The design

### 3.1 Why the two defects need different fixes — and why only one needs minimizers

This separation is the clearest way to see what the change buys, and it is
easy to get wrong.

- **Defect A (per-worker duplication, §1.2)** is fixed by *any* partition of
  k-mer space into disjoint bins owned by one consumer each. High-bit bucketing
  already does this — `disk_spill.rs` already implements it.
- **Defect B (every occurrence materialised as 8 bytes, §1.1)** requires
  super-k-mers, and **super-k-mers require a minimizer**. A super-k-mer is a
  maximal run of consecutive k-mers sharing a bin; such runs exist only if the
  bin function is *stable along a read*. A minimizer is stable by construction
  (§2.1, locality). High-bit bucketing of canonical k-mers is not: consecutive
  k-mers have essentially unrelated high bits, so every run has length 1 and
  there is nothing to compress.

But the two are coupled, and this is the actual argument for the whole design:

> **A partition-then-sort design must buffer every occurrence of a bin before
> it can count that bin, and a bin only closes at end of input. So it must
> buffer *all* occurrences.** As `u64` that is 6.26 GiB — which is why
> `disk_spill.rs` has to go to disk, and why the disk strategy is 2.1× slower.
> As super-k-mers it is **0.81 GiB**, which fits in memory comfortably.
>
> **Super-k-mers are what make the partitioned design viable in RAM.** That,
> and not the raw compression ratio, is the thesis of this document.

### 3.2 Shape

```
Phase 1 (parallel over reads, N workers)
  read → QC → quality-trim → for each maximal N-free stretch:
      rolling canonical minimizer per k-mer window
      cut into super-k-mers at every minimizer change
      append each super-k-mer to its signature's bin
  ⇒ NUM_BINS append-only chunk lists, ~0.81 GiB total

Phase 2 (parallel over bins, one bin per worker)
  expand a bin's super-k-mers → Vec<u64> canonical k-mers   (~1.6 M)
  sort_unstable + compact → sorted (kmer, count)            (~105 k)
  free that bin's super-k-mer chunks

Merge
  streaming k-way merge over NUM_BINS sorted, disjoint tables
  → globally sorted table → KmerCounter::from_sorted_entries → export
```

The `crossbeam_channel::bounded(64)` producer/consumer front end
(`src/pipeline.rs:214-239`), all QC handling, `quality_trim_end`, the
`catch_unwind` worker guard and the whole of `export.rs` are **untouched**.
Only what a worker does with a record's sequence changes.

### 3.3 Minimizer choice — scheme, window, canonicality, exclusions

**Chosen: canonical m-mer, random 64-bit hash order, `m = 7`, window
`w = k − m + 1 = 25` for k=31, ties broken by leftmost position.**

#### Canonical m-mers — the load-bearing correctness argument

FastDNA canonicalises each k-mer independently: `canon(x) = min(x, rc(x))`
(`src/kmer.rs:44`) **[code]**. A bin function must therefore satisfy
`bin(x) == bin(rc(x))`, or the two strand orientations of one physical k-mer
land in different bins, each bin counts its own half, and the exported table
contains the same canonical k-mer twice with split counts. **That failure is
silent**: the output is a well-formed sorted Parquet file with plausible
numbers.

Define

```
sig(x) = min over the k−m+1 positions of  h( canon_m(mmer_i) )
```

where `canon_m(y) = min(y, rc_m(y))` on m-mers and `h` is a fixed 64-bit mixer.

**Claim: `sig(x) == sig(rc(x))` for every k-mer `x`.** The multiset of m-mers of
`rc(x)` is exactly the multiset of reverse complements of the m-mers of `x`,
and `canon_m` maps a value and its reverse complement to the same
representative. The two multisets of canonical m-mers are therefore identical,
and so are their minima under any fixed `h`. **[judgment: the argument is mine;
the technique is KMC's per-m-mer canonicalisation (§2.2), which the KMC2 paper
states explicitly, and Kraken2 documents the same ordering constraint.]**

This is why we follow **KMC's per-m-mer canonicalisation and not Roberts'
window canonicalisation** (§2.1). Roberts' rule — take the smaller of the
window's minimizer and its reverse complement's minimizer — is also
strand-invariant, but it requires evaluating a second reverse-complemented
window, which is strictly more work for the same guarantee.

**Known subtlety, flagged not hidden.** Canonicalisation makes the hash
distribution over m-mers slightly non-uniform (each canonical value stands for
two forward values, except palindromes), which perturbs density. Marçais et al.,
"k-nonical space: sketching with reverse complements", *Bioinformatics* 2024,
<https://doi.org/10.1093/bioinformatics/btae629> **[paper]** treats this. Two
mitigations: choose **odd `m`** so no m-mer can equal its own reverse
complement (`m = 7` satisfies this), and measure the realised density in step 1
rather than assuming it (§5).

#### Random hash order, not lexicographic

Roberts §2.1 and Marçais 2017 §2.4 both say lexicographic order on DNA is
pathological on poly-A, and Marçais 2017 measures it as *above* `2/(w+1)`.
Zheng 2020 shows a hash order hits `2/(w+1)` essentially exactly at any k we
use. Kraken2 reached the same conclusion independently with its XOR shuffle.
**The mixer must be applied to the *canonical* m-mer**, or strand invariance
(above) is lost.

#### Exclusions: what we adopt, what we decline, and the price — quantified

KMC's rule buys two things (§2.2), and under a hash order they separate:

- **Reason 2 (fewer super-k-mers) is a lexicographic-order artefact.** KMC's
  own worked example — `AAAAAAC` → `AAAAACX` → `AAAACXY` — is a cascade that
  only occurs because an A-rich prefix makes the *successor* m-mer also likely
  minimal under lexicographic order. **[judgment] Under a random hash order
  that cascade does not exist, so most of the measured 10–15% density benefit
  is not available to us and adopting the full rule would not reproduce it.**
- **Reason 1 (bin balance) survives any ordering.** A hash order removes the
  *a priori* bias — poly-A is no longer preferentially smallest — but not the
  *empirical* one: poly-A tracts are genuinely over-represented in real reads,
  so whichever bin holds `A^m` still receives far more than `1/4^m` of the
  data.

Therefore this design adopts a **narrow rule aimed only at Reason 1**:

> A canonical m-mer is **ineligible** as a signature if it is a homopolymer
> (all A, C, G or T), or if it begins with `AAA`.

Applied to the *canonical* m-mer this is automatically symmetric across strands
— `canon_m(A^7) = A^7` because `rc(A^7) = T^7 > A^7` — so it cannot reintroduce
the asymmetry just eliminated. (KMC needs explicit reverse-complement duals in
its rule for the same reason; we get them free by testing the canonical form.)

KMC's "no interior `AA`" clause is **deliberately declined**: it disqualifies a
large fraction of the m-mer space, and its benefit is the lexicographic cascade
we do not have.

**The price, stated in numbers rather than adjectives.** KMC's full rule buys
~15% fewer super-k-mers. On this input that is 829 MiB versus ~773 MiB
(§3.5) — **56 MiB on a term that is under 10% of predicted peak memory, or
under 2% of the total.** Not worth reimplementing a rule whose density benefit
we cannot inherit anyway. If measured bin skew is bad on real data (§6, R3),
adopting more of KMC's rule, or FastK's sampled-and-balanced approach, is a
contained follow-up.

**Fallback when every m-mer in a window is ineligible** — a pure poly-A read,
which real data contains. Assign signature `0`, route to a dedicated
**overflow bin 0**. This mirrors KMC, which maps ineligible signatures to a
sentinel and a dedicated last bin (§2.2). That bin can be large in
*occurrences* but is tiny in *distinct* k-mers, so it costs sort time, not
memory. It needs its own test.

#### Bin mapping and count

`bin = ((h >> 32) as usize) & (NUM_BINS − 1)`, reusing the already-computed
hash. `NUM_BINS = 512` by default, a power of two — the same default as KMC.
This is a **static** map, unlike FastK's learned trie and KMC's
frequency-histogram merge; adopting a sampled balancing pass is a later
refinement, not part of this landing.

| Bins | occurrences/bin | expansion buffer | distinct/bin | phase-1 open chunks (8 threads × 16 KiB) |
|---:|---:|---:|---:|---:|
| 256 | 3.28 M | 25.0 MiB | 210 k | 32 MiB |
| **512** | **1.64 M** | **12.5 MiB** | **105 k** | **64 MiB** |
| 1024 | 0.82 M | 6.3 MiB | 53 k | 128 MiB |

512 puts a bin's whole expansion buffer at 12.5 MiB — L3-resident on most
machines FastDNA runs on, so the per-bin sort is cache-local — while keeping
`threads × bins × chunk_bytes` at a defensible 64 MiB. A starting point sized
the same way `DEFAULT_BUCKET_BITS` and `RAW_FINALIZE_THRESHOLD` were, not a
value proven by a sweep. It must be configurable.

**On `m = 7` vs KMC3's `9`.** KMC3 raised its default to 9 for finer-grained
bin assignment (4^9 = 262,144 signatures to distribute over 512 bins, versus
16,384). Smaller `m` gives a larger window and therefore lower density and
better compression: `m=7` gives `2/26 = 0.0769` against `m=9`'s
`2/24 = 0.0833`, an 8% difference in super-k-mer count. Since we derive the bin
from a hash rather than from a frequency-balanced signature map, the
finer-grained signature space buys us less than it buys KMC. `m = 7`,
configurable. **[judgment]**

#### Rolling computation

A deque-based sliding-window minimum, as Kraken2 documents ("an average of O(1)
time to calculate a new minimizer"), or the deterministic two-stacks variant
that simd-minimizers uses. Amortised O(1) per base. Per §2.4, the cost is
~0.3 s across 8 workers on this input — negligible against a 110 s baseline.

### 3.4 Super-k-mer representation — concrete type, byte layout, measured size

A bin is a list of fixed-size chunks; a chunk is a flat byte buffer of
back-to-back variable-length super-k-mer records.

```
Record layout — byte-aligned, no padding between records:

  offset 0   : u8   n_bases      (k ..= 255; 0 terminates a chunk)
  offset 1.. : u8[] packed bases, ceil(n_bases / 4) bytes,
               2 bits per base, A=00 C=01 G=10 T=11 — the same encoding as
               src/kmer.rs::base_to_bits — first base in the high bits of
               the first byte

Record size = 1 + ceil(n_bases / 4) bytes.
```

Rust types (declarations only; this is a design document):

```rust
/// One bin's accumulated super-k-mers. Chunks are handed over whole, so
/// the hot path never touches a lock.
pub struct BinStore { chunks: Vec<Chunk> }

/// A fixed-capacity byte buffer. CHUNK_BYTES = 16 * 1024.
pub struct Chunk { bytes: Box<[u8; CHUNK_BYTES]>, len: usize }

/// Per-worker open chunk per bin; filled lock-free, published on overflow.
pub struct BinWriter { open: Vec<Chunk> /* len == NUM_BINS */ }
```

This is the same shape FastK uses for its sort records — 2-bit packed sequence
plus an explicit length field **[fastk-src]** — arrived at independently
because there is not much choice.

**Why `u8` for the length.** `n_bases ≤ 255` covers any Illumina read. Long
reads are handled by capping a super-k-mer at 255 bases and starting a new one
— always *safe* (it only splits a run; every k-mer still reaches its correct
bin) at a cost of one extra `k−1` overlap per 255 bases, ~12% expansion on
ONT-length reads. FastK does the same thing with its `MAX_SUPER` cap
**[fastk-src]**. **[judgment]** A `u16` length would cost one extra byte on
each of 77 million records (74 MiB) to save 12% on a read type FastDNA does not
properly support anyway (`docs/feature-gap-analysis.md` lists long-read mode as
a separate project).

**Size for the benchmark input.** `m = 7`, random hash order, `w = 25`, density
`2/26 = 0.0769`, so `(150 − 7 + 1) × 0.0769 = 11.08` super-k-mers per read
→ **77,538,461 super-k-mers**, averaging `120 / 11.08 = 10.83` k-mers and
therefore `10.83 + 30 = 40.83` bases each.

| Quantity | Value |
|---|---:|
| Total bases stored | 3.166 × 10⁹ |
| 2-bit packed payload | 755 MiB |
| Length bytes (1 per record) | 74 MiB |
| **Super-k-mer store** | **829 MiB** |
| Current representation (840M × 8 B) | **6,406 MiB** |
| **Byte reduction** | **7.73×** |
| Bytes per k-mer occurrence | 1.035 |

At KMC's *observed* density (70,635,757 super-k-mers) the same arithmetic gives
**773 MiB and 8.29×**.

**The 11.9× claim, checked and corrected [judgment].** 11.9× is the reduction
in the *number of items* (840M → 70.6M). The reduction in *bytes moved* is
**7.7–8.3×**, because each super-k-mer carries `k−1 = 30` redundant head bases.
Both are true; the byte figure predicts memory-bandwidth cost and is the one
this design commits to. Note also that the super-k-mer store is **3.3× larger
than the reads themselves** 2-bit packed (1.05 Gbase = 250 MiB) — that overlap
is not free, and a claim of "we just store the reads" would be wrong.

Roberts' own caveat (§2.1) applies: on real DNA the realised density can run "a
few percent above 2/(w+1)". Budget accordingly; §4.3 sets 900 MiB as the
falsification threshold.

### 3.5 The sort-order problem — resolved explicitly

`src/disk_spill.rs`'s module header (lines 29-45) already states the constraint,
and it is correct **[code]**:

> because canonical k-mers are compared as plain integers, high-bit bucketing
> has the property that bucket 0 through bucket `2^bits − 1` covers the k-mer
> space in ascending order — concatenating each bucket's sorted merge output in
> bucket order is *already* the final globally sorted table … A minimizer
> scheme loses that property.

It is lost, irrecoverably: a minimizer bin holds k-mers scattered across the
whole `u64` range by construction. Note that **KMC simply accepts this** — its
database is grouped by signature and a lookup binary-searches within a
signature's range (§2.2). We cannot copy that ending, because
`export.rs`, the GenomeScope histogram path and the planned S1 binary table all
assume ascending `kmer_u64`. **FastK does keep global sort**, at the cost of a
redistribution into ordered output parts **[fastk-src]**.

| Option | Cost | Verdict |
|---|---|---|
| **A. Final streaming k-way merge over the 512 sorted bins** | 53.8M items × log₂512 = 9 comparisons; one pass over the **compacted** 821 MiB table, not over the 6.26 GiB of occurrences | **Chosen** |
| B. Keep a separate index, export grouped by bin (KMC's ending) | Breaks the sorted-output contract three consumers depend on | Rejected |
| C. Sort the whole 53.8M-entry table at the end | O(n log n) over 821 MiB instead of O(n log B), for no benefit | Rejected |

**Cost of A, honestly.**

- *Time:* one pass over 53,776,394 already-sorted entries with a 512-entry
  binary heap. **[judgment]** ~1–3 s single-threaded; parallelisable by key
  range if it ever matters. Noise against a 110 s baseline.
- *Machinery:* already exists. `counter.rs::k_way_merge_sorted_counts`
  (`src/counter.rs:226`) and `disk_spill.rs::merge_sources_into`
  (`src/disk_spill.rs:344`) are exactly this algorithm, over in-memory slices
  and over file readers respectively **[code]**.
- *Memory:* the merge is streaming, so nothing new is materialised — **except**
  that `export.rs` takes `&KmerCounter` (`src/export.rs:52`, `:130`)
  **[code]**, so the merged table must be materialised. During the merge the
  512 bin tables (821 MiB) and the growing output (up to 821 MiB) are both
  alive: a **transient peak of ~1.6 GiB**. Refactoring the exporters to consume
  an `Iterator<Item = (u64, u32)>` removes it entirely; it is a separate change
  with its own test surface and is listed as optional in §5.

**Consequence for `disk_spill.rs`.** If the disk strategy later adopts the same
bin function (§5, step 8), `merge_buckets` (`src/disk_spill.rs:407`) can no
longer append buckets in bucket order — it needs the same final merge. The test
`bucket_of_is_monotonic_in_kmer_value_for_a_fixed_k` (`src/disk_spill.rs:494`)
pins the property that would be lost and must be **replaced deliberately and
visibly**, never silently deleted. That test is the clearest signal in the
codebase that this is a change of contract, not a drop-in.

### 3.6 Per-worker duplication — solved, and why

**Yes, the minimizer partition eliminates it, and this is the larger half of
the win.**

Not because workers "share" bins under contention, but because **what a worker
accumulates changes kind**. Today a worker accumulates a *count table* — a
summary of the k-mers it happened to see — and since batches are randomly
distributed, every worker's table converges to the full 53.8M-entry set
(`src/mem_estimate.rs:44-60`). In the new design a worker accumulates
*super-k-mer bytes*, which are a **partition** of the input, not a replicated
summary: worker 3's and worker 5's contributions to bin 17 are disjoint pieces
of one 829 MiB total. Eight workers hold 829 MiB *between* them, not each.

**Contention.** The hot path is lock-free: a worker appends to its own open
`Chunk` for the target bin — a bounds check and a ~12-byte memcpy. Only when a
chunk fills does it publish, taking one `Mutex<Vec<Chunk>>` per bin per 16 KiB.
Over the whole run that is `829 MiB / 16 KiB ≈ 53,000` acquisitions spread
across 8 threads and 512 independent mutexes. **[judgment]** Uncontended by any
reasonable measure; a lock-free stack would be premature.

**What survives of the `threads ×` scaling.** The `threads × occurrences` term
vanishes. Two smaller thread-proportional terms remain, both bounded and both
independent of input size:

- open chunks: `threads × NUM_BINS × CHUNK_BYTES` = 8 × 512 × 16 KiB = **64 MiB**
- phase-2 transients: `threads × (bin_occ × 8 B + bin_distinct × 16 B)` = **113 MiB**

> **This is the falsifiable structural claim of the design: after this change,
> peak RSS on a fixed input should vary by under ~250 MiB between 1 and 8
> threads.** Today it varies by gigabytes.

### 3.7 One free win found while reading the code

**[code]** `Vec<(u64, u32)>` is the count-table representation everywhere
(`counter.rs`, `disk_spill.rs`). `size_of::<(u64,u32)>()` is **16**, not 12 —
alignment padding wastes 4 bytes per entry. On 53,776,394 distinct k-mers that
is **205 MiB thrown away**, and 25% of the bytes moved in every merge pass.
Parallel `Vec<u64>` keys and `Vec<u32>` counts recover it. **[judgment]** This
is independent of everything else here, is a few hours of work, and should
probably be done first regardless of whether the rest is approved.

---

## 4. Predicted result — recorded before implementation

All predictions are for the **native Windows** configuration of the 2026-08-25
benchmark, because that is the configuration with a clean, non-swapping
baseline: **133.1 s total / 110.1 s counting / 8.34 GB peak**.

### 4.1 Memory — high confidence

| Term | Predicted |
|---|---:|
| Base overhead (channel, allocator, OS), from `mem_estimate.rs` | 777 MiB |
| Super-k-mer store | 829 MiB |
| Open chunks (8 × 512 × 16 KiB) | 64 MiB |
| Phase-2 per-bin transients | 113 MiB |
| Final table (53.8M × 16 B) | 821 MiB |
| Transient during the cross-bin merge | +821 MiB |

Phase 1 peaks near **1.63 GiB**; phase 2 near **2.08 GiB** (the super-k-mer
store is freed bin by bin as it is consumed, so it and the growing output
overlap only partially); the merge near **2.36 GiB**.

> **Prediction: peak RSS 2.2–3.0 GB. I will call the design wrong above
> 3.5 GB.** That is a 2.8–3.8× reduction from 8.34 GB and puts FastDNA in the
> same class as FastK's measured 2.86 GB.
>
> **Prediction: peak RSS becomes essentially thread-independent** — under
> 250 MiB spread across 1, 4 and 8 threads on the same input.

This is the prediction I am most confident in: it is arithmetic over sizes I
can compute exactly, not a guess about a machine.

### 4.2 Time — lower confidence, and here is why

**[judgment]** I tried to account for the current 110.1 s counting phase from
first principles and **could not**. Summing the plausible costs — 6.26 GiB of
sequential `Vec` pushes, 840M elements sorted once in 2M-element chunks, ~6.5
consolidation passes per worker over a table growing to ~640 MiB, and a final
8-way reduce over near-full tables — gives roughly 60–100 GiB of memory
traffic, which at a realistic 2–5 GiB/s effective rate is 20–50 s, not 110 s.
The residual is presumably allocator behaviour and page pressure at 8.34 GB on
a 16 GB machine. **Any time prediction inherits that uncertainty, and I am not
going to hide it.**

What I can bound is the *ratio*. The new design's counting traffic is: write
0.81 GiB of super-k-mers; read them back; expand to 6.26 GiB of `u64` in
12.5 MiB cache-resident slices; sort in place (2–3 effective passes); compact
to 0.80 GiB; merge once over 0.80 GiB. Total ≈ **22–28 GiB** against 60–100 GiB
today — a **3–4× traffic reduction**, all sequential, most of it L3-resident,
with the repeated whole-table re-merges (§1.3) gone entirely. Per §2.4 the new
minimizer work adds ~0.3 s.

> **Prediction: counting phase 30–50 s (from 110.1 s). Total wall clock
> 55–75 s (from 133.1 s), since the ~23 s export is unchanged.**
>
> **Prediction on WSL2 Linux: 70–110 s (from 388.2 s).** Most of that reduction
> is *not* algorithmic — it is that a ~2.5 GB peak fits in a 12 GB VM and a
> 10.34 GB peak does not. If this holds, most of the WSL gap was swapping,
> which is itself a testable sub-claim.

### 4.3 What result would mean the design is wrong

In falsification order:

1. **Peak RSS above 3.5 GB.** The representation or the partition is not doing
   what §3.4/§3.6 claim. The cleanest failure — the memory arithmetic has the
   least slack of anything here.
2. **Peak RSS still scales visibly with thread count** (>500 MiB spread across
   1/4/8 threads). Then per-worker duplication was not eliminated, most likely
   because something in phase 2 accumulates per worker.
3. **Total wall clock above 100 s on Windows.** Then the memory-bandwidth
   diagnosis (§1.1) is wrong or badly weighted and the bottleneck is elsewhere.
   §5 step 0 exists to make this outcome unlikely *before* the work is done.
4. **Counts differ from the current implementation on any input.** Not a
   performance failure — a correctness failure, and the one that must be
   impossible to ship (§5 step 2; §6 R2).
5. **Super-k-mer store materially above 900 MiB.** The density assumption is
   wrong for this data — check `m`, the exclusion rule, the canonical-m-mer
   density perturbation (§3.3), and the `N`-cut policy.

---

## 5. Migration path

The counter is working, tested and correct: **252 `#[test]` functions across
`src/` and `tests/`**, counted 2026-08-25 **[code]**. Every step below is
independently testable and leaves all 252 green until the step that
deliberately changes a contract.

**Step 0 — measure before building (mandatory gate).** Instrument the existing
in-memory path to split wall time three ways: producer thread (FASTQ read +
parse), worker loop, reduce phase. Run on the benchmark file. **If the
single-threaded producer alone accounts for more than ~40 s of the 110.1 s,
stop** — this design cannot deliver its predicted win and the correct fix is
parallel FASTQ parsing instead. One day of work that can save weeks, and it
directly de-risks failure mode 4.3(3).

**Step 1 — `src/minimizer.rs`, no callers.** Canonical m-mer, 64-bit mixer,
eligibility rule, deque sliding-window minimum, `signature_of`, `bin_of`. Pure
functions. New tests only:
- `sig(x) == sig(rc(x))` over pseudo-random k-mers — the §3.3 invariant.
- `bin` is total: every k-mer maps into `0..NUM_BINS`.
- An all-ineligible window falls back to bin 0.
- Realised density on synthetic sequence, reported and asserted within a band
  around `2/(w+1)` — this is where Roberts' "a few percent above" caveat and
  the canonical-m-mer perturbation get measured rather than assumed.

**Step 2 — `src/superkmer.rs`, no callers.** Encode/decode of the §3.4 record
layout; `split_into_superkmers(seq, k, m)`. The single most important test in
the project lives here:

> Expanding every super-k-mer of a read, canonicalising each k-mer, and
> collecting yields **exactly the same multiset** as
> `kmer::extract_canonical_kmers(seq, k)` — including for reads containing `N`,
> reads shorter than `k`, all-`N` reads, pure-homopolymer reads, and reads at
> the 255-base super-k-mer cap.

Plus: every k-mer expanded from one super-k-mer has the same `bin_of`. If those
two properties hold, the design cannot produce a wrong count.

**Step 3 — `src/binned.rs`, standalone.** `BinStore`, `BinWriter`, phase 2, the
cross-bin merge; entry point `count_records(records, k) -> Vec<(u64, u32)>`.
Tested *against `KmerCounter`* using the deterministic synthetic FASTQ
generator `tests/dual_strategy.rs:46` already provides — a differential test,
not a golden file.

**Step 4 — wire it as a third strategy, opt-in.** `CountStrategy::Binned`
alongside `InMemory` and `Disk` (`src/pipeline.rs:454`) **[code]**, reachable
only via explicit `--strategy binned` / `FASTDNA_STRATEGY=binned`;
`resolve_strategy` still never chooses it. Extend `tests/dual_strategy.rs` —
whose stated acceptance criterion is already "a caller must never be able to
tell, from the counts alone, which strategy actually ran" — from two strategies
to three. **The memory and disk paths are not modified at all, so all 252 tests
pass by construction.** This is where §4's benchmark gets run for real.

**Step 5 — new memory model.** Add `estimate_peak_bytes_binned` to
`mem_estimate.rs`, calibrated against step 4's measurements. **Add, do not
change**: `estimate_matches_calibration_runs_within_reported_error`
(`src/mem_estimate.rs:286`) pins the *old* model's arithmetic against real runs
and must stay green.

**Step 6 — promote to default in `auto`,** above an input-size threshold, once
§4's predictions are confirmed or refuted. Only now does user-visible behaviour
change.

**Step 7 (optional, later) — the two known further reductions.** Both are
independently testable against step 3's output, and both are contingent on
data shape:
- **FastK's weighted super-mer sort** (§2.3): sort and deduplicate super-k-mers
  within a bin, then expand each *distinct* super-k-mer once with a
  multiplicity weight. Gated on low error rate; FastK scopes it to ≤1%.
- **KMC's (k,x)-mers** (§2.2): "with setting x = 3 the number of (k,x)-mers
  becomes about twice smaller than the number of k-mers", reported to save
  >20% total time.

**Step 8 (optional, later) — unify `disk_spill.rs`** on the same bin function.
Its module header already anticipates this: the spill/merge machinery "does not
care how `bucket_of` computes its answer" (`src/disk_spill.rs:44`) **[code]**.

### 5.1 Behaviour that must not change

| Invariant | Pinned by |
|---|---|
| Canonical k-mer = `min(kmer, rc(kmer))`, 2-bit packed, `k ∈ 1..=32` | `src/kmer.rs::test_base_encoding_and_decoding`, `::test_reverse_complement_symmetry` |
| Ambiguous bases reset the window; no corrupt k-mers | `src/kmer.rs::test_ambiguous_base_reset` — **the binned path must cut super-k-mers at every `N`**, a new place this invariant can break |
| Counts saturate at `u32::MAX`, never wrap or panic | `src/counter.rs:176-179`, `:267`, `:300` — every compaction and merge in the new path must use `saturating_add` too |
| QC merge associative; percentages only in `finalize()` | `src/qc.rs` — **untouched**; workers keep their private `QcSummary` exactly as now |
| Output table ascending by `kmer_u64` | `export.rs`, `tests/histogram_format.rs` — preserved by §3.5 option A |
| Output schema `kmer_u64 / kmer_sequence / frequency` | `src/export.rs::counts_schema` — the contract with `fastdna-ml` |
| Strategies indistinguishable from their counts alone | `tests/dual_strategy.rs` |
| Worker panics become `FastDnaError::Internal`, never cross FFI | `src/pipeline.rs:320`, `:379-385` |

### 5.2 Tests that legitimately must change

- `src/disk_spill.rs::bucket_of_is_monotonic_in_kmer_value_for_a_fixed_k`
  (line 494) — **only if step 8 is taken.** It pins a property the minimizer
  partition deliberately gives up (§3.5). It must be *replaced* by tests of the
  new invariant, not deleted.
- The `disk_spill.rs` module header's "no additional sort needed" paragraph
  (lines 29-45) becomes false at step 8 and must be rewritten in the same
  commit.
- `mem_estimate.rs`'s model is **extended**, not modified (step 5).

---

## 6. Risks, and what would make this not worth doing

**R1 — the diagnosis could be wrong. (Highest impact.)** Everything here
assumes the 110.1 s counting phase is dominated by memory traffic from
materialising occurrences. §4.2 admits I cannot account for 110 s from first
principles. If the real cost is the single-threaded producer parsing 2.14 GB
and allocating three `Vec<u8>` per record × 7M records, this design changes the
wrong thing. **Mitigated entirely by step 0**, which is why step 0 is a gate.

**R2 — silent split counts. This is the single biggest risk in the design.**
The whole architecture rests on one new global invariant: *every occurrence of
a canonical k-mer reaches exactly one bin*. Break it — a non-strand-invariant
signature, a mishandled ineligibility fallback, an off-by-one at a super-k-mer
boundary, an `N` cut that drops or duplicates a window — and the output is a
well-formed sorted Parquet file with the correct schema, a plausible k-mer
count, and **wrong numbers**. Nothing crashes; `train_classifier.py` downstream
trains happily on it. This is categorically worse than any failure the current
code can produce, and it is why steps 1–3 front-load property tests before a
differential test before a benchmark. It is also why §3.3's strand-invariance
argument is written out as a proof rather than asserted.

**R3 — bin skew on real (non-synthetic) data.** The benchmark input is
synthetic (`scripts/bench/generate_reads_large.py`, seed 9001) and therefore
compositionally well-behaved. Real amplicon panels, poly-A-rich RNA-seq and
low-complexity metagenomes are exactly what KMC's exclusion rules and FastK's
learned trie exist to survive, and Discount (§2.4) names the problem outright:
"minimizers are known to generate bins of very different sizes". Our narrower
exclusion rule (§3.3) and static hash-to-bin map are the least defended point
of the design. Mitigation: report per-bin occupancy behind a debug flag from
step 4, and benchmark on at least one real SRA sample before step 6.

**R4 — we will probably still be behind FastK. Say it plainly.** Comparing
across operating systems is not sound, but at face value: FastK 38.3 s (WSL)
against my predicted 55–75 s (Windows) / 70–110 s (WSL). **We would very likely
remain roughly 2× behind FastK**, because we would have adopted KMC's
architecture without FastK's weighted super-mer sort (§2.3, step 7) or its
data-adaptive partition trie. Approve this as "reach and beat KMC3, cut memory
by 3×", not as "beat FastK". For calibration, ntStat 2026 (§2.4) finds KMC3 is
*still* the tool to beat on speed nine years on — matching it is not a modest
target.

**R5 — most of the memory win, and some of the time win, is available for far
less work. This must be on the record.**

| Smaller change | Effort | Predicted effect |
|---|---|---|
| SoA count tables instead of `Vec<(u64,u32)>` (§3.7) | hours | −205 MiB peak, −25% merge traffic. Independent of everything. |
| Single k-way reduce over all N worker tables instead of rayon's pairwise `reduce` (`src/pipeline.rs:422`) — `k_way_merge_sorted_counts` already exists | ~1 day | **[judgment]** several seconds; removes the worst-case sequential merge chain |
| Make `RAW_FINALIZE_THRESHOLD` adaptive to observed diversity | ~2 days | `docs/BENCHMARKS.md:261-279` measures the *unbounded* buffer at 87.6–109.9 s against the bounded 125.3–173.1 s on this very file — **a 20–30% time win is sitting in one constant** |
| Just use the existing disk strategy | zero | already 1.10 GB peak; 2.1× slower, not 3× |

**[judgment] If the goal is only "stop using 8 GB", the disk strategy already
solves it and §3.7 sharpens it. This design is justified only if the goal is
low memory *and* competitive speed at once — which is precisely the trade the
current two strategies cannot make, one being fast and memory-hungry and the
other frugal and slow. That, and only that, is the case for doing it.**

**R6 — hot-path risk.** This touches the most-tested, most load-bearing code in
the crate, under a clippy configuration that denies `unwrap`/`expect` and a
`panic = "unwind"` FFI contract. The step-4 opt-in strategy structure exists
specifically so a half-finished version cannot regress anyone.

---

## 7. Recommendation

1. **Do step 0 now.** One day. It gates everything else and resolves R1, the
   risk that would waste the most effort.
2. **Do §3.7 (SoA tables) regardless.** Free and independent.
3. **If step 0 confirms the worker loop dominates, do steps 1–4.** Steps 1–3
   are purely additive modules with no risk to existing behaviour; step 4 is
   opt-in. The decision point with real data is after step 4, against the
   predictions in §4 — recorded here in advance precisely so that decision can
   be made on evidence rather than on sunk cost.
4. **Do not start step 6 until a real SRA sample has been counted correctly and
   a per-bin occupancy report has been examined** (R3).

---

## 8. Sources

**Papers**

- Roberts M, Hayes W, Hunt BR, Mount SM, Yorke JA. "Reducing storage
  requirements for biological sequence comparison." *Bioinformatics*
  20(18):3363–3369, 2004. <https://doi.org/10.1093/bioinformatics/bth408>
- Deorowicz S, Kokot M, Grabowski Sz, Debudaj-Grabysz A. "KMC 2: Fast and
  resource-frugal k-mer counting." *Bioinformatics* 31(10):1569–1576, 2015.
  <https://doi.org/10.1093/bioinformatics/btv022> · preprint
  <https://arxiv.org/abs/1407.1507>
- Kokot M, Długosz M, Deorowicz S. "KMC 3: counting and manipulating k-mer
  statistics." *Bioinformatics* 33(17):2759–2761, 2017.
  <https://doi.org/10.1093/bioinformatics/btx304>
- Marçais G, Pellow D, Bork D, Orenstein Y, Shamir R, Kingsford C. "Improving
  the performance of minimizers and winnowing schemes." *Bioinformatics*
  33(14):i110–i117, 2017. <https://doi.org/10.1093/bioinformatics/btx235>
- Marçais G, DeBlasio D, Kingsford C. "Asymptotically optimal minimizers
  schemes." *Bioinformatics* 34(13):i13–i22, 2018.
  <https://doi.org/10.1093/bioinformatics/bty258>
- Zheng H, Kingsford C, Marçais G. "Improved design and analysis of practical
  minimizers." *Bioinformatics* 36(Suppl 1):i119–i127, 2020.
  <https://doi.org/10.1093/bioinformatics/btaa472>
- Orenstein Y, Pellow D, Marçais G, Shamir R, Kingsford C. "Designing small
  universal k-mer hitting sets for improved analysis of high-throughput
  sequencing." *PLoS Comput Biol* 13(10):e1005777, 2017.
  <https://doi.org/10.1371/journal.pcbi.1005777>
- Marçais G et al. "k-nonical space: sketching with reverse complements."
  *Bioinformatics*, 2024. <https://doi.org/10.1093/bioinformatics/btae629>
- Edgar R. "Syncmers are more sensitive than minimizers for selecting conserved
  k-mers in biological sequences." *PeerJ* 9:e10805, 2021.
  <https://doi.org/10.7717/peerj.10805>
- Pibiri GE. "Sparse and skew hashing of k-mers." *Bioinformatics*
  38(Suppl 1):i185–i194, 2022.
  <https://doi.org/10.1093/bioinformatics/btac245>
- Wood DE, Lu J, Langmead B. "Improved metagenomic analysis with Kraken 2."
  *Genome Biology* 20:257, 2019. <https://doi.org/10.1186/s13059-019-1891-0>
- Ndiaye M et al. "When less is more: sketching with minimizers in genomics."
  *Genome Biology*, 2024. <https://doi.org/10.1186/s13059-024-03414-4>
- Pasquale G et al. "The open-closed mod-minimizer algorithm." *Algorithms Mol
  Biol*, 2025. <https://doi.org/10.1186/s13015-025-00270-0>
- Groot Koerkamp R, Pibiri GE. "SimdMinimizers: computing random minimizers,
  fast." Preprint, 2025. <https://doi.org/10.1101/2025.01.27.634998>
- Nikolić V et al. "Discount: distributed k-mer counting with universal
  frequency ordering." *Bioinformatics*, 2021.
  <https://doi.org/10.1093/bioinformatics/btab156>
- ntStat. *PLoS Comput Biol*, 2026.
  <https://doi.org/10.1371/journal.pcbi.1014158>

**Repositories and licences (verified 2026-08-25)**

- KMC — <https://github.com/refresh-bio/KMC>. Licence fetched from
  <https://raw.githubusercontent.com/refresh-bio/KMC/master/LICENSE>: **GNU
  General Public License, Version 3, 29 June 2007**. Read-the-paper-only for
  this project.
- FastK — <https://github.com/thegenemyers/FASTK>. Licence read from
  `~/fastdna-bench/tools/FASTK/LICENSE` in WSL: permissive,
  BSD-3-Clause-*style*, © 2020 Dr. Eugene W. Myers, with a modified clause 1
  ("or derivative codes"); GitHub's detector classifies it `NOASSERTION`.
  Unpublished — no paper exists.

**FastDNA files read for this document (all 2026-08-25)**

`CLAUDE.md`, `src/kmer.rs`, `src/counter.rs`, `src/pipeline.rs`,
`src/disk_spill.rs`, `src/mem_estimate.rs`, `src/export.rs`,
`tests/dual_strategy.rs`, `docs/BENCHMARKS.md`,
`docs/feature-gap-analysis.md`.
