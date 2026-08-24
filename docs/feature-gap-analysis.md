# Feature-gap analysis: FastDNA vs. KMC3, FastK, Jellyfish, Mash/sourmash, ntCard

Researched 2026-08-24 against the tools' current repos and docs. Every "FastDNA
lacks X" claim below was verified in this repo's code on that date (files listed
at the end). Companion documents: `ml-genomics-roadmap.md` (ML side),
`BENCHMARKS.md` (measured performance).

## Quick wins (small effort, high value) — ranked

| # | Feature | Who has it | Why it matters | Sketch | Effort |
|---|---------|-----------|----------------|--------|--------|
| Q1 | **FASTA input** | KMC3, FastK, Jellyfish | Cannot count a genome/assembly today; `assembly_qc.py` fakes FASTQ to work around it | Format sniffer (`>` vs `@`), `FastaReader` producing `FastqRecord` with synthetic qual; multi-line FASTA handled | S |
| Q2 | **Multi-file + stdin input** | KMC3 (`@list`), FastK, Jellyfish (pipes) | Real samples are R1/R2 (+lanes); `fasterq-dump \| fastdna -i -` is standard HPC practice | `--input` becomes `Vec<PathBuf>`; producer iterates files into the same channel; `-` = stdin with gzip magic-byte sniff; R1/R2 pairing already exists in `cohort/discovery.rs` | S |
| Q3 | **GenomeScope-compatible histogram** | Jellyfish `histo`, `kmc_tools transform histogram`, ntCard, meryl | GenomeScope2 (genome size/heterozygosity/ploidy) expects headerless space-separated `coverage count`; our CSV header breaks it | `--histogram-format genomescope\|csv`, plus a `-cx`-style tail cap | S |
| Q4 | **CLI subcommands for existing library features** | n/a | sketch/dist/cardinality/peek exist in Rust but are Python-only; pipeline users live on the CLI | clap subcommands: `count` (default), `sketch`, `dist`, `card`, `peek` | S–M |
| Q5 | **Submit Bioconda recipe** | all competitors are on Bioconda | It is how pipelines install tools; nf-core effectively requires it | `recipe/meta.yaml` exists; needs real URL/sha256 + submission | S |

## Strategic items — ranked

- **S1. Binary k-mer database + random-access query API** (KMC `.kmc_pre/.kmc_suf` + kmc_api; FastK `.ktab` + libfastk; Jellyfish `.jf`). Foundation for everything below. FastDNA's merge already yields a globally sorted `(u64,u32)` table, so a sorted fixed-record file + sampled index gives O(log n) queries without hashing — or declare sorted Parquet + row-group stats the format and ship a query layer (DuckDB-friendly). New `src/ktab.rs`, `fastdna query`, Python `KmerTable.open(...)`. Effort M; unlocks S2–S4.
- **S2. Set operations between tables** (kmc_tools simple; FastK Logex). Case-vs-control k-mers, reference/vector subtraction, trio binning. With S1, every op is a linear merge-join — machinery `disk_spill.rs` already has. Best value/effort of the strategic set. Effort M.
- **S3. Per-read k-mer profiles** (FastK's signature feature; no KMC equivalent). Error detection, QV estimation, exact Merqury-style QC (today `assembly_qc.py` approximates with set membership). Two-pass: count, then stream reads with S1 lookups, RLE-compressed count vectors. Effort L.
- **S4. Read filtering by k-mer content** (kmc_tools filter, BBDuk). Host/contamination removal in metagenomics. Streaming pass + S1 lookups + FASTQ writer. Effort M.
- **S5. Minimizer/super-k-mer partitioning** (KMC3 bins, FastK supermers). Pure performance at 100GB+ scale; `disk_spill.rs`'s module boundary already anticipates swapping `bucket_of`. Do after completeness items. Effort L.
- **S6. Multi-sample cohort matrix as a first-class Rust artifact** (`fastdna matrix samples/ -o cohort.parquet`). Nobody in the C/C++ field does this natively; feeds FastDNA's ML story — an area where FastDNA can be first. Effort M.
- **S7. ntCard-style streaming spectrum estimate; k up to 64 via u128.** Lower priority. Wire or delete `cms.rs` (dead code is a liability).

## Deliberately not copying

- Jellyfish's lock-free hash table (our sort-and-compact beat a hash design 9x in our own benchmarks).
- BAM/CRAM input (heavy htslib/noodles dependency; `samtools fastq | fastdna -i -` covers it once Q2 lands).
- Full sourmash ecosystem (SBT/LCA databases); consider exporting sketches in sourmash signature JSON instead.
- Histex/Vennex-style micro-tool zoo (subsumed by subcommands + Python).
- FastK homopolymer compression (long-read mode is a real project — quality trimming itself is wrong for ONT — not a flag).
- KMC-style memory knob forest (auto strategy + `--max-ram` is a better UX; keep it).

## Suggested execution order

1. Q1+Q2+Q3 in one release (removes the "can't even" objections).
2. Q4+Q5 (surface + distribution).
3. S1 → S2 → S4 (database → set ops → filtering backbone).
4. S6 (the differentiator play, with the ML layer).
5. S3 and S5 (FastK parity and speed-at-scale endgame).

Sources: KMC3 repo/paper, kmc_tools usage docs, FastK repo, Jellyfish repo,
GenomeScope 2.0 repo, sourmash docs. Repo files verified: `src/cli.rs`,
`src/lib.rs`, `src/fastq.rs`, `src/export.rs`, `src/main.rs`,
`src/disk_spill.rs`, `src/sketch.rs`, `src/hll.rs`, `src/cms.rs`,
`src/preview.rs`, `src/cohort/`, `python/fastdna/*.py`, `workflow_templates/`,
`README.md`.
