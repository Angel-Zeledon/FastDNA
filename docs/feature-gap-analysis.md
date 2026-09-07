# Feature-gap analysis: FastDNA vs. KMC3, FastK, Jellyfish, Mash/sourmash, ntCard

> **Status, 2026-09-05**: this is a researched *menu*, not a set of
> commitments -- see `docs/goal-fast-kmer-counter.md`, which supersedes the
> goal doc cited in the audit note below. It is the one gap document that
> survived the removal of the ML layer, because its subject is the counting
> engine and its neighbours rather than anything that was removed. An item
> earns its place by being counting, and by being checkable against one of
> the tools in the title.
>
> Cross-references below to `ml-genomics-roadmap.md`,
> `ml-differentiation-roadmap.md` and
> `goal-most-complete-genomics-ml-library.md` point at documents deleted on
> that date; they are left in place rather than rewritten, because editing
> the body would misrepresent when the analysis was done. Read them as
> "there used to be a companion document here".

> **Audit pass, 2026-08-27**: every status marker below was re-checked against
> the actual code on this date (not just against other planning docs), per
> `docs/goal-most-complete-genomics-ml-library.md`'s instruction to verify
> before executing anything on this list. Several items this doc listed as
> missing on 2026-08-24 had already shipped by 2026-08-27 (confirmed exactly
> the pattern `docs/CHECKPOINT-2026-08-26.md` warned about); Q4 was
> genuinely missing and was implemented in this pass. The analysis and
> reasoning below are otherwise unchanged from the original 2026-08-24
> research -- this is a status correction, not a rewrite.

Researched 2026-08-24 against the tools' current repos and docs. Every "FastDNA
lacks X" claim below was verified in this repo's code on that date (files listed
at the end). Companion documents: `ml-genomics-roadmap.md` (ML side),
`BENCHMARKS.md` (measured performance).

## Quick wins (small effort, high value) — ranked

| # | Feature | Status | Evidence |
|---|---------|--------|----------|
| Q1 | **FASTA input** | **Shipped.** | `src/fastq.rs`'s `Format::Fasta` + content-sniffing reader (not extension-based); `tests/fasta_input.rs` and the `fastq::tests::fasta_*` unit tests (multi-line FASTA, CRLF, synthetic Q40 quality, gzip). |
| Q2 | **Multi-file + stdin input** | **Shipped.** | `src/cli.rs`'s `--input` is `Vec<PathBuf>` (`num_args = 1..`); `src/fastq.rs::MultiSourceReader` iterates files into one channel; `-` is stdin with gzip magic-byte sniffing (`fastq::STDIN_ARG`); R1/R2 pairing lives in `src/cohort/discovery.rs`. Confirmed by `tests/multi_input.rs`, `tests/stdin_input.rs`. |
| Q3 | **GenomeScope-compatible histogram** | **Shipped.** | `src/cli.rs::CliHistogramFormat::GenomeScope` (clap value `genomescope`), `src/export.rs::HistogramFormat`, `--histogram-max` for the `-cx`-style depth cap. Confirmed by `tests/histogram_format.rs`. |
| Q4 | **CLI subcommands for existing library features** | **Shipped (this pass, 2026-08-27).** | Was genuinely missing as of the audit: `src/cli.rs` had no `clap::Subcommand`, only the flat counting flags. Added `fastdna sketch\|dist\|card\|peek` plus an explicit `fastdna count` (identical to giving no subcommand -- backward compatible, per `CHANGELOG.md`'s CLI compatibility contract) in `src/cli.rs` (`Command`, `SketchArgs`, `DistArgs`, `CardArgs`, `PeekArgs`) and `src/main.rs` (`run_sketch`/`run_dist`/`run_card`/`run_peek`), wrapping `sketch.rs`/`hll.rs`/`preview.rs`. Tests: `tests/cli_subcommands.rs` (18 tests, parsing + real-binary end-to-end). |
| Q5 | **Submit Bioconda recipe** | **Partially shipped.** | `recipe/meta.yaml` exists and is well-formed, but `source.url`/`source.sha256` are still literal placeholders (`recipe/meta.yaml` lines 36-46: a 64-zero sha256 with a comment saying so), and it has never been run through `conda build` or submitted to bioconda-recipes. The recipe's own header comments document the remaining steps. |

## Strategic items — ranked

- **S1. Binary k-mer database + random-access query API.** **Shipped
  (this pass, 2026-08-27).** `src/ktab.rs`'s `KmerTable` implements exactly
  the design `docs/superpowers/plans/2026-08-24-completeness-phase.md`'s
  Task 1 reasoned through: no new file format -- the sorted Parquet
  `export.rs` already writes is declared the k-mer table format, and
  `export_counts_parquet` now always attaches `fastdna.sorted_by=
  kmer_u64`/`fastdna.k=<k>` footer metadata so every counting run's default
  output is directly queryable, no conversion step. `KmerTable::open`
  validates that metadata plus per-row-group `kmer_u64` statistics (cheap,
  footer-only); `get`/`range`/`iter` use those statistics to prune which
  row groups a query decodes, confirmed against the real `parquet` 53.4
  crate (`ParquetRecordBatchReaderBuilder::with_row_groups`) rather than
  assumed. CLI: `fastdna query --table FILE --kmer <sequence-or-u64>`
  (`cli::QueryArgs`). Python: `fastdna.KmerTable` (`.open`, `.get`,
  `__getitem__`, `__contains__`, `__len__`), wrapping `fastdna._core.
  KmerTable` (`src/ffi.rs`). Tests: 14 inline unit tests in `src/ktab.rs`
  (point lookup hit/miss, multi-row-group sort/completeness, a
  cross-tool-written table with no fastdna-specific writer, a real
  zero-row row group), `tests/ktab_cli.rs` (5 tests, real-binary
  count -> query end to end), `python/tests/test_ktab.py` (11 tests).
  Scope: covers both counting strategies' output (in-memory and disk both
  converge on the same sorted Parquet via `export.rs`); `binned.rs`'s S5
  strategy is not wired into that export path either, so it is unaffected
  either way. Unlocks S2-S4, all still open.
- **S2. Set operations between tables.** **Shipped (this pass,
  2026-08-27).** `src/setops.rs` (new `pub mod setops`): `union`,
  `intersect`, `diff`, each a single linear merge-join over `KmerTable::
  iter`'s already-sorted streams (`MultiTableMerge`, a binary-heap k-way
  merge -- the same algorithm `disk_spill.rs::merge_sources_into` already
  uses for its own per-bucket run files, adapted to pull from Parquet-backed
  `RangeIter`s instead of flat run files), exactly the "every op is a linear
  merge-join" design this entry originally called for. No table is ever
  loaded into a hash set: each call holds at most one Arrow batch per input
  table in memory, regardless of table size. Output is written by
  `export::export_pairs_parquet` (new, in `export.rs`, sharing
  `export_counts_parquet`'s chunking/writer so the two cannot drift) in the
  same `(kmer_u64, frequency)` Parquet shape S1 already establishes, so a
  set operation's result is immediately a valid `KmerTable` -- confirmed by
  `setops.rs`'s own `union_output_written_and_reopened_composes_with_diff`
  test and `tests/setops_cli.rs`'s
  `setops_output_composes_as_a_valid_input_to_a_further_setop`.
  - Union: every k-mer present in any input table; `CombineOp::Sum`
    (default) sums counts across tables that have it -- the natural
    "combine these samples" reading; `Min`/`Max` are also available.
  - Intersect: only k-mers present in *every* input table; the merge always
    computes each table's own count for a shared k-mer (not just presence),
    folded into the output's single `frequency` column via `CombineOp`
    (`Min` is the default, matching `kmc_tools simple`'s own default
    reducer for `intersect`).
  - Diff: asymmetric A-minus-(one or more B tables) -- the reference-
    subtraction/host-removal case. A `max_subtract_count` threshold (0 by
    default) tolerates a bounded amount of reference noise before a k-mer is
    treated as contamination and dropped; kept rows keep A's own original
    count. Accepts several subtract tables at once (a host genome, an
    adapter set, a trio's other members, ...), dropping a k-mer flagged by
    *any* of them.
  - CLI: `fastdna union --input FILE... -o OUT.parquet [--combine
    sum|min|max]`, `fastdna intersect --input FILE... -o OUT.parquet
    [--combine min|sum|max]`, `fastdna diff --input FILE --subtract
    FILE... -o OUT.parquet [--max-subtract-count N]` (`cli::{UnionArgs,
    IntersectArgs, DiffArgs}`, `main.rs::{run_union, run_intersect,
    run_diff}`).
  - Python: `KmerTable.union/.intersect/.difference` (`src/ffi.rs`'s
    `ktab_union`/`ktab_intersect`/`ktab_diff`, wrapped in
    `python/fastdna/__init__.py`), each returning a freshly opened
    `KmerTable`.
  - Tests: `src/setops.rs`'s inline unit tests (union/intersect/diff
    correctness, empty tables, full/no overlap, 3+ tables, mismatched-`k`
    rejection, composability), `tests/setops_cli.rs` (real-binary
    count -> setops -> query end to end), `python/tests/test_setops.py`.
- **S3. Per-read k-mer profiles.** **Shipped (this pass, 2026-08-27),
  by this item's original design.** `src/read_profile.rs` (new `pub mod
  read_profile`) implements exactly the "two-pass: count, then stream reads
  with S1 lookups, RLE-compressed count vectors" design this entry named,
  now unblocked by S1: the reference table is built first (`count` ->
  `export.rs`), then opened as a `ktab::KmerTable` and streamed against
  every read, position `i` of a read mapping to the reference count of the
  k-mer starting there. This is distinct from, and does not replace, either
  of the two things already covered here before this pass:
  `metagenomics.rs`'s `ReadClassification` (Kraken-style taxonomic
  assignment against a tagged reference, still the right tool for species
  classification) and `assembly_qc.py::evaluate_assembly`'s set-membership
  QV approximation (still its own module, unchanged by this pass -- wiring
  it to consume real per-read profiles instead of set membership is a
  natural follow-up, not done here).
  - **Output format: RLE**, one row per `(read_id, start, run_length,
    count)`, chosen over a one-row-per-base "tidy" table specifically
    because the latter is enormous at real scale (illustratively, on the
    order of 90 billion rows for a 30x human whole-genome run at 150bp
    reads, before Parquet's own encoding, against tens of millions of rows
    for `export_counts_parquet`'s one-row-per-distinct-k-mer shape on the
    same input) while RLE exploits the same empirical regularity FastK's
    own `.prof` format is built around: consecutive overlapping k-mers in a
    non-repetitive, error-free stretch of a read are frequently supported
    by the same reads and so carry identical reference counts, making a run
    boundary itself the informative signal (a real coverage transition or a
    sequencing error). `read_profile::expand_rle` reconstructs the exact
    full per-position sequence losslessly. See `read_profile.rs`'s own
    module doc comment for the full reasoning, including what was and was
    not independently measured in this session.
  - A small, always-emitted per-read summary table (`read_id, n_kmers,
    n_present_kmers, min_count, median_count, max_count`, the distribution
    computed over the full per-position sequence including absent/count-0
    positions) ships alongside the RLE profile, so the common "which reads
    look erroneous" question never requires decompressing a single run.
    Column naming (`n_kmers`/`n_present_kmers`) deliberately matches
    `metagenomics.rs::ReadClassification`'s own `n_kmers`/
    `n_classified_kmers` convention rather than diverging from it.
  - `ProfileIndex` (new, `read_profile.rs`): a resident, sorted
    `(k-mer, count)` index built once per run via `KmerTable::iter` -- the
    same trick `read_filter::ReferenceIndex` already uses to avoid
    `KmerTable::get` in a per-k-mer tight loop, kept as a small parallel
    type (not a generalized `ReferenceIndex`) so as not to disturb that
    type's own tested, shipped behavior for `fastdna filter`.
  - `kmer.rs` gains `extract_canonical_kmers_with_positions_into`, the
    positional counterpart of `extract_canonical_kmers_into` (same rolling
    window, same ambiguous-base reset semantics, plus each k-mer's read
    position) that a profile needs and the plain extractor's caller-order
    output does not provide.
  - Streaming, memory bounded by the resident reference index plus one read
    -- the same tradeoff S4 (`read_filter.rs`) makes and documents.
  - CLI: `fastdna profile --input FILE... --table REFERENCE.parquet -o
    PROFILE.parquet [--summary SUMMARY.parquet]` (`cli::ProfileArgs`,
    `main.rs::run_profile`).
  - Python: `KmerTable.profile_reads(inputs, *, output, summary=
    "read_profile_summary.parquet")`, wrapping `fastdna._core.
    profile_reads` (`src/ffi.rs`), returning a `ProfileStats` (`reads_total`,
    `reads_profiled`).
  - **Scope: single-end only**, the same reason `read_filter.rs` documents
    for its own lack of paired-end (R1/R2) synchronization. Wiring
    `assembly_qc.py` to consume real profiles instead of its current
    set-membership approximation is explicitly deferred, not attempted
    here -- that module's own docstring already documents the
    approximation it makes and why, and changing it is separate, additional
    scope.
  - Tests: inline unit tests in `src/read_profile.rs` (a fully-covered read
    at a uniform count collapses to one run; a read entirely absent from
    the reference; a read shorter than `k`; a count change starts a new run
    at the exact position; a run never bridges an ambiguous-base gap even
    when the count matches on both sides; `expand_rle` round-trips losslessly,
    including against `build_read_profile`'s own real output; ambiguous-base
    handling matches the shared extractor) and in `src/kmer.rs` (the new
    positional extractor pinned against the plain one at every `k`, the gap
    an ambiguous base leaves, a too-short read), plus `tests/
    read_profile_cli.rs` (real-binary `count -> profile` end to end) and
    `python/tests/test_read_profile.py`.
## B4 (`ml-differentiation-roadmap.md`'s finishing touches list)

- **B4. Wire `assembly_qc.py`'s FASTA path through `fastdna.count()`.**
  **Shipped (2026-08-27).** `python/fastdna/assembly_qc.py::
  _assembly_kmer_counts` now always builds the assembly's k-mer multiset
  via `fastdna.count(assembly_path, k=k, min_count=1, min_quality=0.0,
  with_sequence=True)` -- the Rust pipeline (Q1's content-sniffed FASTA
  support) -- instead of dispatching on the file's extension between that
  and a pure-Python FASTA extractor. No dispatch is needed any more: the
  Rust core decides FASTA vs. FASTQ from the stream's own first non-blank
  byte (`src/fastq.rs::sniff_format`), so `.fasta`/`.fa`/`.fna`/`.fastq`/
  `.fq` (`.gz` or not) all take the same path. The old pure-Python
  helpers (`_count_fasta_kmers`, `_canonical_kmers`, `_iter_fasta_sequences`)
  are kept, unused by `evaluate_assembly()` itself, as a manual building
  block for `evaluate_kmers()`'s lower-level entry point. Verified
  equivalent: `tests/fasta_input.rs`'s
  `a_fasta_file_counts_identically_to_the_equivalent_fastq` (Rust,
  pipeline-level) and `python/tests/test_assembly_qc.py::
  TestFastPathMatchesPurePythonFallback` (Python, compares the new
  extractor against the old one on the same fixture, including an
  ambiguous-base/wrapped-line case). **Measured performance caveat**: on
  single-contig fixtures from 20 kb to 2 Mb, the new path was consistently
  *slower* in wall time than the pure-Python extractor it replaces, not
  faster -- the counting itself is faster in Rust, but decoding every
  distinct k-mer back to a Python string (`with_sequence=True`, required
  so this function can hand back a `{kmer: count}` mapping) and building
  the resulting Python dict is real per-element FFI/object-creation work
  that does not disappear just because counting moved to Rust. This
  change is a correctness/maintenance win (one native-FASTA-capable
  counting core instead of two independently-maintained k-mer extractors)
  confirmed by the tests above; it is not a proven wall-clock win at every
  input size -- see `assembly_qc.py`'s module docstring for the full
  explanation.
- **S4. Read filtering by k-mer content.** **Shipped (this pass,
  2026-08-27).** `src/read_filter.rs` (new `pub mod read_filter`): streams a
  FASTQ/FASTA input against a reference `ktab::KmerTable` and keeps or
  discards each *read* based on the fraction of its own canonical k-mers
  found in that reference -- `kmc_tools filter`/BBDuk's `ref=`/`k=`
  filtering, exactly the gap this entry named. Built directly on S1
  (`ktab::KmerTable`); `setops::diff`'s own output composes with this
  unchanged (a `diff` result is a valid `KmerTable`, so it can be used as
  `--table` here for a two-step "subtract, then filter" pipeline).
  - **Threshold semantics**: a read "matches" the reference when the
    fraction of its own canonical k-mers found in the table is `>=
    --min-fraction` (inclusive; default `0.1`). A read that yields *no*
    canonical k-mers at all (shorter than the table's `k`, or entirely
    ambiguous bases) never matches, regardless of `--min-fraction`
    (including `0.0`) -- distinct from a read that genuinely computes a
    `0.0` fraction against a real, non-empty read, which *does* clear an
    inclusive `>= 0.0` threshold. See `read_filter.rs`'s module doc comment
    for the full reasoning.
  - **Modes**: `--mode keep` writes only matching reads (targeted
    enrichment); `--mode discard` writes only non-matching reads
    (host/contaminant removal) -- the two directions BBDuk's own `ref=`
    filtering supports.
  - **Performance design**: `ktab.rs`'s own `KmerTable::get` reopens the
    Parquet file and re-decodes a row group *per call*, which its own doc
    comment already warns against in a tight loop -- and a filtering run is
    exactly that tight loop, at the scale of every k-mer of every read.
    `read_filter::ReferenceIndex` instead loads the reference table once
    per run (via `KmerTable::iter`, already streaming/tested) into a
    resident sorted `Vec<u64>` (no frequency column, no hash overhead --
    only presence is ever asked), queried with `binary_search`. Memory
    scales with the *reference* table's size, not the input read stream's
    (which stays fully streamed, one record at a time); an unbounded
    resident reference is the one tradeoff this design makes, deliberately,
    the same way `setops.rs` explicitly refuses to load *either* side of a
    set operation into memory for the opposite reason (either side there
    could be the large one).
  - **Output**: always FASTQ (`@id`/seq/`+`/quality), even for FASTA input
    -- reusing `fastq.rs`'s existing `SYNTHETIC_FASTA_QUALITY` (Q40)
    convention rather than inventing a FASTA writer. Gzipped iff the output
    path's own extension says so, written atomically (`atomic::AtomicFile`).
  - **Scope: single-end only.** Each `--input` file is filtered
    independently, read by read, and several files are filtered as one
    concatenated stream into one output -- the same convention `count`'s
    own multi-file `--input` already uses. **Paired-end (R1/R2)
    synchronized filtering -- keeping/discarding a pair as a unit if either
    mate matches, the standard real-pipeline convention -- is not
    implemented.** Passing both mates via `--input` filters each
    independently and can desynchronize them; wiring `cohort::discovery`'s
    pairing logic through a genuinely paired-aware two-stream writer is
    real, additional scope, deliberately left as an explicit follow-up
    (documented in `read_filter.rs`'s module doc comment and
    `cli::FilterArgs`'s) rather than shipped half-correct.
  - CLI: `fastdna filter --input FILE... --table REFERENCE.parquet --mode
    keep|discard [--min-fraction 0.1] -o OUT.fastq[.gz]` (`cli::FilterArgs`,
    `main.rs::run_filter`).
  - Python: `KmerTable.filter_reads(inputs, mode=, output=,
    min_fraction=0.1)`, wrapping `fastdna._core.filter_reads` (`src/
    ffi.rs`), returning a `FilterStats` (`reads_total`, `reads_written`).
  - Tests: 15 inline unit tests in `src/read_filter.rs` (`ReferenceIndex`
    membership, `matching_fraction`/`read_is_match` threshold semantics
    including the zero-k-mers-vs-zero-fraction distinction, `filter_records`
    under both modes, FASTA-to-FASTQ output, multi-file input, real gzip
    output, `min_fraction` validation), `tests/read_filter_cli.rs` (7 tests,
    real-binary `count -> filter` end to end).
- **S5. Minimizer/super-k-mer partitioning.** **Shipped and promoted to the
  default (2026-09-05).** `auto` selects it for a sized input whose
  predicted peak fits the budget and whose sampled bin balance is under
  `pipeline::MAX_ACCEPTABLE_BIN_SKEW`; measured at 2.4x the previous
  default's speed and half its memory on 840M occurrences. The paragraph
  below describes the state before that decision. `src/binned.rs`, `src/minimizer.rs`, `src/superkmer.rs`,
  `src/adaptive_bins.rs` all exist with substantial test coverage;
  `src/cli.rs::CliStrategy::Binned` exposes it as `--strategy binned` /
  `FASTDNA_STRATEGY=binned`. `pipeline::resolve_strategy`'s `Auto` never
  selects it, by design (`cli.rs`'s own doc comment: "Promoting it is a
  separate, later decision", `docs/design-minimizer-counting.md` §5 step 6).
  What remains open is exactly that promotion decision, not the
  implementation.
- **S6. Multi-sample cohort matrix as a first-class Rust artifact.**
  **Shipped.** `src/cohort/matrix.rs::CohortMatrix` is a real first-class
  Rust artifact -- it builds a cohort-wide k-mer presence/count matrix
  directly from each sample's already-sorted `KmerCounter` table,
  specifically to avoid the double-decode/double-copy the old
  `gwas.py`-calls-`count()`-per-sample approach paid (see that file's own
  module doc comment for the byte-accounting). It is wired into Python as
  `python/fastdna/gwas.py::cohort_presence_matrix()`, returning a
  `scipy.sparse.csr_matrix`. The one remaining piece -- the literal `fastdna
  matrix` CLI verb and a Parquet-file export of the matrix independent of
  the GWAS module -- has now landed:
  - `CohortMatrix` gains a `kmer_u64: Vec<u64>` field, parallel to
    `kmer_sequences` (computed alongside it at no extra cost), so the raw
    2-bit-packed k-mer of every column is available without re-encoding the
    decoded sequence.
  - `cohort::matrix::build_cohort_matrix_from_directory` (reuses
    `discover_samples`'s lenient R1/R2 pairing, distinct from `--paired-
    dir`'s strict variant) and `build_cohort_matrix_from_files` (an
    explicit, unpaired file list) wire sample discovery/counting into
    `build_cohort_matrix` -- the library half of the new CLI verb.
  - `export::export_cohort_matrix_parquet` (`src/export.rs`) writes the
    matrix as a long/"tidy"/COO Parquet table -- one row per nonzero
    `(sample_id, kmer_u64, count)` entry, `kmer_sequence` opt-in via
    `--with-sequence` -- rather than a dense `samples x kmers` grid, which
    would be enormous and almost entirely zero for a real cohort and is not
    the shape `CohortMatrix` holds in memory anyway. `sample_id` is written
    as a string column (not a row index) specifically so the file is usable
    directly from DuckDB/pandas/polars with no FastDNA-specific reader.
    Footer metadata (`fastdna.k`, `fastdna.n_samples`, `fastdna.n_kmers`,
    `fastdna.n_candidates`, `fastdna.truncation_cutoff`) lets a caller
    recover the same facts `gwas.py`'s truncation warning is built from
    without recomputing them. This file is deliberately *not* claimed to be
    a `ktab::KmerTable` (no `fastdna.sorted_by` metadata): its rows are
    grouped by sample, not globally sorted by `kmer_u64`.
  - CLI: `fastdna matrix --input DIR|--sample FILE... -o cohort.parquet
    [-k 31] [--min-count 2] [--min-samples 2] [--max-kmers N]
    [--with-sequence]` (`cli::MatrixArgs`, `main.rs::run_matrix`).
    `--input DIR` reuses `discover_samples`, the same convention `count
    --paired-dir` already uses for directory-of-samples discovery;
    `--sample FILE...` names each sample's file explicitly (no pairing),
    deriving its id the same way `gwas.py::_sample_id_from_path` does so
    ids agree between the CLI verb and the Python module.
  - Python: `gwas.py::cohort_presence_matrix()` is intentionally left
    unchanged -- it remains the in-memory `scipy.sparse` path, and a
    dedicated Python "export to Parquet" entry point was scoped out as
    beyond this item (the CLI verb already covers the file-artifact use
    case S6 asked for).
  - Tests: inline unit tests in `src/cohort/matrix.rs` (the new `kmer_u64`
    field, both new build-from-samples functions, including the
    `min_samples`-above-cohort-size and orphan-warning-not-fatal cases) and
    `src/export.rs` (schema with/without `--with-sequence`, exact COO
    round-trip, `sample_ids` length-mismatch rejection, empty-matrix
    validity, footer metadata including truncation), plus
    `tests/matrix_cli.rs` (real-binary end to end: directory discovery,
    explicit `--sample`, `--with-sequence`, and cross-flag validation
    rejections). (Same underlying evidence as
    `ml-differentiation-roadmap.md`'s B2, now also closed.)
- **S7. ntCard-style streaming spectrum estimate; k up to 64 via u128.**
  **Both halves shipped** -- the spectrum estimator on 2026-08-27, the
  `u128` engine on 2026-09-05 (see the entry below, which was written while
  the second half was still open and is left in place because it records
  why the cascade it feared was avoided rather than paid).
  "Wire or delete `cms.rs`" was resolved by
  deletion: `src/cms.rs` no longer exists (absent from `src/lib.rs`'s
  module list; `CHANGELOG.md`'s `[Unreleased] Removed` section records it).
  - **Shipped**: an ntCard-style streaming k-mer frequency-spectrum
    estimator (Mohamadi, Khan & Birol, *Bioinformatics* 2017) -- `src/
    ntcard.rs`'s `NtCardSketch`/`estimate_spectrum`/`write_spectrum`,
    `fastdna spectrum` (`src/cli.rs`'s `SpectrumArgs`, `src/main.rs`'s
    `run_spectrum`), and `fastdna.estimate_spectrum()` (`src/ffi.rs`,
    `python/fastdna/__init__.py`). Estimates both F0 (distinct k-mers) and
    the full frequency spectrum (f1, f2, ...) in one streaming pass, in
    `2^precision * 16` bytes regardless of input size (256 KB at the
    shared default precision 14 -- 16x `hll::HyperLogLog`'s footprint at
    the same precision, since each bucket keeps a full 64-bit sub-hash plus
    an exact occurrence count instead of one leading-zero byte; see `src/
    ntcard.rs`'s module doc comment for why). Output matches `KmerCounter::
    generate_histogram`/`KmerCounts.spectrum()`'s exact `{depth: distinct
    k-mers}` shape, so it is directly consumable by `python/fastdna/
    genomescope.py::profile_genome` wherever an exact spectrum is too
    expensive to compute first. Accuracy is measured, not assumed, on a
    synthetic dataset shaped like real sequencing data (a large low-depth
    error class plus several deeper coverage classes) against `KmerCounter`'s
    exact spectrum: at precision 14, f1 (the error/noise class, the hardest
    to estimate) is within 15%, and every other frequency class present in
    both spectra is within 30% -- see `src/ntcard.rs`'s
    `ntcard_matches_the_exact_spectrum_within_a_measured_tolerance` test and
    its module doc comment for the full characterization and why these are
    per-run, not universally-guaranteed, bounds.
  - **Shipped 2026-09-05, to k=64 rather than KMC3's 256**: `k > 32` via
    `u128` k-mers (`src/wide_kmer.rs`, `src/wide_counter.rs`, `--engine`).
    Two bits per base in 128 bits is 64 bases; going further needs a
    byte-array key, which changes the sort, the Parquet schema and every
    comparison in the counter, and is a different project. What the
    paragraph below predicted -- that this "would require changing the
    2-bit-per-`u64` representation at its core, which cascades into
    `counter.rs`" -- is exactly what was avoided: the narrow engine is
    untouched, and the wide one sits beside it. The cascade the paragraph
    feared is real, which is why nothing generic was attempted.

    Still narrow-only, and refused by name rather than misread
    (`ktab::SORTED_BY_WIDE_VALUE`): `query`, `union`/`intersect`/`diff`,
    `filter`, `similarity`, sketching, and the Python API. A wide table is
    a Parquet file keyed on 16 big-endian bytes; those operations are all
    `u64`-keyed.

  - **The original deferral, for the record**: k > 32 support via `u128`
    k-mers. This would require changing the 2-bit-per-base `u64` k-mer
    representation at its core (`src/kmer.rs`'s encoding/rolling-extraction
    functions), which cascades into `src/counter.rs` (the table that stores
    those k-mers). `src/counter.rs` had real, unrelated, uncommitted work in
    flight on the host at the time this pass ran, which this change was
    explicitly required not to touch or conflict with -- so the u128 half
    was scoped out rather than attempted half-correct or landed in a state
    that would collide with that in-flight work. Picking it up later means
    starting from `src/kmer.rs`'s `base_to_bits`/`extract_canonical_
    kmers_into`/`canonical_kmer_u64` (all currently hard-coded to `u64` and
    `k <= 32`) and propagating the wider type through `counter.rs`,
    `export.rs`'s schema, and every CLI/FFI surface that currently validates
    `k <= 32` (`FastDnaError::InvalidK`'s own range).

## Deliberately not copying

- Jellyfish's lock-free hash table (our sort-and-compact beat a hash design 9x in our own benchmarks).
- BAM/CRAM input (heavy htslib/noodles dependency; `samtools fastq | fastdna -i -` covers it now that Q2 has landed).
- Full sourmash ecosystem (SBT/LCA databases); consider exporting sketches in sourmash signature JSON instead.
- Histex/Vennex-style micro-tool zoo (subsumed by subcommands + Python -- Q4 landed this pass).
- FastK homopolymer compression (long-read mode is a real project — quality trimming itself is wrong for ONT — not a flag).
- KMC-style memory knob forest (auto strategy + `--max-ram` is a better UX; keep it).

## Suggested execution order

> **Spent, 2026-09-07.** Every item below has shipped except Q5 (Bioconda),
> and Q5's stated justification -- "per
> `docs/goal-most-complete-genomics-ml-library.md` publishability is now a
> completeness blocker" -- cites a goal document that was superseded and
> deleted on 2026-09-05. Publishing is no longer a completeness question
> (`docs/goal-fast-kmer-counter.md`).
>
> The list is kept rather than rewritten because it records the order
> things were actually done in, and because a plan that turned out to be
> right is worth more as evidence than as instructions. Read it as history.
> References below to `still open` items describe the state on the date
> each line was written.

Re-derived 2026-08-27 against what is actually still open (see status column
above); the original document's foundation-first logic ("database before set
ops before filtering") is unchanged, just re-anchored to what remains:

1. **Finish Q5** (Bioconda: real source URL/sha256, `conda build`,
   bioconda-recipes PR) -- cheapest remaining item, and per
   `docs/goal-most-complete-genomics-ml-library.md` publishability is now a
   completeness blocker, not a nice-to-have.
2. **S1** (binary k-mer database + query API) -- a concrete plan already
   exists (`docs/superpowers/plans/2026-08-24-completeness-phase.md`);
   nothing else in this list can start without it.
3. **S2 → S4** (set operations, then filtering). Both landed: S2 this pass
   (`src/setops.rs`), S4 (the read-filtering CLI/Python surface,
   `src/read_filter.rs`) in a later pass the same day, exactly as
   originally ordered. S4's paired-end (R1/R2) synchronized filtering is a
   documented, explicit gap -- see its own entry above.
4. Both small, concrete finishing touches named here have now **shipped**:
   exposing `cohort::matrix::CohortMatrix` as a `fastdna matrix` CLI verb
   plus a generic Parquet export (closes S6/B2 for real; see this doc's own
   S6 entry above, `src/export.rs::export_cohort_matrix_parquet`,
   `cli::MatrixArgs`), and wiring `assembly_qc.py`'s FASTA path through
   `fastdna.count()` now that Q1's native FASTA support exists instead of
   its pure-Python fallback (closes B4; see this doc's own B4 entry above
   and `ml-differentiation-roadmap.md`).
5. **S3 proper** (FastK-style full per-read profiles/QV over a generic
   per-sample database) has now **shipped**, once S1 existed to build it on
   -- see this doc's own S3 entry above (`src/read_profile.rs`).
6. **S7** (ntCard streaming spectrum, k up to 64 via u128) -- lowest
   priority, unchanged from the original ranking.
7. Snakemake/Nextflow workflow templates (the one genuinely-missing piece
   of `ml-genomics-roadmap.md`'s wave-1 feature 4) -- small and concrete,
   worth folding into this pass rather than leaving indefinitely open.

Sources: KMC3 repo/paper, kmc_tools usage docs, FastK repo, Jellyfish repo,
GenomeScope 2.0 repo, sourmash docs. Repo files verified: `src/cli.rs`,
`src/lib.rs`, `src/fastq.rs`, `src/export.rs`, `src/main.rs`,
`src/disk_spill.rs`, `src/sketch.rs`, `src/hll.rs`, `src/metagenomics.rs`,
`src/cohort/matrix.rs`, `src/preview.rs`, `src/cohort/`,
`python/fastdna/*.py`, `recipe/meta.yaml`, `README.md`. `src/cms.rs` was
verified absent (deleted; see S7).
