# FastDNA cohort k-mer counting -- Snakemake template

A minimal, runnable Snakemake pipeline that counts canonical k-mers across a
cohort of paired-end FASTQ samples and combines every sample's table into
one cohort-level k-mer table.

## Layout

```
workflow_templates/snakemake/
├── config.yaml          # edit this: input/output paths, k, thread count, ...
├── workflow/
│   └── Snakefile         # the pipeline itself -- should not need editing
└── README.md             # this file
```

This matches Snakemake's own recommended project layout (`workflow/Snakefile`
alongside a top-level config file), so `snakemake` run from this directory
auto-discovers `workflow/Snakefile` with no `--snakefile` flag needed.

## What it expects as input

A directory of paired-end FASTQ(.gz) files, one pair per sample, named:

```
data/
├── sample_a_R1.fastq.gz
├── sample_a_R2.fastq.gz
├── sample_b_R1.fastq.gz
└── sample_b_R2.fastq.gz
```

This is the same `_R1`/`_R2` convention FastDNA's own `fastdna --paired-dir`
batch mode recognizes (see the main repository `README.md`'s "Paired-end
batch counting" section); this Snakemake template globs directly on it so
each sample becomes its own wildcard-addressable rule instance, rather than
one whole-directory CLI call. `_1`/`_2` and Illumina's `_R1_001`/`_R2_001`
naming are not matched by this template's glob pattern -- rename/symlink
those files to the `_R1`/`_R2` form first, or edit the `glob_wildcards(...)`
pattern in `workflow/Snakefile` to match your own convention.

## What it produces

- `results/<sample>.counts.parquet` -- one k-mer frequency table per sample
  (`fastdna count`'s own output format).
- `results/<sample>.qc.json` -- one QC report per sample.
- `results/cohort_union.parquet` -- every sample's k-mer table folded into
  one cohort-level table via `fastdna union` (counts for the same k-mer
  summed across samples by default; see `config.yaml`'s `combine` setting).
- `results/logs/<sample>.fastdna_count.log` -- each sample's counting-run
  stdout/stderr.

## How to run it

1. Install the `fastdna` CLI (see the main repository `README.md`'s
   "Building from source") so `fastdna` is on `PATH`, and install Snakemake
   itself (`pip install snakemake`).
2. Edit `config.yaml`: point `fastq_dir` at your directory of paired-end
   FASTQ files, and adjust `k`/`min_quality`/`min_count`/`threads`/`combine`
   if the defaults don't fit your data.
3. From this directory:

   ```bash
   snakemake --cores 4 --configfile config.yaml
   ```

   `--configfile` is required explicitly (rather than baked into the
   `Snakefile` via a `configfile:` directive) because that directive's path
   resolution differs across Snakemake versions depending on whether it is
   relative to the current working directory or to the `Snakefile` itself --
   naming it explicitly on the command line sidesteps that ambiguity for any
   Snakemake version.

4. A dry run (`snakemake --cores 4 --configfile config.yaml -n`) prints the
   planned rule graph without running anything -- useful for checking that
   your `fastq_dir` was discovered correctly before committing to a real run.

## Notes

- Every sample uses the same `k`/`min_quality`/`min_count`/`threads`
  settings (set once in `config.yaml`), matching the same constraint
  FastDNA's own `--paired-dir` batch mode documents.
- A single-sample "cohort" has nothing to union against (`fastdna union`
  requires at least two tables); in that case the pipeline just copies the
  one sample's table forward as `cohort_union.parquet` instead of invoking
  `fastdna union` (see `workflow/Snakefile`'s `fastdna_union` rule).
- Verification: this template's syntax and the `fastdna` invocations inside
  it were checked by hand against `fastdna --help`/`fastdna count --help`/
  `fastdna union --help` output from a real release build of this
  repository's `src/cli.rs`, not from memory of the README alone. A live
  `snakemake -n` dry run against a real FASTQ fixture directory was not
  performed in the environment this was written in (no Snakemake-enabled
  Docker image was available in this session) -- treat that as the one
  remaining verification step before production use.
