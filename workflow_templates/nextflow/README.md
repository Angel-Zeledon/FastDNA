# FastDNA cohort k-mer counting -- Nextflow template

A minimal, runnable Nextflow (DSL2) pipeline that counts canonical k-mers
across a cohort of paired-end FASTQ samples and combines every sample's
table into one cohort-level k-mer table. Conceptually identical to
`workflow_templates/snakemake/` -- pick whichever workflow manager your
pipeline already uses.

## Layout

```
workflow_templates/nextflow/
├── main.nf              # the pipeline itself -- should not need editing
├── nextflow.config       # edit this: input/output paths, k, thread count, ...
└── README.md             # this file
```

## What it expects as input

A directory of paired-end FASTQ(.gz) files, one pair per sample, named:

```
data/
├── sample_a_R1.fastq.gz
├── sample_a_R2.fastq.gz
├── sample_b_R1.fastq.gz
└── sample_b_R2.fastq.gz
```

`main.nf` discovers samples with `Channel.fromFilePairs("data/*_R{1,2}.fastq.gz",
size: 2)`, Nextflow's standard idiom for a directory of mate pairs -- the
same `_R1`/`_R2` convention FastDNA's own `fastdna --paired-dir` batch mode
recognizes (see the main repository `README.md`'s "Paired-end batch
counting"). `size: 2` makes an unpaired R1 with no R2 (or vice versa) a hard
error at channel-construction time rather than a silently accepted
single-end sample.

## What it produces

Published under `params.output_dir` (`results/` by default):

- `<sample_id>.counts.parquet` -- one k-mer frequency table per sample
  (`fastdna count`'s own output format).
- `<sample_id>.qc.json` -- one QC report per sample.
- `cohort_union.parquet` -- every sample's k-mer table folded into one
  cohort-level table via `fastdna union` (counts for the same k-mer summed
  across samples by default; see `nextflow.config`'s `combine` parameter).

## How to run it

1. Install the `fastdna` CLI (see the main repository `README.md`'s
   "Building from source") so `fastdna` is on `PATH`, and install Nextflow
   itself (`curl -s https://get.nextflow.io | bash`, or see
   [nextflow.io](https://www.nextflow.io/)).
2. Edit `nextflow.config`'s `params` block: point `fastq_dir` at your
   directory of paired-end FASTQ files, and adjust `k`/`min_quality`/
   `min_count`/`threads`/`combine` if the defaults don't fit your data --
   or override any of them on the command line instead, e.g.:

   ```bash
   nextflow run main.nf --fastq_dir /path/to/my/data --k 25
   ```

3. From this directory, with the defaults in `nextflow.config`:

   ```bash
   nextflow run main.nf
   ```

4. A preview run (`nextflow run main.nf -preview`) validates the workflow
   and prints the process graph without executing anything -- useful for
   checking that your `fastq_dir` resolves correctly before committing to a
   real run.

## Notes

- Every sample uses the same `k`/`min_quality`/`min_count`/`threads`
  parameters (set once in `nextflow.config`), matching the same constraint
  FastDNA's own `--paired-dir` batch mode documents.
- `process.executor = 'local'` in `nextflow.config` runs everything on the
  machine invoking `nextflow run`; add a `-profile`/site config to target a
  cluster or cloud executor instead -- nothing in `main.nf` assumes local
  execution.
- A single-sample "cohort" has nothing to union against (`fastdna union`
  requires at least two tables); in that case `FASTDNA_UNION` just copies
  the one sample's table forward as `cohort_union.parquet` instead of
  invoking `fastdna union` (see `main.nf`'s `FASTDNA_UNION` process).
- Verification: this template's syntax and the `fastdna` invocations inside
  it were checked by hand against `fastdna --help`/`fastdna count --help`/
  `fastdna union --help` output from a real release build of this
  repository's `src/cli.rs`, not from memory of the README alone. A live
  `nextflow run` (or `-preview`) against a real FASTQ fixture directory was
  not performed in the environment this was written in (no Nextflow-enabled
  Docker image was available in this session) -- treat that as the one
  remaining verification step before production use.
