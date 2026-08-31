#!/usr/bin/env nextflow
// main.nf -- cohort k-mer counting pipeline for a directory of paired-end
// FASTQ samples (DSL2). Conceptually identical to
// workflow_templates/snakemake/workflow/Snakefile -- see this directory's
// README.md for how to run it and what it expects as input.
//
// End to end, this pipeline:
//   1. Discovers every sample under params.fastq_dir via
//      Channel.fromFilePairs's "<sample>_R{1,2}.fastq.gz" glob (Nextflow's
//      own idiomatic way to turn a directory of mate pairs into a channel
//      of (sample_id, [R1, R2]) tuples -- the direct equivalent of the
//      Snakemake template's glob_wildcards call).
//   2. Counts each sample's canonical k-mers with `fastdna count`, feeding
//      both mates in as two --input files in one run -- FastDNA already
//      aggregates multiple --input files into a single counting pass (see
//      the main repository README.md's "Paired-end batch counting"), so
//      this process gets the exact same per-sample result FastDNA's own
//      --paired-dir batch mode would, one sample per process instance
//      instead of one directory-wide subprocess.
//   3. Combines every sample's Parquet table into one cohort-level k-mer
//      table with `fastdna union` -- this pipeline's "combined/summary
//      output" deliverable (see README.md's Subcommands table and
//      src/setops.rs for what union actually does).

nextflow.enable.dsl = 2


process FASTDNA_COUNT {
    // tag makes `nextflow log`/the execution report readable per-sample
    // instead of per opaque task hash.
    tag "${sample_id}"
    cpus params.threads
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}.counts.parquet"), emit: counts
    path "${sample_id}.qc.json"

    script:
    // reads[0]/reads[1] are R1/R2 respectively: Channel.fromFilePairs below
    // sorts each pair's two paths lexically, and "_R1" sorts before "_R2"
    // for every naming convention this glob matches.
    """
    fastdna count \\
        --input ${reads[0]} --input ${reads[1]} \\
        --output ${sample_id}.counts.parquet \\
        --kmer-size ${params.k} \\
        --min-quality ${params.min_quality} \\
        --min-count ${params.min_count} \\
        --threads ${task.cpus} \\
        --qc ${sample_id}.qc.json
    """
}


process FASTDNA_UNION {
    // A single process over the whole cohort (not one instance per sample):
    // folding every sample's table into one cohort table is inherently a
    // many-to-one step, unlike FASTDNA_COUNT above.
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path(count_tables)

    output:
    path "cohort_union.parquet"

    script:
    // `fastdna union --input` requires at least two tables ("a union of
    // fewer than two tables is not a combination of anything" --
    // src/cli.rs's UnionArgs doc comment): a single-sample cohort has
    // nothing to union against, so that case just copies the one sample's
    // table forward as the "cohort" output instead of calling a command
    // that would reject it -- the same fallback the Snakemake template's
    // fastdna_union rule takes.
    """
    set -euo pipefail
    n_tables=\$(ls -1 *.parquet | wc -l)
    if [ "\${n_tables}" -eq 1 ]; then
        cp *.parquet cohort_union.parquet
    else
        fastdna union --input ${count_tables} --output cohort_union.parquet --combine ${params.combine}
    fi
    """
}


workflow {
    // fromFilePairs(..., size: 2) enforces exactly two files per discovered
    // sample id -- an unpaired R1 with no R2 (or vice versa) fails fast here
    // rather than silently being treated as single-end, matching the same
    // "an unpaired file is a hard error, not a fallback" stance FastDNA's own
    // --paired-dir mode documents for its own cohort discovery.
    sample_ch = Channel.fromFilePairs("${params.fastq_dir}/*_R{1,2}.fastq.gz", size: 2)

    FASTDNA_COUNT(sample_ch)

    // .map { it[1] } drops the sample_id, keeping just each sample's
    // Parquet path; .collect() gathers every sample's output into the one
    // list FASTDNA_UNION's single process instance consumes.
    FASTDNA_UNION(FASTDNA_COUNT.out.counts.map { sample_id, table -> table }.collect())
}
