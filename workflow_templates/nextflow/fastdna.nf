#!/usr/bin/env nextflow
// fastdna.nf -- minimal FastDNA process/module for Nextflow (DSL2)
//
// Equivalent to workflow_templates/snakemake/Snakefile.example's
// `fastdna_count` rule: one process invocation per sample, shelling out to
// the fastdna CLI. CLI flags match the current README's "Command-line
// interface" section exactly:
//
//   fastdna --input sample.fastq.gz --output counts.parquet -k 31
//   -i/--input, -o/--output, -k/--kmer-size, -q/--min-quality,
//   -m/--min-count, -M/--max-count, -t/--threads, --qc, --histogram
//
// Copy this file into your pipeline (e.g. modules/fastdna.nf), `include`
// it, and wire FASTDNA_COUNT into your own workflow block with your own
// sample channel -- nothing else here should need to change to run.
//
// Verification note: written by hand against current Nextflow DSL2 module
// conventions (process block with input:/output:/script:); the `nextflow`
// CLI was not available to actually run `nextflow run` against this file
// in the environment this was written in -- see this repo's contribution
// notes for what was and wasn't run.

nextflow.enable.dsl = 2

params.k           = 31
params.min_quality = 20.0
params.min_count   = 1
params.threads     = 4

process FASTDNA_COUNT {
    tag "${sample_id}"
    cpus params.threads
    publishDir "results", mode: 'copy'

    input:
    tuple val(sample_id), path(fastq)

    output:
    tuple val(sample_id), path("${sample_id}.counts.parquet"), emit: counts
    tuple val(sample_id), path("${sample_id}.qc.json"),        emit: qc

    script:
    """
    fastdna \\
        --input ${fastq} \\
        --output ${sample_id}.counts.parquet \\
        --kmer-size ${params.k} \\
        --min-quality ${params.min_quality} \\
        --min-count ${params.min_count} \\
        --threads ${task.cpus} \\
        --qc ${sample_id}.qc.json
    """
}

// Example wiring -- point `sample_ch` at your own FASTQ(.gz) files:
//
// workflow {
//     sample_ch = Channel.fromFilePairs("data/*.fastq.gz", size: 1)
//         .map { sample_id, files -> tuple(sample_id, files[0]) }
//
//     FASTDNA_COUNT(sample_ch)
// }
