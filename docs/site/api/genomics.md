# Genomics utilities

Analyses built on the same k-mer engine that are not about fitting a model:
identifying what a sample is, profiling a genome without assembling it,
grading an assembly without a reference, mapping a k-mer back to a gene,
translating nucleotides to protein, and checking whether generated sequence
is plausible.

Each of these is pure Python on top of the stable counting and sketching API.
None of them adds a new file format or a new Rust surface — that is this
package's own rule, not a coincidence.

## Identifying a sample

::: fastdna.taxonomy

## Per-read metagenomic classification

::: fastdna.metagenomics

## Reference-free genome profiling

::: fastdna.genomescope

## Reference-free assembly QC

::: fastdna.assembly_qc

## From k-mer back to gene

::: fastdna.annotate

<!-- Both paragraphs below are carried over by hand because they are
     currently dropped in rendering: in each case the docstring continues in
     prose *inside* the `Parameters` block instead of closing the section
     first, so griffe's NumPy parser reads the paragraph as a run of nameless
     parameters and emits nothing for it. Delete this note (and the
     `warnings: false` in mkdocs.yml) once python/fastdna/annotate.py moves
     those paragraphs out of their sections. -->
!!! note "Two behaviours documented in the source but not rendered above"

    **`load_annotation` raises `ValueError`** if none of the annotation's
    `seqid`s match any record id in the reference FASTA. That mismatch is
    almost always a sign that the two files describe different assemblies, or
    that one uses accession numbers where the other uses plain contig names —
    and discovering it later, from a silent all-intergenic result, would be a
    far worse experience.

    **`export_bed` always writes `0` in BED's `score` field (column 5).**
    Nothing in the input table is a score in BED's `0-1000` sense, and
    inventing one would be exactly the kind of fabricated threshold this
    project's other modules (`gwas.prefilter_association`,
    `plotting.plot_significance`) are written to avoid.

## DNA to protein

::: fastdna.translate

## Checking generated sequence

::: fastdna.validate_generated
