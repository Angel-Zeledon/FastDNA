# K-mer tables

The operations on counts that already exist. A `.parquet` file written by
`fastdna count` **is** a k-mer table -- there is no separate build step and
no proprietary format -- so anything here can be pointed at the output of a
counting run, or at the output of another operation here.

Every one of them reads both table widths: `kmer_u64` for `k <= 32` and the
16-byte `kmer_bits` above it. The reader is chosen from the file's own
Parquet footer, never from an argument. Inputs to a single operation must
share a width and a counting convention -- and mixing either is refused by
name rather than allowed to produce a well-formed wrong answer, since two
widths never share a `k` and a canonical table and a non-canonical one
disagree about what a key means.

::: fastdna.KmerTable

## Exact similarity between tables

::: fastdna.similarity

!!! info "Exact, where `compare_all` is approximate"

    [`compare_all`](sketching.md#fastdna.compare_all) builds a MinHash
    sketch per sample and compares those, which is what makes it
    `O(sketch_size)` per pair instead of `O(genome)`. `similarity` reads the
    real tables: it costs a pass over them and answers with no sampling
    error at all.

    It also reports `bray_curtis`, which needs the counts a sketch throws
    away and is therefore not computable from sketches even in principle.

    Validated against `kmc_tools`' own set operations on real reads, at
    k=31 and k=41, agreeing exactly on every set size and every derived
    ratio.
