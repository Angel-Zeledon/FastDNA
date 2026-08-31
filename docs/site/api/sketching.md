# Sketching and comparison

A sketch is a fixed-size fingerprint of a sample's canonical k-mer set. It
answers *"how similar are these two samples"*, not *"what exactly is in
each"* — and that restriction is what makes cohort-scale comparison
affordable.

`compare_all()` is the entry point for cohort work: it builds one sketch per
sample and then compares every pair, which costs `O(N)` FASTQ reads plus
`O(N²)` sketch comparisons. Each comparison is `O(sketch_size)`, not
`O(genome size)`. Comparing full k-mer sets pair by pair would instead cost
`O(N²)` FASTQ reads — the exact cost sketching exists to avoid.

Its output is also the input to the leakage-aware cross-validation in
[`fastdna.cv`](evaluation.md#fastdna.cv): all-pairs Mash distances are a
serviceable stand-in for a phylogeny at the resolution that question needs.

## MinHash

::: fastdna.sketch

::: fastdna.Sketch

::: fastdna.load_sketch

## FracMinHash (scaled)

::: fastdna.frac_sketch

::: fastdna.FracSketch

::: fastdna.load_frac_sketch

## Comparing

::: fastdna.compare

::: fastdna.compare_all

## Visualizing a cohort

::: fastdna.embed
